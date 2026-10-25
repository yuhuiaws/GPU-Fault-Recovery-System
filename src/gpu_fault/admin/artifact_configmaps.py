"""How release artifacts are stored in Kubernetes ConfigMaps.

A ConfigMap holds at most 1 MiB of data (Kubernetes ``MaxSecretSize``), and the
release artifacts have crossed that ceiling twice: the raw control-plane wheel
on 2026-09-07 (1,106,539 bytes; xz took it back under) and the xz-compressed
one on 2026-10-25 (1,051,472 bytes; the release preflight died in
``kubectl create configmap`` with "Too long: may not be more than 1048576
bytes" and nothing could be released until the engine learned to split).
Nothing installs a wheel from the ConfigMap at runtime -- the runtime image
already carries it -- so the ConfigMap only has to keep the artifact
recoverable and digest-checkable.

Three layouts; every reader accepts all of them and the writer picks by size:

* raw: one ConfigMap, ``binaryData[<artifact filename>]``. Node installer
  bundles, which the installer Job reads straight from its mount; releases
  before 2026-09-07 stored wheels this way too.
* xz: one ConfigMap, ``binaryData[<artifact filename>.xz]``. A wheel whose xz
  stream fits in one object.
* chunked set: the named ConfigMap carries only ``data["artifact.json"]`` --
  the artifact's name, size and sha256, the stream encoding and the ordered
  list of parts -- and each part ``<name>-part-<k>`` carries one slice of the
  (xz or raw) stream under ``binaryData``, at most ``ARTIFACT_PART_BYTES``
  long. A Deployment keeps mounting the named ConfigMap, so the name stays the
  pin the rollout, the rollback and the pruner reason about; the parts are
  reached only through the manifest. Parts are created before the manifest,
  so a manifest's existence implies a complete set, and every reader re-hashes
  what it read: each part against the manifest, the stream against the
  manifest, the decoded artifact against the manifest.

An artifact a Pod reads from its mount (the bundle) cannot be split: the writer
refuses it with its size and the limit instead of letting an installer Job
start against a mount that holds only a manifest.

The digest recorded next to a ConfigMap (``gpu-fault.io/artifact-sha256`` and
the release manifest) is always the artifact's own sha256, whichever layout the
ConfigMap set holds.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMPRESSED_ARTIFACT_SUFFIX = ".xz"
#: Kubernetes ``core.MaxSecretSize``: the sum of a ConfigMap's ``data`` and
#: ``binaryData`` values (decoded bytes; keys are not counted).
CONFIG_MAP_PAYLOAD_LIMIT = 1_048_576
#: Slice length of one part of a chunked set. Well under the limit, so a part
#: never has to be measured against it, and large enough that a wheel takes a
#: handful of objects: the 2026-10-25 control-plane wheel (1,051,472 bytes xz)
#: is two parts.
ARTIFACT_PART_BYTES = 768 * 1024
#: The ``data`` key of a chunked set's named ConfigMap.
MANIFEST_KEY = "artifact.json"
MANIFEST_SCHEMA_VERSION = 1
PART_NAME_INFIX = "-part-"
#: Beyond this a release is shipping something other than a wheel (48 MiB of
#: xz); refuse rather than paper a namespace with ConfigMaps.
MAX_PARTS = 64
ENCODING_XZ = "xz"
ENCODING_NONE = "none"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PART_NAME = re.compile(rf"^(?P<primary>.+){re.escape(PART_NAME_INFIX)}(?P<index>\d+)$")
_CONFIG_MAP_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?$")


class ArtifactLayoutError(ValueError):
    """A ConfigMap set does not hold the artifact it claims to hold."""


class ArtifactTooLarge(ValueError):
    """The artifact cannot be shipped through a ConfigMap set."""


@dataclass(frozen=True)
class ArtifactObject:
    """One ConfigMap of an artifact's set: its name, its single key and payload."""

    name: str
    key: str
    payload: bytes
    #: ``binaryData`` (True) or ``data`` (False; the manifest only).
    binary: bool
    #: The artifact file itself when the payload is its unencoded content (a
    #: raw single object): kubectl reads it in place, nothing is copied.
    source: Path | None = None

    @property
    def size(self) -> int:
        return len(self.payload)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


@dataclass(frozen=True)
class ArtifactPlan:
    """The ConfigMap objects that ship one artifact, parts first, primary last."""

    name: str
    artifact: str
    sha256: str
    size: int
    encoding: str
    payload_size: int
    objects: tuple[ArtifactObject, ...]

    @property
    def chunked(self) -> bool:
        return len(self.objects) > 1

    @property
    def primary(self) -> ArtifactObject:
        return self.objects[-1]

    @property
    def parts(self) -> tuple[ArtifactObject, ...]:
        return self.objects[:-1]

    @property
    def layout(self) -> str:
        if self.chunked:
            return f"chunked:{len(self.parts)}"
        return self.encoding

    @property
    def limit_ratio(self) -> float:
        """How much of one ConfigMap the whole stream would take."""

        return self.payload_size / CONFIG_MAP_PAYLOAD_LIMIT


def part_config_map_name(name: str, index: int) -> str:
    return f"{name}{PART_NAME_INFIX}{index}"


def artifact_set_name(name: str) -> str:
    """The named (primary) ConfigMap of ``name``: itself unless it is a part."""

    match = _PART_NAME.match(name)
    return match.group("primary") if match else name


def compress_artifact(path: Path, destination: Path) -> Path:
    """Write ``path`` xz-compressed to ``destination`` and return it."""

    destination.write_bytes(encode_artifact(path.read_bytes(), compress=True))
    return destination


def encode_artifact(data: bytes, *, compress: bool) -> bytes:
    if not compress:
        return data
    return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)


def plan_artifact_config_maps(
    path: Path,
    *,
    name: str,
    key: str,
    compress: bool,
    splittable: bool,
    label: str = "artifact",
) -> ArtifactPlan:
    """Decide how ``path`` ships under the ConfigMap ``name``.

    One object when the (xz or raw) stream fits the limit, a chunked set
    otherwise. ``splittable`` is False for an artifact a Pod reads from its
    mount; such an artifact over the limit raises ``ArtifactTooLarge`` with the
    numbers an operator needs, before anything is created.
    """

    if not _CONFIG_MAP_NAME.match(name) or len(name) > 253 - len(
        part_config_map_name("", MAX_PARTS)
    ):
        raise ArtifactLayoutError(f"{label} ConfigMap name is invalid: {name}")
    if not key or "/" in key or key.startswith("."):
        raise ArtifactLayoutError(f"{label} ConfigMap key is invalid: {key}")
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    payload = encode_artifact(data, compress=compress)
    encoding = ENCODING_XZ if compress else ENCODING_NONE
    stored_key = key + COMPRESSED_ARTIFACT_SUFFIX if compress else key
    if len(payload) <= CONFIG_MAP_PAYLOAD_LIMIT:
        objects: tuple[ArtifactObject, ...] = (
            ArtifactObject(
                name,
                stored_key,
                payload,
                binary=True,
                source=None if compress else path,
            ),
        )
        return ArtifactPlan(
            name, key, sha256, len(data), encoding, len(payload), objects
        )
    described = (
        f"{label} {path.name} is {len(payload):,} bytes"
        + (f" xz-compressed ({len(data):,} raw)" if compress else "")
        + f"; a ConfigMap holds at most {CONFIG_MAP_PAYLOAD_LIMIT:,} bytes"
    )
    if not splittable:
        raise ArtifactTooLarge(
            described + " and this artifact is read from its mount, so it cannot "
            "be split across ConfigMaps -- shrink it before releasing"
        )
    slices = [
        payload[offset : offset + ARTIFACT_PART_BYTES]
        for offset in range(0, len(payload), ARTIFACT_PART_BYTES)
    ]
    if len(slices) > MAX_PARTS:
        raise ArtifactTooLarge(
            described + f" and splitting it would take {len(slices)} parts; the "
            f"engine ships at most {MAX_PARTS}"
        )
    parts = tuple(
        ArtifactObject(
            part_config_map_name(name, index),
            f"{stored_key}.{index:02d}",
            chunk,
            binary=True,
        )
        for index, chunk in enumerate(slices)
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact": key,
        "sha256": sha256,
        "size": len(data),
        "encoding": encoding,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_size": len(payload),
        "parts": [
            {
                "config_map": part.name,
                "key": part.key,
                "size": part.size,
                "sha256": part.sha256,
            }
            for part in parts
        ],
    }
    primary = ArtifactObject(
        name,
        MANIFEST_KEY,
        (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode(),
        binary=False,
    )
    return ArtifactPlan(
        name, key, sha256, len(data), encoding, len(payload), (*parts, primary)
    )


def write_artifact_object(item: ArtifactObject, directory: Path) -> Path:
    """Materialise ``item`` for ``kubectl create configmap --from-file``.

    kubectl files a UTF-8 file under ``data`` and anything else under
    ``binaryData``; an xz stream starts with ``\\xfd``, a gzip stream with
    ``\\x1f\\x8b``, so a part lands in ``binaryData`` and the JSON manifest in
    ``data`` -- the readers accept either placement for a part regardless.
    """

    if item.source is not None:
        return item.source
    target = directory / item.key
    target.write_bytes(item.payload)
    return target


def _mapping(value: object, what: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ArtifactLayoutError(f"{what} is invalid")
    return dict(value)


def artifact_manifest(document: Mapping[str, object]) -> dict[str, Any] | None:
    """The validated chunked-set manifest of ``document``, or None for a single-object layout."""

    data = _mapping(document.get("data"), "ConfigMap data")
    if MANIFEST_KEY not in data:
        return None
    raw = data[MANIFEST_KEY]
    if not isinstance(raw, str):
        raise ArtifactLayoutError(f"{MANIFEST_KEY} is not text")
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise ArtifactLayoutError(f"{MANIFEST_KEY} is not JSON: {exc}") from None
    if not isinstance(manifest, dict):
        raise ArtifactLayoutError(f"{MANIFEST_KEY} is not an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactLayoutError(
            f"{MANIFEST_KEY} schema_version {manifest.get('schema_version')!r} is "
            f"not {MANIFEST_SCHEMA_VERSION}"
        )
    artifact = manifest.get("artifact")
    if not isinstance(artifact, str) or not artifact or "/" in artifact:
        raise ArtifactLayoutError(f"{MANIFEST_KEY} names no artifact")
    for field in ("sha256", "payload_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ArtifactLayoutError(f"{MANIFEST_KEY} {field} is not a sha256")
    for field in ("size", "payload_size"):
        value = manifest.get(field)
        if type(value) is not int or value < 0:
            raise ArtifactLayoutError(f"{MANIFEST_KEY} {field} is not a size")
    if manifest.get("encoding") not in {ENCODING_XZ, ENCODING_NONE}:
        raise ArtifactLayoutError(f"{MANIFEST_KEY} encoding is unknown")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts or len(parts) > MAX_PARTS:
        raise ArtifactLayoutError(f"{MANIFEST_KEY} lists no usable parts")
    for part in parts:
        if (
            not isinstance(part, dict)
            or not isinstance(part.get("config_map"), str)
            or _CONFIG_MAP_NAME.match(part["config_map"]) is None
            or not isinstance(part.get("key"), str)
            or not part["key"]
            or type(part.get("size")) is not int
            or part["size"] <= 0
            or not isinstance(part.get("sha256"), str)
            or _SHA256.fullmatch(part["sha256"]) is None
        ):
            raise ArtifactLayoutError(f"{MANIFEST_KEY} has a malformed part")
    return manifest


def artifact_part_names(document: Mapping[str, object]) -> tuple[str, ...]:
    """The part ConfigMaps a chunked set's primary names; empty otherwise."""

    manifest = artifact_manifest(document)
    if manifest is None:
        return ()
    return tuple(str(part["config_map"]) for part in manifest["parts"])


def stored_artifact_key(document: Mapping[str, object]) -> str | None:
    """The artifact a ConfigMap stores, as its readers name it.

    A chunked set names the artifact file in its manifest; a single-object
    layout is identified by its sole ``binaryData`` key (``<file>`` or
    ``<file>.xz``). None when neither applies.
    """

    manifest = artifact_manifest(document)
    if manifest is not None:
        return str(manifest["artifact"])
    keys = sorted(_mapping(document.get("binaryData"), "ConfigMap binaryData"))
    return keys[0] if len(keys) == 1 else None


def _binary_values(document: Mapping[str, object], what: str) -> dict[str, str]:
    binary = _mapping(document.get("binaryData"), f"{what} binaryData")
    if any(not isinstance(value, str) for value in binary.values()):
        raise ArtifactLayoutError(f"{what} binaryData is invalid")
    return {key: str(value) for key, value in binary.items()}


def _part_bytes(document: Mapping[str, object], key: str, name: str) -> bytes:
    binary = _binary_values(document, name)
    if key in binary:
        try:
            return base64.b64decode(binary[key], validate=True)
        except ValueError:
            raise ArtifactLayoutError(f"{name}/{key} is not base64") from None
    data = _mapping(document.get("data"), f"{name} data")
    if isinstance(data.get(key), str):
        # kubectl files a slice that happens to be valid UTF-8 under ``data``;
        # the text round-trips to the same bytes.
        return str(data[key]).encode("utf-8")
    raise ArtifactLayoutError(f"{name}/{key} is missing")


def read_artifact_digest(
    document: Mapping[str, object],
    key: str,
    fetch: Callable[[str], Mapping[str, object]],
    *,
    name: str = "ConfigMap",
) -> tuple[str, str] | None:
    """(artifact key, sha256 of the artifact) stored by ``document``'s set.

    ``fetch`` reads a part ConfigMap by name; it is only called for a chunked
    set. A single-object layout answers like ``artifact_binary_sha``; None when
    the document holds nothing recognisable. Every byte is re-hashed: a part,
    stream or manifest digest that disagrees with the content raises
    ``ArtifactLayoutError`` rather than returning the manifest's claim.
    """

    manifest = artifact_manifest(document)
    if manifest is None:
        return artifact_binary_sha(_binary_values(document, name), key)
    stream = bytearray()
    for part in manifest["parts"]:
        part_name = str(part["config_map"])
        chunk = _part_bytes(fetch(part_name), str(part["key"]), part_name)
        if (
            len(chunk) != part["size"]
            or hashlib.sha256(chunk).hexdigest() != part["sha256"]
        ):
            raise ArtifactLayoutError(
                f"{name} part {part_name}/{part['key']} does not match its manifest"
            )
        stream += chunk
    if (
        len(stream) != manifest["payload_size"]
        or hashlib.sha256(stream).hexdigest() != manifest["payload_sha256"]
    ):
        raise ArtifactLayoutError(f"{name} parts do not reassemble to the manifest")
    if manifest["encoding"] == ENCODING_XZ:
        try:
            data = lzma.decompress(bytes(stream))
        except lzma.LZMAError as exc:
            raise ArtifactLayoutError(f"{name} xz stream is corrupt: {exc}") from None
    else:
        data = bytes(stream)
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["sha256"] or len(data) != manifest["size"]:
        raise ArtifactLayoutError(f"{name} manifest digest disagrees with its parts")
    return str(manifest["artifact"]), digest


def artifact_binary_sha(binary: dict[str, str], key: str) -> tuple[str, str] | None:
    """(stored key, sha256 of the artifact) for ``key`` in a ConfigMap's binaryData.

    Accepts the compressed form (``<key>.xz``), the raw form (``<key>``) and,
    when the ConfigMap holds exactly one entry under another name, that entry.
    Returns ``None`` when nothing matches.
    """

    for candidate in (key + COMPRESSED_ARTIFACT_SUFFIX, key):
        encoded = binary.get(candidate)
        if encoded is not None:
            return candidate, _decoded_sha(candidate, encoded)
    if len(binary) == 1:
        ((candidate, encoded),) = binary.items()
        return candidate, _decoded_sha(candidate, encoded)
    return None


def _decoded_sha(key: str, encoded: str) -> str:
    data = base64.b64decode(encoded)
    if key.endswith(COMPRESSED_ARTIFACT_SUFFIX):
        data = lzma.decompress(data)
    return hashlib.sha256(data).hexdigest()


# --- operator CLI -----------------------------------------------------------
#
# The deployment manual's manual steps and apply-control-plane-role-split.sh
# used to spell the layout themselves (``xz -9 | kubectl create configmap
# --from-file=<wheel>.xz=...``), which is exactly the single-object assumption
# that failed live. They call this instead, so there is one writer and one
# reader of the layout.


def _kubectl(
    prefix: Sequence[str], *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*prefix, *arguments], text=True, capture_output=True, check=False
    )


def _get_config_map(
    prefix: Sequence[str], namespace: str, name: str
) -> dict[str, Any] | None:
    completed = _kubectl(
        prefix, "-n", namespace, "get", "configmap", name, "-o", "json"
    )
    if completed.returncode:
        if "NotFound" in completed.stderr:
            return None
        raise ArtifactLayoutError(
            f"cannot read configmap/{name}: {completed.stderr.strip()}"
        )
    document = json.loads(completed.stdout or "{}")
    if not isinstance(document, dict):
        raise ArtifactLayoutError(f"configmap/{name} is not an object")
    return document


def _require_config_map(
    prefix: Sequence[str], namespace: str, name: str
) -> dict[str, Any]:
    document = _get_config_map(prefix, namespace, name)
    if document is None:
        raise ArtifactLayoutError(f"configmap/{name} does not exist")
    return document


def _kubectl_prefix(arguments: argparse.Namespace) -> list[str]:
    prefix = ["kubectl"]
    if arguments.kubeconfig:
        prefix.extend(["--kubeconfig", arguments.kubeconfig])
    if arguments.context:
        prefix.extend(["--context", arguments.context])
    return prefix


def upload_artifact_set(
    plan: ArtifactPlan,
    *,
    namespace: str,
    prefix: Sequence[str],
    log: Callable[[str], None],
) -> str:
    """Create the missing objects of ``plan``, parts first, then verify the set.

    An object that already exists is kept when it holds exactly the planned
    bytes and refused otherwise: content-addressed names never change meaning
    under an operator's feet.
    """

    with tempfile.TemporaryDirectory(prefix="gpu-fault-artifact-") as scratch:
        for item in plan.objects:
            existing = _get_config_map(prefix, namespace, item.name)
            if existing is not None:
                if item is plan.primary:
                    break
                if _part_bytes(existing, item.key, item.name) != item.payload:
                    raise ArtifactLayoutError(
                        f"configmap/{item.name} exists with different content; "
                        "delete the stale part set before re-uploading"
                    )
                log(f"configmap/{item.name} already holds {item.key}")
                continue
            source = write_artifact_object(item, Path(scratch))
            completed = _kubectl(
                prefix,
                "-n",
                namespace,
                "create",
                "configmap",
                item.name,
                f"--from-file={item.key}={source}",
            )
            if completed.returncode:
                raise ArtifactLayoutError(
                    f"creating configmap/{item.name} failed: {completed.stderr.strip()}"
                )
            log(f"created configmap/{item.name} ({item.key}, {item.size:,} bytes)")
    found = read_artifact_digest(
        _require_config_map(prefix, namespace, plan.name),
        plan.artifact,
        lambda part: _require_config_map(prefix, namespace, part),
        name=plan.name,
    )
    if found is None or found[1] != plan.sha256:
        raise ArtifactLayoutError(
            f"configmap/{plan.name} does not hold {plan.artifact} with sha256 "
            f"{plan.sha256}"
        )
    return found[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m gpu_fault.admin.artifact_configmaps",
        description=(
            "Store a release artifact in its ConfigMap set the way the release "
            "engine does, or read the artifact digest back from one."
        ),
    )
    parser.add_argument("--kubeconfig", default="")
    parser.add_argument("--context", default="")
    parser.add_argument("--namespace", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("upload", help="create the set if it is missing")
    upload.add_argument("--name", required=True, help="the named (primary) ConfigMap")
    upload.add_argument("--artifact", type=Path, required=True)
    upload.add_argument(
        "--compress",
        action="store_true",
        help="store the xz stream (wheels); omit for a bundle read from its mount",
    )
    upload.add_argument(
        "--splittable",
        action="store_true",
        help="allow a chunked set; refused for an artifact a Pod reads from its mount",
    )
    digest = commands.add_parser("digest", help="print the stored artifact's sha256")
    digest.add_argument("--name", required=True)
    digest.add_argument("--key", default="", help="the artifact filename, if known")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    prefix = _kubectl_prefix(arguments)
    try:
        if arguments.command == "upload":
            plan = plan_artifact_config_maps(
                arguments.artifact,
                name=arguments.name,
                key=arguments.artifact.name,
                compress=arguments.compress,
                splittable=arguments.splittable,
            )
            digest = upload_artifact_set(
                plan,
                namespace=arguments.namespace,
                prefix=prefix,
                log=lambda line: print(line, file=sys.stderr),
            )
            print(
                json.dumps(
                    {
                        "name": plan.name,
                        "artifact": plan.artifact,
                        "sha256": digest,
                        "layout": plan.layout,
                        "payload_bytes": plan.payload_size,
                        "objects": [item.name for item in plan.objects],
                    },
                    sort_keys=True,
                )
            )
        else:
            found = read_artifact_digest(
                _require_config_map(prefix, arguments.namespace, arguments.name),
                arguments.key,
                lambda part: _require_config_map(prefix, arguments.namespace, part),
                name=arguments.name,
            )
            if found is None:
                raise ArtifactLayoutError(
                    f"configmap/{arguments.name} holds no recognisable artifact"
                )
            print(found[1])
    except (ArtifactLayoutError, ArtifactTooLarge, OSError, lzma.LZMAError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
