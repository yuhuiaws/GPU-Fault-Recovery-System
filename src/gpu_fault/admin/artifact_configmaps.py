"""How release artifacts are stored in Kubernetes ConfigMaps.

A ConfigMap tops out at 1 MiB and the control-plane wheel outgrew it on
2026-09-07 (1,106,539 bytes). Nothing installs a wheel from the ConfigMap at
runtime -- the runtime image already carries it -- so the ConfigMap only has to
keep the artifact recoverable and digest-checkable. Wheels are therefore stored
xz-compressed under ``<wheel filename>.xz``; a wheel is a zip with per-file
deflate, and xz over the whole file still takes ~40% off. Node installer
bundles stay raw: the installer Job reads them straight from the mount.

The digest recorded next to a ConfigMap (``gpu-fault.io/artifact-sha256`` and
the release manifest) is always the wheel's own sha256, whichever form the
ConfigMap holds.
"""

from __future__ import annotations

import base64
import hashlib
import lzma
from pathlib import Path

COMPRESSED_ARTIFACT_SUFFIX = ".xz"


def compress_artifact(path: Path, destination: Path) -> Path:
    """Write ``path`` xz-compressed to ``destination`` and return it."""

    destination.write_bytes(
        lzma.compress(path.read_bytes(), preset=9 | lzma.PRESET_EXTREME)
    )
    return destination


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
