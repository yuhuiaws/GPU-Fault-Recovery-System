"""The artifact ConfigMap reader and CLI on the sets and API answers that go wrong.

``test_artifact_configmaps`` covers the three layouts, their re-hashing and the
CLI's happy paths. The manual steps and ``apply-control-plane-role-split.sh``
call this module instead of spelling the layout themselves, so every way a set
or the API can disagree with the plan has to come back as one ``error:`` line
and exit status 1: a ConfigMap kubectl cannot read or that is not an object, a
manifest that is not text or not an object, a part that is not base64, parts
that do not reassemble, an xz claim over a stream that is not xz, a stale part
or a foreign object under a content-addressed name, a create the API refuses.
"""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import os
import runpy
import stat
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import artifact_configmaps as module
from gpu_fault.admin.artifact_configmaps import (
    MANIFEST_KEY,
    ArtifactLayoutError,
    artifact_manifest,
    artifact_part_names,
    compress_artifact,
    plan_artifact_config_maps,
    read_artifact_digest,
)

FAKE_KUBECTL = """#!PYTHON_EXECUTABLE
import base64, json, os, sys
from pathlib import Path
state = Path(os.environ["FAKE_STATE"])
args = sys.argv[1:]
with (state / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps(args) + "\\n")
name = args[args.index("configmap") + 1]
if "get" in args:
    for suffix, stream, code in ((".error", sys.stderr, 1), (".raw", sys.stdout, 0)):
        answer = state / (name + suffix)
        if answer.exists():
            stream.write(answer.read_text())
            raise SystemExit(code)
    target = state / (name + ".json")
    if not target.exists():
        print('Error from server (NotFound): configmaps "%s" not found' % name,
              file=sys.stderr)
        raise SystemExit(1)
    print(target.read_text())
elif "create" in args:
    refusal = state / "create.error"
    if refusal.exists():
        sys.stderr.write(refusal.read_text())
        raise SystemExit(1)
    field = next(item for item in args if item.startswith("--from-file="))
    key, path = field[len("--from-file="):].split("=", 1)
    payload = Path(path).read_bytes()
    document = {"kind": "ConfigMap", "metadata": {"name": name}}
    try:
        document["data"] = {key: payload.decode("utf-8")}
    except UnicodeDecodeError:
        document["binaryData"] = {key: base64.b64encode(payload).decode()}
    (state / (name + ".json")).write_text(json.dumps(document))
else:
    raise SystemExit("unexpected kubectl use: " + " ".join(args))
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A kubectl on PATH whose answers are files in the returned directory."""

    state = tmp_path / "api"
    state.mkdir()
    binary = tmp_path / "bin"
    binary.mkdir()
    kubectl = binary / "kubectl"
    kubectl.write_text(
        FAKE_KUBECTL.replace("PYTHON_EXECUTABLE", sys.executable), encoding="utf-8"
    )
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_STATE", str(state))
    return state


def calls(state: Path) -> list[list[str]]:
    journal = state / "calls.jsonl"
    if not journal.exists():
        return []
    return [json.loads(line) for line in journal.read_text().splitlines()]


def stored(state: Path, name: str, binary: dict[str, bytes]) -> None:
    document = {
        "kind": "ConfigMap",
        "metadata": {"name": name},
        "binaryData": {
            key: base64.b64encode(value).decode() for key, value in binary.items()
        },
    }
    (state / f"{name}.json").write_text(json.dumps(document))


def chunked_set(
    chunks: list[bytes], *, encoding: str, artifact: bytes, payload_sha256: str = ""
) -> dict[str, dict[str, Any]]:
    """The documents of a chunked set named ``cm`` whose parts are ``chunks``."""

    stream = b"".join(chunks)
    parts = [
        {
            "config_map": f"cm-part-{index}",
            "key": f"a.whl.xz.{index:02d}",
            "size": len(chunk),
            "sha256": sha256(chunk),
        }
        for index, chunk in enumerate(chunks)
    ]
    manifest = {
        "schema_version": 1,
        "artifact": "a.whl",
        "sha256": sha256(artifact),
        "size": len(artifact),
        "encoding": encoding,
        "payload_sha256": payload_sha256 or sha256(stream),
        "payload_size": len(stream),
        "parts": parts,
    }
    documents: dict[str, dict[str, Any]] = {
        "cm": {"data": {MANIFEST_KEY: json.dumps(manifest)}}
    }
    for part, chunk in zip(parts, chunks, strict=True):
        documents[part["config_map"]] = {
            "binaryData": {part["key"]: base64.b64encode(chunk).decode()}
        }
    return documents


def test_compress_artifact_writes_the_xz_stream_the_readers_decode(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "a.whl"
    wheel.write_bytes(b"PK\x03\x04" + b"wheel" * 500)

    written = compress_artifact(wheel, tmp_path / "a.whl.xz")

    assert written == tmp_path / "a.whl.xz", written
    assert lzma.decompress(written.read_bytes()) == wheel.read_bytes()
    assert written.stat().st_size < wheel.stat().st_size, "the stream is compressed"


@pytest.mark.parametrize("key", ["", "wheels/a.whl", ".a.whl"])
def test_a_key_kubectl_cannot_file_is_refused_before_anything_is_read(
    tmp_path: Path, key: str
) -> None:
    missing = tmp_path / "never-read.whl"
    with pytest.raises(ArtifactLayoutError, match="bundle ConfigMap key is invalid"):
        plan_artifact_config_maps(
            missing, name="cm", key=key, compress=False, splittable=True, label="bundle"
        )


@pytest.mark.parametrize(
    ("raw", "problem"),
    [(5, "artifact.json is not text"), ("[1, 2]", "artifact.json is not an object")],
    ids=["number", "list"],
)
def test_a_manifest_that_is_not_a_json_object_is_refused(
    raw: Any, problem: str
) -> None:
    with pytest.raises(ArtifactLayoutError, match=problem):
        artifact_manifest({"data": {MANIFEST_KEY: raw}})


def test_a_single_object_layout_names_no_parts() -> None:
    document = {"binaryData": {"a.whl.xz": base64.b64encode(b"xz").decode()}}

    assert artifact_part_names(document) == (), "a single object has no part set"
    assert artifact_part_names({}) == (), "an empty ConfigMap has no part set"


def test_binary_data_that_is_not_text_is_refused_without_fetching() -> None:
    fetched: list[str] = []

    with pytest.raises(ArtifactLayoutError, match="ConfigMap binaryData is invalid"):
        read_artifact_digest({"binaryData": {"a.whl": 5}}, "a.whl", fetched.append)
    assert fetched == [], "a single-object layout never reads a part"


def test_a_part_that_is_not_base64_is_refused() -> None:
    artifact = b"wheel bytes"
    documents = chunked_set([artifact], encoding="none", artifact=artifact)
    documents["cm-part-0"]["binaryData"]["a.whl.xz.00"] = "%%% not base64 %%%"

    with pytest.raises(
        ArtifactLayoutError, match="cm-part-0/a.whl.xz.00 is not base64"
    ):
        read_artifact_digest(documents["cm"], "a.whl", documents.__getitem__, name="cm")


def test_parts_that_each_match_but_do_not_reassemble_are_refused() -> None:
    chunks = [b"first half ", b"second half"]
    documents = chunked_set(
        chunks, encoding="none", artifact=b"".join(chunks), payload_sha256="f" * 64
    )

    with pytest.raises(ArtifactLayoutError, match="cm parts do not reassemble"):
        read_artifact_digest(documents["cm"], "a.whl", documents.__getitem__, name="cm")


def test_an_xz_claim_over_a_stream_that_is_not_xz_is_refused() -> None:
    chunks = [b"definitely not ", b"an xz stream"]
    documents = chunked_set(chunks, encoding="xz", artifact=b"the wheel")

    with pytest.raises(ArtifactLayoutError, match="cm xz stream is corrupt"):
        read_artifact_digest(documents["cm"], "a.whl", documents.__getitem__, name="cm")


@pytest.mark.allows_cluster_binaries("kubectl")
def test_digest_reports_what_it_cannot_read_parse_or_recognise(
    api: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (api / "cm-a.error").write_text("Unable to connect to the server: i/o timeout\n")
    (api / "cm-b.raw").write_text("[]\n")
    stored(api, "cm-c", {"other.tar.gz": b"one", "notes.txt": b"two"})
    common = ["--context", "unit-context", "--namespace", "gpu-fault-system"]

    errors = {}
    for name in ("cm-a", "cm-b", "cm-c"):
        assert module.main([*common, "digest", "--name", name]) == 1, name
        captured = capsys.readouterr()
        assert captured.out == "", captured.out
        errors[name] = captured.err.strip()

    assert errors == {
        "cm-a": "error: cannot read configmap/cm-a: "
        "Unable to connect to the server: i/o timeout",
        "cm-b": "error: configmap/cm-b is not an object",
        "cm-c": "error: configmap/cm-c holds no recognisable artifact",
    }, errors
    assert [call[:6] for call in calls(api)] == [
        ["--context", "unit-context", "-n", "gpu-fault-system", "get", "configmap"]
    ] * 3, "every read carries the operator's context and namespace"


@pytest.mark.allows_cluster_binaries("kubectl")
def test_upload_refuses_a_stale_part_under_the_set_name(
    tmp_path: Path,
    api: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(module, "CONFIG_MAP_PAYLOAD_LIMIT", 4096)
    monkeypatch.setattr(module, "ARTIFACT_PART_BYTES", 1024)
    wheel = tmp_path / "a.whl"
    wheel.write_bytes(os.urandom(4096 + 100))
    name = "wheel-" + sha256(wheel.read_bytes())[:12]
    stored(api, f"{name}-part-0", {"a.whl.00": b"a part of another upload"})

    status = module.main(
        [
            *("--namespace", "gpu-fault-system", "upload", "--name", name),
            *("--artifact", str(wheel), "--splittable"),
        ]
    )

    assert status == 1
    assert capsys.readouterr().err.strip() == (
        f"error: configmap/{name}-part-0 exists with different content; delete "
        "the stale part set before re-uploading"
    )
    assert not any("create" in call for call in calls(api)), (
        "nothing is created next to a stale part"
    )


@pytest.mark.allows_cluster_binaries("kubectl")
def test_upload_reports_a_create_the_api_refuses(
    tmp_path: Path, api: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"\x1f\x8b" + b"bundle" * 64)
    (api / "create.error").write_text(
        'Error from server (Forbidden): configmaps is forbidden: User "ops" cannot '
        'create resource "configmaps"\n'
    )

    status = module.main(
        [
            *("--namespace", "gpu-fault-system", "upload", "--name", "bundle-a"),
            *("--artifact", str(bundle)),
        ]
    )

    assert status == 1
    error = capsys.readouterr().err.strip()
    assert error.startswith(
        "error: creating configmap/bundle-a failed: Error from server (Forbidden)"
    ), error
    assert not (api / "bundle-a.json").exists(), "nothing was stored"


@pytest.mark.allows_cluster_binaries("kubectl")
def test_upload_never_trusts_a_foreign_object_under_the_content_addressed_name(
    tmp_path: Path, api: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"\x1f\x8b" + b"bundle" * 64)
    digest = sha256(bundle.read_bytes())
    stored(api, "bundle-a", {"other.tar.gz": b"one", "notes.txt": b"two"})

    status = module.main(
        [
            *("--namespace", "gpu-fault-system", "upload", "--name", "bundle-a"),
            *("--artifact", str(bundle)),
        ]
    )

    assert status == 1
    assert capsys.readouterr().err.strip() == (
        f"error: configmap/bundle-a does not hold bundle.tar.gz with sha256 {digest}"
    )
    assert not any("create" in call for call in calls(api)), (
        "an existing primary is verified, never overwritten"
    )


@pytest.mark.allows_cluster_binaries("kubectl")
def test_the_module_entry_point_exits_with_the_status_main_returns(
    api: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``python3 -m gpu_fault.admin.artifact_configmaps``, as the manual runs it."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["artifact_configmaps", "--namespace", "ns", "digest", "--name", "cm-z"],
    )
    with warnings.catch_warnings():
        # The module is already imported by this test file; runpy says so.
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as exited:
            runpy.run_module("gpu_fault.admin.artifact_configmaps", run_name="__main__")

    assert exited.value.code == 1, exited.value
    assert capsys.readouterr().err.strip() == "error: configmap/cm-z does not exist"
    assert [call[-4:] for call in calls(api)] == [["configmap", "cm-z", "-o", "json"]]
