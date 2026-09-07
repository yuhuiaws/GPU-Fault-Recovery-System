"""Wheel ConfigMaps hold the artifact xz-compressed; digests are of the wheel."""

from __future__ import annotations

import base64
import hashlib
import lzma
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_artifacts as ARTIFACTS
from gpu_fault_release.regional_release_config import ReleaseError


def _wheel(tmp_path: Path) -> tuple[Path, str]:
    wheel = tmp_path / "gpu_fault_control_plane-0.10.0-py3-none-any.whl"
    wheel.write_bytes(b"PK\x03\x04" + b"wheel bytes " * 4096)
    return wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_artifact_binary_sha_reads_compressed_raw_and_single_entries() -> None:
    payload = b"artifact"
    sha = hashlib.sha256(payload).hexdigest()
    compressed = base64.b64encode(lzma.compress(payload)).decode()
    raw = base64.b64encode(payload).decode()

    assert ARTIFACTS.artifact_binary_sha({"a.whl.xz": compressed}, "a.whl") == (
        "a.whl.xz",
        sha,
    )
    assert ARTIFACTS.artifact_binary_sha({"a.whl": raw}, "a.whl") == ("a.whl", sha)
    assert ARTIFACTS.artifact_binary_sha({"other.whl.xz": compressed}, "a.whl") == (
        "other.whl.xz",
        sha,
    )
    assert ARTIFACTS.artifact_binary_sha({}, "a.whl") is None, "empty binaryData"
    assert ARTIFACTS.artifact_binary_sha({"x": raw, "y": raw}, "a.whl") is None, (
        "two unrelated entries must not be guessed"
    )


def test_upload_config_map_compresses_wheels_and_verifies_the_wheel_digest(
    tmp_path: Path,
) -> None:
    """The control-plane wheel outgrew the 1 MiB ConfigMap ceiling live on
    2026-09-07 (1,106,539 bytes); xz keeps the same ConfigMap name and a
    digest that is still the wheel's own sha256."""
    wheel, sha = _wheel(tmp_path)
    created: list[list[str]] = []
    store: dict[str, dict] = {}

    def run(command: list[str]) -> None:
        created.append(command)
        key, source = command[-1].removeprefix("--from-file=").split("=", 1)
        store["cm"] = {
            "binaryData": {key: base64.b64encode(Path(source).read_bytes()).decode()}
        }

    release = SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system"),
        runner=SimpleNamespace(
            probe_output=lambda _command: (1, "", 'configmaps "cm" not found NotFound'),
            run=run,
            dry_run=False,
        ),
        _get_json=lambda _command: store["cm"],
    )

    ARTIFACTS.upload_config_map(
        release, ["kubectl"], "cm", wheel.name, wheel, sha, compress=True
    )

    assert created[0][-1].startswith(f"--from-file={wheel.name}.xz="), created[0]
    (stored_key,) = store["cm"]["binaryData"].keys()
    assert stored_key == f"{wheel.name}.xz", stored_key
    stored = base64.b64decode(store["cm"]["binaryData"][stored_key])
    assert len(stored) < wheel.stat().st_size, (len(stored), wheel.stat().st_size)
    assert lzma.decompress(stored) == wheel.read_bytes(), "xz round trip"

    with pytest.raises(ReleaseError, match="digest mismatch"):
        ARTIFACTS.upload_config_map(
            release, ["kubectl"], "cm", wheel.name, wheel, "0" * 64, compress=True
        )


def test_upload_config_map_keeps_bundles_raw(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"\x1f\x8b" + b"bundle" * 100)
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    created: list[list[str]] = []
    release = SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system"),
        runner=SimpleNamespace(
            probe_output=lambda _command: (1, "", "NotFound"),
            run=lambda command: created.append(command),
            dry_run=False,
        ),
        _get_json=lambda _command: {
            "binaryData": {bundle.name: base64.b64encode(bundle.read_bytes()).decode()}
        },
    )

    ARTIFACTS.upload_config_map(release, ["kubectl"], "cm", bundle.name, bundle, sha)

    assert created[0][-1] == f"--from-file={bundle.name}={bundle}", created[0]
