from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from scripts import component_artifact_cache, component_artifacts


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, bytes]]:
    root = tmp_path / "repo"
    release_dir = root / "dist/release-a"
    release_dir.mkdir(parents=True)
    wheel_bytes = {
        "control_plane": b"control-wheel",
        "executor": b"executor-wheel",
        "node_runtime": b"node-wheel",
    }
    wheel_paths = {}
    for name, content in wheel_bytes.items():
        path = release_dir / f"{name}.whl"
        path.write_bytes(content)
        wheel_paths[name] = path
    bundle = release_dir / "node-bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        member = tarfile.TarInfo("gpu-fault-node-installer/dist/node_runtime.whl")
        member.size = len(wheel_bytes["node_runtime"])
        archive.addfile(member, io.BytesIO(wheel_bytes["node_runtime"]))
    module_digests = {
        "control_plane": "a" * 64,
        "executor": "b" * 64,
        "node_runtime": "c" * 64,
    }
    manifest = {
        "schema_version": 3,
        "component_build_identity_sha256": "f" * 64,
        "deployable": False,
        "staging_only": False,
        "release_id": "release-a",
        "delivery": {"runtime_prebuilt": False, "sha256": "d" * 64},
        "bundle": "dist/release-a/node-bundle.tar.gz",
        "bundle_sha256": _sha256(bundle.read_bytes()),
        "components": {
            name: {
                "wheel": f"dist/release-a/{path.name}",
                "wheel_sha256": _sha256(wheel_bytes[name]),
                "module_digest": module_digests[name],
                "module_count": index,
            }
            for index, (name, path) in enumerate(wheel_paths.items(), 1)
        },
    }
    content = json.dumps(manifest, sort_keys=True).encode()
    immutable = release_dir / "release.json"
    current = root / "dist/current-release.json"
    immutable.write_bytes(content)
    current.write_bytes(content)
    monkeypatch.setattr(
        component_artifacts,
        "build_release_identity",
        lambda _root: {"sha256": "d" * 64},
    )
    monkeypatch.setattr(
        component_artifacts,
        "component_source_digest",
        lambda name: module_digests[name],
    )
    monkeypatch.setattr(
        component_artifacts, "component_build_identity", lambda _root: "f" * 64
    )
    monkeypatch.setattr(
        component_artifact_cache, "component_build_identity", lambda _root: "f" * 64
    )
    return current, wheel_bytes


def test_component_artifacts_validate_source_and_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, wheel_bytes = _artifact_tree(tmp_path, monkeypatch)

    artifacts = component_artifacts.load_component_artifacts(
        tmp_path / "repo", manifest
    )

    assert artifacts.module_counts == {
        "control_plane": 1,
        "executor": 2,
        "node_runtime": 3,
    }
    assert artifacts.wheels["executor"].read_bytes() == wheel_bytes["executor"]


def test_component_artifacts_reject_tampered_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _wheel_bytes = _artifact_tree(tmp_path, monkeypatch)
    (tmp_path / "repo/dist/release-a/executor.whl").write_bytes(b"tampered")

    with pytest.raises(
        component_artifacts.ComponentArtifactError, match="executor wheel SHA-256"
    ):
        component_artifacts.load_component_artifacts(tmp_path / "repo", manifest)


def test_component_artifact_cache_restores_physical_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _wheel_bytes = _artifact_tree(tmp_path, monkeypatch)
    root = tmp_path / "repo"
    cache_root = tmp_path / "cache"

    cached = component_artifact_cache.store_component_artifacts(
        root, cache_root, manifest
    )
    shutil.rmtree(root / "dist")

    restored = component_artifact_cache.find_cached_manifest(root, cache_root)

    assert restored == cached
    artifacts = component_artifacts.load_component_artifacts(
        root,
        restored,
        artifact_root=restored.parent.parent,
        require_delivery_identity=False,
    )
    assert artifacts.wheels["control_plane"].read_bytes() == b"control-wheel"


def test_component_build_identity_covers_node_bundle_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for relative in (
        "LICENSE",
        "pyproject.toml",
        "requirements/build.lock",
        "scripts/component_artifacts.py",
        "scripts/component_wheels.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    node_identity = {"sha256": "a" * 64}
    monkeypatch.setattr(
        component_artifacts,
        "build_release_identity",
        lambda _root: {"node_template_inputs": dict(node_identity)},
    )
    monkeypatch.setattr(
        component_artifacts,
        "component_source_digest",
        lambda name: _sha256(name.encode()),
    )

    first = component_artifacts.component_build_identity(tmp_path)
    node_identity["sha256"] = "b" * 64
    second = component_artifacts.component_build_identity(tmp_path)

    assert first != second
