from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.release_identity import (
    ReleaseIdentityError,
    bind_runtime_image,
    build_release_identity,
    sha256_bytes,
)


def identity_root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "manifests").mkdir()
    (tmp_path / "renderers").mkdir()
    (tmp_path / "rendered").mkdir()
    (tmp_path / "node").mkdir()
    (tmp_path / "deploy/image").mkdir(parents=True)
    (tmp_path / "requirements").mkdir()
    (tmp_path / "manifests/cpu.yaml").write_text("kind: Deployment\n")
    (tmp_path / "manifests/gpu.yaml").write_text("kind: DaemonSet\n")
    (tmp_path / "renderers/render.py").write_text("print('render')\n")
    (tmp_path / "rendered/cpu.yaml").write_text("kind: Deployment\n")
    (tmp_path / "node/install.sh").write_text("#!/bin/sh\n")
    (tmp_path / "deploy/image/Dockerfile").write_text("FROM python\n")
    (tmp_path / "requirements/runtime.lock").write_text("fastapi==1\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    images = {
        "schema_version": 1,
        "images": {
            name: {
                "source": f"registry.example/{name}:tag",
                "reference": f"registry.example/{name}@sha256:" + char * 64,
            }
            for name, char in {
                "runtime": "1",
                "node_installer": "2",
                "dcgm_exporter": "3",
                "adot": "4",
            }.items()
        },
    }
    (tmp_path / "config/images.json").write_text(json.dumps(images))
    config = {
        "schema_version": 1,
        "schema_rollback_compatible": True,
        "dependency_lock": "uv.lock",
        "images_lock": "config/images.json",
        "manifest_inputs": ["manifests/*.yaml"],
        "renderer_inputs": ["renderers/*.py"],
        "rendered_manifest_inputs": ["rendered/*.yaml"],
        "runtime_image_inputs": ["renderers/*.py", "manifests/*.yaml"],
        "node_template_inputs": ["node/*.sh"],
        "component_inputs": {
            "collector": ["manifests/gpu.yaml"],
            "cpu": ["manifests/cpu.yaml"],
            "dcgm": ["manifests/gpu.yaml"],
            "endpoint": ["manifests/cpu.yaml"],
            "executor": ["manifests/gpu.yaml"],
            "node": ["node/*.sh"],
            "observability": ["manifests/cpu.yaml"],
            "schema": ["manifests/gpu.yaml"],
            "watcher": ["manifests/gpu.yaml"],
        },
    }
    (tmp_path / "config/release-identity.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    return tmp_path


def test_release_identity_changes_with_component_inputs(tmp_path: Path) -> None:
    root = identity_root(tmp_path)
    before = build_release_identity(root)

    path = root / "manifests/cpu.yaml"
    path.write_text(path.read_text() + "metadata: {}\n")
    after = build_release_identity(root)

    assert before["sha256"] != after["sha256"]
    assert before["components"]["cpu"]["sha256"] != after["components"]["cpu"]["sha256"]
    assert (
        before["components"]["executor"]["sha256"]
        == after["components"]["executor"]["sha256"]
    )
    assert (
        before["runtime_image_inputs"]["sha256"]
        != after["runtime_image_inputs"]["sha256"]
    )


def test_release_identity_includes_file_modes(tmp_path: Path) -> None:
    root = identity_root(tmp_path)
    path = root / "node/install.sh"
    path.chmod(0o644)
    before = build_release_identity(root)
    path.chmod(0o755)
    after = build_release_identity(root)

    assert (
        before["node_template_inputs"]["sha256"]
        != after["node_template_inputs"]["sha256"]
    )


def test_release_identity_rejects_mutable_images(tmp_path: Path) -> None:
    root = identity_root(tmp_path)
    path = root / "config/images.json"
    value = json.loads(path.read_text())
    value["images"]["runtime"]["reference"] = "registry.example/runtime:latest"
    path.write_text(json.dumps(value))

    with pytest.raises(ReleaseIdentityError, match="immutable"):
        build_release_identity(root)


def test_runtime_image_descriptor_binds_deployable_identity(tmp_path: Path) -> None:
    root = identity_root(tmp_path)
    identity = build_release_identity(root)
    descriptor = {
        "schema_version": 2,
        "deployable": True,
        "repository": "registry.example/gpu-fault-runtime",
        "reference": "registry.example/gpu-fault-runtime@sha256:" + "a" * 64,
        "source_identity_sha256": identity["sha256"],
        "dockerfile_sha256": sha256_bytes(
            (root / "deploy/image/Dockerfile").read_bytes()
        ),
        "dependency_lock_sha256": sha256_bytes(
            (root / "requirements/runtime.lock").read_bytes()
        ),
        "components": {
            "control_plane": {
                "distribution": "gpu-fault-control-plane",
                "wheel_sha256": "b" * 64,
                "module_digest": "c" * 64,
            },
            "executor": {
                "distribution": "gpu-fault-cluster-executor",
                "wheel_sha256": "d" * 64,
                "module_digest": "e" * 64,
            },
        },
    }

    bound = bind_runtime_image(root, identity, descriptor)

    assert bound["runtime_prebuilt"] is True
    assert bound["images"]["runtime"]["reference"] == descriptor["reference"]
    assert bound["images"]["runtime"]["components"] == descriptor["components"]
    assert bound["sha256"] != identity["sha256"]


def test_runtime_image_descriptor_rejects_local_only_build(tmp_path: Path) -> None:
    root = identity_root(tmp_path)
    identity = build_release_identity(root)
    descriptor = {
        "schema_version": 2,
        "deployable": False,
        "repository": "registry.example/gpu-fault-runtime",
        "reference": "registry.example/gpu-fault-runtime@sha256:" + "a" * 64,
        "source_identity_sha256": identity["sha256"],
        "dockerfile_sha256": sha256_bytes(
            (root / "deploy/image/Dockerfile").read_bytes()
        ),
        "dependency_lock_sha256": sha256_bytes(
            (root / "requirements/runtime.lock").read_bytes()
        ),
        "components": {
            "control_plane": {
                "distribution": "gpu-fault-control-plane",
                "wheel_sha256": "b" * 64,
                "module_digest": "c" * 64,
            },
            "executor": {
                "distribution": "gpu-fault-cluster-executor",
                "wheel_sha256": "d" * 64,
                "module_digest": "e" * 64,
            },
        },
    }

    with pytest.raises(ReleaseIdentityError, match="local-only"):
        bind_runtime_image(root, identity, descriptor)


def test_runtime_image_descriptor_requires_component_identities(tmp_path: Path) -> None:
    root = identity_root(tmp_path)
    identity = build_release_identity(root)
    descriptor = {
        "schema_version": 2,
        "deployable": True,
        "repository": "registry.example/gpu-fault-runtime",
        "reference": "registry.example/gpu-fault-runtime@sha256:" + "a" * 64,
        "source_identity_sha256": identity["sha256"],
        "dockerfile_sha256": sha256_bytes(
            (root / "deploy/image/Dockerfile").read_bytes()
        ),
        "dependency_lock_sha256": sha256_bytes(
            (root / "requirements/runtime.lock").read_bytes()
        ),
        "components": {},
    }

    with pytest.raises(ReleaseIdentityError, match="component identities"):
        bind_runtime_image(root, identity, descriptor)
