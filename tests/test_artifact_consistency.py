"""Source, wheel and node bundle must be the same code.

A wheel SHA-256 identifies a file, and manifest annotations record an
intent -- neither proves what a process imported. Both control plane
Deployments have run with
``gpu-fault.io/artifact-sha256=8baa9d7c...`` while the wheel ConfigMap
they mounted held ``60c7deb6b7bc...``, so an audit that trusts the label
reaches the wrong conclusion. ``gpu_fault.module_digest`` is computed
from the package's own files, which makes source, wheel, bundle and the
running process directly comparable, and /v1/version publishes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from gpu_fault import module_digest
from scripts.component_wheels import component_source_digest
from scripts.release_identity import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = ROOT / "src/gpu_fault"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_manifest() -> dict:
    if os.getenv("GPU_FAULT_REQUIRE_BUILD_ARTIFACTS") != "1":
        pytest.skip("build artifact verification is opt-in; run make artifact-check")
    path = ROOT / "dist/current-release.json"
    if not path.is_file():
        pytest.fail("content-addressed release is required; run make artifact-check")
    return json.loads(path.read_text(encoding="utf-8"))


def test_module_digest_reacts_to_a_behaviour_change(tmp_path) -> None:
    copy = tmp_path / "gpu_fault"
    shutil.copytree(SOURCE_PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    baseline = module_digest(copy)
    assert baseline == module_digest(SOURCE_PACKAGE)

    policy_engine = copy / "policy" / "engine.py"
    policy_engine.write_text(
        policy_engine.read_text(encoding="utf-8") + "\n# behaviour change\n",
        encoding="utf-8",
    )
    assert module_digest(copy) != baseline


def test_module_digest_ignores_caches_and_strays(tmp_path) -> None:
    copy = tmp_path / "gpu_fault"
    shutil.copytree(SOURCE_PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    baseline = module_digest(copy)

    (copy / "__pycache__").mkdir(exist_ok=True)
    (copy / "__pycache__" / "policy.cpython-312.pyc").write_bytes(b"\x00compiled")
    (copy / "NOTES.txt").write_text("scratch", encoding="utf-8")
    assert module_digest(copy) == baseline


def test_built_component_wheels_match_their_source_closures(tmp_path) -> None:
    manifest = release_manifest()
    wheels = sorted((ROOT / "dist").rglob("*.whl"))
    assert len(wheels) == 3
    for name in ("control_plane", "executor", "node_runtime"):
        component = manifest["components"][name]
        wheel = ROOT / component["wheel"]
        assert wheel in wheels
        assert wheel.parent.name == manifest["release_id"]
        assert sha256(wheel) == component["wheel_sha256"]
        unpacked = tmp_path / name
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(unpacked)
            file_modes = {
                path.filename: (path.external_attr >> 16) & 0o777
                for path in archive.infolist()
                if not path.is_dir()
            }
            assert all(
                mode == 0o644 or (path.endswith(".dist-info/RECORD") and mode == 0o664)
                for path, mode in file_modes.items()
            ), f"wheel contains caller-umask-dependent modes: {file_modes}"
            metadata_name = next(
                path
                for path in archive.namelist()
                if path.endswith(".dist-info/METADATA")
            )
            metadata = archive.read(metadata_name).decode("utf-8")
            license_names = [
                path
                for path in archive.namelist()
                if path.endswith(".dist-info/licenses/LICENSE")
            ]
            assert "License-Expression: Apache-2.0" in metadata
            assert "License-File: LICENSE" in metadata
            assert len(license_names) == 1
            assert archive.read(license_names[0]) == (ROOT / "LICENSE").read_bytes()
        assert module_digest(unpacked / "gpu_fault") == component["module_digest"]
        assert component["module_digest"] == component_source_digest(name)


def test_node_bundle_contains_the_exact_release_wheel(tmp_path: Path) -> None:
    manifest = release_manifest()
    wheel = ROOT / manifest["components"]["node_runtime"]["wheel"]
    bundles = sorted((ROOT / "dist").rglob("gpu-fault-node-installer-*.tar.gz"))
    assert len(bundles) == 1
    bundle = ROOT / manifest["bundle"]
    assert bundles == [bundle]
    assert bundle.parent == wheel.parent
    assert sha256(bundle) == manifest["bundle_sha256"]

    extracted = tmp_path / "bundle"
    with tarfile.open(bundle, "r:gz") as archive:
        modes = {member.name: member.mode for member in archive.getmembers()}
        assert set(modes.values()).issubset({0o644, 0o755}), (
            f"node bundle contains non-canonical modes: {modes}"
        )
        assert all(
            mode == (0o755 if name.endswith(".sh") else 0o644)
            for name, mode in modes.items()
            if not name.endswith("/")
            and (
                name.endswith(".sh")
                or "/dist/" in name
                or name.endswith((".json", ".service", ".timer", ".csv"))
            )
        ), f"node bundle file modes do not match file roles: {modes}"
        archive.extractall(extracted, filter="data")
    inner_wheels = sorted(extracted.rglob("*.whl"))
    assert len(inner_wheels) == 1
    assert sha256(inner_wheels[0]) == sha256(wheel)

    unpacked = tmp_path / "bundle-wheel"
    with zipfile.ZipFile(inner_wheels[0]) as archive:
        archive.extractall(unpacked)
    assert (
        module_digest(unpacked / "gpu_fault")
        == manifest["components"]["node_runtime"]["module_digest"]
    )


def test_component_wheels_expose_only_their_runtime_surfaces(tmp_path: Path) -> None:
    manifest = release_manifest()
    expected_imports = {
        "control_plane": ("gpu_fault.app", "gpu_fault.admin_cli"),
        "executor": (
            "gpu_fault.cluster_executor",
            "gpu_fault.completion_controller",
            "gpu_fault.node_installer_reconciler",
        ),
        "node_runtime": ("gpu_fault.node_agent.app", "gpu_fault.collectors_cli"),
    }
    forbidden = {
        "control_plane": ("gpu_fault.cluster_executor", "gpu_fault.node_agent.app"),
        "executor": ("gpu_fault.app.factory", "gpu_fault.node_agent.app"),
        "node_runtime": ("gpu_fault.app.factory", "gpu_fault.cluster_executor"),
    }
    for name, imports in expected_imports.items():
        wheel = ROOT / manifest["components"][name]["wheel"]
        installed = tmp_path / f"{name}-installed"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(installed),
                str(wheel),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        command = [
            sys.executable,
            "-c",
            ";".join(f"import {module}" for module in imports),
        ]
        subprocess.run(
            command, env={**os.environ, "PYTHONPATH": str(installed)}, check=True
        )
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
        for module in forbidden[name]:
            path = module.replace(".", "/")
            assert f"{path}.py" not in names
            assert f"{path}/__init__.py" not in names


def test_release_manifest_is_complete_and_no_sdist_exists() -> None:
    manifest = release_manifest()
    release_dir = ROOT / "dist" / manifest["release_id"]

    assert release_dir.is_dir()
    assert not (ROOT / "build").exists()
    assert (
        json.loads((release_dir / "release.json").read_text(encoding="utf-8"))
        == manifest
    )
    assert manifest["module_digest"] == component_source_digest("control_plane")
    assert manifest["schema_version"] == 3
    delivery = dict(manifest["delivery"])
    delivery_sha256 = delivery.pop("sha256")
    assert canonical_sha256(delivery) == delivery_sha256
    assert manifest["delivery"]["rendered_manifests"]["file_count"] >= 3
    assert (
        manifest["components"]["node_bundle"]["template_sha256"]
        == (manifest["delivery"]["node_template_inputs"]["sha256"])
    )
    hashes = {
        "control_plane": manifest["components"]["control_plane"]["wheel_sha256"],
        "executor": manifest["components"]["executor"]["wheel_sha256"],
        "node_runtime": manifest["components"]["node_runtime"]["wheel_sha256"],
        "node_bundle": manifest["bundle_sha256"],
    }
    assert (
        manifest["release_id"]
        == hashlib.sha256(
            json.dumps(
                {"artifacts": hashes, "delivery": delivery_sha256},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:12]
    )
    assert all(
        "@sha256:" in item["reference"]
        for item in manifest["delivery"]["images"].values()
    ), "release delivery contains a mutable image reference"
    if manifest["deployable"]:
        runtime_components = manifest["delivery"]["images"]["runtime"]["components"]
        for name in ("control_plane", "executor"):
            assert (
                runtime_components[name]["wheel_sha256"]
                == (manifest["components"][name]["wheel_sha256"])
            )
            assert (
                runtime_components[name]["module_digest"]
                == (manifest["components"][name]["module_digest"])
            )
    assert list((ROOT / "dist").glob("*.whl")) == []
    assert list((ROOT / "dist").glob("*.tar.gz")) == []
    assert not [
        path
        for path in (ROOT / "dist").rglob("*.tar.gz")
        if not path.name.startswith("gpu-fault-node-installer-")
    ]
    assert sorted(path.name for path in (ROOT / "dist").iterdir() if path.is_dir()) == [
        manifest["release_id"]
    ]
