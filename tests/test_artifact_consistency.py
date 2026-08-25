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
import tarfile
import zipfile
from pathlib import Path

import pytest

from gpu_fault import module_digest

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


def test_built_wheel_matches_the_source_tree(tmp_path) -> None:
    """The wheel in dist/ must be built from this checkout.

    This is the gate that was missing: ``deploy.sh`` rebuilds the wheel
    on every deploy, but nothing ever asserted that a wheel already
    sitting in dist/ -- the one a manual roll uploads into the wheel
    ConfigMap -- came from the current source.
    """

    manifest = release_manifest()
    wheels = sorted((ROOT / "dist").rglob("*.whl"))
    assert len(wheels) == 1
    wheel = ROOT / manifest["wheel"]
    assert wheels == [wheel]
    assert wheel.parent.name == manifest["release_id"]
    assert sha256(wheel) == manifest["wheel_sha256"]

    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(unpacked)

    assert module_digest(unpacked / "gpu_fault") == module_digest(SOURCE_PACKAGE), (
        f"{wheel.name} was not built from this source tree; rebuild it "
        "before deploying or the fleet runs code nobody reviewed"
    )


def test_node_bundle_contains_the_exact_release_wheel(tmp_path: Path) -> None:
    manifest = release_manifest()
    wheel = ROOT / manifest["wheel"]
    bundles = sorted((ROOT / "dist").rglob("gpu-fault-node-installer-*.tar.gz"))
    assert len(bundles) == 1
    bundle = ROOT / manifest["bundle"]
    assert bundles == [bundle]
    assert bundle.parent == wheel.parent
    assert sha256(bundle) == manifest["bundle_sha256"]

    extracted = tmp_path / "bundle"
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    inner_wheels = sorted(extracted.rglob("*.whl"))
    assert len(inner_wheels) == 1
    assert sha256(inner_wheels[0]) == sha256(wheel)

    unpacked = tmp_path / "bundle-wheel"
    with zipfile.ZipFile(inner_wheels[0]) as archive:
        archive.extractall(unpacked)
    assert module_digest(unpacked / "gpu_fault") == module_digest(SOURCE_PACKAGE)


def test_release_manifest_is_complete_and_no_sdist_exists() -> None:
    manifest = release_manifest()
    release_dir = ROOT / "dist" / manifest["release_id"]

    assert release_dir.is_dir()
    assert not (ROOT / "build").exists()
    assert (
        json.loads((release_dir / "release.json").read_text(encoding="utf-8"))
        == manifest
    )
    assert manifest["module_digest"] == module_digest(SOURCE_PACKAGE)
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
