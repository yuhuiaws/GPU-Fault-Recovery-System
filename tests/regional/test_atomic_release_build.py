from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
BUILD = lazy_script_module(
    "atomic_release_build", ROOT / "scripts/build-release-artifacts.py"
)


def test_release_builder_resolves_python_from_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "bin/python3"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent))

    assert BUILD.resolve_python_executable("python3") == str(executable.absolute())
    with pytest.raises(RuntimeError, match="was not found"):
        BUILD.resolve_python_executable("missing-python")


def test_failed_release_build_preserves_current_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    current = dist / "current-release.json"
    current.write_text('{"release_id":"stable"}\n', encoding="utf-8")
    globals_ = BUILD.build.__globals__
    monkeypatch.setitem(globals_, "DIST", dist)
    monkeypatch.setitem(globals_, "BUILD", tmp_path / "build")

    def fail_component(**_kwargs):
        raise RuntimeError("component build failed")

    monkeypatch.setitem(globals_, "build_component", fail_component)

    with pytest.raises(RuntimeError, match="component build failed"):
        BUILD.build(sys.executable)

    assert current.read_text(encoding="utf-8") == '{"release_id":"stable"}\n'
    assert not list(dist.glob(".build-*")), "failed staging directory was retained"


def test_release_builder_never_deletes_published_dist() -> None:
    source = (ROOT / "scripts/build-release-artifacts.py").read_text(encoding="utf-8")

    assert "shutil.rmtree(DIST" not in source
    assert 'os.replace(staged_current, DIST / "current-release.json")' in source


def test_release_pruning_preserves_signed_ci_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist = tmp_path / "dist"
    release = dist / "release-current"
    domains = dist / "ci-domains/unit"
    stale = dist / "release-stale"
    for path in (release, domains, stale):
        path.mkdir(parents=True, exist_ok=True)
    globals_ = BUILD.prune_dist.__globals__
    monkeypatch.setitem(globals_, "DIST", dist)

    BUILD.prune_dist(release)

    assert release.is_dir(), "current content-addressed release was pruned"
    assert domains.is_dir(), "signed CI domain evidence was pruned"
    assert not stale.exists(), "stale content-addressed release was retained"


def test_runtime_component_identity_must_match_release_artifacts() -> None:
    descriptor = {
        "components": {
            "control_plane": {"wheel_sha256": "a" * 64, "module_digest": "b" * 64},
            "executor": {"wheel_sha256": "c" * 64, "module_digest": "d" * 64},
        }
    }
    BUILD.validate_runtime_components(
        descriptor,
        hashes={"control_plane": "a" * 64, "executor": "c" * 64},
        module_digests={"control_plane": "b" * 64, "executor": "d" * 64},
    )

    descriptor["components"]["executor"]["module_digest"] = "e" * 64
    with pytest.raises(RuntimeError, match="executor module digest"):
        BUILD.validate_runtime_components(
            descriptor,
            hashes={"control_plane": "a" * 64, "executor": "c" * 64},
            module_digests={"control_plane": "b" * 64, "executor": "d" * 64},
        )


def test_release_builder_reuses_validated_component_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheels = {}
    for name in ("control_plane", "executor", "node_runtime"):
        path = tmp_path / f"{name}.whl"
        path.write_bytes(name.encode())
        wheels[name] = path
    bundle = tmp_path / "node-bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    cached = SimpleNamespace(
        manifest={"staging_only": False},
        wheels=wheels,
        bundle=bundle,
        module_digests={
            "control_plane": "a" * 64,
            "executor": "b" * 64,
            "node_runtime": "c" * 64,
        },
        module_counts={"control_plane": 10, "executor": 11, "node_runtime": 12},
    )
    globals_ = BUILD.build.__globals__
    monkeypatch.setitem(globals_, "DIST", dist)
    monkeypatch.setitem(globals_, "BUILD", tmp_path / "build")
    monkeypatch.setitem(
        globals_,
        "build_release_identity",
        lambda _root: {
            "sha256": "d" * 64,
            "runtime_prebuilt": False,
            "schema_rollback_compatible": False,
            "node_template_inputs": {"sha256": "e" * 64},
        },
    )
    monkeypatch.setitem(
        globals_, "load_component_artifacts", lambda *_args, **_kwargs: cached
    )
    monkeypatch.setitem(
        globals_,
        "build_component",
        lambda **_kwargs: pytest.fail("validated component artifact was rebuilt"),
    )
    monkeypatch.setitem(
        globals_,
        "run",
        lambda *_args, **_kwargs: pytest.fail("node bundle was rebuilt"),
    )
    monkeypatch.setitem(globals_, "project_version", lambda: "0.10.0")

    def run(command, **_kwargs):
        script = command[-1]
        output = (
            "7\n"
            if "LATEST_POSTGRES_SCHEMA_VERSION" in script
            else json.dumps({"agent": 3, "executor": 2})
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(BUILD.subprocess, "run", run)

    manifest = BUILD.build(
        sys.executable, reuse_artifacts_from=tmp_path / "current-release.json"
    )

    assert manifest["components"]["control_plane"]["module_count"] == 10
    assert manifest["components"]["executor"]["module_count"] == 11
    assert manifest["components"]["node_runtime"]["module_count"] == 12


def test_repository_local_component_cache_uses_its_own_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    cache_entry = root / ".cache/components/identity"
    manifest = cache_entry / "dist/current-release.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    wheels = {}
    for name in ("control_plane", "executor", "node_runtime"):
        path = cache_entry / f"{name}.whl"
        path.write_bytes(name.encode())
        wheels[name] = path
    bundle = cache_entry / "node-bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    observed: dict[str, object] = {}
    globals_ = BUILD.prepare_component_artifacts.__globals__
    monkeypatch.setitem(globals_, "ROOT", root)

    def load(_root, path, **kwargs):
        observed["path"] = path
        observed.update(kwargs)
        return SimpleNamespace(
            wheels=wheels,
            bundle=bundle,
            module_digests={name: name for name in wheels},
            module_counts={name: 1 for name in wheels},
        )

    monkeypatch.setitem(globals_, "load_component_artifacts", load)
    staging = tmp_path / "staging"
    staging.mkdir()

    BUILD.prepare_component_artifacts(
        sys.executable, staging=staging, reuse_artifacts_from=manifest
    )

    assert observed["artifact_root"] == cache_entry
    assert observed["require_delivery_identity"] is False
