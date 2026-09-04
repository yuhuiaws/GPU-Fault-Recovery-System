from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
BUILD = lazy_script_module(ROOT / "scripts/build-release-artifacts.py")


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
    monkeypatch.setattr(BUILD, "DIST", dist)
    monkeypatch.setattr(BUILD, "BUILD", tmp_path / "build")

    def fail_component(**_kwargs):
        raise RuntimeError("component build failed")

    monkeypatch.setattr(BUILD, "build_component", fail_component)

    with pytest.raises(RuntimeError, match="component build failed"):
        BUILD.build(sys.executable)

    assert current.read_text(encoding="utf-8") == '{"release_id":"stable"}\n'
    assert not list(dist.glob(".build-*")), "failed staging directory was retained"


def test_release_builder_never_deletes_published_dist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful build swaps the manifest in without emptying dist.

    Deploys read ``dist/current-release.json`` and the release directory it
    points at. Rebuilding by clearing dist first would leave a window where a
    concurrent deploy resolves a manifest whose artifacts are gone, so the
    publish is a rename over the live tree and preserved directories such as the
    signed CI evidence survive it.
    """

    dist = tmp_path / "dist"
    preserved = dist / "ci-domains/unit"
    preserved.mkdir(parents=True)
    current = dist / "current-release.json"
    current.write_text('{"release_id":"stable"}\n', encoding="utf-8")
    monkeypatch.setattr(BUILD, "DIST", dist)
    monkeypatch.setattr(BUILD, "BUILD", tmp_path / "build")

    def prepare(_python, *, staging: Path, reuse_artifacts_from=None):
        names = {
            "control_plane": "control.whl",
            "executor": "executor.whl",
            "node_runtime": "node.whl",
        }
        wheels = {}
        for component, name in names.items():
            path = staging / name
            path.write_text(component, encoding="utf-8")
            wheels[component] = path
        bundle = staging / "node-bundle.tar.gz"
        bundle.write_text("bundle", encoding="utf-8")
        return BUILD.PreparedArtifacts(
            wheels=wheels,
            bundle=bundle,
            module_digests=dict.fromkeys(names, "d" * 64),
            module_counts=dict.fromkeys(names, 1),
        )

    monkeypatch.setattr(BUILD, "prepare_component_artifacts", prepare)
    removed: list[Path] = []
    real_rmtree = BUILD.shutil.rmtree
    monkeypatch.setattr(
        BUILD.shutil,
        "rmtree",
        lambda path, **kwargs: (
            removed.append(Path(path)) or real_rmtree(path, **kwargs)
        ),
    )

    manifest = BUILD.build(sys.executable, staging_only=True)

    release_id = str(manifest["release_id"])
    assert json.loads(current.read_text(encoding="utf-8"))["release_id"] == release_id
    assert (dist / release_id / "release.json").is_file(), (
        "the published release directory has no manifest"
    )
    assert preserved.is_dir(), "the signed CI evidence directory was deleted"
    assert dist not in removed, "the builder deleted the published dist tree"


def test_a_killed_build_does_not_wedge_the_next_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staging directory nothing cleaned up has to be cleared, not counted.

    ``artifact-check`` rebuilds and then counts what is in dist, so a staging
    directory from a build that was killed rather than failed -- an interrupted
    deploy, an OOM, a reboot -- makes the gate report six wheels where the
    release has three, blaming the build that is actually fine. The reuse
    shortcut has to clear it too: a rebuild of the current release publishes
    nothing new and would otherwise leave the leftover in place forever.
    """

    dist = tmp_path / "dist"
    abandoned = dist / ".build-killedrun/687057570a62"
    abandoned.mkdir(parents=True)
    (abandoned / "gpu_fault_control_plane-0.10.0-py3-none-any.whl").write_bytes(b"old")
    current = dist / "current-release.json"
    current.write_text('{"release_id":"stable"}\n', encoding="utf-8")
    monkeypatch.setattr(BUILD, "DIST", dist)
    monkeypatch.setattr(BUILD, "BUILD", tmp_path / "build")
    monkeypatch.setattr(
        BUILD,
        "resolve_current_component_artifacts",
        lambda **_kwargs: ({"release_id": "stable"}, None),
    )

    manifest = BUILD.build(sys.executable, reuse_if_current=True)

    assert manifest == {"release_id": "stable"}, "the reuse shortcut was not taken"
    assert not list(dist.glob(".build-*")), "an abandoned staging directory survived"
    assert current.read_text(encoding="utf-8") == '{"release_id":"stable"}\n'


def test_release_pruning_preserves_signed_ci_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist = tmp_path / "dist"
    release = dist / "release-current"
    domains = dist / "ci-domains/unit"
    stale = dist / "release-stale"
    for path in (release, domains, stale):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(BUILD, "DIST", dist)

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
    monkeypatch.setattr(BUILD, "DIST", dist)
    monkeypatch.setattr(BUILD, "BUILD", tmp_path / "build")
    monkeypatch.setattr(
        BUILD,
        "build_release_identity",
        lambda _root: {
            "sha256": "d" * 64,
            "runtime_prebuilt": False,
            "schema_rollback_compatible": False,
            "node_template_inputs": {"sha256": "e" * 64},
        },
    )
    monkeypatch.setattr(
        BUILD, "load_component_artifacts", lambda *_args, **_kwargs: cached
    )
    monkeypatch.setattr(
        BUILD,
        "build_component",
        lambda **_kwargs: pytest.fail("validated component artifact was rebuilt"),
    )
    monkeypatch.setattr(
        BUILD, "run", lambda *_args, **_kwargs: pytest.fail("node bundle was rebuilt")
    )
    monkeypatch.setattr(BUILD, "project_version", lambda: "0.10.0")

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
    monkeypatch.setattr(BUILD, "ROOT", root)

    def load(_root, path, **kwargs):
        observed["path"] = path
        observed.update(kwargs)
        return SimpleNamespace(
            wheels=wheels,
            bundle=bundle,
            module_digests={name: name for name in wheels},
            module_counts={name: 1 for name in wheels},
        )

    monkeypatch.setattr(BUILD, "load_component_artifacts", load)
    staging = tmp_path / "staging"
    staging.mkdir()

    BUILD.prepare_component_artifacts(
        sys.executable, staging=staging, reuse_artifacts_from=manifest
    )

    assert observed["artifact_root"] == cache_entry
    assert observed["require_delivery_identity"] is False
