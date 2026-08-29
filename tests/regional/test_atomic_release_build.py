from __future__ import annotations

import sys
from pathlib import Path

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

    assert BUILD.resolve_python_executable("python3") == str(executable.resolve())
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
