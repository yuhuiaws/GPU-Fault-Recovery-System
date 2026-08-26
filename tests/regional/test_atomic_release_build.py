from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
BUILD = lazy_script_module(
    "atomic_release_build", ROOT / "scripts/build-release-artifacts.py"
)


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
