from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from gpu_fault.admin import bootstrap_common, site
from gpu_fault.admin.config import default_admin_config
from scripts.e2e.regional import boot020_release_candidates as candidates
from tests.regional._cov95_boot_candidates import CandidateBuild
from tests.regional._cov95_boot_extra_safety import (
    boot_extra_isolation as boot_extra_isolation,
)


@pytest.mark.parametrize("corrupt_summary", [False, True])
def test_build_entrypoint_rebuilds_unreadable_cache_with_only_fake_git_and_builds(
    tmp_path, monkeypatch, capsys, corrupt_summary
):
    fixture = CandidateBuild(tmp_path, monkeypatch)
    args = fixture.arguments
    if corrupt_summary:
        args.work_dir.mkdir()
        (args.work_dir / "candidates.json").write_text("{", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "boot-candidates",
            "build",
            "--snapshot-repo",
            str(args.snapshot_repo),
            "--work-dir",
            str(args.work_dir),
            "--state-dir",
            str(args.state_dir),
            "--region",
            args.region,
            "--runtime-repository",
            args.runtime_repository,
            "--executor-module",
            args.executor_module,
            "--node-module",
            args.node_module,
        ],
    )
    assert candidates.main() == 0, "candidate construction failed behind fake builds"
    saved = json.loads((args.work_dir / "candidates.json").read_text(encoding="utf-8"))
    assert list(saved["candidates"]) == ["B", "C", "D"]
    assert len(fixture.build_calls) == 3
    assert len(fixture.git_calls) == 6
    assert all(
        call["repository_root"].is_relative_to(args.work_dir)
        for call in fixture.build_calls
    ), "a candidate build escaped the test-owned checkout"
    assert '"base_release_id": "base"' in capsys.readouterr().out


def test_configs_then_clean_entrypoints_preserve_unowned_artifacts(
    tmp_path, monkeypatch
):
    fixture = CandidateBuild(tmp_path, monkeypatch)
    args = fixture.arguments
    result = candidates.build_candidates(args)
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(candidates, "ROOT", repository)
    admin = default_admin_config()
    base = {
        "release": {"manifest": "unused", "agent_config_digest": "a" * 64},
        "runtime_profile": {"version": "unit-profile"},
        "admin_config": {
            "config": admin.as_dict(),
            "config_sha256": admin.sha256(),
            "role_sha256": admin.role_sha256(),
        },
    }
    monkeypatch.setattr(
        site,
        "load_site",
        lambda *_args, **_kwargs: SimpleNamespace(release_config=base),
    )
    digest_calls = []

    def digest(runner, **kwargs):
        digest_calls.append(kwargs)
        return "f" * 64

    monkeypatch.setattr(bootstrap_common, "compute_agent_config_digest", digest)
    out_dir = tmp_path / "configs"
    common = [
        "--snapshot-repo",
        str(args.snapshot_repo),
        "--work-dir",
        str(args.work_dir),
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "boot-candidates",
            "configs",
            *common,
            "--site",
            str(tmp_path / "site.yaml"),
            "--out-dir",
            str(out_dir),
            "--replicas-delta",
            "-1",
        ],
    )
    assert candidates.main() == 0, "config derivation failed"
    full = json.loads((out_dir / "full.json").read_text(encoding="utf-8"))
    assert full["release"]["agent_config_digest"] == "f" * 64
    assert full["runtime_profile"]["version"] == "unit-profile-boot020"
    assert full["release"]["manifest"] == result["candidates"]["D"]["manifest"]
    assert len(digest_calls) == 1
    assert sorted(path.name for path in out_dir.iterdir()) == sorted(
        [*(f"{name}.json" for name in candidates.CONFIG_NAMES), "final-noop.json"]
    )
    assert json.loads((out_dir / "final-noop.json").read_text()) == full
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in out_dir.iterdir()), (
        "derived configs must remain private"
    )
    unrelated = repository / "dist" / "foreign-release"
    unrelated.mkdir()
    sentinel = unrelated / "sentinel"
    sentinel.write_text("preexisting artifact", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["boot-candidates", "clean", *common])
    assert candidates.main() == 0, "cleanup did not finish"
    assert list((repository / "dist").iterdir()) == [unrelated]
    assert sentinel.read_text(encoding="utf-8") == "preexisting artifact"
    assert candidates.main() == 0, "repeated cleanup must be idempotent"


@pytest.mark.parametrize("failure", [None, "executor"])
def test_check_entrypoint_finishes_all_classifications_after_a_read_failure(
    tmp_path, monkeypatch, capsys, failure
):
    calls = []

    class Backend:
        def __init__(self, configs):
            assert set(configs) == set(candidates.DRIVER_KEYS.values())

        def classify(self, scenario):
            calls.append(scenario)
            if scenario == failure:
                raise RuntimeError("unit classification unavailable")
            return {"kind": "NOOP", "changed": []}

    monkeypatch.setitem(
        sys.modules,
        "run_boot020_release_rolling",
        SimpleNamespace(LiveReleaseRollingBackend=Backend),
    )
    monkeypatch.setattr(candidates, "sys", SimpleNamespace(path=list(sys.path)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "boot-candidates",
            "check",
            "--out-dir",
            str(tmp_path),
            "--gpu-kubeconfig",
            str(tmp_path / "unused-kubeconfig"),
        ],
    )
    assert candidates.main() == (1 if failure else 0)
    assert calls == list(candidates.DRIVER_KEYS.values())
    output = capsys.readouterr().out
    assert "full -> NOOP" in output
    assert ("data-plane -> ERROR" in output) is (failure is not None)


@pytest.mark.parametrize("unreadable", ["json", "io"])
def test_candidate_cache_read_errors_cannot_authorize_reuse(
    tmp_path, monkeypatch, unreadable
):
    manifest = tmp_path / "candidate.json"
    manifest.write_text("{", encoding="utf-8")
    if unreadable == "io":
        from pathlib import Path

        read_text = Path.read_text

        def read(path, *args, **kwargs):
            if path == manifest:
                raise OSError("unit candidate read failed")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
    assert (
        candidates.reusable_candidate(
            manifest,
            base_release_id="base",
            recorded={"release_id": "candidate", "edits_sha256": "e" * 64},
            expected_edits_digest="e" * 64,
        )
        is False
    )
