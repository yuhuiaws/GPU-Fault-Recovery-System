"""BOOT-020's release-candidate precondition, satisfied by the runner (2026-09-30).

Round 4 lost BOOT-020 and BOOT-023 because the five chained configs were an
operator input the campaign forgot to rebuild for the freshly deployed site;
the runner now derives the candidate inputs from the administrator state and
builds when the configs are missing or bound to another release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import boot020_release_prerequisites as prerequisites
from scripts.e2e.regional import run_boot020_release_rolling as boot020
from scripts.e2e.regional import run_boot023_release_history as boot023
from scripts.e2e.regional.regional_commands import RegionalFixtureError

RUNTIME = "1.dkr.ecr.us-west-2.amazonaws.com/gpu-fault/runtime-abc@sha256:" + "0" * 64


def _state_dir(
    tmp_path: Path, *, live: str = "rel-live", extra_repo: bool = True
) -> Path:
    state = tmp_path / "state"
    snapshot = state / "source-snapshots" / "deadbeef"
    for name, release_id in (
        ("repository-1-live", live),
        ("repository-0-old", "rel-old"),
    ):
        if name.endswith("old") and not extra_repo:
            continue
        dist = snapshot / name / "dist"
        dist.mkdir(parents=True)
        (dist / "current-release.json").write_text(
            json.dumps({"release_id": release_id})
        )
    state.mkdir(exist_ok=True)
    (state / "source-deploy-success.json").write_text(
        json.dumps(
            {
                "live": {"release_id": live},
                "prepared_repository_root": str(snapshot / "repository-1-live"),
            }
        )
    )
    (state / "site.yaml").write_text(
        "apiVersion: gpu-fault.aws/v1alpha1\nkind: Site\nspec:\n"
        f"  awsRegion: us-west-2\n  repositoryRoot: {snapshot}\n"
        f"  images:\n    runtime: {RUNTIME}\n"
    )
    return state


def _inputs(state: Path) -> prerequisites.CandidateInputs:
    return prerequisites.CandidateInputs(
        snapshot_repo=prerequisites.live_snapshot_repository(state),
        region="us-west-2",
        runtime_repository=RUNTIME.split("@")[0],
        cache_repository=RUNTIME.split("@")[0].replace("runtime-", "runtime-cache-"),
        runtime_profile="hyperpod-v1",
        site_file=state / "site.yaml",
    )


def _write_candidates(
    out_dir: Path, snapshot_repo: Path, *, state: Path | None = None
) -> None:
    """Five configs bound to ``snapshot_repo``; with ``state`` also the metadata
    recording that state's site inputs, which reuse requires since 2026-09-30."""

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = str(snapshot_repo / "dist" / "current-release.json")
    for key, name in prerequisites.CONFIG_FILES.items():
        (out_dir / name).write_text(
            json.dumps({"release": {"manifest": manifest}, "k": key})
        )
    if state is not None:
        prerequisites.write_candidate_metadata(
            out_dir, out_dir, site=prerequisites.site_inputs(state)
        )


def test_the_live_snapshot_is_the_prepared_repository_of_the_last_deploy(tmp_path):
    state = _state_dir(tmp_path)
    repo = prerequisites.live_snapshot_repository(state)
    assert repo.name == "repository-1-live", repo
    assert prerequisites.snapshot_release_id(repo) == "rel-live", "manifest release id"
    # A `gpu-fault-admin config` release moves the live id; the source stays.
    (state / "source-deploy-success.json").write_text(
        json.dumps(
            {
                "live": {"release_id": "cfg-release"},
                "prepared_repository_root": str(repo),
            }
        )
    )
    assert prerequisites.live_snapshot_repository(state) == repo.resolve(), (
        "a config release must not hide the deployed snapshot"
    )


def test_without_a_deploy_record_the_site_repository_root_is_used(tmp_path):
    state = _state_dir(tmp_path)
    (state / "source-deploy-success.json").unlink()
    repo = prerequisites.live_snapshot_repository(state)
    assert repo.name in {"repository-0-old", "repository-1-live"}, repo
    (state / "site.yaml").write_text(
        "spec:\n  awsRegion: us-west-2\n  repositoryRoot: /nonexistent\n"
        f"  images:\n    runtime: {RUNTIME}\n"
    )
    with pytest.raises(RegionalFixtureError, match="no deployed source snapshot"):
        prerequisites.live_snapshot_repository(state)


def test_candidates_are_current_only_when_bound_to_the_live_snapshot(tmp_path):
    state = _state_dir(tmp_path)
    live = prerequisites.live_snapshot_repository(state)
    out = tmp_path / "boot020-releases"
    assert prerequisites.candidates_bound_to(out, live) is False, "missing files"
    _write_candidates(out, live)
    assert prerequisites.candidates_bound_to(out, live) is True
    stale = state / "source-snapshots" / "deadbeef" / "repository-0-old"
    _write_candidates(out, stale)
    assert prerequisites.candidates_bound_to(out, live) is False, (
        "configs bound to a previous release must not be reused after a redeploy"
    )


def test_ensure_reuses_bound_candidates_without_building(tmp_path):
    state = _state_dir(tmp_path)
    live = prerequisites.live_snapshot_repository(state)
    out = tmp_path / "boot020-releases"
    _write_candidates(out, live, state=state)
    calls: list[str] = []
    record = prerequisites.ensure_release_candidates(
        state,
        out,
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        build=lambda ns: calls.append("build"),
        write=lambda ns: calls.append("write"),
        check=lambda ns: calls.append("check") or 0,
        inputs=_inputs,
    )
    assert record["action"] == "reused" and calls == [], (record, calls)
    assert record["live_release_id"] == "rel-live", record


def test_ensure_builds_writes_and_checks_when_candidates_are_stale(tmp_path):
    state = _state_dir(tmp_path)
    live = prerequisites.live_snapshot_repository(state)
    stale = state / "source-snapshots" / "deadbeef" / "repository-0-old"
    out = tmp_path / "boot020-releases"
    _write_candidates(out, stale)
    seen: dict[str, argparse.Namespace] = {}

    def build(ns: argparse.Namespace) -> Any:
        seen["build"] = ns

    def write(ns: argparse.Namespace) -> Any:
        seen["write"] = ns
        _write_candidates(ns.out_dir, ns.snapshot_repo)

    def check(ns: argparse.Namespace) -> int:
        seen["check"] = ns
        return 0

    record = prerequisites.ensure_release_candidates(
        state,
        out,
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        replicas_delta=-1,
        build=build,
        write=write,
        check=check,
        inputs=_inputs,
    )
    assert record["action"] == "built", record
    assert seen["build"].snapshot_repo == live, seen["build"]
    assert seen["build"].runtime_repository == RUNTIME.split("@")[0], seen["build"]
    assert seen["build"].cache_repository.endswith("/gpu-fault/runtime-cache-abc"), (
        seen["build"].cache_repository
    )
    assert seen["build"].region == "us-west-2", seen["build"]
    assert seen["build"].state_dir == state, seen["build"]
    assert seen["write"].replicas_delta == -1 and seen["write"].out_dir == out, seen[
        "write"
    ]
    assert seen["check"].out_dir == out, seen["check"]
    assert prerequisites.candidates_bound_to(out, live) is True


def test_ensure_refuses_candidates_that_fail_the_live_classification(tmp_path):
    state = _state_dir(tmp_path)
    live = prerequisites.live_snapshot_repository(state)
    out = tmp_path / "boot020-releases"
    with pytest.raises(RegionalFixtureError, match="do not classify"):
        prerequisites.ensure_release_candidates(
            state,
            out,
            gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
            build=lambda ns: None,
            write=lambda ns: _write_candidates(ns.out_dir, live),
            check=lambda ns: 1,
            inputs=_inputs,
        )


def test_candidate_inputs_derive_repository_region_and_snapshot(tmp_path, monkeypatch):
    state = _state_dir(tmp_path)
    monkeypatch.setattr(
        prerequisites, "candidate_inputs", prerequisites.candidate_inputs
    )
    # load_site needs a full site; only the pure pieces are checked here.
    spec = prerequisites.site_spec(state)
    assert spec["awsRegion"] == "us-west-2", spec
    assert prerequisites.live_release_id(state) == "rel-live", "live release id"


def test_runner_parser_accepts_omitted_configs_and_a_candidates_dir(tmp_path):
    arguments = boot020.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--plan",
            "--admin-state-dir",
            str(tmp_path),
            "--admin-reference",
            "REF-1",
            "--release-candidates-dir",
            str(tmp_path / "cands"),
        ]
    )
    assert arguments.noop_config is None and arguments.full_config is None, (
        "the five configs must be optional"
    )
    assert arguments.release_candidates_dir == tmp_path / "cands", arguments
    assert arguments.replicas_delta == -1, arguments


def test_plan_mode_records_a_pending_build_instead_of_building(tmp_path):
    state = _state_dir(tmp_path)
    arguments = argparse.Namespace(
        noop_config=None,
        control_plane_config=None,
        data_plane_config=None,
        agent_config=None,
        full_config=None,
        admin_state_dir=state,
        release_candidates_dir=None,
        replicas_delta=-1,
        execute=False,
        gpu_kubeconfig=None,
    )
    configs, record = prerequisites.resolve_release_configs(arguments)
    assert record["action"] == "build-at-execute", record
    assert configs["full"] == state / prerequisites.CANDIDATES_DIR_NAME / "full.json", (
        configs
    )
    # Execute mode resolves the same way: the build is deferred to after the
    # live-driver authorization (ensure_for_execute), never done while resolving.
    arguments.execute = True
    configs_exec, record_exec = prerequisites.resolve_release_configs(arguments)
    assert record_exec["action"] == "build-at-execute" and configs_exec == configs, (
        record_exec
    )
    arguments.execute = False
    _write_candidates(
        state / prerequisites.CANDIDATES_DIR_NAME,
        prerequisites.live_snapshot_repository(state),
        state=state,
    )
    configs, record = prerequisites.resolve_release_configs(arguments)
    assert record["action"] == "reused", record
    assert configs["noop"] == state / prerequisites.CANDIDATES_DIR_NAME / "noop.json", (
        configs
    )


def test_ensure_for_execute_only_acts_on_a_pending_build(tmp_path):
    record = {"action": "reused", "candidates_dir": str(tmp_path)}
    arguments = argparse.Namespace(
        admin_state_dir=tmp_path, gpu_kubeconfig=None, replicas_delta=-1
    )
    assert prerequisites.ensure_for_execute(arguments, record) is record, (
        "a reused or explicit record must not trigger a build"
    )


def test_boot023_defaults_to_boot020s_handoff_in_the_run(tmp_path):
    assert boot023.default_noop_config(tmp_path) == str(
        tmp_path / "cases" / "GF-REGIONAL-BOOT-020" / "boot023-noop.json"
    )
