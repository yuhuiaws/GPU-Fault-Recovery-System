"""DESTR015 scope and refusal edges using only fake I/O and regular temp files."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from scripts.e2e.regional import late_ownership_entry as companion_entry
from scripts.e2e.regional import late_ownership_live as companion
from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional import run_destr015_parallel_branch_join as case
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
)
from tests.regional._cov95_destr_branches import BranchHarness, BranchProbe
from tests.regional._cov95_destr_warm import NOW
from tests.regional.test_destr015_parallel_branch_join import NODES

COMPANION_CASE = "GF-REGIONAL-PREEMPT-033"


@pytest.fixture(autouse=True)
def refuse_real_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        pytest.fail("DESTR015 edge tests must never execute an unfaked transport")

    monkeypatch.setattr(RegionalLiveFixture, "run", staticmethod(refuse))


def late_scope() -> dict[str, Any]:
    return {
        "case_id": COMPANION_CASE,
        "scenario": "ownership-drift",
        "protocol_ready": True,
        "bundle_sha256": "a" * 64,
        "executor": {
            "name": "executor-owned",
            "uid": "executor-original-uid",
            "container_id": "containerd://original",
            "image_id": "registry.example/executor@sha256:" + "b" * 64,
        },
    }


def companion_plan(
    harness: BranchHarness, run_dir: Path
) -> tuple[companion_entry.Settings, dict[str, Any], Path]:
    preflight = harness.plan(run_dir / "ordinary")
    preflight["late_ownership"] = late_scope()
    directory = run_dir / "companion" / "cases" / COMPANION_CASE
    write_json_atomic(
        directory / "plan.json",
        {"details": case.plan_details(harness.settings, preflight)},
    )
    settings = companion_entry.Settings(
        harness.settings,
        COMPANION_CASE,
        "ownership-drift",
        run_dir / "ordinary" / "cases" / case.CASE_ID / f"{case.CASE_ID}.json",
    )
    return settings, preflight, directory


def test_optional_scope_is_absent_from_the_ordinary_identity() -> None:
    preflight = {
        "release_id": "release-original",
        "nodes": {
            node: {"uid": f"uid-{node}", "boot_id": f"boot-{node}"} for node in NODES
        },
        "store": {"profile": {"profile_version": "profile-original"}},
        "runtime_identity": {"generation": 7},
    }
    expected = {
        "release_id": "release-original",
        "node_uids": {node: f"uid-{node}" for node in NODES},
        "node_boot_ids": {node: f"boot-{node}" for node in NODES},
        "runtime_profile_version": "profile-original",
        "runtime_identity": {"generation": 7},
    }
    assert case.plan_identity(preflight, nodes=NODES) == expected
    supplied = {**preflight, "late_ownership": None}
    identity = case.plan_identity(supplied, nodes=NODES)
    assert identity == {**expected, "late_ownership": None}
    assert case.identity_digest(identity) != case.identity_digest(expected)
    assert "late_ownership" not in preflight


@pytest.mark.parametrize(
    "field", ["case_id", "scenario", "protocol_ready", "bundle_sha256", "executor"]
)
def test_every_companion_scope_dimension_changes_the_plan_identity(field: str) -> None:
    preflight = {"late_ownership": late_scope()}
    original = case.plan_identity(preflight, nodes=NODES)
    changed = deepcopy(preflight)
    changed["late_ownership"][field] = {
        "case_id": case.CASE_ID,
        "scenario": "late-sibling",
        "protocol_ready": False,
        "bundle_sha256": "c" * 64,
        "executor": {**late_scope()["executor"], "uid": "recreated-executor"},
    }[field]
    current = case.plan_identity(changed, nodes=NODES)
    assert current["late_ownership"] == changed["late_ownership"]
    assert case.identity_digest(current) != case.identity_digest(original)
    assert original["late_ownership"] == late_scope()


def test_companion_setup_uses_its_own_case_scope_and_supplied_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BranchHarness(tmp_path, monkeypatch)
    settings, preflight, directory = companion_plan(harness, tmp_path)
    ordinary = tmp_path / "ordinary" / "cases" / case.CASE_ID / "plan.json"
    ordinary_before = ordinary.read_bytes()
    probe_settings: list[Any] = []
    prewarm_settings: list[dict[str, Any]] = []
    preflight_calls: list[dict[str, Any]] = []
    prepared: list[Any] = []

    def supplied(
        current: companion_entry.Settings, path: Path, **kwargs: Any
    ) -> dict[str, Any]:
        assert current is settings and path == directory
        preflight_calls.append(kwargs)
        return deepcopy(preflight)

    def no_second_preflight(*args: Any, **kwargs: Any) -> None:
        pytest.fail(
            "supplied companion preflight must not be replaced by ordinary preflight"
        )

    def probe(value: Any) -> BranchProbe:
        probe_settings.append(value)
        return BranchProbe(harness, value.node)

    def prewarm(regional: Any, **kwargs: Any) -> Any:
        assert regional is harness.regional
        prewarm_settings.append(kwargs)
        return harness.prewarm

    def before_companion_action(run: Any, **kwargs: Any) -> None:
        assert kwargs == {"case_id": COMPANION_CASE, "scenario": "ownership-drift"}
        prepared.append(run)
        raise RegionalFixtureError("intentional stop before companion action")

    monkeypatch.setattr(companion, "read_only_preflight", supplied)
    monkeypatch.setattr(case, "read_only_preflight", no_second_preflight)
    monkeypatch.setattr(case, "HostProbeFixture", probe)
    monkeypatch.setattr(case, "ImagePrewarmFixture", prewarm)
    monkeypatch.setattr(companion, "build_scope", before_companion_action)
    with pytest.raises(RegionalFixtureError, match="intentional stop"):
        companion.execute_case(
            settings, directory.parents[1], 3, NOW + timedelta(hours=1)
        )

    assert preflight_calls == [{"reuse_focused_tests": True}]
    assert len(prepared) == 1
    run = prepared[0]
    assert run.case_dir == directory and run.preflight == preflight
    assert run.settings is harness.settings
    assert set(run.source_uids) == {f"old-{node}" for node in NODES}
    assert prewarm_settings == [{"case_id": COMPANION_CASE, "run_id": run.run_id}]
    assert [value.node for value in probe_settings] == list(NODES)
    for value in probe_settings:
        assert value.case_id == COMPANION_CASE
        assert value.run_id == run.run_id
        assert value.state_directory == directory / "host-probes"
        assert value.kubeconfig == harness.settings.regional.gpu_kubeconfig
        assert value.context == harness.settings.regional.gpu_context
    names = [name for name, _ in harness.calls]
    assert "workload.submit" in names
    assert "workload.delete" in names
    assert names.count("probe.cleanup") == 2
    assert "probe.write-xid46" not in names
    assert ordinary.read_bytes() == ordinary_before
    assert not (directory / "result.json").exists(), (
        "aborted companion setup must not publish a result"
    )
    assert not (ordinary.parent / f"{case.CASE_ID}.json").exists(), (
        "companion setup must not publish an ordinary-case verdict"
    )
    assert case.CASE.case_id == case.CASE_ID
    assert case.CASE.confirmation == case.CONFIRMATION == "DESTR015_EXECUTE"


@pytest.mark.parametrize(
    "defect", ["preflight-error", "scenario", "executor", "bundle"]
)
def test_companion_drift_is_refused_before_any_resource_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    harness = BranchHarness(tmp_path, monkeypatch)
    settings, preflight, directory = companion_plan(harness, tmp_path)
    original_plan = (directory / "plan.json").read_bytes()
    if defect == "preflight-error":
        preflight["errors"] = ["final ownership protocol is unavailable"]
        message = "preflight failed"
    else:
        if defect == "scenario":
            preflight["late_ownership"]["scenario"] = "late-sibling"
        elif defect == "executor":
            preflight["late_ownership"]["executor"]["uid"] = "recreated-executor"
        else:
            preflight["late_ownership"]["bundle_sha256"] = "d" * 64
        message = "plan identity drifted"
    harness.calls.clear()
    monkeypatch.setattr(
        companion, "read_only_preflight", lambda *args, **kwargs: deepcopy(preflight)
    )
    with pytest.raises(RegionalFixtureError, match=message):
        companion.execute_case(
            settings, directory.parents[1], 1, NOW + timedelta(hours=1)
        )
    assert harness.calls == []
    assert not harness.injected, "companion scope drift must prevent fault injection"
    assert (directory / "plan.json").read_bytes() == original_plan
    assert not (directory / "pinned-workload.yaml").exists(), (
        "refused companion scope must not materialize a workload"
    )


def test_companion_expired_window_cannot_begin_image_prewarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BranchHarness(tmp_path, monkeypatch)
    settings, preflight, directory = companion_plan(harness, tmp_path)
    monkeypatch.setattr(
        companion, "read_only_preflight", lambda *args, **kwargs: deepcopy(preflight)
    )
    harness.clock.sleep(61)
    with pytest.raises(RegionalFixtureError, match="window ended before image prewarm"):
        companion.execute_case(
            settings, directory.parents[1], 1, NOW + timedelta(seconds=60)
        )
    names = [name for name, _ in harness.calls]
    assert "prewarm.create" not in names
    assert "workload.submit" not in names
    assert "probe.write-xid46" not in names
    assert "prewarm.cleanup" in names
    assert names.count("probe.cleanup") == 2


@pytest.mark.parametrize("explicit", [False, True])
def test_configuration_and_environment_keep_the_ordinary_identity_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    harness = BranchHarness(tmp_path, monkeypatch)
    regional = harness.settings.regional
    predecessor = tmp_path / "explicit-predecessor.json"
    arguments = argparse.Namespace(
        run_dir=tmp_path / "run-owned",
        attempt=4,
        cpu_kubeconfig=str(regional.cpu_kubeconfig),
        gpu_kubeconfig=str(regional.gpu_kubeconfig),
        gpu_context=regional.gpu_context,
        namespace=regional.namespace,
        cluster_id=regional.cluster_id,
        region=regional.region,
        site_file=str(harness.settings.site_file) if explicit else "",
        host_probe_image=harness.settings.host_probe_image if explicit else "",
        node_a=NODES[0] if explicit else "",
        node_b=NODES[1] if explicit else "",
        manifest=harness.settings.manifest,
        pci_bdf_a=" 0000:03:00.0 " if explicit else "",
        pci_bdf_b=" 0000:04:00.0 " if explicit else "",
        job_id=" explicit-job " if explicit else "",
        attempt_id=" explicit-attempt " if explicit else "",
        predecessor_evidence=str(predecessor) if explicit else "",
    )
    for name, value in harness.settings.environment().items():
        monkeypatch.setenv(name, value)
    configured = case.configure(arguments)
    default_job, default_attempt = case.derived_identity(arguments.run_dir, 4)
    assert configured.nodes == NODES
    assert configured.regional == regional
    assert configured.job_id == ("explicit-job" if explicit else default_job)
    assert configured.attempt_id == (
        "explicit-attempt" if explicit else default_attempt
    )
    assert configured.site_file == harness.settings.site_file.resolve()
    assert configured.manifest == harness.settings.manifest.resolve()
    assert configured.pci_bdf_a == ("0000:03:00.0" if explicit else "")
    assert configured.pci_bdf_b == ("0000:04:00.0" if explicit else "")
    assert configured.predecessor_path == (
        predecessor.resolve()
        if explicit
        else (
            arguments.run_dir
            / "cases"
            / case.PREDECESSOR_CASE_ID
            / f"{case.PREDECESSOR_CASE_ID}.json"
        ).resolve()
    )
    assert configured.environment() == {
        **regional.environment(),
        "GPU_FAULT_SITE_FILE": str(configured.site_file),
        "GPU_FAULT_TRAINING_MANIFEST": str(configured.manifest),
        "GPU_FAULT_HOST_PROBE_IMAGE": configured.host_probe_image,
        "GPU_FAULT_NODE_A": NODES[0],
        "GPU_FAULT_NODE_B": NODES[1],
        "GPU_FAULT_TEST_JOB_ID": configured.job_id,
        "GPU_FAULT_TEST_ATTEMPT_ID": configured.attempt_id,
        "GPU_FAULT_PREDECESSOR_EVIDENCE": str(configured.predecessor_path),
    }
    assert harness.calls == []


@pytest.mark.parametrize("mode", ["disabled", "missing", "failed", "source-drift"])
@pytest.mark.parametrize("returncode", [0, 2])
def test_focused_cache_miss_records_the_fresh_result_without_running_a_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, returncode: int
) -> None:
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "a" * 64)
    if mode != "missing":
        details: dict[str, Any] = {}
        live_driver_guard.record_focused_tests(
            details,
            {"passed": mode != "failed", "returncode": 0, "command": ["cached"]},
        )
        if mode == "source-drift":
            details["focused_tests_source_digest"] = "b" * 64
        write_json_atomic(tmp_path / "plan.json", {"details": details})
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> CompletedProcess[str]:
        calls.append(command)
        assert command[:4] == [sys.executable, "-m", "pytest", "-q"]
        assert command[-1] == "tests/regional/test_destr015_parallel_branch_join.py"
        assert kwargs == {"cwd": case.ROOT, "check": False, "timeout": 600}
        return CompletedProcess(
            command, returncode, stdout="fake stdout\n", stderr="fake stderr\n"
        )

    monkeypatch.setattr(RegionalLiveFixture, "run", staticmethod(run))
    result = case.focused_tests(tmp_path, reuse=mode != "disabled")
    assert len(calls) == 1
    assert result == {
        "passed": returncode == 0,
        "returncode": returncode,
        "command": calls[0],
    }
    assert "focused_tests_reused" not in result
    path = tmp_path / "focused-tests.log"
    assert path.read_text() == "fake stdout\nfake stderr\n"
    assert path.stat().st_mode & 0o777 == 0o600


def test_explicit_gpu_selection_reaches_only_its_node_in_the_normal_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BranchHarness(tmp_path, monkeypatch)
    selected = {NODES[0]: "0000:03:00.0", NODES[1]: "0000:04:00.0"}
    harness.settings = replace(
        harness.settings, pci_bdf_a=selected[NODES[0]], pci_bdf_b=selected[NODES[1]]
    )
    harness.plan(tmp_path)
    code, report = harness.execute(tmp_path)
    assert code == 0 and report["verdict"] == "PASS"
    assert report["errors"] == report["cleanup"]["errors"] == []
    writes = [detail for name, detail in harness.calls if name == "probe.write-xid46"]
    assert len(writes) == 2
    assert {value["node"] for value in writes} == set(NODES)
    for value in writes:
        arguments = value["args"]
        assert arguments[arguments.index("--pci-bdf") + 1] == selected[value["node"]]


def test_observation_budget_expiry_keeps_nonterminal_workflow_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BranchHarness(tmp_path, monkeypatch)
    harness.plan(tmp_path)
    harness.pending_reads = 1
    original = harness.regional.store_snapshot
    observations: list[dict[str, Any]] = []

    def delayed(**kwargs: Any) -> dict[str, Any]:
        state = original(**kwargs)
        if kwargs.get("workflow_request_ids") == [harness.workflow["request_id"]]:
            observations.append(kwargs)
            harness.clock.sleep(case.OBSERVATION_BUDGET_SECONDS + 1)
        return state

    monkeypatch.setattr(harness.regional, "store_snapshot", delayed)
    code, report = harness.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL"
    assert report["errors"] == ["workflow status is not SUCCEEDED: RUNNING"]
    assert report["cleanup"]["errors"] == []
    assert len(observations) == 1
    state_path = tmp_path / "cases" / case.CASE_ID / "workflow-state.json"
    assert json.loads(state_path.read_text())["workflow"]["status"] == "RUNNING"
    names = [name for name, _ in harness.calls]
    assert "workload.restarted" not in names
    assert "workload.delete" in names


def test_cpu_blast_change_cannot_be_hidden_by_successful_parallel_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = BranchHarness(tmp_path, monkeypatch)
    harness.cpu_after = {"nodes": ["cpu-a", "unexpected-node"]}
    harness.plan(tmp_path)
    code, report = harness.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL"
    assert report["errors"] == ["control-plane EKS state differs from baseline"]
    assert report["cleanup"]["errors"] == []
    assert report["restart_budget"]["restart_count"] == 1
    path = tmp_path / "cases" / case.CASE_ID / "cpu-blast-after.json"
    assert json.loads(path.read_text()) == harness.cpu_after
