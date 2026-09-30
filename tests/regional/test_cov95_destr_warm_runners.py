from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr003_warm_spare_failover as failover
from scripts.e2e.regional import run_destr008_warm_spare_shortage as shortage
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from scripts.e2e.regional.warm_spare_fixture import (
    HYPERPOD_HEALTH_LABEL,
    INSTANCE_GROUP_LABEL,
    QUARANTINE_TAINT,
    SPARE_LABEL,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
)
from tests.regional._cov95_destr_warm import (
    FAULT,
    INCIDENT,
    NOW,
    SPARE,
    WarmHarness,
    node_snapshot,
)


@pytest.mark.parametrize("module", [failover, shortage])
def test_preflight_reads_real_guards_and_binds_predecessor(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(module, tmp_path, monkeypatch)
    result = h.plan(tmp_path)
    assert result["errors"] == [], result
    assert result["store"]["profile"]["profile_version"] == "profile-v1", result
    assert result["store"]["remote_commands"]["open_by_cluster"] == {}, result
    predecessor = next(detail for name, detail in h.calls if name == "predecessor")
    assert predecessor == {
        "case_id": module.PREDECESSOR_CASE_ID,
        "release_id": "release-test",
        "cluster_id": "cluster-a",
        "region": "us-west-2",
    }, predecessor
    h.calls.clear()
    reused = module.read_only_preflight(
        h.settings, tmp_path, reusable_tests=h.test_result
    )
    assert reused["errors"] == [], reused
    assert reused["focused_tests"]["reused"] is True, reused
    assert not any(name == "focused" for name, _ in h.calls), h.calls


def _execution_journal(path: Path, *, completed: bool, started: bool) -> None:
    record = {
        "schema_version": 1,
        "binding": {"inputs": {"scenarios": ["no-spare"]}},
        "attempt": 1,
        "maintenance_expires_at": 1_800_000_000,
        "release_id": "release-test",
        "profile_version": "profile-v1",
        "fault_uid": "fault-uid",
        "spare_uid": "spare-uid",
        "scenarios": {
            "no-spare": {
                "state": "STARTED" if started else "CLEANED",
                "workload_started": True,
                "fixture_started": True,
                "post_started": True,
            }
        },
        "prewarm_started": True,
        "prewarm_cleaned": True,
        "completed": completed,
    }
    path.write_text(json.dumps(record))
    path.chmod(0o600)


def test_shortage_preflight_defers_readiness_only_for_an_unfinished_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Attempts 11 and 15 (2026-09-19) died mid-scenario and left their pinned
    # workload, a quarantined fault node and an unlabelled spare; the driver's
    # --plan refused those readiness errors, so the guard never authorized the
    # cleanup-only continuation that removes them. An UNFINISHED journal defers
    # them; a finished one (the next attempt runs fresh) keeps every check.
    h = WarmHarness(shortage, tmp_path, monkeypatch)
    h.gpu_workloads = [{"node": FAULT, "name": "gpu-fault-single-node-warm-spare"}]
    h.nodes[SPARE]["labels"].pop(SPARE_LABEL)
    journal = tmp_path / shortage.EXECUTION_OWNER_JOURNAL
    _execution_journal(journal, completed=False, started=True)
    pending = shortage.read_only_preflight(
        h.settings, tmp_path, reusable_tests=h.test_result
    )
    assert pending["errors"] == [] and pending["cleanup_continuation"] is True, pending
    assert set(pending["deferred_readiness_errors"]) >= {
        "fault or spare node already has a GPU workload",
        "spare node is not declared by the spare label",
    }, "the plan must still record what the continuation is expected to remove"
    assert pending["occupied_targets"] == h.gpu_workloads, pending
    _execution_journal(journal, completed=True, started=False)
    finished = shortage.read_only_preflight(
        h.settings, tmp_path, reusable_tests=h.test_result
    )
    assert finished["cleanup_continuation"] is False, finished
    assert "fault or spare node already has a GPU workload" in finished["errors"], (
        "a finished journal owns nothing: the next attempt runs fresh with every check"
    )
    journal.write_text("not json")
    assert (
        shortage.read_only_preflight(
            h.settings, tmp_path, reusable_tests=h.test_result
        )["cleanup_continuation"]
        is False
    ), "an unreadable journal never relaxes readiness"


@pytest.mark.parametrize("module", [failover, shortage])
@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        ("predecessor", "predecessor evidence"),
        ("same-node", "identical"),
        ("fault-ready", "fault node is not Ready"),
        ("fault-spare", "fault node is labeled"),
        ("fault-taint", "quarantine ownership"),
        ("fault-owner", "quarantine ownership"),
        ("spare-ready", "spare node is not Ready"),
        ("spare-label", "not declared"),
        ("spare-health", "health label"),
        ("reservation", "already has a reservation"),
        ("pool-state", "not AVAILABLE"),
        ("spare-set", "declared spare set"),
        ("topology", "topology do not match"),
        ("workload", "already has a GPU workload"),
        ("agent", "exactly one ACTIVE"),
        ("profile-absent", "no nodeReplace"),
        ("profile-owner", "wrong owner"),
        ("warnings", "profile has warnings"),
        ("queue", "processor queue"),
        ("commands-unknown", "queue state is unknown"),
        ("commands-open", "remote command queue is not empty"),
        ("cluster", "NodeRecovery=None"),
        ("executor-empty", "safety environment"),
        ("executor-unsafe", "safety environment"),
        ("gates-empty", "not enabled on every"),
        ("gates-false", "not enabled on every"),
        ("tests", "regression tests failed"),
    ],
)
def test_preflight_rejects_unproven_or_unsafe_inputs(
    module: Any,
    fault: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = WarmHarness(module, tmp_path, monkeypatch)
    if fault == "predecessor":
        h.predecessor_valid = False
    elif fault == "same-node":
        h.settings = replace(h.settings, spare_node=FAULT)
    elif fault == "fault-ready":
        h.nodes[FAULT]["ready"] = "False"
    elif fault == "fault-spare":
        h.nodes[FAULT]["labels"][SPARE_LABEL] = "true"
    elif fault == "fault-taint":
        h.nodes[FAULT]["taints"] = [{"key": QUARANTINE_TAINT}]
    elif fault == "fault-owner":
        h.nodes[FAULT]["annotations"]["gpu-fault.io/incident-id"] = "foreign"
    elif fault == "spare-ready":
        h.nodes[SPARE]["unschedulable"] = False
    elif fault == "spare-label":
        h.nodes[SPARE]["labels"].pop(SPARE_LABEL)
    elif fault == "spare-health":
        h.nodes[SPARE]["labels"][HYPERPOD_HEALTH_LABEL] = "Unknown"
    elif fault == "reservation":
        h.nodes[SPARE]["annotations"][SPARE_RESERVATION_ANNOTATION] = "foreign"
    elif fault == "pool-state":
        h.nodes[SPARE]["annotations"][SPARE_POOL_STATE_ANNOTATION] = "ALLOCATED"
    elif fault == "spare-set":
        h.declared_spares.append("node-foreign")
    elif fault == "topology":
        h.nodes[SPARE]["labels"][INSTANCE_GROUP_LABEL] = "other-group"
    elif fault == "workload":
        h.gpu_workloads = [{"node": SPARE}]
    elif fault == "agent":
        h.agents.append(deepcopy(h.agents[0]))
    elif fault == "profile-absent":
        h.fault_state["profile"] = None
    elif fault == "profile-owner":
        h.fault_state["profile"]["capabilities"][0]["owner"] = "untrusted"
    elif fault == "warnings":
        h.fault_state["profile"]["warnings"] = ["drift"]
    elif fault == "queue":
        h.fault_state["queue"]["fault_backlog_depth"] = 1
    elif fault == "commands-unknown":
        h.fault_state["remote_commands"] = None
    elif fault == "commands-open":
        h.fault_state["remote_commands"]["open_by_cluster"] = {"cluster-a": 1}
    elif fault == "cluster":
        h.cluster["node_recovery"] = "Automatic"
    elif fault == "executor-empty":
        h.executor_env = []
    elif fault == "executor-unsafe":
        h.executor_env[0]["allow_replace"] = "true"
    elif fault == "gates-empty":
        h.gates = []
    elif fault == "gates-false":
        h.gates[0]["enabled"] = "false"
    elif fault == "tests":
        h.test_result["passed"] = False
    result = module.read_only_preflight(h.settings, tmp_path)
    assert any(expected in item for item in result["errors"]), result
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        h.execute(tmp_path)
    assert not any(
        name.endswith((".create", ".submit", ".post")) for name, _ in h.calls
    ), h.calls


def test_failover_executes_and_restores_only_owned_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(failover, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.calls.clear()
    code, report = h.execute(tmp_path)
    assert code == 0 and report["verdict"] == "PASS", report
    assert report["errors"] == report["cleanup"]["errors"] == [], report
    assert report["source_pod_uids"] == ["source-uid"], report
    assert report["target_pod_uids"] == ["target-uid"], report
    assert report["provider_events_provisional"] is True, report
    assert h.posted["replacement_strategy"] == "HEALTHY_WARM_SPARE_ONLY", h.posted
    assert h.posted["cluster_id"] == "cluster-a" and h.posted["node_id"] == FAULT, (
        h.posted
    )
    assert h.posted["gpu_uuids"] == [f"GPU-{i}" for i in range(8)], h.posted
    phases = [name for name, _ in h.calls]
    ordered = [
        "prewarm.create",
        "workload.submit",
        "workload.running",
        "replacement.post",
        "workflow.wait",
        "workload.authorize_restart",
        "workload.restarted",
        "incident.idle",
        "workload.delete",
        "spares.release",
        "agent.reactivate",
        "agent.active",
        "restore.create",
        "restore.wait",
        "prewarm.cleanup",
    ]
    assert [phases.index(name) for name in ordered] == sorted(
        phases.index(name) for name in ordered
    ), phases
    release = next(detail for name, detail in h.calls if name == "spares.release")
    assert release == {"nodes": [SPARE], "incident_id": INCIDENT}, release
    assert h.nodes == {node: node_snapshot(node) for node in (FAULT, SPARE)}, h.nodes


@pytest.mark.parametrize(
    ("phase", "error_key"),
    [
        ("prewarm.create", "error"),
        ("workload.submit", "error"),
        ("workload.running", "error"),
        ("replacement.post", "error"),
        ("workflow.wait", "error"),
        ("workload.authorize_restart", "error"),
        ("workload.restarted", "error"),
        ("workload.delete", "cleanup"),
        ("spares.release", "cleanup"),
        ("agent.reactivate", "cleanup"),
        ("agent.active", "cleanup"),
        ("restore.create", "cleanup"),
        ("restore.wait", "cleanup"),
        ("prewarm.cleanup", "cleanup"),
    ],
)
def test_failover_phase_failure_is_reported_and_cleanup_still_runs(
    phase: str, error_key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(failover, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    h.calls.clear()
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    assert phase in str(report[error_key]), report
    phases = [name for name, _ in h.calls]
    assert "prewarm.cleanup" in phases, phases
    if phase == "workload.authorize_restart":
        assert "workload.restarted" not in phases
    if phase in {"replacement.post", "workflow.wait"}:
        assert report["incident_id_recovered_from_event"] is True, report
        assert phases.index("incident.lookup") < phases.index("workload.delete"), phases
    if phase in {"workload.submit", "workload.running", "prewarm.create"}:
        assert "replacement.post" not in phases, phases
        assert "spares.release" not in phases, phases


@pytest.mark.parametrize("lookup", ["missing", "error", "busy"])
def test_failover_defers_workload_and_spare_cleanup_without_quiescence(
    lookup: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(failover, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["workflow.wait"] = TimeoutError("terminal state unavailable")
    if lookup == "missing":
        h.lookup_incident = ""
    elif lookup == "error":
        h.failures["incident.lookup"] = RuntimeError("lookup unavailable")
    else:
        h.failures["incident.idle"] = TimeoutError("lease still active")
    h.calls.clear()
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert report["cleanup"]["workload_and_spare_cleanup_deferred"] is True, report
    names = [name for name, _ in h.calls]
    assert "prewarm.cleanup" in names, names
    assert not {"workload.delete", "spares.release", "restore.create"} & set(names), (
        names
    )
    if lookup == "error":
        assert "lookup unavailable" in report["incident_lookup_error"], report


@pytest.mark.parametrize("phase", ["prewarm.create", "workload.running"])
def test_failover_rechecks_window_before_the_next_mutation(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(failover, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.advance_at[phase] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "window ended" in report["error"], report
    names = [name for name, _ in h.calls]
    assert "replacement.post" not in names, names
    assert "prewarm.cleanup" in names, names


def test_failover_abort_runs_cleanup_and_propagates_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(failover, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["workflow.wait"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _ in h.calls]
    assert names.index("incident.lookup") < names.index("workload.delete"), names
    assert "prewarm.cleanup" in names, names


@pytest.mark.parametrize(
    ("variation", "expected"),
    [
        ("budget", "restart budget"),
        ("target", "not on the warm spare"),
        ("quarantine", "not quarantined"),
        ("owner", "ownership is incorrect"),
        ("reservation", "does not reference"),
        ("pool", "not ALLOCATED"),
        ("cordoned", "remained cordoned"),
        ("agent", "not revoked"),
        ("provider", "provider replace/delete"),
    ],
)
def test_failover_evidence_failure_cannot_become_pass(
    variation: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(failover, tmp_path, monkeypatch)
    h.plan(tmp_path)
    if variation == "budget":
        h.workflow["restart_budget"]["restart_count"] = 2
    elif variation == "target":
        h.target_nodes = [FAULT]
    elif variation == "agent":
        h.workflow["agents"] = []
    elif variation == "provider":
        h.events = [{"event_name": "BatchReplaceClusterNodes"}]
    else:
        node = FAULT if variation in {"quarantine", "owner"} else SPARE
        value = node_snapshot(node)
        if node == FAULT:
            value.update(
                unschedulable=True,
                taints=[{"key": QUARANTINE_TAINT}],
                annotations={"gpu-fault.io/incident-id": INCIDENT},
            )
        else:
            value.update(
                unschedulable=False,
                annotations={
                    SPARE_RESERVATION_ANNOTATION: INCIDENT,
                    SPARE_POOL_STATE_ANNOTATION: "ALLOCATED",
                },
            )
        if variation == "quarantine":
            value["taints"] = []
        elif variation == "owner":
            value["annotations"]["gpu-fault.io/incident-id"] = "different"
        elif variation == "reservation":
            value["annotations"][SPARE_RESERVATION_ANNOTATION] = "different"
        elif variation == "pool":
            value["annotations"][SPARE_POOL_STATE_ANNOTATION] = "AVAILABLE"
        else:
            value["unschedulable"] = True
        original_wait = h.warm.wait_for_workflow

        def wait(**kwargs: Any) -> dict[str, Any]:
            result = original_wait(**kwargs)
            h.nodes[node] = value
            return result

        monkeypatch.setattr(h.warm, "wait_for_workflow", wait)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert any(expected in error for error in report["errors"]), report


def scenario_run(h: WarmHarness, tmp_path: Path, scenario: str) -> dict[str, Any]:
    return shortage.run_scenario(
        h.settings,
        warm=h.warm,
        regional=h.regional,
        case_dir=tmp_path,
        run_dir=tmp_path,
        attempt=2,
        scenario=scenario,
        maintenance_window_end=NOW + timedelta(hours=1),
        provider_baseline=deepcopy(h.provider),
        profile_version="profile-v1",
    )


@pytest.mark.parametrize(
    "scenario", ["no-spare", "topology-mismatch", "reserved-by-other"]
)
def test_safe_shortage_scenario_keeps_refusal_and_restores_after_terminal(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(shortage, tmp_path, monkeypatch)
    report = scenario_run(h, tmp_path, scenario)
    assert report["verdict"] == "PASS" and report["errors"] == [], report
    assert (
        report["scenario_cleanup"]["errors"] == report["fault_cleanup"]["errors"] == []
    ), report
    assert h.posted["replacement_strategy"] == "HEALTHY_WARM_SPARE_ONLY", h.posted
    names = [name for name, _ in h.calls]
    order = [
        "shortage.apply",
        "workload.submit",
        "workload.running",
        "replacement.post",
        "workflow.wait",
        "incident.idle",
        "workload.delete",
        "shortage.restore",
        "spares.release",
        "restore.create",
        "incident.close",
    ]
    assert [names.index(name) for name in order] == sorted(
        names.index(name) for name in order
    ), names
    assert h.nodes == {node: node_snapshot(node) for node in (FAULT, SPARE)}, h.nodes
    # The shortage incident is closed once its node is released, with the
    # evidence read from the restored node; nothing is left for the operator.
    closed = [payload for name, payload in h.calls if name == "incident.close"]
    assert [item["incident_id"] for item in closed] == [INCIDENT], closed
    assert closed[0]["evidence"][0]["node_id"] == FAULT, closed
    assert closed[0]["evidence"][0]["unschedulable"] is False, closed
    assert report["incident_close"][INCIDENT]["closed"] is True, report
    assert "residual_incidents" not in report, report


@pytest.mark.parametrize(
    "scenario", ["active-gpu-pod", "kubernetes-not-ready", "agent-unavailable"]
)
def test_bounded_shortage_is_refused_before_any_scenario_mutation(
    scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(shortage, tmp_path, monkeypatch)
    report = scenario_run(h, tmp_path, scenario)
    assert report["verdict"] == "FAIL", report
    assert "independent cancellation capability" in report["error"], report
    names = {name for name, _ in h.calls}
    assert not {"shortage.apply", "workload.submit", "replacement.post"} & names, (
        h.calls
    )


@pytest.mark.parametrize(
    "phase",
    [
        "shortage.apply",
        "workload.submit",
        "replacement.post",
        "workflow.wait",
        "workload.delete",
        "shortage.restore",
        "spares.release",
        "agent.reactivate",
        "restore.wait",
    ],
)
def test_shortage_phase_failure_does_not_skip_remaining_cleanup(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(shortage, tmp_path, monkeypatch)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    report = scenario_run(h, tmp_path, "reserved-by-other")
    assert report["verdict"] == "FAIL", report
    assert phase in str(report), report
    names = [name for name, _ in h.calls]
    assert "workload.delete" in names and "shortage.restore" in names, names
    if phase in {"replacement.post", "workflow.wait"}:
        assert report["incident_id_recovered_from_event"] is True, report
        assert names.index("incident.idle") < names.index("shortage.restore"), names


@pytest.mark.parametrize("lookup", ["missing", "error", "busy"])
def test_shortage_does_not_remove_the_shortage_while_commands_can_act(
    lookup: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(shortage, tmp_path, monkeypatch)
    h.failures["workflow.wait"] = TimeoutError("workflow lease unresolved")
    if lookup == "missing":
        h.lookup_incident = ""
    elif lookup == "error":
        h.failures["incident.lookup"] = RuntimeError("incident lookup failed")
    else:
        h.failures["incident.idle"] = TimeoutError("command remains leased")
    report = scenario_run(h, tmp_path, "no-spare")
    assert report["verdict"] == "FAIL" and report["cleanup_deferred"] is True, report
    names = {name for name, _ in h.calls}
    assert not {"workload.delete", "shortage.restore", "spares.release"} & names, (
        h.calls
    )
    assert SPARE_LABEL not in h.nodes[SPARE]["labels"], h.nodes


def test_partial_matrix_never_claims_formal_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(shortage, tmp_path, monkeypatch)
    h.plan(tmp_path)
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    assert [item["scenario"] for item in report["scenarios"]] == list(
        h.settings.scenarios
    ), report
    assert all(item["verdict"] == "PASS" for item in report["scenarios"]), report
    assert report["errors"] == [
        "selected scenarios do not cover the complete matrix"
    ], report
    assert report["prewarm_residuals"] == {"pods": False}, report


def test_matrix_stops_after_first_failed_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = WarmHarness(shortage, tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["workflow.wait"] = RuntimeError("no terminal workflow")
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert len(report["scenarios"]) == 1, report
    assert (
        "scenario matrix stopped before all selected scenarios" in report["errors"]
    ), report
    assert "one or more shortage scenarios failed" in report["errors"], report
    assert sum(name == "replacement.post" for name, _ in h.calls) == 1, h.calls
