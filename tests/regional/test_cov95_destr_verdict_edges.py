from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr014_verdicts as v14
from scripts.e2e.regional import destr015_verdicts as v15
from scripts.e2e.regional import destr016_verdicts as v16
from scripts.e2e.regional import destr017_verdicts as v17
from scripts.e2e.regional import destr018_verdicts as v18
from scripts.e2e.regional import destr019_verdicts as v19
from scripts.e2e.regional import destr020_verdicts as v20
from scripts.e2e.regional import destr021_verdicts as v21
from scripts.e2e.regional import destr023_verdicts as v23
from tests.regional import test_destr014_branch_exhaustion as d14
from tests.regional import test_destr015_parallel_branch_join as d15
from tests.regional import test_destr016_preempting_reboot as d16
from tests.regional import test_destr017_out_of_band_reboot_fence as d17
from tests.regional._cov95_destr_agent import (
    COMMAND_ID,
    GENERATION,
    INCIDENT,
    NODE,
    WORKFLOW,
    AgentHarness,
    audit,
    health,
    journal,
)
from tests.regional._cov95_destr_branches import branch_host
from tests.regional._cov95_destr_edge_data import change
from tests.regional._cov95_destr_metadata import recovery_bundle
from tests.regional.test_destr023_idle_cluster_reset import coverage


@pytest.mark.parametrize("module", [v18, v19, v20, v21])
@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "unparseable",
        datetime(2026, 9, 12),
        datetime(2026, 9, 12, tzinfo=timezone.utc),
    ],
)
def test_timestamp_parsers_preserve_declared_naive_time_policy(
    module: Any, value: Any
) -> None:
    result = module.parse_time(value)
    if not isinstance(value, datetime):
        assert result is None, (module.CASE_ID, value, result)
    elif value.tzinfo is None and module is v18:
        assert result is None, "lifetime evidence requires explicit timezone"
    else:
        assert result == value.replace(tzinfo=timezone.utc), (
            module.CASE_ID,
            value,
            result,
        )


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("node", "ready"), "False", "not Ready"),
        (("node", "taints"), [{"key": "foreign"}], "pre-existing taints"),
        (("node", "ownership_annotations"), {"owner": "foreign"}, "workflow ownership"),
        (("agent", "generation"), None, "no generation"),
        (("profile", "warnings"), ["drift"], "profile has warnings"),
        (("host", "agent", "ActiveState"), "inactive", "unit is not active"),
        (("host", "ledger", "present"), False, "ledger is absent"),
    ],
)
def test_agent_audit_preflight_rejects_unproven_identity_and_ledger(
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    frame = {
        "node": h.node,
        "agent": h.agent,
        "profile": {"warnings": []},
        "workloads": [],
        "queue": h.queue,
        "remote_commands": h.commands,
        "recent_event": None,
        "host": h.host,
        "tests_passed": True,
    }
    assert v19.preflight_errors(**frame) == [], frame
    change(frame, path, value)
    errors = v19.preflight_errors(**frame)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("event", "xid"), 46, "not XID"),
        (("decision", "official_action"), "RESET_GPU", "RESTART_FM"),
        (
            ("workflow", "official_steps", 0, "execution_owner"),
            "foreign",
            "workflow owners",
        ),
        (("workflow", "official_steps"), [], "workflow steps"),
        (("commands",), [], "command"),
    ],
)
def test_agent_audit_requires_the_exact_fabric_restart_workflow(
    path: tuple[str | int, ...],
    value: Any,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = AgentHarness(tmp_path, monkeypatch)
    assert v19.workflow_errors(h.state, node=NODE) == [], h.state
    change(h.state, path, value)
    errors = v19.workflow_errors(h.state, node=NODE)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("rows", 0, "attempt"), 2, "attempt"),
        (("rows", 0, "state"), "FAILED", "state"),
        (("rows", 0, "operation"), "RESET_GPU", "operation"),
        (("rows", 0, "started_at"), "bad", "ordered"),
        (("rows", 0, "gpu_uuids_present"), False, "gpu_uuids column"),
        (("rows", 0, "completed_at"), None, "ordered"),
        (("rows", 0, "exit_code"), 7, "SUCCEEDED row"),
    ],
)
def test_agent_ledger_audit_rejects_wrong_attempt_outcome_and_incomplete_fields(
    path: tuple[str | int, ...], value: Any, expected: str
) -> None:
    document = audit()
    kwargs = {
        "command_id": COMMAND_ID,
        "incident_id": INCIDENT,
        "workflow_request_id": WORKFLOW,
        "fencing_token": 2,
        "baseline_interrupted": 0,
    }
    assert v19.ledger_row_errors(document, **kwargs) == [], document
    change(document, path, value)
    errors = v19.ledger_row_errors(document, **kwargs)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("field", "expected"),
    [("gpu_count", "numeric gpu_count"), ("duration_ms", "numeric duration_ms")],
)
def test_agent_journal_requires_numeric_observation_fields(
    field: str, expected: str
) -> None:
    lines = journal()["lines"]
    lines[-1]["fields"][field] = "unavailable"
    errors = v19.journal_errors(
        lines,
        command_id=COMMAND_ID,
        incident_id=INCIDENT,
        workflow_request_id=WORKFLOW,
        node=NODE,
        fencing_token=2,
        generation=GENERATION,
    )
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("change_fields", "expected"),
    [
        ({"configured": False}, "configured heartbeat"),
        ({"consecutive_failures": 1}, "consecutive_failures"),
    ],
)
def test_agent_health_requires_configured_successful_heartbeat(
    change_fields: dict[str, Any], expected: str
) -> None:
    document = health(dict.fromkeys(v19.COUNTER_NAMES, 0))
    document["payload"]["heartbeat"].update(change_fields)
    errors = v19.health_errors(document, label="unit")
    assert any(expected in error for error in errors), errors
    assert (
        v19.health_errors(document, label="offline", heartbeat_required=False) == []
    ), document


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("dag_enabled",), False, "not dag_enabled"),
        (("step_executions", 5, "status"), "SUCCEEDED", "no FAILED RESET_GPU"),
        (("step_executions", 5, "details"), {}, "failure carries none"),
        (("step_executions", 7, "details"), {}, "confirmed no-start"),
        (
            ("official_steps", 21, "parameters", "replacement_strategy"),
            "PROVIDER",
            "HEALTHY_WARM_SPARE_ONLY",
        ),
        (("step_executions", 13, "status"), "SUCCEEDED", "no FAILED REPLACE_NODE"),
    ],
)
def test_exhaustion_verdict_requires_failure_at_the_specified_branch_rungs(
    path: tuple[str | int, ...], value: Any, expected: str
) -> None:
    workflow = d14.happy_workflow()
    kwargs = {
        "fault_node": d14.FAULT,
        "sibling_node": d14.SIBLING,
        "failure_reason": f"node branch escalation exhausted: branch:{d14.SIBLING}",
    }
    assert v14.workflow_errors(workflow, d14.happy_incident(), **kwargs) == [], workflow
    change(workflow, path, value)
    errors = v14.workflow_errors(workflow, d14.happy_incident(), **kwargs)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("terminal_failure_reason",), "failure", "failure reason"),
        (
            ("step_executions", 16, "status"),
            "FAILED",
            "RESTART_WORKLOAD is not SUCCEEDED",
        ),
        (("step_executions", 16, "details", "source_gpu_count"), 8, "source GPU count"),
        (("step_executions", 2, "started_at"), "bad-time", "unordered timestamps"),
        (
            ("step_executions", 2, "updated_at"),
            "2026-09-06T10:00:00",
            "unordered timestamps",
        ),
    ],
)
def test_parallel_verdict_refuses_failed_join_and_invalid_timing(
    path: tuple[str | int, ...], value: Any, expected: str
) -> None:
    workflow = d15.happy_workflow()
    kwargs = {
        "nodes": d15.NODES,
        "expected_gpu_count": 16,
        "restart_budget": d15.happy_budget(),
    }
    assert v15.workflow_errors(workflow, d15.happy_incident(), **kwargs) == [], workflow
    change(workflow, path, value)
    errors = v15.workflow_errors(workflow, d15.happy_incident(), **kwargs)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    "defect", ["no-gpu", "fabric-reset", "duplicate-command", "timer"]
)
def test_parallel_host_verdict_rejects_unexpected_physical_effects(defect: str) -> None:
    hosts = {
        node: {"before": branch_host(node), "after": branch_host(node, after=True)}
        for node in d15.NODES
    }
    target = hosts[d15.NODES[0]]["after"]
    if defect == "no-gpu":
        target["gpu_inventory"] = []
    elif defect == "fabric-reset":
        target["ledger"].append(
            {"command_id": "extra", "operation": "RESET_ALL_GPUS_NVSWITCHES"}
        )
    elif defect == "duplicate-command":
        target["ledger"].append(dict(target["ledger"][0]))
    else:
        target["gpu_fault_timers"] = ["unexpected.timer"]
    errors = v15.host_errors(hosts, nodes=d15.NODES, expected_gpu_count=8)
    assert errors, (defect, hosts)


def test_parallel_host_skips_inactive_optional_service_at_baseline() -> None:
    hosts = {
        node: {"before": branch_host(node), "after": branch_host(node, after=True)}
        for node in d15.NODES
    }
    hosts[d15.NODES[0]]["before"]["services"]["optional.service"] = {
        "ActiveState": "inactive"
    }
    assert v15.host_errors(hosts, nodes=d15.NODES, expected_gpu_count=8) == [], hosts


@pytest.mark.parametrize(
    "defect", ["incident-action", "terminal", "missing-barrier", "remote-status"]
)
def test_preemption_verdict_requires_a_running_unrestored_barrier(defect: str) -> None:
    state = d16.barrier_snapshot()
    if defect == "incident-action":
        state["incident"]["official_action"] = "RESTART_BM"
    elif defect == "terminal":
        state["workflow"]["status"] = "FAILED"
    elif defect == "missing-barrier":
        state["workflow"]["step_executions"].pop()
    else:
        state["workflow"]["step_executions"][-1]["details"]["remote_status"] = "PENDING"
    errors = v16.reset_workflow_errors(
        state["workflow"], state["incident"], state["decision"]
    )
    assert errors, (defect, state)


@pytest.mark.parametrize("defect", ["missing-verify", "bad-owner", "reboot-same"])
def test_reboot_fence_identity_and_local_ledger_proof_must_be_complete(
    defect: str,
) -> None:
    if defect == "missing-verify":
        after = d17.host_after()
        after["ledger"] = [
            row
            for row in after["ledger"]
            if row["operation"] != "VERIFY_NO_GPU_CLIENTS"
        ]
        errors = v17.ledger_errors(d17.host_before(), after, node=d17.NODE)
    else:
        before, after = d17.agent_before(), d17.agent_after()
        if defect == "bad-owner":
            after["node_id"] = "foreign"
        else:
            after["boot_id"] = before["boot_id"]
        errors = v17.agent_errors(before, after, node=d17.NODE)
    assert errors, (defect, errors)


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (
            (
                "remote_commands",
                0,
                "result_details",
                "node_baselines",
                "node-a",
                "before",
                "unschedulable",
            ),
            True,
            "schedulable node",
        ),
        (
            (
                "remote_commands",
                0,
                "result_details",
                "node_baselines",
                "node-a",
                "after",
                "unschedulable",
            ),
            False,
            "node cordoned",
        ),
        (
            (
                "remote_commands",
                0,
                "result_details",
                "node_baselines",
                "node-a",
                "after",
                "taint_keys",
            ),
            [],
            "quarantine taint",
        ),
        (
            (
                "remote_commands",
                0,
                "result_details",
                "node_baselines",
                "node-a",
                "after",
                "resource_version",
            ),
            "1",
            "re-read after",
        ),
        (
            (
                "remote_commands",
                2,
                "result_details",
                "node_baselines",
                "node-a",
                "before",
                "unschedulable",
            ),
            False,
            "node cordoned",
        ),
    ],
)
def test_metadata_baselines_must_prove_observed_isolation_and_release(
    path: tuple[str | int, ...], value: Any, expected: str
) -> None:
    bundle = recovery_bundle("foreign-test")
    assert v21.baseline_errors(bundle, node="node-a") == [], bundle
    change(bundle, path, value)
    errors = v21.baseline_errors(bundle, node="node-a")
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    "defect",
    [
        "missing-key",
        "future",
        "age-missing",
        "counts",
        "orphan-age",
        "observation-mismatch",
        "zero-with-time",
    ],
)
def test_coverage_heartbeat_schema_never_accepts_partial_or_inconsistent_evidence(
    defect: str,
) -> None:
    document = coverage()
    if defect == "missing-key":
        document.pop("latest_observation_at")
    elif defect == "future":
        document["heartbeat"]["observed_at"] = "2099-01-01T00:00:00+00:00"
    elif defect == "age-missing":
        document["heartbeat_age_seconds"] = None
    elif defect == "counts":
        document["heartbeat"]["watched_attempts"] = True
    elif defect == "orphan-age":
        document["heartbeat"] = None
    elif defect == "observation-mismatch":
        document.update(
            observation_count=1,
            latest_observation_at=document["probed_at"],
            latest_observation_age_seconds=50,
        )
    else:
        document["latest_observation_at"] = document["probed_at"]
    errors = v23.coverage_supported_errors(document)
    assert errors, (defect, document)


def test_coverage_heartbeat_age_is_not_inferred_from_an_unrelated_timestamp() -> None:
    document = coverage()
    document["heartbeat_age_seconds"] = 1
    errors = v23.coverage_supported_errors(document)
    assert "coverage heartbeat age disagrees with its timestamp" in errors, errors


def test_fence_final_host_allows_inactive_optional_service_but_not_clients() -> None:
    before, after = d17.host_before(), d17.host_after()
    before["services"]["optional.service"] = {"ActiveState": "inactive"}
    after["compute_clients"] = [{"pid": "100"}]
    errors = v17.host_final_errors(before, after, node=d17.NODE, expected_gpu_count=8)
    assert any("compute clients" in error for error in errors), errors
