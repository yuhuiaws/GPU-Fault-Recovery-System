from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import executor_env_window as env_window
from scripts.e2e.regional import run_destr014_branch_exhaustion as destr014
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata

yaml = importlib.import_module("yaml")

ROOT = Path(__file__).resolve().parents[2]
FAULT = "node-b"
SIBLING = "node-c"
INCIDENT = "inc-destr014-test"
REQUEST = "workflow-destr014-test"


def _step(
    operation: str,
    nodes: list[str],
    branch_id: str,
    *,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "execution_owner": "test",
        "node_ids": list(nodes),
        "gpu_uuids": [],
        "workload_ids": ["job"],
        "parameters": dict(parameters or {}),
        "depends_on_step_indexes": [],
        "branch_id": branch_id,
        "branch_node_ids": list(nodes) if branch_id.startswith("branch:") else [],
    }


def _execution(
    index: int,
    operation: str,
    status: str,
    *,
    error: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "step_index": index,
        "operation": operation,
        "status": status,
        "phase": "official",
        "adapter_operation_id": f"remote/{index}",
        "error": error,
        "details": dict(details or {}),
        "started_at": "2026-09-06T10:00:00+00:00",
        "updated_at": "2026-09-06T10:05:00+00:00",
    }


def happy_workflow() -> dict[str, Any]:
    b = f"branch:{FAULT}"
    c = f"branch:{SIBLING}"
    steps = [
        _step("FREEZE_EVIDENCE", [FAULT, SIBLING], "shared"),  # 0
        _step("STOP_WORKLOADS", [FAULT, SIBLING], "shared"),  # 1
        _step("MARK_UNSCHEDULABLE", [FAULT], b),  # 2
        _step("QUIESCE_GPU_SERVICES", [FAULT], b),  # 3
        _step("VERIFY_NO_GPU_CLIENTS", [FAULT], b),  # 4
        _step("RESET_GPU", [FAULT], b),  # 5
        _step("RESTORE_GPU_SERVICES", [FAULT], b),  # 6
        _step("VALIDATE_GPU", [FAULT], b),  # 7
        _step("RESTORE_SCHEDULING", [FAULT], b),  # 8
        _step("MARK_UNSCHEDULABLE", [SIBLING], c),  # 9
        _step("RESTART_NODE", [SIBLING], c),  # 10
        _step("VALIDATE_GPU", [SIBLING], c),  # 11
        _step("VALIDATE_HOST", [SIBLING], c),  # 12
        _step("VALIDATE_FABRIC", [SIBLING], c),  # 13
        _step("RESTORE_SCHEDULING", [SIBLING], c),  # 14
        _step("RESTART_WORKLOAD", [FAULT, SIBLING], "join"),  # 15
        _step("RESTART_NODE", [FAULT], b),  # 16 escalation rung 1
        _step("VALIDATE_GPU", [FAULT], b),  # 17
        _step("VALIDATE_HOST", [FAULT], b),  # 18
        _step("VALIDATE_FABRIC", [FAULT], b),  # 19
        _step("RESTORE_SCHEDULING", [FAULT], b),  # 20
        _step(
            "REPLACE_NODE",
            [SIBLING],
            c,
            parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
        ),  # 21 escalation rung 1 for node-c
        _step("VALIDATE_GPU", [SIBLING], c),  # 22
        _step("VALIDATE_HOST", [SIBLING], c),  # 23
        _step("VALIDATE_FABRIC", [SIBLING], c),  # 24
        _step("RESTORE_SCHEDULING", [SIBLING], c),  # 25
    ]
    completed = [0, 1, 2, 3, 4, 9, 16, 17, 18, 19, 20]
    superseded = [5, 6, 7, 8, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25]
    executions = [
        _execution(0, "FREEZE_EVIDENCE", "SUCCEEDED"),
        _execution(1, "STOP_WORKLOADS", "SUCCEEDED"),
        _execution(2, "MARK_UNSCHEDULABLE", "SUCCEEDED"),
        _execution(3, "QUIESCE_GPU_SERVICES", "SUCCEEDED"),
        _execution(4, "VERIFY_NO_GPU_CLIENTS", "SUCCEEDED"),
        _execution(
            5,
            "RESET_GPU",
            "FAILED",
            error=(
                "barrier failed: node-b: GPU device clients are still active: "
                "GPU-1:4242:python3"
            ),
            details={"gpu_reset_commit_attempt": 6, "barrier_state": "FAILED"},
        ),
        _execution(9, "MARK_UNSCHEDULABLE", "SUCCEEDED"),
        _execution(
            10,
            "RESTART_NODE",
            "FAILED",
            error=(
                "step 10/RESTART_NODE stayed non-terminal for 305s, past the "
                "300s per-step cap"
            ),
            details={"step_waiting_seconds": 305, "step_waiting_timeout_seconds": 300},
        ),
        _execution(
            16,
            "RESTART_NODE",
            "SUCCEEDED",
            details={"confirmation_source": "agent-boot-id"},
        ),
        _execution(17, "VALIDATE_GPU", "SUCCEEDED"),
        _execution(18, "VALIDATE_HOST", "SUCCEEDED"),
        _execution(19, "VALIDATE_FABRIC", "SUCCEEDED"),
        _execution(20, "RESTORE_SCHEDULING", "SUCCEEDED"),
        _execution(
            21, "REPLACE_NODE", "FAILED", error="insufficient healthy HyperPod spares"
        ),
    ]
    return {
        "request_id": REQUEST,
        "incident_id": INCIDENT,
        "status": "FAILED",
        "dag_enabled": True,
        "official_steps": steps,
        "completed_step_indexes": completed,
        "superseded_step_indexes": superseded,
        "step_executions": executions,
        "branch_escalation_counts": {FAULT: 1, SIBLING: 1},
        "exhausted_branch_ids": [f"branch:{SIBLING}"],
        "terminal_failure_reason": None,
        "completed_operations": [
            "FREEZE_EVIDENCE",
            "STOP_WORKLOADS",
            "MARK_UNSCHEDULABLE",
            "QUIESCE_GPU_SERVICES",
            "VERIFY_NO_GPU_CLIENTS",
            "RESTART_NODE",
            "VALIDATE_GPU",
            "VALIDATE_HOST",
            "VALIDATE_FABRIC",
            "RESTORE_SCHEDULING",
        ],
    }


def happy_incident() -> dict[str, Any]:
    return {
        "incident_id": INCIDENT,
        "state": "QUARANTINED",
        "node_ids": [FAULT, SIBLING],
        "workflow_request_id": REQUEST,
        "job_id": "destr014-job",
    }


def _errors(workflow: dict[str, Any], incident: dict[str, Any]) -> list[str]:
    return destr014.workflow_errors(
        workflow,
        incident,
        fault_node=FAULT,
        sibling_node=SIBLING,
        failure_reason=f"node branch escalation exhausted: branch:{SIBLING}",
    )


def test_confirmation_token_matches_the_case_contract_derivation() -> None:
    metadata = RegionalCaseMetadata(
        case_id=destr014.CASE_ID,
        title="",
        category="regional-destructive-acceptance",
        level="staging",
        risk="destructive-provider-reboot",
        automation="manual",
        procedure="x",
        predecessor=destr014.PREDECESSOR_CASE_ID,
    )
    assert destr014.CONFIRMATION == metadata.confirmation
    assert destr014.CONFIRMATION == "DESTR014_EXECUTE"
    assert destr014.CASE_ID == "GF-REGIONAL-DESTR-014"
    assert destr014.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-008"


def test_destr014_workflow_contract_requires_sibling_exhaustion_without_restart() -> (
    None
):
    assert _errors(happy_workflow(), happy_incident()) == []


def test_a_join_that_ran_fails() -> None:
    workflow = happy_workflow()
    workflow["step_executions"].append(_execution(15, "RESTART_WORKLOAD", "SUCCEEDED"))
    workflow["completed_step_indexes"].append(15)
    errors = _errors(workflow, happy_incident())
    assert any("RESTART_WORKLOAD" in item for item in errors), errors


def test_a_missing_exhausted_id_fails() -> None:
    workflow = happy_workflow()
    workflow["exhausted_branch_ids"] = []
    errors = _errors(workflow, happy_incident())
    assert any("exhausted" in item for item in errors), errors


def test_an_exhausted_id_for_the_wrong_node_fails() -> None:
    workflow = happy_workflow()
    workflow["exhausted_branch_ids"] = [f"branch:{FAULT}"]
    errors = _errors(workflow, happy_incident())
    assert any("exhausted" in item for item in errors), errors


def test_a_reset_that_failed_for_another_reason_fails() -> None:
    workflow = happy_workflow()
    workflow["step_executions"][5]["error"] = (
        "GPU reset is disabled by node configuration"
    )
    errors = _errors(workflow, happy_incident())
    assert any("clients are still active" in item for item in errors), errors


def test_the_agreed_escalation_counts_are_pinned() -> None:
    workflow = happy_workflow()
    workflow["branch_escalation_counts"] = {FAULT: 1, SIBLING: 2}
    errors = _errors(workflow, happy_incident())
    assert any("branch_escalation_counts" in item for item in errors), errors


def test_the_sibling_reboot_must_fail_by_bounded_waiting() -> None:
    workflow = happy_workflow()
    workflow["step_executions"][7]["details"] = {}
    workflow["step_executions"][7]["error"] = "HyperPod preflight failed"
    errors = _errors(workflow, happy_incident())
    assert any("step_waiting_timeout_seconds" in item for item in errors), errors


def test_the_sibling_replacement_must_fail_for_want_of_a_spare() -> None:
    workflow = happy_workflow()
    workflow["step_executions"][-1]["error"] = "HyperPod preflight failed"
    errors = _errors(workflow, happy_incident())
    assert any("insufficient healthy HyperPod spares" in item for item in errors), (
        errors
    )
    workflow = happy_workflow()
    workflow["official_steps"][21]["parameters"] = {}
    errors = _errors(workflow, happy_incident())
    assert any("HEALTHY_WARM_SPARE_ONLY" in item for item in errors), errors


def test_a_succeeded_workflow_or_recovered_incident_fails() -> None:
    workflow = happy_workflow()
    workflow["status"] = "SUCCEEDED"
    errors = _errors(workflow, happy_incident())
    assert any("FAILED" in item for item in errors), errors
    incident = happy_incident()
    incident["state"] = "RECOVERED"
    errors = _errors(happy_workflow(), incident)
    assert any("QUARANTINED" in item for item in errors), errors


def test_an_unresolved_non_join_step_fails() -> None:
    workflow = happy_workflow()
    workflow["superseded_step_indexes"].remove(25)
    errors = _errors(workflow, happy_incident())
    assert any("resolved" in item for item in errors), errors


def test_a_failure_reason_without_the_exhaustion_prefix_fails() -> None:
    errors = destr014.workflow_errors(
        happy_workflow(),
        happy_incident(),
        fault_node=FAULT,
        sibling_node=SIBLING,
        failure_reason="workflow lifetime exceeded",
    )
    assert any("node branch escalation exhausted" in item for item in errors), errors


def test_the_fault_node_branch_must_end_with_restore_scheduling() -> None:
    workflow = happy_workflow()
    workflow["step_executions"] = [
        item for item in workflow["step_executions"] if item["step_index"] != 20
    ]
    workflow["completed_step_indexes"].remove(20)
    workflow["superseded_step_indexes"].append(20)
    errors = _errors(workflow, happy_incident())
    assert any("RESTORE_SCHEDULING" in item and FAULT in item for item in errors), (
        errors
    )


def test_quarantine_taint_value_is_the_incident_digest() -> None:
    assert (
        destr014.quarantine_taint_value("inc-1")
        == hashlib.sha256(b"inc-1").hexdigest()[:24]
    )


def _node(name: str, *, cordoned: bool, taints: list[dict[str, str]]) -> dict:
    return {
        "name": name,
        "uid": f"uid-{name}",
        "boot_id": "boot-2",
        "ready": "True",
        "unschedulable": cordoned,
        "taints": taints,
        "gpu_allocatable": "8",
        "ownership_annotations": (
            {"gpu-fault.io/incident-id": INCIDENT} if cordoned else {}
        ),
    }


def _quarantine_taint(incident_id: str = INCIDENT) -> dict[str, str]:
    return {
        "key": "gpu-fault.io/quarantined",
        "value": destr014.quarantine_taint_value(incident_id),
        "effect": "NoSchedule",
    }


def test_schedulability_happy_shape_passes() -> None:
    snapshots = {
        FAULT: _node(FAULT, cordoned=False, taints=[]),
        SIBLING: _node(SIBLING, cordoned=True, taints=[_quarantine_taint()]),
    }
    assert (
        destr014.schedulability_errors(
            snapshots, fault_node=FAULT, sibling_node=SIBLING, incident_id=INCIDENT
        )
        == []
    )


def test_schedulability_rejects_a_cordoned_fault_node_and_a_wrong_taint() -> None:
    snapshots = {
        FAULT: _node(FAULT, cordoned=True, taints=[_quarantine_taint()]),
        SIBLING: _node(SIBLING, cordoned=True, taints=[_quarantine_taint("other")]),
    }
    errors = destr014.schedulability_errors(
        snapshots, fault_node=FAULT, sibling_node=SIBLING, incident_id=INCIDENT
    )
    assert any(FAULT in item for item in errors), errors
    assert any("taint" in item and SIBLING in item for item in errors), errors
    snapshots[SIBLING] = _node(SIBLING, cordoned=False, taints=[])
    errors = destr014.schedulability_errors(
        snapshots, fault_node=FAULT, sibling_node=SIBLING, incident_id=INCIDENT
    )
    assert any("cordon" in item and SIBLING in item for item in errors), errors


def _event(name: str) -> dict[str, str]:
    return {
        "event_name": name,
        "event_time": "t",
        "username": "executor",
        "session_issuer_role_name": "executor-role",
    }


def test_cloudtrail_requires_exactly_two_reboots_and_no_replace() -> None:
    two = [_event("BatchRebootClusterNodes"), _event("BatchRebootClusterNodes")]
    assert destr014.cloudtrail_errors(two, FAULT, SIBLING) == []
    one = destr014.cloudtrail_errors(two[:1], FAULT, SIBLING)
    assert any("BatchRebootClusterNodes" in item for item in one), one
    three = destr014.cloudtrail_errors(
        [*two, _event("RebootClusterNodes")], FAULT, SIBLING
    )
    assert three, "three reboots must be refused"
    replaced = destr014.cloudtrail_errors(
        [*two, _event("BatchReplaceClusterNodes")], FAULT, SIBLING
    )
    assert any("BatchReplaceClusterNodes" in item for item in replaced), replaced
    deleted = destr014.cloudtrail_errors(
        [*two, _event("BatchDeleteClusterNodes")], FAULT, SIBLING
    )
    assert any("BatchDeleteClusterNodes" in item for item in deleted), deleted


def _ledger_row(command_id: str, operation: str, state: str, completed: str) -> dict:
    return {
        "command_id": command_id,
        "operation": operation,
        "state": state,
        "attempt": 1,
        "started_at": completed,
        "completed_at": completed,
    }


def _host_inputs() -> dict[str, Any]:
    baseline_rows = [
        _ledger_row(
            "old/4/VERIFY_NO_GPU_CLIENTS/c",
            "VERIFY_NO_GPU_CLIENTS",
            "SUCCEEDED",
            "2026-09-06T09:00:00+00:00",
        )
    ]
    after_rows = [
        *baseline_rows,
        _ledger_row(
            "wf/3/QUIESCE_GPU_SERVICES/c",
            "QUIESCE_GPU_SERVICES",
            "SUCCEEDED",
            "2026-09-06T10:00:10+00:00",
        ),
        _ledger_row(
            "wf/4/VERIFY_NO_GPU_CLIENTS/c",
            "VERIFY_NO_GPU_CLIENTS",
            "SUCCEEDED",
            "2026-09-06T10:00:20+00:00",
        ),
        _ledger_row(
            "wf/5/RESET_GPU/commit", "RESET_GPU", "FAILED", "2026-09-06T10:00:40+00:00"
        ),
    ]
    return {
        "fault_baseline": {"boot_id": "boot-b-1", "ledger": baseline_rows},
        "fault_after": {"boot_id": "boot-b-2", "ledger": after_rows},
        "sibling_baseline": {
            "boot_id": "boot-c-1",
            "agent_unit": {"UnitFileState": "enabled", "ActiveState": "active"},
        },
        "sibling_after": {
            "boot_id": "boot-c-2",
            "agent_unit": {"UnitFileState": "enabled", "ActiveState": "active"},
        },
        "holder_status": {
            "matched_row": {"command_id": "wf/4/VERIFY_NO_GPU_CLIENTS/c"},
            "hold_started_at": "2026-09-06T10:00:21+00:00",
            "hold_ended_at": "2026-09-06T10:03:00+00:00",
        },
        "sibling_agent_during": {"UnitFileState": "disabled", "ActiveState": "active"},
        "sibling_agent_after": {"UnitFileState": "enabled", "ActiveState": "active"},
    }


def test_host_happy_shape_passes() -> None:
    assert destr014.host_errors(**_host_inputs()) == []


def test_host_rejects_unchanged_boot_ids() -> None:
    inputs = _host_inputs()
    inputs["fault_after"]["boot_id"] = "boot-b-1"
    inputs["sibling_after"]["boot_id"] = "boot-c-1"
    errors = destr014.host_errors(**inputs)
    assert any("boot" in item and "fault" in item for item in errors), errors
    assert any("boot" in item and "sibling" in item for item in errors), errors


def test_host_rejects_an_agent_that_was_never_disabled_or_never_restored() -> None:
    inputs = _host_inputs()
    inputs["sibling_agent_during"] = {
        "UnitFileState": "enabled",
        "ActiveState": "active",
    }
    errors = destr014.host_errors(**inputs)
    assert any("disabled" in item for item in errors), errors
    inputs = _host_inputs()
    inputs["sibling_agent_after"] = {
        "UnitFileState": "disabled",
        "ActiveState": "inactive",
    }
    errors = destr014.host_errors(**inputs)
    assert any("restored" in item for item in errors), errors


def test_host_rejects_a_reset_that_succeeded_or_a_holder_that_never_armed() -> None:
    inputs = _host_inputs()
    inputs["fault_after"]["ledger"].append(
        _ledger_row(
            "wf/5/RESET_GPU/commit2",
            "RESET_GPU",
            "SUCCEEDED",
            "2026-09-06T10:01:00+00:00",
        )
    )
    errors = destr014.host_errors(**inputs)
    assert any("RESET_GPU" in item for item in errors), errors
    inputs = _host_inputs()
    inputs["holder_status"] = {"matched_row": None, "hold_started_at": None}
    errors = destr014.host_errors(**inputs)
    assert any("holder" in item for item in errors), errors
    inputs = _host_inputs()
    inputs["fault_after"]["ledger"] = inputs["fault_after"]["ledger"][:2]
    errors = destr014.host_errors(**inputs)
    assert any("RESET_GPU" in item for item in errors), errors


def test_injection_must_land_in_one_dag_workflow() -> None:
    fault_state = {"workflow": {"request_id": REQUEST, "dag_enabled": True}}
    sibling_state = {"workflow": {"request_id": REQUEST, "dag_enabled": True}}
    assert destr014.injection_errors(fault_state, sibling_state) == []
    other = {"workflow": {"request_id": "other", "dag_enabled": True}}
    errors = destr014.injection_errors(fault_state, other)
    assert any("workflow_request_id" in item for item in errors), errors
    flat = {"workflow": {"request_id": REQUEST, "dag_enabled": False}}
    errors = destr014.injection_errors(flat, copy.deepcopy(flat))
    assert any("dag_enabled" in item for item in errors), errors


def test_follow_up_requires_a_support_escalation_scoped_to_the_sibling() -> None:
    follow_up = {
        "incident": {
            "incident_id": f"inc-support-after-{REQUEST}",
            "node_ids": [SIBLING],
            "effective_action": "ESCALATE_OPERATOR",
            "reasons": ["replacement remediation failed; ..."],
        },
        "workflow": {
            "official_steps": [
                {"operation": "FREEZE_EVIDENCE"},
                {"operation": "ESCALATE_SUPPORT"},
            ]
        },
    }
    assert (
        destr014.follow_up_errors(follow_up, fault_node=FAULT, sibling_node=SIBLING)
        == []
    )
    missing = destr014.follow_up_errors(
        {"incident": None, "workflow": None}, fault_node=FAULT, sibling_node=SIBLING
    )
    assert any("follow-up" in item for item in missing), missing
    widened = copy.deepcopy(follow_up)
    widened["incident"]["node_ids"] = [FAULT, SIBLING]
    errors = destr014.follow_up_errors(widened, fault_node=FAULT, sibling_node=SIBLING)
    assert any(FAULT in item for item in errors), errors


def test_workload_must_be_gone_and_not_recreated() -> None:
    assert (
        destr014.workload_errors(
            pods=[],
            restart_budget={"budget": 1, "restart_count": 0},
            source_uids={"a", "b"},
        )
        == []
    )
    errors = destr014.workload_errors(
        pods=[{"uid": "z", "phase": "Running", "node": FAULT}],
        restart_budget={"budget": 1, "restart_count": 1},
        source_uids={"a", "b"},
    )
    assert any("Pod" in item for item in errors), errors
    assert any("restart" in item for item in errors), errors


def test_step_transitions_record_only_changes() -> None:
    previous: dict[str, str] = {}
    current, changes = destr014.step_transitions(
        previous,
        [
            _execution(5, "RESET_GPU", "WAITING"),
            _execution(10, "RESTART_NODE", "WAITING"),
        ],
    )
    assert len(changes) == 2
    again, changes = destr014.step_transitions(
        current,
        [
            _execution(5, "RESET_GPU", "FAILED"),
            _execution(10, "RESTART_NODE", "WAITING"),
        ],
    )
    assert [item["operation"] for item in changes] == ["RESET_GPU"]
    assert changes[0]["from"] == "WAITING"
    assert changes[0]["to"] == "FAILED"
    assert again["5:RESET_GPU"] == "FAILED"


def test_lifetime_arithmetic_refuses_an_estimate_over_the_lifetime() -> None:
    estimate = destr014.estimated_duration_seconds(
        verify_max_attempts=6,
        poll_interval_seconds=5.0,
        managed_recovery_timeout_seconds=300,
    )
    assert estimate == (
        6 * 5
        + 300
        + 2 * destr014.REBOOT_ALLOWANCE_SECONDS
        + destr014.VALIDATION_ALLOWANCE_SECONDS
        + destr014.CONTAINMENT_ALLOWANCE_SECONDS
    )
    assert (
        destr014.lifetime_errors(estimated_seconds=estimate, lifetime_seconds=3600)
        == []
    )
    refused = destr014.lifetime_errors(
        estimated_seconds=estimate, lifetime_seconds=estimate
    )
    assert any("lifetime" in item for item in refused), refused
    unknown = destr014.lifetime_errors(
        estimated_seconds=estimate, lifetime_seconds=None
    )
    assert any("lifetime" in item for item in unknown), unknown
    large = destr014.estimated_duration_seconds(
        verify_max_attempts=60,
        poll_interval_seconds=5.0,
        managed_recovery_timeout_seconds=1800,
    )
    assert destr014.lifetime_errors(estimated_seconds=large, lifetime_seconds=3600), (
        "the shipped defaults must not fit inside a one-hour lifetime"
    )


def test_budget_headroom_fails_closed() -> None:
    full = {
        "readable": True,
        "scopes": {
            "region": {"limit": 20, "active": 0},
            "class:c:NODE_REBOOT": {"limit": 2, "active": 2},
        },
    }
    errors = destr014.budget_headroom_errors(full)
    assert any("class:c:NODE_REBOOT" in item for item in errors), errors
    unknown = destr014.budget_headroom_errors({"readable": False, "scopes": {}})
    assert any("budget_headroom_unknown" in item for item in unknown), unknown
    assert (
        destr014.budget_headroom_errors(
            {"readable": True, "scopes": {"region": {"limit": 20, "active": 1}}}
        )
        == []
    )
    assert destr014.budget_headroom_errors({"readable": True, "scopes": {}}), (
        "no scopes at all means the probe did not compute the claims"
    )


def _node_snapshot(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "uid": f"uid-{name}",
        "boot_id": f"boot-{name}",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "gpu_allocatable": "8",
        "ownership_annotations": {},
    }


def _warm_node(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "uid": f"uid-{name}",
        "ready": "True",
        "gpu_allocatable": 8,
        "unschedulable": False,
        "taints": [],
        "labels": {
            "gpu-fault.io/spare": None,
            "sagemaker.amazonaws.com/node-health-status": "Schedulable",
            "sagemaker.amazonaws.com/instance-group-name": "gpu",
            "node.kubernetes.io/instance-type": "ml.p5en.48xlarge",
        },
        "annotations": {},
    }


def _executor_pod(name: str) -> dict[str, str | None]:
    return {
        "pod": name,
        "spare_failover": "true",
        "remote_state": "true",
        "allow_replace": "false",
        "allow_reboot": "true",
        "spare_label": "gpu-fault.io/spare",
        "verify_max_attempts": None,
        "managed_recovery_timeout": None,
    }


def _happy_preflight_kwargs() -> dict[str, Any]:
    return {
        "fault_node": FAULT,
        "sibling_node": SIBLING,
        "fault": _warm_node(FAULT),
        "sibling": _warm_node(SIBLING),
        "fault_agent": {"lifecycle_state": "ACTIVE", "boot_id": "boot-b-1"},
        "sibling_agent": {"lifecycle_state": "ACTIVE", "boot_id": "boot-c-1"},
        "spare_nodes": [],
        "cluster": {"status": "InService", "node_recovery": "None"},
        "executor_env": [_executor_pod("exec-0"), _executor_pod("exec-1")],
        "reboot_probe_errors": [],
        "control_env": {
            "max_rungs": 2,
            "job_lifetime_seconds": 3600,
            "aggregation_window_seconds": 5,
            "poll_interval_seconds": 5.0,
        },
        "budget": {"readable": True, "scopes": {"region": {"limit": 20, "active": 0}}},
        "fault_workloads": [],
        "sibling_workloads": [],
        "open_workflows": [],
        "gpu_workloads": [],
        "predecessor": {"valid": True},
        "tests": {"passed": True},
        "verify_max_attempts": 6,
        "managed_recovery_timeout_seconds": 300,
    }


def test_preflight_happy_shape_passes() -> None:
    assert destr014.preflight_errors(**_happy_preflight_kwargs()) == []


def test_preflight_refuses_a_stray_spare_label() -> None:
    kwargs = _happy_preflight_kwargs()
    kwargs["spare_nodes"] = ["node-d"]
    errors = destr014.preflight_errors(**kwargs)
    assert any("spare" in item for item in errors), errors
    kwargs = _happy_preflight_kwargs()
    kwargs["fault"]["labels"]["gpu-fault.io/spare"] = "true"
    errors = destr014.preflight_errors(**kwargs)
    assert any("spare" in item for item in errors), errors


def test_preflight_refuses_disabled_reboot_and_low_rung_ceiling() -> None:
    kwargs = _happy_preflight_kwargs()
    kwargs["executor_env"][1]["allow_reboot"] = "false"
    errors = destr014.preflight_errors(**kwargs)
    assert any("reboot" in item for item in errors), errors
    kwargs = _happy_preflight_kwargs()
    kwargs["control_env"]["max_rungs"] = 1
    errors = destr014.preflight_errors(**kwargs)
    assert any("GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS" in item for item in errors), (
        errors
    )
    kwargs = _happy_preflight_kwargs()
    kwargs["reboot_probe_errors"] = ["HyperPod NodeRecovery is not None"]
    errors = destr014.preflight_errors(**kwargs)
    assert "HyperPod NodeRecovery is not None" in errors


def test_preflight_refuses_busy_nodes_open_workflows_and_full_budget() -> None:
    kwargs = _happy_preflight_kwargs()
    kwargs["fault_workloads"] = [{"namespace": "kube-system", "name": "coredns"}]
    kwargs["open_workflows"] = [{"request_id": "wf-open", "status": "RUNNING"}]
    kwargs["budget"] = {"readable": False, "scopes": {}}
    kwargs["gpu_workloads"] = [{"node": FAULT, "name": "other"}]
    errors = destr014.preflight_errors(**kwargs)
    assert any("critical" in item or "system" in item for item in errors), errors
    assert any("open workflow" in item for item in errors), errors
    assert any("budget_headroom_unknown" in item for item in errors), errors
    assert any("GPU workload" in item for item in errors), errors


def test_preflight_refuses_an_estimate_that_does_not_fit_the_lifetime() -> None:
    kwargs = _happy_preflight_kwargs()
    kwargs["managed_recovery_timeout_seconds"] = 3600
    errors = destr014.preflight_errors(**kwargs)
    assert any("lifetime" in item for item in errors), errors


def test_preflight_requires_the_same_topology_and_a_none_recovery_cluster() -> None:
    kwargs = _happy_preflight_kwargs()
    kwargs["sibling"]["labels"]["node.kubernetes.io/instance-type"] = "ml.p5.48xlarge"
    kwargs["cluster"]["node_recovery"] = "Automatic"
    kwargs["sibling_node"] = FAULT
    errors = destr014.preflight_errors(**kwargs)
    assert any("topology" in item for item in errors), errors
    assert any("NodeRecovery" in item for item in errors), errors
    assert any("identical" in item for item in errors), errors


def test_plan_identity_digest_is_stable_across_key_order() -> None:
    preflight = {
        "release_id": "rel-1",
        "fault_node": {"uid": "uid-b", "boot_id": "boot-b"},
        "sibling_node": {"uid": "uid-c", "boot_id": "boot-c"},
        "store": {"profile": {"profile_version": "hyperpod-v9"}},
        "provider_inventory": {"sha256": "abc"},
        "executor_env_window": {"baseline_sha256": "def"},
    }
    identity = destr014.plan_identity(preflight)
    reordered = {key: identity[key] for key in reversed(sorted(identity))}
    assert destr014.identity_digest(identity) == destr014.identity_digest(reordered)
    drifted = dict(identity)
    drifted["release_id"] = "rel-2"
    assert destr014.identity_digest(identity) != destr014.identity_digest(drifted)
    assert set(identity) >= {
        "release_id",
        "fault_node_uid",
        "fault_node_boot_id",
        "sibling_node_uid",
        "sibling_node_boot_id",
        "runtime_profile_version",
        "provider_inventory_sha256",
    }


def test_two_node_manifest_pins_master_and_worker_to_the_two_nodes(
    tmp_path: Path,
) -> None:
    destination = destr014.render_two_node_manifest(
        destr014.DEFAULT_MANIFEST,
        tmp_path / "pinned.yaml",
        master_node=FAULT,
        worker_node=SIBLING,
    )
    document = yaml.safe_load(destination.read_text(encoding="utf-8"))
    replicas = document["spec"]["pytorchReplicaSpecs"]
    assert replicas["Master"]["replicas"] == 1
    assert replicas["Worker"]["replicas"] == 1
    assert replicas["Master"]["template"]["spec"]["nodeName"] == FAULT
    assert replicas["Worker"]["template"]["spec"]["nodeName"] == SIBLING
    assert destination.stat().st_mode & 0o077 == 0


def test_two_node_manifest_refuses_the_same_node_twice(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        destr014.render_two_node_manifest(
            destr014.DEFAULT_MANIFEST,
            tmp_path / "pinned.yaml",
            master_node=FAULT,
            worker_node=FAULT,
        )


def test_derived_identity_is_deterministic_per_run_and_attempt(tmp_path: Path) -> None:
    first = destr014.derived_identity(tmp_path, 1)
    assert first == destr014.derived_identity(tmp_path, 1)
    assert first != destr014.derived_identity(tmp_path, 2)
    assert first[0].startswith("destr014-"), first
    assert first[1] == f"{first[0]}-a001"


def test_evidence_components_carry_digests_only() -> None:
    components = destr014.evidence_components(
        {
            "preflight_identity": {"release_id": "rel"},
            "workflow": {"request_id": REQUEST, "status": "FAILED"},
            "incident": {"incident_id": INCIDENT},
            "provider_events": [_event("BatchRebootClusterNodes")],
        }
    )
    assert set(components) == {
        "preflight_identity",
        "workflow",
        "incident",
        "provider_events",
    }
    for value in components.values():
        assert len(value) == 64, value
        int(value, 16)
    digest = destr014.case_digest(components)
    assert len(digest) == 64
    assert digest == destr014.case_digest(dict(reversed(list(components.items()))))
    assert REQUEST not in json.dumps(components)


def test_parser_accepts_the_documented_arguments() -> None:
    arguments = destr014.parser().parse_args(
        [
            "--run-dir",
            "/tmp/destr014-run",
            "--attempt",
            "2",
            "--plan",
            "--cpu-kubeconfig",
            "/tmp/cpu",
            "--gpu-kubeconfig",
            "/tmp/gpu",
            "--gpu-context",
            "ctx",
            "--namespace",
            "gpu-fault-system",
            "--cluster-id",
            "cluster-a",
            "--region",
            "us-west-2",
            "--site-file",
            "/tmp/site.yaml",
            "--hyperpod-cluster",
            "hp",
            "--host-probe-image",
            "img",
            "--predecessor-evidence",
            "/tmp/pred.json",
            "--job-id",
            "job",
            "--attempt-id",
            "job-a001",
            "--fault-node",
            FAULT,
            "--fault-pci-bdf",
            "0000:53:00",
            "--fault-device",
            "/dev/nvidia0",
            "--sibling-node",
            SIBLING,
            "--sibling-pci-bdf",
            "0000:64:00",
            "--verify-max-attempts",
            "6",
            "--managed-recovery-timeout-seconds",
            "300",
            "--variant",
            "sibling-exhausted",
            "--maintenance-window-end",
            "2026-09-06T12:00:00Z",
        ]
    )
    assert arguments.execute is False
    assert arguments.plan is True
    assert arguments.variant == "sibling-exhausted"
    assert arguments.verify_max_attempts == 6
    assert arguments.managed_recovery_timeout_seconds == 300
    defaults = destr014.parser().parse_args(["--run-dir", "/tmp/destr014-run"])
    assert defaults.execute is False
    assert defaults.variant == "sibling-exhausted"
    assert defaults.verify_max_attempts == 6
    assert defaults.managed_recovery_timeout_seconds == 300
    assert hasattr(defaults, "predecessor_evidence"), defaults
    with pytest.raises(SystemExit):
        destr014.parser().parse_args(
            ["--run-dir", "/tmp/destr014-run", "--variant", "warm-spare"]
        )


def test_env_window_allowlist_and_assignments() -> None:
    assignments = env_window.parse_assignments(
        [
            "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS=6",
            "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS=300",
        ]
    )
    assert assignments == {
        "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": "6",
        "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS": "300",
    }
    for bad in (
        ["GPU_FAULT_ALLOW_HYPERPOD_REBOOT=false"],
        ["GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS="],
        ["GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS=six"],
        ["GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS"],
        ["GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS=0"],
    ):
        with pytest.raises(env_window.RegionalFixtureError):
            env_window.parse_assignments(bad)
    assert env_window.DEPLOYMENT == "gpu-fault-cluster-executor"
    assert env_window.CONTAINER == "executor"
    assert env_window.OPEN_CONFIRMATION == "OPEN_EXECUTOR_ENV_WINDOW"
    assert env_window.CLOSE_CONFIRMATION == "CLOSE_EXECUTOR_ENV_WINDOW"


def test_env_window_restores_unset_as_delete_not_false() -> None:
    baseline = {
        "variables": {
            "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": {
                "present": False,
                "value": None,
            },
            "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS": {
                "present": True,
                "value": "1800",
            },
        }
    }
    assert env_window.restore_arguments(baseline) == [
        "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS-",
        "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS=1800",
    ]
    assert env_window.open_arguments(
        {"GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": "6"}
    ) == ["GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS=6"]


def test_env_window_convergence_requires_every_ready_replica() -> None:
    expected = {"GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": "6"}
    settled = [
        {"pod": "a", "values": {"GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": "6"}},
        {"pod": "b", "values": {"GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": "6"}},
    ]
    assert env_window.converged(settled, expected) is True
    lagging = [
        settled[0],
        {"pod": "b", "values": {"GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": "60"}},
    ]
    assert env_window.converged(lagging, expected) is False
    assert env_window.converged([], expected) is False
    restored = {"GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": None}
    assert (
        env_window.converged(
            [
                {
                    "pod": "a",
                    "values": {"GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": None},
                }
            ],
            restored,
        )
        is True
    )


def test_env_window_parser_is_read_only_by_default() -> None:
    parser = env_window.parser()
    arguments = parser.parse_args(["--baseline", "/tmp/b.json"])
    assert arguments.open is False
    assert arguments.close is False
    arguments = parser.parse_args(
        [
            "--baseline",
            "/tmp/b.json",
            "--open",
            "--confirm",
            "OPEN_EXECUTOR_ENV_WINDOW",
            "--set",
            "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS=6",
            "--set",
            "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS=300",
        ]
    )
    assert arguments.open is True
    assert len(arguments.set) == 2
    with pytest.raises(SystemExit):
        parser.parse_args(["--baseline", "/tmp/b.json", "--open", "--close"])


def test_env_window_open_refuses_unrecorded_live_values_and_resumes_its_own() -> None:
    baseline = {
        "variables": {
            "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": {
                "present": False,
                "value": None,
            }
        }
    }
    live_changed = {
        "variables": {
            "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": {"present": True, "value": "6"}
        }
    }
    assignments = {"GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": "6"}
    assert env_window.open_decision(None, live_changed, assignments) == "refuse"
    assert env_window.open_decision(None, baseline, assignments) == "open"
    record = {"opened_at": "x", "baseline": baseline, "assignments": assignments}
    assert env_window.open_decision(record, live_changed, assignments) == "resume"
    assert env_window.open_decision(record, baseline, assignments) == "refuse"
    closed = {**record, "closed_at": "y"}
    assert env_window.open_decision(closed, baseline, assignments) == "open"
    other = {"GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": "7"}
    assert env_window.open_decision(record, live_changed, other) == "refuse"
