"""Contract tests for GF-REGIONAL-DESTR-016.

Every verdict of the case is exercised against synthetic control-plane, node
and CloudTrail snapshots: once on a run that must pass, and once per way the
run can be wrong. Nothing here touches a cluster.

The fixtures encode the two shapes the case would otherwise assert wrongly:
the successor's step graph is rewired at claim time (the appended
``RESTORE_GPU_SERVICES`` is the *last* index and ``VALIDATE_GPU`` depends on
it), and the superseded reset's barrier step record stays WAITING while its
*remote command* is the thing that gets cancelled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr016_verdicts as verdicts
from scripts.e2e.regional import run_destr016_preempting_reboot as destr016
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata

NODE = "node-b"
INCIDENT = "inc-destr016-test"
RESET_ID = "wf-destr016-reset"
REBOOT_ID = "wf-destr016-reboot"
BARRIER_COMMAND = "cmd-destr016-verify"


def _at(minute: int, second: int = 0) -> str:
    return f"2026-09-06T10:{minute:02d}:{second:02d}+00:00"


def _step(
    operation: str,
    *,
    depends: list[int] | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "execution_owner": "regional-cluster-executor",
        "node_ids": [NODE],
        "gpu_uuids": [],
        "workload_ids": [],
        "parameters": dict(parameters or {}),
        "depends_on_step_indexes": list(depends or []),
    }


def _execution(
    index: int,
    operation: str,
    status: str,
    *,
    started_at: str,
    updated_at: str,
    error: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "step_index": index,
        "operation": operation,
        "status": status,
        "phase": "official",
        "adapter_operation_id": f"remote/{operation.lower()}",
        "error": error,
        "details": dict(details or {}),
        "started_at": started_at,
        "updated_at": updated_at,
    }


# --------------------------------------------------------------------------- #
# Phase 1 fixtures: the reset workflow parked at the dirty boundary
# --------------------------------------------------------------------------- #
def parked_reset_workflow() -> dict[str, Any]:
    steps = []
    for index, operation in enumerate(verdicts.RESET_OPERATIONS):
        steps.append(_step(operation, depends=[index - 1] if index else []))
    return {
        "request_id": RESET_ID,
        "incident_id": INCIDENT,
        "status": "RUNNING",
        "node_ids": [NODE],
        "official_steps": steps,
        "step_executions": [
            _execution(
                0,
                "FREEZE_EVIDENCE",
                "SUCCEEDED",
                started_at=_at(0),
                updated_at=_at(0, 20),
            ),
            _execution(
                1,
                "MARK_UNSCHEDULABLE",
                "SUCCEEDED",
                started_at=_at(0, 25),
                updated_at=_at(0, 40),
            ),
            _execution(
                2,
                "QUIESCE_GPU_SERVICES",
                "SUCCEEDED",
                started_at=_at(1),
                updated_at=_at(1, 50),
            ),
            _execution(
                3,
                verdicts.BARRIER_OPERATION,
                "WAITING",
                started_at=_at(2),
                updated_at=_at(3, 10),
                details={
                    "remote_command_id": BARRIER_COMMAND,
                    "remote_cluster_id": "cluster-a",
                    "remote_status": "WAITING",
                    "mutation_submitted_by_control_plane": False,
                },
            ),
        ],
    }


def parked_reset_incident() -> dict[str, Any]:
    return {
        "incident_id": INCIDENT,
        "node_id": NODE,
        "state": "RECOVERING",
        "official_action": "RESET_GPU",
        "workflow_request_id": RESET_ID,
        "reasons": [f"{NODE} XID 46 GPU stopped processing"],
    }


def barrier_commands() -> list[dict[str, Any]]:
    return [
        {
            "command_id": BARRIER_COMMAND,
            "workflow_request_id": RESET_ID,
            "step_index": 3,
            "step": {"operation": verdicts.BARRIER_OPERATION, "node_ids": [NODE]},
            "status": "WAITING",
            "status_source": "executor-result",
            "error": None,
            "result_details": {
                "gpu_client_quiesce_attempt": 4,
                "waiting_node": NODE,
                "reason": (
                    "GPU compute clients are still active: /dev/nvidia0 pid 4242"
                ),
            },
        }
    ]


def barrier_snapshot() -> dict[str, Any]:
    return {
        "workflow": parked_reset_workflow(),
        "incident": parked_reset_incident(),
        "decision": {"official_action": "RESET_GPU"},
        "commands": barrier_commands(),
        "event": {"xid": 46, "node_id": NODE},
    }


def absorbed_snapshot() -> dict[str, Any]:
    state = barrier_snapshot()
    state["incident"]["reasons"] = [
        f"{NODE} XID 46 GPU stopped processing",
        f"{NODE} XID 46 GPU stopped processing (merged)",
    ]
    return state


# --------------------------------------------------------------------------- #
# Phase B fixtures: the preempting successor
# --------------------------------------------------------------------------- #
def successor_workflow() -> dict[str, Any]:
    """The graph claim-time adoption leaves behind: the appended restore is the
    last index, depends on RESTART_NODE, and VALIDATE_GPU depends on it."""

    steps = [
        _step("FREEZE_EVIDENCE"),
        _step("MARK_UNSCHEDULABLE", depends=[0]),
        _step("RESTART_NODE", depends=[1]),
        _step("VALIDATE_GPU", depends=[7]),
        _step("VALIDATE_HOST", depends=[3]),
        _step("VALIDATE_FABRIC", depends=[4]),
        _step("RESTORE_SCHEDULING", depends=[5]),
        _step(
            "RESTORE_GPU_SERVICES",
            depends=[2],
            parameters={
                verdicts.HANDOFF_PARAMETER: True,
                "handoff_from_workflow_id": RESET_ID,
            },
        ),
    ]
    return {
        "request_id": REBOOT_ID,
        "incident_id": INCIDENT,
        "status": "RUNNING",
        "node_ids": [NODE],
        "predecessor_workflow_id": RESET_ID,
        "quiesce_handoff_from_workflow_id": RESET_ID,
        "preemption_reason": "strictly stronger recovery action: rank 30 -> 50",
        "completed_operations": ["MARK_UNSCHEDULABLE"],
        "inherited_step_indexes": [1],
        "dag_enabled": True,
        "dag_revision": 2,
        "official_steps": steps,
        "step_executions": [
            _execution(
                0,
                "FREEZE_EVIDENCE",
                "SUCCEEDED",
                started_at=_at(4),
                updated_at=_at(4, 20),
            )
        ],
    }


def superseded_reset_workflow() -> dict[str, Any]:
    workflow = parked_reset_workflow()
    workflow["status"] = "SUPERSEDED"
    workflow["preempted_by_workflow_id"] = REBOOT_ID
    return workflow


def escalated_incident() -> dict[str, Any]:
    incident = parked_reset_incident()
    incident["official_action"] = "RESTART_NODE"
    incident["workflow_request_id"] = REBOOT_ID
    incident["reasons"] = [
        f"{NODE} XID 46 GPU stopped processing",
        f"{NODE} XID 46 GPU stopped processing (merged)",
        f"{NODE} XID 79 GPU has fallen off the bus",
    ]
    return incident


def cancelled_commands() -> list[dict[str, Any]]:
    commands = barrier_commands()
    commands[0].update(
        {
            "status": "FAILED",
            "status_source": "workflow-preempted",
            "error": f"remote command cancelled by stronger workflow {REBOOT_ID}",
        }
    )
    return commands


def terminal_state() -> dict[str, Any]:
    workflow = successor_workflow()
    workflow["status"] = "SUCCEEDED"
    workflow["step_executions"] = [
        _execution(
            0, "FREEZE_EVIDENCE", "SUCCEEDED", started_at=_at(4), updated_at=_at(4, 20)
        ),
        _execution(2, "RESTART_NODE", "WAITING", started_at=_at(5), updated_at=_at(6)),
        _execution(
            2, "RESTART_NODE", "SUCCEEDED", started_at=_at(5), updated_at=_at(14)
        ),
        _execution(
            7,
            "RESTORE_GPU_SERVICES",
            "SUCCEEDED",
            started_at=_at(15),
            updated_at=_at(16),
        ),
        _execution(
            3, "VALIDATE_GPU", "SUCCEEDED", started_at=_at(17), updated_at=_at(18)
        ),
        _execution(
            4, "VALIDATE_HOST", "SUCCEEDED", started_at=_at(18), updated_at=_at(19)
        ),
        _execution(
            5, "VALIDATE_FABRIC", "SUCCEEDED", started_at=_at(19), updated_at=_at(20)
        ),
        _execution(
            6, "RESTORE_SCHEDULING", "SUCCEEDED", started_at=_at(20), updated_at=_at(21)
        ),
    ]
    return {
        "workflow": workflow,
        "incident": {**escalated_incident(), "state": "RECOVERED"},
        "submission": {"state": "SUBMITTED", "action": "REBOOT"},
        "agent": {
            "node_id": NODE,
            "boot_id": "boot-after",
            "artifact_sha256": "artifact-sha",
            "lifecycle_state": "ACTIVE",
        },
    }


# --------------------------------------------------------------------------- #
# Data-plane fixtures
# --------------------------------------------------------------------------- #
def _row(
    command_id: str, operation: str, state: str, *, attempt: int = 1
) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "completed_at": _at(3),
        "attempt": attempt,
        "state": state,
        "operation": operation,
        "started_at": _at(2),
    }


def host_baseline() -> dict[str, Any]:
    return {
        "boot_id": "host-boot-before",
        "kmsg_writable": True,
        "gpu_inventory": [
            {"index": 0, "pci_bdf": "0000:0a:00.0"},
            {"index": 1, "pci_bdf": "0000:0b:00.0"},
        ],
        "compute_clients": [],
        "quiesce_states": [],
        "services": {
            "nvidia-fabricmanager.service": {"ActiveState": "active"},
            "gpu-fault-node-agent.service": {"ActiveState": "active"},
        },
        "ledger": [
            _row("wf-old/4/VERIFY/commit", "VERIFY_NO_GPU_CLIENTS", "SUCCEEDED")
        ],
    }


def host_after() -> dict[str, Any]:
    baseline = host_baseline()
    return {
        **baseline,
        "boot_id": "host-boot-after",
        "ledger": [
            *baseline["ledger"],
            _row(f"{RESET_ID}/2/QUIESCE/commit", "QUIESCE_GPU_SERVICES", "SUCCEEDED"),
            _row(f"{BARRIER_COMMAND}", "VERIFY_NO_GPU_CLIENTS", "FAILED", attempt=1),
            _row(f"{BARRIER_COMMAND}", "VERIFY_NO_GPU_CLIENTS", "FAILED", attempt=2),
            _row(f"{REBOOT_ID}/7/RESTORE/commit", "RESTORE_GPU_SERVICES", "SUCCEEDED"),
            _row(f"{REBOOT_ID}/3/VALIDATE/commit", "VALIDATE_GPU", "SUCCEEDED"),
        ],
    }


def holder_status() -> dict[str, Any]:
    return {
        "run_id": "destr016-run-a1",
        "holder_unit": "gpu-fault-destr016-holder-abc.service",
        "unit_state": {"LoadState": "not-found", "ActiveState": "inactive"},
        "pid": "0",
        "matched_row": {"command_id": f"{RESET_ID}/2/QUIESCE/commit"},
        "hold_started_at": _at(1, 55),
        "arm_race_lost": False,
        "holder_error": None,
        "boot_id": "host-boot-after",
    }


def node_baseline() -> dict[str, Any]:
    return {
        "name": NODE,
        "uid": "node-uid",
        "boot_id": "node-boot-before",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
        "gpu_allocatable": 8,
    }


def node_after_boot() -> dict[str, Any]:
    return {**node_baseline(), "boot_id": "node-boot-after"}


def provider_events() -> list[dict[str, Any]]:
    return [
        {
            "event_name": "BatchRebootClusterNodes",
            "username": "gpu-fault-executor-role",
            "session_issuer_role_name": "gpu-fault-executor-role",
        },
        {"event_name": "DescribeClusterNode", "username": "someone-else"},
    ]


# --------------------------------------------------------------------------- #
# Case contract
# --------------------------------------------------------------------------- #
def test_confirmation_token_matches_the_case_contract_derivation() -> None:
    metadata = RegionalCaseMetadata(
        case_id=destr016.CASE_ID,
        title="",
        category="regional-destructive-acceptance",
        level="staging",
        risk="destructive-provider-reboot",
        automation="manual",
        procedure="docs/x.md#gf-regional-destr-016",
        predecessor=destr016.PREDECESSOR_CASE_ID,
    )
    assert destr016.CONFIRMATION == metadata.confirmation == "DESTR016_EXECUTE"
    assert destr016.CASE_ID == "GF-REGIONAL-DESTR-016"
    assert destr016.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-002"


def test_destr016_workflow_contract_preempts_the_reset_and_reboots_once() -> None:
    """The whole case contract, on fixtures that model one correct run.

    This is the entry the catalog's ``related_pytest`` points at. It fails if any
    stage of the chain stops holding: the dirty WAITING boundary with GPU
    services quiesced, the absorbed same-rank fault, the preemption that
    supersedes the reset and hands the quiesce to a rebooting successor, the
    cancelled barrier command, exactly one real reboot, and a recovered node.
    """

    barrier = barrier_snapshot()
    assert (
        verdicts.reset_workflow_errors(
            barrier["workflow"], barrier["incident"], barrier["decision"]
        )
        == []
    )
    assert verdicts.barrier_reason_errors(barrier["commands"]) == []
    assert verdicts.absorb_errors(barrier, absorbed_snapshot(), node=NODE) == []
    assert _escalation_errors() == []
    assert verdicts.successor_step_graph_errors(successor_workflow()) == []
    assert verdicts.superseded_predecessor_errors(superseded_reset_workflow()) == []
    assert (
        verdicts.cancelled_command_errors(
            cancelled_commands(), successor_request_id=REBOOT_ID
        )
        == []
    )
    assert _terminal_errors(terminal_state()) == []
    assert verdicts.restore_after_reboot_errors(terminal_state()["workflow"]) == []
    assert _host_errors(host_after()) == []
    assert verdicts.holder_errors(holder_status()) == []
    assert verdicts.schedulability_errors(node_after_boot(), node=NODE) == []
    assert (
        verdicts.node_recovery_errors(
            node_baseline(), node_after_boot(), node_after_boot()
        )
        == []
    )
    assert verdicts.provider_errors(provider_events(), actor_matches_role=True) == []


def test_runner_probe_scripts_exist() -> None:
    assert destr016.HOLDER_PROBE.name == "destr016_node_probe.py"
    assert destr016.INJECT_PROBE.name == "destructive_node_probe.py"
    assert destr016.HOLDER_PROBE.is_file() is True
    assert destr016.INJECT_PROBE.is_file() is True


# --------------------------------------------------------------------------- #
# Phase 1: the dirty boundary
# --------------------------------------------------------------------------- #
def test_parked_reset_workflow_is_the_boundary_the_case_needs() -> None:
    state = barrier_snapshot()
    assert (
        verdicts.reset_workflow_errors(
            state["workflow"], state["incident"], state["decision"]
        )
        == []
    )
    assert verdicts.barrier_reason_errors(state["commands"]) == []


def test_a_reset_that_already_committed_is_not_a_dirty_boundary() -> None:
    workflow = parked_reset_workflow()
    workflow["step_executions"].append(
        _execution(4, "RESET_GPU", "SUCCEEDED", started_at=_at(4), updated_at=_at(5))
    )
    errors = verdicts.waiting_boundary_errors(workflow)
    assert any("RESET_GPU already ran" in item for item in errors), errors


def test_a_boundary_without_a_quiesce_is_clean_not_dirty() -> None:
    workflow = parked_reset_workflow()
    workflow["step_executions"] = [
        item
        for item in workflow["step_executions"]
        if item["operation"] != "QUIESCE_GPU_SERVICES"
    ]
    errors = verdicts.waiting_boundary_errors(workflow)
    assert any("the boundary is clean" in item for item in errors), errors


def test_a_restored_quiesce_is_not_a_dirty_boundary() -> None:
    workflow = parked_reset_workflow()
    workflow["step_executions"].append(
        _execution(
            5, "RESTORE_GPU_SERVICES", "SUCCEEDED", started_at=_at(4), updated_at=_at(5)
        )
    )
    errors = verdicts.waiting_boundary_errors(workflow)
    assert any("not unrestored" in item for item in errors), errors


def test_a_barrier_that_is_not_waiting_fails() -> None:
    workflow = parked_reset_workflow()
    workflow["step_executions"][3]["status"] = "SUCCEEDED"
    errors = verdicts.waiting_boundary_errors(workflow)
    assert any("is not WAITING" in item for item in errors), errors


def test_a_barrier_without_a_remote_command_id_fails() -> None:
    workflow = parked_reset_workflow()
    workflow["step_executions"][3]["details"] = {"remote_status": "WAITING"}
    errors = verdicts.waiting_boundary_errors(workflow)
    assert any("no remote command id" in item for item in errors), errors


def test_a_reset_workflow_with_other_steps_fails() -> None:
    workflow = parked_reset_workflow()
    workflow["official_steps"].append(_step("REPLACE_NODE"))
    errors = verdicts.reset_workflow_errors(
        workflow, parked_reset_incident(), {"official_action": "RESET_GPU"}
    )
    assert any("reset workflow steps are not" in item for item in errors), errors


def test_a_policy_decision_other_than_reset_fails() -> None:
    errors = verdicts.reset_workflow_errors(
        parked_reset_workflow(),
        parked_reset_incident(),
        {"official_action": "RESTART_NODE"},
    )
    assert any("did not resolve the first XID 46" in item for item in errors), errors


def test_a_barrier_command_that_waits_for_another_reason_fails() -> None:
    commands = barrier_commands()
    commands[0]["result_details"]["reason"] = "node agent lease expired"
    errors = verdicts.barrier_reason_errors(commands)
    assert any("clients are still active" in item for item in errors), errors


def test_a_barrier_command_with_no_verification_attempt_fails() -> None:
    commands = barrier_commands()
    commands[0]["result_details"].pop("gpu_client_quiesce_attempt")
    errors = verdicts.barrier_reason_errors(commands)
    assert any("no client-verification attempt" in item for item in errors), errors


def test_two_barrier_commands_are_refused() -> None:
    commands = [*barrier_commands(), *barrier_commands()]
    assert verdicts.barrier_reason_errors(commands) == [
        "there is not exactly one barrier remote command: 2"
    ]


# --------------------------------------------------------------------------- #
# Phase A: the same-rank fault is absorbed
# --------------------------------------------------------------------------- #
def test_an_absorbed_same_rank_fault_keeps_one_workflow() -> None:
    assert (
        verdicts.absorb_errors(barrier_snapshot(), absorbed_snapshot(), node=NODE) == []
    )


def test_a_second_workflow_for_the_same_rank_fault_fails() -> None:
    after = absorbed_snapshot()
    after["workflow"]["request_id"] = "wf-second"
    errors = verdicts.absorb_errors(barrier_snapshot(), after, node=NODE)
    assert any("created a second workflow" in item for item in errors), errors


def test_a_second_incident_for_the_same_rank_fault_fails() -> None:
    after = absorbed_snapshot()
    after["incident"]["incident_id"] = "inc-second"
    errors = verdicts.absorb_errors(barrier_snapshot(), after, node=NODE)
    assert any("created a second incident" in item for item in errors), errors


def test_an_extra_step_from_the_absorbed_fault_fails() -> None:
    after = absorbed_snapshot()
    after["workflow"]["official_steps"].append(_step("COLLECT_DIAGNOSTIC_BUNDLE"))
    errors = verdicts.absorb_errors(barrier_snapshot(), after, node=NODE)
    assert any("changed the step list" in item for item in errors), errors


def test_an_extra_step_execution_from_the_absorbed_fault_fails() -> None:
    after = absorbed_snapshot()
    after["workflow"]["step_executions"].append(
        _execution(
            3,
            verdicts.BARRIER_OPERATION,
            "WAITING",
            started_at=_at(4),
            updated_at=_at(4),
        )
    )
    errors = verdicts.absorb_errors(barrier_snapshot(), after, node=NODE)
    assert any("added a step execution" in item for item in errors), errors


def test_an_escalated_official_action_is_not_an_absorption() -> None:
    after = absorbed_snapshot()
    after["incident"]["official_action"] = "RESTART_NODE"
    errors = verdicts.absorb_errors(barrier_snapshot(), after, node=NODE)
    assert any("changed the official action" in item for item in errors), errors


def test_an_absorbed_fault_that_left_no_reason_fails() -> None:
    after = barrier_snapshot()
    errors = verdicts.absorb_errors(barrier_snapshot(), after, node=NODE)
    assert any("reasons did not grow" in item for item in errors), errors


def test_a_reason_that_names_another_node_or_xid_fails() -> None:
    before = parked_reset_incident()
    after = parked_reset_incident()
    after["reasons"] = [*before["reasons"], "node-c XID 46 on another node"]
    errors = verdicts.incident_reason_errors(before, after, node=NODE, xid=46)
    assert any("no incident reason records" in item for item in errors), errors


def test_the_absorbed_event_must_be_the_same_xid() -> None:
    after = absorbed_snapshot()
    after["event"] = {"xid": 79, "node_id": NODE}
    errors = verdicts.absorb_errors(barrier_snapshot(), after, node=NODE)
    assert any("absorbed event is not XID 46" in item for item in errors), errors


# --------------------------------------------------------------------------- #
# Phase B: preemption, handoff, cancellation
# --------------------------------------------------------------------------- #
def _escalation_errors(
    predecessor: dict[str, Any] | None = None,
    successor: dict[str, Any] | None = None,
    incident: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
) -> list[str]:
    return verdicts.escalation_errors(
        predecessor if predecessor is not None else superseded_reset_workflow(),
        successor if successor is not None else successor_workflow(),
        incident if incident is not None else escalated_incident(),
        decision=decision
        if decision is not None
        else {"official_action": "RESTART_NODE"},
    )


def test_the_escalation_contract_holds_on_a_correct_preemption() -> None:
    assert _escalation_errors() == []
    assert verdicts.successor_step_graph_errors(successor_workflow()) == []
    assert verdicts.superseded_predecessor_errors(superseded_reset_workflow()) == []
    assert (
        verdicts.cancelled_command_errors(
            cancelled_commands(), successor_request_id=REBOOT_ID
        )
        == []
    )


def test_a_reset_workflow_still_running_after_the_escalation_fails() -> None:
    predecessor = superseded_reset_workflow()
    predecessor["status"] = "RUNNING"
    errors = _escalation_errors(predecessor=predecessor)
    assert any("is not SUPERSEDED" in item for item in errors), errors


def test_a_successor_that_reused_the_reset_workflow_row_fails() -> None:
    successor = successor_workflow()
    successor["request_id"] = RESET_ID
    assert _escalation_errors(successor=successor) == [
        "the escalation reused the reset workflow row"
    ]


def test_an_unlinked_predecessor_and_successor_pair_fails() -> None:
    predecessor = superseded_reset_workflow()
    predecessor["preempted_by_workflow_id"] = "wf-other"
    successor = successor_workflow()
    successor["predecessor_workflow_id"] = "wf-other"
    errors = _escalation_errors(predecessor=predecessor, successor=successor)
    assert any("does not point at its successor" in item for item in errors), errors
    assert any("does not point back at" in item for item in errors), errors


def test_an_incident_still_pointing_at_the_superseded_workflow_fails() -> None:
    incident = escalated_incident()
    incident["workflow_request_id"] = RESET_ID
    errors = _escalation_errors(incident=incident)
    assert any("still points at the superseded" in item for item in errors), errors


def test_a_second_incident_for_the_escalation_fails() -> None:
    successor = successor_workflow()
    successor["incident_id"] = "inc-other"
    errors = _escalation_errors(successor=successor)
    assert any("belongs to another incident" in item for item in errors), errors


def test_a_successor_without_the_rank_reason_fails() -> None:
    successor = successor_workflow()
    successor["preemption_reason"] = "operator requested"
    errors = _escalation_errors(successor=successor)
    assert any("strictly stronger recovery action" in item for item in errors), errors


def test_a_successor_that_inherited_the_wrong_steps_fails() -> None:
    successor = successor_workflow()
    successor["completed_operations"] = ["MARK_UNSCHEDULABLE", "QUIESCE_GPU_SERVICES"]
    errors = _escalation_errors(successor=successor)
    assert any("did not inherit exactly" in item for item in errors), errors


def test_a_successor_that_records_no_inherited_indexes_fails() -> None:
    successor = successor_workflow()
    successor["inherited_step_indexes"] = []
    errors = _escalation_errors(successor=successor)
    assert any("no inherited step indexes" in item for item in errors), errors


def test_a_policy_decision_other_than_restart_node_fails() -> None:
    errors = _escalation_errors(decision={"official_action": "REPLACE_NODE"})
    assert any("did not resolve XID 79" in item for item in errors), errors


def test_a_successor_whose_restore_is_not_the_last_step_fails() -> None:
    """A plain appended restore in prose order is the wrong graph."""

    successor = successor_workflow()
    steps = successor["official_steps"]
    steps.insert(3, steps.pop(7))
    errors = verdicts.successor_step_graph_errors(successor)
    assert any("successor steps are not" in item for item in errors), errors


def test_a_successor_whose_restore_does_not_carry_the_handoff_parameter_fails() -> None:
    successor = successor_workflow()
    successor["official_steps"][7]["parameters"] = {}
    errors = verdicts.successor_step_graph_errors(successor)
    assert any(verdicts.HANDOFF_PARAMETER in item for item in errors), errors
    assert any("does not name the workflow" in item for item in errors), errors


def test_a_restore_that_does_not_wait_for_the_reboot_fails() -> None:
    successor = successor_workflow()
    successor["official_steps"][7]["depends_on_step_indexes"] = [1]
    errors = verdicts.successor_step_graph_errors(successor)
    assert any("does not depend on RESTART_NODE alone" in item for item in errors), (
        errors
    )


def test_a_validate_gpu_still_pointing_at_the_reboot_fails() -> None:
    successor = successor_workflow()
    successor["official_steps"][3]["depends_on_step_indexes"] = [2]
    errors = verdicts.successor_step_graph_errors(successor)
    assert any("was not repointed" in item for item in errors), errors


def test_a_successor_that_is_not_dag_enabled_fails() -> None:
    successor = successor_workflow()
    successor["dag_enabled"] = False
    errors = verdicts.successor_step_graph_errors(successor)
    assert any("not dag_enabled" in item for item in errors), errors


def test_a_successor_without_the_handoff_source_fails() -> None:
    successor = successor_workflow()
    successor["quiesce_handoff_from_workflow_id"] = None
    errors = verdicts.successor_step_graph_errors(successor)
    assert any(
        "does not record the quiesce handoff source" in item for item in errors
    ), errors


def test_a_superseded_reset_whose_barrier_record_changed_fails() -> None:
    """The step record stays WAITING; only the remote command is cancelled."""

    predecessor = superseded_reset_workflow()
    predecessor["step_executions"][3]["status"] = "FAILED"
    errors = verdicts.superseded_predecessor_errors(predecessor)
    assert any("single WAITING row" in item for item in errors), errors


def test_a_superseded_reset_that_ran_a_reset_or_restore_fails() -> None:
    predecessor = superseded_reset_workflow()
    predecessor["step_executions"].append(
        _execution(4, "RESET_GPU", "FAILED", started_at=_at(4), updated_at=_at(5))
    )
    predecessor["step_executions"].append(
        _execution(
            5, "RESTORE_GPU_SERVICES", "SUCCEEDED", started_at=_at(5), updated_at=_at(6)
        )
    )
    errors = verdicts.superseded_predecessor_errors(predecessor)
    assert "the superseded reset ran RESET_GPU" in errors
    assert "the superseded reset ran RESTORE_GPU_SERVICES" in errors


def test_a_barrier_command_left_waiting_after_the_preemption_fails() -> None:
    commands = barrier_commands()
    errors = verdicts.cancelled_command_errors(commands, successor_request_id=REBOOT_ID)
    assert any("is not FAILED" in item for item in errors), errors


def test_a_barrier_command_failed_by_something_else_fails() -> None:
    commands = cancelled_commands()
    commands[0]["status_source"] = "executor-result"
    commands[0]["error"] = "node agent reported a hardware error"
    errors = verdicts.cancelled_command_errors(commands, successor_request_id=REBOOT_ID)
    assert any("was not cancelled by the preemption" in item for item in errors), errors
    assert any("does not name the successor" in item for item in errors), errors


def test_a_reset_command_issued_by_the_superseded_workflow_fails() -> None:
    commands = [
        *cancelled_commands(),
        {
            "command_id": "cmd-reset",
            "step": {"operation": "RESET_GPU"},
            "status": "SUCCEEDED",
        },
    ]
    errors = verdicts.cancelled_command_errors(commands, successor_request_id=REBOOT_ID)
    assert "the superseded reset issued a RESET_GPU command" in errors


# --------------------------------------------------------------------------- #
# Terminal state
# --------------------------------------------------------------------------- #
def _terminal_errors(state: dict[str, Any]) -> list[str]:
    return verdicts.terminal_errors(
        state, expected_boot_id="boot-before", expected_artifact="artifact-sha"
    )


def test_the_terminal_contract_holds_on_a_recovered_incident() -> None:
    assert _terminal_errors(terminal_state()) == []
    assert verdicts.restore_after_reboot_errors(terminal_state()["workflow"]) == []


def test_a_workflow_that_did_not_succeed_fails() -> None:
    state = terminal_state()
    state["workflow"]["status"] = "FAILED"
    state["workflow"]["terminal_failure_reason"] = "reboot timed out"
    errors = _terminal_errors(state)
    assert any("is not SUCCEEDED" in item for item in errors), errors
    assert any("carries a failure reason" in item for item in errors), errors


def test_an_incident_that_is_not_recovered_fails() -> None:
    state = terminal_state()
    state["incident"]["state"] = "ISOLATED"
    errors = _terminal_errors(state)
    assert any("not RECOVERED" in item for item in errors), errors


@pytest.mark.parametrize(
    "operation",
    ["RESTART_NODE", "RESTORE_GPU_SERVICES", "VALIDATE_GPU", "RESTORE_SCHEDULING"],
)
def test_every_terminal_step_must_end_succeeded(operation: str) -> None:
    state = terminal_state()
    state["workflow"]["step_executions"] = [
        item
        for item in state["workflow"]["step_executions"]
        if item["operation"] != operation
    ]
    errors = _terminal_errors(state)
    assert any(f"{operation} did not end SUCCEEDED" in item for item in errors), errors


def test_a_submission_that_is_not_a_reboot_fails() -> None:
    state = terminal_state()
    state["submission"] = {"state": "PENDING", "action": "REPLACE"}
    errors = _terminal_errors(state)
    assert any("not SUBMITTED" in item for item in errors), errors
    assert any("action is not REBOOT" in item for item in errors), errors


def test_an_unchanged_agent_boot_id_means_no_real_reboot() -> None:
    state = terminal_state()
    state["agent"]["boot_id"] = "boot-before"
    errors = _terminal_errors(state)
    assert any("did not change; no real reboot" in item for item in errors), errors


def test_a_changed_agent_artifact_or_inactive_agent_fails() -> None:
    state = terminal_state()
    state["agent"]["artifact_sha256"] = "another-sha"
    state["agent"]["lifecycle_state"] = "DEGRADED"
    errors = _terminal_errors(state)
    assert any("artifact changed" in item for item in errors), errors
    assert any("did not return ACTIVE" in item for item in errors), errors


def test_a_restore_that_ran_before_the_reboot_finished_fails() -> None:
    workflow = terminal_state()["workflow"]
    for item in workflow["step_executions"]:
        if item["operation"] == "RESTORE_GPU_SERVICES":
            item["started_at"] = _at(7)
    errors = verdicts.restore_after_reboot_errors(workflow)
    assert any("started before the reboot finished" in item for item in errors), errors


def test_a_missing_restore_or_reboot_execution_fails() -> None:
    workflow = terminal_state()["workflow"]
    workflow["step_executions"] = [
        item
        for item in workflow["step_executions"]
        if item["operation"] != "RESTORE_GPU_SERVICES"
    ]
    errors = verdicts.restore_after_reboot_errors(workflow)
    assert any("one successful RESTORE_GPU_SERVICES" in item for item in errors), errors


def test_a_reboot_that_parked_waiting_before_succeeding_is_still_one_reboot() -> None:
    """A managed recovery polls WAITING before it completes; that is normal."""

    workflow = terminal_state()["workflow"]
    assert [
        item["status"] for item in verdicts.executions_of(workflow, "RESTART_NODE")
    ] == ["WAITING", "SUCCEEDED"]
    assert verdicts.restore_after_reboot_errors(workflow) == []


# --------------------------------------------------------------------------- #
# Data plane
# --------------------------------------------------------------------------- #
def _host_errors(after: dict[str, Any]) -> list[str]:
    return verdicts.host_errors(host_baseline(), after, expected_gpu_count=2)


def test_the_host_contract_holds_on_a_real_reboot() -> None:
    assert _host_errors(host_after()) == []
    assert verdicts.holder_errors(holder_status()) == []


def test_an_unchanged_host_boot_id_fails() -> None:
    after = host_after()
    after["boot_id"] = host_baseline()["boot_id"]
    errors = _host_errors(after)
    assert any("did not reboot" in item for item in errors), errors


def test_a_successful_reset_row_in_the_ledger_fails() -> None:
    after = host_after()
    after["ledger"].append(_row("cmd-reset", "RESET_GPU", "SUCCEEDED"))
    errors = _host_errors(after)
    assert any("successful RESET_GPU" in item for item in errors), errors


def test_a_successful_full_fabric_reset_row_fails() -> None:
    after = host_after()
    after["ledger"].append(_row("cmd-fabric", "RESET_ALL_GPUS_NVSWITCHES", "SUCCEEDED"))
    errors = _host_errors(after)
    assert any("RESET_ALL_GPUS_NVSWITCHES" in item for item in errors), errors


@pytest.mark.parametrize(
    "operation", ["QUIESCE_GPU_SERVICES", "RESTORE_GPU_SERVICES", "VALIDATE_GPU"]
)
def test_the_ledger_must_show_every_node_side_step(operation: str) -> None:
    after = host_after()
    after["ledger"] = [row for row in after["ledger"] if row["operation"] != operation]
    errors = _host_errors(after)
    assert f"the Node Agent ledger has no successful {operation}" in errors


def test_a_client_verification_that_only_ever_succeeded_means_no_holder() -> None:
    after = host_after()
    for row in after["ledger"]:
        if row["operation"] == verdicts.BARRIER_OPERATION:
            row["state"] = "SUCCEEDED"
    errors = _host_errors(after)
    assert any("the holder never held" in item for item in errors), errors


def test_barrier_retries_under_one_command_id_are_not_collapsed() -> None:
    """Keyed by (command_id, attempt): dropping the attempt would hide retries."""

    baseline = [_row("cmd", verdicts.BARRIER_OPERATION, "FAILED", attempt=1)]
    after = [
        *baseline,
        _row("cmd", verdicts.BARRIER_OPERATION, "FAILED", attempt=2),
        _row("cmd", verdicts.BARRIER_OPERATION, "FAILED", attempt=3),
    ]
    added = verdicts.added_ledger_rows(baseline, after)
    assert [row["attempt"] for row in added] == [2, 3]


def test_a_surviving_quiesce_state_or_client_fails() -> None:
    after = host_after()
    after["quiesce_states"] = ["/var/lib/gpu-fault/quiesce-abc.json"]
    after["compute_clients"] = [{"pid": 4242, "device": "/dev/nvidia0"}]
    errors = _host_errors(after)
    assert any("quiesce state survives" in item for item in errors), errors
    assert any("still has NVIDIA compute clients" in item for item in errors), errors


def test_a_service_that_did_not_return_active_fails() -> None:
    after = host_after()
    after["services"] = {
        **after["services"],
        "nvidia-fabricmanager.service": {"ActiveState": "failed"},
    }
    errors = _host_errors(after)
    assert (
        "a service did not return active after the reboot: "
        "nvidia-fabricmanager.service" in errors
    )


def test_a_missing_gpu_after_the_reboot_fails() -> None:
    after = host_after()
    after["gpu_inventory"] = after["gpu_inventory"][:1]
    errors = _host_errors(after)
    assert any("GPU inventory is not 2" in item for item in errors), errors


def test_a_holder_that_lost_the_arming_race_fails() -> None:
    status = holder_status()
    status["arm_race_lost"] = True
    errors = verdicts.holder_errors(status)
    assert any("no dirty boundary to preempt" in item for item in errors), errors


def test_a_holder_that_never_started_or_errored_fails() -> None:
    status = holder_status()
    status["hold_started_at"] = None
    status["matched_row"] = None
    status["holder_error"] = "arm ledger row never appeared"
    errors = verdicts.holder_errors(status)
    assert any("never started" in item for item in errors), errors
    assert any("never matched a ledger row" in item for item in errors), errors
    assert any("reported an error" in item for item in errors), errors


def test_a_holder_still_active_after_the_reboot_fails() -> None:
    status = holder_status()
    status["unit_state"] = {"LoadState": "loaded", "ActiveState": "active"}
    errors = verdicts.holder_errors(status)
    assert any("still active after the reboot" in item for item in errors), errors


def test_the_node_must_come_back_schedulable_and_unowned() -> None:
    assert verdicts.schedulability_errors(node_after_boot(), node=NODE) == []
    assert (
        verdicts.node_recovery_errors(
            node_baseline(), node_after_boot(), node_after_boot()
        )
        == []
    )


def test_a_cordoned_tainted_or_owned_node_fails() -> None:
    snapshot = node_after_boot()
    snapshot["unschedulable"] = True
    snapshot["taints"] = [{"key": "gpu-fault.io/quarantine", "value": "abc"}]
    snapshot["ownership_annotations"] = {"gpu-fault.io/incident-id": INCIDENT}
    errors = verdicts.schedulability_errors(snapshot, node=NODE)
    assert any("not schedulable" in item for item in errors), errors
    assert any("still carries a gpu-fault taint" in item for item in errors), errors
    assert any("ownership annotation" in item for item in errors), errors


def test_an_unrebooted_or_recreated_node_fails() -> None:
    errors = verdicts.node_recovery_errors(
        node_baseline(), node_baseline(), {**node_after_boot(), "uid": "another-uid"}
    )
    assert any("Node boot id did not change" in item for item in errors), errors
    assert any("Node UID changed" in item for item in errors), errors


def test_gpu_capacity_must_return_to_baseline() -> None:
    errors = verdicts.node_recovery_errors(
        node_baseline(), node_after_boot(), {**node_after_boot(), "gpu_allocatable": 7}
    )
    assert any("GPU capacity did not return" in item for item in errors), errors


def test_exactly_one_reboot_call_by_the_executor_role_passes() -> None:
    events = provider_events()
    assert len(verdicts.reboot_events(events)) == 1
    assert verdicts.provider_errors(events, actor_matches_role=True) == []


def test_no_reboot_or_two_reboots_fail() -> None:
    assert verdicts.provider_errors([], actor_matches_role=None) == [
        "CloudTrail does not contain exactly one reboot event: []"
    ]
    twice = [*provider_events(), provider_events()[0]]
    errors = verdicts.provider_errors(twice, actor_matches_role=None)
    assert any("exactly one reboot event" in item for item in errors), errors


def test_a_reboot_by_another_actor_fails() -> None:
    errors = verdicts.provider_errors(provider_events(), actor_matches_role=False)
    assert errors == ["the CloudTrail reboot actor is not the executor role"]


def test_a_replace_or_delete_mutation_fails() -> None:
    events = [*provider_events(), {"event_name": "BatchDeleteClusterNodes"}]
    errors = verdicts.provider_errors(events, actor_matches_role=True)
    assert any("replace/delete mutation" in item for item in errors), errors


# --------------------------------------------------------------------------- #
# Timeline and arithmetic
# --------------------------------------------------------------------------- #
def test_step_transitions_only_report_changes() -> None:
    executions = terminal_state()["workflow"]["step_executions"]
    state, changes = verdicts.step_transitions({}, executions)
    assert state["2/RESTART_NODE#1"] == "SUCCEEDED"
    assert state["2/RESTART_NODE#0"] == "WAITING"
    # One entry per status change, so the reboot's WAITING poll and its success
    # are both on the timeline even though they share a step key.
    assert [item["step"] for item in changes].count("2/RESTART_NODE") == 2
    assert len(changes) == len(executions)
    again_state, again_changes = verdicts.step_transitions(state, executions)
    assert again_changes == []
    assert again_state == state


def test_the_case_must_fit_one_node_workflow_lifetime() -> None:
    estimated = verdicts.estimated_duration_seconds()
    assert estimated == 2340
    assert (
        verdicts.lifetime_errors(estimated_seconds=estimated, lifetime_seconds=3600)
        == []
    )
    errors = verdicts.lifetime_errors(
        estimated_seconds=estimated, lifetime_seconds=1800
    )
    assert any("does not fit the 1800s" in item for item in errors), errors
    assert verdicts.lifetime_errors(
        estimated_seconds=estimated, lifetime_seconds=None
    ) == ["node workflow lifetime is unknown; cannot bound the case duration"]


def test_both_injections_must_fit_the_barrier_step_timeout() -> None:
    assert verdicts.step_timeout_errors(step_timeout_seconds=600) == []
    errors = verdicts.step_timeout_errors(step_timeout_seconds=120)
    assert any("is below 300s" in item for item in errors), errors
    assert verdicts.step_timeout_errors(step_timeout_seconds=None) == [
        "the deployed workflow step timeout is unknown"
    ]
    # The smallest ceiling the case accepts still contains the park it plans.
    assert verdicts.MIN_STEP_TIMEOUT_SECONDS > verdicts.WAITING_PARK_ALLOWANCE_SECONDS
    assert verdicts.step_timeout_errors(step_timeout_seconds=300) == []


# --------------------------------------------------------------------------- #
# Runner helpers
# --------------------------------------------------------------------------- #
def happy_preflight_arguments() -> dict[str, Any]:
    return {
        "node": NODE,
        "node_snapshot": node_baseline(),
        "agent": {
            "lifecycle_state": "ACTIVE",
            "allowed_operations": list(verdicts.AGENT_OPERATIONS),
            "boot_id": "boot-before",
        },
        "profile": {
            "profile_version": "v7",
            "warnings": [],
            "capabilities": [
                {
                    "capability": "gpuReset",
                    "mode": "OWN",
                    "owner": "gpu-fault-node-agent",
                },
                {
                    "capability": "nodeReboot",
                    "mode": "OWN",
                    "owner": "gpu-fault-hyperpod-adapter",
                    "adapter": "regional-cluster-executor",
                },
            ],
        },
        "queue": {"depth": 0},
        "remote_commands": {"open_by_cluster": {}},
        "gpu_workloads": [],
        "business_workloads": [],
        "event": None,
        "predecessor": {"valid": True},
        "tests": {"passed": True},
        "control_env": {
            "step_timeout_seconds": 600,
            "node_lifetime_seconds": 3600,
            "preemption_enabled": True,
        },
        "identity_errors": [],
    }


def test_a_healthy_idle_node_passes_the_preflight() -> None:
    assert destr016.preflight_errors(**happy_preflight_arguments()) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"predecessor": {"valid": False}}, "predecessor evidence is not PASS"),
        ({"tests": {"passed": False}}, "focused regression tests failed"),
        ({"event": {"xid": 46}}, "has a recent XID event"),
        ({"queue": {"depth": 3}}, "processor queue is not empty"),
        (
            {"remote_commands": {"open_by_cluster": {"cluster-a": 1}}},
            "remote command queue is not empty",
        ),
        ({"gpu_workloads": [{"name": "job"}]}, "already has a GPU workload"),
        ({"business_workloads": [{"name": "pod"}]}, "carries a non-system workload"),
        (
            {"agent": {"lifecycle_state": "DEGRADED", "allowed_operations": []}},
            "Node Agent is not ACTIVE",
        ),
        (
            {
                "control_env": {
                    "step_timeout_seconds": 60,
                    "node_lifetime_seconds": 3600,
                }
            },
            "is below 300s",
        ),
        (
            {
                "control_env": {
                    "step_timeout_seconds": 600,
                    "node_lifetime_seconds": 600,
                }
            },
            "does not fit the 600s node workflow lifetime",
        ),
        (
            {
                "control_env": {
                    "step_timeout_seconds": 600,
                    "node_lifetime_seconds": 3600,
                    "preemption_enabled": False,
                }
            },
            "preemption is disabled",
        ),
        ({"identity_errors": ["release rollback is active"]}, "rollback is active"),
    ],
)
def test_the_preflight_refuses_every_unsafe_start(
    override: dict[str, Any], expected: str
) -> None:
    arguments = {**happy_preflight_arguments(), **override}
    errors = destr016.preflight_errors(**arguments)
    assert any(expected in item for item in errors), errors


def test_the_preflight_refuses_a_node_that_is_not_idle_and_clean() -> None:
    arguments = happy_preflight_arguments()
    arguments["node_snapshot"] = {
        **node_baseline(),
        "unschedulable": True,
        "gpu_allocatable": 0,
    }
    errors = destr016.preflight_errors(**arguments)
    assert any(
        "is not Ready, schedulable, untainted and unowned" in i for i in errors
    ), errors
    assert any("allocates no GPU" in item for item in errors), errors


@pytest.mark.parametrize(
    ("capability", "override", "expected"),
    [
        (
            "gpuReset",
            {"owner": "someone-else"},
            "gpuReset is not OWN by the Node Agent",
        ),
        ("gpuReset", {"mode": "OBSERVE"}, "gpuReset is not OWN by the Node Agent"),
        (
            "nodeReboot",
            {"adapter": "legacy-hyperpod"},
            "nodeReboot is not OWN by the HyperPod adapter",
        ),
        (
            "nodeReboot",
            {"mode": "OBSERVE"},
            "nodeReboot is not OWN by the HyperPod adapter",
        ),
    ],
)
def test_the_preflight_refuses_a_profile_that_does_not_own_the_operations(
    capability: str, override: dict[str, str], expected: str
) -> None:
    arguments = happy_preflight_arguments()
    profile = json.loads(json.dumps(arguments["profile"]))
    for item in profile["capabilities"]:
        if item["capability"] == capability:
            item.update(override)
    arguments["profile"] = profile
    errors = destr016.preflight_errors(**arguments)
    assert any(expected in item for item in errors), errors


def test_the_preflight_refuses_a_missing_capability_or_a_warned_profile() -> None:
    arguments = happy_preflight_arguments()
    arguments["profile"] = {"capabilities": [], "warnings": ["driver mismatch"]}
    errors = destr016.preflight_errors(**arguments)
    assert any("no gpuReset capability" in item for item in errors), errors
    assert any("no nodeReboot capability" in item for item in errors), errors
    assert any("profile has warnings" in item for item in errors), errors


def test_plan_identity_is_order_independent_and_names_the_release() -> None:
    preflight = {
        "release_id": "rel-1",
        "node": {"uid": "node-uid", "boot_id": "node-boot-before"},
        "store": {"profile": {"profile_version": "v7"}},
        "runtime_identity": {"release_state": {"phase": "deployed"}},
    }
    identity = destr016.plan_identity(preflight, node=NODE)
    shuffled = dict(reversed(list(identity.items())))
    assert destr016.identity_digest(identity) == destr016.identity_digest(shuffled)
    assert identity["node"] == NODE
    assert identity["node_boot_id"] == "node-boot-before"
    assert identity["runtime_profile_version"] == "v7"


def test_evidence_components_carry_digests_only() -> None:
    details = {
        "preflight_identity": {"release_id": "rel-1"},
        "workflow": terminal_state()["workflow"],
        "incident": escalated_incident(),
        "hosts": {"boot_id_after": "host-boot-after"},
    }
    components = destr016.evidence_components(details)
    assert set(components) == set(details)
    expected = hashlib.sha256(
        json.dumps({"release_id": "rel-1"}, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert components["preflight_identity"] == expected
    assert len(destr016.case_digest(components)) == 64


def test_control_env_record_folds_disagreement_to_the_shipped_default() -> None:
    agreed = destr016.control_env_record(
        {destr016.STEP_TIMEOUT_VARIABLE: "900", destr016.NODE_LIFETIME_VARIABLE: "5400"}
    )
    assert agreed["step_timeout_seconds"] == 900
    assert agreed["node_lifetime_seconds"] == 5400
    unknown = destr016.control_env_record(
        {destr016.STEP_TIMEOUT_VARIABLE: None, destr016.NODE_LIFETIME_VARIABLE: "oops"}
    )
    assert unknown["step_timeout_seconds"] == destr016.DEFAULT_STEP_TIMEOUT_SECONDS
    assert unknown["node_lifetime_seconds"] == destr016.DEFAULT_NODE_LIFETIME_SECONDS
    assert unknown["observed"] == {
        destr016.STEP_TIMEOUT_VARIABLE: None,
        destr016.NODE_LIFETIME_VARIABLE: "oops",
    }


def test_the_window_must_have_room_for_a_real_reboot() -> None:
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 9, 6, 10, tzinfo=timezone.utc)
    assert (
        destr016.window_errors(now=now, maintenance_window_end=now + timedelta(hours=2))
        == []
    )
    errors = destr016.window_errors(
        now=now, maintenance_window_end=now + timedelta(minutes=20)
    )
    assert any("maintenance window has" in item for item in errors), errors


def test_each_injection_gets_its_own_marker() -> None:
    values = destr016.markers("destr016-1757000000-a1")
    assert set(values) == {"reset", "absorb", "escalate"}
    assert len(set(values.values())) == 3
    prefix = "destr016-1757000000-a1-"
    assert sorted(values.values()) == [prefix + suffix for suffix in ("e", "r", "s")]


def test_the_faulted_gpu_and_the_held_device_come_from_the_inventory() -> None:
    inventory = host_baseline()["gpu_inventory"]
    assert destr016.target_bdf("", inventory) == "0000:0a:00.0"
    assert destr016.target_bdf("0000:0c:00.0", inventory) == "0000:0c:00.0"
    assert destr016.holder_device("", inventory) == "/dev/nvidia0"
    assert destr016.holder_device("/dev/nvidia3", []) == "/dev/nvidia3"
    assert destr016.holder_device("", [{"index": 5, "pci_bdf": "x"}]) == "/dev/nvidia5"


def test_an_empty_inventory_cannot_be_faulted() -> None:
    from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError

    with pytest.raises(RegionalFixtureError):
        destr016.target_bdf("", [])
    with pytest.raises(RegionalFixtureError):
        destr016.holder_device("", [])


def test_parser_accepts_the_documented_arguments() -> None:
    parser = destr016.parser()
    arguments = parser.parse_args(["--run-dir", "/tmp/run"])
    assert arguments.execute is False and arguments.plan is False
    assert arguments.max_hold_seconds == 1800
    arguments = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            "DESTR016_EXECUTE",
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--node",
            NODE,
            "--pci-bdf",
            "0000:0a:00.0",
            "--holder-device",
            "/dev/nvidia0",
            "--host-probe-image",
            "image",
            "--hyperpod-cluster",
            "cluster",
            "--executor-role-arn",
            "arn:aws:iam::1:role/executor",
        ]
    )
    assert arguments.node == NODE
    assert arguments.confirm == "DESTR016_EXECUTE"
    assert arguments.holder_device == "/dev/nvidia0"
    help_text = parser.format_help()
    for option in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert option in help_text


def test_plan_details_declare_the_destructive_risk_and_the_stop_conditions() -> None:
    settings = destr016.Settings(
        regional=None,  # type: ignore[arg-type]
        node=NODE,
        host_probe_image="image",
        hyperpod_cluster="cluster",
        executor_role_arn="arn:aws:iam::1:role/executor",
        pci_bdf="0000:0a:00.0",
        device="/dev/nvidia0",
        max_hold_seconds=1800,
        predecessor_path=Path("/tmp/predecessor.json"),
    )
    preflight = {
        "release_id": "rel-1",
        "node": node_baseline(),
        "store": {"profile": {"profile_version": "v7"}},
        "runtime_identity": {},
        "predecessor": {"valid": True},
        "control_env": {"step_timeout_seconds": 600},
    }
    details = destr016.plan_details(settings, preflight)
    assert details["risk"] == "destructive-provider-reboot"
    assert details["preflight_identity_digest"] == destr016.identity_digest(
        destr016.plan_identity(preflight, node=NODE)
    )
    assert sorted(details["rollback"].values()) == [True] * 6
    joined = " ".join(details["stop_conditions"])
    for expected in (
        "predecessor evidence is not PASS",
        "arming race",
        "second workflow",
        "quiesce handoff",
        "RESET_GPU",
        "CloudTrail",
        "isolated",
    ):
        assert expected in joined, expected
    assert "BatchRebootClusterNodes" in details["mutation"]


def test_agent_operations_only_name_node_action_operations() -> None:
    """VALIDATE_GPU runs through the GPU_VALIDATION adapter; requiring it in the
    Agent's allowed_operations refused every live node (2026-09-08)."""
    from gpu_fault.models import WorkflowOperation
    from gpu_fault.operation_registry import OperationAdapter, operations_for_adapter

    node_actions = {
        item.value for item in operations_for_adapter(OperationAdapter.NODE_ACTION)
    }
    assert set(verdicts.AGENT_OPERATIONS) <= node_actions, verdicts.AGENT_OPERATIONS
    assert WorkflowOperation.VALIDATE_GPU.value not in verdicts.AGENT_OPERATIONS
