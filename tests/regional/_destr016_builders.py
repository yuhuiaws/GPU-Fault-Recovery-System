"""Synthetic snapshot builders for the GF-REGIONAL-DESTR-016 contract tests.

The fixtures encode the two shapes the case would otherwise assert wrongly:
the successor's step graph is rewired at claim time (the appended
``RESTORE_GPU_SERVICES`` is the *last* index and ``VALIDATE_GPU`` depends on
it), and the superseded reset's barrier step record stays WAITING while its
*remote command* is the thing that gets cancelled. Nothing here touches a
cluster.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import destr016_verdicts as verdicts

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
    incident["official_action"] = "RESTART_BM"
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
