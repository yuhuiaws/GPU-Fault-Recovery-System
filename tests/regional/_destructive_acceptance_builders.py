"""Pure builders extracted without replacing the stronger local reset fixtures."""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import regional_live_fixture as live_fixture_module
from scripts.e2e.regional import run_destr001_gpu_reset as destr001
from tests.regional._destructive_acceptance_support import restart_state


def _reset_state() -> dict[str, Any]:
    waiting = []
    for operation in (
        "QUIESCE_GPU_SERVICES",
        "VERIFY_NO_GPU_CLIENTS",
        "RESET_GPU",
        "RESTORE_GPU_SERVICES",
    ):
        waiting.append(
            {
                "operation": operation,
                "status": "WAITING",
                "details": {"mutation_submitted_by_control_plane": False},
            }
        )
    return {
        "event": {"xid": 46, "evidence_ref": "kmsg://node/boot/1"},
        "decision": {"official_action": "RESET_GPU"},
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [
                {"operation": operation} for operation in destr001.EXPECTED_STEPS
            ],
            "completed_operations": list(destr001.EXPECTED_STEPS),
            "step_executions": [
                {
                    "operation": operation,
                    "status": "SUCCEEDED",
                    "adapter_operation_id": f"remote/{operation.lower()}",
                }
                for operation in (
                    "QUIESCE_GPU_SERVICES",
                    "VERIFY_NO_GPU_CLIENTS",
                    "RESET_GPU",
                    "RESTORE_GPU_SERVICES",
                )
            ],
        },
        "observed_waiting_step_executions": waiting,
        "commands": [
            {"status": "SUCCEEDED", "step": {"operation": operation}}
            for operation in (
                "MARK_UNSCHEDULABLE",
                "QUIESCE_GPU_SERVICES",
                "VERIFY_NO_GPU_CLIENTS",
                "RESET_GPU",
                "RESTORE_GPU_SERVICES",
                "RESTORE_SCHEDULING",
            )
        ],
    }


def _restart_state(gpu_count: int) -> dict[str, Any]:
    return restart_state(gpu_count)


# What a no-summary-may-copy-it field looks like in the planted records below.
PLANTED_SECRET = "planted-secret-0123456789abcdef"
RESET_BUSY_ERROR = (
    "node agent {node}: ResetProgressError: nvidia-smi exited with status 255: "
    "The following GPUs could not be reset:\n  GPU 00000000:59:00.0: In use by "
    "another client\n\n1 device is currently being used by one or more other "
    "processes (e.g., Fabric Manager, CUDA application, graphics application such "
    "as an X server, or a monitoring application such as another instance of "
    "nvidia-smi). Please first kill all processes using this device and all "
    "compute applications running in the system.; device holders: "
    "44878:nvidia-persiste:/dev/nvidia1,/dev/nvidia2,/dev/nvidiactl "
    "(GPU GPU-0, 1 of 1)"
)
ESCALATION_STEPS = (
    "RESTART_NODE",
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
)


def escalated_reset_state(
    node: str = "node-a", *, restart_status: str = "SUCCEEDED"
) -> dict[str, Any]:
    """A store snapshot shaped like DESTR-001's escalated reset on plain EKS.

    The batched node-action command [QUIESCE, VERIFY, RESET, RESTORE] FAILED at
    RESET_GPU (a busy GPU), the product escalated the branch to RESTART_NODE
    (BRANCH_ESCALATED plus the PLAN_REWRITE that replaced the branch), rebooted
    and validated the node and closed the workflow SUCCEEDED. With another
    ``restart_status`` the reboot rung did not complete and the workflow
    FAILED. Lease digests, fencing tokens, challenge nonces and signatures are
    planted in the command and step details; no summary may copy them.
    """

    error = RESET_BUSY_ERROR.format(node=node)
    restarted = restart_status == "SUCCEEDED"
    initial = [*destr001.EXPECTED_STEPS]
    steps = [
        *(
            {"operation": item, "node_ids": [node], "branch_id": "branch:initial"}
            for item in initial
        ),
        *(
            {"operation": item, "node_ids": [node], "branch_id": f"branch:{node}"}
            for item in ESCALATION_STEPS
        ),
    ]
    planted = {
        "remote_fencing_token": 7,
        "physical_ownership_checks": [
            {"challenge_nonce": PLANTED_SECRET, "signature": PLANTED_SECRET}
        ],
    }
    executions: list[dict[str, Any]] = [
        {
            "operation": item,
            "step_index": index,
            "status": "SUCCEEDED",
            "adapter_operation_id": f"remote/{item.lower()}",
            "error": None,
            "details": {},
        }
        for index, item in enumerate(initial[:4])
    ]
    executions.append(
        {
            "operation": "RESET_GPU",
            "step_index": 4,
            "status": "FAILED",
            "adapter_operation_id": "remote/quiesce_gpu_services",
            "error": error,
            "details": {"remote_status": "FAILED", "reset_failed": ["GPU-0"]} | planted,
        }
    )
    executions.append(
        {
            "operation": "RESTART_NODE",
            "step_index": 8,
            "status": restart_status,
            "adapter_operation_id": "remote/restart_node",
            "error": None if restarted else "reboot was not confirmed",
            "details": {"action": "REBOOT"},
        }
    )
    if restarted:
        executions += [
            {"operation": item, "step_index": 8 + index, "status": "SUCCEEDED"}
            for index, item in enumerate(ESCALATION_STEPS)
            if index
        ]
    escalation_reason = f"RESET_GPU failed on {node} ({error[:420]}…"
    events = [
        {
            "kind": "STEP_ATTEMPT",
            "code": "STEP_FAILED",
            "operation": "RESET_GPU",
            "step_index": 4,
            "status": "FAILED",
            "reason": error,
            "details": {"attempt": 2},
        },
        {
            "kind": "PLAN_REWRITE",
            "code": "BRANCH_REPLACED",
            "operation": None,
            "step_index": None,
            "status": None,
            "reason": f"branch replaced: branch:{node}",
            "details": {
                "branch_id": f"branch:{node}",
                "node_id": node,
                "superseded_indexes": [4, 5, 6, 7],
                "appended_indexes": [8, 9, 10, 11, 12],
            },
        },
        {
            "kind": "BRANCH_ESCALATION",
            "code": "BRANCH_ESCALATED",
            "operation": "RESET_GPU",
            "phase": "official",
            "step_index": 4,
            "status": "FAILED",
            "at": "2030-01-01T00:01:00Z",
            "reason": escalation_reason,
            "details": {
                "branch_id": "branch:initial",
                "from_operation": "RESET_GPU",
                "to_operation": "RESTART_NODE",
                "node_id": node,
                "lineage_node_id": node,
                "rung_count": 1,
                "max_rungs": 2,
                "to_branch_id": f"branch:{node}",
            },
        },
    ]
    commands = [
        {
            "command_id": "remote-cordon",
            "status": "SUCCEEDED",
            "step_index": 1,
            "step": {"operation": "MARK_UNSCHEDULABLE", "node_ids": [node]},
        },
        {
            "command_id": "remote-batch",
            "status": "FAILED",
            "step_index": 2,
            "error": error,
            "step": {"operation": "QUIESCE_GPU_SERVICES", "node_ids": [node]},
            "batched_steps": [
                {
                    "step_index": index,
                    "idempotency_key": f"wf-destr001-escalated/{index}/{item}",
                    "step": {"operation": item, "node_ids": [node]},
                }
                for index, item in ((3, "VERIFY_NO_GPU_CLIENTS"), (4, "RESET_GPU"))
            ],
            "fencing_token": 7,
            "lease_token_sha256": PLANTED_SECRET,
            "result_details": planted,
        },
        {
            "command_id": "remote-reboot",
            "status": restart_status,
            "step_index": 8,
            "step": {"operation": "RESTART_NODE", "node_ids": [node]},
            "result_details": {"action": "REBOOT"},
        },
    ]
    completed = [*initial[:4], *(ESCALATION_STEPS if restarted else ())]
    return {
        "event": {"xid": 46, "evidence_ref": f"kmsg://{node}/1"},
        "decision": {"official_action": "RESET_GPU"},
        "incident": {
            "incident_id": "inc-destr001-escalated",
            "state": "RECOVERED" if restarted else "QUARANTINED",
        },
        "workflow": {
            "request_id": "wf-destr001-escalated",
            "status": "SUCCEEDED" if restarted else "FAILED",
            "official_action": "RESTART_NODE",
            "dag_enabled": True,
            "official_steps": steps,
            "completed_operations": completed,
            "superseded_step_indexes": [4, 5, 6, 7],
            "branch_escalation_counts": {node: 1},
            "step_executions": executions,
            "events": events,
        },
        "observed_waiting_step_executions": [],
        "commands": commands,
        "notifications": [],
    }


def _runtime_identity(*, phase: str = "complete") -> dict[str, Any]:
    deployments = {}
    for plane, names in live_fixture_module.RUNTIME_IDENTITY_DEPLOYMENTS.items():
        deployments[plane] = {
            name: {
                "generation": 1,
                "desired_replicas": 2,
                "observed_generation": 1,
                "updated_replicas": 2,
                "ready_replicas": 2,
                "available_replicas": 2,
                "template_sha256": "a" * 64,
                "images": ["registry.example/runtime@sha256:" + "b" * 64],
            }
            for name in names
        }
    return {
        "release_state": {
            "release_id": "release-a",
            "phase": phase,
            "transaction_committed": phase == "complete",
        },
        "deployments": deployments,
    }
