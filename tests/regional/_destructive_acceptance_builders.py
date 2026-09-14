"""Pure state builders shared by the destructive acceptance fixture tests.

Split out of ``test_destructive_acceptance_fixtures.py`` so that file stays
under the architecture size ceiling; these helpers build only dicts.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import regional_live_fixture as live_fixture_module
from scripts.e2e.regional import run_destr001_gpu_reset as destr001


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
    waiting: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for operation in ("STOP_WORKLOADS", "RESTART_WORKLOAD"):
        waiting.append(
            {
                "operation": operation,
                "status": "WAITING",
                "details": {"mutation_submitted_by_control_plane": False},
            }
        )
        details: dict[str, Any] = {}
        if operation == "RESTART_WORKLOAD":
            details = {
                "notification_context": {
                    "source_gpu_count": gpu_count,
                    "target_gpu_count": gpu_count,
                    "restart_count": 1,
                }
            }
        executions.append(
            {
                "operation": operation,
                "status": "SUCCEEDED",
                "adapter_operation_id": f"remote/{operation.lower()}",
                "details": details,
            }
        )
    return {
        "event": {"xid": 11},
        "decision": {"official_action": "RESTART_APP"},
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [
                {
                    "operation": "FREEZE_EVIDENCE",
                    "execution_owner": "gpu-fault-control-plane",
                },
                {
                    "operation": "STOP_WORKLOADS",
                    "execution_owner": "gpu-fault-kubernetes-adapter",
                },
                {
                    "operation": "RESTART_WORKLOAD",
                    "execution_owner": "gpu-fault-kubernetes-adapter",
                },
            ],
            "step_executions": executions,
        },
        "observed_waiting_step_executions": waiting,
        "commands": [
            {"status": "SUCCEEDED", "step": {"operation": operation}}
            for operation in ("STOP_WORKLOADS", "RESTART_WORKLOAD")
        ],
        "restart_budget": {"budget": 1, "restart_count": 1},
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
        "release_state": {"release_id": "release-a", "phase": phase},
        "deployments": deployments,
    }
