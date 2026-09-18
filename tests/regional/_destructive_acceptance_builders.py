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
