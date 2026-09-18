from __future__ import annotations

from typing import Any


def restart_state(gpu_count: int) -> dict[str, Any]:
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


def reset_host_pair(
    *, minimum: int, last: int, journal_resets: int = 0
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger_before = [
        {"command_id": "cmd-old", "operation": "QUIESCE_GPU_SERVICES", "attempt": 1}
    ]
    baseline = {
        "gpu_inventory": [
            {"pci_bdf": f"0000:{index + 0x59:02x}:00", "uuid": f"GPU-{index}"}
            for index in range(8)
        ],
        "ledger": list(ledger_before),
        "services": {"kubelet.service": {"ActiveState": "active"}},
        "gpu_fault_timers": ["gpu-fault-certificate-check.timer"],
    }
    after = {
        **baseline,
        "compute_clients": [],
        "quiesce_states": [],
        "ledger": ledger_before
        + [
            {
                "command_id": "cmd-reset",
                "operation": "RESET_GPU",
                "attempt": 1,
                "state": "SUCCEEDED",
                "gpu_uuids": ["GPU-0"],
            }
        ],
        "kernel_reset_journal": {"target_reset_count": journal_resets},
        "sampler": {
            "sample_count": 40,
            "min_gpu_count": minimum,
            "observed_gpu_uuid_sets": [
                [f"GPU-{index}" for index in range(8)],
                [f"GPU-{index}" for index in range(8 - minimum, 8)],
            ],
            "last": {
                "gpu_count": last,
                "gpu_uuids": [f"GPU-{index}" for index in range(last)],
            },
        },
    }
    return baseline, after
