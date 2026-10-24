"""A telemetry check is neither an active fabric test nor proof of NCCL recovery."""

from __future__ import annotations

import math
from typing import Any

from gpu_fault.models import WorkflowOperation


def validation_evidence(
    operation: WorkflowOperation, clamped: dict[str, Any]
) -> dict[str, Any]:
    return {
        **({"inventory_requirement_clamped": clamped} if clamped else {}),
        **(
            {
                "fabric_evidence_scope": "SCOPED_TELEMETRY_ONLY",
                "active_communication_test": False,
                "nccl_recovery_proven": False,
            }
            if operation is WorkflowOperation.VALIDATE_FABRIC
            else {}
        ),
    }


def validate_task_fabric(
    latest: list[Any], scope: Any
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(scope, dict) or scope.get("basis") != "TASK_DEVICE_BINDING":
        return {
            "fabric_status": "UNKNOWN",
            "reason": "TASK_NIC_PORT_SCOPE_UNAVAILABLE",
        }, []
    interfaces, ports = scope.get("network_interfaces"), scope.get("rdma_ports")
    if (
        not isinstance(interfaces, list)
        or not isinstance(ports, list)
        or not interfaces
        and not ports
        or any(
            not isinstance(value, str) or not value for value in [*interfaces, *ports]
        )
    ):
        return {"fabric_status": "UNKNOWN", "reason": "INVALID_TASK_DEVICE_BINDING"}, []
    missing, failures = [], []
    for name, devices, healthy in (
        ("network_link_up", interfaces, 1),
        ("rdma_link_down", ports, 0),
        ("network_errors_delta", interfaces, 0),
        ("network_drops_delta", interfaces, 0),
        ("rdma_errors_delta", ports, 0),
    ):
        for device in set(devices):
            matches = [
                item for item in latest if item.name == name and item.device == device
            ]
            if len(matches) != 1 or not math.isfinite(matches[0].value):
                missing.append(f"{name}:{device}")
            elif matches[0].value != healthy:
                failures.append(f"{name}:{device}")
    for item in latest:
        if (
            item.device in {*interfaces, *ports}
            and item.name
            in {
                "rdma_errors_delta",
                "efa_rnr_errors_delta",
                "efa_retry_errors_delta",
                "efa_cq_errors_delta",
                "network_errors_delta",
                "network_drops_delta",
            }
            and item.value > 0
        ):
            failures.append(f"{item.name}:{item.device}")
    pending = (
        {"fabric_status": "UNKNOWN", "missing_task_fabric_metrics": sorted(missing)}
        if missing
        else None
    )
    return pending, sorted(set(failures))
