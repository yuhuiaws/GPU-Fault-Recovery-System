"""A telemetry check is neither an active fabric test nor proof of NCCL recovery."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from gpu_fault.models import WorkflowOperation

#: The diagnostics whose VALIDATE_FABRIC judges the task's own NICs and RDMA
#: ports; a healthy management link must not pass them.
TASK_SCOPED_REASONS = frozenset(
    {"EFA_TRAFFIC_HUNG_SUSPECTED", "TRAINING_HUNG", "APPLICATION_NCCL_ERROR"}
)
SCOPE_UNAVAILABLE = "TASK_NIC_PORT_SCOPE_UNAVAILABLE"
#: What the recheck survey said about each node it could not bind
#: (``hung_survey.after_summary``), kept on the VALIDATE_FABRIC step.
UNAVAILABLE_SCOPES = "training_fabric_unavailable_by_node"
_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")


def requires_task_scope(parameters: dict[str, Any]) -> bool:
    return bool(parameters.get("require_task_fabric_scope")) or (
        parameters.get("diagnostic_reason") in TASK_SCOPED_REASONS
    )


def task_scope_problem(scope: Any) -> str | None:
    """Why ``scope`` cannot name the task's NICs and ports, ``None`` if it can."""

    if not isinstance(scope, dict) or scope.get("basis") != "TASK_DEVICE_BINDING":
        return SCOPE_UNAVAILABLE
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
        return "INVALID_TASK_DEVICE_BINDING"
    return None


_EVIDENCE_CODES = (
    ("basis", "scope_basis"),
    ("reason", "scope_reason"),
    ("source", "scope_source"),
)
_EVIDENCE_COUNTS = (
    ("expected_pids", "expected_process_count"),
    ("covered_pids", "covered_process_count"),
)


def scope_evidence(scope: Any) -> dict[str, Any]:
    """A node survey's own account of its unbound scope, bounded to codes and counts."""

    evidence: dict[str, Any] = {"scope_basis": "NOT_REPORTED"}
    if not isinstance(scope, dict):
        return evidence
    for key, name in _EVIDENCE_CODES:
        value = scope.get(key)
        if isinstance(value, str) and _CODE.fullmatch(value):
            evidence[name] = value
    for key, name in _EVIDENCE_COUNTS:
        if isinstance(scope.get(key), list):
            evidence[name] = len(scope[key])
    return evidence


def reported_evidence(value: Any) -> dict[str, Any]:
    """``scope_evidence`` as stored on the step, read back with the same bounds."""

    evidence: dict[str, Any] = {"scope_basis": "NOT_REPORTED"}
    if not isinstance(value, dict):
        return evidence
    for _, name in _EVIDENCE_CODES:
        code = value.get(name)
        if isinstance(code, str) and _CODE.fullmatch(code):
            evidence[name] = code
    for _, name in _EVIDENCE_COUNTS:
        count = value.get(name)
        if type(count) is int and count >= 0:
            evidence[name] = count
    return evidence


def unknown_task_fabric(step: Any) -> dict[str, dict[str, Any]]:
    """The explicit UNKNOWN verdict of each step node without a usable task scope.

    Only a task-scoped VALIDATE_FABRIC has one. The scope comes from the hung
    DAG's recheck survey (or an explicit binding); once that step has run
    nothing fills it in, so waiting for it only runs out the per-step cap. The
    caller decides whether the verdict may resolve the step: only when a human
    hand-off follows, so an UNKNOWN can never end a workflow RECOVERED.
    """

    parameters = step.parameters
    if step.operation is not WorkflowOperation.VALIDATE_FABRIC or not (
        requires_task_scope(parameters)
    ):
        return {}
    bindings = parameters.get("training_fabric_by_node")
    bindings = bindings if isinstance(bindings, dict) else {}
    reported = parameters.get(UNAVAILABLE_SCOPES)
    reported = reported if isinstance(reported, dict) else {}
    verdicts: dict[str, dict[str, Any]] = {}
    for node_id in step.node_ids:
        problem = task_scope_problem(bindings.get(node_id))
        if problem is None:
            continue
        verdicts[node_id] = {
            "fabric_status": "UNKNOWN",
            "reason": problem,
            **(
                reported_evidence(reported.get(node_id))
                if problem == SCOPE_UNAVAILABLE
                else {}
            ),
        }
    return verdicts


def validation_evidence(
    operation: WorkflowOperation,
    clamped: dict[str, Any],
    unknown: dict[str, dict[str, Any]] | None = None,
    node_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """What a passed validation proved. ``unknown``: nodes whose task scope could
    not be determined; they are reported, never counted as validated."""

    evidence = {
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
    if unknown:
        validated = [node_id for node_id in node_ids if node_id not in unknown]
        evidence.update(
            fabric_status="UNKNOWN",
            fabric_evidence_scope=(
                "PARTIAL_TASK_SCOPE" if validated else "TASK_SCOPE_UNAVAILABLE"
            ),
            validated_nodes=validated,
            unvalidated_nodes=sorted(unknown),
            task_fabric_unknown_nodes=unknown,
        )
    return evidence


def validate_task_fabric(
    latest: list[Any], scope: Any
) -> tuple[dict[str, Any] | None, list[str]]:
    problem = task_scope_problem(scope)
    if problem is not None:
        return {"fabric_status": "UNKNOWN", "reason": problem}, []
    interfaces, ports = scope["network_interfaces"], scope["rdma_ports"]
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
