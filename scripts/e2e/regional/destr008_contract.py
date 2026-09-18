"""Pure shortage verdicts and shared fixture bounds; no live operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

from scripts.e2e.regional.destr008_journal import SCENARIOS
from scripts.e2e.regional.warm_spare_fixture import (
    INSTANCE_TYPE_LABELS,
    QUARANTINE_TAINT,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
)

EXPECTED_REASON = {
    "no-spare": (
        "warm-spare replacement is required; "
        "provider node replacement API fallback is disabled"
    ),
    "topology-mismatch": "insufficient healthy HyperPod spares",
    "kubernetes-not-ready": "Kubernetes node is not Ready",
    "reserved-by-other": "reserved by incident",
    "active-gpu-pod": "active GPU resource pods exist",
    "agent-unavailable": "node agent is not fleet-ready",
}
ALERT_SCENARIOS = set(SCENARIOS) - {"no-spare"}
SERVICE_UNIT = {
    "kubernetes-not-ready": "kubelet.service",
    "agent-unavailable": "gpu-fault-node-agent.service",
}
WORKFLOW_WAIT_SECONDS = 900
BOUND_MARGIN_SECONDS = 60


def instance_type(snapshot: dict[str, Any]) -> str | None:
    return next(
        (
            snapshot["labels"].get(key)
            for key in INSTANCE_TYPE_LABELS
            if snapshot["labels"].get(key)
        ),
        None,
    )


def agent_by_node(state: dict[str, Any], node: str) -> dict[str, Any] | None:
    matches = [
        item for item in state.get("agents") or [] if item.get("node_id") == node
    ]
    return cast(dict[str, Any], matches[0]) if len(matches) == 1 else None


def observation_gpu_uuids(observation: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(gpu_uuid)
            for container in observation.get("containers", [])
            for gpu_uuid in container.get("gpu_uuids", [])
        }
    )


def capability(profile: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    for item in (profile or {}).get("capabilities", []):
        if isinstance(item, dict) and item.get("capability") == name:
            return cast(dict[str, Any], item)
    return None


def profile_errors(profile: dict[str, Any] | None) -> list[str]:
    expected = {
        "nodeReplace": ("gpu-fault-hyperpod-adapter", "regional-cluster-executor"),
        "workloadStop": (
            "gpu-fault-kubernetes-adapter",
            "regional-cluster-executor",
        ),
        "workloadRestart": (
            "gpu-fault-kubernetes-adapter",
            "regional-cluster-executor",
        ),
    }
    errors = []
    for name, (owner, adapter) in expected.items():
        item = capability(profile, name)
        if item is None:
            errors.append(f"runtime profile has no {name} capability")
        elif (
            item.get("mode") != "OWN"
            or item.get("owner") != owner
            or item.get("adapter") != adapter
        ):
            errors.append(f"{name} has the wrong owner or adapter")
    return errors


def scenario_admission_errors(
    scenarios: tuple[str, ...] | list[str],
    *,
    capabilities: dict[str, Any] | None = None,
) -> list[str]:
    bounded = sorted(set(scenarios) & {*SERVICE_UNIT, "active-gpu-pod"})
    if not bounded:
        return []
    populations = (capabilities or {}).get("populations")
    if (
        (capabilities or {}).get("supported") is True
        and isinstance(populations, list)
        and len(populations) == 2
        and {row.get("plane") for row in populations if isinstance(row, dict)}
        == {"cpu", "gpu"}
        and all(row.get("pods") and row.get("probes") for row in populations)
    ):
        return []
    return [
        "bounded shortage requires deployed activation inhibition and independent "
        "cancellation capability: " + ", ".join(bounded)
    ]


def terminal_execution(
    workflow: dict[str, Any], operation: str
) -> dict[str, Any] | None:
    matches = [
        item
        for item in workflow.get("step_executions", [])
        if item.get("operation") == operation
        and item.get("status") in {"SUCCEEDED", "FAILED"}
    ]
    return cast(dict[str, Any], matches[-1]) if matches else None


def wait_timeout_seconds(
    bound_at: datetime | None,
    now: datetime,
    *,
    default: int = WORKFLOW_WAIT_SECONDS,
    margin: int = BOUND_MARGIN_SECONDS,
) -> int:
    """Keep the workflow wait inside the fixture's original shortage window."""
    if bound_at is None:
        return default
    remaining = int((bound_at - now).total_seconds()) - margin
    return max(1, min(default, remaining))


def bound_errors(
    replace: dict[str, Any] | None,
    *,
    bound_at: datetime | None,
    label: str,
) -> list[str]:
    """A conclusion after fixture expiry does not prove the intended shortage."""
    if bound_at is None or replace is None:
        return []
    concluded_text = str(replace.get("updated_at") or replace.get("started_at") or "")
    if not concluded_text:
        return [f"REPLACE_NODE has no timestamp to compare with the {label}"]
    concluded = datetime.fromisoformat(concluded_text.replace("Z", "+00:00"))
    if concluded.tzinfo is None:
        concluded = concluded.replace(tzinfo=timezone.utc)
    if concluded >= bound_at:
        return [
            f"the {label} fired at {bound_at.isoformat()} before REPLACE_NODE "
            f"concluded at {concluded.isoformat()}; the shortage was not holding"
        ]
    return []


def scenario_errors(
    state: dict[str, Any],
    settings: object,
    scenario: str,
    *,
    event_id: str,
) -> list[str]:
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    errors = []
    if workflow.get("status") != "FAILED":
        errors.append("shortage workflow is not FAILED")
    stop = terminal_execution(workflow, "STOP_WORKLOADS")
    if stop is None or stop.get("status") != "SUCCEEDED":
        errors.append("STOP_WORKLOADS did not execute successfully")
    replace = terminal_execution(workflow, "REPLACE_NODE")
    if replace is None or replace.get("status") != "FAILED":
        errors.append("REPLACE_NODE did not execute and fail")
    elif EXPECTED_REASON[scenario] not in str(replace.get("error") or ""):
        errors.append("REPLACE_NODE failure reason does not match the scenario")
    replace_step = next(
        (
            item
            for item in workflow.get("official_steps", [])
            if item.get("operation") == "REPLACE_NODE"
        ),
        None,
    )
    if (replace_step or {}).get("parameters") != {
        "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY",
        "activation_forbidden": True,
    }:
        errors.append(
            "REPLACE_NODE lacks its nonactivating warm-spare authority "
            "(HEALTHY_WARM_SPARE_ONLY with activation_forbidden=true)"
        )
    if any(
        "ACTIVATION_FORBIDDEN:" in str(execution.get("error") or "")
        or any(
            key in (execution.get("details") or {})
            and execution["details"][key] is not False
            for key in ("activation_inhibited", "cached_activation_rejected")
        )
        for execution in workflow.get("step_executions") or []
    ):
        errors.append(
            "activation inhibition fired instead of the intended shortage gate"
        )
    alerts = [
        item
        for item in state.get("notifications") or []
        if "hyperpod-spare-insufficient" in str(item.get("deduplication_key") or "")
    ]
    expected_alerts = 1 if scenario in ALERT_SCENARIOS else 0
    if len(alerts) != expected_alerts:
        errors.append(
            f"expected {expected_alerts} spare-insufficient alerts, got {len(alerts)}"
        )
    # Only the injected finding's marker may own this drill's incident.
    marker_ids = sorted(
        str(item.get("marker_id") or "") for item in state.get("markers") or []
    )
    if marker_ids != [f"marker-{event_id}"]:
        errors.append(
            "incident markers are not exactly the injected finding's own marker: "
            f"{marker_ids}"
        )
    fault = state.get("fault_node") or {}
    if not fault.get("unschedulable") or not any(
        item.get("key") == QUARANTINE_TAINT for item in fault.get("taints") or []
    ):
        errors.append("fault node is not kept quarantined")
    incident_id = str(incident.get("incident_id") or "")
    spare = state.get("spare_node") or {}
    if spare.get("annotations", {}).get(SPARE_RESERVATION_ANNOTATION) == incident_id:
        errors.append("failed scenario left an incident spare reservation")
    if spare.get("annotations", {}).get(SPARE_POOL_STATE_ANNOTATION) == "ALLOCATED":
        errors.append("failed scenario left the spare ALLOCATED")
    return errors
