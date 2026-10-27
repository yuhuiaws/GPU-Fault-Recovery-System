"""Shared interpretation of a collector's missing, failed or stale observations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from gpu_fault.collector_kinds import CollectorKind, collector_silent_thresholds
from gpu_fault.telemetry_models import TelemetryMetricLatest


def observation_reason(
    status: Any,
    now: datetime,
    max_age_seconds: float,
    *,
    after: datetime | None = None,
) -> str | None:
    if status is None or status.last_success_at is None:
        return "NO_SUCCESSFUL_OBSERVATION"
    age = (now - status.last_success_at).total_seconds()
    if age < 0:
        return "FUTURE_OBSERVATION"
    if age > max_age_seconds:
        return "STALE_OBSERVATION"
    if after is not None and status.last_success_at <= after:
        return "NO_POST_ACTION_OBSERVATION"
    if status.last_error_at is not None and (
        status.last_error_at >= status.last_success_at
    ):
        return "COLLECTION_ERROR"
    if status.errors:
        return "COLLECTION_ERROR"
    return None


def reported_since(
    status: Any,
    now: datetime,
    max_age_seconds: float,
    *,
    after: datetime | None = None,
) -> bool:
    """Whether the collector delivered a batch -- clean or carrying a
    collection error -- no older than ``max_age_seconds`` and after ``after``.

    ``observation_reason`` judges the last *clean* batch (``last_success_at``);
    this reads the newest batch of either kind: ``CollectorStatus.observed_at``
    is the batch's own time and the store's merge keeps the newest
    (``telemetry.merge_collector_status``). A validation that may fail a node
    on what an erroring batch still read -- its inventory rows -- asks this
    before it asks for a clean one.
    """

    if status is None or status.observed_at is None:
        return False
    age = (now - status.observed_at).total_seconds()
    if age < 0 or age > max_age_seconds:
        return False
    return after is None or status.observed_at > after


def record_observation(
    store: Any,
    cluster_id: str,
    node_id: str,
    collector: CollectorKind,
    status: Any,
    now: datetime,
    max_age_seconds: float | None = None,
) -> str | None:
    reason = observation_reason(
        status,
        now,
        collector_silent_thresholds()[collector]
        if max_age_seconds is None
        else max_age_seconds,
    )
    store.observe_telemetry_metrics(
        [
            TelemetryMetricLatest(
                cluster_id=cluster_id,
                node_id=node_id,
                observed_at=now,
                name="collector_observation_valid",
                value=0 if reason else 1,
                device=collector.value,
                labels={
                    "state": "UNKNOWN" if reason else "HEALTHY",
                    "reason": reason or "FRESH_SUCCESS",
                },
            )
        ]
    )
    return reason


def request_observation_refresh(
    context: Any, agent: Any, collectors: list[CollectorKind], now: datetime
) -> None:
    from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
    from gpu_fault.models import RecoveryAction, Severity
    from gpu_fault.orchestration.families.identity import derived_record_id

    profile = getattr(agent, "runtime_profile_version", None)
    if not profile or not collectors:
        return
    event_id = derived_record_id(
        "observation-refresh",
        "collector-health",
        agent.cluster_id,
        agent.node_id,
        str(int(now.timestamp() // 3600)),
    )
    finding = NodeHealthFinding(
        finding_id=f"finding-{event_id}",
        event_id=event_id,
        cluster_id=agent.cluster_id,
        node_id=agent.node_id,
        observed_at=now,
        category=NodeHealthCategory.SYSTEM_LOG,
        severity=Severity.WARNING,
        reason="collector observations are UNKNOWN; request a bounded fresh snapshot",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        metric_name="collector_observation_unknown",
        runtime_profile_version=profile,
        policy_source="SITE_OBSERVABILITY",
        diagnostic_parameters={"collectors": sorted(item.value for item in collectors)},
    )
    # One immutable request per node/hour; ordinary execution owns retries.
    context.orchestrator.ingest_node_health(finding)
