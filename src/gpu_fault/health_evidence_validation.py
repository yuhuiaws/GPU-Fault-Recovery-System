"""Read-only evidence gates for monitoring, training progress and thermal recovery."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Callable

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation
from gpu_fault.health_recovery_contract import (
    COOLDOWN_MAX_GAP_SECONDS,
    COOLDOWN_SECONDS,
    HEALTH_CONTRACT,
)
from gpu_fault.host_diagnostics import HOST_DIAGNOSTIC_FAILURE
from gpu_fault.observation_health import observation_reason
from gpu_fault.telemetry import CollectorKind


def pending(reason: str, **details: Any) -> WorkflowStepOutcome:
    return WorkflowStepOutcome.waiting(
        details={
            HOST_DIAGNOSTIC_FAILURE: True,
            "health_evidence_pending": reason,
            **details,
        }
    )


def validation_success(context: Any, **details: Any) -> WorkflowStepOutcome:
    return WorkflowStepOutcome.succeeded(
        operation_id=context.idempotency_key,
        details={
            "validated_nodes": context.step.node_ids,
            "validation": context.step.operation.value,
            **details,
        },
    )


def special_health_validation(
    context: Any, store: Any, now: datetime
) -> WorkflowStepOutcome | None:
    contract = context.step.parameters.get(HEALTH_CONTRACT, {})
    if not isinstance(contract, dict):
        return pending("invalid recovery contract")
    mode = contract.get("mode")
    if mode not in {"OBSERVABILITY", "TRAINING"}:
        return None
    try:
        after = datetime.fromisoformat(contract["detected_at"])
        if after.tzinfo is None:
            raise ValueError("naive event time")
        if mode == "OBSERVABILITY":
            return validate_observability(context, store, contract, now, after)
        return validate_training(context, store, contract, now, after)
    except (KeyError, TypeError, ValueError):
        return pending("invalid recovery contract")


def validate_observability(
    context: Any, store: Any, contract: dict[str, Any], now: datetime, after: datetime
) -> WorkflowStepOutcome:
    channels = [CollectorKind(value) for value in contract["collectors"]]
    if not channels or not context.step.node_ids:
        return pending("observation scope is empty")
    missing = {}
    for node in context.step.node_ids:
        statuses = {
            item.collector: item
            for item in store.list_collector_statuses(context.incident.cluster_id, node)
        }
        for channel in channels:
            status = statuses.get(channel)
            reason = observation_reason(status, now, 120, after=after)
            if (
                reason is None
                and status is not None
                and channel in {CollectorKind.GPU_METRICS, CollectorKind.GPU_INVENTORY}
                and status.sample_count == 0
            ):
                reason = "EMPTY_GPU_OBSERVATION"
            if reason is not None:
                missing[f"{node}/{channel.value}"] = reason
    if missing:
        return pending("collector state is UNKNOWN", missing_observations=missing)
    return validation_success(context, observation_state="HEALTHY")


def validate_training(
    context: Any, store: Any, contract: dict[str, Any], now: datetime, after: datetime
) -> WorkflowStepOutcome:
    attempt = contract["attempt_id"]
    rank = int(contract["rank"])
    states = [
        state
        for state in store.list_training_progress_states(context.incident.cluster_id)
        if state.heartbeat.attempt_id == attempt and state.heartbeat.rank == rank
    ]
    if len(states) != 1:
        return pending("missing progress for the original attempt/rank")
    heartbeat = states[0].heartbeat
    if (
        heartbeat.node_id not in context.step.node_ids
        or heartbeat.observed_at <= after
        or not 0 <= (now - heartbeat.observed_at).total_seconds() <= 120
        or (contract.get("pod_uid") and heartbeat.pod_uid != contract["pod_uid"])
    ):
        return pending("progress identity or freshness is not proven")
    step = contract.get("step")
    progressed = (
        states[0].last_progress_at > after
        if step is None
        else heartbeat.step is not None and heartbeat.step > step
    )
    if (
        heartbeat.numerical_error
        or heartbeat.loss is not None
        and not math.isfinite(heartbeat.loss)
        or heartbeat.step is None
        or not progressed
    ):
        return pending("original rank has not made valid application progress")
    if contract.get("kind") == "hang":
        missing = stalled_attempt_reporters(store, context, attempt, now, after)
        if missing:
            return pending(
                "attempt progress coverage is incomplete", missing_progress=missing
            )
    return validation_success(
        context,
        validated_attempt=attempt,
        validated_rank=rank,
        validated_step=heartbeat.step,
    )


def stalled_attempt_reporters(
    store: Any, context: Any, attempt: str, now: datetime, after: datetime
) -> list[str]:
    allocations = [
        state.observation
        for state in store.list_attempt_observation_states(context.incident.cluster_id)
        if state.observation.attempt_id == attempt
        and state.observation.job_id == context.incident.job_id
    ]
    if len(allocations) != 1:
        return ["allocation_unknown"]
    observation = allocations[0]
    if not 0 <= (now - observation.observed_at).total_seconds() <= 120:
        return ["allocation_stale"]
    containers = [
        item for item in observation.containers if item.critical and not item.terminated
    ]
    if not containers or len(containers) < observation.expected_critical_ranks:
        return ["allocation_incomplete"]
    states = {
        state.heartbeat.rank: state
        for state in store.list_training_progress_states(
            context.incident.cluster_id, attempt
        )
    }
    survey: dict[str, Any] = next(
        (
            item.details.get("hung_survey", {})
            for item in context.workflow.step_executions
            if item.operation is WorkflowOperation.COLLECT_HUNG_SUMMARY
        ),
        {},
    )
    expected = survey.get("expected_rank_count")
    if type(expected) is int and 0 < expected <= 65536:
        identities = {
            (item.node_id, item.pod_uid, item.container_name) for item in containers
        }
        absent = [
            str(rank)
            for rank in range(expected)
            if rank not in states
            or (
                states[rank].heartbeat.node_id,
                states[rank].heartbeat.pod_uid,
                states[rank].heartbeat.container_name,
            )
            not in identities
            or states[rank].heartbeat.step is None
            or states[rank].last_progress_at <= after
            or not 0
            <= (now - states[rank].heartbeat.observed_at).total_seconds()
            <= 120
            or states[rank].heartbeat.numerical_error
            or states[rank].heartbeat.loss is not None
            and not math.isfinite(states[rank].heartbeat.loss)
        ]
        if absent:
            return absent
    missing = []
    for container in containers:
        state = states.get(container.rank)
        if state is None:
            missing.append(str(container.rank))
            continue
        heartbeat = state.heartbeat
        if (
            heartbeat.node_id != container.node_id
            or heartbeat.pod_uid != container.pod_uid
            or heartbeat.container_name != container.container_name
            or not 0 <= (now - heartbeat.observed_at).total_seconds() <= 120
            or state.last_progress_at <= after
            or heartbeat.step is None
            or heartbeat.numerical_error
            or heartbeat.loss is not None
            and not math.isfinite(heartbeat.loss)
        ):
            missing.append(str(container.rank))
    return missing


def cooldown_key(context: Any, node_id: str) -> str:
    return f"thermal-cooldown/{context.workflow.request_id}/{node_id}"


def reset_cooldown(store: Any, context: Any, node_id: str, now: datetime) -> None:
    store.claim_health_signal_transition(cooldown_key(context, node_id), False, now)


def reset_unproven_cooldown(
    store: Any,
    context: Any,
    now: datetime,
    pending_nodes: dict[str, Any],
    failures: dict[str, Any],
) -> None:
    contract = context.step.parameters.get(HEALTH_CONTRACT, {})
    if isinstance(contract, dict) and contract.get("mode") == "THERMAL_COOLDOWN":
        for node in set(pending_nodes) | set(failures):
            reset_cooldown(store, context, node, now)


def thermal_cooldown(
    context: Any,
    metrics: Any,
    now: datetime,
    resolve_targets: Callable[[Any, str, list[Any]], list[str]],
) -> WorkflowStepOutcome | None:
    contract = context.step.parameters.get(HEALTH_CONTRACT, {})
    if not isinstance(contract, dict) or contract.get("mode") != "THERMAL_COOLDOWN":
        return None
    targets = set(context.step.gpu_uuids)
    if not targets:
        return pending("thermal scope has no GPU identity")
    node_latest = {
        node: metrics.latest(context.incident.cluster_id, node)
        for node in context.step.node_ids
    }
    owned = {
        node: set(resolve_targets(context, node, latest))
        for node, latest in node_latest.items()
    }
    if set().union(*owned.values()) != targets:
        return pending("thermal GPU ownership is incomplete")
    waiting = {}
    for node in context.step.node_ids:
        node_targets = owned[node]
        if not node_targets:
            continue
        latest = [
            item
            for item in node_latest[node]
            if item.sample.gpu_uuid in node_targets
            and item.sample.canonical_name
            in {"gpu_temperature_c", "gpu_tlimit_c", "memory_temperature_c"}
            and math.isfinite(item.sample.value)
            and 0 <= (now - item.observed_at).total_seconds() <= 120
            and item.observed_at >= context.workflow.created_at
        ]
        if {item.sample.gpu_uuid for item in latest} != node_targets:
            reset_cooldown(metrics.store, context, node, now)
            waiting[node] = "missing fresh temperature for every affected GPU"
            continue
        observed_at = min(item.observed_at for item in latest)
        key = cooldown_key(context, node)
        previous = metrics.store.get_health_signal_state(key)
        if previous is not None and observed_at <= previous.observed_at:
            waiting[node] = "waiting for a new healthy sample"
            continue
        if (
            previous is not None
            and (observed_at - previous.observed_at).total_seconds()
            > COOLDOWN_MAX_GAP_SECONDS
        ):
            reset_cooldown(metrics.store, context, node, observed_at)
        ready = metrics.store.claim_health_signal_transition(
            key, True, observed_at, minimum_active_seconds=COOLDOWN_SECONDS
        )
        if not ready:
            waiting[node] = f"healthy observations must span {COOLDOWN_SECONDS}s"
    return (
        pending("thermal cooldown not complete", node_pending=waiting)
        if waiting
        else None
    )
