from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from pydantic import ValidationError

from gpu_fault.attempt_observation_state import terminal_attempt_observation
from gpu_fault.completion_observation import (
    TERMINAL_ORIGIN_MISSING_TOMBSTONE,
    TERMINAL_ORIGIN_OBSERVED,
)
from gpu_fault.models import Environment, TerminalEvent
from gpu_fault.telemetry import (
    ATTEMPT_COVERAGE_PATH,
    WorkloadCoverageHeartbeat,
)
from gpu_fault.watcher import AttemptObservation, WorkloadPhase

LOGGER = logging.getLogger(__name__)
ACTIVE_PHASES = frozenset({WorkloadPhase.PENDING, WorkloadPhase.RUNNING})


@dataclass(frozen=True)
class AttemptSpec:
    cluster_id: str
    environment: Environment
    job_id: str
    attempt_id: str
    expected_critical_ranks: int
    runtime_profile_version: str
    cleanup_timeout_seconds: int
    workload_ids: tuple[str, ...] = ()
    checkpoint_manifest_ref: str | None = None
    termination_initiator_incident_id: str | None = None
    restart_budget: int = 1


def attempt_spec_from_observation(observation: AttemptObservation) -> AttemptSpec:
    return AttemptSpec(
        cluster_id=observation.cluster_id,
        environment=observation.environment,
        job_id=observation.job_id,
        attempt_id=observation.attempt_id,
        expected_critical_ranks=observation.expected_critical_ranks,
        runtime_profile_version=observation.runtime_profile_version,
        cleanup_timeout_seconds=observation.cleanup_timeout_seconds,
        workload_ids=tuple(observation.workload_ids),
        checkpoint_manifest_ref=observation.checkpoint_manifest_ref,
        termination_initiator_incident_id=(
            observation.termination_initiator_incident_id
        ),
        restart_budget=observation.restart_budget,
    )


def restore_persisted_attempt_observations(controller: Any) -> None:
    """Rebuild in-memory attempt state from the sink's persisted records.

    A record that cannot be parsed, or that belongs to another cluster,
    environment or is no longer active, is skipped and counted in
    ``controller.restore_skipped_total`` rather than raised: the persisted
    state is shared by every Watcher generation, and one foreign record used
    to keep the whole process from starting (F-G4 / P1-24B).
    """
    controller.restore_skipped_total = 0
    load = getattr(controller.sink, "load_attempt_observations", None)
    if load is None:
        return
    for payload in load():
        try:
            observation = AttemptObservation.model_validate(payload)
        except ValidationError as exc:
            controller.restore_skipped_total += 1
            LOGGER.warning(
                "skipping unparsable persisted attempt observation %r: %s",
                (payload or {}).get("attempt_id")
                if isinstance(payload, dict)
                else None,
                exc,
            )
            continue
        if (
            observation.cluster_id != controller.cluster_id
            or observation.environment is not controller.environment
            or observation.workload_phase not in ACTIVE_PHASES
        ):
            controller.restore_skipped_total += 1
            LOGGER.warning(
                "skipping persisted attempt observation that does not belong to "
                "this watcher: attempt=%s cluster=%s environment=%s phase=%s",
                observation.attempt_id,
                observation.cluster_id,
                observation.environment.value,
                observation.workload_phase.value,
            )
            continue
        try:
            controller.watcher.observe(observation)
        except ValueError as exc:
            controller.restore_skipped_total += 1
            LOGGER.warning(
                "skipping persisted attempt observation rejected by the watcher "
                "core: attempt=%s: %s",
                observation.attempt_id,
                exc,
            )
            continue
        controller._attempt_specs[observation.attempt_id] = (
            attempt_spec_from_observation(observation)
        )
        controller._last_observations[observation.attempt_id] = observation
        # This process has never seen a Pod of this attempt, so the absence of
        # Pods proves nothing about how it ended: the tombstone path has to ask
        # the workload object first (F7). Cleared as soon as one of its Pods is
        # listed.
        controller._restored_attempts.add(observation.attempt_id)


def cache_terminal_attempt_observation(
    controller: Any,
    attempt_id: str,
    event: TerminalEvent,
    observation: AttemptObservation,
) -> AttemptObservation:
    # The origin is decided before the tracker is cleared: only a terminal the
    # tracker itself produced from missing Pods is a tombstone, and only a
    # tombstone may be evicted later by Pods that come back (F2). A terminal
    # that was read off Pod status -- including the TIMED_OUT of an attempt
    # whose failure we did observe -- stays authoritative.
    origin = (
        TERMINAL_ORIGIN_MISSING_TOMBSTONE
        if (
            controller._missing_attempts.is_tombstoned(attempt_id)
            and observation.workload_phase is WorkloadPhase.STOPPED
        )
        else TERMINAL_ORIGIN_OBSERVED
    )
    controller._missing_attempts.clear(attempt_id)
    return cast(
        AttemptObservation,
        controller._terminal_observations.setdefault(
            attempt_id,
            terminal_attempt_observation(event, observation),
            origin,
        ),
    )


def publish_coverage_heartbeat(
    controller: Any,
    *,
    watched_pods: int,
    watched_attempts: int,
    resource_version: str | None,
) -> None:
    """Tell the control plane this pass watched the whole cluster (F4).

    Only a completed full pass calls this, which is what makes the
    statement true: a pass that raised never reaches it, and a debounced
    pass over one attempt looked at one attempt. A watcher that does not
    publish observations stays silent as well -- claiming coverage while
    the control plane cannot see the attempts that *are* running would turn
    a busy node into an IDLE one, which is the fail-open direction.

    Delivery failures are counted and dropped: the heartbeat is weak
    evidence with a natural retry (the next pass), and a control plane that
    is refusing writes must not also fail the reconcile.
    """

    if not controller.publish_observations:
        return
    heartbeat = WorkloadCoverageHeartbeat(
        cluster_id=controller.cluster_id,
        observed_at=controller.now(),
        watched_pods=watched_pods,
        watched_attempts=watched_attempts,
        resource_version=resource_version,
        watcher_instance=controller.watcher_instance,
    )
    try:
        controller.sink.post(ATTEMPT_COVERAGE_PATH, heartbeat.model_dump(mode="json"))
    except Exception:
        controller.coverage_heartbeat_failures_total += 1
        LOGGER.warning(
            "cannot publish coverage heartbeat for cluster %s",
            controller.cluster_id,
            exc_info=True,
        )
        return
    controller.coverage_heartbeats_total += 1
    controller.last_coverage_heartbeat_at = heartbeat.observed_at


def publish_attempt_observation(
    controller: Any,
    observation: AttemptObservation,
    attempt_id: str,
) -> None:
    payload = observation.model_dump(mode="json")
    if observation.workload_phase in ACTIVE_PHASES:
        persist = getattr(controller.sink, "save_attempt_observation", None)
        if persist is not None:
            try:
                persist(payload)
            except Exception:
                LOGGER.exception(
                    "cannot persist active attempt observation for %s",
                    attempt_id,
                )
    try:
        controller.sink.post("/v1/workload-observations", payload)
    except Exception:
        LOGGER.exception(
            "cannot publish workload observation for %s",
            attempt_id,
        )
        return
    if observation.workload_phase in ACTIVE_PHASES:
        return
    remove = getattr(controller.sink, "remove_attempt_observation", None)
    if remove is not None:
        try:
            remove(payload)
        except Exception:
            LOGGER.exception(
                "cannot remove persisted attempt observation for %s",
                attempt_id,
            )
