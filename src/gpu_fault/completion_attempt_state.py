from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from pydantic import ValidationError

from gpu_fault.attempt_observation_state import terminal_attempt_observation
from gpu_fault.models import Environment, TerminalEvent
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


def cache_terminal_attempt_observation(
    controller: Any,
    attempt_id: str,
    event: TerminalEvent,
    observation: AttemptObservation,
) -> AttemptObservation:
    controller._missing_attempts.clear(attempt_id)
    return cast(
        AttemptObservation,
        controller._terminal_observations.setdefault(
            attempt_id,
            terminal_attempt_observation(event, observation),
        ),
    )


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
