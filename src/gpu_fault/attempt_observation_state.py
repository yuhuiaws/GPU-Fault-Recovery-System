from __future__ import annotations

from gpu_fault.models import TerminalEvent, TerminalStatus
from gpu_fault.telemetry_models import WorkloadObservationState
from gpu_fault.watcher import AttemptObservation, WorkloadPhase

TERMINAL_WORKLOAD_PHASES = frozenset(
    {
        WorkloadPhase.SUCCEEDED,
        WorkloadPhase.FAILED,
        WorkloadPhase.STOPPED,
    }
)


def attempt_observation_state_is_terminal(
    state: WorkloadObservationState | None,
) -> bool:
    return (
        state is not None
        and state.observation.workload_phase in TERMINAL_WORKLOAD_PHASES
    )


def terminal_workload_phase(status: TerminalStatus) -> WorkloadPhase:
    if status is TerminalStatus.SUCCEEDED:
        return WorkloadPhase.SUCCEEDED
    if status in {TerminalStatus.FAILED, TerminalStatus.TIMED_OUT}:
        return WorkloadPhase.FAILED
    return WorkloadPhase.STOPPED


def terminal_attempt_observation(
    event: TerminalEvent,
    previous: AttemptObservation | None,
) -> AttemptObservation:
    rank_statuses = {item.rank: item for item in event.rank_exit_status}
    containers = []
    if previous is not None:
        for container in previous.containers:
            status = rank_statuses.get(container.rank)
            if status is None:
                continue
            containers.append(
                container.model_copy(
                    update={
                        "terminated": True,
                        "exit_code": status.exit_code,
                        "signal": status.signal,
                        "finished_at": status.finished_at or event.ended_at,
                    }
                )
            )
    known_ranks = {
        *rank_statuses,
        *(item.rank for item in event.allocation if item.rank is not None),
    }
    expected_ranks = (
        previous.expected_critical_ranks
        if previous is not None
        else max(1, max(known_ranks, default=-1) + 1)
    )
    observed_at = max(
        event.ended_at,
        previous.observed_at if previous is not None else event.ended_at,
    )
    return AttemptObservation(
        cluster_id=event.cluster_id,
        environment=event.environment,
        job_id=event.job_id,
        attempt_id=event.attempt_id,
        workload_phase=terminal_workload_phase(event.terminal_status),
        observed_at=observed_at,
        started_at=previous.started_at if previous is not None else None,
        expected_critical_ranks=expected_ranks,
        containers=containers,
        workload_ids=(
            list(event.workload_ids)
            if event.workload_ids
            else list(previous.workload_ids)
            if previous is not None
            else []
        ),
        cleanup_timeout_seconds=(
            previous.cleanup_timeout_seconds if previous is not None else 120
        ),
        checkpoint_manifest_ref=(
            event.checkpoint_manifest_ref
            if event.checkpoint_manifest_ref is not None
            else previous.checkpoint_manifest_ref
            if previous is not None
            else None
        ),
        termination_initiator_incident_id=(
            event.termination_initiator_incident_id
            if event.termination_initiator_incident_id is not None
            else previous.termination_initiator_incident_id
            if previous is not None
            else None
        ),
        runtime_profile_version=event.runtime_profile_version,
        restart_budget=event.restart_budget,
    )


def terminal_attempt_observation_state(
    event: TerminalEvent,
    previous: WorkloadObservationState | None,
) -> WorkloadObservationState:
    observation = terminal_attempt_observation(
        event,
        previous.observation if previous is not None else None,
    )
    return WorkloadObservationState(
        first_observed_at=(
            previous.first_observed_at
            if previous is not None
            else observation.observed_at
        ),
        observation=observation,
    )
