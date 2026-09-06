from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from enum import StrEnum
from threading import RLock

from pydantic import Field, model_validator

from gpu_fault.models import (
    AllocationEntry,
    Environment,
    RankExitStatus,
    StrictModel,
    TerminalEvent,
    TerminalStatus,
)


class WorkloadPhase(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class AttemptPhase(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILURE_DETECTED = "FAILURE_DETECTED"
    TERMINAL_SUCCEEDED = "TERMINAL_SUCCEEDED"
    TERMINAL_FAILED = "TERMINAL_FAILED"
    TERMINAL_STOPPED = "TERMINAL_STOPPED"
    TERMINAL_TIMED_OUT = "TERMINAL_TIMED_OUT"


class AllocationCompleteness(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    MISSING = "MISSING"


class ContainerObservation(StrictModel):
    pod_uid: str
    pod_name: str
    container_name: str
    container_id: str | None = None
    host_pid: int | None = Field(default=None, ge=1)
    cgroup_path: str | None = None
    role: str
    rank: int = Field(ge=0)
    critical: bool = True
    node_id: str | None = None
    instance_id: str | None = None
    gpu_uuids: list[str] = Field(default_factory=list)
    gpu_count: int = Field(default=0, ge=0)
    workload_log_snapshot: dict | None = None
    fabric_partition: str | None = None
    terminated: bool = False
    exit_code: int | None = None
    signal: int | None = Field(default=None, ge=0)
    finished_at: datetime | None = None
    restart_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_termination(self) -> ContainerObservation:
        if self.terminated and self.exit_code is None:
            raise ValueError("terminated container observation requires exit_code")
        if not self.terminated and (
            self.exit_code is not None or self.finished_at is not None
        ):
            raise ValueError("running container cannot have exit_code or finished_at")
        return self

    @property
    def observation_key(self) -> str:
        return f"{self.pod_uid}/{self.container_name}"


class AttemptObservation(StrictModel):
    cluster_id: str
    environment: Environment
    job_id: str
    attempt_id: str
    workload_phase: WorkloadPhase
    observed_at: datetime
    started_at: datetime | None = None
    expected_critical_ranks: int = Field(ge=1)
    containers: list[ContainerObservation] = Field(default_factory=list)
    workload_ids: list[str] = Field(default_factory=list)
    cleanup_timeout_seconds: int = Field(default=120, ge=1, le=3600)
    checkpoint_manifest_ref: str | None = None
    termination_initiator_incident_id: str | None = None
    runtime_profile_version: str
    restart_budget: int = Field(default=1, ge=0)

    @property
    def gpu_count(self) -> int:
        declared = sum(container.gpu_count for container in self.containers)
        if declared:
            return declared
        return len(
            {
                gpu_uuid
                for container in self.containers
                for gpu_uuid in container.gpu_uuids
            }
        )


class FailureDetectedEvent(StrictModel):
    cluster_id: str
    job_id: str
    attempt_id: str
    detected_at: datetime
    runtime_profile_version: str | None = None
    workload_ids: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)
    gpu_uuids: list[str] = Field(default_factory=list)
    first_failed_rank: int | None = None
    node_id: str | None = None
    exit_code: int | None = None
    reason: str
    allocation_completeness: AllocationCompleteness
    workload_log_snapshots: list[dict] = Field(default_factory=list)

    @property
    def event_key(self) -> str:
        return f"{self.cluster_id}/{self.attempt_id}/TrainingAttemptFailureDetected"


def failure_containment_ids(event_key: str) -> tuple[str, str]:
    suffix = hashlib.sha256(event_key.encode()).hexdigest()[:20]
    return (
        f"inc-attempt-failure-{suffix}",
        f"workflow-attempt-stop-{suffix}",
    )


class FailureContainmentDecision(StrictModel):
    attempt_id: str
    event_key: str
    incident_id: str
    workflow_request_id: str
    duplicate: bool = False


class WatcherResult(StrictModel):
    attempt_id: str
    phase: AttemptPhase
    allocation_completeness: AllocationCompleteness
    failure_detected: FailureDetectedEvent | None = None
    terminal_event: TerminalEvent | None = None
    commands: list[str] = Field(default_factory=list)
    duplicate_terminal: bool = False


class _AttemptState:
    def __init__(self, observation: AttemptObservation) -> None:
        self.phase = AttemptPhase.PENDING
        self.containers: dict[str, ContainerObservation] = {}
        self.failure: FailureDetectedEvent | None = None
        self.cleanup_deadline: datetime | None = None
        self.terminal: TerminalEvent | None = None
        self.last_observation = observation


class CompletionWatcher:
    """Runtime-neutral core for a Kubernetes/Slurm workload adapter."""

    def __init__(
        self,
        *,
        terminal_retention_seconds: int = 3600,
        max_attempts: int = 10000,
    ) -> None:
        if terminal_retention_seconds < 60:
            raise ValueError("terminal attempt retention must be at least 60 seconds")
        if max_attempts < 100:
            raise ValueError("watcher max attempts must be at least 100")
        self._attempts: dict[str, _AttemptState] = {}
        self._lock = RLock()
        self.terminal_retention_seconds = terminal_retention_seconds
        self.max_attempts = max_attempts
        self.pruned_attempts_total = 0
        # Attempt ids pruned since the owner last asked. The controller keeps
        # its own per-attempt state (specs, cached terminal observations, sent
        # keys) and has to let go of the same attempts, or it re-feeds the
        # cached terminal every pass and this core re-creates what it pruned.
        self._pruned_attempt_ids: set[str] = set()

    def reset_attempt(self, attempt_id: str) -> None:
        with self._lock:
            self._attempts.pop(attempt_id, None)

    def take_pruned_attempt_ids(self) -> set[str]:
        """Attempt ids pruned since the previous call; the set is then cleared."""
        with self._lock:
            pruned = set(self._pruned_attempt_ids)
            self._pruned_attempt_ids.clear()
            return pruned

    def observe(self, observation: AttemptObservation) -> WatcherResult:
        with self._lock:
            self._prune(
                observation.observed_at,
                keep_attempt_id=observation.attempt_id,
            )
            state = self._attempts.get(observation.attempt_id)
            if state is None:
                state = _AttemptState(observation)
                self._attempts[observation.attempt_id] = state
            else:
                self._validate_identity(state.last_observation, observation)

            if state.terminal is not None:
                return WatcherResult(
                    attempt_id=observation.attempt_id,
                    phase=state.phase,
                    allocation_completeness=self._completeness(
                        state, observation.expected_critical_ranks
                    ),
                    terminal_event=state.terminal,
                    duplicate_terminal=True,
                )

            state.last_observation = observation
            for container in observation.containers:
                # A Pod replaced by the job controller (evicted, preempted,
                # recreated after a non-zero exit) takes over its rank. The
                # predecessor is not a separate rank of the attempt; keeping
                # it made a still-"running" ghost block the terminal forever
                # and a failed ghost put a stale exit into the terminal event.
                for key, previous in list(state.containers.items()):
                    if (
                        previous.rank == container.rank
                        and key != container.observation_key
                    ):
                        del state.containers[key]
                state.containers[container.observation_key] = container

            critical = [item for item in state.containers.values() if item.critical]
            completeness = self._completeness(
                state, observation.expected_critical_ranks
            )
            failed = sorted(
                (
                    item
                    for item in critical
                    if item.terminated
                    and item.exit_code is not None
                    and item.exit_code != 0
                ),
                key=lambda item: (
                    item.finished_at or observation.observed_at,
                    item.pod_uid,
                    item.container_name,
                ),
            )
            initiated_stop = bool(observation.termination_initiator_incident_id)
            passive_failure_incident, _ = failure_containment_ids(
                (
                    f"{observation.cluster_id}/"
                    f"{observation.attempt_id}/"
                    "TrainingAttemptFailureDetected"
                )
            )
            passive_failure_stop = (
                observation.termination_initiator_incident_id
                == passive_failure_incident
            )
            has_failure = (not initiated_stop or passive_failure_stop) and (
                bool(failed) or observation.workload_phase is WorkloadPhase.FAILED
            )
            emitted_failure: FailureDetectedEvent | None = None
            commands: list[str] = []

            if has_failure and state.failure is None:
                first = failed[0] if failed else None
                state.failure = FailureDetectedEvent(
                    cluster_id=observation.cluster_id,
                    job_id=observation.job_id,
                    attempt_id=observation.attempt_id,
                    detected_at=observation.observed_at,
                    runtime_profile_version=(observation.runtime_profile_version),
                    workload_ids=observation.workload_ids,
                    node_ids=sorted(
                        {item.node_id for item in critical if item.node_id}
                    ),
                    gpu_uuids=sorted(
                        {gpu_uuid for item in critical for gpu_uuid in item.gpu_uuids}
                    ),
                    first_failed_rank=first.rank if first else None,
                    node_id=first.node_id if first else None,
                    exit_code=first.exit_code if first else None,
                    reason=(
                        "critical container exited non-zero"
                        if first
                        else "workload adapter reported FAILED"
                    ),
                    allocation_completeness=completeness,
                    workload_log_snapshots=[
                        item.workload_log_snapshot
                        for item in critical
                        if item.workload_log_snapshot is not None
                    ],
                )
                state.cleanup_deadline = observation.observed_at + timedelta(
                    seconds=observation.cleanup_timeout_seconds
                )
                state.phase = AttemptPhase.FAILURE_DETECTED
                emitted_failure = state.failure
                commands = [
                    "FREEZE_EVIDENCE",
                    "STOP_DISTRIBUTED_ATTEMPT",
                ]

            all_terminal = len(critical) >= observation.expected_critical_ranks and all(
                item.terminated for item in critical
            )
            terminal_status: TerminalStatus | None = None
            if initiated_stop and all_terminal:
                terminal_status = TerminalStatus.STOPPED
            elif state.failure is not None and all_terminal:
                terminal_status = TerminalStatus.FAILED
            elif (
                state.failure is not None
                and state.cleanup_deadline is not None
                and observation.observed_at >= state.cleanup_deadline
            ):
                terminal_status = TerminalStatus.TIMED_OUT
            elif (
                observation.workload_phase is WorkloadPhase.SUCCEEDED
                and all_terminal
                and all(item.exit_code == 0 for item in critical)
            ):
                terminal_status = TerminalStatus.SUCCEEDED
            elif observation.workload_phase is WorkloadPhase.STOPPED and (
                all_terminal or not observation.containers
            ):
                terminal_status = TerminalStatus.STOPPED

            if terminal_status is not None:
                state.terminal = self._terminal_event(
                    observation, state, terminal_status
                )
                state.phase = {
                    TerminalStatus.SUCCEEDED: (AttemptPhase.TERMINAL_SUCCEEDED),
                    TerminalStatus.FAILED: AttemptPhase.TERMINAL_FAILED,
                    TerminalStatus.STOPPED: AttemptPhase.TERMINAL_STOPPED,
                    TerminalStatus.TIMED_OUT: (AttemptPhase.TERMINAL_TIMED_OUT),
                }[terminal_status]
            elif state.failure is None:
                state.phase = (
                    AttemptPhase.PENDING
                    if observation.workload_phase is WorkloadPhase.PENDING
                    else AttemptPhase.RUNNING
                )

            return WatcherResult(
                attempt_id=observation.attempt_id,
                phase=state.phase,
                allocation_completeness=completeness,
                failure_detected=emitted_failure,
                terminal_event=state.terminal,
                commands=commands,
            )

    def _prune(
        self,
        observed_at: datetime,
        *,
        keep_attempt_id: str,
    ) -> None:
        cutoff = observed_at - timedelta(seconds=self.terminal_retention_seconds)
        removable = [
            (attempt_id, state)
            for attempt_id, state in self._attempts.items()
            if attempt_id != keep_attempt_id and state.terminal is not None
        ]
        expired = {
            attempt_id
            for attempt_id, state in removable
            if state.terminal is not None and state.terminal.ended_at <= cutoff
        }
        remaining_count = len(self._attempts) - len(expired)
        overflow = max(0, remaining_count - self.max_attempts)
        if overflow > 0:
            # Only the overflow path needs the oldest-first order; sorting on
            # every observation made each pass O(N log N) for nothing.
            removable.sort(
                key=lambda item: (
                    item[1].terminal.ended_at if item[1].terminal else observed_at,
                    item[0],
                )
            )
            for attempt_id, _state in removable:
                if attempt_id in expired:
                    continue
                if overflow <= 0:
                    break
                expired.add(attempt_id)
                overflow -= 1
        for attempt_id in expired:
            self._attempts.pop(attempt_id, None)
        self.pruned_attempts_total += len(expired)
        self._pruned_attempt_ids.update(expired)

    def _terminal_event(
        self,
        observation: AttemptObservation,
        state: _AttemptState,
        terminal_status: TerminalStatus,
    ) -> TerminalEvent:
        critical = sorted(
            (item for item in state.containers.values() if item.critical),
            key=lambda item: (item.rank, item.observation_key),
        )
        allocation = [
            AllocationEntry(
                node_id=item.node_id,
                instance_id=item.instance_id,
                rank=item.rank,
                gpu_uuids=item.gpu_uuids,
                gpu_count=item.gpu_count,
                fabric_partition=item.fabric_partition,
            )
            for item in critical
            if item.node_id
        ]
        exits = [
            RankExitStatus(
                rank=item.rank,
                exit_code=item.exit_code,
                node_id=item.node_id or "UNKNOWN",
                signal=item.signal,
                finished_at=item.finished_at,
            )
            for item in critical
            if item.terminated and item.exit_code is not None
        ]
        return TerminalEvent(
            cluster_id=observation.cluster_id,
            environment=observation.environment,
            job_id=observation.job_id,
            attempt_id=observation.attempt_id,
            terminal_status=terminal_status,
            ended_at=observation.observed_at,
            rank_exit_status=exits,
            allocation=allocation,
            workload_ids=observation.workload_ids,
            checkpoint_manifest_ref=(observation.checkpoint_manifest_ref),
            termination_initiator_incident_id=(
                observation.termination_initiator_incident_id
            ),
            runtime_profile_version=(observation.runtime_profile_version),
            restart_budget=observation.restart_budget,
        )

    @staticmethod
    def _completeness(state: _AttemptState, expected: int) -> AllocationCompleteness:
        critical = [item for item in state.containers.values() if item.critical]
        if not critical or not any(item.node_id for item in critical):
            return AllocationCompleteness.MISSING
        if (
            len(critical) < expected
            or any(not item.node_id for item in critical)
            or any(not item.gpu_uuids for item in critical)
        ):
            return AllocationCompleteness.INCOMPLETE
        return AllocationCompleteness.COMPLETE

    @staticmethod
    def _validate_identity(
        previous: AttemptObservation,
        current: AttemptObservation,
    ) -> None:
        identity = (
            "cluster_id",
            "environment",
            "job_id",
            "attempt_id",
            "runtime_profile_version",
        )
        changed = [
            field
            for field in identity
            if getattr(previous, field) != getattr(current, field)
        ]
        if changed:
            raise ValueError(
                "attempt identity changed across observations: " + ", ".join(changed)
            )
