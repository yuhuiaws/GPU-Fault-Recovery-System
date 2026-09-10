from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.models import Environment, TerminalStatus
from gpu_fault.watcher import (
    AllocationCompleteness,
    AttemptObservation,
    AttemptPhase,
    CompletionWatcher,
    ContainerObservation,
    WorkloadPhase,
    failure_containment_ids,
)
from tests._builders import attempt_observation, container_observation

NOW = datetime(2026, 7, 20, 3, 0, tzinfo=timezone.utc)


def container(
    rank: int,
    *,
    terminated: bool,
    exit_code: int | None = None,
    node_id: str | None = None,
    gpu_uuid: str | None = None,
    gpu_count: int = 0,
    workload_log_snapshot: dict | None = None,
    finished_at: datetime | None = None,
    deletion_requested: bool = False,
) -> ContainerObservation:
    return container_observation(
        f"pod-{rank}",
        f"worker-{rank}",
        rank,
        node_id,
        gpu_uuids=[gpu_uuid] if gpu_uuid else [],
        gpu_count=gpu_count,
        workload_log_snapshot=workload_log_snapshot,
        terminated=terminated,
        exit_code=exit_code,
        finished_at=finished_at,
        deletion_requested=deletion_requested,
    )


def observation(
    *,
    phase: WorkloadPhase,
    observed_at: datetime,
    containers: list[ContainerObservation],
    attempt_id: str = "train-a1",
    initiator: str | None = None,
    restart_budget: int = 1,
) -> AttemptObservation:
    return attempt_observation(
        "train",
        attempt_id,
        observed_at,
        environment=Environment.KUBERNETES,
        workload_phase=phase,
        expected_critical_ranks=2,
        containers=containers,
        cleanup_timeout_seconds=60,
        checkpoint_manifest_ref="s3://bucket/checkpoint.json",
        termination_initiator_incident_id=initiator,
        restart_budget=restart_budget,
    )


def test_failure_detected_precedes_unique_terminal_event() -> None:
    watcher = CompletionWatcher()
    running = watcher.observe(
        observation(
            phase=WorkloadPhase.RUNNING,
            observed_at=NOW,
            containers=[
                container(0, terminated=False, node_id="node-a", gpu_uuid="GPU-a"),
                container(1, terminated=False, node_id="node-b", gpu_uuid="GPU-b"),
            ],
        )
    )
    failed = watcher.observe(
        observation(
            phase=WorkloadPhase.FAILED,
            observed_at=NOW + timedelta(seconds=10),
            containers=[
                container(
                    0,
                    terminated=True,
                    exit_code=1,
                    node_id="node-a",
                    gpu_uuid="GPU-a",
                    finished_at=NOW + timedelta(seconds=9),
                )
            ],
        )
    )
    terminal = watcher.observe(
        observation(
            phase=WorkloadPhase.FAILED,
            observed_at=NOW + timedelta(seconds=20),
            containers=[
                container(
                    1,
                    terminated=True,
                    exit_code=143,
                    node_id="node-b",
                    gpu_uuid="GPU-b",
                    finished_at=NOW + timedelta(seconds=19),
                )
            ],
        )
    )
    duplicate = watcher.observe(
        observation(
            phase=WorkloadPhase.FAILED,
            observed_at=NOW + timedelta(seconds=21),
            containers=[],
        )
    )

    assert running.phase is AttemptPhase.RUNNING
    assert failed.phase is AttemptPhase.FAILURE_DETECTED
    assert failed.failure_detected.first_failed_rank == 0
    assert failed.terminal_event is None
    assert failed.commands == ["FREEZE_EVIDENCE", "STOP_DISTRIBUTED_ATTEMPT"]
    assert terminal.phase is AttemptPhase.TERMINAL_FAILED
    assert terminal.terminal_event.terminal_status is TerminalStatus.FAILED
    assert len(terminal.terminal_event.allocation) == 2
    assert duplicate.duplicate_terminal
    assert duplicate.terminal_event.event_key == terminal.terminal_event.event_key


def test_cleanup_timeout_produces_timed_out_terminal() -> None:
    watcher = CompletionWatcher()
    watcher.observe(
        observation(
            phase=WorkloadPhase.FAILED,
            observed_at=NOW,
            containers=[
                container(
                    0,
                    terminated=True,
                    exit_code=1,
                    node_id="node-a",
                    gpu_uuid="GPU-a",
                    finished_at=NOW,
                )
            ],
        )
    )

    result = watcher.observe(
        observation(
            phase=WorkloadPhase.FAILED,
            observed_at=NOW + timedelta(seconds=61),
            containers=[],
        )
    )

    assert result.phase is AttemptPhase.TERMINAL_TIMED_OUT
    assert result.terminal_event.terminal_status is TerminalStatus.TIMED_OUT


def test_controller_initiated_nonzero_exits_are_stopped() -> None:
    watcher = CompletionWatcher()
    result = watcher.observe(
        observation(
            phase=WorkloadPhase.STOPPED,
            observed_at=NOW,
            initiator="incident-reset",
            containers=[
                container(
                    0,
                    terminated=True,
                    exit_code=143,
                    node_id="node-a",
                    gpu_uuid="GPU-a",
                    finished_at=NOW,
                ),
                container(
                    1,
                    terminated=True,
                    exit_code=143,
                    node_id="node-b",
                    gpu_uuid="GPU-b",
                    finished_at=NOW,
                ),
            ],
        )
    )

    assert result.failure_detected is None
    assert result.phase is AttemptPhase.TERMINAL_STOPPED
    assert result.terminal_event.terminal_status is TerminalStatus.STOPPED


def test_passive_fallback_initiator_replays_failure_after_restart() -> None:
    watcher = CompletionWatcher()
    incident_id, _ = failure_containment_ids(
        "cluster-a/train-a1/TrainingAttemptFailureDetected"
    )
    snapshot = {
        "record_id": "workload-log/replayed",
        "node_id": "node-a",
        "captured_at": NOW.isoformat(),
        "tail": "training output",
    }

    detected = watcher.observe(
        observation(
            phase=WorkloadPhase.STOPPED,
            observed_at=NOW,
            initiator=incident_id,
            containers=[
                container(
                    0,
                    terminated=True,
                    exit_code=1,
                    node_id="node-a",
                    gpu_count=8,
                    workload_log_snapshot=snapshot,
                    finished_at=NOW,
                ),
                container(1, terminated=False, node_id="node-b", gpu_count=8),
            ],
        )
    )
    terminal = watcher.observe(
        observation(
            phase=WorkloadPhase.STOPPED,
            observed_at=NOW + timedelta(seconds=10),
            initiator=incident_id,
            containers=[
                container(
                    1,
                    terminated=True,
                    exit_code=143,
                    node_id="node-b",
                    gpu_count=8,
                    finished_at=NOW + timedelta(seconds=9),
                )
            ],
        )
    )

    assert detected.failure_detected is not None
    assert detected.failure_detected.workload_log_snapshots == [snapshot]
    assert terminal.terminal_event is not None
    assert terminal.terminal_event.terminal_status is TerminalStatus.STOPPED


def test_terminal_event_uses_declared_gpu_count_without_uuids() -> None:
    watcher = CompletionWatcher()

    result = watcher.observe(
        observation(
            phase=WorkloadPhase.STOPPED,
            observed_at=NOW,
            initiator="incident-reset",
            containers=[
                container(
                    0,
                    terminated=True,
                    exit_code=143,
                    node_id="node-a",
                    gpu_count=8,
                    finished_at=NOW,
                ),
                container(
                    1,
                    terminated=True,
                    exit_code=143,
                    node_id="node-b",
                    gpu_count=8,
                    finished_at=NOW,
                ),
            ],
        )
    )

    assert result.terminal_event.gpu_count == 16
    assert [item.gpu_count for item in result.terminal_event.allocation] == [8, 8]


def test_restart_budget_change_is_not_attempt_identity_change() -> None:
    watcher = CompletionWatcher()
    watcher.observe(
        observation(
            phase=WorkloadPhase.RUNNING,
            observed_at=NOW,
            restart_budget=1,
            containers=[
                container(0, terminated=False, node_id="node-a", gpu_uuid="GPU-a")
            ],
        )
    )

    result = watcher.observe(
        observation(
            phase=WorkloadPhase.RUNNING,
            observed_at=NOW + timedelta(seconds=1),
            restart_budget=2,
            containers=[
                container(1, terminated=False, node_id="node-b", gpu_uuid="GPU-b")
            ],
        )
    )

    assert result.phase is AttemptPhase.RUNNING


def test_terminal_attempt_state_is_pruned_after_retention() -> None:
    watcher = CompletionWatcher(terminal_retention_seconds=60, max_attempts=100)
    watcher.observe(
        observation(
            phase=WorkloadPhase.SUCCEEDED,
            observed_at=NOW,
            attempt_id="old-attempt",
            containers=[
                container(
                    0,
                    terminated=True,
                    exit_code=0,
                    node_id="node-a",
                    gpu_uuid="GPU-a",
                    finished_at=NOW,
                ),
                container(
                    1,
                    terminated=True,
                    exit_code=0,
                    node_id="node-b",
                    gpu_uuid="GPU-b",
                    finished_at=NOW,
                ),
            ],
        )
    )
    watcher.observe(
        observation(
            phase=WorkloadPhase.RUNNING,
            observed_at=NOW + timedelta(seconds=61),
            attempt_id="new-attempt",
            containers=[
                container(0, terminated=False, node_id="node-a", gpu_uuid="GPU-a")
            ],
        )
    )

    assert "old-attempt" not in watcher._attempts
    assert watcher.pruned_attempts_total == 1


def test_missing_gpu_mapping_is_incomplete_not_guessed() -> None:
    watcher = CompletionWatcher()

    result = watcher.observe(
        observation(
            phase=WorkloadPhase.RUNNING,
            observed_at=NOW,
            containers=[
                container(0, terminated=False, node_id="node-a"),
                container(1, terminated=False, node_id="node-b"),
            ],
        )
    )

    assert result.allocation_completeness is AllocationCompleteness.INCOMPLETE


def test_success_requires_all_expected_critical_ranks() -> None:
    watcher = CompletionWatcher()
    partial = watcher.observe(
        observation(
            phase=WorkloadPhase.SUCCEEDED,
            observed_at=NOW,
            containers=[
                container(
                    0,
                    terminated=True,
                    exit_code=0,
                    node_id="node-a",
                    gpu_uuid="GPU-a",
                    finished_at=NOW,
                )
            ],
        )
    )
    complete = watcher.observe(
        observation(
            phase=WorkloadPhase.SUCCEEDED,
            observed_at=NOW + timedelta(seconds=1),
            containers=[
                container(
                    1,
                    terminated=True,
                    exit_code=0,
                    node_id="node-b",
                    gpu_uuid="GPU-b",
                    finished_at=NOW + timedelta(seconds=1),
                )
            ],
        )
    )

    assert partial.terminal_event is None
    assert complete.phase is AttemptPhase.TERMINAL_SUCCEEDED
    assert complete.terminal_event.terminal_status is TerminalStatus.SUCCEEDED


def test_nonzero_exits_of_pods_being_deleted_are_a_stop_not_a_failure() -> None:
    """A graceful delete SIGTERMs torchrun, which exits 1.

    COLLECT-021's teardown deleted its PyTorchJob; the watcher read the exit
    as a training failure, the passive path spent the last restart on it and
    ESCALATED. A Pod the API was asked to remove is a stop, like a Pod that
    has already vanished -- no FAILED, no budget, no escalation.
    """

    watcher = CompletionWatcher()
    result = watcher.observe(
        observation(
            phase=WorkloadPhase.STOPPED,
            observed_at=NOW,
            containers=[
                container(
                    0,
                    terminated=True,
                    exit_code=1,
                    node_id="node-a",
                    gpu_uuid="GPU-a",
                    finished_at=NOW,
                    deletion_requested=True,
                ),
                container(
                    1,
                    terminated=True,
                    exit_code=1,
                    node_id="node-b",
                    gpu_uuid="GPU-b",
                    finished_at=NOW,
                    deletion_requested=True,
                ),
            ],
        )
    )

    assert result.failure_detected is None, "a requested deletion is not a failure"
    assert result.phase is AttemptPhase.TERMINAL_STOPPED
    assert result.terminal_event.terminal_status is TerminalStatus.STOPPED
