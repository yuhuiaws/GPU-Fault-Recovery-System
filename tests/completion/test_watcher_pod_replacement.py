"""A replaced Pod replaces its predecessor in the attempt state (F-G4).

``_AttemptState.containers`` was the union of every container ever observed
for the attempt, keyed by Pod UID. A Pod that the job controller replaced
(evicted, preempted, or recreated after a non-zero exit) therefore stayed in
the state forever: a still-"running" ghost kept the attempt from ever reaching
a terminal, and a failed ghost put a stale exit into the terminal event even
after its replacement succeeded.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.models import Environment, TerminalStatus
from gpu_fault.watcher import (
    AttemptPhase,
    CompletionWatcher,
    ContainerObservation,
    WorkloadPhase,
)
from tests._builders import attempt_observation, container_observation

NOW = datetime(2026, 7, 20, 3, 0, tzinfo=timezone.utc)


def container(
    pod_uid: str,
    rank: int,
    node_id: str,
    *,
    exit_code: int | None = None,
    finished_at: datetime | None = None,
) -> ContainerObservation:
    return container_observation(
        pod_uid,
        f"worker-{rank}",
        rank,
        node_id,
        gpu_uuids=[f"GPU-{node_id}"],
        terminated=exit_code is not None,
        exit_code=exit_code,
        finished_at=finished_at,
    )


def observe(watcher: CompletionWatcher, phase: WorkloadPhase, at: datetime, containers):
    return watcher.observe(
        attempt_observation(
            "train",
            "train-a1",
            at,
            environment=Environment.KUBERNETES,
            workload_phase=phase,
            expected_critical_ranks=2,
            containers=containers,
            cleanup_timeout_seconds=60,
        )
    )


def test_a_replaced_running_pod_does_not_block_the_terminal() -> None:
    watcher = CompletionWatcher()
    observe(
        watcher,
        WorkloadPhase.RUNNING,
        NOW,
        [container("pod-0a", 0, "node-a"), container("pod-1", 1, "node-b")],
    )

    result = observe(
        watcher,
        WorkloadPhase.SUCCEEDED,
        NOW + timedelta(seconds=10),
        [
            container("pod-0b", 0, "node-a", exit_code=0, finished_at=NOW),
            container("pod-1", 1, "node-b", exit_code=0, finished_at=NOW),
        ],
    )

    assert result.phase is AttemptPhase.TERMINAL_SUCCEEDED
    assert result.terminal_event is not None
    assert result.terminal_event.terminal_status is TerminalStatus.SUCCEEDED
    assert [(item.rank, item.node_id) for item in result.terminal_event.allocation] == [
        (0, "node-a"),
        (1, "node-b"),
    ]


def test_a_replaced_failed_pod_leaves_no_stale_exit_in_the_terminal() -> None:
    watcher = CompletionWatcher()
    failed = observe(
        watcher,
        WorkloadPhase.FAILED,
        NOW,
        [
            container("pod-0a", 0, "node-a", exit_code=1, finished_at=NOW),
            container("pod-1", 1, "node-b"),
        ],
    )
    assert failed.failure_detected is not None

    result = observe(
        watcher,
        WorkloadPhase.FAILED,
        NOW + timedelta(seconds=10),
        [
            container("pod-0b", 0, "node-c", exit_code=137, finished_at=NOW),
            container("pod-1", 1, "node-b", exit_code=143, finished_at=NOW),
        ],
    )

    assert result.terminal_event is not None
    assert result.terminal_event.terminal_status is TerminalStatus.FAILED
    assert [
        (item.rank, item.exit_code) for item in result.terminal_event.rank_exit_status
    ] == [(0, 137), (1, 143)]
    assert [item.node_id for item in result.terminal_event.allocation] == [
        "node-c",
        "node-b",
    ]
