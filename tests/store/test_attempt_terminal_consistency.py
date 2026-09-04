from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import Environment, RankExitStatus, TerminalEvent, TerminalStatus
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.telemetry import WorkloadTopologyService
from gpu_fault.watcher import WorkloadPhase
from tests._builders import attempt_observation, container_observation

NOW = datetime(2026, 9, 3, 15, 1, 11, tzinfo=timezone.utc)


class LegacyInMemoryStore(InMemoryStore):
    def seed_terminal_event(self, event: TerminalEvent) -> None:
        with self._lock:
            self._events[event.event_key] = event
            self._attempt_event_keys[(event.cluster_id, event.attempt_id)] = (
                event.event_key
            )


class LegacySqliteStore(SqliteStore):
    def seed_terminal_event(self, event: TerminalEvent) -> None:
        storage_key = self._state_key((event.cluster_id, event.attempt_id))
        with self._state_transaction(f"test-terminal/{storage_key}"):
            self._put("event", event.event_key, event)
            self._link("attempt_event", storage_key, event.event_key)


def _observation(
    observed_at: datetime = NOW, *, job_id: str = "job-a", attempt_id: str = "attempt-a"
):
    return attempt_observation(
        job_id,
        attempt_id,
        observed_at,
        cluster_id="cluster-a",
        expected_critical_ranks=2,
        containers=[
            container_observation(
                "pod-0", "worker-0", 0, "node-a", gpu_uuids=["GPU-0"]
            ),
            container_observation(
                "pod-1",
                "worker-1",
                1,
                "node-b",
                terminated=True,
                exit_code=1,
                finished_at=NOW,
                gpu_uuids=["GPU-1"],
            ),
        ],
        workload_ids=["training/job/job-a"],
        runtime_profile_version="profile-a",
    )


def _terminal(
    *, job_id: str = "job-a", attempt_id: str = "attempt-a", ended_at: datetime = NOW
) -> TerminalEvent:
    return TerminalEvent(
        cluster_id="cluster-a",
        environment=Environment.HYPERPOD_EKS,
        job_id=job_id,
        attempt_id=attempt_id,
        terminal_status=TerminalStatus.TIMED_OUT,
        ended_at=ended_at,
        rank_exit_status=[
            RankExitStatus(rank=1, exit_code=1, node_id="node-b", finished_at=ended_at)
        ],
        workload_ids=[f"training/job/{job_id}"],
        runtime_profile_version="profile-a",
    )


@pytest.fixture(params=("memory", "sqlite"))
def store(request, tmp_path):
    value = (
        LegacyInMemoryStore()
        if request.param == "memory"
        else LegacySqliteStore(str(tmp_path / "state.db"))
    )
    try:
        yield value
    finally:
        close = getattr(value, "close", None)
        if close is not None:
            close()


def test_terminal_event_terminalizes_observation_and_rejects_late_active(store) -> None:
    active = _observation()
    assert store.save_attempt_observation(active), "active observation was not stored"

    assert store.save_event_if_absent(_terminal()), "terminal event was not inserted"

    state = store.list_attempt_observation_states("cluster-a")[0]
    assert state.observation.workload_phase is WorkloadPhase.FAILED
    assert [item.rank for item in state.observation.containers] == [1]
    assert state.observation.containers[0].terminated is True
    assert (
        WorkloadTopologyService(store)
        .resolve("cluster-a", "node-a", NOW + timedelta(seconds=1))
        .attempt_ids
        == []
    )

    late = active.model_copy(update={"observed_at": NOW + timedelta(minutes=1)})
    assert store.save_attempt_observation(late) is False
    assert (
        store.list_attempt_observation_states("cluster-a")[0].observation.workload_phase
        is WorkloadPhase.FAILED
    )


def test_duplicate_terminal_event_keeps_terminal_observation(store) -> None:
    store.save_attempt_observation(_observation())
    event = _terminal()

    assert store.save_event_if_absent(event), "first terminal event was not inserted"
    assert store.save_event_if_absent(event) is False
    assert (
        store.list_attempt_observation_states("cluster-a")[0].observation.workload_phase
        is WorkloadPhase.FAILED
    )


def test_cleanup_reconciles_later_contradiction_without_limit_starvation(store) -> None:
    older_observation = _observation(
        NOW - timedelta(minutes=3), job_id="job-old", attempt_id="attempt-old"
    )
    older_event = _terminal(
        job_id="job-old", attempt_id="attempt-old", ended_at=NOW - timedelta(minutes=2)
    )
    store.save_attempt_observation(older_observation)
    store.save_event_if_absent(older_event)

    active = _observation()
    store.save_attempt_observation(active)
    store.seed_terminal_event(_terminal())
    before = {
        item.observation.attempt_id: item.observation.workload_phase
        for item in store.list_attempt_observation_states("cluster-a")
    }
    assert before["attempt-a"] is WorkloadPhase.RUNNING

    result = store.cleanup_hot_state(now=NOW + timedelta(minutes=1), limit=1)

    assert result["attempt_observation_terminalized"] == 1
    states = {
        item.observation.attempt_id: item.observation.workload_phase
        for item in store.list_attempt_observation_states("cluster-a")
    }
    assert states["attempt-a"] is WorkloadPhase.FAILED
