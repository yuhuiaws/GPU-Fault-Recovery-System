"""F-G7: the attempt-observation terminalization sweep on PostgreSQL.

The sweep is driven by the *observation* rows that are still non-terminal, not
by every terminal event the cluster has ever recorded (P1-50D); an event whose
observation row was removed by retention is not re-materialized (P2-50G); in
``dual`` mode the two tables are compared and written independently so the
sweep converges (P1-50E); each row is written under the row's own advisory
lock (P1-44L); a row the sweep cannot decode is skipped and does not block the
rest (progress cursor).

Needs ``GPU_FAULT_TEST_POSTGRES_URL``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from gpu_fault.models import Environment, TerminalEvent, TerminalStatus
from gpu_fault.store import PostgresStore
from gpu_fault.telemetry_models import WorkloadObservationState
from gpu_fault.watcher import WorkloadPhase
from tests._builders import attempt_observation
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is required"
)
NOW = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)


class SeedingStore(PostgresStore):
    def seed_terminal_event(self, event: TerminalEvent) -> None:
        """A terminal event row without its terminalized observation."""

        storage_key = self._state_key((event.cluster_id, event.attempt_id))
        with self._state_transaction(f"test-terminal/{storage_key}"):
            self._put("event", event.event_key, event)
            self._link("attempt_event", storage_key, event.event_key)

    def observation_key(self, cluster_id: str, attempt_id: str) -> str:
        return str(self._state_key((cluster_id, attempt_id)))

    def legacy_row(self, key: str) -> WorkloadObservationState | None:
        row = self._get_optional("attempt_observation", key)
        return row

    def dedicated_row(self, key: str) -> WorkloadObservationState | None:
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT payload FROM gpu_fault_attempt_observations WHERE key=%s",
                (key,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return WorkloadObservationState.model_validate(row[0])

    def row_lock_is_free(self, key: str) -> bool:
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT pg_try_advisory_xact_lock(
                        hashtextextended(%s, 0)
                    )
                    """,
                    (f"attempt_observation/{key}",),
                )
                return bool(cursor.fetchone()[0])

    def insert_raw_event(self, key: str, payload: dict[str, object]) -> None:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO gpu_fault_objects(kind, key, payload)
                VALUES ('event', %s, %s::jsonb)
                """,
                (key, json.dumps(payload)),
            )


def _store(mode: str) -> SeedingStore:
    assert POSTGRES_URL is not None
    for _ in postgres_store_instance():
        pass
    return SeedingStore(POSTGRES_URL, initialize_schema=False, hot_state_mode=mode)


def _terminal(cluster_id: str, attempt_id: str, ended_at: datetime) -> TerminalEvent:
    return TerminalEvent(
        cluster_id=cluster_id,
        environment=Environment.HYPERPOD_EKS,
        job_id=f"job-{attempt_id}",
        attempt_id=attempt_id,
        terminal_status=TerminalStatus.TIMED_OUT,
        ended_at=ended_at,
        runtime_profile_version="profile-a",
    )


@pytest.fixture(params=["dedicated", "dual", "legacy"])
def mode(request) -> str:
    return request.param


def test_sweep_does_not_rematerialize_an_observation_removed_by_retention(
    mode: str,
) -> None:
    """P2-50G: retention deleted the terminal row; the sweep used to read
    "event without observation" as an inconsistency and write it back, every
    tick, forever."""

    store = _store(mode)
    cluster_id = f"cluster-{uuid4().hex}"
    try:
        active = attempt_observation("job-a", "attempt-a", NOW, cluster_id=cluster_id)
        assert store.save_attempt_observation(active), "active row not stored"
        assert store.save_event_if_absent(
            _terminal(cluster_id, "attempt-a", NOW + timedelta(seconds=30))
        ), "terminal event not inserted"
        assert (
            store.list_attempt_observation_states(cluster_id)[
                0
            ].observation.workload_phase
            is WorkloadPhase.FAILED
        )

        first = store.cleanup_hot_state(
            now=NOW + timedelta(days=1), terminal_retention=timedelta(hours=1)
        )
        assert first["attempt_observation_terminalized"] == 0
        # The legacy kind in gpu_fault_objects had no retention at all, so in
        # ``legacy`` and ``dual`` mode a terminal row lived forever (P3-50J).
        assert store.list_attempt_observation_states(cluster_id) == [], (
            "retention did not remove the terminal observation"
        )

        second = store.cleanup_hot_state(
            now=NOW + timedelta(days=1), terminal_retention=timedelta(hours=1)
        )

        assert second["attempt_observation_terminalized"] == 0, (
            "the sweep re-created an observation row that retention removed"
        )
        assert store.list_attempt_observation_states(cluster_id) == []
    finally:
        store.close()
        _truncate()


def test_sweep_still_terminalizes_a_contradicted_active_row(mode: str) -> None:
    store = _store(mode)
    cluster_id = f"cluster-{uuid4().hex}"
    try:
        store.save_attempt_observation(
            attempt_observation("job-a", "attempt-a", NOW, cluster_id=cluster_id)
        )
        store.seed_terminal_event(
            _terminal(cluster_id, "attempt-a", NOW + timedelta(seconds=30))
        )

        result = store.cleanup_hot_state(now=NOW + timedelta(minutes=1))

        assert result["attempt_observation_terminalized"] == 1
        states = store.list_attempt_observation_states(cluster_id)
        assert [item.observation.workload_phase for item in states] == [
            WorkloadPhase.FAILED
        ]
        assert (
            store.cleanup_hot_state(now=NOW + timedelta(minutes=2))[
                "attempt_observation_terminalized"
            ]
            == 0
        ), "a terminalized row was picked up again"
    finally:
        store.close()
        _truncate()


def test_dual_mode_sweep_converges_when_only_the_legacy_side_is_stale() -> None:
    """P1-50E: the dedicated row is terminal, the legacy row still says RUNNING.
    The old comparison looked only at the dedicated side, found nothing to
    write, and left the legacy row as a permanent candidate."""

    dedicated_store = _store("dedicated")
    cluster_id = f"cluster-{uuid4().hex}"
    key = dedicated_store.observation_key(cluster_id, "attempt-a")
    try:
        active = attempt_observation("job-a", "attempt-a", NOW, cluster_id=cluster_id)
        dedicated_store.save_attempt_observation(active)
        dedicated_store.save_event_if_absent(
            _terminal(cluster_id, "attempt-a", NOW + timedelta(seconds=30))
        )
        # A legacy row written before the hot-state switch, never terminalized.
        dedicated_store._put(
            "attempt_observation",
            key,
            WorkloadObservationState(first_observed_at=NOW, observation=active),
        )
    finally:
        dedicated_store.close()

    dual = SeedingStore(POSTGRES_URL, initialize_schema=False, hot_state_mode="dual")
    try:
        assert dual.legacy_row(key) is not None
        assert dual.legacy_row(key).observation.workload_phase is WorkloadPhase.RUNNING

        result = dual.cleanup_hot_state(now=NOW + timedelta(minutes=1), limit=1)

        assert result["attempt_observation_terminalized"] == 1, (
            "the dual-mode sweep found nothing to write for a stale legacy row"
        )
        legacy = dual.legacy_row(key)
        assert legacy is not None
        assert legacy.observation.workload_phase is WorkloadPhase.FAILED
        dedicated = dual.dedicated_row(key)
        assert dedicated is not None
        assert dedicated.observation.workload_phase is WorkloadPhase.FAILED
        assert (
            dual.cleanup_hot_state(now=NOW + timedelta(minutes=2), limit=1)[
                "attempt_observation_terminalized"
            ]
            == 0
        ), "the dual-mode sweep did not converge"
    finally:
        dual.close()
        _truncate()


def test_sweep_writes_each_row_under_its_own_advisory_lock() -> None:
    """P1-44L: every other writer of an observation row takes
    ``attempt_observation/<key>``; the sweep took a global lock instead."""

    store = _store("dedicated")
    observer = SeedingStore(POSTGRES_URL, initialize_schema=False)
    cluster_id = f"cluster-{uuid4().hex}"
    key = store.observation_key(cluster_id, "attempt-a")
    seen: list[bool] = []
    original = store._terminalize_attempt_observation

    def terminalize_and_probe(event: TerminalEvent) -> bool:
        seen.append(observer.row_lock_is_free(key))
        return original(event)

    try:
        store.save_attempt_observation(
            attempt_observation("job-a", "attempt-a", NOW, cluster_id=cluster_id)
        )
        store.seed_terminal_event(
            _terminal(cluster_id, "attempt-a", NOW + timedelta(seconds=30))
        )
        store._terminalize_attempt_observation = terminalize_and_probe  # type: ignore[method-assign]

        result = store.cleanup_hot_state(now=NOW + timedelta(minutes=1))

        assert result["attempt_observation_terminalized"] == 1
        assert seen == [False], (
            "the sweep wrote the observation row without holding its row lock"
        )
        assert observer.row_lock_is_free(key) is True, "the row lock leaked"
    finally:
        store.close()
        observer.close()
        _truncate()


def test_sweep_skips_a_row_it_cannot_decode_and_fixes_the_rest() -> None:
    """A poisoned head used to abort every sweep at the same row."""

    store = _store("dedicated")
    cluster_id = f"cluster-{uuid4().hex}"
    try:
        # Sorted by observation key, "attempt-0-bad" precedes "attempt-1-good".
        bad = attempt_observation(
            "job-bad", "attempt-0-bad", NOW, cluster_id=cluster_id
        )
        good = attempt_observation(
            "job-good", "attempt-1-good", NOW, cluster_id=cluster_id
        )
        store.save_attempt_observation(bad)
        store.save_attempt_observation(good)
        store.insert_raw_event(
            f"{cluster_id}/attempt-0-bad/TrainingAttemptTerminal",
            {"cluster_id": cluster_id, "attempt_id": "attempt-0-bad"},
        )
        store.seed_terminal_event(
            _terminal(cluster_id, "attempt-1-good", NOW + timedelta(seconds=30))
        )

        result = store.cleanup_hot_state(now=NOW + timedelta(minutes=1), limit=1)
        result_next = store.cleanup_hot_state(now=NOW + timedelta(minutes=1), limit=1)

        terminalized = (
            result["attempt_observation_terminalized"]
            + result_next["attempt_observation_terminalized"]
        )
        assert terminalized == 1, "the healthy row behind the poisoned one never healed"
        states = {
            item.observation.attempt_id: item.observation.workload_phase
            for item in store.list_attempt_observation_states(cluster_id)
        }
        assert states["attempt-1-good"] is WorkloadPhase.FAILED
        assert states["attempt-0-bad"] is WorkloadPhase.RUNNING
    finally:
        store.close()
        _truncate()
