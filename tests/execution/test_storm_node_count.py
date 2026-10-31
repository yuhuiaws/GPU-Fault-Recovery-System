"""The remediation storm breaker's GPU-node count when the fleet read fails (§9.5).

``storm_admission.cluster_gpu_nodes`` reads a cluster's ACTIVE/DRAINING fleet
Agents at most once per ``NODE_COUNT_TTL_SECONDS``. A failed read used to be
cached as ``None`` for that minute, so one transient store error made a
256-node cluster's breaker judge at the floor of 3 for 60 s. A failure is
never cached now: the last good count, while younger than
``NODE_COUNT_LAST_GOOD_SECONDS``, judges instead (the floor once it is that
old, or when there is none), the next read follows a short back-off, and every
failed read is logged and counted by what judged instead.

The tests drive the count with explicit ``time.monotonic()`` readings (``now``)
and give every case its own store, so no count carries over between them.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.app.metric_aggregation import STRATEGIES, Strategy
from gpu_fault.app.metrics import METRIC_CONTRIBUTORS
from gpu_fault.app.remediation_budget_metrics import (
    STORM_NODE_COUNT_READ_FAILURES,
    storm_node_count_metric_lines,
)
from gpu_fault.departed_nodes import DEPARTED_AGENT_GUARD_AGE
from gpu_fault.execution import storm_admission
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.fleet import AgentLifecycleState, AgentRecord
from gpu_fault.models import WorkflowStatus
from gpu_fault.remediation_storm import STORM_MIN_NODES, storm_threshold
from gpu_fault.store import InMemoryStore
from tests.execution._budget_hold_support import (
    REBOOT_PLAN,
    ScriptedAdapter,
    budget_dispatcher,
    seed,
)

CLUSTER = "cluster-a"
TTL = storm_admission.NODE_COUNT_TTL_SECONDS
LAST_GOOD = storm_admission.NODE_COUNT_LAST_GOOD_SECONDS
RETRY = storm_admission.NODE_COUNT_RETRY_SECONDS
LAST_GOOD_FALLBACK = storm_admission.FALLBACK_LAST_GOOD
FLOOR_FALLBACK = storm_admission.FALLBACK_FLOOR
ADOT_CONTROL_PLANE = (
    Path(__file__).resolve().parents[2] / "deploy/observability/adot-control-plane.yaml"
)


class FleetStore:
    """``list_agents`` answering ``nodes`` ACTIVE Agents, or raising ``error``.

    A read takes the answer the store holds when it begins, then runs
    ``inside_read`` (where the thread cases park their readers) and answers;
    ``reads`` counts the reads that reached the store.
    """

    def __init__(self, nodes: int) -> None:
        self.nodes = nodes
        self.error: Exception | None = None
        self.inside_read: Callable[[], object] = lambda: None
        self.reads = 0
        self._reads_lock = threading.Lock()

    def list_agents(self, cluster_id: str) -> list[SimpleNamespace]:
        with self._reads_lock:
            self.reads += 1
        error, nodes = self.error, self.nodes
        self.inside_read()
        if error is not None:
            raise error
        agent = SimpleNamespace(
            cluster_id=cluster_id, lifecycle_state=AgentLifecycleState.ACTIVE
        )
        return [agent] * nodes


class SlottedFleetStore:
    """A store that cannot be weakly referenced (``__slots__``, no ``__weakref__``)."""

    __slots__ = ("error", "nodes", "reads")

    def __init__(self, nodes: int) -> None:
        self.nodes = nodes
        self.error: Exception | None = None
        self.reads = 0

    def list_agents(self, cluster_id: str) -> list[SimpleNamespace]:
        self.reads += 1
        if self.error is not None:
            raise self.error
        agent = SimpleNamespace(
            cluster_id=cluster_id, lifecycle_state=AgentLifecycleState.ACTIVE
        )
        return [agent] * self.nodes


def count(store: object, now: float) -> int | None:
    return storm_admission.cluster_gpu_nodes(store, CLUSTER, now=now)


def failed_reads_since(before: dict[str, int]) -> dict[str, int]:
    after = storm_admission.node_count_read_failures()
    return {fallback: after[fallback] - before[fallback] for fallback in after}


def unavailable() -> RuntimeError:
    return RuntimeError("agent table unavailable")


def test_a_failed_read_after_a_good_one_judges_with_the_last_good_count() -> None:
    store = FleetStore(256)
    assert count(store, 0.0) == 256
    store.error = unavailable()
    before = storm_admission.node_count_read_failures()

    nodes = count(store, TTL + 1)

    assert nodes == 256, "one transient store error must not drop the count"
    assert storm_threshold(nodes) == 52
    assert store.reads == 2, "the expired count was read again"
    assert failed_reads_since(before) == {LAST_GOOD_FALLBACK: 1, FLOOR_FALLBACK: 0}


def test_a_failed_read_with_no_good_count_is_the_floor() -> None:
    store = FleetStore(256)
    store.error = unavailable()
    before = storm_admission.node_count_read_failures()

    nodes = count(store, 0.0)

    assert nodes is None, "nothing was ever counted: the count is unknown"
    assert storm_threshold(nodes) == STORM_MIN_NODES == 3
    assert failed_reads_since(before) == {LAST_GOOD_FALLBACK: 0, FLOOR_FALLBACK: 1}


def test_the_read_after_a_failure_goes_to_the_store_not_to_a_cached_none() -> None:
    store = FleetStore(256)
    store.error = unavailable()
    assert count(store, 0.0) is None, "the floor"
    # During the short back-off the failing store is left alone, and the
    # answer is the failure's own (the floor here), not a count kept for a TTL.
    assert count(store, RETRY / 2) is None, "the floor during the back-off"
    assert store.reads == 1, "no read during the back-off"
    store.error = None

    assert count(store, RETRY) == 256, "the store answers again: so does the count"
    assert store.reads == 2, "the first read after the back-off goes to the store"
    assert RETRY < TTL, "before, the failure was the answer for the whole TTL"
    assert count(store, RETRY + TTL - 1) == 256
    assert store.reads == 2, "a good count is kept for the TTL"


def test_the_last_good_count_expires_into_the_floor() -> None:
    store = FleetStore(256)
    assert count(store, 0.0) == 256
    store.error = unavailable()
    # A failed read just inside the bound still has the last good count, and
    # so does its back-off while the count is younger than the bound ...
    assert count(store, LAST_GOOD - RETRY) == 256
    assert count(store, LAST_GOOD - 0.1) == 256
    before = storm_admission.node_count_read_failures()

    # ... and the first failed read as old as the bound has none: the floor.
    assert count(store, LAST_GOOD) is None, "past the bound: the floor"
    assert failed_reads_since(before) == {LAST_GOOD_FALLBACK: 0, FLOOR_FALLBACK: 1}
    assert store.reads == 3, "the back-off answer read nothing"


def test_the_bounds_stay_within_what_the_count_already_tolerates() -> None:
    assert 0 < RETRY < TTL, "the back-off is short and never as long as the TTL"
    # A departed node's Agent stays counted for at least the 10-minute
    # departure guard, until the departed-node cleanup retires it: a last good
    # count is never older than the staleness the count already accepts.
    assert TTL < LAST_GOOD < DEPARTED_AGENT_GUARD_AGE.total_seconds()


def test_a_failure_streak_logs_its_traceback_once(caplog) -> None:
    store = FleetStore(8)
    assert count(store, 0.0) == 8
    store.error = unavailable()
    caplog.set_level(logging.WARNING, logger=storm_admission.__name__)

    count(store, TTL + 1)  # the streak's first failure
    count(store, TTL + 1 + RETRY)  # the same streak
    store.error = None
    count(store, TTL + 1 + 2 * RETRY)  # a good read ends it
    store.error = unavailable()
    count(store, 2 * TTL + 2 + 2 * RETRY)  # a new streak

    records = [r for r in caplog.records if r.name == storm_admission.__name__]
    assert [record.exc_info is not None for record in records] == [True, False, True]
    assert {"agent table unavailable" in r.getMessage() for r in records} == {True}
    assert "last good count, 8 nodes" in records[0].getMessage()


def test_concurrent_failed_reads_are_each_counted_under_the_module_lock() -> None:
    readers = 8
    store = FleetStore(64)
    assert count(store, 0.0) == 64
    store.error = unavailable()
    barrier = threading.Barrier(readers, timeout=10)
    store.inside_read = barrier.wait
    before = storm_admission.node_count_read_failures()
    answers: list[int | None] = []
    answers_lock = threading.Lock()

    def judge() -> None:
        nodes = count(store, TTL + 1)
        with answers_lock:
            answers.append(nodes)

    workers = [threading.Thread(target=judge) for _ in range(readers)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)

    assert not [worker for worker in workers if worker.is_alive()], "a reader hung"
    assert not barrier.broken, (
        "every reader was inside the store read at once: the lock guards the "
        "count, never the read"
    )
    assert answers == [64] * readers
    assert store.reads == 1 + readers
    assert failed_reads_since(before) == {
        LAST_GOOD_FALLBACK: readers,
        FLOOR_FALLBACK: 0,
    }
    assert count(store, TTL + 2) == 64, "the back-off they set answers"
    assert store.reads == 1 + readers, "and reads nothing"


def test_a_slow_read_never_replaces_a_newer_count() -> None:
    store = FleetStore(10)
    entered, release = threading.Event(), threading.Event()

    def park_the_slow_read() -> None:
        if threading.current_thread().name == "slow-read":
            entered.set()
            release.wait(timeout=10)

    store.inside_read = park_the_slow_read
    slow_answers: list[int | None] = []
    slow = threading.Thread(
        target=lambda: slow_answers.append(count(store, 0.0)), name="slow-read"
    )
    slow.start()
    assert entered.wait(timeout=10), "the slow read never reached the store"
    store.nodes = 20  # ten nodes joined; a read that begins now sees them

    assert count(store, 1.0) == 20
    release.set()
    slow.join(timeout=10)

    assert not slow.is_alive(), "the slow read never finished"
    assert slow_answers == [10], "the slow read answers its own caller"
    assert count(store, 2.0) == 20, "the newer count stays"
    assert store.reads == 2, "and stays fresh"


def test_a_store_that_cannot_be_weakly_referenced_is_read_every_time() -> None:
    store = SlottedFleetStore(12)

    assert count(store, 0.0) == 12
    assert count(store, 1.0) == 12
    assert store.reads == 2, "nothing is kept for it"
    store.error = unavailable()
    assert count(store, 2.0) is None, "and a failure has no last good count"


def adot_keep_regex() -> str:
    documents = yaml.safe_load_all(ADOT_CONTROL_PLANE.read_text(encoding="utf-8"))
    config = next(item for item in documents if item and item["kind"] == "ConfigMap")
    scrape = yaml.safe_load(config["data"]["collector.yaml"])["receivers"][
        "prometheus"
    ]["config"]["scrape_configs"][0]
    return str(
        next(
            item["regex"]
            for item in scrape["metric_relabel_configs"]
            if item["action"] == "keep" and item["source_labels"] == ["__name__"]
        )
    )


def test_failed_reads_render_as_a_per_process_counter_by_fallback() -> None:
    store = FleetStore(4)
    store.error = unavailable()
    assert count(store, 0.0) is None, "the floor"
    failures = storm_admission.node_count_read_failures()

    lines = storm_node_count_metric_lines(None)

    name = STORM_NODE_COUNT_READ_FAILURES
    assert f"# TYPE {name} counter" in lines
    # Both series are always rendered, so rate() sees a fallback's first one.
    assert f'{name}{{fallback="last_good"}} {failures[LAST_GOOD_FALLBACK]}' in lines
    assert f'{name}{{fallback="floor"}} {failures[FLOOR_FALLBACK]}' in lines
    assert failures[FLOOR_FALLBACK] >= 1
    assert STRATEGIES[name] is Strategy.PER_PROCESS, "each process is a reset domain"
    assert re.fullmatch(adot_keep_regex(), name), "the ADOT keep list ships it"
    # The count is read on the claim path, i.e. where the dispatcher runs.
    assert "remediation-storm-node-count" in METRIC_CONTRIBUTORS.fleet_level_names


class AgentReadFailingStore(InMemoryStore):
    """An in-memory store whose Agent reads fail while ``agent_reads_fail``."""

    def __init__(self) -> None:
        super().__init__()
        self.agent_reads_fail = False

    def list_agents(self, cluster_id: str | None = None) -> list[AgentRecord]:
        if self.agent_reads_fail:
            raise unavailable()
        return list(super().list_agents(cluster_id))


def fleet_agent(index: int, now: datetime) -> AgentRecord:
    return AgentRecord(
        cluster_id=CLUSTER,
        node_id=f"fleet-{index:03d}",
        endpoint="https://10.0.0.7:9099",
        agent_version="1.0.0",
        artifact_sha256="a" * 64,
        policy_version="610",
        runtime_profile_version="active-v1",
        config_digest="b" * 64,
        allowed_operations=[],
        first_seen_at=now,
        last_seen_at=now,
    )


@pytest.mark.parametrize("readers", [1, 4])
def test_a_large_cluster_whose_fleet_read_fails_still_trips_at_twenty_percent(
    readers: int,
) -> None:
    now = datetime.now(timezone.utc)
    store = AgentReadFailingStore()
    for index in range(100):
        store.save_agent(fleet_agent(index, now))
    # This process counted the fleet a TTL ago; the re-read the claims make
    # fails, as one transient store error does.
    stale = time.monotonic() - TTL - 1
    assert storm_admission.cluster_gpu_nodes(store, CLUSTER, now=stale) == 100
    store.agent_reads_fail = True
    start = now - timedelta(seconds=30)
    ids = [
        seed(
            store, f"idle-{index}", created_at=start + timedelta(milliseconds=index)
        ).request_id
        for index in range(21)
    ]
    before = storm_admission.node_count_read_failures()

    budget_dispatcher(
        store,
        ScriptedAdapter(REBOOT_PLAN),
        RemediationBudgetPolicy(),
        max_workers=readers,
    ).run_once()

    statuses = [store.get_workflow(item).status for item in ids]
    assert statuses.count(WorkflowStatus.RUNNING) == 19, (
        "100 nodes: the 20th node trips it; the floor of 3 would have held the third"
    )
    breaker = store.get_remediation_storm_breaker(CLUSTER)
    assert breaker is not None, "the claims judged the breaker"
    assert (breaker.trip_threshold, breaker.trip_cluster_nodes) == (20, 100)
    failed = failed_reads_since(before)
    assert failed[LAST_GOOD_FALLBACK] >= 1, failed
    assert failed[FLOOR_FALLBACK] == 0, failed
