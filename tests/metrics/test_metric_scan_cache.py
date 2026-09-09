"""What a Prometheus scrape is allowed to cost.

The closed-loop and attempt-ownership families both need the whole agent table
and the closed-loop family needs the workflow table, so an unshared render reads
the same rows several times on every scrape and grows with the fleet. These tests
pin the sharing, the budget on the workflow read, and the requirement that a
budget that is too small says so instead of quietly reporting a slice.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpu_fault.app import ApplicationContext
from gpu_fault.app.builtin_metric_contributors import (
    closed_loop_metric_lines,
    orchestration_metric_lines,
)
from gpu_fault.app.metric_scan_cache import MetricScanCache, metric_scan_cache
from gpu_fault.models import WorkflowStatus
from tests._builders import (
    attempt_observation,
    build_store,
    container_observation,
    fault_incident,
    workflow_request,
)

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


class CountingStore:
    """A store that only records how often the expensive scans were called."""

    def __init__(
        self, *, agents: int = 0, workflows: int = 0, observations: int = 0
    ) -> None:
        self.agent_calls = 0
        self.workflow_calls = 0
        self.observation_calls = 0
        self.workflow_limits: list[int] = []
        self.observation_limits: list[int | None] = []
        self.observation_orders: list[bool] = []
        self._agents = [
            SimpleNamespace(node_id=f"node-{index}") for index in range(agents)
        ]
        self._workflows = [
            SimpleNamespace(request_id=f"wf-{index}") for index in range(workflows)
        ]
        self._observations = [
            SimpleNamespace(attempt_id=f"attempt-{index}")
            for index in range(observations)
        ]

    def list_agents(self, cluster_id: str | None = None) -> list:
        self.agent_calls += 1
        return list(self._agents)

    def list_workflows(self, statuses=None, *, limit=100, newest_first=False) -> list:
        self.workflow_calls += 1
        self.workflow_limits.append(limit)
        rows = (
            list(reversed(self._workflows)) if newest_first else list(self._workflows)
        )
        return rows[:limit]

    def list_attempt_observation_states(
        self,
        cluster_id: str | None = None,
        *,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list:
        self.observation_calls += 1
        self.observation_limits.append(limit)
        self.observation_orders.append(newest_first)
        rows = (
            list(reversed(self._observations))
            if newest_first
            else list(self._observations)
        )
        return rows if limit is None else rows[:limit]


def test_repeated_reads_inside_one_ttl_cost_one_scan() -> None:
    store = CountingStore(agents=3, workflows=3)
    clock = [0.0]
    cache = MetricScanCache(store, ttl_seconds=10.0, monotonic=lambda: clock[0])

    assert len(cache.agents()) == 3
    cache.agents()
    cache.workflows()
    cache.workflows()

    # One workflow scan is two bounded reads (open statuses, newest terminal;
    # G-2), shared for the TTL like the agent read.
    assert store.agent_calls == 1
    assert store.workflow_calls == 2

    clock[0] = 11.0
    cache.agents()
    cache.workflows()

    assert store.agent_calls == 2
    assert store.workflow_calls == 4


def test_zero_ttl_disables_sharing_so_a_caller_can_demand_fresh_rows() -> None:
    store = CountingStore(agents=1, workflows=1)
    cache = MetricScanCache(store, ttl_seconds=0.0, monotonic=lambda: 0.0)

    cache.agents()
    cache.agents()

    assert store.agent_calls == 2


def test_workflow_scan_is_bounded_and_reports_its_truncation() -> None:
    store = CountingStore(workflows=5)
    cache = MetricScanCache(store, workflow_limit=3, ttl_seconds=0.0)

    scan = cache.workflows()

    # One row past the budget is requested on each of the two slices (open
    # statuses, then newest terminal; G-2), which is how truncation is observed
    # instead of guessed from a full page. This fake ignores the status filter,
    # so both slices see the same five rows and each is truncated to three.
    assert store.workflow_limits == [4, 4]
    assert len(scan.workflows) == 6
    assert scan.limit == 3
    assert scan.truncated is True

    cache.invalidate()
    within = MetricScanCache(store, workflow_limit=5, ttl_seconds=0.0).workflows()

    assert len(within.workflows) == 10
    assert within.truncated is False


def test_observation_scan_is_shared_bounded_and_newest_first() -> None:
    """The ownership family's table read gets the workflow treatment.

    It used to be the one remaining scrape-path scan with neither a cache nor a
    budget, over a table that retains a week of training attempts by default.
    """

    store = CountingStore(observations=5)
    cache = MetricScanCache(store, observation_limit=3, ttl_seconds=0.0)

    scan = cache.observation_states()

    # One row past the budget, so truncation is observed rather than guessed.
    assert store.observation_limits == [4]
    # Newest first: ownership counts fresh attempts and staleness counts the ones
    # just past the freshness window, and both live at the recent end, so the
    # oldest rows are the right ones to drop.
    assert store.observation_orders == [True]
    assert len(scan.states) == 3
    assert scan.limit == 3
    assert scan.truncated is True


def test_observation_scan_reports_no_truncation_within_budget() -> None:
    store = CountingStore(observations=2)

    scan = MetricScanCache(store, observation_limit=50, ttl_seconds=0.0)
    result = scan.observation_states()

    assert len(result.states) == 2
    assert result.truncated is False


def test_repeated_observation_reads_inside_one_ttl_cost_one_scan() -> None:
    store = CountingStore(observations=3)
    clock = [0.0]
    cache = MetricScanCache(store, ttl_seconds=10.0, monotonic=lambda: clock[0])

    cache.observation_states()
    cache.observation_states()

    assert store.observation_calls == 1

    clock[0] = 11.0
    cache.observation_states()

    assert store.observation_calls == 2


def test_invalidate_forces_the_next_read_to_hit_the_store() -> None:
    store = CountingStore(agents=1)
    cache = MetricScanCache(store, ttl_seconds=600.0, monotonic=lambda: 0.0)

    cache.agents()
    cache.invalidate()
    cache.agents()

    assert store.agent_calls == 2


def test_a_runtime_without_a_cache_still_renders() -> None:
    """Plugin contributors and tests build the runtime themselves.

    Missing sharing must degrade to an uncached read; it must not fail a scrape.
    """

    store = CountingStore(agents=2)
    cache = metric_scan_cache(SimpleNamespace(context=SimpleNamespace(store=store)))

    assert isinstance(cache, MetricScanCache), (
        "a runtime without a shared cache must still get a working accessor"
    )
    assert len(cache.agents()) == 2


def _persist_workflows(store, count: int) -> None:
    for index in range(count):
        incident = fault_incident(
            incident_id=f"incident-{index}",
            event_id=f"event-{index}",
            node_ids=[f"node-{index}"],
        )
        store.save_incident(incident)
        store.save_workflow(
            workflow_request(
                request_id=f"wf-{index}",
                incident_id=incident.incident_id,
                status=WorkflowStatus.SUCCEEDED,
                created_at=NOW,
                updated_at=NOW + timedelta(seconds=index),
            )
        )


def test_status_gauge_stays_exact_when_the_detail_scan_is_truncated() -> None:
    """Truncating the detail read must not corrupt the workflow census.

    The status gauge drives capacity and backlog alerting, so it is a
    server-side aggregate; only the step, duration and milestone families,
    which need whole objects, are bounded.
    """

    store = build_store()
    _persist_workflows(store, 5)
    context = ApplicationContext(store=store)
    runtime = SimpleNamespace(
        context=context,
        metric_scan_cache=MetricScanCache(store, workflow_limit=2, ttl_seconds=0.0),
    )

    lines = closed_loop_metric_lines(runtime)

    assert 'gpu_fault_workflow_total{status="SUCCEEDED"} 5' in lines
    assert "gpu_fault_workflow_scan_limit 2" in lines
    assert "gpu_fault_workflow_scan_size 2" in lines
    assert "gpu_fault_workflow_scan_truncated 1" in lines


def test_untruncated_scan_publishes_a_zero_truncation_gauge() -> None:
    store = build_store()
    _persist_workflows(store, 2)
    context = ApplicationContext(store=store)
    runtime = SimpleNamespace(
        context=context,
        metric_scan_cache=MetricScanCache(store, workflow_limit=50, ttl_seconds=0.0),
    )

    lines = closed_loop_metric_lines(runtime)

    assert "gpu_fault_workflow_scan_truncated 0" in lines
    assert "gpu_fault_workflow_scan_size 2" in lines


def _persist_observations(store, count: int) -> None:
    # Timestamped against the wall clock rather than the fixed ``NOW`` the
    # workflow helper uses: the ownership family classifies an observation as
    # fresh or stale against the real clock, and a week-old fixture would land
    # every row in the staleness gauge instead of the ownership one.
    recent = datetime.now(timezone.utc)
    for index in range(count):
        store.save_attempt_observation(
            attempt_observation(
                job_id="job-a",
                attempt_id=f"attempt-{index}",
                observed_at=recent - timedelta(seconds=count - index),
                containers=[
                    container_observation(
                        pod_uid=f"pod-{index}",
                        pod_name=f"trainer-{index}",
                        rank=0,
                        node_id=f"node-{index}",
                    )
                ],
            )
        )


def test_ownership_family_reads_the_observation_table_once_and_bounded() -> None:
    """The ownership gauges come from a cached, bounded, newest-first slice.

    Previously ``ownership_metric_snapshot`` reached into the store itself and
    read every retained attempt observation on every scrape, on the thread
    serving ``/metrics``.
    """

    store = build_store()
    _persist_observations(store, 5)
    context = ApplicationContext(store=store)
    runtime = SimpleNamespace(
        context=context,
        metric_scan_cache=MetricScanCache(store, observation_limit=2, ttl_seconds=0.0),
    )

    lines = orchestration_metric_lines(runtime)

    assert "gpu_fault_attempt_observation_scan_limit 2" in lines
    assert "gpu_fault_attempt_observation_scan_size 2" in lines
    assert "gpu_fault_attempt_observation_scan_truncated 1" in lines
    # The newest two observations are attempt-3 and attempt-4, on node-3/node-4,
    # so only those nodes carry an ownership series.
    assert (
        'gpu_fault_ambiguous_attempt_ownership_current{cluster_id="cluster-a",'
        'gpu_node="node-4"} 1'
    ) in lines
    assert not any('gpu_node="node-0"' in line for line in lines), (
        "the oldest observation fell outside the scan budget, so its node must "
        f"carry no ownership series: {lines}"
    )


def test_ownership_family_reports_no_truncation_within_budget() -> None:
    store = build_store()
    _persist_observations(store, 2)
    runtime = SimpleNamespace(
        context=ApplicationContext(store=store),
        metric_scan_cache=MetricScanCache(store, observation_limit=50, ttl_seconds=0.0),
    )

    lines = orchestration_metric_lines(runtime)

    assert "gpu_fault_attempt_observation_scan_size 2" in lines
    assert "gpu_fault_attempt_observation_scan_truncated 0" in lines


def test_an_unbounded_caller_still_sees_the_whole_table_in_storage_order() -> None:
    """Only the scrape path is bounded.

    Six other callers read this method for their own filtering and have always
    received every row in storage order. Imposing the metrics slice or ordering on
    them would be a silent behaviour change, so the default stays as it was.
    """

    store = build_store()
    _persist_observations(store, 4)

    states = store.list_attempt_observation_states()

    assert [item.observation.attempt_id for item in states] == [
        "attempt-0",
        "attempt-1",
        "attempt-2",
        "attempt-3",
    ]
    assert [
        item.observation.attempt_id
        for item in store.list_attempt_observation_states(limit=2, newest_first=True)
    ] == ["attempt-3", "attempt-2"]


def test_store_status_counts_cover_every_declared_status() -> None:
    """A missing status must read as zero, not as an absent series."""

    store = build_store()
    _persist_workflows(store, 1)

    counts = store.workflow_status_counts()

    assert set(counts) == set(WorkflowStatus)
    assert counts[WorkflowStatus.SUCCEEDED] == 1
    assert counts[WorkflowStatus.BLOCKED] == 0
