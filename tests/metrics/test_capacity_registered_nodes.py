"""The registered GPU fleet next to the declared capacity topology (b229).

``spec.capacity.largestClusterNodeCount`` / ``managedNodeCount`` size four
runtime limits, and node groups grow without the administrator. The two
``gpu_fault_capacity_registered_*`` families put the real fleet beside the
declaration: one series per cluster and the site total, tallied from the
worker tier's shared agent scan, the same Agents the storm breaker counts.
These tests pin the count, its cost (one tally per scan-cache TTL, no read of
its own), the b227 rule (a failed read is never rendered or kept as a zero),
the Pod-coherent merge and the keep filter.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.app.capacity_metrics import (
    REGISTERED_CLUSTER_NODES,
    REGISTERED_MANAGED_NODES,
    managed_gpu_nodes,
    registered_gpu_nodes,
    registered_node_metric_lines,
)
from gpu_fault.app.metric_aggregation import STRATEGIES, Strategy
from gpu_fault.app.metric_contributors import MetricContributorRegistry
from gpu_fault.app.metric_scan_cache import MetricScanCache
from gpu_fault.app.metrics import METRIC_CONTRIBUTORS
from gpu_fault.app.process_metrics import aggregate, parse_lines
from gpu_fault.execution.storm_admission import cluster_gpu_nodes
from gpu_fault.fleet import AgentLifecycleState

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = (REGISTERED_CLUSTER_NODES, REGISTERED_MANAGED_NODES)
ACTIVE, DRAINING, REVOKED = (
    AgentLifecycleState.ACTIVE,
    AgentLifecycleState.DRAINING,
    AgentLifecycleState.REVOKED,
)


def agents(cluster_id: str, *states: AgentLifecycleState) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            cluster_id=cluster_id,
            node_id=f"{cluster_id}-node-{index}",
            lifecycle_state=state,
        )
        for index, state in enumerate(states)
    ]


class AgentStore:
    """The one read the tally makes; ``fail`` makes the next reads raise."""

    def __init__(self, records=()) -> None:
        self.records = list(records)
        self.reads = 0
        self.fail = False

    def list_agents(self, cluster_id=None):
        self.reads += 1
        if self.fail:
            raise ConnectionError("store unavailable")
        return [
            item
            for item in self.records
            if cluster_id is None or item.cluster_id == cluster_id
        ]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def registry(**synthetic: bool) -> SimpleNamespace:
    """A regional registry runtime whose snapshot marks the given clusters."""

    registrations = {
        cluster_id: SimpleNamespace(synthetic=flag)
        for cluster_id, flag in synthetic.items()
    }
    return SimpleNamespace(
        snapshot=lambda: SimpleNamespace(registrations=registrations)
    )


def runtime(store: AgentStore, *, clock: Clock | None = None, registry_runtime=None):
    cache = MetricScanCache(store, ttl_seconds=60.0, monotonic=clock or Clock())
    return SimpleNamespace(
        context=SimpleNamespace(
            store=store, regional_registry_runtime=registry_runtime
        ),
        metric_scan_cache=cache,
    )


def values(lines: list[str]) -> dict[str, dict[tuple, str]]:
    parsed = parse_lines(lines)
    return {
        family: {
            sample.labels: sample.value for sample in parsed.samples.get(family, [])
        }
        for family in FAMILIES
        if family in parsed.families
    }


def test_registered_nodes_are_the_active_and_draining_agents_per_cluster() -> None:
    fleet = [
        *agents("hp-a", ACTIVE, ACTIVE, DRAINING, REVOKED),
        *agents("eks-b", ACTIVE, REVOKED, REVOKED),
        *agents("gone-c", REVOKED),
    ]

    assert registered_gpu_nodes(fleet) == {"hp-a": 3, "eks-b": 1}, (
        "a retired (REVOKED) Agent is not a registered GPU node"
    )


@pytest.mark.parametrize("state", list(AgentLifecycleState))
def test_the_count_is_the_remediation_storm_breakers_count(
    state: AgentLifecycleState,
) -> None:
    """Both judge "the cluster's GPU nodes"; one definition, whatever the state."""

    store = AgentStore([*agents("hp-a", ACTIVE, state, state), *agents("eks-b", state)])

    for cluster_id in ("hp-a", "eks-b"):
        assert registered_gpu_nodes(store.list_agents()).get(
            cluster_id, 0
        ) == cluster_gpu_nodes(store, cluster_id, now=0.0), (
            f"the capacity count and the storm breaker disagree on {state} Agents"
        )


def test_the_contributor_renders_one_series_per_cluster_and_the_site_total() -> None:
    store = AgentStore(
        [*agents("hp-a", ACTIVE, ACTIVE, DRAINING), *agents("eks-b", ACTIVE, REVOKED)]
    )

    rendered = values(registered_node_metric_lines(runtime(store)))

    assert rendered[REGISTERED_CLUSTER_NODES] == {
        (("cluster_id", "eks-b"),): "1",
        (("cluster_id", "hp-a"),): "3",
    }, "one series per cluster, never one per node"
    assert rendered[REGISTERED_MANAGED_NODES] == {(): "4"}, rendered


def test_an_empty_fleet_declares_the_cluster_family_and_totals_zero() -> None:
    lines = registered_node_metric_lines(runtime(AgentStore()))
    parsed = parse_lines(lines)

    assert set(FAMILIES) <= set(parsed.families), (
        "an empty fleet must still declare both families (an empty local "
        "family wins the Pod merge)"
    )
    assert parsed.samples.get(REGISTERED_CLUSTER_NODES, []) == [], lines
    assert values(lines)[REGISTERED_MANAGED_NODES] == {(): "0"}, lines


def test_synthetic_capacity_drill_clusters_are_left_out() -> None:
    store = AgentStore(
        [*agents("hp-a", ACTIVE, ACTIVE), *agents("perf-cap-000", *[ACTIVE] * 256)]
    )
    drills = registry(**{"hp-a": False, "perf-cap-000": True})

    rendered = values(
        registered_node_metric_lines(runtime(store, registry_runtime=drills))
    )

    assert rendered[REGISTERED_CLUSTER_NODES] == {(("cluster_id", "hp-a"),): "2"}, (
        "a drill's synthetic Agents are not managed GPU nodes"
    )
    assert rendered[REGISTERED_MANAGED_NODES] == {(): "2"}, rendered
    assert managed_gpu_nodes({"a": 1, "b": 2}, ["b"]) == {"a": 1}, "filter only"


def test_a_registry_that_has_not_loaded_renders_nothing_rather_than_drills() -> None:
    def unavailable():
        raise RuntimeError("regional registry snapshot is unavailable")

    store = AgentStore(agents("perf-cap-000", ACTIVE))

    with pytest.raises(RuntimeError, match="unavailable"):
        registered_node_metric_lines(
            runtime(store, registry_runtime=SimpleNamespace(snapshot=unavailable))
        )


def test_the_tally_costs_one_shared_agent_read_per_scan_ttl() -> None:
    clock = Clock()
    store = AgentStore(agents("hp-a", ACTIVE))
    process = runtime(store, clock=clock)

    registered_node_metric_lines(process)
    store.records.extend(agents("hp-b", ACTIVE, ACTIVE))
    clock.now += 59.0
    cached = values(registered_node_metric_lines(process))
    clock.now += 2.0
    fresh = values(registered_node_metric_lines(process))

    assert store.reads == 2, f"one agent read per 60-s TTL, saw {store.reads}"
    assert cached[REGISTERED_MANAGED_NODES] == {(): "1"}, cached
    assert fresh[REGISTERED_MANAGED_NODES] == {(): "3"}, fresh


def test_the_tally_reuses_the_agent_scan_the_other_families_already_read() -> None:
    store = AgentStore(agents("hp-a", ACTIVE, ACTIVE))
    process = runtime(store)

    process.metric_scan_cache.agents()
    registered_node_metric_lines(process)

    assert store.reads == 1, "the contributor must not read the agent table again"


def test_a_failed_read_is_never_rendered_or_kept_as_a_zero() -> None:
    """b227: a failed count read is never kept as the count. The contributor
    raises, its families are skipped and counted, and the next render reads
    again instead of serving a cached zero."""

    store = AgentStore(agents("hp-a", ACTIVE, ACTIVE, ACTIVE))
    process = runtime(store)
    contributors = MetricContributorRegistry()
    contributors.register(
        "capacity-registered-nodes", registered_node_metric_lines, fleet_level=True
    )

    store.fail = True
    failed = parse_lines(contributors.render(process))
    store.fail = False
    recovered = values(contributors.render(process))

    assert not set(FAMILIES) & set(failed.families), (
        "a failed read rendered the registered families"
    )
    assert contributors.error_counts() == {"capacity-registered-nodes": 1}, (
        "the failed render must be counted in gpu_fault_metrics_contributor_errors_total"
    )
    assert recovered[REGISTERED_MANAGED_NODES] == {(): "3"}, (
        "the read after the failure must answer the real count, not a kept zero"
    )
    assert store.reads == 2, store.reads


def test_the_pod_merge_answers_with_a_siblings_count_when_the_local_read_failed() -> (
    None
):
    sibling = parse_lines(
        registered_node_metric_lines(
            runtime(AgentStore(agents("hp-a", ACTIVE, ACTIVE)))
        )
    )
    sibling.slot = 1
    failed_local = parse_lines(
        [
            "# TYPE gpu_fault_capacity_largest_cluster_node_count gauge",
            "gpu_fault_capacity_largest_cluster_node_count 512",
        ]
    )
    failed_local.slot = 0

    merged = aggregate(failed_local, [sibling])

    assert (
        'gpu_fault_capacity_registered_cluster_node_count{cluster_id="hp-a"} 2'
        in merged
    ), merged
    assert "gpu_fault_capacity_registered_managed_node_count 2" in merged, merged


def test_an_empty_local_fleet_beats_a_siblings_older_series() -> None:
    sibling = parse_lines(
        registered_node_metric_lines(runtime(AgentStore(agents("hp-a", ACTIVE))))
    )
    sibling.slot = 1
    local = parse_lines(registered_node_metric_lines(runtime(AgentStore())))
    local.slot = 0

    merged = aggregate(local, [sibling])

    assert [
        line for line in merged if line.startswith(REGISTERED_CLUSTER_NODES + "{")
    ] == [], "a sibling's older per-cluster series survived the merge"
    assert "gpu_fault_capacity_registered_managed_node_count 0" in merged, merged


def test_both_families_merge_any_and_render_on_the_worker_tier_only() -> None:
    assert {STRATEGIES[family] for family in FAMILIES} == {Strategy.ANY}, (
        "the counts come from one shared table: a SUM would multiply them by "
        "the processes of a Pod"
    )
    assert "capacity-registered-nodes" in METRIC_CONTRIBUTORS.fleet_level_names, (
        "fleet-level: ingress and spool-worker must not repeat the agent read"
    )


def control_plane_keep_regex() -> str:
    documents = yaml.safe_load_all(
        (ROOT / "deploy/observability/adot-control-plane.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = next(item for item in documents if item and item["kind"] == "ConfigMap")
    jobs = yaml.safe_load(config["data"]["collector.yaml"])["receivers"]["prometheus"][
        "config"
    ]["scrape_configs"]
    job = next(item for item in jobs if item["job_name"] == "gpu-fault-control-plane")
    return next(
        item["regex"]
        for item in job["metric_relabel_configs"]
        if item["action"] == "keep" and item["source_labels"] == ["__name__"]
    )


def test_both_families_reach_amp_through_the_keep_filter() -> None:
    keep = control_plane_keep_regex()

    dropped = [family for family in FAMILIES if not re.fullmatch(keep, family)]

    assert dropped == [], f"the ADOT keep filter drops: {dropped}"
