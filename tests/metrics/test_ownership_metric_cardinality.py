"""The node-ownership families cost what the alerts read, not one line per Agent.

``orchestration_metric_lines`` rendered ``gpu_fault_ambiguous_attempt_ownership_current``
and ``gpu_fault_stale_attempt_observations`` for every evaluated node: a zero
on an idle node and ``1`` on a busy one, two lines per registered Agent in
every publication of every control-plane process (25,600 lines at 12,800
Agents). The per-node families now carry only the nodes that meet the alert
condition (more than one fresh attempt; at least one stale observation), and
four per-cluster gauges -- always present, zero included -- carry the coverage
and the counts the other lines used to imply.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.app import ApplicationContext
from gpu_fault.app.builtin_metric_contributors import orchestration_metric_lines
from gpu_fault.app.metric_aggregation import STRATEGIES, Strategy
from gpu_fault.app.metric_scan_cache import MetricScanCache
from gpu_fault.app.process_metrics import aggregate, parse_lines
from tests._builders import attempt_observation, build_store, container_observation

ROOT = Path(__file__).resolve().parents[2]
CURRENT = "gpu_fault_ambiguous_attempt_ownership_current"
STALE = "gpu_fault_stale_attempt_observations"
EVALUATED = "gpu_fault_attempt_observation_scan_nodes"
OWNED = "gpu_fault_attempt_observation_scan_owned_nodes"
AMBIGUOUS = "gpu_fault_ambiguous_attempt_ownership_nodes"
STALE_NODES = "gpu_fault_stale_attempt_observation_nodes"
PER_NODE = (CURRENT, STALE)
PER_CLUSTER = (EVALUATED, OWNED, AMBIGUOUS, STALE_NODES)
# Past the evidence service's 120-s ownership freshness window.
STALE_AGE_SECONDS = 600.0


class FleetStore:
    """The two reads ``MetricScanCache`` makes for the ownership families."""

    def __init__(self, agents=(), observations=()) -> None:
        self.agents = list(agents)
        # Oldest first, the way the store keeps them.
        self.observations = list(observations)

    def list_agents(self):
        return list(self.agents)

    def list_attempt_observation_states(self, *, limit=None, newest_first=False):
        rows = [SimpleNamespace(observation=item) for item in self.observations]
        if newest_first:
            rows.reverse()
        return rows if limit is None else rows[:limit]


def fleet(cluster_id: str, nodes: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(cluster_id=cluster_id, node_id=f"node-{index}")
        for index in range(nodes)
    ]


def claim(job_id: str, *nodes: str, cluster_id: str = "cluster-a", age: float = 5.0):
    """One RUNNING attempt of ``job_id`` with a container on each node.

    The evidence service classifies against the wall clock, so ``age`` is
    relative to now: a few seconds is fresh, ``STALE_AGE_SECONDS`` is stale.
    """

    return attempt_observation(
        job_id=job_id,
        attempt_id=f"{job_id}-attempt",
        observed_at=datetime.now(timezone.utc) - timedelta(seconds=age),
        cluster_id=cluster_id,
        containers=[
            container_observation(
                pod_uid=f"{job_id}-{node}",
                pod_name=f"{job_id}-{node}",
                rank=rank,
                node_id=node,
            )
            for rank, node in enumerate(nodes)
        ],
    )


def render(store: FleetStore, *, observation_limit: int = 20_000) -> list[str]:
    runtime = SimpleNamespace(
        context=ApplicationContext(store=build_store()),
        metric_scan_cache=MetricScanCache(
            store, observation_limit=observation_limit, ttl_seconds=0.0
        ),
    )
    return orchestration_metric_lines(runtime)


def per_node(lines: list[str], family: str) -> dict[tuple[str, str], int]:
    nodes: dict[tuple[str, str], int] = {}
    for sample in parse_lines(lines).samples[family]:
        labels = dict(sample.labels)
        assert set(labels) == {"cluster_id", "gpu_node"}, sample
        nodes[(labels["cluster_id"], labels["gpu_node"])] = int(sample.value)
    return nodes


def per_cluster(lines: list[str]) -> dict[str, dict[str, int]]:
    parsed = parse_lines(lines)
    clusters: dict[str, dict[str, int]] = {}
    for family in PER_CLUSTER:
        for sample in parsed.samples[family]:
            labels = dict(sample.labels)
            assert set(labels) == {"cluster_id"}, (
                f"{family} is labelled by cluster only"
            )
            clusters.setdefault(labels["cluster_id"], {})[family] = int(sample.value)
    return clusters


def counts(*, evaluated: int, owned: int = 0, ambiguous: int = 0, stale: int = 0):
    return {
        EVALUATED: evaluated,
        OWNED: owned,
        AMBIGUOUS: ambiguous,
        STALE_NODES: stale,
    }


def without(lines: list[str], family: str) -> list[str]:
    """``lines`` as if ``family`` had not been rendered at all."""

    prefixes = (f"# HELP {family} ", f"# TYPE {family} ", f"{family}{{", f"{family} ")
    return [line for line in lines if not line.startswith(prefixes)]


def control_plane_keep_regex() -> str:
    documents = yaml.safe_load_all(
        (ROOT / "deploy/observability/adot-control-plane.yaml").read_text(
            encoding="utf-8"
        )
    )
    config_map = next(
        item for item in documents if item and item["kind"] == "ConfigMap"
    )
    collector = yaml.safe_load(config_map["data"]["collector.yaml"])
    job = next(
        item
        for item in collector["receivers"]["prometheus"]["config"]["scrape_configs"]
        if item["job_name"] == "gpu-fault-control-plane"
    )
    return next(
        item["regex"]
        for item in job["metric_relabel_configs"]
        if item["action"] == "keep" and item["source_labels"] == ["__name__"]
    )


def test_an_idle_fleet_renders_no_per_node_series_and_zero_counts_per_cluster() -> None:
    lines = render(FleetStore(fleet("cluster-a", 256)))
    parsed = parse_lines(lines)

    for family in PER_NODE:
        assert parsed.samples[family] == [], f"{family} rendered idle nodes"
        help_line, type_line = parsed.families[family]
        assert type_line == f"# TYPE {family} gauge", (
            f"{family} must stay declared without samples"
        )
        assert help_line is not None and "rendered only for a node" in help_line, (
            f"{family}'s HELP does not say when a node has a series: {help_line}"
        )
    assert per_cluster(lines) == {"cluster-a": counts(evaluated=256)}


@pytest.mark.parametrize("busy", [False, True], ids=["idle", "one-attempt-per-node"])
def test_the_render_does_not_grow_with_the_registered_fleet(busy: bool) -> None:
    def lines_for(nodes: int) -> list[str]:
        agents = fleet("cluster-a", nodes)
        observations = (
            [claim(f"job-{index}", agent.node_id) for index, agent in enumerate(agents)]
            if busy
            else []
        )
        return render(FleetStore(agents, observations))

    small, large = lines_for(1), lines_for(512)

    assert len(large) == len(small), (
        "the ownership families grew with the fleet: "
        f"{len(small)} lines for one Agent, {len(large)} for 512"
    )
    assert per_cluster(large) == {
        "cluster-a": counts(evaluated=512, owned=512 if busy else 0)
    }


def test_single_owners_render_no_per_node_series_and_are_counted_as_owned() -> None:
    agents = fleet("cluster-a", 8)
    observations = [
        claim("job-0", "node-0"),
        claim("job-1", "node-1"),
        # One attempt across three nodes owns each of them once.
        claim("job-wide", "node-2", "node-3", "node-4"),
    ]

    lines = render(FleetStore(agents, observations))

    assert per_node(lines, CURRENT) == {}
    assert per_node(lines, STALE) == {}
    assert per_cluster(lines) == {"cluster-a": counts(evaluated=8, owned=5)}


def test_a_double_owner_is_the_one_per_node_series_and_is_counted() -> None:
    agents = fleet("cluster-a", 4)
    observations = [
        claim("job-a", "node-1"),
        claim("job-b", "node-1"),
        claim("job-c", "node-2"),
    ]

    lines = render(FleetStore(agents, observations))

    assert per_node(lines, CURRENT) == {("cluster-a", "node-1"): 2}
    assert per_cluster(lines) == {
        "cluster-a": counts(evaluated=4, owned=2, ambiguous=1)
    }


def test_a_stale_observation_renders_its_node_and_is_counted() -> None:
    agents = fleet("cluster-a", 4)
    observations = [
        claim("job-gone", "node-3", age=STALE_AGE_SECONDS),
        # node-0 keeps a stale observation beside its fresh single owner.
        claim("job-old", "node-0", age=STALE_AGE_SECONDS),
        claim("job-live", "node-0"),
    ]

    lines = render(FleetStore(agents, observations))

    assert per_node(lines, STALE) == {
        ("cluster-a", "node-0"): 1,
        ("cluster-a", "node-3"): 1,
    }
    assert per_node(lines, CURRENT) == {}, "a stale observation is not an owner"
    assert per_cluster(lines) == {"cluster-a": counts(evaluated=4, owned=1, stale=2)}


def test_a_truncated_scan_counts_only_its_newest_slice() -> None:
    agents = fleet("cluster-a", 4)
    # The double owner on node-0 is the two oldest rows.
    observations = [
        claim("job-a", "node-0", age=30),
        claim("job-b", "node-0", age=20),
        claim("job-c", "node-1", age=10),
        claim("job-d", "node-2", age=5),
    ]

    truncated = render(FleetStore(agents, observations), observation_limit=2)
    complete = render(FleetStore(agents, observations), observation_limit=4)

    assert "gpu_fault_attempt_observation_scan_truncated 1" in truncated
    assert per_node(truncated, CURRENT) == {}, (
        "the double owner fell outside the scan budget, so no node may claim it"
    )
    # Registered Agents are evaluated whatever the budget; ownership is the slice's.
    assert per_cluster(truncated) == {"cluster-a": counts(evaluated=4, owned=2)}
    assert "gpu_fault_attempt_observation_scan_truncated 0" in complete
    assert per_node(complete, CURRENT) == {("cluster-a", "node-0"): 2}
    assert per_cluster(complete) == {
        "cluster-a": counts(evaluated=4, owned=3, ambiguous=1)
    }


def test_two_clusters_are_counted_apart() -> None:
    # node ids repeat across clusters; the key is (cluster, node).
    agents = [*fleet("cluster-a", 3), *fleet("cluster-b", 2)]
    observations = [
        claim("job-x", "node-0", cluster_id="cluster-b"),
        claim("job-y", "node-0", cluster_id="cluster-b"),
        claim("job-z", "node-1", cluster_id="cluster-b", age=STALE_AGE_SECONDS),
        # A cluster known only from an observation is evaluated too.
        claim("job-w", "node-9", cluster_id="cluster-c"),
    ]

    lines = render(FleetStore(agents, observations))

    assert per_node(lines, CURRENT) == {("cluster-b", "node-0"): 2}
    assert per_node(lines, STALE) == {("cluster-b", "node-1"): 1}
    assert per_cluster(lines) == {
        "cluster-a": counts(evaluated=3),
        "cluster-b": counts(evaluated=2, owned=1, ambiguous=1, stale=1),
        "cluster-c": counts(evaluated=1, owned=1),
    }


def test_an_empty_local_family_beats_a_siblings_older_series_in_the_pod_merge() -> None:
    """Every process of a Pod publishes its render and the answering one merges
    them. The ownership families are ANY: the first process that declares a
    family answers for the Pod. A sibling whose scan cache still holds a double
    owner that this process no longer sees must not resurrect it, which is why
    a family without samples keeps its HELP/TYPE lines."""

    local_lines = render(FleetStore(fleet("cluster-a", 2)))
    sibling = parse_lines(
        [
            f"# HELP {CURRENT} an older render of another process",
            f"# TYPE {CURRENT} gauge",
            f'{CURRENT}{{cluster_id="cluster-a",gpu_node="node-1"}} 2',
            f"# TYPE {STALE} gauge",
            f'{STALE}{{cluster_id="cluster-a",gpu_node="node-0"}} 1',
            f"# TYPE {AMBIGUOUS} gauge",
            f'{AMBIGUOUS}{{cluster_id="cluster-a"}} 1',
        ]
    )
    sibling.slot = 1
    local = parse_lines(local_lines)
    local.slot = 0

    merged = aggregate(local, [sibling])

    assert [
        line for line in merged if line.startswith((f"{CURRENT}{{", f"{STALE}{{"))
    ] == [], "a sibling's older per-node series survived the merge"
    assert f'{AMBIGUOUS}{{cluster_id="cluster-a"}} 0' in merged
    assert {STRATEGIES[family] for family in (*PER_NODE, *PER_CLUSTER)} == {
        Strategy.ANY
    }, "a MAX or SUM merge would keep the sibling's older series"
    # Undeclared, the family would fall through to the sibling's publication.
    undeclared = parse_lines(without(local_lines, CURRENT))
    undeclared.slot = 0
    assert f'{CURRENT}{{cluster_id="cluster-a",gpu_node="node-1"}} 2' in aggregate(
        undeclared, [sibling]
    ), "the contrast case no longer falls through: the test proves nothing"


def test_every_orchestration_family_reaches_amp_through_the_keep_filter() -> None:
    """ADOT's keep list is the only way into AMP, and the stale entry there is an
    exact name: a per-cluster family missing from it is dropped silently."""

    keep = control_plane_keep_regex()
    lines = render(
        FleetStore(
            fleet("cluster-a", 2),
            [
                claim("job-a", "node-0"),
                claim("job-b", "node-0"),
                claim("job-c", "node-1", age=STALE_AGE_SECONDS),
            ],
        )
    )
    parsed = parse_lines(lines)
    rendered = set(parsed.families) | {
        sample.name for samples in parsed.samples.values() for sample in samples
    }

    assert set(PER_NODE) | set(PER_CLUSTER) <= set(parsed.families)
    dropped = sorted(name for name in rendered if not re.fullmatch(keep, name))
    assert dropped == [], f"the ADOT keep filter drops: {dropped}"
