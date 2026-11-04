"""The latest-metric reads touch one node's rows, whatever the statistics say.

The state every burst-1c perf run starts in, rebuilt from the run records: the
burst cluster has no partition of its own (it is published through the
durable registry), so its 32,000 rows go to the twin's DEFAULT partition,
which the previous run's cleanup emptied of them and autovacuum then analysed
-- the partition's statistics know only the production clusters. Until the
next autovacuum pass, the planner estimated the burst cluster at one row, and
both reads walked the primary key's cluster range: every row the cluster had
written, per batch (``store/postgres/metric_latest_reads.py``).

The state is built here exactly (autovacuum off for the partitions, explicit
ANALYZE, ``pg_class``/``pg_stats`` checked before anything is measured), with
the index geometry the store-compaction job leaves when it rebuilds one
flagged index: a compact primary key beside a sparse node index -- the
geometry in which the previous listing statement, a cost tie between the two
indexes, took the primary key's range. The store's own calls are then
measured by the index tuples they read (``pg_stat_user_indexes``), and the
statements' plans by EXPLAIN under custom and generic plans alike; counts are
recorded as test properties.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.gpu_metrics import GpuMetricLatest, GpuMetricSample, GpuMetricSource
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres import hot_layout
from gpu_fault.store.postgres.metric_latest_reads import (
    node_rows_query,
    previous_rows_query,
)
from gpu_fault.store.shared.primitives import state_key
from tests.store._postgres_processor_claim_support import (
    _ensure_postgres_schema,
    _truncate,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
TWIN = "gpu_fault_gpu_metric_latest_clusters"
DEFAULT = "gpu_fault_gpu_metric_latest_clusters_default"
BURST = "perf-burst"
PRODUCTION = "production-default"  # rows in the DEFAULT partition
PARTITIONED = "production-own"  # a cluster with a partition of its own
GPUS = tuple(f"GPU-00000000-0000-4000-8000-00000000000{index}" for index in range(8))
METRICS = (
    "sm_clock_mhz",
    "memory_clock_mhz",
    "clock_throttle_reasons",
    "memory_temp_c",
)
ROWS_PER_NODE = len(GPUS) * len(METRICS)
BURST_NODES = 200
PROBE_NODE = 7


def _node(index: int) -> str:
    return f"burst-node-{index:04d}"


def _connect():
    import psycopg

    assert POSTGRES_URL is not None
    return psycopg.connect(POSTGRES_URL, autocommit=True)


def _latest(cluster: str, node: str, gpu: int, metric: str, at: datetime):
    return GpuMetricLatest(
        cluster_id=cluster,
        node_id=node,
        observed_at=at,
        source=GpuMetricSource.DCGM_EXPORTER,
        sample=GpuMetricSample(
            metric_name=f"DCGM_FI_DEV_{metric.upper()}",
            canonical_name=metric,
            value=345.0,
            gpu_index=str(gpu),
            gpu_uuid=GPUS[gpu],
            pci_bdf=f"00000000:{0x59 + gpu:02X}:00.0",
        ),
    )


def _batch(cluster: str, node: str, at: datetime):
    """One GPU_METRICS batch's store items, the perf template's shape."""

    return [
        ((cluster, node, GPUS[gpu], metric), _latest(cluster, node, gpu, metric, at))
        for metric in METRICS
        for gpu in range(len(GPUS))
    ]


def _copy(connection: Any, cluster: str, nodes: list[str], at: datetime) -> None:
    with connection.cursor() as cursor:
        with cursor.copy(
            f"COPY {TWIN} (key, cluster_id, node_id, observed_at, payload) FROM STDIN"
        ) as copy:
            for node in nodes:
                for key, latest in _batch(cluster, node, at):
                    copy.write_row(
                        (state_key(key), cluster, node, at, latest.model_dump_json())
                    )


def _statistics(connection: Any) -> dict[str, Any]:
    reltuples, modified = connection.execute(
        "SELECT c.reltuples, s.n_mod_since_analyze FROM pg_class c "
        "JOIN pg_stat_user_tables s ON s.relid = c.oid WHERE c.relname = %s",
        (DEFAULT,),
    ).fetchone()
    common = connection.execute(
        "SELECT most_common_vals::text FROM pg_stats "
        "WHERE tablename = %s AND attname = 'cluster_id'",
        (DEFAULT,),
    ).fetchone()
    return {
        "reltuples": float(reltuples),
        "modified_since_analyze": int(modified),
        "cluster_mcv": common[0] if common else None,
    }


@pytest.fixture(scope="module")
def burst_state():
    """The DEFAULT partition with the burst's rows written after its last
    ANALYZE; torn down to an empty twin."""

    _truncate()
    _ensure_postgres_schema()
    at = datetime.now(timezone.utc) - timedelta(minutes=1)
    with _connect() as connection:
        with connection.cursor() as cursor:
            hot_layout.write_layout_mode(cursor, "detached")
        with connection.transaction(), connection.cursor() as cursor:
            hot_layout.ensure_cluster_partition(cursor, PARTITIONED)
        for (name,) in connection.execute(
            "SELECT inhrelid::regclass::text FROM pg_inherits "
            "WHERE inhparent = %s::regclass",
            (TWIN,),
        ).fetchall():
            connection.execute(f"ALTER TABLE {name} SET (autovacuum_enabled = false)")
        # An earlier working set's dead space, then the compaction job's
        # rebuild of the one flagged index: a compact primary key next to a
        # sparse node index.
        old = [f"old-node-{index:04d}" for index in range(1250)]
        _copy(connection, "earlier-perf-cluster", old, at - timedelta(days=1))
        # Production rows written after it keep the heap at its high-water
        # mark, as the live store's continuously updated rows do.
        _copy(connection, PRODUCTION, [f"{PRODUCTION}-{n}" for n in range(10)], at)
        connection.execute(
            f"DELETE FROM {DEFAULT} WHERE cluster_id = 'earlier-perf-cluster'"
        )
        connection.execute(f"VACUUM {DEFAULT}")
        connection.execute(f"REINDEX INDEX {DEFAULT}_pkey")
        _copy(connection, PARTITIONED, [f"{PARTITIONED}-{n}" for n in range(4)], at)
        connection.execute(f"ANALYZE {TWIN}")
        _copy(connection, BURST, [_node(n) for n in range(BURST_NODES)], at)
    # The seeding backend's row counters reach pg_stat_user_tables when it
    # exits; read them from a new connection.
    with _connect() as connection:
        for _ in range(50):
            state = _statistics(connection)
            if state["modified_since_analyze"] >= BURST_NODES * ROWS_PER_NODE:
                break
            connection.execute("SELECT pg_sleep(0.1), pg_stat_clear_snapshot()")
    try:
        yield state
    finally:
        with _connect() as connection:
            with connection.transaction(), connection.cursor() as cursor:
                hot_layout.drop_cluster_partition(cursor, PARTITIONED)
            connection.execute(f"ALTER TABLE {DEFAULT} RESET (autovacuum_enabled)")
            with connection.cursor() as cursor:
                hot_layout.write_layout_mode(cursor, "inline")
        _truncate()


def test_the_burst_state_is_the_one_every_perf_run_starts_in(burst_state) -> None:
    """The precondition the measurements below rely on."""

    assert BURST not in (burst_state["cluster_mcv"] or ""), burst_state
    assert PRODUCTION in (burst_state["cluster_mcv"] or ""), burst_state
    assert burst_state["reltuples"] < BURST_NODES * ROWS_PER_NODE, burst_state
    assert burst_state["modified_since_analyze"] >= BURST_NODES * ROWS_PER_NODE, (
        f"the burst rows arrived after the last ANALYZE: {burst_state}"
    )


def _index_tuples_read(connection: Any) -> dict[str, int]:
    connection.execute("SELECT pg_stat_clear_snapshot()")
    return {
        name.removeprefix(DEFAULT + "_"): int(read)
        for name, read in connection.execute(
            "SELECT indexrelname, idx_tup_read FROM pg_stat_user_indexes "
            "WHERE relname = %s",
            (DEFAULT,),
        ).fetchall()
    }


def test_a_burst_batch_reads_only_its_own_rows(burst_state, request) -> None:
    """The store's two calls for one batch of a node the partition's
    statistics have never seen read that node's rows and no others: the
    primary key once per key (read-back, then the upsert's conflict check),
    the node index once per row (listing)."""

    with _connect() as connection:
        before = _index_tuples_read(connection)
    store = PostgresStore(POSTGRES_URL, initialize_schema=False, pool_max_size=1)
    try:
        node = _node(PROBE_NODE)
        at = datetime.now(timezone.utc)
        previous = store.observe_gpu_metrics(_batch(BURST, node, at))
        listed = store.list_gpu_metrics_latest(BURST, node)
    finally:
        # Index counters reach pg_stat_user_indexes when the backend exits.
        store.close()
    assert all(isinstance(item, GpuMetricLatest) for item in previous), previous
    assert [item.observed_at for item in listed] == [at] * ROWS_PER_NODE
    with _connect() as connection:
        for _ in range(50):
            after = _index_tuples_read(connection)
            if after != before:
                break
            connection.execute("SELECT pg_sleep(0.1)")
    read = {name: after[name] - before.get(name, 0) for name in after}
    request.node.user_properties.append(("index_tuples_read", json.dumps(read)))
    burst_rows = BURST_NODES * ROWS_PER_NODE
    assert read["pkey"] <= 4 * ROWS_PER_NODE, (
        f"the primary key read {read['pkey']} entries for one batch "
        f"(the cluster has {burst_rows} rows): {read}"
    )
    assert read["node_id_key_idx"] <= 2 * ROWS_PER_NODE, (
        f"the node index read {read['node_id_key_idx']} entries: {read}"
    )


def _touched(plan: Any) -> int:
    """Rows every scan of a twin partition produced or threw away."""

    def nodes(value: Any):
        if isinstance(value, dict):
            yield value
            for nested in value.values():
                yield from nodes(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from nodes(nested)

    return sum(
        (
            int(node.get("Actual Rows", 0))
            + int(node.get("Rows Removed by Filter", 0))
            + int(node.get("Rows Removed by Index Recheck", 0))
        )
        * int(node.get("Actual Loops", 1))
        for node in nodes(plan)
        if str(node.get("Relation Name", "")).startswith(TWIN)
        and "Scan" in node.get("Node Type", "")
    )


def _positional(sql: str) -> str:
    parts = sql.split("%s")
    return "".join(
        part + (f"${index}" if index < len(parts) else "")
        for index, part in enumerate(parts, start=1)
    )


def _explain(connection: Any, query: tuple[str, tuple[Any, ...]], mode: str) -> dict:
    """EXPLAIN ANALYZE of a store statement as a prepared statement under
    ``plan_cache_mode`` = ``mode``, in a rolled-back transaction."""

    from psycopg import sql

    statement, params = query
    plan: dict = {}
    with connection.transaction(force_rollback=True):
        connection.execute(f"SET LOCAL plan_cache_mode = {mode}")
        connection.execute(f"PREPARE b247_probe AS {_positional(statement)}")
        arguments = sql.SQL(", ").join(sql.Literal(value) for value in params)
        plan = connection.execute(
            sql.SQL(
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) EXECUTE b247_probe({})"
            ).format(arguments)
        ).fetchone()[0][0]
    return plan


@pytest.mark.parametrize("mode", ["force_custom_plan", "force_generic_plan"])
@pytest.mark.parametrize("read", ["previous_rows", "node_rows"])
def test_the_read_touches_one_node_s_rows_in_custom_and_generic_plans(
    burst_state, read: str, mode: str, request
) -> None:
    node = _node(PROBE_NODE)
    keys = [
        state_key(key) for key, _ in _batch(BURST, node, datetime.now(timezone.utc))
    ]
    query = (
        previous_rows_query(TWIN, keys, BURST)
        if read == "previous_rows"
        else node_rows_query(TWIN, BURST, node)
    )
    with _connect() as connection:
        plan = _explain(connection, query, mode)
    touched = _touched(plan)
    request.node.user_properties.extend(
        [
            ("touched_rows", touched),
            ("shared_hit_blocks", int(plan["Plan"].get("Shared Hit Blocks", 0))),
            ("execution_ms", round(float(plan.get("Execution Time", 0)), 3)),
        ]
    )
    assert int(plan["Plan"]["Actual Rows"]) == ROWS_PER_NODE, plan["Plan"]
    assert touched <= ROWS_PER_NODE, (
        f"{read} touched {touched} rows of a cluster with "
        f"{BURST_NODES * ROWS_PER_NODE}: {json.dumps(plan)[:2000]}"
    )


def _same_key(latest: GpuMetricLatest, key: tuple[str, str, str, str]) -> bool:
    sample = latest.sample
    return (
        latest.cluster_id,
        latest.node_id,
        sample.gpu_uuid,
        sample.canonical_name,
    ) == key


def test_reads_keep_clusters_apart_in_the_default_partition(burst_state) -> None:
    """Two clusters in the DEFAULT partition share a node name: each read
    answers for its own cluster, the listing in key order, and a batch that
    names a key twice reads it once."""

    store = PostgresStore(POSTGRES_URL, initialize_schema=False, pool_max_size=1)
    other = "other-burst"
    node = _node(PROBE_NODE)
    at = datetime.now(timezone.utc)
    try:
        store.observe_gpu_metrics(_batch(other, node, at))
        listed = store.list_gpu_metrics_latest(other, node)
        assert {item.cluster_id for item in listed} == {other}, listed
        assert len(listed) == ROWS_PER_NODE, listed
        keys = [
            state_key(
                (other, item.node_id, item.sample.gpu_uuid, item.sample.canonical_name)
            )
            for item in listed
        ]
        assert keys == sorted(keys), "the listing is in key order"
        burst = store.list_gpu_metrics_latest(BURST, node)
        assert {item.cluster_id for item in burst} == {BURST}, burst
        key, latest = _batch(other, node, at + timedelta(seconds=1))[0]
        previous = next(item for item in listed if _same_key(item, key))
        assert store.observe_gpu_metrics([(key, latest), (key, latest)]) == [
            previous,
            False,
        ]
    finally:
        store.close()
        with _connect() as connection:
            connection.execute(f"DELETE FROM {TWIN} WHERE cluster_id = %s", (other,))
