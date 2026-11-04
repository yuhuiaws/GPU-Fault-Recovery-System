"""The latest-metric table's two hot reads, shaped so their plans do not depend
on the table's statistics.

Every GPU metric batch reads its own rows back before the upsert
(``observe_gpu_metrics``) and then lists its node's rows for the composite
rules (``list_gpu_metrics_latest``, also the kernel-XID GPU lookup). On the
partitioned twin (schema v23) both reads name one cluster, and its statistics
can predate that cluster: a cluster published through the durable regional
registry has no partition of its own, so its rows land in the DEFAULT
partition, whose last ANALYZE may have run before the cluster wrote anything
(every perf burst starts so: the harness deletes the previous run's rows and
autovacuum then analyses the partition without them). The MCV list then
covers the whole partition with other clusters and the planner estimates the
cluster at one row.

With that estimate both reads walked the primary key's ``cluster_id`` range
and filtered the rest on the heap -- every row the cluster had written so
far, per call, until autovacuum analysed the partition again. On PostgreSQL
16 with the live layout rebuilt locally (a 56k-page DEFAULT partition, 32,000
rows of a 1,000-node burst, statistics from before it) the read-back took
20-25 ms and 33,002 shared buffers per batch instead of 0.8 ms and 128; the
node listing is a cost tie between that range and the node index, decided by
index geometry, and took 21 ms and 33,002 buffers when the range won. A burst of
1,000 such batches is quadratic, and it lasted until the next autovacuum pass
analysed the partition (in every slow burst run the perf samplers show the
read storm ending with an autovacuum worker).

The read-back below leaves the planner no choice that statistics decide, and
the listing no longer depends on the cluster's own entry in them:

* the read-back is one ``(cluster_id, key)`` primary-key probe per key, a
  LATERAL subquery whose ``LIMIT 1`` keeps it from being flattened into a join
  (flattened, the same one-row estimate hashes the whole range: 54 ms); a
  probe that names every column of a unique index costs one descent with or
  without statistics, in custom and generic plans alike, nothing else the
  planner can build is cheaper, and the literal ``cluster_id`` still prunes to
  one partition;
* the listing passes the cluster as an uncorrelated sub-select, so the
  planner prices the range at an average cluster's share of the partition
  (``1 / n_distinct``) instead of the stale one row, and the node index, which
  reads only the node's rows, wins; partitions are pruned at run time instead
  of at planning.

What stays: a partition that has never been analysed has no column
statistics at all, and the listing there may still combine both indexes
(index pages of the cluster's range, no heap) until its first ANALYZE.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.store.postgres.hot_layout import METRIC_LATEST_TABLE


def previous_rows_query(
    table: str, keys: list[str], cluster_id: str
) -> tuple[str, tuple[Any, ...]]:
    """The batch's current rows, one primary-key probe per distinct key.

    The flat v22 table's key is unique on its own; the partitioned twin's
    primary key is ``(cluster_id, key)``.
    """

    wanted = list(dict.fromkeys(keys))
    params: tuple[Any, ...] = (wanted,)
    scope = ""
    if table != METRIC_LATEST_TABLE:
        params, scope = (wanted, cluster_id), "cluster_id=%s AND "
    return (
        f"""
        SELECT latest.key, latest.payload
        FROM unnest(%s::text[]) AS wanted(key)
        CROSS JOIN LATERAL (
            SELECT key, payload
            FROM {table}
            WHERE {scope}key=wanted.key
            LIMIT 1
        ) AS latest
        """,
        params,
    )


def node_rows_query(
    table: str, cluster_id: str, node_id: str
) -> tuple[str, tuple[Any, ...]]:
    """One node's rows in key order, the cluster priced as an average one."""

    return (
        f"""
        SELECT key, payload
        FROM {table}
        WHERE cluster_id=(SELECT %s::text)
          AND node_id=%s
        ORDER BY key
        """,
        (cluster_id, node_id),
    )
