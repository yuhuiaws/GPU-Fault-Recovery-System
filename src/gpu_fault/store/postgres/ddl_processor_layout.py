"""Schema v23 stage: the detached hot-table layout (detailed design v2 §4.4.2).

A ``ddl*.py`` module like the others so the migration registry checksums it
and ``declared_index_names`` / ``declared_index_statements`` scrape it; kept
apart from ``ddl.py`` only because that file is at its size ratchet.

What the live catalog said after one perf day on one release: every one of
the queue's 248,350 updates was a non-HOT update (``n_tup_hot_upd`` 0) because
``status`` sat in every partial index predicate, ``lease_expires_at`` in the
lane index and ``updated_at`` in the completed index, so each request left
three heap versions and about eighteen index entries that the retention sweep
deleted row by row; the request and response bodies made 65 % of a perf
round's rows TOAST. This stage changes the data model, not the maintenance:

* ``gpu_fault_processor_queue`` gains the two columns the hot statements read
  out of the payload today (``path``, ``scope_keys``) so no index has to
  reference ``payload`` any more. A BEFORE INSERT trigger fills them from the
  payload when a writer of the previous release leaves them NULL, which is
  what keeps the roll online: old and new Pods write the same rows.
* The index set loses ``status``, the lease columns and ``updated_at``
  entirely (claim, path claim, lane, cluster, and a GIN over ``scope_keys``
  that covers only rows carrying scope keys), so a claim, a completion and a
  coalescing rewrite are HOT updates. The eight status-predicated indexes are
  dropped by the v23 migration callback.
* ``gpu_fault_processor_payloads`` holds request bodies (``BYTEA``, inline up
  to the page size) and ``gpu_fault_processor_replays`` -- RANGE-partitioned
  on ``completed_at`` in five-minute buckets with a DEFAULT catch-all -- holds
  completed requests, so the queue heap carries only in-flight rows and the
  retention sweep drops whole partitions instead of deleting rows.
* ``gpu_fault_gpu_metric_latest_clusters`` is the LIST-by-cluster twin of
  ``gpu_fault_gpu_metric_latest``; a cluster's partition is created at
  registration and dropped at removal or perf teardown.
* ``gpu_fault_hot_layout_mode`` is the one-row fact that decides which layout
  writers use: ``inline`` (the v22 layout, seeded on an upgraded database) or
  ``detached`` (seeded on a fresh one, switched by ``gpu-fault-store-migrate
  --set-hot-layout detached`` after the control-plane roll). Readers always
  read both layouts.

Partitioned parents accept no storage parameters, so each partition is
created with the parent's intended reloptions (``hot_layout.py``).
"""

from __future__ import annotations

from typing import Any

from gpu_fault.store.postgres.ddl_helpers import (
    _add_column_if_missing,
    _declare_index,
    _ensure_trigger,
)

QUEUE_TABLE = "gpu_fault_processor_queue"
PAYLOADS_TABLE = "gpu_fault_processor_payloads"
REPLAYS_TABLE = "gpu_fault_processor_replays"
REPLAYS_DEFAULT_PARTITION = "gpu_fault_processor_replays_default"
METRIC_LATEST_CLUSTERS_TABLE = "gpu_fault_gpu_metric_latest_clusters"
METRIC_LATEST_CLUSTERS_DEFAULT_PARTITION = (
    "gpu_fault_gpu_metric_latest_clusters_default"
)
LAYOUT_MODE_TABLE = "gpu_fault_hot_layout_mode"
LAYOUT_INLINE = "inline"
LAYOUT_DETACHED = "detached"

#: The v22 queue indexes whose keys or predicates blocked HOT updates. Dropped
#: once by the v23 migration callback; never declared again.
RETIRED_QUEUE_INDEXES: tuple[str, ...] = (
    "gpu_fault_processor_queue_priority_claim",
    "gpu_fault_processor_queue_path_priority_claim",
    "gpu_fault_processor_queue_lane",
    "gpu_fault_processor_queue_cluster",
    "gpu_fault_processor_queue_incomplete_cluster",
    "gpu_fault_processor_queue_available",
    "gpu_fault_processor_queue_completed",
    "gpu_fault_processor_queue_correlation_scopes",
)

#: Reloptions every partition of the metric-latest twin is created with (the
#: parent cannot carry them): the same values v19 gave the flat table.
METRIC_LATEST_PARTITION_OPTIONS = (
    "fillfactor=70, autovacuum_vacuum_scale_factor=0, "
    "autovacuum_vacuum_threshold=50, autovacuum_analyze_scale_factor=0, "
    "autovacuum_analyze_threshold=50, autovacuum_vacuum_cost_delay=0"
)


def create_processor_queue_columns(cursor: Any) -> None:
    """The two hot predicate columns and the trigger that fills them for a
    previous-release writer (expand phase of the v23 expand/contract)."""

    _add_column_if_missing(cursor, QUEUE_TABLE, "path", "TEXT")
    _add_column_if_missing(cursor, QUEUE_TABLE, "scope_keys", "TEXT[]")
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION
        gpu_fault_processor_queue_fill_columns()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.path IS NULL THEN
                NEW.path := NEW.payload->>'path';
            END IF;
            IF NEW.scope_keys IS NULL THEN
                NEW.scope_keys := ARRAY(
                    SELECT jsonb_array_elements_text(
                        coalesce(NEW.payload->'correlation_scope_keys', '[]'::jsonb)
                    )
                );
            END IF;
            RETURN NEW;
        END
        $$
        """
    )
    # The WHEN clause is evaluated in C before any plpgsql call: a row written
    # by this release carries both columns and never enters the function, so
    # the trigger costs the admission INSERT nothing after the roll. Spelled
    # the way ``pg_get_triggerdef`` deparses it so a second ensure compares
    # equal and never recreates the trigger (item J).
    _ensure_trigger(
        cursor,
        "gpu_fault_processor_queue_fill_columns_trigger",
        QUEUE_TABLE,
        """
        CREATE TRIGGER
        gpu_fault_processor_queue_fill_columns_trigger
        BEFORE INSERT
        ON gpu_fault_processor_queue
        FOR EACH ROW
        WHEN (((new.path IS NULL) OR (new.scope_keys IS NULL)))
        EXECUTE FUNCTION
        gpu_fault_processor_queue_fill_columns()
        """,
    )


def create_processor_queue_indexes(cursor: Any) -> None:
    """The v23 queue index set: no ``status``, no lease column, no ``updated_at``
    anywhere, so claims, completions and coalescing rewrites stay HOT.

    The claim window orders by (priority, created_at, request_id) with no
    predicate: in the detached layout only in-flight rows live in the queue,
    and during the inline window a completed row is a residual filter for the
    minutes it survives. The per-path pools walk the same order under a path
    equality; lane and cluster probes fetch the handful of rows of one key;
    the GIN covers only rows with scope keys and runs without a pending list
    (``fastupdate = off``): a row carries one or two short keys, so inserting
    them into the entry tree directly costs little, while 190 concurrent
    admission flushes appending to one pending list all queued on its
    metapage lock. Readers spell ``cardinality(scope_keys) > 0`` so the
    planner can prove the partial predicate.
    """

    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_claim_window
        ON gpu_fault_processor_queue (
            priority,
            created_at,
            request_id
        )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_path_claim_window
        ON gpu_fault_processor_queue (
            path,
            priority,
            created_at,
            request_id
        )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_lane_rows
        ON gpu_fault_processor_queue (ordering_key)
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_cluster_rows
        ON gpu_fault_processor_queue (cluster_id, created_at)
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_scope_keys
        ON gpu_fault_processor_queue
        USING GIN (scope_keys)
        WITH (fastupdate = off)
        WHERE cardinality(scope_keys) > 0
        """,
    )


def create_detached_layout_tables(cursor: Any) -> None:
    """The payloads table, the replay partition parent, the metric-latest
    cluster-partitioned twin and the layout mode row."""

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_processor_payloads (
            request_id TEXT PRIMARY KEY,
            body BYTEA NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_processor_replays (
            request_id TEXT NOT NULL,
            cluster_id TEXT,
            completed_at TIMESTAMPTZ NOT NULL,
            envelope JSONB NOT NULL,
            body BYTEA,
            response_status INTEGER,
            response_content_type TEXT,
            response_body BYTEA,
            PRIMARY KEY (request_id, completed_at)
        ) PARTITION BY RANGE (completed_at)
        """
    )
    cursor.execute(f"SELECT to_regclass('{REPLAYS_DEFAULT_PARTITION}')")
    if cursor.fetchone()[0] is None:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {REPLAYS_DEFAULT_PARTITION}
            PARTITION OF gpu_fault_processor_replays DEFAULT
            """
        )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_gpu_metric_latest_clusters (
            key TEXT NOT NULL,
            cluster_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL,
            PRIMARY KEY (cluster_id, key)
        ) PARTITION BY LIST (cluster_id)
        """
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_gpu_metric_latest_clusters_node
        ON gpu_fault_gpu_metric_latest_clusters (node_id, key)
        """,
    )
    cursor.execute(f"SELECT to_regclass('{METRIC_LATEST_CLUSTERS_DEFAULT_PARTITION}')")
    if cursor.fetchone()[0] is None:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {METRIC_LATEST_CLUSTERS_DEFAULT_PARTITION}
            PARTITION OF gpu_fault_gpu_metric_latest_clusters DEFAULT
            WITH ({METRIC_LATEST_PARTITION_OPTIONS})
            """
        )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_hot_layout_mode (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE
                CHECK (singleton),
            mode TEXT NOT NULL
                CHECK (mode IN ('inline', 'detached')),
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    # Read before write (item J): an upgraded database starts in the v22
    # layout; a fresh bootstrap moves the row to ``detached`` afterwards.
    cursor.execute("SELECT 1 FROM gpu_fault_hot_layout_mode WHERE singleton")
    if cursor.fetchone() is None:
        cursor.execute(
            """
            INSERT INTO gpu_fault_hot_layout_mode(singleton, mode, updated_at)
            VALUES (TRUE, 'inline', now())
            ON CONFLICT(singleton) DO NOTHING
            """
        )


def create_processor_layout_stage(cursor: Any) -> None:
    """Schema v23 (module docstring): columns and trigger first, then the
    tables, then the index set that relies on the new columns."""

    create_processor_queue_columns(cursor)
    create_detached_layout_tables(cursor)
    create_processor_queue_indexes(cursor)
