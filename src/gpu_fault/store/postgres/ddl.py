from __future__ import annotations

from pathlib import Path
from typing import Any

from gpu_fault.store.postgres.ddl_helpers import (
    _CREATE_INDEX,
    _add_column_if_missing,
    _declare_index,
    _drop_column_if_present,
    _drop_index_if_present,
    _enable_trigger_if_disabled,
    _ensure_trigger,
    _recorded_schema_version,
)
from gpu_fault.store.postgres.ddl_processor_retry import (
    upgrade_processor_retry_schedule,
)
from gpu_fault.store.postgres.ddl_spool import _create_telemetry_spool


def create_postgres_schema(cursor) -> None:
    """Idempotent DDL, serialized by the caller advisory lock.

    Idempotent means lock-free when nothing changes (store review 2026-09-07,
    item J). Every statement here runs in one transaction with a 60 s
    ``lock_timeout``, and ``--ensure-schema`` runs on every deploy against a
    live database. Postgres takes the lock before it checks whether the
    statement is a no-op: ``CREATE INDEX ... IF NOT EXISTS`` opens the table in
    ShareLock (blocks every writer) and only then finds the index present;
    ``ALTER TABLE ... ADD/DROP COLUMN IF [NOT] EXISTS`` and ``DROP TRIGGER`` +
    ``CREATE TRIGGER`` take AccessExclusiveLock; ``ALTER TABLE ... ENABLE
    TRIGGER`` takes ShareRowExclusiveLock. Each lock is held until commit, so
    ~50 such statements against a busy ``gpu_fault_objects`` stall the whole
    control plane for the DDL duration, or fail on ``lock_timeout``. The
    helpers below (``_declare_index``, ``_add_column_if_missing``,
    ``_ensure_trigger``, ...) read the catalog first and issue the DDL only
    when the object is actually missing or different; the two legacy
    partition sweeps are gated on the recorded schema version. A routine
    deploy therefore holds only AccessShareLock on the catalogs and the
    schema-version table.
    """
    _create_base_tables(cursor)
    _create_base_indexes(cursor)
    _create_domain_indexes_one(cursor)
    _create_dispatcher_indexes(cursor)
    _create_domain_indexes_two(cursor)
    _create_domain_tables_and_indexes(cursor)
    _create_processor_tables(cursor)
    _create_processor_counter_functions(cursor)
    _create_processor_triggers(cursor)
    _create_priority_counter_function(cursor)
    _create_priority_counter_triggers(cursor)
    _seed_processor_counters(cursor)
    _create_processor_indexes(cursor)
    _create_telemetry_spool(cursor)


def _create_base_tables(cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_schema_version (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE
                CHECK (singleton),
            version INTEGER NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            checksum TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL,
            CHECK (version > 0),
            CHECK (length(checksum) = 64)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_objects (
            kind TEXT NOT NULL,
            key TEXT NOT NULL,
            payload JSONB NOT NULL,
            PRIMARY KEY (kind, key)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_links (
            kind TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (kind, key)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
        gpu_fault_gpu_metric_latest (
            key TEXT PRIMARY KEY,
            cluster_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL
        )
        """
    )


def _create_base_indexes(cursor: Any) -> None:
    # Split from ``_create_base_tables`` so each stage stays readable; the
    # order (tables first, then their indexes) is unchanged.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_gpu_metric_latest_node
        ON gpu_fault_gpu_metric_latest (
            cluster_id, node_id, key
        )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_marker_incident
        ON gpu_fault_objects (
            (payload->>'incident_id'),
            (payload->>'observed_at'),
            key
        )
        WHERE kind='marker'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_gpu_metric_latest_observed
        ON gpu_fault_gpu_metric_latest (observed_at)
        """,
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
        gpu_fault_gpu_metrics_batches (
            key TEXT PRIMARY KEY,
            cluster_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL
        )
        """
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_gpu_metrics_batches_created
        ON gpu_fault_gpu_metrics_batches (created_at, key)
        """,
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
        gpu_fault_attempt_observations (
            key TEXT PRIMARY KEY,
            cluster_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL
        )
        """
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_attempt_observations_cluster
        ON gpu_fault_attempt_observations (
            cluster_id, observed_at, key
        )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_attempt_observations_observed
        ON gpu_fault_attempt_observations (observed_at, key)
        """,
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
        gpu_fault_training_progress (
            key TEXT PRIMARY KEY,
            cluster_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            rank BIGINT NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL
        )
        """
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_training_progress_attempt
        ON gpu_fault_training_progress (
            cluster_id, attempt_id, rank, key
        )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_training_progress_observed
        ON gpu_fault_training_progress (observed_at, key)
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_raw_evidence_lookup
        ON gpu_fault_objects (
            (payload->>'cluster_id'),
            (payload->>'node_id'),
            (payload->>'observed_at') DESC
        )
        WHERE kind='raw_evidence'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_raw_evidence_expiry
        ON gpu_fault_objects (
            (payload->>'expires_at'),
            key
        )
        WHERE kind='raw_evidence'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_xid_correlation_lookup
        ON gpu_fault_objects (
            (payload->>'cluster_id'),
            (payload->>'node_id'),
            (payload->>'observed_at')
        )
        WHERE kind='xid_correlation_event'
        """,
    )


def _create_domain_indexes_one(cursor) -> None:
    # Every executor polls claim every 2 seconds per cluster, so
    # this is the hottest read in regional mode. Without a
    # partial index the claim scans all remote commands ever
    # written, including terminal ones, and the cost grows with
    # cluster lifetime rather than with the open backlog.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_remote_command_claim
        ON gpu_fault_objects (
            (payload->>'cluster_id'),
            (payload->'step'->>'execution_owner'),
            (payload->>'created_at'),
            key
        )
        WHERE kind='remote_command'
          AND payload->>'status' IN (
              'PENDING', 'WAITING', 'LEASED'
          )
        """,
    )
    # Every workflow reconcile lists that workflow's commands in every
    # status (``list_remote_commands(workflow_request_ids=...)``), inside
    # the transaction that holds the workflow's advisory and row locks;
    # the only workflow_request_id index used to be partial on the open
    # statuses, so that read was a full remote_command scan under lock
    # (store review 2026-09-07, item H1). This all-status index carries
    # the query's ORDER BY, and the timeout cancel path (open rows of one
    # workflow) walks the same prefix with a status filter, so the old
    # open-only partial ``gpu_fault_remote_command_workflow`` is dropped
    # by the v12 migration.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_remote_command_workflow_all
        ON gpu_fault_objects (
            (payload->>'workflow_request_id'),
            (payload->>'created_at'),
            key
        )
        WHERE kind='remote_command'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_remote_command_terminal
        ON gpu_fault_objects (
            (payload->>'updated_at')
        )
        WHERE kind='remote_command'
          AND payload->>'status' IN ('SUCCEEDED', 'FAILED')
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_active_workflow_scope
        ON gpu_fault_objects (
            (payload->>'status'),
            (payload->>'incident_id'),
            (payload->>'updated_at') DESC
        )
        WHERE kind='workflow'
          AND payload->>'status' IN (
              'PENDING', 'RUNNING', 'SAFETY_PENDING'
          )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_failed_workflow_updated
        ON gpu_fault_objects (
            (payload->>'updated_at') DESC,
            key
        )
        WHERE kind='workflow'
          AND payload->>'status'='FAILED'
        """,
    )


def _create_dispatcher_indexes(cursor: Any) -> None:
    # Second half of ``_create_domain_indexes_one``: the dispatcher, join-key
    # and /metrics count indexes of ``gpu_fault_objects``.
    # Dispatcher hot queries (F-A9). Each partial index carries exactly the
    # predicate its query uses, and the queries spell the status list as
    # literals so the planner can prove the predicate (P0-73C). Text order on
    # ``payload->>'updated_at'`` is deliberate: the stored value is
    # ``isoformat()`` and a ``::timestamptz`` cast would bypass the index.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_executable_workflow_order
        ON gpu_fault_objects (
            (payload->>'updated_at'),
            key
        )
        WHERE kind='workflow'
          AND payload->>'status' IN (
              'PENDING', 'RUNNING', 'SAFETY_PENDING'
          )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_unhandled_failed_workflow_updated
        ON gpu_fault_objects (
            (payload->>'updated_at'),
            key
        )
        WHERE kind='workflow'
          AND payload->>'status'='FAILED'
          AND (
              payload->>'failure_handled_at' IS NULL
              OR payload->>'failure_handled_at'=''
          )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_blocked_workflow_updated
        ON gpu_fault_objects (
            (payload->>'updated_at'),
            key
        )
        WHERE kind='workflow'
          AND payload->>'status'='BLOCKED'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_preempting_successor
        ON gpu_fault_objects (
            (payload->>'predecessor_workflow_id'),
            (payload->>'created_at'),
            key
        )
        WHERE kind='workflow'
          AND payload->>'preempt_predecessor'='true'
          AND payload->>'status' IN (
              'PENDING', 'SAFETY_PENDING'
          )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_incident_scope
        ON gpu_fault_objects (
            (payload->>'cluster_id'),
            (payload->>'job_id'),
            key
        )
        WHERE kind='incident'
        """,
    )
    # incident -> workflow join key in every status. The job-recovery
    # reader (``list_job_recovery_workflow_incidents``, once per attempt
    # observation) joins ``w.payload->>'incident_id' = i.key`` and the
    # only incident_id index was the active-status partial above, so the
    # terminal history was scanned per observation (store review
    # 2026-09-07, item H2).
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_workflow_incident
        ON gpu_fault_objects (
            (payload->>'incident_id'),
            key
        )
        WHERE kind='workflow'
        """,
    )
    # The paged dispatch scan (F-A2a/c) orders by the eligibility key
    # ``GREATEST(created_at, not_before)`` with a row-value cursor; the code
    # and its tests assumed this index while the DDL never declared it, so
    # production sorted the executable set on every tick (store review
    # 2026-09-07, item H3).
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_executable_workflow_dispatch_order
        ON gpu_fault_objects (
            (GREATEST(payload->>'created_at', payload->>'not_before')),
            key
        )
        WHERE kind='workflow'
          AND payload->>'status' IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')
        """,
    )
    # /metrics aggregates ``count(*) GROUP BY payload->>'status'`` over every
    # workflow, incident and decision on every scrape, and control-record
    # retention is off, so each scrape read and detoasted the whole kind
    # (store review 2026-09-07, item G). One extra small index entry per
    # write buys one index range scan per status value (``count(*) WHERE
    # payload->>'status' = $1``: an Index Cond on the expression, no payload
    # evaluated, no TOAST fetch). Not an index-only scan: Postgres ignores
    # expression columns when deciding whether an index can return a tuple
    # (indxpath.c check_index_only), so the aggregate must ask per status
    # rather than GROUP BY the expression. A trigger-maintained counter
    # table was rejected: it would put every workflow write on one hot
    # counter row, the contention shape the processor counters had to shard
    # their way out of.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_workflow_status_count
        ON gpu_fault_objects ((payload->>'status'))
        WHERE kind='workflow'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_incident_state_count
        ON gpu_fault_objects ((payload->>'state'))
        WHERE kind='incident'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_decision_status_count
        ON gpu_fault_objects ((payload->>'status'))
        WHERE kind='decision'
        """,
    )


def _create_domain_indexes_two(cursor) -> None:
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_incident_nodes
        ON gpu_fault_objects
        USING GIN ((payload->'node_ids'))
        WHERE kind='incident'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_attempt_observation_cluster
        ON gpu_fault_objects (
            (payload->'observation'->>'cluster_id'),
            (payload->'observation'->>'observed_at') DESC,
            key
        )
        WHERE kind='attempt_observation'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_gpu_metric_latest_scope
        ON gpu_fault_objects (
            (payload->>'cluster_id'),
            (payload->>'node_id'),
            key
        )
        WHERE kind='gpu_metric_latest'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_gpu_finding_state_scope
        ON gpu_fault_objects (
            (payload->'finding'->>'cluster_id'),
            (payload->'finding'->>'node_id'),
            key
        )
        WHERE kind='gpu_finding_state'
          AND payload->'finding' IS NOT NULL
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_gpu_finding_history_scope
        ON gpu_fault_objects (
            (payload->>'cluster_id'),
            (payload->>'node_id'),
            key
        )
        WHERE kind='gpu_finding_history'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_gpu_finding_history_observed
        ON gpu_fault_objects (
            (payload->>'observed_at'),
            key
        )
        WHERE kind='gpu_finding_history'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_xid_correlation_due
        ON gpu_fault_objects (
            (payload->>'deadline'),
            (payload->>'lease_expires_at'),
            key
        )
        WHERE kind='xid_correlation'
          AND payload->>'status'='PENDING'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_agent_cluster
        ON gpu_fault_objects (
            (payload->>'cluster_id'),
            (payload->>'node_id'),
            key
        )
        WHERE kind='agent'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_fleet_deployment_created
        ON gpu_fault_objects (
            (payload->>'created_at'),
            key
        )
        WHERE kind='fleet_deployment'
        """,
    )


def _create_domain_tables_and_indexes(cursor) -> None:
    # Every agent heartbeat asks for its cluster's open deployments.
    # The index above is ordered by created_at across all clusters
    # and all statuses, so that lookup had to walk every deployment
    # ever recorded and filter; this one matches the query exactly
    # and only carries the open rows.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_fleet_deployment_active_scope
        ON gpu_fault_objects (
            (payload->>'cluster_id'),
            (payload->>'created_at'),
            key
        )
        WHERE kind='fleet_deployment'
          AND payload->>'status' NOT IN ('SUCCEEDED', 'FAILED')
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_collector_status_scope
        ON gpu_fault_objects (
            (payload->>'cluster_id'),
            (payload->>'node_id'),
            (payload->>'collector'),
            key
        )
        WHERE kind='collector_status'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_telemetry_metric_scope
        ON gpu_fault_objects (
            (payload->>'cluster_id'),
            (payload->>'node_id'),
            (payload->>'device'),
            (payload->>'name'),
            key
        )
        WHERE kind='telemetry_metric_latest'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_training_progress_scope
        ON gpu_fault_objects (
            (payload->'heartbeat'->>'cluster_id'),
            (payload->'heartbeat'->>'attempt_id'),
            ((payload->'heartbeat'->>'rank')::bigint),
            key
        )
        WHERE kind='training_progress'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_remote_command_status_scope
        ON gpu_fault_objects (
            (payload->>'status'),
            (payload->>'cluster_id'),
            (payload->>'created_at'),
            key
        )
        WHERE kind='remote_command'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_notification_delivery_due
        ON gpu_fault_objects (
            (payload->>'status'),
            (payload->>'available_at'),
            (payload->>'lease_expires_at'),
            key
        )
        WHERE kind='notification_delivery'
          AND payload->>'status' IN (
              'PENDING', 'RETRY', 'LEASED'
          )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_notification_created
        ON gpu_fault_objects (
            ((payload->>'priority')::integer),
            (payload->>'created_at'),
            key
        )
        WHERE kind='notification'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_notification_result_status
        ON gpu_fault_objects (
            (payload->>'status'),
            key
        )
        WHERE kind='notification_result'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_hyperpod_identity_cluster
        ON gpu_fault_objects (
            (payload->>'cluster_name'),
            (payload->>'node_logical_id'),
            key
        )
        WHERE kind='hyperpod_node_identity'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_barrier_created
        ON gpu_fault_objects (
            (payload->>'created_at'),
            key
        )
        WHERE kind='barrier'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_active_marker_action
        ON gpu_fault_objects (
            (payload->>'recommended_action'),
            (payload->>'observed_at') DESC,
            key
        )
        WHERE kind='marker'
          AND payload->>'active'='true'
          AND payload->>'trusted'='true'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_active_marker_nodes
        ON gpu_fault_objects
        USING GIN ((payload->'scope'->'node_ids'))
        WHERE kind='marker'
          AND payload->>'active'='true'
          AND payload->>'trusted'='true'
        """,
    )


def _create_processor_tables(cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_processor_queue (
            request_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            cluster_id TEXT,
            correlation_key TEXT,
            ordering_key TEXT NOT NULL,
            priority INTEGER NOT NULL,
            lease_owner TEXT,
            leader_epoch BIGINT,
            lease_token TEXT,
            lease_expires_at TIMESTAMPTZ,
            response_status INTEGER,
            response_content_type TEXT,
            response_body_base64 TEXT,
            not_before TIMESTAMPTZ,
            retry_count INTEGER NOT NULL DEFAULT 0
                CHECK (retry_count >= 0),
            lane_policy TEXT NOT NULL DEFAULT 'STRICT'
                CHECK (lane_policy IN ('STRICT', 'REORDERABLE')),
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL
        )
        """
    )
    _add_column_if_missing(
        cursor, "gpu_fault_processor_queue", "response_status", "INTEGER"
    )
    _add_column_if_missing(
        cursor, "gpu_fault_processor_queue", "response_content_type", "TEXT"
    )
    _add_column_if_missing(
        cursor, "gpu_fault_processor_queue", "response_body_base64", "TEXT"
    )
    upgrade_processor_retry_schedule(cursor)
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION
        gpu_fault_processor_queue_notify_pending()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.status = 'PENDING' THEN
                PERFORM pg_notify(
                    'gpu_fault_processor_queue',
                    json_build_object(
                        'priority', NEW.priority,
                        'path', NEW.payload->>'path',
                        'request_id', NEW.request_id
                    )::text
                );
            END IF;
            RETURN NULL;
        END
        $$
        """
    )
    _ensure_trigger(
        cursor,
        "gpu_fault_processor_queue_notify_pending_trigger",
        "gpu_fault_processor_queue",
        """
        CREATE TRIGGER
        gpu_fault_processor_queue_notify_pending_trigger
        AFTER INSERT OR UPDATE OF status
        ON gpu_fault_processor_queue
        FOR EACH ROW
        EXECUTE FUNCTION
        gpu_fault_processor_queue_notify_pending()
        """,
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_processor_lanes (
            ordering_key TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            epoch BIGINT NOT NULL,
            lease_token TEXT NOT NULL,
            lease_expires_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
        gpu_fault_processor_queue_counts (
            cluster_id TEXT PRIMARY KEY,
            incomplete_count BIGINT NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL,
            CHECK (incomplete_count >= 0)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
        gpu_fault_processor_counter_mode (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE
                CHECK (singleton),
            mode TEXT NOT NULL
                CHECK (mode IN ('dual', 'partitioned')),
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    # Read before write: the no-op upsert still took RowExclusiveLock on the
    # mode table and a row lock on the singleton every bootstrap (item J).
    cursor.execute("SELECT 1 FROM gpu_fault_processor_counter_mode WHERE singleton")
    if cursor.fetchone() is None:
        cursor.execute(
            """
            INSERT INTO gpu_fault_processor_counter_mode(
                singleton, mode, updated_at
            ) VALUES (TRUE, 'dual', now())
            ON CONFLICT(singleton) DO NOTHING
            """
        )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS
        gpu_fault_processor_priority_count_shards (
            cluster_id TEXT NOT NULL,
            priority_bucket SMALLINT NOT NULL
                CHECK (priority_bucket IN (0, 50, 100)),
            shard_id SMALLINT NOT NULL
                CHECK (shard_id >= 0 AND shard_id < 16),
            incomplete_count BIGINT NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (
                cluster_id, priority_bucket, shard_id
            ),
            CHECK (incomplete_count >= 0)
        )
        """
    )


def _create_processor_counter_functions(cursor) -> None:
    # One counter row per cluster, updated once per statement instead
    # of once per row. The row trigger below it used to run two
    # upserts against the same counter row for every queue row - a
    # 1000-row admission batch or cleanup delete meant up to 2000
    # extra index writes and 2000 chances to serialize behind another
    # writer on the same counter. Aggregating the delta from the
    # transition table collapses that to one upsert per cluster per
    # statement, which is what makes bulk enqueue and bulk cleanup
    # scale with rows instead of with lock waits.
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION
        gpu_fault_processor_queue_count_apply(
            p_scopes TEXT[],
            p_deltas BIGINT[]
        )
        RETURNS void
        LANGUAGE plpgsql
        AS $$
        BEGIN
            -- Claim the counter rows in cluster_id order before
            -- touching them. One statement now carries deltas for
            -- every cluster it touched, and the UPDATE below joins
            -- an unordered set, so two transactions whose batches
            -- span the same two clusters could take those two row
            -- locks in opposite order. A 32-cluster burst does that
            -- thousands of times a second: the deadlock aborts the
            -- completion batch, the requests stay in the queue and
            -- the whole ingest path stalls behind them. Locking in
            -- a total order first makes the cycle impossible.
            PERFORM 1
            FROM gpu_fault_processor_queue_counts
            WHERE cluster_id = ANY (p_scopes)
            ORDER BY cluster_id
            FOR UPDATE;
            WITH deltas AS (
                SELECT scope, delta
                FROM unnest(p_scopes, p_deltas)
                    AS pairs(scope, delta)
                WHERE delta <> 0
            ),
            updated AS (
                UPDATE gpu_fault_processor_queue_counts AS counts
                SET incomplete_count=greatest(
                        0, counts.incomplete_count + deltas.delta
                    ),
                    updated_at=now()
                FROM deltas
                WHERE counts.cluster_id=deltas.scope
                RETURNING counts.cluster_id
            )
            INSERT INTO gpu_fault_processor_queue_counts(
                cluster_id, incomplete_count, updated_at
            )
            SELECT deltas.scope, greatest(0, deltas.delta), now()
            FROM deltas
            WHERE deltas.scope NOT IN (
                SELECT cluster_id FROM updated
            )
            -- Same reason as the FOR UPDATE above: rows are
            -- inserted in the order this SELECT produces them, and
            -- a conflicting row is locked at that point, so the
            -- order has to be total here too.
            ORDER BY deltas.scope
            -- Only reachable if a concurrent transaction created the
            -- row after the UPDATE above missed it. A scope with no
            -- row cannot have counted work, so a negative delta
            -- there is already floored at zero.
            ON CONFLICT(cluster_id) DO UPDATE SET
                incomplete_count=greatest(
                    0,
                    gpu_fault_processor_queue_counts
                        .incomplete_count
                    + excluded.incomplete_count
                ),
                updated_at=excluded.updated_at;
        END
        $$
        """
    )
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION
        gpu_fault_processor_queue_count_sync()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            scopes TEXT[];
            deltas BIGINT[];
            include_legacy BOOLEAN;
        BEGIN
            SELECT mode='dual'
            INTO include_legacy
            FROM gpu_fault_processor_counter_mode
            WHERE singleton=TRUE;
            include_legacy := coalesce(include_legacy, TRUE);
            IF TG_OP = 'INSERT' THEN
                IF current_setting(
                    'gpu_fault.skip_processor_queue_count_insert',
                    true
                ) = 'on' THEN
                    RETURN NULL;
                END IF;
                SELECT
                    array_agg(scope), array_agg(delta)
                INTO scopes, deltas
                FROM (
                    SELECT
                        coalesce(
                            cluster_id, '__unscoped__'
                        ) AS scope,
                        count(*)::bigint AS delta
                    FROM added
                    WHERE status IN ('PENDING', 'LEASED')
                      AND include_legacy
                    GROUP BY 1
                ) AS grouped;
            ELSIF TG_OP = 'DELETE' THEN
                SELECT
                    array_agg(scope), array_agg(delta)
                INTO scopes, deltas
                FROM (
                    SELECT
                        coalesce(
                            cluster_id, '__unscoped__'
                        ) AS scope,
                        -count(*)::bigint AS delta
                    FROM removed
                    WHERE status IN ('PENDING', 'LEASED')
                      AND include_legacy
                    GROUP BY 1
                ) AS grouped;
            ELSE
                SELECT
                    array_agg(scope), array_agg(delta)
                INTO scopes, deltas
                FROM (
                    SELECT scope, sum(delta)::bigint AS delta
                    FROM (
                        SELECT
                            coalesce(
                                cluster_id, '__unscoped__'
                            ) AS scope,
                            1 AS delta
                        FROM added
                        WHERE status IN ('PENDING', 'LEASED')
                          AND include_legacy
                        UNION ALL
                        SELECT
                            coalesce(
                                cluster_id, '__unscoped__'
                            ),
                            -1
                        FROM removed
                        WHERE status IN ('PENDING', 'LEASED')
                          AND include_legacy
                    ) AS signed
                    GROUP BY scope
                    HAVING sum(delta) <> 0
                ) AS grouped;
            END IF;
            IF scopes IS NOT NULL THEN
                PERFORM gpu_fault_processor_queue_count_apply(
                    scopes, deltas
                );
            END IF;
            RETURN NULL;
        END
        $$
        """
    )


def _create_processor_triggers(cursor) -> None:
    # The row trigger and its function are replaced, not kept
    # alongside: running both would double every delta.
    cursor.execute(
        """
        DROP TRIGGER IF EXISTS
            gpu_fault_processor_queue_count_trigger
        ON gpu_fault_processor_queue
        """
    )
    cursor.execute(
        """
        DROP FUNCTION IF EXISTS
            gpu_fault_processor_queue_count_adjust()
        """
    )
    # A transition-table trigger may only cover one event, hence
    # three definitions over the one function.
    for name, event, referencing in (
        (
            "gpu_fault_processor_queue_count_insert",
            "INSERT",
            "NEW TABLE AS added",
        ),
        (
            # No column list: Postgres rejects one together with
            # transition tables. A lease renewal therefore still
            # fires the trigger, but its rows contribute +1 and -1
            # to the same scope and the HAVING clause drops the
            # group, so no counter row is touched.
            "gpu_fault_processor_queue_count_update",
            "UPDATE",
            "OLD TABLE AS removed NEW TABLE AS added",
        ),
        (
            "gpu_fault_processor_queue_count_delete",
            "DELETE",
            "OLD TABLE AS removed",
        ),
    ):
        cursor.execute(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger
                    WHERE tgname='{name}'
                ) THEN
                    CREATE TRIGGER {name}
                    AFTER {event}
                    ON gpu_fault_processor_queue
                    REFERENCING {referencing}
                    FOR EACH STATEMENT
                    EXECUTE FUNCTION
                        gpu_fault_processor_queue_count_sync();
                END IF;
            END
            $$
            """
        )
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION
        gpu_fault_processor_priority_count_apply(
            p_scopes TEXT[],
            p_priorities SMALLINT[],
            p_shards SMALLINT[],
            p_deltas BIGINT[]
        )
        RETURNS void
        LANGUAGE plpgsql
        AS $$
        DECLARE
            target RECORD;
        BEGIN
            -- A set-based SELECT ... ORDER BY ... FOR UPDATE did
            -- not establish a stable lock order under Aurora's
            -- plan for the composite IN predicate. Admission and
            -- completion transactions each held a different shard
            -- tuple and waited for the other, producing 103
            -- deadlocks in the 32-cluster mixed burst. A procedural
            -- loop makes the order observable and therefore
            -- mandatory: every writer reaches each
            -- (cluster,bucket,shard) key in the same total order.
            FOR target IN
                SELECT
                    scope,
                    priority_bucket,
                    shard,
                    sum(delta)::bigint AS delta
                FROM unnest(
                    p_scopes,
                    p_priorities,
                    p_shards,
                    p_deltas
                ) AS pairs(
                    scope, priority_bucket, shard, delta
                )
                WHERE delta <> 0
                GROUP BY scope, priority_bucket, shard
                HAVING sum(delta) <> 0
                ORDER BY scope, priority_bucket, shard
            LOOP
                INSERT INTO
                    gpu_fault_processor_priority_count_shards(
                        cluster_id,
                        priority_bucket,
                        shard_id,
                        incomplete_count,
                        updated_at
                    )
                VALUES (
                    target.scope,
                    target.priority_bucket,
                    target.shard,
                    greatest(0, target.delta),
                    now()
                )
                ON CONFLICT(
                    cluster_id, priority_bucket, shard_id
                ) DO UPDATE SET
                    incomplete_count=greatest(
                        0,
                        gpu_fault_processor_priority_count_shards
                            .incomplete_count
                        + target.delta
                    ),
                    updated_at=excluded.updated_at;
            END LOOP;
        END
        $$
        """
    )


def _create_priority_counter_function(cursor) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION
        gpu_fault_processor_priority_count_sync()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            scopes TEXT[];
            priorities SMALLINT[];
            shards SMALLINT[];
            deltas BIGINT[];
        BEGIN
            IF TG_OP = 'INSERT' THEN
                SELECT
                    array_agg(
                        scope ORDER BY
                            scope, priority_bucket, shard
                    ),
                    array_agg(
                        priority_bucket ORDER BY
                            scope, priority_bucket, shard
                    ),
                    array_agg(
                        shard ORDER BY
                            scope, priority_bucket, shard
                    ),
                    array_agg(
                        delta ORDER BY
                            scope, priority_bucket, shard
                    )
                INTO scopes, priorities, shards, deltas
                FROM (
                    SELECT
                        coalesce(
                            cluster_id, '__unscoped__'
                        ) AS scope,
                        (
                            CASE
                                WHEN priority <= 10 THEN 0
                                WHEN priority <= 50 THEN 50
                                ELSE 100
                            END
                        )::smallint AS priority_bucket,
                        mod(
                            (
                                hashtextextended(request_id, 0)
                                & 9223372036854775807
                            ),
                            16
                        )::smallint AS shard,
                        count(*)::bigint AS delta
                    FROM added
                    WHERE status IN ('PENDING', 'LEASED')
                    GROUP BY 1, 2, 3
                ) AS grouped;
            ELSIF TG_OP = 'DELETE' THEN
                SELECT
                    array_agg(
                        scope ORDER BY
                            scope, priority_bucket, shard
                    ),
                    array_agg(
                        priority_bucket ORDER BY
                            scope, priority_bucket, shard
                    ),
                    array_agg(
                        shard ORDER BY
                            scope, priority_bucket, shard
                    ),
                    array_agg(
                        delta ORDER BY
                            scope, priority_bucket, shard
                    )
                INTO scopes, priorities, shards, deltas
                FROM (
                    SELECT
                        coalesce(
                            cluster_id, '__unscoped__'
                        ) AS scope,
                        (
                            CASE
                                WHEN priority <= 10 THEN 0
                                WHEN priority <= 50 THEN 50
                                ELSE 100
                            END
                        )::smallint AS priority_bucket,
                        mod(
                            (
                                hashtextextended(request_id, 0)
                                & 9223372036854775807
                            ),
                            16
                        )::smallint AS shard,
                        -count(*)::bigint AS delta
                    FROM removed
                    WHERE status IN ('PENDING', 'LEASED')
                    GROUP BY 1, 2, 3
                ) AS grouped;
            ELSE
                SELECT
                    array_agg(
                        scope ORDER BY
                            scope, priority_bucket, shard
                    ),
                    array_agg(
                        priority_bucket ORDER BY
                            scope, priority_bucket, shard
                    ),
                    array_agg(
                        shard ORDER BY
                            scope, priority_bucket, shard
                    ),
                    array_agg(
                        delta ORDER BY
                            scope, priority_bucket, shard
                    )
                INTO scopes, priorities, shards, deltas
                FROM (
                    SELECT
                        scope,
                        priority_bucket,
                        shard,
                        sum(delta)::bigint AS delta
                    FROM (
                        SELECT
                            coalesce(
                                cluster_id, '__unscoped__'
                            ) AS scope,
                            (
                                CASE
                                    WHEN priority <= 10 THEN 0
                                    WHEN priority <= 50 THEN 50
                                    ELSE 100
                                END
                            )::smallint AS priority_bucket,
                            mod(
                                (
                                    hashtextextended(request_id, 0)
                                    & 9223372036854775807
                                ),
                                16
                            )::smallint AS shard,
                            1 AS delta
                        FROM added
                        WHERE status IN ('PENDING', 'LEASED')
                        UNION ALL
                        SELECT
                            coalesce(
                                cluster_id, '__unscoped__'
                            ),
                            (
                                CASE
                                    WHEN priority <= 10 THEN 0
                                    WHEN priority <= 50 THEN 50
                                    ELSE 100
                                END
                            )::smallint,
                            mod(
                                (
                                    hashtextextended(request_id, 0)
                                    & 9223372036854775807
                                ),
                                16
                            )::smallint,
                            -1
                        FROM removed
                        WHERE status IN ('PENDING', 'LEASED')
                    ) AS signed
                    GROUP BY scope, priority_bucket, shard
                    HAVING sum(delta) <> 0
                ) AS grouped;
            END IF;
            IF scopes IS NOT NULL THEN
                PERFORM gpu_fault_processor_priority_count_apply(
                    scopes, priorities, shards, deltas
                );
            END IF;
            RETURN NULL;
        END
        $$
        """
    )


def _create_priority_counter_triggers(cursor) -> None:
    for name, event, referencing in (
        (
            "gpu_fault_processor_priority_count_insert",
            "INSERT",
            "NEW TABLE AS added",
        ),
        (
            "gpu_fault_processor_priority_count_update",
            "UPDATE",
            "OLD TABLE AS removed NEW TABLE AS added",
        ),
        (
            "gpu_fault_processor_priority_count_delete",
            "DELETE",
            "OLD TABLE AS removed",
        ),
    ):
        cursor.execute(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger
                    WHERE tgname='{name}'
                ) THEN
                    CREATE TRIGGER {name}
                    AFTER {event}
                    ON gpu_fault_processor_queue
                    REFERENCING {referencing}
                    FOR EACH STATEMENT
                    EXECUTE FUNCTION
                        gpu_fault_processor_priority_count_sync();
                END IF;
            END
            $$
            """
        )
        _enable_trigger_if_disabled(cursor, "gpu_fault_processor_queue", name)


def _seed_processor_counters(cursor) -> None:
    # Seeding the counter table takes a lock that conflicts with the
    # ROW EXCLUSIVE lock of every queue write, so while it runs the
    # whole region's admission, claim, completion and cleanup stall.
    # That is acceptable once, on the empty table; it is not
    # acceptable on every process start, and uvicorn's
    # ``--limit-max-requests`` turns process start into a recurring
    # event. Skipping the reseed when the table already has rows
    # keeps the lock off the steady state path - the row trigger
    # maintains the counters from there, and
    # ``processor_queue_count_status`` reports any drift.
    #
    # An empty queue has nothing to count either: seeding an empty counter
    # table from it is a TRUNCATE plus a LOCK TABLE that both end where they
    # started, so that case skips the locks as well (store review
    # 2026-09-07, item J).
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM gpu_fault_processor_queue
            WHERE status IN ('PENDING', 'LEASED')
        )
        """
    )
    queue_has_incomplete = bool(cursor.fetchone()[0])
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM gpu_fault_processor_queue_counts
        )
        """
    )
    counts_seeded = bool(cursor.fetchone()[0])
    if not counts_seeded and queue_has_incomplete:
        cursor.execute(
            """
            LOCK TABLE gpu_fault_processor_queue
            IN SHARE ROW EXCLUSIVE MODE
            """
        )
        cursor.execute("TRUNCATE gpu_fault_processor_queue_counts")
        cursor.execute(
            """
            INSERT INTO gpu_fault_processor_queue_counts(
                cluster_id, incomplete_count, updated_at
            )
            SELECT
                coalesce(cluster_id, '__unscoped__'),
                count(*),
                now()
            FROM gpu_fault_processor_queue
            WHERE status IN ('PENDING', 'LEASED')
              AND (
                  priority <> 0
                  OR (
                      SELECT mode='dual'
                      FROM gpu_fault_processor_counter_mode
                      WHERE singleton=TRUE
                  )
              )
            GROUP BY coalesce(cluster_id, '__unscoped__')
            """
        )
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM gpu_fault_processor_priority_count_shards
        )
        """
    )
    priority_counts_seeded = bool(cursor.fetchone()[0])
    if not priority_counts_seeded and queue_has_incomplete:
        cursor.execute(
            """
            LOCK TABLE gpu_fault_processor_queue
            IN SHARE ROW EXCLUSIVE MODE
            """
        )
        cursor.execute("TRUNCATE gpu_fault_processor_priority_count_shards")
        cursor.execute(
            """
            INSERT INTO gpu_fault_processor_priority_count_shards(
                cluster_id,
                priority_bucket,
                shard_id,
                incomplete_count,
                updated_at
            )
            SELECT
                coalesce(cluster_id, '__unscoped__'),
                (
                    CASE
                        WHEN priority <= 10 THEN 0
                        WHEN priority <= 50 THEN 50
                        ELSE 100
                    END
                )::smallint,
                mod(
                    (
                        hashtextextended(request_id, 0)
                        & 9223372036854775807
                    ),
                    16
                )::smallint,
                count(*),
                now()
            FROM gpu_fault_processor_queue
            WHERE status IN ('PENDING', 'LEASED')
            GROUP BY 1, 2, 3
            """
        )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_lanes_expiry
        ON gpu_fault_processor_lanes (lease_expires_at)
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_incomplete_cluster
        ON gpu_fault_processor_queue (cluster_id, priority)
        WHERE status IN ('PENDING', 'LEASED')
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_correlation_scopes
        ON gpu_fault_processor_queue
        USING GIN ((payload->'correlation_scope_keys'))
        WHERE status IN ('PENDING', 'LEASED')
        """,
    )
    _add_column_if_missing(
        cursor, "gpu_fault_processor_queue", "correlation_key", "TEXT"
    )
    # Virtual processor partitions no longer participate in claiming.
    # Strip the legacy JSON field before the strict ProcessorRequest model
    # reads an existing queue row, then remove both obsolete indexes and
    # the constant-valued column. The v5 migration removed partition state,
    # so a database recorded at v5 or later has already been swept; the two
    # sweeps have no index and read every row, so they only run on a fresh
    # database or one still below v5 (item J).
    recorded_version = _recorded_schema_version(cursor)
    if recorded_version is None or recorded_version < 5:
        cursor.execute(
            """
            UPDATE gpu_fault_processor_queue
            SET payload=payload - 'partition_id'
            WHERE payload ? 'partition_id'
            """
        )
        cursor.execute(
            """
            UPDATE gpu_fault_objects
            SET payload=payload - 'partition_id'
            WHERE kind='processor_request'
              AND payload ? 'partition_id'
            """
        )
    _drop_index_if_present(cursor, "gpu_fault_processor_queue_claim")
    _drop_index_if_present(cursor, "gpu_fault_processor_queue_partition_claim")
    _drop_column_if_present(cursor, "gpu_fault_processor_queue", "partition_id")


def _create_processor_indexes(cursor) -> None:
    # The claim window orders by (priority, created_at, request_id)
    # with no partition or path predicate. This is the index the two ordered
    # walks of the claim window (F-D2) rely on: partial on the two claimable
    # statuses so COMPLETED history never enters it. F-D2 briefly declared an
    # identical twin (``..._claim_order``); the planner picked either at
    # random, so v11 drops the twin and this one stays the only declaration.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_priority_claim
        ON gpu_fault_processor_queue (
            priority,
            created_at,
            request_id
        )
        WHERE status IN ('PENDING', 'LEASED')
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_cluster
        ON gpu_fault_processor_queue (
            cluster_id,
            status,
            created_at
        )
        WHERE status IN ('PENDING', 'LEASED')
        """,
    )
    # Per-pool claims filter on one path and then take the oldest
    # window in (priority, created_at) order. ordering_key sat
    # between the two in the previous shape, which forced a sort over
    # the whole path backlog; lane lookups use
    # gpu_fault_processor_queue_lane instead.
    _drop_index_if_present(cursor, "gpu_fault_processor_queue_path_claim")
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_path_priority_claim
        ON gpu_fault_processor_queue (
            (payload->>'path'),
            priority,
            created_at,
            request_id
        )
        WHERE status IN ('PENDING', 'LEASED')
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_lane
        ON gpu_fault_processor_queue (
            ordering_key,
            status,
            lease_expires_at
        )
        WHERE status IN ('PENDING', 'LEASED')
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_processor_queue_completed
        ON gpu_fault_processor_queue (updated_at)
        WHERE status='COMPLETED'
        """,
    )


def declared_index_names() -> frozenset[str]:
    """Every index the DDL modules declare, read from their source.

    The schema check compares this set with ``pg_indexes`` so a forgotten
    ``CREATE INDEX CONCURRENTLY`` (the three-step method of F-J3) fails the
    replica at startup instead of degrading every hot query to a sequential
    scan. Derived from the source rather than maintained by hand so a new
    index cannot be declared without also being validated.
    """

    names: set[str] = set()
    for path in sorted(Path(__file__).parent.glob("ddl*.py")):
        names.update(_CREATE_INDEX.findall(path.read_text(encoding="utf-8")))
    return frozenset(names)
