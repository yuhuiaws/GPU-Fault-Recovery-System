from __future__ import annotations

from gpu_fault.store.postgres.ddl_helpers import _declare_index, _ensure_trigger


def _create_telemetry_spool(cursor) -> None:
    # Telemetry's own spool, deliberately not the processor queue.
    #
    # All three telemetry channels write under an ``observed_at``
    # comparison that is repeated in SQL and taken under a row lock
    # (``observe_gpu_inventory_snapshots``,
    # ``save_collector_statuses_batch``, ``observe_telemetry_metrics``,
    # ``update_gpu_findings``, ``claim_health_signal_transitions``), so
    # a replay is a no-op and a reordered sample loses to the newer one
    # it would have overwritten. Neither the queue's lease nor its
    # per-node ordering key buys them anything, and correlation never
    # looks at them at all (``is_correlated_fault`` is fault-paths
    # only) - what the queue charges them is a lane row, a
    # ``FOR EACH STATEMENT`` counter trigger that takes the cluster's
    # counter row until commit, a share of the claim's O(depth) window
    # scan, and a completion round trip carrying a response body no
    # collector reads.
    #
    # So: no lane table, no counter table, no trigger, no COMPLETED
    # tombstone. What is left is the part that actually has to be
    # durable - the payload - plus the two columns that make a
    # crashed consumer recoverable.
    #
    # Coalescing comes from the primary key instead of from a
    # ``SELECT ... FOR UPDATE`` followed by an update: ``spool_key`` is
    # the ordering key for a channel that supersedes its own samples
    # and carries the request id for one that does not, so one
    # ``INSERT ... ON CONFLICT`` both admits and coalesces.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_telemetry_spool (
            spool_key TEXT PRIMARY KEY,
            cluster_id TEXT,
            path TEXT NOT NULL,
            request_id TEXT NOT NULL,
            revision BIGINT NOT NULL DEFAULT 0,
            attempts INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            available_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            payload JSONB NOT NULL
        )
        """
    )
    # A lease is an ``available_at`` in the future, exactly as in the
    # notification delivery spool, so an expired lease needs no reaper
    # to become claimable again - it simply falls back inside the
    # claim window. This index is the claim: one range scan from the
    # front, and the ``spool_key`` tiebreak keeps two consumers taking
    # the same rows in the same order rather than crossing.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_telemetry_spool_available
        ON gpu_fault_telemetry_spool (available_at, spool_key)
        """,
    )
    # Dedicated consumers claim one telemetry path at a time to give
    # inventory, GPU metrics and host summaries bounded service even
    # when one stream dominates. Keep that claim as one ordered index
    # range scan rather than filtering the global availability index.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_telemetry_spool_path_available
        ON gpu_fault_telemetry_spool (
            path, available_at, spool_key
        )
        """,
    )
    # Depth is reported and capped per cluster, and the cap is the
    # only thing on the admission path that has to read the table.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_telemetry_spool_cluster
        ON gpu_fault_telemetry_spool (cluster_id)
        """,
    )
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION
        gpu_fault_telemetry_spool_notify_available()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.available_at <= clock_timestamp() THEN
                PERFORM pg_notify(
                    'gpu_fault_telemetry_spool',
                    json_build_object(
                        'path', NEW.path
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
        "gpu_fault_telemetry_spool_notify_available_trigger",
        "gpu_fault_telemetry_spool",
        """
        CREATE TRIGGER
        gpu_fault_telemetry_spool_notify_available_trigger
        AFTER INSERT OR UPDATE OF available_at, revision
        ON gpu_fault_telemetry_spool
        FOR EACH ROW
        EXECUTE FUNCTION
            gpu_fault_telemetry_spool_notify_available()
        """,
    )
