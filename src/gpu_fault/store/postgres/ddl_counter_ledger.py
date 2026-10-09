"""Schema v20 stage: per-writer-backend processor counter rows.

A ``ddl*.py`` module like the others so the migration registry checksums it
and the startup validation scrapes the trigger-function bodies from it; kept
apart from ``ddl.py`` only because that file is at its size ratchet.

Five perf r1 runs on one release and one database split into two PASS
(1449 / 1542 req/s) and two FAIL (663 / 490 req/s, 2.75 % / 21.6 % reserved
503s) with balanced api-ha load and no node or Aurora CPU saturation. The RDS
log (``log_lock_waits``) told them apart: lock waits of a second or more,
4665 / 6984 per FAIL window against 603 / 523 per PASS window, almost all
"while inserting index tuple" into ``gpu_fault_processor_priority_count_shards``
(6990 / 8907 vs 1080 / 988) and ``gpu_fault_processor_queue_counts``
(196 / 551 vs 0) -- ShareLock on transaction and AccessExclusiveLock on tuple
inside the statement-level counter triggers of the admission flush and the
completion CTE. Every concurrent flush and completion for one cluster upserted
the same (cluster, bucket, hash-of-request shard) rows and the same
per-cluster count row, so the transactions queued behind each other for their
whole duration; the total lock order added earlier turned 103 deadlocks into
this convoy, and a 1 ms rise in commit latency tipped a system already 62 %
of its active statements waiting on Lock:transactionid into expiring
admission batches -- the 503s.

The counter rows are now a per-writer ledger. ``shard_id`` is the writer's
``mod(pg_backend_pid(), 128)`` in both tables (the per-cluster count table
gains the same column and a composite key), so a backend only ever touches
its own rows, and the rows hold signed partial sums: backend B completing a
request backend A admitted drives B's row negative while the aggregate stays
exact, which is why the ``>= 0`` checks and the ``greatest(0, ...)`` clamps
are gone (with per-writer rows a clamp would make the aggregate drift
upward). Every reader already sums per cluster or per (cluster, bucket). Two
live backends share rows only when their pids collide modulo 128; they then
apply in the same total (scope, bucket, shard) order, so a collision costs a
wait, never a deadlock. The statement-level triggers, their transition-table
aggregation and the mode semantics (``dual`` maintains both tables,
``partitioned`` only the shards) are unchanged. Existing hash-sharded rows
stay valid partial sums, so nothing is reseeded.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.store.postgres.ddl_helpers import (
    _add_column_if_missing,
    _enable_trigger_if_disabled,
)

#: Writer shards per (cluster[, bucket]). The CHECK constraints below and the
#: ``mod(pg_backend_pid(), ...)`` in the trigger functions must agree.
COUNTER_WRITER_SHARDS = 128
COUNTS_TABLE = "gpu_fault_processor_queue_counts"
SHARDS_TABLE = "gpu_fault_processor_priority_count_shards"
COUNTS_SHARD_CHECK = "gpu_fault_processor_queue_counts_shard_id_check"
SHARDS_SHARD_CHECK = "gpu_fault_processor_priority_count_shards_shard_id_check"


def _check_constraints(cursor: Any, table: str, marker: str) -> list[str]:
    """Names of ``table``'s CHECK constraints whose definition matches ``marker``.

    One aggregated row, read with ``fetchone``: the DDL test doubles answer
    every catalog read with a one-column row, and a non-text answer is read
    as "none".
    """

    cursor.execute(
        """
        SELECT string_agg(conname, ',' ORDER BY conname)
        FROM pg_constraint
        WHERE conrelid = to_regclass(%s)
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE %s
        """,
        (table, marker),
    )
    row = cursor.fetchone()
    value = row[0] if row else None
    return value.split(",") if isinstance(value, str) and value else []


def _ensure_shard_check(cursor: Any, table: str, name: str) -> None:
    """``shard_id`` bounded by the current writer-shard count, replacing a
    stale bound; nothing is issued when the current bound is in place."""

    if _check_constraints(cursor, table, f"%shard_id < {COUNTER_WRITER_SHARDS})%"):
        return
    for stale in _check_constraints(cursor, table, "%shard_id%"):
        cursor.execute(f"ALTER TABLE {table} DROP CONSTRAINT {stale}")
    cursor.execute(
        f"ALTER TABLE {table} ADD CONSTRAINT {name} "
        f"CHECK (shard_id >= 0 AND shard_id < {COUNTER_WRITER_SHARDS})"
    )


def _drop_check_constraints(cursor: Any, table: str, marker: str) -> None:
    for name in _check_constraints(cursor, table, marker):
        cursor.execute(f"ALTER TABLE {table} DROP CONSTRAINT {name}")


def _ensure_primary_key(cursor: Any, table: str, columns: tuple[str, ...]) -> None:
    """The primary key on exactly ``columns``; replaced in one ALTER when it
    differs, left alone when it matches (the ALTER takes AccessExclusiveLock)."""

    cursor.execute(
        """
        SELECT string_agg(conname || ':' || pg_get_constraintdef(oid), ',')
        FROM pg_constraint
        WHERE conrelid = to_regclass(%s)
          AND contype = 'p'
        """,
        (table,),
    )
    row = cursor.fetchone()
    value = row[0] if row else None
    rendered = ", ".join(columns)
    if value is None:
        cursor.execute(f"ALTER TABLE {table} ADD PRIMARY KEY ({rendered})")
        return
    if not isinstance(value, str):
        return
    name, _, definition = value.partition(":")
    if definition == f"PRIMARY KEY ({rendered})":
        return
    cursor.execute(
        f"ALTER TABLE {table} DROP CONSTRAINT {name}, ADD PRIMARY KEY ({rendered})"
    )


def create_counter_ledger_tables_stage(cursor: Any) -> None:
    """Schema v20: the two counter tables' per-writer shape (module docstring).

    Catalog-checked, so a database already in this shape sees only reads: the
    per-cluster count table gains ``shard_id`` (existing rows become shard 0)
    and the key ``(cluster_id, shard_id)``, both shard checks bound the writer
    count, and the ``>= 0`` checks go because the rows are signed partial
    sums.
    """

    _add_column_if_missing(
        cursor, COUNTS_TABLE, "shard_id", "SMALLINT NOT NULL DEFAULT 0"
    )
    _ensure_shard_check(cursor, COUNTS_TABLE, COUNTS_SHARD_CHECK)
    _ensure_primary_key(cursor, COUNTS_TABLE, ("cluster_id", "shard_id"))
    _drop_check_constraints(cursor, COUNTS_TABLE, "%incomplete_count >= 0%")
    _ensure_shard_check(cursor, SHARDS_TABLE, SHARDS_SHARD_CHECK)
    _drop_check_constraints(cursor, SHARDS_TABLE, "%incomplete_count >= 0%")


def _create_processor_counter_functions(cursor: Any) -> None:
    # One upsert per cluster per statement (the transition table carries the
    # whole batch), on the writer's own row: ``shard_id`` is this backend's
    # pid modulo the writer count, so no other transaction waits on it. Two
    # live backends with the same residue reach the scopes in the same total
    # order, so even that collision cannot deadlock.
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
        DECLARE
            shard SMALLINT := mod(pg_backend_pid(), 128)::smallint;
            target RECORD;
        BEGIN
            FOR target IN
                SELECT scope, sum(delta)::bigint AS delta
                FROM unnest(p_scopes, p_deltas)
                    AS pairs(scope, delta)
                WHERE delta <> 0
                GROUP BY scope
                HAVING sum(delta) <> 0
                ORDER BY scope
            LOOP
                -- Signed partial sum: a completion of work another
                -- backend admitted legitimately takes this row below
                -- zero; the per-cluster aggregate stays exact.
                INSERT INTO gpu_fault_processor_queue_counts(
                    cluster_id, shard_id, incomplete_count, updated_at
                )
                VALUES (target.scope, shard, target.delta, now())
                ON CONFLICT(cluster_id, shard_id) DO UPDATE SET
                    incomplete_count=
                        gpu_fault_processor_queue_counts.incomplete_count
                        + excluded.incomplete_count,
                    updated_at=excluded.updated_at;
            END LOOP;
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


def _create_processor_triggers(cursor: Any) -> None:
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
            -- The trigger passes this backend's own writer shard for
            -- every row, so concurrent transactions touch disjoint
            -- keys. The procedural loop keeps the total
            -- (scope, bucket, shard) order all the same: it is what
            -- makes two writers that collide on a pid residue wait
            -- rather than deadlock, and what a caller supplying
            -- arbitrary shards still relies on.
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
                -- Signed partial sum per writer; see the module
                -- docstring for why no clamp may apply here.
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
                    target.delta,
                    now()
                )
                ON CONFLICT(
                    cluster_id, priority_bucket, shard_id
                ) DO UPDATE SET
                    incomplete_count=
                        gpu_fault_processor_priority_count_shards
                            .incomplete_count
                        + target.delta,
                    updated_at=excluded.updated_at;
            END LOOP;
        END
        $$
        """
    )


def _create_priority_counter_function(cursor: Any) -> None:
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
                        mod(pg_backend_pid(), 128)::smallint AS shard,
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
                        mod(pg_backend_pid(), 128)::smallint AS shard,
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
                            mod(pg_backend_pid(), 128)::smallint AS shard,
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
                            mod(pg_backend_pid(), 128)::smallint,
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


def _create_priority_counter_triggers(cursor: Any) -> None:
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
