"""Shared migration metadata and fences for control-state tables."""

from __future__ import annotations

from typing import Any

from gpu_fault.store.postgres.ddl_helpers import _ensure_trigger


def create_control_state_registry(cursor: Any) -> None:
    cursor.execute(
        "SELECT to_regclass('gpu_fault_control_state_modes'), "
        "to_regclass('gpu_fault_remote_commands'), to_regclass('gpu_fault_workflows'), "
        "EXISTS(SELECT 1 FROM gpu_fault_schema_migrations WHERE version>=15)"
    )
    registry, commands, workflows, migrated = cursor.fetchone()
    if registry is None and (commands is not None or workflows is not None or migrated):
        raise RuntimeError(
            "control-state migration metadata is missing; restore it before ensure"
        )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_control_state_modes (
            kind TEXT PRIMARY KEY CHECK (kind IN ('remote_command', 'workflow')),
            mode TEXT NOT NULL DEFAULT 'legacy'
                CHECK (mode IN ('legacy', 'dual', 'dedicated')),
            revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0),
            dedicated_at TIMESTAMPTZ,
            backfill_after_key TEXT,
            backfill_complete BOOLEAN NOT NULL DEFAULT FALSE,
            legacy_purged BOOLEAN NOT NULL DEFAULT FALSE,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (mode <> 'dedicated' OR dedicated_at IS NOT NULL),
            CHECK (NOT legacy_purged OR mode = 'dedicated')
        )
        """
    )
    cursor.execute("SELECT kind, mode FROM gpu_fault_control_state_modes")
    modes = dict(cursor.fetchall())
    if set(modes) != {"remote_command", "workflow"}:
        if registry is not None:
            raise RuntimeError(
                "control-state migration metadata is missing or invalid; restore it before ensure"
            )
        cursor.execute(
            """
            INSERT INTO gpu_fault_control_state_modes(kind)
            VALUES ('remote_command'), ('workflow')
            """
        )
    if any(
        modes.get(kind) == "dedicated" and table is None
        for kind, table in (("remote_command", commands), ("workflow", workflows))
    ):
        raise RuntimeError(
            "dedicated control-state table is missing; restore it before ensure"
        )
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_control_state_mode(record_kind TEXT)
        RETURNS TEXT LANGUAGE plpgsql STABLE AS $$
        DECLARE result TEXT;
        BEGIN
            SELECT mode INTO result FROM gpu_fault_control_state_modes
            WHERE kind=record_kind;
            IF result IS NULL OR result NOT IN ('legacy', 'dual', 'dedicated') THEN
                RAISE EXCEPTION 'control-state migration mode is unavailable'
                    USING ERRCODE='55000';
            END IF;
            RETURN result;
        END
        $$
        """
    )
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_control_state_lock_mode(record_kind TEXT)
        RETURNS TEXT LANGUAGE plpgsql VOLATILE AS $$
        DECLARE result TEXT;
        BEGIN
            -- Compound writers can touch both kinds; one lock order avoids
            -- a cutover deadlocking with a workflow -> command transaction.
            PERFORM pg_advisory_xact_lock_shared(hashtextextended(
                'gpu_fault_control_state/remote_command', 0));
            PERFORM pg_advisory_xact_lock_shared(hashtextextended(
                'gpu_fault_control_state/workflow', 0));
            -- A pre-cutover repeatable snapshot must fail rather than read the old mode.
            IF current_setting('transaction_isolation')
               IN ('repeatable read', 'serializable') THEN
                SELECT mode INTO result FROM gpu_fault_control_state_modes
                WHERE kind=record_kind FOR SHARE;
            ELSE
                SELECT mode INTO result FROM gpu_fault_control_state_modes
                WHERE kind=record_kind;
            END IF;
            IF result IS NULL OR result NOT IN ('legacy', 'dual', 'dedicated') THEN
                RAISE EXCEPTION 'control-state migration mode is unavailable'
                    USING ERRCODE='55000';
            END IF;
            RETURN result;
        END
        $$
        """
    )


def create_control_state_time_functions(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_state_datetime_text(
            instant TIMESTAMPTZ, is_naive BOOLEAN
        ) RETURNS TEXT LANGUAGE SQL IMMUTABLE PARALLEL SAFE AS $$
            SELECT CASE WHEN instant IS NULL THEN NULL ELSE
                to_char(instant AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US')
                || CASE WHEN is_naive THEN '' ELSE 'Z' END END
        $$
        """
    )
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_state_datetime_naive(value TEXT)
        RETURNS BOOLEAN LANGUAGE SQL IMMUTABLE PARALLEL SAFE AS $$
            SELECT value IS NOT NULL AND value !~ '(Z|[+-][0-9]{2}:[0-9]{2})$'
        $$
        """
    )
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_control_state_descriptor(record_kind TEXT)
        RETURNS TABLE(table_name TEXT, key_field TEXT, payload_function TEXT,
                      columns_function TEXT, snapshot_field TEXT)
        LANGUAGE SQL IMMUTABLE PARALLEL SAFE AS $$
            SELECT table_name, key_field, payload_function, columns_function, snapshot_field
            FROM (VALUES
                ('remote_command', 'gpu_fault_remote_commands', 'command_id',
                 'gpu_fault_remote_command_payload', 'gpu_fault_remote_command_columns',
                 'snapshot'),
                ('workflow', 'gpu_fault_workflows', 'request_id',
                 'gpu_fault_workflow_payload', 'gpu_fault_workflow_columns', 'payload')
            ) AS definitions(kind, table_name, key_field, payload_function,
                             columns_function, snapshot_field)
            WHERE kind=record_kind
        $$
        """
    )


def create_legacy_control_state_fence(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_objects_control_state_fence()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE record_kind TEXT; current_mode TEXT;
        BEGIN
            IF TG_OP='UPDATE' AND
               (OLD.kind, OLD.key) IS DISTINCT FROM (NEW.kind, NEW.key) AND
               (OLD.kind IN ('remote_command', 'workflow') OR
                NEW.kind IN ('remote_command', 'workflow')) THEN
                RAISE EXCEPTION 'control-record identity cannot change in place'
                    USING ERRCODE='55000';
            END IF;
            record_kind := CASE WHEN TG_OP='DELETE' THEN OLD.kind ELSE NEW.kind END;
            IF record_kind NOT IN ('remote_command', 'workflow') THEN
                RETURN CASE WHEN TG_OP='DELETE' THEN OLD ELSE NEW END;
            END IF;
            current_mode := gpu_fault_control_state_lock_mode(record_kind);
            IF current_mode='dedicated' THEN
                IF TG_OP='DELETE' AND
                   current_setting('gpu_fault.control_state_cleanup', true)=record_kind THEN
                    RETURN OLD;
                END IF;
                RAISE EXCEPTION 'legacy control-state writer is fenced after cutover'
                    USING ERRCODE='55000';
            END IF;
            RETURN CASE WHEN TG_OP='DELETE' THEN OLD ELSE NEW END;
        END
        $$
        """
    )
    _ensure_trigger(
        cursor,
        "gpu_fault_objects_control_state_fence_trigger",
        "gpu_fault_objects",
        """
        CREATE TRIGGER gpu_fault_objects_control_state_fence_trigger
        BEFORE INSERT OR DELETE OR UPDATE ON gpu_fault_objects
        FOR EACH ROW EXECUTE FUNCTION gpu_fault_objects_control_state_fence()
        """,
    )
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_native_control_state_fence()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE current_mode TEXT; expected_table TEXT;
        BEGIN
            SELECT table_name INTO expected_table
            FROM gpu_fault_control_state_descriptor(TG_ARGV[0]);
            IF expected_table IS DISTINCT FROM TG_TABLE_NAME THEN
                RAISE EXCEPTION 'control-state trigger identity differs'
                    USING ERRCODE='55000';
            END IF;
            current_mode := gpu_fault_control_state_lock_mode(TG_ARGV[0]);
            IF current_mode='legacy' AND TG_OP='DELETE' AND
               current_setting('gpu_fault.control_state_reset', true)=TG_ARGV[0] THEN
                RETURN OLD;
            END IF;
            IF current_mode='legacy' OR
               (current_mode='dual' AND pg_trigger_depth()<2) THEN
                RAISE EXCEPTION 'dedicated control-state write is not authoritative'
                    USING ERRCODE='55000';
            END IF;
            RETURN CASE WHEN TG_OP='DELETE' THEN OLD ELSE NEW END;
        END
        $$
        """
    )


def ensure_control_state_view(
    cursor: Any,
    name: str,
    query: str,
    *,
    validate_only: bool = False,
) -> None:
    """Compare server-deparsed definitions before replacing a live view."""
    if not name.startswith("gpu_fault_") or not name.replace("_", "").isalnum():
        raise ValueError("invalid control-state view name")
    cursor.execute("SELECT pg_get_viewdef(to_regclass(%s), true)", (name,))
    row = cursor.fetchone()
    previous = row[0] if row else None
    if previous is not None:
        cursor.execute(f"CREATE TEMP VIEW gf_control_state_view_probe AS {query}")
        cursor.execute(
            "SELECT pg_get_viewdef('gf_control_state_view_probe'::regclass, true)"
        )
        expected = cursor.fetchone()[0]
        cursor.execute("DROP VIEW gf_control_state_view_probe")
        if previous == expected:
            return
    if validate_only:
        raise RuntimeError(f"control-state view {name} differs; run --ensure-schema")
    cursor.execute(f"CREATE OR REPLACE VIEW {name} AS {query}")
