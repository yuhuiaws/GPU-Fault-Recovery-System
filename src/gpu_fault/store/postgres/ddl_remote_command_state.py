"""Typed remote-command state and legacy-compatible read projections."""

from __future__ import annotations

from typing import Any

from gpu_fault.store.postgres.ddl_control_state import ensure_control_state_view
from gpu_fault.store.postgres.ddl_helpers import _declare_index, _ensure_trigger


def create_remote_command_state_table(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_remote_commands (
            command_id TEXT PRIMARY KEY,
            cluster_id TEXT NOT NULL,
            workflow_request_id TEXT NOT NULL,
            incident_id TEXT NOT NULL,
            step_index BIGINT NOT NULL CHECK (step_index >= 0),
            fencing_token BIGINT NOT NULL CHECK (fencing_token >= 1),
            status TEXT NOT NULL
                CHECK (status IN ('PENDING', 'LEASED', 'WAITING', 'SUCCEEDED', 'FAILED')),
            lease_owner TEXT,
            last_lease_owner TEXT,
            lease_token TEXT,
            lease_expires_at TIMESTAMPTZ,
            cancellation_requested_at TIMESTAMPTZ,
            cancellation_reason TEXT,
            result_details JSONB NOT NULL DEFAULT '{}'::jsonb,
            error TEXT,
            status_source TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            lease_expires_at_naive BOOLEAN NOT NULL DEFAULT FALSE,
            cancellation_requested_at_naive BOOLEAN NOT NULL DEFAULT FALSE,
            created_at_naive BOOLEAN NOT NULL DEFAULT FALSE,
            updated_at_naive BOOLEAN NOT NULL DEFAULT FALSE,
            execution_owner TEXT NOT NULL,
            snapshot JSONB NOT NULL CHECK (jsonb_typeof(snapshot)='object')
        ) WITH (fillfactor=70)
        """
    )
    _ensure_trigger(
        cursor,
        "gpu_fault_remote_commands_fence",
        "gpu_fault_remote_commands",
        """
        CREATE TRIGGER gpu_fault_remote_commands_fence
        BEFORE INSERT OR DELETE OR UPDATE ON gpu_fault_remote_commands
        FOR EACH ROW EXECUTE FUNCTION gpu_fault_native_control_state_fence('remote_command')
        """,
    )


def create_remote_command_payload_functions(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_remote_command_payload(
            value gpu_fault_remote_commands
        ) RETURNS JSONB LANGUAGE SQL STABLE PARALLEL SAFE AS $$
            SELECT value.snapshot || jsonb_build_object(
                'command_id', value.command_id,
                'cluster_id', value.cluster_id,
                'workflow_request_id', value.workflow_request_id,
                'incident_id', value.incident_id,
                'step_index', value.step_index,
                'fencing_token', value.fencing_token,
                'status', value.status,
                'lease_owner', value.lease_owner,
                'last_lease_owner', value.last_lease_owner,
                'lease_token', value.lease_token,
                'lease_expires_at',
                    gpu_fault_state_datetime_text(value.lease_expires_at, value.lease_expires_at_naive),
                'cancellation_requested_at',
                    gpu_fault_state_datetime_text(value.cancellation_requested_at, value.cancellation_requested_at_naive),
                'cancellation_reason', value.cancellation_reason,
                'result_details', value.result_details,
                'error', value.error,
                'status_source', value.status_source,
                'created_at', gpu_fault_state_datetime_text(value.created_at, value.created_at_naive),
                'updated_at', gpu_fault_state_datetime_text(value.updated_at, value.updated_at_naive)
            )
        $$
        """
    )
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_remote_command_columns(value JSONB)
        RETURNS gpu_fault_remote_commands LANGUAGE SQL STABLE
        SET timezone='UTC' AS $$
            SELECT jsonb_populate_record(NULL::gpu_fault_remote_commands,
                value || jsonb_build_object(
                    'execution_owner', value->'step'->>'execution_owner',
                    'result_details', coalesce(value->'result_details', '{}'::jsonb),
                    'lease_expires_at_naive',
                        gpu_fault_state_datetime_naive(value->>'lease_expires_at'),
                    'cancellation_requested_at_naive',
                        gpu_fault_state_datetime_naive(value->>'cancellation_requested_at'),
                    'created_at_naive', gpu_fault_state_datetime_naive(value->>'created_at'),
                    'updated_at_naive', gpu_fault_state_datetime_naive(value->>'updated_at'),
                    'snapshot', value - ARRAY[
                        'command_id', 'cluster_id', 'workflow_request_id', 'incident_id',
                        'step_index', 'fencing_token', 'status', 'lease_owner', 'last_lease_owner',
                        'lease_token', 'lease_expires_at', 'cancellation_requested_at',
                        'cancellation_reason', 'result_details', 'error', 'status_source',
                        'created_at', 'updated_at'
                    ]
                )
            )
        $$
        """
    )


def create_remote_command_state_indexes(cursor: Any) -> None:
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_remote_commands_claim
        ON gpu_fault_remote_commands (
            cluster_id, execution_owner,
            gpu_fault_state_datetime_text(created_at, created_at_naive), command_id
        )
        WHERE status IN ('PENDING', 'LEASED', 'WAITING')
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_remote_commands_workflow
        ON gpu_fault_remote_commands (
            workflow_request_id,
            gpu_fault_state_datetime_text(created_at, created_at_naive), command_id
        )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_remote_commands_incident
        ON gpu_fault_remote_commands (incident_id, command_id)
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_remote_commands_status
        ON gpu_fault_remote_commands (status, cluster_id)
        """,
    )


def create_remote_command_state_view(
    cursor: Any, *, validate_only: bool = False
) -> None:
    # Keep the UNION branches unfiltered so ordered claim scans can use
    # Merge Append; the outer mode gate still prunes the inactive storage.
    ensure_control_state_view(
        cursor,
        "gpu_fault_remote_command_records",
        """
        SELECT kind, key, payload, cluster_id, workflow_request_id, incident_id,
               step_index, fencing_token, status, lease_owner, last_lease_owner,
               lease_token, lease_expires_at, cancellation_requested_at,
               cancellation_reason, result_details, error, status_source,
               created_at, updated_at, execution_owner
        FROM (
        SELECT o.kind, o.key, o.payload,
               o.payload->>'cluster_id' AS cluster_id,
               o.payload->>'workflow_request_id' AS workflow_request_id,
               o.payload->>'incident_id' AS incident_id,
               (o.payload->>'step_index')::bigint AS step_index,
               (o.payload->>'fencing_token')::bigint AS fencing_token,
               o.payload->>'status' AS status,
               o.payload->>'lease_owner' AS lease_owner,
               o.payload->>'last_lease_owner' AS last_lease_owner,
               o.payload->>'lease_token' AS lease_token,
               o.payload->>'lease_expires_at' AS lease_expires_at,
               o.payload->>'cancellation_requested_at' AS cancellation_requested_at,
               o.payload->>'cancellation_reason' AS cancellation_reason,
               o.payload->'result_details' AS result_details,
               o.payload->>'error' AS error,
               o.payload->>'status_source' AS status_source,
               o.payload->>'created_at' AS created_at,
               o.payload->>'updated_at' AS updated_at,
               o.payload->'step'->>'execution_owner' AS execution_owner,
               FALSE AS native
        FROM gpu_fault_objects o
        UNION ALL
        SELECT 'remote_command', d.command_id, gpu_fault_remote_command_payload(d),
               d.cluster_id, d.workflow_request_id, d.incident_id,
               d.step_index, d.fencing_token, d.status, d.lease_owner,
               d.last_lease_owner, d.lease_token,
               gpu_fault_state_datetime_text(d.lease_expires_at, d.lease_expires_at_naive),
               gpu_fault_state_datetime_text(d.cancellation_requested_at, d.cancellation_requested_at_naive),
               d.cancellation_reason, d.result_details, d.error, d.status_source,
               gpu_fault_state_datetime_text(d.created_at, d.created_at_naive),
               gpu_fault_state_datetime_text(d.updated_at, d.updated_at_naive),
               d.execution_owner, TRUE
        FROM gpu_fault_remote_commands d
        ) r
        WHERE kind='remote_command'
          AND (
            (native AND (SELECT gpu_fault_control_state_mode('remote_command')) IN ('dual', 'dedicated'))
            OR (
                NOT native
                AND (SELECT gpu_fault_control_state_mode('remote_command')) IN ('legacy', 'dual')
                AND (
                    (SELECT gpu_fault_control_state_mode('remote_command'))='legacy'
                    OR NOT EXISTS (SELECT 1 FROM gpu_fault_remote_commands d WHERE d.command_id=r.key)
                )
            )
          )
        """,
        validate_only=validate_only,
    )


def create_control_records_view(
    cursor: Any,
    *,
    workflow_enabled: bool = False,
    validate_only: bool = False,
) -> None:
    if workflow_enabled:
        ensure_control_state_view(
            cursor,
            "gpu_fault_control_records",
            """
            SELECT kind, key, payload FROM gpu_fault_objects
            WHERE kind NOT IN ('remote_command', 'workflow')
            UNION ALL
            SELECT kind, key, payload FROM gpu_fault_remote_command_records
            UNION ALL
            SELECT kind, key, payload FROM gpu_fault_workflow_records
            """,
            validate_only=validate_only,
        )
        return
    ensure_control_state_view(
        cursor,
        "gpu_fault_control_records",
        """
        SELECT kind, key, payload FROM gpu_fault_objects WHERE kind <> 'remote_command'
        UNION ALL
        SELECT kind, key, payload FROM gpu_fault_remote_command_records
        """,
        validate_only=validate_only,
    )


def create_remote_command_state_wakeup(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_remote_commands_notify_wakeup()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF gpu_fault_control_state_lock_mode('remote_command')<>'dedicated' THEN
                RETURN NULL;
            END IF;
            IF TG_OP='UPDATE' AND OLD.status IS NOT DISTINCT FROM NEW.status THEN
                RETURN NULL;
            END IF;
            PERFORM pg_notify(
                'gpu_fault_remote_command',
                json_build_object(
                    'command_id', NEW.command_id,
                    'cluster_id', NEW.cluster_id,
                    'workflow_request_id', NEW.workflow_request_id,
                    'status', NEW.status
                )::text
            );
            RETURN NULL;
        END
        $$
        """
    )
    _ensure_trigger(
        cursor,
        "gpu_fault_remote_commands_wakeup_trigger",
        "gpu_fault_remote_commands",
        """
        CREATE TRIGGER gpu_fault_remote_commands_wakeup_trigger
        AFTER INSERT OR UPDATE OF status ON gpu_fault_remote_commands
        FOR EACH ROW EXECUTE FUNCTION gpu_fault_remote_commands_notify_wakeup()
        """,
    )
