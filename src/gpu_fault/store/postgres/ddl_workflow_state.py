"""Workflow state columns and cursor-compatible reads; schema migration v16."""

from __future__ import annotations

from typing import Any

from gpu_fault.store.postgres.ddl_control_state import ensure_control_state_view
from gpu_fault.store.postgres.ddl_helpers import _declare_index, _ensure_trigger


def create_workflow_state_table(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_fault_workflows (
            request_id TEXT PRIMARY KEY,
            incident_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('PENDING', 'SAFETY_PENDING', 'BLOCKED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'SUPERSEDED')
            ),
            blocked_kind TEXT,
            fencing_token BIGINT NOT NULL CHECK (fencing_token>=1),
            execution_epoch BIGINT NOT NULL DEFAULT 0 CHECK (execution_epoch>=0),
            execution_owner_id TEXT,
            execution_lease_expires_at TIMESTAMPTZ,
            merge_revision BIGINT NOT NULL DEFAULT 0 CHECK (merge_revision>=0),
            predecessor_workflow_id TEXT,
            preempt_predecessor BOOLEAN NOT NULL DEFAULT FALSE,
            preemption_pending_by_workflow_id TEXT,
            runtime_profile_version TEXT,
            safety_only BOOLEAN NOT NULL DEFAULT FALSE,
            not_before TIMESTAMPTZ,
            failure_handled_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            execution_lease_expires_at_naive BOOLEAN NOT NULL DEFAULT FALSE,
            not_before_naive BOOLEAN NOT NULL DEFAULT FALSE,
            failure_handled_at_naive BOOLEAN NOT NULL DEFAULT FALSE,
            created_at_naive BOOLEAN NOT NULL DEFAULT FALSE,
            updated_at_naive BOOLEAN NOT NULL DEFAULT FALSE,
            payload JSONB NOT NULL CHECK (jsonb_typeof(payload)='object')
        ) WITH (fillfactor=70)
        """
    )
    _ensure_trigger(
        cursor,
        "gpu_fault_workflows_fence",
        "gpu_fault_workflows",
        """
        CREATE TRIGGER gpu_fault_workflows_fence
        BEFORE INSERT OR DELETE OR UPDATE ON gpu_fault_workflows
        FOR EACH ROW EXECUTE FUNCTION gpu_fault_native_control_state_fence('workflow')
        """,
    )


def create_workflow_payload_functions(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_workflow_payload(value gpu_fault_workflows)
        RETURNS JSONB LANGUAGE SQL STABLE PARALLEL SAFE AS $$
            SELECT value.payload || jsonb_build_object(
                'request_id', value.request_id,
                'incident_id', value.incident_id,
                'status', value.status,
                'blocked_kind', value.blocked_kind,
                'fencing_token', value.fencing_token,
                'execution_epoch', value.execution_epoch,
                'execution_owner_id', value.execution_owner_id,
                'execution_lease_expires_at',
                    gpu_fault_state_datetime_text(value.execution_lease_expires_at, value.execution_lease_expires_at_naive),
                'merge_revision', value.merge_revision,
                'predecessor_workflow_id', value.predecessor_workflow_id,
                'preempt_predecessor', value.preempt_predecessor,
                'preemption_pending_by_workflow_id', value.preemption_pending_by_workflow_id,
                'runtime_profile_version', value.runtime_profile_version,
                'safety_only', value.safety_only,
                'not_before', gpu_fault_state_datetime_text(value.not_before, value.not_before_naive),
                'failure_handled_at',
                    gpu_fault_state_datetime_text(value.failure_handled_at, value.failure_handled_at_naive),
                'created_at', gpu_fault_state_datetime_text(value.created_at, value.created_at_naive),
                'updated_at', gpu_fault_state_datetime_text(value.updated_at, value.updated_at_naive)
            )
        $$
        """
    )
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_workflow_columns(value JSONB)
        RETURNS gpu_fault_workflows LANGUAGE SQL STABLE SET timezone='UTC' AS $$
            SELECT jsonb_populate_record(NULL::gpu_fault_workflows,
                value || jsonb_build_object(
                    'execution_epoch', coalesce(value->'execution_epoch', '0'::jsonb),
                    'merge_revision', coalesce(value->'merge_revision', '0'::jsonb),
                    'preempt_predecessor', coalesce(value->'preempt_predecessor', 'false'::jsonb),
                    'safety_only', coalesce(value->'safety_only', 'false'::jsonb),
                    'execution_lease_expires_at_naive',
                        gpu_fault_state_datetime_naive(value->>'execution_lease_expires_at'),
                    'not_before_naive', gpu_fault_state_datetime_naive(value->>'not_before'),
                    'failure_handled_at_naive', gpu_fault_state_datetime_naive(value->>'failure_handled_at'),
                    'created_at_naive', gpu_fault_state_datetime_naive(value->>'created_at'),
                    'updated_at_naive', gpu_fault_state_datetime_naive(value->>'updated_at'),
                    'payload', value - ARRAY[
                        'request_id', 'incident_id', 'status', 'blocked_kind', 'fencing_token',
                        'execution_epoch', 'execution_owner_id', 'execution_lease_expires_at',
                        'merge_revision', 'predecessor_workflow_id', 'preempt_predecessor',
                        'preemption_pending_by_workflow_id', 'runtime_profile_version',
                        'safety_only', 'not_before', 'failure_handled_at', 'created_at', 'updated_at'
                    ]
                )
            )
        $$
        """
    )


def create_workflow_state_view(cursor: Any, *, validate_only: bool = False) -> None:
    # Filters belong above UNION ALL so PostgreSQL can pull up both branches
    # and preserve their ordered index paths instead of sorting every page.
    ensure_control_state_view(
        cursor,
        "gpu_fault_workflow_records",
        """
        SELECT kind, key, payload, incident_id, status, blocked_kind, fencing_token,
               execution_epoch, execution_owner_id, execution_lease_expires_at,
               merge_revision, predecessor_workflow_id, preempt_predecessor,
               preemption_pending_by_workflow_id, runtime_profile_version,
               safety_only, not_before, failure_handled_at, created_at, updated_at,
               failure_unhandled, dispatch_eligible_at
        FROM (
        SELECT o.kind, o.key, o.payload,
               o.payload->>'incident_id' AS incident_id,
               o.payload->>'status' AS status,
               o.payload->>'blocked_kind' AS blocked_kind,
               (o.payload->>'fencing_token')::bigint AS fencing_token,
               coalesce((o.payload->>'execution_epoch')::bigint, 0) AS execution_epoch,
               o.payload->>'execution_owner_id' AS execution_owner_id,
               o.payload->>'execution_lease_expires_at' AS execution_lease_expires_at,
               coalesce((o.payload->>'merge_revision')::bigint, 0) AS merge_revision,
               o.payload->>'predecessor_workflow_id' AS predecessor_workflow_id,
               o.payload->>'preempt_predecessor' AS preempt_predecessor,
               o.payload->>'preemption_pending_by_workflow_id' AS preemption_pending_by_workflow_id,
               o.payload->>'runtime_profile_version' AS runtime_profile_version,
               o.payload->>'safety_only' AS safety_only,
               o.payload->>'not_before' AS not_before,
               o.payload->>'failure_handled_at' AS failure_handled_at,
               o.payload->>'created_at' AS created_at,
               o.payload->>'updated_at' AS updated_at,
               (o.payload->>'failure_handled_at' IS NULL OR o.payload->>'failure_handled_at'='') AS failure_unhandled,
               GREATEST(o.payload->>'created_at', o.payload->>'not_before') AS dispatch_eligible_at,
               FALSE AS native
        FROM gpu_fault_objects o
        UNION ALL
        SELECT 'workflow', d.request_id, gpu_fault_workflow_payload(d),
               d.incident_id, d.status, d.blocked_kind, d.fencing_token,
               d.execution_epoch, d.execution_owner_id,
               gpu_fault_state_datetime_text(d.execution_lease_expires_at, d.execution_lease_expires_at_naive),
               d.merge_revision, d.predecessor_workflow_id, d.preempt_predecessor::text,
               d.preemption_pending_by_workflow_id, d.runtime_profile_version, d.safety_only::text,
               gpu_fault_state_datetime_text(d.not_before, d.not_before_naive),
               gpu_fault_state_datetime_text(d.failure_handled_at, d.failure_handled_at_naive),
               gpu_fault_state_datetime_text(d.created_at, d.created_at_naive),
               gpu_fault_state_datetime_text(d.updated_at, d.updated_at_naive),
               d.failure_handled_at IS NULL,
               GREATEST(
                   gpu_fault_state_datetime_text(d.created_at, d.created_at_naive),
                   gpu_fault_state_datetime_text(d.not_before, d.not_before_naive)
               ), TRUE
        FROM gpu_fault_workflows d
        ) r
        WHERE kind='workflow'
          AND (
            (native AND (SELECT gpu_fault_control_state_mode('workflow')) IN ('dual', 'dedicated'))
            OR (
                NOT native
                AND (SELECT gpu_fault_control_state_mode('workflow')) IN ('legacy', 'dual')
                AND (
                    (SELECT gpu_fault_control_state_mode('workflow'))='legacy'
                    OR NOT EXISTS (SELECT 1 FROM gpu_fault_workflows d WHERE d.request_id=r.key)
                )
            )
          )
        """,
        validate_only=validate_only,
    )


def create_workflow_state_indexes(cursor: Any) -> None:
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_workflows_status
        ON gpu_fault_workflows (status)
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_workflows_incident
        ON gpu_fault_workflows (incident_id, request_id)
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_workflows_predecessor
        ON gpu_fault_workflows (predecessor_workflow_id, request_id)
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_workflows_updated
        ON gpu_fault_workflows (
            gpu_fault_state_datetime_text(updated_at, updated_at_naive), request_id
        )
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_workflows_executable_updated
        ON gpu_fault_workflows (
            gpu_fault_state_datetime_text(updated_at, updated_at_naive), request_id
        )
        WHERE status IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_workflows_dispatch
        ON gpu_fault_workflows (
            (GREATEST(
                gpu_fault_state_datetime_text(created_at, created_at_naive),
                gpu_fault_state_datetime_text(not_before, not_before_naive)
            )), request_id
        )
        WHERE status IN ('PENDING', 'RUNNING', 'SAFETY_PENDING')
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_workflows_blocked_updated
        ON gpu_fault_workflows (
            gpu_fault_state_datetime_text(updated_at, updated_at_naive), request_id
        )
        WHERE status='BLOCKED'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_workflows_unhandled_failed
        ON gpu_fault_workflows (
            gpu_fault_state_datetime_text(updated_at, updated_at_naive), request_id
        )
        WHERE status='FAILED' AND failure_handled_at IS NULL
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_workflows_preempting_successor
        ON gpu_fault_workflows (
            predecessor_workflow_id,
            gpu_fault_state_datetime_text(created_at, created_at_naive), request_id
        )
        WHERE preempt_predecessor::text='true' AND status IN ('PENDING', 'SAFETY_PENDING')
        """,
    )


def create_workflow_state_wakeup(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE OR REPLACE FUNCTION gpu_fault_workflows_notify_wakeup()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF gpu_fault_control_state_lock_mode('workflow')<>'dedicated'
               OR NEW.status NOT IN ('PENDING', 'RUNNING', 'SAFETY_PENDING') THEN
                RETURN NULL;
            END IF;
            IF TG_OP='UPDATE'
               AND OLD.status IS NOT DISTINCT FROM NEW.status
               AND OLD.not_before IS NOT DISTINCT FROM NEW.not_before
               AND OLD.not_before_naive IS NOT DISTINCT FROM NEW.not_before_naive
               AND OLD.merge_revision IS NOT DISTINCT FROM NEW.merge_revision
               AND OLD.execution_owner_id IS NOT DISTINCT FROM NEW.execution_owner_id
               AND OLD.fencing_token IS NOT DISTINCT FROM NEW.fencing_token THEN
                RETURN NULL;
            END IF;
            PERFORM pg_notify(
                'gpu_fault_workflow_dispatch',
                json_build_object(
                    'request_id', NEW.request_id, 'cluster_id', NULL,
                    'status', NEW.status,
                    'not_before', gpu_fault_state_datetime_text(NEW.not_before, NEW.not_before_naive)
                )::text
            );
            RETURN NULL;
        END
        $$
        """
    )
    _ensure_trigger(
        cursor,
        "gpu_fault_workflows_wakeup_trigger",
        "gpu_fault_workflows",
        """
        CREATE TRIGGER gpu_fault_workflows_wakeup_trigger
        AFTER INSERT OR UPDATE OF status, not_before, not_before_naive,
            merge_revision, execution_owner_id, fencing_token
        ON gpu_fault_workflows
        FOR EACH ROW EXECUTE FUNCTION gpu_fault_workflows_notify_wakeup()
        """,
    )
