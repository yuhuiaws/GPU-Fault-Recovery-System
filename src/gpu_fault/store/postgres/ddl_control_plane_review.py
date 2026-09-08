"""Schema v13 stage (control-plane review 2026-09-08, G-2 / G-9 / F-9 / F-I1).

A ``ddl*.py`` module like the others so the migration registry checksums it
and ``declared_index_names`` / ``declared_index_statements`` scrape it; kept
apart from ``ddl.py`` only because that file is at its size ratchet.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.store.postgres.ddl_helpers import (
    _declare_index,
    _set_table_options_if_different,
)


def create_v13_stage(cursor: Any) -> None:
    """Schema v13 (control-plane review 2026-09-08, G-2 / G-9 / F-9 / F-I1).

    Every ``updated_at`` index on workflows was partial per status, so the
    /metrics detail scan's newest-first read across statuses sorted the whole
    kind on disk on every scrape; the two join keys the orphan inspections
    probe (incident -> workflow pointer, workflow -> predecessor) had no
    index; the fleet-deployment retention sweep and the incident archiver
    ordered by expressions no index carried. Text order on
    ``payload->>'updated_at'`` throughout: the stored value is ``isoformat()``
    in UTC and a ``::timestamptz`` cast is STABLE, not IMMUTABLE, so it can
    neither be indexed nor use one -- the archiver's candidate query has to
    compare text, as the fleet sweep already does.
    """

    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_workflow_updated_all
        ON gpu_fault_objects (
            (payload->>'updated_at'),
            key
        )
        WHERE kind='workflow'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_workflow_predecessor
        ON gpu_fault_objects (
            (payload->>'predecessor_workflow_id'),
            key
        )
        WHERE kind='workflow'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_incident_workflow_pointer
        ON gpu_fault_objects (
            (payload->>'workflow_request_id'),
            key
        )
        WHERE kind='incident'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_incident_archive_candidate
        ON gpu_fault_objects (
            (payload->>'updated_at'),
            key
        )
        WHERE kind='incident'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_fleet_deployment_terminal
        ON gpu_fault_objects (
            (payload->>'updated_at'),
            key
        )
        WHERE kind='fleet_deployment'
          AND payload->>'status' IN ('SUCCEEDED', 'FAILED')
        """,
    )
    # Retention sweeps the review switched on (F-8, Agent 4): each orders by
    # the text timestamp it filters on, so each gets the matching partial index.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_marker_inactive_observed
        ON gpu_fault_objects (
            (payload->>'observed_at'),
            key
        )
        WHERE kind='marker'
          AND payload->>'active'='false'
        """,
    )
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_notification_created_at
        ON gpu_fault_objects (
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
        gpu_fault_event_ended
        ON gpu_fault_objects (
            (payload->>'ended_at'),
            key
        )
        WHERE kind='event'
        """,
    )
    # ``cleanup_completion_records`` asks per candidate whether an incident
    # still names the event; without this the anti-join read the incident kind
    # per batch.
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS
        gpu_fault_incident_event
        ON gpu_fault_objects (
            (payload->>'event_id'),
            key
        )
        WHERE kind='incident'
        """,
    )
    _set_table_options_if_different(
        cursor,
        "gpu_fault_objects",
        {
            "autovacuum_vacuum_scale_factor": "0.02",
            "autovacuum_analyze_scale_factor": "0.01",
        },
    )
