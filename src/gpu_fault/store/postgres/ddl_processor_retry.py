from __future__ import annotations

from typing import Protocol

from gpu_fault.store.postgres.ddl_helpers import (
    _add_column_if_missing,
)


class MigrationCursor(Protocol):
    def execute(
        self,
        query: str,
        params: object | None = None,
    ) -> object: ...


def upgrade_processor_retry_schedule(cursor: MigrationCursor) -> None:
    _add_column_if_missing(
        cursor, "gpu_fault_processor_queue", "not_before", "TIMESTAMPTZ"
    )
    _add_column_if_missing(
        cursor,
        "gpu_fault_processor_queue",
        "retry_count",
        "INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0)",
    )
    _add_column_if_missing(
        cursor,
        "gpu_fault_processor_queue",
        "lane_policy",
        "TEXT NOT NULL DEFAULT 'STRICT' CHECK (lane_policy IN ('STRICT', 'REORDERABLE'))",
    )
    # ``gpu_fault_processor_queue_available`` (status, not_before, priority,
    # created_at, request_id) was declared here from v7 to v22; it led with
    # ``status`` and so blocked every HOT update. Schema v23 drops it: the
    # expired-lease reclaim it served reads an in-flight-only queue now.
