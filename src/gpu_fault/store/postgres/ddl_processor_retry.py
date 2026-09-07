from __future__ import annotations

from typing import Protocol

from gpu_fault.store.postgres.ddl_helpers import (
    _add_column_if_missing,
    _declare_index,
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
    _declare_index(
        cursor,
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_processor_queue_available
        ON gpu_fault_processor_queue (
            status,
            not_before,
            priority,
            created_at,
            request_id
        )
        """,
    )
