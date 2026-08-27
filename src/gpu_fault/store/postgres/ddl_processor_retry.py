from __future__ import annotations

from typing import Protocol


class MigrationCursor(Protocol):
    def execute(
        self,
        query: str,
        params: object | None = None,
    ) -> object: ...


def upgrade_processor_retry_schedule(cursor: MigrationCursor) -> None:
    cursor.execute(
        """
        ALTER TABLE gpu_fault_processor_queue
        ADD COLUMN IF NOT EXISTS not_before TIMESTAMPTZ
        """
    )
    cursor.execute(
        """
        ALTER TABLE gpu_fault_processor_queue
        ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0
            CHECK (retry_count >= 0)
        """
    )
    cursor.execute(
        """
        ALTER TABLE gpu_fault_processor_queue
        ADD COLUMN IF NOT EXISTS lane_policy TEXT NOT NULL DEFAULT 'STRICT'
            CHECK (lane_policy IN ('STRICT', 'REORDERABLE'))
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS gpu_fault_processor_queue_available
        ON gpu_fault_processor_queue (
            status,
            not_before,
            priority,
            created_at,
            request_id
        )
        """
    )
