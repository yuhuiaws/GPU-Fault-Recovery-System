from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from threading import RLock


from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionResult,
    NodeActionStatus,
)


class NodeActionLedger:
    def __init__(
        self,
        path: str,
        *,
        retention_seconds: int = 604800,
        max_results: int = 10000,
    ) -> None:
        if retention_seconds < 600:
            raise ValueError("node action retention must be at least 600 seconds")
        if max_results < 100:
            raise ValueError("node action max results must be at least 100")
        self._lock = RLock()
        self.retention_seconds = retention_seconds
        self.max_results = max_results
        self._last_cleanup_monotonic = 0.0
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                command_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS fencing (
                incident_id TEXT PRIMARY KEY,
                token INTEGER NOT NULL,
                updated_at TEXT
            )
            """
        )
        self._ensure_column("results", "completed_at", "TEXT")
        self._ensure_column("results", "attempt", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column(
            "results",
            "state",
            "TEXT NOT NULL DEFAULT 'COMPLETED'",
        )
        self._ensure_column("results", "operation", "TEXT")
        self._ensure_column("results", "started_at", "TEXT")
        self._ensure_column("fencing", "updated_at", "TEXT")
        self._interrupt_in_progress()
        self.cleanup()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def get(self, command_id: str) -> NodeActionResult | None:
        with self._lock:
            row = self._db.execute(
                "SELECT state, payload FROM results WHERE command_id=?",
                (command_id,),
            ).fetchone()
        if row and row[0] == "IN_PROGRESS":
            return None
        return NodeActionResult.model_validate_json(row[1]) if row else None

    def mark_in_progress(
        self,
        command: NodeActionCommand,
        attempt: int,
    ) -> None:
        marker = NodeActionResult(
            command_id=command.command_id,
            operation=command.operation,
            status=NodeActionStatus.INTERRUPTED,
            error="node action is in progress",
            retryable=False,
            attempt=attempt,
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._db.execute(
                """
                INSERT INTO results(
                    command_id, payload, completed_at, attempt,
                    state, operation, started_at
                ) VALUES (?, ?, NULL, ?, 'IN_PROGRESS', ?, ?)
                ON CONFLICT(command_id) DO UPDATE SET
                    payload=excluded.payload,
                    completed_at=NULL,
                    attempt=excluded.attempt,
                    state='IN_PROGRESS',
                    operation=excluded.operation,
                    started_at=excluded.started_at
                WHERE excluded.attempt >= results.attempt
                """,
                (
                    command.command_id,
                    marker.model_dump_json(),
                    attempt,
                    command.operation.value,
                    now,
                ),
            )

    def _interrupt_in_progress(self) -> None:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT command_id, payload, attempt
                FROM results
                WHERE state='IN_PROGRESS'
                """
            ).fetchall()
            for command_id, payload, attempt in rows:
                marker = NodeActionResult.model_validate_json(payload)
                interrupted = marker.model_copy(
                    update={
                        "status": NodeActionStatus.INTERRUPTED,
                        "error": (
                            "agent restarted while the action was in "
                            "progress; manual confirmation is required"
                        ),
                        "retryable": False,
                        "attempt": attempt,
                        "completed_at": datetime.now(timezone.utc),
                    }
                )
                self._db.execute(
                    """
                    UPDATE results
                    SET payload=?, completed_at=?, state='INTERRUPTED'
                    WHERE command_id=?
                    """,
                    (
                        interrupted.model_dump_json(),
                        interrupted.completed_at.isoformat(),
                        command_id,
                    ),
                )

    def save(self, result: NodeActionResult) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT INTO results(
                    command_id, payload, completed_at, attempt,
                    state, operation
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(command_id) DO UPDATE SET
                    payload=excluded.payload,
                    completed_at=excluded.completed_at,
                    attempt=excluded.attempt,
                    state=excluded.state,
                    operation=excluded.operation
                WHERE excluded.attempt >= results.attempt
                """,
                (
                    result.command_id,
                    result.model_dump_json(),
                    result.completed_at.isoformat(),
                    result.attempt,
                    result.status.value,
                    result.operation.value,
                ),
            )
        self._maybe_cleanup()

    def delete(self, command_id: str) -> None:
        with self._lock:
            self._db.execute(
                "DELETE FROM results WHERE command_id=?",
                (command_id,),
            )

    def _maybe_cleanup(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup_monotonic < 300:
            return
        self.cleanup()

    def cleanup(self, now: datetime | None = None) -> dict[str, int]:
        timestamp = now or datetime.now(timezone.utc)
        cutoff = (timestamp - timedelta(seconds=self.retention_seconds)).isoformat()
        with self._lock:
            results = self._db.execute(
                """
                DELETE FROM results
                WHERE completed_at IS NOT NULL
                  AND completed_at < ?
                """,
                (cutoff,),
            ).rowcount
            overflow = self._db.execute(
                """
                DELETE FROM results
                WHERE command_id IN (
                    SELECT command_id
                    FROM results
                    WHERE state != 'IN_PROGRESS'
                    ORDER BY completed_at DESC, command_id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.max_results,),
            ).rowcount
            fencing = self._db.execute(
                """
                DELETE FROM fencing
                WHERE updated_at IS NOT NULL
                  AND updated_at < ?
                """,
                (cutoff,),
            ).rowcount
            self._last_cleanup_monotonic = time.monotonic()
        return {
            "results": results + overflow,
            "fencing": fencing,
        }

    def accept_fencing(self, incident_id: str, token: int) -> bool:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT token FROM fencing WHERE incident_id=?",
                    (incident_id,),
                ).fetchone()
                if row and token < row[0]:
                    self._db.execute("ROLLBACK")
                    return False
                self._db.execute(
                    """
                    INSERT INTO fencing(
                        incident_id, token, updated_at
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(incident_id)
                    DO UPDATE SET
                        token=MAX(token, excluded.token),
                        updated_at=excluded.updated_at
                    """,
                    (
                        incident_id,
                        token,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                self._db.execute("COMMIT")
                return True
            except Exception:
                self._db.execute("ROLLBACK")
                raise
