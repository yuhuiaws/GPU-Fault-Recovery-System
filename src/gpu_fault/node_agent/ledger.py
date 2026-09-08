from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any

from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionResult,
    NodeActionStatus,
)

LOGGER = logging.getLogger(__name__)

# ``PRAGMA user_version`` of a ledger whose ``results`` table is keyed by
# ``(command_id, attempt)``. Ledgers written before that carry version 0 and a
# single-row-per-command table; they are migrated in place on open.
LEDGER_SCHEMA_VERSION = 2
DEFAULT_RETENTION_SECONDS = 30 * 24 * 3600

# ``state`` of a row whose handler has been dispatched but whose result has not
# been written yet.
IN_PROGRESS_STATE = "IN_PROGRESS"
RESTART_INTERRUPTED_ERROR = (
    "agent restarted while the action was in progress; manual confirmation is required"
)

_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS results (
    command_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    payload TEXT NOT NULL,
    completed_at TEXT,
    state TEXT NOT NULL DEFAULT 'COMPLETED',
    operation TEXT,
    started_at TEXT,
    incident_id TEXT,
    workflow_request_id TEXT,
    fencing_token INTEGER,
    gpu_uuids TEXT,
    parameters_digest TEXT,
    signature_digest TEXT,
    exit_code INTEGER,
    PRIMARY KEY (command_id, attempt)
)
"""

_AUDIT_COLUMNS = (
    "attempt",
    "state",
    "operation",
    "started_at",
    "completed_at",
    "incident_id",
    "workflow_request_id",
    "fencing_token",
    "gpu_uuids",
    "parameters_digest",
    "signature_digest",
    "exit_code",
)


def canonical_digest(value: Any) -> str:
    """sha256 of the canonical JSON form; what the ledger keeps of parameters."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class NodeActionLedger:
    def __init__(
        self,
        path: str,
        *,
        retention_seconds: int = DEFAULT_RETENTION_SECONDS,
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
            CREATE TABLE IF NOT EXISTS fencing (
                incident_id TEXT PRIMARY KEY,
                token INTEGER NOT NULL,
                updated_at TEXT
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_health (
                id INTEGER PRIMARY KEY,
                probed_at TEXT
            )
            """
        )
        self._ensure_column("fencing", "updated_at", "TEXT")
        self._migrate_results_table()
        self._interrupt_in_progress()
        self.cleanup()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _migrate_results_table(self) -> None:
        """Bring ``results`` to the per-attempt schema without losing rows.

        Nodes upgrade with a populated ledger. SQLite cannot change a primary
        key in place, so the old table is renamed, the new one created, and
        the rows copied -- one transaction, so a crash mid-way leaves either
        the old or the new table, never neither.
        """

        version = int(self._db.execute("PRAGMA user_version").fetchone()[0])
        if version >= LEDGER_SCHEMA_VERSION:
            return
        has_results = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='results'"
        ).fetchone()
        if not has_results:
            self._db.execute(_RESULTS_TABLE)
            self._db.execute(f"PRAGMA user_version={LEDGER_SCHEMA_VERSION}")
            return
        # The pre-audit code added these lazily; a ledger opened only by a
        # very old agent may still be missing some.
        self._ensure_column("results", "completed_at", "TEXT")
        self._ensure_column("results", "attempt", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("results", "state", "TEXT NOT NULL DEFAULT 'COMPLETED'")
        self._ensure_column("results", "operation", "TEXT")
        self._ensure_column("results", "started_at", "TEXT")
        legacy_rows = int(
            self._db.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        )
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute("ALTER TABLE results RENAME TO results_legacy")
            self._db.execute(_RESULTS_TABLE)
            self._db.execute(
                """
                INSERT INTO results(
                    command_id, attempt, payload, completed_at, state,
                    operation, started_at
                )
                SELECT command_id, attempt, payload, completed_at, state,
                       operation, started_at
                FROM results_legacy
                """
            )
            self._db.execute("DROP TABLE results_legacy")
            self._db.execute(f"PRAGMA user_version={LEDGER_SCHEMA_VERSION}")
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise
        LOGGER.info(
            "node action ledger migrated to per-attempt schema "
            "schema_version=%d rows=%d",
            LEDGER_SCHEMA_VERSION,
            legacy_rows,
        )

    def get(self, command_id: str) -> NodeActionResult | None:
        """The latest attempt's result, or None while it is still running."""

        row = self.latest_row(command_id)
        return row[2] if row else None

    def latest_row(
        self, command_id: str
    ) -> tuple[str, int, NodeActionResult | None] | None:
        """State, attempt and result of the latest attempt, IN_PROGRESS included.

        ``get`` answers ``None`` for an IN_PROGRESS row, which reads exactly
        like "this command was never run" -- and a caller that believes that
        runs the destructive handler a second time. This is the raw row: the
        result is ``None`` only while the attempt is still marked in progress.
        """

        with self._lock:
            row = self._db.execute(
                """
                SELECT state, attempt, payload FROM results
                WHERE command_id=?
                ORDER BY attempt DESC
                LIMIT 1
                """,
                (command_id,),
            ).fetchone()
        if row is None:
            return None
        state, attempt, payload = row
        if state == IN_PROGRESS_STATE:
            return str(state), int(attempt), None
        return str(state), int(attempt), NodeActionResult.model_validate_json(payload)

    def attempt_history(self, command_id: str) -> list[dict[str, Any]]:
        """Every attempt of one command, oldest first: the audit view."""

        with self._lock:
            rows = self._db.execute(
                "SELECT "
                + ", ".join(_AUDIT_COLUMNS)
                + " FROM results WHERE command_id=? ORDER BY attempt",
                (command_id,),
            ).fetchall()
        history = []
        for row in rows:
            entry = dict(zip(_AUDIT_COLUMNS, row))
            entry["command_id"] = command_id
            raw_uuids = entry.get("gpu_uuids")
            entry["gpu_uuids"] = (
                json.loads(raw_uuids) if isinstance(raw_uuids, str) else None
            )
            history.append(entry)
        return history

    def mark_in_progress(
        self,
        command: NodeActionCommand,
        attempt: int,
        *,
        signature: str | None = None,
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
                    command_id, attempt, payload, completed_at, state,
                    operation, started_at, incident_id, workflow_request_id,
                    fencing_token, gpu_uuids, parameters_digest,
                    signature_digest
                ) VALUES (?, ?, ?, NULL, 'IN_PROGRESS', ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(command_id, attempt) DO UPDATE SET
                    payload=excluded.payload,
                    completed_at=NULL,
                    state='IN_PROGRESS',
                    operation=excluded.operation,
                    started_at=excluded.started_at,
                    incident_id=excluded.incident_id,
                    workflow_request_id=excluded.workflow_request_id,
                    fencing_token=excluded.fencing_token,
                    gpu_uuids=excluded.gpu_uuids,
                    parameters_digest=excluded.parameters_digest,
                    signature_digest=excluded.signature_digest
                """,
                (
                    command.command_id,
                    attempt,
                    marker.model_dump_json(),
                    command.operation.value,
                    now,
                    command.incident_id,
                    command.workflow_request_id,
                    command.fencing_token,
                    json.dumps(list(command.gpu_uuids)),
                    canonical_digest(command.parameters),
                    (
                        hashlib.sha256(signature.encode()).hexdigest()
                        if signature
                        else None
                    ),
                ),
            )

    def mark_interrupted(
        self, command_id: str, attempt: int, error: str
    ) -> NodeActionResult | None:
        """Rewrite one IN_PROGRESS row as INTERRUPTED and return the marker.

        This is a plain UPDATE of a row that already exists, not the
        ``INSERT .. ON CONFLICT`` ``save`` performs, so it stays available as
        the last resort when writing the result is exactly what failed.
        Answers ``None`` when the row is gone or already carries a result.
        """

        now = datetime.now(timezone.utc)
        with self._lock:
            row = self._db.execute(
                """
                SELECT payload FROM results
                WHERE command_id=? AND attempt=? AND state=?
                """,
                (command_id, attempt, IN_PROGRESS_STATE),
            ).fetchone()
            if row is None:
                return None
            marker = NodeActionResult.model_validate_json(row[0]).model_copy(
                update={
                    "status": NodeActionStatus.INTERRUPTED,
                    "error": error,
                    "retryable": False,
                    "attempt": attempt,
                    "completed_at": now,
                }
            )
            self._db.execute(
                """
                UPDATE results
                SET payload=?, completed_at=?, state='INTERRUPTED'
                WHERE command_id=? AND attempt=?
                """,
                (marker.model_dump_json(), now.isoformat(), command_id, attempt),
            )
        return marker

    def _interrupt_in_progress(self) -> None:
        with self._lock:
            rows = self._db.execute(
                "SELECT command_id, attempt FROM results WHERE state=?",
                (IN_PROGRESS_STATE,),
            ).fetchall()
            for command_id, attempt in rows:
                self.mark_interrupted(
                    str(command_id), int(attempt), RESTART_INTERRUPTED_ERROR
                )

    def save(self, result: NodeActionResult, *, exit_code: int | None = None) -> None:
        """Record one attempt's result; audit columns set at start are kept."""

        with self._lock:
            self._db.execute(
                """
                INSERT INTO results(
                    command_id, attempt, payload, completed_at, state,
                    operation, exit_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(command_id, attempt) DO UPDATE SET
                    payload=excluded.payload,
                    completed_at=excluded.completed_at,
                    state=excluded.state,
                    operation=excluded.operation,
                    exit_code=COALESCE(excluded.exit_code, results.exit_code)
                """,
                (
                    result.command_id,
                    result.attempt,
                    result.model_dump_json(),
                    result.completed_at.isoformat(),
                    result.status.value,
                    result.operation.value,
                    exit_code,
                ),
            )
        self._maybe_cleanup()

    def delete(self, command_id: str) -> None:
        with self._lock:
            self._db.execute(
                "DELETE FROM results WHERE command_id=?",
                (command_id,),
            )

    def probe_writable(self) -> None:
        """Raise ``sqlite3.Error`` unless a write commits right now.

        ``/healthz`` calls this: a read-only or full filesystem under the
        ledger means every accepted command will fail at ``mark_in_progress``,
        which is worth a 503 before the first command arrives.
        """

        with self._lock:
            self._db.execute(
                """
                INSERT INTO ledger_health(id, probed_at) VALUES (1, ?)
                ON CONFLICT(id) DO UPDATE SET probed_at=excluded.probed_at
                """,
                (datetime.now(timezone.utc).isoformat(),),
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()

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
                WHERE (command_id, attempt) IN (
                    SELECT command_id, attempt
                    FROM results
                    WHERE state != 'IN_PROGRESS'
                    ORDER BY completed_at DESC, command_id DESC, attempt DESC
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
        removed = {"results": results + overflow, "fencing": fencing}
        LOGGER.info(
            "node action ledger retention purge results=%d fencing=%d "
            "retention_seconds=%d max_results=%d",
            removed["results"],
            removed["fencing"],
            self.retention_seconds,
            self.max_results,
        )
        return removed

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
