"""The health-signal retention's two primitives on SQLite.

Every claim runs under ``_state_transaction`` -- the process RLock and a
``BEGIN IMMEDIATE`` write transaction -- so the guarded delete below, inside
one such transaction, can never interleave with a claim's read and write.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from threading import RLock

from pydantic import BaseModel

from gpu_fault.store.shared.signal_retention import (
    SignalPage,
    SignalRow,
    decoded_signal_rows,
)


class SqliteSignalRetentionMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: sqlite3.Connection
    _lock: RLock
    _models: dict[str, type[BaseModel]]
    _state_transaction: Callable[[str], AbstractContextManager[None]]

    def _signal_state_page(
        self,
        kind: str,
        *,
        after_key: str,
        limit: int,
        stale_before: datetime,
    ) -> SignalPage:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT key, payload FROM objects
                WHERE kind=? AND key>?
                ORDER BY key
                LIMIT ?
                """,
                (kind, after_key, limit),
            ).fetchall()
        stale, undecodable = decoded_signal_rows(
            kind,
            rows,
            lambda name, payload: self._models[name].model_validate_json(payload),
            stale_before,
        )
        return SignalPage(
            rows=stale,
            scanned=len(rows),
            last_key=rows[-1][0] if rows else None,
            undecodable=undecodable,
        )

    def _delete_unchanged_signal_states(
        self, kind: str, victims: Sequence[SignalRow]
    ) -> list[str]:
        """Delete each row whose stored payload is still the one judged."""

        deleted: list[str] = []
        if not victims:
            return deleted
        with self._state_transaction(f"{kind}/retention"):
            for row in victims:
                cursor = self._db.execute(
                    "DELETE FROM objects WHERE kind=? AND key=? AND payload=?",
                    (kind, row.key, row.token),
                )
                if cursor.rowcount == 1:
                    deleted.append(row.key)
        return deleted
