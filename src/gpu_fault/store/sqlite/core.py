from __future__ import annotations

from typing import Any

import json
from contextlib import contextmanager

from gpu_fault.store.shared.errors import NotFoundError


class SqliteCoreMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _lock: Any
    _models: Any

    def close(self) -> None:
        self._db.close()

    def _put(self, kind: str, key: str, value) -> None:
        self._db.execute(
            """
            INSERT INTO objects(kind, key, payload) VALUES (?, ?, ?)
            ON CONFLICT(kind, key) DO UPDATE SET payload=excluded.payload
            """,
            (kind, key, value.model_dump_json()),
        )

    def _delete(self, kind: str, key: str) -> None:
        self._db.execute(
            "DELETE FROM objects WHERE kind=? AND key=?",
            (kind, key),
        )

    def _get(self, kind: str, key: str):
        row = self._db.execute(
            "SELECT payload FROM objects WHERE kind=? AND key=?",
            (kind, key),
        ).fetchone()
        if row is None:
            raise NotFoundError(key)
        return self._models[kind].model_validate_json(row[0])

    def _get_optional(self, kind: str, key: str):
        try:
            return self._get(kind, key)
        except NotFoundError:
            return None

    def _list(self, kind: str):
        rows = self._db.execute(
            "SELECT payload FROM objects WHERE kind=?",
            (kind,),
        ).fetchall()
        model = self._models[kind]
        return [model.model_validate_json(row[0]) for row in rows]

    def _link(self, kind: str, key: str, value: str) -> None:
        self._db.execute(
            """
            INSERT INTO links(kind, key, value) VALUES (?, ?, ?)
            ON CONFLICT(kind, key) DO UPDATE SET value=excluded.value
            """,
            (kind, key, value),
        )

    def _get_link(self, kind: str, key: str) -> str | None:
        row = self._db.execute(
            "SELECT value FROM links WHERE kind=? AND key=?",
            (kind, key),
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _state_key(parts) -> str:
        return json.dumps(
            list(parts),
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        )

    @contextmanager
    def _state_transaction(self, _lock_key: str):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
