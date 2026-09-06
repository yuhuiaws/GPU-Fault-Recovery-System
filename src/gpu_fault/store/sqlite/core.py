from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from pydantic import BaseModel

from gpu_fault.store.shared.errors import NotFoundError


class SqliteCoreMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _lock: Any
    _models: dict[str, type[BaseModel]]

    # `_get`, `_get_optional` and `_list` return `Any` on purpose: the model
    # class is looked up in `_models` at run time from a string kind, so the
    # result type is only knowable at the call site. Callers state it with a
    # `cast`, which is what makes the mixins above them checkable.

    def close(self) -> None:
        self._db.close()

    def _put(self, kind: str, key: str, value: BaseModel) -> None:
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

    def _get(self, kind: str, key: str) -> Any:
        row = self._db.execute(
            "SELECT payload FROM objects WHERE kind=? AND key=?",
            (kind, key),
        ).fetchone()
        if row is None:
            raise NotFoundError(key)
        return self._models[kind].model_validate_json(row[0])

    def _get_optional(self, kind: str, key: str) -> Any:
        try:
            return self._get(kind, key)
        except NotFoundError:
            return None

    def _list(self, kind: str) -> list[Any]:
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
    def _state_key(parts: Iterable[Any]) -> str:
        return json.dumps(
            list(parts),
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        )

    # How many ``_state_transaction`` entries are nested inside the open
    # transaction; names the savepoint each nested entry opens.
    _savepoint_depth: int = 0

    @contextmanager
    def _state_transaction(self, _lock_key: str) -> Iterator[None]:
        """One write transaction, re-entrant within the process lock.

        The outermost entry is a ``BEGIN IMMEDIATE`` transaction. An entry made
        while one is open (``collector_ingestion_transaction`` or
        ``completion_transaction`` around the store's own writes) is a savepoint
        of it: it commits or rolls back with the outer transaction, and its own
        failure undoes only its writes so the caller can catch and go on. The
        lock is an ``RLock`` and every transaction is opened under it, so the
        shared connection never sees two threads' transactions interleave.
        """

        with self._lock:
            if self._db.in_transaction:
                name = f"gpu_fault_sp_{self._savepoint_depth}"
                self._savepoint_depth += 1
                self._db.execute(f"SAVEPOINT {name}")
                try:
                    yield
                    self._db.execute(f"RELEASE SAVEPOINT {name}")
                except Exception:
                    self._db.execute(f"ROLLBACK TO SAVEPOINT {name}")
                    self._db.execute(f"RELEASE SAVEPOINT {name}")
                    raise
                finally:
                    self._savepoint_depth -= 1
                return
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
