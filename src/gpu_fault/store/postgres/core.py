from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, cast

from pydantic import BaseModel

from gpu_fault.store.shared.errors import NotFoundError


class PostgresCoreMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _models: dict[str, type[BaseModel]]

    # `_decode`, `_get`, `_get_for_update` and `_list` return `Any` on purpose:
    # the model class is looked up in `_models` at run time from a string kind,
    # so the row type is only knowable at the call site. Callers state it with a
    # `cast`, which is what makes the mixins above them checkable.

    def close(self) -> None:
        executor = getattr(self, "_processor_completion_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        self._db.close()

    def pool_metrics(self) -> dict[str, Any]:
        return cast("dict[str, Any]", self._db.metrics_snapshot())

    def _put(self, kind: str, key: str, value: BaseModel) -> None:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO gpu_fault_objects(kind, key, payload)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT(kind, key)
                DO UPDATE SET payload=excluded.payload
                """,
                (kind, key, value.model_dump_json()),
            )

    def _delete(self, kind: str, key: str) -> None:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM gpu_fault_objects
                WHERE kind=%s AND key=%s
                """,
                (kind, key),
            )

    def _decode(self, kind: str, payload: Any) -> Any:
        if isinstance(payload, str):
            return self._models[kind].model_validate_json(payload)
        return self._models[kind].model_validate(payload)

    def _get(self, kind: str, key: str) -> Any:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload FROM gpu_fault_objects
                WHERE kind=%s AND key=%s
                """,
                (kind, key),
            )
            row = cursor.fetchone()
        if row is None:
            raise NotFoundError(key)
        return self._decode(kind, row[0])

    def _get_for_update(self, kind: str, key: str) -> Any:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload FROM gpu_fault_objects
                WHERE kind=%s AND key=%s
                FOR UPDATE
                """,
                (kind, key),
            )
            row = cursor.fetchone()
        if row is None:
            raise NotFoundError(key)
        return self._decode(kind, row[0])

    def _list(self, kind: str) -> list[Any]:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload FROM gpu_fault_objects
                WHERE kind=%s
                """,
                (kind,),
            )
            rows = cursor.fetchall()
        return [self._decode(kind, row[0]) for row in rows]

    def _link(self, kind: str, key: str, value: str) -> None:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO gpu_fault_links(kind, key, value)
                VALUES (%s, %s, %s)
                ON CONFLICT(kind, key)
                DO UPDATE SET value=excluded.value
                """,
                (kind, key, value),
            )

    def _get_link(self, kind: str, key: str) -> str | None:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT value FROM gpu_fault_links
                WHERE kind=%s AND key=%s
                """,
                (kind, key),
            )
            row = cursor.fetchone()
        return row[0] if row else None

    @contextmanager
    def _state_transaction(self, lock_key: str) -> Iterator[None]:
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                        hashtextextended(%s, 0)
                    )
                    """,
                    (lock_key,),
                )
            yield
