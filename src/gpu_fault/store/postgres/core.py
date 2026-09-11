from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Iterator, cast

from pydantic import BaseModel

from gpu_fault.store.shared.errors import (
    NotFoundError,
    StaleWriteError,
    TransactionRequiredError,
)


class PostgresCoreMixin:
    """The PostgreSQL half of :class:`gpu_fault.store.shared.primitives.StorePrimitives`.

    ``_get_optional`` and ``_state_key`` come from ``SharedRecordAccessMixin``.
    """

    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _models: dict[str, type[BaseModel]]

    def _statement_guard(self) -> AbstractContextManager[object]:
        """Deliberately a no-op.

        The shared single-row writers -- save_workflow, save_incident,
        save_agent and the rest -- take this guard around one statement.
        On SQLite it is the process RLock that keeps the shared connection
        single-threaded. Here every statement gets its own pooled
        connection and three replicas times four uvicorn workers already
        run these paths concurrently, so a process-level lock could not
        mean anything: all it would do is funnel every store I/O thread of
        a process through one RLock. Anything that needs mutual exclusion
        uses ``_state_transaction``'s advisory lock instead.
        """

        return nullcontext()

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

    def _put(
        self,
        kind: str,
        key: str,
        value: BaseModel,
        *,
        expected: BaseModel | None = None,
    ) -> None:
        """Write one row; with ``expected`` only if the row still equals it.

        The unconditional form is the historical whole-row upsert. The
        conditional form is the CAS a writer uses when it read the row without
        a lock and must not overwrite a concurrent change (F-J4): the update
        matches on the full JSON payload it read, and a miss -- changed or
        deleted -- raises :class:`StaleWriteError` instead of landing.
        """

        if expected is None:
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
            return
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                UPDATE gpu_fault_objects
                SET payload=%s::jsonb
                WHERE kind=%s AND key=%s AND payload=%s::jsonb
                """,
                (value.model_dump_json(), kind, key, expected.model_dump_json()),
            )
            if cursor.rowcount == 1:
                return
            # The literal payload did not match. A row written before a model
            # field existed decodes with that field's default and re-encodes
            # with it, so its stored JSON never equals the copy the caller
            # read even though nothing moved: 18 BLOCKED records from a week
            # earlier failed every sweep tick this way (2026-09-11). Compare
            # the decoded row with the caller's copy instead, and condition
            # the write on the exact stored payload so a writer that lands
            # between these two statements still makes this one miss.
            cursor.execute(
                """
                SELECT payload::text FROM gpu_fault_objects
                WHERE kind=%s AND key=%s
                """,
                (kind, key),
            )
            row = cursor.fetchone()
            if row is None:
                raise StaleWriteError(f"{kind}/{key} changed since it was read")
            stored = str(row[0])
            if (
                self._decode(kind, stored).model_dump_json()
                != expected.model_dump_json()
            ):
                raise StaleWriteError(f"{kind}/{key} changed since it was read")
            cursor.execute(
                """
                UPDATE gpu_fault_objects
                SET payload=%s::jsonb
                WHERE kind=%s AND key=%s AND payload=%s::jsonb
                """,
                (value.model_dump_json(), kind, key, stored),
            )
            if cursor.rowcount != 1:
                raise StaleWriteError(f"{kind}/{key} changed since it was read")

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
        # The row lock lives as long as the transaction. Outside one, the
        # autocommit pool releases it before the caller sees the row (F-J4).
        if not getattr(self._db, "in_transaction", True):
            raise TransactionRequiredError(
                f"_get_for_update({kind}/{key}) requires an enclosing transaction"
            )
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
