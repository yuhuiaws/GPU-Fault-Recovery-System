"""The key/value primitives every ``Shared*Mixin`` template is written over.

The SQLite and PostgreSQL stores keep their records in one ``(kind, key) ->
JSON payload`` table plus one ``(kind, key) -> value`` link table. Everything
a store does that is not a query-planner-specific statement -- fetch one
record, upsert one record, run a few of those under one lock -- is the same
on both, and lives in the ``gpu_fault.store.shared`` mixins. Those mixins are
written against this Protocol and nothing else of the dialect.

A shared mixin declares the primitives it uses as class-level annotations
under the ``# Attributes supplied by the composed concrete implementation.``
convention, spelled with the aliases below (``_put: PutRecord``). mypy then
checks each dialect's definition against the alias where the concrete store
composes the two, and the ``TYPE_CHECKING`` assertion at the bottom of each
store module checks the store as a whole against :class:`StorePrimitives`.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from typing import Any, Callable, Iterable, Protocol

from pydantic import BaseModel

from gpu_fault.store.shared.errors import NotFoundError

# `_get`, `_get_optional` and `_list` return `Any` on purpose: the model class
# is looked up in `_models` at run time from a string kind, so the row type is
# only knowable at the call site, which states it with a `cast`.
PutRecord = Callable[[str, str, BaseModel], None]
DeleteRecord = Callable[[str, str], None]
GetRecord = Callable[[str, str], Any]
GetOptionalRecord = Callable[[str, str], Any]
ListRecords = Callable[[str], list[Any]]
LinkRecord = Callable[[str, str, str], None]
GetLink = Callable[[str, str], str | None]
StateKey = Callable[[Iterable[Any]], str]
StateTransaction = Callable[[str], AbstractContextManager[None]]
StatementGuard = Callable[[], AbstractContextManager[object]]


class StorePrimitives(Protocol):
    """What a dialect supplies so the shared templates can run on it."""

    _models: dict[str, type[BaseModel]]

    def _put(self, kind: str, key: str, value: BaseModel) -> None:
        """Upsert one record."""

    def _delete(self, kind: str, key: str) -> None:
        """Delete one record; a miss is not an error."""

    def _get(self, kind: str, key: str) -> Any:
        """Fetch one record or raise :class:`NotFoundError`."""

    def _get_optional(self, kind: str, key: str) -> Any:
        """Fetch one record or return ``None``."""

    def _list(self, kind: str) -> list[Any]:
        """Every record of one kind, in no particular order."""

    def _link(self, kind: str, key: str, value: str) -> None:
        """Upsert one ``key -> value`` link."""

    def _get_link(self, kind: str, key: str) -> str | None:
        """Read one link or return ``None``."""

    def _state_key(self, parts: Iterable[Any]) -> str:
        """Canonical text key for a composite identity.

        A plain member rather than ``@staticmethod`` on purpose: the dialect
        mixins declare it under the annotation convention, so the concrete
        store's first definition in the MRO is a ``Callable`` attribute, and
        only a method member accepts one.
        """

    def _state_transaction(self, lock_key: str) -> AbstractContextManager[None]:
        """One write transaction, serialised on ``lock_key`` across writers.

        This is the mutual exclusion the multi-statement templates rely on:
        a ``BEGIN IMMEDIATE`` under the process lock on SQLite, an advisory
        transaction lock on PostgreSQL.
        """

    def _statement_guard(self) -> AbstractContextManager[object]:
        """Guard around one statement that runs outside a transaction.

        SQLite hands one connection to every thread of the process, so the
        guard is its ``RLock``. PostgreSQL gives each statement its own pooled
        connection and serialises writers in the database, so its guard is a
        deliberate no-op: a process-level lock there would only funnel every
        store thread through one RLock without protecting anything.
        """


def state_key(parts: Iterable[Any]) -> str:
    """The canonical composite key both key/value dialects store under."""

    return json.dumps(
        list(parts),
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )


class SharedRecordAccessMixin:
    """The two primitives that derive from ``_get`` the same way everywhere."""

    # Attributes supplied by the composed concrete implementation.
    _get: GetRecord

    def _get_optional(self, kind: str, key: str) -> Any:
        try:
            return self._get(kind, key)
        except NotFoundError:
            return None

    @staticmethod
    def _state_key(parts: Iterable[Any]) -> str:
        return state_key(parts)
