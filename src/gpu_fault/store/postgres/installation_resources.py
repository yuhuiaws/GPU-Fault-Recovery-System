"""The installation resource registry on PostgreSQL: one write, one transaction.

``/v1/installation-resources/sync`` used to save its snapshot row by row, each
in its own transaction, so a deterministic identity conflict surfaced only at
its row, after every row ahead of it had been rewritten -- and the processor
replayed the request, rewriting them again, until its retry horizon (live
2026-10-04..13). A write now takes the advisory lock of every row it names, in
key order and with the key text ``save_installation_resource`` always used, so a
batch and a single-row writer of the same row exclude each other; it reads all
recorded rows, refuses an identity change before writing anything, and commits
all rows or none.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Protocol, Sequence, cast

from gpu_fault.installation_resources import (
    InstallationResource,
    batch_keys,
    refuse_identity_changes,
)
from gpu_fault.store.postgres.core import take_advisory_xact_locks
from gpu_fault.store.shared.errors import NotFoundError


class _Database(Protocol):
    """The two ``PooledPostgresDatabase`` entry points this mixin uses."""

    def cursor(self) -> AbstractContextManager[Any]: ...

    def transaction(self) -> AbstractContextManager[None]: ...


def _lock_key(key: str) -> str:
    return f"installation_resource/{key}"


class PostgresInstallationResourceMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: _Database
    _decode: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _put: Callable[..., Any]

    def _recorded_installation_resources(
        self, cursor: Any, keys: Sequence[str]
    ) -> dict[str, InstallationResource]:
        cursor.execute(
            """
            SELECT key, payload FROM gpu_fault_objects
            WHERE kind='installation_resource' AND key = ANY(%s)
            """,
            (list(keys),),
        )
        return {
            row[0]: self._decode("installation_resource", row[1])
            for row in cursor.fetchall()
        }

    def save_installation_resource(
        self,
        resource: InstallationResource,
    ) -> InstallationResource:
        return self.save_installation_resources([resource])[0]

    def save_installation_resources(
        self,
        resources: Sequence[InstallationResource],
    ) -> list[InstallationResource]:
        keys = batch_keys(resources)
        with self._db.transaction():
            with self._db.cursor() as cursor:
                take_advisory_xact_locks(cursor, (_lock_key(key) for key in keys))
                recorded = self._recorded_installation_resources(cursor, keys)
            refuse_identity_changes(recorded, resources)
            for key, resource in zip(keys, resources):
                self._put("installation_resource", key, resource)
        return list(resources)

    def get_installation_resource(
        self,
        site_id: str,
        resource_key: str,
    ) -> InstallationResource:
        resource = self._get_optional(
            "installation_resource",
            f"{site_id}/{resource_key}",
        )
        if resource is None:
            raise NotFoundError(resource_key)
        return cast(InstallationResource, resource)

    def list_installation_resources(
        self,
        site_id: str | None = None,
    ) -> list[InstallationResource]:
        clauses = ["kind='installation_resource'"]
        parameters: list[object] = []
        if site_id is not None:
            clauses.append("payload->>'site_id'=%s")
            parameters.append(site_id)
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT payload FROM gpu_fault_objects WHERE "
                + " AND ".join(clauses)
                + " ORDER BY payload->>'site_id', payload->>'resource_key'",
                parameters,
            )
            rows = cursor.fetchall()
        return [self._decode("installation_resource", row[0]) for row in rows]
