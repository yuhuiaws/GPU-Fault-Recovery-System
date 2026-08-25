"""Static conformance assertions for the store Protocol.

Runtime contract tests can compare method names and signatures, but only
mypy can check the complete structural relationship between each concrete
store and ``ControlPlaneStore``. These assignments are never executed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpu_fault.store.contracts import ControlPlaneStore
    from gpu_fault.store.memory.store import InMemoryStore
    from gpu_fault.store.postgres.store import PostgresStore
    from gpu_fault.store.sqlite.store import SqliteStore

    def _assert_store_conformance(
        memory: InMemoryStore,
        sqlite: SqliteStore,
        postgres: PostgresStore,
    ) -> None:
        _memory: ControlPlaneStore = memory
        _sqlite: ControlPlaneStore = sqlite
        _postgres: ControlPlaneStore = postgres
