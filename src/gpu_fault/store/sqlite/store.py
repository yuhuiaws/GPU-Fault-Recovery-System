from __future__ import annotations

import os
import sqlite3
from threading import RLock
from typing import TYPE_CHECKING

from gpu_fault.store.memory.telemetry_spool import MemoryTelemetrySpoolMixin
from gpu_fault.store.shared.compositions import SharedCompositionMixin
from gpu_fault.store.shared.control_records import SharedControlRecordMixin
from gpu_fault.store.shared.efa import (
    SharedEfaTrafficMixin,
    SharedEfaTrafficRulesMixin,
)
from gpu_fault.store.shared.errors import (
    EfaTrafficAdminConflict as EfaTrafficAdminConflict,
)
from gpu_fault.store.shared.errors import (
    NotFoundError as NotFoundError,
)
from gpu_fault.store.shared.errors import (
    WorkflowLeaseError as WorkflowLeaseError,
)
from gpu_fault.store.shared.fleet import SharedFleetMixin
from gpu_fault.store.shared.notifications import SharedNotificationMixin
from gpu_fault.store.shared.primitives import (
    SharedRecordAccessMixin,
    StorePrimitives,
)
from gpu_fault.store.shared.processor_leases import SharedProcessorLeaseMixin
from gpu_fault.store.shared.record_models import record_models
from gpu_fault.store.shared.remote_commands import SharedRemoteCommandMixin
from gpu_fault.store.shared.telemetry_records import SharedTelemetryRecordMixin
from gpu_fault.store.shared.transactional_workflows import TransactionalWorkflowMixin
from gpu_fault.store.shared.workflow_records import SharedWorkflowRecordMixin
from gpu_fault.store.shared.xid import SharedXidMixin, SharedXidSignalMixin
from gpu_fault.store.sqlite.control_records import SqliteControlRecordMixin
from gpu_fault.store.sqlite.core import SqliteCoreMixin
from gpu_fault.store.sqlite.fleet import SqliteFleetMixin
from gpu_fault.store.sqlite.notifications import SqliteNotificationMixin
from gpu_fault.store.sqlite.processor_leases import SqliteProcessorLeaseMixin
from gpu_fault.store.sqlite.processor_queue import SqliteProcessorQueueMixin
from gpu_fault.store.sqlite.remote_commands import SqliteRemoteCommandMixin
from gpu_fault.store.sqlite.telemetry import SqliteTelemetryMixin
from gpu_fault.store.sqlite.workflows import SqliteWorkflowMixin
from gpu_fault.store.sqlite.xid import SqliteXidMixin


class SqliteStore(
    # SQLite-specific statements first: they override the shared templates.
    SqliteCoreMixin,
    SqliteControlRecordMixin,
    SqliteFleetMixin,
    SqliteNotificationMixin,
    SqliteRemoteCommandMixin,
    TransactionalWorkflowMixin,
    SqliteWorkflowMixin,
    SqliteXidMixin,
    SqliteTelemetryMixin,
    SqliteProcessorQueueMixin,
    SqliteProcessorLeaseMixin,
    # Dialect-neutral templates over the key/value primitives.
    SharedRecordAccessMixin,
    SharedControlRecordMixin,
    SharedEfaTrafficMixin,
    SharedFleetMixin,
    SharedNotificationMixin,
    SharedRemoteCommandMixin,
    SharedWorkflowRecordMixin,
    SharedXidMixin,
    SharedTelemetryRecordMixin,
    SharedProcessorLeaseMixin,
    # Pure rules and public-contract compositions every store shares.
    SharedEfaTrafficRulesMixin,
    SharedXidSignalMixin,
    SharedCompositionMixin,
    # The telemetry spool is not durable on SQLite; see the mixin docstring.
    MemoryTelemetrySpoolMixin,
):
    """Durable single-writer store for active executor deployments.

    SQLite is suitable for one control-plane replica and canary use. A
    multi-replica production deployment should implement the same contract
    with PostgreSQL or DynamoDB conditional writes.
    """

    # SQLite state that lives on disk (not an in-memory or shared-cache DB).
    _ON_DISK = staticmethod(
        lambda path: bool(path)
        and path != ":memory:"
        and not path.startswith("file::memory:")
    )

    @staticmethod
    def _restrict_state_dir(path: str) -> None:
        if not SqliteStore._ON_DISK(path):
            return
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, mode=0o700, exist_ok=True)

    @staticmethod
    def _restrict_state_file(path: str) -> None:
        if not SqliteStore._ON_DISK(path):
            return
        for suffix in ("", "-wal", "-shm", "-journal"):
            candidate = f"{path}{suffix}"
            if os.path.exists(candidate):
                os.chmod(candidate, 0o600)

    def __init__(self, path: str) -> None:
        # One connection for the process, so every statement and every
        # transaction runs under this lock (see ``_statement_guard`` and
        # ``_state_transaction``).
        self._lock = RLock()
        self._telemetry_spool = {}
        self._models = record_models()
        self.path = path
        # The state file holds tenant fault data and cluster tokens, so it
        # must never inherit a permissive process umask. Create the parent
        # directory 0700 and the DB file (plus its WAL sidecars) 0600, and
        # do so before any sensitive row is written (security review M-7).
        self._restrict_state_dir(path)
        self._db = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._restrict_state_file(path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        # journal_mode=WAL creates the -wal/-shm sidecars, which carry the
        # same tenant data; tighten them too now that they exist.
        self._restrict_state_file(path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS objects (
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (kind, key)
            )
            """
        )
        self._db.execute(
            """
            UPDATE objects
            SET payload=json_remove(payload, '$.partition_id')
            WHERE kind='processor_request'
              AND json_type(payload, '$.partition_id') IS NOT NULL
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS links (
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (kind, key)
            )
            """
        )
        self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS objects_active_workflow_scope
            ON objects (
                kind,
                json_extract(payload, '$.status'),
                json_extract(payload, '$.incident_id'),
                json_extract(payload, '$.updated_at') DESC
            )
            WHERE kind='workflow'
            """
        )
        self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS objects_incident_scope
            ON objects (
                kind,
                json_extract(payload, '$.cluster_id'),
                json_extract(payload, '$.job_id')
            )
            WHERE kind='incident'
            """
        )
        self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS objects_marker_incident
            ON objects (
                json_extract(payload, '$.incident_id'),
                json_extract(payload, '$.observed_at'),
                key
            )
            WHERE kind='marker'
            """
        )


if TYPE_CHECKING:

    def _assert_primitives(store: SqliteStore) -> None:
        _primitives: StorePrimitives = store
