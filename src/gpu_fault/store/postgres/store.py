from __future__ import annotations

import os
from threading import Condition
from typing import TYPE_CHECKING

from gpu_fault.schema_migrations import (
    LATEST_POSTGRES_SCHEMA_VERSION,
)
from gpu_fault.store.postgres.pool import (
    PooledPostgresDatabase,
    configure_writer_connection,
)
from gpu_fault.store.shared.errors import (
    EfaTrafficAdminConflict as EfaTrafficAdminConflict,
    NotFoundError as NotFoundError,
    WorkflowLeaseError as WorkflowLeaseError,
)
from gpu_fault.store.postgres.collector_telemetry import PostgresCollectorTelemetryMixin
from gpu_fault.store.postgres.core import PostgresCoreMixin
from gpu_fault.store.postgres.control_records import PostgresControlRecordMixin
from gpu_fault.store.postgres.fleet import PostgresFleetMixin
from gpu_fault.store.postgres.gpu_telemetry import PostgresGpuTelemetryMixin
from gpu_fault.store.postgres.notifications import PostgresNotificationMixin
from gpu_fault.store.postgres.processor_admin import PostgresProcessorAdminMixin
from gpu_fault.store.postgres.processor_admission import PostgresProcessorAdmissionMixin
from gpu_fault.store.postgres.processor_claims import PostgresProcessorClaimsMixin
from gpu_fault.store.postgres.processor_completion import (
    PostgresProcessorCompletionMixin,
)
from gpu_fault.store.postgres.processor_completion_runtime import (
    configure_processor_completion_runtime,
)
from gpu_fault.store.postgres.processor_leases import PostgresProcessorLeaseMixin
from gpu_fault.store.postgres.processor_storage import PostgresProcessorStorageMixin
from gpu_fault.store.postgres.remote_commands import PostgresRemoteCommandMixin
from gpu_fault.store.postgres.schema_state import PostgresSchemaMixin
from gpu_fault.store.postgres.telemetry_spool import PostgresTelemetrySpoolMixin
from gpu_fault.store.postgres.workflows import PostgresWorkflowMixin
from gpu_fault.store.postgres.xid import PostgresXidMixin
from gpu_fault.store.shared.compositions import SharedCompositionMixin
from gpu_fault.store.shared.control_records import SharedControlRecordMixin
from gpu_fault.store.shared.efa import (
    SharedEfaTrafficMixin,
    SharedEfaTrafficRulesMixin,
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

_PooledPostgresDatabase = PooledPostgresDatabase
POSTGRES_SCHEMA_VERSION = LATEST_POSTGRES_SCHEMA_VERSION


class PostgresStore(
    # PostgreSQL-specific statements first: they override the shared templates.
    PostgresCoreMixin,
    PostgresSchemaMixin,
    PostgresControlRecordMixin,
    PostgresFleetMixin,
    PostgresNotificationMixin,
    PostgresRemoteCommandMixin,
    PostgresWorkflowMixin,
    PostgresXidMixin,
    PostgresGpuTelemetryMixin,
    PostgresCollectorTelemetryMixin,
    PostgresTelemetrySpoolMixin,
    PostgresProcessorAdminMixin,
    PostgresProcessorAdmissionMixin,
    PostgresProcessorClaimsMixin,
    PostgresProcessorCompletionMixin,
    PostgresProcessorLeaseMixin,
    PostgresProcessorStorageMixin,
    # Dialect-neutral templates over the key/value primitives.
    TransactionalWorkflowMixin,
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
):
    """Shared active-active store with transactional workflow leases."""

    def __init__(
        self,
        url: str,
        *,
        pool_min_size: int = 1,
        pool_max_size: int = 8,
        pool_timeout_seconds: float = 2,
        initialize_schema: bool = True,
        hot_state_mode: str | None = None,
    ) -> None:
        if (
            pool_min_size < 0
            or pool_max_size < 1
            or pool_min_size > pool_max_size
            or pool_timeout_seconds <= 0
        ):
            raise ValueError("invalid PostgreSQL pool configuration")
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError("install gpu-fault-control-plane[postgres]") from exc
        self._models = record_models()
        self.url = url
        self.hot_state_mode = (
            (
                hot_state_mode
                or os.getenv(
                    "GPU_FAULT_POSTGRES_HOT_STATE_MODE",
                    "dedicated",
                )
            )
            .strip()
            .lower()
        )
        if self.hot_state_mode not in {
            "legacy",
            "dual",
            "dedicated",
        }:
            raise ValueError(
                "GPU_FAULT_POSTGRES_HOT_STATE_MODE must be legacy, dual, or dedicated"
            )
        self.processor_queue_state_mode = (
            os.getenv(
                "GPU_FAULT_PROCESSOR_QUEUE_STATE_MODE",
                "dedicated",
            )
            .strip()
            .lower()
        )
        if self.processor_queue_state_mode not in {
            "legacy",
            "dual",
            "dedicated",
        }:
            raise ValueError(
                "GPU_FAULT_PROCESSOR_QUEUE_STATE_MODE must be "
                "legacy, dual, or dedicated"
            )
        # Headroom for rows the claim window later discards: several
        # queued requests can share one lane and collapse to a single
        # candidate, and an interlocked fault drops out entirely. Larger
        # values raise the per-claim yield, smaller ones cap the cost of
        # a claim at high queue depth.
        self.claim_window_multiplier = max(
            1,
            int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_CLAIM_WINDOW_MULTIPLIER",
                    "8",
                )
            ),
        )
        # Server-side backstop for the request deadline: a statement that
        # outlives every client waiting on it still holds its connection,
        # its locks and its share of the ACU budget. Bounding it in the
        # database is the only place that can actually stop the work, so
        # this is set well above any healthy statement and only catches
        # the pathological ones. Bootstrap DDL exempts itself below.
        statement_timeout_seconds = float(
            os.getenv(
                "GPU_FAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS",
                "60",
            )
        )
        connection_kwargs = {"autocommit": True}
        if statement_timeout_seconds > 0:
            connection_kwargs["options"] = (
                "-c statement_timeout="
                f"{int(statement_timeout_seconds * 1000)}"
                " -c idle_in_transaction_session_timeout="
                f"{int(statement_timeout_seconds * 1000)}"
            )
        # The fleet's connection ceiling is per-process, times uvicorn
        # workers, times replicas - 12 ingress processes and 6 workers
        # here - so a burst that grows every pool to max_size and leaves
        # it there parks hundreds of idle backends on the writer, each
        # holding its own work_mem allocation, until the pods restart.
        # max_idle shrinks after bursts; max_lifetime bounds stale
        # connections to the old Aurora writer.
        pool_max_idle_seconds = float(
            os.getenv(
                "GPU_FAULT_POSTGRES_POOL_MAX_IDLE_SECONDS",
                "300",
            )
        )
        pool_max_lifetime_seconds = float(
            os.getenv(
                "GPU_FAULT_POSTGRES_POOL_MAX_LIFETIME_SECONDS",
                "3600",
            )
        )
        pool_kwargs = {}
        if pool_max_idle_seconds > 0:
            pool_kwargs["max_idle"] = pool_max_idle_seconds
        if pool_max_lifetime_seconds > 0:
            pool_kwargs["max_lifetime"] = pool_max_lifetime_seconds
        self._pool = ConnectionPool(
            conninfo=url,
            min_size=pool_min_size,
            max_size=pool_max_size,
            timeout=pool_timeout_seconds,
            kwargs=connection_kwargs,
            configure=configure_writer_connection,
            open=True,
            **pool_kwargs,
        )
        self._db = PooledPostgresDatabase(self._pool)
        configure_processor_completion_runtime(self, pool_max_size)
        self._processor_completion_condition = Condition()
        self._processor_completion_queue = []
        self._processor_completion_leader_active = False
        self._attempt_observation_condition = Condition()
        self._attempt_observation_queue = []
        self._attempt_observation_leader_active = False
        self._processor_counter_mode_cache = "dual"
        self._processor_counter_mode_checked_at = 0.0
        try:
            self._initialize_schema_state(initialize_schema)
        except Exception:
            if self._processor_completion_executor is not None:
                self._processor_completion_executor.shutdown(
                    wait=True,
                    cancel_futures=True,
                )
            self._pool.close()
            raise


if TYPE_CHECKING:

    def _assert_primitives(store: PostgresStore) -> None:
        _primitives: StorePrimitives = store
