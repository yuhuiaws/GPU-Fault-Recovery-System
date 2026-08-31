from __future__ import annotations

import os
from contextlib import nullcontext
from threading import Condition

from gpu_fault.models import (
    EfaTrafficAdminDecision,
    EfaTrafficState,
)
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
from gpu_fault.policy import Nvlink74BitOccurrenceState
from gpu_fault.store.memory.store import InMemoryStore
from gpu_fault.store.sqlite.store import SqliteStore

_PooledPostgresDatabase = PooledPostgresDatabase
POSTGRES_SCHEMA_VERSION = LATEST_POSTGRES_SCHEMA_VERSION


class PostgresStore(
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
    SqliteStore,
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
        InMemoryStore.__init__(self)
        # The in-memory and SQLite stores guard their dict/file writes
        # with one process-wide RLock, and 19 small writers inherit it
        # unchanged - save_workflow, save_incident, save_agent and the
        # rest. On PostgreSQL that lock cannot mean anything: three
        # replicas times four uvicorn workers already run these paths
        # concurrently, so anything that needs mutual exclusion uses
        # _state_transaction's advisory lock instead. All it did here was
        # funnel all 32 store I/O threads of a process through one
        # RLock. Cross-replica atomicity comes from the transaction; the
        # multi-statement writers below take one explicitly.
        self._lock = nullcontext()
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
        from gpu_fault.gpu_metrics import (
            GpuFindingState,
            GpuHealthFinding,
            GpuInventorySnapshot,
            GpuMetricLatest,
            GpuMetricsIngestionResult,
        )
        from gpu_fault.fleet import (
            AgentRecord,
            FleetDeployment,
            MultiNodeBarrier,
        )
        from gpu_fault.telemetry import (
            CollectorMetricsSnapshotRecord,
            CollectorStatus,
            RawEvidenceRecord,
            TelemetryMetricLatest,
            WorkloadObservationState,
        )
        from gpu_fault.training_health import (
            TrainingProgressState,
        )
        from gpu_fault.managed_recovery import (
            HyperPodNodeIdentity,
        )
        from gpu_fault.hyperpod import (
            HyperPodSubmissionRecord,
        )
        from gpu_fault import regional as regional_models
        from gpu_fault.installation_resources import InstallationResource
        from gpu_fault.processor import (
            PeriodicTaskLease,
            ProcessorLaneLease,
            ProcessorLeadership,
            ProcessorRequest,
        )
        from gpu_fault.policy import (
            FaultPolicyDecision,
            XidCorrelationRecord,
            XidEvent,
        )

        self._models = {
            **self._MODELS,
            "agent": AgentRecord,
            "fleet_deployment": FleetDeployment,
            "barrier": MultiNodeBarrier,
            "gpu_metric_latest": GpuMetricLatest,
            "gpu_inventory_snapshot": GpuInventorySnapshot,
            "gpu_finding_state": GpuFindingState,
            "gpu_finding_history": GpuHealthFinding,
            "gpu_metrics_batch": GpuMetricsIngestionResult,
            "collector_status": CollectorStatus,
            "collector_metrics_snapshot": CollectorMetricsSnapshotRecord,
            "telemetry_metric_latest": TelemetryMetricLatest,
            "attempt_observation": WorkloadObservationState,
            "training_progress": TrainingProgressState,
            "raw_evidence": RawEvidenceRecord,
            "hyperpod_node_identity": HyperPodNodeIdentity,
            "hyperpod_submission": HyperPodSubmissionRecord,
            "regional_cluster": regional_models.RegionalClusterRegistration,
            "regional_registry_head": regional_models.RegionalRegistryHead,
            "regional_registry_member": regional_models.RegionalRegistryMember,
            "regional_registry_revision": regional_models.RegionalRegistryRevision,
            "installation_resource": InstallationResource,
            "remote_command": regional_models.RemoteActionCommand,
            "processor_leadership": ProcessorLeadership,
            "periodic_task_lease": PeriodicTaskLease,
            "processor_lane": ProcessorLaneLease,
            "processor_request": ProcessorRequest,
            "xid_correlation_event": XidEvent,
            "xid_policy_decision": FaultPolicyDecision,
            "xid_correlation": XidCorrelationRecord,
            "xid74_occurrence_state": Nvlink74BitOccurrenceState,
            "efa_traffic_state": EfaTrafficState,
            "efa_traffic_admin_decision": EfaTrafficAdminDecision,
        }
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
        self._attempt_observation_condition = Condition()
        self._attempt_observation_queue = []
        self._legacy_reconciled_at = 0.0
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
