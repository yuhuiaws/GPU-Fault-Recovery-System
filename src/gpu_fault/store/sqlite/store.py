from __future__ import annotations

import sqlite3

from gpu_fault.models import (
    AdvisoryNotification,
    CompletionDecision,
    DiagnosticRequest,
    EfaTrafficAdminDecision,
    EfaTrafficState,
    EffectiveRuntimeProfile,
    FaultIncident,
    HealthSignalState,
    NodeMarker,
    NotificationDelivery,
    NotificationDispatchWatermark,
    NotificationResult,
    RecoveryPlan,
    RestartBudgetState,
    TerminalEvent,
    TriageReport,
    WorkflowRequest,
    XidMetricBaseline,
)
from gpu_fault.store.shared.errors import (
    EfaTrafficAdminConflict as EfaTrafficAdminConflict,
    NotFoundError as NotFoundError,
    WorkflowLeaseError as WorkflowLeaseError,
)
from gpu_fault.store.sqlite.control_records import SqliteControlRecordMixin
from gpu_fault.store.sqlite.core import SqliteCoreMixin
from gpu_fault.store.sqlite.efa import SqliteEfaTrafficMixin
from gpu_fault.store.sqlite.fleet import SqliteFleetMixin
from gpu_fault.store.sqlite.notifications import SqliteNotificationMixin
from gpu_fault.store.sqlite.processor_leases import SqliteProcessorLeaseMixin
from gpu_fault.store.sqlite.processor_queue import SqliteProcessorQueueMixin
from gpu_fault.store.sqlite.remote_commands import SqliteRemoteCommandMixin
from gpu_fault.store.sqlite.telemetry import SqliteTelemetryMixin
from gpu_fault.store.sqlite.workflows import SqliteWorkflowMixin
from gpu_fault.store.sqlite.xid import SqliteXidMixin
from gpu_fault.store.shared.transactional_workflows import TransactionalWorkflowMixin
from gpu_fault.policy import Nvlink74BitOccurrenceState
from gpu_fault.store.memory.store import InMemoryStore


class SqliteStore(
    SqliteCoreMixin,
    SqliteControlRecordMixin,
    SqliteEfaTrafficMixin,
    SqliteFleetMixin,
    SqliteNotificationMixin,
    SqliteRemoteCommandMixin,
    TransactionalWorkflowMixin,
    SqliteWorkflowMixin,
    SqliteXidMixin,
    SqliteTelemetryMixin,
    SqliteProcessorQueueMixin,
    SqliteProcessorLeaseMixin,
    InMemoryStore,
):
    """Durable single-writer store for active executor deployments.

    SQLite is suitable for one control-plane replica and canary use. A
    multi-replica production deployment should implement the same contract
    with PostgreSQL or DynamoDB conditional writes.
    """

    _MODELS = {
        "event": TerminalEvent,
        "decision": CompletionDecision,
        "marker": NodeMarker,
        "diagnostic": DiagnosticRequest,
        "triage": TriageReport,
        "plan": RecoveryPlan,
        "profile": EffectiveRuntimeProfile,
        "incident": FaultIncident,
        "workflow": WorkflowRequest,
        "notification": AdvisoryNotification,
        "notification_delivery": NotificationDelivery,
        "notification_result": NotificationResult,
        "notification_watermark": NotificationDispatchWatermark,
        "restart_budget": RestartBudgetState,
        "xid_metric_baseline": XidMetricBaseline,
        "health_signal_state": HealthSignalState,
        "efa_traffic_state": EfaTrafficState,
        "efa_traffic_admin_decision": EfaTrafficAdminDecision,
    }

    def __init__(self, path: str) -> None:
        super().__init__()
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
        from gpu_fault.regional import (
            RegionalClusterRegistration,
            RemoteActionCommand,
        )
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
            "regional_cluster": RegionalClusterRegistration,
            "installation_resource": InstallationResource,
            "remote_command": RemoteActionCommand,
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
        self.path = path
        self._db = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
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
