from __future__ import annotations

from threading import RLock

from gpu_fault.installation_resources import InstallationResource
from gpu_fault.models import (
    AdvisoryNotification,
    CompletionDecision,
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
    WorkflowRequest,
    XidMetricBaseline,
)
from gpu_fault.store.memory.control_records import MemoryControlRecordMixin
from gpu_fault.store.memory.efa import MemoryEfaTrafficMixin
from gpu_fault.store.memory.fleet import MemoryFleetMixin
from gpu_fault.store.memory.notifications import MemoryNotificationMixin
from gpu_fault.store.memory.processor_leases import MemoryProcessorLeaseMixin
from gpu_fault.store.memory.processor_queue import MemoryProcessorQueueMixin
from gpu_fault.store.memory.remote_commands import MemoryRemoteCommandMixin
from gpu_fault.store.memory.telemetry import MemoryTelemetryMixin
from gpu_fault.store.memory.telemetry_spool import MemoryTelemetrySpoolMixin
from gpu_fault.store.memory.workflows import MemoryWorkflowMixin
from gpu_fault.store.memory.xid import MemoryXidMixin
from gpu_fault.store.shared.compositions import SharedCompositionMixin
from gpu_fault.store.shared.efa import SharedEfaTrafficRulesMixin
from gpu_fault.store.shared.errors import (
    EfaTrafficAdminConflict as EfaTrafficAdminConflict,
)
from gpu_fault.store.shared.errors import (
    NotFoundError as NotFoundError,
)
from gpu_fault.store.shared.errors import (
    WorkflowLeaseError as WorkflowLeaseError,
)
from gpu_fault.store.shared.xid import SharedXidSignalMixin


class InMemoryStore(
    # Dict-backed implementations first: they override the shared compositions.
    MemoryControlRecordMixin,
    MemoryEfaTrafficMixin,
    MemoryNotificationMixin,
    MemoryRemoteCommandMixin,
    MemoryWorkflowMixin,
    MemoryXidMixin,
    MemoryTelemetryMixin,
    MemoryTelemetrySpoolMixin,
    MemoryProcessorQueueMixin,
    MemoryProcessorLeaseMixin,
    MemoryFleetMixin,
    # Pure rules and public-contract compositions every store shares.
    SharedEfaTrafficRulesMixin,
    SharedXidSignalMixin,
    SharedCompositionMixin,
):
    """Thread-safe store for tests and single-process tooling.

    Everything lives in dicts guarded by one RLock and is lost with the
    process. ``SqliteStore`` and ``PostgresStore`` are its peers behind the
    same ``ControlPlaneStore`` contract, not subclasses: production runs on
    ``PostgresStore``, which shares code with this store only through the
    ``gpu_fault.store.shared`` mixins composed above.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._events: dict[str, TerminalEvent] = {}
        self._decisions: dict[str, CompletionDecision] = {}
        self._attempt_event_keys: dict[tuple[str, str], str] = {}
        self._markers: dict[str, NodeMarker] = {}
        self._plans: dict[str, RecoveryPlan] = {}
        self._profiles: dict[str, EffectiveRuntimeProfile] = {}
        self._installation_resources: dict[str, InstallationResource] = {}
        self._incidents: dict[str, FaultIncident] = {}
        self._incident_by_event: dict[str, str] = {}
        self._workflows: dict[str, WorkflowRequest] = {}
        self._notifications: dict[str, AdvisoryNotification] = {}
        self._collector_metrics_snapshot = None
        self._notification_by_deduplication_key: dict[str, str] = {}
        self._notification_results: dict[str, NotificationResult] = {}
        self._notification_deliveries: dict[str, NotificationDelivery] = {}
        self._notification_watermarks: dict[str, NotificationDispatchWatermark] = {}
        self._agents = {}
        self._fleet_deployments = {}
        self._barriers = {}
        self._xid_metric_baselines: dict[tuple[str, str, str], XidMetricBaseline] = {}
        self._health_signal_states: dict[str, HealthSignalState] = {}
        self._efa_traffic_states: dict[str, EfaTrafficState] = {}
        self._efa_traffic_admin_decisions: dict[str, EfaTrafficAdminDecision] = {}
        self._gpu_metric_latest = {}
        self._gpu_inventory_snapshots = {}
        self._gpu_finding_states = {}
        self._gpu_finding_history = {}
        self._gpu_metrics_batches = {}
        self._collector_statuses = {}
        self._telemetry_metric_latest = {}
        self._attempt_observations = {}
        self._workload_coverage = {}
        self._training_progress = {}
        self._raw_evidence = {}
        self._hyperpod_node_identities = {}
        self._hyperpod_submissions = {}
        self._regional_clusters = {}
        self._regional_registry_revisions = {}
        self._regional_registry_head = None
        self._regional_registry_members = {}
        self._remote_commands = {}
        self._processor_leadership = None
        self._periodic_task_leases = {}
        self._processor_lanes = {}
        self._processor_members = {}
        self._processor_requests = {}
        self._telemetry_spool: dict[str, dict] = {}
        self._restart_budgets: dict[tuple[str, str], RestartBudgetState] = {}
        self._replacement_fault_groups: dict[str, str] = {}
        self._sxid_fault_groups: dict[str, str] = {}
        self._xid_correlation_events = {}
        self._xid_policy_decisions = {}
        self._xid_correlations = {}
        self._xid74_occurrence_states = {}
        self._xid74_counted_events: set[tuple[str, tuple]] = set()
