"""The ``kind -> model`` table both key/value stores decode rows with."""

from __future__ import annotations

from pydantic import BaseModel

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


def record_models() -> dict[str, type[BaseModel]]:
    """Every record kind a store may be asked to decode.

    The imports are deferred because several of these modules import the
    store package themselves; resolving them when a store is constructed,
    as the stores always did, keeps the module graph acyclic.
    """

    from gpu_fault.fleet import (
        AgentRecord,
        FleetDeployment,
        MultiNodeBarrier,
    )
    from gpu_fault.gpu_metrics import (
        GpuFindingState,
        GpuHealthFinding,
        GpuInventorySnapshot,
        GpuMetricLatest,
        GpuMetricsIngestionResult,
    )
    from gpu_fault.hyperpod import HyperPodSubmissionRecord
    from gpu_fault.installation_resources import InstallationResource
    from gpu_fault.managed_recovery import HyperPodNodeIdentity
    from gpu_fault.policy import (
        FaultPolicyDecision,
        Nvlink74BitOccurrenceState,
        XidCorrelationRecord,
        XidEvent,
    )
    from gpu_fault.processor import (
        PeriodicTaskLease,
        ProcessorLaneLease,
        ProcessorLeadership,
        ProcessorRequest,
    )
    from gpu_fault.regional import (
        RegionalClusterRegistration,
        RegionalRegistryHead,
        RegionalRegistryMember,
        RegionalRegistryRevision,
        RemoteActionCommand,
    )
    from gpu_fault.telemetry import (
        CollectorMetricsSnapshotRecord,
        CollectorStatus,
        RawEvidenceRecord,
        TelemetryMetricLatest,
        WorkloadObservationState,
    )
    from gpu_fault.training_health import TrainingProgressState

    return {
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
        "regional_registry_head": RegionalRegistryHead,
        "regional_registry_member": RegionalRegistryMember,
        "regional_registry_revision": RegionalRegistryRevision,
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
    }
