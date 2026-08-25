from __future__ import annotations

from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Callable,
    Protocol,
    TypedDict,
    runtime_checkable,
)

if TYPE_CHECKING:
    from gpu_fault.fleet import (
        AgentRecord,
        FleetDeployment,
        MultiNodeBarrier,
    )
    from gpu_fault.models import (
        FaultIncident,
        WorkflowRequest,
    )
    from gpu_fault.models import (
        AdvisoryNotification,
        CompletionDecision,
        DiagnosticRequest,
        EfaTrafficState,
        EffectiveRuntimeProfile,
        NodeMarker,
        NotificationDelivery,
        NotificationResult,
        RecoveryPlan,
        TerminalEvent,
        TriageReport,
        WorkflowStatus,
    )
    from gpu_fault.processor.models import (
        PeriodicTaskLease,
        ProcessorRequest,
    )
    from gpu_fault.regional import (
        RegionalClusterRegistration,
        RemoteActionCommand,
    )
    from gpu_fault.telemetry_models import (
        WorkloadObservationState,
    )
    from gpu_fault.telemetry import CollectorStatus


class ProcessorQueueStats(TypedDict):
    depth: int
    oldest_age_seconds: float
    by_cluster: dict[str, int]


class ProcessorQueueCountStatus(TypedDict):
    expected_total: int
    counter_total: int
    mismatched_clusters: int
    ready: bool


@runtime_checkable
class ProcessorStore(Protocol):
    def try_enqueue_processor_request(
        self,
        request: ProcessorRequest,
        *,
        max_depth: int,
        max_cluster_depth: int,
        reserved_fault_depth: int = 0,
        reserved_cluster_fault_depth: int = 0,
        global_admission_guard: int = 256,
    ) -> tuple[ProcessorRequest | None, str | None]: ...

    def try_enqueue_processor_requests_batch(
        self,
        requests: list[ProcessorRequest],
        *,
        max_depth: int,
        max_cluster_depth: int,
        reserved_fault_depth: int = 0,
        reserved_cluster_fault_depth: int = 0,
        global_admission_guard: int = 256,
    ) -> list[tuple[ProcessorRequest | None, str | None]]: ...

    def claim_active_processor_requests(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
        include_paths: set[str] | None = None,
        exclude_paths: set[str] | None = None,
        routine_starvation_seconds: float = 30,
    ) -> list[ProcessorRequest]: ...

    def complete_active_processor_request(
        self,
        request_id: str,
        owner_id: str,
        lane_epoch: int,
        lease_token: str,
        *,
        response_status: int,
        response_content_type: str | None,
        response_body_base64: str,
        path: str | None = None,
    ) -> ProcessorRequest: ...

    def processor_queue_stats(
        self, *, now: datetime | None = None
    ) -> ProcessorQueueStats: ...

    def processor_queue_count_status(
        self,
    ) -> ProcessorQueueCountStatus: ...

    def processor_fault_backlog_depth(self) -> int: ...


@runtime_checkable
class FleetStore(Protocol):
    def save_regional_cluster(
        self, registration: RegionalClusterRegistration
    ) -> RegionalClusterRegistration: ...

    def delete_regional_cluster(self, cluster_id: str) -> None: ...

    def get_regional_cluster(self, cluster_id: str) -> RegionalClusterRegistration: ...

    def list_regional_clusters(
        self,
    ) -> list[RegionalClusterRegistration]: ...

    def save_agent(self, agent: AgentRecord) -> None: ...

    def get_agent(self, cluster_id: str, node_id: str) -> AgentRecord: ...

    def list_agents(self, cluster_id: str | None = None) -> list[AgentRecord]: ...

    def replace_agent_if_matches(
        self,
        replacement: AgentRecord,
        expected: AgentRecord | None,
    ) -> bool: ...

    def save_fleet_deployment(self, deployment: FleetDeployment) -> None: ...

    def get_fleet_deployment(self, deployment_id: str) -> FleetDeployment: ...

    def list_fleet_deployments(
        self,
    ) -> list[FleetDeployment]: ...

    def list_active_fleet_deployments(
        self, cluster_id: str
    ) -> list[FleetDeployment]: ...

    def save_barrier(self, barrier: MultiNodeBarrier) -> None: ...

    def get_barrier(self, barrier_id: str) -> MultiNodeBarrier: ...

    def list_barriers(self) -> list[MultiNodeBarrier]: ...


@runtime_checkable
class WorkflowStore(Protocol):
    def save_workflow(self, workflow: WorkflowRequest) -> None: ...

    def get_workflow(self, request_id: str) -> WorkflowRequest: ...

    def save_incident_and_workflow(
        self,
        incident: FaultIncident,
        workflow: WorkflowRequest,
    ) -> None: ...

    def get_incident(self, incident_id: str) -> FaultIncident: ...

    def save_incident(self, incident: FaultIncident) -> None: ...

    def get_incident_by_event(self, event_id: str) -> FaultIncident | None: ...

    def link_event_to_incident(self, event_id: str, incident_id: str) -> None: ...

    def list_workflows(
        self,
        statuses: set[WorkflowStatus] | None = None,
        *,
        limit: int = 100,
        newest_first: bool = False,
    ) -> list[WorkflowRequest]: ...

    def list_active_workflow_incidents(
        self,
        cluster_id: str,
        *,
        node_ids: set[str] | None = None,
        job_id: str | None = None,
    ) -> list[tuple[FaultIncident, WorkflowRequest]]: ...

    def list_job_recovery_workflow_incidents(
        self,
        cluster_id: str,
        job_id: str,
        attempt_id: str,
        *,
        limit: int = 100,
    ) -> list[tuple[FaultIncident, WorkflowRequest]]: ...

    def list_unhandled_failed_workflows(
        self, *, limit: int = 1000
    ) -> list[WorkflowRequest]: ...

    def get_preempting_successor(
        self, predecessor_workflow_id: str
    ) -> WorkflowRequest | None: ...

    def claim_workflow(
        self,
        request_id: str,
        executor_id: str,
        fencing_token: int,
        *,
        now: datetime | None = None,
        lease_duration: timedelta = timedelta(minutes=3),
    ) -> WorkflowRequest: ...

    def renew_workflow_lease(
        self,
        request_id: str,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
        lease_duration: timedelta = timedelta(minutes=3),
    ) -> WorkflowRequest: ...

    def save_workflow_if_leased(
        self,
        workflow: WorkflowRequest,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
    ) -> None: ...

    def save_workflow_and_incident_if_leased(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
    ) -> None: ...

    def create_incident_workflow_if_absent(
        self,
        event_id: str,
        builder: Callable[[], tuple[FaultIncident, WorkflowRequest]],
        *,
        serialization_key: str | None = None,
    ) -> tuple[FaultIncident, WorkflowRequest, bool]: ...

    def cancel_remote_commands_for_workflow(
        self,
        workflow_request_id: str,
        *,
        reason: str,
    ) -> dict[str, int]: ...

    def cancel_remote_command(
        self,
        command_id: str,
        *,
        reason: str,
    ) -> bool: ...

    def get_remote_command(self, command_id: str) -> RemoteActionCommand: ...

    def list_remote_commands(self) -> list[RemoteActionCommand]: ...

    def has_incomplete_processor_requests_for_scopes(
        self,
        cluster_id: str,
        scope_keys: set[str],
    ) -> bool: ...

    def acquire_periodic_task_lease(
        self,
        task_key: str,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
    ) -> PeriodicTaskLease: ...


@runtime_checkable
class CompletionStore(Protocol):
    def add_marker(self, marker: NodeMarker) -> None: ...

    def list_markers(self) -> list[NodeMarker]: ...

    def list_markers_for_incident(self, incident_id: str) -> list[NodeMarker]: ...

    def list_recent_markers_for_nodes(
        self,
        node_ids: set[str],
        observed_after: datetime,
        *,
        source_boot_id: str | None = None,
        limit: int = 1000,
    ) -> list[NodeMarker]: ...

    def save_event_if_absent(self, event: TerminalEvent) -> bool: ...

    def get_event_by_attempt(
        self, cluster_id: str, attempt_id: str
    ) -> TerminalEvent: ...

    def save_decision(self, decision: CompletionDecision) -> None: ...

    def get_decision_by_event(self, event_key: str) -> CompletionDecision | None: ...

    def get_decision_by_attempt(
        self, cluster_id: str, attempt_id: str
    ) -> CompletionDecision: ...

    def save_plan(self, plan: RecoveryPlan) -> None: ...

    def get_plan(self, plan_id: str) -> RecoveryPlan: ...

    def save_triage_report(self, report: TriageReport) -> None: ...

    def get_diagnostic(self, request_id: str) -> DiagnosticRequest: ...

    def get_profile(self, version: str) -> EffectiveRuntimeProfile: ...

    def save_profile(self, profile: EffectiveRuntimeProfile) -> None: ...

    def list_attempt_observation_states(
        self, cluster_id: str | None = None
    ) -> list[WorkloadObservationState]: ...

    def get_efa_traffic_state(self, state_key: str) -> EfaTrafficState: ...

    @staticmethod
    def efa_traffic_state_key(
        cluster_id: str,
        node_id: str,
        job_id: str,
        attempt_id: str,
    ) -> str: ...


@runtime_checkable
class TelemetryStore(Protocol):
    def list_collector_statuses(
        self,
        cluster_id: str,
        node_id: str | None = None,
    ) -> list[CollectorStatus]: ...


@runtime_checkable
class NotificationStore(Protocol):
    def save_notification_if_absent(
        self, notification: AdvisoryNotification
    ) -> AdvisoryNotification: ...

    def get_notification(self, notification_id: str) -> AdvisoryNotification: ...

    def list_notifications(
        self,
    ) -> list[AdvisoryNotification]: ...

    def save_notification_result(self, result: NotificationResult) -> None: ...

    def get_notification_result(
        self, notification_id: str
    ) -> NotificationResult | None: ...

    def enqueue_notification_delivery(
        self,
        notification_id: str,
        *,
        now: datetime | None = None,
    ) -> NotificationDelivery: ...

    def claim_notification_deliveries(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
    ) -> list[NotificationDelivery]: ...

    def complete_notification_delivery(
        self,
        notification_id: str,
        *,
        owner_id: str,
        lease_epoch: int,
        result: NotificationResult,
        now: datetime,
        retry_at: datetime | None = None,
        terminal: bool = False,
    ) -> NotificationDelivery: ...

    def release_notification_delivery(
        self,
        notification_id: str,
        *,
        owner_id: str,
        lease_epoch: int,
        now: datetime,
        retry_at: datetime,
    ) -> NotificationDelivery | None: ...


@runtime_checkable
class ControlPlaneStore(
    ProcessorStore,
    FleetStore,
    WorkflowStore,
    CompletionStore,
    TelemetryStore,
    NotificationStore,
    Protocol,
):
    """Structural contract consumed by the application layer."""
