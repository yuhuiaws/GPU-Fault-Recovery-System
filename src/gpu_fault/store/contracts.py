from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    TypedDict,
    runtime_checkable,
)

from gpu_fault.installation_resources import InstallationResource

if TYPE_CHECKING:
    from gpu_fault.fleet import (
        AgentRecord,
        FleetDeployment,
        MultiNodeBarrier,
    )
    from gpu_fault.models import (
        AdvisoryNotification,
        CompletionDecision,
        DecisionStatus,
        DiagnosticRequest,
        EfaTrafficState,
        EffectiveRuntimeProfile,
        FaultIncident,
        HealthSignalState,
        IncidentState,
        NodeMarker,
        NotificationDelivery,
        NotificationResult,
        NotificationStatus,
        RecoveryPlan,
        TerminalEvent,
        TriageReport,
        WorkflowEvent,
        WorkflowRequest,
        WorkflowStatus,
    )
    from gpu_fault.processor.models import (
        PeriodicTaskLease,
        ProcessorRequest,
    )
    from gpu_fault.regional import (
        RegionalClusterRegistration,
        RegionalRegistryHead,
        RegionalRegistryMember,
        RegionalRegistryRevision,
        RemoteActionCommand,
    )
    from gpu_fault.telemetry import CollectorStatus
    from gpu_fault.telemetry_models import (
        WorkloadObservationState,
    )
    from gpu_fault.watcher import WorkloadCoverageHeartbeat


# Upper bound for one cluster's active (PENDING/RUNNING/SAFETY_PENDING)
# workflow listing. The ingest path runs this jsonb self-join inside the
# aggregation lock, so the read must not grow with the cluster's history
# (F-J5). Any real active set is orders of magnitude smaller.
ACTIVE_WORKFLOW_INCIDENTS_LIMIT = 500


class ProcessorQueueStats(TypedDict):
    depth: int
    oldest_age_seconds: float
    by_cluster: dict[str, int]
    # Age of the oldest PENDING/LEASED request per cluster (S6). The claim
    # window is FIFO across the region, so the region-wide age alone cannot say
    # which cluster's storm is delaying the others. Clusters with no incomplete
    # request are absent, not zero.
    oldest_age_by_cluster: dict[str, float]


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

    def count_fault_rows_blocked_by_observation(self, *, now: datetime) -> int: ...

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

    def release_active_processor_request(
        self,
        request_id: str,
        owner_id: str,
        lane_epoch: int,
        lease_token: str,
        *,
        not_before: datetime | None = None,
        retry_count: int | None = None,
    ) -> None: ...

    def reclaim_expired_processor_leases(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> int: ...

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

    def list_regional_cluster_ids(self) -> list[str]: ...

    def get_regional_registry_head(self) -> RegionalRegistryHead: ...

    def get_regional_registry_revision(
        self, generation: int
    ) -> RegionalRegistryRevision: ...

    def publish_regional_registry_revision(
        self,
        revision: RegionalRegistryRevision,
        *,
        expected_generation: int,
    ) -> RegionalRegistryHead: ...

    def save_regional_registry_member(
        self,
        member: RegionalRegistryMember,
    ) -> RegionalRegistryMember: ...

    def list_regional_registry_members(
        self,
    ) -> list[RegionalRegistryMember]: ...

    def save_agent(self, agent: AgentRecord) -> None: ...

    def get_agent(self, cluster_id: str, node_id: str) -> AgentRecord: ...

    def list_agents(self, cluster_id: str | None = None) -> list[AgentRecord]: ...

    def replace_agent_if_matches(
        self,
        replacement: AgentRecord,
        expected: AgentRecord | None,
    ) -> bool: ...

    def save_fleet_deployment(self, deployment: FleetDeployment) -> None: ...

    def replace_fleet_deployment_if_matches(
        self,
        replacement: FleetDeployment,
        expected: FleetDeployment | None,
    ) -> bool: ...

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
class InstallationResourceStore(Protocol):
    def save_installation_resource(
        self,
        resource: InstallationResource,
    ) -> InstallationResource: ...

    def get_installation_resource(
        self,
        site_id: str,
        resource_key: str,
    ) -> InstallationResource: ...

    def list_installation_resources(
        self,
        site_id: str | None = None,
    ) -> list[InstallationResource]: ...


@runtime_checkable
class WorkflowStore(Protocol):
    """Incident, workflow and plan records and the transactions over them.

    Every method that takes a ``builder`` (``create_incident_workflow_if_absent``
    and the three ``merge_*`` methods) runs it inside the store transaction, with
    the group's advisory lock held and one pooled connection pinned to the
    calling thread. The builder must therefore be pure computation over the
    records it is handed: no network, no adapter or HTTP call, no sleep, no
    independent store round-trip beyond the nested writes the store itself
    makes. A remote call there holds the lock and the connection for its full
    latency, so one slow adapter becomes a stall of every ingest for that group
    and, once the pool is exhausted, of the whole control plane (store review
    2026-09-07, item D).
    """

    def save_workflow(
        self,
        workflow: WorkflowRequest,
        *,
        expected: WorkflowRequest | None = None,
    ) -> None:
        """Write one workflow row without overwriting a row that has moved.

        Without ``expected`` the write is version-guarded: a missing row is
        inserted; an existing row is replaced only while its
        ``merge_revision``, ``execution_epoch`` and ``fencing_token`` still
        equal the caller's copy. Any of the three moving means a merge widened
        the record, an executor re-leased it (epoch bump) or the incident moved
        to a new generation since the caller read it, and writing the copy back
        would erase that; the store raises :class:`StaleWriteError` naming the
        field so the caller re-reads and reapplies. With ``expected`` the write
        is a full compare-and-set on the whole payload the caller read.

        Callers that hold a read copy should pass it as ``expected``. Paths that
        create a row should prefer ``create_incident_workflow_if_absent`` or
        ``save_incident_and_workflow``, which link the incident in the same
        transaction. Before this guard PostgreSQL inherited the SQLite blind
        upsert under a process lock that ``PostgresStore`` neutralizes, so a
        concurrent merge or lease was silently overwritten (store review
        2026-09-07, item B).
        """
        ...

    def get_workflow(self, request_id: str) -> WorkflowRequest: ...

    def save_incident_and_workflow(
        self,
        incident: FaultIncident,
        workflow: WorkflowRequest,
        *,
        extra_event_ids: Sequence[str] = (),
    ) -> None:
        """Write the pair and link ``incident.event_id`` plus every
        ``extra_event_ids`` member event in the same transaction (control-plane
        review 2026-09-08, C-10)."""
        ...

    def reconcile_restored_workflow(
        self,
        workflow_request_id: str,
        successor_workflow_id: str,
        *,
        expected_fencing_token: int,
        expected_execution_epoch: int,
        expected_workflow_updated_at: datetime | None = None,
        reference: str,
        reconciled_at: datetime,
        actor: str | None = None,
        approval: Mapping[str, object] | None = None,
    ) -> tuple[WorkflowRequest, FaultIncident, RecoveryPlan]:
        """Terminalize a BLOCKED record whose successor restored the node.

        ``actor`` and ``approval`` go on the ``OPERATOR_RECONCILED`` event
        appended in the same transaction (audit trail I1).
        """
        ...

    def reconcile_retired_generation_workflow(
        self,
        workflow_request_id: str,
        successor_workflow_id: str,
        *,
        expected_fencing_token: int,
        reference: str | None,
        reconciled_at: datetime,
        actor: str | None = None,
        approval: Mapping[str, object] | None = None,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        """Terminalize a workflow whose incident moved to a later generation.

        ``actor`` and ``approval`` go on the ``OPERATOR_RETIRED_GENERATION``
        event an operator write (one with a ``reference``) appends; the
        dispatcher's sweep passes neither and records none (audit trail I1).
        """
        ...

    def get_incident(self, incident_id: str) -> FaultIncident: ...

    def save_incident(
        self,
        incident: FaultIncident,
        *,
        expected: FaultIncident | None = None,
        extra_event_ids: Sequence[str] = (),
    ) -> None:
        """Write one incident row and its event link without overwriting a row
        that has moved (architecture review 2026-09-07, item D1).
        ``extra_event_ids`` are linked to the incident in the same transaction
        (control-plane review 2026-09-08, C-09).

        Without ``expected`` the write is version-guarded like
        ``save_workflow``: a missing row is inserted; an existing row is
        replaced only while its ``fencing_token`` is not ahead of the caller's
        copy -- the generation only moves forward, so a lower token is a copy
        read before a generation change. That raises :class:`StaleWriteError`
        naming the field. ``state``, ``reasons`` and
        ``workflow_request_id`` are what callers and the reconcile paths change
        through this method, so they are not guarded this way: a caller that
        read the row and writes it back (the coordinator's reason trail, the
        executor's terminal state, the retired-generation audit) must pass its
        read copy as ``expected``, which makes the write a full compare-and-set
        on the payload it read.
        """
        ...

    def get_incident_by_event(self, event_id: str) -> FaultIncident | None: ...

    def link_event_to_incident(self, event_id: str, incident_id: str) -> None: ...

    def list_workflows(
        self,
        statuses: set[WorkflowStatus] | None = None,
        *,
        limit: int = 100,
        newest_first: bool = False,
        dispatchable_at: datetime | None = None,
        exclude_request_ids: Collection[str] = (),
        after: WorkflowRequest | None = None,
    ) -> list[WorkflowRequest]:
        """Workflows in ``statuses``, oldest ``updated_at`` first.

        ``dispatchable_at`` pushes the dispatcher's permanent filters below
        the LIMIT (F-A2b): rows whose ``not_before`` is still in the future
        and rows whose predecessor is still open are excluded in the store,
        as are ``exclude_request_ids`` (retired generations). Without it the
        oldest held rows occupied the front of every scan window.

        In that dispatch mode the order is ``dispatch_eligible_at`` --
        max(``created_at``, ``not_before``), then ``request_id`` -- which no
        merge rewrites (F-A2a), and ``after`` is the last row of the previous
        page: the result starts strictly after it in that order (F-A2c).
        ``after`` without ``dispatchable_at`` is a ``ValueError``; the cursor
        is only defined over the dispatch order.
        """
        ...

    def amend_workflow(
        self,
        request_id: str,
        updates: Mapping[str, object],
        *,
        event: WorkflowEvent | None = None,
    ) -> WorkflowRequest:
        """Apply ``updates`` to a live workflow row from outside its lease.

        Bumps ``merge_revision`` like a merge does, so an executor holding a
        stale copy fails its next leased write and re-reads (F-B1). Used for
        out-of-band verdicts: the job was withdrawn (F-N1 §7), the plan was
        rewritten after a node-busy timeout (F-N1 §8). Raises NotFoundError.
        ``event`` is appended to the row's audit history in the same write,
        bounded, without a second merge-revision bump (audit trail I1).
        """
        ...

    def count_held_workflows(
        self,
        statuses: set[WorkflowStatus] | None,
        *,
        dispatchable_at: datetime,
        exclude_request_ids: Collection[str] = (),
    ) -> dict[str, int]:
        """How many rows the pushdown of ``list_workflows`` held back, by
        reason (``not_before``, ``predecessor``, ``retired``); zero counts are
        omitted. Counts are independent, so one row may count under two."""
        ...

    def workflow_status_counts(self) -> dict[WorkflowStatus, int]: ...

    def incident_state_counts(self) -> dict[IncidentState, int]:
        """Count every persisted incident by state without decoding any;
        every state is present, absent ones as 0."""
        ...

    def blocked_workflows_without_verified_restore(self) -> int: ...

    def list_active_workflow_incidents(
        self,
        cluster_id: str,
        *,
        node_ids: set[str] | None = None,
        job_id: str | None = None,
        limit: int = ACTIVE_WORKFLOW_INCIDENTS_LIMIT,
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

    def has_workflow_successor(self, predecessor_workflow_id: str) -> bool: ...

    def list_orphan_workflows(
        self, *, created_before: datetime, limit: int = 1000
    ) -> list[WorkflowRequest]:
        """PENDING / SAFETY_PENDING workflows nothing will ever act on (F-B3).

        A workflow is an orphan when its incident is gone or names a different
        workflow *and* no other workflow names it as predecessor: the
        dispatcher hands out one workflow per incident, the fences and both
        sweepers require a strictly higher generation, so the record sits
        forever. ``created_before`` excludes rows still inside the aggregation
        window, where the twin shape is normal churn. Oldest ``created_at``
        first. Read-only.
        """
        ...

    def list_incidents_with_missing_workflow(
        self, *, limit: int = 1000
    ) -> list[FaultIncident]:
        """Incidents whose ``workflow_request_id`` names no workflow row (F-B3).

        The mirror image of ``list_orphan_workflows``: a pointer to a record that
        was never persisted or has been cleaned up. Read-only.
        """
        ...

    def list_incidents_by_state(
        self,
        cluster_id: str,
        states: Collection[IncidentState],
        *,
        node_ids: set[str] | None = None,
        limit: int = ACTIVE_WORKFLOW_INCIDENTS_LIMIT,
    ) -> list[FaultIncident]:
        """Incidents of ``cluster_id`` in one of ``states``, newest
        ``updated_at`` first, optionally only those naming one of ``node_ids``
        (an empty set matches nothing). Independent of any workflow row: the
        incidents this serves (``IncidentClosureService``) are ESCALATED behind
        a workflow that already ended. Read-only.
        """
        ...

    def claim_workflow(
        self,
        request_id: str,
        executor_id: str,
        fencing_token: int,
        *,
        now: datetime | None = None,
        lease_duration: timedelta = timedelta(minutes=3),
        remediation_budget_claims: dict[str, int] | None = None,
    ) -> WorkflowRequest: ...

    def extend_remediation_budget(
        self,
        request_id: str,
        executor_id: str,
        claims: dict[str, int],
        *,
        now: datetime | None = None,
    ) -> WorkflowRequest: ...

    def renew_workflow_lease(
        self,
        request_id: str,
        executor_id: str,
        execution_epoch: int,
        *,
        now: datetime | None = None,
        lease_duration: timedelta = timedelta(minutes=3),
    ) -> WorkflowRequest:
        """Confirm the caller's lease and return the current row.

        Owner, epoch and expiry are validated exactly as for a leased save; a
        mismatch is a ``WorkflowLeaseError``. The lease is only extended when
        less than half of ``lease_duration`` remains; otherwise the row is
        returned unchanged and nothing is written. The executor renews before
        and after every step, seconds into a 3-minute lease, and every renewal
        re-serialized the whole record (step executions, events, steps) into a
        new heap tuple, its TOAST chunks and every partial index entry. The
        read stays because the executor picks merges up through the row this
        returns (store review 2026-09-07, item F1).
        """
        ...

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
    ) -> tuple[FaultIncident, WorkflowRequest, bool]:
        """Create an event's incident and workflow once, or return the pair
        the event already resolves to.

        ``builder`` runs inside the store transaction with the
        ``incident_workflow/<key>`` advisory lock and a pinned connection held:
        it must be pure computation on the caller's own inputs -- no network,
        no adapter or HTTP call, no sleep, no store reads of its own (store
        review 2026-09-07, item D; see the class docstring for why).
        """
        ...

    def merge_replacement_workflow(
        self,
        group_key: str,
        event_id: str,
        builder: Callable[
            [FaultIncident | None, WorkflowRequest | None],
            tuple[FaultIncident, WorkflowRequest],
        ],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        """Merge one node fault into the replacement group ``group_key``.

        ``builder`` receives the group's current incident and workflow (both
        row-locked, or ``None`` for a new group) and returns the merged pair.
        It runs inside the store transaction with the
        ``replacement_fault_group/<key>`` advisory lock and a pinned connection
        held, so it must be pure computation over those two records: no
        network, no adapter or HTTP call, no sleep, no independent store
        round-trip (store review 2026-09-07, item D).
        """
        ...

    def merge_attempt_fault_workflow(
        self,
        group_key: str,
        event_id: str,
        builder: Callable[
            [FaultIncident | None, WorkflowRequest | None],
            tuple[FaultIncident, WorkflowRequest],
        ],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        """Merge one fault into the attempt-scoped group ``group_key``.

        Same contract as ``merge_replacement_workflow`` under the
        ``sxid_fault_group/<key>`` advisory lock: ``builder`` is pure
        computation over the locked records it is handed, nothing external
        (store review 2026-09-07, item D).
        """
        ...

    def merge_sxid_workflow(
        self,
        group_key: str,
        event_id: str,
        builder: Callable[
            [FaultIncident | None, WorkflowRequest | None],
            tuple[FaultIncident, WorkflowRequest],
        ],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        """Alias of ``merge_attempt_fault_workflow`` kept for the SXID ingest
        path; the same builder purity rule applies (store review 2026-09-07,
        item D)."""
        ...

    def find_open_remote_command(
        self,
        workflow_request_id: str,
        step_index: int,
        command_step_space: str,
        *,
        exclude_command_id: str | None = None,
    ) -> RemoteActionCommand | None:
        """The oldest open (PENDING/LEASED/WAITING) command for one step identity.

        The step identity is (workflow, step index, step space) -- safety and
        official steps share indexes (F-C2); ``command_step_space`` is
        ``"official"`` or ``"safety"`` (``remote_helpers.workflow_step_space``).
        Command ids are digests of the whole step, so a merge that rewrites a
        step's targets or parameters gives the next dispatch a new id while the
        old command may still be executing; the adapter asks this before it
        creates a command and holds instead (architecture review 2026-09-07,
        item D5). ``exclude_command_id`` leaves the caller's own id out.
        """
        ...

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

    def expire_stale_fenced_remote_commands(
        self,
        *,
        lease_expired_before: datetime,
        limit: int,
    ) -> int:
        """Fail LEASED commands whose lease lapsed before ``lease_expired_before``
        while their workflow sits at another ``fencing_token`` (control-plane
        review 2026-09-08, D-9): ``status_source="stale-fence"``."""
        ...

    def list_remote_commands(
        self,
        *,
        workflow_request_ids: Iterable[str] | None = None,
    ) -> list[RemoteActionCommand]: ...

    # Every backend has implemented this since remote commands existed; it was
    # missing here only because its callers reached the store through `Any`.
    # `deploy/control-plane/regional/probes/` made them typed, which is the
    # point of those probes being real modules rather than string literals.
    def remote_command_stats(
        self, *, now: datetime | None = None
    ) -> dict[str, Any]: ...

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

    def list_markers(self) -> list[NodeMarker]:
        """Every marker ever recorded. Not for a request path.

        The marker table grows with every observation on every node, so no
        production caller uses this: the scoped reads below answer the two
        questions that are actually asked of it, and one of them is pinned by a
        test that replaces this method with a raising stub. It stays on the
        contract because the acceptance fixtures and tests that assert over the
        whole table (``scripts/e2e/regional/q118_fallback_marker.py``,
        ``scripts/e2e/regional/warm_spare_fixture.py``) legitimately want all of
        it against a fixture-sized store.
        """
        ...

    def list_markers_for_incident(self, incident_id: str) -> list[NodeMarker]: ...

    def list_recent_markers_for_nodes(
        self,
        node_ids: set[str],
        observed_after: datetime,
        *,
        source_boot_id: str | None = None,
        limit: int = 1000,
    ) -> list[NodeMarker]: ...

    def list_markers_in_scope_window(
        self,
        *,
        node_ids: set[str],
        gpu_uuids: set[str],
        fabric_partitions: set[str],
        observed_from: datetime,
        observed_to: datetime,
        limit: int = 1000,
    ) -> list[NodeMarker]: ...

    def completion_transaction(self, event_key: str) -> AbstractContextManager[None]:
        """Serialize and (where the backend can) atomize one attempt's decision.

        The completion service decides a terminal event -- event row, plan with
        its incident and workflow, decision -- inside this context, keyed by the
        event (F-G2). PostgreSQL takes the ``completion/<event_key>`` advisory
        lock in one transaction, so two replicas cannot both decide the same
        event and a crash mid-way leaves no event row without a decision. The
        in-memory and SQLite stores serialize on their process lock. Nested
        store writes (each with their own transaction) must be allowed inside.

        The body runs with that advisory lock and one pooled connection held,
        so it must be pure computation over records already in hand plus the
        nested store writes: no network, no adapter or HTTP call, no sleep, no
        independent store round-trip. A remote call inside holds the lock and
        the connection for its latency, and one slow adapter becomes a stall of
        every completion for that event key and, with the pool drained, of the
        control plane (store review 2026-09-07, item D).
        """
        ...

    def save_event_if_absent(self, event: TerminalEvent) -> bool: ...

    def get_event_by_attempt(
        self, cluster_id: str, attempt_id: str
    ) -> TerminalEvent: ...

    def save_decision(self, decision: CompletionDecision) -> None: ...

    def get_decision_by_event(self, event_key: str) -> CompletionDecision | None: ...

    def get_decision_by_attempt(
        self, cluster_id: str, attempt_id: str
    ) -> CompletionDecision: ...

    def list_decisions_by_status(
        self,
        status: DecisionStatus,
        *,
        older_than: datetime | None = None,
        limit: int = 100,
    ) -> list[CompletionDecision]:
        """Decisions in ``status``, oldest first.

        A decision carries no timestamp of its own, so its age is that of the
        diagnostic request it points at (``diagnostic_request_id`` ->
        ``DiagnosticRequest.created_at``). With ``older_than`` only decisions
        whose request was created at or before it are returned; a decision
        whose request cannot be found is returned too -- nothing can ever
        report on it, so it is stale by definition (F-G2 (4)).
        """
        ...

    def decision_status_counts(self) -> dict[DecisionStatus, int]: ...

    def count_completion_events_without_decision(self) -> int:
        """Terminal event rows with no decision row: the poisoned shape of
        P0-48B, exported as a gauge (F-G2 (6))."""
        ...

    def save_diagnostic(self, request: DiagnosticRequest) -> None: ...

    def save_plan(
        self,
        plan: RecoveryPlan,
        *,
        expected: RecoveryPlan | None = None,
    ) -> None:
        """Upsert one plan row; with ``expected`` only while it still equals it.

        ``RecoveryPlan`` has no version field, so the unconditional form stays
        a blind upsert. Callers that read the plan and write it back -- the
        dispatcher's status mirror -- pass the copy they read as ``expected``
        and get :class:`StaleWriteError` instead of overwriting a reconcile
        that landed in between (architecture review 2026-09-07, item D2).
        """
        ...

    def get_plan(self, plan_id: str) -> RecoveryPlan: ...

    def save_triage_report(self, report: TriageReport) -> None: ...

    def get_diagnostic(self, request_id: str) -> DiagnosticRequest: ...

    def get_profile(self, version: str) -> EffectiveRuntimeProfile: ...

    def save_profile(self, profile: EffectiveRuntimeProfile) -> None: ...

    def list_attempt_observation_states(
        self,
        cluster_id: str | None = None,
        *,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[WorkloadObservationState]: ...

    def save_workload_coverage(self, heartbeat: WorkloadCoverageHeartbeat) -> bool: ...

    def get_workload_coverage(
        self, cluster_id: str
    ) -> WorkloadCoverageHeartbeat | None: ...

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

    def get_health_signal_state(self, signal_key: str) -> HealthSignalState | None: ...

    def mark_health_signal_notified(
        self, signal_key: str, *, notified_at: datetime
    ) -> None:
        """Set a signal's ``notified`` latch once its notification was delivered.

        ``claim_health_signal_transitions`` decides to emit but does not latch
        (P0-38B); the deliverer calls this after the commit that carried the
        incident. A missing or inactive signal, or one whose activation began
        after ``notified_at``, is left alone.
        """
        ...


@runtime_checkable
class NotificationStore(Protocol):
    def save_notification_if_absent(
        self, notification: AdvisoryNotification
    ) -> AdvisoryNotification: ...

    def get_notification(self, notification_id: str) -> AdvisoryNotification: ...

    def list_notifications(
        self,
        *,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[AdvisoryNotification]: ...

    def notification_status_counts(self) -> dict[NotificationStatus, int]: ...

    def notification_delivery_stats(
        self, *, now: datetime | None = None
    ) -> dict[str, Any]:
        """The outbox state machine as an aggregate (ARCH-E E1).

        ``by_status`` counts delivery rows per ``NotificationDeliveryStatus`` as
        the outbox treats them (a row whose result is already SENT counts as
        SENT, a SKIPPED verdict on an undelivered row counts as DEAD);
        ``pending`` is the undelivered count and ``oldest_pending_age_seconds``
        how long the oldest of them has waited since it was queued or last
        re-queued. Read by ``/metrics``; never decodes a notification body.
        """
        ...

    def save_notification_result(self, result: NotificationResult) -> None: ...

    def get_notification_result(
        self, notification_id: str
    ) -> NotificationResult | None: ...

    def get_notification_delivery(
        self, notification_id: str
    ) -> NotificationDelivery | None: ...

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
    InstallationResourceStore,
    WorkflowStore,
    CompletionStore,
    TelemetryStore,
    NotificationStore,
    Protocol,
):
    """Structural contract consumed by the application layer."""
