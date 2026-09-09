from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from gpu_fault.markers import marker_is_diagnostic, retire_markers_for_incident
from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    FaultIncident,
    IncidentState,
    NodeMarker,
    RecoveryAction,
    RecoveryPlan,
    TerminalEvent,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    recovery_action_sort_key,
)
from gpu_fault.planner import PlanBuilder
from gpu_fault.store import NotFoundError
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.telemetry import (
    EvidenceKind,
    EvidenceService,
)
from gpu_fault.watcher import (
    FailureContainmentDecision,
    FailureDetectedEvent,
    failure_containment_ids,
)

LOGGER = logging.getLogger(__name__)


class CompletionService:
    def __init__(
        self,
        store: ControlPlaneStore,
        planner: PlanBuilder | None = None,
        workflow_compiler=None,
        marker_window: timedelta = timedelta(minutes=10),
        evidence_service: EvidenceService | None = None,
        marker_ttl_seconds: int | None = None,
    ) -> None:
        if marker_window <= timedelta(0):
            raise ValueError("marker_window must be positive")
        if (
            marker_ttl_seconds is not None
            and marker_window.total_seconds() > marker_ttl_seconds
        ):
            # A marker observed at t correlates with terminals up to
            # t + marker_window, but the correlator also requires it to be
            # unexpired at the terminal's ended_at; a window longer than the
            # TTL is therefore a configuration that silently never matches
            # the tail of its own window (F-G6 / P2-50I).
            raise ValueError(
                f"marker_window ({marker_window.total_seconds():.0f}s) must not "
                f"exceed the marker TTL ({marker_ttl_seconds}s)"
            )
        self.store = store
        self.planner = planner or PlanBuilder()
        self.workflow_compiler = workflow_compiler
        self.marker_window = marker_window
        self.marker_ttl_seconds = marker_ttl_seconds
        self.evidence_service = evidence_service
        # No process lock (F-G2 / P1-62G): active-active replicas serialize
        # on ``store.completion_transaction(event_key)`` instead, which is
        # also what makes the event row and its decision one write.

    def add_marker(self, marker: NodeMarker) -> NodeMarker:
        self.store.add_marker(marker)
        return marker

    def _ensure_containment(
        self,
        *,
        cluster_id: str,
        job_id: str,
        attempt_id: str,
        runtime_profile_version: str,
        workload_ids: list[str],
        node_ids: list[str],
        gpu_uuids: list[str],
        reasons: list[str],
    ) -> tuple[FaultIncident, WorkflowRequest, bool]:
        """Create or retrieve containment incident and workflow for a failure.

        Builds a passive-containment-v1 incident with STOP_WORKLOAD action and
        a two-step workflow (FREEZE_EVIDENCE, STOP_WORKLOADS). Shared by both
        the failure-detected handler and the terminal handler (when a terminal
        arrives before its failure-detected event).

        Args:
            cluster_id: Target cluster
            job_id: Job identifier (not currently persisted)
            attempt_id: Attempt identifier
            runtime_profile_version: Runtime profile version
            workload_ids: Workload identifiers to stop
            node_ids: Affected node identifiers
            gpu_uuids: Affected GPU UUIDs
            reasons: Human-readable reason strings for the incident

        Returns:
            Tuple of (incident, workflow, created) where created is True if
            this call created the incident, False if it already existed.
        """
        event_key = f"{cluster_id}/{attempt_id}/TrainingAttemptFailureDetected"
        incident_id, workflow_id = failure_containment_ids(event_key)

        def build() -> tuple[FaultIncident, WorkflowRequest]:
            now = datetime.now(timezone.utc)
            workflow = WorkflowRequest(
                request_id=workflow_id,
                incident_id=incident_id,
                runtime_profile_version=runtime_profile_version,
                status=WorkflowStatus.PENDING,
                official_action="STOP_WORKLOAD",
                fencing_token=1,
                official_steps=[
                    WorkflowStepSpec(
                        operation=(WorkflowOperation.FREEZE_EVIDENCE),
                        execution_owner=("gpu-fault-control-plane"),
                        node_ids=node_ids,
                        gpu_uuids=gpu_uuids,
                        workload_ids=workload_ids,
                    ),
                    WorkflowStepSpec(
                        operation=(WorkflowOperation.STOP_WORKLOADS),
                        execution_owner=("gpu-fault-kubernetes-adapter"),
                        node_ids=node_ids,
                        gpu_uuids=gpu_uuids,
                        workload_ids=workload_ids,
                        parameters={"termination_initiator_incident_id": incident_id},
                    ),
                ],
                created_at=now,
                updated_at=now,
            )
            incident = FaultIncident(
                incident_id=incident_id,
                event_id=event_key,
                event_type=("TRAINING_ATTEMPT_FAILURE_DETECTED"),
                cluster_id=cluster_id,
                node_ids=node_ids,
                gpu_uuids=gpu_uuids,
                policy_version="passive-containment-v1",
                policy_source="completion-watcher",
                official_action="STOP_WORKLOAD",
                effective_action=RecoveryAction.STOP_WORKLOAD,
                state=IncidentState.ACTION_PENDING,
                workflow_request_id=workflow_id,
                reasons=reasons,
                created_at=now,
                updated_at=now,
            )
            return incident, workflow

        return self.store.create_incident_workflow_if_absent(event_key, build)

    def handle_failure_detected(
        self, event: FailureDetectedEvent
    ) -> FailureContainmentDecision:
        if not event.runtime_profile_version:
            raise ValueError("failure detection requires runtime profile version")
        self.store.get_profile(event.runtime_profile_version)
        if not event.workload_ids:
            raise ValueError("failure detection requires owning workload IDs")

        reasons = [
            event.reason,
            f"first_failed_rank={event.first_failed_rank}",
            f"exit_code={event.exit_code}",
            *[
                (
                    "workload log capture failed: "
                    f"{item.get('namespace', '')}/"
                    f"{item.get('pod_name', '')}: "
                    f"{item['capture_error']}"
                )
                if item.get("capture_error")
                else (
                    "workload log evidence: "
                    f"{item.get('record_id')}"
                    + (f" ({item.get('s3_uri')})" if item.get("s3_uri") else "")
                )
                for item in event.workload_log_snapshots
            ],
        ]

        incident, workflow, created = self._ensure_containment(
            cluster_id=event.cluster_id,
            job_id=event.job_id,
            attempt_id=event.attempt_id,
            runtime_profile_version=event.runtime_profile_version,
            workload_ids=event.workload_ids,
            node_ids=event.node_ids,
            gpu_uuids=event.gpu_uuids,
            reasons=reasons,
        )

        if self.evidence_service is not None:
            for snapshot in event.workload_log_snapshots:
                if (
                    snapshot.get("capture_error")
                    or not snapshot.get("record_id")
                    or not snapshot.get("node_id")
                ):
                    continue
                try:
                    observed_at = datetime.fromisoformat(
                        str(snapshot["captured_at"]).replace("Z", "+00:00")
                    )
                    self.evidence_service.capture(
                        record_id=str(snapshot["record_id"]),
                        cluster_id=event.cluster_id,
                        node_id=str(snapshot["node_id"]),
                        kind=EvidenceKind.WORKLOAD_LOG,
                        observed_at=observed_at,
                        attempt_ids=[event.attempt_id],
                        payload=dict(snapshot),
                    )
                except Exception:
                    LOGGER.exception(
                        "cannot persist emergency workload log evidence for attempt %s",
                        event.attempt_id,
                    )
        return FailureContainmentDecision(
            attempt_id=event.attempt_id,
            event_key=event.event_key,
            incident_id=incident.incident_id,
            workflow_request_id=workflow.request_id,
            duplicate=not created,
        )

    def handle_terminal(self, event: TerminalEvent) -> CompletionDecision:
        """Decide a terminal event immediately, whatever its containment is doing.

        The terminal used to be held (an HTTP conflict the data plane retried)
        while the attempt's passive containment workflow was open or not yet
        persisted.
        That queued the cluster's terminal events behind one stuck STOP
        (P0-62C) and left the decision to a retry loop. Now the containment
        is created here when it is missing, and the recovery workflow names it
        as ``predecessor_workflow_id``, so the dispatcher sequences
        stop-then-restart without anyone waiting.
        """

        existing = self.store.get_decision_by_event(event.event_key)
        if existing is not None:
            return existing.model_copy(update={"duplicate": True})
        self.store.get_profile(event.runtime_profile_version)
        containment_workflow_id = self._ensure_terminal_containment(event)
        explicit_initiator, passive_containment = self._containment_context(event)

        # Everything persisted about this event -- the event row, the plan
        # with its incident and workflow, the decision -- is written inside
        # one store transaction keyed by the event (F-G2 (3)(5)). On
        # PostgreSQL that is one advisory lock and one commit: a second
        # replica deciding the same event waits, and a crash mid-way leaves
        # no event row without a decision (P0-48B).
        with self.store.completion_transaction(event.event_key):
            existing = self.store.get_decision_by_event(event.event_key)
            if existing is not None:
                return existing.model_copy(update={"duplicate": True})
            if not self.store.save_event_if_absent(event):
                # The event row was written but the decision never was: an
                # earlier attempt died between the two writes (before the
                # two shared a transaction) or was written by a legacy
                # release. Raising here poisoned the attempt permanently
                # (P0-62A). Fall through and decide now.
                LOGGER.warning(
                    "completion event exists without a decision; deciding on "
                    "redelivery: event_key=%s",
                    event.event_key,
                )
            decision = self._decide(
                event,
                explicit_initiator,
                passive_containment,
                predecessor_workflow_id=containment_workflow_id,
            )
            self.store.save_decision(decision)
        return decision

    def _passive_event_key(self, event: TerminalEvent) -> str:
        """The failure-detected event key the passive containment is keyed by."""

        return f"{event.cluster_id}/{event.attempt_id}/TrainingAttemptFailureDetected"

    def _ensure_terminal_containment(self, event: TerminalEvent) -> str | None:
        """The containment workflow id for this attempt, created here when a
        failure terminal arrives before (or without) its failure-detected event.

        Runs before the completion transaction: it is its own idempotent
        store transaction keyed by the passive event id, and a second replica
        racing on it gets ``created=False``. Nothing is created for a clean
        STOPPED/SUCCEEDED terminal (nothing failed), for a terminal with no
        workload ids (nothing to stop), or for a terminal another incident's
        workflow stopped (``_decide`` answers NO_ACTION for it).
        """

        existing = self.store.get_incident_by_event(self._passive_event_key(event))
        if existing is not None:
            return existing.workflow_request_id
        if not event.is_failure or not event.workload_ids:
            return None
        passive_incident_id, _ = failure_containment_ids(self._passive_event_key(event))
        if event.termination_initiator_incident_id not in (None, passive_incident_id):
            return None
        _incident, workflow, _created = self._ensure_containment(
            cluster_id=event.cluster_id,
            job_id=event.job_id,
            attempt_id=event.attempt_id,
            runtime_profile_version=event.runtime_profile_version,
            workload_ids=list(event.workload_ids),
            node_ids=sorted({item.node_id for item in event.allocation}),
            gpu_uuids=sorted(
                {gpu for item in event.allocation for gpu in item.gpu_uuids}
            ),
            reasons=[
                f"terminal {event.terminal_status.value} reported before "
                "failure detection",
                *[
                    f"rank {item.rank} exit_code={item.exit_code}"
                    for item in event.rank_exit_status
                    if item.exit_code
                ],
            ],
        )
        return workflow.request_id

    def _containment_context(
        self, event: TerminalEvent
    ) -> tuple[FaultIncident | None, FaultIncident | None]:
        """The explicitly named initiator incident (if any) and the passive
        containment incident (if any) for this attempt.

        The passive incident is looked up by the attempt, not only through
        the terminal's initiator annotation: the DESTR-015 tombstone race can
        lose that annotation on a job our own containment stopped, and the
        store knows better than the tombstone.
        """

        explicit_initiator: FaultIncident | None = None
        if event.termination_initiator_incident_id:
            try:
                explicit_initiator = self.store.get_incident(
                    event.termination_initiator_incident_id
                )
            except NotFoundError:
                explicit_initiator = None
        passive_containment = (
            explicit_initiator
            if (
                explicit_initiator is not None
                and explicit_initiator.event_type == "TRAINING_ATTEMPT_FAILURE_DETECTED"
            )
            else self.store.get_incident_by_event(self._passive_event_key(event))
        )
        return explicit_initiator, passive_containment

    def _decide(
        self,
        event: TerminalEvent,
        explicit_initiator: FaultIncident | None,
        passive_containment: FaultIncident | None,
        *,
        predecessor_workflow_id: str | None = None,
    ) -> CompletionDecision:
        """Decide a terminal event. Runs inside the completion transaction.

        Returns the decision. Explicit foreign initiator → NO_ACTION; user stop
        or success → NO_ACTION and withdraw; trusted marker → marker plan; no
        allocation → evidence + escalate; otherwise one budgeted restart. A
        STOPPED terminal is a user stop only when no passive containment
        exists for the attempt: with one, the stop was ours, tagged or not.
        Every compiled plan chains behind ``predecessor_workflow_id`` (the
        containment workflow) so the dispatcher orders stop-then-restart.
        """

        if event.termination_initiator_incident_id and (
            explicit_initiator is None
            or explicit_initiator.event_type != "TRAINING_ATTEMPT_FAILURE_DETECTED"
        ):
            return CompletionDecision(
                cluster_id=event.cluster_id,
                attempt_id=event.attempt_id,
                event_key=event.event_key,
                status=DecisionStatus.NO_ACTION,
                reason=(
                    "termination was initiated by incident "
                    f"{event.termination_initiator_incident_id}; "
                    "continue that workflow without recursive recovery"
                ),
            )

        passive_failure_stop = (
            passive_containment is not None
            and passive_containment.event_type == "TRAINING_ATTEMPT_FAILURE_DETECTED"
        )
        if (event.terminal_status.value == "STOPPED" and not passive_failure_stop) or (
            not event.is_failure and not passive_failure_stop
        ):
            reason = (
                "controller-initiated or user stop; no automatic restart"
                if event.terminal_status.value == "STOPPED"
                else "training attempt succeeded"
            )
            self._withdraw_job_workflows(event, reason)
            return CompletionDecision(
                cluster_id=event.cluster_id,
                attempt_id=event.attempt_id,
                event_key=event.event_key,
                status=DecisionStatus.NO_ACTION,
                reason=reason,
            )

        matched = self._live_matching_markers(event)
        if matched:
            selected = max(
                matched,
                key=lambda item: recovery_action_sort_key(
                    item.recommended_action or RecoveryAction.QUARANTINE
                ),
            )
            profile = self.store.get_profile(event.runtime_profile_version)
            try:
                incident = self.store.get_incident(selected.incident_id)
            except NotFoundError:
                incident = None
            if (
                incident is not None
                and incident.workflow_request_id
                and self._workflow_owns_workload_restart(incident.workflow_request_id)
            ):
                return CompletionDecision(
                    cluster_id=event.cluster_id,
                    attempt_id=event.attempt_id,
                    event_key=event.event_key,
                    status=DecisionStatus.NO_ACTION,
                    reason=(
                        "workload recovery is already owned by "
                        "the existing incident workflow; observe "
                        "that workflow without a second restart"
                    ),
                    matched_marker_ids=[item.marker_id for item in matched],
                )
            plan = (
                self.planner.after_incident(event, incident, profile)
                if incident is not None
                else self.planner.from_marker(event, selected, profile)
            )
            plan = self._save_plan(
                plan, event, predecessor_workflow_id=predecessor_workflow_id
            )
            return CompletionDecision(
                cluster_id=event.cluster_id,
                attempt_id=event.attempt_id,
                event_key=event.event_key,
                status=DecisionStatus.PLAN_CREATED,
                reason=(
                    "trusted allocation marker matched; "
                    + (
                        "node remediation is owned by the existing "
                        "incident and only workload recovery was planned"
                        if incident is not None
                        else "reused existing incident action"
                    )
                ),
                matched_marker_ids=[item.marker_id for item in matched],
                recovery_plan_id=plan.plan_id,
            )

        if not event.allocation:
            profile = self.store.get_profile(event.runtime_profile_version)
            plan = self.planner.from_missing_allocation(event, profile)
            plan = self._save_plan(
                plan, event, predecessor_workflow_id=predecessor_workflow_id
            )
            return CompletionDecision(
                cluster_id=event.cluster_id,
                attempt_id=event.attempt_id,
                event_key=event.event_key,
                status=DecisionStatus.PLAN_CREATED,
                reason="allocation snapshot is missing; automatic restart is blocked",
                recovery_plan_id=plan.plan_id,
            )

        profile = self.store.get_profile(event.runtime_profile_version)
        plan = self._save_plan(
            self.planner.without_hardware_evidence(event, profile),
            event,
            predecessor_workflow_id=predecessor_workflow_id,
        )
        return CompletionDecision(
            cluster_id=event.cluster_id,
            attempt_id=event.attempt_id,
            event_key=event.event_key,
            status=DecisionStatus.PLAN_CREATED,
            reason=(
                "no trusted marker matched the allocation; restarting once "
                "within the job restart budget"
                if plan.trigger == "no-hardware-evidence:RESTART"
                else "no trusted marker matched and the profile cannot restart "
                "the workload; escalated to an operator"
            ),
            recovery_plan_id=plan.plan_id,
        )

    def _save_plan(
        self,
        plan: RecoveryPlan,
        event: TerminalEvent,
        *,
        predecessor_workflow_id: str | None = None,
    ) -> RecoveryPlan:
        if self.workflow_compiler is not None:
            plan = self.workflow_compiler.compile(
                plan, event, predecessor_workflow_id=predecessor_workflow_id
            )
        self.store.save_plan(plan)
        return plan

    def _withdraw_job_workflows(self, event: TerminalEvent, reason: str) -> None:
        """Tell the workflows repairing this job that the job is gone (F-N1 §7).

        The workflow winds down: in-flight actions finish, touched nodes are
        released, nothing new starts and the job is not restarted. Recorded
        through ``amend_workflow`` so the executor's stale copy is fenced.
        """

        try:
            pairs = self.store.list_active_workflow_incidents(
                event.cluster_id, job_id=event.job_id
            )
        except Exception:  # noqa: BLE001 - the decision must still be recorded
            LOGGER.exception(
                "could not look up job workflows to withdraw: job=%s", event.job_id
            )
            return
        now = datetime.now(timezone.utc)
        for incident, workflow in pairs:
            if incident.attempt_id not in (None, event.attempt_id):
                continue
            if workflow.workload_withdrawn_at is not None or not any(
                step.operation is WorkflowOperation.RESTART_WORKLOAD
                for step in workflow.official_steps
            ):
                continue
            try:
                self.store.amend_workflow(
                    workflow.request_id,
                    {
                        "workload_withdrawn_at": now,
                        "workload_withdrawn_reason": reason,
                    },
                )
            except Exception:  # noqa: BLE001 - one workflow must not block the rest
                LOGGER.exception(
                    "could not withdraw workflow %s after job %s stopped",
                    workflow.request_id,
                    event.job_id,
                )
                continue
            LOGGER.warning(
                "workflow %s withdrawn: job %s attempt %s ended (%s)",
                workflow.request_id,
                event.job_id,
                event.attempt_id,
                reason,
            )

    def _workflow_owns_workload_restart(self, workflow_request_id: str) -> bool:
        workflow = self.store.get_workflow(workflow_request_id)
        return any(
            step.operation is WorkflowOperation.RESTART_WORKLOAD
            for step in workflow.official_steps
        )

    def _live_matching_markers(self, event: TerminalEvent) -> list[NodeMarker]:
        """Matching markers, minus those whose repair was for another attempt.

        A marker whose incident was recovered (workflow SUCCEEDED) *and* whose
        incident names a different attempt says nothing about why this attempt
        died; matching it planned a same-allocation restart with no triage,
        inside the marker TTL (F-G6 / P2-50H). Such a marker is retired here
        (``active=False``) so it stops matching and stops disqualifying the
        node as a spare. An incident about this attempt, or one that names no
        attempt, still matches: that is the pinned after-incident restart.

        A diagnostic marker (see ``marker_is_diagnostic``) that names a stored
        incident is skipped: it observes the node, it does not repair it.
        """
        live: list[NodeMarker] = []
        for marker in self._matching_markers(event):
            if not marker.incident_id:
                live.append(marker)
                continue
            try:
                incident = self.store.get_incident(marker.incident_id)
            except NotFoundError:
                live.append(marker)
                continue
            if marker_is_diagnostic(marker):
                LOGGER.info(
                    "diagnostic marker %s of incident %s does not own attempt %s",
                    marker.marker_id,
                    incident.incident_id,
                    event.attempt_id,
                )
                continue
            if (
                incident.attempt_id is None
                or incident.attempt_id == event.attempt_id
                or not incident.workflow_request_id
            ):
                live.append(marker)
                continue
            try:
                workflow = self.store.get_workflow(incident.workflow_request_id)
            except NotFoundError:
                live.append(marker)
                continue
            if workflow.status is not WorkflowStatus.SUCCEEDED:
                live.append(marker)
                continue
            retire_markers_for_incident(
                self.store,
                incident.incident_id,
                reason=(
                    f"incident recovered for attempt {incident.attempt_id}; "
                    f"terminal of attempt {event.attempt_id} goes to triage"
                ),
                retired_by="completion-service",
            )
        return live

    def _matching_markers(self, event: TerminalEvent) -> list[NodeMarker]:
        """Actionable markers that correlate with a finished training attempt.

        Scope and the ``marker_window`` are pushed into the Store: the marker
        table accumulates an entry per observation per node, so scanning it once
        per terminal event made the cost of correlating one attempt grow with the
        whole fleet's history. Only the expiry test stays here, because it is
        relative to this event rather than to now.
        """

        nodes = {item.node_id for item in event.allocation}
        gpus = {gpu for item in event.allocation for gpu in item.gpu_uuids}
        fabrics = {
            item.fabric_partition for item in event.allocation if item.fabric_partition
        }
        candidates = self.store.list_markers_in_scope_window(
            node_ids=nodes,
            gpu_uuids=gpus,
            fabric_partitions=fabrics,
            observed_from=event.ended_at - self.marker_window,
            observed_to=event.ended_at + self.marker_window,
        )
        return [marker for marker in candidates if marker.expires_at >= event.ended_at]
