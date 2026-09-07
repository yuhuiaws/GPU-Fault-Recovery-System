from __future__ import annotations

from collections.abc import Callable

from typing import NoReturn

import logging
from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    DiagnosticRequest,
    EffectiveRuntimeProfile,
    FaultIncident,
    IncidentState,
    NodeMarker,
    RecoveryAction,
    RecoveryPlan,
    TerminalEvent,
    TriageFinding,
    TriageOutcome,
    TriageReport,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    recovery_action_sort_key,
)
from gpu_fault.markers import retire_markers_for_incident
from gpu_fault.planner import PlanBuilder
from gpu_fault.ports import DiagnosticPort
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

QUICK_CHECKS = [
    "gpu-enumeration",
    "dcgm-passive-health",
    "xid-sxid-window",
    "ecc-row-remap",
    "nvlink-pcie",
    "rdma-network",
    "host-mce-oom",
]
LOGGER = logging.getLogger(__name__)


class CompletionPendingError(RuntimeError):
    pass


class CompletionService:
    def __init__(
        self,
        store: ControlPlaneStore,
        diagnostics: DiagnosticPort,
        planner: PlanBuilder | None = None,
        workflow_compiler=None,
        marker_window: timedelta = timedelta(minutes=10),
        evidence_service: EvidenceService | None = None,
        marker_ttl_seconds: int | None = None,
        pending_triage_deadline: timedelta = timedelta(minutes=15),
    ) -> None:
        if marker_window <= timedelta(0):
            raise ValueError("marker_window must be positive")
        if pending_triage_deadline <= timedelta(0):
            raise ValueError("pending_triage_deadline must be positive")
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
        self.diagnostics = diagnostics
        self.planner = planner or PlanBuilder()
        self.workflow_compiler = workflow_compiler
        self.marker_window = marker_window
        self.marker_ttl_seconds = marker_ttl_seconds
        self.evidence_service = evidence_service
        # How long a decision may wait in PENDING_TRIAGE before
        # ``reconcile_pending_triage`` expires it into a conservative plan.
        self.pending_triage_deadline = pending_triage_deadline
        # No process lock (F-G2 / P1-62G): active-active replicas serialize
        # on ``store.completion_transaction(event_key)`` instead, which is
        # also what makes the event row and its decision one write.

    def add_marker(self, marker: NodeMarker) -> NodeMarker:
        self.store.add_marker(marker)
        return marker

    def handle_failure_detected(
        self, event: FailureDetectedEvent
    ) -> FailureContainmentDecision:
        if not event.runtime_profile_version:
            raise ValueError("failure detection requires runtime profile version")
        self.store.get_profile(event.runtime_profile_version)
        if not event.workload_ids:
            raise ValueError("failure detection requires owning workload IDs")
        incident_id, workflow_id = failure_containment_ids(event.event_key)

        def build() -> tuple[FaultIncident, WorkflowRequest]:
            now = datetime.now(timezone.utc)
            workflow = WorkflowRequest(
                request_id=workflow_id,
                incident_id=incident_id,
                runtime_profile_version=(event.runtime_profile_version),
                status=WorkflowStatus.PENDING,
                official_action="STOP_WORKLOAD",
                fencing_token=1,
                official_steps=[
                    WorkflowStepSpec(
                        operation=(WorkflowOperation.FREEZE_EVIDENCE),
                        execution_owner=("gpu-fault-control-plane"),
                        node_ids=event.node_ids,
                        gpu_uuids=event.gpu_uuids,
                        workload_ids=event.workload_ids,
                    ),
                    WorkflowStepSpec(
                        operation=(WorkflowOperation.STOP_WORKLOADS),
                        execution_owner=("gpu-fault-kubernetes-adapter"),
                        node_ids=event.node_ids,
                        gpu_uuids=event.gpu_uuids,
                        workload_ids=event.workload_ids,
                        parameters={"termination_initiator_incident_id": (incident_id)},
                    ),
                ],
                created_at=now,
                updated_at=now,
            )
            incident = FaultIncident(
                incident_id=incident_id,
                event_id=event.event_key,
                event_type=("TRAINING_ATTEMPT_FAILURE_DETECTED"),
                cluster_id=event.cluster_id,
                node_ids=event.node_ids,
                gpu_uuids=event.gpu_uuids,
                policy_version="passive-containment-v1",
                policy_source="completion-watcher",
                official_action="STOP_WORKLOAD",
                effective_action=RecoveryAction.STOP_WORKLOAD,
                state=IncidentState.ACTION_PENDING,
                workflow_request_id=workflow_id,
                reasons=[
                    event.reason,
                    (f"first_failed_rank={event.first_failed_rank}"),
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
                ],
                created_at=now,
                updated_at=now,
            )
            return incident, workflow

        incident, workflow, created = self.store.create_incident_workflow_if_absent(
            event.event_key, build
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
        explicit_initiator, passive_containment = self._containment_gate(event)
        existing = self.store.get_decision_by_event(event.event_key)
        if existing is not None:
            return existing.model_copy(update={"duplicate": True})
        self.store.get_profile(event.runtime_profile_version)

        # Everything persisted about this event -- the event row, the plan
        # with its incident and workflow, the diagnostic request, the
        # decision -- is written inside one store transaction keyed by the
        # event (F-G2 (3)(5)). On PostgreSQL that is one advisory lock and
        # one commit: a second replica deciding the same event waits, and a
        # crash mid-way leaves no event row without a decision (P0-48B).
        # The only remote call, quick-triage submission, comes after it.
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
            decision, triage_request = self._decide(
                event, explicit_initiator, passive_containment
            )
            self.store.save_decision(decision)
        if triage_request is None:
            return decision
        return self._submit_quick_triage(decision, triage_request)

    def _containment_gate(
        self, event: TerminalEvent
    ) -> tuple[FaultIncident | None, FaultIncident | None]:
        """Hold the terminal event while its passive containment is open.

        Returns the explicitly named initiator incident (if any) and the
        passive containment incident (if any). Raises
        ``CompletionPendingError`` -- a 409 to the data plane, which retries
        -- while the containment workflow is still open.
        """

        expected_passive_incident, _ = failure_containment_ids(
            f"{event.cluster_id}/{event.attempt_id}/TrainingAttemptFailureDetected"
        )
        explicit_initiator = None
        if event.termination_initiator_incident_id:
            try:
                explicit_initiator = self.store.get_incident(
                    event.termination_initiator_incident_id
                )
            except NotFoundError:
                explicit_initiator = None
        if (
            event.termination_initiator_incident_id == expected_passive_incident
            and explicit_initiator is None
        ):
            raise CompletionPendingError(
                "passive containment incident is not persisted yet: "
                f"{expected_passive_incident}"
            )
        passive_containment = (
            explicit_initiator
            if (
                explicit_initiator is not None
                and explicit_initiator.event_type == "TRAINING_ATTEMPT_FAILURE_DETECTED"
            )
            else self.store.get_incident_by_event(
                f"{event.cluster_id}/{event.attempt_id}/TrainingAttemptFailureDetected"
            )
        )
        if (
            passive_containment is not None
            and passive_containment.event_type == "TRAINING_ATTEMPT_FAILURE_DETECTED"
            and passive_containment.workflow_request_id
        ):
            containment = self.store.get_workflow(
                passive_containment.workflow_request_id
            )
            if containment.status not in {
                WorkflowStatus.SUCCEEDED,
                WorkflowStatus.FAILED,
                WorkflowStatus.SUPERSEDED,
            }:
                # Open (or BLOCKED awaiting an operator): the terminal
                # event is retried later. A containment that FAILED or
                # was SUPERSEDED will never succeed; answering 409 forever
                # queued the whole cluster's terminal events behind it
                # (P0-62C).
                raise CompletionPendingError(
                    "passive containment workflow is not complete: "
                    f"{containment.request_id}/{containment.status.value}"
                )
        return explicit_initiator, passive_containment

    def _decide(
        self,
        event: TerminalEvent,
        explicit_initiator: FaultIncident | None,
        passive_containment: FaultIncident | None,
    ) -> tuple[CompletionDecision, DiagnosticRequest | None]:
        """Decide a terminal event. Runs inside the completion transaction.

        Returns the decision and, when quick triage is needed, the diagnostic
        request to submit *after* the transaction commits. The request is
        persisted here so the decision's ``diagnostic_request_id`` resolves
        (and ages) even if the submission itself fails.
        """

        if event.termination_initiator_incident_id and (
            explicit_initiator is None
            or explicit_initiator.event_type != "TRAINING_ATTEMPT_FAILURE_DETECTED"
        ):
            return (
                CompletionDecision(
                    cluster_id=event.cluster_id,
                    attempt_id=event.attempt_id,
                    event_key=event.event_key,
                    status=DecisionStatus.NO_ACTION,
                    reason=(
                        "termination was initiated by incident "
                        f"{event.termination_initiator_incident_id}; "
                        "continue that workflow without recursive recovery"
                    ),
                ),
                None,
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
            return (
                CompletionDecision(
                    cluster_id=event.cluster_id,
                    attempt_id=event.attempt_id,
                    event_key=event.event_key,
                    status=DecisionStatus.NO_ACTION,
                    reason=reason,
                ),
                None,
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
                return (
                    CompletionDecision(
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
                    ),
                    None,
                )
            plan = (
                self.planner.after_incident(event, incident, profile)
                if incident is not None
                else self.planner.from_marker(event, selected, profile)
            )
            plan = self._save_plan(plan, event)
            return (
                CompletionDecision(
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
                ),
                None,
            )

        if not event.allocation:
            profile = self.store.get_profile(event.runtime_profile_version)
            plan = self.planner.from_missing_allocation(event, profile)
            plan = self._save_plan(plan, event)
            return (
                CompletionDecision(
                    cluster_id=event.cluster_id,
                    attempt_id=event.attempt_id,
                    event_key=event.event_key,
                    status=DecisionStatus.PLAN_CREATED,
                    reason=(
                        "allocation snapshot is missing; triage is "
                        "inconclusive and automatic restart is blocked"
                    ),
                    recovery_plan_id=plan.plan_id,
                ),
                None,
            )

        node_ids = sorted({item.node_id for item in event.allocation})
        request = DiagnosticRequest(
            cluster_id=event.cluster_id,
            attempt_id=event.attempt_id,
            node_ids=node_ids,
            checks=QUICK_CHECKS,
        )
        self.store.save_diagnostic(request)
        return (
            CompletionDecision(
                cluster_id=event.cluster_id,
                attempt_id=event.attempt_id,
                event_key=event.event_key,
                status=DecisionStatus.PENDING_TRIAGE,
                reason=(
                    "no trusted marker matched the allocation; quick triage requested"
                ),
                diagnostic_request_id=request.request_id,
            ),
            request,
        )

    def _submit_quick_triage(
        self, decision: CompletionDecision, request: DiagnosticRequest
    ) -> CompletionDecision:
        """Submit quick triage after the PENDING_TRIAGE decision is committed.

        A failed submission is logged, not raised: the decision is already
        durable and ``reconcile_pending_triage`` owns it from here. Re-raising
        would only make the data plane redeliver an event that is now a
        duplicate.
        """

        try:
            operation_id = self.diagnostics.submit(request)
        except Exception:
            LOGGER.exception(
                "quick triage submission failed for %s; the decision stays "
                "PENDING_TRIAGE until reconcile_pending_triage expires it",
                decision.event_key,
            )
            return decision
        if operation_id != request.request_id:
            LOGGER.warning(
                "diagnostic adapter returned operation %s for request %s; the "
                "decision keeps the request id",
                operation_id,
                request.request_id,
            )
        # ``result`` is the seam of adapters that diagnose synchronously
        # (the DCGM adapter); asynchronous ones report via /v1/triage-results.
        result = getattr(self.diagnostics, "result", None)
        report = result(operation_id) if result is not None else None
        return self.handle_triage(report) if report is not None else decision

    def handle_triage(self, report: TriageReport) -> CompletionDecision:
        request = self.store.get_diagnostic(report.request_id)
        if (
            report.cluster_id is not None and request.cluster_id != report.cluster_id
        ) or request.attempt_id != report.attempt_id:
            self._reject_triage(
                report, "triage report cluster/attempt does not match request"
            )
        if request.cluster_id is None:
            self._reject_triage(report, "diagnostic request has no cluster identity")
        report = report.model_copy(update={"cluster_id": request.cluster_id})
        event_key = f"{request.cluster_id}/{report.attempt_id}/TrainingAttemptTerminal"
        with self.store.completion_transaction(event_key):
            decision = self.store.get_decision_by_attempt(
                request.cluster_id, report.attempt_id
            )
            if decision.status is DecisionStatus.PLAN_CREATED:
                return decision.model_copy(update={"duplicate": True})
            if decision.status is not DecisionStatus.PENDING_TRIAGE:
                self._reject_triage(report, "attempt is not waiting for quick triage")

            expected_nodes = set(request.node_ids)
            report_nodes = {item.node_id for item in report.findings}
            if not report_nodes.issubset(expected_nodes):
                self._reject_triage(
                    report, "triage report contains nodes outside the allocation"
                )

            event = self.store.get_event_by_attempt(
                request.cluster_id, report.attempt_id
            )
            profile = self.store.get_profile(event.runtime_profile_version)
            plan = self.planner.from_triage(event, report.findings, profile)
            self.store.save_triage_report(report)
            plan = self._save_plan(plan, event)
            updated = decision.model_copy(
                update={
                    "status": DecisionStatus.PLAN_CREATED,
                    "reason": f"quick triage resolved as {plan.trigger}",
                    "recovery_plan_id": plan.plan_id,
                }
            )
            self.store.save_decision(updated)
            return updated

    @staticmethod
    def _reject_triage(report: TriageReport, message: str) -> NoReturn:
        # A rejected report is a 422 the data plane does not retry; without
        # this line the control plane kept no trace of it (P1-62F (c)).
        LOGGER.error(
            "triage report rejected: %s (request=%s attempt=%s)",
            message,
            report.request_id,
            report.attempt_id,
        )
        raise ValueError(message)

    def reconcile_pending_triage(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> list[CompletionDecision]:
        """Expire decisions stuck in PENDING_TRIAGE (F-G2 (4) / P1-48E).

        A decision leaves PENDING_TRIAGE only when a triage report arrives.
        If it never does -- diagnostic pod evicted, report rejected with a
        422 the data plane does not retry, submission failed -- the attempt
        neither restarts nor escalates and nothing shows it. After
        ``pending_triage_deadline`` each such decision is closed with the
        conservative plan of ``from_missing_allocation`` (collect evidence,
        escalate to an operator, automatic restart blocked); a synchronous
        adapter's late result is applied instead when it is available.
        """

        observed = now or datetime.now(timezone.utc)
        stale = self.store.list_decisions_by_status(
            DecisionStatus.PENDING_TRIAGE,
            older_than=observed - self.pending_triage_deadline,
            limit=limit,
        )
        resolved: list[CompletionDecision] = []
        for decision in stale:
            try:
                updated = self._expire_pending_triage(decision, observed)
            except Exception:  # noqa: BLE001 - one attempt must not block the rest
                LOGGER.exception(
                    "could not expire PENDING_TRIAGE decision %s", decision.event_key
                )
                continue
            if updated is not None:
                resolved.append(updated)
        return resolved

    def _expire_pending_triage(
        self, decision: CompletionDecision, now: datetime
    ) -> CompletionDecision | None:
        # The diagnostics adapter is read *before* the transaction: the body of
        # a store transaction must be pure computation, and an HTTP adapter
        # here held the ``completion/<key>`` advisory lock and a pooled
        # connection for its full latency (store review 2026-09-07, item D).
        # The decision is read once outside for the request id, then re-read
        # under the lock, which decides whether the pre-fetched report is used.
        current = self.store.get_decision_by_event(decision.event_key)
        if current is None or current.status is not DecisionStatus.PENDING_TRIAGE:
            return None
        report: TriageReport | None = None
        result: Callable[[str], TriageReport | None] | None = getattr(
            self.diagnostics, "result", None
        )
        if result is not None and current.diagnostic_request_id:
            report = result(current.diagnostic_request_id)
        with self.store.completion_transaction(decision.event_key):
            current = self.store.get_decision_by_event(decision.event_key)
            if current is None or current.status is not DecisionStatus.PENDING_TRIAGE:
                return None
            if report is not None:
                # ``handle_triage`` writes, so it stays inside the transaction.
                try:
                    return self.handle_triage(report)
                except ValueError:
                    LOGGER.exception(
                        "late triage result for %s is unusable; expiring",
                        decision.event_key,
                    )
            event = self.store.get_event_by_attempt(
                current.cluster_id, current.attempt_id
            )
            profile = self.store.get_profile(event.runtime_profile_version)
            deadline = int(self.pending_triage_deadline.total_seconds())
            reason = (
                f"quick triage did not report within {deadline}s; expired by the "
                "control plane, automatic restart blocked"
            )
            self._record_triage_timeout(current, reason, now)
            plan = self._save_plan(self._timeout_plan(event, profile, reason), event)
            updated = current.model_copy(
                update={
                    "status": DecisionStatus.PLAN_CREATED,
                    "reason": reason,
                    "recovery_plan_id": plan.plan_id,
                }
            )
            self.store.save_decision(updated)
            LOGGER.warning(
                "PENDING_TRIAGE decision %s expired into plan %s",
                decision.event_key,
                plan.plan_id,
            )
            return updated

    def _record_triage_timeout(
        self, decision: CompletionDecision, reason: str, now: datetime
    ) -> None:
        """Leave an INCONCLUSIVE report for the request that never reported,
        so the audit trail shows why the plan exists."""

        if not decision.diagnostic_request_id:
            return
        try:
            request = self.store.get_diagnostic(decision.diagnostic_request_id)
        except NotFoundError:
            return
        if not request.node_ids:
            return
        self.store.save_triage_report(
            TriageReport(
                request_id=request.request_id,
                cluster_id=request.cluster_id,
                attempt_id=request.attempt_id,
                findings=[
                    TriageFinding(
                        node_id=node_id,
                        outcome=TriageOutcome.INCONCLUSIVE,
                        reason=reason,
                    )
                    for node_id in request.node_ids
                ],
                completed_at=now,
            )
        )

    def _timeout_plan(
        self, event: TerminalEvent, profile: EffectiveRuntimeProfile, reason: str
    ) -> RecoveryPlan:
        # Same conservative shape as a missing allocation: collect evidence,
        # escalate, no automatic restart. A control-plane timeout is not
        # evidence against the nodes, so nothing is quarantined.
        plan = self.planner.from_missing_allocation(event, profile)
        return plan.model_copy(
            update={
                "trigger": "quick-triage:TIMEOUT",
                "steps": [
                    step.model_copy(
                        update={"parameters": {**step.parameters, "reason": reason}}
                    )
                    for step in plan.steps
                ],
            }
        )

    def _save_plan(self, plan: RecoveryPlan, event: TerminalEvent) -> RecoveryPlan:
        if self.workflow_compiler is not None:
            plan = self.workflow_compiler.compile(plan, event)
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
