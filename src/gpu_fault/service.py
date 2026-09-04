from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from threading import RLock

from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    DiagnosticRequest,
    FaultIncident,
    IncidentState,
    NodeMarker,
    RecoveryAction,
    TerminalEvent,
    TriageReport,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    recovery_action_sort_key,
)
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
    ) -> None:
        self.store = store
        self.diagnostics = diagnostics
        self.planner = planner or PlanBuilder()
        self.workflow_compiler = workflow_compiler
        self.marker_window = marker_window
        self.evidence_service = evidence_service
        self._lock = RLock()

    def add_marker(self, marker: NodeMarker) -> NodeMarker:
        self.store.add_marker(marker)
        return marker

    def handle_failure_detected(
        self, event: FailureDetectedEvent
    ) -> FailureContainmentDecision:
        with self._lock:
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
                            parameters={
                                "termination_initiator_incident_id": (incident_id)
                            },
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
                                + (
                                    f" ({item.get('s3_uri')})"
                                    if item.get("s3_uri")
                                    else ""
                                )
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
                            "cannot persist emergency workload log "
                            "evidence for attempt %s",
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
        with self._lock:
            expected_passive_incident, _ = failure_containment_ids(
                (
                    f"{event.cluster_id}/{event.attempt_id}/"
                    "TrainingAttemptFailureDetected"
                )
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
                    and explicit_initiator.event_type
                    == "TRAINING_ATTEMPT_FAILURE_DETECTED"
                )
                else self.store.get_incident_by_event(
                    f"{event.cluster_id}/{event.attempt_id}/"
                    "TrainingAttemptFailureDetected"
                )
            )
            if (
                passive_containment is not None
                and passive_containment.event_type
                == "TRAINING_ATTEMPT_FAILURE_DETECTED"
                and passive_containment.workflow_request_id
            ):
                containment = self.store.get_workflow(
                    passive_containment.workflow_request_id
                )
                if containment.status is not WorkflowStatus.SUCCEEDED:
                    raise CompletionPendingError(
                        "passive containment workflow is not complete: "
                        f"{containment.request_id}/"
                        f"{containment.status.value}"
                    )
            existing = self.store.get_decision_by_event(event.event_key)
            if existing is not None:
                return existing.model_copy(update={"duplicate": True})

            self.store.get_profile(event.runtime_profile_version)
            if not self.store.save_event_if_absent(event):
                existing = self.store.get_decision_by_event(event.event_key)
                if existing is None:
                    raise RuntimeError("event exists without completion decision")
                return existing.model_copy(update={"duplicate": True})

            if event.termination_initiator_incident_id and (
                explicit_initiator is None
                or explicit_initiator.event_type != "TRAINING_ATTEMPT_FAILURE_DETECTED"
            ):
                decision = CompletionDecision(
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
                self.store.save_decision(decision)
                return decision

            passive_failure_stop = (
                passive_containment is not None
                and passive_containment.event_type
                == "TRAINING_ATTEMPT_FAILURE_DETECTED"
            )
            if (
                event.terminal_status.value == "STOPPED" and not passive_failure_stop
            ) or (not event.is_failure and not passive_failure_stop):
                reason = (
                    "controller-initiated or user stop; no automatic restart"
                    if event.terminal_status.value == "STOPPED"
                    else "training attempt succeeded"
                )
                decision = CompletionDecision(
                    cluster_id=event.cluster_id,
                    attempt_id=event.attempt_id,
                    event_key=event.event_key,
                    status=DecisionStatus.NO_ACTION,
                    reason=reason,
                )
                self.store.save_decision(decision)
                return decision

            matched = self._matching_markers(event)
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
                    and self._workflow_owns_workload_restart(
                        incident.workflow_request_id
                    )
                ):
                    decision = CompletionDecision(
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
                    self.store.save_decision(decision)
                    return decision
                plan = (
                    self.planner.after_incident(event, incident, profile)
                    if incident is not None
                    else self.planner.from_marker(event, selected, profile)
                )
                plan = self._save_plan(plan, event)
                decision = CompletionDecision(
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
                self.store.save_decision(decision)
                return decision

            if not event.allocation:
                profile = self.store.get_profile(event.runtime_profile_version)
                plan = self.planner.from_missing_allocation(event, profile)
                plan = self._save_plan(plan, event)
                decision = CompletionDecision(
                    cluster_id=event.cluster_id,
                    attempt_id=event.attempt_id,
                    event_key=event.event_key,
                    status=DecisionStatus.PLAN_CREATED,
                    reason=(
                        "allocation snapshot is missing; triage is "
                        "inconclusive and automatic restart is blocked"
                    ),
                    recovery_plan_id=plan.plan_id,
                )
                self.store.save_decision(decision)
                return decision

            node_ids = sorted({item.node_id for item in event.allocation})
            request = DiagnosticRequest(
                cluster_id=event.cluster_id,
                attempt_id=event.attempt_id,
                node_ids=node_ids,
                checks=QUICK_CHECKS,
            )
            operation_id = self.diagnostics.submit(request)
            decision = CompletionDecision(
                cluster_id=event.cluster_id,
                attempt_id=event.attempt_id,
                event_key=event.event_key,
                status=DecisionStatus.PENDING_TRIAGE,
                reason=(
                    "no trusted marker matched the allocation; quick triage requested"
                ),
                diagnostic_request_id=operation_id,
            )
            self.store.save_decision(decision)
            result = getattr(self.diagnostics, "result", None)
            report = result(operation_id) if result is not None else None
            return self.handle_triage(report) if report is not None else decision

    def handle_triage(self, report: TriageReport) -> CompletionDecision:
        with self._lock:
            request = self.store.get_diagnostic(report.request_id)
            if (
                report.cluster_id is not None
                and request.cluster_id != report.cluster_id
            ) or request.attempt_id != report.attempt_id:
                raise ValueError("triage report cluster/attempt does not match request")
            if request.cluster_id is None:
                raise ValueError("diagnostic request has no cluster identity")
            report = report.model_copy(update={"cluster_id": request.cluster_id})
            decision = self.store.get_decision_by_attempt(
                request.cluster_id, report.attempt_id
            )
            if decision.status is DecisionStatus.PLAN_CREATED:
                return decision.model_copy(update={"duplicate": True})
            if decision.status is not DecisionStatus.PENDING_TRIAGE:
                raise ValueError("attempt is not waiting for quick triage")

            expected_nodes = set(request.node_ids)
            report_nodes = {item.node_id for item in report.findings}
            if not report_nodes.issubset(expected_nodes):
                raise ValueError("triage report contains nodes outside the allocation")

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

    def _save_plan(self, plan, event: TerminalEvent):
        if self.workflow_compiler is not None:
            plan = self.workflow_compiler.compile(plan, event)
        self.store.save_plan(plan)
        return plan

    def _workflow_owns_workload_restart(self, workflow_request_id: str) -> bool:
        workflow = self.store.get_workflow(workflow_request_id)
        return any(
            step.operation is WorkflowOperation.RESTART_WORKLOAD
            for step in workflow.official_steps
        )

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
