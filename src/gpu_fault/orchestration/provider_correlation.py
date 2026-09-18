"""Separate provider evidence association from execution identity and replay."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    bounded_reasons,
)
from gpu_fault.orchestration.workflow_merge import (
    WorkflowMergeService,
    workflow_is_mutable,
)
from gpu_fault.policy import FaultPolicyDecision, SxidEvent, XidEvent
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.watcher import AttemptObservation

if TYPE_CHECKING:
    from gpu_fault.store import InMemoryStore, PostgresStore, SqliteStore


def covered_correlated_workflow(
    store: InMemoryStore | SqliteStore | PostgresStore,
    event: XidEvent | SxidEvent,
    decision: FaultPolicyDecision,
    *,
    attempt_observation: Callable[[XidEvent | SxidEvent], AttemptObservation | None],
    linked_workflow: Callable[
        [FaultIncident, str], tuple[WorkflowRequest | None] | None
    ],
    build_candidate: Callable[
        [XidEvent | SxidEvent, FaultPolicyDecision], WorkflowRequest
    ],
    merger: WorkflowMergeService,
) -> tuple[FaultIncident, WorkflowRequest] | None:
    if decision.correlated_event_id:
        incident = store.get_incident_by_event(decision.correlated_event_id)
    elif decision.duplicate and isinstance(event, XidEvent):
        try:
            incident = store.get_incident(decision.marker.incident_id)
        except NotFoundError:
            return None
        try:
            previous = store.get_xid_event(incident.event_id)
        except NotFoundError:
            previous = None
        previous_decision = store.get_xid_policy_decision(incident.event_id)
        if previous_decision is not None and (
            previous_decision.event_id != incident.event_id
            or previous_decision.workflow_request_id != incident.workflow_request_id
        ):
            return None
        if not equivalent_xid_observation(
            previous,
            event,
            previous_decision,
            decision,
        ):
            return None
    else:
        return None
    if incident is None or incident.cluster_id != event.cluster_id:
        return None
    with store.completion_transaction(
        f"provider-correlation/{event.cluster_id}/{incident.incident_id}"
    ):
        linked = linked_workflow(incident, event.event_id)
        workflow = linked[0] if linked is not None else None
        event_time = event.source_event_time or event.observed_at
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        if workflow is None or not companion_generation_matches(
            event,
            incident,
            workflow,
            attempt_observation(event),
            event_time=event_time,
        ):
            return None
        if not decision.correlated_event_id and (
            not workflow_is_mutable(workflow)
            or workflow.execution_epoch
            or workflow.completed_operations
            or workflow.superseded_step_indexes
        ):
            return None
        candidate = build_candidate(event, decision)
        if not decision.correlated_event_id and not equivalent_pending_plan(
            workflow, candidate, event, incident.event_id
        ):
            return None
        if candidate.status is WorkflowStatus.PENDING:
            if (
                merger.disposition(
                    workflow,
                    candidate,
                    event.node_id,
                    set(decision.marker.scope.gpu_uuids),
                    allow_job_branch_merge=False,
                )
                != "ABSORB"
            ):
                return None
        elif not identical_unexecuted_companion(workflow, candidate):
            return None
        # Lock incident before workflow. Publish the link only after both full
        # snapshots pass CAS, so a concurrent claim cannot become stale coverage.
        store.save_incident(incident, expected=incident)
        store.save_workflow(workflow, expected=workflow)
        store.save_incident(
            incident, expected=incident, extra_event_ids=[event.event_id]
        )
        return incident, workflow


def equivalent_xid_observation(
    previous: XidEvent | None,
    event: XidEvent,
    previous_decision: FaultPolicyDecision | None,
    decision: FaultPolicyDecision,
) -> bool:
    """Only transport provenance may differ for broad-marker execution reuse."""
    provenance = {
        "event_id",
        "observed_at",
        "source_event_time",
        "source_monotonic_us",
        "collected_at",
        "ingested_at",
        "event_source",
        "raw_message",
        "evidence_ref",
    }
    decision_identity = {
        "event_id",
        "marker",
        "incident_id",
        "workflow_request_id",
        "advisory_notification_id",
        "investigatory_notification_id",
        "duplicate",
        "correlated_event_id",
        "reasons",
    }
    marker_provenance = {
        "marker_id",
        "incident_id",
        "observed_at",
        "source_event_time",
        "source_monotonic_us",
        "collected_at",
        "ingested_at",
        "event_source",
        "expires_at",
        "raw_evidence_ref",
    }
    return (
        previous_decision is not None
        and previous_decision.incident_id == decision.marker.incident_id
        and previous_decision.marker.incident_id == previous_decision.incident_id
        and decision.event_id == event.event_id
        and not event.synthetic
        and previous_decision.marker.trusted
        and decision.marker.trusted
        and (
            (
                previous is not None
                and previous_decision.event_id == previous.event_id
                and not previous.synthetic
                and previous.model_dump(exclude=provenance)
                == event.model_dump(exclude=provenance)
            )
            or (
                previous is None
                and event.event_source == previous_decision.marker.event_source
                and event.source_boot_id is not None
                and event.source_monotonic_us is not None
                and previous_decision.marker.source_monotonic_us is not None
                and not any(
                    (
                        event.job_id,
                        event.attempt_id,
                        event.pod_uid,
                        event.container_id,
                        event.host_pid,
                        event.cgroup_path,
                    )
                )
            )
        )
        and previous_decision.model_dump(exclude=decision_identity)
        == decision.model_dump(exclude=decision_identity)
        and previous_decision.marker.model_dump(exclude=marker_provenance)
        == decision.marker.model_dump(exclude=marker_provenance)
    )


def equivalent_pending_plan(
    workflow: WorkflowRequest,
    candidate: WorkflowRequest,
    event: XidEvent | SxidEvent,
    previous_event_id: str,
) -> bool:
    steps = []
    for step in candidate.official_steps:
        if (
            step.operation is WorkflowOperation.RESTART_WORKLOAD
            and event.attempt_id is None
            and step.parameters.get("source_attempt_id") == event.event_id
        ):
            # Without a watcher attempt the builder uses the transport event id.
            # Normalize only this fallback for comparison, never the stored plan.
            step = step.model_copy(
                update={
                    "parameters": {
                        **step.parameters,
                        "source_attempt_id": previous_event_id,
                    }
                }
            )
        steps.append(step)
    return (
        candidate.status is WorkflowStatus.PENDING
        and workflow.official_action == candidate.official_action
        and workflow.runtime_profile_version == candidate.runtime_profile_version
        and workflow.safety_only == candidate.safety_only
        and workflow.official_steps == steps
        and workflow.safety_steps == candidate.safety_steps
    )


def execution_decision(
    event: XidEvent | SxidEvent,
    decision: FaultPolicyDecision,
    existing: FaultIncident | None,
) -> FaultPolicyDecision:
    """Give a new event its own identity; only the Store chooses its group."""
    incident_id = (
        existing.incident_id
        if existing is not None
        else f"inc-{event.event_id}"
        if decision.duplicate or decision.correlated_event_id
        else decision.marker.incident_id
    )
    return bind_marker_incident(decision, incident_id)


def bind_marker_incident(
    decision: FaultPolicyDecision, incident_id: str
) -> FaultPolicyDecision:
    """Bind execution ownership without discarding the evidence association."""
    if decision.marker.incident_id == incident_id:
        return decision
    return decision.model_copy(
        update={
            "marker": decision.marker.model_copy(update={"incident_id": incident_id}),
            "reasons": bounded_reasons(
                [
                    *decision.reasons,
                    "Provider marker association with incident "
                    f"{decision.marker.incident_id}; execution coverage "
                    "is decided by workflow arbitration",
                ]
            ),
        }
    )


def companion_generation_matches(
    event: XidEvent | SxidEvent,
    incident: FaultIncident,
    workflow: WorkflowRequest,
    observation: AttemptObservation | None,
    *,
    event_time: datetime,
) -> bool:
    job_id = observation.job_id if observation is not None else event.job_id
    attempt_id = observation.attempt_id if observation is not None else event.attempt_id
    started_at = observation.started_at if observation is not None else None
    if started_at is not None and started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    return (
        incident.cluster_id == event.cluster_id
        and (incident.job_id, incident.attempt_id) == (job_id, attempt_id)
        and (started_at is None or event_time >= started_at)
        and workflow.status
        not in {
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.SUPERSEDED,
        }
        and workflow.incident_id == incident.incident_id
        and workflow.fencing_token == incident.fencing_token
        and workflow.runtime_profile_version == event.runtime_profile_version
        and (
            incident.source_boot_id is None
            or event.source_boot_id is None
            or incident.source_boot_id == event.source_boot_id
        )
    )


def identical_unexecuted_companion(
    workflow: WorkflowRequest, candidate: WorkflowRequest
) -> bool:
    """An exact companion may share an identical plan that still fails closed."""
    return (
        workflow.status == candidate.status
        and not workflow.step_executions
        and not workflow.completed_step_indexes
        and not workflow.superseded_step_indexes
        and workflow.execution_owner_id is None
        and workflow.official_steps == candidate.official_steps
        and workflow.safety_steps == candidate.safety_steps
        and workflow.safety_only == candidate.safety_only
    )
