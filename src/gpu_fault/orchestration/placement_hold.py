"""Placement hold: a running attempt observed on a node under repair.

Rule A (F-N1 §8) bounds the wait of a job workflow whose nodes another
incident is repairing. That covered the job workflows that already existed.
A *new* attempt scheduled onto a node whose GPU is mid-repair -- the race
before the cordon lands, or a pod pinned with ``nodeName`` -- is only observed
through ``POST /v1/workload-observations``; nothing looked the node's workflow
up, so the job ran into the repair and the node workflow's
VERIFY_NO_GPU_CLIENTS yielded to the job's GPU processes (the wrong side
yielded).

The hold is a job workflow the orchestrator opens for that attempt. It has
one STOP_WORKLOADS step carrying the controller-initiated marker and waits in
the dispatcher like any job workflow: inside the window it dissolves the
moment the nodes are freed (``WorkflowDispatcher._dissolve_placement_hold``);
past the window ``_fail_node_busy`` keeps the STOP, which executes, and the
workflow ends FAILED with the incident ESCALATED -- the operator signal.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    record_workflow_event,
    resolved_step_indexes,
)
from gpu_fault.operation_registry import (
    NODE_MUTATING_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration.workflow_builder import WorkflowBuilder
from gpu_fault.store import NotFoundError
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.watcher import AttemptObservation, WorkloadPhase

LOGGER = logging.getLogger(__name__)

EVENT_TYPE = "WORKLOAD_PLACED_ON_REPAIRING_NODE"
POLICY_SOURCE = "SITE_PLACEMENT_HOLD"
POLICY_VERSION = "site-placement-hold/v1"
ACTOR = "orchestrator-placement-hold"
# A repair acts on the node's GPU runtime, driver or lifecycle. Containment
# (cordon, taint) and the workload-scoped operations (another job's STOP or
# RESTART, including another hold's own STOP) are not repairs of the node.
REPAIR_OPERATIONS = frozenset(NODE_MUTATING_OPERATIONS - WORKLOAD_SCOPED_OPERATIONS)
_LIVE_PHASES = frozenset({WorkloadPhase.PENDING, WorkloadPhase.RUNNING})
_UNSAFE_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9._:-]+")


def hold_incident_id(cluster_id: str, attempt_id: str) -> str:
    """Deterministic: one hold per attempt, however often it is observed."""

    return _UNSAFE_ID_CHARACTERS.sub("-", f"hold-{cluster_id}-{attempt_id}")


def _restarted_attempt(workflow: WorkflowRequest) -> str | None:
    for execution in reversed(workflow.step_executions):
        if (
            execution.operation is WorkflowOperation.RESTART_WORKLOAD
            and execution.details.get("restart_attempt_id")
        ):
            return str(execution.details["restart_attempt_id"])
    return None


class PlacementHoldService:
    """Open a placement hold for an observed attempt (see module docstring)."""

    def __init__(self, store: ControlPlaneStore, builder: WorkflowBuilder) -> None:
        self.store = store
        self._builder = builder

    def hold(
        self, observation: AttemptObservation
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        """Return the (incident, workflow) pair created now, else ``None``.

        ``None`` covers every "nothing to do": the attempt is not live, it was
        stopped by the control plane (initiator marker), it already has a job
        workflow, its nodes are not under a mutating repair, or the hold
        already exists (deterministic event id; the store's if-absent create
        is the race guard).
        """

        if observation.termination_initiator_incident_id is not None:
            return None
        if observation.workload_phase not in _LIVE_PHASES:
            return None
        nodes = sorted(
            {
                container.node_id
                for container in observation.containers
                if container.node_id and not container.terminated
            }
        )
        if not nodes:
            return None
        if self._attempt_has_job_workflow(observation):
            return None
        repairing = self._repairing_workflows(observation.cluster_id, set(nodes))
        if not repairing:
            return None
        incident_id = hold_incident_id(observation.cluster_id, observation.attempt_id)
        step = self._stop_step(observation, nodes, incident_id)
        if step is None:
            return None
        pair = self._build(observation, nodes, repairing, incident_id, step)
        _, _, created = self.store.create_incident_workflow_if_absent(
            incident_id, lambda: pair
        )
        if not created:
            return None
        LOGGER.warning(
            "placement hold %s opened: attempt %s of job %s runs on nodes under "
            "remediation %s",
            incident_id,
            observation.attempt_id,
            observation.job_id,
            {node: repairing[node] for node in sorted(repairing)},
        )
        return pair

    def _attempt_has_job_workflow(self, observation: AttemptObservation) -> bool:
        for incident, workflow in self.store.list_active_workflow_incidents(
            observation.cluster_id, job_id=observation.job_id
        ):
            if incident.attempt_id in (None, observation.attempt_id):
                return True
            if _restarted_attempt(workflow) == observation.attempt_id:
                return True
        return False

    def _repairing_workflows(self, cluster_id: str, nodes: set[str]) -> dict[str, str]:
        """Node -> id of the open workflow whose unresolved steps repair it."""

        repairing: dict[str, str] = {}
        for incident, workflow in self.store.list_active_workflow_incidents(
            cluster_id, node_ids=nodes
        ):
            resolved = resolved_step_indexes(workflow)
            steps = (
                workflow.safety_steps
                if workflow.executes_safety_steps
                else workflow.official_steps
            )
            for index, step in enumerate(steps):
                if index in resolved or step.operation not in REPAIR_OPERATIONS:
                    continue
                for node in sorted(nodes & set(step.node_ids or incident.node_ids)):
                    repairing.setdefault(node, workflow.request_id)
        return repairing

    def _stop_step(
        self,
        observation: AttemptObservation,
        nodes: list[str],
        incident_id: str,
    ) -> WorkflowStepSpec | None:
        try:
            profile = self.store.get_profile(observation.runtime_profile_version)
        except NotFoundError:
            LOGGER.error(
                "placement hold for attempt %s not opened: runtime profile %s "
                "does not exist",
                observation.attempt_id,
                observation.runtime_profile_version,
            )
            return None
        workload_ids = list(observation.workload_ids) or [observation.job_id]
        gpu_uuids = sorted(
            {
                uuid
                for container in observation.containers
                for uuid in container.gpu_uuids
            }
        )
        steps, errors = self._builder.compile_steps(
            [WorkflowOperation.STOP_WORKLOADS], profile, nodes, gpu_uuids, workload_ids
        )
        if errors or len(steps) != 1:
            LOGGER.error(
                "placement hold for attempt %s not opened: %s",
                observation.attempt_id,
                "; ".join(errors) or "STOP_WORKLOADS did not compile",
            )
            return None
        return steps[0].model_copy(
            update={"parameters": {"termination_initiator_incident_id": incident_id}}
        )

    def _build(
        self,
        observation: AttemptObservation,
        nodes: list[str],
        repairing: dict[str, str],
        incident_id: str,
        step: WorkflowStepSpec,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        now = datetime.now(timezone.utc)
        observed_at = observation.observed_at
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        # A data-plane clock ahead of ours must not shorten the window the
        # dispatcher measures from ``created_at``.
        created_at = min(observed_at, now)
        remediation_ids = sorted(set(repairing.values()))
        request_id = f"workflow-{incident_id}"
        reasons = [
            "attempt observed running on nodes under another remediation: "
            + ", ".join(f"{node} ({repairing[node]})" for node in sorted(repairing)),
            "placement hold: waits one node-busy window, dissolves when the "
            "nodes are freed, otherwise stops the job",
        ]
        incident = FaultIncident(
            incident_id=incident_id,
            event_id=incident_id,
            event_type=EVENT_TYPE,
            event_source="workload-observation",
            cluster_id=observation.cluster_id,
            node_ids=nodes,
            gpu_uuids=step.gpu_uuids,
            job_id=observation.job_id,
            attempt_id=observation.attempt_id,
            workload_identity_source="ATTEMPT_OBSERVATION",
            policy_version=POLICY_VERSION,
            policy_source=POLICY_SOURCE,
            official_action=WorkflowOperation.STOP_WORKLOADS.value,
            effective_action=RecoveryAction.STOP_WORKLOAD,
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=request_id,
            fencing_token=1,
            reasons=reasons,
            created_at=created_at,
            updated_at=now,
        )
        workflow = WorkflowRequest(
            request_id=request_id,
            incident_id=incident_id,
            runtime_profile_version=observation.runtime_profile_version,
            status=WorkflowStatus.PENDING,
            official_action=WorkflowOperation.STOP_WORKLOADS.value,
            fencing_token=1,
            official_steps=[step],
            placement_hold=True,
            created_at=created_at,
            updated_at=now,
        )
        workflow = record_workflow_event(
            workflow,
            WorkflowEventKind.HOLD,
            code=WorkflowEventCode.PLACEMENT_HOLD_OPENED.value,
            reason=reasons[0],
            actor=ACTOR,
            details={
                "reason": WorkflowEventCode.PLACEMENT_HOLD_OPENED.value,
                "remediation_workflow_id": remediation_ids[0],
                "remediation_workflow_ids": remediation_ids,
                "nodes": sorted(repairing),
                "attempt_nodes": nodes,
            },
            at=now,
        )
        return incident, workflow
