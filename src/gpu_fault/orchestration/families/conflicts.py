from __future__ import annotations

from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.models import (
    IncidentState,
    lifetime_exceeded,
    resolved_step_indexes,
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.operation_registry import (
    NODE_EXCLUSIVE_OPERATIONS,
    OPERATION_RESOURCE_CLAIMS,
    TRANSIENT_GPU_INVENTORY_OPERATIONS,
)
from gpu_fault.store import NotFoundError


class NodeConflictService:
    EXECUTABLE_STATUSES = frozenset(
        {
            WorkflowStatus.PENDING,
            WorkflowStatus.RUNNING,
            WorkflowStatus.SAFETY_PENDING,
        }
    )
    TERMINAL_STATUSES = frozenset(
        {
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.SUPERSEDED,
        }
    )

    def __init__(self, store, arbiter) -> None:
        self.store = store
        self.arbiter = arbiter

    @staticmethod
    def operation_resource_claims(
        operation: WorkflowOperation,
    ) -> frozenset[str]:
        return OPERATION_RESOURCE_CLAIMS[operation]

    @classmethod
    def steps_resource_claims_by_node(
        cls,
        steps: list[WorkflowStepSpec],
    ) -> dict[str, frozenset[str]]:
        claims: dict[str, set[str]] = {}
        for step in steps:
            for node_id in step.node_ids:
                claims.setdefault(node_id, set()).update(
                    cls.operation_resource_claims(step.operation)
                )
        return {
            node_id: frozenset(values) for node_id, values in claims.items() if values
        }

    @classmethod
    def workflow_resource_claims_by_node(
        cls,
        workflow: WorkflowRequest,
    ) -> dict[str, frozenset[str]]:
        return cls.steps_resource_claims_by_node(workflow.official_steps)

    @classmethod
    def reopen_if_terminal(
        cls,
        incident: FaultIncident | None,
        workflow: WorkflowRequest | None,
    ) -> tuple[FaultIncident | None, WorkflowRequest | None]:
        # Anything that is not PENDING / RUNNING / SAFETY_PENDING will never
        # execute again on its own -- BLOCKED included. Merging a new fault into
        # it absorbed the fault into a record nobody dispatches (P0-56A).
        if workflow is not None and workflow.status not in cls.EXECUTABLE_STATUSES:
            if (
                lifetime_exceeded(workflow)
                and incident is not None
                and incident.state is not IncidentState.RECOVERED
            ):
                # Out of lifetime and with an operator (F-N1): the pair stays
                # the merge target so the event is recorded on this incident
                # instead of opening a new remediation.
                return incident, workflow
            return None, None
        return incident, workflow

    @staticmethod
    def claims_node_exclusively(
        steps: list[WorkflowStepSpec],
    ) -> bool:
        return any(step.operation in NODE_EXCLUSIVE_OPERATIONS for step in steps)

    def active_node_exclusive_workflow(
        self,
        cluster_id: str,
        node_ids: set[str],
        *,
        exclude_request_ids: frozenset[str] = frozenset(),
        candidate_steps: list[WorkflowStepSpec] | None = None,
    ) -> WorkflowRequest | None:
        candidate_claims = (
            self.steps_resource_claims_by_node(candidate_steps)
            if candidate_steps is not None
            else None
        )
        for incident, workflow in self.store.list_active_workflow_incidents(
            cluster_id,
            node_ids=node_ids,
        ):
            if (
                workflow.request_id in exclude_request_ids
                or not self.claims_node_exclusively(workflow.official_steps)
            ):
                continue
            if candidate_claims is not None:
                existing = self.workflow_resource_claims_by_node(workflow)
                if not any(
                    self.arbiter.resource_claims_conflict(
                        existing.get(node_id, frozenset()),
                        candidate_claims.get(node_id, frozenset()),
                    )
                    for node_id in node_ids.intersection(incident.node_ids)
                ):
                    continue
            return workflow
        return None

    def active_workflow_covers_inventory_finding(
        self,
        finding: NodeHealthFinding,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        if finding.metric_name != "gpu_inventory_mismatch":
            return None
        workflow = self.active_node_exclusive_workflow(
            finding.cluster_id,
            {finding.node_id},
        )
        if workflow is None or workflow.status is not WorkflowStatus.RUNNING:
            return None
        operations = {step.operation for step in workflow.official_steps}
        if not operations.intersection(TRANSIENT_GPU_INVENTORY_OPERATIONS):
            return None
        validation = {
            index
            for index, step in enumerate(workflow.official_steps)
            if step.operation is WorkflowOperation.VALIDATE_GPU
        }
        # A superseded validation will never run either (F-C2).
        if not validation - set(resolved_step_indexes(workflow)):
            return None
        try:
            incident = self.store.get_incident(workflow.incident_id)
        except NotFoundError:
            # Listed a moment ago, gone now. "No incumbent" is the safe answer:
            # the finding takes the normal path and at worst repeats one
            # exclusivity check, instead of failing the whole batch (P2-51H).
            return None
        return incident, workflow
