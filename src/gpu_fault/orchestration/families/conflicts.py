from __future__ import annotations

from collections.abc import Iterator

from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    lifetime_exceeded,
    resolved_step_indexes,
)
from gpu_fault.operation_registry import (
    NODE_EXCLUSIVE_OPERATIONS,
    OPERATION_RESOURCE_CLAIMS,
    TRANSIENT_GPU_INVENTORY_OPERATIONS,
)
from gpu_fault.orchestration.workflow_merge import never_executed_operator_block
from gpu_fault.store import NotFoundError

#: Operations during which the node's GPU device plugin is stopped or being
#: replaced on purpose: the quiesce stops its container, the restore starts it,
#: the plugin restart deletes its Pod. A ``gpu_kubernetes_allocatable_mismatch``
#: raised while one of them runs is the workflow's own side effect, not a new
#: fault (live 2026-10-05, gpu-eks-cluster: the quiesce of a RESET_GPU chain
#: raised it, the planner queued a RESTART_GPU_DEVICE_PLUGIN successor on the
#: same incident, and that successor took the incident pointer -- the reset's
#: QUARANTINED verdict never reached the incident).
#: The eks-managed quiesce taint (NoExecute) evicts every device plugin that
#: does not tolerate it -- the EFA plugin included (live, gpu-eks-cluster: the
#: quiesce raised ``efa_kubernetes_allocatable_mismatch`` first, the drain route
#: queued a RESTART_EFA_DEVICE_PLUGIN successor behind the RUNNING reset, and
#: the GPU finding that followed merged into that PENDING successor instead of
#: the running incumbent).
DEVICE_PLUGIN_UNAVAILABLE_OPERATIONS = frozenset(
    {
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
    }
)
#: Finding -> the operations of a RUNNING node-exclusive workflow that make the
#: finding transient while they run (validated by the workflow's own validation
#: step, see ``INVENTORY_FINDING_VALIDATIONS``).
TRANSIENT_INVENTORY_FINDINGS = {
    "gpu_inventory_mismatch": TRANSIENT_GPU_INVENTORY_OPERATIONS,
    "gpu_kubernetes_allocatable_mismatch": (
        TRANSIENT_GPU_INVENTORY_OPERATIONS | DEVICE_PLUGIN_UNAVAILABLE_OPERATIONS
    ),
    "efa_inventory_mismatch": TRANSIENT_GPU_INVENTORY_OPERATIONS,
    "efa_kubernetes_allocatable_mismatch": (
        TRANSIENT_GPU_INVENTORY_OPERATIONS | DEVICE_PLUGIN_UNAVAILABLE_OPERATIONS
    ),
}
#: The validation step of the incumbent that must still be ahead for the finding
#: to be absorbed: a GPU finding is judged by VALIDATE_GPU; an EFA one by
#: VALIDATE_FABRIC when the plan has it, else by the GPU validation that ends a
#: reset chain (the EFA plugin returns with the quiesce taint's removal, and a
#: mismatch that outlives the chain is raised again by the next scan).
_GPU_VALIDATIONS = frozenset({WorkflowOperation.VALIDATE_GPU})
_EFA_VALIDATIONS = _GPU_VALIDATIONS | {WorkflowOperation.VALIDATE_FABRIC}
INVENTORY_FINDING_VALIDATIONS = {
    "gpu_inventory_mismatch": _GPU_VALIDATIONS,
    "gpu_kubernetes_allocatable_mismatch": _GPU_VALIDATIONS,
    "efa_inventory_mismatch": _EFA_VALIDATIONS,
    "efa_kubernetes_allocatable_mismatch": _EFA_VALIDATIONS,
}


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
            if incident is not None and never_executed_operator_block(workflow):
                # C-03: the pair stays the merge target so ``disposition`` can
                # replace the never-run block in place under its own id,
                # instead of leaving it behind as a second record on the node.
                return incident, workflow
            return None, None
        return incident, workflow

    @staticmethod
    def claims_node_exclusively(
        steps: list[WorkflowStepSpec],
    ) -> bool:
        return any(step.operation in NODE_EXCLUSIVE_OPERATIONS for step in steps)

    def active_node_exclusive_workflows(
        self,
        cluster_id: str,
        node_ids: set[str],
        *,
        exclude_request_ids: frozenset[str] = frozenset(),
        candidate_steps: list[WorkflowStepSpec] | None = None,
    ) -> Iterator[WorkflowRequest]:
        """Every active node-exclusive workflow on ``node_ids``, store order.

        A node can carry more than one: a RUNNING incumbent and the PENDING
        successor queued behind it. ``active_node_exclusive_workflow`` names the
        first; a caller that needs the one actually executing iterates.
        """

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
                # C-03: a BLOCKED(NEEDS_OPERATOR) row that never ran a step
                # holds nothing on the node; naming it as the incumbent made
                # every successor's ``predecessor_workflow_id`` point at a row
                # the dispatcher treats as open forever (F-A4).
                or never_executed_operator_block(workflow)
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
            yield workflow

    def active_node_exclusive_workflow(
        self,
        cluster_id: str,
        node_ids: set[str],
        *,
        exclude_request_ids: frozenset[str] = frozenset(),
        candidate_steps: list[WorkflowStepSpec] | None = None,
    ) -> WorkflowRequest | None:
        return next(
            self.active_node_exclusive_workflows(
                cluster_id,
                node_ids,
                exclude_request_ids=exclude_request_ids,
                candidate_steps=candidate_steps,
            ),
            None,
        )

    def active_workflow_covers_inventory_finding(
        self,
        finding: NodeHealthFinding,
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        covering = TRANSIENT_INVENTORY_FINDINGS.get(finding.metric_name or "")
        if covering is None:
            return None
        validations = INVENTORY_FINDING_VALIDATIONS[finding.metric_name or ""]
        # The RUNNING incumbent, not the first active row: a PENDING successor
        # already queued on the node (live, gpu-eks-cluster: the EFA plugin
        # restart the quiesce taint provoked) hid the running reset, so the GPU
        # finding merged into the successor and replaced its plan -- the
        # generation bump that fenced every later command of the reset.
        for workflow in self.active_node_exclusive_workflows(
            finding.cluster_id, {finding.node_id}
        ):
            if workflow.status is not WorkflowStatus.RUNNING:
                continue
            operations = {step.operation for step in workflow.official_steps}
            if not operations.intersection(covering):
                continue
            validation = {
                index
                for index, step in enumerate(workflow.official_steps)
                if step.operation in validations
            }
            # A superseded validation will never run either (F-C2).
            if not validation - set(resolved_step_indexes(workflow)):
                continue
            try:
                incident = self.store.get_incident(workflow.incident_id)
            except NotFoundError:
                # Listed a moment ago, gone now. "No incumbent" is the safe
                # answer: the finding takes the normal path and at worst
                # repeats one exclusivity check, instead of failing the whole
                # batch (P2-51H).
                return None
            return incident, workflow
        return None
