"""The merge disposition vocabulary and the one place that applies it.

``WorkflowMergeService.disposition`` decides how a candidate workflow relates
to the workflow already open for the same node or job. Three ingestion
families used to re-implement how each verdict is applied and drifted apart
(F-B5): only one of them let a stronger queued branch preempt the weaker
one, one dropped the predecessor pointer when it replaced a plan in place,
and all three treated WIDEN_IN_PLACE as a no-op. The verdicts are an enum so
an unknown value is a bug at the boundary rather than a silent successor,
and :class:`DispositionApplier` is the single implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
    resolved_step_indexes,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher

AggregationDeadlines = Callable[[datetime, WorkflowRequest], tuple[datetime, datetime]]
PrepareSuccessor = Callable[[WorkflowRequest, WorkflowRequest], WorkflowRequest]
PreemptBranch = Callable[[WorkflowRequest, WorkflowRequest, str], WorkflowRequest]


class Disposition(StrEnum):
    """How a candidate workflow merges with the workflow already open."""

    # The candidate adds nothing the existing plan does not already do; the
    # existing workflow only has its aggregation window extended.
    ABSORB = "ABSORB"
    # The event is recorded on the incident and nothing is scheduled: a
    # read-only candidate already covered by a running node-mutating action,
    # or an event past the workflow's lifetime / after its workload left.
    ABSORB_RECORD_ONLY = "ABSORB_RECORD_ONLY"
    # Same node, same recovery: pending steps take the candidate's GPUs.
    WIDEN_IN_PLACE = "WIDEN_IN_PLACE"
    # Same for one node branch of a job workflow.
    WIDEN_BRANCH = "WIDEN_BRANCH"
    # The candidate's stronger plan takes over the not-yet-started workflow.
    REPLACE_IN_PLACE = "REPLACE_IN_PLACE"
    # Same for one node branch of a job workflow.
    REPLACE_BRANCH = "REPLACE_BRANCH"
    # A new node joins the job workflow as a parallel branch.
    PARALLEL_BRANCH = "PARALLEL_BRANCH"
    # The candidate runs after the node's current branch finishes.
    QUEUE_BRANCH_SUCCESSOR = "QUEUE_BRANCH_SUCCESSOR"
    # The candidate becomes a successor workflow behind the existing one.
    QUEUE_SUCCESSOR = "QUEUE_SUCCESSOR"


# Dispositions that leave the existing workflow's identity in charge.
MERGING_DISPOSITIONS = frozenset(
    {
        Disposition.ABSORB,
        Disposition.ABSORB_RECORD_ONLY,
        Disposition.WIDEN_BRANCH,
        Disposition.WIDEN_IN_PLACE,
    }
)


def widen_in_place(
    existing: WorkflowRequest,
    candidate: WorkflowRequest,
    node_id: str,
    gpu_uuids: set[str],
    *,
    workload_scoped_operations: frozenset[WorkflowOperation] | set[WorkflowOperation],
) -> list[WorkflowStepSpec]:
    """Give the pending same-node steps of a flat workflow the new GPUs.

    The non-DAG twin of ``DagBrancher.widen_parallel_job_branch``: every
    step that still has to run on ``node_id`` takes the union of its own
    GPUs, the matching candidate step's GPUs and ``gpu_uuids``. Steps that
    completed, were superseded or already have an execution record are left
    alone -- rewriting a step the agent is running would make the ledger
    disagree with the command in flight. Workload-scoped steps do not name
    GPUs.
    """
    resolved = resolved_step_indexes(existing)
    in_flight = {execution.step_index for execution in existing.step_executions}
    candidate_by_operation = {
        step.operation: step
        for step in candidate.official_steps
        if node_id in step.node_ids and step.operation not in workload_scoped_operations
    }
    steps: list[WorkflowStepSpec] = []
    for index, step in enumerate(existing.official_steps):
        if (
            index in resolved
            or index in in_flight
            or node_id not in step.node_ids
            or step.operation in workload_scoped_operations
        ):
            steps.append(step)
            continue
        matching = candidate_by_operation.get(step.operation)
        merged = sorted(
            set(step.gpu_uuids)
            | set(matching.gpu_uuids if matching else [])
            | gpu_uuids
        )
        parameters = dict(step.parameters)
        raw = parameters.get("gpu_uuids_by_node")
        if isinstance(raw, Mapping):
            mapping = {
                str(key): list(values)
                for key, values in raw.items()
                if isinstance(values, list)
            }
            widened_gpus = set(mapping.get(node_id, [])) | set(merged)
            if widened_gpus:
                mapping[node_id] = sorted(widened_gpus)
            parameters["gpu_uuids_by_node"] = mapping
        steps.append(
            step.model_copy(update={"gpu_uuids": merged, "parameters": parameters})
        )
    return steps


@dataclass
class DispositionApplier:
    """Apply a :class:`Disposition` to an (existing, candidate) pair.

    Returns the workflow to persist and the incident whose fields win the
    subsequent incident merge. Every family constructs one of these with its
    own callbacks; the branching, ranking and successor rules are shared.
    """

    arbiter: RecoveryArbiter
    brancher: DagBrancher
    aggregation_deadlines: AggregationDeadlines
    prepare_preempting_successor: PrepareSuccessor
    preempt_parallel_job_branch: PreemptBranch
    workflow_preemption_enabled: bool

    def apply(
        self,
        disposition: Disposition | str,
        *,
        node_id: str,
        candidate: FaultIncident,
        candidate_workflow: WorkflowRequest,
        existing_incident: FaultIncident,
        existing_workflow: WorkflowRequest,
        gpu_uuids: set[str],
        mutable: bool,
        now: datetime,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        verdict = Disposition(disposition)  # ValueError on an unknown value
        match verdict:
            case Disposition.ABSORB_RECORD_ONLY:
                # The caller's incident merge records the event; the
                # workflow gains no step and no successor.
                return existing_workflow, existing_incident
            case Disposition.ABSORB:
                return self._absorb(existing_incident, existing_workflow, mutable, now)
            case Disposition.WIDEN_IN_PLACE:
                steps = widen_in_place(
                    existing_workflow,
                    candidate_workflow,
                    node_id,
                    gpu_uuids,
                    workload_scoped_operations=(
                        self.arbiter.WORKLOAD_SCOPED_OPERATIONS
                    ),
                )
                return (
                    existing_workflow.model_copy(
                        update={"official_steps": steps, "updated_at": now}
                    ),
                    existing_incident,
                )
            case Disposition.WIDEN_BRANCH:
                return (
                    self.brancher.widen_parallel_job_branch(
                        existing_workflow,
                        candidate_workflow,
                        node_id,
                        gpu_uuids,
                    ),
                    existing_incident,
                )
            case Disposition.REPLACE_IN_PLACE:
                return self._replace_in_place(
                    candidate,
                    candidate_workflow,
                    existing_incident,
                    existing_workflow,
                    now,
                )
            case Disposition.PARALLEL_BRANCH:
                workflow = self.brancher.append_parallel_job_branch(
                    existing_workflow,
                    candidate_workflow,
                )
            case Disposition.REPLACE_BRANCH:
                workflow = self.brancher.replace_parallel_job_branch(
                    existing_workflow,
                    candidate_workflow,
                    node_id,
                )
            case Disposition.QUEUE_BRANCH_SUCCESSOR:
                workflow = self._queue_branch_successor(
                    node_id,
                    candidate_workflow,
                    existing_workflow,
                )
            case Disposition.QUEUE_SUCCESSOR:
                return self._successor(
                    candidate,
                    candidate_workflow,
                    existing_incident,
                    existing_workflow,
                    now,
                )
        return workflow, self.winner(
            candidate,
            candidate_workflow,
            existing_incident,
            existing_workflow,
        )

    def winner(
        self,
        candidate: FaultIncident,
        candidate_workflow: WorkflowRequest,
        existing_incident: FaultIncident,
        existing_workflow: WorkflowRequest,
    ) -> FaultIncident:
        if self.arbiter.workflow_recovery_rank(
            candidate_workflow
        ) > self.arbiter.workflow_recovery_rank(existing_workflow):
            return candidate
        return existing_incident

    def _queue_branch_successor(
        self,
        node_id: str,
        candidate: WorkflowRequest,
        existing: WorkflowRequest,
    ) -> WorkflowRequest:
        indexes = self.brancher.node_branch_step_indexes(existing, node_id)
        rank = (
            self.brancher.step_indexes_recovery_rank(existing, indexes)
            if indexes
            else self.arbiter.workflow_recovery_rank(existing)
        )
        if (
            self.workflow_preemption_enabled
            and self.arbiter.workflow_recovery_rank(candidate) > rank
        ):
            return self.preempt_parallel_job_branch(existing, candidate, node_id)
        return self.brancher.append_parallel_job_branch_successor(
            existing,
            candidate,
            node_id,
        )

    def _absorb(
        self,
        incident: FaultIncident,
        workflow: WorkflowRequest,
        mutable: bool,
        now: datetime,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        updates: dict[str, object] = {"updated_at": now}
        if mutable:
            not_before, maximum = self.aggregation_deadlines(now, workflow)
            updates.update(
                {"not_before": not_before, "aggregation_max_deadline": maximum}
            )
        return workflow.model_copy(update=updates), incident

    def _replace_in_place(
        self,
        candidate: FaultIncident,
        candidate_workflow: WorkflowRequest,
        existing_incident: FaultIncident,
        existing_workflow: WorkflowRequest,
        now: datetime,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        not_before, maximum = self.aggregation_deadlines(now, existing_workflow)
        return (
            candidate_workflow.model_copy(
                update={
                    "request_id": existing_workflow.request_id,
                    "incident_id": existing_incident.incident_id,
                    # The plan changes; its place in the serialization chain
                    # and its lifetime do not (F-B5, F-N1).
                    "predecessor_workflow_id": (
                        existing_workflow.predecessor_workflow_id
                    ),
                    "lifetime_deadline_at": existing_workflow.lifetime_deadline_at,
                    "fencing_token": existing_workflow.fencing_token + 1,
                    "not_before": not_before,
                    "aggregation_max_deadline": maximum,
                    "created_at": existing_workflow.created_at,
                    "updated_at": now,
                }
            ),
            candidate,
        )

    def _successor(
        self,
        candidate: FaultIncident,
        candidate_workflow: WorkflowRequest,
        existing_incident: FaultIncident,
        existing_workflow: WorkflowRequest,
        now: datetime,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        workflow = candidate_workflow.model_copy(
            update={
                "incident_id": existing_incident.incident_id,
                "predecessor_workflow_id": existing_workflow.request_id,
                "lifetime_deadline_at": existing_workflow.lifetime_deadline_at,
                "fencing_token": existing_workflow.fencing_token,
                "not_before": None,
                "updated_at": now,
            }
        )
        return (
            self.prepare_preempting_successor(existing_workflow, workflow),
            candidate,
        )
