from __future__ import annotations

import logging
from datetime import datetime, timezone
from threading import RLock

from gpu_fault.execution.config import MAX_DAG_STEPS
from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.operation_registry import (
    SHARED_DAG_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher, mint_branch_id
from gpu_fault.orchestration.families.identity import derived_record_id
from gpu_fault.policy import (
    ActionDisposition,
    DistributedXidBatch,
    FaultPolicyDecision,
)
from gpu_fault.store import NotFoundError

LOGGER = logging.getLogger(__name__)


class ResetOperationService:
    def __init__(self, store, builder, lock: RLock) -> None:
        self.store = store
        self.builder = builder
        self.lock = lock

    def ingest_distributed_xids(
        self,
        batch: DistributedXidBatch,
        decisions: list[FaultPolicyDecision],
    ) -> tuple[FaultIncident, WorkflowRequest]:
        with self.lock:
            existing = self.store.get_incident_by_event(batch.batch_id)
            if existing is not None:
                workflow = self._workflow_if_present(existing.workflow_request_id)
                if workflow is not None:
                    return existing, workflow
                # A re-posted batch whose workflow row is gone used to fail
                # here on every retry (P2-51H). The ids below are derived from
                # the batch id, so building again rewrites the same incident
                # and re-creates its workflow rather than adding a second pair.
                LOGGER.warning(
                    "incident %s for re-posted XID batch %s points at workflow %s "
                    "which is missing; rebuilding the workflow",
                    existing.incident_id,
                    batch.batch_id,
                    existing.workflow_request_id,
                )
            if len(decisions) != len(batch.events):
                raise ValueError("every distributed XID event needs a decision")
            if {item.event_id for item in decisions} != {
                item.event_id for item in batch.events
            }:
                raise ValueError("distributed XID decisions do not match events")
            unsupported = [
                item.event_id
                for item in decisions
                if (
                    item.action is not RecoveryAction.RESET_GPU
                    or item.disposition is not ActionDisposition.EXECUTABLE
                )
            ]
            if unsupported:
                raise ValueError(
                    "distributed reset batch contains non-executable "
                    "XID decisions: " + ",".join(sorted(unsupported))
                )
            affected_gpus: dict[str, list[str]] = {}
            for event in batch.events:
                values = affected_gpus.setdefault(event.node_id, [])
                if event.gpu_uuid not in values:
                    values.append(event.gpu_uuid)
            affected_nodes = sorted(affected_gpus)
            allocation_nodes = list(
                dict.fromkeys(entry.node_id for entry in batch.allocation)
            )
            gpu_uuids = [
                gpu_uuid for node in affected_nodes for gpu_uuid in affected_gpus[node]
            ]
            workload_ids = sorted(set(batch.affected_workload_ids))
            representative = batch.events[0].model_copy(
                update={
                    "job_id": batch.job_id,
                    "affected_workload_ids": workload_ids,
                    "checkpoint_manifest_ref": (batch.checkpoint_manifest_ref),
                }
            )
            profile_version = representative.runtime_profile_version
            profile = None
            profile_errors: list[str] = []
            try:
                profile = self.store.get_profile(profile_version)
            except NotFoundError:
                # Every other family compiles a missing profile into a
                # BLOCKED(NEEDS_OPERATOR) plan; here ``NotFoundError`` left
                # the route as a 500 and the data plane isolated the batch
                # (C-10).
                profile_errors.append(
                    "runtime profile does not exist: " + str(profile_version)
                )
            operations = self.builder.official_operations(representative, decisions[0])
            official_steps, errors = self.builder.compile_steps(
                operations,
                profile,
                affected_nodes,
                gpu_uuids,
                workload_ids,
            )
            incident_id = derived_record_id(
                "incident", "distributed-xid", batch.batch_id
            )
            now = datetime.now(timezone.utc)
            incident = FaultIncident(
                incident_id=incident_id,
                event_id=batch.batch_id,
                event_type="XID_BATCH",
                cluster_id=representative.cluster_id,
                job_id=batch.job_id,
                attempt_id=batch.attempt_id,
                node_ids=affected_nodes,
                gpu_uuids=gpu_uuids,
                policy_version=decisions[0].policy_version,
                policy_source=decisions[0].source.value,
                official_action="RESET_GPU",
                effective_action=RecoveryAction.RESET_GPU,
                drill_id=representative.drill_id,
                reasons=[
                    reason for decision in decisions for reason in decision.reasons
                ],
                created_at=now,
                updated_at=now,
            )
            allocated_gpu_count = len(
                {gpu_uuid for entry in batch.allocation for gpu_uuid in entry.gpu_uuids}
            )
            scoped_steps = []
            for step in official_steps:
                parameters = self.builder.step_parameters(
                    step.operation,
                    representative,
                    incident,
                    decisions[0],
                )
                if step.operation in {
                    WorkflowOperation.QUIESCE_GPU_SERVICES,
                    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                    WorkflowOperation.RESET_GPU,
                    WorkflowOperation.RESTORE_GPU_SERVICES,
                }:
                    parameters = {
                        **parameters,
                        "gpu_uuids_by_node": affected_gpus,
                    }
                if step.operation is WorkflowOperation.RESTART_WORKLOAD:
                    parameters = {
                        **parameters,
                        "job_id": batch.job_id,
                        "source_attempt_id": batch.attempt_id,
                        "source_gpu_count": allocated_gpu_count,
                        "restart_budget": batch.restart_budget,
                    }
                scoped_steps.append(
                    step.model_copy(
                        update={
                            "node_ids": (
                                allocation_nodes
                                if step.operation in WORKLOAD_SCOPED_OPERATIONS
                                else affected_nodes
                            ),
                            "parameters": parameters,
                        }
                    )
                )
            safety_steps, safety_errors = self.builder.compile_steps(
                [
                    WorkflowOperation.FREEZE_EVIDENCE,
                    WorkflowOperation.MARK_UNSCHEDULABLE,
                    WorkflowOperation.QUARANTINE,
                ],
                profile,
                affected_nodes,
                gpu_uuids,
                workload_ids,
            )
            status, errors = self._compilation_outcome(
                scoped_steps,
                safety_steps,
                len(affected_nodes),
                [*profile_errors, *errors],
                safety_errors,
            )
            workflow = WorkflowRequest(
                request_id=derived_record_id(
                    "workflow", "distributed-xid", batch.batch_id
                ),
                incident_id=incident_id,
                runtime_profile_version=profile_version,
                status=status,
                official_action="RESET_GPU",
                fencing_token=1,
                safety_steps=safety_steps,
                official_steps=scoped_steps,
                blocked_reasons=list(dict.fromkeys(errors)),
                safety_only=status is WorkflowStatus.SAFETY_PENDING,
                blocked_kind=(
                    BlockedKind.NEEDS_OPERATOR
                    if status is WorkflowStatus.BLOCKED
                    else None
                ),
                created_at=now,
                updated_at=now,
            )
            if status is WorkflowStatus.PENDING and len(affected_nodes) > 1:
                workflow = self._per_node_dag(workflow, affected_gpus)
            incident = incident.model_copy(
                update={
                    "state": (
                        IncidentState.ACTION_PENDING
                        if workflow.status is WorkflowStatus.PENDING
                        else IncidentState.SAFETY_PENDING
                        if workflow.status is WorkflowStatus.SAFETY_PENDING
                        else IncidentState.ESCALATED
                    ),
                    "workflow_request_id": workflow.request_id,
                }
            )
            # Incident + workflow in one transaction; the store links
            # ``incident.event_id`` (the batch id) itself, the member events are
            # linked below. Two blind autocommit writes let a crash between them
            # leave an orphan workflow (store review 2026-09-07, item B).
            self.store.save_incident_and_workflow(
                incident,
                workflow,
                extra_event_ids=[event.event_id for event in batch.events],
            )
            return incident, workflow

    @staticmethod
    def _compilation_outcome(
        official_steps: list[WorkflowStepSpec],
        safety_steps: list[WorkflowStepSpec],
        node_count: int,
        errors: list[str],
        safety_errors: list[str],
    ) -> tuple[WorkflowStatus, list[str]]:
        expanded_count = sum(
            1 if step.operation in SHARED_DAG_OPERATIONS else node_count
            for step in official_steps
        )
        if node_count > 1 and expanded_count > MAX_DAG_STEPS:
            errors = [
                *errors,
                "DISTRIBUTED_RESET_DAG_STEP_LIMIT: "
                f"expanded plan requires {expanded_count} steps, limit {MAX_DAG_STEPS}; "
                "automatic reset and workload restart withheld; manual recovery required",
            ]
        if not errors:
            status = WorkflowStatus.PENDING
        elif safety_steps and not safety_errors:
            status = WorkflowStatus.SAFETY_PENDING
        else:
            status = WorkflowStatus.BLOCKED
        return status, [*errors, *safety_errors]

    @staticmethod
    def _per_node_dag(
        workflow: WorkflowRequest, affected_gpus: dict[str, list[str]]
    ) -> WorkflowRequest:
        """Use the normal job DAG: one shared STOP, node branches, one restart."""

        brancher = DagBrancher(RecoveryArbiter())
        combined: WorkflowRequest | None = None
        for node_id in sorted(affected_gpus):
            steps = []
            for step in workflow.official_steps:
                shared = step.operation in SHARED_DAG_OPERATIONS
                parameters = dict(step.parameters)
                if not shared and "gpu_uuids_by_node" in parameters:
                    parameters["gpu_uuids_by_node"] = {node_id: affected_gpus[node_id]}
                steps.append(
                    step.model_copy(
                        update={
                            "node_ids": step.node_ids if shared else [node_id],
                            "gpu_uuids": (
                                step.gpu_uuids if shared else affected_gpus[node_id]
                            ),
                            "parameters": parameters,
                            "branch_id": (
                                "join"
                                if step.operation is WorkflowOperation.RESTART_WORKLOAD
                                else "shared"
                                if shared
                                else mint_branch_id([node_id])
                            ),
                            "branch_node_ids": [] if shared else [node_id],
                        }
                    )
                )
            candidate = workflow.model_copy(update={"official_steps": steps})
            combined = (
                candidate
                if combined is None
                else brancher.append_parallel_job_branch(combined, candidate)
            )
        assert combined is not None
        return combined

    def _workflow_if_present(self, request_id: str | None) -> WorkflowRequest | None:
        if not request_id:
            return None
        try:
            workflow: WorkflowRequest = self.store.get_workflow(request_id)
        except NotFoundError:
            return None
        return workflow
