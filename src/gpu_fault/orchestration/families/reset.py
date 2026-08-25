from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.operation_registry import WORKLOAD_SCOPED_OPERATIONS
from gpu_fault.policy import (
    ActionDisposition,
    DistributedXidBatch,
    FaultPolicyDecision,
)


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
                return (
                    existing,
                    self.store.get_workflow(existing.workflow_request_id),
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
                gpu_uuid
                for node_id in affected_nodes
                for gpu_uuid in affected_gpus[node_id]
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
            profile = self.store.get_profile(profile_version)
            operations = self.builder.official_operations(representative, decisions[0])
            official_steps, errors = self.builder.compile_steps(
                operations,
                profile,
                affected_nodes,
                gpu_uuids,
                workload_ids,
            )
            incident_id = f"incident-{uuid4()}"
            now = datetime.now(timezone.utc)
            incident = FaultIncident(
                incident_id=incident_id,
                event_id=batch.batch_id,
                event_type="XID_BATCH",
                cluster_id=representative.cluster_id,
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
            errors.extend(safety_errors)
            workflow = WorkflowRequest(
                incident_id=incident_id,
                runtime_profile_version=profile_version,
                status=(
                    WorkflowStatus.PENDING
                    if not errors
                    else WorkflowStatus.SAFETY_PENDING
                ),
                official_action="RESET_GPU",
                fencing_token=1,
                safety_steps=safety_steps,
                official_steps=scoped_steps,
                blocked_reasons=errors,
                created_at=now,
                updated_at=now,
            )
            incident = incident.model_copy(
                update={
                    "state": (
                        IncidentState.ACTION_PENDING
                        if workflow.status is WorkflowStatus.PENDING
                        else IncidentState.SAFETY_PENDING
                    ),
                    "workflow_request_id": workflow.request_id,
                }
            )
            self.store.save_workflow(workflow)
            self.store.save_incident(incident)
            self.store.link_event_to_incident(batch.batch_id, incident.incident_id)
            for event in batch.events:
                self.store.link_event_to_incident(event.event_id, incident.incident_id)
            return incident, workflow
