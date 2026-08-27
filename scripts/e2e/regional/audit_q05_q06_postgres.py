from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

from gpu_fault.execution import (
    ProductionExecutorConfig,
    ProductionWorkflowExecutor,
    WorkflowStepOutcome,
)
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.store import PostgresStore


class AuditAdapter:
    owner = "audit-owner"

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == self.owner

    def execute(self, context):
        if context.step.operation is WorkflowOperation.REPLACE_NODE:
            return WorkflowStepOutcome.succeeded(
                details={"node_rebindings": {"audit-node-old": "audit-node-spare"}}
            )
        if context.step.operation in {
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.RESTART_NODE,
        }:
            return WorkflowStepOutcome.waiting(
                operation_id=(f"audit/{context.step.operation.value}")
            )
        return WorkflowStepOutcome.succeeded()


def block_leased(
    store: PostgresStore,
    workflow: WorkflowRequest,
    owner: str,
) -> None:
    blocked = workflow.model_copy(
        update={
            "status": WorkflowStatus.BLOCKED,
            "blocked_reasons": ["audit completed"],
            "execution_owner_id": None,
            "execution_lease_expires_at": None,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    store.save_workflow_if_leased(
        blocked,
        owner,
        workflow.execution_epoch,
    )


def run_q06_audit(
    store: PostgresStore,
    executor: ProductionWorkflowExecutor,
    *,
    stamp: str,
    now: datetime,
    later: datetime,
    owner: str,
) -> None:
    incident = FaultIncident(
        incident_id=f"audit-q06-inc-{stamp}",
        event_id=f"audit-q06-event-{stamp}",
        event_type="AUDIT",
        cluster_id="audit-cluster",
        node_ids=["audit-node-old"],
        policy_version="audit",
        policy_source="AUDIT",
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
        created_at=now,
        updated_at=now,
    )
    workflow = WorkflowRequest(
        request_id=f"audit-q06-wf-{stamp}",
        incident_id=incident.incident_id,
        runtime_profile_version="audit",
        status=WorkflowStatus.PENDING,
        fencing_token=1,
        dag_enabled=True,
        dag_revision=1,
        not_before=later,
        official_steps=[
            WorkflowStepSpec(
                operation=WorkflowOperation.REPLACE_NODE,
                execution_owner="audit-owner",
                node_ids=["audit-node-old"],
            ),
            WorkflowStepSpec(
                operation=WorkflowOperation.VALIDATE_GPU,
                execution_owner="audit-owner",
                node_ids=["audit-node-old"],
                depends_on_step_indexes=[0],
            ),
        ],
        created_at=now,
        updated_at=now,
    )
    incident = incident.model_copy(update={"workflow_request_id": workflow.request_id})
    store.save_incident(incident)
    store.save_workflow(workflow)
    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=1),
    )
    saved = store.get_workflow(workflow.request_id)
    saved_incident = store.get_incident(incident.incident_id)
    print(
        "q06",
        result.status.value,
        saved_incident.node_ids,
        result.waiting_step_index,
    )
    assert result.status is WorkflowStatus.RUNNING
    assert saved_incident.node_ids == ["audit-node-spare"]
    assert saved.official_steps[1].node_ids == ["audit-node-spare"]
    block_leased(store, saved, owner)


def main() -> None:
    stamp = str(int(time.time()))
    now = datetime.now(timezone.utc)
    later = now + timedelta(hours=1)
    owner = f"audit-q056-{stamp}"
    store = PostgresStore(
        os.environ["GPU_FAULT_STORE_URL"],
        pool_min_size=1,
        pool_max_size=2,
        pool_timeout_seconds=5,
    )
    executor = ProductionWorkflowExecutor(
        store,
        [AuditAdapter()],
        ProductionExecutorConfig(
            enabled=True,
            executor_id=owner,
            allowed_operations=frozenset(
                {
                    WorkflowOperation.REPLACE_NODE,
                    WorkflowOperation.VALIDATE_GPU,
                    WorkflowOperation.QUIESCE_GPU_SERVICES,
                    WorkflowOperation.RESET_GPU,
                    WorkflowOperation.RESTORE_GPU_SERVICES,
                    WorkflowOperation.RESTART_NODE,
                }
            ),
            workflow_preemption_enabled=True,
        ),
    )
    try:
        run_q06_audit(
            store,
            executor,
            stamp=stamp,
            now=now,
            later=later,
            owner=owner,
        )

        q05_incident = FaultIncident(
            incident_id=f"audit-q05-inc-{stamp}",
            event_id=f"audit-q05-event-{stamp}",
            event_type="AUDIT",
            cluster_id="audit-cluster",
            node_ids=["audit-node"],
            policy_version="audit",
            policy_source="AUDIT",
            state=IncidentState.ACTION_PENDING,
            fencing_token=1,
            created_at=now,
            updated_at=now,
        )
        predecessor = WorkflowRequest(
            request_id=f"audit-q05-pred-{stamp}",
            incident_id=q05_incident.incident_id,
            runtime_profile_version="audit",
            status=WorkflowStatus.RUNNING,
            fencing_token=1,
            not_before=later,
            official_steps=[
                WorkflowStepSpec(
                    operation=WorkflowOperation.QUIESCE_GPU_SERVICES,
                    execution_owner="audit-owner",
                    node_ids=["audit-node"],
                ),
                WorkflowStepSpec(
                    operation=WorkflowOperation.RESET_GPU,
                    execution_owner="audit-owner",
                    node_ids=["audit-node"],
                    gpu_uuids=["GPU-a"],
                ),
                WorkflowStepSpec(
                    operation=WorkflowOperation.RESTORE_GPU_SERVICES,
                    execution_owner="audit-owner",
                    node_ids=["audit-node"],
                ),
            ],
            completed_step_indexes=[0],
            completed_operations=[WorkflowOperation.QUIESCE_GPU_SERVICES],
            step_executions=[
                WorkflowStepExecution(
                    step_index=0,
                    operation=(WorkflowOperation.QUIESCE_GPU_SERVICES),
                    status=WorkflowStepStatus.SUCCEEDED,
                    adapter_operation_id="audit-quiesce",
                    details={
                        "agent_generations": {"audit-node": 1},
                        "maintenance_window_expires_at": (
                            now + timedelta(minutes=5)
                        ).isoformat(),
                    },
                )
            ],
            created_at=now,
            updated_at=now,
        )
        successor = WorkflowRequest(
            request_id=f"audit-q05-succ-{stamp}",
            incident_id=q05_incident.incident_id,
            predecessor_workflow_id=predecessor.request_id,
            preempt_predecessor=True,
            preemption_reason="audit stronger reboot",
            runtime_profile_version="audit",
            status=WorkflowStatus.PENDING,
            fencing_token=1,
            not_before=later,
            official_steps=[
                WorkflowStepSpec(
                    operation=WorkflowOperation.RESTART_NODE,
                    execution_owner="audit-owner",
                    node_ids=["audit-node"],
                )
            ],
            created_at=now,
            updated_at=now,
        )
        q05_incident = q05_incident.model_copy(
            update={"workflow_request_id": successor.request_id}
        )
        store.save_incident(q05_incident)
        store.save_workflow(predecessor)
        store.save_workflow(successor)

        predecessor_result = executor.execute(
            predecessor.request_id,
            WorkflowExecutionRequest(expected_fencing_token=1),
        )
        before_claim = store.get_workflow(successor.request_id)
        print(
            "q05-before-claim",
            predecessor_result.status.value,
            len(before_claim.official_steps),
            before_claim.quiesce_handoff_from_workflow_id,
        )
        assert predecessor_result.status is WorkflowStatus.SUPERSEDED
        assert len(before_claim.official_steps) == 1
        assert before_claim.quiesce_handoff_from_workflow_id is None

        successor_result = executor.execute(
            successor.request_id,
            WorkflowExecutionRequest(expected_fencing_token=1),
        )
        after_claim = store.get_workflow(successor.request_id)
        print(
            "q05-after-claim",
            successor_result.status.value,
            [step.operation.value for step in after_claim.official_steps],
            [step.depends_on_step_indexes for step in after_claim.official_steps],
            after_claim.quiesce_handoff_from_workflow_id,
        )
        assert successor_result.status is WorkflowStatus.RUNNING
        assert [step.operation for step in after_claim.official_steps] == [
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        ]
        assert after_claim.official_steps[1].depends_on_step_indexes == [0]
        assert after_claim.quiesce_handoff_from_workflow_id == predecessor.request_id
        block_leased(store, after_claim, owner)
        store.save_incident(
            store.get_incident(q05_incident.incident_id).model_copy(
                update={
                    "state": IncidentState.ESCALATED,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
