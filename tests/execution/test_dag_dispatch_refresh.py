"""An ingestion rewrite observed at a dispatch boundary replaces the ready snapshot."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    RecoveryAction,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.orchestration import IncidentOrchestrator
from tests._builders import (
    active_workflow_executor,
    attempt_observation,
    container_observation,
    node_health_finding,
)
from tests.regional.test_distributed_reset_regional import WORKLOAD, distributed_plan

OP = WorkflowOperation


@pytest.mark.parametrize("mutation", ["retire", "widen", "dependencies"])
def test_ready_steps_follow_the_fresh_compiled_workflow(mutation: str) -> None:
    store, _incident, workflow = distributed_plan()
    store.save_attempt_observation(
        attempt_observation(
            "job",
            "attempt",
            datetime.now(timezone.utc),
            expected_critical_ranks=3,
            workload_ids=[WORKLOAD],
            containers=[
                container_observation(
                    f"pod-{node}",
                    f"worker-{node}",
                    rank,
                    node,
                    gpu_uuids=[f"GPU-{node}"],
                )
                for rank, node in enumerate(("node-a", "node-b", "node-c"))
            ],
            restart_budget=1,
        )
    )
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)
    target = next(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is OP.RESET_GPU and step.node_ids == ["node-b"]
    )
    source_restore = next(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is OP.RESTORE_GPU_SERVICES and step.node_ids == ["node-a"]
    )
    calls: list[WorkflowStepContext] = []
    rewritten = False

    class Boundary:
        def supports(self, step) -> bool:
            return True

        def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
            nonlocal rewritten
            calls.append(context)
            if (
                not rewritten
                and context.step.operation is OP.RESET_GPU
                and context.step.node_ids == ["node-a"]
            ):
                rewritten = True
                if mutation == "retire":
                    _, updated = orchestrator.ingest_node_health(
                        node_health_finding(
                            "efa-finding",
                            "efa-event",
                            node_id="node-b",
                            observed_at=datetime.now(timezone.utc),
                            category=NodeHealthCategory.RDMA,
                            severity="critical",
                            reason="EFA PCI inventory is below the configured invariant",
                            metric_name="efa_inventory_mismatch",
                            recommended_action=RecoveryAction.REBOOT_NODE,
                            gpu_uuids=["GPU-node-b"],
                            runtime_profile_version=workflow.runtime_profile_version,
                            workload_state=WorkloadState.ACTIVE,
                            affected_workload_ids=[WORKLOAD],
                            diagnostic_parameters={"expected_efa_device_count": 1},
                        )
                    )
                    assert (
                        updated is not None
                        and updated.request_id == workflow.request_id
                    )
                    assert target in updated.superseded_step_indexes
                else:
                    current = store.get_workflow(workflow.request_id)
                    steps = list(current.official_steps)
                    step = steps[target]
                    steps[target] = step.model_copy(
                        update=(
                            {"gpu_uuids": [*step.gpu_uuids, "GPU-new"]}
                            if mutation == "widen"
                            else {
                                "depends_on_step_indexes": [
                                    *step.depends_on_step_indexes,
                                    source_restore,
                                ]
                            }
                        )
                    )
                    store.save_workflow(
                        current.model_copy(
                            update={
                                "official_steps": steps,
                                "dag_revision": current.dag_revision + 1,
                            }
                        ),
                        expected=current,
                    )
            return WorkflowStepOutcome.succeeded()

    executor = active_workflow_executor(
        store,
        [Boundary()],
        {
            *(step.operation for step in workflow.official_steps),
            OP.RESTART_NODE,
            OP.QUARANTINE,
            OP.VALIDATE_HOST,
            OP.VALIDATE_FABRIC,
        },
    )
    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )
    saved = store.get_workflow(workflow.request_id)
    target_calls = [call for call in calls if call.step_index == target]

    assert result.status is WorkflowStatus.SUCCEEDED
    assert not set(saved.completed_step_indexes) & set(saved.superseded_step_indexes), (
        "retired work must not be executed and recorded as completed"
    )
    if mutation == "retire":
        assert not target_calls, (
            "the obsolete reset must not cross the adapter boundary"
        )
    elif mutation == "widen":
        assert len(target_calls) == 1 and "GPU-new" in target_calls[0].step.gpu_uuids, (
            "the pending action must consume its newly committed scope"
        )
    else:
        indexes = [call.step_index for call in calls]
        assert indexes.index(source_restore) < indexes.index(target), (
            "a newly committed dependency must finish before dispatch"
        )
