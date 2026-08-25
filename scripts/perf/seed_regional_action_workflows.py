from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)


KUBERNETES_OWNER = "gpu-fault-kubernetes-adapter"
NODE_OWNER = "gpu-fault-node-agent"
HYPERPOD_OWNER = "gpu-fault-hyperpod-adapter"


def workflow_steps(
    *,
    nodes: list[str],
    workload_id: str,
    run_id: str,
) -> list[WorkflowStepSpec]:
    common = {
        "node_ids": nodes,
        "parameters": {
            "action_capacity_test": True,
            "run_id": run_id,
        },
    }
    return [
        WorkflowStepSpec(
            operation=WorkflowOperation.MARK_UNSCHEDULABLE,
            execution_owner=KUBERNETES_OWNER,
            **common,
        ),
        WorkflowStepSpec(
            operation=WorkflowOperation.CHECKPOINT_WORKLOADS,
            execution_owner=KUBERNETES_OWNER,
            workload_ids=[workload_id],
            depends_on_step_indexes=[0],
            **common,
        ),
        WorkflowStepSpec(
            operation=WorkflowOperation.STOP_WORKLOADS,
            execution_owner=KUBERNETES_OWNER,
            workload_ids=[workload_id],
            depends_on_step_indexes=[1],
            **common,
        ),
        WorkflowStepSpec(
            operation=WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            execution_owner=NODE_OWNER,
            depends_on_step_indexes=[2],
            branch_id="diagnostics",
            **common,
        ),
        WorkflowStepSpec(
            operation=WorkflowOperation.QUIESCE_GPU_SERVICES,
            execution_owner=NODE_OWNER,
            depends_on_step_indexes=[2],
            branch_id="mutation",
            **common,
        ),
        WorkflowStepSpec(
            operation=WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            execution_owner=NODE_OWNER,
            depends_on_step_indexes=[3, 4],
            **common,
        ),
        WorkflowStepSpec(
            operation=WorkflowOperation.RESET_GPU,
            execution_owner=NODE_OWNER,
            gpu_uuids=[f"GPU-{index:02d}" for index in range(len(nodes))],
            depends_on_step_indexes=[5],
            **common,
        ),
        WorkflowStepSpec(
            operation=WorkflowOperation.RESTORE_GPU_SERVICES,
            execution_owner=NODE_OWNER,
            depends_on_step_indexes=[6],
            **common,
        ),
        WorkflowStepSpec(
            operation=WorkflowOperation.RESTART_NODE,
            execution_owner=HYPERPOD_OWNER,
            depends_on_step_indexes=[7],
            **common,
        ),
        WorkflowStepSpec(
            operation=WorkflowOperation.RESTORE_SCHEDULING,
            execution_owner=KUBERNETES_OWNER,
            depends_on_step_indexes=[8],
            **common,
        ),
    ]


def main() -> None:
    run_id = os.environ["ACTION_RUN_ID"]
    clusters = int(os.environ["ACTION_CLUSTERS"])
    workflows_per_cluster = int(os.environ["ACTION_WORKFLOWS_PER_CLUSTER"])
    nodes_per_workflow = int(os.environ["ACTION_NODES_PER_WORKFLOW"])
    context = ApplicationContext.from_environment()
    created = 0
    now = datetime.now(timezone.utc)
    try:
        for cluster_index in range(clusters):
            cluster_id = f"perf-cap-{cluster_index:03d}"
            for workflow_index in range(workflows_per_cluster):
                scope = (
                    f"actionperf-{run_id}-c{cluster_index:03d}-w{workflow_index:03d}"
                )
                incident_id = f"incident-{scope}"
                request_id = f"workflow-{scope}"
                nodes = [
                    f"{scope}-node-{index:02d}" for index in range(nodes_per_workflow)
                ]
                workload_id = f"training/PyTorchJob/{scope}"
                workflow = WorkflowRequest(
                    request_id=request_id,
                    incident_id=incident_id,
                    runtime_profile_version="hyperpod-v1",
                    status=WorkflowStatus.PENDING,
                    official_action="ACTION_CAPACITY_TEST",
                    fencing_token=1,
                    dag_enabled=True,
                    dag_revision=1,
                    official_steps=workflow_steps(
                        nodes=nodes,
                        workload_id=workload_id,
                        run_id=run_id,
                    ),
                    created_at=now,
                    updated_at=now,
                )
                incident = FaultIncident(
                    incident_id=incident_id,
                    event_id=f"event-{scope}",
                    event_type="ACTION_CAPACITY_TEST",
                    cluster_id=cluster_id,
                    node_ids=nodes,
                    policy_version="action-capacity-v1",
                    policy_source="CONTROLLED_DRILL",
                    official_action="ACTION_CAPACITY_TEST",
                    state=IncidentState.ACTION_PENDING,
                    workflow_request_id=request_id,
                    drill_id=run_id,
                    created_at=now,
                    updated_at=now,
                )
                context.store.save_incident_and_workflow(incident, workflow)
                created += 1
    finally:
        context.store.close()
    print(
        json.dumps(
            {
                "run_id": run_id,
                "clusters": clusters,
                "workflows_per_cluster": workflows_per_cluster,
                "nodes_per_workflow": nodes_per_workflow,
                "steps_per_workflow": 10,
                "workflows_created": created,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
