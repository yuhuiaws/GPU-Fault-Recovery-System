from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.fleet import AgentLifecycleState, AgentRecord
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
IDENTITY_KEYS = {
    "agent_protocol_version",
    "node_action_key_version",
    "agent_version",
    "artifact_sha256",
    "compatibility_digest",
    "installer_bundle_sha256",
    "policy_version",
    "runtime_profile_version",
    "config_digest",
    "allowed_operations",
}


def synthetic_agent(
    template: AgentRecord,
    *,
    cluster_id: str,
    node_id: str,
    run_id: str,
    now: datetime,
) -> AgentRecord:
    return template.model_copy(
        update={
            "cluster_id": cluster_id,
            "node_id": node_id,
            "endpoint": "http://127.0.0.1:9",
            "boot_id": f"boot-{run_id}-{node_id}",
            "node_instance_id": f"instance-{run_id}-{node_id}",
            "agent_incarnation_id": f"incarnation-{run_id}-{node_id}",
            "retired_incarnation_ids": [],
            "first_seen_at": now,
            "last_seen_at": now,
            "lease_expires_at": now + timedelta(minutes=30),
            "generation": 1,
            "lifecycle_state": AgentLifecycleState.ACTIVE,
            "transition_id": None,
            "transition_reason": None,
            "transition_started_at": None,
        }
    )


def release_bound_synthetic_agent(
    identity: Mapping[str, object],
    *,
    cluster_id: str,
    node_id: str,
    run_id: str,
    now: datetime,
) -> AgentRecord:
    if set(identity) != IDENTITY_KEYS:
        raise ValueError("release-bound synthetic Agent identity fields are invalid")
    raw_operations = identity["allowed_operations"]
    if not isinstance(raw_operations, list) or not raw_operations:
        raise ValueError("release-bound synthetic Agent operations are invalid")
    return AgentRecord(
        cluster_id=cluster_id,
        node_id=node_id,
        endpoint="http://127.0.0.1:9",
        agent_protocol_version=int(str(identity["agent_protocol_version"])),
        node_action_key_version=int(str(identity["node_action_key_version"])),
        agent_version=str(identity["agent_version"]),
        artifact_sha256=str(identity["artifact_sha256"]),
        compatibility_digest=str(identity["compatibility_digest"]),
        installer_bundle_sha256=str(identity["installer_bundle_sha256"]),
        policy_version=str(identity["policy_version"]),
        runtime_profile_version=str(identity["runtime_profile_version"]),
        config_digest=str(identity["config_digest"]),
        allowed_operations=[
            WorkflowOperation(str(operation)) for operation in raw_operations
        ],
        boot_id=f"boot-{run_id}-{node_id}",
        node_instance_id=f"instance-{run_id}-{node_id}",
        agent_incarnation_id=f"incarnation-{run_id}-{node_id}",
        first_seen_at=now,
        last_seen_at=now,
        lease_expires_at=now + timedelta(minutes=30),
        generation=1,
        lifecycle_state=AgentLifecycleState.ACTIVE,
    )


def agent_source(
    context: ApplicationContext,
    now: datetime,
) -> tuple[AgentRecord | None, Mapping[str, object] | None, str]:
    raw_identity = os.getenv("ACTION_AGENT_IDENTITY_JSON")
    if raw_identity:
        identity = json.loads(raw_identity)
        if not isinstance(identity, dict):
            raise RuntimeError(
                "release-bound synthetic Agent identity is not an object"
            )
        return None, identity, "release-state"
    live_agents = [
        agent
        for agent in context.store.list_agents()
        if not agent.cluster_id.startswith("perf-cap-")
        and agent.lifecycle_state is AgentLifecycleState.ACTIVE
        and agent.lease_expires_at is not None
        and agent.lease_expires_at > now
    ]
    if not live_agents:
        raise RuntimeError(
            "action capacity seed requires one live production Agent identity"
        )
    return max(live_agents, key=lambda item: item.last_seen_at), None, "live-agent"


def save_synthetic_agent(
    context: ApplicationContext,
    *,
    template: AgentRecord | None,
    identity: Mapping[str, object] | None,
    cluster_id: str,
    node_id: str,
    run_id: str,
    now: datetime,
) -> None:
    if identity is not None:
        record = release_bound_synthetic_agent(
            identity,
            cluster_id=cluster_id,
            node_id=node_id,
            run_id=run_id,
            now=now,
        )
    elif template is not None:
        record = synthetic_agent(
            template,
            cluster_id=cluster_id,
            node_id=node_id,
            run_id=run_id,
            now=now,
        )
    else:
        raise RuntimeError("synthetic Agent identity source is missing")
    lease_seconds = int(os.getenv("ACTION_AGENT_LEASE_SECONDS", "1800"))
    record = record.model_copy(
        update={"lease_expires_at": now + timedelta(seconds=lease_seconds)}
    )
    context.store.save_agent(record)


def integrated_node_id(run_id: str, cluster_index: int, workflow_index: int) -> str:
    return f"integrated-node-{run_id}-c{cluster_index:03d}-w{workflow_index:02d}"


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
    seed_mode = os.getenv("ACTION_SEED_MODE", "workflows")
    context = ApplicationContext.from_environment()
    created = 0
    agents_created = 0
    now = datetime.now(timezone.utc)
    try:
        agent_template, identity, identity_source = agent_source(context, now)
        if identity is not None:
            runtime_profile_version = str(identity["runtime_profile_version"])
        elif agent_template is not None:
            runtime_profile_version = agent_template.runtime_profile_version
        else:
            runtime_profile_version = ""
        if not runtime_profile_version:
            raise RuntimeError("synthetic Agent runtime Profile is missing")
        if seed_mode in {"correlated-agents", "integrated-agents"}:
            workflows_per_cluster = int(os.getenv("ACTION_WORKFLOWS_PER_CLUSTER", "2"))
            for cluster_index in range(clusters):
                cluster_id = f"perf-cap-{cluster_index:03d}"
                lanes = (
                    ("preempt", "reset")
                    if seed_mode == "correlated-agents"
                    else tuple(
                        f"w{index:02d}" for index in range(workflows_per_cluster)
                    )
                )
                for lane in lanes:
                    node_id = (
                        f"corr-node-{run_id}-c{cluster_index:03d}-{lane}"
                        if seed_mode == "correlated-agents"
                        else integrated_node_id(
                            run_id,
                            cluster_index,
                            int(lane.removeprefix("w")),
                        )
                    )
                    save_synthetic_agent(
                        context,
                        template=agent_template,
                        identity=identity,
                        cluster_id=cluster_id,
                        node_id=node_id,
                        run_id=run_id,
                        now=now,
                    )
                    agents_created += 1
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "clusters": clusters,
                        "agents_created": agents_created,
                        "agent_identity_source": identity_source,
                        "runtime_profile_version": runtime_profile_version,
                    },
                    sort_keys=True,
                )
            )
            return
        if seed_mode != "workflows":
            raise RuntimeError(f"unsupported action seed mode: {seed_mode}")
        workflows_per_cluster = int(os.environ["ACTION_WORKFLOWS_PER_CLUSTER"])
        nodes_per_workflow = int(os.environ["ACTION_NODES_PER_WORKFLOW"])
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
                for node_id in nodes:
                    save_synthetic_agent(
                        context,
                        template=agent_template,
                        identity=identity,
                        cluster_id=cluster_id,
                        node_id=node_id,
                        run_id=run_id,
                        now=now,
                    )
                    agents_created += 1
                workload_id = f"training/PyTorchJob/{scope}"
                workflow = WorkflowRequest(
                    request_id=request_id,
                    incident_id=incident_id,
                    runtime_profile_version=runtime_profile_version,
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
                "agents_created": agents_created,
                "agent_identity_source": identity_source,
                "runtime_profile_version": runtime_profile_version,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
