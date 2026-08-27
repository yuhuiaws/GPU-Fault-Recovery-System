from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from gpu_fault.cluster_executor import ClusterActionExecutor
from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import RemoteActionCommand


def _executor(cluster_id: str) -> ClusterActionExecutor:
    executor = object.__new__(ClusterActionExecutor)
    executor.client = SimpleNamespace(cluster_id=cluster_id)
    executor.allowed_namespaces = set()
    executor.fleet_registry = None
    executor.adapters = []
    executor.executor_id = "regional-local-guard-audit"
    executor.unexpected_failures = 0
    return executor


def _command(
    *,
    cluster_id: str,
    command_fencing_token: int,
    workflow_fencing_token: int,
    incident_fencing_token: int,
    suffix: str,
) -> RemoteActionCommand:
    incident = FaultIncident(
        incident_id=f"audit-{suffix}-incident",
        event_id=f"audit-{suffix}-event",
        event_type="REGIONAL_LOCAL_GUARD_AUDIT",
        cluster_id=cluster_id,
        node_ids=["audit-nonexistent-node"],
        policy_version="audit-v1",
        policy_source="audit",
        fencing_token=incident_fencing_token,
    )
    step = WorkflowStepSpec(
        operation=WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
        execution_owner="gpu-fault-node-agent",
        node_ids=["audit-nonexistent-node"],
    )
    workflow = WorkflowRequest(
        request_id=f"audit-{suffix}-workflow",
        incident_id=incident.incident_id,
        status=WorkflowStatus.RUNNING,
        fencing_token=workflow_fencing_token,
        official_steps=[step],
    )
    return RemoteActionCommand(
        command_id=f"audit-{suffix}-command",
        cluster_id=cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=0,
        fencing_token=command_fencing_token,
        idempotency_key=f"audit/{suffix}",
        lease_token=f"audit-{suffix}-lease-token",
        step=step,
        workflow=workflow,
        incident=incident,
    )


def run(cluster_id: str, other_cluster_id: str) -> dict:
    executor = _executor(cluster_id)
    cross_cluster = executor._execute(
        _command(
            cluster_id=other_cluster_id,
            command_fencing_token=1,
            workflow_fencing_token=1,
            incident_fencing_token=1,
            suffix="iso002",
        )
    )
    stale_fencing = executor._execute(
        _command(
            cluster_id=cluster_id,
            command_fencing_token=1,
            workflow_fencing_token=1,
            incident_fencing_token=2,
            suffix="cmd011",
        )
    )

    assert cross_cluster.status.value == "FAILED"
    assert cross_cluster.status_source == "executor-rejected"
    assert "command cluster does not match executor cluster" in (
        cross_cluster.error or ""
    )
    assert stale_fencing.status.value == "FAILED"
    assert stale_fencing.status_source == "executor-rejected"
    assert "command fencing token does not match workflow/incident" in (
        stale_fencing.error or ""
    )
    assert executor.unexpected_failures == 0

    return {
        "GF-REGIONAL-ISO-002": {
            "status": cross_cluster.status.value,
            "status_source": cross_cluster.status_source,
            "error": cross_cluster.error,
        },
        "GF-REGIONAL-CMD-011": {
            "status": stale_fencing.status.value,
            "status_source": stale_fencing.status_source,
            "error": stale_fencing.error,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-id", default="cluster-a")
    parser.add_argument("--other-cluster-id", default="cluster-b")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.cluster_id, arguments.other_cluster_id),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
