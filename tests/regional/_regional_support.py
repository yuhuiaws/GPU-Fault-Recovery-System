"""Shared fixtures and builders for split test shards."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    AdvisoryNotification,
    Environment,
    IncidentState,
    NotificationResult,
    NotificationStatus,
    TerminalEvent,
    TerminalStatus,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.regional import (
    RegionalClusterRegistration,
    RemoteActionCommand,
    cluster_token_sha256,
)
from tests._builders import copy_model, fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 7, 28, tzinfo=timezone.utc)

TOKEN_A = "a" * 32

TOKEN_B = "b" * 32


class RecordingNotifier:
    def __init__(self) -> None:
        self.notifications = []

    def send(self, notification):
        self.notifications.append(notification)
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id=f"message-{len(self.notifications)}",
        )


def terminal(cluster_id: str) -> TerminalEvent:
    return TerminalEvent(
        cluster_id=cluster_id,
        environment=Environment.HYPERPOD_EKS,
        job_id="same-job",
        attempt_id="same-attempt",
        terminal_status=TerminalStatus.SUCCEEDED,
        ended_at=NOW,
        runtime_profile_version="profile-v1",
    )


def registration(cluster_id: str, token: str):
    return RegionalClusterRegistration(
        cluster_id=cluster_id,
        region="us-west-2",
        hyperpod_cluster_name=f"hp-{cluster_id}",
        eks_cluster_arn=(f"arn:aws:eks:us-west-2:123456789012:cluster/{cluster_id}"),
        token_sha256=cluster_token_sha256(token),
        allowed_namespaces=["training", "gpu-fault-system"],
        agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
    )


def workflow_state():
    incident = fault_incident(
        "incident-a",
        "event-a",
        policy_version="policy-v1",
        policy_source="test",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
        created_at=NOW,
        updated_at=NOW,
    )
    step = workflow_step(
        WorkflowOperation.MARK_UNSCHEDULABLE, "gpu-fault-kubernetes-adapter"
    )
    workflow = workflow_request(
        "workflow-a",
        incident.incident_id,
        WorkflowStatus.RUNNING,
        official_steps=[step],
        created_at=NOW,
        updated_at=NOW,
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="workflow-a/0/MARK_UNSCHEDULABLE",
    )
    return context


def enqueue_remote_command(
    store,
    command_id: str,
    *,
    cluster_id: str = "cluster-a",
    owner: str = "gpu-fault-kubernetes-adapter",
    created_at: datetime = NOW,
    **overrides,
) -> RemoteActionCommand:
    """Write one command straight into the regional backlog.

    The dispatcher normally produces these, but a boundary sweep needs dozens of
    them with chosen ids and creation times, and driving the dispatcher that many
    times would only prove the dispatcher again. ``ensure_remote_command`` is the
    store's own public entry point, so this is still the real write path.
    """

    context = workflow_state()
    command = RemoteActionCommand(
        command_id=command_id,
        cluster_id=cluster_id,
        workflow_request_id=f"workflow-{command_id}",
        incident_id=f"incident-{command_id}",
        step_index=0,
        fencing_token=context.workflow.fencing_token,
        idempotency_key=f"{command_id}/0/MARK_UNSCHEDULABLE",
        step=copy_model(context.step, execution_owner=owner),
        workflow=copy_model(context.workflow, request_id=f"workflow-{command_id}"),
        incident=copy_model(context.incident, incident_id=f"incident-{command_id}"),
        created_at=created_at,
        updated_at=created_at,
        **overrides,
    )
    return store.ensure_remote_command(command)


def remote_context(suffix: str, execution_owner: str) -> WorkflowStepContext:
    context = workflow_state()
    incident = copy_model(
        context.incident, incident_id=f"incident-{suffix}", event_id=f"event-{suffix}"
    )
    step = copy_model(context.step, execution_owner=execution_owner)
    workflow = copy_model(
        context.workflow,
        request_id=f"workflow-{suffix}",
        incident_id=incident.incident_id,
        official_steps=[step],
    )
    return replace(
        context,
        workflow=workflow,
        incident=incident,
        step=step,
        idempotency_key=(f"workflow-{suffix}/0/MARK_UNSCHEDULABLE"),
    )


def regional_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GPU_FAULT_EXECUTOR_MODE", "active")
    monkeypatch.setenv("GPU_FAULT_ALLOWED_OPERATIONS", "MARK_UNSCHEDULABLE")
    monkeypatch.setenv(
        "GPU_FAULT_STORE_URL", f"sqlite:///{tmp_path / 'regional-context.db'}"
    )
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "e" * 32)
    monkeypatch.setenv("GPU_FAULT_DEPLOYMENT_MODE", "regional")
    monkeypatch.setenv(
        "GPU_FAULT_REGIONAL_CLUSTERS_JSON",
        json.dumps(
            [
                {
                    "cluster_id": "cluster-a",
                    "region": "us-west-2",
                    "hyperpod_cluster_name": "hp-cluster-a",
                    "eks_cluster_arn": (
                        "arn:aws:eks:us-west-2:123456789012:cluster/cluster-a"
                    ),
                    "token": TOKEN_A,
                    "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
                }
            ]
        ),
    )
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER", "false")
    monkeypatch.setenv("GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER", "false")
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", "false")
    monkeypatch.delenv("GPU_FAULT_HYPERPOD_CLUSTER", raising=False)
    # The regional control plane runs with email off and alerts through
    # the metrics path, which this process cannot see; the deployment
    # manifests state that with the same variable.
    monkeypatch.setenv("GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL", "true")


def ownership_fixture(store) -> None:
    """A finished and a still-running incident on cluster-a."""

    for incident_id, request_id, status in (
        ("inc-dead", "workflow-dead", WorkflowStatus.FAILED),
        ("inc-live", "workflow-live", WorkflowStatus.RUNNING),
    ):
        store.save_incident(
            fault_incident(
                incident_id,
                f"event-{incident_id}",
                official_action="RESTART_BM",
                state=IncidentState.ACTION_PENDING,
                workflow_request_id=request_id,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        store.save_workflow(
            workflow_request(
                request_id,
                incident_id,
                status,
                1,
                runtime_profile_version="profile-v1",
                official_action="RESTART_BM",
                official_steps=[workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE)],
                created_at=NOW,
                updated_at=NOW,
            )
        )
    store.save_incident(
        fault_incident(
            "inc-quarantined",
            "event-quarantined",
            "NODE_HEALTH",
            node_ids=["node-q"],
            policy_version="site-v1",
            policy_source="SITE_NODE_HEALTH",
            official_action="QUARANTINE",
            state=IncidentState.QUARANTINED,
            workflow_request_id="workflow-quarantined",
            created_at=NOW,
            updated_at=NOW,
        )
    )
    store.save_workflow(
        workflow_request(
            "workflow-quarantined",
            "inc-quarantined",
            WorkflowStatus.SUCCEEDED,
            1,
            runtime_profile_version="profile-v1",
            official_action="QUARANTINE",
            official_steps=[
                workflow_step(WorkflowOperation.QUARANTINE, node_ids=["node-q"])
            ],
            completed_step_indexes=[0],
            completed_operations=[WorkflowOperation.QUARANTINE],
            created_at=NOW,
            updated_at=NOW,
        )
    )
    # Another cluster's incident must never be readable by cluster-a.
    store.save_incident(
        fault_incident(
            "inc-other-cluster",
            "event-other",
            cluster_id="cluster-b",
            node_ids=["node-b"],
            official_action="RESTART_BM",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id="workflow-dead",
            created_at=NOW,
            updated_at=NOW,
        )
    )


def spare_alert(cluster_id: str, incident_id: str = "inc-spare-1"):
    body = "HyperPod warm-spare capacity alert"
    return AdvisoryNotification(
        deduplication_key=(f"{incident_id}/hyperpod-spare-insufficient"),
        cluster_name=cluster_id,
        incident_id=incident_id,
        subject=(
            f"[GPU action required] {cluster_id}: insufficient HyperPod spare nodes"
        ),
        body_text=body,
        support_case_draft=body,
    )
