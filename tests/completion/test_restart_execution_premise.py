"""The Kubernetes restart adapter reads the planner's two safety premises (F-G5).

``requires_incident_state`` was honoured only by the simulated executor, so a
production restart went ahead while the incident it depended on was still
being repaired. ``avoid_node_ids`` had no reader at all.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.regional import RemoteIncidentOwnershipReport
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)

NOW = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)


class BatchApi:
    def __init__(self, *, node_name: str | None = None) -> None:
        self.created: dict[str, dict[str, Any]] = {}
        self.node_name = node_name

    def read_namespaced_job(self, name: str, _namespace: str) -> dict[str, Any]:
        if name in self.created:
            return self.created[name]
        if name != "training-job":
            error = KeyError(name)
            error.status = 404  # type: ignore[attr-defined]
            raise error
        pod_spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "containers": [
                {"name": "trainer", "resources": {"limits": {"nvidia.com/gpu": "1"}}}
            ],
        }
        if self.node_name is not None:
            pod_spec["nodeName"] = self.node_name
        return {
            "metadata": {"name": name, "resourceVersion": "1", "annotations": {}},
            "spec": {
                "completions": 1,
                "parallelism": 1,
                "template": {
                    "metadata": {
                        "labels": {
                            "gpu-fault.io/managed": "true",
                            "gpu-fault.io/job-id": "train-1",
                            "gpu-fault.io/attempt-id": "attempt-a",
                        },
                        "annotations": {},
                    },
                    "spec": pod_spec,
                },
            },
            "status": {"active": 0, "failed": 1},
        }

    def create_namespaced_job(self, _namespace: str, body: dict[str, Any]) -> None:
        self.created[body["metadata"]["name"]] = body


class UnusedApi:
    pass


def restart_context(extra_parameters: dict[str, Any]) -> WorkflowStepContext:
    incident = fault_incident(
        "incident-derived",
        "event-derived",
        "TRAINING_ATTEMPT_TERMINAL",
        policy_version="passive-recovery-v1",
        policy_source="completion-handler",
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
        created_at=NOW,
        updated_at=NOW,
    )
    step = workflow_step(
        WorkflowOperation.RESTART_WORKLOAD,
        "gpu-fault-kubernetes-adapter",
        workload_ids=["training/job/training-job"],
        parameters={
            "cluster_id": "cluster-a",
            "job_id": "train-1",
            "source_attempt_id": "attempt-a",
            "source_gpu_count": 1,
            "restart_budget": 3,
            **extra_parameters,
        },
    )
    workflow = workflow_request(
        "workflow-derived",
        incident.incident_id,
        WorkflowStatus.RUNNING,
        1,
        official_steps=[step],
        created_at=NOW,
        updated_at=NOW,
    )
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=1),
        idempotency_key="workflow-derived/0/RESTART_WORKLOAD",
    )


def adapter(batch: BatchApi, store: Any) -> KubernetesWorkflowAdapter:
    return KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )


def test_restart_waits_until_the_source_incident_is_recovered() -> None:
    store = build_store()
    source = fault_incident(
        "inc-source", "event-source", state=IncidentState.ACTION_PENDING
    )
    store.save_incident(source)
    batch = BatchApi()
    context = restart_context(
        {"requires_incident_state": "RECOVERED", "incident_id": "inc-source"}
    )

    waiting = adapter(batch, store).execute(context)

    assert waiting.status is WorkflowStepStatus.WAITING
    # RF-4: the hold speaks the shared "node under remediation" vocabulary;
    # the premise it waits on is kept beside it.
    assert waiting.details["reason"] == "NODE_UNDER_REMEDIATION"
    assert waiting.details["premise_reason"] == "INCIDENT_NOT_RECOVERED"
    assert waiting.details["incident_state"] == "ACTION_PENDING"
    assert batch.created == {}

    store.save_incident(copy_model(source, state=IncidentState.RECOVERED))
    completed = adapter(batch, store).execute(context)

    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert len(batch.created) == 1


def test_restart_fails_when_the_source_incident_can_never_recover() -> None:
    store = build_store()
    store.save_incident(
        fault_incident("inc-source", "event-source", state=IncidentState.ESCALATED)
    )
    batch = BatchApi()

    outcome = adapter(batch, store).execute(
        restart_context(
            {"requires_incident_state": "RECOVERED", "incident_id": "inc-source"}
        )
    )

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["reason"] == "INCIDENT_NOT_RECOVERABLE"
    assert batch.created == {}


def test_storeless_restart_reads_the_premise_from_the_ownership_provider() -> None:
    class Provider:
        def incident_ownership(self, incident_id: str) -> RemoteIncidentOwnershipReport:
            return RemoteIncidentOwnershipReport(
                incident_id=incident_id, known=True, incident_state="SAFETY_PENDING"
            )

    batch = BatchApi()
    subject = KubernetesWorkflowAdapter(
        core_api=UnusedApi(),
        batch_api=batch,
        custom_api=UnusedApi(),
        store=None,
        ownership_provider=Provider(),
    )

    outcome = subject.execute(
        restart_context(
            {"requires_incident_state": "RECOVERED", "incident_id": "inc-source"}
        )
    )

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["incident_state"] == "SAFETY_PENDING"
    assert batch.created == {}


def test_storeless_restart_without_a_provider_fails_closed() -> None:
    batch = BatchApi()
    subject = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=None
    )

    outcome = subject.execute(
        restart_context(
            {"requires_incident_state": "RECOVERED", "incident_id": "inc-source"}
        )
    )

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["reason"] == "INCIDENT_STATE_UNVERIFIABLE"
    assert batch.created == {}


def test_restart_refuses_a_template_pinned_to_an_avoided_node() -> None:
    batch = BatchApi(node_name="node-a")

    outcome = adapter(batch, build_store()).execute(
        restart_context({"avoid_node_ids": ["node-a"]})
    )

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["reason"] == "RESTART_TARGET_AVOIDED"
    assert outcome.details["avoided_node_ids"] == ["node-a"]
    assert batch.created == {}


def test_restart_keeps_the_new_attempt_off_avoided_nodes() -> None:
    batch = BatchApi()

    outcome = adapter(batch, build_store()).execute(
        restart_context({"avoid_node_ids": ["node-a", "node-z"]})
    )

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    (body,) = batch.created.values()
    terms = body["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    assert terms == [
        {
            "matchExpressions": [
                {
                    "key": "kubernetes.io/hostname",
                    "operator": "NotIn",
                    "values": ["node-a", "node-z"],
                }
            ]
        }
    ]
