from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Final

import pytest

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.execution.restart_budget_preflight import reserve_restart_budgets
from gpu_fault.models import (
    IncidentState,
    RestartAuthorization,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.store import NotFoundError, SqliteStore
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
#: ``restart_context`` signs the authorization the preflight would have issued
#: for its parameters; pass ``authorization=None`` for a request without one.
MATCHING_AUTHORIZATION: Final = object()


class BatchApi:
    def __init__(self, gpu_count: int) -> None:
        self.annotations: dict[str, str] = {}
        self.created: dict[str, dict] = {}
        self.gpu_count = gpu_count

    def read_namespaced_job(self, name: str, _namespace: str):
        if name in self.created:
            return self.created[name]
        if name != "training-job":
            error = KeyError(name)
            error.status = 404
            raise error
        return {
            "metadata": {
                "name": name,
                "resourceVersion": "1",
                "annotations": dict(self.annotations),
            },
            "spec": {
                "completions": self.gpu_count or 1,
                "parallelism": self.gpu_count or 1,
                "template": {
                    "metadata": {
                        "labels": {
                            "gpu-fault.io/managed": "true",
                            "gpu-fault.io/job-id": "train-1",
                            "gpu-fault.io/attempt-id": "attempt-a",
                        },
                        "annotations": {},
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "trainer",
                                "resources": {
                                    "limits": (
                                        {"nvidia.com/gpu": "1"}
                                        if self.gpu_count
                                        else {}
                                    )
                                },
                            }
                        ],
                    },
                },
            },
            "status": {"active": 0, "failed": 1},
        }

    def create_namespaced_job(self, _namespace: str, body: dict):
        self.created[body["metadata"]["name"]] = body


class UnusedApi:
    pass


def restart_context(
    operation_id: str,
    *,
    source_gpu_count: int,
    restart_budget: int,
    workload_id: str = "training/job/training-job",
    source_attempt_id: str | None = None,
    authorization: RestartAuthorization | None | object = MATCHING_AUTHORIZATION,
) -> WorkflowStepContext:
    if source_attempt_id is None:
        source_attempt_id = f"attempt-{operation_id}"
    incident = fault_incident(
        f"incident-{operation_id}",
        f"event-{operation_id}",
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
        workload_ids=[workload_id],
        parameters={
            "cluster_id": "cluster-a",
            "job_id": "train-1",
            "source_attempt_id": source_attempt_id,
            "source_gpu_count": source_gpu_count,
            "restart_budget": restart_budget,
        },
    )
    workflow = workflow_request(
        f"workflow-{operation_id}",
        incident.incident_id,
        WorkflowStatus.RUNNING,
        1,
        official_steps=[step],
        created_at=NOW,
        updated_at=NOW,
    )
    idempotency_key = f"workflow-{operation_id}/0/RESTART_WORKLOAD"
    if authorization is MATCHING_AUTHORIZATION:
        authorization = RestartAuthorization(
            cluster_id="cluster-a",
            job_id="train-1",
            source_attempt_id=source_attempt_id,
            source_gpu_count=source_gpu_count,
            restart_budget=restart_budget,
            restart_count=1,
            reservation_id=idempotency_key,
        )
    assert authorization is None or isinstance(authorization, RestartAuthorization)
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(
            expected_fencing_token=1, restart_authorization=authorization
        ),
        idempotency_key=idempotency_key,
    )


def test_gpu_count_change_waits_for_explicit_admin_approval() -> None:
    store = build_store()
    batch = BatchApi(gpu_count=1)
    sent: list[str] = []
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(),
        batch_api=batch,
        custom_api=UnusedApi(),
        store=store,
        alert_sender=sent.append,
    )
    context = restart_context("gpu-change", source_gpu_count=2, restart_budget=2)
    # The preflight's reservation; the adapter only compares the authorization.
    store.reserve_job_restart("cluster-a", "train-1", 2, context.idempotency_key)

    waiting = adapter.execute(context)

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["source_gpu_count"] == 2
    assert waiting.details["target_gpu_count"] == 1
    assert waiting.details["required_approval_annotation"] == "2:1"
    assert batch.created == {}
    assert len(store.list_notifications()) == 1
    assert sent == [waiting.details["notification_id"]]
    body = store.list_notifications()[0].body_text
    assert "一、发生了什么" in body
    assert "二、建议管理员做什么" in body
    assert "global batch size" in body
    assert "learning rate" in body
    assert (
        'kubectl --context "${GPU_FAULT_KUBE_CONTEXT:'
        '?set GPU_FAULT_KUBE_CONTEXT for cluster cluster-a}" '
        "-n training annotate job training-job "
        "gpu-fault.io/approve-gpu-count-change='2:1' --overwrite" in body
    )

    batch.annotations["gpu-fault.io/approve-gpu-count-change"] = "2:1"
    completed = adapter.execute(context)

    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert len(batch.created) == 1
    state = store.get_restart_budget("cluster-a", "train-1")
    assert state.restart_count == 1
    assert state.budget == 2
    assert len(store.list_notifications()) == 2
    assert len(sent) == 2


def test_storeless_gpu_count_change_uses_notification_sink() -> None:
    class NotificationSink:
        def __init__(self) -> None:
            self.notifications = []

        def save_notification_if_absent(self, notification):
            self.notifications.append(notification)
            return notification

    sink = NotificationSink()
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(),
        batch_api=BatchApi(gpu_count=1),
        custom_api=UnusedApi(),
        store=None,
        notification_sink=sink,
    )

    waiting = adapter.execute(
        restart_context("storeless-gpu-change", source_gpu_count=2, restart_budget=1)
    )

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["reason"] == "GPU_COUNT_CHANGED"
    assert len(sink.notifications) == 1


def test_restart_budget_blocks_second_restart_for_same_job() -> None:
    """The preflight owns the budget; the adapter neither reserves nor refuses."""

    store = build_store()
    batch = BatchApi(gpu_count=1)
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )
    first = restart_context("first", source_gpu_count=1, restart_budget=1)
    second = restart_context("second", source_gpu_count=1, restart_budget=1)

    reserved = reserve_restart_budgets(
        store, first.workflow, first.incident, first.workflow.official_steps
    )
    exhausted = reserve_restart_budgets(
        store, second.workflow, second.incident, second.workflow.official_steps
    )
    outcome = adapter.execute(first)

    assert reserved is None
    assert exhausted is not None
    assert exhausted.outcome.status is WorkflowStepStatus.FAILED
    assert exhausted.outcome.details["reason"] == "RESTART_BUDGET_EXHAUSTED"
    assert "restart budget exhausted" in (exhausted.outcome.error or "")
    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert len(batch.created) == 1
    state = store.get_restart_budget("cluster-a", "train-1")
    assert state.restart_count == 1
    assert state.reservation_ids == [first.idempotency_key]
    # Only the restarted-workload mail: budget exhaustion is the preflight's.
    assert len(store.list_notifications()) == 1


def test_restart_without_authorization_fails_closed() -> None:
    store = build_store()
    batch = BatchApi(gpu_count=1)
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )

    outcome = adapter.execute(
        restart_context(
            "noauth", source_gpu_count=1, restart_budget=1, authorization=None
        )
    )

    assert outcome.status is WorkflowStepStatus.FAILED
    assert "restart authorization" in (outcome.error or "")
    assert outcome.details["restart_submitted"] is False
    assert batch.created == {}
    with pytest.raises(NotFoundError):
        store.get_restart_budget("cluster-a", "train-1")


def test_restart_rejects_an_authorization_for_another_job() -> None:
    store = build_store()
    batch = BatchApi(gpu_count=1)
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )
    context = restart_context("wrong-job", source_gpu_count=1, restart_budget=1)
    assert context.request.restart_authorization is not None
    forged = context.request.restart_authorization.model_copy(
        update={"job_id": "other-job"}
    )
    context = replace(
        context,
        request=context.request.model_copy(update={"restart_authorization": forged}),
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert "does not match" in (outcome.error or "")
    assert outcome.details["restart_submitted"] is False
    assert batch.created == {}


def test_restart_rejections_before_submission_say_so() -> None:
    """Every guard refusal carries ``restart_submitted: False``.

    The control plane's terminal write reads it to hand the preflight's
    reservation back (``release_unattempted_restart_reservations``); here the
    refusal comes from the incident premise rather than the authorization.
    """

    store = build_store()
    store.save_incident(
        fault_incident("inc-source", "event-source", state=IncidentState.ESCALATED)
    )
    batch = BatchApi(gpu_count=1)
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )
    base = restart_context("premise", source_gpu_count=1, restart_budget=1)
    step = copy_model(
        base.step,
        parameters={
            **base.step.parameters,
            "requires_incident_state": "RECOVERED",
            "incident_id": "inc-source",
        },
    )
    context = replace(
        base, step=step, workflow=copy_model(base.workflow, official_steps=[step])
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["reason"] == "INCIDENT_NOT_RECOVERABLE"
    assert outcome.details["restart_submitted"] is False
    assert batch.created == {}


class PyTorchCustomApi:
    def __init__(self) -> None:
        self.patches: list[dict] = []
        self.created: dict[str, dict] = {}
        template = {
            "metadata": {
                "labels": {
                    "gpu-fault.io/managed": "true",
                    "gpu-fault.io/attempt-id": "attempt-a",
                },
                "annotations": {},
            },
            "spec": {
                "containers": [
                    {
                        "name": "trainer",
                        "resources": {"limits": {"nvidia.com/gpu": "8"}},
                    }
                ]
            },
        }
        self.workload = {
            "metadata": {
                "name": "training-job",
                "resourceVersion": "1",
                "labels": {"gpu-fault.io/attempt-id": "attempt-a"},
                "annotations": {},
            },
            "spec": {
                "runPolicy": {"suspend": True},
                "pytorchReplicaSpecs": {
                    "Master": {"replicas": 1, "template": template},
                    "Worker": {
                        "replicas": 2,
                        "template": {
                            "metadata": {
                                "labels": {
                                    "gpu-fault.io/managed": "true",
                                    "gpu-fault.io/attempt-id": ("attempt-a"),
                                },
                                "annotations": {},
                            },
                            "spec": template["spec"],
                        },
                    },
                },
            },
            "status": {"conditions": [{"type": "Suspended", "status": "True"}]},
        }

    def get_namespaced_custom_object(self, _group, _version, _namespace, _plural, name):
        if name == "training-job":
            return self.workload
        if name in self.created:
            return self.created[name]
        error = KeyError(name)
        error.status = 404
        raise error

    def patch_namespaced_custom_object(self, *_args, **_kwargs):
        self.patches.append(_args[-1])

    def create_namespaced_custom_object(
        self, _group, _version, _namespace, _plural, body
    ):
        self.created[body["metadata"]["name"]] = body


def _execute_pytorch_restart(custom: PyTorchCustomApi, operation_id: str):
    store = build_store()
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=UnusedApi(), custom_api=custom, store=store
    )
    # The PyTorchJob fixture is labelled attempt-a; the authorization the
    # preflight signs names the same attempt, as the planner's parameter does.
    context = restart_context(
        operation_id,
        source_gpu_count=24,
        restart_budget=1,
        workload_id="training/pytorchjob/training-job",
        source_attempt_id="attempt-a",
    )
    return adapter.execute(context), store


def test_restart_missing_source_fails_closed() -> None:
    class MissingCustom(PyTorchCustomApi):
        def get_namespaced_custom_object(
            self, _group, _version, _namespace, _plural, name
        ):
            error = KeyError(name)
            error.status = 404
            raise error

    custom = MissingCustom()

    outcome, store = _execute_pytorch_restart(custom, "missing-source")

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["reason"] == "RESTART_SOURCE_WORKLOAD_NOT_FOUND"
    assert outcome.details["source_workload_ids"] == [
        "training/pytorchjob/training-job"
    ]
    assert custom.created == {}
    with pytest.raises(NotFoundError):
        store.get_restart_budget("cluster-a", "train-1")


def test_restart_rejects_source_uid_drift_before_budget_or_mutation() -> None:
    class ReplacedCustom(PyTorchCustomApi):
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0
            self.workload["metadata"]["uid"] = "source-uid-a"

        def get_namespaced_custom_object(self, group, version, namespace, plural, name):
            value = super().get_namespaced_custom_object(
                group, version, namespace, plural, name
            )
            if name == "training-job":
                self.reads += 1
                value["metadata"]["uid"] = (
                    "source-uid-a" if self.reads == 1 else "source-uid-b"
                )
            return value

    custom = ReplacedCustom()

    outcome, store = _execute_pytorch_restart(custom, "uid-drift")

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["reason"] == "RESTART_SOURCE_WORKLOAD_IDENTITY_DRIFT"
    assert outcome.details["identity_drift"] == [
        {
            "workload_id": "training/pytorchjob/training-job",
            "expected_uid": "source-uid-a",
            "current_uid": "source-uid-b",
        }
    ]
    assert custom.patches == []
    assert custom.created == {}
    with pytest.raises(NotFoundError):
        store.get_restart_budget("cluster-a", "train-1")


def test_restart_rechecks_managed_recovery_owner_after_final_read() -> None:
    class OwnershipChangedCustom(PyTorchCustomApi):
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0
            self.workload["metadata"]["uid"] = "source-uid-a"

        def get_namespaced_custom_object(self, group, version, namespace, plural, name):
            value = super().get_namespaced_custom_object(
                group, version, namespace, plural, name
            )
            if name == "training-job":
                self.reads += 1
                if self.reads == 2:
                    value["metadata"]["annotations"][
                        "sagemaker.amazonaws.com/enable-job-auto-resume"
                    ] = "true"
            return value

    custom = OwnershipChangedCustom()

    outcome, store = _execute_pytorch_restart(custom, "owner-changed")

    assert outcome.status is WorkflowStepStatus.FAILED
    assert "enable-job-auto-resume is enabled" in (outcome.error or "")
    assert custom.patches == []
    assert custom.created == {}
    with pytest.raises(NotFoundError):
        store.get_restart_budget("cluster-a", "train-1")


def test_restart_rejects_source_already_being_deleted() -> None:
    custom = PyTorchCustomApi()
    custom.workload["metadata"].update(
        {"uid": "source-uid-a", "deletionTimestamp": "2026-09-01T16:00:00Z"}
    )

    outcome, store = _execute_pytorch_restart(custom, "source-deleting")

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["reason"] == "RESTART_SOURCE_WORKLOAD_DELETING"
    assert custom.patches == []
    assert custom.created == {}
    with pytest.raises(NotFoundError):
        store.get_restart_budget("cluster-a", "train-1")


def test_restart_source_disappearing_during_patch_is_controlled_failure() -> None:
    class VanishingCustom(PyTorchCustomApi):
        def __init__(self) -> None:
            super().__init__()
            self.workload["metadata"]["uid"] = "source-uid-a"

        def patch_namespaced_custom_object(self, *_args, **_kwargs):
            error = KeyError("training-job")
            error.status = 404
            raise error

    custom = VanishingCustom()

    outcome, _store = _execute_pytorch_restart(custom, "source-vanished")

    assert outcome.status is WorkflowStepStatus.FAILED
    assert (
        outcome.details["reason"] == "RESTART_SOURCE_WORKLOAD_NOT_FOUND_DURING_MUTATION"
    )
    assert custom.created == {}


class PodCoreApi:
    def __init__(self) -> None:
        self.patches: list[tuple[str, str, dict]] = []
        self.deleted: list[tuple[str, str, int]] = []
        self.items = [{"metadata": {"name": "training-master-0"}}]
        self.list_calls = 0

    def list_namespaced_pod(self, namespace, *, label_selector):
        assert namespace == "training"
        assert "gpu-fault.io/attempt-id=attempt-a" in label_selector
        self.list_calls += 1
        return {"items": self.items}

    def patch_namespaced_pod(self, name, namespace, body):
        self.patches.append((name, namespace, body))

    def delete_namespaced_pod(self, name, namespace, grace_period_seconds):
        self.deleted.append((name, namespace, grace_period_seconds))
        self.items = [item for item in self.items if item["metadata"]["name"] != name]


def test_incident_stop_marks_source_pods_before_suspend() -> None:
    custom = PyTorchCustomApi()
    core = PodCoreApi()
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=custom, store=build_store()
    )
    base = restart_context(
        "stop",
        source_gpu_count=24,
        restart_budget=1,
        workload_id="training/pytorchjob/training-job",
    )
    step = copy_model(
        base.step,
        operation=WorkflowOperation.STOP_WORKLOADS,
        parameters={"termination_initiator_incident_id": "incident-xid-11"},
    )
    context = WorkflowStepContext(
        workflow=copy_model(base.workflow, official_steps=[step]),
        incident=base.incident,
        step=step,
        step_index=0,
        request=base.request,
        idempotency_key="workflow-stop/0/STOP_WORKLOADS",
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert core.patches == [
        (
            "training-master-0",
            "training",
            {
                "metadata": {
                    "annotations": {
                        "gpu-fault.io/termination-initiator-incident-id": (
                            "incident-xid-11"
                        )
                    }
                }
            },
        )
    ]
    assert custom.patches[0]["spec"]["runPolicy"]["suspend"] is True
    assert core.deleted == [("training-master-0", "training", 0)]


def test_incident_stop_poll_succeeds_after_source_pods_disappear() -> None:
    custom = PyTorchCustomApi()
    core = PodCoreApi()
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=custom, store=build_store()
    )
    base = restart_context(
        "stop-poll",
        source_gpu_count=24,
        restart_budget=1,
        workload_id="training/pytorchjob/training-job",
    )
    step = copy_model(
        base.step,
        operation=WorkflowOperation.STOP_WORKLOADS,
        parameters={"termination_initiator_incident_id": "incident-xid-11"},
    )
    workflow = copy_model(base.workflow, official_steps=[step])
    context = WorkflowStepContext(
        workflow=workflow,
        incident=base.incident,
        step=step,
        step_index=0,
        request=base.request,
        idempotency_key=("workflow-stop-poll/0/STOP_WORKLOADS"),
    )

    first = adapter.execute(context)
    assert first.status is WorkflowStepStatus.WAITING
    assert core.list_calls == 1

    custom.workload["metadata"]["annotations"] = custom.patches[0]["metadata"][
        "annotations"
    ]
    core.items = []
    polled_workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                0, WorkflowOperation.STOP_WORKLOADS, WorkflowStepStatus.WAITING
            )
        ],
    )
    polled_context = WorkflowStepContext(
        workflow=polled_workflow,
        incident=base.incident,
        step=step,
        step_index=0,
        request=base.request,
        idempotency_key=("workflow-stop-poll/0/STOP_WORKLOADS"),
    )

    completed = adapter.execute(polled_context)

    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert core.list_calls == 2
    assert len(custom.patches) == 1


def test_pytorch_restart_creates_new_attempt_metadata() -> None:
    store = build_store()
    custom = PyTorchCustomApi()
    sent: list[str] = []
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(),
        batch_api=UnusedApi(),
        custom_api=custom,
        store=store,
        alert_sender=sent.append,
    )
    context = restart_context(
        "pytorch",
        source_gpu_count=24,
        restart_budget=2,
        workload_id="training/pytorchjob/training-job",
        source_attempt_id="attempt-a",
    )

    completed = adapter.execute(context)

    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert len(custom.patches) == 1
    patch = custom.patches[0]
    attempt_id = completed.details["restart_attempt_id"]
    assert attempt_id.startswith("attempt-a-r-")
    assert patch["metadata"]["labels"]["gpu-fault.io/attempt-id"] == attempt_id
    assert patch["spec"]["runPolicy"]["suspend"] is False
    for replica in patch["spec"]["pytorchReplicaSpecs"].values():
        metadata = replica["template"]["metadata"]
        assert metadata["labels"]["gpu-fault.io/attempt-id"] == attempt_id
        assert metadata["annotations"]["gpu-fault.io/restart-budget"] == "2"
        assert metadata["annotations"]["gpu-fault.io/restart-count"] == "1"
    notifications = store.list_notifications()
    assert len(notifications) == 1
    assert sent == [notifications[0].notification_id]
    assert completed.details["notification_id"] == (notifications[0].notification_id)
    assert "系统动作：RESTART_WORKLOAD" in notifications[0].body_text
    assert attempt_id in notifications[0].body_text


def test_terminal_pytorch_restart_creates_retry_object() -> None:
    store = build_store()
    custom = PyTorchCustomApi()
    custom.workload["status"]["conditions"] = [{"type": "Failed", "status": "True"}]
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=UnusedApi(), custom_api=custom, store=store
    )
    context = restart_context(
        "pytorch-terminal",
        source_gpu_count=24,
        restart_budget=1,
        workload_id="training/pytorchjob/training-job",
        source_attempt_id="attempt-a",
    )

    completed = adapter.execute(context)
    duplicate = adapter.execute(context)

    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert duplicate.status is WorkflowStepStatus.SUCCEEDED
    assert custom.patches == []
    assert len(custom.created) == 1
    retry_name, retry = next(iter(custom.created.items()))
    assert retry_name.startswith("training-job-r-")
    assert "status" not in retry
    attempt_id = completed.details["restart_attempt_id"]
    assert retry["metadata"]["labels"]["gpu-fault.io/attempt-id"] == attempt_id
    assert retry["spec"]["runPolicy"]["suspend"] is False
    assert retry["spec"]["pytorchReplicaSpecs"]["Master"]["replicas"] == 1
    worker_template = retry["spec"]["pytorchReplicaSpecs"]["Worker"]["template"]
    assert (
        worker_template["metadata"]["labels"]["gpu-fault.io/attempt-id"] == attempt_id
    )
    assert json.loads(
        worker_template["metadata"]["annotations"]["gpu-fault.io/workload-ids"]
    ) == [f"training/pytorchjob/{retry_name}"]
    assert completed.details["restarted_workload_ids"] == [
        f"training/pytorchjob/{retry_name}"
    ]


def test_zero_gpu_workload_can_restart_when_counts_match() -> None:
    store = build_store()
    batch = BatchApi(gpu_count=0)
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=batch, custom_api=UnusedApi(), store=store
    )

    context = restart_context("zero-gpu", source_gpu_count=0, restart_budget=1)
    store.reserve_job_restart("cluster-a", "train-1", 1, context.idempotency_key)

    completed = adapter.execute(context)

    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert len(batch.created) == 1
    assert store.get_restart_budget("cluster-a", "train-1").restart_count == 1
    notifications = store.list_notifications()
    assert len(notifications) == 1
    assert "原 GPU 数量：0" in notifications[0].body_text


def test_restart_reservation_is_idempotent_and_persistent(tmp_path) -> None:
    path = tmp_path / "restart-budget.db"
    first = SqliteStore(str(path))
    state, reserved = first.reserve_job_restart(
        "cluster-a", "train-1", 2, "workflow-a/restart"
    )
    duplicate, duplicate_reserved = first.reserve_job_restart(
        "cluster-a", "train-1", 2, "workflow-a/restart"
    )
    first.close()

    reopened = SqliteStore(str(path))
    persisted = reopened.get_restart_budget("cluster-a", "train-1")
    reopened.close()

    assert reserved
    assert duplicate_reserved
    assert state.restart_count == 1
    assert duplicate.restart_count == 1
    assert persisted.restart_count == 1
    assert persisted.reservation_ids == ["workflow-a/restart"]
