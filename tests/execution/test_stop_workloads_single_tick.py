"""STOP_WORKLOADS finishes in one dispatcher tick when the workload is
already inactive after the suspend patch.

Before this, ``_suspend_outcome`` returned WAITING unconditionally on the
first call and only checked ``_workload_active`` on the next poll, so every
stop cost at least one extra tick (5 s locally, plus a remote-command round
trip on the regional path) even when the controller had already reconciled
the suspend.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests._builders import build_store, copy_model, workflow_step_execution

from ._support import (
    KubernetesWorkflowAdapter,
    UnusedApi,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepContext,
    WorkflowStepStatus,
    workflow_state,
)

NAMESPACE = "gpu-fault-system"
JOB_ID = f"{NAMESPACE}/job/training-job"
PYTORCH_ID = f"{NAMESPACE}/pytorchjob/training-job"


class AttemptPodCore:
    """Core API holding the attempt Pods the stop step marks and deletes."""

    def __init__(self, count: int = 2) -> None:
        self.pods = [
            {"metadata": {"name": f"worker-{index}", "namespace": NAMESPACE}}
            for index in range(count)
        ]
        self.patched: list[tuple[str, str, dict]] = []
        self.deleted: list[tuple[str, str, int]] = []
        self.list_calls = 0

    def list_namespaced_pod(self, *_args, **_kwargs):
        self.list_calls += 1
        return SimpleNamespace(items=list(self.pods))

    def patch_namespaced_pod(self, name, namespace, body):
        self.patched.append((namespace, name, body))

    def delete_namespaced_pod(self, name, namespace, grace_period_seconds):
        self.deleted.append((namespace, name, grace_period_seconds))
        self.pods = [pod for pod in self.pods if pod["metadata"]["name"] != name]


class ReconcilingBatch:
    """Batch API whose Job reports ``status.active`` per ``active_after_patch``
    once the suspend patch has landed, mimicking a controller that has (or
    has not) reconciled by the time the step reads the Job back."""

    def __init__(self, *, active_after_patch: int | None) -> None:
        self.active: int | None = 2
        self.active_after_patch = active_after_patch
        self.annotations: dict[str, str] = {}
        self.suspend_patches: list[bool] = []
        self.reads = 0

    def patch_namespaced_job(self, _name, _namespace, body):
        self.suspend_patches.append(body["spec"]["suspend"])
        self.annotations.update(body["metadata"].get("annotations", {}))
        self.active = self.active_after_patch

    def read_namespaced_job(self, name, _namespace):
        self.reads += 1
        return {
            "metadata": {
                "name": name,
                "resourceVersion": "1",
                "labels": {"gpu-fault.io/attempt-id": "attempt-a"},
                "annotations": dict(self.annotations),
            },
            "spec": {
                "template": {
                    "metadata": {
                        "labels": {
                            "gpu-fault.io/managed": "true",
                            "gpu-fault.io/attempt-id": "attempt-a",
                        }
                    },
                    "spec": {"containers": [{"name": "trainer"}]},
                }
            },
            "status": {"active": self.active, "terminating": 0},
        }


class ReconcilingCustom:
    """Custom API whose PyTorchJob flips to the given status after the patch."""

    def __init__(self, *, status_after_patch: dict | None) -> None:
        self.status: dict | None = {
            "replicaStatuses": {"Master": {"active": 1}, "Worker": {"active": 1}}
        }
        self.status_after_patch = status_after_patch
        self.annotations: dict[str, str] = {}
        self.patches: list[dict] = []

    def get_namespaced_custom_object(self, *_args):
        workload = {
            "metadata": {
                "name": "training-job",
                "resourceVersion": "1",
                "labels": {"gpu-fault.io/attempt-id": "attempt-a"},
                "annotations": dict(self.annotations),
            },
            "spec": {"runPolicy": {"suspend": False}},
        }
        if self.status is not None:
            workload["status"] = self.status
        return workload

    def patch_namespaced_custom_object(self, *_args, **_kwargs):
        body = _args[-1]
        self.patches.append(body)
        self.annotations.update(body["metadata"].get("annotations", {}))
        self.status = self.status_after_patch


def _stop_context(adapter, store, incident, workflow, workload_id: str):
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        workload_ids=[workload_id],
        parameters={"termination_initiator_incident_id": incident.incident_id},
    )
    workflow = copy_model(workflow, official_steps=[step])
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        idempotency_key="workflow-stop/0/STOP_WORKLOADS",
    )


def _polled(context: WorkflowStepContext, waiting) -> WorkflowStepContext:
    """The context the dispatcher hands back after recording a WAITING step."""
    return WorkflowStepContext(
        workflow=copy_model(
            context.workflow,
            step_executions=[
                workflow_step_execution(
                    0,
                    WorkflowOperation.STOP_WORKLOADS,
                    WorkflowStepStatus.WAITING,
                    adapter_operation_id=waiting.adapter_operation_id,
                    details=waiting.details,
                )
            ],
        ),
        incident=context.incident,
        step=context.step,
        step_index=0,
        request=context.request,
        idempotency_key=context.idempotency_key,
    )


def _job_adapter(batch: ReconcilingBatch, core: AttemptPodCore):
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.STOP_WORKLOADS])
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=batch, custom_api=UnusedApi(), store=store
    )
    return adapter, _stop_context(adapter, store, incident, workflow, JOB_ID)


def _pytorch_adapter(custom: ReconcilingCustom, core: AttemptPodCore):
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.STOP_WORKLOADS])
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=custom, store=store
    )
    return adapter, _stop_context(adapter, store, incident, workflow, PYTORCH_ID)


# --- (1) workload already inactive right after the patch ---------------------


def test_job_stop_succeeds_in_one_call_when_job_reports_inactive() -> None:
    batch = ReconcilingBatch(active_after_patch=0)
    core = AttemptPodCore()
    adapter, context = _job_adapter(batch, core)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert batch.suspend_patches == [True]
    assert core.deleted == [(NAMESPACE, "worker-0", 0), (NAMESPACE, "worker-1", 0)]
    # The evidence the WAITING record used to carry rides on the SUCCEEDED
    # outcome instead, so skipping the tick loses nothing.
    assert outcome.details["workloads"] == [JOB_ID]
    assert outcome.details["suspended"] is True
    assert outcome.details["deleted_pods"] == [
        f"{NAMESPACE}/worker-0",
        f"{NAMESPACE}/worker-1",
    ]
    assert "workload_log_evidence" in outcome.details
    assert "workload_log_errors" in outcome.details
    assert outcome.details["already_absent_workloads"] == []
    assert "waiting_for_active_workloads" not in outcome.details


def test_pytorch_stop_succeeds_in_one_call_when_suspended_condition_is_true() -> None:
    custom = ReconcilingCustom(
        status_after_patch={
            "conditions": [{"type": "Suspended", "status": "True"}],
            "replicaStatuses": {"Master": {"active": 0}, "Worker": {"active": 0}},
        }
    )
    core = AttemptPodCore()
    adapter, context = _pytorch_adapter(custom, core)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert len(custom.patches) == 1
    assert custom.patches[0]["spec"]["runPolicy"]["suspend"] is True
    assert len(core.deleted) == 2
    assert outcome.details["suspended"] is True


def test_immediate_stop_reads_the_workload_back_after_the_patch() -> None:
    # The pre-patch object still says active=2; only a fresh read shows the
    # reconciled state, so the fast path must not judge the stale copy.
    batch = ReconcilingBatch(active_after_patch=0)
    adapter, context = _job_adapter(batch, AttemptPodCore())
    reads_before = batch.reads

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    # One read to prepare the mutation, one to confirm the stop.
    assert batch.reads - reads_before == 2


def test_immediate_stop_is_idempotent_on_a_second_call() -> None:
    batch = ReconcilingBatch(active_after_patch=0)
    core = AttemptPodCore()
    adapter, context = _job_adapter(batch, core)

    first = adapter.execute(context)
    # Nothing was recorded as WAITING; the dispatcher may still re-run the
    # step (e.g. after a crash before the SUCCEEDED record was persisted).
    second = adapter.execute(context)

    assert first.status is WorkflowStepStatus.SUCCEEDED
    assert second.status is WorkflowStepStatus.SUCCEEDED
    # _operation_already_applied skips the patch the second time round.
    assert batch.suspend_patches == [True]
    assert second.details["deleted_pods"] == []


# --- (2) workload still active after the patch: existing two-tick path -------


def test_job_stop_waits_while_job_still_active_then_succeeds_once_inactive() -> None:
    batch = ReconcilingBatch(active_after_patch=2)
    core = AttemptPodCore()
    adapter, context = _job_adapter(batch, core)

    waiting = adapter.execute(context)

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["waiting_for_active_workloads"] == [JOB_ID]
    assert waiting.details["unknown_stop_state"] == []
    assert waiting.details["deleted_pods"] == [
        f"{NAMESPACE}/worker-0",
        f"{NAMESPACE}/worker-1",
    ]

    batch.active = 0
    completed = adapter.execute(_polled(context, waiting))

    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert batch.suspend_patches == [True]


def test_job_stop_keeps_waiting_on_poll_while_still_active() -> None:
    batch = ReconcilingBatch(active_after_patch=2)
    core = AttemptPodCore()
    adapter, context = _job_adapter(batch, core)

    waiting = adapter.execute(context)
    # A Pod the controller has not let go of yet is re-listed on the poll;
    # with a Pod still present, Job activity decides (Pod absence alone would
    # have been authoritative via the attempt_pods_absent branch).
    core.pods = [{"metadata": {"name": "worker-9", "namespace": NAMESPACE}}]
    still_waiting = adapter.execute(_polled(context, waiting))

    assert waiting.status is WorkflowStepStatus.WAITING
    assert still_waiting.status is WorkflowStepStatus.WAITING
    assert still_waiting.details["waiting_for_active_workloads"] == [JOB_ID]


def test_pytorch_stop_waits_while_replicas_still_active() -> None:
    custom = ReconcilingCustom(
        status_after_patch={
            "conditions": [{"type": "Running", "status": "True"}],
            "replicaStatuses": {"Master": {"active": 1}, "Worker": {"active": 1}},
        }
    )
    adapter, context = _pytorch_adapter(custom, AttemptPodCore())

    waiting = adapter.execute(context)

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["waiting_for_active_workloads"] == [PYTORCH_ID]


# --- (3) unknown stop state stays WAITING -----------------------------------


def test_pytorch_stop_waits_when_status_is_unknown_after_patch() -> None:
    # No status block at all: _workload_active returns None.
    custom = ReconcilingCustom(status_after_patch=None)
    adapter, context = _pytorch_adapter(custom, AttemptPodCore())

    waiting = adapter.execute(context)

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["waiting_for_active_workloads"] == []
    assert waiting.details["unknown_stop_state"] == [PYTORCH_ID]


def test_pytorch_stop_waits_when_status_has_no_counts_or_conditions() -> None:
    custom = ReconcilingCustom(status_after_patch={})
    adapter, context = _pytorch_adapter(custom, AttemptPodCore())

    waiting = adapter.execute(context)

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["unknown_stop_state"] == [PYTORCH_ID]


def test_read_back_failure_falls_back_to_waiting_instead_of_failing() -> None:
    # The patch already landed; a transient read error must not fail the
    # step, it just costs the tick the old code always paid.
    batch = ReconcilingBatch(active_after_patch=0)
    adapter, context = _job_adapter(batch, AttemptPodCore())
    original_read = batch.read_namespaced_job
    calls = {"n": 0}

    def flaky_read(name, namespace):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("apiserver hiccup")
        return original_read(name, namespace)

    batch.read_namespaced_job = flaky_read

    waiting = adapter.execute(context)

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["unknown_stop_state"] == [JOB_ID]
