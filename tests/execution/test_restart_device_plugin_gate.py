"""A workload restart waits until kubelet reports the repaired node's GPUs healthy.

Live 2026-10-07 (gpu-eks-cluster DESTR-015, attempt 9): both branches reset,
restored, validated and released their node, and the join's RESTART_WORKLOAD
created the PyTorchJob attempt one second after RESTORE_SCHEDULING. Its Pods,
pinned by ``nodeName``, were admitted by kubelet onto devices the upstream
NVIDIA device plugin still reported unhealthy across the reset
(``UnexpectedAdmissionError: Allocate failed due to no healthy devices present``)
and the attempt ended Failed; node allocatable read 8/8 again a little later.
The restart now reads the node's ``nvidia.com/gpu`` capacity and allocatable,
the plugin Pod and the node status heartbeat, holds (bounded, owing no restart
budget) while they disagree, and carries the receipt on the step it then runs.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.adapters.kubernetes import device_plugin_health as gate
from gpu_fault.execution import WorkflowExecutionRequest
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    workflow_step_execution,
)
from tests.execution._support import FakeBatchApi, UnusedApi, workflow_state
from tests.execution.test_restart_safety import BatchApi, restart_context

NODE = "node-a"
PLUGIN_POD = "nvidia-device-plugin-00001"


def _stamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _plugin_pod(*, ready: bool, ready_since: datetime, terminating: bool = False):
    metadata: dict[str, Any] = {"name": PLUGIN_POD, "namespace": gate.PLUGIN_NAMESPACE}
    if terminating:
        metadata["deletionTimestamp"] = _stamp(ready_since)
    return {
        "metadata": metadata,
        "spec": {"nodeName": NODE},
        "status": {
            "conditions": [
                {
                    "type": "Ready",
                    "status": "True" if ready else "False",
                    "lastTransitionTime": _stamp(ready_since),
                }
            ]
        },
    }


def _node(*, capacity: int, allocatable: int, heartbeat: datetime | None):
    status: dict[str, Any] = {
        "capacity": {"cpu": "96", gate.GPU_RESOURCE: str(capacity)},
        "allocatable": {"cpu": "95", gate.GPU_RESOURCE: str(allocatable)},
    }
    if heartbeat is not None:
        status["conditions"] = [
            {"type": "Ready", "status": "True", "lastHeartbeatTime": _stamp(heartbeat)}
        ]
    return {
        "metadata": {"name": NODE, "resourceVersion": "1", "annotations": {}},
        "spec": {"unschedulable": False, "taints": []},
        "status": status,
    }


class GpuNodeCore:
    """The one GPU node kubelet publishes, and the plugin Pod on it."""

    def __init__(self, *, capacity: int = 8, allocatable: int = 8) -> None:
        now = datetime.now(timezone.utc)
        self.ready_since = (now - timedelta(minutes=5)).replace(microsecond=0)
        self.node = _node(
            capacity=capacity,
            allocatable=allocatable,
            heartbeat=self.ready_since + timedelta(seconds=30),
        )
        self.pods: list[Any] = [_plugin_pod(ready=True, ready_since=self.ready_since)]
        self.reads: list[str] = []
        self.pod_lists: list[tuple[str, str | None, str | None]] = []

    def read_node(self, name: str) -> Any:
        self.reads.append(name)
        if name != NODE:
            error = KeyError(name)
            error.status = 404  # type: ignore[attr-defined]
            raise error
        return self.node

    def list_namespaced_pod(
        self,
        namespace: str,
        label_selector: str | None = None,
        field_selector: str | None = None,
    ) -> Any:
        self.pod_lists.append((namespace, label_selector, field_selector))
        return SimpleNamespace(items=list(self.pods))

    def set_allocatable(self, value: int) -> None:
        self.node["status"]["allocatable"][gate.GPU_RESOURCE] = str(value)


class PinnedBatchApi(BatchApi):
    """The restart safety fixture's Job, pinned to the repaired node by nodeName."""

    def read_namespaced_job(self, name: str, namespace: str):
        job = super().read_namespaced_job(name, namespace)
        if name == "training-job":
            job["spec"]["template"]["spec"]["nodeName"] = NODE
        return job


# ------------------------------------------------------------- the receipt


_CORE_PODS = object()


def _receipt(core: GpuNodeCore, pods: Any = _CORE_PODS, **kwargs: Any):
    return gate.gpu_resource_health(
        core.node,
        core.pods if pods is _CORE_PODS else pods,
        now=datetime.now(timezone.utc),
        **kwargs,
    )


def test_a_node_whose_kubelet_reports_every_device_healthy_passes() -> None:
    core = GpuNodeCore()

    receipt = _receipt(core)

    assert receipt["healthy"] is True, receipt
    assert (receipt["capacity"], receipt["allocatable"]) == (8, 8)
    assert receipt["plugin_pods_ready"] == [PLUGIN_POD]
    assert receipt["plugin_ready_since"] == core.ready_since.isoformat()
    assert "reasons" not in receipt


@pytest.mark.parametrize(
    "shape,phrase",
    [
        ("allocatable-short", "allocatable 7 != capacity 8"),
        ("plugin-not-ready", "device plugin Pod on the node is not Ready"),
        ("status-predates-plugin", "predates the device plugin's readiness"),
    ],
)
def test_the_receipt_names_what_keeps_the_devices_unhealthy(
    shape: str, phrase: str
) -> None:
    core = GpuNodeCore()
    pods = core.pods
    if shape == "allocatable-short":
        core.set_allocatable(7)
    elif shape == "plugin-not-ready":
        pods = [_plugin_pod(ready=False, ready_since=core.ready_since)]
    else:
        core.node["status"]["conditions"][0]["lastHeartbeatTime"] = _stamp(
            core.ready_since - timedelta(seconds=30)
        )

    receipt = _receipt(core, pods)

    assert receipt["healthy"] is False, receipt
    assert any(phrase in reason for reason in receipt["reasons"]), receipt["reasons"]


def test_a_terminating_plugin_pod_is_the_old_one_not_a_second_opinion() -> None:
    core = GpuNodeCore()
    pods = [
        _plugin_pod(ready=False, ready_since=core.ready_since, terminating=True),
        *core.pods,
    ]

    receipt = _receipt(core, pods)

    assert receipt["healthy"] is True, receipt
    assert receipt["plugin_pods"] == [PLUGIN_POD]


def test_without_plugin_pods_the_allocatable_comparison_stands_alone() -> None:
    core = GpuNodeCore()

    unread = _receipt(core, None, pods_error="RuntimeError: list refused")
    unmatched = _receipt(core, [])

    assert unread["healthy"] is True and unread["plugin_pods"] is None, unread
    assert unread["plugin_pods_error"] == "RuntimeError: list refused"
    assert unmatched["healthy"] is True and unmatched["plugin_pods"] == [], unmatched
    core.set_allocatable(0)
    assert _receipt(core, None)["healthy"] is False


def test_a_node_without_gpu_capacity_is_outside_the_gate() -> None:
    core = GpuNodeCore(capacity=0, allocatable=0)

    receipt = _receipt(core)

    assert receipt["not_applicable"] is True
    assert receipt["healthy"] is True


# ------------------------------------------------------------- the adapter


def _adapter(core: Any, batch: Any, store: Any) -> KubernetesWorkflowAdapter:
    return KubernetesWorkflowAdapter(
        core_api=core, batch_api=batch, custom_api=UnusedApi(), store=store
    )


def test_a_pinned_restart_waits_until_kubelet_reports_the_devices_healthy() -> None:
    store = build_store()
    core = GpuNodeCore(allocatable=7)
    batch = PinnedBatchApi(gpu_count=1)
    adapter = _adapter(core, batch, store)
    context = restart_context("plugin-gate", source_gpu_count=1, restart_budget=1)
    store.reserve_job_restart("cluster-a", "train-1", 1, context.idempotency_key)

    held = adapter.execute(context)

    assert held.status is WorkflowStepStatus.WAITING, held
    assert held.details["restart_submitted"] is False, "a hold owes no budget"
    pending = held.details[gate.PENDING_KEY]
    assert pending["nodes"][NODE]["allocatable"] == 7, pending
    assert pending["timeout_seconds"] == gate.DEFAULT_TIMEOUT_SECONDS
    assert "allocatable 7 != capacity 8" in pending["reason"], pending
    assert batch.created == {}, "nothing was submitted onto unhealthy devices"
    assert core.reads == [NODE]
    assert core.pod_lists == [
        (gate.PLUGIN_NAMESPACE, gate.PLUGIN_SELECTOR, f"spec.nodeName={NODE}")
    ]

    core.set_allocatable(8)
    proceeded = adapter.execute(context)

    assert proceeded.status is WorkflowStepStatus.SUCCEEDED, proceeded
    assert len(batch.created) == 1, batch.created
    receipt = proceeded.details[gate.RECEIPT_KEY][NODE]
    assert (receipt["healthy"], receipt["allocatable"]) == (True, 8), receipt
    assert receipt["plugin_pods_ready"] == [PLUGIN_POD]


def test_the_hold_is_bounded_and_refuses_without_submitting() -> None:
    store = build_store()
    core = GpuNodeCore(allocatable=7)
    batch = PinnedBatchApi(gpu_count=1)
    adapter = _adapter(core, batch, store)
    context = restart_context("plugin-bound", source_gpu_count=1, restart_budget=1)
    store.reserve_job_restart("cluster-a", "train-1", 1, context.idempotency_key)
    first_hold = workflow_step_execution(
        0,
        WorkflowOperation.RESTART_WORKLOAD,
        WorkflowStepStatus.WAITING,
        details={gate.PENDING_KEY: {"nodes": {}}, "restart_submitted": False},
        started_at=datetime.now(timezone.utc) - timedelta(seconds=400),
    )
    context = replace(
        context, workflow=copy_model(context.workflow, step_executions=[first_hold])
    )

    refused = adapter.execute(context)

    assert refused.status is WorkflowStepStatus.FAILED, refused
    assert refused.details["reason"] == gate.UNHEALTHY_REASON
    assert refused.details["restart_submitted"] is False
    assert refused.details["waited_seconds"] >= 400, refused.details
    assert refused.details[gate.RECEIPT_KEY][NODE]["healthy"] is False
    assert "did not converge before the workload restart" in str(refused.error)
    assert batch.created == {}


def test_the_bound_is_a_step_parameter() -> None:
    store = build_store()
    core = GpuNodeCore(allocatable=7)
    batch = PinnedBatchApi(gpu_count=1)
    adapter = _adapter(core, batch, store)
    context = restart_context("plugin-parameter", source_gpu_count=1, restart_budget=1)
    step = copy_model(
        context.step, parameters={**context.step.parameters, gate.TIMEOUT_PARAMETER: 0}
    )
    context = replace(
        context, step=step, workflow=copy_model(context.workflow, official_steps=[step])
    )

    refused = adapter.execute(context)

    assert refused.status is WorkflowStepStatus.FAILED, refused
    assert refused.details["timeout_seconds"] == 0


def test_a_restart_with_no_pinned_or_repaired_node_reads_no_node() -> None:
    """The gate judges the nodes the Pods come back to; a free-floating
    restart on a record that repaired nothing has none, and the core API is
    never touched (``UnusedApi`` has no ``read_node``)."""

    store = build_store()
    batch = BatchApi(gpu_count=1)
    adapter = _adapter(UnusedApi(), batch, store)
    context = restart_context("plugin-free", source_gpu_count=1, restart_budget=1)
    store.reserve_job_restart("cluster-a", "train-1", 1, context.idempotency_key)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert gate.RECEIPT_KEY not in outcome.details, outcome.details
    assert len(batch.created) == 1


def test_a_pinned_node_without_gpu_capacity_passes_with_a_receipt() -> None:
    store = build_store()
    core = GpuNodeCore(capacity=0, allocatable=0)
    batch = PinnedBatchApi(gpu_count=1)
    adapter = _adapter(core, batch, store)
    context = restart_context("plugin-cpu", source_gpu_count=1, restart_budget=1)
    store.reserve_job_restart("cluster-a", "train-1", 1, context.idempotency_key)

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert outcome.details[gate.RECEIPT_KEY][NODE]["not_applicable"] is True


# ------------------------------------------------------------ the executor


def test_the_join_restart_holds_behind_the_repaired_nodes_device_plugin() -> None:
    """The DESTR-015 shape through the executor: RESTORE_SCHEDULING released
    the node, RESTART_WORKLOAD follows; the restart holds while kubelet still
    reports a device unhealthy and runs once allocatable equals capacity, the
    receipt on its record."""

    store = build_store()
    operations = [
        WorkflowOperation.RESTORE_SCHEDULING,
        WorkflowOperation.RESTART_WORKLOAD,
    ]
    _, workflow = workflow_state(store, operations)
    core = GpuNodeCore(allocatable=7)
    batch = FakeBatchApi()
    batch.active = 0
    adapter = _adapter(core, batch, store)
    steps = [
        copy_model(
            step,
            execution_owner=adapter.owner,
            workload_ids=["gpu-fault-system/job/training-job"],
            parameters={
                "cluster_id": "cluster-a",
                "job_id": "training-job",
                "source_attempt_id": "attempt-a",
                "source_gpu_count": 1,
                "restart_budget": 1,
            }
            if step.operation is WorkflowOperation.RESTART_WORKLOAD
            else {},
        )
        for step in workflow.official_steps
    ]
    workflow = copy_model(
        workflow,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.RESTORE_SCHEDULING],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.RESTORE_SCHEDULING,
                details={"restored_nodes": [NODE]},
            )
        ],
    )
    store.save_workflow(workflow)
    executor = active_workflow_executor(store, [adapter], operations)
    request = WorkflowExecutionRequest(expected_fencing_token=3)

    held = executor.execute(workflow.request_id, request)

    assert held.status is WorkflowStatus.RUNNING, held
    assert held.waiting_step_index == 1, held
    assert batch.created == {}, "no attempt was created onto unhealthy devices"
    record = next(
        item
        for item in store.get_workflow(workflow.request_id).step_executions
        if item.step_index == 1 and item.status is WorkflowStepStatus.WAITING
    )
    assert record.details[gate.PENDING_KEY]["nodes"][NODE]["allocatable"] == 7
    assert record.details["restart_submitted"] is False

    core.set_allocatable(8)
    completed = executor.execute(workflow.request_id, request)

    assert completed.status is WorkflowStatus.SUCCEEDED, completed
    assert len(batch.created) == 1, batch.created
    restart = next(
        item
        for item in store.get_workflow(workflow.request_id).step_executions
        if item.step_index == 1 and item.status is WorkflowStepStatus.SUCCEEDED
    )
    assert restart.details[gate.RECEIPT_KEY][NODE]["healthy"] is True, restart.details
    assert core.reads == [NODE, NODE], "the repaired node is the one judged"
