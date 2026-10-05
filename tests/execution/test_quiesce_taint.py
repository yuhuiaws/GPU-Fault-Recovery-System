"""The eks-managed quiesce taint (``gpu-fault.io/quiesce``, NoExecute).

Live 2026-10-05, gpu-eks-cluster DESTR-001 (XID 46 drill on ip-10-0-41-164):
kubelet stays up on a plain EKS node, so every GPU client container the
quiesce stopped -- the product's own DCGM exporter with its embedded
hostengine, the NVIDIA device plugin, GPU feature discovery -- was back within
seconds, VERIFY_NO_GPU_CLIENTS retried for seven minutes and the step failed
with "quiesce maintenance window expired". The executor now taints the node
before the quiesce and untaints it before the restore; DaemonSets that do not
tolerate the key are evicted for the window. HyperPod (no core client wired)
is byte-identical.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.adapters.common import QUIESCE_TAINT, quarantine_taint_value
from gpu_fault.adapters.node_action.quiesce_taint import (
    quiesce_taint,
    quiesce_taint_patch,
    quiesce_untaint_patch,
)
from gpu_fault.cluster_executor import executor_from_environment
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    node_action_result,
    workflow_step_execution,
)

from ._support import (
    FakeCoreApi,
    KubernetesWorkflowAdapter,
    NodeActionWorkflowAdapter,
    StubFleetRegistry,
    UnusedApi,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepContext,
    WorkflowStepStatus,
    workflow_state,
)

INCIDENT = "inc-xid46"
GPU_TAINT = {"key": "nvidia.com/gpu", "value": "present", "effect": "NoSchedule"}
NOW = datetime(2026, 10, 5, 15, 0, tzinfo=timezone.utc)


def _node(*taints: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": {"resourceVersion": "41", "annotations": {}},
        "spec": {"taints": list(taints)},
    }


def test_the_quiesce_taint_is_noexecute_and_keyed_on_the_incident() -> None:
    taint = quiesce_taint(INCIDENT)

    assert taint == {
        "key": QUIESCE_TAINT,
        "value": quarantine_taint_value(INCIDENT),
        "effect": "NoExecute",
    }, "NoSchedule would never evict the DaemonSet Pods already on the node"


def test_the_taint_patch_keeps_other_taints_and_is_idempotent() -> None:
    patch = quiesce_taint_patch(_node(GPU_TAINT), INCIDENT)

    assert patch is not None
    assert patch["metadata"]["resourceVersion"] == "41", "optimistic concurrency"
    assert patch["spec"]["taints"] == [GPU_TAINT, quiesce_taint(INCIDENT)]
    tainted = _node(GPU_TAINT, quiesce_taint(INCIDENT))
    assert quiesce_taint_patch(tainted, INCIDENT) is None, "already there: no patch"


def test_a_stale_quiesce_taint_of_another_incident_is_replaced() -> None:
    patch = quiesce_taint_patch(_node(quiesce_taint("inc-old"), GPU_TAINT), INCIDENT)

    assert patch is not None
    assert patch["spec"]["taints"] == [GPU_TAINT, quiesce_taint(INCIDENT)]


def test_the_untaint_patch_removes_only_the_quiesce_taint() -> None:
    patch = quiesce_untaint_patch(_node(GPU_TAINT, quiesce_taint(INCIDENT)))

    assert patch is not None
    assert patch["spec"]["taints"] == [GPU_TAINT]
    assert quiesce_untaint_patch(_node(GPU_TAINT)) is None, "nothing to remove"


def _steps_and_context(
    store: Any, *, operation: WorkflowOperation, registry: StubFleetRegistry
) -> WorkflowStepContext:
    incident, workflow = workflow_state(
        store,
        [
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        ],
    )
    steps = [
        copy_model(
            step,
            execution_owner="gpu-fault-node-agent",
            node_ids=["node-a"],
            gpu_uuids=["GPU-a"],
        )
        for step in workflow.official_steps
    ]
    executions = []
    if operation is WorkflowOperation.RESTORE_GPU_SERVICES:
        executions.append(
            workflow_step_execution(
                0,
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                details={
                    "agent_generations": {"node-a": 7},
                    "maintenance_window_expires_at": (
                        NOW + timedelta(hours=1)
                    ).isoformat(),
                },
            )
        )
    workflow = copy_model(workflow, official_steps=steps, step_executions=executions)
    store.save_workflow(workflow)
    registry.now = lambda: NOW
    index = 0 if operation is WorkflowOperation.QUIESCE_GPU_SERVICES else 1
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=steps[index],
        step_index=index,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key=f"workflow/quiesce-taint/{operation.value}",
    )


def _sender(core: FakeCoreApi, seen: list[list[str]]):
    def sender(_, envelope):
        seen.append([item["key"] for item in core.node["spec"]["taints"]])
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"stopped_services": ["dcgm-exporter"]},
        )

    return sender


def test_the_quiesce_taints_the_node_before_the_agent_is_told() -> None:
    core = FakeCoreApi()
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    context = _steps_and_context(
        build_store(),
        operation=WorkflowOperation.QUIESCE_GPU_SERVICES,
        registry=registry,
    )
    seen: list[list[str]] = []

    outcome = NodeActionWorkflowAdapter(
        {},
        "s" * 32,
        sender=_sender(core, seen),
        registry=registry,
        quiesce_taint_core=core,
    ).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert seen == [["sagemaker.amazonaws.com/node-health-status", QUIESCE_TAINT]], (
        "the DaemonSets must already be evicting when the agent stops the containers"
    )
    assert quiesce_taint(context.incident.incident_id) in core.node["spec"]["taints"]
    assert outcome.details["quiesce_taint_nodes"] == ["node-a"]


def test_a_quiesce_whose_taint_cannot_be_applied_is_refused_before_dispatch() -> None:
    class RefusingCore(FakeCoreApi):
        def patch_node(self, _, body):
            raise RuntimeError("nodes is forbidden: patch")

    core = RefusingCore()
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    context = _steps_and_context(
        build_store(),
        operation=WorkflowOperation.QUIESCE_GPU_SERVICES,
        registry=registry,
    )
    seen: list[list[str]] = []

    outcome = NodeActionWorkflowAdapter(
        {},
        "s" * 32,
        sender=_sender(core, seen),
        registry=registry,
        quiesce_taint_core=core,
    ).execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert "quiesce taint could not be applied before the quiesce" in (
        outcome.error or ""
    )
    assert "nodes is forbidden" in (outcome.error or "")
    assert seen == [], "fail closed: no agent command without the taint"
    assert outcome.details["quiesce_taint_error"].endswith(
        "nodes is forbidden: patch"
    ), "the API error is the evidence"


def test_the_restore_untaints_the_node_before_the_services_come_back() -> None:
    core = FakeCoreApi()
    core.node["spec"]["taints"].append(quiesce_taint("incident-active"))
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    context = _steps_and_context(
        build_store(),
        operation=WorkflowOperation.RESTORE_GPU_SERVICES,
        registry=registry,
    )
    seen: list[list[str]] = []

    outcome = NodeActionWorkflowAdapter(
        {},
        "s" * 32,
        sender=_sender(core, seen),
        registry=registry,
        quiesce_taint_core=core,
    ).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert seen == [["sagemaker.amazonaws.com/node-health-status"]], (
        "the DaemonSet Pods must be allowed back before the agent waits for them"
    )
    assert outcome.details["quiesce_taint_removed_nodes"] == ["node-a"]


def test_a_restore_whose_untaint_fails_still_restores_and_ends_failed() -> None:
    class RefusingCore(FakeCoreApi):
        def patch_node(self, _, body):
            raise RuntimeError("nodes is forbidden: patch")

    core = RefusingCore()
    core.node["spec"]["taints"].append(quiesce_taint("incident-active"))
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    context = _steps_and_context(
        build_store(),
        operation=WorkflowOperation.RESTORE_GPU_SERVICES,
        registry=registry,
    )
    seen: list[list[str]] = []

    outcome = NodeActionWorkflowAdapter(
        {},
        "s" * 32,
        sender=_sender(core, seen),
        registry=registry,
        quiesce_taint_core=core,
    ).execute(context)

    assert len(seen) == 1, "the host services are restored regardless"
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "quiesce taint could not be removed" in (outcome.error or "")
    assert outcome.details["quiesce_taint_error"].endswith(
        "nodes is forbidden: patch"
    ), "the API error is the evidence"
    assert outcome.details["node_results"]["node-a"]["stopped_services"] == [
        "dcgm-exporter"
    ], "the agent's receipt is kept next to the taint error"


@pytest.mark.parametrize(
    "operation",
    [WorkflowOperation.QUIESCE_GPU_SERVICES, WorkflowOperation.RESTORE_GPU_SERVICES],
)
def test_without_a_core_client_hyperpod_behaviour_is_unchanged(operation) -> None:
    core = FakeCoreApi()
    before = [dict(item) for item in core.node["spec"]["taints"]]
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    context = _steps_and_context(build_store(), operation=operation, registry=registry)
    seen: list[list[str]] = []

    outcome = NodeActionWorkflowAdapter(
        {}, "s" * 32, sender=_sender(core, seen), registry=registry
    ).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert core.node["spec"]["taints"] == before
    assert len(seen) == 1
    assert "quiesce_taint_nodes" not in outcome.details
    assert "quiesce_taint_removed_nodes" not in outcome.details


def test_restore_scheduling_strips_a_lingering_quiesce_taint() -> None:
    """A reset that FAILED between the quiesce and the restore leaves the
    NoExecute taint on the node; the validated restore's RESTORE_SCHEDULING
    (same incident, same fencing token) releases it with the cordon."""

    core = FakeCoreApi()
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.RESTORE_SCHEDULING])
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi()
    )
    workflow = copy_model(
        workflow,
        official_steps=[
            copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
        ],
    )
    store.save_workflow(workflow)
    core.node["metadata"]["annotations"] = {
        "gpu-fault.io/incident-id": incident.incident_id,
        "gpu-fault.io/fencing-token": str(workflow.fencing_token),
        "gpu-fault.io/previous-unschedulable": "false",
    }
    core.node["spec"]["unschedulable"] = True
    core.node["spec"]["taints"].extend(
        [
            {
                "key": "gpu-fault.io/quarantined",
                "value": quarantine_taint_value(incident.incident_id),
                "effect": "NoSchedule",
            },
            quiesce_taint(incident.incident_id),
        ]
    )

    result = execute_workflow(
        active_workflow_executor(
            store, [adapter], {WorkflowOperation.RESTORE_SCHEDULING}
        ),
        workflow.request_id,
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    assert core.node["spec"]["unschedulable"] is False
    assert core.node["spec"]["taints"] == [
        {
            "key": "sagemaker.amazonaws.com/node-health-status",
            "value": "Unschedulable",
            "effect": "NoSchedule",
        }
    ]
    assert core.node["metadata"]["annotations"] == {}


@pytest.mark.parametrize(
    ("provider", "wired"), [("eks-managed", True), ("hyperpod", False), (None, False)]
)
def test_bootstrap_wires_the_core_client_only_on_eks_managed(
    monkeypatch, tmp_path, provider: str | None, wired: bool
) -> None:
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", "token")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "gpu-eks-cluster")
    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(tmp_path / "claims.json")
    )
    monkeypatch.setenv("GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER", "true")
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_SECRET", "s" * 32)
    monkeypatch.setenv("GPU_FAULT_NODE_KEY_SYNC_SECONDS", "0")
    monkeypatch.delenv("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", raising=False)
    if provider is None:
        monkeypatch.delenv("GPU_FAULT_CLUSTER_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("GPU_FAULT_CLUSTER_PROVIDER", provider)
    sentinel = object()

    class FakeKubernetesAdapter(KubernetesWorkflowAdapter):
        owner = "gpu-fault-kubernetes-adapter"
        core = sentinel

        def __init__(self, **_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.bootstrap.KubernetesWorkflowAdapter",
        FakeKubernetesAdapter,
    )

    executor = executor_from_environment()

    (node_action,) = [
        adapter
        for adapter in executor.adapters
        if isinstance(adapter, NodeActionWorkflowAdapter)
    ]
    assert (node_action.quiesce_taint_core is sentinel) is wired, (
        "the taint is an eks-managed mechanism; HyperPod stops kubelet instead"
    )
