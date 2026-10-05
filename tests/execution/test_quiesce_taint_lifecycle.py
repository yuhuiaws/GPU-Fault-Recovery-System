"""Every exit of a quiesce window removes the eks-managed quiesce taint.

Live (gpu-eks-cluster, XID 46 on ip-10-0-41-164): the reset chain's
RESTORE_GPU_SERVICES compensation was refused by the executor (a stale
generation), the ``gpu-fault.io/quiesce`` NoExecute taint stayed on the node,
the exporter was evicted, the reboot successor's VALIDATE_GPU waited 600 s for
telemetry that could not come, and no product exit removed the taint -- an
operator did, by hand. The taint's life is now recorded on every outcome of
the step that applied it, a failed quiesce lifts it at once, a quiesce that
never completed still owes its restore, the successors' isolation strips an
orphaned one, and a restore under the owning incident strips it even when
another incident has since re-isolated the node.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.adapters.common import QUIESCE_TAINT, quarantine_taint_value
from gpu_fault.adapters.node_action.quiesce_taint import quiesce_taint
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.execution.node_action_uncertainty import unrestored_quiesces
from gpu_fault.models import (
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import NodeActionStatus
from gpu_fault.workflow_quarantine import quiesce_taint_nodes_held
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    node_action_result,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import (
    FakeCoreApi,
    NodeActionWorkflowAdapter,
    StubFleetRegistry,
    UnusedApi,
    workflow_state,
)
from tests.execution.test_isolation_precheck import OWNED, TwoNodeCore

OP = WorkflowOperation
NOW = datetime(2026, 10, 5, 20, 11, tzinfo=timezone.utc)


# ------------------------------------------------------------ node-action step


def _quiesce_context(store, *, registry: StubFleetRegistry) -> WorkflowStepContext:
    incident, workflow = workflow_state(
        store, [OP.QUIESCE_GPU_SERVICES, OP.RESTORE_GPU_SERVICES]
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
    workflow = copy_model(workflow, official_steps=steps)
    store.save_workflow(workflow)
    registry.now = lambda: NOW
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=steps[0],
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="workflow/quiesce-lifecycle/QUIESCE_GPU_SERVICES",
    )


def _agent(answer):
    def sender(_, envelope):
        return answer(envelope)

    return sender


def _adapter(core: FakeCoreApi, registry: StubFleetRegistry, sender):
    return NodeActionWorkflowAdapter(
        {}, "s" * 32, sender=sender, registry=registry, quiesce_taint_core=core
    )


def test_a_quiesce_the_agent_refuses_untaints_at_once_and_says_so() -> None:
    core = FakeCoreApi()
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    context = _quiesce_context(build_store(), registry=registry)
    taints_seen: list[list[str]] = []

    def answer(envelope):
        taints_seen.append([item["key"] for item in core.node["spec"]["taints"]])
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            status=NodeActionStatus.FAILED,
            error="RuntimeError: quiesce state belongs to another workflow",
            retryable=False,
        )

    outcome = _adapter(core, registry, _agent(answer)).execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert QUIESCE_TAINT in taints_seen[0], "tainted before the agent was told"
    assert outcome.details["quiesce_taint_nodes"] == ["node-a"]
    assert outcome.details["quiesce_taint_removed_nodes"] == ["node-a"]
    assert QUIESCE_TAINT not in [item["key"] for item in core.node["spec"]["taints"]], (
        "a window that never opened leaves no NoExecute taint behind"
    )


def test_a_waiting_quiesce_records_the_taint_it_applied() -> None:
    core = FakeCoreApi()
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    context = _quiesce_context(build_store(), registry=registry)

    def answer(envelope):
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            status=NodeActionStatus.FAILED,
            error="agent busy",
            retryable=True,
        )

    outcome = _adapter(core, registry, _agent(answer)).execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome.error
    assert outcome.details["quiesce_taint_nodes"] == ["node-a"]
    assert "quiesce_taint_removed_nodes" not in outcome.details
    assert quiesce_taint(context.incident.incident_id) in core.node["spec"]["taints"], (
        "a quiesce still in flight keeps its taint"
    )


def test_a_completed_quiesce_lifts_the_agent_service_evidence_into_the_step() -> None:
    core = FakeCoreApi()
    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    context = _quiesce_context(build_store(), registry=registry)

    def answer(envelope):
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={
                "quiesced": True,
                "active_services": ["nvidia-persistenced", "gpu-fault-host-collector"],
                "inactive_services": ["nvidia-fabricmanager", "nvidia-dcgm"],
                "protected_services": ["kubelet", "containerd"],
                "stopped_containers": [
                    {"selector": "kube-system/nvidia-device-plugin-ctr"}
                ],
                "failsafe_seconds": 420,
            },
        )

    outcome = _adapter(core, registry, _agent(answer)).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert outcome.details["stopped_services"] == {
        "node-a": ["nvidia-persistenced", "gpu-fault-host-collector"]
    }
    assert outcome.details["inactive_services"] == {
        "node-a": ["nvidia-fabricmanager", "nvidia-dcgm"]
    }
    assert outcome.details["protected_services"] == {
        "node-a": ["kubelet", "containerd"]
    }
    assert outcome.details["stopped_containers"] == {
        "node-a": [{"selector": "kube-system/nvidia-device-plugin-ctr"}]
    }
    assert outcome.details["quiesce_taint_nodes"] == ["node-a"]


# ------------------------------------------------------ executor compensation


def _chain(executions, *, completed: list[int]):
    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
        workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=["node-a"]),
        workflow_step(OP.RESET_GPU, node_ids=["node-a"]),
        workflow_step(OP.RESTORE_GPU_SERVICES, node_ids=["node-a"]),
    ]
    return workflow_request(
        "wf-reset",
        "inc-reset",
        status=WorkflowStatus.RUNNING,
        official_steps=steps,
        completed_step_indexes=completed,
        step_executions=executions,
    )


def test_a_quiesce_that_tainted_but_never_completed_still_owes_its_restore() -> None:
    failed = _chain(
        [
            workflow_step_execution(0, OP.MARK_UNSCHEDULABLE),
            workflow_step_execution(
                1,
                OP.QUIESCE_GPU_SERVICES,
                WorkflowStepStatus.FAILED,
                details={"quiesce_taint_nodes": ["node-a"]},
            ),
        ],
        completed=[0],
    )

    owed = unrestored_quiesces(failed, None, settling_operations={OP.RESTART_NODE})

    assert quiesce_taint_nodes_held(failed) == {"node-a"}
    assert [(index, sorted(nodes)) for index, _, nodes in owed] == [(1, ["node-a"])]


def test_a_failed_quiesce_that_already_untainted_owes_nothing() -> None:
    cleaned = _chain(
        [
            workflow_step_execution(0, OP.MARK_UNSCHEDULABLE),
            workflow_step_execution(
                1,
                OP.QUIESCE_GPU_SERVICES,
                WorkflowStepStatus.FAILED,
                details={
                    "quiesce_taint_nodes": ["node-a"],
                    "quiesce_taint_removed_nodes": ["node-a"],
                },
            ),
        ],
        completed=[0],
    )

    assert quiesce_taint_nodes_held(cleaned) == set()
    assert unrestored_quiesces(cleaned, None, settling_operations=set()) == []


def test_a_completed_quiesce_is_owed_as_before_whatever_the_taint_record_says() -> None:
    completed = _chain(
        [
            workflow_step_execution(0, OP.MARK_UNSCHEDULABLE),
            workflow_step_execution(1, OP.QUIESCE_GPU_SERVICES, details={}),
        ],
        completed=[0, 1],
    )

    owed = unrestored_quiesces(completed, None, settling_operations=set())

    assert [(index, sorted(nodes)) for index, _, nodes in owed] == [(1, ["node-a"])], (
        "HyperPod records (no taint) keep the completed-quiesce rule"
    )


# ------------------------------------------------- kubernetes isolation steps

PREDECESSOR = "inc-reset-drill"
PREDECESSOR_WORKFLOW = "wf-reset-drill"
SUCCESSOR = f"inc-reboot-after-{PREDECESSOR_WORKFLOW}"


def _records(store, *, taints: list[dict], own_quiesce_held: bool = False):
    predecessor = workflow_request(
        PREDECESSOR_WORKFLOW,
        PREDECESSOR,
        status=WorkflowStatus.FAILED,
        fencing_token=1,
        official_steps=[
            workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=[OWNED]),
            workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=[OWNED]),
            workflow_step(OP.RESET_GPU, node_ids=[OWNED]),
        ],
        completed_step_indexes=[0, 1],
        completed_operations=[OP.MARK_UNSCHEDULABLE, OP.QUIESCE_GPU_SERVICES],
    )
    store.save_incident(
        fault_incident(
            PREDECESSOR,
            "event-reset-drill",
            state=IncidentState.QUARANTINED,
            workflow_request_id=PREDECESSOR_WORKFLOW,
            node_ids=[OWNED],
            fencing_token=1,
        )
    )
    store.save_workflow(predecessor)
    step = workflow_step(
        OP.MARK_UNSCHEDULABLE, "gpu-fault-kubernetes-adapter", node_ids=[OWNED]
    )
    executions = (
        [
            workflow_step_execution(
                1, OP.QUIESCE_GPU_SERVICES, details={"quiesce_taint_nodes": [OWNED]}
            )
        ]
        if own_quiesce_held
        else []
    )
    successor_workflow = workflow_request(
        f"workflow-{SUCCESSOR}",
        SUCCESSOR,
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[
            step,
            workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=[OWNED]),
            workflow_step(OP.RESTART_NODE, node_ids=[OWNED]),
        ],
        step_executions=executions,
    )
    successor = fault_incident(
        SUCCESSOR,
        f"reboot-after-{PREDECESSOR_WORKFLOW}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=successor_workflow.request_id,
        node_ids=[OWNED],
        fencing_token=1,
    )
    store.save_incident(successor)
    store.save_workflow(successor_workflow)
    core = TwoNodeCore()
    core.own(OWNED, PREDECESSOR)
    core.nodes[OWNED]["spec"]["taints"].extend(taints)
    context = WorkflowStepContext(
        workflow=successor_workflow,
        incident=successor,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=1),
        idempotency_key="reboot-after/mark",
    )
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi(), store=store
    )
    return core, adapter, context


def _taint_values(core: TwoNodeCore) -> list[str]:
    return [
        item["value"]
        for item in core.nodes[OWNED]["spec"]["taints"]
        if item["key"] == QUIESCE_TAINT
    ]


def test_the_successor_isolation_strips_the_taken_over_owners_quiesce_taint() -> None:
    core, adapter, context = _records(
        build_store(), taints=[quiesce_taint(PREDECESSOR)]
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert _taint_values(core) == [], "the owner ended without its restore: orphaned"
    assert core.nodes[OWNED]["metadata"]["annotations"]["gpu-fault.io/incident-id"] == (
        SUCCESSOR
    )


def test_the_successor_isolation_strips_its_own_incidents_older_quiesce_taint() -> None:
    core, adapter, context = _records(build_store(), taints=[quiesce_taint(SUCCESSOR)])

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert _taint_values(core) == [], (
        "an older generation of this incident left it; this workflow holds none"
    )


def test_a_quiesce_taint_this_workflow_holds_stays_through_its_isolation() -> None:
    core, adapter, context = _records(
        build_store(), taints=[quiesce_taint(SUCCESSOR)], own_quiesce_held=True
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert _taint_values(core) == [quarantine_taint_value(SUCCESSOR)], (
        "the record says this workflow applied it and has not removed it"
    )


def test_a_live_strangers_quiesce_taint_is_left_alone() -> None:
    core, adapter, context = _records(
        build_store(), taints=[quiesce_taint("inc-elsewhere")]
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert _taint_values(core) == [quarantine_taint_value("inc-elsewhere")], (
        "a value nobody here can vouch for is not this step's to lift"
    )


# ------------------------------------------------- RESTORE_SCHEDULING release


def test_the_owner_incidents_restore_strips_only_its_quiesce_taint() -> None:
    """Another incident cordoned the node after the reset chain died; the
    chain's incident still owns -- and only it may lift -- the quiesce taint."""

    store = build_store()
    core = TwoNodeCore()
    core.own(OWNED, "inc-support-later", token="1")
    core.nodes[OWNED]["spec"]["taints"].append(quiesce_taint(PREDECESSOR))
    step = workflow_step(
        OP.RESTORE_SCHEDULING, "gpu-fault-kubernetes-adapter", node_ids=[OWNED]
    )
    workflow = workflow_request(
        "workflow-validated-restore-1",
        PREDECESSOR,
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[step],
    )
    incident = fault_incident(
        PREDECESSOR,
        "event-reset-drill",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=workflow.request_id,
        node_ids=[OWNED],
        fencing_token=1,
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi(), store=store
    )
    before = dict(core.nodes[OWNED]["metadata"]["annotations"])

    outcome = adapter.execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=1),
            idempotency_key="restore/release",
        )
    )

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert outcome.details["restored_nodes"] == [OWNED]
    node = core.nodes[OWNED]
    assert _taint_values(core) == []
    assert node["spec"]["unschedulable"] is True, "the other incident's cordon stays"
    assert node["metadata"]["annotations"] == before, "and so do its annotations"
    assert [item["key"] for item in node["spec"]["taints"]] == [
        "gpu-fault.io/quarantined"
    ], "the other incident's quarantine taint is untouched"
