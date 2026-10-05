"""A multi-node isolation is judged on every node before any node is patched.

Live on HyperPod (GF-REGIONAL-COLLECT-016): a ``reboot-after`` successor's
MARK_UNSCHEDULABLE named two nodes, cordoned the first (a healthy sibling),
was refused on the second -- its predecessor incident still owned that node's
isolation -- and ended FAILED half a second after it started. The incident
stayed ESCALATED (no isolation step completed) while the sibling kept the
successor's cordon, taint and annotations: an orphaned isolation with no
operator exit. The refusal is now raised before the first patch, and a
refusal that still lands mid-step (the node moved between reads) names the
nodes it did isolate so the terminal rule keeps the incident QUARANTINED.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from gpu_fault.adapters import KubernetesWorkflowAdapter, quarantine_taint_value
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.execution.terminal_state import failure_incident_state
from gpu_fault.models import (
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import UnusedApi

PREDECESSOR = "inc-reset-drill"
SUCCESSOR = "inc-reboot-after-wf-reset-drill"
HEALTHY = "node-sibling"
OWNED = "node-reset"


class TwoNodeCore:
    """Two Nodes; ``read_node`` and ``patch_node`` by name, 404 for others."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {
            name: {
                "metadata": {"name": name, "resourceVersion": "1", "annotations": {}},
                "spec": {"unschedulable": False, "taints": []},
            }
            for name in (HEALTHY, OWNED)
        }
        self.patches: list[str] = []
        self.before_patch: list[Any] = []

    def own(self, name: str, incident_id: str, token: str = "1") -> None:
        node = self.nodes[name]
        node["metadata"]["annotations"] = {
            "gpu-fault.io/incident-id": incident_id,
            "gpu-fault.io/fencing-token": token,
            "gpu-fault.io/previous-unschedulable": "false",
        }
        node["spec"]["unschedulable"] = True
        node["spec"]["taints"] = [
            {
                "key": "gpu-fault.io/quarantined",
                "value": quarantine_taint_value(incident_id),
                "effect": "NoSchedule",
            }
        ]

    def read_node(self, name: str) -> dict[str, Any]:
        if name not in self.nodes:
            error = KeyError(name)
            error.status = 404  # type: ignore[attr-defined]
            raise error
        return deepcopy(self.nodes[name])

    def patch_node(self, name: str, body: dict[str, Any]) -> None:
        for hook in self.before_patch:
            hook(name)
        self.patches.append(name)
        node = self.nodes[name]
        node["metadata"]["annotations"].update(body["metadata"]["annotations"])
        node["spec"].update(body["spec"])


def _records(store, *, successor_nodes: list[str]):
    predecessor_workflow = workflow_request(
        "wf-reset-drill",
        PREDECESSOR,
        status=WorkflowStatus.FAILED,
        fencing_token=1,
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=[OWNED]),
            workflow_step(WorkflowOperation.RESET_GPU, node_ids=[OWNED]),
            workflow_step(WorkflowOperation.RESTORE_SCHEDULING, node_ids=[OWNED]),
        ],
        completed_step_indexes=[0, 1],
        completed_operations=[
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.RESET_GPU,
        ],
    )
    predecessor = fault_incident(
        PREDECESSOR,
        "event-reset-drill",
        state=IncidentState.QUARANTINED,
        workflow_request_id=predecessor_workflow.request_id,
        node_ids=[OWNED],
        fencing_token=1,
    )
    store.save_incident(predecessor)
    store.save_workflow(predecessor_workflow)
    step = workflow_step(
        WorkflowOperation.MARK_UNSCHEDULABLE,
        "gpu-fault-kubernetes-adapter",
        node_ids=successor_nodes,
    )
    successor_workflow = workflow_request(
        "workflow-reboot-after-wf-reset-drill",
        SUCCESSOR,
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[
            step,
            workflow_step(WorkflowOperation.RESTART_NODE, node_ids=successor_nodes),
            workflow_step(
                WorkflowOperation.RESTORE_SCHEDULING, node_ids=successor_nodes
            ),
        ],
    )
    successor = fault_incident(
        SUCCESSOR,
        "reboot-after-wf-reset-drill",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=successor_workflow.request_id,
        node_ids=successor_nodes,
        fencing_token=1,
    )
    store.save_incident(successor)
    store.save_workflow(successor_workflow)
    return WorkflowStepContext(
        workflow=successor_workflow,
        incident=successor,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=1),
        idempotency_key="reboot-after/mark",
    )


def _adapter(core: TwoNodeCore, store) -> KubernetesWorkflowAdapter:
    return KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi(), store=store
    )


def test_a_refused_node_stops_the_step_before_any_sibling_is_cordoned() -> None:
    store = build_store()
    core = TwoNodeCore()
    core.own(OWNED, PREDECESSOR)
    context = _records(store, successor_nodes=[HEALTHY, OWNED])

    outcome = _adapter(core, store).execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert "already isolated by another incident" in (outcome.error or "")
    assert outcome.details["safety_rejection"] is True
    assert outcome.details["node_id"] == OWNED
    assert outcome.details["isolated_nodes"] == []
    assert core.patches == [], "nothing was written for a step that is refused"
    sibling = core.nodes[HEALTHY]
    assert sibling["spec"]["unschedulable"] is False
    assert sibling["spec"]["taints"] == [] and sibling["metadata"]["annotations"] == {}
    owned = core.nodes[OWNED]["metadata"]["annotations"]
    assert owned["gpu-fault.io/incident-id"] == PREDECESSOR, (
        "the predecessor keeps the node it isolated"
    )


def test_an_absent_node_is_refused_before_its_siblings_are_cordoned() -> None:
    store = build_store()
    core = TwoNodeCore()
    context = _records(store, successor_nodes=[HEALTHY, "node-gone"])

    outcome = _adapter(core, store).execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert (
        outcome.details["absent"] is True and outcome.details["node_id"] == "node-gone"
    )
    assert outcome.details["isolated_nodes"] == []
    assert core.patches == [] and core.nodes[HEALTHY]["spec"]["unschedulable"] is False


def test_an_unowned_step_still_isolates_every_node_it_names() -> None:
    store = build_store()
    core = TwoNodeCore()
    context = _records(store, successor_nodes=[HEALTHY, OWNED])

    outcome = _adapter(core, store).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert outcome.details["isolated_nodes"] == [HEALTHY, OWNED]
    assert core.patches == [HEALTHY, OWNED]
    for name in (HEALTHY, OWNED):
        annotations = core.nodes[name]["metadata"]["annotations"]
        assert annotations["gpu-fault.io/incident-id"] == SUCCESSOR
        assert core.nodes[name]["spec"]["unschedulable"] is True


def test_a_refusal_that_lands_after_the_precheck_names_the_nodes_it_isolated() -> None:
    """The node moved between the pre-check and its patch: the nodes already
    cordoned are recorded, and the terminal rule reads them as a quarantine."""

    store = build_store()
    core = TwoNodeCore()
    context = _records(store, successor_nodes=[HEALTHY, OWNED])
    # Another incident takes the second node the moment the first is patched.
    core.before_patch.append(
        lambda name: core.own(OWNED, PREDECESSOR) if name == HEALTHY else None
    )

    outcome = _adapter(core, store).execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["safety_rejection"] is True
    assert outcome.details["node_id"] == OWNED
    assert outcome.details["isolated_nodes"] == [HEALTHY]
    assert core.nodes[HEALTHY]["metadata"]["annotations"][
        "gpu-fault.io/incident-id"
    ] == (SUCCESSOR)
    failed = context.workflow.model_copy(
        update={
            "status": WorkflowStatus.FAILED,
            "step_executions": [
                workflow_step_execution(
                    0,
                    WorkflowOperation.MARK_UNSCHEDULABLE,
                    WorkflowStepStatus.FAILED,
                    phase="official",
                    error=outcome.error,
                    details=dict(outcome.details),
                )
            ],
        }
    )
    assert failure_incident_state(failed) is IncidentState.QUARANTINED, (
        "a partial isolation holds the nodes it cordoned; the exit is the "
        "validated restore, not a close that orphans them"
    )
    clean = failed.model_copy(
        update={
            "step_executions": [
                workflow_step_execution(
                    0,
                    WorkflowOperation.MARK_UNSCHEDULABLE,
                    WorkflowStepStatus.FAILED,
                    phase="official",
                    details={"safety_rejection": True, "isolated_nodes": []},
                )
            ]
        }
    )
    assert failure_incident_state(clean) is IncidentState.ESCALATED, (
        "a refusal that isolated nothing leaves nothing to restore"
    )
