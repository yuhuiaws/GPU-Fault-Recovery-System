"""A node's isolation transfers down the whole escalation chain, not one rung.

Live (gpu-eks-cluster): a plugin-restart workflow cordoned the node and FAILED
("expected at most one device plugin Pod, found 2"); its ``reboot-after``
successor was refused before it re-annotated the node; the ``support-after``
successor of *that* then asked whether it may take the node over from the
original incident. The one-hop check (``escalation_continues``: is my event id
``<rung>-after-<the owner's current workflow>``) said no -- the owner's workflow
was two rungs back -- and MARK_UNSCHEDULABLE failed "node is already isolated
by another incident/token". The ownership verdict now carries
``escalation_lineage``: the control plane (or the local store) walks the
successor's ``predecessor_workflow_id`` pointers and ``-after-`` event ids
back to the owner, over any number of rungs.
"""

from __future__ import annotations

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.escalation import escalation_descends_from
from gpu_fault.regional import RemoteIncidentOwnershipReport
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.execution._support import UnusedApi
from tests.execution.test_isolation_precheck import OWNED, TwoNodeCore

OWNER = "inc-plugin-restart"
OWNER_WORKFLOW = "wf-plugin-restart"
REBOOT_WORKFLOW = f"workflow-reboot-after-{OWNER_WORKFLOW}"
SUPPORT = f"inc-support-after-{REBOOT_WORKFLOW}"
SUPPORT_WORKFLOW = f"workflow-support-after-{REBOOT_WORKFLOW}"


def _chain(store) -> WorkflowStepContext:
    """owner (FAILED after its cordon) -> reboot-after (FAILED, refused before
    it could re-annotate) -> support-after (RUNNING, about to isolate)."""

    owner_workflow = workflow_request(
        OWNER_WORKFLOW,
        OWNER,
        status=WorkflowStatus.FAILED,
        fencing_token=1,
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=[OWNED]),
            workflow_step(
                WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN, node_ids=[OWNED]
            ),
        ],
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.MARK_UNSCHEDULABLE],
    )
    store.save_incident(
        fault_incident(
            OWNER,
            "event-allocatable",
            state=IncidentState.QUARANTINED,
            workflow_request_id=OWNER_WORKFLOW,
            node_ids=[OWNED],
            fencing_token=1,
        )
    )
    store.save_workflow(owner_workflow)
    reboot_incident = f"inc-reboot-after-{OWNER_WORKFLOW}"
    store.save_incident(
        fault_incident(
            reboot_incident,
            f"reboot-after-{OWNER_WORKFLOW}",
            state=IncidentState.ESCALATED,
            workflow_request_id=REBOOT_WORKFLOW,
            node_ids=[OWNED],
            fencing_token=1,
        )
    )
    store.save_workflow(
        workflow_request(
            REBOOT_WORKFLOW,
            reboot_incident,
            status=WorkflowStatus.FAILED,
            fencing_token=1,
            official_steps=[
                workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=[OWNED]),
                workflow_step(WorkflowOperation.RESTART_NODE, node_ids=[OWNED]),
            ],
        )
    )
    step = workflow_step(
        WorkflowOperation.MARK_UNSCHEDULABLE,
        "gpu-fault-kubernetes-adapter",
        node_ids=[OWNED],
    )
    support_workflow = workflow_request(
        SUPPORT_WORKFLOW,
        SUPPORT,
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[
            step,
            workflow_step(WorkflowOperation.ESCALATE_SUPPORT, node_ids=[OWNED]),
        ],
    )
    support = fault_incident(
        SUPPORT,
        f"support-after-{REBOOT_WORKFLOW}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=SUPPORT_WORKFLOW,
        node_ids=[OWNED],
        fencing_token=1,
    )
    store.save_incident(support)
    store.save_workflow(support_workflow)
    return WorkflowStepContext(
        workflow=support_workflow,
        incident=support,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=1),
        idempotency_key="support-after/mark",
    )


def test_the_lineage_walk_crosses_two_rungs_and_stops_at_strangers() -> None:
    store = build_store()
    _chain(store)

    assert escalation_descends_from(store, SUPPORT_WORKFLOW, OWNER) is True
    assert escalation_descends_from(store, REBOOT_WORKFLOW, OWNER) is True
    assert escalation_descends_from(store, SUPPORT_WORKFLOW, "inc-unrelated") is False
    assert escalation_descends_from(store, OWNER_WORKFLOW, OWNER) is False, (
        "a workflow is not its own descendant"
    )
    assert escalation_descends_from(store, "wf-missing", OWNER) is False
    assert escalation_descends_from(store, None, OWNER) is False


def test_the_two_hop_successor_takes_the_node_over_with_a_store() -> None:
    store = build_store()
    core = TwoNodeCore()
    core.own(OWNED, OWNER)
    context = _chain(store)
    adapter = KubernetesWorkflowAdapter(
        core_api=core, batch_api=UnusedApi(), custom_api=UnusedApi(), store=store
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    annotations = core.nodes[OWNED]["metadata"]["annotations"]
    assert annotations["gpu-fault.io/incident-id"] == SUPPORT
    assert annotations["gpu-fault.io/previous-unschedulable"] == "false", (
        "the baseline from before the first cordon is kept across the takeover"
    )


class ReportProvider:
    def __init__(self, report: RemoteIncidentOwnershipReport) -> None:
        self.report = report
        self.queries: list[tuple[str, str | None]] = []

    def incident_ownership(
        self, incident_id: str, *, successor_workflow_id: str | None = None
    ) -> RemoteIncidentOwnershipReport:
        self.queries.append((incident_id, successor_workflow_id))
        return self.report


def _report(**overrides: object) -> RemoteIncidentOwnershipReport:
    values: dict[str, object] = {
        "incident_id": OWNER,
        "known": True,
        "workflow_request_id": OWNER_WORKFLOW,
        "workflow_status": "FAILED",
        "terminal": True,
        "incident_state": "QUARANTINED",
        "quarantine_hold": True,
        "terminal_quarantine": False,
        "escalation_lineage": True,
    }
    values.update(overrides)
    return RemoteIncidentOwnershipReport.model_validate(values)


def _storeless(
    core: TwoNodeCore, provider: ReportProvider
) -> KubernetesWorkflowAdapter:
    return KubernetesWorkflowAdapter(
        core_api=core,
        batch_api=UnusedApi(),
        custom_api=UnusedApi(),
        store=None,
        ownership_provider=provider,
    )


def test_the_storeless_executor_asks_about_its_own_lineage_and_takes_over() -> None:
    core = TwoNodeCore()
    core.own(OWNED, OWNER)
    context = _chain(build_store())
    provider = ReportProvider(_report())

    outcome = _storeless(core, provider).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    assert provider.queries == [(OWNER, SUPPORT_WORKFLOW)], (
        "one lookup, naming the asking workflow so the control plane walks the chain"
    )
    assert core.nodes[OWNED]["metadata"]["annotations"]["gpu-fault.io/incident-id"] == (
        SUPPORT
    )


def test_a_control_plane_that_denies_or_cannot_say_the_lineage_refuses() -> None:
    for lineage in (False, None):
        core = TwoNodeCore()
        core.own(OWNED, OWNER)
        context = _chain(build_store())

        outcome = _storeless(
            core, ReportProvider(_report(escalation_lineage=lineage))
        ).execute(context)

        assert outcome.status is WorkflowStepStatus.FAILED, lineage
        assert "already isolated by another incident" in (outcome.error or "")
        assert (
            core.nodes[OWNED]["metadata"]["annotations"]["gpu-fault.io/incident-id"]
            == OWNER
        ), "two rungs away and unvouched: the hold stays with its owner"


def test_an_older_provider_without_the_keyword_is_still_asked() -> None:
    class OldProvider:
        def __init__(self) -> None:
            self.queried: list[str] = []

        def incident_ownership(self, incident_id: str) -> RemoteIncidentOwnershipReport:
            self.queried.append(incident_id)
            return _report(escalation_lineage=None)

    core = TwoNodeCore()
    core.own(OWNED, OWNER)
    provider = OldProvider()

    outcome = _storeless(core, provider).execute(_chain(build_store()))

    assert provider.queried == [OWNER]
    assert outcome.status is WorkflowStepStatus.FAILED, (
        "no lineage answer, two rungs away: refused, exactly as before"
    )
