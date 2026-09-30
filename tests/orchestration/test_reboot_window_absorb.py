"""A node the product is rebooting must not grow a second incident.

Live 2026-09-24: while RESTART_NODE was rebooting a node, (1) a site-node-health
``gpu_kubernetes_allocatable_mismatch`` marker (COLLECT-004) and (2) a kernel-log
XID 46 line replayed with the node's pre-reboot boot id (DESTR-014) each opened
their own incident and a RESTART_GPU_DEVICE_PLUGIN workflow that failed on the
rebooting node; the second one escalated to support and quarantined the node
under the support incident's taint. Both signals belong to the reboot that is
already under way: they are absorbed into the owning incident as evidence, and
a kernel-log event from a boot the node has since left opens no workflow.
"""

from __future__ import annotations

from datetime import timedelta

from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    build_context,
    fault_incident,
    node_health_finding,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.orchestration._support import NOW, _active_agent, event, ingest

NODE = "node-a"
TAIL = (
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.VALIDATE_HOST,
    WorkflowOperation.VALIDATE_FABRIC,
    WorkflowOperation.RESTORE_SCHEDULING,
)


def _seed_reboot(store, *, tail_done: bool = False) -> None:
    """One RUNNING reboot workflow owning node-a: RESTART_NODE has run, the
    validation tail has not (or has, when ``tail_done``)."""

    operations = [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.RESTART_NODE,
        *TAIL,
    ]
    done = list(range(3)) + (list(range(3, len(operations))) if tail_done else [])
    incident = fault_incident(
        "inc-reboot",
        "event-reboot",
        event_type="NODE_HEALTH",
        node_ids=[NODE],
        state=IncidentState.ACTION_PENDING,
        official_action="REBOOT_NODE",
        workflow_request_id="wf-reboot",
    )
    workflow = workflow_request(
        "wf-reboot",
        "inc-reboot",
        status=WorkflowStatus.SUCCEEDED if tail_done else WorkflowStatus.RUNNING,
        official_action="REBOOT_NODE",
        official_steps=[workflow_step(op, node_ids=[NODE]) for op in operations],
        completed_step_indexes=done,
        completed_operations=[operations[i] for i in done],
        step_executions=[
            workflow_step_execution(i, operations[i], WorkflowStepStatus.SUCCEEDED)
            for i in done
        ],
        updated_at=NOW - timedelta(seconds=30),
    )
    store.save_incident_and_workflow(incident, workflow)


def _allocatable_finding(event_id: str):
    return node_health_finding(
        f"finding-{event_id}",
        event_id,
        node_id=NODE,
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        metric_name="gpu_kubernetes_allocatable_mismatch",
        reason="Node allocatable nvidia.com/gpu 0 below the host inventory 8",
        recommended_action=RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
        runtime_profile_version="simulated-v1",
    )


def test_health_marker_on_a_rebooting_node_joins_the_owning_incident() -> None:
    context = build_context()
    _seed_reboot(context.store)

    incident, workflow = context.orchestrator.ingest_node_health(
        _allocatable_finding("event-allocatable")
    )

    assert incident.incident_id == "inc-reboot", (
        "the allocatable mismatch seen mid-reboot belongs to the reboot's incident"
    )
    assert workflow is not None and workflow.request_id == "wf-reboot", (
        "the owning reboot workflow is the only one returned"
    )
    assert [w.request_id for w in context.store.list_workflows()] == ["wf-reboot"], (
        "no second workflow was planned for a node the product is rebooting"
    )
    linked = context.store.get_incident_by_event("event-allocatable")
    assert linked is not None and linked.incident_id == "inc-reboot", (
        "the event is linked to the owning incident so a re-post stays absorbed"
    )


def test_kernel_log_xid_on_a_rebooting_node_joins_the_owning_incident() -> None:
    context = build_context()
    _seed_reboot(context.store)
    context.store.save_agent(_active_agent(boot_id="boot-old"))
    xid = event(46, event_id="xid-mid-reboot").model_copy(
        update={"source_boot_id": "boot-old"}
    )

    _, incident, workflow = ingest(context, xid)

    assert incident.incident_id == "inc-reboot", (
        "a kernel-log fault during the product's own reboot is that reboot's evidence"
    )
    assert workflow is not None and workflow.request_id == "wf-reboot", (
        "no plugin-restart or reset workflow is opened beside the reboot"
    )
    assert [w.request_id for w in context.store.list_workflows()] == ["wf-reboot"], (
        "no second workflow exists after the mid-reboot XID"
    )


def test_stale_boot_kernel_log_event_opens_no_workflow() -> None:
    context = build_context()
    context.store.save_agent(_active_agent(boot_id="boot-new"))
    # The line's kernel time is from the previous boot; the Agent has since
    # heartbeated on the new boot.
    xid = event(46, event_id="xid-stale-boot").model_copy(
        update={
            "source_boot_id": "boot-old",
            "source_event_time": NOW - timedelta(minutes=10),
        }
    )

    _, incident, workflow = ingest(context, xid)

    assert workflow is None, (
        "a journal line replayed from a boot the node has left plans nothing"
    )
    assert incident.state is IncidentState.RECOVERED, (
        "the stale event is recorded and closed, not left pending"
    )
    assert any("boot" in reason for reason in incident.reasons), (
        "the incident says why: the event's boot id is not the node's current one"
    )
    assert context.store.list_workflows() == [], "no workflow row was created"


def test_current_boot_kernel_log_event_still_plans_its_reset() -> None:
    context = build_context()
    context.store.save_agent(_active_agent(boot_id="boot-new"))
    xid = event(46, event_id="xid-current-boot").model_copy(
        update={"source_boot_id": "boot-new"}
    )

    _, incident, workflow = ingest(context, xid)

    assert workflow is not None, "an event from the current boot is planned as before"
    assert incident.state is not IncidentState.RECOVERED, (
        "a live fault is not written off as stale"
    )


def test_a_finished_reboot_no_longer_absorbs_new_findings() -> None:
    context = build_context()
    _seed_reboot(context.store, tail_done=True)

    incident, workflow = context.orchestrator.ingest_node_health(
        _allocatable_finding("event-after-reboot")
    )

    assert incident.incident_id != "inc-reboot", (
        "once the validation tail succeeded the node is released; new findings are their own"
    )
    assert workflow is not None and workflow.request_id != "wf-reboot", (
        "a finding after the reboot plans its own remediation"
    )
