"""Events on a node the product is rebooting belong to that reboot.

While a workflow's RESTART_NODE / REPLACE_NODE is under way -- dispatched or
finished, but the release behind it (RESTORE_SCHEDULING for that node) has not
succeeded -- the node emits signals the reboot itself explains: the kernel logs
XIDs while devices tear down, Kubernetes reports the GPUs unallocatable until
the plugin returns, the host inventory reads empty. Live 2026-09-24 those
signals opened their own incidents and RESTART_GPU_DEVICE_PLUGIN workflows,
which failed on the rebooting node; one escalated to support and quarantined
the node under the support incident (COLLECT-004, DESTR-014). Here they are
absorbed into the owning incident as evidence instead.

A kernel-log event whose ``source_boot_id`` is not the node's current boot --
the collector re-reading a previous boot's journal tail after the node came
back -- describes a boot the node has left; it is recorded and closed, never
planned. The rule is fail-closed: unknown ownership or an unknown agent keeps
today's behaviour.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    workflow_is_open,
)
from gpu_fault.policy import FaultPolicyDecision, SxidEvent, XidEvent
from gpu_fault.store import NotFoundError

LOGGER = logging.getLogger(__name__)

RebootOwner = tuple[FaultIncident, WorkflowRequest] | None

NODE_LIFECYCLE_MUTATIONS = frozenset(
    {WorkflowOperation.RESTART_NODE, WorkflowOperation.REPLACE_NODE}
)
STALE_BOOT_REASON = (
    "kernel-log event carries a boot id the node has since left; "
    "recorded as evidence, no remediation planned"
)


def reboot_in_progress(workflow: WorkflowRequest, node_id: str) -> bool:
    """Whether ``workflow`` is rebooting/replacing ``node_id`` right now.

    True from the moment a RESTART_NODE / REPLACE_NODE step for the node has
    been dispatched (it has an execution record or is completed) until every
    RESTORE_SCHEDULING planned for the node behind it has completed. Superseded
    steps never run and do not count. A closed workflow owns nothing.
    """

    if not workflow_is_open(workflow.status, workflow.blocked_kind):
        return False
    superseded = set(workflow.superseded_step_indexes)
    completed = set(workflow.completed_step_indexes)
    dispatched = completed | {
        execution.step_index for execution in workflow.step_executions
    }
    steps = workflow.official_steps
    mutations = [
        index
        for index, step in enumerate(steps)
        if step.operation in NODE_LIFECYCLE_MUTATIONS
        and node_id in step.node_ids
        and index in dispatched
        and index not in superseded
    ]
    if not mutations:
        return False
    last = max(mutations)
    releases = [
        index
        for index, step in enumerate(steps)
        if index > last
        and step.operation is WorkflowOperation.RESTORE_SCHEDULING
        and node_id in step.node_ids
        and index not in superseded
    ]
    if not releases:
        return True
    return not all(index in completed for index in releases)


def owning_reboot(store: Any, cluster_id: str, node_id: str) -> RebootOwner:
    """The open incident/workflow pair rebooting ``node_id``, if any."""

    for incident, workflow in store.list_active_workflow_incidents(
        cluster_id, node_ids={node_id}
    ):
        if node_id in incident.node_ids and reboot_in_progress(workflow, node_id):
            return incident, workflow
    return None


def absorb_finding(
    store: Any,
    finding: NodeHealthFinding,
    owner: tuple[FaultIncident, WorkflowRequest],
    *,
    persist: bool = True,
) -> tuple[FaultIncident, WorkflowRequest]:
    incident, workflow = owner
    if persist:
        store.link_event_to_incident(finding.event_id, incident.incident_id)
    LOGGER.info(
        "absorbed node-health finding %s (%s) into workflow %s: node %s is being "
        "rebooted by the product",
        finding.event_id,
        finding.metric_name or finding.category.value,
        workflow.request_id,
        finding.node_id,
    )
    return incident, workflow


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def stale_boot(store: Any, event: XidEvent | SxidEvent) -> bool:
    """Whether ``event`` was written by a boot the node's Agent has since left.

    The Agent record carries the node's current boot id and the time of its
    last heartbeat. An event from another boot whose kernel time predates that
    heartbeat is a journal line of a previous boot: the node had already come
    back on its current boot when the line was reported. An event from a boot
    the Agent has not reported yet (a fresh boot before re-registration) has a
    kernel time after the last heartbeat and is kept.
    """

    if not event.source_boot_id:
        return False
    try:
        agent = store.get_agent(event.cluster_id, event.node_id)
    except NotFoundError:
        return False
    if (
        not agent.boot_id
        or agent.boot_id == event.source_boot_id
        or agent.lifecycle_state is not AgentLifecycleState.ACTIVE
    ):
        return False
    event_time = _utc(event.source_event_time or event.observed_at)
    return event_time < _utc(agent.last_seen_at)


def cover_fault(
    store: Any,
    event: XidEvent | SxidEvent,
    decision: FaultPolicyDecision,
    build_incident: Callable[..., FaultIncident],
) -> tuple[FaultIncident, WorkflowRequest | None] | None:
    """Absorb a fault on a rebooting node, or close a stale-boot fault.

    Returns the owning (incident, workflow) when the node is under the
    product's own reboot; a RECOVERED incident without a workflow when the
    event belongs to a boot the node has left; ``None`` otherwise.
    """

    owner = owning_reboot(store, event.cluster_id, event.node_id)
    if owner is not None:
        incident, workflow = owner
        store.link_event_to_incident(event.event_id, incident.incident_id)
        LOGGER.info(
            "absorbed fault event %s into workflow %s: node %s is being rebooted "
            "by the product",
            event.event_id,
            workflow.request_id,
            event.node_id,
        )
        return incident, workflow
    if not stale_boot(store, event):
        return None
    incident = build_incident(event, decision, extra_reasons=[STALE_BOOT_REASON])
    incident = incident.model_copy(
        update={
            "state": IncidentState.RECOVERED,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    store.save_incident(incident)
    LOGGER.warning(
        "fault event %s on %s carries boot id %s but the node runs boot %s; "
        "recorded as RECOVERED evidence, nothing planned",
        event.event_id,
        event.node_id,
        event.source_boot_id,
        store.get_agent(event.cluster_id, event.node_id).boot_id,
    )
    return incident, None
