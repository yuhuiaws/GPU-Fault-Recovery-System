"""The product's own GPU/fabric windows on a node, as the metric rules see them.

Batch 235: an NVLink5 counter increment (``gpu_nvlink5_policy``) and an NVLink
fabric health state (``gpu_fabric_health``) are graded only *outside* the
windows in which the product itself takes the node's GPUs out of the NVLink
fabric and brings them back: the quiesce (``QUIESCE_GPU_SERVICES`` stops Fabric
Manager -- on NVL5 the same unit runs NVLSM, see ``gpu_fabric_manager`` -- and
the GPU services), a GPU reset (``RESET_GPU``, the full GPU + NVSwitch reset,
the driver's drain-P2P recovery), the restore that starts Fabric Manager again,
a Fabric Manager restart alone, a reboot and a replacement. Links retrain in
those windows, so a link-recovery count, a link that left the active set or a
fabric route that is still recovering there is the window's own effect: the
window's validation step (VALIDATE_GPU) judges the GPU once it is over.

A window is read from the node's workflows: a step execution of one of
:data:`PRODUCT_LINK_WINDOW_OPERATIONS` whose node is this one and whose run
-- from ``started_at`` to ``updated_at`` plus :data:`PRODUCT_WINDOW_SETTLE`,
or still waiting -- overlaps the interval a reading covers. The settle margin
is the DCGM path's lag behind the link: the exporter refreshes its fields every
15 s and the collector scrapes every 15 s, so a count the window caused can
reach the control plane one refresh and one scrape after the step ended; twice
that is the margin. Reboots are also caught without a workflow: a counter delta
never crosses a boot (``gpu_composites.metric_counter_delta``).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStepStatus
from gpu_fault.store.shared.errors import NotFoundError

PRODUCT_LINK_WINDOW_OPERATIONS = frozenset(
    {
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RECOVER_DRAIN_P2P,
        WorkflowOperation.RESTART_FABRIC_MANAGER,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.REPLACE_NODE,
    }
)
PRODUCT_WINDOW_SETTLE = timedelta(seconds=60)
#: How many of the node's incidents (newest first) the lookup reads beside its
#: active workflows.
WINDOW_LOOKUP_INCIDENT_LIMIT = 50


def _overlapping(
    workflow: Any, node_id: str, since: datetime, until: datetime
) -> Any | None:
    steps = workflow.official_steps
    for execution in workflow.step_executions:
        if execution.operation not in PRODUCT_LINK_WINDOW_OPERATIONS:
            continue
        step = (
            steps[execution.step_index] if execution.step_index < len(steps) else None
        )
        if step is not None and step.node_ids and node_id not in step.node_ids:
            continue
        if execution.started_at > until:
            continue
        if (
            execution.status is WorkflowStepStatus.WAITING
            or execution.updated_at + PRODUCT_WINDOW_SETTLE >= since
        ):
            return execution
    return None


def product_link_window(
    store: Any, cluster_id: str, node_id: str, since: datetime, until: datetime
) -> Any | None:
    """The step execution of a product window on ``node_id`` that overlaps
    ``[since, until]`` (module docstring), or ``None``.

    Read from the node's active workflows (a window still running) and from
    the incidents updated since the interval began less the settle margin (a
    window that ended).
    """

    workflows: dict[str, Any] = {
        workflow.request_id: workflow
        for _incident, workflow in store.list_active_workflow_incidents(
            cluster_id, node_ids={node_id}
        )
    }
    for incident in store.list_incidents_by_state(
        cluster_id,
        set(IncidentState),
        node_ids={node_id},
        limit=WINDOW_LOOKUP_INCIDENT_LIMIT,
    ):
        if incident.updated_at + PRODUCT_WINDOW_SETTLE < since:
            break
        request_id = incident.workflow_request_id
        if not request_id or request_id in workflows:
            continue
        try:
            workflows[request_id] = store.get_workflow(request_id)
        except NotFoundError:
            continue
    for workflow in workflows.values():
        execution = _overlapping(workflow, node_id, since, until)
        if execution is not None:
            return execution
    return None


def describe_window(execution: Any) -> str:
    """``RESET_GPU (step 3, 12:00:05Z-12:00:40Z)`` for a finding's reason."""

    start = execution.started_at.strftime("%H:%M:%SZ")
    end = (
        "running"
        if execution.status is WorkflowStepStatus.WAITING
        else execution.updated_at.strftime("%H:%M:%SZ")
    )
    return f"{execution.operation.value} (step {execution.step_index}, {start}-{end})"
