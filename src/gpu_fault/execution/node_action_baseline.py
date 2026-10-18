"""The boot a node ran when one of this workflow's node actions started.

``confirm-node-action`` settles a node action whose answer was lost by a boot
transition -- for a QUIESCE or a RESET, "a reboot since the action"
(``node_action_confirmation``) -- and needs the boot the action ran on. The
incident's fault-time ``source_boot_id`` is the wrong baseline once a workflow
stopped workloads: a re-image or a reboot between the XID and the STOP made a
QUIESCE that ran on the boot the node still runs read as rebooted since, and a
multi-node incident carries no source boot to read at all (batch 138).

The workflow's own contained STOP receipt binds every participant's UID and
boot (``adapters/kubernetes/stop_ownership.py``), and every ownership-guarded
node action after it was checked against that boot -- or against the boot this
workflow's own confirmed reboot chain moved it to, the rule the guard applies
(``stop_boot_transition.authorized_boot_transition``). ``stop_receipt_boot``
reads it:

* ``(boot, source)`` -- the receipt's boot for the node, carried forward along
  the SUCCEEDED, externally confirmed ``RESTART_NODE`` records of this
  workflow on the node that ended before the action started (each link's
  ``agent_baselines`` boot must be the boot the chain reached, its
  ``agent_observations`` boot the next one);
* ``UNPROVABLE`` -- the receipt names the node but the chain cannot be
  followed (a reboot of the node before the action that is not confirmed, or a
  link that does not chain): the caller refuses rather than falling back to the
  incident's older boot;
* ``None`` -- the workflow holds no receipt of its own that names the node
  (no STOP, an inherited or uncontained receipt, a STOP after the action), so
  the caller's other sources apply as before.

Pure over the records; nothing here reads a source.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Final

from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepExecution,
    WorkflowStepStatus,
    execution_phase,
)
from gpu_fault.restart_containment import STOP_OWNERSHIP_RECEIPT_KEY

#: ``previous_boot_id_source`` of a baseline the STOP receipt gave, with or
#: without confirmed reboots carrying it forward.
STOP_RECEIPT_SOURCE = f"stop.{STOP_OWNERSHIP_RECEIPT_KEY}"
STOP_RECEIPT_CHAIN_SOURCE = f"{STOP_RECEIPT_SOURCE}+confirmed_reboots"


class _Unprovable:
    def __repr__(self) -> str:
        return "UNPROVABLE"


UNPROVABLE: Final = _Unprovable()


def _instant(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed: datetime | None = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed is not None and parsed.tzinfo is not None else None


def _receipt(
    workflow: WorkflowRequest, incident: FaultIncident
) -> tuple[Mapping[str, Any], datetime] | None:
    """The workflow's own contained STOP receipt and when it completed."""

    phase = execution_phase(workflow)
    steps = workflow.safety_steps if phase == "safety" else workflow.official_steps
    stops = [
        index
        for index, step in enumerate(steps)
        if step.operation is WorkflowOperation.STOP_WORKLOADS
    ]
    if len(stops) != 1 or stops[0] not in workflow.completed_step_indexes:
        return None
    execution = next(
        (
            item
            for item in reversed(workflow.step_executions)
            if item.step_index == stops[0]
            and item.operation is WorkflowOperation.STOP_WORKLOADS
            and item.phase in (None, phase)
        ),
        None,
    )
    if execution is None or execution.status is not WorkflowStepStatus.SUCCEEDED:
        return None
    receipt = execution.details.get(STOP_OWNERSHIP_RECEIPT_KEY)
    if not isinstance(receipt, Mapping):
        return None
    completed = _instant(receipt.get("completed_at"))
    if (
        completed is None
        or receipt.get("contained") is not True
        or receipt.get("workflow_id") != workflow.request_id
        or receipt.get("incident_id") != incident.incident_id
        or receipt.get("fencing_token") != workflow.fencing_token
    ):
        return None
    return receipt, completed


def _receipt_boot(receipt: Mapping[str, Any], node_id: str) -> str | None:
    nodes = receipt.get("nodes")
    for node in nodes if isinstance(nodes, list) else []:
        if isinstance(node, Mapping) and node.get("name") == node_id:
            boot = node.get("boot_id")
            return boot if isinstance(boot, str) and boot else None
    return None


def _observed_boot(details: Mapping[str, Any], node_id: str) -> str | None:
    observations = details.get("agent_observations")
    for item in observations if isinstance(observations, list) else []:
        if isinstance(item, Mapping) and item.get("node_id") == node_id:
            boot = item.get("boot_id")
            return boot if isinstance(boot, str) and boot else None
    return None


def _baseline_boot(details: Mapping[str, Any], node_id: str) -> str | None:
    baselines = details.get("agent_baselines")
    baseline = baselines.get(node_id) if isinstance(baselines, Mapping) else None
    boot = baseline.get("boot_id") if isinstance(baseline, Mapping) else None
    return boot if isinstance(boot, str) and boot else None


def _never_submitted(execution: WorkflowStepExecution) -> bool:
    details = execution.details
    return (
        execution.status is WorkflowStepStatus.FAILED
        and details.get("node_action_not_started") is True
        and not details.get("requires_external_confirmation")
        and not details.get("submitted_nodes")
        and not details.get("provider_operation_id")
    )


def stop_receipt_boot(
    workflow: WorkflowRequest,
    incident: FaultIncident,
    node_id: str,
    execution: WorkflowStepExecution,
) -> tuple[str, str] | _Unprovable | None:
    """The boot ``node_id`` ran when ``execution`` started (module docstring)."""

    bound = _receipt(workflow, incident)
    if bound is None:
        return None
    receipt, completed = bound
    boot = _receipt_boot(receipt, node_id)
    if boot is None or not completed <= execution.started_at:
        return None
    steps = workflow.official_steps
    latest: dict[int, WorkflowStepExecution] = {}
    for item in workflow.step_executions:
        if (
            item.operation is WorkflowOperation.RESTART_NODE
            and item.phase in (None, "official")
            and 0 <= item.step_index < len(steps)
            and steps[item.step_index].operation is WorkflowOperation.RESTART_NODE
            and node_id in steps[item.step_index].node_ids
            and not (
                item.step_index == execution.step_index
                and item.operation is execution.operation
            )
        ):
            latest[item.step_index] = item
    source = STOP_RECEIPT_SOURCE
    for reboot in sorted(latest.values(), key=lambda item: item.updated_at):
        if reboot.started_at >= execution.started_at or _never_submitted(reboot):
            continue
        details = reboot.details
        new_boot = _observed_boot(details, node_id)
        if (
            reboot.status is not WorkflowStepStatus.SUCCEEDED
            or reboot.step_index not in workflow.completed_step_indexes
            or details.get("externally_confirmed") is not True
            or reboot.updated_at > execution.started_at
            or _baseline_boot(details, node_id) != boot
            or new_boot is None
            or new_boot == boot
        ):
            # A reboot of the node before the action that this record cannot
            # follow: the boot the action ran on is not known.
            return UNPROVABLE
        boot = new_boot
        source = STOP_RECEIPT_CHAIN_SOURCE
    return boot, source
