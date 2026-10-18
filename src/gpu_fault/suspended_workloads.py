"""The training jobs a remediation suspended and nothing restarted.

A remediation stops the job on a faulty node (STOP_WORKLOADS suspends it and
records the receipt, ``adapters/kubernetes/stop_ownership.py``) and restarts it
in its last step (RESTART_WORKLOAD). A workflow that fails or is refused after
the suspend landed never reaches that step -- a reset that failed, a node the
platform replaced or re-imaged under the STOP (the FAILED STOP then keeps its
receipt with ``workloads_suspended``, batch 139) -- and the job stays
suspended. The operator's validated restore puts the node back into
scheduling and deliberately does not restart the job (user decision G7,
batch 139): who restarts a job, and with which checkpoint, is its owner's
call. The suspended job is therefore named wherever the operator reads the end
of the remediation: the ``restore`` plan and its result
(``admin/remediation_plans.py``) and the hand-off email
(``adapters/support_escalation.py``).

``suspended_workloads`` reads the records only. Each STOP receipt names its
workloads (namespace, kind, name); a workload a later SUCCEEDED RESTART_WORKLOAD
of the same records restarted is not listed. A job its owner restarted outside
the product, or a passive restart under another incident, is not visible here,
so the line says what the records show.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from gpu_fault.models import WorkflowRequest
from gpu_fault.restart_containment import STOP_OWNERSHIP_RECEIPT_KEY

_SUCCEEDED = "SUCCEEDED"
_FAILED = "FAILED"
_STOP = "STOP_WORKLOADS"
_RESTART = "RESTART_WORKLOAD"


def _record(workflow: WorkflowRequest | Mapping[str, Any]) -> Mapping[str, Any]:
    """The JSON shape of a workflow; the admin side holds inspection dumps."""

    if isinstance(workflow, WorkflowRequest):
        return workflow.model_dump(mode="json")
    return workflow


def _executions(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in record.get("step_executions") or []
        if isinstance(item, Mapping)
    ]


def _stop_receipts(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The receipts of the workflow's STOPs whose suspend landed."""

    latest: dict[tuple[Any, Any], Mapping[str, Any]] = {}
    for execution in _executions(record):
        if execution.get("operation") == _STOP:
            latest[(execution.get("phase"), execution.get("step_index"))] = execution
    receipts = []
    for execution in latest.values():
        details = execution.get("details")
        details = details if isinstance(details, Mapping) else {}
        receipt = details.get(STOP_OWNERSHIP_RECEIPT_KEY)
        if not isinstance(receipt, Mapping):
            continue
        status = execution.get("status")
        if status == _SUCCEEDED or (
            status == _FAILED and details.get("workloads_suspended") is True
        ):
            receipts.append(receipt)
    return receipts


def _restarted(record: Mapping[str, Any]) -> set[str]:
    restarted: set[str] = set()
    for execution in _executions(record):
        if execution.get("operation") != _RESTART or execution.get("status") != (
            _SUCCEEDED
        ):
            continue
        steps = record.get(
            "safety_steps" if execution.get("phase") == "safety" else "official_steps"
        )
        index = execution.get("step_index")
        if (
            isinstance(steps, list)
            and isinstance(index, int)
            and 0 <= index < len(steps)
        ):
            step = steps[index]
            if isinstance(step, Mapping):
                restarted.update(str(item) for item in step.get("workload_ids") or [])
    return restarted


def suspended_workloads(
    workflows: Iterable[WorkflowRequest | Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Workloads suspended by ``workflows`` (newest first) and not restarted.

    A restart in a newer workflow -- or later in the same one -- answers an
    older suspend of the same workload; one in an older workflow does not
    answer a newer suspend. Each entry names the workload as its receipt does
    and the workflow whose STOP suspended it. Takes models or their JSON dumps
    (the admin side's inspection).
    """

    restarted: set[str] = set()
    found: dict[str, dict[str, str]] = {}
    for workflow in workflows:
        record = _record(workflow)
        restarted |= _restarted(record)
        for receipt in _stop_receipts(record):
            for item in receipt.get("workloads") or ():
                if not isinstance(item, Mapping):
                    continue
                workload_id = str(item.get("workload_id") or "")
                if not workload_id or workload_id in restarted or workload_id in found:
                    continue
                found[workload_id] = {
                    "workload_id": workload_id,
                    "namespace": str(item.get("namespace") or ""),
                    "kind": str(item.get("kind") or ""),
                    "name": str(item.get("name") or ""),
                    "stopped_by_workflow_id": str(record.get("request_id") or ""),
                }
    return [found[key] for key in sorted(found)]


def suspended_workload_line(item: Mapping[str, str]) -> str:
    """The one line every surface prints for a suspended workload."""

    return (
        f"workload {item['namespace']}/{item['kind']}/{item['name']} stays "
        f"suspended (stopped by workflow {item['stopped_by_workflow_id']}): the "
        "restore does not restart it; its owner must restart it"
    )
