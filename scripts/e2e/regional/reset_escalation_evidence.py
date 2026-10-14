"""What the product did when a case's single-GPU reset did not hold.

GF-REGIONAL-DESTR-001 on a plain EKS node: the batched node-action command
[QUIESCE, VERIFY, RESET, RESTORE] FAILED at RESET_GPU ("In use by another
client"), the product escalated the branch to RESTART_NODE (a BRANCH_ESCALATED
workflow event and the PLAN_REWRITE that replaced the branch), rebooted the
node, validated it and closed the workflow SUCCEEDED. The reboot also ended the
runner's owned host probe Pod and its reset sampler, so the post-reset host
snapshot raised a probe transport error and the result named only that error:
neither the failed reset, nor the escalation, nor the unrelated incident that
cordoned and quarantined the node again before the case ended.

The readers here are pure: the store snapshot ``wait_for_workflow`` returned and
the node snapshots the runner already takes. Every string they copy comes from a
named field and is bounded; no ``details`` mapping is copied wholesale (command
and step details carry fencing tokens, lease digests and signed challenges).
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.remote_command_shapes import command_operations

TEXT_LIMIT = 400
# The escalation rungs that end the node's boot. The owned probe Pod and the
# host sampler's transient systemd unit live in that boot and end with it.
NODE_LIFECYCLE_OPERATIONS = ("RESTART_NODE", "REPLACE_NODE")
ISOLATION_TAINT_PREFIX = "gpu-fault.io/"
INCIDENT_ANNOTATION = "gpu-fault.io/incident-id"


def bounded(value: object, limit: int = TEXT_LIMIT) -> str | None:
    """``value`` as one line of at most ``limit`` characters; ``None`` stays."""

    if value is None:
        return None
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _records(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _integer(value: object) -> int | None:
    return value if type(value) is int else None


def _escalation(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("details")
    details: dict[str, Any] = raw if isinstance(raw, dict) else {}
    return {
        "from_operation": bounded(
            details.get("from_operation") or event.get("operation")
        ),
        "to_operation": bounded(details.get("to_operation")),
        "node_id": bounded(details.get("node_id") or details.get("lineage_node_id")),
        "rung_count": _integer(details.get("rung_count")),
        "max_rungs": _integer(details.get("max_rungs")),
        "step_index": _integer(event.get("step_index")),
        "at": bounded(event.get("at")),
        "reason": bounded(event.get("reason")),
    }


def product_escalation(state: dict[str, Any]) -> dict[str, Any]:
    """The branch escalations, failed steps and node lifecycle rungs of the run.

    Escalations are the workflow events of kind BRANCH_ESCALATION (code
    BRANCH_ESCALATED). A lifecycle rung counts as completed when the workflow
    lists it in ``completed_operations``, a step execution of it SUCCEEDED, or a
    SUCCEEDED remote command carried it (head or batched step).
    """

    workflow = state.get("workflow") or {}
    executions = _records(workflow.get("step_executions"))
    escalations = [
        _escalation(event)
        for event in _records(workflow.get("events"))
        if event.get("kind") == "BRANCH_ESCALATION"
        or event.get("code") == "BRANCH_ESCALATED"
    ]
    failed_steps = [
        {
            "operation": bounded(item.get("operation")),
            "step_index": _integer(item.get("step_index")),
            "error": bounded(item.get("error")),
        }
        for item in executions
        if item.get("status") == "FAILED"
    ]
    completed = {str(item) for item in workflow.get("completed_operations") or []}
    completed |= {
        str(item.get("operation"))
        for item in executions
        if item.get("status") == "SUCCEEDED"
    }
    completed |= {
        operation
        for command in _records(state.get("commands"))
        if command.get("status") == "SUCCEEDED"
        for operation in command_operations(command)
    }
    return {
        "escalated": bool(escalations),
        "escalations": escalations,
        "failed_steps": failed_steps,
        "node_lifecycle_completed": [
            operation
            for operation in NODE_LIFECYCLE_OPERATIONS
            if operation in completed
        ],
        "workflow_status": bounded(workflow.get("status")),
    }


def escalation_errors(summary: dict[str, Any]) -> list[str]:
    """``RESET_GPU failed on <node>: <error>; the product escalated ...`` lines.

    One per escalation away from RESET_GPU. The error is the failed RESET_GPU
    execution's own (the one the escalation names by step index, else any),
    falling back to the escalation's reason when no failed execution remains.
    """

    failures = [
        item for item in summary["failed_steps"] if item["operation"] == "RESET_GPU"
    ]
    lines = []
    for escalation in summary["escalations"]:
        if escalation["from_operation"] != "RESET_GPU":
            continue
        matching = [
            item for item in failures if item["step_index"] == escalation["step_index"]
        ]
        failed = (matching or failures or [{}])[0]
        node = escalation["node_id"] or "the target node"
        error = str(failed.get("error") or escalation["reason"] or "no error recorded")
        error = error.removeprefix(f"node agent {node}: ")
        target = escalation["to_operation"] or "an unnamed operation"
        lines.append(
            f"RESET_GPU failed on {node}: {error}; "
            f"the product escalated the branch to {target}"
        )
    return lines


def product_reboot(
    summary: dict[str, Any], before: dict[str, Any], after: dict[str, Any]
) -> str | None:
    """Why the node's boot ended during the case, or ``None`` if it did not.

    ``before`` is the preflight node snapshot, ``after`` a later one: a completed
    lifecycle rung or a changed ``boot_id`` each say the probe Pod is gone.
    """

    reasons = []
    if summary["node_lifecycle_completed"]:
        reasons.append(" and ".join(summary["node_lifecycle_completed"]) + " completed")
    old, new = before.get("boot_id"), after.get("boot_id")
    if old and new and old != new:
        reasons.append(f"boot id changed from {bounded(old)} to {bounded(new)}")
    return "; ".join(reasons) or None


def reboot_proof_error(summary: dict[str, Any], reason: str) -> str:
    """The error that stands in for the host verdict after the node rebooted."""

    rebooted = (
        "the product rebooted the node"
        if summary["node_lifecycle_completed"]
        else "the node rebooted during the case"
    )
    return (
        f"the physical single-GPU reset proof is unavailable because {rebooted} "
        f"({reason}): the host probe Pod and the reset sampler ended with that boot"
    )


def post_case_isolation(node: dict[str, Any], incident_id: str) -> dict[str, Any]:
    """Whether the final node is still cordoned, quarantined or owned, and by whom."""

    raw = node.get("ownership_annotations")
    annotations: dict[str, Any] = raw if isinstance(raw, dict) else {}
    owner = bounded(annotations.get(INCIDENT_ANNOTATION)) or None
    return {
        "unschedulable": bool(node.get("unschedulable")),
        "taints": [
            {field: bounded(item.get(field)) for field in ("key", "value", "effect")}
            for item in _records(node.get("taints"))
            if str(item.get("key") or "").startswith(ISOLATION_TAINT_PREFIX)
        ],
        "ownership_incident": owner,
        "owned_by_case_incident": owner is not None and owner == incident_id,
        "other_incident_owner": owner
        if owner is not None and owner != incident_id
        else None,
    }


def isolation_errors(isolation: dict[str, Any]) -> list[str]:
    """One error naming the incident and taints that still hold the node."""

    owner = isolation["ownership_incident"]
    if not (isolation["unschedulable"] or isolation["taints"] or owner):
        return []
    if isolation["other_incident_owner"]:
        holder = f"owned by another incident {owner}"
    elif owner:
        holder = f"owned by this case's incident {owner}"
    else:
        holder = "no incident ownership annotation"
    taints = ", ".join(
        f"{item['key']}={item['value']}:{item['effect']}"
        if item["value"]
        else f"{item['key']}:{item['effect']}"
        for item in isolation["taints"]
    )
    parts = [
        holder,
        f"taint {taints}" if taints else "no gpu-fault.io taint",
        "unschedulable" if isolation["unschedulable"] else "schedulable",
    ]
    return ["target node is still isolated at the end of the case: " + "; ".join(parts)]
