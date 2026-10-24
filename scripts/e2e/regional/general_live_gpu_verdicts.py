"""Chain checks of the GF-GENERAL-LIVE DCGM / GPU-inventory categories.

A DCGM finding reaches the control plane's node-health path as
``gpu-<finding_id>`` (``app/ingest/gpu_findings.py``), so its NODE_HEALTH
incident is ``inc-gpu-<finding_id>``. The finding id names the batch, the
scope and the reading (``gpu_metrics.py`` / ``gpu_composites.py`` /
``gpu_fabric_manager.py`` / ``gpu_collection_errors.py``):

* a metric finding:     ``<batch>-<gpu_key>-<canonical>``
* a composite:          ``<batch>-<scope>-composite-<RULE_ID>``
* the fabric state:     ``<batch>-node-fabric_manager_status``
* field completeness:   ``<batch>-dcgm-fields`` (the streak's first batch)

The functions here select those incidents from the documents
``general_live_store.node_activity`` and ``general_live_gpu_store.gpu_state``
return and answer with human-readable errors (empty means the check holds),
like ``general_live_verdicts``. They read only the product's own records and
the replay's audit receipts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from scripts.e2e.regional.general_live_registry import HARDWARE_OPERATIONS
from scripts.e2e.regional.general_live_verdicts import (
    TERMINAL_WORKFLOWS,
    final_execution,
    operations,
    subsequence,
    workflows_of,
)

FABRIC_FINDING = "fabric_manager_status"
COMPLETENESS_FINDING = "dcgm_field_completeness"
#: What a read-only GPU diagnostic chain may contain; anything isolating or
#: mutating is outside it.
ISOLATING_OPERATIONS = frozenset(
    {"MARK_UNSCHEDULABLE", "QUARANTINE", "STOP_WORKLOADS", "CHECKPOINT_WORKLOADS"}
)
RESET_OPERATIONS = frozenset({"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES"})


def _gpu_incidents(activity: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            item
            for item in activity.get("incidents") or []
            if item.get("event_type") == "NODE_HEALTH"
            and str(item.get("event_id") or "").startswith("gpu-")
        ),
        key=lambda item: str(item.get("created_at")),
    )


def metric_incidents(
    activity: dict[str, Any], canonical: str, *, gpu: str | None = None
) -> list[dict[str, Any]]:
    """NODE_HEALTH incidents of a DCGM metric finding on ``canonical``."""

    suffix = f"-{gpu}-{canonical}" if gpu else f"-{canonical}"
    return [
        item
        for item in _gpu_incidents(activity)
        if str(item.get("event_id")).endswith(suffix)
        and "-composite-" not in str(item.get("event_id"))
    ]


def composite_incidents(
    activity: dict[str, Any], rule_id: str, *, scope: str | None = None
) -> list[dict[str, Any]]:
    suffix = f"-{scope}-composite-{rule_id}" if scope else f"-composite-{rule_id}"
    return [
        item
        for item in _gpu_incidents(activity)
        if str(item.get("event_id")).endswith(suffix)
    ]


def fabric_incidents(activity: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in _gpu_incidents(activity)
        if str(item.get("event_id")).endswith(f"-node-{FABRIC_FINDING}")
    ]


def completeness_incidents(activity: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in _gpu_incidents(activity)
        if str(item.get("event_id")).endswith("-dcgm-fields")
    ]


def inventory_incidents(activity: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    """NODE_HEALTH incidents of an inventory finding (``gpu_inventory_*``)."""

    return [
        item
        for item in activity.get("incidents") or []
        if item.get("event_type") == "NODE_HEALTH"
        and metric in str(item.get("event_id") or "")
    ]


@dataclass
class GpuChain:
    """What the product must have done with one set of incidents."""

    label: str
    present: bool = True
    single: bool = True
    #: Operations that must appear, in this order, in the first workflow.
    operations: tuple[str, ...] = ()
    #: A validation step that must have run, and whether it must have failed.
    validation: str | None = None
    validation_must_fail: bool = False
    incident_states: frozenset[str] = frozenset()
    #: Operations no workflow of these incidents may plan.
    forbidden: frozenset[str] = HARDWARE_OPERATIONS
    #: Effective actions the incident may carry (``effective_action``).
    actions: frozenset[str] = frozenset()
    notes: list[str] = field(default_factory=list)


def chain_errors(
    activity: dict[str, Any], incidents: Sequence[dict[str, Any]], chain: GpuChain
) -> list[str]:
    """finding -> incident -> workflow -> step trace of one GPU expectation."""

    label = chain.label
    if not chain.present:
        return [
            f"unexpected {label} incident {item['incident_id']}" for item in incidents
        ]
    if not incidents:
        return [f"no {label} finding reached an incident"]
    errors = []
    if chain.single and len(incidents) > 1:
        errors.append(
            f"{len(incidents)} {label} incidents (expected one): "
            + ", ".join(str(item["incident_id"]) for item in incidents)
        )
    incident = incidents[0]
    if chain.actions and str(incident.get("effective_action")) not in chain.actions:
        errors.append(
            f"{label} incident action {incident.get('effective_action')}, expected "
            f"one of {sorted(chain.actions)}"
        )
    workflows = [
        workflow
        for item in incidents
        for workflow in workflows_of(activity, str(item["incident_id"]))
    ]
    if not workflows:
        return [*errors, f"{label} incident {incident['incident_id']} has no workflow"]
    first = workflows_of(activity, str(incident["incident_id"]))
    workflow = first[0] if first else workflows[0]
    steps = operations(workflow)
    if chain.operations and not subsequence(chain.operations, steps):
        errors.append(f"{label} workflow steps {steps} lack {list(chain.operations)}")
    for item in workflows:
        found = sorted(set(operations(item)) & set(chain.forbidden))
        if found:
            errors.append(f"{label} workflow {item.get('request_id')} plans {found}")
    if chain.validation is not None:
        execution = final_execution(workflow, chain.validation)
        if execution is None:
            errors.append(f"{label} workflow never executed {chain.validation}")
        elif chain.validation_must_fail:
            if execution.get("status") == "SUCCEEDED":
                errors.append(
                    f"{chain.validation} succeeded while the replayed reading was held"
                )
        elif execution.get("status") != "SUCCEEDED":
            errors.append(
                f"{chain.validation} ended {execution.get('status')}: "
                f"{str(execution.get('error') or '')[:200]}"
            )
    if chain.incident_states and incident.get("state") not in chain.incident_states:
        errors.append(
            f"{label} incident is {incident.get('state')}, expected one of "
            f"{sorted(chain.incident_states)}"
        )
    return errors


def read_only_errors(
    activity: dict[str, Any], incidents: Sequence[dict[str, Any]], label: str
) -> list[str]:
    """A diagnostic-only chain isolates nothing and mutates no hardware."""

    errors = []
    for incident in incidents:
        for workflow in workflows_of(activity, str(incident["incident_id"])):
            steps = set(operations(workflow))
            isolating = sorted(steps & ISOLATING_OPERATIONS)
            hardware = sorted(steps & HARDWARE_OPERATIONS)
            if isolating:
                errors.append(
                    f"{label} workflow {workflow.get('request_id')} isolates: {isolating}"
                )
            if hardware:
                errors.append(
                    f"{label} workflow {workflow.get('request_id')} plans {hardware}"
                )
    return errors


def node_hardware_errors(
    activity: dict[str, Any], *, allowed: Iterable[str] = ()
) -> list[str]:
    """No workflow anywhere on the node planned an unallowed hardware step."""

    permitted = set(allowed)
    return [
        f"workflow {workflow.get('request_id')} plans {sorted(found)}"
        for workflow in activity.get("workflows") or []
        if (found := (set(operations(workflow)) & HARDWARE_OPERATIONS) - permitted)
    ]


def executions_of(
    activity: dict[str, Any], operation: str, incidents: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for incident in incidents:
        for workflow in workflows_of(activity, str(incident["incident_id"])):
            result.extend(
                item
                for item in workflow.get("step_executions") or []
                if item.get("operation") == operation
            )
    return result


def operation_count(
    activity: dict[str, Any], operation: str, incidents: Sequence[dict[str, Any]]
) -> int:
    return sum(
        operations(workflow).count(operation)
        for incident in incidents
        for workflow in workflows_of(activity, str(incident["incident_id"]))
    )


def settled(activity: dict[str, Any], incidents: Sequence[dict[str, Any]]) -> bool:
    workflows = [
        workflow
        for incident in incidents
        for workflow in workflows_of(activity, str(incident["incident_id"]))
    ]
    return bool(workflows) and all(
        str(item.get("status")) in TERMINAL_WORKFLOWS for item in workflows
    )


def step_started(
    activity: dict[str, Any], operation: str, incidents: Sequence[dict[str, Any]]
) -> bool:
    """Whether a workflow of ``incidents`` has begun executing ``operation``."""

    return bool(executions_of(activity, operation, incidents))


# --------------------------------------------------------------------------- #
# Replay audit and product readings
# --------------------------------------------------------------------------- #
def served_values(records: Sequence[dict[str, Any]], key: str) -> list[Any]:
    """What the replay served for ``<gpu>/<field>`` (``None``: unchanged)."""

    return [
        (record.get("served") or {}).get(key)
        for record in records
        if record.get("read") is not None and record.get("armed")
    ]


def replay_effective(records: Sequence[dict[str, Any]], key: str) -> list[str]:
    """The oracle: the replay really served ``key`` and never aborted."""

    errors = []
    aborted = sorted(
        {str(record["aborted"]) for record in records if record.get("aborted")}
    )
    if aborted:
        errors.append("the replay aborted: " + "; ".join(aborted))
    if not any(value is not None for value in served_values(records, key)):
        errors.append(f"the replay never served {key}")
    return errors


def latest_value(
    state: dict[str, Any], canonical: str, gpu: str
) -> dict[str, Any] | None:
    for item in state.get("latest") or []:
        if item.get("name") == canonical and item.get("gpu") == gpu:
            return item
    return None


def accepted_samples(
    state: dict[str, Any], canonical: str, gpu: str
) -> list[dict[str, Any]]:
    """The product's accepted batch samples of ``canonical`` on ``gpu``."""

    result = []
    for batch in state.get("batches") or []:
        for sample in batch.get("samples") or []:
            if sample.get("name") == canonical and sample.get("gpu") == gpu:
                result.append({**sample, "observed_at": batch.get("observed_at")})
    return sorted(result, key=lambda item: str(item.get("observed_at")))


def active_findings(
    state: dict[str, Any], canonical: str, *, gpu: str | None = None
) -> list[dict[str, Any]]:
    return [
        item
        for item in state.get("active_findings") or []
        if item.get("canonical_name") == canonical
        and (gpu is None or item.get("gpu_uuid") == gpu)
    ]


def history_findings(
    state: dict[str, Any], canonical: str, *, gpu: str | None = None
) -> list[dict[str, Any]]:
    return [
        item
        for item in state.get("findings") or []
        if item.get("canonical_name") == canonical
        and (gpu is None or item.get("gpu_uuid") == gpu)
    ]


def collection_errors(state: dict[str, Any]) -> list[str]:
    return [
        str(error)
        for batch in state.get("batches") or []
        for error in batch.get("collection_errors") or []
    ]


__all__ = [
    "COMPLETENESS_FINDING",
    "FABRIC_FINDING",
    "GpuChain",
    "ISOLATING_OPERATIONS",
    "RESET_OPERATIONS",
    "accepted_samples",
    "active_findings",
    "chain_errors",
    "collection_errors",
    "completeness_incidents",
    "composite_incidents",
    "executions_of",
    "fabric_incidents",
    "history_findings",
    "inventory_incidents",
    "latest_value",
    "metric_incidents",
    "node_hardware_errors",
    "operation_count",
    "read_only_errors",
    "replay_effective",
    "served_values",
    "settled",
    "step_started",
]
