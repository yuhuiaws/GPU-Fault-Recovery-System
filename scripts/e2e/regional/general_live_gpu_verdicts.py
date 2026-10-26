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
#: The dispatcher's wording on a record it closed because compilation blocked
#: it before any step ran (``compile_blocked.CLOSE_MARKER``); such a record
#: planned its steps and executed none of them.
COMPILE_TIME_BLOCKED = "compile-time BLOCKED"


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
    #: Operations that may be planned but must not have executed on this site
    #: (a capability the Runtime Profile holds at OBSERVE: 019 remap_failure's
    #: RUN_FIELD_DIAGNOSTIC, memoryDiagnostics with no executable owner).
    unexecuted: frozenset[str] = frozenset()
    notes: list[str] = field(default_factory=list)


def planned_operations(workflow: dict[str, Any]) -> list[str]:
    """Every step the record plans, safety steps first: a plan the product runs
    as a SAFETY plan (b192: the containment prefix of a DRAIN whose diagnostic
    has no executable owner) lists them under ``safety_steps``, the rest under
    ``official_steps``; a chain is judged on both."""

    return [
        str(step.get("operation"))
        for step in [
            *(workflow.get("safety_steps") or []),
            *(workflow.get("official_steps") or []),
        ]
    ]


def isolating_steps(chain: GpuChain) -> list[str]:
    """The containment steps the chain expects (``ISOLATING_OPERATIONS``), in order."""

    return [item for item in chain.operations if item in ISOLATING_OPERATIONS]


def never_executed_errors(workflow: dict[str, Any], chain: GpuChain) -> list[str]:
    """Why ``workflow`` planned the chain's containment but did not do it.

    ``chain_errors`` used to read the PLANNED steps (``official_steps``) only,
    so a record that compilation blocked before any step ran passed the chain
    stage: live 2026-10-26 (gpu-eks-cluster, 019 remap_failure) the DRAIN record
    was BLOCKED/NEEDS_OPERATOR at compile time (``no executable owner for
    memoryDiagnostics``), the dispatcher sweep superseded it 35 ms later, no
    command ever reached the node, and the stage read PASS on a node nothing
    had touched. Three readings of the product's own record catch that:

    * a ``preemption_reason`` naming the compile-time close
      (``COMPILE_TIME_BLOCKED``), or ``blocked_reasons`` on a record with no
      step execution at all, is a chain that was planned and never run;
    * every isolating step of ``chain.operations`` needs a SUCCEEDED final
      execution once the record has ended (``TERMINAL_WORKFLOWS``): a missing
      one is "planned but never executed", a FAILED one is its own error;
    * a record that may still execute (PENDING/RUNNING) is judged on what it
      has executed so far -- a FAILED containment is an error, a step still
      ahead is not; a caller that needs the executions in hand waits for
      ``chain_executed`` first (019 does).

    A chain without isolating steps (the diagnostic-only shapes) is unchanged:
    its validation step is already judged on its execution.
    """

    label = chain.label
    reasons = [str(item) for item in workflow.get("blocked_reasons") or []]
    preempted = COMPILE_TIME_BLOCKED in str(workflow.get("preemption_reason") or "")
    executed_any = bool(workflow.get("step_executions"))
    if preempted or (reasons and not executed_any):
        return [
            f"{label} chain planned but never executed: {COMPILE_TIME_BLOCKED} "
            f"({', '.join(reasons) or 'no blocked reason recorded'})"
        ]
    ended = str(workflow.get("status")) in TERMINAL_WORKFLOWS
    errors = []
    for operation in isolating_steps(chain):
        execution = final_execution(workflow, operation)
        if execution is None:
            if ended:
                errors.append(
                    f"{label} chain planned but never executed: {operation} has no "
                    "execution"
                )
        elif execution.get("status") != "SUCCEEDED":
            errors.append(
                f"{label} {operation} ended {execution.get('status')}: "
                f"{str(execution.get('error') or '')[:200]}"
            )
    return errors


def chain_executed(
    activity: dict[str, Any], incidents: Sequence[dict[str, Any]], chain: GpuChain
) -> bool:
    """Whether the chain's containment can be judged on executions.

    True once the first workflow of ``incidents`` has a terminal execution of
    every isolating step the chain expects, or has ended (``settled``: nothing
    more will execute, and a record that ended without them is the hollow chain
    ``chain_errors`` names). A detection wait that gates on this judges steps
    the product did, not steps it planned.
    """

    if not incidents:
        return False
    workflows = workflows_of(activity, str(incidents[0]["incident_id"]))
    if not workflows:
        return False
    if settled(activity, incidents):
        return True
    workflow = workflows[0]
    return all(
        (final_execution(workflow, operation) or {}).get("status")
        in {"SUCCEEDED", "FAILED"}
        for operation in isolating_steps(chain)
    )


def chain_errors(
    activity: dict[str, Any], incidents: Sequence[dict[str, Any]], chain: GpuChain
) -> list[str]:
    """finding -> incident -> workflow -> step trace of one GPU expectation.

    Planned steps, forbidden steps, the validation's execution, the incident
    state -- and, for a chain with isolating steps, that the first workflow
    executed them (``never_executed_errors``).
    """

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
    steps = planned_operations(workflow)
    if chain.operations and not subsequence(chain.operations, steps):
        errors.append(f"{label} workflow steps {steps} lack {list(chain.operations)}")
    errors.extend(never_executed_errors(workflow, chain))
    for item in workflows:
        found = sorted(set(planned_operations(item)) & set(chain.forbidden))
        if found:
            errors.append(f"{label} workflow {item.get('request_id')} plans {found}")
        for operation in sorted(chain.unexecuted):
            execution = final_execution(item, operation)
            if execution is not None:
                errors.append(
                    f"{label} workflow {item.get('request_id')} executed {operation} "
                    f"({execution.get('status')}); it must not run on this site"
                )
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
            steps = set(planned_operations(workflow))
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
        if (
            found := (set(planned_operations(workflow)) & HARDWARE_OPERATIONS)
            - permitted
        )
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
        planned_operations(workflow).count(operation)
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


def counter_reset_ticks(
    state: dict[str, Any], canonical: str, gpu: str
) -> list[dict[str, Any]]:
    """The accepted batches in which ``canonical`` on ``gpu`` fell below its
    previous accepted value: the product's own record of a counter reset.

    Each tick carries the drop (``previous`` -> ``value``), the batch's
    ``observed_at`` / ``batch_id`` and its ``edge_filter_reasons`` -- the
    collector delivers such a tick as a ``counter-reset`` edge.
    """

    ticks: list[dict[str, Any]] = []
    previous: float | None = None
    for batch in state.get("batches") or []:
        for sample in batch.get("samples") or []:
            if sample.get("name") != canonical or sample.get("gpu") != gpu:
                continue
            value = sample.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if previous is not None and value < previous:
                ticks.append(
                    {
                        "value": value,
                        "previous": previous,
                        "observed_at": batch.get("observed_at"),
                        "batch_id": batch.get("batch_id"),
                        "edge_filter_reasons": list(
                            batch.get("edge_filter_reasons") or []
                        ),
                    }
                )
            previous = value
    return ticks


def incident_finding(
    state: dict[str, Any], incident: dict[str, Any]
) -> dict[str, Any] | None:
    """The finding (``gpu_state`` history) a ``gpu-<finding_id>`` NODE_HEALTH
    incident was minted from, when the history since the moment holds it."""

    event_id = str(incident.get("event_id") or "")
    finding_id = event_id[4:] if event_id.startswith("gpu-") else event_id
    for item in state.get("findings") or []:
        if item.get("finding_id") == finding_id:
            return item
    return None


def finding_text(finding: dict[str, Any] | None) -> str:
    """``value=.. delta=.. rate=../min at <observed_at>`` of a finding view."""

    if finding is None:
        return "finding not in the product's history since the replay was armed"
    return (
        f"value={finding.get('value')} delta={finding.get('delta')} "
        f"rate={finding.get('rate_per_minute')}/min at {finding.get('observed_at')}"
    )


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


# --------------------------------------------------------------------------- #
# Composite findings (``gpu_state`` finding history) and their de-duplication
# --------------------------------------------------------------------------- #
def composite_findings(
    state: dict[str, Any], rule_id: str, *, gpu: str | None = None
) -> list[dict[str, Any]]:
    """The product's composite findings of ``rule_id`` since the read (history)."""

    return sorted(
        (
            item
            for item in [
                *(state.get("findings") or []),
                *(state.get("active_findings") or []),
            ]
            if item.get("correlation_rule_id") == rule_id
            and (gpu is None or item.get("gpu_uuid") in (None, gpu))
        ),
        key=lambda item: str(item.get("observed_at")),
    )


def composite_grade_errors(
    findings: Sequence[dict[str, Any]],
    *,
    rule_id: str,
    severity: str,
    action: str,
    required_components: Iterable[str] = (),
) -> list[str]:
    """A composite stayed at its diagnostic grade and named its components.

    ``severity``/``action`` are what every finding of the rule must carry
    (a THERMAL_STRESS that escalated to CRITICAL/DRAIN fails here);
    ``required_components`` must all appear in ``component_metrics``.
    """

    if not findings:
        return [f"no {rule_id} composite finding was recorded"]
    errors = []
    for finding in findings:
        if str(finding.get("severity")) != severity:
            errors.append(
                f"{rule_id} finding {finding.get('finding_id')} is "
                f"{finding.get('severity')}, expected {severity}"
            )
        if str(finding.get("automatic_action")) != action:
            errors.append(
                f"{rule_id} finding {finding.get('finding_id')} plans "
                f"{finding.get('automatic_action')}, expected {action}"
            )
    components = {
        str(name)
        for finding in findings
        for name in finding.get("component_metrics") or []
    }
    missing = sorted(set(required_components) - components)
    if missing:
        errors.append(f"{rule_id} components {components or '{}'} lack {missing}")
    return errors


def composite_component_errors(
    findings: Sequence[dict[str, Any]],
    *,
    rule_id: str,
    allowed_components: Iterable[str],
) -> list[str]:
    """A composite is built only from its own metric components.

    A kernel XID, another GPU's finding or a finding of another family inside
    ``component_metrics``/``component_finding_ids`` means the composite covered
    an event that was not its own.
    """

    allowed = set(allowed_components)
    errors = []
    for finding in findings:
        foreign = sorted(
            str(name)
            for name in finding.get("component_metrics") or []
            if str(name) not in allowed
        )
        if foreign:
            errors.append(
                f"{rule_id} finding {finding.get('finding_id')} composed foreign "
                f"components {foreign}"
            )
        xid_components = sorted(
            str(item)
            for item in finding.get("component_finding_ids") or []
            if "xid" in str(item).lower()
        )
        if xid_components:
            errors.append(
                f"{rule_id} finding {finding.get('finding_id')} absorbed XID "
                f"components {xid_components}"
            )
    return errors


def component_dedup_errors(
    activity: dict[str, Any],
    composites: Sequence[dict[str, Any]],
    *,
    components: Iterable[str],
    gpu: str,
) -> list[str]:
    """The composite's components were suppressed into it, not minted beside it.

    ``gpu_metrics.ingest`` suppresses a component finding's own incident once
    its composite exists (``suppressed_by_composite``); a component that still
    minted a NODE_HEALTH incident after the composite's creation was handled
    twice. A component incident minted *before* the composite formed is the
    product's design (the composite needed a later batch) and is not an error.
    """

    if not composites:
        return [f"no composite incident to de-duplicate against on {gpu}"]
    formed = min(str(item.get("created_at")) for item in composites)
    errors = []
    for name in components:
        late = [
            item
            for item in metric_incidents(activity, name, gpu=gpu)
            if str(item.get("created_at")) > formed
        ]
        if late:
            errors.append(
                f"{name} on {gpu} minted {len(late)} incident(s) after the composite "
                "formed (" + ", ".join(str(item["incident_id"]) for item in late) + ")"
            )
    return errors


__all__ = [
    "COMPILE_TIME_BLOCKED",
    "COMPLETENESS_FINDING",
    "FABRIC_FINDING",
    "GpuChain",
    "ISOLATING_OPERATIONS",
    "RESET_OPERATIONS",
    "accepted_samples",
    "active_findings",
    "chain_errors",
    "chain_executed",
    "collection_errors",
    "completeness_incidents",
    "component_dedup_errors",
    "composite_component_errors",
    "composite_findings",
    "composite_grade_errors",
    "composite_incidents",
    "counter_reset_ticks",
    "executions_of",
    "fabric_incidents",
    "finding_text",
    "history_findings",
    "incident_finding",
    "inventory_incidents",
    "isolating_steps",
    "latest_value",
    "metric_incidents",
    "never_executed_errors",
    "node_hardware_errors",
    "operation_count",
    "planned_operations",
    "read_only_errors",
    "replay_effective",
    "served_values",
    "settled",
    "step_started",
]
