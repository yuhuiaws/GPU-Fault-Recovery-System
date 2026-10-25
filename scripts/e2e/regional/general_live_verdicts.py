"""Chain checks shared by the GF-GENERAL-LIVE category modules.

Every function reads documents ``general_live_store.node_activity`` returned --
the product's accepted batches, incidents, markers, workflows with their step
executions, health signals and notifications -- or the runner's own oracle
records, and answers with a list of human-readable errors (empty means the
check holds). The verdict of a variant is ``PASS`` only when every stage list
is empty; ``INCONCLUSIVE`` is reserved for an injection the oracle did not see.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from scripts.e2e.regional.general_live_registry import HARDWARE_OPERATIONS

TERMINAL_WORKFLOWS = frozenset({"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"})
VALIDATIONS = frozenset({"VALIDATE_GPU", "VALIDATE_HOST", "VALIDATE_FABRIC"})
HOST_DIAGNOSTIC_FAILURE = "host_diagnostic_failure"


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def safe_device(value: str | None) -> str:
    """``NodeHealthPolicy._safe`` -- the device segment of a finding's event id."""

    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value or "node")[:160]


#: Findings whose event id does not name their metric, as (prefix, ending).
#: The DCGM field-completeness finding reaches the node-health path as
#: ``gpu-<finding id>`` (``app/ingest/gpu_findings.py``) and its finding id is
#: the error streak's first batch plus ``-dcgm-fields``
#: (``gpu_collection_errors.py``): ``inc-gpu-dcgm-<node>-<us>-dcgm-fields``,
#: never ``-dcgm_field_completeness-``. It is node-level, with no rule.
UNNAMED_FINDINGS = {"dcgm_field_completeness": ("gpu-", "-dcgm-fields")}

#: The event types an incident minted by a node-health finding carries. The
#: health family mints ``NODE_HEALTH`` (``families/health.py``), and the
#: product re-labels that same record -- same ``inc-<event id>``, same minting
#: ``event_id`` -- when it routes more evidence into it instead of opening a
#: second incident: a critical QUARANTINE that joins the node's in-flight
#: isolation makes the incumbent ``NODE_HEALTH_GROUP``
#: (``families/drain.py`` ``ingest_terminal_node_quarantine``), a device-resource
#: finding ``NODE_RESOURCE_GROUP`` (``ingest_grouped_node_resource_finding``),
#: and a finding on a node a managed attempt holds is opened as
#: ``GPU_FAULT_GROUP`` (``families/grouped_health.py`` ``_candidate``). Live
#: 2026-10-25 008 ``required_link_down``: the test veth's peer, also declared
#: required, read ``lowerlayerdown`` in the same tick, its finding joined the
#: ``-network_link_down-<a>`` incident, which became NODE_HEALTH_GROUP and was
#: QUARANTINED -- and the runner, taking only NODE_HEALTH, reported "no
#: network_link_down on <a> finding reached an incident". The event id, not the
#: label, says which finding minted an incident.
NODE_HEALTH_EVENT_TYPES = frozenset(
    {"NODE_HEALTH", "NODE_HEALTH_GROUP", "NODE_RESOURCE_GROUP", "GPU_FAULT_GROUP"}
)


def _moment(value: Any) -> datetime | None:
    try:
        moment = parse_time(value)
    except ValueError:
        return None
    if moment is not None and moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def minted_in_read(activity: dict[str, Any], incident: dict[str, Any]) -> bool:
    """Whether ``incident`` is one the activity read itself was asked for.

    ``general_live_store.ACTIVITY_PROBE`` lists the node's incidents created
    since the read's ``since`` in every state, but it also adds the incident
    of any workflow created since, however old that incident is (a validated
    restore, a re-dispatch). The read names its ``since``, ``cluster_id`` and
    ``node_id``: an incident created before that moment, or not naming that
    node and cluster, is not the finding a variant caused. The state never
    counts -- a diagnostic-only incident is often RECOVERED before the next
    poll. A scripted document that names none of the three is taken as it is.
    """

    since = _moment(activity.get("since"))
    if since is not None:
        created = _moment(incident.get("created_at"))
        if created is None or created < since:
            return False
    node = activity.get("node_id")
    if node and node not in (incident.get("node_ids") or []):
        return False
    cluster = activity.get("cluster_id")
    return not cluster or incident.get("cluster_id") == cluster


def finding_incidents(
    activity: dict[str, Any],
    metric: str,
    *,
    device: str | None = None,
    rule_id: str | None = None,
) -> list[dict[str, Any]]:
    """Node-health incidents minted by ``metric`` (on ``device``) since the read.

    A metric finding's event id is ``<batch>-<metric>-<device|node>`` and a
    sustained finding's adds ``-<rule id lower>`` (``host_health.py``); the
    incident id is ``inc-<event id>``. ``rule_id`` is the product's rule id
    (``LOW_MEMORY_AVAILABLE``), never its threshold family (``memory``): the
    runner matched on the family and never saw the LOW_MEMORY_AVAILABLE
    incident the product opened and closed RECOVERED (live 2026-10-25).
    Incidents of any state count, under any label the product's grouping gives
    the minted record (``NODE_HEALTH_EVENT_TYPES``); ``minted_in_read`` keeps
    out the ones the read did not select for itself. A finding the product
    folded into an incident another finding minted names no incident of its
    own here. A finding whose event id does not name its metric is matched on
    its own shape (``UNNAMED_FINDINGS``).
    """

    suffix = f"-{metric}-{safe_device(device)}" if device is not None else f"-{metric}-"
    unnamed = (
        UNNAMED_FINDINGS.get(metric) if device is None and rule_id is None else None
    )
    result = []
    for incident in activity.get("incidents") or []:
        if incident.get("event_type") not in NODE_HEALTH_EVENT_TYPES:
            continue
        if not minted_in_read(activity, incident):
            continue
        event_id = str(incident.get("event_id") or "")
        if unnamed is not None:
            prefix, ending = unnamed
            if event_id.startswith(prefix) and event_id.endswith(ending):
                result.append(incident)
            continue
        if rule_id is not None:
            if not event_id.endswith(f"-{rule_id.lower()}"):
                continue
            event_id = event_id[: -len(rule_id) - 1]
        if device is not None and not event_id.endswith(suffix):
            continue
        if device is None and suffix not in event_id + "-":
            continue
        result.append(incident)
    return sorted(result, key=lambda item: str(item.get("created_at")))


def workflows_of(activity: dict[str, Any], incident_id: str) -> list[dict[str, Any]]:
    return sorted(
        (
            item
            for item in activity.get("workflows") or []
            if item.get("incident_id") == incident_id
        ),
        key=lambda item: str(item.get("created_at")),
    )


def operations(workflow: dict[str, Any]) -> list[str]:
    return [str(step.get("operation")) for step in workflow.get("official_steps") or []]


def executions(workflow: dict[str, Any], operation: str) -> list[dict[str, Any]]:
    return [
        item
        for item in workflow.get("step_executions") or []
        if item.get("operation") == operation
    ]


def final_execution(workflow: dict[str, Any], operation: str) -> dict[str, Any] | None:
    items = executions(workflow, operation)
    terminal = [item for item in items if item.get("status") in {"SUCCEEDED", "FAILED"}]
    if terminal:
        return sorted(terminal, key=lambda item: str(item.get("updated_at")))[-1]
    return items[-1] if items else None


def step_parameters(workflow: dict[str, Any], operation: str) -> list[dict[str, Any]]:
    return [
        dict(step.get("parameters") or {})
        for step in workflow.get("official_steps") or []
        if step.get("operation") == operation
    ]


def samples(
    activity: dict[str, Any],
    metric: str,
    *,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Every accepted batch sample of ``metric`` with its batch's time."""

    result = []
    for batch in activity.get("batches") or []:
        for sample in batch.get("samples") or []:
            if sample.get("name") != metric:
                continue
            if device is not None and sample.get("device") != device:
                continue
            result.append(
                {
                    **sample,
                    "observed_at": batch.get("observed_at"),
                    "batch_id": batch.get("batch_id"),
                    "edge_filter_reasons": batch.get("edge_filter_reasons") or [],
                }
            )
    return sorted(result, key=lambda item: str(item["observed_at"]))


def breaching(
    sample: dict[str, Any], threshold: float, comparison: str = "gte"
) -> bool:
    value = sample.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return value >= threshold if comparison == "gte" else value <= threshold


@dataclass
class ChainExpectation:
    """What the product must have done for one finding (or must not have)."""

    metric: str
    device: str | None = None
    rule_id: str | None = None
    present: bool = True
    #: Operations that must appear, in this order, in the incident's workflow.
    operations: tuple[str, ...] = ()
    #: The validation that must have run, and whether it may succeed.
    validation: str | None = None
    validation_must_fail: bool = False
    #: Terminal incident states accepted for the incident.
    incident_states: frozenset[str] = frozenset()
    #: Operations that must not appear anywhere in the incident's workflows.
    forbidden: frozenset[str] = HARDWARE_OPERATIONS
    single: bool = True
    probe_metric: str | None = None
    notes: list[str] = field(default_factory=list)


#: Terminal states of an incident whose workflow only collects evidence and
#: validates (docs/详细设计-v2.md §6.3): RECOVERED once the workflow ended --
#: a failed validation closes it RECOVERED with reason "diagnostic
#: inconclusive" and a DIAGNOSTIC_INCONCLUSIVE notice, never ESCALATED (that
#: would hold the job's restart for nothing) -- or ACTION_PENDING while the
#: workflow still runs.
DIAGNOSTIC_ONLY_STATES = frozenset({"ACTION_PENDING", "RECOVERED"})
DIAGNOSTIC_INCONCLUSIVE = "diagnostic inconclusive"


def _names_held_fault(details: dict[str, Any], metric: str) -> bool:
    """A failed validation that judged the held fault, not something else.

    The product marks ``host_diagnostic_failure`` only for steps whose health
    contract forbids hardware escalation; a diagnostic-only VALIDATE_HOST on
    another contract (live 2026-10-24: load1_per_cpu) still names the metric
    or its ``host_diagnostic:<metric>`` probe in ``node_failures``.
    """

    if details.get(HOST_DIAGNOSTIC_FAILURE) is True:
        return True
    wanted = {metric, f"host_diagnostic:{metric}"}
    return any(
        reason in wanted
        for reasons in (details.get("node_failures") or {}).values()
        if isinstance(reasons, list)
        for reason in reasons
    )


def subsequence(wanted: Sequence[str], actual: Sequence[str]) -> bool:
    iterator = iter(actual)
    return all(item in iterator for item in wanted)


def chain_errors(activity: dict[str, Any], expectation: ChainExpectation) -> list[str]:
    """The finding -> incident -> workflow -> step trace of one expectation."""

    incidents = finding_incidents(
        activity,
        expectation.metric,
        device=expectation.device,
        rule_id=expectation.rule_id,
    )
    label = expectation.metric + (
        f" on {expectation.device}" if expectation.device is not None else ""
    )
    if not expectation.present:
        return [
            f"unexpected {label} incident {item['incident_id']}" for item in incidents
        ]
    if not incidents:
        return [f"no {label} finding reached an incident"]
    errors = []
    if expectation.single and len(incidents) > 1:
        errors.append(
            f"{len(incidents)} {label} incidents (expected one): "
            + ", ".join(str(item["incident_id"]) for item in incidents)
        )
    incident = incidents[0]
    workflows = workflows_of(activity, str(incident["incident_id"]))
    if not workflows:
        return [*errors, f"{label} incident {incident['incident_id']} has no workflow"]
    workflow = workflows[0]
    if expectation.single and len(workflows) > 1:
        errors.append(f"{label} incident has {len(workflows)} workflows")
    steps = operations(workflow)
    if expectation.operations and not subsequence(expectation.operations, steps):
        errors.append(
            f"{label} workflow steps {steps} lack {list(expectation.operations)}"
        )
    for item in workflows:
        found = sorted(set(operations(item)) & set(expectation.forbidden))
        if found:
            errors.append(f"{label} workflow {item['request_id']} plans {found}")
    if expectation.validation is not None:
        execution = final_execution(workflow, expectation.validation)
        if execution is None:
            errors.append(f"{label} workflow never executed {expectation.validation}")
        elif expectation.validation_must_fail:
            if execution.get("status") == "SUCCEEDED":
                errors.append(
                    f"{expectation.validation} succeeded while the fault was still held"
                )
            elif execution.get("status") == "FAILED" and not _names_held_fault(
                execution.get("details") or {},
                expectation.probe_metric or expectation.metric,
            ):
                errors.append(
                    f"{expectation.validation} failed without naming the held fault"
                )
        elif execution.get("status") != "SUCCEEDED":
            errors.append(
                f"{expectation.validation} ended {execution.get('status')}: "
                f"{str(execution.get('error') or '')[:200]}"
            )
    if expectation.probe_metric is not None:
        probes = [
            parameters.get("host_diagnostic_probe")
            for name in VALIDATIONS
            for parameters in step_parameters(workflow, name)
        ]
        if not any(
            isinstance(probe, dict) and probe.get("metric") == expectation.probe_metric
            for probe in probes
        ):
            errors.append(
                f"no validation step carries a host_diagnostic_probe on "
                f"{expectation.probe_metric}"
            )
    if (
        expectation.incident_states
        and incident.get("state") not in expectation.incident_states
    ):
        errors.append(
            f"{label} incident is {incident.get('state')}, expected one of "
            f"{sorted(expectation.incident_states)}"
        )
    elif (
        expectation.validation_must_fail
        and incident.get("state") == "RECOVERED"
        and not any(
            DIAGNOSTIC_INCONCLUSIVE in str(reason)
            for reason in incident.get("reasons") or []
        )
    ):
        errors.append(
            f"{label} incident is RECOVERED while the fault was held, without "
            f"the {DIAGNOSTIC_INCONCLUSIVE!r} reason"
        )
    return errors


def hardware_errors(
    activity: dict[str, Any], *, allowed: Iterable[str] = ()
) -> list[str]:
    """No workflow on the node planned a hardware operation."""

    permitted = set(allowed)
    errors = []
    for workflow in activity.get("workflows") or []:
        found = sorted((set(operations(workflow)) & HARDWARE_OPERATIONS) - permitted)
        if found:
            errors.append(f"workflow {workflow.get('request_id')} plans {found}")
    return errors


def isolation_operations(activity: dict[str, Any]) -> list[str]:
    found = []
    for workflow in activity.get("workflows") or []:
        for operation in operations(workflow):
            if operation in {"MARK_UNSCHEDULABLE", "QUARANTINE", "STOP_WORKLOADS"}:
                found.append(f"{workflow.get('request_id')}:{operation}")
    return found


def settled(activity: dict[str, Any], incident_ids: Iterable[str]) -> bool:
    """Every workflow of ``incident_ids`` has reached a terminal status."""

    wanted = set(incident_ids)
    workflows = [
        item
        for item in activity.get("workflows") or []
        if item.get("incident_id") in wanted
    ]
    return bool(workflows) and all(
        str(item.get("status")) in TERMINAL_WORKFLOWS for item in workflows
    )


def detection_seconds(
    activity: dict[str, Any], incident: dict[str, Any], started: datetime
) -> float | None:
    created = parse_time(incident.get("created_at"))
    return (created - started).total_seconds() if created is not None else None


def oracle_values(
    records: Sequence[dict[str, Any]], getter: Callable[[dict[str, Any]], Any]
) -> list[float]:
    values = []
    for record in records:
        value = getter(record)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def sustained(
    values: Sequence[float], predicate: Callable[[float], bool], minimum: int
) -> bool:
    """At least ``minimum`` consecutive readings satisfy ``predicate``."""

    run = 0
    for value in values:
        run = run + 1 if predicate(value) else 0
        if run >= minimum:
            return True
    return False


def notification_errors(
    activity: dict[str, Any],
    *,
    incident_ids: Iterable[str],
    expected_at_most: int = 1,
) -> list[str]:
    wanted = set(incident_ids)
    mails = [
        item
        for item in activity.get("notifications") or []
        if item.get("incident_id") in wanted
    ]
    if len(mails) > expected_at_most:
        return [
            f"{len(mails)} notifications for one incident (expected at most {expected_at_most})"
        ]
    return []


def node_isolated(node: dict[str, Any]) -> bool:
    return bool(
        node.get("unschedulable")
        or node.get("ownership_annotations")
        or [
            item
            for item in node.get("taints") or []
            if str(item.get("key", "")).startswith("gpu-fault.io/")
        ]
    )


def verdict(
    stages: dict[str, list[str]], *, inconclusive: list[str] | None = None
) -> str:
    if inconclusive:
        return "INCONCLUSIVE"
    return "PASS" if not any(stages.values()) else "FAIL"


__all__ = [
    "ChainExpectation",
    "HOST_DIAGNOSTIC_FAILURE",
    "NODE_HEALTH_EVENT_TYPES",
    "TERMINAL_WORKFLOWS",
    "UNNAMED_FINDINGS",
    "breaching",
    "chain_errors",
    "detection_seconds",
    "executions",
    "final_execution",
    "finding_incidents",
    "hardware_errors",
    "isolation_operations",
    "minted_in_read",
    "node_isolated",
    "notification_errors",
    "operations",
    "oracle_values",
    "parse_time",
    "safe_device",
    "samples",
    "settled",
    "step_parameters",
    "subsequence",
    "sustained",
    "verdict",
    "workflows_of",
]
