"""The EFA driver chain's reboot rung on the product's records
(GF-GENERAL-LIVE-013 ``bind_failure`` / ``post_repair_failure``).

A failed ``REMEDIATE_EFA_DRIVER`` chain is designed to end in a reboot. The
step that failed -- the remediation's bind, or a post-repair
``RESTART_EFA_DEVICE_PLUGIN`` / ``VALIDATE_FABRIC`` that observed the fault --
carries ``failure_escalation_action=REBOOT_NODE`` for a DRIVER_UNBOUND finding
(``orchestration/families/health.py`` ``_efa_parameters``), the ladder
classifies the record ``efa_recovery`` and opens exactly one successor,
``workflow-reboot-after-<record>`` (``orchestration/escalation.py``
``_classify``, ``emit``), whose plan is the REBOOT_NODE chain of every
escalation successor: FREEZE_EVIDENCE, MARK_UNSCHEDULABLE, QUARANTINE,
[STOP_WORKLOADS with a workload], RESTART_NODE, VALIDATE_GPU, VALIDATE_HOST,
VALIDATE_FABRIC, RESTORE_SCHEDULING (``_escalation_operations``). The
predecessor is parked QUARANTINED with its FAILED record and closed RECOVERED
("node restored via escalation successor") once the successor restored
scheduling (``orchestration/incident_closure.py``
``_close_escalation_predecessors``).

Live 2026-10-27 06:55-07:06Z (HyperPod, R13g) the product did exactly that and
the variant graded FAIL for the harness's own expectation: its isolation check
read the successor's QUARANTINE as isolation beyond the chain. The checks here
grade the designed shape instead -- on the record before and at the reboot
(``escalation_shape_errors``), the chain's own cordon against isolation beyond
it (``isolation_errors``) and the settled exit (``exit_errors``) -- and read
the trap's record in the shape the probe really writes (``trap_record``).
Every function reads a ``node_activity`` document and answers with a list of
errors; empty means the check holds.
"""

from __future__ import annotations

import json
import re
from typing import Any

from scripts.e2e.regional.general_live_common import (
    node_residual_isolation,
    unsettled_workflows,
    workflow_open,
)
from scripts.e2e.regional.general_live_verdicts import (
    executions,
    finding_incidents,
    operations,
    subsequence,
    workflows_of,
)

EFA_METRIC = "efa_inventory_mismatch"
REMEDIATE_STEP = "REMEDIATE_EFA_DRIVER"
PLUGIN_STEP = "RESTART_EFA_DEVICE_PLUGIN"
VALIDATE_STEP = "VALIDATE_FABRIC"
RESTART_NODE = "RESTART_NODE"
REPLACE_NODE = "REPLACE_NODE"
RELEASE_STEP = "RESTORE_SCHEDULING"
QUARANTINE = "QUARANTINE"
STOP_WORKLOADS = "STOP_WORKLOADS"
#: The product's REMEDIATE_EFA_DRIVER chain (``families/health.py``; a
#: STOP_WORKLOADS / RESTART_WORKLOAD pair joins it only with an active job).
REMEDIATION_CHAIN = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    REMEDIATE_STEP,
    PLUGIN_STEP,
    "TRIGGER_HEALTH_SNAPSHOT",
    VALIDATE_STEP,
    RELEASE_STEP,
)
#: The REBOOT_NODE successor's plan (``escalation.py`` ``_escalation_operations``).
REBOOT_SUCCESSOR_CHAIN = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    QUARANTINE,
    RESTART_NODE,
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    VALIDATE_STEP,
    RELEASE_STEP,
)
#: An escalation successor's event id is ``<rung>-after-<source workflow>`` and
#: its workflow id ``workflow-<event id>`` (``escalation.py`` ``emit``).
ESCALATION_SEPARATOR = "-after-"
REBOOT_SUCCESSOR_PREFIX = "workflow-reboot" + ESCALATION_SEPARATOR
#: Isolation steps. The chain's own cordon is MARK_UNSCHEDULABLE in both
#: records and QUARANTINE in the reboot successor; STOP_WORKLOADS belongs to a
#: step that names the workloads it stops.
ISOLATION_OPERATIONS = frozenset({QUARANTINE, STOP_WORKLOADS})
#: ``trap_witness`` of a trap the product's own record witnessed.
JUDGED_BY_PRODUCT_RECORD = "product record"
JUDGED_BY_PROBE = "probe"
#: The operator hand-off of a withheld escalation (``escalation.py``
#: ``_no_verdict_classification`` -> VALIDATION_EVIDENCE_MISSING_STAGE ->
#: ESCALATE_OPERATOR): ``workflow-support-after-<record>`` planning evidence
#: and the ticket only, the node left with the chain's own cordon.
SUPPORT_SUCCESSOR_PREFIX = "workflow-support" + ESCALATION_SEPARATOR
SUPPORT_HANDOFF_CHAIN = ("FREEZE_EVIDENCE", "ESCALATE_SUPPORT")
VALIDATIONS = frozenset({"VALIDATE_GPU", "VALIDATE_HOST", VALIDATE_STEP})
#: ``post_repair_verdict`` values: the step looked and found the fault, or a
#: bound failed it before it observed anything (the product withholds the
#: reboot it asked for, ``validation_evidence.withheld_escalation``).
OBSERVED = "observed"
WITHHELD = "withheld"
#: The step-detail keys ``validation_without_observed_failure`` reads.
_VERDICT_KEYS = ("failed_nodes", "node_failures")
_OBSERVED_FAILURE_KEYS = ("failed_nodes_observed", "node_failures_observed")
_BOUND_FAILURE_KEYS = ("step_waiting_timeout_seconds", "workflow_execution_deadline")
#: The step-detail keys the evidence record keeps beside status and error.
RECORDED_DETAIL_KEYS = (
    "node_pending",
    "node_failures",
    "failed_nodes",
    "inventory_evidence",
    "step_waiting_timeout_seconds",
    "workflow_execution_deadline",
    "completed_nodes",
    "allocatable",
    "expected",
    "already_healthy",
)

Chain = tuple[dict[str, Any], dict[str, Any]]


def efa_chain(observed: dict[str, Any]) -> Chain | None:
    """The EFA finding's incident and its first workflow, or ``None``."""

    incidents = finding_incidents(observed, EFA_METRIC)
    if not incidents:
        return None
    incident = incidents[0]
    workflows = workflows_of(observed, str(incident["incident_id"]))
    if not workflows:
        return None
    return incident, workflows[0]


def incident_state(observed: dict[str, Any], incident_id: str) -> str:
    """The state the document records for ``incident_id`` (``""`` if absent)."""

    for item in observed.get("incidents") or []:
        if str(item.get("incident_id")) == incident_id:
            return str(item.get("state") or "")
    return ""


def successors_of(observed: dict[str, Any], request_id: str) -> list[dict[str, Any]]:
    """The escalation successors the product opened for ``request_id``
    (``workflow-<rung>-after-<request id>``, one generation: a successor's
    own successor ends with the same id and is not one of these), oldest
    first."""

    shape = re.compile(
        rf"workflow-[a-z_]+{re.escape(ESCALATION_SEPARATOR)}{re.escape(request_id)}"
    )
    return sorted(
        (
            item
            for item in observed.get("workflows") or []
            if shape.fullmatch(str(item.get("request_id") or ""))
        ),
        key=lambda item: str(item.get("created_at")),
    )


def reboot_successors(
    observed: dict[str, Any], request_id: str
) -> list[dict[str, Any]]:
    return [
        item
        for item in successors_of(observed, request_id)
        if str(item.get("request_id")).startswith(REBOOT_SUCCESSOR_PREFIX)
    ]


def node_executions(observed: dict[str, Any], operation: str) -> list[dict[str, Any]]:
    """Every execution of ``operation`` on the node's workflows."""

    return [
        {**item, "workflow_request_id": workflow.get("request_id")}
        for workflow in observed.get("workflows") or []
        for item in executions(workflow, operation)
    ]


def terminal_steps(items: list[dict[str, Any]]) -> int:
    """Distinct (workflow, step) pairs that ended SUCCEEDED or FAILED."""

    return len(
        {
            (item.get("workflow_request_id"), item.get("step_index"))
            for item in items
            if item.get("status") in {"SUCCEEDED", "FAILED"}
        }
    )


def reboot_planned_errors(observed: dict[str, Any]) -> list[str]:
    """The spec'd approved branch: a RESTART_NODE planned on the node.

    A failed EFA remediation (bind_failure) or a post-repair validation that
    observed the failure (post_repair_failure) carries
    ``failure_escalation_action=REBOOT_NODE``; the product's ladder opens an
    escalation incident whose workflow plans RESTART_NODE. Found by scanning
    every workflow on the node for the reboot step -- the escalation incident's
    event id does not name the EFA metric.
    """

    for workflow in observed.get("workflows") or []:
        if RESTART_NODE in operations(workflow):
            return []
    return ["no RESTART_NODE escalation was planned (the spec's approved branch)"]


def _record_errors(observed: dict[str, Any], workflow: dict[str, Any]) -> list[str]:
    """The remediation record itself: one attempt, the chain, FAILED, no release."""

    request_id = str(workflow.get("request_id"))
    errors = []
    remediations = [
        item
        for item in observed.get("workflows") or []
        if REMEDIATE_STEP in operations(item)
    ]
    if len(remediations) > 1:
        errors.append(
            f"{len(remediations)} workflows plan {REMEDIATE_STEP} "
            f"({', '.join(str(item.get('request_id')) for item in remediations)}): "
            "the remediation was re-run instead of escalated"
        )
    attempts = terminal_steps(
        [
            {**item, "workflow_request_id": request_id}
            for item in executions(workflow, REMEDIATE_STEP)
        ]
    )
    if attempts != 1:
        errors.append(
            f"{REMEDIATE_STEP} ended {attempts} times in the remediation record "
            "(expected exactly one attempt)"
        )
    planned = operations(workflow)
    if not subsequence(REMEDIATION_CHAIN, planned):
        errors.append(
            f"the remediation record plans {planned}, not the chain "
            f"{list(REMEDIATION_CHAIN)}"
        )
    status = str(workflow.get("status") or "")
    if status != "FAILED":
        errors.append(
            f"the remediation record ended {status or 'unknown'}, expected FAILED "
            "(the failed step hands the node to the reboot rung)"
        )
    if any(
        item.get("status") == "SUCCEEDED" for item in executions(workflow, RELEASE_STEP)
    ):
        errors.append(
            "the remediation record restored scheduling although its step failed"
        )
    return errors


def _successor_errors(observed: dict[str, Any], request_id: str) -> list[str]:
    """Exactly one reboot-after successor with the REBOOT_NODE chain, no other
    rung, nothing after the reboot."""

    errors = []
    successors = successors_of(observed, request_id)
    reboots = reboot_successors(observed, request_id)
    for item in successors:
        if item not in reboots:
            errors.append(
                f"successor {item.get('request_id')} of the remediation record is "
                "not the REBOOT_NODE rung"
            )
    if len(reboots) != 1:
        errors.append(
            f"{len(reboots)} reboot-after successors of the remediation record "
            f"(expected exactly one): {[item.get('request_id') for item in reboots]}"
        )
    for item in reboots:
        action = item.get("official_action")
        if action not in (None, "REBOOT_NODE"):
            errors.append(
                f"successor {item.get('request_id')} carries official_action {action}"
            )
        steps = operations(item)
        if not subsequence(REBOOT_SUCCESSOR_CHAIN, steps):
            errors.append(
                f"the reboot successor plans {steps}, not the REBOOT_NODE chain "
                f"{list(REBOOT_SUCCESSOR_CHAIN)}"
            )
        further = successors_of(observed, str(item.get("request_id")))
        if further:
            errors.append(
                "the reboot did not end the chain: "
                f"{[entry.get('request_id') for entry in further]} followed the rung"
            )
    return errors


def escalation_shape_errors(observed: dict[str, Any], pair: Chain | None) -> list[str]:
    """The ladder's designed shape on the record, before and at the reboot.

    One remediation record (no second workflow re-running
    REMEDIATE_EFA_DRIVER), one terminal REMEDIATE_EFA_DRIVER attempt, the
    product's own chain, the record ended FAILED without restoring scheduling
    on the failed reading, exactly one ``workflow-reboot-after-<record>``
    successor planning the REBOOT_NODE chain (FREEZE_EVIDENCE,
    MARK_UNSCHEDULABLE, QUARANTINE, RESTART_NODE, the three validations,
    RESTORE_SCHEDULING), RESTART_NODE executed once and SUCCEEDED, no
    REPLACE_NODE anywhere and no successor of the successor (the reboot ended
    the fault: the fixture does not survive a boot).
    """

    if pair is None:
        return ["no efa_inventory_mismatch incident with a workflow to grade"]
    _, workflow = pair
    errors = _record_errors(observed, workflow)
    errors.extend(_successor_errors(observed, str(workflow.get("request_id"))))
    rungs = node_executions(observed, RESTART_NODE)
    succeeded = [item for item in rungs if item.get("status") == "SUCCEEDED"]
    if terminal_steps(rungs) != 1 or len(succeeded) != 1:
        errors.append(
            f"RESTART_NODE executed {terminal_steps(rungs)} time(s) with "
            f"{len(succeeded)} SUCCEEDED (expected exactly one SUCCEEDED rung)"
        )
    replacements = [
        str(item.get("request_id"))
        for item in observed.get("workflows") or []
        if REPLACE_NODE in operations(item)
    ]
    if replacements:
        errors.append(f"REPLACE_NODE planned by {replacements}: the ladder went on")
    return errors


def isolation_errors(observed: dict[str, Any], pair: Chain | None) -> list[str]:
    """Isolation beyond the chain's own cordon.

    The remediation record cordons the node (MARK_UNSCHEDULABLE) and never
    quarantines it (``families/health.py``); its REBOOT_NODE successor cordons
    and quarantines the node it reboots (``escalation.py``
    ``_escalation_operations``) -- that QUARANTINE is the chain's own. Either
    stops workloads only on a step that names them. A QUARANTINE in the
    remediation record, a STOP_WORKLOADS naming no workload, and any isolation
    step of a workflow outside the chain are beyond it.
    """

    chain_ids: set[str] = set()
    source_id = ""
    if pair is not None:
        source_id = str(pair[1].get("request_id"))
        chain_ids.add(source_id)
        chain_ids.update(
            str(item.get("request_id"))
            for item in reboot_successors(observed, source_id)
        )
    errors: list[str] = []
    for workflow in observed.get("workflows") or []:
        request_id = str(workflow.get("request_id"))
        for step in workflow.get("official_steps") or []:
            operation = str(step.get("operation"))
            if operation not in ISOLATION_OPERATIONS:
                continue
            if request_id not in chain_ids:
                errors.append(
                    f"workflow {request_id} plans {operation} beyond the chain"
                )
            elif operation == STOP_WORKLOADS and not step.get("workload_ids"):
                errors.append(
                    f"workflow {request_id} plans STOP_WORKLOADS on a step that "
                    "names no workload"
                )
            elif operation == QUARANTINE and request_id == source_id:
                errors.append(
                    f"the remediation record {request_id} itself plans QUARANTINE "
                    "(its cordon is MARK_UNSCHEDULABLE; QUARANTINE belongs to the "
                    "reboot successor)"
                )
    return list(dict.fromkeys(errors))


def exit_errors(
    observed: dict[str, Any], pair: Chain | None, node: dict[str, Any]
) -> list[str]:
    """The settled exit after the rung.

    The reboot successor ended SUCCEEDED with its RESTORE_SCHEDULING done and
    its incident RECOVERED; the EFA incident, parked QUARANTINED when its record
    failed, was closed by the successor's restore
    (``incident_closure._close_escalation_predecessors``: "node restored via
    escalation successor"); the node carries none of the product's isolation.
    """

    if pair is None:
        return []
    incident, workflow = pair
    errors = []
    for item in reboot_successors(observed, str(workflow.get("request_id"))):
        request_id = str(item.get("request_id"))
        status = str(item.get("status") or "")
        if status != "SUCCEEDED":
            errors.append(
                f"the reboot successor {request_id} ended {status or 'unknown'}, "
                "expected SUCCEEDED (its validations grade a node whose function "
                "rebound at boot)"
            )
        if not any(
            entry.get("status") == "SUCCEEDED"
            for entry in executions(item, RELEASE_STEP)
        ):
            errors.append(
                f"the reboot successor {request_id} did not restore scheduling"
            )
        state = incident_state(observed, str(item.get("incident_id")))
        if state != "RECOVERED":
            errors.append(
                f"the escalation incident {item.get('incident_id')} is "
                f"{state or 'unknown'}, expected RECOVERED"
            )
    source_id = str(incident.get("incident_id"))
    state = incident_state(observed, source_id) or str(incident.get("state") or "")
    if state != "RECOVERED":
        errors.append(
            f"the EFA incident {source_id} is {state or 'unknown'}: the successor's "
            "restore closes its QUARANTINED predecessor (expected RECOVERED)"
        )
    residual = node_residual_isolation(node)
    if any(residual.values()):
        errors.append(f"the node still carries the product's isolation: {residual}")
    return errors


def ladder_stopped(observed: dict[str, Any]) -> bool:
    """The product will not reboot: every workflow on the node ended -- none
    open by the product's own rule (``workflow_open``: executable, or BLOCKED
    with an occupying kind, a successor still waiting for its budget) -- no
    RESTART_NODE SUCCEEDED on it, and the finding's incident left
    ACTION_PENDING. The successor follows its predecessor's failure within
    milliseconds, so a caller confirms this on two polls before acting on it."""

    workflows = observed.get("workflows") or []
    if not workflows or unsettled_workflows(observed):
        return False
    if any(workflow_open(item) for item in workflows):
        return False
    if any(
        item.get("status") == "SUCCEEDED"
        for item in node_executions(observed, RESTART_NODE)
    ):
        return False
    pair = efa_chain(observed)
    if pair is None:
        return False
    state = incident_state(observed, str(pair[0].get("incident_id")))
    return state not in {"", "ACTION_PENDING"}


def trap_record(reading: dict[str, Any]) -> dict[str, Any]:
    """The trap's own record from a probe reading.

    ``summary`` carries it as ``trap_fired_at`` -- the JSON text of
    ``/run/.../trap-fired`` (``{"fired": true, "bound": false, "at": ...,
    "trigger": "collector_restart" | "grace_fallback", "seconds_after_bind":
    ...}`` plus the collector readings the trigger rests on, or ``fired:
    false`` with a ``reason``), ``None`` until the unit wrote it (the live
    shape) -- and ``release`` as ``trap_fired``; a scripted reading may carry
    the object itself.
    """

    value = reading.get("trap_fired_at", reading.get("trap_fired"))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {"raw": value}
    return dict(value) if isinstance(value, dict) else {}


def trap_fired_errors(
    outcome: dict[str, Any], *, post_repair_failed: bool
) -> tuple[list[str], str]:
    """Whether the trap fired, and who witnessed it: the probe's own record,
    or -- when the reboot took the probe before the record was read -- the
    product's record of a SUCCEEDED remediation followed by a post-repair
    failure on the same function (nothing else re-unbinds it)."""

    record = outcome.get("record") or {}
    if record.get("fired") is True:
        return [], JUDGED_BY_PROBE
    lost = outcome.get("lost")
    if lost is not None:
        if post_repair_failed:
            return [], JUDGED_BY_PRODUCT_RECORD
        return [
            "the probe was lost before the trap's record was read "
            f"({lost.get('probe_lost')}) and the product's record shows no "
            "post-repair failure to witness it"
        ], ""
    return [
        "the re-unbind trap did not fire after the product's bind: "
        f"{record or 'no record within the window'}"
    ], ""


def support_successors(
    observed: dict[str, Any], request_id: str
) -> list[dict[str, Any]]:
    return [
        item
        for item in successors_of(observed, request_id)
        if str(item.get("request_id")).startswith(SUPPORT_SUCCESSOR_PREFIX)
    ]


def post_repair_verdict(step: dict[str, Any]) -> str:
    """How a FAILED post-repair step failed, by the product's own rule.

    A node action or adapter step (RESTART_EFA_DEVICE_PLUGIN) that failed
    looked at the node: ``OBSERVED``, and its REBOOT_NODE request stands. A
    validation FAILED with a verdict (``node_failures`` / ``failed_nodes`` or
    the ``*_observed`` keys) is ``OBSERVED`` too. A validation a bound failed
    before it observed any failure -- a ``node_pending`` wait, the per-step
    cap's ``step_waiting_timeout_seconds``, the deadline -- is ``WITHHELD``
    (``validation_evidence.validation_without_observed_failure``): the
    product keeps the reboot it asked for and hands the node to an operator.
    """

    if str(step.get("operation")) not in VALIDATIONS:
        return OBSERVED
    details = step.get("details") or {}
    if any(details.get(key) for key in _VERDICT_KEYS + _OBSERVED_FAILURE_KEYS):
        return OBSERVED
    pending = details.get("node_pending")
    if isinstance(pending, dict) and pending:
        return WITHHELD
    if any(key in details for key in _BOUND_FAILURE_KEYS):
        return WITHHELD
    if "per-step cap" in str(step.get("error") or ""):
        return WITHHELD
    return OBSERVED


def handoff_shape_errors(observed: dict[str, Any], pair: Chain | None) -> list[str]:
    """The designed hand-off of a withheld escalation on the record.

    The remediation record FAILED without restoring scheduling, exactly one
    ``workflow-support-after-<record>`` successor planning FREEZE_EVIDENCE and
    ESCALATE_SUPPORT (no isolation step: the chain's cordon is already on the
    node), no reboot-after successor, no RESTART_NODE executed, no
    REPLACE_NODE; the successor SUCCEEDED (the ticket went out).
    """

    if pair is None:
        return ["no efa_inventory_mismatch incident with a workflow to grade"]
    _, workflow = pair
    request_id = str(workflow.get("request_id"))
    errors = []
    status = str(workflow.get("status") or "")
    if status != "FAILED":
        errors.append(
            f"the remediation record ended {status or 'unknown'}, expected FAILED"
        )
    if any(
        item.get("status") == "SUCCEEDED" for item in executions(workflow, RELEASE_STEP)
    ):
        errors.append(
            "the remediation record restored scheduling although its step failed"
        )
    supports = support_successors(observed, request_id)
    if len(supports) != 1:
        errors.append(
            f"{len(supports)} support-after successors of the remediation record "
            f"(expected exactly one): {[item.get('request_id') for item in supports]}"
        )
    for item in supports:
        steps = operations(item)
        if not subsequence(SUPPORT_HANDOFF_CHAIN, steps):
            errors.append(
                f"the hand-off {item.get('request_id')} plans {steps}, not "
                f"{list(SUPPORT_HANDOFF_CHAIN)}"
            )
        if set(steps) & (ISOLATION_OPERATIONS | {RESTART_NODE, REPLACE_NODE}):
            errors.append(
                f"the hand-off {item.get('request_id')} plans isolation or a rung: {steps}"
            )
        if str(item.get("status") or "") != "SUCCEEDED":
            errors.append(
                f"the hand-off {item.get('request_id')} ended {item.get('status')}, "
                "expected SUCCEEDED (the ticket)"
            )
    reboots = reboot_successors(observed, request_id)
    if reboots:
        errors.append(
            "a reboot-after successor followed a validation that observed nothing: "
            f"{[item.get('request_id') for item in reboots]}"
        )
    rungs = node_executions(observed, RESTART_NODE)
    if rungs:
        errors.append(
            f"RESTART_NODE executed {len(rungs)} time(s) on a withheld reboot"
        )
    replacements = [
        str(item.get("request_id"))
        for item in observed.get("workflows") or []
        if REPLACE_NODE in operations(item)
    ]
    if replacements:
        errors.append(f"REPLACE_NODE planned by {replacements}")
    return errors


def node_result(
    observed: dict[str, Any],
    workflow: dict[str, Any] | None,
    *,
    operation: str,
    node: str,
) -> dict[str, Any]:
    """The Node Agent's result for ``operation`` on ``node``.

    The node-action fold writes it as ``node_results[<node>]`` into the step
    execution's details and the remote command's ``result_details``
    (``adapters/node_action/step_execution.py``); a ``RemoteActionCommand``
    names its operation under ``step.operation`` (``regional.py``), never at
    the top level -- the first live ``post_repair_failure`` (2026-10-27) read
    a top-level ``operation`` and found no receipt at all. The step execution
    is read first, then the command of that operation (``step.operation`` or
    a scripted top-level ``operation``) with the matching ``step_index``.
    """

    for item in executions(workflow or {}, operation):
        results = (item.get("details") or {}).get("node_results") or {}
        value = results.get(node)
        if isinstance(value, dict) and value:
            return value
    indexes = {item.get("step_index") for item in executions(workflow or {}, operation)}
    for command in observed.get("commands") or []:
        step = command.get("step") or {}
        named = command.get("operation") or step.get("operation")
        if named != operation:
            continue
        if indexes and command.get("step_index") not in indexes | {None}:
            continue
        results = (command.get("result_details") or {}).get("node_results") or {}
        value = results.get(node)
        if isinstance(value, dict) and value:
            return value
    return {}


def _compact(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in sorted(items, key=lambda entry: entry.get("step_index") or 0):
        details = item.get("details") or {}
        kept = {key: details[key] for key in RECORDED_DETAIL_KEYS if key in details}
        result.append(
            {
                "step_index": item.get("step_index"),
                "operation": item.get("operation"),
                "status": item.get("status"),
                "updated_at": item.get("updated_at"),
                "error": str(item.get("error") or "")[:200] or None,
                **({"details": kept} if kept else {}),
            }
        )
    return result


def escalation_record(
    observed: dict[str, Any],
    pair: Chain | None,
    *,
    boot_before: str,
    boot_now: str,
) -> dict[str, Any]:
    """What a reader needs to see the ladder: the remediation record's
    executions, every successor with its plan and incident state, the
    RESTART_NODE executions and the boot ids."""

    incident, workflow = pair if pair is not None else ({}, {})
    request_id = str(workflow.get("request_id") or "")
    return {
        "incident_id": incident.get("incident_id"),
        "incident_state": incident_state(observed, str(incident.get("incident_id")))
        or incident.get("state"),
        "workflow_request_id": workflow.get("request_id"),
        "workflow_status": workflow.get("status"),
        "executions": _compact(workflow.get("step_executions") or []),
        "successors": [
            {
                "request_id": item.get("request_id"),
                "incident_id": item.get("incident_id"),
                "incident_state": incident_state(
                    observed, str(item.get("incident_id"))
                ),
                "status": item.get("status"),
                "official_action": item.get("official_action"),
                "operations": operations(item),
                "executions": _compact(item.get("step_executions") or []),
            }
            for item in (successors_of(observed, request_id) if request_id else [])
        ],
        "restart_node_executions": _compact(node_executions(observed, RESTART_NODE)),
        "boot_id_before": boot_before,
        "boot_id_now": boot_now,
        "rebooted": bool(boot_before and boot_now and boot_before != boot_now),
    }


__all__ = [
    "EFA_METRIC",
    "ESCALATION_SEPARATOR",
    "ISOLATION_OPERATIONS",
    "JUDGED_BY_PROBE",
    "JUDGED_BY_PRODUCT_RECORD",
    "OBSERVED",
    "RECORDED_DETAIL_KEYS",
    "SUPPORT_HANDOFF_CHAIN",
    "SUPPORT_SUCCESSOR_PREFIX",
    "WITHHELD",
    "PLUGIN_STEP",
    "QUARANTINE",
    "REBOOT_SUCCESSOR_CHAIN",
    "REBOOT_SUCCESSOR_PREFIX",
    "RELEASE_STEP",
    "REMEDIATE_STEP",
    "REMEDIATION_CHAIN",
    "REPLACE_NODE",
    "RESTART_NODE",
    "STOP_WORKLOADS",
    "VALIDATE_STEP",
    "Chain",
    "efa_chain",
    "escalation_record",
    "escalation_shape_errors",
    "exit_errors",
    "handoff_shape_errors",
    "incident_state",
    "isolation_errors",
    "ladder_stopped",
    "node_executions",
    "node_result",
    "post_repair_verdict",
    "reboot_planned_errors",
    "reboot_successors",
    "successors_of",
    "support_successors",
    "terminal_steps",
    "trap_fired_errors",
    "trap_record",
]
