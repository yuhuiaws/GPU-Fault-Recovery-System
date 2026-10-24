"""What a read-only hung diagnostic hands to a human, read from its records.

ESCALATE_SUPPORT of the read-only hung DAG (``hung_handoff``) reports every
other step as it ended (``notifications/hung_support.py``): its record id,
status and result, the per-node digests of the retained summary and triage
documents, the diagnostic bundle archives, and the task-scoped fabric verdict
-- an explicit UNKNOWN or partial task scope names each node's reason and the
node's own survey account (``fabric_scope.unknown_task_fabric``). A step the
continuation retired reads NOT RUN and names the failed step it needed.

Read conservatively: a missing or malformed field reads UNKNOWN or "not
recorded", never healthy or zero; a VALIDATE_FABRIC passes only when it
SUCCEEDED without a ``fabric_status`` of its own; every list and text is
bounded so a 128-node job still yields a readable mail (the step records keep
everything). Control-plane only, like the hand-off it serves.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from gpu_fault.execution.hung_handoff import HUNG_SUPPORT_HANDOFF
from gpu_fault.fabric_scope import reported_evidence
from gpu_fault.models import WorkflowEventKind
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import (
    WorkflowRequest,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    execution_matches_step,
)
from gpu_fault.notifications.hung_support import (
    FABRIC_FAILED,
    FABRIC_NOT_PLANNED,
    FABRIC_NOT_RUN,
    FABRIC_PASSED,
    FABRIC_UNKNOWN,
    FAILED,
    NOT_RUN,
    SKIPPED,
    SUCCEEDED,
    HungFabricResult,
    HungHandoffEvidence,
    HungStepResult,
)

#: Per-node lines one step reports; the rest are counted (the record keeps all).
MAX_NODE_LINES = 8
#: Head of a free-text value (an error, a verdict reason) the mail keeps.
TEXT_LIMIT = 300
NODE_LIMIT = 128
_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REFERENCE = re.compile(r"(?:s3|file)://\S{1,1024}")
#: ``hung_lifecycle.skip_inactive_hung``: the attempt stopped being live and
#: its exit handling owns any recovery.
_RELEASED_OWNER = "COMPLETION_WATCHER"
#: What the codes of an unbound task scope mean (``fabric_scope``,
#: ``node_agent/operations/network_bindings.py``).
_SCOPE_HINTS = {
    "TASK_NIC_PORT_SCOPE_UNAVAILABLE": "no task NIC/RDMA port binding",
    "INVALID_TASK_DEVICE_BINDING": "the binding names no usable NIC or port",
    "TASK_RDMA_RESOURCES_UNAVAILABLE": (
        "the node found no verified rank process or could not read "
        "`rdma -j resource show qp`"
    ),
    "TASK_RDMA_COVERAGE_INCOMPLETE": (
        "not every rank process holds an active QP on a present EFA port"
    ),
    "NOT_REPORTED": "the recheck survey reported no scope for the node",
}


def _text(value: Any, limit: int = TEXT_LIMIT) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _code(value: Any) -> str:
    return value if isinstance(value, str) and _CODE.fullmatch(value) else "UNKNOWN"


def _count(value: Any) -> str:
    return str(value) if type(value) is int and value >= 0 else "UNKNOWN"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _node(value: Any) -> str:
    return _text(value, NODE_LIMIT) or "UNKNOWN"


def _by_node(value: Any) -> list[tuple[Any, Any]]:
    return sorted(_mapping(value).items(), key=lambda item: str(item[0]))


def _bounded(lines: list[str]) -> list[str]:
    if len(lines) <= MAX_NODE_LINES:
        return lines
    extra = len(lines) - MAX_NODE_LINES
    return [*lines[:MAX_NODE_LINES], f"... and {extra} more node(s) in the step record"]


def _reason(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(item) for item in value if isinstance(item, (str, int)))
    return _text(value) or "no reason recorded"


def _records(workflow: WorkflowRequest) -> dict[int, WorkflowStepExecution]:
    steps = workflow.official_steps
    return {
        item.step_index: item
        for item in workflow.step_executions
        if item.step_index < len(steps)
        and execution_matches_step(
            item, item.step_index, steps[item.step_index].operation, "official"
        )
    }


def _retired_by(workflow: WorkflowRequest) -> dict[int, int]:
    """Retired step -> the failed step whose output it needed (``hung_handoff``)."""

    retired: dict[int, int] = {}
    for event in workflow.events:
        if (
            event.kind is not WorkflowEventKind.PLAN_REWRITE
            or event.code != HUNG_SUPPORT_HANDOFF
        ):
            continue
        failed = event.details.get("failed_step_index")
        if type(failed) is not int:
            continue
        for index in event.details.get("retired_step_indexes") or []:
            if type(index) is int:
                retired.setdefault(index, failed)
    return retired


def _not_run(index: int, retired_by: Mapping[int, int], superseded: set[int]) -> str:
    if index in retired_by:
        return f"retired: it needs the output of failed step {retired_by[index]}"
    return "superseded" if index in superseded else "no execution record"


def _label(step: WorkflowStepSpec) -> str:
    survey, wave = step.parameters.get("survey_pass"), step.parameters.get("hung_wave")
    if step.operation is Op.COLLECT_HUNG_SUMMARY and survey is not None:
        return f"{step.operation.value} ({_code(survey)})"
    if step.operation is Op.COLLECT_HUNG_TRIAGE and type(wave) is int:
        return f"{step.operation.value} (wave {wave})"
    return step.operation.value


def _failure_lines(details: Mapping[str, Any]) -> list[str]:
    failures = _mapping(details.get("node_failures"))
    lines = [
        f"{_node(node)}: no result ({_reason(reason)})"
        for node, reason in _by_node(failures)
    ]
    undetermined = details.get("undetermined_nodes")
    for node in undetermined if isinstance(undetermined, list) else []:
        if isinstance(node, str) and node not in failures:
            lines.append(f"{_node(node)}: undetermined (no usable answer)")
    return lines


def _digest_lines(details: Mapping[str, Any], what: str) -> list[str]:
    lines = []
    for node, value in _by_node(details.get("node_results")):
        digest = _mapping(value).get("sha256")
        lines.append(
            f"{_node(node)}: {what} sha256 {digest}"
            if isinstance(digest, str) and _SHA256.fullmatch(digest)
            else f"{_node(node)}: {what} (digest not recorded)"
        )
    return lines


#: (one-line result, lines before the per-node lines, per-node lines, refs)
_Read = tuple[str, list[str], list[str], list[str]]


def _freeze(details: Mapping[str, Any]) -> _Read:
    task = _mapping(details.get("hung_task_summary"))
    parts = [f"captured {_text(details.get('captured_at'), 64) or 'UNKNOWN'}"]
    if task:
        parts.append(
            f"task summary {_code(task.get('status'))}, "
            f"{_count(task.get('expected_node_count'))} node(s) expected"
        )
    return "; ".join(parts), [], [], []


def _survey(details: Mapping[str, Any]) -> _Read:
    analysis = _mapping(details.get("hung_survey"))
    summary = (
        f"{_count(analysis.get('responding_node_count'))}/"
        f"{_count(analysis.get('expected_node_count'))} node(s) answered; ranks "
        f"{_count(analysis.get('valid_rank_count'))}/"
        f"{_count(analysis.get('expected_rank_count'))}; diagnosis "
        f"{_code(analysis.get('diagnosis'))}"
        if analysis
        else "no survey analysis recorded"
    )
    nodes = [*_failure_lines(details), *_digest_lines(details, "summary")]
    return summary, [], nodes, []


def _triage(details: Mapping[str, Any], rank_nodes: Mapping[str, str]) -> _Read:
    decision = _mapping(details.get("hung_triage_decision"))
    nodes = [*_failure_lines(details), *_digest_lines(details, "triage")]
    if not decision:
        return "no triage verdict recorded", [], nodes, []
    ranks = [rank for rank in decision.get("culprit_ranks") or [] if type(rank) is int]
    candidates = ", ".join(
        f"{rank}@{rank_nodes[str(rank)]}" if str(rank) in rank_nodes else str(rank)
        for rank in ranks[:MAX_NODE_LINES]
    )
    if len(ranks) > MAX_NODE_LINES:
        candidates += f", ... ({len(ranks)} in total)"
    confirmed = "yes" if decision.get("root_cause_confirmed") is True else "no"
    summary = (
        f"verdict {_code(decision.get('classification'))} (confidence "
        f"{_code(decision.get('confidence'))}); candidate ranks (leads, not "
        f"proof): {candidates or 'NONE'}; root cause confirmed: {confirmed}; "
        f"{_count(decision.get('active_responding_node_count'))}/"
        f"{_count(decision.get('active_requested_node_count'))} sampled node(s) "
        "answered"
    )
    reason = _text(decision.get("reason"))
    return summary, [f"Reason: {reason}"] if reason else [], nodes, []


def _bundle(step: WorkflowStepSpec, details: Mapping[str, Any]) -> _Read:
    nodes, refs = _failure_lines(details), []
    results = _by_node(details.get("node_results"))
    for node, value in results:
        value = _mapping(value)
        ref = value.get("evidence_ref")
        ref = ref if isinstance(ref, str) and _REFERENCE.fullmatch(ref) else None
        digest = value.get("sha256")
        if not (isinstance(digest, str) and _SHA256.fullmatch(digest)):
            digest = "not recorded"
        manifest = _mapping(value.get("manifest_summary"))
        captures = (
            f"; {_count(manifest.get('capture_count'))} capture(s), "
            f"{_count(manifest.get('failed_capture_count'))} failed"
            if manifest
            else ""
        )
        nodes.append(
            f"{_node(node)}: {ref or 'archive reference not recorded'} sha256 "
            f"{digest} ({_count(value.get('size_bytes'))} bytes{captures})"
        )
        if ref:
            refs.append(ref)
    parts = [f"{len(results)} archive(s)"]
    if step.parameters.get("bundle_scope") is not None:
        parts.append(f"scope {_code(step.parameters['bundle_scope'])}")
    outside = step.parameters.get("bundle_not_collected_nodes")
    if isinstance(outside, list) and outside:
        parts.append(f"{len(outside)} job node(s) outside the bounded scope")
    return "; ".join(parts), [], nodes, refs


def _unknown_reason(value: Any) -> str:
    value = _mapping(value)
    reason = _code(value.get("reason"))
    account = reported_evidence(value)
    survey = [f"scope_basis={account['scope_basis']}"]
    for key in ("scope_reason", "scope_source"):
        if key in account:
            survey.append(f"{key}={account[key]}")
    if "expected_process_count" in account or "covered_process_count" in account:
        survey.append(
            "rank processes covered "
            f"{_count(account.get('covered_process_count'))}/"
            f"{_count(account.get('expected_process_count'))}"
        )
    hint = _SCOPE_HINTS.get(account.get("scope_reason") or account["scope_basis"])
    return (
        f"{reason} ({_SCOPE_HINTS.get(reason, 'unrecognized reason')}); node "
        f"survey: {', '.join(survey)}" + (f" -- {hint}" if hint else "")
    )


def _port_lines(step: WorkflowStepSpec) -> list[str]:
    lines = []
    for node, scope in _by_node(step.parameters.get("training_fabric_by_node")):
        scope = _mapping(scope)
        ports = [_text(item, 64) for item in scope.get("rdma_ports") or []]
        nics = [_text(item, 64) for item in scope.get("network_interfaces") or []]
        lines.append(
            f"Task ports of {_node(node)}: RDMA "
            f"{', '.join(filter(None, ports)) or 'NONE'}; NICs "
            f"{', '.join(filter(None, nics)) or 'NONE'}"
        )
    return lines


def _fabric(
    workflow: WorkflowRequest,
    records: Mapping[int, WorkflowStepExecution],
    retired_by: Mapping[int, int],
) -> tuple[int | None, HungFabricResult]:
    indexes = [
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is Op.VALIDATE_FABRIC
    ]
    if not indexes:
        return None, HungFabricResult(FABRIC_NOT_PLANNED)
    index = indexes[-1]
    record = records.get(index)
    if record is None:
        why = _not_run(index, retired_by, set(workflow.superseded_step_indexes))
        return index, HungFabricResult(
            FABRIC_NOT_RUN, summary=f"{why}; the task's ports were not checked"
        )
    details = _mapping(record.details)
    scope = (
        _code(details["fabric_evidence_scope"])
        if "fabric_evidence_scope" in details
        else "NONE"
    )
    validated = details.get("validated_nodes")
    validated_nodes = (
        tuple(_node(node) for node in validated)
        if isinstance(validated, list)
        else None
    )
    if record.status is WorkflowStepStatus.FAILED:
        failures = [
            f"{_node(node)}: {_reason(reason)}"
            for node, reason in _by_node(details.get("node_failures"))
        ]
        return index, HungFabricResult(
            FABRIC_FAILED,
            scope=scope,
            summary=_text(record.error) or "no error recorded",
            validated_nodes=validated_nodes,
            lines=tuple(_bounded(failures)),
        )
    if record.status is not WorkflowStepStatus.SUCCEEDED or details.get("skipped"):
        why = (
            f"skipped: {_code(details.get('skipped'))}"
            if record.status is WorkflowStepStatus.SUCCEEDED
            else f"the step did not finish ({record.status.value})"
        )
        return index, HungFabricResult(
            FABRIC_NOT_RUN, summary=f"{why}; the task's ports were not checked"
        )
    unknown = _by_node(details.get("task_fabric_unknown_nodes"))
    if "fabric_status" in details or unknown:
        # Only a validation that SUCCEEDED with no verdict of its own passed;
        # an UNKNOWN -- or a status this reader does not know -- is missing
        # evidence, never a healthy fabric.
        pairs = [(_node(node), _unknown_reason(value)) for node, value in unknown]
        if len(pairs) > MAX_NODE_LINES:
            extra = len(pairs) - MAX_NODE_LINES
            pairs = [*pairs[:MAX_NODE_LINES], ("...", f"{extra} more in the record")]
        whom = "some nodes" if validated_nodes else "any node"
        return index, HungFabricResult(
            FABRIC_UNKNOWN,
            scope=scope,
            summary=(
                f"the task's NIC/RDMA port scope could not be determined for "
                f"{whom}; this is missing evidence, not a healthy fabric"
            ),
            validated_nodes=validated_nodes,
            unknown_nodes=tuple(pairs),
        )
    return index, HungFabricResult(
        FABRIC_PASSED,
        scope=scope,
        summary="no fault on the task's NICs and RDMA ports in the latest telemetry",
        validated_nodes=validated_nodes,
        lines=tuple(_bounded(_port_lines(workflow.official_steps[index]))),
    )


def _read(
    step: WorkflowStepSpec, details: Mapping[str, Any], rank_nodes: Mapping[str, str]
) -> _Read:
    if step.operation is Op.FREEZE_EVIDENCE:
        return _freeze(details)
    if step.operation is Op.COLLECT_HUNG_SUMMARY:
        return _survey(details)
    if step.operation is Op.COLLECT_HUNG_TRIAGE:
        return _triage(details, rank_nodes)
    if step.operation is Op.COLLECT_DIAGNOSTIC_BUNDLE:
        return _bundle(step, details)
    return "", [], _failure_lines(details), []


def _step_result(
    index: int,
    step: WorkflowStepSpec,
    record: WorkflowStepExecution | None,
    *,
    workflow_id: str,
    retired_by: Mapping[int, int],
    superseded: set[int],
    rank_nodes: Mapping[str, str],
    fabric: tuple[int | None, HungFabricResult],
) -> HungStepResult:
    label = _label(step)
    if record is None:
        return HungStepResult(
            index, label, NOT_RUN, _not_run(index, retired_by, superseded)
        )
    details = _mapping(record.details)
    # The step's own key (the executor's idempotency key): a FAILED or skipped
    # record carries no adapter operation id.
    record_id = (
        _text(record.adapter_operation_id, 256)
        or f"{workflow_id}/{index}/{step.operation.value}"
    )
    if record.status is WorkflowStepStatus.SUCCEEDED and details.get("skipped"):
        owner = details.get("recovery_owner")
        reason = _code(details.get("skipped"))
        if owner:
            reason += f" (recovery owner {_code(owner)})"
        return HungStepResult(index, label, SKIPPED, reason, record_id)
    if record.status not in {WorkflowStepStatus.SUCCEEDED, WorkflowStepStatus.FAILED}:
        return HungStepResult(
            index, label, f"NOT FINISHED ({record.status.value})", "", record_id
        )
    read: _Read
    if index == fabric[0]:
        verdict = fabric[1]
        scope = f" ({verdict.scope})" if verdict.scope != "NONE" else ""
        read = (f"fabric {verdict.verdict}{scope}, see section 4", [], [], [])
    else:
        read = _read(step, details, rank_nodes)
    summary, head, nodes, refs = read
    if record.status is WorkflowStepStatus.FAILED:
        if summary and index != fabric[0]:
            head = [f"Collected before it failed: {summary}", *head]
        summary = _text(record.error) or "no error recorded"
    return HungStepResult(
        index,
        label,
        FAILED if record.status is WorkflowStepStatus.FAILED else SUCCEEDED,
        summary,
        record_id,
        (*head, *_bounded(nodes)),
        tuple(refs),
    )


def _rank_nodes(records: Mapping[int, WorkflowStepExecution]) -> dict[str, str]:
    """Rank -> node from the latest survey analysis that has the map."""

    for index in sorted(records, reverse=True):
        record = records[index]
        if record.operation is not Op.COLLECT_HUNG_SUMMARY:
            continue
        survey = _mapping(_mapping(record.details).get("hung_survey"))
        found = _mapping(survey.get("rank_node_map"))
        if found:
            return {
                str(rank): _node(node)
                for rank, node in found.items()
                if isinstance(node, str)
            }
    return {}


def _attempt_released(records: Mapping[int, WorkflowStepExecution]) -> str | None:
    for index in sorted(records):
        details = _mapping(records[index].details)
        if details.get("skipped") and details.get("recovery_owner") == _RELEASED_OWNER:
            return _code(details.get("skipped"))
    return None


def _parameter(workflow: WorkflowRequest, name: str) -> str | None:
    for step in workflow.official_steps:
        value = step.parameters.get(name)
        if isinstance(value, str) and value:
            return _text(value, NODE_LIMIT)
    return None


def handoff_evidence(
    workflow: WorkflowRequest, handoff_index: int
) -> HungHandoffEvidence:
    """Every step but the hand-off at ``handoff_index``, as it ended."""

    records = _records(workflow)
    retired_by = _retired_by(workflow)
    fabric = _fabric(workflow, records, retired_by)
    rank_nodes = _rank_nodes(records)
    superseded = set(workflow.superseded_step_indexes)
    steps = tuple(
        _step_result(
            index,
            step,
            records.get(index),
            workflow_id=workflow.request_id,
            retired_by=retired_by,
            superseded=superseded,
            rank_nodes=rank_nodes,
            fabric=fabric,
        )
        for index, step in enumerate(workflow.official_steps)
        if index != handoff_index
    )
    return HungHandoffEvidence(
        steps=steps,
        fabric=fabric[1],
        attempt_released=_attempt_released(records),
        failure=_text(workflow.terminal_failure_reason) or None,
        job_id=_parameter(workflow, "job_id"),
        attempt_id=_parameter(workflow, "attempt_id"),
    )
