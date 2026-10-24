"""The support hand-off notice of a read-only NCCL/EFA hung diagnostic.

The hung DAG (``orchestration/hung_dag.py``) only collects evidence and ends
with ESCALATE_SUPPORT (``execution/hung_handoff.py``); it never cordons,
drains, quarantines, resets or reboots a node and never stops, suspends or
restarts a workload. The hand-off used to send the hardware-escalation mail
(``hardware-escalation-zh-v3``), whose body and support case say the nodes
are out of service, unschedulable and quarantined -- false for every hung
case, and sent on every one of them since the hand-off always runs. This
notice says what the diagnostic is: read-only, the job/attempt and nodes it
looked at, each evidence step's result with its record id and archive
references, the task-scoped fabric verdict (an explicit UNKNOWN or partial
task scope names each node's reason) and what the operator does next.

The builder only formats facts read elsewhere
(``execution/hung_handoff_evidence.py``). Its template and version live here
rather than in ``common.py``: ``common`` also ships in the cluster executor
wheel, while this notice is built and sent by the control plane alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gpu_fault.models import AdvisoryNotification

HUNG_SUPPORT_HANDOFF_TEMPLATE_VERSION = "hung-support-handoff-zh-v1"
#: The ``Disposition`` the support case states: nothing was taken offline.
HUNG_SUPPORT_DISPOSITION = "READ_ONLY_HUNG_DIAGNOSTIC"
#: An operator acts on it; nothing was done on the operator's behalf.
HUNG_SUPPORT_CATEGORY = "OPERATOR_REVIEW"

HUNG_SUPPORT_HANDOFF_EMAIL_TEMPLATE = """\
GPU training attempt suspected hung: read-only diagnostic handed to an operator

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Original event: {event_id}
- Training job: {job_id}
- Attempt: {attempt_id}
- Affected workloads: {workloads}
- Nodes in the diagnostic: {node_ids}
- Policy source: {policy_source}
- Detection reasons:
{reasons}

2. System handling
- Read-only diagnostic: it did not cordon, drain, quarantine, reset or reboot
  any node, and it did not stop, suspend or restart any workload. Hung
  suspicion alone never authorizes an isolation, a restart or a hardware
  action; other faults on these nodes are handled by incidents of their own.
- Training attempt: {attempt_state}
- Diagnostic result: {diagnostic_result}
- The incident is handed to an operator as ESCALATED. An internal support
  hand-off record was created: {ticket_id}

3. Evidence by step
{step_results}

4. Fabric validation (the task's own NICs and RDMA ports)
{fabric_validation}

5. Next steps for the operator
{next_steps}

6. Ticket
- Internal Ticket ID: {ticket_id}
- Disposition: {disposition}
- Workflow operation: ESCALATE_SUPPORT

Template: {template_version}
"""

#: Per-step status words; anything else counts as not run.
SUCCEEDED, SKIPPED, FAILED = "SUCCEEDED", "SKIPPED", "FAILED"
NOT_RUN = "NOT RUN"
#: Fabric verdicts.
FABRIC_PASSED, FABRIC_UNKNOWN, FABRIC_FAILED = "PASSED", "UNKNOWN", "FAILED"
FABRIC_NOT_RUN, FABRIC_NOT_PLANNED = "NOT RUN", "NOT PLANNED"

_TELEMETRY_ONLY = (
    "- Telemetry check only: no active communication test ran and NCCL "
    "recovery is not proven."
)
_FABRIC_NEXT_STEP = {
    FABRIC_UNKNOWN: (
        "The fabric verdict is UNKNOWN for the nodes listed in section 4 -- "
        "missing evidence, not a healthy fabric. Check those nodes' task EFA "
        "ports (rdma -j resource show qp; /sys/class/infiniband/*/ports/*/state) "
        "before the job runs on them again."
    ),
    FABRIC_FAILED: (
        "Fabric validation FAILED on the ports listed in section 4: check them "
        "before the job runs on these nodes again."
    ),
    FABRIC_NOT_RUN: (
        "Fabric validation did not run, so the task's ports were not checked: "
        "check them before the job runs on these nodes again."
    ),
    FABRIC_PASSED: (
        "The task's ports passed a telemetry check only (no active "
        "communication test); it does not show that the training resumed."
    ),
    FABRIC_NOT_PLANNED: "No fabric validation was planned for this diagnostic.",
}


@dataclass(frozen=True)
class HungStepResult:
    """One evidence step of the hung DAG as the hand-off reports it."""

    index: int
    label: str
    status: str
    summary: str = ""
    record_id: str | None = None
    lines: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class HungFabricResult:
    """The task-scoped VALIDATE_FABRIC verdict.

    ``unknown_nodes`` pairs each node the verdict could not judge with its
    reason; ``validated_nodes`` is ``None`` when the record does not say.
    """

    verdict: str
    scope: str = "NONE"
    summary: str = ""
    validated_nodes: tuple[str, ...] | None = None
    unknown_nodes: tuple[tuple[str, str], ...] = ()
    lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class HungHandoffEvidence:
    """What the read-only hung diagnostic collected, step by step.

    ``attempt_released``: the cleanup reason when the attempt stopped being
    live during the diagnostic (its exit handling owns any recovery).
    ``failure``: the workflow's first failure, which it ends FAILED with.
    """

    steps: tuple[HungStepResult, ...]
    fabric: HungFabricResult
    attempt_released: str | None = None
    failure: str | None = None
    job_id: str | None = None
    attempt_id: str | None = None


def _step_block(step: HungStepResult) -> list[str]:
    head = f"- Step {step.index} {step.label}: {step.status}"
    lines = [f"{head} -- {step.summary}" if step.summary else head]
    if step.record_id:
        lines.append(f"  Record: {step.record_id}")
    lines.extend(f"  - {line}" for line in step.lines)
    return lines


def _fabric_block(fabric: HungFabricResult) -> list[str]:
    head = f"- Result: {fabric.verdict}"
    lines = [f"{head} -- {fabric.summary}" if fabric.summary else head]
    if fabric.scope != "NONE":
        lines.append(f"- Evidence scope: {fabric.scope}")
    if fabric.validated_nodes is not None:
        validated = ", ".join(fabric.validated_nodes) or "NONE"
        lines.append(f"- Validated nodes: {validated}")
    if fabric.unknown_nodes:
        lines.append("- Not validated (UNKNOWN):")
        lines.extend(f"  - {node}: {reason}" for node, reason in fabric.unknown_nodes)
    lines.extend(f"- {line}" for line in fabric.lines)
    if fabric.verdict not in {FABRIC_NOT_RUN, FABRIC_NOT_PLANNED}:
        lines.append(_TELEMETRY_ONLY)
    return lines


def _outcome(evidence: HungHandoffEvidence) -> str:
    failed = sum(step.status == FAILED for step in evidence.steps)
    missing = sum(
        step.status not in {SUCCEEDED, SKIPPED, FAILED} for step in evidence.steps
    )
    if not failed and not missing:
        return (
            "every evidence step finished. Collected evidence is not a "
            "recovery: the training is not known to have resumed."
        )
    result = f"incomplete: {failed} step(s) FAILED and {missing} step(s) did not run"
    if failed or evidence.failure:
        first = f" ({evidence.failure})" if evidence.failure else ""
        result += f"; the workflow ends FAILED with its first failure{first}"
    return f"{result}. The evidence that was collected is listed below."


def _attempt_state(evidence: HungHandoffEvidence) -> str:
    if evidence.attempt_released:
        return (
            "no longer live during the diagnostic "
            f"({evidence.attempt_released}): its exit handling (the Completion "
            "Watcher) owns any restart, not this hand-off."
        )
    return (
        "left as it was. It was stalled when the diagnostic started and, unless "
        "it ended on its own since, it still holds its GPUs."
    )


def _next_steps(
    evidence: HungHandoffEvidence, workflow_id: str, incident_id: str
) -> list[str]:
    steps = [
        "Review the triage verdict and the evidence above. The full step "
        f"records are in GET /v1/workflows/{workflow_id} "
        "(step_executions[].details); the diagnostic bundles are the archives "
        "listed under their step.",
        "Decide with the job owner whether the attempt really hangs. The "
        "system does not stop, checkpoint or restart it: the owner stops or "
        "restarts a hung attempt (for example from its last checkpoint).",
        _FABRIC_NEXT_STEP.get(
            evidence.fabric.verdict, _FABRIC_NEXT_STEP[FABRIC_NOT_RUN]
        ),
    ]
    if any(step.status not in {SUCCEEDED, SKIPPED} for step in evidence.steps):
        steps.append(
            "Evidence the FAILED or not-run steps would have collected is "
            "missing; a node that did not answer may itself be unhealthy, so "
            "check it before the job runs there again."
        )
    steps.append(
        "No restore is needed: nothing was isolated. Once the job is handled, "
        "close this incident:\n   gpu-fault-admin workflow-reconcile "
        f'--state-dir "$STATE_DIR" --close-incident {incident_id} '
        '--reason "<what was done>" --reference <change or ticket>'
    )
    return [f"{number}. {text}" for number, text in enumerate(steps, 1)]


class HungSupportHandoffEmailBuilder:
    """Renders the read-only hung diagnostic's support hand-off.

    One notice per workflow: a later evidence round of the same attempt is
    a new workflow with evidence of its own, and a retried hand-off step
    finds its notice already there.
    """

    def build(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        job_id: str | None,
        attempt_id: str | None,
        node_ids: Sequence[str],
        workload_ids: Sequence[str],
        reasons: Sequence[str],
        policy_source: str,
        ticket_id: str,
        evidence: HungHandoffEvidence,
    ) -> AdvisoryNotification:
        nodes = ", ".join(node_ids) or "UNKNOWN"
        job, attempt = job_id or "UNKNOWN", attempt_id or "UNKNOWN"
        step_lines = [line for step in evidence.steps for line in _step_block(step)]
        fabric_lines = _fabric_block(evidence.fabric)
        outcome = _outcome(evidence)
        body = HUNG_SUPPORT_HANDOFF_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            event_id=event_id,
            job_id=job,
            attempt_id=attempt,
            workloads=", ".join(workload_ids) or "NONE",
            node_ids=nodes,
            policy_source=policy_source,
            reasons="\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN",
            attempt_state=_attempt_state(evidence),
            diagnostic_result=outcome,
            ticket_id=ticket_id,
            step_results="\n".join(step_lines) or "- NONE",
            fabric_validation="\n".join(fabric_lines),
            next_steps="\n".join(_next_steps(evidence, workflow_id, incident_id)),
            disposition=HUNG_SUPPORT_DISPOSITION,
            template_version=HUNG_SUPPORT_HANDOFF_TEMPLATE_VERSION,
        )
        refs = list(
            dict.fromkeys(ref for step in evidence.steps for ref in step.evidence_refs)
        )
        case_draft = "\n".join(
            [
                f"Ticket ID: {ticket_id}",
                f"Cluster: {cluster_id}",
                f"Incident: {incident_id}",
                f"Workflow: {workflow_id}",
                f"Training job: {job} attempt {attempt}",
                f"Nodes: {nodes}",
                f"Disposition: {HUNG_SUPPORT_DISPOSITION} (no node isolated, "
                "no workload stopped or restarted)",
                f"Diagnostic result: {outcome}",
                "Evidence:",
                *step_lines,
                "Fabric validation:",
                *fabric_lines,
                "Reasons:",
                *[f"- {item}" for item in reasons],
            ]
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{cluster_id}/{incident_id}/hung-support-handoff/{workflow_id}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                f"[GPU FAULT][training hung: read-only hand-off] {cluster_id} "
                f"{job}/{attempt}: {nodes}"
            ),
            body_text=body,
            support_case_draft=case_draft,
            evidence_refs=refs,
            category=HUNG_SUPPORT_CATEGORY,
        )
