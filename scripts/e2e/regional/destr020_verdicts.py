"""Pure verdict functions and constants of GF-REGIONAL-DESTR-020.

The case proves ARCH-A5 on a real regional deployment: an isolation step whose
target node id the Kubernetes API does not know must fail *closed*. Before A5,
``KubernetesWorkflowAdapter._isolate`` recorded a 404 as ``already_absent_nodes``
and let the workflow proceed on the memory that "an absent node is isolated";
now it returns ``FAILED`` with ``safety_rejection`` and ``absent`` so nothing
downstream can act on a node nobody can observe.

The injection is one API-replay XID whose code is absent from the NVIDIA
catalog, posted for an alias node id that exists nowhere: not as a Kubernetes
node, not as a registered Node Agent, not in any inventory. The policy answers
with the site-safety QUARANTINE (``SITE_SAFETY`` / ``safety_action=QUARANTINE``),
so the workflow is safety-only -- ``FREEZE_EVIDENCE -> MARK_UNSCHEDULABLE ->
QUARANTINE`` -- and contains no physical operation at all. That, plus the
absence of any Node Agent or provider instance for the alias, is the hard stop:
there is nothing on which a reset or reboot could ever be armed.

An XID that maps to RESET_GPU (DESTR-001's XID 46) cannot be used here: the
kernel event model carries no GPU UUID, the alias has no DCGM inventory to
resolve one from, and ``workflow_builder`` blocks RESET_GPU without an explicit
UUID -- the workflow would be BLOCKED before MARK_UNSCHEDULABLE ever ran.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

CASE_ID = "GF-REGIONAL-DESTR-020"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-001"
CONFIRMATION = "DESTR020_IDENTITY_MISMATCH_ISOLATION"
# Absent from the NVIDIA catalog for every product family, so the policy falls
# to the site-safety quarantine (tests/policy: unknown XID is quarantined).
INJECTED_XID = 999
ALIAS_PREFIX = "acceptance-alias-"
EXPECTED_DISPOSITION = "SITE_SAFETY"
EXPECTED_SAFETY_ACTION = "QUARANTINE"
SAFETY_STEPS = ("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE")
ISOLATION_STEP = "MARK_UNSCHEDULABLE"
ABSENT_ERROR_FRAGMENT = "isolation cannot be observed"
# Anything with a physical or provider effect. None of these may ever appear as
# a step execution or a remote command of the alias workflow.
PHYSICAL_OPERATIONS = frozenset(
    {
        "QUIESCE_GPU_SERVICES",
        "VERIFY_NO_GPU_CLIENTS",
        "RESET_GPU",
        "RESET_ALL_GPUS_NVSWITCHES",
        "RESTORE_GPU_SERVICES",
        "RESTART_FABRIC_MANAGER",
        "RESTART_NODE",
        "REPLACE_NODE",
        "RESTART_VM",
        "STOP_WORKLOADS",
        "RESTART_WORKLOAD",
        "REMEDIATE_EFA_DRIVER",
        "RESTART_EFA_DEVICE_PLUGIN",
        "RESTART_GPU_DEVICE_PLUGIN",
    }
)
# The support escalation ``orchestration/escalation.py`` emits for a failed
# containment step whose every FAILED execution carries ``safety_rejection``
# (here: ``absent``): the stage is ``containment_refused`` -> ESCALATE_OPERATOR,
# and its workflow is only FREEZE_EVIDENCE -> ESCALATE_SUPPORT. Planning
# MARK_UNSCHEDULABLE / QUARANTINE again on a node nobody can observe would only
# reproduce the refusal, so the escalation does not plan them.
SUPPORT_ESCALATION_OPERATIONS = ("FREEZE_EVIDENCE", "ESCALATE_SUPPORT")
SUPPORT_ESCALATION_STAGE = "containment_refused"
SUPPORT_ESCALATION_ACTION = "ESCALATE_OPERATOR"
# A support escalation terminates the chain (``CHAIN_TERMINAL_ESCALATIONS``):
# a failed support workflow is handed over, never escalated again. The chain
# therefore has at most one support pair; depth 2 is the runaway.
MAX_ESCALATION_CHAIN_DEPTH = 1
ISOLATION_OPERATIONS = frozenset({ISOLATION_STEP, "QUARANTINE"})
WORKFLOW_TIMEOUT_SECONDS = 600
ESCALATION_TIMEOUT_SECONDS = 180
# How long the runner keeps watching for a second-order escalation after the
# first one appeared. The dispatcher reconciles failed workflows every tick, so
# a chain shows up within seconds; two minutes is generous.
CHAIN_WATCH_SECONDS = 120


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def alias_for_run(run_id: str) -> str:
    """The node id the case injects: deterministic per run, never a real name.

    A digest keeps it a valid DNS label (lower-case hex) while making it
    impossible to collide with a site's node naming by accident; the prefix
    makes it recognisable in every record it leaves behind.
    """

    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    return f"{ALIAS_PREFIX}{digest}"


# --------------------------------------------------------------------------- #
# Alias identity and preflight
# --------------------------------------------------------------------------- #
def alias_errors(
    alias: str,
    *,
    gpu_node_names: list[str],
    kubernetes_node_present: bool,
    agent: dict[str, Any] | None,
    agents: list[dict[str, Any]],
) -> list[str]:
    """The alias must resolve to nothing: no node, no Agent, no fleet entry."""

    errors: list[str] = []
    if not alias.startswith(ALIAS_PREFIX):
        errors.append(f"alias {alias!r} does not carry the {ALIAS_PREFIX} prefix")
    if alias in gpu_node_names:
        errors.append(f"alias {alias} is a GPU node in the cluster")
    if kubernetes_node_present:
        errors.append(f"the Kubernetes API knows a node named {alias}")
    if agent:
        errors.append(f"a Node Agent is registered for {alias}")
    if any(item.get("node_id") == alias for item in agents):
        errors.append(f"the fleet registry lists an Agent for {alias}")
    return errors


def preflight_errors(
    *,
    alias: str,
    alias_facts: dict[str, Any],
    reference_node: dict[str, Any],
    reference_agent: dict[str, Any],
    reference_profile: dict[str, Any],
    reference_workloads: list[dict[str, str]],
    queue: dict[str, Any],
    remote_commands: dict[str, Any],
    alias_event: dict[str, Any] | None,
    tests_passed: bool,
) -> list[str]:
    errors = alias_errors(
        alias,
        gpu_node_names=list(alias_facts.get("gpu_node_names") or []),
        kubernetes_node_present=bool(alias_facts.get("kubernetes_node_present")),
        agent=alias_facts.get("agent"),
        agents=list(alias_facts.get("agents") or []),
    )
    if reference_node.get("ready") != "True":
        errors.append("reference node is not Ready")
    if reference_node.get("ownership_annotations"):
        errors.append("reference node has pre-existing workflow ownership")
    if reference_workloads:
        errors.append("reference node has non-system running Pods")
    if reference_agent.get("lifecycle_state") != "ACTIVE":
        errors.append("reference Node Agent is not ACTIVE")
    if not reference_agent.get("runtime_profile_version"):
        errors.append("reference Node Agent names no runtime profile version")
    if reference_profile.get("warnings"):
        errors.append("runtime profile has warnings")
    if int(queue.get("depth") or 0):
        errors.append("processor queue is not empty")
    if remote_commands.get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if alias_event is not None:
        errors.append(f"an earlier event already exists for {alias}")
    if not tests_passed:
        errors.append("focused regression tests failed")
    return errors


# --------------------------------------------------------------------------- #
# Injection and policy
# --------------------------------------------------------------------------- #
def injection_errors(injection: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    accepted = injection.get("accepted") or {}
    if not accepted.get("processor_request_id"):
        errors.append("the control plane returned no processor request id")
    receipt = injection.get("receipt") or {}
    if receipt.get("status") != 200:
        errors.append(f"processor receipt status is {receipt.get('status')}, not 200")
    return errors


def decision_errors(
    decision: dict[str, Any] | None, event: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if event.get("xid") != INJECTED_XID:
        errors.append(f"matched event is not XID {INJECTED_XID}: {event.get('xid')}")
    if not decision:
        errors.append("no policy decision was recorded for the alias event")
        return errors
    if decision.get("disposition") != EXPECTED_DISPOSITION:
        errors.append(
            f"decision disposition is {decision.get('disposition')!r}, not "
            f"{EXPECTED_DISPOSITION}"
        )
    if decision.get("safety_action") != EXPECTED_SAFETY_ACTION:
        errors.append(
            f"decision safety_action is {decision.get('safety_action')!r}, not "
            f"{EXPECTED_SAFETY_ACTION}"
        )
    if decision.get("action") is not None:
        errors.append(
            f"an unknown XID resolved to an executable action {decision.get('action')!r}"
        )
    return errors


# --------------------------------------------------------------------------- #
# Workflow and remote command
# --------------------------------------------------------------------------- #
def _operations(items: list[dict[str, Any]] | None) -> list[str]:
    return [str(item.get("operation")) for item in items or []]


def workflow_errors(state: dict[str, Any], *, alias: str) -> list[str]:
    errors: list[str] = []
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    if not workflow:
        errors.append("no workflow was created for the alias event")
        return errors
    steps = _operations(workflow.get("safety_steps"))
    if steps != list(SAFETY_STEPS):
        errors.append(f"safety steps {steps} != {list(SAFETY_STEPS)}")
    if workflow.get("status") != "FAILED":
        errors.append(
            f"workflow is {workflow.get('status')}, not FAILED; an unobservable "
            "node must fail the isolation closed"
        )
    executions = workflow.get("step_executions") or []
    isolation = [item for item in executions if item.get("operation") == ISOLATION_STEP]
    if not isolation:
        errors.append(f"{ISOLATION_STEP} never executed")
    for item in isolation:
        if item.get("status") != "FAILED":
            errors.append(
                f"{ISOLATION_STEP} is {item.get('status')}: the isolation of an "
                "absent node was remembered, not observed"
            )
    quarantine = [
        item
        for item in executions
        if item.get("operation") == "QUARANTINE"
        and item.get("status") in {"SUCCEEDED", "WAITING", "RUNNING"}
    ]
    if quarantine:
        errors.append("QUARANTINE ran after a failed isolation")
    reached = sorted(
        {str(item.get("operation")) for item in executions} & PHYSICAL_OPERATIONS
    )
    if reached:
        errors.append(f"workflow reached physical operations: {reached}")
    completed = sorted(set(workflow.get("completed_operations") or []))
    if ISOLATION_STEP in completed or "QUARANTINE" in completed:
        errors.append(f"completed_operations records an isolation: {completed}")
    if list(incident.get("node_ids") or []) != [alias]:
        errors.append(f"incident covers {incident.get('node_ids')}, not just {alias}")
    if incident.get("state") != "ESCALATED":
        errors.append(f"incident state is {incident.get('state')!r}, not ESCALATED")
    return errors


def remote_command_errors(commands: list[dict[str, Any]], *, alias: str) -> list[str]:
    errors: list[str] = []
    by_operation: dict[str, list[dict[str, Any]]] = {}
    for item in commands:
        by_operation.setdefault(str(item.get("operation")), []).append(item)
    others = sorted(set(by_operation) - {ISOLATION_STEP})
    if others:
        errors.append(f"remote commands exist for other operations: {others}")
    isolation = by_operation.get(ISOLATION_STEP) or []
    if len(isolation) != 1:
        errors.append(
            f"expected one {ISOLATION_STEP} remote command, found {len(isolation)}"
        )
        return errors
    command = isolation[0]
    if command.get("status") != "FAILED":
        errors.append(
            f"{ISOLATION_STEP} remote command is {command.get('status')}, not FAILED"
        )
    details = command.get("result_details") or {}
    if details.get("safety_rejection") is not True:
        errors.append("the isolation failure is not marked safety_rejection")
    if details.get("absent") is not True:
        errors.append("the isolation failure does not record the node as absent")
    if details.get("node_id") != alias:
        errors.append(
            f"the isolation failure names {details.get('node_id')!r}, not {alias}"
        )
    if ABSENT_ERROR_FRAGMENT not in str(command.get("error") or ""):
        errors.append(
            f"the isolation error does not say {ABSENT_ERROR_FRAGMENT!r}: "
            f"{command.get('error')!r}"
        )
    return errors


# --------------------------------------------------------------------------- #
# Escalation
# --------------------------------------------------------------------------- #
def escalation_chain_depth(chain: dict[str, Any]) -> int:
    """How many escalation pairs hang off the alias workflow: 0, 1 or 2.

    The runner resolves ``inc-support-after-<wf>`` / ``workflow-support-after-<wf>``
    (depth 1) and the pair keyed after that support workflow (depth 2). Anything
    at depth 2 means the chain did not terminate at support.
    """

    depth = 0
    if chain.get("incident") or chain.get("workflow"):
        depth = 1
    if chain.get("second_order_incident") or chain.get("second_order_workflow"):
        depth = 2
    return depth


def escalation_errors(chain: dict[str, Any], *, alias: str) -> list[str]:
    """Exactly one support handoff for the alias, and it refuses to isolate.

    The failed isolation carries ``safety_rejection``, so the escalation is
    classified ``containment_refused`` and its workflow is only
    ``FREEZE_EVIDENCE -> ESCALATE_SUPPORT``: nothing on the alias is cordoned or
    quarantined again, the workflow SUCCEEDS (both steps are control-plane
    records) and the chain ends there. A second-order
    ``support-after-support-after-...`` pair is the runaway this case exists to
    rule out: one new FAILED workflow per dispatcher pass, forever.
    """

    errors: list[str] = []
    incident = chain.get("incident") or {}
    workflow = chain.get("workflow") or {}
    if not incident or not workflow:
        errors.append(
            "the failed isolation raised no support escalation; "
            f"{SUPPORT_ESCALATION_STAGE} must hand the alias to a human"
        )
        return errors
    if incident.get("effective_action") != SUPPORT_ESCALATION_ACTION:
        errors.append(
            "support incident action is not "
            f"{SUPPORT_ESCALATION_ACTION}: {incident.get('effective_action')}"
        )
    reasons = [str(item) for item in (incident.get("reasons") or [])]
    if not reasons or not reasons[0].startswith(SUPPORT_ESCALATION_STAGE):
        errors.append(
            "support incident reasons[0] does not start with "
            f"{SUPPORT_ESCALATION_STAGE}: {reasons[:1]}; the refused isolation "
            "was classified as an ordinary containment failure"
        )
    if list(incident.get("node_ids") or []) != [alias]:
        errors.append(
            f"support incident covers {incident.get('node_ids')}, not just {alias}"
        )
    operations = _operations(workflow.get("official_steps"))
    if operations != list(SUPPORT_ESCALATION_OPERATIONS):
        errors.append(
            f"support workflow steps are {operations}, not "
            f"{list(SUPPORT_ESCALATION_OPERATIONS)}"
        )
    executed = _operations(workflow.get("step_executions"))
    isolations = sorted(set(executed) & ISOLATION_OPERATIONS)
    if isolations:
        errors.append(
            f"the support workflow planned an isolation on the alias: {isolations}; "
            "a refused containment must not be re-planned"
        )
    physical = sorted(set(executed) & PHYSICAL_OPERATIONS)
    if physical:
        errors.append(f"the support workflow reached a physical operation: {physical}")
    if workflow.get("status") != "SUCCEEDED":
        errors.append(
            f"support workflow is {workflow.get('status')}, not SUCCEEDED; "
            "FREEZE_EVIDENCE and ESCALATE_SUPPORT are control-plane records and "
            "must complete"
        )
    depth = escalation_chain_depth(chain)
    if depth > MAX_ESCALATION_CHAIN_DEPTH:
        second = chain.get("second_order_workflow") or chain.get(
            "second_order_incident"
        )
        errors.append(
            f"a second-order escalation exists at chain depth {depth} "
            f"({second.get('request_id') or second.get('incident_id')}); the "
            "failed isolation escalated its own escalation and will chain on "
            "every reconcile pass"
        )
    return errors


# --------------------------------------------------------------------------- #
# Fleet and leftovers
# --------------------------------------------------------------------------- #
def fleet_errors(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    *,
    alias: str,
) -> list[str]:
    """No real node's scheduling state may move, and the alias must not appear."""

    errors: list[str] = []
    if alias in after:
        errors.append(f"the alias {alias} appeared as a node after the case")
    if set(before) != set(after):
        errors.append(
            "the GPU node set changed during the case: "
            f"{sorted(set(before) ^ set(after))}"
        )
    for name in sorted(set(before) & set(after)):
        for key in ("unschedulable", "taints", "ownership_annotations"):
            if before[name].get(key) != after[name].get(key):
                errors.append(
                    f"node {name} {key} moved from {before[name].get(key)!r} to "
                    f"{after[name].get(key)!r}"
                )
    return errors


def leftover_records(state: dict[str, Any], chain: dict[str, Any]) -> dict[str, Any]:
    """The control records the case leaves behind, by design.

    Control-record retention is off, and a FAILED incident stays as a FAILED
    incident (DESTR-012 group D is the precedent). They are recorded so the
    operator can find them; there is no node to restore.
    """

    incident = state.get("incident") or {}
    workflow = state.get("workflow") or {}
    support_incident = chain.get("incident") or {}
    support_workflow = chain.get("workflow") or {}
    return {
        "incident_id": incident.get("incident_id"),
        "workflow_request_id": workflow.get("request_id"),
        "support_incident_id": support_incident.get("incident_id"),
        "support_workflow_request_id": support_workflow.get("request_id"),
    }
