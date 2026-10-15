"""Pure verdict functions and constants of GF-REGIONAL-DESTR-014.

Split out of ``run_destr014_branch_exhaustion.py`` so the runner stays a
driver: everything here is unit-tested against synthetic control-plane,
node and CloudTrail snapshots and touches no cluster.
"""

from __future__ import annotations

import sys
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_fault.adapters.common import (  # noqa: E402
    quarantine_taint_value as product_quarantine_taint_value,
)
from gpu_fault.node_provider import parse_provider_id  # noqa: E402
from gpu_fault.recovery_safety import (  # noqa: E402
    recovery_safety_errors,
    unresolved_details,
)
from scripts.e2e.regional.collector_reboot_evidence import (  # noqa: E402
    partition_provider_events,
    provider_event_source,
    reboot_event_names,
)
from scripts.e2e.regional.destr015_verdicts import (  # noqa: E402
    event_observed_at,
    spread_errors,
)
from scripts.e2e.regional.guardrail_audit_evidence import (  # noqa: E402
    EC2_REPLACEMENT_MUTATIONS,
)
from scripts.e2e.regional.warm_spare_fixture import QUARANTINE_TAINT  # noqa: E402

# The HyperPod vocabulary; ``provider_reboot_events`` / ``provider_forbidden_events``
# select the EC2 one (RebootInstances; Run/Terminate/Stop/StartInstances in
# ``ec2.amazonaws.com``) for an eks-managed cluster, so both providers run the
# case under one id.
REBOOT_EVENTS = {"BatchRebootClusterNodes", "RebootClusterNodes"}
FORBIDDEN_EVENTS = {
    "BatchDeleteClusterNodes",
    "BatchReplaceClusterNodes",
    "DeleteClusterNodes",
    "ReplaceClusterNodes",
}

# Wall-clock allowances (seconds) for the lifetime arithmetic. A branch takes a
# real reboot plus a validation-and-release tail; the shared containment runs
# once. Deliberately generous so a passing estimate is a real safety margin.
REBOOT_ALLOWANCE_SECONDS = 600
VALIDATION_ALLOWANCE_SECONDS = 300
CONTAINMENT_ALLOWANCE_SECONDS = 120

EXHAUSTION_PREFIX = "node branch escalation exhausted"
# With no ``gpu-fault.io/spare`` label anywhere (the preflight insists on zero
# spares) ``hyperpod_spares`` returns ``applicable=False`` and the lifecycle
# adapter fails REPLACE_NODE with this message. ``insufficient healthy HyperPod
# spares`` is the *other* branch -- a spare is labelled but none is healthy or
# matching -- which this case never reaches (docs/区域模式端到端验收测试用例.md,
# DESTR-014 preconditions).
ZERO_SPARE_REPLACE_ERROR = (
    "warm-spare replacement is required; provider node replacement API "
    "fallback is disabled"
)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    store = preflight.get("store") or {}
    return {
        "release_id": preflight.get("release_id"),
        "fault_node_uid": (preflight.get("fault_node") or {}).get("uid"),
        "fault_node_boot_id": (preflight.get("fault_node") or {}).get("boot_id"),
        "sibling_node_uid": (preflight.get("sibling_node") or {}).get("uid"),
        "sibling_node_boot_id": (preflight.get("sibling_node") or {}).get("boot_id"),
        "sibling_recovery_identity_sha256": canonical_digest(
            preflight.get("recovery_agent")
        ),
        "runtime_profile_version": (store.get("profile") or {}).get("profile_version"),
        "provider_inventory_sha256": (preflight.get("provider_inventory") or {}).get(
            "sha256"
        ),
        "executor_env_baseline_sha256": (
            preflight.get("executor_env_window") or {}
        ).get("baseline_sha256"),
    }


def identity_digest(identity: dict[str, Any]) -> str:
    return canonical_digest(identity)


def evidence_components(details: dict[str, Any]) -> dict[str, str]:
    return {
        name: canonical_digest(details.get(name))
        for name in ("preflight_identity", "workflow", "incident", "provider_events")
        if name in details
    }


def case_digest(components: dict[str, str]) -> str:
    return canonical_digest(components)


# --------------------------------------------------------------------------- #
# Pure verdict functions (unit-tested)
# --------------------------------------------------------------------------- #
def quarantine_taint_value(incident_id: str) -> str:
    """The taint value the product writes: ``incident-<sha256[:24]>``.

    Delegated to the product so the two cannot drift; a local copy without the
    ``incident-`` prefix judged every correct live run's taint as wrong.
    """

    return product_quarantine_taint_value(incident_id)


def _step_node(step: dict[str, Any]) -> list[str]:
    return list(step.get("branch_node_ids") or step.get("node_ids") or [])


def _branch_executions(
    steps: list[dict[str, Any]],
    executions: list[dict[str, Any]],
    node: str,
) -> list[dict[str, Any]]:
    result = []
    for item in executions:
        index = item.get("step_index")
        if isinstance(index, int) and 0 <= index < len(steps):
            if _step_node(steps[index]) == [node]:
                result.append(item)
    return result


def _first(
    executions: list[dict[str, Any]],
    operation: str,
    status: str,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in executions
            if item.get("operation") == operation and item.get("status") == status
        ),
        None,
    )


# Where node-b's RESET_GPU met the armed device holder. ``gpu_reset_commit_attempt``:
# the reset was committed and the node agent's nvidia-smi refused it.
# ``gpu_client_quiesce_attempt``: the step's own client barrier
# (``adapters/node_action/barriers.py``) re-verified the device before
# committing and refused to reset a GPU with a live client -- the agent never
# ran the reset. Both are the "clients are still active" failure the case arms
# for; the product moved from the first to the second on 2026-09-08 (attempt 7).
RESET_FAILURE_DETAIL_KEYS = frozenset(
    {"gpu_reset_commit_attempt", "gpu_client_quiesce_attempt"}
)


def reset_reached_commit(workflow: dict[str, Any], *, fault_node: str) -> bool:
    """True when node-b's failed RESET_GPU got as far as committing the reset.

    Decides whether the host ledger must show a RESET_GPU row: a barrier
    refusal leaves none, by design.
    """

    steps = workflow.get("official_steps") or []
    executions = workflow.get("step_executions") or []
    for item in _branch_executions(steps, executions, fault_node):
        if item.get("operation") == "RESET_GPU" and item.get("status") == "FAILED":
            return "gpu_reset_commit_attempt" in (item.get("details") or {})
    return True


def workflow_errors(
    workflow: dict[str, Any],
    incident: dict[str, Any],
    *,
    fault_node: str,
    sibling_node: str,
    failure_reason: str | None,
    reboot_outcome: str = "confirmed-failure",
) -> list[str]:
    errors: list[str] = []
    if reboot_outcome not in {"confirmed-failure", "unknown"}:
        return ["DESTR-014 reboot outcome scenario is unsupported"]
    unknown = reboot_outcome == "unknown"
    status = "BLOCKED" if unknown else "FAILED"
    if workflow.get("status") != status:
        errors.append(f"workflow status is not {status}")
    if unknown and workflow.get("blocked_kind") != "NEEDS_OPERATOR":
        errors.append("unknown reboot did not retain NEEDS_OPERATOR ownership")
    expected_states = {"ESCALATED", "QUARANTINED"} if unknown else {"QUARANTINED"}
    if incident.get("state") not in expected_states:
        errors.append(f"incident state is not one of {sorted(expected_states)}")
    if not workflow.get("dag_enabled"):
        errors.append("workflow is not dag_enabled")
    counts = workflow.get("branch_escalation_counts") or {}
    expected_counts = {fault_node: 1} if unknown else {fault_node: 1, sibling_node: 1}
    if counts != expected_counts:
        errors.append(f"branch_escalation_counts is not {expected_counts}: {counts}")
    exhausted = workflow.get("exhausted_branch_ids") or []
    # Step 8 asks for "node-c's branch", however the DAG named it: the sibling
    # is ``branch:initial`` when its finding opened the incident (the runner
    # injects it first; attempt 10, 2026-09-19) and ``branch:<node>`` when it
    # joined later. Resolve every exhausted id through the steps it owns.
    steps = workflow.get("official_steps") or []
    branch_nodes: dict[str, set[str]] = {}
    for step in steps:
        branch = step.get("branch_id")
        if branch:
            branch_nodes.setdefault(str(branch), set()).update(
                step.get("branch_node_ids") or step.get("node_ids") or []
            )
    exhausted_nodes: set[str] = set()
    for branch in exhausted:
        exhausted_nodes |= branch_nodes.get(str(branch), set())
    if len(exhausted) != 1 or exhausted_nodes != {sibling_node}:
        errors.append(
            "exhausted_branch_ids is not exactly the sibling's branch: "
            f"{exhausted} owned by {sorted(exhausted_nodes)}"
        )
    if not unknown and not str(failure_reason or "").startswith(EXHAUSTION_PREFIX):
        errors.append(
            f"failure reason does not start with {EXHAUSTION_PREFIX!r}: "
            f"{failure_reason!r}"
        )

    executions = workflow.get("step_executions") or []
    if any(item.get("operation") == "RESTART_WORKLOAD" for item in executions):
        errors.append("RESTART_WORKLOAD has an execution record; the join ran")

    resolved = set(workflow.get("completed_step_indexes") or []) | set(
        workflow.get("superseded_step_indexes") or []
    )
    for index, step in enumerate(steps):
        if step.get("branch_id") == "join":
            continue
        if index not in resolved:
            errors.append(
                f"non-join step {index}/{step.get('operation')} is not resolved"
            )

    fault_execs = _branch_executions(steps, executions, fault_node)
    sibling_execs = _branch_executions(steps, executions, sibling_node)

    reset = next(
        (
            item
            for item in fault_execs
            if item.get("operation") == "RESET_GPU" and item.get("status") == "FAILED"
        ),
        None,
    )
    if reset is None:
        errors.append(f"{fault_node} has no FAILED RESET_GPU execution")
    else:
        if "clients are still active" not in str(reset.get("error") or ""):
            errors.append(
                f"{fault_node} RESET_GPU did not fail with 'clients are still active'"
            )
        if not (RESET_FAILURE_DETAIL_KEYS & set(reset.get("details") or {})):
            errors.append(
                f"{fault_node} RESET_GPU failure carries none of "
                f"{sorted(RESET_FAILURE_DETAIL_KEYS)}"
            )

    if _first(fault_execs, "RESTART_NODE", "SUCCEEDED") is None:
        errors.append(f"{fault_node} in-place RESTART_NODE did not SUCCEED")
    for operation in (
        "VALIDATE_GPU",
        "VALIDATE_HOST",
        "VALIDATE_FABRIC",
        "RESTORE_SCHEDULING",
    ):
        if _first(fault_execs, operation, "SUCCEEDED") is None:
            errors.append(f"{fault_node} branch did not complete {operation}")

    sibling_reboot = next(
        (
            item
            for item in sibling_execs
            if item.get("operation") == "RESTART_NODE"
            and item.get("status") == "FAILED"
        ),
        None,
    )
    if sibling_reboot is None:
        errors.append(f"{sibling_node} has no FAILED RESTART_NODE execution")
    else:
        details = sibling_reboot.get("details") or {}
        if unknown:
            if not unresolved_details(details):
                errors.append(
                    "unknown reboot lacks physical-outcome uncertainty evidence"
                )
            if "step_waiting_timeout_seconds" not in details:
                errors.append("unknown reboot lacks its bounded waiting receipt")
        elif unresolved_details(details) or (
            details.get("node_action_not_started") is not True
        ):
            errors.append(
                "replacement requires a confirmed no-start reboot failure; "
                "timeout or absent confirmation cannot authorize another action"
            )

    if unknown:
        forbidden = {"REPLACE_NODE", "RESTORE_SCHEDULING", "RESTART_WORKLOAD"}
        if any(
            step.get("operation") == "REPLACE_NODE"
            and _step_node(step) == [sibling_node]
            for step in steps
        ):
            errors.append("unknown reboot incorrectly added a replacement rung")
        if any(item.get("operation") in forbidden for item in sibling_execs):
            errors.append("unknown sibling executed replacement or readmission")
        return errors

    replace_step = next(
        (
            step
            for step in steps
            if step.get("operation") == "REPLACE_NODE"
            and _step_node(step) == [sibling_node]
        ),
        None,
    )
    if replace_step is None:
        errors.append(f"{sibling_node} branch has no REPLACE_NODE step")
    elif (replace_step.get("parameters") or {}).get("replacement_strategy") != (
        "HEALTHY_WARM_SPARE_ONLY"
    ):
        errors.append(f"{sibling_node} REPLACE_NODE is not HEALTHY_WARM_SPARE_ONLY")
    replace_exec = next(
        (
            item
            for item in sibling_execs
            if item.get("operation") == "REPLACE_NODE"
            and item.get("status") == "FAILED"
        ),
        None,
    )
    if replace_exec is None:
        errors.append(f"{sibling_node} has no FAILED REPLACE_NODE execution")
    elif ZERO_SPARE_REPLACE_ERROR not in str(replace_exec.get("error") or ""):
        errors.append(
            f"{sibling_node} REPLACE_NODE did not fail with the zero-spare "
            f"error {ZERO_SPARE_REPLACE_ERROR!r}: {replace_exec.get('error')!r}"
        )
    return errors


def product_hold_reasons(state: dict[str, Any]) -> list[str]:
    """Why the product still owns recovery, spelled out.

    The same complete-inventory rule the CPU restore gate applies
    (``recovery_safety_errors``): a workflow that is not terminal or retains
    NEEDS_OPERATOR, a remote command that is not terminal, or a receipt whose
    physical outcome is unresolved. Empty means the product has let go.
    """
    workflow, commands = state.get("workflow"), state.get("commands")
    if not isinstance(workflow, dict) or not isinstance(commands, list):
        return ["workflow or remote command inventory is incomplete"]
    reasons: list[str] = []
    status, kind = workflow.get("status"), workflow.get("blocked_kind")
    if status not in {"SUCCEEDED", "FAILED", "SUPERSEDED"}:
        reasons.append(f"workflow is {status}" + (f"/{kind}" if kind else ""))
    elif kind == "NEEDS_OPERATOR":
        reasons.append("workflow retains NEEDS_OPERATOR")
    for error in recovery_safety_errors([workflow], commands):
        if error not in reasons:
            reasons.append(error)
    return reasons


def recovery_cleanup_hold(state: dict[str, Any]) -> bool:
    """No cancellation, timeout or incomplete read proves physical quiescence."""
    return bool(product_hold_reasons(state))


def operator_hold_reasons(
    product_reasons: list[str], sibling_proof: dict[str, Any]
) -> list[str]:
    """Everything an operator would still have to confirm before restoration.

    Two sources: the product's own hold (``product_hold_reasons``) and the
    sibling's physical outcome, which the host recovery record proves when
    the node runs a new boot with its Agent restored. Both empty releases the
    hold; a truly unknown outcome (node never back, boot id unchanged, Agent
    neither restored nor restorable) keeps it.
    """
    reasons = [f"product: {reason}" for reason in product_reasons]
    if not sibling_proof.get("proven"):
        gaps = sibling_proof.get("gaps") or ["no host recovery record"]
        reasons.append("sibling reboot is unproven: " + "; ".join(gaps))
    return reasons


def host_errors(
    *,
    fault_baseline: dict[str, Any],
    fault_after: dict[str, Any],
    sibling_baseline: dict[str, Any],
    sibling_after: dict[str, Any],
    holder_status: dict[str, Any],
    sibling_agent_during: dict[str, Any],
    sibling_agent_after: dict[str, Any],
    reset_reached_commit: bool = True,
) -> list[str]:
    errors: list[str] = []
    if fault_after.get("boot_id") == fault_baseline.get("boot_id"):
        errors.append("fault node boot id did not change across its reboot")
    if sibling_after.get("boot_id") == sibling_baseline.get("boot_id"):
        errors.append("sibling node boot id did not change across its reboot")

    baseline_ids = {
        (row.get("command_id"), row.get("operation"))
        for row in fault_baseline.get("ledger") or []
    }
    new_reset_rows = [
        row
        for row in fault_after.get("ledger") or []
        if row.get("operation") == "RESET_GPU"
        and (row.get("command_id"), row.get("operation")) not in baseline_ids
    ]
    # A reset the client barrier refused never reaches the agent, so no ledger
    # row is the expected shape there; only a committed reset owes one.
    if reset_reached_commit and not new_reset_rows:
        errors.append("fault node ledger has no new RESET_GPU row; the reset never ran")
    if any(row.get("state") == "SUCCEEDED" for row in new_reset_rows):
        errors.append("fault node RESET_GPU succeeded; the holder did not break it")

    if not holder_status.get("matched_row") or not holder_status.get("hold_started_at"):
        errors.append("GPU device holder never armed against a VERIFY ledger row")

    if str(sibling_agent_during.get("UnitFileState") or "") != "disabled":
        errors.append("sibling Node Agent was not disabled during the run")
    if (
        str(sibling_agent_after.get("UnitFileState") or "") != "enabled"
        or sibling_agent_after.get("ActiveState") != "active"
    ):
        errors.append("sibling Node Agent was not restored after the run")
    return errors


def provider_reboot_events(provider: str | None) -> frozenset[str]:
    """The CloudTrail event names the product's reboot leaves on ``provider``."""

    return frozenset(reboot_event_names(provider or "hyperpod"))


def provider_forbidden_events(provider: str | None) -> frozenset[str]:
    """The provider mutations that would mean a replacement, not a reboot."""

    if (provider or "hyperpod") == "eks-managed":
        return EC2_REPLACEMENT_MUTATIONS
    return frozenset(FORBIDDEN_EVENTS)


def node_reboot_identities(
    provider_inventory: Mapping[str, Any] | None,
    nodes: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, dict[str, str | None]]:
    """Per node name, the identities a provider reboot request can carry.

    ``instance_id`` is read from the Node's ``providerID`` (the warm-spare node
    view); ``node_logical_id`` is the provider inventory row of that instance
    (the HyperPod logical id a ``BatchRebootClusterNodes`` names; on
    eks-managed the row's logical id is the node name and EC2
    ``RebootInstances`` names the instance). Either may be ``None``; the
    verdict then refuses to attribute a reboot to that node.
    """

    rows = (provider_inventory or {}).get("nodes") or []
    by_instance = {
        str(row.get("instance_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("instance_id")
    }
    identities: dict[str, dict[str, str | None]] = {}
    for name, view in nodes.items():
        parsed = parse_provider_id((view or {}).get("provider_id"))
        instance_id = parsed[1] if parsed else None
        row = by_instance.get(instance_id) if instance_id else None
        logical = (row or {}).get("node_logical_id")
        identities[name] = {
            "instance_id": instance_id,
            "node_logical_id": str(logical) if logical else None,
        }
    return identities


def _names_node(event: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    """True when the reboot request is the singleton naming this node."""

    instance_id = identity.get("instance_id")
    logical_id = identity.get("node_logical_id")
    requested_instances = event.get("request_node_ids")
    requested_logical = event.get("request_node_logical_ids")
    return bool(
        (instance_id and requested_instances == [instance_id])
        or (logical_id and requested_logical == [logical_id])
    )


def cloudtrail_errors(
    events: list[dict[str, Any]],
    fault_node: str,
    sibling_node: str,
    *,
    provider: str | None = None,
    identities: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """One accepted provider reboot per node, in the provider's own CloudTrail
    vocabulary and source, and no replacement.

    ``identities`` is ``node_reboot_identities``: a reboot is attributed to a
    node by the request's node logical id (HyperPod) or EC2 instance id
    (eks-managed), so two reboots of one node, a batch naming both, or a reboot
    of a node outside the case all fail. Calls AWS refused (an ``errorCode``:
    Karpenter's ``RunInstances`` dry-runs on plain EKS) are set aside and named
    in the count, never counted as a reboot or a replacement.
    """

    provider_name = provider or "hyperpod"
    names = provider_reboot_events(provider_name)
    source = provider_event_source(provider_name)
    accepted, refused = partition_provider_events(events)
    mutations = [item for item in accepted if isinstance(item, Mapping)]
    reboots = [item for item in mutations if item.get("event_name") in names]
    per_node: dict[str, list[Mapping[str, Any]]] = {fault_node: [], sibling_node: []}
    stray = 0
    for item in reboots:
        targets = [
            node
            for node in (fault_node, sibling_node)
            if _names_node(item, identities.get(node) or {})
        ]
        if item.get("event_source") == source and len(targets) == 1:
            per_node[targets[0]].append(item)
        else:
            stray += 1
    errors: list[str] = []
    aside = f"{len(refused)} refused provider calls set aside"
    for node in (fault_node, sibling_node):
        identity = identities.get(node) or {}
        if not identity.get("instance_id") and not identity.get("node_logical_id"):
            errors.append(
                f"{node} has no provider identity (providerID or inventory row); "
                "its reboot cannot be attributed"
            )
        elif len(per_node[node]) != 1:
            errors.append(
                f"CloudTrail does not contain exactly one accepted {provider_name} "
                f"reboot {sorted(names)} for {node}: got {len(per_node[node])} "
                f"({aside})"
            )
    if stray:
        errors.append(
            f"CloudTrail contains {stray} accepted reboot call(s) in {source} "
            "not attributable to exactly one case node"
        )
    forbidden = provider_forbidden_events(provider_name)
    for item in mutations:
        if item.get("event_name") in forbidden:
            errors.append(
                f"CloudTrail contains a forbidden mutation: {item['event_name']}"
            )
    return errors


def schedulability_errors(
    snapshots: dict[str, dict[str, Any]],
    *,
    fault_node: str,
    sibling_node: str,
    incident_id: str,
) -> list[str]:
    """``incident_id`` is the incident that owns the sibling's quarantine: the
    support-after escalation when the exhaustion opened one (its QUARANTINE
    writes the taint), else the case incident."""

    errors: list[str] = []
    fault = snapshots.get(fault_node) or {}
    sibling = snapshots.get(sibling_node) or {}
    if fault.get("unschedulable"):
        errors.append(f"fault node {fault_node} is still cordoned")
    if any(item.get("key") == QUARANTINE_TAINT for item in fault.get("taints") or []):
        errors.append(f"fault node {fault_node} still carries the quarantine taint")
    if not sibling.get("unschedulable"):
        errors.append(f"sibling node {sibling_node} is not cordoned")
    taint = next(
        (
            item
            for item in sibling.get("taints") or []
            if item.get("key") == QUARANTINE_TAINT
        ),
        None,
    )
    if taint is None:
        errors.append(f"sibling node {sibling_node} is missing the quarantine taint")
    elif taint.get("value") != quarantine_taint_value(incident_id):
        errors.append(
            f"sibling node {sibling_node} quarantine taint value is not the "
            "incident digest"
        )
    return errors


def injection_errors(
    fault_state: dict[str, Any],
    sibling_state: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    fault_workflow = fault_state.get("workflow") or {}
    sibling_workflow = sibling_state.get("workflow") or {}
    request = fault_workflow.get("request_id")
    if not request or request != sibling_workflow.get("request_id"):
        errors.append("the two XID injections did not join one workflow_request_id")
    if not (fault_workflow.get("dag_enabled") and sibling_workflow.get("dag_enabled")):
        errors.append("the joined workflow is not dag_enabled")
    return errors


def injection_window_errors(
    sibling_state: dict[str, Any],
    fault_state: dict[str, Any],
    *,
    sibling_node: str,
    fault_node: str,
    aggregation_window_seconds: int,
) -> list[str]:
    """Step 4, judged on the stored events' own ``observed_at``: node-c's XID
    79 first, node-b's XID 46 inside the same aggregation window. A pair that
    misses the window was never a candidate for the in-window merge, so the
    join verdicts alone would read like a product defect (an eks-managed run
    wrote the two 9 s apart and judged 28 consequences of that)."""

    observed = event_observed_at({sibling_node: sibling_state, fault_node: fault_state})
    errors = spread_errors(
        observed, aggregation_window_seconds=aggregation_window_seconds
    )
    if not errors and datetime.fromisoformat(
        observed[fault_node].replace("Z", "+00:00")
    ) <= datetime.fromisoformat(observed[sibling_node].replace("Z", "+00:00")):
        errors.append(
            "node-b's XID 46 was observed before node-c's XID 79; the "
            "injection order was not kept"
        )
    return errors


def follow_up_errors(
    follow_up: dict[str, Any],
    *,
    fault_node: str,
    sibling_node: str,
) -> list[str]:
    errors: list[str] = []
    incident = follow_up.get("incident")
    workflow = follow_up.get("workflow")
    if not incident or not workflow:
        errors.append("hardware escalation follow-up is missing for the incident")
        return errors
    operations = [
        step.get("operation") for step in workflow.get("official_steps") or []
    ]
    if "ESCALATE_SUPPORT" not in operations:
        errors.append("follow-up workflow has no ESCALATE_SUPPORT step")
    nodes = incident.get("node_ids") or []
    if nodes != [sibling_node]:
        errors.append(
            f"follow-up incident is not scoped to {sibling_node} alone: {nodes}"
        )
    return errors


def workload_errors(
    *,
    pods: list[dict[str, Any]],
    restart_budget: dict[str, Any],
    source_uids: set[str],
) -> list[str]:
    errors: list[str] = []
    live = [item for item in pods if item.get("phase") not in {"Succeeded", "Failed"}]
    if live:
        errors.append(f"the job still has {len(live)} Pod(s) after the failed workflow")
    if (restart_budget or {}).get("restart_count") != 0:
        errors.append("restart budget advanced; the job was restarted")
    return errors


def step_transitions(
    previous: dict[str, str],
    executions: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    current = dict(previous)
    changes: list[dict[str, Any]] = []
    for item in executions:
        key = f"{item.get('step_index')}:{item.get('operation')}"
        status = str(item.get("status") or "")
        if current.get(key) != status:
            changes.append(
                {
                    "step_index": item.get("step_index"),
                    "operation": item.get("operation"),
                    "from": current.get(key),
                    "to": status,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            current[key] = status
    return current, changes


def estimated_duration_seconds(
    *,
    verify_max_attempts: int,
    poll_interval_seconds: float,
    managed_recovery_timeout_seconds: int,
) -> int:
    return int(
        round(verify_max_attempts * poll_interval_seconds)
        + managed_recovery_timeout_seconds
        + 2 * REBOOT_ALLOWANCE_SECONDS
        + VALIDATION_ALLOWANCE_SECONDS
        + CONTAINMENT_ALLOWANCE_SECONDS
    )


def lifetime_errors(
    *,
    estimated_seconds: int,
    lifetime_seconds: int | None,
) -> list[str]:
    if lifetime_seconds is None:
        return ["workflow lifetime is unknown; refusing to run fail-closed"]
    if estimated_seconds >= lifetime_seconds:
        return [
            f"estimated duration {estimated_seconds}s meets or exceeds the job "
            f"workflow lifetime {lifetime_seconds}s"
        ]
    return []


def budget_headroom_errors(budget: dict[str, Any]) -> list[str]:
    if not budget.get("readable"):
        return ["budget_headroom_unknown: remediation budget could not be read"]
    scopes = budget.get("scopes") or {}
    if not scopes:
        return ["budget_headroom_unknown: no remediation budget scopes were computed"]
    errors: list[str] = []
    for name, scope in scopes.items():
        if int(scope.get("active") or 0) >= int(scope.get("limit") or 0):
            errors.append(f"remediation budget scope is full: {name}")
    return errors
