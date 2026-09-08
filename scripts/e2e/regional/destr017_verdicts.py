"""Pure verdict functions and constants of GF-REGIONAL-DESTR-017.

Split out of ``run_destr017_out_of_band_reboot_fence.py`` so the runner stays a
driver: everything here is unit-tested against synthetic control-plane, node and
CloudTrail snapshots and touches no cluster.

The case proves the *generation fence*: a node that is rebooted by something
outside the control plane while a RESET_GPU workflow waits on it comes back with
a new boot id, the Node Agent re-registers with a higher generation, and every
in-flight maintenance command addressed to the old generation is refused. The
new boot must therefore never execute a GPU reset, and nothing may be recorded
as SUCCEEDED on its behalf.

The fail-closed literals below are the four the code can produce for a waiting
maintenance step, in the order the barrier checks them
(``src/gpu_fault/adapters/node_action/barriers.py``,
``src/gpu_fault/node_agent/operations/clients.py``, and the executor's per-step
waiting cap). Which one fires depends on how the reboot raced the pinned
maintenance window, so the case records the variant instead of pinning it -- but
whichever one ends the waiting step, the restore compensation that follows is
window-exempt and generation-checked, so the *generation* fence has to appear in
the terminal record. That is the assertion; the variant is evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# The official steps a single idle node's RESET_GPU workflow compiles to.
OFFICIAL_OPERATIONS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
)
# The step the device holder keeps WAITING, and the step the executor resumes as
# failure compensation once it fails.
FENCE_STEP_OPERATION = "VERIFY_NO_GPU_CLIENTS"
COMPENSATION_OPERATION = "RESTORE_GPU_SERVICES"
# Operations that must never appear as an execution: the whole point is that the
# reset never happens, and no hardware rung may be climbed inside this workflow.
FORBIDDEN_EXECUTIONS = frozenset(
    {
        "RESET_GPU",
        "RESET_ALL_GPUS_NVSWITCHES",
        "RESTART_NODE",
        "REPLACE_NODE",
        "RESTART_VM",
    }
)
RESET_OPERATIONS = frozenset({"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES"})
# Node Agent operations the node must advertise before the case starts.
AGENT_OPERATIONS = (
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
)
# The runtime profile must hand gpuReset to this owner in OWN mode.
AGENT_OWNER = "gpu-fault-node-agent"

# Fail-closed literals. Every one is a substring of a real error the product
# raises; none is reconstructed from a format string here.
GENERATION_FENCE_LITERAL = "agent generation changed from"
MAINTENANCE_FENCE_LITERAL = "maintenance agent fence failed for"
WINDOW_EXPIRY_LITERAL = "quiesce maintenance window expired at"
DEVICE_CLIENTS_LITERAL = "GPU device clients are still active"
WAITING_CAP_LITERAL = "stayed non-terminal for"

FENCE_AGENT_GENERATION = "agent_generation"
FENCE_WINDOW_EXPIRED = "maintenance_window_expired"
FENCE_DEVICE_CLIENTS = "device_clients_active"
FENCE_WAITING_CAP = "step_waiting_cap"
FENCE_TRANSPORT = "agent_unreachable"
FENCE_UNKNOWN = "unknown"
# Every variant that is a fail-closed refusal. ``unknown`` is not one.
ACCEPTED_FENCES = (
    FENCE_AGENT_GENERATION,
    FENCE_WINDOW_EXPIRED,
    FENCE_DEVICE_CLIENTS,
    FENCE_WAITING_CAP,
    FENCE_TRANSPORT,
)

# The escalation the failed containment step must open, and the rungs it must
# not: an OS reboot the control plane did not order may not be answered with
# another reboot or a replacement.
SUPPORT_EVENT_PREFIX = "support-after-"
FORBIDDEN_ESCALATION_PREFIXES = ("reboot-after-", "replace-after-", "drain-after-")
SUPPORT_OPERATIONS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "ESCALATE_SUPPORT",
)

# The shipped plan mode of ``gpu-fault-admin workflow-reconcile``.
RETIRED_GENERATION_PLAN_MODE = "retired-generation-plan"
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
EXPECTED_XID = 46

# Wall-clock allowances (seconds) for the lifetime arithmetic. Deliberately
# generous so a passing estimate is a real safety margin: containment, the
# waiting window the holder buys, the reboot and re-registration, the terminal
# compensation, and the support successor that follows it.
CONTAINMENT_ALLOWANCE_SECONDS = 180
WAITING_ALLOWANCE_SECONDS = 660
REBOOT_ALLOWANCE_SECONDS = 420
SUCCESSOR_ALLOWANCE_SECONDS = 300


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _executions_of(
    executions: list[dict[str, Any]],
    operation: str,
) -> list[dict[str, Any]]:
    return [item for item in executions if item.get("operation") == operation]


def _error_text(item: dict[str, Any]) -> str:
    return str(item.get("error") or "")


def _added_ledger_rows(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline = {
        (row.get("command_id"), row.get("operation"))
        for row in before.get("ledger") or []
    }
    return [
        row
        for row in after.get("ledger") or []
        if (row.get("command_id"), row.get("operation")) not in baseline
    ]


def fence_variant(error: str) -> str:
    """Which fail-closed refusal ``error`` is.

    Ordered the way the barrier and the agent produce them: the generation
    check is the most specific, so it is tested before the maintenance-fence
    wrapper it travels inside.
    """

    text = str(error or "")
    if not text:
        return FENCE_UNKNOWN
    if GENERATION_FENCE_LITERAL in text:
        return FENCE_AGENT_GENERATION
    if WINDOW_EXPIRY_LITERAL in text:
        return FENCE_WINDOW_EXPIRED
    if DEVICE_CLIENTS_LITERAL in text:
        return FENCE_DEVICE_CLIENTS
    if WAITING_CAP_LITERAL in text:
        return FENCE_WAITING_CAP
    if MAINTENANCE_FENCE_LITERAL in text:
        # A maintenance fence that is not a generation change: the endpoint
        # could not be resolved at all, i.e. the Agent was not back yet.
        return FENCE_TRANSPORT
    return FENCE_UNKNOWN


def fence_evidence(workflow: dict[str, Any]) -> dict[str, Any]:
    """The variant of every terminal error the case reads, digest-free.

    Recorded whatever the verdict, so a run that fails on an unexpected variant
    still says which one it saw.
    """

    executions = workflow.get("step_executions") or []
    verify = _executions_of(executions, FENCE_STEP_OPERATION)
    compensation = _executions_of(executions, COMPENSATION_OPERATION)
    return {
        "workflow_status": workflow.get("status"),
        "workflow_error_variant": fence_variant(str(workflow.get("error") or "")),
        "verify_attempts": len(verify),
        "verify_statuses": [item.get("status") for item in verify],
        "verify_variants": [fence_variant(_error_text(item)) for item in verify],
        "compensation_statuses": [item.get("status") for item in compensation],
        "compensation_variants": [
            fence_variant(_error_text(item)) for item in compensation
        ],
        "executed_operations": sorted(
            {str(item.get("operation")) for item in executions}
        ),
        "completed_operations": workflow.get("completed_operations") or [],
    }


# --------------------------------------------------------------------------- #
# Control-plane verdicts
# --------------------------------------------------------------------------- #
def workflow_errors(
    workflow: dict[str, Any],
    incident: dict[str, Any],
    *,
    node: str,
) -> list[str]:
    """The terminal contract of the fenced RESET_GPU workflow.

    Not "it failed": it must fail *closed*, at the waiting maintenance step, with
    the generation fence recorded somewhere in the terminal record, having never
    executed a reset and never marked an old-generation command SUCCEEDED.
    """

    errors: list[str] = []
    status = workflow.get("status")
    if status != "FAILED":
        errors.append(
            f"workflow status is not FAILED: {status}; an out-of-band reboot "
            "inside a maintenance window must fail the workflow closed"
        )
    if incident.get("state") != "QUARANTINED":
        errors.append(
            f"incident state is not QUARANTINED: {incident.get('state')}; the "
            "node was cordoned and never released"
        )

    steps = workflow.get("official_steps") or []
    operations = [str(step.get("operation")) for step in steps]
    if operations != list(OFFICIAL_OPERATIONS):
        errors.append(
            f"workflow did not compile the idle-node reset plan: {operations}"
        )
    for step in steps:
        nodes = list(step.get("node_ids") or [])
        if nodes not in ([], [node]):
            errors.append(f"a step addresses another node: {step.get('operation')}")

    executions = workflow.get("step_executions") or []
    executed = {str(item.get("operation")) for item in executions}
    forbidden = sorted(FORBIDDEN_EXECUTIONS.intersection(executed))
    if forbidden:
        errors.append(
            f"the fenced workflow executed a node mutation it must never reach: "
            f"{forbidden}"
        )
    if RESET_OPERATIONS.intersection(workflow.get("completed_operations") or []):
        errors.append("the workflow recorded a completed GPU reset")

    verify = _executions_of(executions, FENCE_STEP_OPERATION)
    if not verify:
        errors.append(
            f"{FENCE_STEP_OPERATION} never executed; the case never reached the "
            "waiting step the reboot has to interrupt"
        )
    else:
        if any(item.get("status") == "SUCCEEDED" for item in verify):
            errors.append(
                f"{FENCE_STEP_OPERATION} succeeded; the device holder did not "
                "hold the step waiting and the reset was not fenced"
            )
        last_verify = verify[-1]
        if last_verify.get("status") != "FAILED":
            errors.append(
                f"{FENCE_STEP_OPERATION} did not end FAILED: "
                f"{last_verify.get('status')}"
            )
        variant = fence_variant(_error_text(last_verify))
        if variant not in ACCEPTED_FENCES:
            errors.append(
                f"{FENCE_STEP_OPERATION} did not fail closed on a known fence: "
                f"{_error_text(last_verify)!r}"
            )

    compensation = _executions_of(executions, COMPENSATION_OPERATION)
    if not compensation:
        errors.append(
            f"{COMPENSATION_OPERATION} compensation never ran after the waiting "
            "step failed; a succeeded quiesce was left unresolved"
        )

    # The proof of the case: the generation fence has to be in the record. The
    # waiting step may have failed on any of the accepted variants, but the
    # window-exempt compensation that follows it is still generation-checked.
    fenced = [
        item
        for item in executions
        if fence_variant(_error_text(item)) == FENCE_AGENT_GENERATION
    ]
    workflow_fenced = (
        fence_variant(str(workflow.get("error") or "")) == FENCE_AGENT_GENERATION
    )
    if not fenced and not workflow_fenced:
        errors.append(
            "no step and not the terminal error carries "
            f"{GENERATION_FENCE_LITERAL!r}: the new boot's Agent was never "
            "fenced out, so this run does not prove the generation fence"
        )
    for item in fenced:
        if node not in _error_text(item):
            errors.append(
                f"a generation fence names another node: {_error_text(item)!r}"
            )

    if workflow.get("superseded_step_indexes"):
        errors.append(
            "the workflow has superseded steps; a retired-generation sweep took "
            f"it over: {workflow.get('superseded_step_indexes')}"
        )
    if workflow.get("blocked_kind") or workflow.get("blocked_reasons"):
        errors.append(
            f"the workflow is BLOCKED rather than FAILED: "
            f"{workflow.get('blocked_kind')} {workflow.get('blocked_reasons')}"
        )
    return errors


def command_errors(
    commands: list[dict[str, Any]],
    *,
    node: str,
) -> list[str]:
    """No remote command may be a reset, and none may be marked SUCCEEDED.

    ``commands`` are the remote command records of the fenced workflow. The old
    generation's waiting command is the one the reboot orphans: it must end
    refused or expired, never completed.
    """

    errors: list[str] = []
    for command in commands:
        operation = str((command.get("step") or {}).get("operation") or "")
        if operation in RESET_OPERATIONS:
            errors.append(
                f"a GPU reset command was dispatched to {node}: "
                f"{command.get('command_id')}"
            )
        if operation == FENCE_STEP_OPERATION and command.get("status") == "SUCCEEDED":
            errors.append(
                f"an old-generation {operation} command is recorded SUCCEEDED: "
                f"{command.get('command_id')}"
            )
    return errors


def successor_errors(
    successor_workflow: dict[str, Any],
    successor_incident: dict[str, Any],
    *,
    node: str,
    predecessor_request_id: str,
    forbidden_escalations: dict[str, Any],
    compensation_failed: bool,
) -> list[str]:
    """The escalation the fenced workflow is allowed to open.

    A failed containment/release step classifies to ``containment_or_release``,
    whose rung is an operator, not hardware: exactly one support escalation, and
    never a reboot or replacement of a node the product did not reboot.
    """

    errors: list[str] = []
    opened = [name for name, value in forbidden_escalations.items() if value]
    if opened:
        errors.append(
            f"a hardware escalation was opened for an out-of-band reboot: {opened}"
        )
    if not compensation_failed:
        if successor_workflow or successor_incident:
            errors.append(
                "a support escalation exists although no containment step failed"
            )
        return errors
    if not successor_workflow:
        errors.append(
            "the failed restore compensation opened no support escalation; the "
            "node was left unrestored with nobody accountable"
        )
        return errors
    expected_request_id = f"workflow-{SUPPORT_EVENT_PREFIX}{predecessor_request_id}"
    if successor_workflow.get("request_id") != expected_request_id:
        errors.append(
            f"the successor is not this workflow's support escalation: "
            f"{successor_workflow.get('request_id')}"
        )
    operations = [
        str(step.get("operation"))
        for step in successor_workflow.get("official_steps") or []
    ]
    if operations != list(SUPPORT_OPERATIONS):
        errors.append(f"the support escalation is not the support plan: {operations}")
    if successor_workflow.get("official_action") != "ESCALATE_OPERATOR":
        errors.append(
            f"the successor's action is not ESCALATE_OPERATOR: "
            f"{successor_workflow.get('official_action')}"
        )
    if node not in (successor_incident.get("node_ids") or []):
        errors.append(
            f"the support escalation does not name {node}: "
            f"{successor_incident.get('node_ids')}"
        )
    if successor_workflow.get("status") not in {"SUCCEEDED", "PENDING", "RUNNING"}:
        errors.append(
            f"the support escalation did not run: {successor_workflow.get('status')}"
        )
    return errors


def _incarnation(agent: dict[str, Any]) -> str:
    """The live AgentRecord calls it ``agent_incarnation_id``."""

    return str(agent.get("agent_incarnation_id") or agent.get("incarnation_id") or "")


def agent_errors(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    node: str,
) -> list[str]:
    """Exactly one generation bump, and the previous incarnation retired.

    ``AgentRecord.generation`` is what the maintenance fence compares, so a run
    where it did not advance by one either never rebooted or rebooted twice.
    """

    errors: list[str] = []
    if before.get("node_id") != node or after.get("node_id") != node:
        return [f"agent records are not both for {node}"]
    first = before.get("generation")
    second = after.get("generation")
    if not isinstance(first, int) or not isinstance(second, int):
        return [f"agent generation is not an integer: {first!r} -> {second!r}"]
    if second != first + 1:
        errors.append(
            f"agent generation did not advance exactly once: {first} -> {second}"
        )
    if after.get("lifecycle_state") != "ACTIVE":
        errors.append(
            f"the re-registered agent is not ACTIVE: {after.get('lifecycle_state')}"
        )
    if before.get("boot_id") and after.get("boot_id") == before.get("boot_id"):
        errors.append("the agent re-registered with the same boot id")
    retired_before = list(before.get("retired_incarnation_ids") or [])
    retired_after = list(after.get("retired_incarnation_ids") or [])
    added = [item for item in retired_after if item not in retired_before]
    if len(added) != 1:
        errors.append(
            f"the reboot did not retire exactly one incarnation: {len(added)}"
        )
    elif _incarnation(before) and added[0] != _incarnation(before):
        errors.append("the retired incarnation is not the pre-reboot one")
    return errors


def boot_errors(
    *,
    node: str,
    baseline_boot_id: str,
    final_boot_id: str,
    host_boot_ids: list[str],
    reboot_status: dict[str, Any],
) -> list[str]:
    """Exactly one boot id change, witnessed twice.

    ``host_boot_ids`` is the probe's durable observation history (marker file);
    the Kubernetes node status is the independent second witness.
    """

    errors: list[str] = []
    if not baseline_boot_id or not final_boot_id:
        return [f"{node} boot ids were not captured on both sides of the reboot"]
    if final_boot_id == baseline_boot_id:
        errors.append(
            f"{node} boot id did not change; the out-of-band reboot never happened"
        )
    unique = list(dict.fromkeys(str(item) for item in host_boot_ids if item))
    if len(unique) != 2:
        errors.append(
            f"{node} host probe observed {len(unique)} boots, not exactly two: "
            f"{len(host_boot_ids)} samples"
        )
    elif unique[0] != baseline_boot_id or unique[-1] != final_boot_id:
        errors.append(
            f"{node} host boot history does not match the Kubernetes node status"
        )
    if not reboot_status.get("fired"):
        errors.append(f"{node} reboot marker does not record a fired reboot")
    if reboot_status.get("boot_id_before_reboot") != baseline_boot_id:
        errors.append(
            f"{node} reboot marker recorded another pre-reboot boot id than the "
            "one the case baselined"
        )
    return errors


# --------------------------------------------------------------------------- #
# Data-plane verdicts
# --------------------------------------------------------------------------- #
def ledger_errors(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    node: str,
) -> list[str]:
    """The Node Agent ledger survives the reboot; read the new boot's truth.

    Assertions 1 and 2 of the case live here: no reset was ever executed, and no
    command of the retired generation is recorded as a success.
    """

    errors: list[str] = []
    added = _added_ledger_rows(before, after)
    resets = [row for row in added if str(row.get("operation")) in RESET_OPERATIONS]
    if resets:
        errors.append(
            f"{node} Node Agent ledger recorded a GPU reset: "
            f"{[(row.get('operation'), row.get('state')) for row in resets]}"
        )
    verify = [row for row in added if str(row.get("operation")) == FENCE_STEP_OPERATION]
    if not verify:
        errors.append(
            f"{node} Node Agent ledger has no {FENCE_STEP_OPERATION} row; the "
            "waiting step never reached the node"
        )
    if any(row.get("state") == "SUCCEEDED" for row in verify):
        errors.append(
            f"{node} Node Agent ledger completed {FENCE_STEP_OPERATION} "
            "successfully; the device holder did not hold"
        )
    quiesce = [
        row
        for row in added
        if str(row.get("operation")) == "QUIESCE_GPU_SERVICES"
        and row.get("state") == "SUCCEEDED"
    ]
    if not quiesce:
        errors.append(
            f"{node} Node Agent ledger has no successful QUIESCE_GPU_SERVICES; "
            "the maintenance window was never pinned"
        )
    unexpected = sorted(
        {
            str(row.get("operation"))
            for row in added
            if row.get("state") == "SUCCEEDED"
            and str(row.get("operation")) != "QUIESCE_GPU_SERVICES"
        }
    )
    if unexpected:
        errors.append(
            f"{node} Node Agent ledger completed an operation the fence should "
            f"have refused: {unexpected}"
        )
    if len({row.get("command_id") for row in added}) != len(added):
        errors.append(f"{node} Node Agent ledger contains duplicate command IDs")
    return errors


def boot_reconcile_errors(after: dict[str, Any], *, node: str) -> list[str]:
    """The Node Agent restores a quiesce the reboot orphaned when it starts.

    The fail-safe timer is transient and dies with the boot; the state file
    does not. ``GpuServiceQuiesceManager.reconcile_after_boot`` restores and
    removes it on the first start of the new boot, so by the time the agent is
    ACTIVE again the post-reboot snapshot must show no quiesce residue -- before
    the runner's own restore-quiesce cleanup ever runs.
    """

    residue = [item.get("name") for item in after.get("quiesce_states") or []]
    if residue:
        return [
            f"{node} Node Agent did not reconcile the quiesce state the reboot "
            f"orphaned: {residue}"
        ]
    return []


def reset_journal_errors(
    after: dict[str, Any],
    *,
    node: str,
) -> list[str]:
    """The kernel is the last word on whether a reset touched the GPU."""

    journal = after.get("kernel_reset_journal") or {}
    if "target_reset_count" not in journal:
        return [f"{node} kernel reset journal was not captured"]
    count = int(journal.get("target_reset_count") or 0)
    if count:
        return [
            f"{node} kernel journal shows {count} reset entries for the target "
            "GPU on the new boot; the fence did not hold"
        ]
    return []


def host_final_errors(
    before: dict[str, Any],
    final: dict[str, Any],
    *,
    node: str,
    expected_gpu_count: int,
) -> list[str]:
    """The node after cleanup: same hardware, services up, no residue."""

    errors: list[str] = []
    if len(final.get("gpu_inventory") or []) != expected_gpu_count:
        errors.append(
            f"{node} GPU inventory is not {expected_gpu_count} after the reboot"
        )
    if final.get("quiesce_states"):
        errors.append(
            f"{node} GPU quiesce state file remains: "
            f"{[item.get('name') for item in final['quiesce_states']]}"
        )
    if final.get("gpu_fault_timers") != before.get("gpu_fault_timers"):
        errors.append(
            f"{node} gpu-fault timer inventory did not return to baseline: "
            f"{final.get('gpu_fault_timers')}"
        )
    for unit, state in (before.get("services") or {}).items():
        if state.get("ActiveState") != "active":
            continue
        current = (final.get("services") or {}).get(unit) or {}
        if current.get("ActiveState") != "active":
            errors.append(f"{node} service did not return active: {unit}")
    if final.get("compute_clients"):
        errors.append(
            f"{node} still has GPU compute clients after cleanup: "
            f"{final.get('compute_clients')}"
        )
    return errors


def holder_errors(status: dict[str, Any], *, node: str) -> list[str]:
    """The holder had to be alive and matched to *this* drill's quiesce row."""

    errors: list[str] = []
    if status.get("holder_error"):
        errors.append(f"{node} device holder failed: {status.get('holder_error')}")
    if not status.get("matched_row"):
        errors.append(
            f"{node} device holder never matched a ledger row; it did not open "
            "after this drill's quiesce"
        )
    if not status.get("hold_started_at"):
        errors.append(f"{node} device holder never started")
    return errors


def schedulability_errors(snapshot: dict[str, Any], *, node: str) -> list[str]:
    errors: list[str] = []
    if snapshot.get("ready") != "True":
        errors.append(f"{node} is not Ready after the case")
    if snapshot.get("unschedulable"):
        errors.append(f"{node} is still unschedulable after the validated restore")
    taints = [
        taint.get("key")
        for taint in snapshot.get("taints") or []
        if str(taint.get("key") or "").startswith("gpu-fault.io/")
    ]
    if taints:
        errors.append(f"{node} still carries a gpu-fault taint: {taints}")
    if snapshot.get("ownership_annotations"):
        errors.append(
            f"{node} still carries an ownership annotation: "
            f"{sorted(snapshot['ownership_annotations'])}"
        )
    return errors


def cloudtrail_errors(events: list[dict[str, Any]]) -> list[str]:
    """An OS reboot is not a provider action.

    ``BatchRebootClusterNodes`` never appears: nothing asked the provider for
    anything, and the successor escalates to an operator instead of a rung.
    """

    if not events:
        return []
    return [
        "provider mutation appeared although the reboot came from outside the "
        "control plane: "
        + ", ".join(sorted({str(item.get("event_name")) for item in events}))
    ]


def reconcile_plan_errors(
    plan: dict[str, Any],
    *,
    request_ids: set[str],
) -> list[str]:
    """The operator reconcile is recorded in plan mode and never applied.

    Applying a retired-generation revocation is an operator decision, so the
    case runs ``build_retired_generation_plan`` only and records NOT_APPLIED.
    What it asserts is that the fence produced nothing for that plan to revoke:
    the fenced workflow ended terminal by itself, so it must not appear as a
    retired generation somebody has to close by hand.
    """

    errors: list[str] = []
    if plan.get("mode") != RETIRED_GENERATION_PLAN_MODE:
        errors.append(
            f"the workflow reconcile did not run in plan mode: {plan.get('mode')}"
        )
    if plan.get("applied"):
        errors.append("the workflow reconcile was applied; it must stay a plan")
    listed = sorted(
        str(item.get("request_id"))
        for item in plan.get("items") or []
        if str(item.get("request_id")) in request_ids
    )
    if listed:
        errors.append(
            "the retired-generation plan wants to revoke this case's workflows; "
            f"the fence left a wedged record: {listed}"
        )
    return errors


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight_errors(
    *,
    node: str,
    node_snapshot: dict[str, Any],
    agent: dict[str, Any],
    profile: dict[str, Any],
    business_workloads: list[dict[str, str]],
    queue: dict[str, Any],
    remote_commands: dict[str, Any],
    recent_events: list[dict[str, Any]],
    host_snapshot: dict[str, Any],
    reboot_status: dict[str, Any],
    expected_gpu_count: int,
) -> list[str]:
    """Read-only gates. An idle, healthy, unfenced node or no case at all."""

    errors: list[str] = []
    if node_snapshot.get("ready") != "True":
        errors.append(f"{node} is not Ready")
    if node_snapshot.get("unschedulable"):
        errors.append(f"{node} is already unschedulable")
    taints = [
        taint.get("key")
        for taint in node_snapshot.get("taints") or []
        if str(taint.get("key") or "").startswith("gpu-fault.io/")
    ]
    if taints:
        errors.append(f"{node} already carries a gpu-fault taint: {taints}")
    if node_snapshot.get("ownership_annotations"):
        errors.append(f"{node} already carries a gpu-fault ownership annotation")
    if business_workloads:
        errors.append(
            f"{node} runs business workloads: "
            f"{[item.get('name') for item in business_workloads]}"
        )
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append(f"{node} agent is not ACTIVE: {agent.get('lifecycle_state')}")
    # The capability mode lives on the runtime profile, not on the Agent record:
    # the reset the fence protects must be OWNed by the Node Agent.
    reset = next(
        (
            item
            for item in profile.get("capabilities") or []
            if item.get("capability") == "gpuReset"
        ),
        None,
    )
    if reset is None or reset.get("mode") != "OWN" or reset.get("owner") != AGENT_OWNER:
        errors.append(
            f"{node} gpuReset capability is not OWN by the Node Agent: {reset!r}"
        )
    if not isinstance(agent.get("generation"), int):
        errors.append(f"{node} agent has no generation to fence on: {agent!r}")
    advertised = (
        agent.get("allowed_operations") or agent.get("supported_operations") or []
    )
    missing = [
        operation for operation in AGENT_OPERATIONS if operation not in advertised
    ]
    if missing:
        errors.append(f"{node} agent does not advertise {missing}")
    if profile.get("warnings"):
        errors.append(f"{node} runtime profile carries warnings: {profile['warnings']}")
    if int(queue.get("depth") or 0):
        errors.append(f"processor queue is not empty: {queue}")
    for key in ("pending", "leased", "in_progress"):
        if int(remote_commands.get(key) or 0):
            errors.append(f"remote commands are not idle: {remote_commands}")
            break
    if recent_events:
        errors.append(
            f"{node} already has a recent XID event: "
            f"{[item.get('event_id') for item in recent_events]}"
        )
    if len(host_snapshot.get("gpu_inventory") or []) != expected_gpu_count:
        errors.append(
            f"{node} does not report {expected_gpu_count} GPUs: "
            f"{len(host_snapshot.get('gpu_inventory') or [])}"
        )
    if not host_snapshot.get("kmsg_writable"):
        errors.append(f"{node} cannot write /dev/kmsg; the injection would be fake")
    if host_snapshot.get("quiesce_states"):
        errors.append(
            f"{node} still holds a GPU quiesce state file: "
            f"{[item.get('name') for item in host_snapshot['quiesce_states']]}"
        )
    if host_snapshot.get("compute_clients"):
        errors.append(
            f"{node} already has GPU compute clients: "
            f"{host_snapshot.get('compute_clients')}"
        )
    if reboot_status.get("armed") and not reboot_status.get("reboot_cancelled_at"):
        errors.append(
            f"{node} already has an armed reboot timer from an earlier run: "
            f"{reboot_status.get('reboot_unit')}"
        )
    return errors


# --------------------------------------------------------------------------- #
# Timeline and arithmetic
# --------------------------------------------------------------------------- #
def step_transitions(
    previous: dict[str, str],
    executions: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Fold step executions into ``{index/operation: status}``; return the new
    state and only the entries whose status changed since ``previous``."""

    state = dict(previous)
    changes: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for item in executions:
        key = f"{item.get('step_index')}/{item.get('operation')}"
        status = str(item.get("status") or "")
        if state.get(key) == status:
            continue
        state[key] = status
        changes.append(
            {
                "observed_at": now,
                "step": key,
                "status": status,
                "fence_variant": fence_variant(_error_text(item)),
                "started_at": item.get("started_at"),
                "updated_at": item.get("updated_at"),
            }
        )
    return state, changes


def estimated_duration_seconds() -> int:
    return (
        CONTAINMENT_ALLOWANCE_SECONDS
        + WAITING_ALLOWANCE_SECONDS
        + REBOOT_ALLOWANCE_SECONDS
        + SUCCESSOR_ALLOWANCE_SECONDS
    )


def lifetime_errors(
    *,
    estimated_seconds: int,
    lifetime_seconds: int | None,
) -> list[str]:
    """The successor shares the predecessor's lifetime deadline (F-N1), so the
    whole chain has to fit one node workflow lifetime or the escalation dies of
    a deadline instead of the fence."""

    if lifetime_seconds is None:
        return ["node workflow lifetime is unknown; cannot bound the case duration"]
    if estimated_seconds >= lifetime_seconds:
        return [
            f"estimated duration {estimated_seconds}s does not fit the "
            f"{lifetime_seconds}s node workflow lifetime"
        ]
    return []


def reboot_window_errors(
    *,
    delay_seconds: int,
    window_remaining_seconds: float | None,
    step_waiting_limit_seconds: int | None,
) -> list[str]:
    """The reboot has to land while the step is still waiting.

    A reboot armed after the pinned maintenance window has already expired, or
    after the per-step waiting cap would have failed the step anyway, proves the
    window fence rather than the generation fence.
    """

    errors: list[str] = []
    if window_remaining_seconds is None:
        return ["the pinned maintenance window is unknown; cannot place the reboot"]
    if delay_seconds >= window_remaining_seconds:
        errors.append(
            f"a reboot armed {delay_seconds}s out fires after the maintenance "
            f"window ends in {window_remaining_seconds:.0f}s"
        )
    if (
        step_waiting_limit_seconds is not None
        and delay_seconds >= step_waiting_limit_seconds
    ):
        errors.append(
            f"a reboot armed {delay_seconds}s out fires after the "
            f"{step_waiting_limit_seconds}s per-step waiting cap"
        )
    return errors
