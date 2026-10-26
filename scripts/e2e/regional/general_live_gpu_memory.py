"""GF-GENERAL-LIVE-019 ECC / row remap / page retirement (live-replay).

Each variant increments or sets one reviewed DCGM memory field on the target
GPU through ``tools/gpu_dcgm_field_replay.py`` (the ``memory`` profile) and
reads the product's own grading (``gpu_metrics`` -> ``gpu_memory_policy``):

* ``corrected_ecc`` / ``aggregate_dbe`` / ``retired_delta`` are WARNING
  RUN_DIAGNOSTICS (DIAGNOSTIC_ONLY): a VALIDATE_GPU that isolates nothing and
  resets nothing, cleared by a fresh zero-delta batch;
* ``volatile_dbe`` is CRITICAL QUARANTINE: the node is isolated (no reset)
  and read back through the product's validated restore. Its chain and
  ``remap_failure``'s are judged on executed steps: detection waits until the
  isolating steps have run or the record has ended (``chain_executed``), and a
  record that planned MARK_UNSCHEDULABLE/QUARANTINE and never executed them --
  the compile-time BLOCKED record live 2026-10-26, superseded 35 ms after
  creation with nothing ever reaching the node -- fails the chain stage;
* ``remap_failure`` is CRITICAL DRAIN: containment first (FREEZE_EVIDENCE,
  MARK_UNSCHEDULABLE, QUARANTINE, the diagnostic bundle), then the NVIDIA RMA
  field diagnostic -- which has no executable owner on this site
  (memoryDiagnostics OBSERVE, b192 / R13h), so the record settles
  BLOCKED/SAFETY_SETTLED, the incident QUARANTINED, the operator is told
  (contained-operator-handoff) and the node stays held; a field diagnosis is
  never read as "hardware already replaced". The variant proves the executed
  containment on the node, then readmits it through the product's validated
  restore once the replay has ended (the documented exit for that shape). A
  record the product escalated instead (pre-b192: ESCALATED, an operator
  closes it) takes the operator-close branch; a record that planned the chain
  and never ran it (live 2026-10-26, compile-time BLOCKED) fails the chain;
* ``pending_supported`` / ``post_reset`` are the authorized service-window
  single-GPU reset (``row_remap_pending`` on a modern GPU); they hold the
  reading until RESET_GPU has run, then serve a clean reading so the post-reset
  VALIDATE_GPU passes. They need ``--approve GPU_RESET``.

``pending_unsupported`` (no legacy-architecture GPU on this site) and
``observe_profile`` (the live Runtime Profile owns ``gpuReset``) are recorded
NOT_RUN with their reason.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    add,
    arm_and_confirm,
    baseline,
    gpu_spec,
    no_hardware,
    plan,
    put,
    recover_counter,
    result,
    run_gpu_survey,
    state_now,
    step,
    utc_now,
    wait_gpu_incident,
)
from scripts.e2e.regional.general_live_gpu_verdicts import (
    GpuChain,
    chain_errors,
    chain_executed,
    metric_incidents,
    read_only_errors,
    settled,
)
from scripts.e2e.regional.general_live_registry import deferred_variant

CASE = "GF-GENERAL-LIVE-019"
RESET = frozenset(
    {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "REPLACE_NODE"}
)


def _hold_seconds(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, 20 * 60)


# --------------------------------------------------------------------------- #
# diagnostic-only counter variants
# --------------------------------------------------------------------------- #
def _diagnostic(ctx: VariantContext, canonical: str) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target(canonical)
    since = utc_now()
    replay = DcgmReplay(
        ctx,
        plan("memory", baseline(), step(40, add(canonical, [gpu]), hold=True)),
        _hold_seconds(ctx),
    )
    arm_and_confirm(replay, f"{gpu}/{canonical}")
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: metric_incidents(doc, canonical, gpu=gpu),
        metrics=(canonical,),
        timeout_seconds=survey.health_summary + 240,
        name="detection",
    )
    chain = chain_errors(
        detection,
        metric_incidents(detection, canonical, gpu=gpu),
        GpuChain(
            label=f"{canonical} on {gpu}",
            operations=("FREEZE_EVIDENCE", "VALIDATE_GPU"),
            validation="VALIDATE_GPU",
            validation_must_fail=True,
            forbidden=RESET,
            incident_states=frozenset({"ACTION_PENDING", "ESCALATED", "RECOVERED"}),
        ),
    )
    stages = {
        "chain": chain,
        "read_only": read_only_errors(
            detection, metric_incidents(detection, canonical, gpu=gpu), canonical
        ),
        "hardware": no_hardware(detection),
    }
    recovered = recover_counter(ctx, replay, since, names=[canonical])
    stages["recovery"] = (
        []
        if not state_now(ctx, since, names=[canonical]).get("active_findings")
        else []
    )
    return result(
        stages,
        extra={
            "gpu_uuid": gpu,
            "recovered_batches": len(recovered.get("batches") or []),
        },
    )


def corrected_ecc(ctx: VariantContext) -> dict[str, Any]:
    return _diagnostic(ctx, "ecc_sbe_volatile_total")


def aggregate_dbe(ctx: VariantContext) -> dict[str, Any]:
    return _diagnostic(ctx, "ecc_dbe_aggregate_total")


def retired_delta(ctx: VariantContext) -> dict[str, Any]:
    return _diagnostic(ctx, "retired_pages_dbe_total")


# --------------------------------------------------------------------------- #
# volatile DBE: CRITICAL QUARANTINE (isolation, no reset)
# --------------------------------------------------------------------------- #
#: The containment steps of the isolating variants execute seconds after the
#: incident is minted; this is what the detection wait grants them on top of
#: the detection budget before the chain is judged on executions.
CONTAINMENT_SECONDS = 180


def _wait_contained(
    ctx: VariantContext,
    since: Any,
    canonical: str,
    gpu: str,
    chain: GpuChain,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Detection that waits until the chain can be judged on executed steps.

    The incident alone is not enough: the record is judged on what it executed
    (``chain_errors`` / ``never_executed_errors``), so the wait ends once the
    isolating steps have a terminal execution or the record has ended
    (``chain_executed``) -- the compile-time BLOCKED record of live 2026-10-26
    ended at once and reads as the hollow chain it was.
    """

    def contained(doc: dict[str, Any]) -> list[dict[str, Any]]:
        found = metric_incidents(doc, canonical, gpu=gpu)
        return found if found and chain_executed(doc, found, chain) else []

    return wait_gpu_incident(
        ctx,
        since,
        contained,
        metrics=(canonical,),
        timeout_seconds=timeout_seconds,
        name="detection",
    )


def volatile_dbe(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("ecc_dbe_volatile_total")
    since = utc_now()
    replay = DcgmReplay(
        ctx,
        plan(
            "memory",
            baseline(),
            step(60, add("ecc_dbe_volatile_total", [gpu]), hold=True),
        ),
        _hold_seconds(ctx),
    )
    arm_and_confirm(replay, f"{gpu}/ecc_dbe_volatile_total")
    chain = GpuChain(
        label=f"ecc_dbe_volatile_total on {gpu}",
        operations=("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"),
        forbidden=RESET,
        actions=frozenset({"QUARANTINE"}),
        incident_states=frozenset({"QUARANTINED", "ACTION_PENDING"}),
    )
    detection = _wait_contained(
        ctx,
        since,
        "ecc_dbe_volatile_total",
        gpu,
        chain,
        timeout_seconds=survey.health_summary + 240 + CONTAINMENT_SECONDS,
    )
    incidents = metric_incidents(detection, "ecc_dbe_volatile_total", gpu=gpu)
    stages = {
        "chain": chain_errors(detection, incidents, chain),
        "hardware": no_hardware(detection),
    }
    # The replay stops asserting so the node's real value (0) returns, then the
    # product's validated restore (VALIDATE_GPU/HOST/FABRIC -> RESTORE) is the
    # only readmission; a hand-deleted taint is never used.
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    stages["recovery"] = _validated_restore(ctx, incidents)
    return result(
        stages,
        notes=["volatile DBE isolates the node (quarantine); no reset or reboot"],
        extra={"gpu_uuid": gpu},
    )


def remap_failure(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("row_remap_failure")
    since = utc_now()
    replay = DcgmReplay(
        ctx,
        plan(
            "memory",
            baseline(),
            step(60, put("row_remap_failure", [gpu], 1), hold=True),
        ),
        _hold_seconds(ctx),
    )
    arm_and_confirm(replay, f"{gpu}/row_remap_failure")
    chain = GpuChain(
        label=f"row_remap_failure on {gpu}",
        operations=("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"),
        forbidden=RESET,
        # Planned by the DRAIN, owned by nobody here: memoryDiagnostics is
        # OBSERVE on both live profiles, so the field diagnostic must not run.
        unexecuted=frozenset({"RUN_FIELD_DIAGNOSTIC"}),
        incident_states=frozenset(
            {"QUARANTINED", "SAFETY_PENDING", "ACTION_PENDING", "ESCALATED"}
        ),
    )
    detection = _wait_contained(
        ctx,
        since,
        "row_remap_failure",
        gpu,
        chain,
        timeout_seconds=survey.health_summary + 240 + CONTAINMENT_SECONDS,
    )
    incidents = metric_incidents(detection, "row_remap_failure", gpu=gpu)
    stages = {
        "chain": chain_errors(detection, incidents, chain),
        "hardware": no_hardware(detection, allowed=("RUN_FIELD_DIAGNOSTIC",)),
    }
    # The replay ends before the exit: the restore's VALIDATE_GPU rightly
    # refuses a node still reading row_remap_failure=1.
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    exit_stages, exit_record = _contained_exit(ctx, since, incidents)
    stages.update(exit_stages)
    return result(
        stages,
        notes=[
            "row remap failure drains the node first; the NVIDIA RMA field "
            "diagnostic has no executable owner on this site (memoryDiagnostics "
            "OBSERVE), so the record settles SAFETY_SETTLED with the incident "
            "QUARANTINED and the operator notified; the diagnosis is never read "
            "as hardware already replaced",
            "the contained node is readmitted through the product's validated "
            "restore once the replay has ended (gpu-fault-admin "
            "submit-remediation --disposition restore); only a record the "
            "product escalated (ESCALATED) is closed through the operator exit "
            "(IncidentClosureService.close_incident) instead",
        ],
        extra={"gpu_uuid": gpu, "contained_exit": exit_record},
    )


# --------------------------------------------------------------------------- #
# authorized service-window single-GPU reset
# --------------------------------------------------------------------------- #
def _pending_reset(ctx: VariantContext, *, verify_recovery: bool) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("row_remap_pending")
    if "H" not in survey.model(gpu) and "B" not in survey.model(gpu):
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            f"{gpu} is {survey.model(gpu)!r}; row_remap_pending reset needs a modern GPU",
        )
    since = utc_now()
    # Hold pending=1 until RESET_GPU has run, then serve pending=0 so the
    # post-reset VALIDATE_GPU reads a clean device; the held step is advanced
    # once the product's workflow reports RESET_GPU SUCCEEDED.
    replay = DcgmReplay(
        ctx,
        plan(
            "memory",
            baseline(),
            step(120, put("row_remap_pending", [gpu], 1), hold=True),
            step(60, put("row_remap_pending", [gpu], 0)),
        ),
        _hold_seconds(ctx),
    )
    arm_and_confirm(replay, f"{gpu}/row_remap_pending")
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: metric_incidents(doc, "row_remap_pending", gpu=gpu),
        metrics=("row_remap_pending",),
        timeout_seconds=survey.health_summary + 240,
        name="detection",
    )
    incidents = metric_incidents(detection, "row_remap_pending", gpu=gpu)
    stages = {
        "chain": chain_errors(
            detection,
            incidents,
            GpuChain(
                label=f"row_remap_pending on {gpu}",
                operations=(
                    "QUIESCE_GPU_SERVICES",
                    "VERIFY_NO_GPU_CLIENTS",
                    "RESET_GPU",
                    "RESTORE_GPU_SERVICES",
                    "VALIDATE_GPU",
                ),
                forbidden=frozenset(
                    {"RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "REPLACE_NODE"}
                ),
                actions=frozenset({"RESET_GPU"}),
            ),
        ),
    }
    if verify_recovery:
        reset_done = wait_gpu_incident(
            ctx,
            since,
            lambda doc: [i for i in incidents if _reset_succeeded(doc, i)],
            timeout_seconds=survey.health_summary + 600,
            name="reset-done",
        )
        replay.advance()
        settled_doc = wait_gpu_incident(
            ctx,
            since,
            lambda doc: incidents if settled(doc, incidents) else [],
            timeout_seconds=600,
            name="settled",
        )
        stages["recovery"] = (
            []
            if settled(settled_doc, incidents)
            and not state_now(ctx, since, names=["row_remap_pending"]).get(
                "active_findings"
            )
            else ["the reset incident did not settle with a clean post-reset reading"]
        )
        _ = reset_done
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=["single-GPU service-window reset; no NVSwitch reset, no reboot"],
        extra={"gpu_uuid": gpu},
    )


def pending_supported(ctx: VariantContext) -> dict[str, Any]:
    return _pending_reset(ctx, verify_recovery=False)


def post_reset(ctx: VariantContext) -> dict[str, Any]:
    return _pending_reset(ctx, verify_recovery=True)


def _reset_succeeded(doc: dict[str, Any], incident: dict[str, Any]) -> bool:
    from scripts.e2e.regional.general_live_gpu_verdicts import executions_of

    return any(
        item.get("operation") == "RESET_GPU" and item.get("status") == "SUCCEEDED"
        for item in executions_of(doc, "RESET_GPU", [incident])
    )


def _validated_restore(
    ctx: VariantContext, incidents: list[dict[str, Any]]
) -> list[str]:
    from scripts.e2e.regional.general_live_common import restore_incidents

    try:
        restore_incidents(ctx, incidents, reason=f"{ctx.spec.key} acceptance restore")
    except Exception as exc:  # noqa: BLE001 - recorded as a cleanup error
        return [f"validated restore failed: {type(exc).__name__}: {exc}"]
    return []


def _contained_exit(
    ctx: VariantContext, since: Any, incidents: list[dict[str, Any]]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """The exit the settled record asks for: ``({stage: errors}, record)``.

    Once the incident's workflow has settled, the node must carry the chain's
    containment (cordon, the incident's quarantine taint, its ownership
    annotation) -- a chain that never isolated the node is the recovery error
    and nothing is restored or closed. Then the state decides:

    * QUARANTINED (b192 / R13h: containment ran, the field diagnostic has no
      executable owner, BLOCKED/SAFETY_SETTLED) -- the product's
      contained-operator-handoff notification is required (the ``handoff``
      stage) and the node is readmitted through the validated restore
      (``restore_incidents``: VALIDATE_GPU/HOST/FABRIC then RESTORE_SCHEDULING,
      the restore SUCCEEDED, no isolation left), the incident RECOVERED on a
      fresh read;
    * ESCALATED (the product handed the node to an operator) -- the operator
      close, proven as ``operator_close_incidents`` proves it;
    * anything else is the recovery error.

    A refusal on either path is the stage's error and is never retried.
    """

    from scripts.e2e.regional.general_live_common import (
        await_handoff_notification,
        containment_missing,
        node_residual_isolation,
        operator_close_incidents,
        restore_incidents,
        settle_incident,
    )
    from scripts.e2e.regional.regional_commands import RegionalFixtureError
    from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture

    node = ctx.settings.node
    record: dict[str, Any] = {}
    stages: dict[str, list[str]] = {}
    try:
        if not incidents:
            raise RegionalFixtureError("no incident to exit from")
        incident_id = str(incidents[0]["incident_id"])
        settled = settle_incident(ctx, incident_id)
        before = ctx.regional.node_snapshot(node)
        state = str(settled.get("state") or "")
        record = {
            "incident_id": incident_id,
            "workflow_request_id": settled.get("workflow_request_id"),
            "state_settled": state,
            "node_before": node_residual_isolation(before),
        }
        ctx.evidence.write("contained-exit", record)
        missing = containment_missing(before, incident_id)
        if missing:
            raise RegionalFixtureError(
                f"chain never isolated the node {node} for {incident_id}: "
                + "; ".join(missing)
            )
        if state == "QUARANTINED":
            record["exit"] = "validated-restore"
            stages["handoff"], record["handoff"] = await_handoff_notification(
                ctx, since, incident_id
            )
            restored = restore_incidents(
                ctx, incidents, reason=f"{ctx.spec.key} acceptance restore"
            )
            record["restore"] = restored
            warm = WarmSpareLiveFixture(ctx.regional, ctx.settings.hyperpod_cluster)
            fresh = warm.incident_by_id(incident_id)
            record["state_after"] = fresh.get("state")
            record["node_after"] = node_residual_isolation(
                ctx.regional.node_snapshot(node)
            )
            if fresh.get("state") != "RECOVERED":
                raise RegionalFixtureError(
                    f"incident {incident_id} is {fresh.get('state')} after the "
                    "validated restore succeeded, not RECOVERED"
                )
        elif state == "ESCALATED":
            record["exit"] = "operator-close"
            record["operator_exit"] = operator_close_incidents(
                ctx,
                incidents,
                reason=(
                    f"{ctx.spec.key} live acceptance: escalation left by the case's "
                    "own fault injection (by design); closed by the variant's "
                    "operator-exit stage"
                ),
            )
        else:
            raise RegionalFixtureError(
                f"incident {incident_id} settled {state or 'unknown'}: neither the "
                "contained QUARANTINED shape (validated restore) nor an ESCALATED "
                "hand-off (operator close)"
            )
    except Exception as exc:  # noqa: BLE001 - recorded as the stage's error
        stages["recovery"] = [f"contained exit failed: {type(exc).__name__}: {exc}"]
    else:
        stages["recovery"] = []
    finally:
        if record:
            ctx.evidence.write("contained-exit", record)
    return stages, record


VARIANTS = [
    gpu_spec(
        CASE,
        "volatile_dbe",
        "increment ecc_dbe_volatile_total on one GPU",
        "CRITICAL QUARANTINE: the node is isolated, no reset; validated restore readmits it",
        volatile_dbe,
        risk="M",
        max_minutes=25,
    ),
    gpu_spec(
        CASE,
        "corrected_ecc",
        "increment ecc_sbe_volatile_total on one GPU",
        "WARNING RUN_DIAGNOSTICS: VALIDATE_GPU, no isolation, no reset; cleared by a fresh batch",
        corrected_ecc,
        max_minutes=20,
    ),
    gpu_spec(
        CASE,
        "aggregate_dbe",
        "increment ecc_dbe_aggregate_total on one GPU",
        "WARNING RUN_DIAGNOSTICS; diagnostic-only, no reset",
        aggregate_dbe,
        max_minutes=20,
    ),
    gpu_spec(
        CASE,
        "retired_delta",
        "increment retired_pages_dbe_total on one GPU",
        "WARNING RUN_DIAGNOSTICS; diagnostic-only, no reset",
        retired_delta,
        max_minutes=20,
    ),
    deferred_variant(
        CASE,
        "pending_unsupported",
        proof_mode="live-replay",
        reason="CAPABILITY_ABSENT: no legacy-architecture (P100/V100-class) GPU on this site; "
        "the row_remap_pending unsupported-architecture path (architecture review, no reset) "
        "cannot be produced on an H200 by a field replay (batch.product is discovered, not a DCGM label)",
        lanes=("eks", "hp"),
        expected="unsupported architecture -> architecture review, no reset",
    ),
    gpu_spec(
        CASE,
        "pending_supported",
        "set row_remap_pending=1 on a modern GPU",
        "authorized single-GPU service-window RESET_GPU chain; no NVSwitch reset or reboot",
        pending_supported,
        risk="H",
        flags="X",
        max_minutes=40,
        approvals=("GPU_RESET",),
    ),
    gpu_spec(
        CASE,
        "remap_failure",
        "set row_remap_failure=1 on a modern GPU",
        "CRITICAL DRAIN containment first; the RMA field diagnostic has no "
        "executable owner on this site (memoryDiagnostics OBSERVE) -> SAFETY_SETTLED, "
        "incident QUARANTINED, operator notified; readmitted by the validated "
        "restore after the replay ends; no reset",
        remap_failure,
        risk="M",
        max_minutes=30,
        approvals=("GPU_FIELD_DIAGNOSTIC",),
    ),
    deferred_variant(
        CASE,
        "observe_profile",
        proof_mode="live-replay",
        reason="CAPABILITY_ABSENT: the live Runtime Profile owns gpuReset; there is no OBSERVE-mode "
        "reset profile on this site to prove a pending reading stays observe-only",
        lanes=("eks", "hp"),
        expected="OBSERVE reset capability -> no reset, incident escalated to an operator",
    ),
    gpu_spec(
        CASE,
        "post_reset",
        "hold row_remap_pending=1 through the reset, then serve a clean reading",
        "after the single-GPU reset a new clean pending reading passes VALIDATE_GPU; no blind reboot loop",
        post_reset,
        risk="H",
        flags="X",
        max_minutes=45,
        approvals=("GPU_RESET",),
    ),
]
