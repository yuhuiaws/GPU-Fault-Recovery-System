"""GF-GENERAL-LIVE-019 ECC / row remap / page retirement (live-replay).

Each variant increments or sets one reviewed DCGM memory field on the target
GPU through ``tools/gpu_dcgm_field_replay.py`` (the ``memory`` profile) and
reads the product's own grading (``gpu_metrics`` -> ``gpu_memory_policy``):

* ``corrected_ecc`` / ``aggregate_dbe`` / ``retired_delta`` are WARNING
  RUN_DIAGNOSTICS (DIAGNOSTIC_ONLY): a VALIDATE_GPU that isolates nothing and
  resets nothing, cleared by a fresh zero-delta batch;
* ``volatile_dbe`` is CRITICAL QUARANTINE: the node is isolated (no reset)
  and read back through the product's validated restore;
* ``remap_failure`` is CRITICAL DRAIN with the NVIDIA RMA field diagnostic and
  ESCALATE_SUPPORT -- the node stays cordoned until an operator closes it, and
  a field diagnosis is never read as "hardware already replaced";
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
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: metric_incidents(doc, "ecc_dbe_volatile_total", gpu=gpu),
        metrics=("ecc_dbe_volatile_total",),
        timeout_seconds=survey.health_summary + 240,
        name="detection",
    )
    incidents = metric_incidents(detection, "ecc_dbe_volatile_total", gpu=gpu)
    stages = {
        "chain": chain_errors(
            detection,
            incidents,
            GpuChain(
                label=f"ecc_dbe_volatile_total on {gpu}",
                operations=("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"),
                forbidden=RESET,
                actions=frozenset({"QUARANTINE"}),
                incident_states=frozenset({"QUARANTINED", "ACTION_PENDING"}),
            ),
        ),
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
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: metric_incidents(doc, "row_remap_failure", gpu=gpu),
        metrics=("row_remap_failure",),
        timeout_seconds=survey.health_summary + 240,
        name="detection",
    )
    incidents = metric_incidents(detection, "row_remap_failure", gpu=gpu)
    stages = {
        "chain": chain_errors(
            detection,
            incidents,
            GpuChain(
                label=f"row_remap_failure on {gpu}",
                operations=("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"),
                forbidden=RESET,
                incident_states=frozenset(
                    {"ESCALATED", "ACTION_PENDING", "QUARANTINED"}
                ),
            ),
        ),
        "hardware": no_hardware(detection, allowed=("RUN_FIELD_DIAGNOSTIC",)),
    }
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    stages["recovery"] = _validated_restore(ctx, incidents)
    return result(
        stages,
        notes=[
            "row remap failure drains and runs the NVIDIA RMA field diagnostic; "
            "the diagnosis is never read as hardware already replaced"
        ],
        extra={"gpu_uuid": gpu},
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
        "CRITICAL DRAIN with the RMA field diagnostic + ESCALATE_SUPPORT; cordoned until an operator closes, no reset",
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
