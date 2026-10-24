"""GF-GENERAL-LIVE-021 NVLink fabric not trained (live-replay).

The fabric state is DCGM field 170, graded per node over every GPU
(``gpu_fabric_manager``): a graded NVSwitch GPU (H100/H200/H800/H20, HGX
B200/B300) that holds an unsettled state (1 not started, 2 in progress, 4
failure) for >= 300 s mints one CRITICAL finding whose recovery is the full
GPU + NVSwitch reset. This module sets the state through the ``fabric`` profile
and reads the product's own grading.

Producing the finding therefore drives a real RESET_ALL_GPUS_NVSWITCHES on the
node, so every variant that crosses the window needs ``--approve
GPU_FULL_FABRIC_RESET`` and runs only on the HyperPod lane's exclusive node
(``recovery_failure`` also needs ``--approve NODE_REBOOT``). The held plan
serves SUCCESS (3) once RESET_ALL has run so the post-reset VALIDATE_FABRIC
reads a trained fabric. ``before_window`` (held under 300 s), ``ungraded_product``
(a non-NVSwitch modelName) and ``monitor_restart`` (the held clock restarts
with the collector) never cross the window, so they isolate nothing and need
no reset approval.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    arm_and_confirm,
    baseline,
    gpu_spec,
    label,
    no_hardware,
    plan,
    put,
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
    executions_of,
    fabric_incidents,
    settled,
)

CASE = "GF-GENERAL-LIVE-021"
WINDOW = 300
STATES = {"not_started": 1, "in_progress": 2, "failed": 4}
FULL_RESET = (
    "QUIESCE_GPU_SERVICES",
    "RESET_ALL_GPUS_NVSWITCHES",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "VALIDATE_FABRIC",
)


def _graded_gpus(survey) -> list[str]:
    graded = [gpu for gpu in survey.gpus if survey.has(gpu, "fabric_manager_status")]
    models = {survey.model(gpu) for gpu in graded}
    if not graded or not any(
        any(token in model for token in ("H100", "H200", "H800", "H20", "B200", "B300"))
        for model in models
    ):
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            "no graded NVSwitch GPU (H100/H200/H800/H20, HGX B200/B300) reports a fabric state",
        )
    return graded


def _hold(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, 45 * 60)


# --------------------------------------------------------------------------- #
# detection + authorized full fabric reset
# --------------------------------------------------------------------------- #
def _unsettled_reset(
    ctx: VariantContext, status: int, *, recovery_failure: bool = False
) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = _graded_gpus(survey)
    since = utc_now()
    tail = put("fabric_manager_status", gpus, 4 if recovery_failure else 3)
    repl = DcgmReplay(
        ctx,
        plan(
            "fabric",
            baseline(),
            step(
                int((WINDOW + 180) / max(1, survey.interval)) + 4,
                put("fabric_manager_status", gpus, status),
                hold=True,
            ),
            step(40, tail),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, *[f"{gpu}/fabric_manager_status" for gpu in gpus])
    detection = wait_gpu_incident(
        ctx,
        since,
        fabric_incidents,
        timeout_seconds=WINDOW + survey.health_summary + 300,
        name="detection",
    )
    incidents = fabric_incidents(detection)
    stages = {
        "chain": chain_errors(
            detection,
            incidents,
            GpuChain(
                label="nvlink fabric not trained",
                operations=FULL_RESET,
                actions=frozenset({"RESET_GPU"}),
                forbidden=frozenset({"REPLACE_NODE"}),
            ),
        ),
    }
    reset_done = wait_gpu_incident(
        ctx,
        since,
        lambda doc: [
            i
            for i in incidents
            if executions_of(doc, "RESET_ALL_GPUS_NVSWITCHES", [i])
            and any(
                e.get("status") == "SUCCEEDED"
                for e in executions_of(doc, "RESET_ALL_GPUS_NVSWITCHES", [i])
            )
        ],
        timeout_seconds=WINDOW + survey.health_summary + 900,
        name="reset-done",
    )
    repl.advance()
    final = wait_gpu_incident(
        ctx,
        since,
        lambda doc: incidents if settled(doc, incidents) else [],
        timeout_seconds=900,
        name="settled",
    )
    if recovery_failure:
        stages["recovery_failure"] = (
            []
            if _escalated(final, incidents)
            else [
                "a fabric that stayed failed after the reset did not escalate/hold isolation"
            ]
        )
    else:
        stages["recovery"] = (
            []
            if settled(final, incidents) and not _fabric_active(ctx, since)
            else ["the fabric reset did not settle with a trained post-reset fabric"]
        )
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    _ = reset_done
    return result(stages, extra={"gpus": gpus, "status": status})


def not_started(ctx: VariantContext) -> dict[str, Any]:
    return _unsettled_reset(ctx, STATES["not_started"])


def in_progress(ctx: VariantContext) -> dict[str, Any]:
    return _unsettled_reset(ctx, STATES["in_progress"])


def failed(ctx: VariantContext) -> dict[str, Any]:
    return _unsettled_reset(ctx, STATES["failed"])


def full_reset(ctx: VariantContext) -> dict[str, Any]:
    return _unsettled_reset(ctx, STATES["in_progress"])


def every_gpu_success(ctx: VariantContext) -> dict[str, Any]:
    # The recovery tail asserts SUCCESS on every graded GPU; a single-PID
    # Fabric Manager restart is not enough, VALIDATE_FABRIC waits for state 3.
    return _unsettled_reset(ctx, STATES["not_started"])


def recovery_failure(ctx: VariantContext) -> dict[str, Any]:
    return _unsettled_reset(ctx, STATES["failed"], recovery_failure=True)


# --------------------------------------------------------------------------- #
# below-window / ungraded / monitor-restart (no finding, no reset)
# --------------------------------------------------------------------------- #
def before_window(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = _graded_gpus(survey)
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "fabric",
            baseline(),
            step(400, put("fabric_manager_status", gpus, 2), hold=True),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, *[f"{gpu}/fabric_manager_status" for gpu in gpus])
    ctx.hold(min(WINDOW - 30, _hold(ctx)), name="under-window")
    activity = wait_gpu_incident(
        ctx, since, lambda d: [], timeout_seconds=1, name="under-window-check"
    )
    stages = {
        "no_finding": []
        if not fabric_incidents(activity)
        else ["an unsettled fabric under the window minted a finding"],
        "hardware": no_hardware(activity),
    }
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=["the 300 s window keeps boot-time training out"],
        extra={"gpus": gpus},
    )


def ungraded_product(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = _graded_gpus(survey)[:1]
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "fabric",
            baseline(),
            step(
                400,
                put("fabric_manager_status", gpus, 2),
                label("fabric_manager_status", gpus, "modelName", "NVIDIA L40S"),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, *[f"{gpu}/fabric_manager_status" for gpu in gpus])
    ctx.hold(min(WINDOW + 60, _hold(ctx)), name="ungraded")
    activity = wait_gpu_incident(
        ctx, since, lambda d: [], timeout_seconds=1, name="ungraded-check"
    )
    stages = {
        "not_graded": []
        if not fabric_incidents(activity)
        else ["a non-NVSwitch product was graded for fabric state"],
        "hardware": no_hardware(activity),
    }
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=["an L40S-class (no NVSwitch) product is not fabric-graded"],
        extra={"gpus": gpus},
    )


def monitor_restart(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = _graded_gpus(survey)
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "fabric",
            baseline(),
            step(800, put("fabric_manager_status", gpus, 2), hold=True),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, *[f"{gpu}/fabric_manager_status" for gpu in gpus])
    ctx.hold(min(WINDOW - 60, _hold(ctx)), name="before-restart")
    restart = repl.collector_restart()
    ctx.hold(min(WINDOW - 60, _hold(ctx)), name="after-restart")
    activity = wait_gpu_incident(
        ctx, since, lambda d: [], timeout_seconds=1, name="restart-check"
    )
    stages = {
        "clock_reset": []
        if not fabric_incidents(activity)
        else ["the held-time clock did not restart with the collector"],
        "hardware": no_hardware(activity),
    }
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=[
            "the held-seconds clock restarts at the collector's first post-restart observation"
        ],
        extra={"gpus": gpus, "restart": restart},
    )


def _fabric_active(ctx: VariantContext, since: Any) -> bool:
    doc = state_now(ctx, since, names=["fabric_manager_status"])
    return any(
        item.get("canonical_name") == "fabric_manager_status"
        for item in doc.get("active_findings") or []
    )


def _escalated(doc: dict[str, Any], incidents: list[dict[str, Any]]) -> bool:
    wanted = {str(item["incident_id"]) for item in incidents}
    return any(
        item.get("incident_id") in wanted
        and item.get("state") in {"ESCALATED", "QUARANTINED"}
        for item in doc.get("incidents") or []
    )


VARIANTS = [
    gpu_spec(
        CASE,
        "not_started",
        "hold DCGM fabric state 1 past the 300 s window",
        "CRITICAL full GPU+NVSwitch reset; trained after",
        not_started,
        risk="H",
        flags="X",
        lanes=("hp",),
        max_minutes=40,
        approvals=("GPU_FULL_FABRIC_RESET",),
    ),
    gpu_spec(
        CASE,
        "in_progress",
        "hold DCGM fabric state 2 past the window",
        "CRITICAL full fabric reset; trained after",
        in_progress,
        risk="H",
        flags="X",
        lanes=("hp",),
        max_minutes=40,
        approvals=("GPU_FULL_FABRIC_RESET",),
    ),
    gpu_spec(
        CASE,
        "failed",
        "hold DCGM fabric state 4 (failure)",
        "CRITICAL full fabric reset at once",
        failed,
        risk="H",
        flags="X",
        lanes=("hp",),
        max_minutes=40,
        approvals=("GPU_FULL_FABRIC_RESET",),
    ),
    gpu_spec(
        CASE,
        "before_window",
        "hold an unsettled fabric state under 300 s",
        "no finding before the window elapses",
        before_window,
        lanes=("hp",),
        max_minutes=12,
    ),
    gpu_spec(
        CASE,
        "ungraded_product",
        "hold an unsettled state on a non-NVSwitch modelName",
        "an ungraded product is not fabric-graded",
        ungraded_product,
        lanes=("hp",),
        max_minutes=12,
    ),
    gpu_spec(
        CASE,
        "monitor_restart",
        "restart the metrics collector mid-hold",
        "the held-time clock restarts; no premature finding",
        monitor_restart,
        lanes=("hp",),
        max_minutes=16,
    ),
    gpu_spec(
        CASE,
        "full_reset",
        "hold an unsettled state and let the full reset run",
        "one reset of every GPU and NVSwitch inside the quiesce window",
        full_reset,
        risk="H",
        flags="X",
        lanes=("hp",),
        max_minutes=45,
        approvals=("GPU_FULL_FABRIC_RESET",),
    ),
    gpu_spec(
        CASE,
        "every_gpu_success",
        "hold unsettled, reset, require SUCCESS on every GPU",
        "VALIDATE_FABRIC waits for state 3 on every graded GPU, not one PID",
        every_gpu_success,
        risk="H",
        flags="X",
        lanes=("hp",),
        max_minutes=45,
        approvals=("GPU_FULL_FABRIC_RESET",),
    ),
    gpu_spec(
        CASE,
        "recovery_failure",
        "keep the fabric failed after the reset",
        "a reset that does not train escalates and holds isolation; no blind loop",
        recovery_failure,
        risk="H",
        flags="X",
        lanes=("hp",),
        max_minutes=45,
        approvals=("GPU_FULL_FABRIC_RESET", "NODE_REBOOT"),
    ),
]
