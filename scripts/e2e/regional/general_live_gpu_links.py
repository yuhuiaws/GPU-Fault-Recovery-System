"""GF-GENERAL-LIVE-020 NVLink error counters and 022 PCIe replay (live-replay).

020 increments the reviewed NVLink error counters through the ``nvlink``
profile and reads the product's grading (``gpu_metrics._link_decision`` ->
WARNING RUN_DIAGNOSTICS per class; two classes on one GPU or one class on two
GPUs form the NVLink composites). A counter reset or a stale (unrefreshed)
scrape must not be read as a new fault or a recovery.

022 drives the PCIe replay counter (``pcie``/``pcie_xid`` profiles). The rate
is graded against the previous accepted sample, so ``at_rate_limit`` stays at
or below 8/min (no finding) and ``over_rate_limit`` clears it (WARNING
RUN_DIAGNOSTICS). ``same_gpu_xid`` / ``wrong_gpu_xid`` / ``expired_xid`` drive
the PCIe<->XID join through the DCGM last-XID gauge (replayed only as XID 32,
catalog RESTART_APP and monitor-only on an idle node; ``--approve
DCGM_XID_GAUGE``). ``expired_xid`` is an expected FAIL: the gauge fallback has
no age check, so a stale gauge keeps re-arming the CRITICAL DRAIN composite.

``independent_fatal_event`` needs a trusted independent kernel XID/SXID; that
source is lane K's, so it is recorded NOT_RUN(FIXTURE_NOT_READY) here.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_fixture import VariantContext
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    add,
    arm_and_confirm,
    at_limit_increment,
    baseline,
    clear,
    gpu_spec,
    no_hardware,
    over_limit_increment,
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
    composite_incidents,
    metric_incidents,
    read_only_errors,
)
from scripts.e2e.regional.general_live_registry import deferred_variant

CASE_020 = "GF-GENERAL-LIVE-020"
CASE_022 = "GF-GENERAL-LIVE-022"
NVLINK = {
    "crc": "nvlink_crc_flit_error_total",
    "replay": "nvlink_replay_error_total",
    "recovery_counter": "nvlink_recovery_error_total",
}
RESET = frozenset(
    {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "REPLACE_NODE"}
)


def _hold(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, 20 * 60)


def _diag_chain(doc: dict[str, Any], canonical: str, gpu: str) -> list[str]:
    return chain_errors(
        doc,
        metric_incidents(doc, canonical, gpu=gpu),
        GpuChain(
            label=f"{canonical} on {gpu}",
            operations=("FREEZE_EVIDENCE", "VALIDATE_GPU"),
            validation="VALIDATE_GPU",
            validation_must_fail=True,
            forbidden=RESET,
            incident_states=frozenset({"ACTION_PENDING", "ESCALATED", "RECOVERED"}),
        ),
    )


# --------------------------------------------------------------------------- #
# 020 NVLink counters
# --------------------------------------------------------------------------- #
def _nvlink_single(ctx: VariantContext, canonical: str) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target(canonical)
    since = utc_now()
    replay = DcgmReplay(
        ctx,
        plan("nvlink", baseline(), step(40, add(canonical, [gpu]), hold=True)),
        _hold(ctx),
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
    stages = {
        "chain": _diag_chain(detection, canonical, gpu),
        "read_only": read_only_errors(
            detection, metric_incidents(detection, canonical, gpu=gpu), canonical
        ),
        "hardware": no_hardware(detection),
    }
    recover_counter(ctx, replay, since, names=[canonical])
    return result(stages, extra={"gpu_uuid": gpu})


def crc(ctx: VariantContext) -> dict[str, Any]:
    return _nvlink_single(ctx, NVLINK["crc"])


def replay(ctx: VariantContext) -> dict[str, Any]:
    return _nvlink_single(ctx, NVLINK["replay"])


def recovery_counter(ctx: VariantContext) -> dict[str, Any]:
    return _nvlink_single(ctx, NVLINK["recovery_counter"])


def multiple_gpus(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    canonical = NVLINK["crc"]
    gpus = survey.gpus_with(canonical)[:2]
    if len(gpus) < 2:
        from scripts.e2e.regional.general_live_fixture import VariantNotRun

        raise VariantNotRun(
            "CAPABILITY_ABSENT", "fewer than two GPUs report NVLink counters"
        )
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan("nvlink", baseline(), step(40, add(canonical, gpus), hold=True)),
        _hold(ctx),
    )
    arm_and_confirm(repl, *[f"{gpu}/{canonical}" for gpu in gpus])
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, "MULTI_GPU_NVLINK_FABRIC_FAILURE"),
        metrics=(canonical,),
        timeout_seconds=survey.health_summary + 240,
        name="detection",
    )
    stages = {
        "chain": chain_errors(
            detection,
            composite_incidents(detection, "MULTI_GPU_NVLINK_FABRIC_FAILURE"),
            GpuChain(
                label="MULTI_GPU_NVLINK_FABRIC_FAILURE",
                operations=("FREEZE_EVIDENCE",),
                forbidden=RESET,
            ),
        ),
        "hardware": no_hardware(detection),
    }
    recover_counter(ctx, repl, since, names=[canonical])
    return result(
        stages,
        notes=["multi-GPU NVLink correlation is diagnostic, never a fatal verdict"],
        extra={"gpus": gpus},
    )


def counter_reset(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    canonical = NVLINK["crc"]
    gpu = survey.target(canonical)
    since = utc_now()
    # Raise the counter (the product baselines above the real value), then drop
    # it: the decrease is a counter-reset edge, never a new fault.
    repl = DcgmReplay(
        ctx,
        plan(
            "nvlink",
            baseline(),
            step(3, add(canonical, [gpu], 5)),
            step(20, clear(canonical, [gpu]), hold=True),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/{canonical}")
    ctx.hold(min(120, _hold(ctx)), name="reset-observe")
    doc = state_now(ctx, since, names=[canonical])
    activity = wait_gpu_incident(
        ctx,
        since,
        lambda d: [] if True else d.get("incidents"),
        metrics=(canonical,),
        timeout_seconds=1,
        name="reset-check",
    )
    stages = {
        "no_fault": []
        if not metric_incidents(activity, canonical, gpu=gpu)
        else ["a counter reset minted a finding"],
        "hardware": no_hardware(activity),
        "active": []
        if not doc.get("active_findings")
        else ["a counter reset left an active finding"],
    }
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=["a counter reset (decrease) is not a new fault or a recovery"],
        extra={"gpu_uuid": gpu},
    )


def stale_sample(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    canonical = NVLINK["crc"]
    gpu = survey.target(canonical)
    since = utc_now()
    # One armed increment, then serve the same scrape again (stale): the
    # unrefreshed read must be neither a new fault nor a recovery.
    repl = DcgmReplay(
        ctx,
        plan(
            "nvlink",
            baseline(),
            step(2, add(canonical, [gpu], 1)),
            step(20, {"op": "stale", "age": 1}, hold=True),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/{canonical}")
    ctx.hold(min(120, _hold(ctx)), name="stale-observe")
    activity = wait_gpu_incident(
        ctx,
        since,
        lambda d: [],
        metrics=(canonical,),
        timeout_seconds=1,
        name="stale-check",
    )
    stages = {
        "single": []
        if len(metric_incidents(activity, canonical, gpu=gpu)) <= 1
        else ["a stale scrape minted extra findings"],
        "hardware": no_hardware(activity),
    }
    recover_counter(ctx, repl, since, names=[canonical])
    return result(
        stages,
        notes=["an unrefreshed (stale) scrape carries no new information"],
        extra={"gpu_uuid": gpu},
    )


def recovered(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    canonical = NVLINK["replay"]
    gpu = survey.target(canonical)
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan("nvlink", baseline(), step(40, add(canonical, [gpu]), hold=True)),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/{canonical}")
    wait_gpu_incident(
        ctx,
        since,
        lambda doc: metric_incidents(doc, canonical, gpu=gpu),
        metrics=(canonical,),
        timeout_seconds=survey.health_summary + 240,
        name="detection",
    )
    recovered_state = recover_counter(ctx, repl, since, names=[canonical])
    stages = {
        "recovery": []
        if recovered_state.get("batches") and not recovered_state.get("active_findings")
        else ["recovery needs a new normal increment, not a counter zeroing"],
    }
    return result(stages, extra={"gpu_uuid": gpu})


# --------------------------------------------------------------------------- #
# 022 PCIe replay
# --------------------------------------------------------------------------- #
def at_rate_limit(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("pcie_replay_total")
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "pcie",
            baseline(),
            step(
                30,
                add("pcie_replay_total", [gpu], at_limit_increment(survey.interval)),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/pcie_replay_total")
    ctx.hold(min(180, _hold(ctx)), name="at-limit-observe")
    activity = wait_gpu_incident(
        ctx,
        since,
        lambda d: [],
        metrics=("pcie_replay_total",),
        timeout_seconds=1,
        name="at-limit",
    )
    stages = {
        "no_fault": []
        if not metric_incidents(activity, "pcie_replay_total", gpu=gpu)
        else ["a rate at the limit minted a finding"],
        "hardware": no_hardware(activity),
    }
    recover_counter(ctx, repl, since, names=["pcie_replay_total"])
    return result(
        stages,
        notes=["a PCIe rate at exactly 8/min does not breach (> is the test)"],
        extra={"gpu_uuid": gpu},
    )


def over_rate_limit(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("pcie_replay_total")
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "pcie",
            baseline(),
            step(
                40,
                add("pcie_replay_total", [gpu], over_limit_increment(survey.interval)),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/pcie_replay_total")
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: metric_incidents(doc, "pcie_replay_total", gpu=gpu),
        metrics=("pcie_replay_total",),
        timeout_seconds=survey.health_summary + 240,
        name="detection",
    )
    stages = {
        "chain": _diag_chain(detection, "pcie_replay_total", gpu),
        "read_only": read_only_errors(
            detection,
            metric_incidents(detection, "pcie_replay_total", gpu=gpu),
            "pcie_replay_total",
        ),
        "hardware": no_hardware(detection),
    }
    recover_counter(ctx, repl, since, names=["pcie_replay_total"])
    return result(stages, extra={"gpu_uuid": gpu})


def pcie_counter_reset(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("pcie_replay_total")
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "pcie",
            baseline(),
            step(3, add("pcie_replay_total", [gpu], 50)),
            step(20, clear("pcie_replay_total", [gpu]), hold=True),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/pcie_replay_total")
    ctx.hold(min(120, _hold(ctx)), name="reset-observe")
    activity = wait_gpu_incident(
        ctx,
        since,
        lambda d: [],
        metrics=("pcie_replay_total",),
        timeout_seconds=1,
        name="reset",
    )
    stages = {
        "no_fault": []
        if not metric_incidents(activity, "pcie_replay_total", gpu=gpu)
        else ["a PCIe counter reset minted a finding"],
        "hardware": no_hardware(activity),
    }
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(stages, extra={"gpu_uuid": gpu})


def source_boot_change(ctx: VariantContext) -> dict[str, Any]:
    from scripts.e2e.regional.general_live_gpu_replay import replay_boot_id

    survey = run_gpu_survey(ctx)
    gpu = survey.target("pcie_replay_total")
    boot, origin = replay_boot_id(survey, ctx.run_id)
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "pcie",
            baseline(),
            step(
                30,
                add("pcie_replay_total", [gpu], over_limit_increment(survey.interval)),
                hold=True,
            ),
        ),
        _hold(ctx),
        boot_id=boot,
    )
    arm_and_confirm(repl, f"{gpu}/pcie_replay_total")
    ctx.hold(min(180, _hold(ctx)), name="boot-observe")
    activity = wait_gpu_incident(
        ctx,
        since,
        lambda d: [],
        metrics=("pcie_replay_total",),
        timeout_seconds=1,
        name="boot",
    )
    stages = {
        "no_cross_boot_rate": []
        if not metric_incidents(activity, "pcie_replay_total", gpu=gpu)
        else ["a source/boot change formed a cross-generation PCIe rate"],
        "hardware": no_hardware(activity),
    }
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=[
            f"replayed boot id from {origin}; a boot change resets the counter baseline"
        ],
        extra={"gpu_uuid": gpu},
    )


def _pcie_xid(ctx: VariantContext, *, same_gpu: bool) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("pcie_replay_total")
    xid_gpu = gpu if same_gpu else survey.target("pcie_replay_total", avoid=[gpu])
    since = utc_now()
    increment = over_limit_increment(survey.interval)
    operations = [
        put("xid_last_error", [xid_gpu], 32),
        add("pcie_replay_total", [gpu], increment),
    ]
    repl = DcgmReplay(
        ctx, plan("pcie_xid", baseline(), step(40, *operations, hold=True)), _hold(ctx)
    )
    arm_and_confirm(repl, f"{gpu}/pcie_replay_total")
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: metric_incidents(doc, "pcie_replay_total", gpu=gpu)
        or composite_incidents(doc, "PCIE_XID_LINK_FAILURE"),
        metrics=("pcie_replay_total",),
        timeout_seconds=survey.health_summary + 240,
        name="detection",
    )
    composite = composite_incidents(detection, "PCIE_XID_LINK_FAILURE")
    if same_gpu:
        stages = {
            "join": []
            if composite
            else [
                "a same-GPU PCIe + XID 32 did not form the PCIE_XID_LINK_FAILURE composite"
            ],
            "hardware": no_hardware(detection, allowed=()),
        }
    else:
        stages = {
            "no_cross_join": []
            if not composite
            else ["a PCIe rate joined an XID on a different GPU"],
            "plain": []
            if metric_incidents(detection, "pcie_replay_total", gpu=gpu)
            else ["the plain PCIe finding did not appear"],
            "hardware": no_hardware(detection),
        }
    recover_counter(ctx, repl, since, names=["pcie_replay_total"])
    return result(stages, extra={"gpu_uuid": gpu, "xid_gpu": xid_gpu})


def same_gpu_xid(ctx: VariantContext) -> dict[str, Any]:
    return _pcie_xid(ctx, same_gpu=True)


def wrong_gpu_xid(ctx: VariantContext) -> dict[str, Any]:
    return _pcie_xid(ctx, same_gpu=False)


def expired_xid(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("pcie_replay_total")
    since = utc_now()
    # Establish the gauge at 32 first (an old event), let the product baseline
    # it, then at a later unrelated window drive a PCIe rate. The gauge is
    # "expired" (no fresh XID), yet the product has no age check and re-arms the
    # CRITICAL DRAIN composite -- the spec says expired XIDs must not join.
    repl = DcgmReplay(
        ctx,
        plan(
            "pcie_xid",
            baseline(),
            step(4, put("xid_last_error", [gpu], 32)),
            step(
                40,
                put("xid_last_error", [gpu], 32),
                add("pcie_replay_total", [gpu], over_limit_increment(survey.interval)),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/pcie_replay_total")
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, "PCIE_XID_LINK_FAILURE")
        or metric_incidents(doc, "pcie_replay_total", gpu=gpu),
        metrics=("pcie_replay_total",),
        timeout_seconds=survey.health_summary + 240,
        name="detection",
    )
    composite = composite_incidents(detection, "PCIE_XID_LINK_FAILURE")
    # expected_outcome=FAIL: PASS only once the product gains an age check and
    # the expired gauge no longer joins (then the composite is absent).
    stages = {
        "expired_not_joined": []
        if not composite
        else [
            "an expired XID gauge joined a PCIe rate into the CRITICAL DRAIN composite (no age check)"
        ],
    }
    recover_counter(ctx, repl, since, names=["pcie_replay_total"])
    return result(stages, extra={"gpu_uuid": gpu})


def recovery_evidence(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("pcie_replay_total")
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "pcie",
            baseline(),
            step(
                40,
                add("pcie_replay_total", [gpu], over_limit_increment(survey.interval)),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/pcie_replay_total")
    wait_gpu_incident(
        ctx,
        since,
        lambda doc: metric_incidents(doc, "pcie_replay_total", gpu=gpu),
        metrics=("pcie_replay_total",),
        timeout_seconds=survey.health_summary + 240,
        name="detection",
    )
    recovered_state = recover_counter(ctx, repl, since, names=["pcie_replay_total"])
    stages = {
        "recovery": []
        if recovered_state.get("batches") and not recovered_state.get("active_findings")
        else [
            "recovery needs a new normal rate and real communication, not a cleared counter"
        ],
    }
    return result(stages, extra={"gpu_uuid": gpu})


VARIANTS = [
    gpu_spec(
        CASE_020,
        "crc",
        "increment nvlink_crc_flit_error_total",
        "WARNING RUN_DIAGNOSTICS; read-only, no reset",
        crc,
        max_minutes=20,
    ),
    gpu_spec(
        CASE_020,
        "replay",
        "increment nvlink_replay_error_total",
        "WARNING RUN_DIAGNOSTICS; read-only",
        replay,
        max_minutes=20,
    ),
    gpu_spec(
        CASE_020,
        "recovery_counter",
        "increment nvlink_recovery_error_total",
        "WARNING RUN_DIAGNOSTICS; read-only",
        recovery_counter,
        max_minutes=20,
    ),
    gpu_spec(
        CASE_020,
        "multiple_gpus",
        "increment one NVLink class on two GPUs",
        "MULTI_GPU_NVLINK_FABRIC_FAILURE diagnostic; no DRAIN/reset",
        multiple_gpus,
        max_minutes=20,
    ),
    gpu_spec(
        CASE_020,
        "counter_reset",
        "raise then drop an NVLink counter",
        "a counter decrease is no new fault or recovery",
        counter_reset,
        max_minutes=18,
    ),
    gpu_spec(
        CASE_020,
        "stale_sample",
        "serve an unrefreshed (stale) NVLink scrape",
        "an unrefreshed scrape is neither a new fault nor a recovery",
        stale_sample,
        max_minutes=18,
    ),
    deferred_variant(
        CASE_020,
        "independent_fatal_event",
        proof_mode="live-replay",
        reason="FIXTURE_NOT_READY: needs the kernel-source replay of lane K; an independent fatal "
        "Xid/SXid must enter through the trusted kernel source and be dispatched by its "
        "catalog, not absorbed by the NVLink counter diagnosis",
        lanes=("eks", "hp"),
        expected="an independent fatal Xid/SXid keeps its own catalog handling, not absorbed by counters",
    ),
    gpu_spec(
        CASE_020,
        "recovered",
        "recover after an NVLink counter fault",
        "recovery needs a new normal increment, not a counter zeroing",
        recovered,
        max_minutes=20,
    ),
    gpu_spec(
        CASE_022,
        "at_rate_limit",
        "drive the PCIe replay rate to exactly 8/min",
        "no finding: the threshold is strictly greater than 8/min",
        at_rate_limit,
        max_minutes=18,
    ),
    gpu_spec(
        CASE_022,
        "over_rate_limit",
        "drive the PCIe replay rate over 8/min",
        "WARNING RUN_DIAGNOSTICS; read-only, no reset",
        over_rate_limit,
        max_minutes=20,
    ),
    gpu_spec(
        CASE_022,
        "counter_reset",
        "raise then drop the PCIe replay counter",
        "a counter reset is no new fault",
        pcie_counter_reset,
        max_minutes=18,
    ),
    gpu_spec(
        CASE_022,
        "source_boot_change",
        "increment PCIe replay under a changed source boot id",
        "a source/boot change forms no cross-generation rate",
        source_boot_change,
        max_minutes=18,
    ),
    gpu_spec(
        CASE_022,
        "same_gpu_xid",
        "PCIe rate + an XID 32 gauge on the same GPU",
        "the same-GPU PCIe<->XID join forms the stronger DRAIN composite",
        same_gpu_xid,
        risk="M",
        max_minutes=22,
        approvals=("DCGM_XID_GAUGE",),
    ),
    gpu_spec(
        CASE_022,
        "wrong_gpu_xid",
        "PCIe rate on one GPU, XID 32 gauge on another",
        "a PCIe rate does not join an XID on a different GPU",
        wrong_gpu_xid,
        max_minutes=22,
        approvals=("DCGM_XID_GAUGE",),
    ),
    gpu_spec(
        CASE_022,
        "expired_xid",
        "drive PCIe rate against a stale XID 32 gauge from an earlier window",
        "an expired XID must not join the PCIe rate into a CRITICAL DRAIN composite "
        "(batch 160: the last-XID gauge is dated by its change and must fall inside the "
        "correlation window)",
        expired_xid,
        risk="M",
        max_minutes=22,
        approvals=("DCGM_XID_GAUGE",),
    ),
    gpu_spec(
        CASE_022,
        "recovery_evidence",
        "recover after a PCIe rate fault",
        "recovery needs a new normal rate and real communication, not a cleared counter",
        recovery_evidence,
        max_minutes=20,
    ),
]
