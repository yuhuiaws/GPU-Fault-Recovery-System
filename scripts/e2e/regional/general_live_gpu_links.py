"""GF-GENERAL-LIVE-020 NVLink error counters and 022 PCIe replay (live-replay).

020 increments the reviewed NVLink error counters through the ``nvlink``
profile and reads the product's grading (``gpu_metrics._link_decision`` ->
WARNING RUN_DIAGNOSTICS per class; two classes on one GPU or one class on two
GPUs form the NVLink composites). A counter reset or a stale (unrefreshed)
scrape must not be read as a new fault or a recovery.

Each 020 variant names an NVLink error *family* (``NVLINK``: crc, replay,
recovery; ``gpu_fault.dcgm_fields.DCGM_FABRIC_CANONICAL_GROUPS``) and the
survey picks the member this node's exporter really publishes
(``GpuSurvey.nvlink_counter``): the legacy per-class counter on NVLink3 and
earlier parts (``nvlink_crc_flit_error_total`` from
``DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL``), the DL aggregate on
Hopper/NVLink4 and later (``nvlink_crc_aggregate_error_total`` from
``DCGM_FI_DEV_NVLINK_ERROR_DL_CRC``) -- live 2026-10-26 the p5en H200 nodes
report only the aggregates and the flit name hard-coded here before left all
eight variants NOT_RUN(CAPABILITY_ABSENT). The product grades every member of
a family alike; the chosen canonical name and its exporter metric go into the
variant's notes and ``canonical_name`` / ``exporter_metrics`` evidence.

022 drives the PCIe replay counter (``pcie``/``pcie_xid`` profiles). The rate
is graded against the previous accepted sample, so ``at_rate_limit`` stays at
or below 8/min (no finding) and ``over_rate_limit`` clears it (WARNING
RUN_DIAGNOSTICS). ``counter_reset`` and ``source_boot_change`` grade a
*boundary*, so the plan around it must not be a storm of its own: the raise
before the drop stays under the rule (``under_limit_increment``) and the drop
must be the product's ``counter-reset`` edge and new baseline; the boot change
is crossed with the plan's first read already served (``arm_before_repoint``),
a jump over the rule against any reading of the real boot
(``cross_boot_jump``), which must form no rate and must become the new
generation's baseline (live 2026-10-26 both plans were storms -- +50 per read,
+5 per read -- and the oracle blamed their legitimate WARNING on the boundary).
``same_gpu_xid`` / ``wrong_gpu_xid`` / ``expired_xid`` drive
the PCIe<->XID join through the DCGM last-XID gauge (replayed only as XID 32,
catalog RESTART_APP and monitor-only on an idle node; ``--approve
DCGM_XID_GAUGE``). ``expired_xid`` is an expected FAIL: the gauge fallback has
no age check, so a stale gauge keeps re-arming the CRITICAL DRAIN composite.

``independent_fatal_event`` delivers a trusted kernel XID beside the held
NVLink counter fault through lane K's kernel-source replay
(``general_live_kernel_xid``): XID 137 NVLINK_PRIV_ERR, an NVLink-class entry
whose catalog action is IGNORE (-> MONITOR_ONLY), so the live run isolates and
resets nothing while proving the XID keeps its own catalog decision and
incident and is not absorbed by the counter diagnosis. The fatal-grade dispatch
(RESET/QUARANTINE) is deliberately not driven here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
)
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    activity_now,
    add,
    arm_and_confirm,
    arm_before_repoint,
    at_limit_increment,
    baseline,
    clear,
    cross_boot_jump,
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
    under_limit_increment,
    utc_now,
    wait_gpu_incident,
)
from scripts.e2e.regional.general_live_gpu_verdicts import (
    GpuChain,
    accepted_samples,
    active_findings,
    chain_errors,
    composite_incidents,
    counter_reset_ticks,
    finding_text,
    history_findings,
    incident_finding,
    latest_value,
    metric_incidents,
    read_only_errors,
    served_values,
)
from scripts.e2e.regional.general_live_kernel_xid import (
    independence_errors,
    inject_kernel_xid,
    kmsg_pci_bdf,
    require_monitor_grade,
    restore_if_isolated,
    wait_xid_decisions,
)

CASE_020 = "GF-GENERAL-LIVE-020"
CASE_022 = "GF-GENERAL-LIVE-022"
#: The NVLink error family each 020 variant drives; the survey resolves the
#: family to the canonical counter the node reports (``NVLINK_FAMILY_MEMBERS``).
NVLINK = {
    "crc": "nvlink_crc",
    "replay": "nvlink_replay",
    "recovery_counter": "nvlink_recovery",
}
#: The plan text of the CRC counter: never a promise of the flit counter alone.
CRC_COUNTER = "the NVLink CRC counter (flit total on NVLink3, DL aggregate on NVLink4+)"
RESET = frozenset(
    {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "REPLACE_NODE"}
)
#: The reviewed trusted kernel XID beside the NVLink counter fault: an NVLink
#: class entry whose catalog Immediate Action is IGNORE (-> NO_ACTION).
INDEPENDENT_XID = 137
KERNEL_SOURCE_REPLAY = "KERNEL_SOURCE_REPLAY"


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
def _nvlink_single(ctx: VariantContext, family: str) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    counter = survey.nvlink_counter(family)
    canonical = counter.canonical
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
    return result(
        stages, notes=[counter.note()], extra={"gpu_uuid": gpu, **counter.evidence()}
    )


def crc(ctx: VariantContext) -> dict[str, Any]:
    return _nvlink_single(ctx, NVLINK["crc"])


def replay(ctx: VariantContext) -> dict[str, Any]:
    return _nvlink_single(ctx, NVLINK["replay"])


def recovery_counter(ctx: VariantContext) -> dict[str, Any]:
    return _nvlink_single(ctx, NVLINK["recovery_counter"])


def multiple_gpus(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    counter = survey.nvlink_counter(NVLINK["crc"])
    canonical = counter.canonical
    gpus = list(counter.gpus[:2])
    if len(gpus) < 2:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            f"fewer than two GPUs report NVLink counters ({canonical})",
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
        notes=[
            "multi-GPU NVLink correlation is diagnostic, never a fatal verdict",
            counter.note(),
        ],
        extra={"gpus": gpus, **counter.evidence()},
    )


def counter_reset(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    counter = survey.nvlink_counter(NVLINK["crc"])
    canonical = counter.canonical
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
        notes=[
            "a counter reset (decrease) is not a new fault or a recovery",
            counter.note(),
        ],
        extra={"gpu_uuid": gpu, **counter.evidence()},
    )


def stale_sample(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    counter = survey.nvlink_counter(NVLINK["crc"])
    canonical = counter.canonical
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
        notes=[
            "an unrefreshed (stale) scrape carries no new information",
            counter.note(),
        ],
        extra={"gpu_uuid": gpu, **counter.evidence()},
    )


def independent_fatal_event(ctx: VariantContext) -> dict[str, Any]:
    """A trusted kernel XID beside the NVLink counter fault keeps its own handling.

    The CRC counter is held on one GPU until its read-only diagnosis has an
    incident; then a reviewed XID 137 (NVLINK_PRIV_ERR, catalog IGNORE) enters
    through the kernel source naming that GPU's PCI BDF. The product must grade
    the XID from its own catalog row (MONITOR_ONLY, own RECOVERED incident, no
    workflow), neither merge it into the counter incident nor drop it, and the
    counter chain must stay the single read-only diagnostic it was.
    """

    entry = require_monitor_grade(INDEPENDENT_XID)
    survey = run_gpu_survey(ctx)
    counter = survey.nvlink_counter(NVLINK["crc"])
    canonical = counter.canonical
    gpu = survey.target(canonical)
    pci_bdf = kmsg_pci_bdf(survey.pci_bdf(gpu))
    if pci_bdf is None:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            f"{gpu} reports no PCI BDF; a kernel XID cannot name it",
        )
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan("nvlink", baseline(), step(40, add(canonical, [gpu]), hold=True)),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/{canonical}")
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: metric_incidents(doc, canonical, gpu=gpu),
        metrics=(canonical,),
        timeout_seconds=survey.health_summary + 240,
        name="detection",
    )
    if not metric_incidents(detection, canonical, gpu=gpu):
        stages = {"counter": [f"no {canonical} finding on {gpu} reached an incident"]}
        recover_counter(ctx, repl, since, names=[canonical])
        return result(
            stages,
            notes=[counter.note()],
            extra={"gpu_uuid": gpu, **counter.evidence()},
        )
    injection = inject_kernel_xid(
        ctx, xid=INDEPENDENT_XID, pci_bdf=pci_bdf, label="020-independent-event"
    )
    decisions = wait_xid_decisions(
        ctx, injection.started, xid=INDEPENDENT_XID, name="independent-xid"
    )
    injection.close()
    final = activity_now(ctx, since, metrics=(canonical,))
    counter_incidents = metric_incidents(final, canonical, gpu=gpu)
    stages = {
        "counter_chain": _diag_chain(final, canonical, gpu),
        "counter_read_only": read_only_errors(final, counter_incidents, canonical),
        "independent": independence_errors(
            decisions,
            entry=entry,
            gpu_uuid=gpu,
            foreign_incident_ids={
                str(item.get("incident_id")) for item in counter_incidents
            },
        ),
        "hardware": no_hardware(final),
    }
    stages["isolation"] = restore_if_isolated(ctx, decisions)
    recover_counter(ctx, repl, since, names=[canonical])
    return result(
        stages,
        notes=[
            f"XID {INDEPENDENT_XID} ({entry.mnemonic}) is monitor grade in catalog "
            f"{entry.catalog_version}: {entry.immediate_action} -> MONITOR_ONLY; the "
            "fatal-grade dispatch beside a counter diagnosis is not driven live here",
            counter.note(),
        ],
        extra={
            "gpu_uuid": gpu,
            "xid": INDEPENDENT_XID,
            "pci_bdf": pci_bdf,
            "decisions": decisions,
            "kernel_replay": injection.evidence(),
            **counter.evidence(),
        },
    )


def recovered(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    counter = survey.nvlink_counter(NVLINK["replay"])
    canonical = counter.canonical
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
    return result(
        stages, notes=[counter.note()], extra={"gpu_uuid": gpu, **counter.evidence()}
    )


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


PCIE = "pcie_replay_total"


def _when(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _values(samples: list[dict[str, Any]]) -> list[float]:
    return [
        item["value"]
        for item in samples
        if isinstance(item.get("value"), (int, float))
        and not isinstance(item.get("value"), bool)
    ]


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _fault_errors(
    state: dict[str, Any],
    activity: dict[str, Any],
    gpu: str,
    *,
    place: Callable[[dict[str, Any] | None], str],
    boundary: str,
) -> list[str]:
    """One error per ``pcie_replay_total`` finding on ``gpu`` -- an incident's
    or a history one's -- worded by ``place`` (where the finding sits relative
    to the boundary), plus any finding still active after the ``boundary``."""

    errors: list[str] = []
    seen: set[str] = set()
    for incident in metric_incidents(activity, PCIE, gpu=gpu):
        finding = incident_finding(state, incident)
        if finding is not None:
            seen.add(str(finding.get("finding_id")))
        errors.append(
            f"{place(finding)}: {finding_text(finding)}; "
            f"incident {incident.get('incident_id')}"
        )
    for finding in history_findings(state, PCIE, gpu=gpu):
        if str(finding.get("finding_id")) not in seen:
            errors.append(f"{place(finding)}: {finding_text(finding)}")
    for finding in active_findings(state, PCIE, gpu=gpu):
        errors.append(f"{boundary} left an active finding: {finding_text(finding)}")
    return errors


def pcie_counter_reset_stages(
    state: dict[str, Any], activity: dict[str, Any], gpu: str
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Grade a PCIe replay counter reset on the product's own records.

    ``raise``: the product accepted a reading above the first accepted one
    (the under-limit raise reached it). ``reset``: a later accepted reading
    fell below its predecessor, its batch is a ``counter-reset`` edge and the
    product's latest reading is that reset value -- the baseline re-armed from
    the drop, not held at the peak. ``no_fault``: no ``pcie_replay_total``
    finding or incident on the GPU, each one named by where it sits -- during
    the raise (the plan's own rate) or at/after the drop (the reset minted it)
    -- and none still active. The readings document is returned for the
    evidence.
    """

    samples = accepted_samples(state, PCIE, gpu)
    values = _values(samples)
    ticks = counter_reset_ticks(state, PCIE, gpu)
    latest = latest_value(state, PCIE, gpu)
    listed = ", ".join(_number(value) for value in values) or "none"
    raised: list[str] = []
    reset: list[str] = []
    if not values or max(values) <= values[0]:
        raised.append(
            f"the product accepted no raised {PCIE} reading of {gpu} "
            f"(accepted values: {listed})"
        )
    elif not ticks:
        reset.append(
            f"the product accepted no {PCIE} reading of {gpu} below its "
            f"predecessor (accepted values: {listed})"
        )
    else:
        tick = ticks[0]
        if "counter-reset" not in tick["edge_filter_reasons"]:
            reasons = ", ".join(tick["edge_filter_reasons"]) or "no reason"
            reset.append(
                f"the drop {_number(tick['previous'])} -> {_number(tick['value'])} "
                f"at {tick['observed_at']} was delivered without a counter-reset "
                f"edge reason ({reasons})"
            )
        latest_number = latest.get("value") if latest is not None else None
        if latest_number != tick["value"]:
            shown = (
                _number(latest_number)
                if isinstance(latest_number, (int, float))
                else "absent"
            )
            reset.append(
                f"the product's latest {PCIE} reading of {gpu} is {shown}, not the "
                f"reset value {_number(tick['value'])}: the baseline was not "
                "re-armed from the drop"
            )
    reset_at = _when(ticks[0]["observed_at"]) if ticks else None

    def place(finding: dict[str, Any] | None) -> str:
        when = _when(finding.get("observed_at")) if finding is not None else None
        if reset_at is not None and when is not None and when < reset_at:
            return (
                "the raise before the reset minted a finding (the plan's increment "
                "must stay under 8/min)"
            )
        return "a PCIe counter reset minted a finding"

    stages = {
        "raise": raised,
        "reset": reset,
        "no_fault": _fault_errors(
            state, activity, gpu, place=place, boundary="a counter reset"
        ),
        "hardware": no_hardware(activity),
    }
    readings = {
        "accepted": samples,
        "reset_ticks": ticks,
        "latest": latest,
        "findings": history_findings(state, PCIE, gpu=gpu),
    }
    return stages, readings


def pcie_counter_reset(ctx: VariantContext) -> dict[str, Any]:
    """A PCIe replay counter that falls below its previous reading is no fault.

    The raise before the drop stays clearly under the 8/min rule
    (``under_limit_increment``: one per read on a 15-s collector, 4/min), so
    the only reading that could mint a finding is the reset itself. Live
    2026-10-26 the plan raised by 50 per read (200/min) and the WARNING that
    legitimate storm minted was read as "a PCIe counter reset minted a
    finding", while the product's records showed the reset tick graded unknown
    and nothing minted at it. The stages read the product's own records
    (``pcie_counter_reset_stages``).
    """

    survey = run_gpu_survey(ctx)
    gpu = survey.target(PCIE)
    increment = under_limit_increment(survey.interval)
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "pcie",
            baseline(),
            step(3, add(PCIE, [gpu], increment)),
            step(20, clear(PCIE, [gpu]), hold=True),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/{PCIE}")
    # Two more raised reads, the drop, and the product's ingest of it.
    ctx.hold(min(max(120, 4 * repl.interval + 60), _hold(ctx)), name="reset-observe")
    state = state_now(ctx, since, names=[PCIE])
    activity = wait_gpu_incident(
        ctx,
        since,
        lambda d: [],
        metrics=(PCIE,),
        timeout_seconds=1,
        name="reset",
    )
    stages, readings = pcie_counter_reset_stages(state, activity, gpu)
    ctx.evidence.write("reset-readings", readings)
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=[
            f"raised by {increment} per read ({increment * 60 / repl.interval:g}/min, "
            "under the 8/min rule) then dropped: the drop is a counter-reset edge, "
            "graded unknown, and the new baseline -- never a fault or a recovery"
        ],
        extra={"gpu_uuid": gpu, "increment": increment},
    )


def source_boot_change_stages(
    state: dict[str, Any],
    activity: dict[str, Any],
    gpu: str,
    *,
    boot: str,
    jump: float | None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Grade a source/boot change on the product's own records.

    ``no_cross_boot_rate``: no ``pcie_replay_total`` finding or incident on the
    GPU (the only reading that could rate over the rule is the jump across the
    boundary). ``new_baseline``: the product accepted a reading under the
    replayed boot and its latest reading of the GPU is the new boot's first
    one -- the boot change started a new baseline instead of being refused.
    Raises ``VariantInconclusive`` when the new boot's first accepted reading
    is not the served jump: the boundary was crossed at the real value and was
    never exercised.
    """

    samples = accepted_samples(state, PCIE, gpu)
    new_boot = [
        item
        for item in samples
        if (item.get("labels") or {}).get("source_boot_id") == boot
    ]
    first = new_boot[0] if new_boot else None
    if first is not None and jump is not None and first.get("value") != jump:
        raise VariantInconclusive(
            f"the new boot's first accepted {PCIE} reading of {gpu} was "
            f"{_number(first['value'])}, not the replayed jump {_number(jump)}: the "
            "collector scraped the tool before the arm reached it, so the boundary "
            "was crossed at the real value and not exercised"
        )
    latest = latest_value(state, PCIE, gpu)
    baseline_errors: list[str] = []
    if first is None:
        baseline_errors.append(
            f"the product accepted no {PCIE} reading of {gpu} under the replayed "
            f"boot {boot}: the new boot's readings never became a baseline"
        )
    elif latest is None or (
        (latest.get("labels") or {}).get("source_boot_id") != boot
        or latest.get("value") != first.get("value")
    ):
        shown = (
            f"value={_number(latest['value'])} under boot "
            f"{(latest.get('labels') or {}).get('source_boot_id') or 'unknown'}"
            if latest is not None and isinstance(latest.get("value"), (int, float))
            else "absent"
        )
        baseline_errors.append(
            f"the product's latest {PCIE} reading of {gpu} is {shown}, not the new "
            f"boot's first reading ({_number(first['value'])} under {boot}): the boot "
            "change did not start a new baseline"
        )
    stages = {
        "no_cross_boot_rate": _fault_errors(
            state,
            activity,
            gpu,
            place=lambda _finding: (
                "a source/boot change formed a cross-generation PCIe rate"
            ),
            boundary="a source/boot change",
        ),
        "new_baseline": baseline_errors,
        "hardware": no_hardware(activity),
    }
    readings = {
        "accepted": samples,
        "first_new_boot_reading": first,
        "latest": latest,
        "findings": history_findings(state, PCIE, gpu=gpu),
    }
    return stages, readings


def source_boot_change(ctx: VariantContext) -> dict[str, Any]:
    """A source/boot change forms no cross-generation PCIe rate and starts a
    new baseline.

    The plan's first read is a jump over the 8/min rule against any reading
    of the real boot (``cross_boot_jump``), held afterwards; the proxy is
    armed *before* the collector is repointed (``arm_before_repoint``), so the
    repointed collector's first scrape -- the product's first reading under
    the replayed boot id -- already carries the jump. A product that rated
    across the boundary would mint; one that starts a new baseline must not,
    and its latest reading must be the new boot's first. Live 2026-10-26 the
    plan ramped +5 per read (20/min) after a pass-through first scrape: the
    boundary was (0, real boot) -> (0, replayed boot) and the ramp inside the
    new boot was the storm the oracle blamed on the boot change.
    """

    from scripts.e2e.regional.general_live_gpu_replay import replay_boot_id

    survey = run_gpu_survey(ctx)
    gpu = survey.target(PCIE)
    boot, origin = replay_boot_id(survey, ctx.run_id)
    jump = cross_boot_jump(survey.interval, survey.health_summary)
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan("pcie", step(1, add(PCIE, [gpu], jump)), step(30, hold=True)),
        _hold(ctx),
        boot_id=boot,
    )
    key = f"{gpu}/{PCIE}"
    arm_before_repoint(repl, key)
    ctx.hold(min(180, _hold(ctx)), name="boot-observe")
    state = state_now(ctx, since, names=[PCIE])
    activity = wait_gpu_incident(
        ctx,
        since,
        lambda d: [],
        metrics=(PCIE,),
        timeout_seconds=1,
        name="boot",
    )
    served = [value for value in served_values(repl.records, key) if value is not None]
    stages, readings = source_boot_change_stages(
        state, activity, gpu, boot=boot, jump=served[0] if served else None
    )
    ctx.evidence.write("boot-readings", {**readings, "served": served})
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=[
            f"replayed boot id from {origin}; the new boot's first reading is a jump "
            f"of {jump} (over 8/min against any reading of the real boot), served "
            "before the repointed collector's first scrape; a boot change starts a "
            "new baseline and forms no cross-generation rate"
        ],
        extra={"gpu_uuid": gpu, "boot_id": boot, "jump": jump},
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
        f"increment {CRC_COUNTER}",
        "WARNING RUN_DIAGNOSTICS; read-only, no reset",
        crc,
        max_minutes=20,
    ),
    gpu_spec(
        CASE_020,
        "replay",
        "increment the NVLink replay counter (per-class total on NVLink3, DL "
        "aggregate on NVLink4+)",
        "WARNING RUN_DIAGNOSTICS; read-only",
        replay,
        max_minutes=20,
    ),
    gpu_spec(
        CASE_020,
        "recovery_counter",
        "increment the NVLink recovery counter (per-class total on NVLink3, DL "
        "aggregate on NVLink4+)",
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
    gpu_spec(
        CASE_020,
        "independent_fatal_event",
        f"hold {CRC_COUNTER} on one GPU, then a reviewed trusted kernel "
        f"XID {INDEPENDENT_XID} (NVLINK_PRIV_ERR, catalog IGNORE) on that GPU through "
        "the kmsg source",
        "the XID keeps its own catalog decision (MONITOR_ONLY, own RECOVERED incident, "
        "no workflow) and is not absorbed by the counter diagnosis, which stays a single "
        "read-only chain. The fatal-grade dispatch (RESET/QUARANTINE) is deliberately "
        "not driven live",
        independent_fatal_event,
        risk="M",
        max_minutes=30,
        approvals=(KERNEL_SOURCE_REPLAY,),
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
        "raise the PCIe replay counter under 8/min, then drop it",
        "a counter reset is no new fault: the drop is the product's counter-reset "
        "edge, graded unknown, and its new baseline",
        pcie_counter_reset,
        max_minutes=18,
    ),
    gpu_spec(
        CASE_022,
        "source_boot_change",
        "a PCIe replay jump over 8/min as the first reading under a replayed "
        "source boot id (armed before the repoint)",
        "a source/boot change forms no cross-generation rate; the new boot's first "
        "reading is the new baseline",
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
