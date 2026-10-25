"""GF-GENERAL-LIVE-024 DCGM composite findings (live-replay).

The composites correlate per-GPU DCGM component findings within the 45 s
window (``gpu_metrics._evaluate_composite_findings``). This module drives the
memory and NVLink components through the ``composite`` profile and checks the
product's real correlation and de-duplication, not just a rule name:

* ``corrected_memory`` -- two correctable memory signals on one GPU form
  CORRECTABLE_MEMORY_DEGRADATION (diagnostic);
* ``memory_failure`` -- an uncorrectable error plus pending repair form a
  well-founded CRITICAL GPU_MEMORY_DEGRADATION that is not weakened, and whose
  components are suppressed into it (no separate reset);
* ``nvlink_multiple_gpus`` -- one NVLink class on two GPUs forms the node-level
  MULTI_GPU_NVLINK_FABRIC_FAILURE (diagnostic);
* ``fresh_components`` -- a composite forms from components activated this batch;
* ``reused_components`` -- holding the same components does not re-mint it;
* ``wrong_scope_source`` -- components on different GPUs do not compose;
* ``independent_xid`` -- a trusted kernel XID on the GPU of a weak composite
  keeps its own catalog decision and incident; the composite never covers it
  (``general_live_kernel_xid``: XID 92 EXCESSIVE_SBE_INTERRUPTS, catalog
  IGNORE -> MONITOR_ONLY, beside CORRECTABLE_MEMORY_DEGRADATION).

``thermal_warning`` / ``power_performance`` reuse the GF-ENV-LIVE environment
replay (temperature / clock / power are on the environment proxy's allow-list,
not this lane's DCGM field tool) and live in ``general_live_gpu_environment``.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    activity_now,
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
    composite_component_errors,
    composite_findings,
    composite_incidents,
    read_only_errors,
)
from scripts.e2e.regional.general_live_kernel_xid import (
    independence_errors,
    inject_kernel_xid,
    kmsg_pci_bdf,
    require_monitor_grade,
    restore_if_isolated,
    wait_xid_decisions,
)

CASE = "GF-GENERAL-LIVE-024"
RESET = frozenset(
    {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "REPLACE_NODE"}
)
#: The reviewed trusted kernel XID beside the weak memory composite: catalog
#: IGNORE (-> NO_ACTION / MONITOR_ONLY), fault class GPU_MEMORY like the
#: correctable-memory components it must not be folded into.
INDEPENDENT_XID = 92
CORRECTABLE_RULE = "CORRECTABLE_MEMORY_DEGRADATION"
CORRECTABLE_COMPONENTS = ("ecc_sbe_volatile_total", "row_remap_correctable_total")
KERNEL_SOURCE_REPLAY = "KERNEL_SOURCE_REPLAY"


def _hold(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, 20 * 60)


def corrected_memory(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("ecc_sbe_volatile_total")
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "composite",
            baseline(),
            step(
                40,
                add("ecc_sbe_volatile_total", [gpu]),
                add("row_remap_correctable_total", [gpu]),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(
        repl, f"{gpu}/ecc_sbe_volatile_total", f"{gpu}/row_remap_correctable_total"
    )
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, "CORRECTABLE_MEMORY_DEGRADATION"),
        timeout_seconds=survey.health_summary + 300,
        name="detection",
    )
    stages = {
        "chain": chain_errors(
            detection,
            composite_incidents(detection, "CORRECTABLE_MEMORY_DEGRADATION"),
            GpuChain(
                label="CORRECTABLE_MEMORY_DEGRADATION",
                operations=("FREEZE_EVIDENCE",),
                forbidden=RESET,
            ),
        ),
        "read_only": read_only_errors(
            detection,
            composite_incidents(detection, "CORRECTABLE_MEMORY_DEGRADATION"),
            "corrected_memory",
        ),
        "hardware": no_hardware(detection),
    }
    recover_counter(
        ctx,
        repl,
        since,
        names=["ecc_sbe_volatile_total", "row_remap_correctable_total"],
    )
    return result(stages, extra={"gpu_uuid": gpu})


def memory_failure(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("ecc_dbe_volatile_total")
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "composite",
            baseline(),
            step(
                60,
                add("ecc_dbe_volatile_total", [gpu]),
                put("row_remap_pending", [gpu], 1),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/ecc_dbe_volatile_total")
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, "GPU_MEMORY_DEGRADATION"),
        timeout_seconds=survey.health_summary + 300,
        name="detection",
    )
    incidents = composite_incidents(detection, "GPU_MEMORY_DEGRADATION")
    stages = {
        "chain": chain_errors(
            detection,
            incidents,
            GpuChain(
                label="GPU_MEMORY_DEGRADATION",
                operations=("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"),
                forbidden=RESET,
                actions=frozenset({"QUARANTINE", "DRAIN"}),
            ),
        ),
        "not_weakened": []
        if incidents
        else [
            "a well-founded memory failure was covered by a weaker composite or dropped"
        ],
        "hardware": no_hardware(detection),
    }
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    stages["recovery"] = _validated_restore(ctx, incidents)
    return result(
        stages,
        notes=[
            "a well-founded memory failure keeps its CRITICAL isolation; components suppressed into it"
        ],
        extra={"gpu_uuid": gpu},
    )


def nvlink_multiple_gpus(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = survey.gpus_with("nvlink_crc_flit_error_total")[:2]
    if len(gpus) < 2:
        raise VariantNotRun(
            "CAPABILITY_ABSENT", "fewer than two GPUs report NVLink counters"
        )
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "composite",
            baseline(),
            step(40, add("nvlink_crc_flit_error_total", gpus), hold=True),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, *[f"{gpu}/nvlink_crc_flit_error_total" for gpu in gpus])
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, "MULTI_GPU_NVLINK_FABRIC_FAILURE"),
        timeout_seconds=survey.health_summary + 300,
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
    recover_counter(ctx, repl, since, names=["nvlink_crc_flit_error_total"])
    return result(stages, extra={"gpus": gpus})


def fresh_components(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("ecc_sbe_volatile_total")
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "composite",
            baseline(),
            step(
                40,
                add("ecc_sbe_volatile_total", [gpu]),
                add("retired_pages_sbe_total", [gpu]),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(
        repl, f"{gpu}/ecc_sbe_volatile_total", f"{gpu}/retired_pages_sbe_total"
    )
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, "CORRECTABLE_MEMORY_DEGRADATION"),
        timeout_seconds=survey.health_summary + 300,
        name="detection",
    )
    stages = {
        "fresh": []
        if composite_incidents(detection, "CORRECTABLE_MEMORY_DEGRADATION")
        else ["fresh components did not form a composite"],
        "hardware": no_hardware(detection),
    }
    recover_counter(
        ctx, repl, since, names=["ecc_sbe_volatile_total", "retired_pages_sbe_total"]
    )
    return result(stages, extra={"gpu_uuid": gpu})


def reused_components(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("ecc_sbe_volatile_total")
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "composite",
            baseline(),
            step(
                80,
                add("ecc_sbe_volatile_total", [gpu]),
                add("retired_pages_sbe_total", [gpu]),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/ecc_sbe_volatile_total")
    wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, "CORRECTABLE_MEMORY_DEGRADATION"),
        timeout_seconds=survey.health_summary + 300,
        name="detection",
    )
    ctx.hold(min(survey.health_summary + 60, _hold(ctx)), name="reuse-observe")
    final = wait_gpu_incident(
        ctx, since, lambda d: [], timeout_seconds=1, name="reuse-check"
    )
    incidents = composite_incidents(final, "CORRECTABLE_MEMORY_DEGRADATION")
    stages = {
        "single": []
        if len(incidents) <= 1
        else [f"reused components minted {len(incidents)} composite incidents"],
        "hardware": no_hardware(final),
    }
    recover_counter(
        ctx, repl, since, names=["ecc_sbe_volatile_total", "retired_pages_sbe_total"]
    )
    return result(
        stages,
        notes=[
            "sustained (reused) components keep one composite incident, not a new one each batch"
        ],
        extra={"gpu_uuid": gpu},
    )


def wrong_scope_source(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = survey.gpus_with("ecc_dbe_volatile_total")[:2]
    if len(gpus) < 2:
        raise VariantNotRun(
            "CAPABILITY_ABSENT", "fewer than two GPUs report memory fields"
        )
    error_gpu, repair_gpu = gpus[0], gpus[1]
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "composite",
            baseline(),
            step(
                60,
                add("ecc_dbe_volatile_total", [error_gpu]),
                put("row_remap_pending", [repair_gpu], 1),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{error_gpu}/ecc_dbe_volatile_total")
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, "GPU_MEMORY_DEGRADATION")
        or _any_incident(doc, error_gpu),
        timeout_seconds=survey.health_summary + 300,
        name="detection",
    )
    stages = {
        "no_cross_gpu_composite": []
        if not composite_incidents(detection, "GPU_MEMORY_DEGRADATION")
        else ["an error on one GPU composed with repair on another GPU"],
        "hardware": no_hardware(detection),
    }
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    stages["recovery"] = _validated_restore(ctx, _any_incident(detection, error_gpu))
    return result(
        stages,
        notes=[
            "GPU_MEMORY_DEGRADATION is per-GPU; components on different GPUs do not compose"
        ],
        extra={"error_gpu": error_gpu, "repair_gpu": repair_gpu},
    )


def independent_xid(ctx: VariantContext) -> dict[str, Any]:
    """A trusted kernel XID on the GPU of a weak composite is not covered by it.

    The weak composite is the diagnostic CORRECTABLE_MEMORY_DEGRADATION held
    on one GPU; while it is active, a reviewed XID 92 (EXCESSIVE_SBE_INTERRUPTS,
    the same GPU_MEMORY class, catalog IGNORE) enters through the kernel source
    naming that GPU's PCI BDF. The product must grade the XID from its own
    catalog row (MONITOR_ONLY, own RECOVERED incident, no workflow), never merge
    it into the composite's incident or list it among the composite's
    components, and the composite must stay the read-only diagnostic it was.
    """

    entry = require_monitor_grade(INDEPENDENT_XID)
    survey = run_gpu_survey(ctx)
    gpu = survey.target("ecc_sbe_volatile_total")
    pci_bdf = kmsg_pci_bdf(survey.pci_bdf(gpu))
    if pci_bdf is None:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            f"{gpu} reports no PCI BDF; a kernel XID cannot name it",
        )
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "composite",
            baseline(),
            step(
                40,
                add("ecc_sbe_volatile_total", [gpu]),
                add("row_remap_correctable_total", [gpu]),
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(
        repl, f"{gpu}/ecc_sbe_volatile_total", f"{gpu}/row_remap_correctable_total"
    )
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, CORRECTABLE_RULE),
        timeout_seconds=survey.health_summary + 300,
        name="detection",
    )
    if not composite_incidents(detection, CORRECTABLE_RULE):
        # Without the weak composite there is nothing to be covered by; the
        # counters are released and the variant reports the missing composite.
        stages = {"composite": [f"no {CORRECTABLE_RULE} composite reached an incident"]}
        recover_counter(ctx, repl, since, names=list(CORRECTABLE_COMPONENTS))
        return result(stages, extra={"gpu_uuid": gpu})
    injection = inject_kernel_xid(
        ctx, xid=INDEPENDENT_XID, pci_bdf=pci_bdf, label="024-independent-xid"
    )
    decisions = wait_xid_decisions(
        ctx, injection.started, xid=INDEPENDENT_XID, name="independent-xid"
    )
    injection.close()
    final = activity_now(ctx, since, metrics=CORRECTABLE_COMPONENTS)
    state = state_now(ctx, since, names=CORRECTABLE_COMPONENTS)
    composites = composite_incidents(final, CORRECTABLE_RULE)
    stages = {
        "composite_chain": chain_errors(
            final,
            composites,
            GpuChain(
                label=CORRECTABLE_RULE,
                operations=("FREEZE_EVIDENCE",),
                forbidden=RESET,
            ),
        ),
        "composite_read_only": read_only_errors(final, composites, CORRECTABLE_RULE),
        "independent": independence_errors(
            decisions,
            entry=entry,
            gpu_uuid=gpu,
            foreign_incident_ids={str(item.get("incident_id")) for item in composites},
        ),
        "not_covered": composite_component_errors(
            composite_findings(state, CORRECTABLE_RULE, gpu=gpu),
            rule_id=CORRECTABLE_RULE,
            allowed_components=CORRECTABLE_COMPONENTS,
        ),
        "hardware": no_hardware(final),
    }
    stages["isolation"] = restore_if_isolated(ctx, decisions)
    recover_counter(ctx, repl, since, names=list(CORRECTABLE_COMPONENTS))
    return result(
        stages,
        notes=[
            f"XID {INDEPENDENT_XID} ({entry.mnemonic}) is monitor grade in catalog "
            f"{entry.catalog_version}: {entry.immediate_action} -> MONITOR_ONLY; the "
            "fatal-grade dispatch beside a weak composite is not driven live here",
        ],
        extra={
            "gpu_uuid": gpu,
            "xid": INDEPENDENT_XID,
            "pci_bdf": pci_bdf,
            "decisions": decisions,
            "kernel_replay": injection.evidence(),
        },
    )


def _any_incident(doc: dict[str, Any], gpu: str) -> list[dict[str, Any]]:
    from scripts.e2e.regional.general_live_gpu_verdicts import metric_incidents

    return metric_incidents(doc, "ecc_dbe_volatile_total", gpu=gpu)


def _validated_restore(
    ctx: VariantContext, incidents: list[dict[str, Any]]
) -> list[str]:
    if not incidents:
        return []
    from scripts.e2e.regional.general_live_common import restore_incidents

    try:
        restore_incidents(ctx, incidents, reason=f"{ctx.spec.key} acceptance restore")
    except Exception as exc:  # noqa: BLE001 - recorded as a cleanup error
        return [f"validated restore failed: {type(exc).__name__}: {exc}"]
    return []


VARIANTS = [
    gpu_spec(
        CASE,
        "corrected_memory",
        "two correctable memory signals on one GPU",
        "CORRECTABLE_MEMORY_DEGRADATION diagnostic; no reset",
        corrected_memory,
        max_minutes=20,
    ),
    gpu_spec(
        CASE,
        "memory_failure",
        "an uncorrectable error plus pending repair on one GPU",
        "CRITICAL GPU_MEMORY_DEGRADATION, not weakened; isolation only",
        memory_failure,
        risk="M",
        max_minutes=25,
    ),
    gpu_spec(
        CASE,
        "nvlink_multiple_gpus",
        "one NVLink class on two GPUs",
        "MULTI_GPU_NVLINK_FABRIC_FAILURE diagnostic",
        nvlink_multiple_gpus,
        max_minutes=20,
    ),
    gpu_spec(
        CASE,
        "fresh_components",
        "components activated in the same batch",
        "a composite forms from fresh components",
        fresh_components,
        max_minutes=20,
    ),
    gpu_spec(
        CASE,
        "reused_components",
        "hold the same components across batches",
        "the composite is not re-minted for reused components",
        reused_components,
        max_minutes=22,
    ),
    gpu_spec(
        CASE,
        "wrong_scope_source",
        "an error on one GPU, repair on another",
        "components on different GPUs do not compose",
        wrong_scope_source,
        risk="M",
        max_minutes=22,
    ),
    gpu_spec(
        CASE,
        "independent_xid",
        "hold CORRECTABLE_MEMORY_DEGRADATION on one GPU, then a reviewed trusted "
        f"kernel XID {INDEPENDENT_XID} (EXCESSIVE_SBE_INTERRUPTS, catalog IGNORE) on "
        "that GPU through the kmsg source",
        "the XID keeps its own catalog decision (MONITOR_ONLY, own RECOVERED incident, "
        "no workflow) and is neither merged into the weak composite's incident nor "
        "listed among its components; the composite stays diagnostic. The fatal-grade "
        "dispatch (RESET/QUARANTINE) is deliberately not driven live",
        independent_xid,
        risk="M",
        max_minutes=30,
        approvals=(KERNEL_SOURCE_REPLAY,),
    ),
]
