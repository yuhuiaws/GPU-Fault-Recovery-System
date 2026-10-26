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
* ``wrong_scope_source`` -- components on different GPUs do not compose,
  judged on the product's records (incidents and findings per GPU); the
  repair GPU's own pending-repair finding may run the product's authorized
  single-GPU service-window reset (``GPU_RESET`` approval, hardware judged per
  incident) and must settle RECOVERED by itself, so the pair is served for one
  correlation window only and the plan returns to the real readings before
  the reset is validated (``window_reads``);
* ``independent_xid`` -- a trusted kernel XID on the GPU of a weak composite
  keeps its own catalog decision and incident; the composite never covers it
  (``general_live_kernel_xid``: XID 92 EXCESSIVE_SBE_INTERRUPTS, catalog
  IGNORE -> MONITOR_ONLY, beside CORRECTABLE_MEMORY_DEGRADATION).

The memory composites replay two readings of one GPU, and the exporter decides
which second reading exists (``GpuSurvey.component_pair``): beside the volatile
SBE counter, ``retired_pages_sbe_total`` where the part still retires pages
(pre-Hopper), else ``row_remap_correctable_total`` (Hopper exposes no page
retirement at all -- live 2026-10-26, p5en H200, ``fresh_components`` ended
INCONCLUSIVE "targeted reading absent upstream; refusing invented samples");
beside the volatile DBE counter, the ``row_remap_pending`` flag where
published, else ``retired_pages_pending``. The chosen component is in every
result's ``second_component`` / ``component_metrics`` / ``exporter_metrics``
and notes, and a plan that names a reading the survey does not show is refused
before anything is installed on the node (``composite_replay``:
``NOT_RUN(CAPABILITY_ABSENT)``, never INCONCLUSIVE mid-run).

``thermal_warning`` / ``power_performance`` reuse the GF-ENV-LIVE environment
replay (temperature / clock / power are on the environment proxy's allow-list,
not this lane's DCGM field tool) and live in ``general_live_gpu_environment``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import (
    ComponentPair,
    DcgmReplay,
    GpuSurvey,
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
from scripts.e2e.regional.general_live_gpu_replay_close import wait_served_or_records
from scripts.e2e.regional.general_live_gpu_verdicts import (
    GpuChain,
    chain_errors,
    composite_component_errors,
    composite_findings,
    composite_incidents,
    metric_incidents,
    node_hardware_errors,
    read_only_errors,
    settled,
)
from scripts.e2e.regional.general_live_kernel_xid import (
    independence_errors,
    inject_kernel_xid,
    kmsg_pci_bdf,
    require_monitor_grade,
    restore_if_isolated,
    wait_xid_decisions,
)
from scripts.e2e.regional.general_live_verdicts import workflows_of

CASE = "GF-GENERAL-LIVE-024"
RESET = frozenset(
    {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "REPLACE_NODE"}
)
#: The reviewed trusted kernel XID beside the weak memory composite: catalog
#: IGNORE (-> NO_ACTION / MONITOR_ONLY), fault class GPU_MEMORY like the
#: correctable-memory components it must not be folded into.
INDEPENDENT_XID = 92
CORRECTABLE_RULE = "CORRECTABLE_MEMORY_DEGRADATION"
MEMORY_RULE = "GPU_MEMORY_DEGRADATION"
#: The anchor of the correctable-memory composite: every exporter generation
#: publishes the volatile SBE counter.
CORRECTABLE_PRIMARY = "ecc_sbe_volatile_total"
#: Its second component, in the order a plan prefers it. The product's rule
#: (``gpu_metrics._add_memory_composites``) forms CORRECTABLE_MEMORY_DEGRADATION
#: from any two distinct members of {ecc_sbe_volatile_total,
#: ecc_sbe_aggregate_total, retired_pages_sbe_total, row_remap_correctable_total};
#: the exporter decides which the node has: the retired-page counter
#: (``DCGM_FI_DEV_RETIRED_SBE``) where the part still retires pages
#: (pre-Hopper), else the correctable row-remap counter
#: (``DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS``, Ampere and later). Hopper exposes
#: no page retirement at all: live 2026-10-26 (plain EKS, p5en H200, R13g)
#: ``fresh_components`` ended INCONCLUSIVE "the replay aborted: targeted reading
#: absent upstream; refusing invented samples: GPU-.../retired_pages_sbe_total"
#: after it was armed, and 019 ``retired_delta`` is CAPABILITY_ABSENT there.
CORRECTABLE_COMPANIONS = ("retired_pages_sbe_total", "row_remap_correctable_total")
#: The error side of GPU_MEMORY_DEGRADATION and the pending-repair gauge beside
#: it (its repair side; ``gpu_memory_policy.memory_decision`` grades either
#: flag at 1 a WARNING finding on every architecture, so the composite forms
#: and keeps the error side's QUARANTINE): the row-remap flag
#: (``DCGM_FI_DEV_ROW_REMAP_PENDING``, Ampere and later) where published, else
#: the page-retirement flag (``DCGM_FI_DEV_RETIRED_PENDING``, older parts).
UNCORRECTABLE_PRIMARY = "ecc_dbe_volatile_total"
REPAIR_COMPANIONS = ("row_remap_pending", "retired_pages_pending")
#: The hardware operations of the product's authorized single-GPU
#: service-window reset chain (its other steps -- the containment,
#: VERIFY_NO_GPU_CLIENTS, RESTORE_GPU_SERVICES, VALIDATE_GPU, RESTORE_SCHEDULING
#: -- are not ``HARDWARE_OPERATIONS``). A pending-repair flag on a modern GPU is
#: graded WARNING RESET_GPU ("reset in a service window",
#: ``gpu_memory_policy``), so in ``wrong_scope_source`` the repair GPU's own
#: incident runs this chain -- live 2026-10-26 07:47-07:50Z (eksb, R13g) it
#: did, every step SUCCEEDED and the incident RECOVERED by itself, while the
#: variant's unscoped hardware oracle failed on the plan although no cross-GPU
#: composite had formed. The variant therefore declares ``GPU_RESET`` like 019
#: ``pending_supported`` and judges hardware per incident. The chain's
#: post-reset VALIDATE_GPU re-grades the fresh readings: a plan still holding
#: ``row_remap_pending=1`` fails it and the product escalates QUARANTINE ->
#: RESTART_NODE (live 2026-10-26 14:54-15:06Z, HyperPod, R13g: the quiesce had
#: stopped kubelet and with it the probe that would have closed the replay),
#: so the plan is bounded (``window_reads``) and never relies on the runner.
SERVICE_WINDOW_RESET = frozenset({"QUIESCE_GPU_SERVICES", "RESET_GPU"})
KERNEL_SOURCE_REPLAY = "KERNEL_SOURCE_REPLAY"
#: The product correlates component findings into a composite within this
#: window (``gpu_metrics.GpuMetricThresholds.correlation_window_seconds``,
#: default 45): a pair served for one whole window that did not compose never
#: will, so a scoping plan serves its pair that long and no longer.
COMPOSITE_WINDOW_SECONDS = 45


def _hold(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, 20 * 60)


def window_reads(interval: int, seconds: int = COMPOSITE_WINDOW_SECONDS) -> int:
    """The collector reads that cover ``seconds`` at ``interval`` s each (at
    least two), the bound of a plan step that must end by itself.

    The reset chain a replayed pending-repair flag starts quiesces the node
    about 20 s after the mint and completes RESET_GPU 70-90 s after it (live
    2026-10-26, both clusters); a bounded step of one correlation window
    (three reads of a 15-s collector) has the tool back on the real readings
    before the post-reset VALIDATE_GPU reads them, whether or not the runner
    can still reach the node, and the product's own grading happens on the
    first replayed scrape (both incidents 4 s after it, live).
    """

    return max(2, -(-int(seconds) // max(1, int(interval))))


def composite_replay(
    ctx: VariantContext, survey: GpuSurvey, *steps: dict[str, Any]
) -> DcgmReplay:
    """A ``composite``-profile replay of ``steps`` after the baseline reads.

    Preflight: every reading a step targets must be in the survey. The replay
    tool derives each served sample from the real upstream one and refuses to
    invent a sample for a field the exporter does not publish -- an abort
    after ``arm`` that ends the variant INCONCLUSIVE with nothing proved. A
    plan naming such a reading is refused here, before anything is installed
    on the node, as ``NOT_RUN(CAPABILITY_ABSENT)`` listing each one.
    """

    document = plan("composite", baseline(), *steps)
    missing = sorted(
        {
            f"{gpu}/{operation['field']}"
            for item in document["steps"]
            for operation in item["ops"]
            if "field" in operation
            for gpu in operation["gpus"]
            if not survey.has(gpu, operation["field"])
        }
    )
    if missing:
        raise VariantNotRun(
            "CAPABILITY_ABSENT", "the node does not report " + ", ".join(missing)
        )
    return DcgmReplay(ctx, document, _hold(ctx), node_boot_id=survey.boot_id)


def _correctable_replay(
    ctx: VariantContext, survey: GpuSurvey, pair: ComponentPair, *, reads: int
) -> DcgmReplay:
    """Both components of ``pair`` incremented in one step and held."""

    return composite_replay(
        ctx,
        survey,
        step(reads, *(add(name, [pair.gpu]) for name in pair.components), hold=True),
    )


def corrected_memory(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    pair = survey.component_pair(CORRECTABLE_PRIMARY, CORRECTABLE_COMPANIONS)
    since = utc_now()
    repl = _correctable_replay(ctx, survey, pair, reads=40)
    arm_and_confirm(repl, *pair.keys())
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, CORRECTABLE_RULE),
        timeout_seconds=survey.health_summary + 300,
        name="detection",
    )
    stages = {
        "chain": chain_errors(
            detection,
            composite_incidents(detection, CORRECTABLE_RULE),
            GpuChain(
                label=CORRECTABLE_RULE,
                operations=("FREEZE_EVIDENCE",),
                forbidden=RESET,
            ),
        ),
        "read_only": read_only_errors(
            detection,
            composite_incidents(detection, CORRECTABLE_RULE),
            "corrected_memory",
        ),
        "hardware": no_hardware(detection),
    }
    recover_counter(ctx, repl, since, names=list(pair.components))
    return result(stages, notes=[pair.note()], extra=pair.evidence())


def memory_failure(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    pair = survey.component_pair(UNCORRECTABLE_PRIMARY, REPAIR_COMPANIONS)
    gpu = pair.gpu
    since = utc_now()
    repl = composite_replay(
        ctx,
        survey,
        step(
            60,
            add(pair.primary, [gpu]),
            put(pair.companion, [gpu], 1),
            hold=True,
        ),
    )
    arm_and_confirm(repl, f"{gpu}/{pair.primary}")
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, MEMORY_RULE),
        timeout_seconds=survey.health_summary + 300,
        name="detection",
    )
    incidents = composite_incidents(detection, MEMORY_RULE)
    stages = {
        "chain": chain_errors(
            detection,
            incidents,
            GpuChain(
                label=MEMORY_RULE,
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
            "a well-founded memory failure keeps its CRITICAL isolation; components suppressed into it",
            pair.note(),
        ],
        extra=pair.evidence(),
    )


def nvlink_multiple_gpus(ctx: VariantContext) -> dict[str, Any]:
    # The CRC family member this node reports (the DL aggregate on Hopper/NVLink4+,
    # the flit counter on older parts); the product grades both alike.
    survey = run_gpu_survey(ctx)
    counter = survey.nvlink_counter("nvlink_crc")
    canonical = counter.canonical
    gpus = list(counter.gpus[:2])
    if len(gpus) < 2:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            f"fewer than two GPUs report NVLink counters ({canonical})",
        )
    since = utc_now()
    repl = composite_replay(ctx, survey, step(40, add(canonical, gpus), hold=True))
    arm_and_confirm(repl, *[f"{gpu}/{canonical}" for gpu in gpus])
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
    recover_counter(ctx, repl, since, names=[canonical])
    return result(
        stages, notes=[counter.note()], extra={"gpus": gpus, **counter.evidence()}
    )


def fresh_components(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    pair = survey.component_pair(CORRECTABLE_PRIMARY, CORRECTABLE_COMPANIONS)
    since = utc_now()
    repl = _correctable_replay(ctx, survey, pair, reads=40)
    arm_and_confirm(repl, *pair.keys())
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, CORRECTABLE_RULE),
        timeout_seconds=survey.health_summary + 300,
        name="detection",
    )
    stages = {
        "fresh": []
        if composite_incidents(detection, CORRECTABLE_RULE)
        else ["fresh components did not form a composite"],
        "hardware": no_hardware(detection),
    }
    recover_counter(ctx, repl, since, names=list(pair.components))
    return result(stages, notes=[pair.note()], extra=pair.evidence())


def reused_components(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    pair = survey.component_pair(CORRECTABLE_PRIMARY, CORRECTABLE_COMPANIONS)
    since = utc_now()
    repl = _correctable_replay(ctx, survey, pair, reads=80)
    arm_and_confirm(repl, f"{pair.gpu}/{pair.primary}")
    wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, CORRECTABLE_RULE),
        timeout_seconds=survey.health_summary + 300,
        name="detection",
    )
    ctx.hold(min(survey.health_summary + 60, _hold(ctx)), name="reuse-observe")
    final = wait_gpu_incident(
        ctx, since, lambda d: [], timeout_seconds=1, name="reuse-check"
    )
    incidents = composite_incidents(final, CORRECTABLE_RULE)
    stages = {
        "single": []
        if len(incidents) <= 1
        else [f"reused components minted {len(incidents)} composite incidents"],
        "hardware": no_hardware(final),
    }
    recover_counter(ctx, repl, since, names=list(pair.components))
    return result(
        stages,
        notes=[
            "sustained (reused) components keep one composite incident, not a new one each batch",
            pair.note(),
        ],
        extra=pair.evidence(),
    )


def scoped_hardware_errors(
    activity: dict[str, Any],
    *,
    repair_incidents: Sequence[dict[str, Any]],
    allowed: Iterable[str] = SERVICE_WINDOW_RESET,
) -> list[str]:
    """Hardware operations judged per incident.

    The workflows of ``repair_incidents`` (the repair GPU's own pending-repair
    finding) may plan the operations in ``allowed`` -- the authorized
    service-window reset chain -- and nothing else; every other workflow on
    the node, the error GPU's isolation first, plans no hardware operation.
    """

    repair_ids = {str(item.get("incident_id")) for item in repair_incidents}
    workflows = list(activity.get("workflows") or [])
    own = [item for item in workflows if str(item.get("incident_id")) in repair_ids]
    others = [
        item for item in workflows if str(item.get("incident_id")) not in repair_ids
    ]
    return [
        *node_hardware_errors({"workflows": others}),
        *node_hardware_errors({"workflows": own}, allowed=allowed),
    ]


def repair_settled_errors(
    activity: dict[str, Any],
    incidents: Sequence[dict[str, Any]],
    *,
    gpu: str,
    component: str,
) -> list[str]:
    """The repair GPU's own incident ended through its own chain.

    Every workflow of it is terminal and the incident is RECOVERED (the chain's
    RESTORE_SCHEDULING released it); the variant never restores that incident
    itself, so a chain that stalled or parked the record is this stage's error
    and nothing is retried.
    """

    if not incidents:
        return [f"the replayed {component} on {gpu} minted no incident of its own"]
    errors = []
    if not settled(activity, incidents):
        statuses = sorted(
            {
                str(workflow.get("status"))
                for incident in incidents
                for workflow in workflows_of(activity, str(incident["incident_id"]))
            }
        )
        errors.append(
            f"the {component} incident on {gpu} did not settle through its own "
            f"chain (workflow statuses {statuses or ['none']})"
        )
    current = {
        str(item.get("incident_id")): item for item in activity.get("incidents") or []
    }
    for incident in incidents:
        incident_id = str(incident["incident_id"])
        state = str(current.get(incident_id, incident).get("state"))
        if state != "RECOVERED":
            errors.append(
                f"the {component} incident {incident_id} on {gpu} is {state}, not "
                "RECOVERED by its own chain"
            )
    return errors


def scoping_errors(
    activity: dict[str, Any],
    state: dict[str, Any],
    *,
    error_gpu: str,
    repair_gpu: str,
    companion: str,
) -> list[str]:
    """The two replayed findings stayed per-GPU records.

    Judged on the product's own records after the pair was served for a whole
    correlation window: no GPU_MEMORY_DEGRADATION finding was recorded for
    either GPU (the finding store, not only the incidents), the error GPU's
    uncorrectable finding minted its own incident, and neither GPU carries an
    incident of the reading the other one replayed.
    """

    errors = [
        f"{MEMORY_RULE} finding {finding.get('finding_id')} formed on "
        f"{finding.get('gpu_uuid') or 'an unnamed GPU'} from "
        f"{sorted(str(name) for name in finding.get('component_metrics') or [])}"
        for finding in composite_findings(state, MEMORY_RULE)
    ]
    if not metric_incidents(activity, UNCORRECTABLE_PRIMARY, gpu=error_gpu):
        errors.append(
            f"the replayed {UNCORRECTABLE_PRIMARY} on {error_gpu} minted no incident "
            "of its own"
        )
    for name, gpu in ((UNCORRECTABLE_PRIMARY, repair_gpu), (companion, error_gpu)):
        errors.extend(
            f"{name} minted incident {item.get('incident_id')} on {gpu}, which did "
            "not replay it"
            for item in metric_incidents(activity, name, gpu=gpu)
        )
    return errors


def _minted(
    activity: dict[str, Any], error_gpu: str, repair_gpu: str, companion: str
) -> bool:
    """The product recorded the replayed pair: either GPU's own incident or a
    (wrong) composite -- the oracle gate's stand-in when the journal is away."""

    return bool(
        composite_incidents(activity, MEMORY_RULE)
        or _any_incident(activity, error_gpu)
        or metric_incidents(activity, companion, gpu=repair_gpu)
    )


def wrong_scope_source(ctx: VariantContext) -> dict[str, Any]:
    """An error on one GPU and pending repair on another do not compose.

    The subject is composite scoping, judged on the product's records: the two
    findings stay separate per-GPU incidents -- the error GPU's CRITICAL
    isolation (QUARANTINE, no hardware operation) and the repair GPU's own
    pending-repair finding -- and no GPU_MEMORY_DEGRADATION finding or
    incident spans the two GPUs (``scoping_errors``). On a modern GPU that
    repair finding is the product's authorized single-GPU service-window
    RESET_GPU chain (``GPU_RESET`` approval, like 019 ``pending_supported``):
    the chain is the product's policy, not this variant's subject, and the
    variant's duty is to let it end RECOVERED. The pair is therefore served for
    one correlation window (``window_reads``) and the plan returns to the real
    readings by itself -- as a reset applies a pending remap -- so the chain's
    post-reset VALIDATE_GPU grades the real device whether or not the runner
    still reaches the node. Live 2026-10-26 14:54-15:06Z (HyperPod, R13g) the
    held plan was still serving ``row_remap_pending=1`` after the reset -- the
    quiesce had stopped kubelet and with it the probe that would have closed
    the replay -- and the product escalated QUARANTINE -> RESTART_NODE, its
    documented policy for a remap a reset did not clear. The same quiesce
    takes the journal away before the oracle gate can read it, so the gate
    falls back to the records (``wait_served_or_records``) and the journal is
    read at the close, which itself waits for the node to return. The hardware
    oracle is scoped per incident, the repair incident is left to settle on
    its own, and only the error GPU's incident is released by the validated
    restore.
    """

    survey = run_gpu_survey(ctx)
    gpus = survey.gpus_with(UNCORRECTABLE_PRIMARY)
    if len(gpus) < 2:
        raise VariantNotRun(
            "CAPABILITY_ABSENT", "fewer than two GPUs report memory fields"
        )
    error_gpu = gpus[0]
    # The repair flag goes to another GPU that really publishes it.
    repair = survey.component_pair(
        UNCORRECTABLE_PRIMARY, REPAIR_COMPANIONS, avoid=[error_gpu]
    )
    repair_gpu = repair.gpu
    reads = window_reads(survey.interval)
    since = utc_now()
    # Bounded, never held: after ``reads`` scrapes the tool serves the real
    # readings again on its own (the pending flag as a reset leaves it).
    repl = composite_replay(
        ctx,
        survey,
        step(
            reads,
            add(UNCORRECTABLE_PRIMARY, [error_gpu]),
            put(repair.companion, [repair_gpu], 1),
        ),
    )
    repl.deploy()
    repl.arm()
    gate = wait_served_or_records(
        repl,
        [f"{error_gpu}/{UNCORRECTABLE_PRIMARY}"],
        records_show=lambda: _minted(
            activity_now(ctx, since), error_gpu, repair_gpu, repair.companion
        ),
        label="wrong_scope_source",
    )
    wait_gpu_incident(
        ctx,
        since,
        lambda doc: composite_incidents(doc, MEMORY_RULE)
        or _any_incident(doc, error_gpu),
        timeout_seconds=survey.health_summary + 300,
        name="detection",
    )
    # The scoping is judged once the pair has been served for the whole
    # correlation window: a composite that did not form within it never forms.
    ctx.hold(COMPOSITE_WINDOW_SECONDS, name="composite-window")
    final = activity_now(ctx, since)
    state = state_now(ctx, since, names=[UNCORRECTABLE_PRIMARY, repair.companion])
    stages = {
        "no_cross_gpu_composite": []
        if not composite_incidents(final, MEMORY_RULE)
        else ["an error on one GPU composed with repair on another GPU"],
        "scoped_incidents": scoping_errors(
            final,
            state,
            error_gpu=error_gpu,
            repair_gpu=repair_gpu,
            companion=repair.companion,
        ),
    }
    # The plan has run out by itself; the close restores the collector (and,
    # on a node the chain quiesced or rebooted, waits for it to come back).
    repl.close()
    ctx.cleanup.discard("dcgm-replay")
    # The repair GPU's own incident runs its chain on the real (clean) reading;
    # it is given the service-window reset's time to settle, and the hardware
    # oracle reads the record after that.
    repair_incidents = metric_incidents(final, repair.companion, gpu=repair_gpu)
    if not repair_incidents:
        final = wait_gpu_incident(
            ctx,
            since,
            lambda doc: metric_incidents(doc, repair.companion, gpu=repair_gpu),
            timeout_seconds=survey.interval * 4 + 60,
            name="repair-incident",
        )
        repair_incidents = metric_incidents(final, repair.companion, gpu=repair_gpu)
    if repair_incidents:
        final = wait_gpu_incident(
            ctx,
            since,
            lambda doc: repair_incidents if settled(doc, repair_incidents) else [],
            timeout_seconds=survey.health_summary + 600,
            name="repair-settled",
        )
    stages["hardware"] = scoped_hardware_errors(
        final, repair_incidents=repair_incidents
    )
    stages["repair_settled"] = repair_settled_errors(
        final, repair_incidents, gpu=repair_gpu, component=repair.companion
    )
    stages["recovery"] = _validated_restore(ctx, _any_incident(final, error_gpu))
    metrics = ", ".join(repair.exporter_metrics) or "exporter name not surveyed"
    return result(
        stages,
        notes=[
            "GPU_MEMORY_DEGRADATION is per-GPU; components on different GPUs do not compose",
            f"repair gauge exercised: {repair.companion} on {repair_gpu} (exporter "
            f"{metrics}) beside {UNCORRECTABLE_PRIMARY} on {error_gpu}",
            f"the pair was served for {reads} scrapes (one {COMPOSITE_WINDOW_SECONDS}-s "
            "correlation window) and the plan returned to the real readings by "
            "itself, so the reset chain's VALIDATE_GPU graded the real device",
            "the repair GPU's own incident may run the authorized single-GPU "
            "service-window reset chain and settles RECOVERED by itself; only the "
            "error GPU's incident is released by the validated restore",
        ],
        extra={
            "error_gpu": error_gpu,
            "repair_gpu": repair_gpu,
            "component_metrics": [UNCORRECTABLE_PRIMARY, repair.companion],
            "second_component": repair.companion,
            "exporter_metrics": list(repair.exporter_metrics),
            "fault_reads": reads,
            "oracle_gate": gate,
            "repair_incidents": [
                str(item.get("incident_id")) for item in repair_incidents
            ],
        },
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
    pair = survey.component_pair(CORRECTABLE_PRIMARY, CORRECTABLE_COMPANIONS)
    gpu = pair.gpu
    pci_bdf = kmsg_pci_bdf(survey.pci_bdf(gpu))
    if pci_bdf is None:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            f"{gpu} reports no PCI BDF; a kernel XID cannot name it",
        )
    since = utc_now()
    repl = _correctable_replay(ctx, survey, pair, reads=40)
    arm_and_confirm(repl, *pair.keys())
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
        recover_counter(ctx, repl, since, names=list(pair.components))
        return result(stages, notes=[pair.note()], extra=pair.evidence())
    injection = inject_kernel_xid(
        ctx, xid=INDEPENDENT_XID, pci_bdf=pci_bdf, label="024-independent-xid"
    )
    decisions = wait_xid_decisions(
        ctx, injection.started, xid=INDEPENDENT_XID, name="independent-xid"
    )
    injection.close()
    final = activity_now(ctx, since, metrics=pair.components)
    state = state_now(ctx, since, names=pair.components)
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
            allowed_components=pair.components,
        ),
        "hardware": no_hardware(final),
    }
    stages["isolation"] = restore_if_isolated(ctx, decisions)
    recover_counter(ctx, repl, since, names=list(pair.components))
    return result(
        stages,
        notes=[
            f"XID {INDEPENDENT_XID} ({entry.mnemonic}) is monitor grade in catalog "
            f"{entry.catalog_version}: {entry.immediate_action} -> MONITOR_ONLY; the "
            "fatal-grade dispatch beside a weak composite is not driven live here",
            pair.note(),
        ],
        extra={
            **pair.evidence(),
            "xid": INDEPENDENT_XID,
            "pci_bdf": pci_bdf,
            "decisions": decisions,
            "kernel_replay": injection.evidence(),
        },
    )


def _any_incident(doc: dict[str, Any], gpu: str) -> list[dict[str, Any]]:
    return metric_incidents(doc, UNCORRECTABLE_PRIMARY, gpu=gpu)


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


#: The second-component wording of the plan texts: the exporter decides.
CORRECTABLE_PAIR_TEXT = (
    "the volatile SBE counter beside the retired-page counter on pre-Hopper parts "
    "or the correctable row-remap counter on Hopper and later"
)
REPAIR_PAIR_TEXT = (
    "the row-remap pending flag on Ampere and later, the page-retirement pending "
    "flag on older parts"
)

VARIANTS = [
    gpu_spec(
        CASE,
        "corrected_memory",
        f"two correctable memory signals on one GPU ({CORRECTABLE_PAIR_TEXT})",
        "CORRECTABLE_MEMORY_DEGRADATION diagnostic; no reset",
        corrected_memory,
        max_minutes=20,
    ),
    gpu_spec(
        CASE,
        "memory_failure",
        f"an uncorrectable error plus pending repair on one GPU ({REPAIR_PAIR_TEXT})",
        "CRITICAL GPU_MEMORY_DEGRADATION, not weakened; isolation only",
        memory_failure,
        risk="M",
        max_minutes=25,
    ),
    gpu_spec(
        CASE,
        "nvlink_multiple_gpus",
        "one NVLink class on two GPUs (the CRC counter: flit total on NVLink3, DL "
        "aggregate on NVLink4+)",
        "MULTI_GPU_NVLINK_FABRIC_FAILURE diagnostic",
        nvlink_multiple_gpus,
        max_minutes=20,
    ),
    gpu_spec(
        CASE,
        "fresh_components",
        f"components activated in the same batch ({CORRECTABLE_PAIR_TEXT})",
        "a composite forms from fresh components",
        fresh_components,
        max_minutes=20,
    ),
    gpu_spec(
        CASE,
        "reused_components",
        f"hold the same components across batches ({CORRECTABLE_PAIR_TEXT})",
        "the composite is not re-minted for reused components",
        reused_components,
        max_minutes=22,
    ),
    gpu_spec(
        CASE,
        "wrong_scope_source",
        f"an error on one GPU, repair on another ({REPAIR_PAIR_TEXT}), served for "
        "one 45-s correlation window and then the real readings again; the repair "
        "GPU's own pending-repair finding runs the product's authorized single-GPU "
        "service-window RESET_GPU chain",
        "components on different GPUs do not compose (per-GPU incidents and "
        "findings on the records); the error GPU's isolation plans no hardware "
        "operation; the repair GPU's reset chain validates the real post-reset "
        "device and settles RECOVERED by itself (no NVSwitch reset, no "
        "RESTART_NODE escalation, no reboot)",
        wrong_scope_source,
        risk="H",
        flags="X",
        max_minutes=35,
        approvals=("GPU_RESET",),
    ),
    gpu_spec(
        CASE,
        "independent_xid",
        "hold CORRECTABLE_MEMORY_DEGRADATION on one GPU "
        f"({CORRECTABLE_PAIR_TEXT}), then a reviewed trusted "
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
