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
  single-GPU reset (``row_remap_pending`` on a modern GPU). The flag is served
  for exactly the node's edge-filter confirmation count of consecutive scrapes
  (``pending_fault_reads``: the last one is the batch the collector delivers
  and the product mints on; fewer never reach the control plane, live
  2026-10-27) and the tool is back on the device's real reading before the
  chain's quiesce stops the collector, so the post-reset VALIDATE_GPU grades
  the real device whether or not the runner can still reach the node. The
  reads before the confirmation are armed clear of the node's health-summary
  phase (``summary_clear_wait_seconds``), the one delivery that could mint the
  flag early and leave a read for the first scrape after RESTORE_GPU_SERVICES.
  Both are judged on the product's records once the record has ended.
  ``pending_supported`` proves the policy and the executed chain
  (QUIESCE_GPU_SERVICES, VERIFY_NO_GPU_CLIENTS, RESET_GPU,
  RESTORE_GPU_SERVICES, VALIDATE_GPU; the incident RECOVERED; no NVSwitch
  reset, reboot or RESTART_NODE escalation), ``post_reset`` the post-action
  state on top (``post_reset_errors``: one RESET_GPU, VALIDATE_GPU SUCCEEDED
  after it on the fresh reading, the pending finding cleared, no follow-up
  workflow). They need ``--approve GPU_RESET``.

No 019 plan holds a step (``until_advance``): every fault is bounded by its
own read count, sized from the wait that needs it (``fault_reads``), and the
tool returns to the real readings by itself. The runner never has to reach
the node while the product acts on it -- on HyperPod the quiesce stops kubelet
and with it the probe Pod (``general_live_gpu_replay_close``).

``pending_unsupported`` (no legacy-architecture GPU on this site) and
``observe_profile`` (the live Runtime Profile owns ``gpuReset``) are recorded
NOT_RUN with their reason.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.env import env_bool
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
)
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    GpuSurvey,
    activity_now,
    add,
    arm_and_confirm,
    baseline,
    gpu_spec,
    leading_passthrough_reads,
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
    chain_executed,
    executions_of,
    metric_incidents,
    planned_operations,
    read_only_errors,
    settled,
)
from scripts.e2e.regional.general_live_registry import deferred_variant
from scripts.e2e.regional.general_live_verdicts import (
    final_execution,
    parse_time,
    workflows_of,
)
from tools.gpu_dcgm_field_replay import MAX_READS_PER_STEP

CASE = "GF-GENERAL-LIVE-019"
RESET = frozenset(
    {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "REPLACE_NODE"}
)
#: The detection wait of every variant: the product's health summary plus a
#: margin for the collector's delivery and the processor's grading.
DETECTION_MARGIN_SECONDS = 240
#: Reads a bounded fault step carries beyond the wait it must outlast: one
#: skipped or late scrape, and one for the scrape in flight when the wait ends.
FAULT_SLACK_READS = 2


def _hold_seconds(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, 20 * 60)


def fault_reads(interval: int, seconds: int) -> int:
    """The read bound of a fault step that must outlast a ``seconds`` wait.

    The scrapes a collector at ``interval`` s makes in ``seconds``, plus
    ``FAULT_SLACK_READS``. The step is never held (``until_advance``): after
    its reads the tool serves the device's real readings by itself, so the
    runner never has to reach the node to end the fault. A bound over the
    tool's ``MAX_READS_PER_STEP`` is refused here, before anything is
    deployed, rather than clipped into a step that ends under the wait.
    """

    needed = -(-int(seconds) // max(1, int(interval))) + FAULT_SLACK_READS
    if needed > MAX_READS_PER_STEP:
        raise VariantInconclusive(
            f"a fault held for {seconds} s needs {needed} reads at {interval} s per "
            f"read; the field tool serves at most {MAX_READS_PER_STEP} per step"
        )
    return needed


def _detection_seconds(survey: GpuSurvey) -> int:
    return survey.health_summary + DETECTION_MARGIN_SECONDS


def _replay(
    ctx: VariantContext, survey: GpuSurvey, *steps: dict[str, Any]
) -> DcgmReplay:
    """A ``memory``-profile replay of ``steps`` after the baseline reads, with
    the node's boot id recorded for a close the node's reboot may have to
    judge (``DcgmReplay.close``)."""

    return DcgmReplay(
        ctx,
        plan("memory", baseline(), *steps),
        _hold_seconds(ctx),
        node_boot_id=survey.boot_id,
    )


# --------------------------------------------------------------------------- #
# diagnostic-only counter variants
# --------------------------------------------------------------------------- #
def _diagnostic(ctx: VariantContext, canonical: str) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target(canonical)
    since = utc_now()
    # The counter rises by one on every read until the detection wait has had
    # its budget; the chain's VALIDATE_GPU runs seconds after the mint, while
    # the finding is active, and ``recover_counter`` ends the replay after it.
    replay = _replay(
        ctx,
        survey,
        step(
            fault_reads(survey.interval, _detection_seconds(survey)),
            add(canonical, [gpu]),
        ),
    )
    arm_and_confirm(replay, f"{gpu}/{canonical}")
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: metric_incidents(doc, canonical, gpu=gpu),
        metrics=(canonical,),
        timeout_seconds=_detection_seconds(survey),
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


def _containment_seconds(survey: GpuSurvey) -> int:
    """The detection wait of an isolating variant: the detection budget plus
    the time its containment steps take to execute (``CONTAINMENT_SECONDS``);
    the fault is served for as long (``fault_reads``) and no longer."""

    return _detection_seconds(survey) + CONTAINMENT_SECONDS


def volatile_dbe(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target("ecc_dbe_volatile_total")
    since = utc_now()
    replay = _replay(
        ctx,
        survey,
        step(
            fault_reads(survey.interval, _containment_seconds(survey)),
            add("ecc_dbe_volatile_total", [gpu]),
        ),
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
        timeout_seconds=_containment_seconds(survey),
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
    """A row remap failure is contained, never reset.

    The product's policy for ``row_remap_failure`` on a modern GPU is CRITICAL
    DRAIN (``gpu_memory_policy.memory_decision``: field diagnosis before any
    RMA conclusion), a chain with no reset step and so no quiesce: the probe
    stays reachable on both providers and nothing has to outlast a reset. The
    failure is served for the containment wait (``fault_reads`` over
    ``_containment_seconds``) -- a remap failure does not clear by itself, and
    the chain is judged on executed containment -- and the replay ends before
    the exit, whose restore VALIDATE_GPU rightly refuses a node still reading
    ``row_remap_failure=1``.
    """

    survey = run_gpu_survey(ctx)
    gpu = survey.target("row_remap_failure")
    since = utc_now()
    replay = _replay(
        ctx,
        survey,
        step(
            fault_reads(survey.interval, _containment_seconds(survey)),
            put("row_remap_failure", [gpu], 1),
        ),
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
        timeout_seconds=_containment_seconds(survey),
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
PENDING = "row_remap_pending"
#: The product's service-window reset chain for a pending remap on a modern
#: GPU (``orchestration/workflow_builder``: RESET_GPU action), in order.
SERVICE_WINDOW_CHAIN = (
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
)
#: The hardware operations that chain may plan; anything else on the node is
#: a hardware error (``no_hardware``).
SERVICE_WINDOW_HARDWARE = ("QUIESCE_GPU_SERVICES", "RESET_GPU")
#: The reset ladder's next rung after a validation that fails on a reset node
#: (``orchestration/escalation.next_rung``), with the rungs beyond it: the
#: escalation 019's reset variants must never see.
RESET_ESCALATION = frozenset(
    {"RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "REPLACE_NODE"}
)
#: How long the settled record is waited for after the detection, beyond the
#: health summary: the chain ran 2 min 6 s live (eksb 2026-10-26 07:47:57-
#: 07:50:03Z); its RESTART_NODE escalation, which must be read to be failed,
#: took until 15:06Z after a 14:54Z mint (HyperPod).
RESET_SETTLE_SECONDS = 600
#: Live 2026-10-26, both clusters: the product graded the first *delivered*
#: replayed scrape within ~4 s (``gpu_memory_policy.memory_decision``: the
#: pending flag mints on one reading > 0 -- no delta, confirmation or sustain)
#: and the chain's QUIESCE_GPU_SERVICES followed within ~20 s of the mint. The
#: quiesce stops the metrics collector itself and the exporter container on
#: both providers (``node_agent.common.DEFAULT_QUIESCE_SERVICES`` /
#: ``DEFAULT_QUIESCE_CONTAINERS``), and on eksb (07:48Z, 024
#: ``wrong_scope_source``) the scrape 15 s after the first replayed one
#: already found the exporter gone ("Connection refused" in the replay
#: journal). No fault read may be left for a scrape after that: the field tool
#: counts only the scrapes it could serve (``gpu_dcgm_field_replay.
#: make_server``: a failed upstream fetch is journaled as an error and
#: ``reads`` does not advance), so a fault read the quiesce finds unconsumed
#: is served to the first scrape after RESTORE_GPU_SERVICES -- the batch the
#: post-reset VALIDATE_GPU grades (``adapters/gpu_validation``: fresh GPU
#: metrics after the RESET_GPU readiness barrier, then the GPU's active
#: findings, no grace for the pending flag) -- and ``escalation.next_rung``
#: turns a validation failed after a reset into RESTART_NODE (live 2026-10-26
#: 14:54-15:06Z, HyperPod, a held plan).
#:
#: Live 2026-10-27 04:05-04:18Z (HyperPod, the hp lane's fault node, R13g):
#: the flag served for ONE scrape (the bound S36 drew from the ingest alone)
#: minted nothing -- oracle read 5 served ``row_remap_pending=1``, reads 6-43
#: were the device's own, no incident, the node untouched. The batch has to
#: be *delivered* first, and the node's collector delivers a changed
#: ``row_remap_pending`` only once the flag has been read on
#: ``GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES`` consecutive scrapes
#: (``collectors/gpu/dcgm.py`` ``_candidate_keys``: ``row_remap_pending > 0``
#: is a "breached" candidate; ``_should_deliver``: ``candidate-confirmed`` at
#: the confirmation streak, 3 on the live nodes). The flag is a gauge, not a
#: counter (``_is_counter``: ``*_total``) and not a state gauge
#: (``STATE_CHANGE_REASONS``: the last XID and the fabric state), so one read
#: of it is no delivery edge at all and the batch is suppressed. 024
#: ``wrong_scope_source`` mints on one read only because the DBE counter in
#: the same batch is a ``counter-increased`` edge (eksb 07:48Z, read 4).
EDGE_FILTER_KEY = "GPU_FAULT_DCGM_EDGE_FILTER_ENABLED"
#: The health summary moves each phase by at most ``ratio x interval / 2``
#: (``collectors/cadence.CadenceJitter``, ``GPU_FAULT_COLLECTOR_CADENCE_JITTER_
#: RATIO`` default 0.1, not set by this site's deploy) and goes out on the
#: first scrape at or after the moved phase (``collectors/gpu/dcgm.py``
#: ``_should_deliver`` "health-summary", ``jittered_next_phase``).
SUMMARY_JITTER_RATIO = 0.1
#: Runner-to-node clock skew plus the arm's probe round trip, both ways.
ARM_SKEW_SECONDS = 10


def edge_filter_enabled(survey: GpuSurvey) -> bool:
    """Whether the node's collector runs the DCGM edge filter (its default).

    Read as the product reads it (``env_bool``); a token the product would
    refuse at start-up cannot belong to a running collector and reads as the
    default.
    """

    environment = {
        str(key): str(value)
        for key, value in (survey.raw.get("collector_env") or {}).items()
    }
    try:
        return env_bool(EDGE_FILTER_KEY, True, environ=environment)
    except ValueError:
        return True


def pending_fault_reads(confirmation: int, *, edge_filter_enabled: bool = True) -> int:
    """The scrapes a replayed ``row_remap_pending`` is served to: exactly the
    node's edge-filter confirmation count, consecutively.

    The ``confirmation``-th consecutive read is the one the collector delivers
    (``candidate-confirmed``) and the product mints on; the plan has no read
    after it, so the tool is back on the device's real reading before the
    chain's quiesce stops the collector. Fewer reads never reach the control
    plane (live 2026-10-27, one read); a spare read is the escalation (a fault
    read the quiesce finds unconsumed is served to the post-reset VALIDATE_GPU's
    batch, RESTART_NODE). With the edge filter disabled every batch is
    delivered and the first read mints: one read.
    """

    if not edge_filter_enabled:
        return 1
    return max(1, int(confirmation))


def require_pending_plan(
    document: dict[str, Any], *, gpu: str, reads: int, variant: str
) -> None:
    """The plan serves the flag on exactly ``reads`` consecutive scrapes after
    pass-through reads, nothing else, and holds nothing (``until_advance``).

    ``reads`` is the node's own count (``pending_fault_reads``): a plan built
    for another count -- one read on a node confirming on three (live
    2026-10-27), or a spare read -- is refused before anything is deployed.
    """

    steps = list(document.get("steps") or [])
    if any(item.get("until_advance") for item in steps):
        raise ValueError(f"{CASE} {variant}: a pending plan never holds a step")
    if len(steps) != 2 or steps[0].get("ops") or int(steps[0].get("reads") or 0) < 1:
        raise ValueError(
            f"{CASE} {variant}: expected pass-through reads then one flag step, "
            f"got {len(steps)} step(s)"
        )
    expected = {"op": "set", "field": PENDING, "gpus": [gpu], "value": 1}
    if steps[1].get("ops") != [expected]:
        raise ValueError(
            f"{CASE} {variant}: the flag step must set {PENDING}=1 on {gpu} and "
            f"nothing else, got {steps[1].get('ops')}"
        )
    if int(steps[1].get("reads") or 0) != reads:
        raise ValueError(
            f"{CASE} {variant}: the flag step serves {steps[1].get('reads')} read(s); "
            f"the node's edge filter delivers the {reads}-sample confirmation and "
            "nothing earlier, and a read past it is served to the post-reset "
            f"VALIDATE_GPU (RESTART_NODE); the step must serve exactly {reads}"
        )


def summary_clear_wait_seconds(
    now: datetime,
    *,
    cluster_id: str,
    node_id: str,
    interval: int,
    health_summary: int,
    passthrough_reads: int,
    fault_reads: int,
    jitter_ratio: float = SUMMARY_JITTER_RATIO,
) -> dict[str, Any]:
    """How long to wait before arming so no health summary can deliver a
    fault read before the confirmation, and the windows that say why.

    A fault read delivered early -- the summary is the one delivery an idle
    node makes on its own -- mints the flag with the later fault reads still
    unconsumed when the quiesce stops the collector, and those are served to
    the post-reset VALIDATE_GPU (RESTART_NODE). The node's summaries go out on
    its restart-stable phase (``collectors/scheduling.next_stable_phase`` of
    cluster, node and the ``gpu-metrics`` channel), moved by at most
    ``jitter_ratio x health_summary / 2`` and delivered on the first scrape at
    or after the moved phase. Armed at ``A``, fault read ``k`` is scraped in
    ``(A + (p + k - 1) x i, A + (p + k) x i]`` for ``p`` pass-through reads at
    ``i`` s (the tick phase is unknown; one more interval covers a late scrape),
    so the reads before the confirmation lie in ``(A + p x i, A + (p + N) x i]``
    plus the skew. Returns the first ``A >= now`` whose window meets no summary
    window, as ``{"wait_seconds", "arm_at", "fault_window", "summary_windows"}``.
    A plan with one fault read has nothing before its confirmation: no wait.
    """

    reads_before = max(0, int(fault_reads) - 1)
    answer: dict[str, Any] = {
        "wait_seconds": 0.0,
        "arm_at": now.isoformat(),
        "fault_window": None,
        "summary_windows": [],
    }
    if reads_before == 0:
        return answer
    tick = max(1, int(interval))
    # The reads before the confirmation: from the first tick after the
    # pass-through reads to one tick past the last of them (a late scrape).
    lead = tick * max(0, int(passthrough_reads))
    span = tick * (reads_before + 1)
    jitter = float(health_summary) / 2 * float(jitter_ratio)
    before = jitter + ARM_SKEW_SECONDS
    after = jitter + tick + ARM_SKEW_SECONDS
    if span + before + after >= int(health_summary):
        raise ValueError(
            f"no summary-clear window: {reads_before + 1} fault reads at {tick} s "
            f"span {span} s and every {health_summary} s summary excludes "
            f"{before + after:.0f} s"
        )
    arm_at = now
    for _ in range(4):
        start = arm_at + timedelta(seconds=lead)
        end = start + timedelta(seconds=span)
        phase = next_stable_phase(
            start - timedelta(seconds=after),
            cluster_id=cluster_id,
            node_id=node_id,
            channel="gpu-metrics",
            interval_seconds=int(health_summary),
        )
        window = (phase - timedelta(seconds=before), phase + timedelta(seconds=after))
        answer["summary_windows"].append([window[0].isoformat(), window[1].isoformat()])
        if window[1] <= start or window[0] >= end:
            answer["wait_seconds"] = (arm_at - now).total_seconds()
            answer["arm_at"] = arm_at.isoformat()
            answer["fault_window"] = [start.isoformat(), end.isoformat()]
            return answer
        arm_at = window[1] - timedelta(seconds=lead)
    raise ValueError("no summary-clear window found within four summary phases")


def _pending_incidents(doc: dict[str, Any], gpu: str) -> list[dict[str, Any]]:
    return metric_incidents(doc, PENDING, gpu=gpu)


def _pending_plan(ctx: VariantContext, survey: GpuSurvey, gpu: str) -> dict[str, Any]:
    """The checked plan: the flag on the node's confirmation count of reads."""

    from scripts.e2e.regional.regional_commands import RegionalFixtureError

    filtered = edge_filter_enabled(survey)
    reads = pending_fault_reads(survey.confirmation, edge_filter_enabled=filtered)
    document = plan("memory", baseline(), step(reads, put(PENDING, [gpu], 1)))
    try:
        require_pending_plan(document, gpu=gpu, reads=reads, variant=ctx.spec.variant)
    except ValueError as exc:
        raise RegionalFixtureError(f"pending replay plan rejected: {exc}") from exc
    ctx.evidence.write(
        "pending-plan-shape",
        {
            "confirmation_samples": survey.confirmation,
            "edge_filter_enabled": filtered,
            "fault_reads": reads,
            "passthrough_reads": leading_passthrough_reads(document),
        },
    )
    return document


def _arm_clear_of_summary(
    ctx: VariantContext, replay: DcgmReplay, survey: GpuSurvey, reads: int
) -> dict[str, Any]:
    """Arm once the reads before the confirmation cannot meet a health summary."""

    from scripts.e2e.regional.regional_commands import RegionalFixtureError

    try:
        clearance = summary_clear_wait_seconds(
            ctx.clock(),
            cluster_id=ctx.settings.regional.cluster_id,
            node_id=ctx.settings.node,
            interval=replay.interval,
            health_summary=survey.health_summary,
            passthrough_reads=leading_passthrough_reads(replay.plan),
            fault_reads=reads,
        )
    except ValueError as exc:
        raise RegionalFixtureError(f"pending replay cannot be armed: {exc}") from exc
    ctx.evidence.write("summary-phase-clearance", clearance)
    if clearance["wait_seconds"] > 0:
        ctx.hold(clearance["wait_seconds"], name="summary-phase-clearance")
    replay.arm()
    return clearance


def _pending_reset(ctx: VariantContext, *, verify_recovery: bool) -> dict[str, Any]:
    """Serve the pending flag for the confirmation, let the chain run, judge
    the records.

    The plan is bounded (``pending_fault_reads``) and never held: after the
    confirming scrape the tool serves the device's real reading (``row_remap_
    pending=0``) by itself, so the chain's post-reset VALIDATE_GPU grades the
    real device and the runner never has to reach the node while the product
    acts on it. The arm waits for a window no health summary can reach
    (``_arm_clear_of_summary``). On HyperPod the quiesce stops kubelet and
    with it the probe Pod, so the oracle gate falls back to the product's
    records when the journal cannot be read (``wait_served_or_records``; the
    journal is read at the close) and the close itself waits for the node to
    return (``DcgmReplay.close``). The record is waited for until it has
    ended, then judged on executed steps and the state it settled in; the
    replay is closed after that, never inside the chain.
    """

    survey = run_gpu_survey(ctx)
    gpu = survey.target(PENDING)
    if "H" not in survey.model(gpu) and "B" not in survey.model(gpu):
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            f"{gpu} is {survey.model(gpu)!r}; row_remap_pending reset needs a modern GPU",
        )
    document = _pending_plan(ctx, survey, gpu)
    reads = int(document["steps"][1]["reads"])
    since = utc_now()
    replay = DcgmReplay(ctx, document, _hold_seconds(ctx), node_boot_id=survey.boot_id)
    replay.deploy()
    clearance = _arm_clear_of_summary(ctx, replay, survey, reads)
    gate = wait_served_or_records(
        replay,
        [f"{gpu}/{PENDING}"],
        records_show=lambda: bool(
            _pending_incidents(activity_now(ctx, since, metrics=(PENDING,)), gpu)
        ),
        label=ctx.spec.variant,
    )
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: _pending_incidents(doc, gpu),
        metrics=(PENDING,),
        timeout_seconds=_detection_seconds(survey),
        name="detection",
    )
    incidents = _pending_incidents(detection, gpu)
    final = detection
    if incidents:
        # The chain is judged once the record has ended: executed steps and the
        # state it settled in, including a RESTART_NODE escalation, which only
        # a settled record shows in full.
        final = wait_gpu_incident(
            ctx,
            since,
            lambda doc: incidents if settled(doc, incidents) else [],
            metrics=(PENDING,),
            timeout_seconds=survey.health_summary + RESET_SETTLE_SECONDS,
            name="settled",
        )
        # The incident as the settled read shows it: the detection's copy still
        # carries the state it was minted in.
        incidents = _pending_incidents(final, gpu) or incidents
    chain = GpuChain(
        label=f"{PENDING} on {gpu}",
        operations=SERVICE_WINDOW_CHAIN,
        validation="VALIDATE_GPU",
        forbidden=RESET_ESCALATION,
        actions=frozenset({"RESET_GPU"}),
        incident_states=frozenset({"RECOVERED"}),
    )
    stages = {
        "chain": [
            *chain_errors(final, incidents, chain),
            *reset_executed_errors(final, incidents, chain.label),
        ],
        "hardware": no_hardware(final, allowed=SERVICE_WINDOW_HARDWARE),
    }
    # The plan has run out by itself; the close restores the collector (and,
    # on a node the chain quiesced or rebooted, waits for it to come back).
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    cleared = state_now(ctx, since, names=[PENDING])
    stages["recovery"] = recovered_errors(final, incidents, cleared, label=chain.label)
    if verify_recovery:
        stages["post_reset"] = post_reset_errors(final, incidents, label=chain.label)
    return result(
        stages,
        notes=[
            "single-GPU service-window reset; no NVSwitch reset, no reboot",
            f"{PENDING}=1 was served for {reads} consecutive scrape"
            f"{'s' if reads != 1 else ''} (the node's edge-filter confirmation "
            "count: the last one is the batch the collector delivers and the "
            "product mints on) and the plan returned to the device's real reading "
            "by itself before the chain's quiesce stopped the collector, so the "
            "post-reset VALIDATE_GPU graded the real device",
            f"armed {clearance['wait_seconds']:.0f} s after the repoint settled, "
            "clear of the node's health-summary phase, so no summary could "
            "deliver a read before the confirmation",
            "judged on the product's records once the record had ended; the "
            "replay was closed after the chain, never inside it",
        ],
        extra={
            "gpu_uuid": gpu,
            "fault_reads": reads,
            "confirmation_samples": survey.confirmation,
            "summary_clearance_wait_seconds": clearance["wait_seconds"],
            "oracle_gate": gate,
            "incident_ids": [str(item.get("incident_id")) for item in incidents],
        },
    )


def pending_supported(ctx: VariantContext) -> dict[str, Any]:
    return _pending_reset(ctx, verify_recovery=False)


def post_reset(ctx: VariantContext) -> dict[str, Any]:
    return _pending_reset(ctx, verify_recovery=True)


def _succeeded_resets(
    activity: dict[str, Any], incidents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        item
        for item in executions_of(activity, "RESET_GPU", incidents)
        if item.get("status") == "SUCCEEDED"
    ]


def reset_executed_errors(
    activity: dict[str, Any], incidents: list[dict[str, Any]], label: str
) -> list[str]:
    """The chain reset the GPU: a SUCCEEDED RESET_GPU execution on the record."""

    if not incidents:
        return []
    executions = executions_of(activity, "RESET_GPU", incidents)
    if not executions:
        return [f"{label} chain planned RESET_GPU but never executed it"]
    if not _succeeded_resets(activity, incidents):
        last = executions[-1]
        if last.get("status") == "FAILED":
            return [
                f"{label} RESET_GPU ended FAILED: {str(last.get('error') or '')[:200]}"
            ]
        return [f"{label} RESET_GPU has not succeeded ({last.get('status')})"]
    return []


def recovered_errors(
    activity: dict[str, Any],
    incidents: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    label: str,
) -> list[str]:
    """The reset recovered the GPU, on the records: the record ended, the
    incident RECOVERED, and no ``row_remap_pending`` finding is active on
    the node -- the real reading cleared it (``gpu_metrics``: a flag reads 0,
    no decision, the finding state is closed on that batch)."""

    if not incidents:
        return []
    errors = []
    if not settled(activity, incidents):
        errors.append(f"{label} workflow had not ended when the settle wait ran out")
    for incident in incidents:
        if incident_state(activity, incident) != "RECOVERED":
            errors.append(
                f"{label} incident {incident.get('incident_id')} is "
                f"{incident_state(activity, incident)}, not RECOVERED"
            )
    active = [
        item
        for item in state.get("active_findings") or []
        if item.get("canonical_name") == PENDING
    ]
    if active:
        errors.append(
            f"{len(active)} {PENDING} finding{'s' if len(active) != 1 else ''} still "
            "active after the chain ended: the device's real reading did not "
            "clear the replayed one"
        )
    return errors


def incident_state(activity: dict[str, Any], incident: dict[str, Any]) -> str:
    """The incident's state as the latest read shows it (the detection's
    document is older than the settled one)."""

    wanted = str(incident.get("incident_id"))
    for item in activity.get("incidents") or []:
        if str(item.get("incident_id")) == wanted:
            return str(item.get("state") or "")
    return str(incident.get("state") or "")


def post_reset_errors(
    activity: dict[str, Any], incidents: list[dict[str, Any]], *, label: str
) -> list[str]:
    """What the records must say after the reset for the post-reset proof.

    One RESET_GPU execution, SUCCEEDED; the chain's VALIDATE_GPU SUCCEEDED and
    not before the reset completed (the readiness barrier: it graded readings
    newer than the reset); and a single workflow for the incident -- a second
    one is the ladder's next rung (QUARANTINE, RESTART_NODE: live 2026-10-26
    14:54Z after a held plan) and the blind reboot loop this variant rules out.
    """

    if not incidents:
        return []
    errors = []
    resets = _succeeded_resets(activity, incidents)
    if len(resets) != 1:
        errors.append(
            f"{label} RESET_GPU succeeded {len(resets)} times (expected exactly one)"
        )
    incident_id = str(incidents[0].get("incident_id"))
    workflows = workflows_of(activity, incident_id)
    if workflows:
        validation = final_execution(workflows[0], "VALIDATE_GPU")
        if validation is None:
            errors.append(f"{label} chain never executed VALIDATE_GPU after the reset")
        elif validation.get("status") != "SUCCEEDED":
            errors.append(
                f"{label} post-reset VALIDATE_GPU ended {validation.get('status')}: "
                f"{str(validation.get('error') or '')[:200]}"
            )
        elif resets:
            reset_at = parse_time(resets[0].get("updated_at"))
            validated_at = parse_time(validation.get("updated_at"))
            if reset_at and validated_at and validated_at < reset_at:
                errors.append(
                    f"{label} VALIDATE_GPU ended at {validated_at.isoformat()}, before "
                    f"RESET_GPU completed at {reset_at.isoformat()}"
                )
    for workflow in workflows[1:]:
        errors.append(
            f"{label} incident went on to workflow {workflow.get('request_id')} "
            f"planning {planned_operations(workflow)}: the reset did not end the "
            "incident"
        )
    return errors


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
        "set row_remap_pending=1 on a modern GPU for exactly the node's edge-filter "
        "confirmation scrapes (the last one mints it); the device's real reading "
        "returns by itself before the chain's quiesce",
        "authorized single-GPU service-window RESET_GPU chain executed "
        "(QUIESCE_GPU_SERVICES, VERIFY_NO_GPU_CLIENTS, RESET_GPU, RESTORE_GPU_SERVICES, "
        "VALIDATE_GPU) and the incident RECOVERED, judged on the records; no NVSwitch "
        "reset, reboot or RESTART_NODE escalation",
        pending_supported,
        risk="H",
        flags="X",
        max_minutes=45,
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
        "set row_remap_pending=1 on a modern GPU for exactly the node's edge-filter "
        "confirmation scrapes (the last one mints it); the post-reset VALIDATE_GPU "
        "grades the device's own clean reading",
        "after the single-GPU reset VALIDATE_GPU passes on the fresh clean reading, "
        "the pending finding clears, exactly one RESET_GPU, the incident RECOVERED; "
        "no blind reboot loop (no follow-up workflow, no RESTART_NODE)",
        post_reset,
        risk="H",
        flags="X",
        max_minutes=45,
        approvals=("GPU_RESET",),
    ),
]
