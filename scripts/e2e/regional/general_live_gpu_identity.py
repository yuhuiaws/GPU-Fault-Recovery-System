"""GF-GENERAL-LIVE-026 GPU identity / expected count (live-replay).

The inventory reading is replayed through ``tools/nvidia_smi_inventory_replay.py``
(a pass-through wrapper first on the one collector unit's PATH): one GPU row
reports a replay UUID (``replace_uuid``) or two rows swap PCI bus ids
(``swap_bdf``). The expected-count inputs (an unknown instance type) and the
boot generation go through the same unit's private environment. The verdict
reads the product's own inventory grading (``app/ingest/telemetry_context`` ->
``orchestration/families/health``): a changed UUID set is a CRITICAL
RUN_DIAGNOSTICS identity finding that plans only FREEZE_EVIDENCE + VALIDATE_GPU
(diagnostic, no reset/reboot) with an operator-review notification; an unknown
count is a WARNING COLLECT_EVIDENCE.

Two variants are expected FAIL against the current product:
``bdf_changed`` (the identity finding compares UUID sets only, so a BDF-only
change is never diagnosed) and ``stale_boot`` (the baseline is keyed on
cluster+node and the snapshot's boot id is never compared, so old-boot
inventory overrides the current one).

The oracle gate (``_replay``) waits for the wrapper's first rewritten real
inventory query from the repoint on, inside ``inventory_window_seconds``: the
restarted collector queries the inventory on its first tick (start-up spread
plus start-up jitter) and then every inventory interval, so a journal judged
once right after the repoint holds nothing (live 2026-10-26, plain EKS,
``bdf_changed`` and ``count_unchanged`` INCONCLUSIVE 14-15 s after the
collector restart, the first real query 6-21 s after it). The reviewed plan's
``not_after`` is composed before the first file lands on the node, as the
deploy allowance plus the variant's hold; a deploy that overran the allowance
is a named fixture failure, never a plan that expires under the variant.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
)
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    GpuSurvey,
    activity_now,
    no_hardware,
    passthrough_plan,
    gpu_spec,
    replay_boot_id,
    replay_uuid,
    result,
    run_gpu_survey,
    utc_now,
    wait_gpu_incident,
)
from scripts.e2e.regional.general_live_gpu_verdicts import (
    GpuChain,
    chain_errors,
    inventory_incidents,
    read_only_errors,
)

CASE = "GF-GENERAL-LIVE-026"
IDENTITY = "gpu_inventory_identity_changed"
UNKNOWN_COUNT = "gpu_expected_count_unknown"
MISMATCH = "gpu_inventory_mismatch"
RESET = frozenset(
    {"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES", "RESTART_NODE", "REPLACE_NODE"}
)
#: The restarted collector's first tick -- and with it its first inventory
#: query (``collectors/gpu/inventory_cadence.GpuInventoryCadence.is_due``: due
#: while nothing was delivered yet) -- follows the start-up spread
#: (``collectors/gpu/dcgm.py``: ``next_stable_phase`` over
#: ``GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS``, product default 15 s) plus
#: the random start-up jitter (``collectors/cadence.CadenceJitter``:
#: ``[0, GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS)``, default 20 s); the next
#: queries come every ``GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS`` (60 s on
#: this site). The gate allows that first tick, two inventory intervals (a
#: failed first query is retried on the next ticks) and a grace for the probe
#: round trips: 185 s on this site, from the repoint receipt.
STARTUP_SPREAD_KEY = "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS"
STARTUP_JITTER_KEY = "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS"
DEFAULT_STARTUP_SPREAD_SECONDS = 15
DEFAULT_STARTUP_JITTER_SECONDS = 20
INVENTORY_GATE_INTERVALS = 2
INVENTORY_GATE_GRACE_SECONDS = 30
#: The time the fixture allows between composing the reviewed plan and the
#: collector's restart (four source installs, deploy-proxy, repoint-collector:
#: about a minute live, each call bounded by its own probe timeout). The plan's
#: ``not_after`` is composed before the first file lands on the node
#: (``install-source`` refuses a second copy), so it is this allowance plus the
#: variant's hold: from the repoint on, the plan stays valid for the whole hold
#: -- the gate window, the detection wait and the rebaseline hold are inside
#: it -- and a deploy that overran the allowance is reported with its duration
#: instead of expiring the plan under the running variant.
INVENTORY_DEPLOY_ALLOWANCE_SECONDS = 600


def _hold(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, 20 * 60)


def _seconds(survey: GpuSurvey, key: str, default: int) -> int:
    """A collector setting the product reads as a number of seconds."""

    raw = (survey.raw.get("collector_env") or {}).get(key)
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def inventory_window_seconds(survey: GpuSurvey) -> int:
    """How long the first rewritten inventory query may take after the repoint."""

    return (
        _seconds(survey, STARTUP_SPREAD_KEY, DEFAULT_STARTUP_SPREAD_SECONDS)
        + _seconds(survey, STARTUP_JITTER_KEY, DEFAULT_STARTUP_JITTER_SECONDS)
        + INVENTORY_GATE_INTERVALS * survey.inventory_interval
        + INVENTORY_GATE_GRACE_SECONDS
    )


def inventory_plan(
    operations: list[dict[str, Any]], *, composed_at: float, hold: int
) -> dict[str, Any]:
    """The reviewed plan: valid for the deploy allowance plus the hold."""

    return {
        "operations": operations,
        "not_after": composed_at + INVENTORY_DEPLOY_ALLOWANCE_SECONDS + hold,
    }


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _replay(
    ctx: VariantContext,
    survey: GpuSurvey,
    operations: list[dict[str, Any]],
    *,
    boot_id: str = "",
    unknown_expected_count: bool = False,
) -> DcgmReplay:
    """Deploy the inventory wrapper and gate on its first rewritten query.

    The clock is the context's (the wall clock live; a virtual one in tests),
    read as an epoch for the plan the node-side wrapper compares against
    ``time.time()``. The gate evidence (``inventory-gate.json``) records the
    window, the deploy duration and the plan's remaining validity.
    """

    hold = _hold(ctx)
    composed_at = ctx.clock().timestamp()
    plan = inventory_plan(operations, composed_at=composed_at, hold=hold)
    replay = DcgmReplay(
        ctx,
        passthrough_plan(),
        hold,
        inventory_plan=plan,
        boot_id=boot_id,
        unknown_expected_count=unknown_expected_count,
    )
    replay.deploy()  # pass-through field tool; the wrapper rewrites on its own
    repointed_at = ctx.clock().timestamp()
    window = inventory_window_seconds(survey)
    deploy_seconds = repointed_at - composed_at
    remaining = plan["not_after"] - repointed_at
    gate: dict[str, Any] = {
        "inventory_interval_seconds": survey.inventory_interval,
        "startup_spread_seconds": _seconds(
            survey, STARTUP_SPREAD_KEY, DEFAULT_STARTUP_SPREAD_SECONDS
        ),
        "startup_jitter_seconds": _seconds(
            survey, STARTUP_JITTER_KEY, DEFAULT_STARTUP_JITTER_SECONDS
        ),
        "window_seconds": window,
        "hold_seconds": hold,
        "deploy_allowance_seconds": INVENTORY_DEPLOY_ALLOWANCE_SECONDS,
        "plan_composed_at": _iso(composed_at),
        "repointed_at": _iso(repointed_at),
        "deploy_seconds": round(deploy_seconds, 1),
        "plan_not_after": _iso(plan["not_after"]),
        "plan_remaining_seconds": round(remaining, 1),
    }
    ctx.evidence.write("inventory-gate", gate)
    required = max(hold, window)
    if remaining < required:
        raise VariantInconclusive(
            "harness fixture failed: the reviewed inventory plan stays valid "
            f"{remaining:.0f} s after the repoint, below the {required} s the variant "
            f"needs (the deploy took {deploy_seconds:.0f} s against a "
            f"{INVENTORY_DEPLOY_ALLOWANCE_SECONDS} s allowance; hold {hold} s, "
            f"first-query window {window} s); nothing was graded"
        )
    gate["rewritten"] = replay.wait_inventory_rewritten(window)
    ctx.evidence.write("inventory-gate", gate)
    return replay


def _notification_errors(
    doc: dict[str, Any], incidents: list[dict[str, Any]], category: str
) -> list[str]:
    wanted = {str(item["incident_id"]) for item in incidents}
    mails = [
        item
        for item in doc.get("notifications") or []
        if item.get("incident_id") in wanted
    ]
    if not mails:
        return [f"no {category} notification for the inventory incident"]
    return []


def uuid_changed(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = survey.inventory
    index = int(gpus[-1]["index"])
    since = utc_now()
    replay = _replay(
        ctx,
        survey,
        [{"op": "replace_uuid", "index": index, "uuid": replay_uuid(ctx.run_id)}],
    )
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: inventory_incidents(doc, IDENTITY),
        timeout_seconds=survey.inventory_interval + survey.health_summary + 300,
        name="detection",
    )
    incidents = inventory_incidents(detection, IDENTITY)
    stages = {
        "chain": chain_errors(
            detection,
            incidents,
            GpuChain(
                label="gpu_inventory_identity_changed",
                operations=("FREEZE_EVIDENCE", "VALIDATE_GPU"),
                forbidden=RESET,
                incident_states=frozenset({"ACTION_PENDING", "ESCALATED", "RECOVERED"}),
            ),
        ),
        "read_only": read_only_errors(detection, incidents, "identity_changed"),
        "notification": _notification_errors(detection, incidents, "OPERATOR_REVIEW"),
        "hardware": no_hardware(detection),
    }
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=[
            "a UUID change is diagnosed read-only; VALIDATE_GPU cannot confirm the replay UUID and ends inconclusive"
        ],
        extra={"index": index},
    )


def bdf_changed(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = survey.inventory
    if len(gpus) < 2:
        raise VariantNotRun("CAPABILITY_ABSENT", "need two GPUs to swap PCI bus ids")
    indexes = [int(gpus[0]["index"]), int(gpus[-1]["index"])]
    since = utc_now()
    replay = _replay(ctx, survey, [{"op": "swap_bdf", "indexes": indexes}])
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: inventory_incidents(doc, IDENTITY)
        or inventory_incidents(doc, MISMATCH),
        timeout_seconds=survey.inventory_interval + survey.health_summary + 120,
        name="detection",
    )
    # expected_outcome=FAIL: PASS only once a BDF-only change is diagnosed.
    stages = {
        "bdf_diagnosed": []
        if inventory_incidents(detection, IDENTITY)
        or inventory_incidents(detection, MISMATCH)
        else [
            "a BDF change with unchanged UUIDs produced no finding (identity compares UUID sets only)"
        ],
        "hardware": no_hardware(detection),
    }
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(stages, extra={"indexes": indexes})


def count_unchanged(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    index = int(survey.inventory[-1]["index"])
    since = utc_now()
    replay = _replay(
        ctx,
        survey,
        [{"op": "replace_uuid", "index": index, "uuid": replay_uuid(ctx.run_id)}],
    )
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: inventory_incidents(doc, IDENTITY),
        timeout_seconds=survey.inventory_interval + survey.health_summary + 300,
        name="detection",
    )
    stages = {
        "identity": []
        if inventory_incidents(detection, IDENTITY)
        else ["a same-count UUID change was not diagnosed"],
        "no_count_mismatch": []
        if not inventory_incidents(detection, MISMATCH)
        else ["a same-count UUID change was treated as a count mismatch (reboot path)"],
        "hardware": no_hardware(detection),
    }
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=["same count: diagnose the identity, never reboot for a count mismatch"],
        extra={"index": index},
    )


def unknown_expected_count(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    index = int(survey.inventory[-1]["index"])
    since = utc_now()
    # An identity change (so a snapshot is accepted and graded) under an unset
    # count and an unknown instance type -> gpu_expected_count_unknown.
    replay = _replay(
        ctx,
        survey,
        [
            {
                "op": "replace_uuid",
                "index": index,
                "uuid": replay_uuid(ctx.run_id, "count"),
            }
        ],
        unknown_expected_count=True,
    )
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: inventory_incidents(doc, UNKNOWN_COUNT),
        timeout_seconds=survey.inventory_interval + survey.health_summary + 300,
        name="detection",
    )
    incidents = inventory_incidents(detection, UNKNOWN_COUNT)
    stages = {
        "chain": chain_errors(
            detection,
            incidents,
            GpuChain(
                label="gpu_expected_count_unknown",
                operations=("FREEZE_EVIDENCE",),
                forbidden=RESET,
            ),
        ),
        "notification": _notification_errors(detection, incidents, "OPERATOR_REVIEW"),
        "hardware": no_hardware(detection),
    }
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=[
            "an unknown expected count is evidence-only; no default is guessed and no reboot"
        ],
        extra={"index": index},
    )


def stale_boot(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    index = int(survey.inventory[-1]["index"])
    boot, origin = replay_boot_id(survey, ctx.run_id)
    since = utc_now()
    replay = _replay(
        ctx,
        survey,
        [
            {
                "op": "replace_uuid",
                "index": index,
                "uuid": replay_uuid(ctx.run_id, "stale"),
            }
        ],
        boot_id=boot,
    )
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: inventory_incidents(doc, IDENTITY),
        timeout_seconds=survey.inventory_interval + survey.health_summary + 300,
        name="detection",
    )
    # expected_outcome=FAIL: PASS only once the product refuses to act on an
    # old-boot snapshot (then the identity finding is absent).
    stages = {
        "stale_boot_ignored": []
        if not inventory_incidents(detection, IDENTITY)
        else [
            "the product acted on inventory from an old boot id (boot is not compared at ingest)"
        ],
    }
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=[f"replayed boot id from {origin}"],
        extra={"index": index, "boot_id": boot},
    )


def approved_rebaseline(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    index = int(survey.inventory[-1]["index"])
    since = utc_now()
    # A stable new identity across several inventory deliveries: the first
    # opens the finding, the next matching snapshot clears the signal, and the
    # old identity is not re-reported for ever.
    replay = _replay(
        ctx,
        survey,
        [
            {
                "op": "replace_uuid",
                "index": index,
                "uuid": replay_uuid(ctx.run_id, "rebaseline"),
            }
        ],
    )
    wait_gpu_incident(
        ctx,
        since,
        lambda doc: inventory_incidents(doc, IDENTITY),
        timeout_seconds=survey.inventory_interval + survey.health_summary + 300,
        name="detection",
    )
    ctx.hold(
        min(survey.inventory_interval * 3 + 30, _hold(ctx)), name="rebaseline-observe"
    )
    final = activity_now(ctx, since)
    incidents = inventory_incidents(final, IDENTITY)
    stages = {
        "single_report": []
        if len(incidents) <= 1
        else [
            f"the stable new identity was re-reported {len(incidents)} times instead of rebaselining"
        ],
    }
    replay.close()
    ctx.cleanup.discard("dcgm-replay")
    return result(
        stages,
        notes=[
            "the baseline advances implicitly (the next matching snapshot clears the signal); "
            "there is no explicit identity-approval API -- see the report's product-gap note"
        ],
        extra={"index": index, "identity_incidents": len(incidents)},
    )


def no_unapproved_reset(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    index = int(survey.inventory[-1]["index"])
    since = utc_now()
    replay = _replay(
        ctx,
        survey,
        [
            {
                "op": "replace_uuid",
                "index": index,
                "uuid": replay_uuid(ctx.run_id, "noreset"),
            }
        ],
    )
    detection = wait_gpu_incident(
        ctx,
        since,
        lambda doc: inventory_incidents(doc, IDENTITY),
        timeout_seconds=survey.inventory_interval + survey.health_summary + 300,
        name="detection",
    )
    stages = {
        "no_reset": no_hardware(detection),
        "diagnosed": []
        if inventory_incidents(detection, IDENTITY)
        else ["the identity change was not even diagnosed"],
    }
    # close() restores the real nvidia-smi and removes the wrapper, so no forged
    # UUID is left on the node to feed a later real reset.
    receipts = replay.close()
    ctx.cleanup.discard("dcgm-replay")
    stages["no_residue"] = (
        []
        if (receipts.get("stop") or {}).get("run_dir_removed")
        else ["the forged inventory fixture was left on the node"]
    )
    return result(
        stages,
        notes=[
            "identity findings never reset or reboot; the forged UUID is removed in cleanup"
        ],
        extra={"index": index},
    )


VARIANTS = [
    gpu_spec(
        CASE,
        "uuid_changed",
        "one GPU row reports a replay UUID",
        "CRITICAL RUN_DIAGNOSTICS identity finding: FREEZE_EVIDENCE + VALIDATE_GPU, operator review, no reset",
        uuid_changed,
        risk="M",
        max_minutes=20,
    ),
    gpu_spec(
        CASE,
        "bdf_changed",
        "two GPU rows swap PCI bus ids (UUIDs unchanged)",
        "a BDF change is diagnosed read-only (batch 160: the identity is the UUID set plus each "
        "UUID's PCI address)",
        bdf_changed,
        max_minutes=16,
    ),
    gpu_spec(
        CASE,
        "count_unchanged",
        "a replay UUID at the same GPU count",
        "diagnose the identity change; never treat a same-count change as a count mismatch/reboot",
        count_unchanged,
        risk="M",
        max_minutes=20,
    ),
    gpu_spec(
        CASE,
        "unknown_expected_count",
        "an identity change under an unset count and unknown instance type",
        "WARNING COLLECT_EVIDENCE: FREEZE_EVIDENCE only, operator review, no guessed count or reboot",
        unknown_expected_count,
        risk="M",
        max_minutes=20,
    ),
    gpu_spec(
        CASE,
        "stale_boot",
        "an identity change carried on an old boot id",
        "old-boot inventory does not override the current one (batch 160: a snapshot from a boot "
        "the node's Agent is not running keeps the baseline and is not graded)",
        stale_boot,
        max_minutes=18,
    ),
    gpu_spec(
        CASE,
        "approved_rebaseline",
        "a stable new identity across several snapshots",
        "the first snapshot diagnoses, the next matching one rebaselines; the old identity is not re-reported for ever",
        approved_rebaseline,
        risk="M",
        max_minutes=22,
    ),
    gpu_spec(
        CASE,
        "no_unapproved_reset",
        "a replay UUID change, then cleanup",
        "no identity finding resets or reboots; the forged UUID is removed so it cannot feed a later real reset",
        no_unapproved_reset,
        risk="M",
        max_minutes=20,
    ),
]
