"""GF-GENERAL-LIVE device-signal case 012: GPU missing / fallen off the bus.

The sibling device-signal categories live in their own modules: 007 SMART in
``general_live_smart.py``, 010 EFA/RDMA counters in ``general_live_efa.py`` and
015 BMC in ``general_live_bmc.py``. The GPU inventory category is, by the
coverage matrix, ``live-replay`` or ``vendor-lab`` only: the product grades it
from a raw reading (the ``nvidia-smi`` GPU inventory and ``/proc/driver/
nvidia``), and proving the processing chain means replacing exactly that
targeted reading in the one host-collector unit's private environment, or
inducing a real hardware fault in a vendor lab.

This site has no vendor lab and must never unbind or remove a real GPU, so the
vendor-lab variants stay NOT_RUN(VENDOR_LAB_ONLY).

The 012 read-path variants are wired on the shared host source replay
(``general_live_source_replay.py``, approval ``HOST_SOURCE_REPLAY``). The host
collector reads the GPU inventory from one ``nvidia-smi --query-gpu=uuid,
utilization.gpu`` call per tick (``collectors/host/gpu_rank.py`` ``GPU_QUERY_ARGV``, memoised for the
utilization and inventory contributors) and, only when that query fails, counts
the driver's own bound GPUs under ``/proc/driver/nvidia/gpus``
(``collectors/host/inventory.py`` ``_gpu_inventory`` / ``_discovered_gpu_count``).
The reviewed wrapper on the unit's PATH drops one *real* UUID from one call
(``single_sample``), or fails the query like an NVML initialisation failure for
a few ticks (``query_failed``); ``proc_unreadable`` gives the unit a copy of
``/proc/driver/nvidia`` without ``gpus`` while that query fails, so the
product's listing of the directory fails too (ENOENT -- the two-read-failure
case the spec marks as the known risk; for a root unit no directory can be made
*unreadable*: ``InaccessiblePaths=`` leaves an empty directory root lists,
which is "zero GPUs bound", the very reading this variant must never feed the
product); ``expectation_wrong`` overrides the expected count in the unit's
environment. The product's own records decide: a ``gpu_inventory_
mismatch`` finding is the CRITICAL REBOOT_NODE rule's input (``host_health.py``
METRIC_RULES), so none of these may ever mint one, plan a hardware step or
isolate the node; a failed query must leave ``failure_mode`` DRIVER_QUERY_FAILED
/ INVENTORY_UNAVAILABLE context, never a zero count. The wrapper's journal
(``replay.audit``) is the raw-source oracle of what the product actually asked
and was served.

``expectation_wrong`` reads the configuration finding batch 174 introduced:
``host_health.gpu_inventory_verdicts`` re-reads every ``gpu_inventory_mismatch``
sample against the control plane's instance table before grading, and a
configured count that contradicts the type while the node lists exactly the
type's GPUs (the driver's bound count agreeing) is re-read as 0 -- no mismatch
finding -- and graded as ``gpu_expected_count_contradicted`` WARNING /
COLLECT_EVIDENCE: a FREEZE_EVIDENCE-only workflow, an OPERATOR_REVIEW
notification, and an incident reason naming the configured count, the
instance type and its count, the listed count and the bound count. The
override is closed as soon as that finding exists; a corrected configuration
reads the contradiction at 0 and the episode closes.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from scripts.e2e.regional.general_live_common import (
    activity,
    run_survey,
    wait_settled,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    HARDWARE_OPERATIONS,
    LIVE_REPLAY,
    VENDOR_LAB,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_source_replay import (
    HOST_SOURCE_REPLAY,
    NVIDIA_PROC_TREE,
    HostSourceReplay,
    command_plan,
    host_source_replay,
    rule,
)
from scripts.e2e.regional.general_live_verdicts import (
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    isolation_operations,
    node_isolated,
    operations,
    samples,
    verdict,
    workflows_of,
)

CASE_012 = "GF-GENERAL-LIVE-012"

#: The host collector's GPU inventory samples (``collectors/host/inventory.py``)
#: plus the per-GPU utilization latest metric whose ``device`` is the UUID.
GPU_METRICS = frozenset(
    {
        "gpu_inventory_mismatch",
        "gpu_inventory_expected_count",
        "gpu_inventory_active_count",
        "gpu_inventory_missing_count",
        "gpu_inventory_excess_count",
        "gpu_inventory_discovered_count",
        "host_gpu_utilization_percent",
    }
)
#: The failing query the wrapper serves: nvidia-smi prints the NVML failure on
#: stdout and exits non-zero (9 = "NVIDIA driver is not loaded" in its manual);
#: the product only needs the non-zero exit and an empty UUID list.
NVML_FAILURE_TEXT = "Failed to initialize NVML: Driver/library version mismatch\n"
NVML_FAILURE_EXIT = 9
#: Consecutive ticks the query fails: more than the inventory rule's 2-sample
#: confirmation (GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES), so the
#: "second failed read minted REBOOT_NODE" history is exercised.
FAILED_TICKS = 4
#: ``failure_mode`` labels the inventory contributor stamps on a failed read.
DRIVER_QUERY_FAILED = "DRIVER_QUERY_FAILED"
INVENTORY_UNAVAILABLE = "INVENTORY_UNAVAILABLE"
#: How long a chain the product minted on the override is waited to settle:
#: the FREEZE_EVIDENCE-only review chain closes within a health summary; the
#: budget still covers a regressed REBOOT_NODE chain so the FAIL is recorded on
#: a settled record (reboot, validations, restore).
REBOOT_SETTLE_BUDGET = 1800
#: The configuration finding of a mistyped expected GPU count
#: (``host_health.GPU_EXPECTED_COUNT_CONTRADICTED_METRIC``, batch 174) and the
#: category of the notification it mails (``app/ingest/node_health.py``).
CONTRADICTION_METRIC = "gpu_expected_count_contradicted"
OPERATOR_REVIEW_CATEGORY = "OPERATOR_REVIEW"
#: Steps a diagnostic-only or evidence-only chain must never contain.
ISOLATION_OPERATIONS = frozenset({"MARK_UNSCHEDULABLE", "QUARANTINE", "STOP_WORKLOADS"})


# --------------------------------------------------------------------------- #
# 012 GPU inventory read paths (host source replay of the nvidia-smi query,
# /proc/driver/nvidia/gpus and the expected count)
# --------------------------------------------------------------------------- #
def _node_gpus(ctx: VariantContext) -> tuple[list[str], int | None]:
    """The node's GPU UUIDs and configured expected count, from the product.

    ``host_gpu_utilization_percent`` is kept per GPU in the latest-metrics
    table with the UUID as its device, and ``gpu_inventory_expected_count`` is
    the count the host collector grades against -- the real UUID a replay drops
    and the real expectation a wrong override is judged by (spec 012: "固定真实
    期望数", a replay must drop a *real* UUID from the full query).
    """

    latest = activity(
        ctx,
        utc_now(),
        metrics={"host_gpu_utilization_percent", "gpu_inventory_expected_count"},
    ).get("metric_latest")
    uuids = sorted(
        str(item["device"])
        for item in latest or []
        if item.get("name") == "host_gpu_utilization_percent"
        and str(item.get("device") or "").startswith("GPU-")
    )
    if not uuids:
        raise VariantBlocked(
            "FIXTURE_NOT_READY",
            "the product's latest host telemetry names no GPU UUID for the node "
            "(host_gpu_utilization_percent); the replay needs a real UUID",
        )
    expected: int | None = None
    for item in latest or []:
        if item.get("name") == "gpu_inventory_expected_count" and isinstance(
            item.get("value"), (int, float)
        ):
            expected = int(item["value"])
    return uuids, expected


def served_rewrites(audit: dict[str, Any]) -> list[dict[str, Any]]:
    """The wrapper journal entries where a rule rewrote the utilization query."""

    return [
        item
        for item in audit.get("entries") or []
        if item.get("query") == "utilization" and item.get("applied_rules")
    ]


def _inventory_samples(observed: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        item
        for item in samples(observed, name)
        if isinstance(item.get("value"), (int, float))
    ]


def never_zero_errors(observed: dict[str, Any]) -> list[str]:
    """No sample ever confirmed a missing GPU or reported zero GPUs."""

    errors = [
        f"gpu_inventory_mismatch={item['value']:g} was reported at {item['observed_at']}"
        for item in _inventory_samples(observed, "gpu_inventory_mismatch")
        if item["value"] >= 1
    ]
    errors += [
        f"gpu_inventory_active_count=0 was reported at {item['observed_at']}"
        for item in _inventory_samples(observed, "gpu_inventory_active_count")
        if item["value"] == 0
    ]
    return errors


def gpu_failure_mode_errors(
    observed: dict[str, Any], mode: str, *, expected: int | None, discovered: bool
) -> list[str]:
    """The failed read left its context: ``failure_mode`` and the bound count."""

    stamped = [
        item
        for item in _inventory_samples(observed, "gpu_inventory_expected_count")
        if (item.get("labels") or {}).get("failure_mode") == mode
    ]
    if not stamped:
        return [f"no gpu_inventory_expected_count sample carries failure_mode={mode}"]
    errors = []
    for item in stamped:
        labels = item.get("labels") or {}
        if discovered:
            if "discovered_count" not in labels:
                errors.append(
                    f"a {mode} sample at {item['observed_at']} lacks discovered_count"
                )
            elif expected is not None and labels["discovered_count"] != str(expected):
                errors.append(
                    f"discovered_count={labels['discovered_count']} at "
                    f"{item['observed_at']} is not the node's {expected} bound GPUs"
                )
        elif "discovered_count" in labels:
            errors.append(
                f"an {mode} sample at {item['observed_at']} fabricated "
                f"discovered_count={labels['discovered_count']}"
            )
    if not discovered and _inventory_samples(
        observed, "gpu_inventory_discovered_count"
    ):
        errors.append(
            "a gpu_inventory_discovered_count was reported from an unreadable procfs"
        )
    return errors


def collection_error_errors(observed: dict[str, Any], *needles: str) -> list[str]:
    """At least one delivered batch names the failed contributor and why."""

    for batch in observed.get("batches") or []:
        for error in batch.get("collection_errors") or []:
            if all(needle in str(error) for needle in needles):
                return []
    return [
        "no delivered batch carries a collection error naming "
        + " and ".join(repr(item) for item in needles)
    ]


def _isolation_errors(ctx: VariantContext, observed: dict[str, Any]) -> list[str]:
    isolation = isolation_operations(observed)
    errors = [f"isolation operations planned: {isolation}"] if isolation else []
    if node_isolated(ctx.regional.node_snapshot(ctx.settings.node)):
        errors.append("the node is isolated after a GPU inventory read failure")
    return errors


def _await_inventory_recovery(
    ctx: VariantContext, since: datetime, expected: int | None, *, name: str
) -> dict[str, Any]:
    """Fresh product evidence that the inventory reads the full count again."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics=GPU_METRICS)
        active = _inventory_samples(last, "gpu_inventory_active_count")
        if active and (expected is None or active[-1]["value"] == expected):
            mismatch = _inventory_samples(last, "gpu_inventory_mismatch")
            if mismatch and mismatch[-1]["value"] == 0:
                return last
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name=name,
    )
    if value is None:
        return {
            "errors": [
                "no fresh gpu_inventory_active_count at the expected count with "
                "gpu_inventory_mismatch=0 reached the product after the window"
            ],
            "activity": last,
        }
    return {"errors": [], "activity": value}


def _nvidia_smi_window(
    ctx: VariantContext,
    replay: HostSourceReplay,
    rules: list[dict[str, Any]],
    *,
    trees: tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    plan = command_plan(
        "nvidia-smi", rules, not_after_seconds=ctx.spec.max_minutes * 60
    )
    ctx.evidence.write("nvidia-smi-plan", plan)
    ctx.evidence.write("wrapper-install", replay.install_wrapper("nvidia-smi", plan))
    opened = replay.open_window(
        wrappers=["nvidia-smi"],
        trees=list(trees),
        restore_seconds=ctx.spec.max_minutes * 60,
    )
    ctx.evidence.write("source-window-open", opened)
    return opened


def _read_audit(ctx: VariantContext, replay: HostSourceReplay) -> list[dict[str, Any]]:
    audit = replay.audit("nvidia-smi")
    served = served_rewrites(audit)
    ctx.evidence.write("nvidia-smi-audit", audit)
    ctx.evidence.append_oracle(
        [
            {
                "utilization_rewrites_served": len(served),
                "calls": audit.get("calls"),
                **replay.evidence(tool="nvidia-smi"),
            }
        ],
        source="nvidia-smi-audit",
    )
    return served


def gpu_single_sample(ctx: VariantContext) -> dict[str, Any]:
    """One tick lists one real UUID less: not confirmed, no finding, no reboot.

    ``max_calls=1`` bounds the drop to the first utilization query after the
    unit restarts (that tick is the collector's baseline delivery, so the
    product's record shows ``gpu_inventory_missing_count=1`` with
    ``consecutive_mismatch_samples=1``); the next tick lists every GPU again and
    the 2-sample confirmation never closes.
    """

    run_survey(ctx)
    uuids, expected = _node_gpus(ctx)
    dropped = uuids[0]
    replay = host_source_replay(ctx)
    started = utc_now()
    _nvidia_smi_window(
        ctx,
        replay,
        [rule("utilization", "drop_row", uuid=dropped, max_calls=1)],
    )
    ctx.hold(ctx.host_interval * 3 + 30, name="single-sample")
    served = _read_audit(ctx, replay)
    if len(served) != 1:
        raise VariantInconclusive(
            f"the wrapper rewrote the utilization query {len(served)} times, not once"
        )
    recovered = _await_inventory_recovery(
        ctx, started, expected, name="single-sample-recovery"
    )
    observed = recovered["activity"] or activity(ctx, started, metrics=GPU_METRICS)
    missing = [
        {
            "observed_at": item["observed_at"],
            "value": item["value"],
            "consecutive": (item.get("labels") or {}).get(
                "consecutive_mismatch_samples"
            ),
        }
        for item in _inventory_samples(observed, "gpu_inventory_missing_count")
        if item["value"] >= 1
    ]
    stages = {
        "not_confirmed": never_zero_errors(observed),
        "no_finding": chain_errors(
            observed, ChainExpectation(metric="gpu_inventory_mismatch", present=False)
        ),
        "no_reboot": hardware_errors(observed),
        "not_isolated": _isolation_errors(ctx, observed),
        "recovered": recovered["errors"],
    }
    return _result_012(
        stages,
        notes=[
            "one missing sample is below the 2-sample confirmation; the drop was "
            "served exactly once (wrapper journal)",
            "the baseline tick after the unit restart is delivered, so the single "
            "missing sample is usually visible in the records",
        ],
        extra={
            "dropped_uuid": dropped,
            "expected_count": expected,
            "missing_samples": missing,
        },
    )


def _failed_query(ctx: VariantContext, *, proc_unreadable: bool) -> dict[str, Any]:
    run_survey(ctx)
    _uuids, expected = _node_gpus(ctx)
    replay = host_source_replay(ctx)
    started = utc_now()
    if proc_unreadable:
        # The unit reads a copy of the real /proc/driver/nvidia that has no
        # ``gpus``: ``_discovered_gpu_count``'s listing raises (ENOENT), the
        # same OSError branch as an unreadable directory, and the version file
        # stays real. InaccessiblePaths= on ``gpus`` would leave root an empty
        # directory -- 0 GPUs bound, a count, never "could not be read".
        ctx.evidence.write(
            "nvidia-proc-fixture", replay.build_nvidia_proc(omit_gpus=True)
        )
    _nvidia_smi_window(
        ctx,
        replay,
        [
            rule(
                "utilization",
                "replace_output",
                stdout=NVML_FAILURE_TEXT,
                stderr=NVML_FAILURE_TEXT,
                exit_code=NVML_FAILURE_EXIT,
                max_calls=FAILED_TICKS,
            )
        ],
        trees=((NVIDIA_PROC_TREE, "/proc/driver/nvidia"),) if proc_unreadable else (),
    )
    ctx.hold(ctx.host_interval * (FAILED_TICKS + 1) + 30, name="failed-query")
    served = _read_audit(ctx, replay)
    if len(served) < 2:
        raise VariantInconclusive(
            f"the product asked the failing utilization query {len(served)} times; "
            "two failed reads are the spec's confirmation window"
        )
    recovered = _await_inventory_recovery(
        ctx, started, expected, name="failed-query-recovery"
    )
    observed = recovered["activity"] or activity(ctx, started, metrics=GPU_METRICS)
    mode = INVENTORY_UNAVAILABLE if proc_unreadable else DRIVER_QUERY_FAILED
    needles = ("GPU inventory query failed",) + (
        ("could not be read", "unknown as well") if proc_unreadable else ()
    )
    stages = {
        "unknown_observed": collection_error_errors(observed, *needles),
        "failure_mode": gpu_failure_mode_errors(
            observed, mode, expected=expected, discovered=not proc_unreadable
        ),
        "never_zero": never_zero_errors(observed),
        "no_finding": chain_errors(
            observed, ChainExpectation(metric="gpu_inventory_mismatch", present=False)
        ),
        "no_reboot": hardware_errors(observed),
        "not_isolated": _isolation_errors(ctx, observed),
        "recovered": recovered["errors"],
    }
    notes = [
        f"the failing query was served {len(served)} times (wrapper journal); "
        "the product reported failure context, never a count",
    ]
    if proc_unreadable:
        notes.append(
            "both reads failed (NVML, and /proc/driver/nvidia/gpus absent from the "
            "unit's copy of the tree: ENOENT): the inventory is "
            "INVENTORY_UNAVAILABLE with no discovered count -- the spec's known "
            "REBOOT_NODE risk must not reproduce"
        )
    return _result_012(
        stages,
        notes=notes,
        extra={"expected_count": expected, "failed_ticks_served": len(served)},
    )


def gpu_query_failed(ctx: VariantContext) -> dict[str, Any]:
    """An NVML initialisation failure for a few ticks: UNKNOWN, never zero GPUs."""

    return _failed_query(ctx, proc_unreadable=False)


def gpu_proc_unreadable(ctx: VariantContext) -> dict[str, Any]:
    """The query fails and ``/proc/driver/nvidia/gpus`` is unreadable in the unit."""

    return _failed_query(ctx, proc_unreadable=True)


def contradiction_reason_errors(
    incident: dict[str, Any], labels: Mapping[str, Any], *, wrong: int
) -> list[str]:
    """The incident's reason names the four counts the operator needs.

    ``host_health.gpu_expected_count_contradicted_rule`` spells the configured
    count (E), the instance type and its count (T), the listed count (O) and,
    when the collector reported one, the driver's bound count (D); the
    operator-review mail carries the reason, not the labels, so the words must
    be there. ``labels`` are the product's own labels on the overridden
    expectation sample (``node_instance_type``, ``observed_count``,
    ``discovered_count``, ``instance_type_gpu_count``).
    """

    reasons = " ".join(str(item) for item in incident.get("reasons") or [])
    wanted = [f"configured expected GPU count {wrong}"]
    instance_type = labels.get("node_instance_type")
    wanted.append(
        f"contradicts instance type {instance_type}"
        if instance_type
        else "contradicts instance type"
    )
    type_count = labels.get("instance_type_gpu_count")
    if type_count:
        wanted.append(f"carries {type_count} GPUs")
    observed = labels.get("observed_count")
    wanted.append(f"the node lists {observed}" if observed else "the node lists")
    bound = labels.get("discovered_count")
    if bound:
        wanted.append(f"driver has {bound} bound")
    return [
        f"the {CONTRADICTION_METRIC} incident {incident.get('incident_id')} reason "
        f"does not say {text!r}: {reasons[:300]!r}"
        for text in wanted
        if text not in reasons
    ]


def operator_review_errors(observed: dict[str, Any], incident_id: str) -> list[str]:
    mails = [
        item
        for item in observed.get("notifications") or []
        if item.get("incident_id") == incident_id
        and item.get("category") == OPERATOR_REVIEW_CATEGORY
    ]
    if not mails:
        return [
            f"no {OPERATOR_REVIEW_CATEGORY} notification names the "
            f"{CONTRADICTION_METRIC} incident {incident_id}"
        ]
    if len(mails) > 1:
        return [f"{len(mails)} operator-review notifications for one contradiction"]
    return []


def gpu_expectation_wrong(ctx: VariantContext) -> dict[str, Any]:
    """A wrong expected count is a configuration finding, not a missing GPU.

    The real count is read from the product (``gpu_inventory_expected_count``);
    the unit's environment gets that count plus one. The window is closed as
    soon as the product has a finding on the override (the contradiction, or a
    mismatch -- the old behaviour) or recorded it twice, so the exposure is
    the minimum the rule needs; the chain the product then runs is its own
    and is waited to settle before the records are judged.
    """

    run_survey(ctx)
    _uuids, expected = _node_gpus(ctx)
    if expected is None:
        raise VariantBlocked(
            "FIXTURE_NOT_READY",
            "the product records no gpu_inventory_expected_count for the node; the "
            "real expectation a wrong override is judged against is unknown",
        )
    wrong = expected + 1
    replay = host_source_replay(ctx)
    started = utc_now()
    opened = replay.open_window(
        env={"GPU_FAULT_EXPECTED_GPU_COUNT": str(wrong)},
        restore_seconds=ctx.spec.max_minutes * 60,
    )
    ctx.evidence.write("source-window-open", opened)
    metrics = GPU_METRICS | {CONTRADICTION_METRIC}

    def overrides(document: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            item
            for item in _inventory_samples(document, "gpu_inventory_expected_count")
            if item["value"] == wrong
        ]

    def reacted() -> dict[str, Any] | None:
        current = activity(ctx, started, metrics=metrics)
        if (
            finding_incidents(current, CONTRADICTION_METRIC)
            or finding_incidents(current, "gpu_inventory_mismatch")
            or len(overrides(current)) >= 2
        ):
            return current
        return None

    ctx.wait_until(
        reacted,
        timeout_seconds=ctx.host_interval * 6 + 120,
        poll_seconds=10,
        name="expectation-reaction",
    )
    ctx.evidence.write("source-window-close", replay.close())
    closed_at = utc_now()
    observed = activity(ctx, started, metrics=metrics)
    recorded = [
        {"observed_at": item["observed_at"], **(item.get("labels") or {})}
        for item in overrides(observed)
    ]
    if not recorded:
        raise VariantInconclusive(
            f"the product never recorded the overridden expectation {wrong}"
        )
    contradictions = finding_incidents(observed, CONTRADICTION_METRIC)
    mismatches = finding_incidents(observed, "gpu_inventory_mismatch")
    if contradictions or mismatches:
        observed = wait_settled(
            ctx,
            started,
            [str(item["incident_id"]) for item in (*contradictions, *mismatches)],
            timeout_seconds=REBOOT_SETTLE_BUDGET,
            metrics=metrics,
            name="expectation-settled",
        )
        contradictions = finding_incidents(observed, CONTRADICTION_METRIC)
    recovered = _await_inventory_recovery(
        ctx, closed_at, expected, name="expectation-recovery"
    )
    labels: dict[str, Any] = dict(recorded[-1])
    stages = {
        "not_a_missing_gpu": chain_errors(
            observed, ChainExpectation(metric="gpu_inventory_mismatch", present=False)
        ),
        "configuration_finding": chain_errors(
            observed,
            ChainExpectation(
                metric=CONTRADICTION_METRIC,
                operations=("FREEZE_EVIDENCE",),
                forbidden=HARDWARE_OPERATIONS | ISOLATION_OPERATIONS,
            ),
        ),
        "evidence_only": [
            f"the {CONTRADICTION_METRIC} incident {item['incident_id']} carries "
            f"effective_action {item.get('effective_action')}, not COLLECT_EVIDENCE"
            for item in contradictions
            if item.get("effective_action") != "COLLECT_EVIDENCE"
        ]
        + [
            f"the {CONTRADICTION_METRIC} workflow {workflow.get('request_id')} plans "
            f"{operations(workflow)}: more than FREEZE_EVIDENCE"
            for item in contradictions
            for workflow in workflows_of(observed, str(item["incident_id"]))
            if set(operations(workflow)) - {"FREEZE_EVIDENCE"}
        ],
        "reason_names_the_counts": [
            error
            for item in contradictions[:1]
            for error in contradiction_reason_errors(item, labels, wrong=wrong)
        ],
        "operator_review": [
            error
            for item in contradictions[:1]
            for error in operator_review_errors(observed, str(item["incident_id"]))
        ],
        "no_reboot": hardware_errors(observed),
        "not_isolated": _isolation_errors(ctx, observed),
        "recovered": recovered["errors"],
    }
    return _result_012(
        stages,
        notes=[
            f"expected count {expected} (product record) overridden to {wrong} in "
            "the host collector unit only; the control plane re-reads the mismatch "
            "against the instance type and grades the contradiction WARNING / "
            "COLLECT_EVIDENCE for an operator, never a reboot",
        ],
        extra={
            "expected_count": expected,
            "overridden_count": wrong,
            "recorded_expectations": recorded[-3:],
            "incidents": [
                str(item["incident_id"]) for item in (*contradictions, *mismatches)
            ],
        },
    )


def _result_012(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        "proof_scope": "live-replay",
        **(extra or {}),
    }


def _gpu_read_path(
    variant: str,
    *,
    injection: str,
    expected: str,
    execute: Any,
    cleanup: str = "close the source window (deadman-backed); confirm clean",
    max_minutes: int = 16,
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE_012,
        variant=variant,
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection=injection,
        expected=expected,
        cleanup=cleanup,
        max_minutes=max_minutes,
        approvals=(HOST_SOURCE_REPLAY,),
        execute=execute,
    )


VARIANTS = [
    # ----------------------------------------------------------------------- #
    # 012 GPU missing / fallen off the bus
    # ----------------------------------------------------------------------- #
    deferred_variant(
        CASE_012,
        "confirmed_missing",
        proof_mode=VENDOR_LAB,
        reason=(
            "VENDOR_LAB_ONLY: a real GPU off the bus; this lane must never unbind or "
            "remove a real GPU, and the 2-sample confirmation feeds a provider reboot"
        ),
        lanes=("eks", "hp"),
        expected="two confirmed missing samples -> approved reboot chain, new boot id",
    ),
    _gpu_read_path(
        "single_sample",
        injection=(
            "drop one real GPU UUID from exactly one nvidia-smi uuid,utilization.gpu "
            "query (reviewed wrapper on the host collector unit's PATH, max_calls=1)"
        ),
        expected=(
            "gpu_inventory_missing_count=1 once, gpu_inventory_mismatch stays 0; no "
            "finding, no REBOOT_NODE, no isolation; the next ticks list every GPU"
        ),
        execute=gpu_single_sample,
    ),
    _gpu_read_path(
        "query_failed",
        injection=(
            f"serve an NVML initialisation failure (exit {NVML_FAILURE_EXIT}) to the "
            f"host collector's uuid,utilization.gpu query for {FAILED_TICKS} ticks"
        ),
        expected=(
            "collection error 'GPU inventory query failed'; gpu_inventory_expected_count "
            "carries failure_mode=DRIVER_QUERY_FAILED and the driver's discovered_count; "
            "no mismatch sample, no zero count, no finding, no reboot; full count after"
        ),
        execute=gpu_query_failed,
    ),
    _gpu_read_path(
        "proc_unreadable",
        injection=(
            "the same failing query while the host collector unit reads a copy of "
            "/proc/driver/nvidia without gpus (its listing fails: ENOENT; two failed "
            "reads) -- not InaccessiblePaths=, which leaves a root unit an empty "
            "directory, i.e. 0 GPUs bound"
        ),
        expected=(
            "collection error names the unreadable procfs ('unknown as well'); "
            "failure_mode=INVENTORY_UNAVAILABLE with no discovered_count; no mismatch "
            "sample, no zero count, no finding, no REBOOT_NODE (spec's known risk)"
        ),
        execute=gpu_proc_unreadable,
    ),
    VariantSpec(
        case_id=CASE_012,
        variant="expectation_wrong",
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="M",
        flags="S",
        injection=(
            "override GPU_FAULT_EXPECTED_GPU_COUNT in the host collector unit's "
            "environment to the product's recorded expected count plus one; the "
            "window closes as soon as the product minted a finding on the override "
            "or recorded it twice"
        ),
        expected=(
            "no gpu_inventory_mismatch finding (the control plane re-reads the mismatch "
            "against the instance type: listed == type count == driver-bound); one "
            "gpu_expected_count_contradicted incident, WARNING / COLLECT_EVIDENCE, "
            "FREEZE_EVIDENCE only, whose reason names the configured count, the "
            "instance type and its count, the listed and the bound count; one "
            "OPERATOR_REVIEW notification; no REBOOT_NODE / RESTART_NODE, no isolation; "
            "the real expectation reads back after the window"
        ),
        cleanup=(
            "close the env window (restores the real expectation); the product's "
            "evidence-only chain is waited to settle, never cancelled"
        ),
        max_minutes=30,
        approvals=(HOST_SOURCE_REPLAY,),
        execute=gpu_expectation_wrong,
    ),
    deferred_variant(
        CASE_012,
        "reboot_verified",
        proof_mode=VENDOR_LAB,
        reason=(
            "VENDOR_LAB_ONLY: a verified reboot of a truly missing GPU needs a real "
            "confirmed-missing fault and an approved provider reboot"
        ),
        lanes=("eks", "hp"),
        expected="STOP, provider reboot, new boot id, full validation, controlled resume",
    ),
    deferred_variant(
        CASE_012,
        "post_reboot_failure",
        proof_mode=VENDOR_LAB,
        reason=(
            "VENDOR_LAB_ONLY: a GPU still missing after a real reboot needs the same "
            "physical fault and provider reboot"
        ),
        lanes=("eks", "hp"),
        expected="a still-missing GPU escalates / stays isolated, no blind reboot loop",
    ),
]
