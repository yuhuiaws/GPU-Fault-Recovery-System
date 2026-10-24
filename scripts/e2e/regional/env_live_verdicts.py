"""Verdict logic for GF-ENV-LIVE, judged on the product's own records.

Every function here is pure: it takes the documents the runner read from the
control plane (the store snapshot's incident / workflow / notifications /
decisions / evidence, the node snapshot, the collector statuses, the replay
audit) and returns the concrete reasons a case failed, or ``[]``. The replay
tool never declares a LIVE PASS; these functions turn the product's workflow
evidence into the verdict, and a missing receipt is a failure, not a pass.
What could not be observed at all -- a replay the collector never consumed, a
reading no accepted record shows -- is INCONCLUSIVE (``case_verdict``), never a
PASS; ``env_live_judges`` decides which is which on the polled records.

The workflow shapes they check come from ``orchestration.families.health``: a
severe environment finding compiles the ENVIRONMENT_REVIEW chain
(FREEZE_EVIDENCE, MARK_UNSCHEDULABLE, STOP_WORKLOADS when ACTIVE, QUARANTINE,
COLLECT_DIAGNOSTIC_BUNDLE, TRIGGER_HEALTH_SNAPSHOT, ESCALATE_SUPPORT) and ends
the incident ESCALATED with no reset/reboot/RMA/restart; a warning or a
duration/brake-below-confirmation stays read-only diagnostic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

CONFIRMATION_SAMPLES = 3
CONFIRMATION_MAX_GAP_SECONDS = 60

#: The ENVIRONMENT_REVIEW chain's operations, in the order health.py compiles
#: them. STOP_WORKLOADS is present only for an ACTIVE attempt.
ENVIRONMENT_REVIEW_OPERATIONS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "STOP_WORKLOADS",
    "QUARANTINE",
    "COLLECT_DIAGNOSTIC_BUNDLE",
    "TRIGGER_HEALTH_SNAPSHOT",
    "ESCALATE_SUPPORT",
)
#: Operations that must never appear for an environment finding: this is a
#: cooling/power-delivery review, not a physical remediation.
FORBIDDEN_OPERATIONS = frozenset(
    {
        "RESET_GPU",
        "RESET_ALL_GPUS_NVSWITCHES",
        "REBOOT_NODE",
        "RESTART_NODE",
        "REPLACE_NODE",
        "RESTART_WORKLOAD",
        "RUN_FIELD_DIAGNOSTIC",
        "REMEDIATE_EFA_DRIVER",
        "RESTART_EFA_DEVICE_PLUGIN",
        "RESTART_GPU_DEVICE_PLUGIN",
    }
)
#: Operations a read-only diagnostic workflow may carry (003/006/007-transient).
DIAGNOSTIC_OPERATIONS = frozenset(
    {
        "FREEZE_EVIDENCE",
        "RUN_DCGM_DIAGNOSTIC",
        "RUN_DIAGNOSTICS",
        "TRIGGER_HEALTH_SNAPSHOT",
        "VALIDATE_GPU",
        "VALIDATE_HOST",
        "VALIDATE_FABRIC",
        "COLLECT_DIAGNOSTIC_BUNDLE",
        "ESCALATE_SUPPORT",
    }
)
#: Operations that isolate or stop; a read-only case must carry none of them.
ISOLATING_OPERATIONS = frozenset(
    {"MARK_UNSCHEDULABLE", "QUARANTINE", "STOP_WORKLOADS", *FORBIDDEN_OPERATIONS}
)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def workflow_operations(workflow: dict[str, Any] | None) -> list[str]:
    if not isinstance(workflow, dict):
        return []
    return [str(step.get("operation")) for step in workflow.get("official_steps") or []]


def forbidden_operation_errors(
    workflow: dict[str, Any] | None, *, label: str
) -> list[str]:
    present = sorted(set(workflow_operations(workflow)) & FORBIDDEN_OPERATIONS)
    return [f"{label}: forbidden operation {op} appeared" for op in present]


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-001: source and health baseline
# --------------------------------------------------------------------------- #
#: The fields the baseline requires a reading or an explicit "unsupported"
#: record for. Temperature thresholds are reported on whichever scale the
#: device uses, so the pair is satisfied by either member.
BASELINE_REQUIRED_READINGS = (
    "gpu_temperature_c",
    "power_usage_w",
    "power_limit_w",
    "clock_throttle_reasons",
)
BASELINE_REQUIRED_SCALE_PAIRS = (
    ("gpu_slowdown_temperature_c", "gpu_slowdown_tlimit_c"),
    ("gpu_shutdown_temperature_c", "gpu_shutdown_tlimit_c"),
)


def baseline_errors(
    *,
    per_gpu_readings: dict[str, set[str]],
    unsupported: dict[str, set[str]],
    valid_sample_count: int,
    source_boot_id: str | None,
    collector_mode: str | None,
    new_environment_actions: Iterable[str],
    task_progressed: bool | None,
) -> list[str]:
    """001: every required field has a reading or an explicit unsupported record,
    at least three fresh samples were accepted, no environment action fired and
    the original task kept moving while the proxy was armed-but-passthrough.
    ``task_progressed=None`` means the node runs no test task; the caller
    records that gap in its evidence instead of asserting progress."""

    errors: list[str] = []
    for gpu, readings in sorted(per_gpu_readings.items()):
        marked = unsupported.get(gpu, set())
        for field in BASELINE_REQUIRED_READINGS:
            if field not in readings and field not in marked:
                errors.append(
                    f"{gpu}: {field} has neither a reading nor an unsupported record"
                )
        for absolute, tlimit in BASELINE_REQUIRED_SCALE_PAIRS:
            if (
                absolute not in readings
                and tlimit not in readings
                and absolute not in marked
                and tlimit not in marked
            ):
                errors.append(
                    f"{gpu}: neither {absolute} nor {tlimit} was read or marked "
                    "unsupported"
                )
    if valid_sample_count < CONFIRMATION_SAMPLES:
        errors.append(
            f"only {valid_sample_count} valid control-plane samples, need "
            f"{CONFIRMATION_SAMPLES}"
        )
    if not source_boot_id:
        errors.append("control plane recorded no source_boot_id for the samples")
    if not collector_mode:
        errors.append("collector mode (DCGM / nvidia-smi) was not recorded")
    actions = sorted(set(new_environment_actions))
    if actions:
        errors.append(f"baseline created environment actions: {actions}")
    if task_progressed is False:
        errors.append("the original training step did not advance during baseline")
    return errors


# --------------------------------------------------------------------------- #
# Sample validity (shared by 004/007/009/011)
# --------------------------------------------------------------------------- #
def valid_severe_samples(
    samples: list[dict[str, Any]],
    *,
    max_gap_seconds: int = CONFIRMATION_MAX_GAP_SECONDS,
) -> tuple[int, list[str]]:
    """The run of severe samples the control plane would count, and why a break.

    A sample counts when its source, boot id and BDF match the previous severe
    one and it is accepted within ``max_gap_seconds`` of it; a wider gap, a
    source/boot/BDF change or a collection error restarts the count, exactly as
    the policy's ``confirm_environment_decision`` does.
    """

    run = 0
    notes: list[str] = []
    previous: dict[str, Any] | None = None
    for index, sample in enumerate(samples):
        if sample.get("collection_error"):
            notes.append(f"sample {index}: collection error resets the count")
            run = 0
            previous = None
            continue
        if not sample.get("severe"):
            run = 0
            previous = None
            continue
        if previous is not None:
            gap = None
            current_at = _parse_time(sample.get("observed_at"))
            previous_at = _parse_time(previous.get("observed_at"))
            if current_at is not None and previous_at is not None:
                gap = (current_at - previous_at).total_seconds()
            identity_changed = any(
                sample.get(key) != previous.get(key)
                for key in ("source", "source_boot_id", "pci_bdf")
            )
            if identity_changed:
                notes.append(f"sample {index}: source/boot/BDF changed; count reset")
                run = 1
                previous = sample
                continue
            if gap is not None and gap > max_gap_seconds:
                notes.append(
                    f"sample {index}: {gap:.0f}s gap exceeds {max_gap_seconds}s; "
                    "count reset"
                )
                run = 1
                previous = sample
                continue
        run += 1
        previous = sample
    return run, notes


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-003 / 006: warning and duration stay read-only
# --------------------------------------------------------------------------- #
def read_only_diagnostic_errors(
    *,
    workflow: dict[str, Any] | None,
    incident_state: str | None,
    node_isolated: bool,
    task_progressed: bool | None,
    label: str,
) -> list[str]:
    """A warning / duration / composite signal is diagnosed, never escalated.

    No isolating operation, the node stays schedulable, the task keeps moving,
    and any workflow carries only diagnostic operations. ``task_progressed``
    is ``None`` when no test task exists to measure (recorded by the caller).
    """

    errors: list[str] = []
    operations = set(workflow_operations(workflow))
    isolating = sorted(operations & ISOLATING_OPERATIONS)
    if isolating:
        errors.append(f"{label}: read-only signal drove isolating steps {isolating}")
    stray = sorted(operations - DIAGNOSTIC_OPERATIONS)
    if stray:
        errors.append(f"{label}: workflow carried non-diagnostic steps {stray}")
    if node_isolated:
        errors.append(f"{label}: node was isolated for a read-only signal")
    if task_progressed is False:
        errors.append(f"{label}: the training task stopped on a read-only signal")
    if incident_state in {"QUARANTINED", "ESCALATED"}:
        errors.append(f"{label}: incident escalated to {incident_state}")
    return errors


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-004 / 011: severe temperature to human review
# --------------------------------------------------------------------------- #
def severe_review_errors(
    *,
    confirmed_sample_run: int,
    required_samples: int,
    pre_confirmation_stopped: bool,
    workflow: dict[str, Any] | None,
    incident_state: str | None,
    node_cordoned: bool,
    quarantine_taint: bool,
    stopped_owned_task: bool,
    diagnostic_bundle: bool,
    health_snapshot: bool,
    notification_sent: bool,
    reset_or_reboot: bool,
    task_restarted: bool,
    held_isolated_after_cooldown: bool,
    label: str,
) -> list[str]:
    """004: the N-th severe sample confirms and the ENVIRONMENT_REVIEW chain runs.

    Real cordon, the owned task suspended/exited, the quarantine taint, the
    diagnostic-bundle and snapshot receipts and a notification all present, the
    incident ESCALATED, and no reset/reboot/RMA/restart; the earlier severe
    samples must not have stopped anything. 011 reuses this with
    ``required_samples=1`` (shutdown protection does not wait for three).
    """

    errors: list[str] = []
    if pre_confirmation_stopped:
        errors.append(f"{label}: a pre-confirmation severe sample already stopped work")
    if confirmed_sample_run < required_samples:
        errors.append(
            f"{label}: confirmation reached {confirmed_sample_run} of "
            f"{required_samples} valid severe samples"
        )
    errors.extend(forbidden_operation_errors(workflow, label=label))
    if reset_or_reboot:
        errors.append(f"{label}: a reset/reboot/RMA action was taken")
    if task_restarted:
        errors.append(f"{label}: the suspended training job was restarted")
    operations = set(workflow_operations(workflow))
    for required in ("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"):
        if required not in operations:
            errors.append(f"{label}: workflow is missing {required}")
    if not node_cordoned:
        errors.append(f"{label}: node was not cordoned")
    if not quarantine_taint:
        errors.append(f"{label}: quarantine taint is absent")
    if not stopped_owned_task:
        errors.append(f"{label}: the owned ACTIVE task was not stopped")
    if not diagnostic_bundle:
        errors.append(f"{label}: no diagnostic-bundle receipt")
    if not health_snapshot:
        errors.append(f"{label}: no health-snapshot receipt")
    if not notification_sent:
        errors.append(f"{label}: no notification receipt")
    if incident_state != "ESCALATED":
        errors.append(f"{label}: incident state is {incident_state}, not ESCALATED")
    if not held_isolated_after_cooldown:
        errors.append(f"{label}: node did not stay isolated past the old 300s cooldown")
    return errors


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-005: transient, clear and repeat
# --------------------------------------------------------------------------- #
def transient_errors(
    *,
    cleared_through_edge_filter: bool,
    confirmation_after_clear: int,
    duplicate_batch_advanced: bool,
    stopped: bool,
    retriggered_run: int,
    required_samples: int,
    label: str,
) -> list[str]:
    """005: a cleared severe sample resets the count, a replayed accepted batch
    does not advance it, nothing stops -- and a later sustained run still
    triggers the 004 chain."""

    errors: list[str] = []
    if not cleared_through_edge_filter:
        errors.append(f"{label}: the clearing sample did not pass the edge filter")
    if confirmation_after_clear >= required_samples:
        errors.append(
            f"{label}: the count reached {confirmation_after_clear} after a clear"
        )
    if duplicate_batch_advanced:
        errors.append(f"{label}: a replayed accepted batch advanced the count")
    if stopped:
        errors.append(f"{label}: a transient/repeat sample stopped work")
    if retriggered_run < required_samples:
        errors.append(
            f"{label}: a later sustained run did not re-trigger "
            f"({retriggered_run}/{required_samples})"
        )
    return errors


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-007: Power Brake vs ordinary throttle bits
# --------------------------------------------------------------------------- #
def power_brake_errors(
    *,
    transient_isolated: bool,
    sustained_review: list[str],
    non_brake_actions: dict[str, str | None],
    mixed_recognized_as_brake: dict[str, bool],
    thermal_warning_downgraded_brake: bool,
    receipt_is_power_check: bool,
    rebooted_or_auto_recovered: bool,
    label: str,
) -> list[str]:
    """007: one brake sample then clear does not isolate; three sustained brake
    samples run the 004 human-review chain as a power-delivery check; ordinary
    bits are not read as a brake, mixed masks with ``0x80`` still are, and a
    thermal warning cannot downgrade a confirmed brake."""

    errors: list[str] = list(sustained_review)
    if transient_isolated:
        errors.append(f"{label}: a single brake sample isolated the node")
    for bits, action in sorted(non_brake_actions.items()):
        if action in {"DRAIN", "QUARANTINE", "STOP_WORKLOAD"}:
            errors.append(f"{label}: ordinary bit {bits} drove {action}")
    for mask, recognized in sorted(mixed_recognized_as_brake.items()):
        if not recognized:
            errors.append(f"{label}: mixed mask {mask} was not recognized as a brake")
    if thermal_warning_downgraded_brake:
        errors.append(f"{label}: a thermal warning downgraded a confirmed brake")
    if not receipt_is_power_check:
        errors.append(f"{label}: the escalation is not a power-delivery check")
    if rebooted_or_auto_recovered:
        errors.append(f"{label}: a reboot or automatic recovery ran")
    return errors


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-008: unavailable fields and aliases
# --------------------------------------------------------------------------- #
def unknown_field_errors(
    *,
    reported_unknown: bool,
    fabricated_brake: bool,
    stopped_or_isolated: bool,
    missing_field_gpu_judged_alone: bool,
    label: str,
) -> list[str]:
    """008: a blank / omitted / non-integer value reads as UNKNOWN or a
    collection error, never a fabricated brake; nothing stops or isolates; and a
    multi-GPU batch judges the field-missing GPU on its own, not by another
    GPU's same-named field."""

    errors: list[str] = []
    if not reported_unknown:
        errors.append(f"{label}: the unusable value was not reported UNKNOWN")
    if fabricated_brake:
        errors.append(f"{label}: an unusable mask was read as a Power Brake")
    if stopped_or_isolated:
        errors.append(f"{label}: an unusable value drove a STOP or isolation")
    if not missing_field_gpu_judged_alone:
        errors.append(
            f"{label}: the field-missing GPU was judged by the batch's other GPUs"
        )
    return errors


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-009: interruption, restart and source switch
# --------------------------------------------------------------------------- #
def interruption_errors(
    *,
    recount_from_one: bool,
    single_producer: bool,
    gap_not_treated_healthy: bool,
    source_change_inherited_confirmation: bool,
    cross_boot_verified: bool,
    label: str,
) -> list[str]:
    """009: a >60s accepted-sample gap recounts from 1, there is only ever one
    producer, a gap is not read as healthy, and a source change does not inherit
    a confirmation. Cross-boot is a gap the runner records rather than drives."""

    errors: list[str] = []
    if not recount_from_one:
        errors.append(f"{label}: confirmation did not restart after the >60s gap")
    if not single_producer:
        errors.append(f"{label}: more than one metrics producer was present")
    if not gap_not_treated_healthy:
        errors.append(f"{label}: the collection gap was treated as healthy")
    if source_change_inherited_confirmation:
        errors.append(f"{label}: a source switch inherited the old confirmation")
    if not cross_boot_verified:
        errors.append(
            f"{label}: cross-boot path was not verified (record the gap; this "
            "case does not reboot the node)"
        )
    return errors


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-010: scope and concurrency
# --------------------------------------------------------------------------- #
def scope_errors(
    *,
    control_metrics_unchanged: bool,
    control_task_unaffected: bool,
    finding_gpu_scope_correct: bool,
    stop_covers_owned_job: bool,
    stop_targeted_single_rank: bool,
    workflows_merged: bool,
    notifications_deduplicated: bool,
    duplicate_stop: bool,
    label: str,
) -> list[str]:
    """010: only the changed GPU is found; the control GPU/node and an
    independent control task are untouched; the STOP covers the affected
    distributed job's owned scope (never a single rank); concurrent reports of
    one fault merge the workflow, dedupe the notification and run one STOP."""

    errors: list[str] = []
    if not control_metrics_unchanged:
        errors.append(f"{label}: the control GPU's metrics changed")
    if not control_task_unaffected:
        errors.append(f"{label}: an independent control task was affected")
    if not finding_gpu_scope_correct:
        errors.append(f"{label}: the finding/diagnostic GPU scope is wrong")
    if not stop_covers_owned_job:
        errors.append(f"{label}: the STOP did not cover the owned distributed job")
    if stop_targeted_single_rank:
        errors.append(f"{label}: the STOP wrongly targeted a single rank")
    if not workflows_merged:
        errors.append(f"{label}: concurrent reports did not merge into one workflow")
    if not notifications_deduplicated:
        errors.append(f"{label}: the notification was not deduplicated")
    if duplicate_stop:
        errors.append(f"{label}: the fault drove a duplicate STOP")
    return errors


# --------------------------------------------------------------------------- #
# Cleanup (every case)
# --------------------------------------------------------------------------- #
def cleanup_errors(
    *,
    collector_endpoint_restored: bool,
    single_producer_after: bool,
    fresh_batches_after: int,
    proxy_closed_after_endpoint: bool,
    probe_residuals_absent: bool,
    label: str,
    minimum_fresh_batches: int = 3,
) -> list[str]:
    """Cleanup check for every case: the collector endpoint, mode and unique
    producer are restored, at least three fresh real batches are observed, the
    proxy closed only after the endpoint was restored, and no probe resource
    remains. A cleanup failure fails the whole case."""

    errors: list[str] = []
    if not collector_endpoint_restored:
        errors.append(f"{label}: collector endpoint was not restored")
    if not single_producer_after:
        errors.append(f"{label}: more than one producer after cleanup")
    if fresh_batches_after < minimum_fresh_batches:
        errors.append(
            f"{label}: only {fresh_batches_after} fresh real batches after cleanup"
        )
    if not proxy_closed_after_endpoint:
        errors.append(f"{label}: proxy closed before the endpoint was restored")
    if not probe_residuals_absent:
        errors.append(f"{label}: a probe resource remained after cleanup")
    return errors


def case_verdict(
    error_groups: dict[str, list[str]], inconclusive: Iterable[str] = ()
) -> str:
    """FAIL on any stage error (a failed cleanup included); INCONCLUSIVE when
    nothing failed but the replay or its ingestion could not be shown; PASS
    only when every stage held on positive evidence."""

    if any(error_groups.values()):
        return "FAIL"
    return "INCONCLUSIVE" if list(inconclusive) else "PASS"
