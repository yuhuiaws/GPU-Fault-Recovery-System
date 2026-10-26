"""What one armed GF-ENV-LIVE plan must make the product do, and how long that takes.

``expectation_for`` turns the case, the armed plan and the node's collector
cadence into an :class:`Expectation`: the kind of reaction the spec requires
(``docs/components/gpu-environment-live-acceptance.md``), the field and values
that must be graded, and the deadline terms derived from the product's own
windows:

* every plan read is one collector interval (the proxy serves one read per
  scrape), plus the most the cadence jitter can stretch those reads, plus the
  start-up spread and start-up jitter a collector restarted inside the window
  needs before its first scrape, plus one more interval and the ingestion path
  as delivery margin -- the same cadence terms the repoint's first-read budget
  is sized from (``probes/env_replay_proxy.first_read_seconds``);
* ``gpu_environment_policy.confirm_environment_decision`` confirms a CRITICAL
  temperature or Power Brake sample on the third consecutive accepted sample at
  most 60 s apart (an absolute shutdown breach on the first); before that it is
  a WARNING with RUN_DIAGNOSTICS and "confirmation k/3";
* the DCGM edge filter delivers an environmental (CRITICAL or brake) sample on
  every read, a warning or thermal-bit candidate after
  ``GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES`` consecutive reads, a cleared
  candidate at once, and any other reading only with the health summary;
* ``health_recovery_contract``: a DRAIN thermal/brake finding compiles the
  ENVIRONMENT_REVIEW chain, whose post-review hold must outlast the old
  ``COOLDOWN_SECONDS`` (300 s) cooldown; a RUN_DIAGNOSTICS one compiles the
  DIAGNOSTIC_ONLY chain, whose VALIDATE_GPU waits a thermal-only WARNING out for
  ``GPU_FAULT_TEMPERATURE_WARNING_GRACE_SECONDS`` (120 s by default).

Everything here is pure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gpu_fault.gpu_environment_policy import (
    CONFIRMATION_MAX_GAP_SECONDS,
    CONFIRMATION_SAMPLES,
    HW_POWER_BRAKE_MASK,
)
from gpu_fault.health_recovery_contract import COOLDOWN_SECONDS

from scripts.e2e.regional import env_live_verdicts as verdicts
from scripts.e2e.regional.env_live_watch import (
    PASSTHROUGH_READS,
    PROCESSING_MARGIN_SECONDS,
    CollectorCadence,
    plan_reads,
)

#: VALIDATE_GPU's wait on a thermal-only WARNING before it fails on the active
#: finding (``app/context.py``, GPU_FAULT_TEMPERATURE_WARNING_GRACE_SECONDS;
#: the ``GpuValidationAdapter`` default is the same two minutes).
TEMPERATURE_WARNING_GRACE_SECONDS = 120.0
#: The plan field the device variant replays (``env_live_plans``).
FIELD_VARIANTS = {
    "gpu_tlimit_c": "core_tlimit",
    "gpu_temperature_c": "core_absolute",
    "memory_temperature_c": "memory_absolute",
}
KINDS = frozenset(
    {
        "baseline",
        "review",
        "warning",
        "transient",
        "duration",
        "no_finding",
        "unknown",
        "interruption",
    }
)
#: GF-ENV-LIVE-009's collection pause: longer than the policy's confirmation
#: gap (``CONFIRMATION_MAX_GAP_SECONDS``) by one collector tick of margin, so
#: the first accepted sample after the resume is unmistakably past the gap.
PAUSE_MARGIN_SECONDS = 15.0
#: How long the pause deadman waits beyond the planned pause before it resumes
#: the unit itself (the runner normally resumes first).
PAUSE_DEADMAN_MARGIN_SECONDS = 120.0
#: GF-ENV-LIVE-009's recount hold: the collector is stopped a second time right
#: after the first post-gap read, for at most this many polling rounds, so the
#: product's regraded finding state is read by at least one poll before the
#: clearing read replaces it. The product keeps a confirmation-count change
#: only in the current ``gpu_finding_state`` row (the next accepted reading
#: overwrites it; ``gpu_finding_history`` records a severity or action change
#: only), so the regrade "2/3 -> 1/3" is a 15 s state at the default cadence
#: while one polling round is 46-56 s (``env_live_watch.POLL_READ_BUDGET_
#: SECONDS``). Live 2026-10-26 (plain EKS, R13g): the post-gap sample was
#: accepted 13:32:57Z, cleared 13:33:12Z, polled 13:32:20Z and 13:33:16Z --
#: INCONCLUSIVE "no finding graded it yet". The judge resumes the hold as soon
#: as it has read the regrade; the bound is for failed reads.
RECOUNT_HOLD_POLLS = 2


def recount_hold_seconds(cadence: CollectorCadence) -> float:
    """The most the 009 recount hold keeps the collector stopped."""

    return RECOUNT_HOLD_POLLS * cadence.polling_round_seconds


def number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def same_value(left: Any, right: Any) -> bool:
    a, b = number_or_none(left), number_or_none(right)
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


def brake_mask(value: Any) -> bool:
    """``mask & 0x80`` on a valid reason mask (``clock_reason_decision``)."""

    number = number_or_none(value)
    return (
        number is not None
        and number.is_integer()
        and 0 <= number < 2**32
        and bool(int(number) & HW_POWER_BRAKE_MASK)
    )


@dataclass(frozen=True)
class Expectation:
    """The observation rule of one armed (or pass-through) plan."""

    case_id: str
    plan_name: str
    kind: str
    node: str
    gpu: str
    field: str | None
    trigger_values: tuple[float, ...]
    planned_reads: int
    names: tuple[str, ...]
    terms: dict[str, float]
    hold_seconds: float
    required_samples: int = CONFIRMATION_SAMPLES
    max_confirmation: int = 0
    threshold_source: str | None = None
    power_delivery: bool = False
    require_diagnostic: bool = False
    require_clear: bool = False
    fallback_field: str | None = None
    boot_id: str | None = None
    notes: tuple[str, ...] = ()
    #: GF-ENV-LIVE-009: the plan read after which the collector is paused and
    #: for how long (0 = no pause step in this expectation).
    pause_after_reads: int = 0
    pause_seconds: float = 0.0
    #: GF-ENV-LIVE-009: the most the recount hold (the second pause, right
    #: after the first post-gap read) keeps the collector stopped (0 = none).
    recount_hold_seconds: float = 0.0
    #: GF-ENV-LIVE-010: the control node whose task and records must not change.
    control_node: str | None = None

    def evidence(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "plan": self.plan_name,
            "gpu": self.gpu,
            "field": self.field,
            "trigger_values": list(self.trigger_values),
            "planned_reads": self.planned_reads,
            "required_samples": self.required_samples,
            "max_confirmation": self.max_confirmation,
            "threshold_source": self.threshold_source,
            "hold_seconds": self.hold_seconds,
            "require_diagnostic": self.require_diagnostic,
            "require_clear": self.require_clear,
            "pause_after_reads": self.pause_after_reads,
            "pause_seconds": self.pause_seconds,
            "recount_hold_seconds": self.recount_hold_seconds,
            "control_node": self.control_node,
            "not_covered_by_this_attempt": list(self.notes),
        }


def replayed_values(plan: Mapping[str, Any]) -> dict[str, list[float | None]]:
    """Every value the plan serves per field, in read order (``None``: omitted)."""

    result: dict[str, list[float | None]] = {}
    for step in plan.get("steps") or []:
        for name, value in (step.get("values") or {}).items():
            served = None if value is None else float(value)
            result.setdefault(str(name), []).extend(
                [served] * int(step.get("reads") or 0)
            )
    return result


def _decision(preflight: Mapping[str, Any], gpu: str, variant: str) -> dict[str, Any]:
    plans = (preflight.get("device_plans") or {}).get(gpu) or {}
    return dict((plans.get("decisions") or {}).get(variant) or {})


def _crosses(value: float, decision: Mapping[str, Any]) -> bool:
    """Past the variant's critical boundary on its own scale."""

    critical = number_or_none(decision.get("critical_value"))
    if critical is None:
        return False
    return value <= critical if decision.get("scale") == "tlimit" else value >= critical


def _single_field(values: Mapping[str, list[float | None]], case_id: str) -> str:
    names = [
        name for name, items in values.items() if any(v is not None for v in items)
    ]
    if len(names) != 1:
        raise ValueError(f"{case_id}: the plan must replay exactly one field: {names}")
    return names[0]


def _served(values: Mapping[str, list[float | None]], name: str) -> list[float]:
    return [value for value in values.get(name) or [] if value is not None]


def _interruption_expectation(
    base: dict[str, Any],
    values: Mapping[str, list[float | None]],
    *,
    plan: Mapping[str, Any],
    case_id: str,
    preflight: Mapping[str, Any],
    gpu: str,
    cadence: CollectorCadence,
    replay: Mapping[str, float],
    hold: float,
) -> Expectation:
    """GF-ENV-LIVE-009: the pause, the recount, the clear and the restart."""

    interval = cadence.interval_seconds
    name = _single_field(values, case_id)
    pause_after = plan_pause_read(plan)
    pause_seconds = CONFIRMATION_MAX_GAP_SECONDS + PAUSE_MARGIN_SECONDS
    recount_hold = recount_hold_seconds(cadence)
    return Expectation(
        **base,
        kind="interruption",
        field=name,
        trigger_values=tuple(sorted(set(_served(values, name)))),
        names=(name, "gpu_temperature_c"),
        terms={
            **replay,
            # The pause stops the proxy reads for its whole length; the recount
            # hold stops them again after the first post-gap read until the
            # judge has read the regraded state (at most ``recount_hold``);
            # the restart's first scrape waits for the startup spread and
            # jitter; its delivery and one hold after the clear complete the
            # case.
            "collection_pause": pause_seconds,
            "recount_hold": recount_hold,
            "collector_restart_first_read": (
                cadence.startup_allowance_seconds + interval
            ),
            "restart_delivery_margin": PROCESSING_MARGIN_SECONDS,
            "hold": hold,
        },
        hold_seconds=hold,
        threshold_source=_decision(preflight, gpu, FIELD_VARIANTS.get(name, "")).get(
            "threshold_source"
        ),
        require_clear=True,
        pause_after_reads=pause_after,
        pause_seconds=pause_seconds,
        recount_hold_seconds=recount_hold,
        boot_id=str((preflight.get("node") or {}).get("boot_id") or "") or None,
        notes=(
            "the DCGM -> nvidia-smi source switch runs only through the normal "
            "installation (GPU_FAULT_METRICS_MODE); this attempt does not "
            "switch the source and records that gap",
            "cross-boot re-confirmation needs a separately approved node "
            "maintenance reboot; this attempt does not reboot the node",
        ),
    )


def _review_expectation(
    base: dict[str, Any],
    values: Mapping[str, list[float | None]],
    *,
    number: int,
    plan_name: str,
    case_id: str,
    preflight: Mapping[str, Any],
    gpu: str,
    interval: float,
    replay: Mapping[str, float],
) -> Expectation:
    """004 / 010 / 011 and the sustained or mixed brake of 007: the review chain."""

    name = _single_field(values, case_id)
    triggers = _served(values, name)
    if number == 7:
        triggers = [value for value in triggers if brake_mask(value)]
        source: Any = "NVIDIA_HW_POWER_BRAKE"
    elif number == 11:
        source = "NVIDIA_DEVICE_LIMIT"
    else:
        source = _decision(preflight, gpu, plan_name.split("/", 1)[0]).get(
            "threshold_source"
        )
    notes: tuple[str, ...] = ()
    if number == 7:
        notes = (
            "the transient brake, the 0x04/0x08/0x10/0x100 bits and the "
            "other mixed masks run as their own attempts",
        )
    elif number == 10:
        notes = (
            "the target node runs no owned distributed job (the preflight admits "
            "only a clean node), so the STOP scope of an affected job is not "
            "measured; the control task runs on the control node",
        )
    return Expectation(
        **base,
        kind="review",
        field=name,
        trigger_values=tuple(sorted(set(triggers))),
        names=(name, "gpu_temperature_c"),
        terms={
            **replay,
            "cooldown_hold": COOLDOWN_SECONDS + interval,
            "workflow_dispatch": PROCESSING_MARGIN_SECONDS,
        },
        hold_seconds=COOLDOWN_SECONDS + interval,
        required_samples=1 if number == 11 else CONFIRMATION_SAMPLES,
        threshold_source=source,
        power_delivery=number == 7,
        control_node=(
            str((preflight.get("control_node") or {}).get("name") or "") or None
        )
        if number == 10
        else None,
        notes=notes,
    )


def plan_pause_read(plan: Mapping[str, Any]) -> int:
    """The read after which 009 pauses: the warmup block plus the first severe
    block (``env_live_plans.pause_read_index``, kept here without the import)."""

    steps = [step for step in plan.get("steps") or [] if isinstance(step, Mapping)]
    if len(steps) < 3:
        raise ValueError("an interruption plan needs warmup, severe and clear steps")
    return int(steps[0].get("reads") or 0) + int(steps[1].get("reads") or 0)


def expectation_for(
    *,
    case_id: str,
    plan_name: str,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    cadence: CollectorCadence,
    node: str,
) -> Expectation:
    """The rule for ``plan_name`` of ``case_id``; ``ValueError`` if none fits."""

    number = int(case_id.rsplit("-", 1)[-1])
    gpu = str((preflight.get("target_gpu") or {}).get("gpu_uuid") or "")
    reads = plan_reads(plan)
    interval = cadence.interval_seconds
    replay = {
        "replay_reads_x_interval": cadence.replay_seconds(reads),
        # The cadence jitter over the plan's reads, and the start-up spread and
        # jitter a collector restarted inside the window needs before its
        # first scrape (``CollectorCadence``): sized from the node's own
        # collector.env like the repoint's first-read budget.
        "cadence_jitter": cadence.jitter_seconds(reads),
        "collector_startup_spread_and_jitter": cadence.startup_allowance_seconds,
        "delivery_margin": cadence.delivery_margin_seconds,
    }
    # A read-only signal is held for one confirmation window after it was
    # graded: a mistaken escalation would show inside it (warning_plan).
    hold = CONFIRMATION_SAMPLES * interval + PROCESSING_MARGIN_SECONDS
    diagnostic_wait = max(
        hold, TEMPERATURE_WARNING_GRACE_SECONDS + PROCESSING_MARGIN_SECONDS
    )
    base: dict[str, Any] = {
        "case_id": case_id,
        "plan_name": plan_name,
        "node": node,
        "gpu": gpu,
        "planned_reads": reads,
    }
    values = replayed_values(plan)
    if number == 1:
        names = (
            *verdicts.BASELINE_REQUIRED_READINGS,
            *(name for pair in verdicts.BASELINE_REQUIRED_SCALE_PAIRS for name in pair),
            "gpu_tlimit_c",
            "memory_temperature_c",
        )
        return Expectation(
            **base,
            kind="baseline",
            field=None,
            trigger_values=(),
            names=tuple(dict.fromkeys(names)),
            terms={
                # The repoint restarts the collector: its first scrape waits for
                # the startup spread and jitter, then one read per interval.
                "collector_restart_first_read": (
                    cadence.startup_allowance_seconds + interval
                ),
                "passthrough_reads_x_interval": PASSTHROUGH_READS * interval,
                "cadence_jitter": cadence.jitter_seconds(PASSTHROUGH_READS),
                "delivery_margin": PROCESSING_MARGIN_SECONDS,
            },
            hold_seconds=0.0,
            boot_id=str((preflight.get("node") or {}).get("boot_id") or "") or None,
        )
    if number == 9:
        return _interruption_expectation(
            base,
            values,
            plan=plan,
            case_id=case_id,
            preflight=preflight,
            gpu=gpu,
            cadence=cadence,
            replay=replay,
            hold=hold,
        )
    if number in {4, 10, 11} or (
        number == 7 and (plan_name == "sustained" or plan_name.startswith("mixed_"))
    ):
        return _review_expectation(
            base,
            values,
            number=number,
            plan_name=plan_name,
            case_id=case_id,
            preflight=preflight,
            gpu=gpu,
            interval=interval,
            replay=replay,
        )
    if number == 3:
        name = _single_field(values, case_id)
        variant = plan_name.split("/", 1)[0]
        return Expectation(
            **base,
            kind="warning",
            field=name,
            trigger_values=tuple(sorted(set(_served(values, name)))),
            names=(name, "gpu_temperature_c"),
            terms={**replay, "validate_gpu_warning_grace_or_hold": diagnostic_wait},
            hold_seconds=hold,
            threshold_source=_decision(preflight, gpu, variant).get("threshold_source"),
            require_diagnostic=True,
            fallback_field="gpu_temperature_c" if name == "gpu_tlimit_c" else None,
        )
    if number == 5 or (number == 7 and plan_name == "transient"):
        name = _single_field(values, case_id)
        if number == 7:
            triggers = [value for value in _served(values, name) if brake_mask(value)]
        else:
            decision = _decision(preflight, gpu, FIELD_VARIANTS.get(name, ""))
            triggers = [
                value for value in _served(values, name) if _crosses(value, decision)
            ]
        return Expectation(
            **base,
            kind="transient",
            field=name,
            trigger_values=tuple(sorted(set(triggers))),
            names=(name, "gpu_temperature_c"),
            terms={**replay, "hold": hold},
            hold_seconds=hold,
            max_confirmation=len(triggers),
            require_clear=True,
            notes=(
                (
                    "the replayed accepted-batch retry and the later sustained run "
                    "that must still trigger the 004 chain are separate attempts",
                )
                if number == 5
                else ()
            ),
        )
    if number == 6:
        bits = [value for value in _served(values, "clock_throttle_reasons") if value]
        return Expectation(
            **base,
            kind="duration",
            field="clock_throttle_reasons",
            trigger_values=tuple(sorted(set(bits))),
            names=(
                "clock_throttle_reasons",
                "thermal_violation_total_us",
                "gpu_temperature_c",
            ),
            terms={**replay, "validate_gpu_warning_grace_or_hold": diagnostic_wait},
            hold_seconds=hold,
            require_diagnostic=True,
            notes=(
                "the repeat under a warning-band temperature (THERMAL_STRESS) is a "
                "separate attempt",
            ),
        )
    edge = (
        f"no delivery edge exists for it: only a health summary (every "
        f"{cadence.health_summary_seconds:.0f}s) inside the {reads}-read plan can "
        "carry it to the control plane"
    )
    if number == 7 and plan_name.startswith("non_brake_"):
        name = _single_field(values, case_id)
        return Expectation(
            **base,
            kind="no_finding",
            field=name,
            trigger_values=tuple(sorted(set(_served(values, name)))),
            names=(name, "gpu_temperature_c"),
            terms={**replay, "hold": hold},
            hold_seconds=hold,
            notes=(f"a non-brake bit is no finding, so {edge}",),
        )
    if number == 8:
        name = next(iter(values), "clock_throttle_reasons")
        return Expectation(
            **base,
            kind="unknown",
            field=name,
            trigger_values=tuple(sorted(set(_served(values, name)))),
            names=(name, "gpu_temperature_c"),
            terms={**replay, "hold": hold},
            hold_seconds=hold,
            notes=(f"an unusable reason mask is no candidate, so {edge}",),
        )
    raise ValueError(f"no observation rule for {case_id} plan {plan_name}")


__all__ = [
    "Expectation",
    "FIELD_VARIANTS",
    "KINDS",
    "PAUSE_DEADMAN_MARGIN_SECONDS",
    "PAUSE_MARGIN_SECONDS",
    "TEMPERATURE_WARNING_GRACE_SECONDS",
    "brake_mask",
    "expectation_for",
    "number_or_none",
    "plan_pause_read",
    "replayed_values",
    "same_value",
]
