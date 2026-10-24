"""What one armed GF-ENV-LIVE plan must make the product do, and how long that takes.

``expectation_for`` turns the case, the armed plan and the node's collector
cadence into an :class:`Expectation`: the kind of reaction the spec requires
(``docs/components/gpu-environment-live-acceptance.md``), the field and values
that must be graded, and the deadline terms derived from the product's own
windows:

* every plan read is one collector interval (the proxy serves one read per
  scrape), plus one more interval and the ingestion path as delivery margin;
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

from gpu_fault.gpu_environment_policy import CONFIRMATION_SAMPLES, HW_POWER_BRAKE_MASK
from gpu_fault.health_recovery_contract import COOLDOWN_SECONDS

from scripts.e2e.regional import env_live_verdicts as verdicts
from scripts.e2e.regional.env_live_watch import (
    PASSTHROUGH_READS,
    PROCESSING_MARGIN_SECONDS,
    STARTUP_SPREAD_SECONDS,
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
    {"baseline", "review", "warning", "transient", "duration", "no_finding", "unknown"}
)


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
                # the startup spread, then one read per interval.
                "collector_restart_first_read": STARTUP_SPREAD_SECONDS + interval,
                "passthrough_reads_x_interval": PASSTHROUGH_READS * interval,
                "delivery_margin": PROCESSING_MARGIN_SECONDS,
            },
            hold_seconds=0.0,
            boot_id=str((preflight.get("node") or {}).get("boot_id") or "") or None,
        )
    if number in {4, 11} or (
        number == 7 and (plan_name == "sustained" or plan_name.startswith("mixed_"))
    ):
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
            notes=(
                (
                    "the transient brake, the 0x04/0x08/0x10/0x100 bits and the "
                    "other mixed masks run as their own attempts",
                )
                if number == 7
                else ()
            ),
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
    "TEMPERATURE_WARNING_GRACE_SECONDS",
    "brake_mask",
    "expectation_for",
    "number_or_none",
    "replayed_values",
    "same_value",
]
