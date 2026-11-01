"""What the DCGM edge filter delivers for a state, a brake time or a rate.

``DcgmMetricsCollector._candidate_keys`` asks these helpers which samples are
candidates; the confirmation streaks, carry-over and recovery stay there.

* :func:`state_breached`: a state that is bad as read -- a pending repair, a
  row-remap failure, a last XID, a fabric state held past its window, and from
  batch 233 a confirmed PCIe downtrain, from batch 235 unrepairable memory and
  a fabric health fault held long enough to grade CRITICAL
  (``gpu_fabric_health.held_breached``). Delivered once confirmed, not at the
  next summary. The slow states of batch 233 -- the ECC mode (it changes only
  across a reboot, whose first scrape is always delivered) and the InfoROM
  answer -- are state-change edges instead (``dcgm.STATE_CHANGE_REASONS``).
* :func:`power_brake_candidate`: any HW power brake time in DCGM field 1424 is
  the brake's own environmental semantics (``NVIDIA_HW_POWER_BRAKE``), so the
  filter delivers every tick while it accrues and the control plane can see two
  consecutive samples (``gpu_field_rules.power_brake_decision``).
* :class:`CorrectableRateGate`: field 1501 is graded by rate, never on any change
  (a healthy link may count a few correctable errors): a rate above R
  (``gpu_pcie_link_policy.correctable_rate_limit``: the lower of the current
  link's expected error rate and DCGM's PCIe diagnostic limit) is the
  ``PCIE_CORRECTABLE_RATE`` semantics, delivered every tick while it holds.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import Any

from gpu_fault.gpu_fabric_health import FABRIC_HEALTH_HELD_SECONDS, held_breached
from gpu_fault.gpu_fabric_manager import (
    FABRIC_MANAGER_HELD_SECONDS,
    FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS,
)
from gpu_fault.gpu_field_rules import DUTY_MAX_PLAUSIBLE, HW_POWER_BRAKE_SLOWDOWN
from gpu_fault.gpu_memory_repair_policy import MEMORY_UNREPAIRABLE
from gpu_fault.gpu_pcie_link_policy import PCIE_CORRECTABLE_ERRORS, PCIE_LINK_DOWNTRAIN

#: States graded bad whenever they are above zero.
POSITIVE_STATES = frozenset(
    {
        "retired_pages_pending",
        "row_remap_failure",
        "row_remap_pending",
        "xid_last_error",
        PCIE_LINK_DOWNTRAIN,
        MEMORY_UNREPAIRABLE,
    }
)
#: Counters graded by rate in the edge filter, never on any change.
RATE_GRADED_COUNTERS = frozenset({PCIE_CORRECTABLE_ERRORS})
POWER_BRAKE_SEMANTICS = "NVIDIA_HW_POWER_BRAKE"


def state_breached(name: str, value: float, peers: dict[str, float]) -> bool:
    """Whether the sample is a bad state as read (module docstring)."""

    if name in POSITIVE_STATES:
        return value > 0
    if name == FABRIC_MANAGER_HELD_SECONDS:
        return value >= FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS
    if name == FABRIC_HEALTH_HELD_SECONDS:
        return held_breached(value, peers)
    return False


def power_brake_candidate(
    sample: Any, duty_cycle: float | None
) -> dict[str, Any] | None:
    """The brake semantics for field 1424 brake time in a plausible duty cycle."""

    if (
        sample.canonical_name != HW_POWER_BRAKE_SLOWDOWN
        or duty_cycle is None
        or not 0 < duty_cycle <= DUTY_MAX_PLAUSIBLE
    ):
        return None
    return {"threshold_source": POWER_BRAKE_SEMANTICS}


class CorrectableRateGate:
    """Field 1501's rate per minute between consecutive readings (bounded).

    Measured from the previous *reading*, not from the first observation of the
    previous value as the violation duty cycles are: a counter flat for an hour
    that jumps in one exporter window is a burst, and dividing it by the hour
    would hide it. An unchanged reading is no news for ``stale_seconds`` after
    the last change -- the exporter may not have refreshed -- and a real 0 after
    that; a decrease is a counter reset (a GPU reset restarts it): 0. Only
    delivery depends on this rate; the control plane grades its own windows.
    """

    def __init__(self, *, stale_seconds: float, max_keys: int) -> None:
        self.stale_seconds = stale_seconds
        self.max_keys = max_keys
        self._last: OrderedDict[str, tuple[float, datetime, datetime]] = OrderedDict()

    def rate(
        self, key: str, sample: Any, observed_at: datetime
    ) -> tuple[float | None, bool]:
        """``(rate per minute | None, no news)`` for one sample."""

        if sample.canonical_name not in RATE_GRADED_COUNTERS:
            return None, False
        last = self._last.get(key)
        value = float(sample.value)
        changed_at = observed_at if last is None or last[0] != value else last[2]
        self._last[key] = (value, observed_at, changed_at)
        self._last.move_to_end(key)
        while len(self._last) > self.max_keys:
            self._last.popitem(last=False)
        if last is None:
            return None, False
        previous_value, seen_at, previous_change = last
        elapsed = (observed_at - seen_at).total_seconds()
        if elapsed <= 0:
            return None, True
        delta = value - previous_value
        if delta < 0:
            return 0.0, False
        if delta == 0:
            quiet = (observed_at - previous_change).total_seconds()
            return (None, True) if quiet <= self.stale_seconds else (0.0, False)
        return delta * 60.0 / elapsed, False
