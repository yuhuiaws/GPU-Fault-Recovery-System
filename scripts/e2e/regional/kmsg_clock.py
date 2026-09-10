"""The kernel clock trails the wall clock a runner reads before an injection.

A line written to ``/dev/kmsg`` is stamped on the kernel's boot-time base and
the collector converts it with a boot-time estimate, so its ``observed_at``
lands a little *before* the ``datetime.now()`` the runner took just before the
write. COLLECT-021's XID 13 read 10:51:08.9 against an injection at 10:51:09.x;
an exact ``observed_after`` cut dropped the event and the runner waited its
whole budget for a decision the control plane had already recorded. A
marker-tagged read identifies its line by the marker, so the time bound is
only a scan limit and is widened by ``KMSG_CLOCK_SKEW_SECONDS`` here.
"""

from __future__ import annotations

from datetime import datetime, timedelta

KMSG_CLOCK_SKEW_SECONDS = 30


def marker_observed_after(
    marker: str, observed_after: datetime | None
) -> datetime | None:
    """The ``observed_after`` bound to send for a marker-tagged store read."""

    if observed_after is None or not marker:
        return observed_after
    return observed_after - timedelta(seconds=KMSG_CLOCK_SKEW_SECONDS)
