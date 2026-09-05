"""Timestamped narration for a release, and the clock behind it.

Separate from `regional_release_state` because these lines are the operator's
view of a release rather than part of its durable state: nothing here reads or
writes the state ConfigMap, and `save_state` is the only caller that needs both.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import Any


class PhaseClock:
    """The one monotonic origin every narration line measures against.

    Wall-clock stamps alone force the reader to subtract two timestamps out of a
    three-thousand-line log to learn that a phase took five minutes, and the
    stamps have second resolution while the checkpoints they bracket are seconds
    apart. A monotonic origin also survives a clock step mid-release, which a
    stamp subtraction does not.
    """

    def __init__(self) -> None:
        self._started: float | None = None
        self._last: float | None = None

    def start(self) -> None:
        self._started = time.monotonic()
        self._last = None

    def split(self) -> tuple[float, float]:
        """Seconds since the previous split, and since the origin."""

        now = time.monotonic()
        if self._started is None:
            self._started = now
        since = now - (self._last if self._last is not None else self._started)
        self._last = now
        return since, now - self._started

    def total(self) -> float:
        if self._started is None:
            return 0.0
        return time.monotonic() - self._started


PHASE_CLOCK = PhaseClock()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def narrate_release_start(mode: str, *, dry_run: bool) -> None:
    """Open the log with what this invocation is, and start its clock.

    One `gpu-fault-admin deploy` drives this script a dozen times -- plan,
    preflight, deploy, verify, commit -- and their outputs run together into a
    single stream, so a slow upgrade cannot be attributed to an invocation
    without a line saying which one began where. The clock starts here rather
    than at the first checkpoint so the phases before the first `save_state`
    (uploads, preflight) are inside the measured window instead of vanishing
    into the first phase's elapsed time.
    """

    PHASE_CLOCK.start()
    print(
        f"release-begin {_stamp()} mode={mode} dry_run={str(dry_run).lower()}",
        file=sys.stderr,
        flush=True,
    )


def narrate_release_end(mode: str, *, exit_code: int) -> None:
    """Close the log with the invocation's outcome and its wall time.

    Printed from a `finally`, so a release that dies still gets its bound: the
    unbounded case is the one where an operator cannot tell whether the last
    command is still running or the process is gone.
    """

    print(
        f"release-end {_stamp()} mode={mode} exit_code={exit_code} "
        f"total={PHASE_CLOCK.total():.1f}s",
        file=sys.stderr,
        flush=True,
    )


def narrate_step(kind: str, **fields: Any) -> None:
    """Narrate work that happens between two durable checkpoints.

    A phase line is printed only when `save_state` commits, and the longest
    stretches of a release live inside one phase rather than between two:
    `data-plane-progress` is checkpointed once per cluster, so a four-wave
    rollout of one cluster prints two lines seven and a half minutes apart while
    the Reconciler installs node by node. Nothing in between says which wave is
    running or what it is waiting for, which is the gap these lines fill.

    They carry `total` but never take a split: a split is the cost of a phase,
    and consuming one here would charge this line's interval to a phase that has
    not ended yet.
    """

    parts = [f"{kind} {_stamp()}"]
    parts.extend(f"{name}={value}" for name, value in fields.items())
    parts.append(f"total={PHASE_CLOCK.total():.1f}s")
    print(" ".join(parts), file=sys.stderr, flush=True)


def narrate_phase(release: Any, phase: str) -> None:
    """Announce a durable checkpoint on the stream the operator is watching.

    `save_state` is the only funnel for release state, but it writes a ConfigMap
    and prints nothing, so an upgrade narrates thousands of `+ kubectl` lines
    without ever saying which of its dozen transaction phases it is in. Reading
    the state back does not recover that either: `completed_phases` is stored
    sorted alphabetically, so `registry-staged` appears before `schema-ready`
    even though the orchestrator reaches them the other way round. A real
    upgrade checkpoints sixteen times, so one line each is cheap next to the
    command trace, and it is the only timestamp in the log -- without it there
    is no way to tell which phase is the slow one.

    Printed after the write rather than before, so a line is a promise that the
    checkpoint survived: resume and rollback both key off persisted phases.
    """

    fields = [f"release-phase {_stamp()} {phase}"]
    lifecycle = release.state.get("release_lifecycle")
    if lifecycle:
        fields.append(f"lifecycle={lifecycle}")
    expected = release.state.get("cluster_ids") or []
    if expected:
        done = release.state.get("completed_cluster_ids") or []
        # Several checkpoints share one phase name and differ only in which
        # cluster just converged, so the phase alone cannot show progress.
        fields.append(f"clusters={len(done)}/{len(expected)}")
    since, total = PHASE_CLOCK.split()
    # `elapsed` is the cost of reaching this checkpoint from the previous one --
    # the number that says which phase to go optimize -- and `total` saves the
    # reader from adding a column of them up.
    fields.append(f"elapsed={since:.1f}s")
    fields.append(f"total={total:.1f}s")
    print(" ".join(fields), file=sys.stderr, flush=True)
