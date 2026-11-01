"""HA-009 keeps the product's hourly Aurora credential refresh out of its rotation.

The product's refresher CronJob ``gpu-fault-aurora-credential-refresh``
(``deploy/control-plane/regional/aurora-credential-refresh.yaml``: schedule
``"17 * * * *"``, ``timeZone: Etc/UTC``, ``concurrencyPolicy: Forbid``) copies
AWSCURRENT into the CPU Secret on every tick that finds it stale. HA-009 rotates
the master password and then proves that its OWN refresh Job published the
rotation. A tick between the rotation request and the end of that first refresh
publishes the rotated credential first: live on 2026-11-01 (round 14, hp lane)
the first refresh's binding proof read the CPU Secret at 09:16:40Z and again
after the 09:17:00Z tick's Job had patched it (``rotated=True
restarted=False``, done 09:17:05Z) and failed "CPU Secret changed during
proof". The product behaved as designed; a run that did not straddle minute 17
passed, which is why earlier rounds were green.

Two read-only guards. The CronJob is never suspended: a suspended refresher
left behind by a dead runner would break the next real rotation.

* Before the rotation, :func:`await_clear_window` reads the CronJob. When a
  tick may fire inside the critical window, it waits until that tick's Job has
  completed (``lastSuccessfulTime >= tick`` and no active Job), then decides
  again; bounded by the caller's phase budget, and the decision is written to
  ``refresh-cronjob-window.json``.
* After the first refresh, :func:`checked_first_refresh` reads the CronJob once
  more (:func:`self_check`). A refresher run inside [rotation requested, end of
  the first refresh] -- a schedule edited mid-case, clock skew, a late
  controller -- may have published the rotation before the case's Job, so the
  failures such a run causes are INCONCLUSIVE, never a product FAIL.

Only the product's own form ``M * * * *`` is supported. Any other schedule is
refused with its reason: a tick that cannot be predicted cannot be kept out of
the window.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.aurora_binding import (
    REFRESHER_NOT_IDLE,
    SECRET_CHANGED_DURING_PROOF,
    BindingError,
)
from scripts.e2e.regional.ha009_refresh import CaseError

EVIDENCE = "refresh-cronjob-window.json"
SUPPORTED_SCHEDULE = "M * * * *"
# ``M * * * *`` fires once per wall-clock hour.
PERIOD_SECONDS = 3600
# A due tick without a Job yet: the CronJob controller requeues itself for the
# exact schedule time, so the Job normally exists within a second (live
# 2026-11-01: created :17:00, completed :17:05). Past this grace the controller
# is late; a Job it still starts later is caught by the self-check.
TICK_START_GRACE_SECONDS = 120
# Poll every 5 s while a tick is due or its Job runs; sleep at most 60 s before
# a future tick, so an edited schedule is read again within a minute.
POLL_SECONDS = 5
IDLE_POLL_SECONDS = 60
# The self-check compares second-precision CronJob times with this host's clock.
SKEW_SECONDS = 5
UTC_ZONES = frozenset({"UTC", "Etc/UTC", "Etc/UCT", "Etc/GMT", "GMT", "Etc/Zulu"})
# The BindingErrors a refresher run explains: its Job patched the CPU Secret
# between the proof's two reads, or was still active when the proof read the
# CronJob. Every other BindingError keeps its FAIL.
REFRESHER_BINDING_ERRORS = frozenset(
    "Aurora binding: " + reason
    for reason in (SECRET_CHANGED_DURING_PROOF, REFRESHER_NOT_IDLE)
)
_MINUTE = re.compile(r"[0-9]{1,2}")


class RefreshWindowError(CaseError):
    """No rotation: the refresher's ticks cannot be kept out of the window."""


class RefreshTickInterference(CaseError):
    """A refresher run inside the window explains the first refresh's failure."""

    def __init__(self, reason: str, error: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.error = error


def iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value is not None else None


def tick_minute(schedule: object) -> int:
    """The minute of the product's hourly ``M * * * *``; refuse any other form."""

    fields = schedule.split() if isinstance(schedule, str) else []
    if (
        len(fields) == 5
        and fields[1:] == ["*", "*", "*", "*"]
        and _MINUTE.fullmatch(fields[0])
        and int(fields[0]) < 60
    ):
        return int(fields[0])
    raise RefreshWindowError(
        f"refresh CronJob schedule {schedule!r} is not the product's hourly "
        f"{SUPPORTED_SCHEDULE!r} form (aurora-credential-refresh.yaml ships "
        "'17 * * * *'): HA-009 cannot predict its ticks, so it cannot keep them "
        "out of the rotation window and does not request the rotation"
    )


def schedule_zone(name: object) -> tzinfo:
    """``spec.timeZone``; unset means the controller's own zone, UTC on EKS."""

    if name is None or name == "":
        return timezone.utc
    if isinstance(name, str):
        if name in UTC_ZONES:
            return timezone.utc
        try:
            return ZoneInfo(name)
        except (ValueError, KeyError, OSError):
            pass
    raise RefreshWindowError(
        f"refresh CronJob timeZone {name!r} is not a time zone this host can "
        "resolve: HA-009 cannot place its ticks and does not request the rotation"
    )


def ticks(minute: int, zone: tzinfo, start: datetime, end: datetime) -> list[datetime]:
    """Every fire time in [start, end] (UTC): the whole minutes whose wall-clock
    minute in ``zone`` is ``minute``. A half-hour zone such as Asia/Kolkata
    moves the UTC minute; a whole-hour DST change does not."""

    current = start.astimezone(timezone.utc).replace(second=0, microsecond=0)
    if current < start:
        current += timedelta(minutes=1)
    found = []
    while current <= end:
        if current.astimezone(zone).minute == minute:
            found.append(current)
        current += timedelta(minutes=1)
    return found


def around(minute: int, zone: tzinfo, now: datetime) -> tuple[datetime, datetime]:
    """The last fire time at or before ``now`` and the first one after it."""

    horizon = timedelta(hours=3)
    before = ticks(minute, zone, now - horizon, now)
    after = ticks(minute, zone, now + timedelta(microseconds=1), now + horizon)
    if not before or not after:
        raise RefreshWindowError("refresh CronJob ticks could not be placed")
    return before[-1], after[0]


def moment(value: object, field: str) -> datetime | None:
    if value is None or value == "":
        return None
    parsed = None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise RefreshWindowError(f"refresh CronJob {field} {value!r} is malformed")
    return parsed.astimezone(timezone.utc)


def cronjob_state(cronjob: dict[str, Any]) -> dict[str, Any]:
    """What the window reads from the CronJob: schedule and status, nothing else."""

    metadata = cronjob.get("metadata") or {}
    spec = cronjob.get("spec") or {}
    status = cronjob.get("status") or {}
    active = status.get("active") or []
    return {
        "name": metadata.get("name"),
        "uid": metadata.get("uid"),
        "created_at": metadata.get("creationTimestamp"),
        "schedule": spec.get("schedule"),
        "time_zone": spec.get("timeZone"),
        "suspend": spec.get("suspend", False),
        "last_schedule_time": status.get("lastScheduleTime"),
        "last_successful_time": status.get("lastSuccessfulTime"),
        # A malformed list fails closed: it reads as a running Job.
        "active": sorted(
            str(item.get("name") or "?") if isinstance(item, dict) else "?"
            for item in active
        )
        if isinstance(active, list)
        else ["?"],
    }


def read_cronjob(control: Callable[..., str], name: str) -> dict[str, Any]:
    """Read-only ``kubectl get cronjob``; the case never writes the CronJob."""

    value = json.loads(control("get", "cronjob", name, "-o", "json", timeout=60))
    if (
        not isinstance(value, dict)
        or value.get("kind") != "CronJob"
        or (value.get("metadata") or {}).get("name") != name
    ):
        raise RefreshWindowError(f"refresh CronJob {name} read returned another object")
    return value


@dataclass(frozen=True)
class Decision:
    """Whether the critical window opening now is free of refresher ticks."""

    previous_tick: datetime
    next_tick: datetime
    wait_for: datetime | None
    why: str


def decide(cronjob: dict[str, Any], now: datetime, critical_seconds: int) -> Decision:
    state = cronjob_state(cronjob)
    minute = tick_minute(state["schedule"])
    zone = schedule_zone(state["time_zone"])
    if state["suspend"] not in (False, None):
        raise RefreshWindowError(
            f"refresh CronJob {state['name']} is suspended or not plainly enabled "
            f"(spec.suspend {state['suspend']!r}): HA-009 never suspends or resumes "
            "the product's refresher and does not rotate the master password while "
            "it is suspended"
        )
    previous, upcoming = around(minute, zone, now)
    scheduled = moment(state["last_schedule_time"], "lastScheduleTime")
    created = moment(state["created_at"], "creationTimestamp")
    if state["active"]:
        return Decision(
            previous,
            upcoming,
            scheduled or previous,
            f"refresher Job {', '.join(state['active'])} is active",
        )
    # The tick of the current minute (or a late one): due, but its Job does not
    # exist yet. A CronJob created after the tick never runs it.
    unrecorded = (scheduled is None or scheduled < previous) and (
        created is None or created <= previous
    )
    if unrecorded and (now - previous).total_seconds() < TICK_START_GRACE_SECONDS:
        return Decision(
            previous, upcoming, previous, "the tick is due and has no Job yet"
        )
    ahead = (upcoming - now).total_seconds()
    if ahead <= critical_seconds:
        return Decision(
            previous,
            upcoming,
            upcoming,
            f"the next tick is {ahead:.0f}s away, inside the "
            f"{critical_seconds}s critical window",
        )
    return Decision(previous, upcoming, None, "clear")


def tick_completed(cronjob: dict[str, Any], tick: datetime) -> bool:
    """The brief's completion: ``lastSuccessfulTime >= tick`` and no active Job."""

    state = cronjob_state(cronjob)
    succeeded = moment(state["last_successful_time"], "lastSuccessfulTime")
    return not state["active"] and succeeded is not None and succeeded >= tick


def await_clear_window(
    case_dir: Path,
    read: Callable[[], dict[str, Any]],
    *,
    critical_seconds: int,
    budget_seconds: int,
    terms: dict[str, int],
    log: Callable[[str], None],
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Return once no refresher tick can fire within ``critical_seconds``.

    A tick inside the window is waited out until its Job has completed, then
    the window is decided again from that moment. ``budget_seconds`` bounds the
    whole wait; running out of it, an unsupported schedule, an unknown time
    zone or a suspended refresher refuse the rotation. The decision and every
    CronJob state read are written to ``refresh-cronjob-window.json``.
    """

    record: dict[str, Any] = {
        "schema_version": 1,
        "supported_schedule": SUPPORTED_SCHEDULE,
        "critical_window_seconds": critical_seconds,
        "critical_window_terms": dict(terms),
        "budget_seconds": budget_seconds,
        "started_at": iso(now()),
        "waited_for": [],
        "polls": 0,
        "waited_seconds": 0.0,
    }
    started = monotonic()
    waiting: datetime | None = None
    try:
        if critical_seconds + TICK_START_GRACE_SECONDS >= PERIOD_SECONDS:
            raise RefreshWindowError(
                f"the {critical_seconds}s critical window leaves no clear window "
                f"between ticks {PERIOD_SECONDS}s apart"
            )
        while True:
            cronjob = read()
            current = now()
            record["polls"] += 1
            record["cronjob_state"] = cronjob_state(cronjob)
            record.setdefault("cronjob_state_initial", record["cronjob_state"])
            record["schedule"] = record["cronjob_state"]["schedule"]
            record["time_zone"] = record["cronjob_state"]["time_zone"]
            # Decided on every read, so a schedule edited or a refresher
            # suspended during the wait refuses at once.
            decision = decide(cronjob, current, critical_seconds)
            record["previous_tick"] = iso(decision.previous_tick)
            record["next_tick"] = iso(decision.next_tick)
            if waiting is None or tick_completed(cronjob, waiting):
                if decision.wait_for is None:
                    window_end = current + timedelta(seconds=critical_seconds)
                    record["clear"] = True
                    record["window"] = {"start": iso(current), "end": iso(window_end)}
                    log(
                        f"refresh CronJob window clear until {iso(window_end)} "
                        f"(next tick {iso(decision.next_tick)})"
                    )
                    return record
                if decision.wait_for != waiting:
                    waiting = decision.wait_for
                    record["waited_for"].append(
                        {
                            "tick": iso(waiting),
                            "why": decision.why,
                            "seen_at": iso(current),
                        }
                    )
                    log(
                        f"refresh CronJob tick {iso(waiting)}: {decision.why}; waiting "
                        f"for its Job to complete (budget {budget_seconds}s)"
                    )
            elapsed = monotonic() - started
            if elapsed >= budget_seconds:
                state = record["cronjob_state"]
                raise RefreshWindowError(
                    f"the refresher tick {iso(waiting)} had not completed after "
                    f"{budget_seconds}s (lastScheduleTime "
                    f"{state['last_schedule_time']}, lastSuccessfulTime "
                    f"{state['last_successful_time']}, active {state['active']}): "
                    "HA-009 does not request the rotation while a refresher run may "
                    "still publish inside its window"
                )
            ahead = (waiting - current).total_seconds() if waiting else 0.0
            pause = (
                min(IDLE_POLL_SECONDS, ahead) if ahead > POLL_SECONDS else POLL_SECONDS
            )
            sleep(min(pause, budget_seconds - elapsed))
    except RefreshWindowError as exc:
        record["refused"] = str(exc)
        raise
    finally:
        record["waited_seconds"] = round(monotonic() - started, 1)
        write_json_atomic(case_dir / EVIDENCE, record)


def self_check(
    cronjob: dict[str, Any], start: datetime, end: datetime
) -> dict[str, Any]:
    """Did a refresher run overlap [rotation requested, end of the first refresh]?

    ``lastScheduleTime`` inside the interval is a tick there;
    ``lastSuccessfulTime`` inside it also catches a late start whose tick lies
    earlier; a Job still active at the check is a run in flight.
    """

    state = cronjob_state(cronjob)
    low = start - timedelta(seconds=SKEW_SECONDS)
    high = end + timedelta(seconds=SKEW_SECONDS)
    scheduled = moment(state["last_schedule_time"], "lastScheduleTime")
    succeeded = moment(state["last_successful_time"], "lastSuccessfulTime")
    runs = []
    if scheduled is not None and low <= scheduled <= high:
        runs.append(f"run scheduled at {iso(scheduled)} (lastScheduleTime)")
    if succeeded is not None and low <= succeeded <= high:
        runs.append(f"run completed at {iso(succeeded)} (lastSuccessfulTime)")
    if state["active"]:
        runs.append(f"Job {', '.join(state['active'])} active at the check")
    reason = ""
    if runs:
        reason = (
            f"the product's refresher CronJob {state['name']} ran inside HA-009's "
            f"window [{iso(start)} rotation requested, {iso(end)} end of the first "
            f"refresh]: {'; '.join(runs)} (schedule {state['schedule']!r}, "
            f"timeZone {state['time_zone']!r}). That run, not the case's own "
            "refresh Job, may have published the rotated credential (a schedule "
            "edited mid-case, clock skew or a late controller), so the first "
            "refresh's outcome does not judge the product"
        )
    return {
        "window": {"start": iso(start), "end": iso(end)},
        "skew_seconds": SKEW_SECONDS,
        "cronjob_state": state,
        "inside": bool(runs),
        "reason": reason,
    }


def record_self_check(
    case_dir: Path,
    record: dict[str, Any],
    read: Callable[[], dict[str, Any]],
    start: datetime | None,
    end: datetime,
) -> dict[str, Any]:
    """Add the self-check to the window evidence; a failed read never hides
    the outcome it was meant to explain."""

    try:
        if start is None:
            raise RefreshWindowError("rotation request time is unknown")
        check = self_check(read(), start, end)
    except ProcessSupervisionLost:
        raise
    except Exception as exc:
        check = {
            "window": {"start": iso(start), "end": iso(end)},
            "inside": False,
            "error": f"{type(exc).__name__}: self-check read failed",
        }
    record["self_check"] = check
    write_json_atomic(case_dir / EVIDENCE, record)
    return check


def checked_first_refresh(
    case_dir: Path,
    record: dict[str, Any],
    read: Callable[[], dict[str, Any]],
    refresh: Callable[[], dict[str, Any]],
    *,
    requested_at: datetime | None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Run the rotation's first refresh, then the self-check (recorded whatever
    the outcome).

    A BindingError a refresher run inside the window explains becomes
    :class:`RefreshTickInterference` (INCONCLUSIVE); every other error, and any
    error while no run overlapped, propagates unchanged.
    """

    try:
        job = refresh()
    except Exception as exc:
        check = record_self_check(case_dir, record, read, requested_at, now())
        if (
            isinstance(exc, BindingError)
            and check["inside"]
            and str(exc) in REFRESHER_BINDING_ERRORS
        ):
            raise RefreshTickInterference(
                check["reason"], f"{type(exc).__name__}: {exc}"
            ) from None
        raise
    record_self_check(case_dir, record, read, requested_at, now())
    return job


def explain_by_refresher(
    result: dict[str, Any], check: dict[str, Any] | None, explained: frozenset[str]
) -> None:
    """A FAIL whose every error a refresher run inside the window explains
    (the first Job found the Secret already rotated) is INCONCLUSIVE."""

    errors = result.get("errors")
    if (
        result.get("verdict") == "FAIL"
        and check
        and check.get("inside") is True
        and errors
        and set(errors) <= explained
    ):
        result["verdict"] = "INCONCLUSIVE"
        result["inconclusive_reason"] = check["reason"]
