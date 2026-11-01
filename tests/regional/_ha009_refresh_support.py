"""Offline refresher CronJob documents and a fake clock for HA-009's window guard."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

CRONJOB = "gpu-fault-aurora-credential-refresh"
NAMESPACE = "unit"
CREATED = "2026-10-31T00:00:00Z"


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def stamp(value: datetime) -> str:
    """Kubernetes' RFC 3339 form, second precision."""

    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cronjob(
    schedule: object = "17 * * * *",
    *,
    time_zone: object = "Etc/UTC",
    suspend: object = False,
    last_schedule: datetime | None = None,
    last_success: datetime | None = None,
    active: Iterable[str] = (),
    created: str = CREATED,
) -> dict[str, Any]:
    """A CronJob as ``kubectl get cronjob -o json`` returns it (no Job template)."""

    spec: dict[str, Any] = {
        "schedule": schedule,
        "suspend": suspend,
        "concurrencyPolicy": "Forbid",
    }
    if time_zone is not None:
        spec["timeZone"] = time_zone
    status: dict[str, Any] = {}
    if last_schedule is not None:
        status["lastScheduleTime"] = stamp(last_schedule)
    if last_success is not None:
        status["lastSuccessfulTime"] = stamp(last_success)
    names = list(active)
    if names:
        status["active"] = [
            {"kind": "Job", "name": name, "namespace": NAMESPACE} for name in names
        ]
    return {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "metadata": {
            "name": CRONJOB,
            "namespace": NAMESPACE,
            "uid": "cronjob-uid",
            "creationTimestamp": created,
        },
        "spec": spec,
        "status": status,
    }


def quiet_cronjob(now: datetime | None = None) -> dict[str, Any]:
    """A refresher whose next tick is 40 minutes after ``now`` (beyond HA-009's
    critical window) and whose previous tick, 20 minutes ago, has completed."""

    current = (now or datetime.now(timezone.utc)).replace(second=0, microsecond=0)
    previous = current - timedelta(minutes=20)
    return cronjob(
        f"{previous.minute} * * * *",
        last_schedule=previous,
        last_success=previous + timedelta(seconds=5),
    )


class FakeClock:
    """Wall clock and monotonic clock that only ``sleep`` moves."""

    def __init__(self, start: datetime) -> None:
        self.wall = start
        self.mono = 1000.0
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0, "a wait must never sleep a negative time"
        self.sleeps.append(seconds)
        self.wall += timedelta(seconds=seconds)
        self.mono += seconds


class Refresher:
    """The CronJob as its controller reports it while ``clock`` runs.

    Each tick's Job is created ``start_lag_seconds`` after the tick (the
    controller records the tick itself as ``lastScheduleTime``), stays active
    for ``job_seconds``, and ``lastSuccessfulTime`` (its completion time)
    appears ``success_lag_seconds`` after it left ``active``: the controller
    records success one sync after it drops the Job. ``previous`` ran and
    completed before the clock started.
    """

    def __init__(
        self,
        clock: FakeClock,
        ticks: Iterable[datetime],
        *,
        previous: datetime,
        start_lag_seconds: float = 0.0,
        job_seconds: float = 5.0,
        success_lag_seconds: float = 0.0,
        never_completes: bool = False,
        schedule: object = "17 * * * *",
        time_zone: object = "Etc/UTC",
    ) -> None:
        self.clock = clock
        self.ticks = sorted(ticks)
        self.previous = previous
        self.start_lag = timedelta(seconds=start_lag_seconds)
        self.job = timedelta(seconds=job_seconds)
        self.success_lag = timedelta(seconds=success_lag_seconds)
        self.never_completes = never_completes
        self.schedule = schedule
        self.time_zone = time_zone
        self.reads: list[datetime] = []

    def __call__(self) -> dict[str, Any]:
        now = self.clock.now()
        self.reads.append(now)
        started = [tick for tick in self.ticks if tick + self.start_lag <= now]
        running = [
            tick
            for tick in started
            if self.never_completes or now < tick + self.start_lag + self.job
        ]
        succeeded = [
            tick + self.start_lag + self.job
            for tick in started
            if not self.never_completes
            and now >= tick + self.start_lag + self.job + self.success_lag
        ]
        return cronjob(
            self.schedule,
            time_zone=self.time_zone,
            last_schedule=max(started, default=self.previous),
            last_success=max(succeeded, default=self.previous + timedelta(seconds=5)),
            active=[f"{CRONJOB}-{int(tick.timestamp()) // 60}" for tick in running],
        )
