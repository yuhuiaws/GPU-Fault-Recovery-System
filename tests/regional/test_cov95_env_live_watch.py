"""GF-ENV-LIVE observation: cadence parsing, replay progress and the poll loop's reads."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.e2e.regional import env_live_watch as watch
from tests.regional._env_live_world import DROPIN, PROXY, UNIT_INVOCATION, World

T0 = datetime(2026, 11, 5, 4, 0, tzinfo=timezone.utc)


def test_a_boolean_collector_setting_keeps_the_default_interval() -> None:
    cadence = watch.CollectorCadence.from_probe(
        {"collector_env": {watch.INTERVAL_KEY: True}}
    )

    assert cadence.interval_seconds == watch.DEFAULT_INTERVAL_SECONDS
    assert watch.INTERVAL_KEY not in cadence.configured


def test_a_gone_drop_in_on_a_restarted_unit_names_the_restart() -> None:
    world = World()
    baseline = watch.repoint_baseline(world.repoint())
    snapshot = world.snapshot()
    snapshot["metrics_unit"].update(DropInPaths="", InvocationID="0f1e2d3c")

    diagnosis = watch.diagnose_zero_reads(baseline, snapshot, {}, proxy_endpoint=PROXY)

    assert diagnosis["reason"] == watch.ZERO_READS_DROPIN_GONE
    assert DROPIN in str(baseline["dropin"])
    assert (
        f"; the unit restarted (InvocationID {UNIT_INVOCATION} -> 0f1e2d3c"
        in (diagnosis["detail"])
    )


def progress(*, armed: bool = True) -> watch.ReplayProgress:
    return watch.ReplayProgress(planned_reads=3, armed=armed, proxy_endpoint=PROXY)


def test_progress_without_reads_has_served_nothing_countable() -> None:
    value = progress()

    assert value.served is None
    value.begin({"status": {"armed": True}}, T0)
    assert (value.start_reads, value.served) == (None, None)
    assert value.inconclusive() == ["the arm receipt carried no proxy read count"]


def test_a_status_read_without_a_status_object_is_an_error_not_a_read() -> None:
    value = progress(armed=False)

    value.begin(None, T0)

    assert value.status_errors == ["a /status read returned no status object"]
    assert value.status_reads == 0
    assert value.inconclusive() == ["no proxy /status read succeeded after arming"]


def test_a_status_without_a_read_count_keeps_the_last_count() -> None:
    value = progress()
    value.begin({"status": {"armed": True, "reads": 4}}, T0)

    value.update({"status": {"armed": True, "failed_reads": 2}}, T0)

    assert (value.reads, value.failed_reads, value.status_reads) == (4, 2, 2)


def test_a_pass_through_proxy_that_reports_itself_armed_is_a_violation() -> None:
    value = progress(armed=False)

    value.begin({"status": {"armed": True, "reads": 1}}, T0)

    assert value.passthrough_violations == [
        f"{T0.isoformat()}: pass-through proxy reported armed=True next_values={{}}"
    ]


def test_refused_reads_without_a_diagnosis_are_named_with_the_last_error() -> None:
    value = progress()
    value.begin({"status": {"armed": True, "reads": 2}}, T0)
    value.update(
        {
            "status": {
                "armed": True,
                "reads": 2,
                "failed_reads": 2,
                "last_error": "upstream timeout",
            }
        },
        T0,
    )

    assert value.inconclusive() == [
        "the collector consumed 0 of 3 planned replay reads before the deadline",
        "the proxy refused 2 collector read(s) with 503 (last error: upstream timeout)",
    ]


class FirstPollJudge:
    reads_node = False

    def __init__(self) -> None:
        self.snapshots: list[watch.Snapshot] = []

    def observe(self, snapshot: watch.Snapshot, progress: Any) -> None:
        self.snapshots.append(snapshot)

    def decide(self, progress: Any, now: datetime, *, final: bool) -> Any:
        return watch.Judgement("INCONCLUSIVE", {}, ["decided on the first poll"], {})

    def wait_until(self, progress: Any, now: datetime) -> datetime | None:
        return None


def window() -> watch.ObservationWindow:
    return watch.ObservationWindow(
        started_at=T0,
        terms={"replay": 60.0},
        cap_at=T0 + timedelta(minutes=5),
        caps={"maintenance": (T0 + timedelta(minutes=5)).isoformat()},
    )


def scripted_clock(*times: datetime) -> Any:
    values: Iterator[datetime] = iter(times)
    last = [times[-1]]

    def clock() -> datetime:
        last[0] = next(values, last[0])
        return last[0]

    return clock


def test_watch_records_a_failed_status_read_and_never_counts_it() -> None:
    value = progress()
    value.begin({"status": {"armed": True, "reads": 0}}, T0)

    def status() -> dict[str, Any]:
        raise RuntimeError("kubectl exec timed out")

    judge = FirstPollJudge()
    result = watch.watch(
        judge,
        value,
        watch.WatchReads(
            status=status,
            gpu_state=lambda: {"gpus": {}},
            activity=lambda: {"incidents": []},
            node=lambda: {"name": "node-a"},
        ),
        window(),
        poll_seconds=15,
        clock=scripted_clock(T0),
        sleep=lambda seconds: None,
    )

    assert result.read_errors == [
        f"{T0.isoformat()} status: RuntimeError: kubectl exec timed out"
    ]
    assert value.status_reads == 1, "only the arm receipt was counted"
    assert judge.snapshots[0].gpu_state == {"gpus": {}}
    assert result.polls == 1


def test_watch_skips_reads_past_the_observation_cap() -> None:
    value = progress()
    value.begin({"status": {"armed": True, "reads": 0}}, T0)
    late = T0 + timedelta(minutes=5, seconds=watch.READ_GRACE_SECONDS + 1)
    reads: list[str] = []

    judge = FirstPollJudge()
    result = watch.watch(
        judge,
        value,
        watch.WatchReads(
            status=lambda: reads.append("status") or {},
            gpu_state=lambda: reads.append("gpu_state") or {},
            activity=lambda: reads.append("activity") or {},
            node=lambda: reads.append("node") or {},
        ),
        window(),
        poll_seconds=15,
        clock=scripted_clock(T0, late),
        sleep=lambda seconds: None,
    )

    assert reads == []
    assert result.read_errors == [
        f"{T0.isoformat()} {name}: skipped, past the observation cap"
        for name in ("status", "gpu_state", "activity")
    ]
    assert judge.snapshots[0].activity is None
