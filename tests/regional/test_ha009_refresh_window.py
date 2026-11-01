"""HA-009 keeps the product's hourly Aurora credential refresh out of its rotation.

Live 2026-11-01 (round 14, hp lane) the refresher CronJob's 09:17:00Z tick
published the rotated credential while the first refresh's binding proof was
running, and the case failed "CPU Secret changed during proof" although the
product behaved as designed. These tests pin the read-only guard: where the
ticks fall, the clear-window wait on a fake clock, the budgets it is charged
to, and the self-check that makes a refresher run inside the window
INCONCLUSIVE instead of a product FAIL.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from scripts.e2e.regional import aurora_binding as binding
from scripts.e2e.regional import ha009_refresh_window as window
from scripts.e2e.regional import ha009_verdicts as verdicts
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from tests.regional._aurora_binding_support import CRONJOB, SECRET, FakeAurora, encoded
from tests.regional._ha009_refresh_support import (
    FakeClock,
    Refresher,
    cronjob,
    quiet_cronjob,
    utc,
)

CRITICAL = ha009.REFRESH_CRITICAL_WINDOW_SECONDS
BUDGET = ha009.PHASE_BUDGETS["refresh_cronjob_window"]
DEADLINE = datetime(2099, 1, 1, tzinfo=timezone.utc)
SECRET_CHANGED = "Aurora binding: " + binding.SECRET_CHANGED_DURING_PROOF
REFRESHER_BUSY = "Aurora binding: " + binding.REFRESHER_NOT_IDLE


def wait(
    tmp_path: Path, clock: FakeClock, read: Any, **overrides: Any
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "critical_seconds": CRITICAL,
        "budget_seconds": BUDGET,
        "terms": ha009.REFRESH_WINDOW_TERMS,
        "log": lambda _line: None,
        "now": clock.now,
        "monotonic": clock.monotonic,
        "sleep": clock.sleep,
    }
    arguments.update(overrides)
    return window.await_clear_window(tmp_path, read, **arguments)


def evidence(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / window.EVIDENCE).read_text(encoding="utf-8"))


# -- where the ticks fall ------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "previous", "upcoming"),
    [
        ("2026-11-01T09:30:00", "2026-11-01T09:17:00", "2026-11-01T10:17:00"),
        ("2026-11-01T09:16:59", "2026-11-01T08:17:00", "2026-11-01T09:17:00"),
        ("2026-11-01T23:40:00", "2026-11-01T23:17:00", "2026-11-02T00:17:00"),
        ("2026-12-31T23:59:30", "2026-12-31T23:17:00", "2027-01-01T00:17:00"),
    ],
)
def test_hourly_ticks_roll_over_the_hour_the_day_and_the_year(
    now: str, previous: str, upcoming: str
) -> None:
    assert window.around(17, timezone.utc, utc(now)) == (utc(previous), utc(upcoming))


@pytest.mark.parametrize("now", ["2026-11-01T09:17:00", "2026-11-01T09:17:30"])
def test_a_tick_in_the_current_minute_is_the_previous_tick_not_the_next(
    now: str,
) -> None:
    assert window.around(17, timezone.utc, utc(now)) == (
        utc("2026-11-01T09:17:00"),
        utc("2026-11-01T10:17:00"),
    )


def test_the_due_tick_of_the_current_minute_blocks_until_its_job_exists() -> None:
    tick = utc("2026-11-01T09:17:00")
    previous = utc("2026-11-01T08:17:00")
    due = window.decide(
        cronjob(last_schedule=previous, last_success=previous + timedelta(seconds=5)),
        tick,
        CRITICAL,
    )
    assert (due.wait_for, due.why) == (tick, "the tick is due and has no Job yet")
    ran = window.decide(
        cronjob(last_schedule=tick, last_success=tick + timedelta(seconds=5)),
        tick + timedelta(seconds=30),
        CRITICAL,
    )
    assert (ran.wait_for, ran.next_tick) == (None, utc("2026-11-01T10:17:00")), (
        "a completed run of this minute's tick leaves 3570 s to the next one"
    )
    fresh = window.decide(
        cronjob(created="2026-11-01T09:17:20Z"), tick + timedelta(seconds=30), CRITICAL
    )
    assert fresh.wait_for is None, "a CronJob created after the tick never runs it"


@pytest.mark.parametrize("name", [None, "", "Etc/UTC", "UTC"])
def test_an_unset_or_utc_time_zone_is_utc_without_a_zone_database(name: object) -> None:
    assert window.schedule_zone(name) is timezone.utc


def test_a_half_hour_time_zone_moves_the_utc_minute_of_the_tick() -> None:
    kolkata = timezone(timedelta(hours=5, minutes=30))
    assert window.around(17, kolkata, utc("2026-11-01T09:30:00")) == (
        utc("2026-11-01T08:47:00"),
        utc("2026-11-01T09:47:00"),
    )


def test_named_zones_resolve_and_a_whole_hour_dst_change_keeps_the_minute() -> None:
    try:
        ZoneInfo("Asia/Kolkata")
        ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError:
        pytest.skip("this host has no IANA time zone database")
    kolkata = window.schedule_zone("Asia/Kolkata")
    assert window.around(17, kolkata, utc("2026-11-01T09:30:00")) == (
        utc("2026-11-01T08:47:00"),
        utc("2026-11-01T09:47:00"),
    )
    # US DST ends 2026-11-01 06:00Z: 01:17 local happens twice, an hour apart.
    new_york = window.schedule_zone("America/New_York")
    assert window.ticks(
        17, new_york, utc("2026-11-01T04:00:00"), utc("2026-11-01T08:00:00")
    ) == [utc(f"2026-11-01T0{hour}:17:00") for hour in (4, 5, 6, 7)]


@pytest.mark.parametrize(
    ("schedule", "minute"),
    [("17 * * * *", 17), ("0 * * * *", 0), ("59 * * * *", 59), (" 07  * * * * ", 7)],
)
def test_the_products_hourly_form_is_supported(schedule: str, minute: int) -> None:
    assert window.tick_minute(schedule) == minute


@pytest.mark.parametrize(
    "schedule",
    [
        "*/5 * * * *",
        "17 * * * 1",
        "17 */2 * * *",
        "17,47 * * * *",
        "17-20 * * * *",
        "@hourly",
        "CRON_TZ=UTC 17 * * * *",
        "17 * * *",
        "60 * * * *",
        "+7 * * * *",
        "007 * * * *",
        "",
        None,
        17,
    ],
)
def test_any_other_schedule_refuses_the_rotation_with_its_reason(
    schedule: object, tmp_path: Path
) -> None:
    with pytest.raises(window.RefreshWindowError, match="not the product's hourly"):
        window.tick_minute(schedule)
    clock = FakeClock(utc("2026-11-01T09:40:00"))
    with pytest.raises(window.RefreshWindowError, match="does not request"):
        wait(tmp_path, clock, lambda: cronjob(schedule))
    record = evidence(tmp_path)
    assert "not the product's hourly" in record["refused"]
    assert record["cronjob_state"]["schedule"] == schedule
    assert "clear" not in record and clock.sleeps == []


@pytest.mark.parametrize(
    ("document", "fragment"),
    [
        (cronjob(time_zone="Mars/Olympus_Mons"), "timeZone"),
        (cronjob(time_zone=["Etc/UTC"]), "timeZone"),
        (cronjob(suspend=True), "suspended"),
        (cronjob(suspend="false"), "suspended"),
    ],
)
def test_unknown_zones_and_a_suspended_refresher_refuse(
    document: dict[str, Any], fragment: str
) -> None:
    with pytest.raises(window.RefreshWindowError, match=fragment):
        window.decide(document, utc("2026-11-01T09:40:00"), CRITICAL)


def test_a_refresher_that_never_ran_only_has_its_next_tick_to_avoid() -> None:
    now = utc("2026-11-01T09:40:00")
    assert window.decide(cronjob(), now, CRITICAL).wait_for is None
    assert window.decide(cronjob(), utc("2026-11-01T09:00:00"), CRITICAL).wait_for == (
        utc("2026-11-01T09:17:00")
    )


def test_the_cronjob_is_only_read(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def control(*args: str, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return json.dumps(cronjob())

    monkeypatch.setattr(ha009.BASE, "control", control)
    monkeypatch.setattr(ha009, "CRONJOB", CRONJOB)
    assert ha009.refresh_cronjob()["metadata"]["name"] == CRONJOB
    assert calls == [(("get", "cronjob", CRONJOB, "-o", "json"), {"timeout": 60})]
    with pytest.raises(window.RefreshWindowError, match="another object"):
        window.read_cronjob(lambda *a, **k: json.dumps({"kind": "Job"}), CRONJOB)


# -- the clear-window wait -------------------------------------------------------


def test_no_wait_when_the_next_tick_lies_beyond_the_critical_window(
    tmp_path: Path,
) -> None:
    clock = FakeClock(utc("2026-11-01T09:40:00"))
    read = Refresher(clock, [], previous=utc("2026-11-01T09:17:00"))
    lines: list[str] = []
    record = wait(tmp_path, clock, read, log=lines.append)
    assert clock.sleeps == [] and len(read.reads) == 1
    assert record["clear"] is True and record["waited_for"] == []
    assert record["window"] == {
        "start": "2026-11-01T09:40:00+00:00",
        "end": "2026-11-01T10:11:40+00:00",
    }
    assert record["next_tick"] == "2026-11-01T10:17:00+00:00"
    assert (record["schedule"], record["time_zone"]) == ("17 * * * *", "Etc/UTC")
    assert record["waited_seconds"] == 0.0 and record["polls"] == 1
    assert evidence(tmp_path) == record, "the decision is the case evidence"
    assert lines == [
        "refresh CronJob window clear until 2026-11-01T10:11:40+00:00 "
        "(next tick 2026-11-01T10:17:00+00:00)"
    ]


def test_a_tick_inside_the_window_is_waited_out_until_its_job_completed(
    tmp_path: Path,
) -> None:
    """The live run's moment: its probe baseline was ready at 09:13:00Z."""

    clock = FakeClock(utc("2026-11-01T09:13:00"))
    tick = utc("2026-11-01T09:17:00")
    read = Refresher(clock, [tick], previous=utc("2026-11-01T08:17:00"))
    lines: list[str] = []
    record = wait(tmp_path, clock, read, log=lines.append)
    assert record["waited_for"] == [
        {
            "tick": "2026-11-01T09:17:00+00:00",
            "why": f"the next tick is 240s away, inside the {CRITICAL}s critical window",
            "seen_at": "2026-11-01T09:13:00+00:00",
        }
    ]
    assert record["window"]["start"] == "2026-11-01T09:17:05+00:00", (
        "the rotation starts only once the tick's Job completed"
    )
    assert record["next_tick"] == "2026-11-01T10:17:00+00:00"
    assert datetime.fromisoformat(record["window"]["end"]) < utc("2026-11-01T10:17:00")
    assert record["waited_seconds"] == 245.0
    assert max(clock.sleeps) <= window.IDLE_POLL_SECONDS
    assert record["cronjob_state_initial"]["last_schedule_time"] == (
        "2026-11-01T08:17:00Z"
    )
    assert record["cronjob_state"]["last_successful_time"] == "2026-11-01T09:17:05Z"
    assert len(lines) == 2 and "waiting for its Job to complete" in lines[0]
    assert evidence(tmp_path) == record


def test_the_current_minutes_tick_without_a_job_is_waited_for(tmp_path: Path) -> None:
    clock = FakeClock(utc("2026-11-01T09:17:00"))
    read = Refresher(
        clock,
        [utc("2026-11-01T09:17:00")],
        previous=utc("2026-11-01T08:17:00"),
        start_lag_seconds=3,
    )
    record = wait(tmp_path, clock, read)
    assert record["waited_for"][0]["why"] == "the tick is due and has no Job yet"
    assert record["window"]["start"] == "2026-11-01T09:17:10+00:00"
    assert record["next_tick"] == "2026-11-01T10:17:00+00:00"


def test_an_active_refresher_job_is_waited_for(tmp_path: Path) -> None:
    clock = FakeClock(utc("2026-11-01T09:17:02"))
    read = Refresher(
        clock, [utc("2026-11-01T09:17:00")], previous=utc("2026-11-01T08:17:00")
    )
    record = wait(tmp_path, clock, read)
    assert record["waited_for"][0]["tick"] == "2026-11-01T09:17:00+00:00"
    assert record["waited_for"][0]["why"].endswith("is active"), record["waited_for"]
    assert record["window"]["start"] == "2026-11-01T09:17:07+00:00"


def test_completion_needs_last_successful_time_not_only_an_empty_active_list(
    tmp_path: Path,
) -> None:
    """The controller drops the Job from ``active`` one sync before it records
    ``lastSuccessfulTime``; the wait ends only on the latter."""

    clock = FakeClock(utc("2026-11-01T09:16:00"))
    read = Refresher(
        clock,
        [utc("2026-11-01T09:17:00")],
        previous=utc("2026-11-01T08:17:00"),
        success_lag_seconds=7,
    )
    record = wait(tmp_path, clock, read)
    assert record["window"]["start"] == "2026-11-01T09:17:15+00:00"


def test_a_tick_that_never_completes_refuses_the_rotation_inside_the_budget(
    tmp_path: Path,
) -> None:
    clock = FakeClock(utc("2026-11-01T09:13:00"))
    read = Refresher(
        clock,
        [utc("2026-11-01T09:17:00")],
        previous=utc("2026-11-01T08:17:00"),
        never_completes=True,
    )
    with pytest.raises(window.RefreshWindowError, match=f"after {BUDGET}s"):
        wait(tmp_path, clock, read)
    assert sum(clock.sleeps) == BUDGET, "the wait never sleeps past its budget"
    record = evidence(tmp_path)
    assert "had not completed" in record["refused"]
    assert record["waited_seconds"] == float(BUDGET)


def test_a_schedule_edited_during_the_wait_refuses_at_the_next_read(
    tmp_path: Path,
) -> None:
    clock = FakeClock(utc("2026-11-01T09:13:00"))
    documents = [cronjob(), cronjob("*/5 * * * *")]
    with pytest.raises(window.RefreshWindowError, match="'\\*/5 \\* \\* \\* \\*'"):
        wait(tmp_path, clock, lambda: documents.pop(0))
    assert evidence(tmp_path)["polls"] == 2


def test_a_window_as_long_as_the_tick_period_is_refused(tmp_path: Path) -> None:
    clock = FakeClock(utc("2026-11-01T09:40:00"))
    with pytest.raises(window.RefreshWindowError, match="no clear window"):
        wait(tmp_path, clock, lambda: cronjob(), critical_seconds=3500)


# -- budgets ---------------------------------------------------------------------


def test_the_critical_window_and_wait_budget_derive_from_phase_budgets() -> None:
    terms = ha009.REFRESH_WINDOW_TERMS
    for phase in ("managed_rotation", "first_refresh_job", "secret_propagation"):
        assert terms[phase] == ha009.PHASE_BUDGETS[phase], phase
    assert CRITICAL == sum(terms.values()) == 1900
    assert BUDGET == CRITICAL + 300, "the wait covers a far tick plus its Job"
    assert CRITICAL + window.TICK_START_GRACE_SECONDS < window.PERIOD_SECONDS, (
        "a clear window must exist between two hourly ticks"
    )
    total = ha009.total_budget_seconds()
    assert total == sum(ha009.PHASE_BUDGETS.values()) + ha009.BUDGET_MARGIN_SECONDS
    assert ha009.rotation_budget_seconds() == total - ha009.BUDGET_MARGIN_SECONDS - (
        BUDGET
    )
    assert ha009.SEED_LEASE_SECONDS == total, "the seed outlives the wait too"
    assert ha009.REFRESH_WATCHDOG_SECONDS == 1900, (
        "the watchdog is armed after the wait: its delay does not include it"
    )


def test_every_window_check_charges_the_wait_where_it_is_still_ahead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    required: list[int] = []
    monkeypatch.setattr(
        ha009.BASE,
        "require_window",
        lambda _end, *, required_seconds: required.append(required_seconds),
    )
    monkeypatch.setattr(ha009, "refresh_cronjob", quiet_cronjob)
    monkeypatch.setattr(
        ha009, "secret_versions", lambda _arn: {"stages": {"AWSCURRENT": "v1"}}
    )
    state: dict[str, Any] = {
        "maintenance_window_end": DEADLINE,
        "aurora_binding": {"unit": "proof"},
    }
    ha009.await_refresh_window(tmp_path, state, "unit-secret", "v1")
    assert state["refresh_window"]["clear"] is True
    monkeypatch.setattr(
        ha009, "aurora_guard", lambda: SimpleNamespace(read=lambda expected: expected)
    )
    monkeypatch.setattr(ha009, "aws", lambda *_a: {})
    ha009.request_rotation(state)
    assert required == [
        sum(ha009.PHASE_BUDGETS.values()),
        ha009.rotation_budget_seconds(),
    ], "before the wait: the wait and every later phase; at the rotation: the rest"
    assert state["rotation_requested_at"].tzinfo is timezone.utc


def test_the_case_refuses_a_maintenance_window_shorter_than_its_wait_included(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Refused(Exception):
        pass

    required: list[int] = []

    def require_window(_end: datetime, *, required_seconds: int) -> None:
        required.append(required_seconds)
        raise Refused

    monkeypatch.setattr(ha009.BASE, "require_window", require_window)
    with pytest.raises(Refused):
        ha009.run_case(tmp_path, 1, DEADLINE)
    assert required == [ha009.total_budget_seconds()]
    assert ha009.total_budget_seconds() > BUDGET + ha009.rotation_budget_seconds()


def test_a_managed_rotation_during_the_wait_voids_the_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ha009.BASE, "require_window", lambda *_a, **_k: None)
    monkeypatch.setattr(ha009, "refresh_cronjob", quiet_cronjob)
    monkeypatch.setattr(
        ha009, "secret_versions", lambda _arn: {"stages": {"AWSCURRENT": "v2"}}
    )
    state: dict[str, Any] = {"maintenance_window_end": DEADLINE}
    with pytest.raises(ha009.CaseError, match="AWSCURRENT changed"):
        ha009.await_refresh_window(tmp_path, state, "unit-secret", "v1")


# -- the self-check after the first refresh ---------------------------------------

REQUESTED = utc("2026-11-01T09:15:14")
ENDED = utc("2026-11-01T09:17:30")


def live_tick() -> dict[str, Any]:
    """The CronJob as the hp lane read it after the failure."""

    return cronjob(
        last_schedule=utc("2026-11-01T09:17:00"),
        last_success=utc("2026-11-01T09:17:05"),
    )


def first_refresh(
    tmp_path: Path, read: Any, outcome: BaseException | dict[str, Any]
) -> dict[str, Any]:
    def refresh() -> dict[str, Any]:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    record: dict[str, Any] = {"schema_version": 1, "clear": True}
    result = window.checked_first_refresh(
        tmp_path, record, read, refresh, requested_at=REQUESTED, now=lambda: ENDED
    )
    assert record["self_check"] == evidence(tmp_path)["self_check"]
    return result


def test_the_live_failure_is_inconclusive_with_the_run_time_and_the_window(
    tmp_path: Path,
) -> None:
    with pytest.raises(window.RefreshTickInterference) as caught:
        first_refresh(tmp_path, live_tick, binding.BindingError(SECRET_CHANGED))
    reason = caught.value.reason
    assert "run scheduled at 2026-11-01T09:17:00+00:00 (lastScheduleTime)" in reason
    assert "[2026-11-01T09:15:14+00:00 rotation requested, " in reason
    assert "2026-11-01T09:17:30+00:00 end of the first refresh]" in reason
    assert caught.value.error == "BindingError: " + SECRET_CHANGED
    check = evidence(tmp_path)["self_check"]
    assert check["inside"] is True and check["reason"] == reason


@pytest.mark.parametrize(
    "document",
    [
        # a late start: its tick lies before the rotation, its completion inside
        cronjob(
            last_schedule=utc("2026-11-01T08:17:00"),
            last_success=utc("2026-11-01T09:16:10"),
        ),
        # a Job still in flight when the self-check reads the CronJob
        cronjob(
            last_schedule=utc("2026-11-01T08:17:00"),
            last_success=utc("2026-11-01T08:17:05"),
            active=["late-run"],
        ),
    ],
)
def test_a_late_or_running_refresher_job_is_a_run_inside_the_window(
    document: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(window.RefreshTickInterference):
        first_refresh(tmp_path, lambda: document, binding.BindingError(REFRESHER_BUSY))


def test_a_binding_error_without_a_refresher_run_inside_the_window_stays_fail(
    tmp_path: Path,
) -> None:
    quiet = cronjob(
        last_schedule=utc("2026-11-01T08:17:00"),
        last_success=utc("2026-11-01T08:17:05"),
    )
    with pytest.raises(binding.BindingError, match="CPU Secret changed"):
        first_refresh(tmp_path, lambda: quiet, binding.BindingError(SECRET_CHANGED))
    assert evidence(tmp_path)["self_check"]["inside"] is False


@pytest.mark.parametrize(
    "error",
    [
        binding.BindingError("Aurora binding: RDS changed during proof"),
        binding.BindingError("Aurora binding: CPU projected Store proof failed"),
        ha009.CaseError("Aurora refresh Job did not succeed"),
    ],
)
def test_every_other_first_refresh_error_keeps_its_verdict(
    error: Exception, tmp_path: Path
) -> None:
    with pytest.raises(type(error)) as caught:
        first_refresh(tmp_path, live_tick, error)
    assert caught.value is error, "an error a refresher run cannot cause propagates"
    assert evidence(tmp_path)["self_check"]["inside"] is True, (
        "the self-check is recorded, but it explains none of these errors"
    )


def test_an_unreadable_self_check_never_hides_the_binding_error(tmp_path: Path) -> None:
    def broken() -> dict[str, Any]:
        raise RuntimeError("kubectl transport lost")

    with pytest.raises(binding.BindingError):
        first_refresh(tmp_path, broken, binding.BindingError(SECRET_CHANGED))
    check = evidence(tmp_path)["self_check"]
    assert check["inside"] is False
    assert check["error"] == "RuntimeError: self-check read failed"


def test_a_successful_first_refresh_records_its_self_check(tmp_path: Path) -> None:
    assert first_refresh(tmp_path, live_tick, {"name": "job-1"}) == {"name": "job-1"}
    assert evidence(tmp_path)["self_check"]["inside"] is True, (
        "recorded for the verdict: a later rotated=False is then explained"
    )


def test_a_first_job_that_found_the_secret_rotated_is_inconclusive_only_alone() -> None:
    inside = window.self_check(live_tick(), REQUESTED, ENDED)
    outside = window.self_check(
        cronjob(last_schedule=utc("2026-11-01T08:17:00")), REQUESTED, ENDED
    )
    explained = frozenset({verdicts.FIRST_REFRESH_NOT_ROTATED})

    def result(*errors: str) -> dict[str, Any]:
        return {"verdict": "FAIL" if errors else "PASS", "errors": list(errors)}

    alone = result(verdicts.FIRST_REFRESH_NOT_ROTATED)
    window.explain_by_refresher(alone, inside, explained)
    assert alone["verdict"] == "INCONCLUSIVE"
    assert alone["inconclusive_reason"] == inside["reason"]
    for value, check in (
        (result(verdicts.FIRST_REFRESH_NOT_ROTATED, "a Pod restarted"), inside),
        (result(verdicts.FIRST_REFRESH_NOT_ROTATED), outside),
        (result(verdicts.FIRST_REFRESH_NOT_ROTATED), None),
        (result(), inside),
    ):
        before = dict(value)
        window.explain_by_refresher(value, check, explained)
        assert value == before, "only a lone explained error changes the verdict"


def test_the_explained_reasons_are_the_binding_proofs_own_messages() -> None:
    environment = FakeAurora()
    expected = environment.guard.read()
    original = environment.control
    reads: list[str] = []

    def rotating(*args: str, **kwargs: Any) -> str:
        value = original(*args, **kwargs)
        if args[:3] == ("get", "secret", SECRET):
            reads.append("secret")
            if len(reads) == 2:
                document = json.loads(value)
                rotated = environment.dsn.replace("REPLACE_WITH", "ROTATED")
                document["data"]["postgres-url"] = encoded(rotated)
                return json.dumps(document)
        return value

    environment.guard.control = rotating
    with pytest.raises(binding.BindingError) as changed:
        environment.guard.refresh_job("unit-job", expected)
    assert str(changed.value) == SECRET_CHANGED
    assert str(changed.value) in window.REFRESHER_BINDING_ERRORS

    environment = FakeAurora()
    expected = environment.guard.read()
    environment.documents["cronjob", CRONJOB]["status"] = {"active": [{"name": "t"}]}
    with pytest.raises(binding.BindingError) as busy:
        environment.guard.refresh_job("unit-job", expected)
    assert str(busy.value) == REFRESHER_BUSY
    assert str(busy.value) in window.REFRESHER_BINDING_ERRORS
