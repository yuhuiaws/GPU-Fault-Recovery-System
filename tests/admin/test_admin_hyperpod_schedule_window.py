"""When a HyperPod scheduled update window is (batch 240).

The deploy defers its CPU schedule step within three hours of the group's next
window and its preflight warns when a deploy may overlap one, so the window
must never be found later than the real one: the live expression, both
readings of a numeric day-of-week (the HyperPod guide's own examples
disagree), ``#`` and ``L``, and every expression it cannot place refused, not
guessed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.admin.hyperpod_schedule_window import (
    REIMAGE_MINUTES,
    SCHEDULE_EVIDENCE,
    UnsupportedSchedule,
    Window,
    current_or_next_window,
    next_start,
    previous_start,
    roll_duration,
)

LIVE = "cron(0 8 ? * SUN *)"
NOW = datetime(2026, 11, 2, 2, 0, tzinfo=UTC)


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


def test_the_live_schedule_is_sunday_0800_utc() -> None:
    assert previous_start(LIVE, before=NOW) == _utc(2026, 11, 1, 8, 0), (
        "the all-at-once re-image of 2026-11-01 started at 08:00Z"
    )
    assert next_start(LIVE, after=NOW) == _utc(2026, 11, 8, 8, 0)
    assert next_start(LIVE, after=_utc(2026, 11, 8, 8, 0)) == _utc(2026, 11, 8, 8, 0), (
        "a start exactly now is the next one"
    )


def test_a_one_at_a_time_roll_of_the_live_group_lasts_two_hours() -> None:
    # Three instances, a 1800 s wait: 3 x (10 min re-image + 30 min wait).
    assert roll_duration(3, 1800) == timedelta(hours=2)
    assert roll_duration(0, -1) == timedelta(minutes=REIMAGE_MINUTES), (
        "never shorter than one re-image"
    )

    inside = current_or_next_window(
        LIVE, now=_utc(2026, 11, 1, 9, 30), instances=3, wait_seconds=1800
    )
    assert inside == Window(_utc(2026, 11, 1, 8, 0), _utc(2026, 11, 1, 10, 0)), (
        "inside a window, that window"
    )
    after = current_or_next_window(
        LIVE, now=_utc(2026, 11, 1, 10, 0), instances=3, wait_seconds=1800
    )
    assert after == Window(_utc(2026, 11, 8, 8, 0), _utc(2026, 11, 8, 10, 0))


def test_overlap_is_half_open() -> None:
    window = Window(_utc(2026, 11, 8, 8, 0), _utc(2026, 11, 8, 10, 0))

    assert window.overlaps(_utc(2026, 11, 8, 7, 0), _utc(2026, 11, 8, 8, 1)), "start"
    assert window.overlaps(_utc(2026, 11, 8, 9, 59), _utc(2026, 11, 8, 12, 0)), "end"
    assert not window.overlaps(_utc(2026, 11, 8, 6, 0), _utc(2026, 11, 8, 8, 0)), (
        "a deploy that ends as the window starts does not overlap it"
    )
    assert not window.overlaps(_utc(2026, 11, 8, 10, 0), _utc(2026, 11, 8, 11, 0)), (
        "nor one that starts as it ends"
    )


def test_a_numeric_day_of_week_matches_under_both_readings() -> None:
    # The guide calls cron(00 12 ? * 1 *) Monday, and 3 Tuesday in its # text:
    # 1 = Monday and 1 = Sunday are both read, so the earlier day wins.
    saturday = _utc(2026, 11, 7, 13, 0)
    assert next_start("cron(00 12 ? * 1 *)", after=saturday) == _utc(2026, 11, 8, 12, 0)
    assert next_start("cron(00 12 ? * 1 *)", after=_utc(2026, 11, 8, 13, 0)) == (
        _utc(2026, 11, 9, 12, 0)
    ), "and the Monday after it"
    # 3#2: the second Tuesday (10 Nov 2026) or the second Wednesday (11 Nov).
    assert next_start("cron(0 3 ? * 3#2 *)", after=NOW) == _utc(2026, 11, 10, 3, 0)
    # 1L: the last Sunday (29 Nov) or the last Monday (30 Nov).
    assert next_start("cron(0 8 ? * 1L *)", after=NOW) == _utc(2026, 11, 29, 8, 0)


@pytest.mark.parametrize(
    "expression,start",
    [
        ("cron(30 4 ? * MON-FRI *)", _utc(2026, 11, 2, 4, 30)),
        ("cron(0 8 ? * SAT,SUN *)", _utc(2026, 11, 7, 8, 0)),
        ("cron(0 8 15 * ? *)", _utc(2026, 11, 15, 8, 0)),
        ("cron(0 8 ? JAN,JUL SUN *)", _utc(2027, 1, 3, 8, 0)),
        ("cron(0 8 ? */6 SUN *)", _utc(2027, 1, 3, 8, 0)),
        ("cron(0 8 ? * SUN 2027)", _utc(2027, 1, 3, 8, 0)),
    ],
)
def test_names_lists_ranges_steps_and_years(expression: str, start: datetime) -> None:
    assert next_start(expression, after=NOW) == start


def test_a_schedule_with_no_start_left_has_no_window() -> None:
    expression = "cron(0 8 ? * SUN 2025)"

    assert next_start(expression, after=NOW) is None
    assert (
        current_or_next_window(expression, now=NOW, instances=3, wait_seconds=1800)
        is None
    )


@pytest.mark.parametrize(
    "expression,reason",
    [
        ("rate(7 days)", "is not a cron"),
        ("cron(0 8 ? * SUN)", "has 5 fields, not 6"),
        ("cron(* 8 ? * SUN *)", "is not a value"),
        ("cron(0 */2 ? * SUN *)", "is not a value"),
        ("cron(0 8 ? * ? *)", "exactly one of day-of-month and day-of-week"),
        ("cron(0 8 1 * SUN *)", "exactly one of day-of-month and day-of-week"),
        ("cron(0 8 ? * 2-6 *)", "a numeric day range is ambiguous"),
        ("cron(0 24 ? * SUN *)", "outside 0-23"),
        ("cron(0 8 ? * FRI-MON *)", "runs backwards"),
    ],
)
def test_an_expression_it_cannot_place_is_refused(expression: str, reason: str) -> None:
    with pytest.raises(UnsupportedSchedule, match=reason):
        current_or_next_window(expression, now=NOW, instances=3, wait_seconds=1800)


def test_the_cron_rules_are_cited() -> None:
    assert SCHEDULE_EVIDENCE["url"] == (
        "https://docs.aws.amazon.com/sagemaker/latest/dg/"
        "sagemaker-hyperpod-release-ami-update.html"
    )
    assert SCHEDULE_EVIDENCE["section"] == "Cron expressions"
    assert any(
        "All six fields are required" in quote for quote in SCHEDULE_EVIDENCE["quotes"]
    ), SCHEDULE_EVIDENCE["quotes"]
