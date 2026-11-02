"""When a HyperPod instance group's scheduled update window is.

A ``ScheduledUpdateConfig.ScheduleExpression`` is a six-field cron
expression, ``cron(Minutes Hours Day-of-month Month Day-of-week Year)``, in
UTC (the live CPU group's ``cron(0 8 ? * SUN *)`` started at 08:00Z on
2026-11-01). The HyperPod developer guide's "Cron expressions" table
(``SCHEDULE_EVIDENCE``) allows no wildcard in Minutes and Hours -- one start
time per matching day -- ``?`` in Day-of-month, ``*`` and ``/`` in Month,
``?``, ``#`` and ``L`` in Day-of-week and ``*`` in Year; this module also
reads lists and ranges. The guide's numeric day-of-week disagrees with itself:
its examples count from Monday (``cron(00 12 ? * 1 *)`` "on every Monday",
``1#2`` the second Monday, ``1L`` the last Monday, ``5L`` "the last Friday"),
while its ``#`` text counts from Sunday (``3#2`` "the second Tuesday", ``3``
"the third day of each week"). So a numeric day matches under either reading:
a window found here is never later than the real one, which is the safe side
for a margin.

A window lasts about ``instances x (re-image + wait)`` with a batch of one
instance; ``REIMAGE_MINUTES`` is a conservative re-image time (the
2026-11-01 all-at-once re-image had the nodes back within about 5 minutes;
a one-at-a-time roll has not been observed). Nothing here reads anything.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

__all__ = [
    "REIMAGE_MINUTES",
    "SCHEDULE_EVIDENCE",
    "UnsupportedSchedule",
    "Window",
    "current_or_next_window",
    "next_start",
    "previous_start",
    "roll_duration",
]

#: The HyperPod developer guide's cron rules (fetched 2026-11-02).
SCHEDULE_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": (
        "https://docs.aws.amazon.com/sagemaker/latest/dg/"
        "sagemaker-hyperpod-release-ami-update.html"
    ),
    "page": "Update your AMI version in your SageMaker HyperPod cluster",
    "section": "Cron expressions",
    "docs_date": "Last-Modified 2026-10-31",
    "quotes": (
        "Cron expressions support six fields and are separated by white space. All "
        "six fields are required.",
        "The following example creates a schedule that runs at 12:00pm on every "
        "Monday regardless of day-of-month. `cron(00 12 ? * 1 *)`",
        "For example, 3#2 would be the second Tuesday of the month: the 3 refers to "
        "Tuesday because it is the third day of each week, and the 2 refers to the "
        "second day of that type within the month.",
        "For example, 5L means the last Friday of the month.",
    ),
}
#: A conservative re-image of one instance (live 2026-11-01: about 5 minutes).
REIMAGE_MINUTES = 10
#: How far a search looks: a yearly or one-time expression still resolves.
_SEARCH_DAYS = 3 * 366
_DAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_MONTHS = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)
_EXPRESSION = re.compile(r"^\s*cron\((?P<fields>[^()]*)\)\s*$")


class UnsupportedSchedule(ValueError):
    """An expression this module cannot place in time."""


@dataclass(frozen=True)
class Window:
    """One scheduled update window, in UTC."""

    start: datetime
    end: datetime

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return start < self.end and self.start < end


DayTest = Callable[[date], bool]


def _number(text: str, low: int, high: int, names: tuple[str, ...] = ()) -> int:
    upper = text.upper()
    if names and upper in names:
        return names.index(upper) + low
    if not text.isdigit():
        raise UnsupportedSchedule(f"{text!r} is not a value")
    value = int(text)
    if not low <= value <= high:
        raise UnsupportedSchedule(f"{value} is outside {low}-{high}")
    return value


def _values(
    field: str, low: int, high: int, names: tuple[str, ...] = ()
) -> frozenset[int]:
    """A list of values, ranges and ``/`` steps (``*`` is every value)."""

    values: set[int] = set()
    for part in field.split(","):
        base, _, step_text = part.partition("/")
        step = _number(step_text, 1, high) if step_text else 1
        if base == "*":
            first, last = low, high
        elif "-" in base:
            first_text, _, last_text = base.partition("-")
            first = _number(first_text, low, high, names)
            last = _number(last_text, low, high, names)
        else:
            first = _number(base, low, high, names)
            last = high if step_text else first
        if first > last:
            raise UnsupportedSchedule(f"{part!r} runs backwards")
        values.update(range(first, last + 1, step))
    return frozenset(values)


def _weekdays(text: str) -> tuple[int, ...]:
    """The ``date.weekday()`` values one day-of-week token can mean: a name is
    exact, a number is read both ways (1 = Monday, and 1 = Sunday)."""

    if text.upper() in _DAYS:
        return (_DAYS.index(text.upper()),)
    value = _number(text, 1, 7)
    return tuple(sorted({(value - 1) % 7, (value - 2) % 7}))


def _day_of_week(field: str) -> DayTest | None:
    if field == "?":
        return None
    if field == "*":
        return lambda day: True
    if "#" in field:
        token, _, nth_text = field.partition("#")
        nth, days = _number(nth_text, 1, 5), _weekdays(token)
        return lambda day: day.weekday() in days and (day.day - 1) // 7 + 1 == nth
    if field.upper().endswith("L"):
        days = _weekdays(field[:-1])
        return lambda day: (
            day.weekday() in days and (day + timedelta(days=7)).month != day.month
        )
    wanted: set[int] = set()
    for part in field.split(","):
        if "-" in part:
            first_text, _, last_text = part.partition("-")
            if not (first_text.upper() in _DAYS and last_text.upper() in _DAYS):
                raise UnsupportedSchedule(f"{part!r}: a numeric day range is ambiguous")
            first, last = (
                _DAYS.index(first_text.upper()),
                _DAYS.index(last_text.upper()),
            )
            if first > last:
                raise UnsupportedSchedule(f"{part!r} runs backwards")
            wanted.update(range(first, last + 1))
        else:
            wanted.update(_weekdays(part))
    return lambda day: day.weekday() in wanted


@dataclass(frozen=True)
class _Spec:
    minute: int
    hour: int
    day: DayTest

    def start_on(self, day: date) -> datetime:
        return datetime(
            day.year, day.month, day.day, self.hour, self.minute, tzinfo=UTC
        )


def _spec(expression: str) -> _Spec:
    match = _EXPRESSION.match(expression or "")
    if match is None:
        raise UnsupportedSchedule(f"{expression!r} is not a cron(...) expression")
    fields = match.group("fields").split()
    if len(fields) != 6:
        raise UnsupportedSchedule(f"{expression!r} has {len(fields)} fields, not 6")
    minute_text, hour_text, dom, month, dow, year = fields
    minute, hour = _number(minute_text, 0, 59), _number(hour_text, 0, 23)
    months = _values(month, 1, 12, _MONTHS)
    years = None if year == "*" else _values(year, 1970, 2199)
    weekday = _day_of_week(dow)
    if (dom == "?") == (weekday is None):
        raise UnsupportedSchedule(
            f"{expression!r}: exactly one of day-of-month and day-of-week must be ?"
        )
    days = None if dom in {"?", "*"} else _values(dom, 1, 31)

    def day_test(day: date) -> bool:
        if day.month not in months or (years is not None and day.year not in years):
            return False
        if weekday is not None:
            return weekday(day)
        return days is None or day.day in days

    return _Spec(minute=minute, hour=hour, day=day_test)


def next_start(expression: str, *, after: datetime) -> datetime | None:
    """The first start at or after ``after``; ``None`` when the expression
    has none left within the search span (a one-time schedule in the past)."""

    spec = _spec(expression)
    day = after.astimezone(UTC).date()
    for offset in range(_SEARCH_DAYS):
        candidate = day + timedelta(days=offset)
        if spec.day(candidate) and spec.start_on(candidate) >= after:
            return spec.start_on(candidate)
    return None


def previous_start(expression: str, *, before: datetime) -> datetime | None:
    """The last start at or before ``before``, or ``None``."""

    spec = _spec(expression)
    day = before.astimezone(UTC).date()
    for offset in range(_SEARCH_DAYS):
        candidate = day - timedelta(days=offset)
        if spec.day(candidate) and spec.start_on(candidate) <= before:
            return spec.start_on(candidate)
    return None


def roll_duration(instances: int, wait_seconds: int) -> timedelta:
    """How long a one-instance-at-a-time window lasts, about."""

    return max(instances, 1) * timedelta(
        minutes=REIMAGE_MINUTES, seconds=max(wait_seconds, 0)
    )


def current_or_next_window(
    expression: str, *, now: datetime, instances: int, wait_seconds: int
) -> Window | None:
    """The window ``now`` is inside, else the next one; ``None`` when there is
    no window left. Raises ``UnsupportedSchedule`` for an expression this
    module cannot place."""

    duration = roll_duration(instances, wait_seconds)
    previous = previous_start(expression, before=now)
    if previous is not None and now < previous + duration:
        return Window(start=previous, end=previous + duration)
    upcoming = next_start(expression, after=now)
    return None if upcoming is None else Window(start=upcoming, end=upcoming + duration)
