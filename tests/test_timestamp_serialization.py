"""Every ``StrictModel`` datetime serializes with fixed six-digit microseconds.

Store review 2026-09-07, item E. The Postgres store compares and orders
payload timestamps as TEXT on purpose (expression indexes on
``payload->>'created_at'``, ``'not_before'``, ``GREATEST(...)``), and pydantic's
default renders ``...:00Z`` for a whole second next to ``...:00.500000Z`` for
the rest, so ``'Z'`` (0x5A) sorted a whole second after every fraction of the
same second (``'.'`` is 0x2E). These tests pin the one rendering both the
models and ``utc_text`` now share, and that text order is time order.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest
from pydantic import BaseModel

from gpu_fault.models import (
    StrictModel,
    WorkflowOperation,
    WorkflowRequest,
    datetime_json_text,
)
from gpu_fault.store.shared.time import utc_text
from tests._builders import workflow_request, workflow_step

UTC = timezone.utc
SECOND = datetime(2026, 9, 7, 6, 0, 0, tzinfo=UTC)
PLUS_TWO = timezone(timedelta(hours=2))

# Whole seconds interleaved with fractions of the same seconds, out of order.
SAMPLE = [
    SECOND + timedelta(microseconds=500_000),
    SECOND,
    SECOND + timedelta(seconds=1),
    SECOND + timedelta(microseconds=1),
    SECOND + timedelta(seconds=1, microseconds=999_999),
    SECOND - timedelta(microseconds=1),
    SECOND + timedelta(seconds=59, microseconds=250_000),
    SECOND + timedelta(minutes=1),
]


class Inner(StrictModel):
    at: datetime


class Sample(StrictModel):
    t: datetime
    optional: Optional[datetime] = None
    union_none: datetime | None = None
    items: list[datetime] = []
    by_key: dict[str, datetime] = {}
    pair: tuple[datetime, int] | None = None
    text_or_time: datetime | str = "text"
    inner: Inner | None = None
    inners: list[Inner] = []


class Node(StrictModel):
    """Recursive: the schema handler returns a reference, not the body."""

    at: datetime
    children: list[Node] = []


def _json(model: BaseModel) -> dict[str, Any]:
    dumped: dict[str, Any] = json.loads(model.model_dump_json())
    return dumped


def test_whole_second_renders_with_six_zero_microseconds() -> None:
    assert datetime_json_text(SECOND) == "2026-09-07T06:00:00.000000Z"
    assert _json(Sample(t=SECOND))["t"] == "2026-09-07T06:00:00.000000Z"
    assert (
        _json(Sample(t=SECOND + timedelta(microseconds=500_000)))["t"]
        == "2026-09-07T06:00:00.500000Z"
    )


def test_pydantic_default_was_not_order_preserving() -> None:
    """The premise of the fix: the stock rendering breaks text order at a
    whole second. If pydantic ever fixes this, the walk becomes redundant."""

    class Stock(BaseModel):
        t: datetime

    stock = [json.loads(Stock(t=value).model_dump_json())["t"] for value in SAMPLE]
    by_time = [
        json.loads(Stock(t=value).model_dump_json())["t"] for value in sorted(SAMPLE)
    ]
    assert stock[1] == "2026-09-07T06:00:00Z"
    assert sorted(stock) != by_time


def test_text_order_is_time_order_across_whole_seconds() -> None:
    texts = [_json(Sample(t=value))["t"] for value in SAMPLE]
    assert sorted(texts) == [_json(Sample(t=value))["t"] for value in sorted(SAMPLE)]
    assert sorted(utc_text(value) for value in SAMPLE) == [
        utc_text(value) for value in sorted(SAMPLE)
    ]


@pytest.mark.parametrize("value", SAMPLE + [SECOND.astimezone(PLUS_TWO)])
def test_utc_text_matches_the_model_rendering(value: datetime) -> None:
    assert utc_text(value) == _json(Sample(t=value))["t"]


def test_non_utc_offsets_are_normalized_to_utc() -> None:
    local = datetime(2026, 9, 7, 8, 0, 0, 250_000, tzinfo=PLUS_TWO)
    assert _json(Sample(t=local))["t"] == "2026-09-07T06:00:00.250000Z"


def test_naive_stays_offset_less_and_round_trips_naive() -> None:
    naive = datetime(2026, 9, 7, 6, 0, 0)
    text = _json(Sample(t=naive))["t"]
    assert text == "2026-09-07T06:00:00.000000"
    parsed = Sample.model_validate_json(Sample(t=naive).model_dump_json())
    assert parsed.t == naive
    assert parsed.t.tzinfo is None
    # utc_text takes a naive value as UTC, as it always did.
    assert utc_text(naive) == "2026-09-07T06:00:00.000000Z"


def test_optional_none_stays_null() -> None:
    dumped = _json(Sample(t=SECOND))
    assert dumped["optional"] is None
    assert dumped["union_none"] is None
    assert dumped["pair"] is None
    assert dumped["inner"] is None


def test_every_container_and_nested_model_uses_the_rendering() -> None:
    fraction = SECOND + timedelta(microseconds=5)
    model = Sample(
        t=SECOND,
        optional=SECOND,
        union_none=fraction,
        items=[SECOND, fraction],
        by_key={"k": SECOND},
        pair=(SECOND, 1),
        text_or_time=SECOND,
        inner=Inner(at=SECOND),
        inners=[Inner(at=fraction)],
    )
    whole = "2026-09-07T06:00:00.000000Z"
    micro = "2026-09-07T06:00:00.000005Z"
    dumped = _json(model)
    assert dumped == {
        "t": whole,
        "optional": whole,
        "union_none": micro,
        "items": [whole, micro],
        "by_key": {"k": whole},
        "pair": [whole, 1],
        "text_or_time": whole,
        "inner": {"at": whole},
        "inners": [{"at": micro}],
    }
    assert model.model_dump(mode="json") == dumped
    assert Sample.model_validate_json(model.model_dump_json()) == model


def test_recursive_model_is_patched_through_the_reference() -> None:
    tree = Node(at=SECOND, children=[Node(at=SECOND, children=[Node(at=SECOND)])])
    dumped = _json(tree)
    whole = "2026-09-07T06:00:00.000000Z"
    assert dumped["at"] == whole
    assert dumped["children"][0]["at"] == whole
    assert dumped["children"][0]["children"][0]["at"] == whole
    assert Node.model_validate_json(tree.model_dump_json()) == tree


def test_python_mode_dump_is_unchanged() -> None:
    model = Sample(t=SECOND, items=[SECOND], inner=Inner(at=SECOND))
    dumped = model.model_dump()
    assert dumped["t"] is SECOND
    assert dumped["items"] == [SECOND]
    assert isinstance(dumped["inner"]["at"], datetime), (
        "python-mode dump must keep datetime objects"
    )


def test_workflow_request_round_trips_through_the_store_encoding() -> None:
    workflow = workflow_request(
        "wf-a",
        "inc-a",
        official_steps=[workflow_step(WorkflowOperation.FREEZE_EVIDENCE)],
        created_at=SECOND,
        updated_at=SECOND + timedelta(microseconds=1),
        not_before=SECOND,
    )
    encoded = workflow.model_dump_json()
    payload = json.loads(encoded)
    assert payload["created_at"] == "2026-09-07T06:00:00.000000Z"
    assert payload["updated_at"] == "2026-09-07T06:00:00.000001Z"
    assert payload["not_before"] == utc_text(SECOND)
    assert payload["created_at"] < payload["updated_at"]
    assert WorkflowRequest.model_validate_json(encoded) == workflow


def test_legacy_text_still_parses() -> None:
    """Rows written before the change carry the stock text; readers accept it."""

    payload = json.loads(Sample(t=SECOND).model_dump_json())
    payload["t"] = "2026-09-07T06:00:00Z"
    assert Sample.model_validate(payload).t == SECOND
