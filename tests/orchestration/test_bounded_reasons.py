"""Incident reasons are deduplicated and bounded on every merge (F-J6).

An incident that lived for an hour of correlated faults appended every
event's reasons to one JSON list without bound. The merge sites now go
through one helper that keeps the reasons the incident opened with and the
most recent ones, inside a fixed budget.
"""

from __future__ import annotations

from gpu_fault.models import INCIDENT_REASONS_LIMIT, bounded_reasons


def test_reasons_are_deduplicated_in_order():
    assert bounded_reasons(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_reasons_over_the_limit_keep_the_opening_and_the_latest():
    values = [f"reason-{index}" for index in range(INCIDENT_REASONS_LIMIT + 50)]

    bounded = bounded_reasons(values)

    assert len(bounded) == INCIDENT_REASONS_LIMIT
    head = INCIDENT_REASONS_LIMIT // 2
    assert bounded[:head] == values[:head]
    assert bounded[head:] == values[-(INCIDENT_REASONS_LIMIT - head) :]
    # Appending one more reason to a full list evicts the oldest of the tail,
    # never the opening reasons.
    again = bounded_reasons([*bounded, "reason-new"])
    assert again[:head] == values[:head]
    assert again[-1] == "reason-new"
    assert len(again) == INCIDENT_REASONS_LIMIT


def test_a_custom_limit_is_respected():
    assert bounded_reasons(["a", "b", "c", "d", "e"], limit=3) == ["a", "d", "e"]
