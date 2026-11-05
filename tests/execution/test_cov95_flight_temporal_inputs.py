"""``gpu_fault.execution.flight_temporal``: what makes two FR snapshots comparable.

``tests/execution/test_flight_temporal.py`` drives the verdicts end to end;
these tests pin each input guard of the time-series reader on its own: a
process group whose identity cannot be proven, a snapshot that is not a clean
dump, a series whose clock or progress counters cannot be compared. Every one
reads as "no evidence", never as progress or a stall.
"""

from __future__ import annotations

import copy
from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.execution.flight_temporal import (
    comparable_progress,
    group_key,
    membership_digest,
    paired_groups,
    snapshot_groups,
)
from tests.execution._flight_series_support import NOW, entry, temporal_signal

DIGEST = membership_digest([0, 1, 2])


def _signal() -> dict[str, Any]:
    return temporal_signal(0, [entry(99, "completed"), entry(100)])


def _progress(**fields: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "comparison_supported": True,
        "last_enqueued": 10,
        "last_started": 9,
        "last_completed": 8,
        "oldest_pending": {"collective_seq_id": 9, "operation_key": "k"},
    }
    values.update(fields)
    return values


def test_a_group_is_keyed_by_name_membership_digest_and_size() -> None:
    assert group_key({"pg_name": "dp", "pg_members": [2, 0, 1]}) == ("dp", DIGEST, 3)
    assert group_key(
        {"pg_name": "dp", "pg_membership_sha256": DIGEST, "pg_member_count": 3}
    ) == ("dp", DIGEST, 3), "a compact membership names the same group"


@pytest.mark.parametrize(
    "group",
    [
        {"pg_name": "dp", "pg_members": [0, 1, 1]},
        {"pg_name": "dp", "pg_members": [0, 1, 2], "pg_membership_sha256": "0" * 64},
        {"pg_name": "dp", "pg_members": [0, 1, 2], "pg_member_count": 4},
        {"pg_members": [0, 1, 2]},
        {"pg_name": ":", "pg_members": [0, 1, 2]},
        {"pg_name": "dp", "pg_membership_sha256": "abc", "pg_member_count": 3},
        {"pg_name": "dp", "pg_membership_sha256": DIGEST, "pg_member_count": "3"},
        {"pg_name": "dp", "pg_membership_sha256": DIGEST, "pg_member_count": 0},
    ],
    ids=[
        "duplicate-member",
        "digest-contradicts-members",
        "count-contradicts-members",
        "no-name",
        "empty-name",
        "short-digest",
        "count-not-int",
        "count-zero",
    ],
)
def test_a_group_whose_identity_cannot_be_proven_has_no_key(
    group: dict[str, Any],
) -> None:
    assert group_key(group) is None, group


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {"status": "failed", "groups": []},
        {"status": "dumped", "parse_error": "truncated pickle", "groups": []},
        {"status": "dumped", "process_groups_truncated": True, "groups": []},
        {"status": "dumped", "groups": {"dp": {}}},
        {"status": "dumped", "groups": [{"pg_name": "dp"}] * 65},
    ],
    ids=["not-a-mapping", "not-dumped", "parse-error", "truncated", "not-a-list", "65"],
)
def test_a_snapshot_that_is_not_a_clean_dump_has_no_groups(snapshot: Any) -> None:
    assert snapshot_groups(snapshot) == {}


def test_a_snapshot_keeps_only_groups_it_can_identify_once() -> None:
    dp = {"pg_name": "dp", "pg_members": [0, 1, 2], "progress": {"n": 1}}
    tp = {"pg_name": "tp", "pg_members": [0, 1]}
    snapshot = {
        "status": "dumped",
        "groups": ["dp", {"pg_name": "x", "pg_members": [0, 0]}, dp, tp, dict(tp)],
    }
    assert snapshot_groups(snapshot) == {("dp", DIGEST, 3): dp}, (
        "malformed and unidentifiable entries are skipped; a group listed twice "
        "is ambiguous and dropped"
    )


def test_a_clean_pair_yields_one_row_per_common_group() -> None:
    (row,) = paired_groups(_signal())
    assert row["key"] == ("dp", DIGEST, 3)
    assert row["pending_persists"] is True, row
    assert row["completion_advanced"] is False, row


@pytest.mark.parametrize(
    "defect",
    [
        "series-not-a-mapping",
        "identity-not-a-mapping",
        "interval-bool",
        "interval-text",
        "interval-nan",
        "interval-too-short",
        "interval-too-long",
        "naive-timestamp",
        "requested-before-first",
        "observed-before-requested",
        "longer-than-30s",
        "missing-timestamp",
        "unparsable-timestamp",
        "timestamp-not-text",
    ],
)
def test_a_series_whose_timing_cannot_be_trusted_pairs_nothing(defect: str) -> None:
    signal = _signal()
    series = signal["flight_recorder_series"]
    first = NOW
    if defect == "series-not-a-mapping":
        signal["flight_recorder_series"] = ["PAIRED"]
    elif defect == "identity-not-a-mapping":
        signal["process_identity"] = "verified"
    elif defect.startswith("interval-"):
        series["interval_seconds"] = {
            "interval-bool": True,
            "interval-text": "2",
            "interval-nan": float("nan"),
            "interval-too-short": 0.5,
            "interval-too-long": 31,
        }[defect]
    elif defect == "naive-timestamp":
        series["first_observed_at"] = first.replace(tzinfo=None).isoformat()
    elif defect == "requested-before-first":
        series["second_requested_at"] = (first - timedelta(seconds=1)).isoformat()
    elif defect == "observed-before-requested":
        series["second_observed_at"] = (first + timedelta(seconds=1)).isoformat()
    elif defect == "longer-than-30s":
        series["second_observed_at"] = (first + timedelta(seconds=31)).isoformat()
    elif defect == "missing-timestamp":
        del series["second_observed_at"]
    elif defect == "unparsable-timestamp":
        series["first_observed_at"] = "yesterday"
    else:
        series["second_requested_at"] = 1_700_000_000
    assert paired_groups(signal) == [], defect


def test_a_group_whose_progress_is_not_a_mapping_is_skipped() -> None:
    signal = _signal()
    second = signal["flight_recorder_series"]["second"]["groups"][0]
    second["progress"] = ["last_completed", 99]
    assert paired_groups(signal) == []


def test_progress_counters_are_comparable_only_when_both_points_carry_them() -> None:
    assert comparable_progress(_progress(), _progress(last_completed=9)) is True


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ({"comparison_supported": False}, {}),
        ({}, {"comparison_supported": None}),
        ({"_drop": "last_started"}, {}),
        ({}, {"_drop": "last_enqueued"}),
        ({"last_completed": -1}, {}),
        ({}, {"last_started": "9"}),
        ({}, {"last_completed": None}),
        ({}, {"last_completed": 7}),
        ({"oldest_pending": "seq 9"}, {}),
        ({}, {"oldest_pending": {"collective_seq_id": "9"}}),
        ({"oldest_pending": {"collective_seq_id": 11}}, {}),
        ({}, {"oldest_pending": {"collective_seq_id": -1}}),
        (
            {
                "last_enqueued": None,
                "last_started": None,
                "last_completed": None,
                "oldest_pending": None,
            },
            {},
        ),
    ],
    ids=[
        "before-unsupported",
        "after-unsupported",
        "before-missing-field",
        "after-missing-field",
        "negative-counter",
        "non-int-counter",
        "counter-lost",
        "counter-regressed",
        "pending-not-a-mapping",
        "pending-seq-not-int",
        "pending-beyond-enqueued",
        "pending-negative",
        "nothing-enqueued-before",
    ],
)
def test_progress_counters_that_cannot_be_compared_are_not_progress(
    before: dict[str, Any], after: dict[str, Any]
) -> None:
    points = []
    for change in (before, after):
        point = _progress(**{k: v for k, v in change.items() if k != "_drop"})
        if "_drop" in change:
            point.pop(change["_drop"])
        points.append(copy.deepcopy(point))
    assert comparable_progress(points[0], points[1]) is False, points
