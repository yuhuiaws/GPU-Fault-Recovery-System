from copy import deepcopy

import pytest

from gpu_fault.execution.flight_temporal import membership_digest
from gpu_fault.execution.hung_classification import classify_hung_signals
from gpu_fault.flight_progress import group_progress
from tests.execution._flight_series_support import entry, temporal_signal


def classify(rows):
    return classify_hung_signals(rows, undetermined_nodes=[])


def lagging():
    return [
        temporal_signal(0, [entry(99, "completed"), entry(100)]),
        temporal_signal(1, [entry(100, "completed")]),
        temporal_signal(2, [entry(100, "completed")]),
    ]


def test_same_enqueue_sequence_can_hide_persistent_completion_lag():
    result = classify(lagging())
    assert result["culprit_ranks"] == [0], result
    assert result["targeted_capture_eligible"] is True, result
    assert result["root_cause_confirmed"] is False, result
    [group] = result["process_groups"]
    assert group["candidate_reasons"]["0"] == ["PERSISTENT_COMPLETION_LAG"], group
    assert group["pending_observed_seconds_by_rank"] == {"0": 2}, group


def test_enqueues_advancing_does_not_conceal_an_older_unfinished_collective():
    rows = lagging()
    rows[0] = temporal_signal(
        0,
        [entry(99, "completed"), entry(100), entry(101, "scheduled")],
        [
            entry(99, "completed"),
            entry(100),
            entry(101, "scheduled"),
            entry(102, "scheduled"),
        ],
    )
    result = classify(rows)
    assert result["culprit_ranks"] == [0], result
    assert result["process_groups"][0]["completion_advanced_ranks"] == [], result


def test_observed_completion_progress_is_not_a_persistent_lag():
    rows = lagging()
    rows[0] = temporal_signal(
        0, [entry(99, "completed"), entry(100)], [entry(100, "completed")]
    )
    result = classify(rows)
    assert result["culprit_ranks"] == [], result
    assert result["process_groups"][0]["completion_advanced_ranks"] == [0], result
    assert result["targeted_capture_eligible"] is False, result


def test_every_member_stalled_is_reported_without_inventing_a_culprit():
    result = classify([temporal_signal(rank, [entry(100)]) for rank in range(3)])
    assert result["stalled_process_groups"] == ["dp"], result
    assert result["culprit_ranks"] == [], result
    assert result["classification"] == "PLAUSIBLE", result
    assert result["targeted_capture_eligible"] is False, result


@pytest.mark.parametrize("kind", ["enqueue", "start"])
def test_repeated_enqueue_and_start_lags_remain_diagnostic_leads(kind):
    rows = [temporal_signal(rank, [entry(100)]) for rank in range(3)]
    rows[0] = temporal_signal(
        0, [entry(99, "completed")] if kind == "enqueue" else [entry(100, "scheduled")]
    )
    result = classify(rows)
    assert result["culprit_ranks"] == [0], result
    assert result["process_groups"][0]["candidate_reasons"]["0"] == [
        f"PERSISTENT_{kind.upper()}_LAG"
    ], result


@pytest.mark.parametrize(
    "defect",
    [
        "missing-second",
        "pid-reused",
        "membership-changed",
        "counter-regressed",
        "history-evicted",
        "unknown-operation-identity",
        "nonoverlapping-times",
        "p2p",
    ],
)
def test_incomplete_or_incomparable_time_series_cannot_authorize_attachment(defect):
    rows = lagging()
    if defect == "nonoverlapping-times":
        rows[0] = temporal_signal(0, [entry(99, "completed"), entry(100)], offset=5)
    else:
        series = rows[0]["flight_recorder_series"]
        group = series["second"]["groups"][0]
        if defect == "missing-second":
            series["status"] = "SECOND_UNAVAILABLE"
        elif defect == "pid-reused":
            rows[0]["process_identity"]["start_time_ticks"] = 9999
        elif defect == "membership-changed":
            group["pg_members"] = [0, 1, 2, 3]
        elif defect == "counter-regressed":
            group["progress"]["last_completed"] = 98
        elif defect == "history-evicted":
            group["progress"]["oldest_pending"] = None
        elif defect == "unknown-operation-identity":
            group["progress"]["oldest_pending"]["operation_key"] = None
        else:
            group["progress"]["comparison_supported"] = False
    result = classify(rows)
    assert result["targeted_capture_eligible"] is False, result
    assert result["root_cause_confirmed"] is False, result


def test_missing_completed_watermark_is_not_fabricated_as_zero():
    rows = lagging()
    rows[0] = temporal_signal(0, [entry(100)])
    result = classify(rows)
    assert result["culprit_ranks"] == [0], result
    assert (
        rows[0]["flight_recorder_series"]["first"]["groups"][0]["progress"][
            "last_completed"
        ]
        is None
    )


def test_progress_in_one_group_does_not_hide_another_group_stall():
    rows = lagging()
    for rank, row in enumerate(rows):
        other = temporal_signal(
            rank, [entry(200)], [entry(200, "completed")], name="other"
        )
        for point in ("first", "second"):
            row["flight_recorder_series"][point]["groups"].extend(
                deepcopy(other["flight_recorder_series"][point]["groups"])
            )
    result = classify(rows)
    assert result["culprit_ranks"] == [0], result
    assert len(result["process_groups"]) == 2, result


def test_single_snapshot_is_a_lead_but_no_longer_authorizes_targeted_capture():
    rows = [
        temporal_signal(rank, [entry(99 if rank == 0 else 100)]) for rank in range(3)
    ]
    for row in rows:
        row.pop("flight_recorder_series")
    result = classify(rows)
    assert result["culprit_ranks"] == [0], result
    assert result["targeted_capture_eligible"] is False, result


def test_1024_member_group_uses_compact_membership_and_preserves_completion_lag():
    members = list(range(1024))
    digest = membership_digest(members)
    rows = [
        temporal_signal(
            rank,
            [entry(99, "completed"), entry(100)]
            if rank == 0
            else [entry(100, "completed")],
        )
        for rank in members
    ]
    for row in rows:
        for point in ("first", "second"):
            group = row["flight_recorder_series"][point]["groups"][0]
            group.pop("pg_members")
            group.update(pg_member_count=1024, pg_membership_sha256=digest)
    result = classify(rows)
    assert result["temporal_rank_count"] == 1024, result
    assert result["culprit_ranks"] == [0], result
    assert result["process_groups"][0]["membership_verified"] is True, result
    assert result["targeted_capture_eligible"] is True, result


@pytest.mark.parametrize(
    "timestamp", [None, -1, float("nan"), float("inf"), "unknown", True]
)
def test_invalid_completion_timestamp_is_not_completion(timestamp):
    progress = group_progress(
        [entry(100, "started", time_discovered_completed=timestamp)]
    )
    assert progress["last_completed"] is None, progress
    assert progress["oldest_pending"]["collective_seq_id"] == 100, progress
