from copy import deepcopy

import pytest

from gpu_fault.execution.hung_classification import classify_hung_signals
from tests.execution._cov95_runtime_triage import signal
from tests.execution._flight_series_support import add_series
from tests.execution._process_evidence_support import process_signal


@pytest.mark.parametrize(
    "status",
    ["not_configured", "dump_missing", "node_unavailable", "pipe_write_failed"],
)
def test_missing_flight_recorder_is_not_a_culprit(status):
    rows = [signal(100 + rank, rank) for rank in range(3)]
    rows[0]["flight_recorder"] = {"status": status}
    decision = classify_hung_signals(rows, undetermined_nodes=[])
    assert decision["culprit_ranks"] == [], decision
    assert decision["classification"] == "UNDETERMINED", decision
    assert decision["targeted_capture_eligible"] is False


@pytest.mark.parametrize("sequences", [[9, 10], [9, 9, 10, 10], [10, 10, 10]])
def test_small_tied_or_symmetric_sample_cannot_confirm_a_rank(sequences):
    decision = classify_hung_signals(
        [signal(100 + rank, rank, seq) for rank, seq in enumerate(sequences)],
        undetermined_nodes=[],
    )
    assert decision["culprit_ranks"] == [], decision
    assert decision["root_cause_confirmed"] is False


def test_all_process_groups_are_compared_with_their_own_members():
    rows = [signal(100 + rank, rank) for rank in range(6)]
    for rank, row in enumerate(rows):
        entry = deepcopy(row["flight_recorder"]["last_entry"])
        entry.update(
            pg_name="reused-local-name",
            pg_members=[0, 1, 2] if rank < 3 else [3, 4, 5],
            collective_seq_id=(9 if rank == 0 else 10)
            if rank < 3
            else (899 if rank == 3 else 900),
        )
        # An unrelated high-sequence group must not hide this group's outlier.
        row["flight_recorder"]["last_entries_by_pg"] = [
            entry,
            {
                "pg_name": f"singleton-{rank}",
                "pg_members": [rank],
                "collective_seq_id": 9999,
                "state": "completed",
            },
        ]
        add_series(row)
    decision = classify_hung_signals(rows, undetermined_nodes=[])
    assert decision["culprit_ranks"] == [0, 3], decision
    assert (
        len([group for group in decision["process_groups"] if group["culprit_ranks"]])
        == 2
    ), decision
    assert decision["classification"] == "PLAUSIBLE", decision
    assert decision["targeted_capture_eligible"] is True
    assert decision["root_cause_confirmed"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "missing-members",
        "missing-peer",
        "duplicate-rank",
        "unknown-group",
        "parse-error",
    ],
)
def test_incomplete_identity_never_authorizes_targeted_capture(defect):
    rows = [signal(100 + rank, rank, 9 if rank == 0 else 10) for rank in range(3)]
    if defect == "duplicate-rank":
        rows.append(deepcopy(rows[0]))
    for row in rows:
        entry = row["flight_recorder"]["last_entry"]
        if defect == "missing-members":
            entry.pop("pg_members")
        elif defect == "missing-peer":
            entry["pg_members"].append(3)
        elif defect == "unknown-group":
            entry["pg_name"] = ":"
        elif defect == "parse-error":
            row["flight_recorder"]["parse_error"] = "bad dump"
    decision = classify_hung_signals(rows, undetermined_nodes=[])
    assert decision["targeted_capture_eligible"] is False, decision


def test_weak_candidates_are_not_truncated_to_look_precise():
    rows = [
        process_signal(
            rank,
            rank=rank,
            state="D" if rank < 4 else "R",
            wchan="io_schedule" if rank < 4 else "0",
        )
        for rank in range(20)
    ]
    decision = classify_hung_signals(rows, undetermined_nodes=[])
    assert decision["candidate_process_count"] == 4, decision
    assert [row["rank"] for row in decision["candidate_processes"]] == [0, 1, 2, 3], (
        decision
    )
    assert decision["targeted_capture_eligible"] is False


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1])
def test_missing_or_invalid_behavior_is_not_zero_utilization(value):
    rows = [
        {
            "rank": rank,
            "proc": {"cpu_ticks_delta": 100},
            "gpu": {"utilization_gpu_percent": 80},
        }
        for rank in range(5)
    ]
    rows[0].update(proc={"cpu_ticks_delta": value}, gpu={})
    decision = classify_hung_signals(rows, undetermined_nodes=[])
    assert decision["culprit_ranks"] == [], decision
