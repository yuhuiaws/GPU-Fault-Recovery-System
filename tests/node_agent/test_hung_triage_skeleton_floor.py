"""The ladder has a floor: a triage result keeps every rank's identity and verdict.

Batch 186 trims an over-budget COLLECT_HUNG_TRIAGE result stepwise (lossless
pointers first, then unfinished entries, the operation lists to 32/16/8, the
lists, the last stack and process sample). A document still over
``TRIAGE_RESULT_BUDGET_BYTES`` after that whole ladder was returned as it was,
and the worker replaced it by the ``RESULT_SIZE_LIMIT`` stand-in with ``ranks:
[]`` -- every rank identity, status, progress watermark and raw-dump reference
lost in that last resort. The operator's requirement (2026-10-26): important
fields stay exposed; never drop everything and just report an error.

The ``skeleton`` step below is that floor. Every rank row is reduced to its
identity and verdict-bearing fields (pid, rank, process identity, status/error,
progress watermarks, recorder status with dump path/size and the ``raw_evidence``
reference, the series' status and counts) and the node-level fields stay whole.
Should even the skeleton exceed the worker's line, the stand-in now names the
ranks it replaced (``ranks_summary``: pid, rank, status for as many as fit in
16 KiB). The live shape is the one ``test_hung_triage_result_budget`` builds
with the product's own reader.
"""

from __future__ import annotations

import copy
from typing import Any

from gpu_fault.execution.hung_classification import classify_hung_signals
from gpu_fault.node_agent import observation_worker
from gpu_fault.node_agent.observation_limits import (
    OBSERVATION_RESULT_BUDGET_BYTES,
    RESULT_STAND_IN_RANKS_BUDGET_BYTES,
    WORKER_OUTPUT,
)
from gpu_fault.node_agent.operations.triage_budget import (
    SKELETON_STEP,
    TRIAGE_RESULT_BUDGET_BYTES,
    bounded_triage_result,
    encoded_size,
    skeleton,
)
from tests.node_agent.test_hung_triage_result_budget import (
    live_shaped_rank,
    live_shaped_result,
    published_line,
)

RANKS = 64


def verdict_branches(ranks: list[dict[str, Any]]) -> dict[str, Any]:
    verdict = classify_hung_signals(ranks, undetermined_nodes=[])
    return {
        "culprit_ranks": verdict["culprit_ranks"],
        "fr_temporal": verdict["evidence_branches"]["fr_temporal"],
        "fr_snapshot": verdict["evidence_branches"]["fr_snapshot"],
        "flight_recorder_missing_count": verdict["flight_recorder_missing_count"],
        "observed_rank_count": verdict["observed_rank_count"],
    }


def test_sixty_four_paired_ranks_fit_under_the_budget_at_the_skeleton_floor():
    """Eight times the live HyperPod node: 64 ranks, two 64-operation dumps each."""

    result = live_shaped_result(RANKS)
    full = copy.deepcopy(result)
    before = encoded_size(result)
    assert before > 8 * TRIAGE_RESULT_BUDGET_BYTES, before

    bounded = bounded_triage_result(result)

    note = bounded["evidence_trimmed"]
    assert note["within_budget"] is True, note
    assert note["bytes_after"] <= TRIAGE_RESULT_BUDGET_BYTES, note
    assert note["bytes_after"] == encoded_size(bounded), note
    assert note["bytes_before"] == before, note
    names = [step["step"] for step in note["steps"]]
    assert names[-1] == SKELETON_STEP == "skeleton", names
    assert names[:-1] == [
        "flight_recorder.semantic_entries_in_series",
        "semantic_entries:32",
        "semantic_entries:16",
        "semantic_entries:8",
        "semantic_entries",
        "stack_series.samples:last",
        "process_series.samples:last",
    ], "the floor comes after the whole ladder"
    floor = note["steps"][-1]
    assert floor["ranks"] == RANKS, floor
    assert floor["bytes"] <= TRIAGE_RESULT_BUDGET_BYTES, floor
    assert floor["bytes"] < note["steps"][-2]["bytes"], note["steps"][-2:]
    for key in full:
        if key != "ranks":
            assert bounded[key] == full[key], f"node-level field {key} is untouched"
    assert len(bounded["ranks"]) == RANKS


def test_a_skeleton_row_keeps_identity_status_watermarks_and_references() -> None:
    full = live_shaped_rank(5)
    row = copy.deepcopy(full)

    assert skeleton([row]) == 1

    assert row["skeleton"] is True
    for key in (
        "node_id",
        "pid",
        "rank",
        "local_rank",
        "gpu_uuid",
        "gpu_uuids",
        "process_name",
        "process_identity",
    ):
        assert row[key] == full[key], key
    assert row["comparison_context"] == {"verified": False}, (
        "scalars of the comparison context stay, the per-thread role maps go"
    )
    recorder, full_recorder = row["flight_recorder"], full["flight_recorder"]
    for key in (
        "status",
        "pipe",
        "dump_file",
        "dump_size_bytes",
        "dump_source",
        "entry_count",
        "unfinished_entry_count",
        "process_groups_truncated",
        "raw_evidence",
    ):
        assert recorder[key] == full_recorder[key], key
    assert recorder["unfinished_entries"] == []
    assert "unfinished_entries_trimmed" not in recorder, (
        "nothing was dropped: the live shape's only unfinished entry is its last"
    )
    [group], [full_group] = (
        recorder["last_entries_by_pg"],
        full_recorder["last_entries_by_pg"],
    )
    for key in (
        "pg_name",
        "pg_members",
        "pg_rank_order",
        "pg_member_count",
        "pg_membership_sha256",
        "collective_seq_id",
        "state",
        "profiling_name",
        "time_discovered_started",
        "progress",
    ):
        assert group[key] == full_group[key], key
    assert "semantic_entries" not in group
    assert group["semantic_entries_total"] == 64
    assert group["semantic_entries_trimmed"] is True
    for heavy in ("input_dims", "output_dims", "sizes", "input_dtypes"):
        assert heavy not in group, heavy
    assert recorder["last_entry"]["collective_seq_id"] == 64
    assert "input_dims" not in recorder["last_entry"]
    series, full_series = row["flight_recorder_series"], full["flight_recorder_series"]
    for key in (
        "version",
        "status",
        "first_observed_at",
        "process_start_ticks",
        "clock_uncertainty_seconds",
        "second_requested_at",
        "interval_seconds",
        "second_observed_at",
    ):
        assert series[key] == full_series[key], key
    for which in ("first", "second"):
        snapshot, full_snapshot = series[which], full_series[which]
        assert snapshot["status"] == "dumped"
        assert snapshot["raw_evidence"] == full_snapshot["raw_evidence"]
        [group], [full_group] = snapshot["groups"], full_snapshot["groups"]
        assert group["progress"] == full_group["progress"], which
        assert group["pg_membership_sha256"] == full_group["pg_membership_sha256"]
        assert "semantic_entries" not in group
        assert group["semantic_entries_total"] == 64
    assert row["process_series"] == {
        "version": "process-series/v1",
        "requested_samples": 3,
        "samples": [],
        "samples_trimmed": 3,
        "windows": [],
        "windows_trimmed": 2,
    }
    assert row["stack_series"] == {
        "version": "stack-series/v1",
        "requested_samples": 3,
        "samples": [],
        "samples_trimmed": 3,
    }
    assert row["gpu_series"] == {
        "scope": "DEVICE_NOT_PID",
        "mapping_verified": True,
        "shared_with_other_processes": False,
        "samples": [],
        "samples_trimmed": 3,
    }
    assert row["python_stack"] == {"status": "COLLECTED", "returncode": 0}
    assert row["proc"] == {"status": "COLLECTED"}
    assert "gpu" not in row
    assert encoded_size(row) <= 6 * 1024, encoded_size(row)
    assert TRIAGE_RESULT_BUDGET_BYTES // encoded_size(row) >= 128, (
        "at least 128 skeleton ranks of the live shape fit under the budget"
    )
    assert skeleton([row]) == 0, "the floor is idempotent"


def test_the_skeleton_keeps_the_recorder_verdict_of_the_full_rows() -> None:
    result = live_shaped_result(RANKS)
    full = copy.deepcopy(result)

    bounded = bounded_triage_result(result)

    assert [step["step"] for step in bounded["evidence_trimmed"]["steps"]][-1] == (
        SKELETON_STEP
    )
    skeleton_verdict = verdict_branches(bounded["ranks"])
    full_verdict = verdict_branches(full["ranks"])
    assert skeleton_verdict == full_verdict, (
        "the temporal and snapshot recorder branches read watermarks, not lists"
    )
    assert skeleton_verdict["observed_rank_count"] == RANKS
    assert skeleton_verdict["flight_recorder_missing_count"] == 0
    assert full_verdict["fr_temporal"] is not None, (
        "the paired series still yields the temporal verdict"
    )


def test_eight_and_sixteen_ranks_never_reach_the_floor() -> None:
    for ranks in (8, 16):
        note = bounded_triage_result(live_shaped_result(ranks))["evidence_trimmed"]
        assert note["within_budget"] is True, note
        assert SKELETON_STEP not in [step["step"] for step in note["steps"]], (
            ranks,
            note,
        )


def test_the_floor_is_applied_and_named_even_when_the_document_is_still_over():
    result = live_shaped_result(8)
    full = copy.deepcopy(result)
    budget = 20_000

    bounded = bounded_triage_result(result, budget=budget)

    note = bounded["evidence_trimmed"]
    assert note["within_budget"] is False, note
    assert note["steps"][-1]["step"] == SKELETON_STEP, note
    assert note["steps"][-1]["ranks"] == 8, note
    assert note["bytes_after"] == encoded_size(bounded)
    assert [row["pid"] for row in bounded["ranks"]] == [
        row["pid"] for row in full["ranks"]
    ]
    for row in bounded["ranks"]:
        assert row["skeleton"] is True
        assert row["flight_recorder_series"]["status"] == "PAIRED"
        assert row["flight_recorder"]["raw_evidence"]["status"] == "PRESERVED"


def test_the_worker_publishes_a_skeleton_result_whole() -> None:
    document = bounded_triage_result(live_shaped_result(RANKS))
    assert document["evidence_trimmed"]["steps"][-1]["step"] == SKELETON_STEP

    message = published_line(document)

    assert message == {"final": True, "result": document}


def skeleton_rows(count: int) -> list[dict[str, Any]]:
    """``count`` distinct skeleton rows of the live shape, cheaply."""

    [template] = rows = [live_shaped_rank(0)]
    skeleton(rows)
    return [
        {**copy.deepcopy(template), "pid": 3000 + rank, "rank": rank}
        for rank in range(count)
    ]


def test_the_stand_in_names_the_ranks_of_a_result_whose_skeleton_exceeds_the_line():
    """Only a document whose skeleton alone is over 1 MiB reaches the stand-in,
    and even that last resort lists its ranks."""

    row_bytes = encoded_size(skeleton_rows(1)[0])
    count = OBSERVATION_RESULT_BUDGET_BYTES // row_bytes + 2
    document = {**live_shaped_result(0), "ranks": skeleton_rows(count)}
    size = encoded_size({"final": True, "result": document}) + 1
    assert size > OBSERVATION_RESULT_BUDGET_BYTES, size
    assert (
        bounded_triage_result(copy.deepcopy(document))["evidence_trimmed"][
            "within_budget"
        ]
        is False
    ), "nothing above the floor is left to trim"

    message = published_line(document)

    result = message["result"]
    assert result["status"] == "RESULT_SIZE_LIMIT", result
    assert result["ranks"] == [] and result["evidence_complete"] is False
    assert result["failed_component"] == "observation_worker"
    assert result["failure_stage"] == WORKER_OUTPUT
    assert result["result_bytes"] == size
    assert result["error"].startswith(
        f"observation result exceeds budget: {size} > {OBSERVATION_RESULT_BUDGET_BYTES} bytes; largest fields: ranks="
    ), result["error"]
    assert result["ranks_total"] == count
    assert result["ranks_summary_truncated"] is False
    assert result["ranks_summary"] == [
        {"pid": 3000 + rank, "rank": rank, "status": "PAIRED"} for rank in range(count)
    ]
    assert encoded_size(result["ranks_summary"]) <= RESULT_STAND_IN_RANKS_BUDGET_BYTES


def test_the_stand_in_lists_as_many_ranks_as_fit_in_sixteen_kibibytes() -> None:
    count = 2000
    document = {
        "status": "COLLECTED",
        "ranks": [
            {"pid": 100 + rank, "rank": rank, "status": "COLLECTED", "pad": "x" * 600}
            for rank in range(count)
        ],
    }
    assert encoded_size(document) > OBSERVATION_RESULT_BUDGET_BYTES

    result = published_line(document)["result"]

    assert result["status"] == "RESULT_SIZE_LIMIT"
    assert result["ranks_total"] == count
    assert result["ranks_summary_truncated"] is True
    listed = result["ranks_summary"]
    assert 0 < len(listed) < count, len(listed)
    assert listed == [
        {"pid": 100 + rank, "rank": rank, "status": "COLLECTED"}
        for rank in range(len(listed))
    ], "the first ranks in the result's own order"
    assert encoded_size(listed) <= RESULT_STAND_IN_RANKS_BUDGET_BYTES, encoded_size(
        listed
    )
    one_more = [
        *listed,
        {"pid": 100 + len(listed), "rank": len(listed), "status": "COLLECTED"},
    ]
    assert encoded_size(one_more) > RESULT_STAND_IN_RANKS_BUDGET_BYTES, (
        "the list stops at the first rank that would not fit"
    )
    assert RESULT_STAND_IN_RANKS_BUDGET_BYTES == 16 * 1024


def test_a_ranks_status_is_its_own_then_the_series_then_the_recorder() -> None:
    rank_status = observation_worker.rank_status
    assert rank_status({"pid": 1, "error": "PROCESS_CHANGED_OR_UNREADABLE"}) == (
        "PROCESS_CHANGED_OR_UNREADABLE"
    )
    assert rank_status({"status": "UNAVAILABLE", "error": "x"}) == "UNAVAILABLE"
    assert rank_status(live_shaped_rank(0)) == "PAIRED"
    assert rank_status(live_shaped_rank(0, recorder=False)) == "not_configured"
    assert rank_status({"flight_recorder": {"status": "process_changed"}}) == (
        "process_changed"
    )
    assert rank_status({"communication_status": "AVAILABLE", "communication": []}) == (
        "AVAILABLE"
    )
    assert rank_status({"pid": 7, "rank": 0}) is None
    assert observation_worker.ranks_summary({"ranks": "not-a-list"}) == {
        "ranks_summary": [],
        "ranks_total": 0,
        "ranks_summary_truncated": False,
    }
