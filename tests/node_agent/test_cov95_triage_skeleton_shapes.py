"""The triage result ladder on the sparse rows a real triage also returns.

``test_hung_triage_result_budget`` and ``test_hung_triage_skeleton_floor`` walk
the ladder (``triage_budget.LADDER``) and its ``skeleton`` floor on the live
HyperPod shape, where every rank carries two dumps, a series and sample lists.
A triage of a real node also returns rows that shape never has: a rank whose
process exited before anything was read (no recorder, no series, no samples), a
rank whose recorder is not configured or whose dump held no operation
(``{"entry_count": 0}``), a rank whose second dump never came
(``SECOND_UNAVAILABLE``, no ``second``), and a series a previous Agent wrote
without ``groups``. Every step must pass such rows through, count only the rows
it changed, and the floor must keep what they have without inventing the
fields they lack.
"""

from __future__ import annotations

import copy
from typing import Any

from gpu_fault.node_agent.operations.triage_budget import (
    SKELETON_STEP,
    bounded_triage_result,
    drop_semantic_entries,
    drop_semantic_entries_carried_by_the_series,
    drop_unfinished_entries,
    encoded_size,
    keep_last_sample,
    limit_semantic_entries,
    skeleton,
)


def operation(sequence: int, *, state: str = "scheduled") -> dict[str, Any]:
    return {
        "pg_name": "0:default_pg",
        "collective_seq_id": sequence,
        "state": state,
        "profiling_name": "nccl:all_reduce",
        "input_dims": [[1024, 1024]],
    }


def vanished_rank() -> dict[str, Any]:
    """A rank whose process exited before the triage read anything of it."""

    return {
        "node_id": "node-a",
        "pid": 4001,
        "rank": 3,
        "status": "PROCESS_EXITED",
        "error": "process 4001 exited during the triage",
    }


def unconfigured_rank() -> dict[str, Any]:
    return {
        "node_id": "node-a",
        "pid": 4002,
        "rank": 4,
        "status": "OK",
        "flight_recorder": {
            "status": "not_configured",
            "trace_buffer": {"variable": None, "enabled": False},
        },
    }


def first_dump_only_rank() -> dict[str, Any]:
    """The first dump read and snapshotted; the second never came."""

    return {
        "node_id": "node-a",
        "pid": 4003,
        "rank": 5,
        "status": "OK",
        "flight_recorder_series": {
            "version": "fr-series/v1",
            "status": "SECOND_UNAVAILABLE",
            "first": {
                "status": "dumped",
                "parse_error": None,
                "groups": [
                    {
                        "pg_name": "0:default_pg",
                        "progress": {"last_started_seq": 7},
                        "semantic_entries": [operation(index) for index in range(12)],
                    }
                ],
            },
        },
    }


def test_rows_without_flight_evidence_pass_every_step_unchanged() -> None:
    rows = [vanished_rank(), unconfigured_rank(), first_dump_only_rank()]
    before = copy.deepcopy(rows)

    assert drop_semantic_entries_carried_by_the_series(rows) == 0, (
        "no recorder list to point at the series: nothing changed"
    )
    assert drop_unfinished_entries(rows) == 0
    assert keep_last_sample("stack_series")(rows) == 0
    assert keep_last_sample("process_series")(rows) == 0
    assert rows == before, "a step that changed nothing must not touch a row"

    # Only the series list of the first-dump-only rank is long enough to cut.
    assert limit_semantic_entries(8)(rows) == 1
    [group] = rows[2]["flight_recorder_series"]["first"]["groups"]
    assert [entry["collective_seq_id"] for entry in group["semantic_entries"]] == list(
        range(8)
    ), "the head of the list stays: pending operations lead"
    assert group["semantic_entries_total"] == 12, group
    assert group["semantic_entries_truncated"] is True, group
    assert rows[:2] == before[:2], "rows without operation lists are left alone"

    assert drop_semantic_entries(rows) == 1
    assert "semantic_entries" not in group, group
    assert group["semantic_entries_total"] == 12, "the count of the first cut stays"
    assert group["semantic_entries_trimmed"] is True, group
    assert group["progress"] == {"last_started_seq": 7}, "watermarks always stay"


def test_unfinished_entries_go_their_count_stays_and_the_floor_says_so() -> None:
    pending = {
        "pid": 4100,
        "rank": 0,
        "status": "OK",
        "flight_recorder": {
            "status": "dumped",
            "entry_count": 9,
            "unfinished_entry_count": 3,
            "unfinished_entries": [operation(7), operation(8), operation(9)],
        },
    }
    settled = {
        "pid": 4101,
        "rank": 1,
        "status": "OK",
        "flight_recorder": {
            "status": "dumped",
            "entry_count": 9,
            "unfinished_entry_count": 0,
            "unfinished_entries": [],
        },
    }
    untouched = copy.deepcopy(settled)

    assert drop_unfinished_entries([pending, settled]) == 1
    assert pending["flight_recorder"] == {
        "status": "dumped",
        "entry_count": 9,
        "unfinished_entry_count": 3,
        "unfinished_entries": [],
        "unfinished_entries_trimmed": True,
    }, pending
    assert settled == untouched, "an empty list was not trimmed and is not marked"

    # At the floor the trimmed list is still named as trimmed ...
    assert skeleton([pending, settled]) == 2
    assert pending["flight_recorder"]["unfinished_entries"] == []
    assert pending["flight_recorder"]["unfinished_entries_trimmed"] is True
    assert "unfinished_entries_trimmed" not in settled["flight_recorder"], (
        "nothing was dropped from the settled rank"
    )

    # ... and so is one the floor itself empties.
    direct = {
        "pid": 4102,
        "rank": 2,
        "flight_recorder": {"status": "dumped", "unfinished_entries": [operation(3)]},
    }
    assert skeleton([direct]) == 1
    assert direct["flight_recorder"] == {
        "status": "dumped",
        "unfinished_entries": [],
        "unfinished_entries_trimmed": True,
    }, direct


def test_the_floor_keeps_what_a_sparse_row_has_and_invents_nothing() -> None:
    vanished = vanished_rank()
    empty_dump = {
        "pid": 4200,
        "rank": 6,
        "status": "OK",
        "comparison_context": {"verified": True, "roles": {"1": "main"}},
        "flight_recorder": {
            "status": "dumped",
            "dump_file": "/proc/4200/root/tmp/nccl_trace_rank_6",
            "dump_size_bytes": 2,
            "entry_count": 0,
        },
        "python_stack": {"status": "OK", "threads": [{"name": "MainThread"}]},
    }
    legacy_series = {
        "pid": 4201,
        "rank": 7,
        "status": "OK",
        "flight_recorder": {
            "status": "dumped",
            "last_entries_by_pg": [
                {
                    "pg_name": "0:default_pg",
                    "semantic_entries_in": "flight_recorder_series.first",
                }
            ],
        },
        "flight_recorder_series": {
            "status": "SECOND_UNAVAILABLE",
            # A series an earlier Agent wrote: a group without its list or its
            # count, and a snapshot without groups.
            "first": {"status": "dumped", "groups": [{"pg_name": "0:default_pg"}]},
            "second": {"status": "dump_missing", "parse_error": None},
        },
        "proc": {"status": "OK", "returncode": 0, "stdout": "x" * 64},
    }

    assert skeleton([vanished, empty_dump, legacy_series]) == 3

    assert vanished == {**vanished_rank(), "skeleton": True}, (
        "a row with no evidence keeps exactly its identity and verdict"
    )
    assert empty_dump == {
        "pid": 4200,
        "rank": 6,
        "status": "OK",
        "skeleton": True,
        "comparison_context": {"verified": True},
        "flight_recorder": {
            "status": "dumped",
            "dump_file": "/proc/4200/root/tmp/nccl_trace_rank_6",
            "dump_size_bytes": 2,
            "entry_count": 0,
        },
        "python_stack": {"status": "OK"},
    }, empty_dump
    recorder = legacy_series["flight_recorder"]
    assert recorder == {
        "status": "dumped",
        "last_entries_by_pg": [
            {"pg_name": "0:default_pg", "semantic_entries_trimmed": True}
        ],
    }, "no count is invented for a list the series never counted"
    assert legacy_series["flight_recorder_series"] == {
        "status": "SECOND_UNAVAILABLE",
        "first": {"status": "dumped", "groups": [{"pg_name": "0:default_pg"}]},
        "second": {"status": "dump_missing", "parse_error": None},
    }, legacy_series["flight_recorder_series"]
    assert legacy_series["proc"] == {"status": "OK", "returncode": 0}
    assert skeleton([vanished, empty_dump, legacy_series]) == 0, (
        "a row already at the floor is left alone"
    )


def test_an_over_budget_document_with_sparse_ranks_reaches_the_floor_intact() -> None:
    heavy = first_dump_only_rank()
    heavy["flight_recorder_series"]["first"]["groups"][0]["semantic_entries"] = [
        {**operation(index), "input_dims": [[4096] * 64]} for index in range(64)
    ]
    heavy["stack_series"] = {
        "version": "stack-series/v1",
        "samples": [{"threads": ["x" * 2048]} for _ in range(3)],
    }
    result = {
        "status": "COMPLETE",
        "node_id": "node-a",
        "ranks": [vanished_rank(), unconfigured_rank(), heavy, "not a row"],
    }
    budget = 2048
    assert encoded_size(result) > 10 * budget, encoded_size(result)

    bounded = bounded_triage_result(result, budget)

    note = bounded["evidence_trimmed"]
    steps = [step["step"] for step in note["steps"]]
    assert steps == [
        "semantic_entries:32",
        "semantic_entries:16",
        "semantic_entries:8",
        "semantic_entries",
        "stack_series.samples:last",
        SKELETON_STEP,
    ], "steps that changed no rank are not recorded"
    assert note["steps"][-1]["ranks"] == 3, "every dict row reaches the floor"
    assert note["within_budget"] is True, note
    assert encoded_size(bounded) <= budget, "the document as sent fits"
    vanished, unconfigured, floor, other = bounded["ranks"]
    assert vanished == {**vanished_rank(), "skeleton": True}, vanished
    assert unconfigured["flight_recorder"] == {
        "status": "not_configured",
        "trace_buffer": {"variable": None, "enabled": False},
    }, "the recorder's status and why it is off are skeleton fields"
    assert other == "not a row", "a malformed row is never rewritten"
    assert floor["stack_series"] == {
        "version": "stack-series/v1",
        "samples": [],
        "samples_trimmed": 3,
    }, floor["stack_series"]
    [group] = floor["flight_recorder_series"]["first"]["groups"]
    assert group == {
        "pg_name": "0:default_pg",
        "progress": {"last_started_seq": 7},
        "semantic_entries_total": 64,
        "semantic_entries_trimmed": True,
    }, group
