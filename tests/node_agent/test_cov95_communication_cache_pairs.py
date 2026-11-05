"""The hung survey's cached communication window and its flight recorder comparison.

``cached_communication`` reads the two passive samples a survey compares --
the requested pair of consecutive epochs or, without one, the newest two of
the last two minutes -- and refuses an implausible request or window instead
of comparing it. ``cached_flight_pairs`` pairs a process generation's two
existing flight recorder files, and ``apply_cached_flights`` turns each
process group whose membership and comparability held across them into the
rank's communication evidence. The history is the real SQLite store under
``tmp_path``.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.node_agent.communication_history import CommunicationHistory
from gpu_fault.node_agent.operations.communication_cache import (
    apply_cached_flights,
    cached_communication,
    cached_flight_pairs,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
STAMP = NOW.timestamp()
EPOCH = int(STAMP // 30)
MEMBERS = hashlib.sha256(b"[0,1]").hexdigest()


def world(tmp_path: Path, *, boot: bool = True) -> SimpleNamespace:
    if boot:
        directory = tmp_path / "proc/sys/kernel/random"
        directory.mkdir(parents=True)
        (directory / "boot_id").write_text("boot\n")
    return SimpleNamespace(
        proc_root=tmp_path / "proc",
        diagnostic_output_dir=tmp_path / "evidence",
        now=lambda: NOW,
    )


def history(tmp_path: Path) -> CommunicationHistory:
    return CommunicationHistory(tmp_path / "evidence" / "communication-history.db")


def record(store: CommunicationHistory, stamp: float, *processes: Any) -> None:
    store.append(
        store.scope("job", "attempt", "boot"),
        int(stamp // 30),
        stamp,
        {
            "observed_at": datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
            "node_addresses": ["10.0.0.1"],
            "processes": list(processes),
        },
    )


def request(**parameters: Any) -> SimpleNamespace:
    return SimpleNamespace(
        parameters={"job_id": "job", "attempt_id": "attempt", **parameters}
    )


def group(
    name: str = "0:default_pg",
    progress: tuple[Any, Any, Any] = (10, 9, 8),
    *,
    members: str | None = MEMBERS,
    count: int = 2,
    supported: bool = True,
) -> dict[str, Any]:
    enqueued, started, completed = progress
    return {
        "pg_name": name,
        "pg_membership_sha256": members,
        "pg_member_count": count,
        "progress": {
            "comparison_supported": supported,
            "last_enqueued": enqueued,
            "last_started": started,
            "last_completed": completed,
        },
    }


def flight_process(pid: int, mtime: int, groups: list[Any] | None = None) -> dict:
    return {
        "pid": pid,
        "rank": pid - 100,
        "process_identity": {"verified": True, "start_time_ticks": 1000 + pid},
        "flight_recorder": {
            "status": "dumped",
            "source_mtime_ns": mtime,
            "last_entries_by_pg": groups if groups is not None else [group()],
        },
    }


def test_without_requested_epochs_the_newest_two_recent_samples_are_compared(tmp_path):
    agent = world(tmp_path)
    store = history(tmp_path)
    record(store, STAMP - 200, flight_process(100, 5))
    record(store, STAMP - 60, flight_process(100, 10))
    record(store, STAMP - 30, flight_process(100, 20, [group(progress=(12, 11, 9))]))

    result = cached_communication(agent, request(clock_uncertainty_seconds=0.5))

    window = result["communication_history"]
    assert window["status"] == "AVAILABLE", result
    assert window["requested_epochs"] is None, window
    assert (window["first_epoch"], window["second_epoch"]) == (EPOCH - 2, EPOCH - 1)
    assert window["interval_seconds"] == 30 and window["age_seconds"] == 30, window
    assert window["clock_uncertainty_seconds"] == 0.5, window
    assert result["node_addresses"] == ["10.0.0.1"], result
    (pair,) = result["cached_fr"]
    assert pair["pid"] == 100 and pair["scope"] == "EXISTING_FILES_ONLY", pair
    assert pair["second"][0]["progress"]["last_enqueued"] == 12, pair

    summary = {
        **result,
        "groups": {},
        "ranks": [
            {
                "pid": 100,
                "process_identity": {"verified": True, "start_time_ticks": 1100},
                "communication_status": "UNAVAILABLE",
                "communication": [],
            }
        ],
    }
    apply_cached_flights(summary)
    (evidence,) = summary["ranks"][0]["communication"]
    assert evidence["before"] == [10, 9, 8] and evidence["after"] == [12, 11, 9]
    assert summary["ranks"][0]["communication_status"] == "AVAILABLE", summary


@pytest.mark.parametrize(
    "epochs",
    [
        pytest.param([EPOCH - 3, EPOCH - 1], id="not-consecutive"),
        pytest.param([EPOCH - 1], id="one-epoch"),
        pytest.param([str(EPOCH - 2), str(EPOCH - 1)], id="strings"),
        pytest.param([True, EPOCH], id="boolean"),
        pytest.param([EPOCH - 100, EPOCH - 99], id="older-than-the-history"),
        pytest.param([EPOCH + 1, EPOCH + 2], id="in-the-future"),
        pytest.param(f"{EPOCH - 2},{EPOCH - 1}", id="not-a-list"),
    ],
)
def test_a_requested_epoch_pair_must_be_two_consecutive_recent_epochs(tmp_path, epochs):
    agent = world(tmp_path)
    store = history(tmp_path)
    record(store, STAMP - 60, flight_process(100, 10))
    record(store, STAMP - 30, flight_process(100, 20))

    result = cached_communication(agent, request(communication_history_epochs=epochs))

    assert result == {"communication_history": {"status": "INVALID_REQUESTED_EPOCHS"}}


@pytest.mark.parametrize(
    "stamps,epochs",
    [
        pytest.param(
            ((EPOCH - 2) * 30 + 29.6, (EPOCH - 1) * 30 + 0.2),
            [EPOCH - 2, EPOCH - 1],
            id="requested-pair-under-a-second-apart",
        ),
        pytest.param((STAMP - 115, STAMP - 10), None, id="latest-pair-too-far-apart"),
    ],
)
def test_a_window_outside_one_to_ninety_seconds_is_never_compared(
    tmp_path, stamps, epochs
):
    agent = world(tmp_path)
    store = history(tmp_path)
    for stamp in stamps:
        record(store, stamp, flight_process(100, int(stamp)))
    parameters = {} if epochs is None else {"communication_history_epochs": epochs}

    result = cached_communication(agent, request(**parameters))

    assert result == {"communication_history": {"status": "INVALID_WINDOW"}}


def corrupt_database(tmp_path: Path) -> None:
    path = tmp_path / "evidence" / "communication-history.db"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not an SQLite database" * 64)


def corrupt_sample(tmp_path: Path) -> None:
    store = history(tmp_path)
    record(store, STAMP - 60, flight_process(100, 10))
    with sqlite3.connect(store.path) as db:
        db.execute(
            "INSERT INTO sample VALUES (?,?,?,?,?)",
            (store.scope("job", "attempt", "boot"), EPOCH - 1, STAMP - 30, b"raw", 3),
        )


@pytest.mark.parametrize(
    "damage,missing,boot,fragment",
    [
        pytest.param(corrupt_database, None, True, "not a database", id="corrupt-db"),
        pytest.param(corrupt_sample, None, True, "decompressing", id="corrupt-sample"),
        pytest.param(history, None, False, "No such file", id="no-boot-id"),
        pytest.param(history, "job_id", True, "'job_id'", id="no-job-id"),
    ],
)
def test_an_unreadable_history_is_unavailable_with_its_bounded_cause(
    tmp_path, damage, missing, boot, fragment
):
    agent = world(tmp_path, boot=boot)
    damage(tmp_path)
    command = request()
    command.parameters.pop(missing, None)

    result = cached_communication(agent, command)

    window = result["communication_history"]
    assert window["status"] == "UNAVAILABLE", result
    assert fragment in window["reason"] and len(window["reason"]) <= 128, window


def test_flight_pairs_are_bounded_to_256_processes():
    first = {"processes": [flight_process(pid, 10) for pid in range(100, 400)]}
    second = {
        "processes": [
            flight_process(pid, 20, [group(progress=(11, 10, 9))])
            for pid in range(100, 400)
        ]
    }

    pairs = cached_flight_pairs(first, second)

    assert len(pairs) == 256, len(pairs)
    assert pairs[0] == {
        "pid": 100,
        "rank": 0,
        "process_identity": {"verified": True, "start_time_ticks": 1100},
        "first": [group()],
        "second": [group(progress=(11, 10, 9))],
        "completion_source": "FR_STATUS_OR_VISIBLE_ENTRIES",
        "scope": "EXISTING_FILES_ONLY",
    }, pairs[0]


def survey_rank(pid: int, *, start: int | None = None, **values: Any) -> dict:
    return {
        "pid": pid,
        "process_identity": {
            "verified": values.pop("verified", True),
            "start_time_ticks": 1000 + pid if start is None else start,
        },
        "communication_status": values.pop("status", "UNAVAILABLE"),
        "communication": values.pop("communication", []),
    }


def pair(pid: int, first: list[Any], second: list[Any]) -> dict[str, Any]:
    return {
        "pid": pid,
        "process_identity": {"verified": True, "start_time_ticks": 1000 + pid},
        "first": first,
        "second": second,
    }


def test_only_comparable_groups_of_the_same_process_become_cached_evidence():
    live = [{"group": "application-group", "source": "APPLICATION_SAMPLE"}]
    good = [group()]
    summary = {
        "communication_history": {
            "status": "AVAILABLE",
            "first_observed_at": "2026-07-20T11:59:00+00:00",
            "second_observed_at": "2026-07-20T11:59:30+00:00",
            "clock_uncertainty_seconds": 0.25,
        },
        "groups": {"application-group": {"group_key": "application-group"}},
        "ranks": [
            survey_rank(100),
            survey_rank(101, communication=list(live), status="AVAILABLE"),
            survey_rank(102, verified=False),
            survey_rank(103, start=9999),
            survey_rank(104),
        ],
        "cached_fr": [
            pair(
                100,
                [
                    group(),
                    group("1:tp"),
                    group("3:resized"),
                    group("4:anon", members=None),
                ],
                [
                    group(progress=(12, None, 11)),
                    group("1:tp", supported=False),
                    group("2:new"),
                    group("3:resized", count=4),
                    group("4:anon", members=None),
                ],
            ),
            pair(101, good, good),
            pair(102, good, good),
            pair(103, good, good),
            pair(104, [group(members="a" * 64)], [group(members="b" * 64)]),
            pair(105, good, good),
        ],
    }

    apply_cached_flights(summary)

    key = "fr-" + hashlib.sha256(f"0:default_pg/{MEMBERS}".encode()).hexdigest()
    assert summary["groups"] == {
        "application-group": {"group_key": "application-group"},
        key: {
            "group_key": key,
            "name": "0:default_pg",
            "membership_sha256": MEMBERS,
            "member_count": 2,
        },
    }, summary["groups"]
    ranks = {row["pid"]: row for row in summary["ranks"]}
    assert ranks[100]["communication"] == [
        {
            "group": key,
            "before": [10, 9, 8],
            "after": [12, -1, 11],
            "first_observed_at": "2026-07-20T11:59:00+00:00",
            "second_observed_at": "2026-07-20T11:59:30+00:00",
            "clock_uncertainty_seconds": 0.25,
            "source": "CACHED_FLIGHT_RECORDER",
        }
    ], ranks[100]
    assert ranks[100]["communication_status"] == "AVAILABLE", ranks[100]
    assert ranks[101]["communication"] == live, "live evidence is never replaced"
    for pid in (102, 103, 104):
        assert ranks[pid]["communication"] == [], ranks[pid]
        assert ranks[pid]["communication_status"] == "UNAVAILABLE", ranks[pid]
    assert "cached_fr" not in summary, "the raw pairs never leave the node"
