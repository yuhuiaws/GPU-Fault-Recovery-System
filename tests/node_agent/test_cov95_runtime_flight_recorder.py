from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.node_agent import NodeActionExecutor, NodeActionStatus
from tests.node_agent._cov95_runtime_flight import collect
from tests.node_agent._cov95_runtime_support import (
    node_factory_fixture as node_factory_fixture,
)
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("mode", ["json", "pickle", "template"])
@pytest.mark.parametrize(
    ("payload", "count", "sequence", "group"),
    [
        (
            {
                "nested": {
                    "trace": [
                        {"collective_seq_id": "12", "process_group": ["0", "unit"]},
                        4,
                    ]
                }
            },
            1,
            12,
            "0:unit",
        ),
        (
            [
                [
                    "ignored",
                    {
                        "nested": {
                            "records": [
                                {
                                    "collective_seq_id_": 2,
                                    "group_name_": "group",
                                    "group_desc_": "unit",
                                }
                            ]
                        }
                    },
                ]
            ],
            1,
            2,
            "group:unit",
        ),
        (
            {
                "entries": [
                    {"collective_seq_id": None},
                    {"collective_seq_id": []},
                    {"collective_seq_id": "invalid"},
                    {"collective_seq_id": 9, "process_group": ["solo"]},
                ]
            },
            1,
            9,
            "solo",
        ),
        ({"records": [None, {}, {"time_created": "unused"}]}, 0, None, None),
    ],
)
def test_real_triage_reads_nested_torch_formats_without_inventing_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    node_factory: Callable[..., NodeActionExecutor],
    mode: str,
    payload: Any,
    count: int,
    sequence: int | None,
    group: str | None,
) -> None:
    result, writes = collect(
        monkeypatch, tmp_path, node_factory, payload=payload, mode=mode
    )
    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["read_only"] is True
    (rank,) = result.details["ranks"]
    assert rank["rank"] == 7
    assert rank["gpu_uuid"] == "GPU-a"
    report = rank["flight_recorder"]
    assert report["status"] == "dumped"
    assert report["entry_count"] == count
    if sequence is not None:
        assert report["last_entry"]["collective_seq_id"] == sequence
        assert report["last_entry"]["pg_name"] == group
    else:
        assert "last_entry" not in report
    assert writes == [b"1\n"]


@pytest.mark.parametrize(
    ("mode", "status", "write_count"),
    [
        ("no-environment", "not_configured", 0),
        ("traversal", "pipe_missing", 0),
        ("nonfifo", "pipe_missing", 0),
        ("write-failure", "pipe_write_failed", 0),
        ("no-dump", "dump_missing", 1),
        ("unchanged", "dump_missing", 1),
        ("corrupt", "dumped", 1),
        ("global-pickle", "dumped", 1),
    ],
)
def test_missing_stale_or_untrusted_dump_remains_unproved_with_a_bounded_wait(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    node_factory: Callable[..., NodeActionExecutor],
    mode: str,
    status: str,
    write_count: int,
) -> None:
    result, writes = collect(monkeypatch, tmp_path, node_factory, mode=mode)
    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["elapsed_seconds"] <= 2.5
    (rank,) = result.details["ranks"]
    report = rank["flight_recorder"]
    assert report["status"] == status
    assert "last_entry" not in report
    if mode in {"corrupt", "global-pickle"}:
        assert "parse_error" in report
    if mode == "global-pickle":
        assert "refusing to resolve" in report["parse_error"]
    assert writes == [b"1\n"] * write_count


def test_many_unfinished_collectives_are_bounded_and_preserve_the_last_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    node_factory: Callable[..., NodeActionExecutor],
) -> None:
    result, _writes = collect(
        monkeypatch,
        tmp_path,
        node_factory,
        payload={
            "entries": [
                {"collective_seq_id": index, "pg_name": "unit"} for index in range(70)
            ]
        },
    )
    assert result.status is NodeActionStatus.SUCCEEDED
    report = result.details["ranks"][0]["flight_recorder"]
    assert report["entry_count"] == 70
    assert report["last_entry"]["collective_seq_id"] == 69
    assert report["unfinished_entry_count"] == 70
    assert len(report["unfinished_entries"]) == 64
    assert report["unfinished_entries_truncated"] is True
