from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.postgres_grant import PostgresGrantError
from scripts import postgres_shard_receipts
from scripts.postgres_shard_receipts import (
    PostgresShardError,
    read_failure,
    read_private_json,
    validate_partition_union,
    validate_targets,
)
from tests._parallel_postgres_support import (
    DISCOVERED,
    checked_receipt,
    receipt_payload,
)


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_complete_disjoint_nodeid_partitions(tmp_path: Path, workers: int) -> None:
    receipts = [
        checked_receipt(
            tmp_path, receipt_payload(index, workers), index=index, workers=workers
        )
        for index in range(workers)
    ]
    durations = validate_partition_union(receipts, workers=workers)
    assert set(durations) == set(DISCOVERED), (
        "every discovered test needs one execution"
    )
    assert sum(durations.values()) == len(DISCOVERED) * 0.25


@pytest.mark.parametrize("offset_microseconds", [-1, 0, 1])
def test_receipt_finish_cannot_be_after_the_same_host_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offset_microseconds: int
) -> None:
    now = datetime(2026, 9, 1, 0, 0, 2, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Clock:
            return cls.fromtimestamp(now.timestamp(), tz)

    monkeypatch.setattr(postgres_shard_receipts, "datetime", Clock)
    value = receipt_payload(0)
    value["session"]["finished_at"] = (
        now + timedelta(microseconds=offset_microseconds)
    ).isoformat()
    if offset_microseconds > 0:
        with pytest.raises(PostgresShardError, match="timestamps"):
            checked_receipt(tmp_path, value)
    else:
        receipt = checked_receipt(tmp_path, value)
        assert receipt.wall_seconds > 0, "a completed receipt was rejected"


@pytest.mark.parametrize(
    "field,value",
    [
        ("nodeid", "tests/test_case.py::test_case\ninjected output"),
        ("nodeid", "::test_case"),
        ("nodeid", "../tests/test_case.py::test_case"),
        ("nodeid", "tests/test_case.py::" + "x" * 4096),
        ("nodeid", ["tests/test_case.py::test_case"]),
        ("phase", "arbitrary output"),
        ("phase", []),
        ("outcome", "arbitrary output"),
        ("outcome", []),
        ("output", "arbitrary output"),
    ],
)
def test_first_failure_only_exposes_bounded_protocol_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = {
        "failed": True,
        "nodeid": "tests/test_case.py::test_case[value]",
        "phase": "call",
        "outcome": "failed",
    }
    payload[field] = value
    path = tmp_path / "failure.json"
    write_json_atomic(path, payload)
    with pytest.raises(PostgresShardError, match="first-failure metadata"):
        read_failure(path)


@pytest.mark.parametrize(
    "defect",
    [
        "source",
        "session-source",
        "aggregate",
        "exit",
        "missing-record",
        "missing-discovery",
        "collection-error",
        "collection-skip",
        "missing-file",
        "filtered",
        "xdist",
        "requested-xdist",
        "partition",
        "boolean-partition",
        "stress-workers",
        "stress-rounds",
        "skip",
        "teardown",
        "setup",
        "nan-duration",
        "infinite-duration",
        "negative-duration",
        "boolean-duration",
        "missing-duration",
        "output",
        "old",
        "unaware",
        "reverse-time",
        "ci-context",
        "alias",
        "unselected",
        "empty-partition",
    ],
)
def test_receipts_fail_closed(tmp_path: Path, defect: str) -> None:
    value = receipt_payload(0)
    session = value["session"]
    selection = session["selection"]
    nodeid = next(iter(value["records"]))
    record = value["records"][nodeid]
    if defect == "source":
        value["source_identity"] = "b" * 64
    elif defect == "session-source":
        session["source_identity"] = "b" * 64
    elif defect == "aggregate":
        value["schema_version"] = 2
    elif defect == "exit":
        session["exitstatus"] = 1
    elif defect == "missing-record":
        del value["records"][nodeid]
    elif defect == "missing-discovery":
        session["discovered_nodeids"].remove(nodeid)
    elif defect == "collection-error":
        session["collection_errors"] = ["tests/test_first.py"]
    elif defect == "collection-skip":
        session["collection_skips"] = ["tests/test_first.py"]
    elif defect == "missing-file":
        session["collected_files"].pop()
    elif defect == "filtered":
        selection["keyword"] = "not stress"
    elif defect == "xdist":
        selection["numprocesses"] = 4
    elif defect == "requested-xdist":
        selection["requested_numprocesses"] = "auto"
    elif defect == "partition":
        selection["partition"] = [4, 1]
    elif defect == "boolean-partition":
        selection["partition"] = [4, False]
    elif defect == "stress-workers":
        selection["stress_workers"] = "1"
    elif defect == "stress-rounds":
        selection["stress_rounds"] = "1"
    elif defect in {"skip", "teardown", "setup"}:
        record["phases"]["call" if defect == "skip" else defect] = "skipped"
    elif defect.endswith("-duration"):
        if defect == "missing-duration":
            del record["duration_seconds"]
        else:
            record["duration_seconds"] = {
                "nan-duration": math.nan,
                "infinite-duration": math.inf,
                "negative-duration": -1,
                "boolean-duration": True,
            }[defect]
    elif defect == "output":
        record["output"] = "example-private-test-output"
    elif defect == "old":
        session["started_at"] = "2026-08-01T00:00:00+00:00"
    elif defect == "unaware":
        session["started_at"] = "2026-09-01T00:00:01"
    elif defect == "reverse-time":
        session["finished_at"] = "2026-09-01T00:00:00+00:00"
    elif defect == "ci-context":
        session["ci_context"] = {}
    elif defect == "alias":
        alias = nodeid.replace("tests/", "tests/../tests/")
        value["records"][alias] = dict(record)
        session["collected_nodeids"].append(alias)
        session["discovered_nodeids"].append(alias)
    elif defect == "unselected":
        extra = next(item for item in DISCOVERED if item not in value["records"])
        value["records"][extra] = dict(record)
        session["collected_nodeids"].append(extra)
    elif defect == "empty-partition":
        value["records"] = {}
        session["collected_nodeids"] = []
    else:
        raise AssertionError("unhandled receipt defect")
    with pytest.raises(PostgresShardError):
        checked_receipt(tmp_path, value)


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "discovery", "overlap", "union"]
)
def test_cross_worker_inventory_is_exact(tmp_path: Path, defect: str) -> None:
    receipts = [
        checked_receipt(tmp_path, receipt_payload(index), index=index)
        for index in range(4)
    ]
    if defect == "missing":
        receipts.pop()
    elif defect == "duplicate":
        receipts[1] = receipts[0]
    elif defect == "discovery":
        receipts[1] = replace(receipts[1], discovered=frozenset())
    elif defect == "overlap":
        receipts[1] = replace(receipts[1], durations=dict(receipts[0].durations))
    elif defect == "union":
        receipts[1] = replace(receipts[1], durations={})
    with pytest.raises(PostgresShardError):
        validate_partition_union(receipts, workers=4)


@pytest.mark.parametrize(
    "defect", ["malformed", "duplicate", "public", "symlink", "oversized"]
)
def test_private_receipt_reader_refuses_unsafe_files(
    tmp_path: Path, defect: str
) -> None:
    tmp_path.chmod(0o700)
    path = tmp_path / "receipt.json"
    path.write_text('{"value": 1}')
    path.chmod(0o600)
    if defect == "malformed":
        path.write_text("{")
    elif defect == "duplicate":
        path.write_text('{"value": 1, "value": 2}')
    elif defect == "public":
        path.chmod(0o644)
    elif defect == "symlink":
        target = tmp_path / "target"
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises((PostgresShardError, PostgresGrantError)):
        read_private_json(path, maximum=1 if defect == "oversized" else 1024)


@pytest.mark.parametrize(
    "targets",
    [
        [],
        ["tests/test_ok.py", "tests/test_ok.py"],
        ["-k"],
        ["tests/test_ok.py::test_one"],
        ["tests/../tests/test_ok.py"],
        ["/tests/test_ok.py"],
        ["tests/missing.py"],
        ["tests"],
    ],
)
def test_targets_cannot_silently_reduce_the_make_inventory(
    tmp_path: Path, targets: list[str]
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_ok.py").touch()
    with pytest.raises(PostgresShardError):
        validate_targets(tmp_path, targets)


def test_target_symlink_cannot_leave_the_source(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "external.py").touch()
    (tmp_path / "tests/test_alias.py").symlink_to(tmp_path / "external.py")
    with pytest.raises(PostgresShardError):
        validate_targets(tmp_path, ["tests/test_alias.py"])
