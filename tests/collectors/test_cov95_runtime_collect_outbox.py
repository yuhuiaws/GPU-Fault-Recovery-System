"""Maintenance validation and durable file behavior, scoped to private files."""

from __future__ import annotations

import errno
import json
import os

import pytest

from gpu_fault.collectors import outbox_file
from gpu_fault.collectors.outbox_maintenance import (
    OutboxMaintenanceRequest,
    describe_record,
    requeue_error_prefix,
)
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"collector": None}, "collector parameter"),
        ({"collector": "retired-producer"}, "unknown collector"),
        ({"action": None}, "action parameter"),
        ({"action": "erase"}, "unknown outbox action"),
        ({"confirm": 1}, "boolean"),
        ({"path": 42}, "string"),
        ({"path": "relative"}, "control-plane path"),
        ({"action": "requeue-dead", "confirm": False}, "requires confirm"),
    ],
)
def test_invalid_maintenance_never_changes_the_outbox(tmp_path, changes, message):
    path = tmp_path / "kernel.ndjson"
    path.write_text('{"replayable":false,"payload":{"event_id":"event-a"}}\n')
    original = path.read_bytes()
    with pytest.raises(ValueError, match=message):
        OutboxMaintenanceRequest.from_parameters(
            {"collector": "kernel", "action": "stats", **changes}
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize("path_filter", [None, "/first"])
@pytest.mark.parametrize(
    "operator,reference",
    [(None, None), ("operator-a", None), ("operator-a", "change-1")],
)
def test_maintenance_requeues_only_whole_selected_records(
    tmp_path, path_filter, operator, reference
):
    request = OutboxMaintenanceRequest.from_parameters(
        {
            "collector": "kernel",
            "action": "requeue-dead",
            "confirm": True,
            "path": path_filter,
        }
    )
    outbox = outbox_file.OutboxFile(tmp_path / request.outbox_name)
    records = [
        {
            "path": "/first",
            "payload": {"event_id": "a"},
            "replayable": False,
            "error": "rejected",
        },
        {"path": "/other", "payload": {"event_id": "b"}, "replayable": False},
        {
            "path": "/first",
            "payload": {"payload_event_key": "c"},
            "replayable": False,
            "payload_truncated": True,
        },
        {"path": "/first", "payload": {}, "replayable": True},
    ]
    outbox.write(records)

    outcome = outbox.requeue_dead_report(
        path_filter=request.path, error_prefix=requeue_error_prefix(operator, reference)
    )

    expected = 2 if path_filter is None else 1
    assert outcome == {
        "requeued": expected,
        "skipped_payload_truncated": 1,
        "dead_before": 3,
        "dead_after": 3 - expected,
    }
    updated = outbox.read()
    assert updated[2] == records[2]
    assert updated[3] == records[3]
    assert updated[0]["error"].startswith(requeue_error_prefix(operator, reference)), (
        "the requeued record must retain the requested operator attribution"
    )
    assert outbox.stats()["replayable"] == expected + 1
    assert OutboxMaintenanceRequest.from_parameters(request.as_parameters()) == request


@pytest.mark.parametrize(
    ("payload", "truncated", "identity"),
    [
        (None, False, None),
        ({}, False, None),
        ({"event_id": "a"}, False, "a"),
        ({"payload_event_key": "b"}, True, "b"),
        ({"payload_event_key": 4}, True, None),
    ],
)
def test_outbox_description_is_bounded_and_does_not_disclose_payload(
    payload, truncated, identity
):
    result = describe_record(
        2,
        {
            "payload": payload,
            "payload_truncated": truncated,
            "replayable": True,
            "path": "/events",
            "failed_at": "stamp",
            "error": "bad\n" + "x" * 200,
        },
    )
    assert result["request_id"] == identity
    assert result["status"] == "replayable"
    assert len(result["error"]) == 120
    assert "\n" not in result["error"]
    assert "payload" not in result


@pytest.mark.parametrize("failure", ["open", "lock", "unlock", "close"])
def test_lock_holder_metadata_probe_closes_descriptors_on_os_errors(
    monkeypatch, tmp_path, failure
):
    outbox = outbox_file.OutboxFile(tmp_path / "kernel.ndjson")
    original_open, original_close = os.open, os.close
    seen = []
    closed = []

    def open_file(path, *args, **kwargs):
        if path == outbox.lock_path and failure == "open":
            raise OSError(errno.EIO, "open failed")
        return original_open(path, *args, **kwargs)

    def flock(fd, flags):
        seen.append(flags)
        if failure == "lock" or (
            failure == "unlock" and flags == outbox_file.fcntl.LOCK_UN
        ):
            raise OSError(errno.EIO, "lock failed")

    def close(fd):
        original_close(fd)
        closed.append(fd)
        if failure == "close":
            raise OSError(errno.EIO, "close reported failure")

    # The fakes replace global os functions, which pytest's own tmp_path
    # teardown also uses; keep them to the call under test.
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", open_file)
        patch.setattr(os, "close", close)
        patch.setattr(outbox_file.fcntl, "flock", flock)
        assert outbox.lock_holder() is None
    assert bool(seen) is (failure != "open")
    assert len(closed) == (0 if failure == "open" else 1)


@pytest.mark.parametrize(
    "pid_error", [ProcessLookupError, PermissionError, OverflowError]
)
def test_recorded_lock_owner_liveness_is_only_a_hint(monkeypatch, tmp_path, pid_error):
    outbox = outbox_file.OutboxFile(tmp_path / "kernel.ndjson")
    outbox.lock_path.write_text(
        json.dumps({"pid": 23, "role": "collector\nfake", "since": None})
    )

    def held(fd, flags):
        raise BlockingIOError(errno.EAGAIN, "held")

    def probe(pid, signal):
        assert (pid, signal) == (23, 0)
        raise pid_error("fake pid probe")

    monkeypatch.setattr(outbox_file.fcntl, "flock", held)
    monkeypatch.setattr(os, "kill", probe)
    description = outbox.lock_holder()
    assert "pid 23" in description
    assert "\n" not in description
    assert ("gone" in description) is (pid_error is ProcessLookupError)
    assert ("alive" in description) is (pid_error is not ProcessLookupError)
