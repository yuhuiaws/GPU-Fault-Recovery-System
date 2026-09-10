"""``COLLECTOR_OUTBOX_MAINTENANCE``: the node-local ``gpu-fault-collector
outbox`` command, run by the node agent for ``gpu-fault-admin collector-outbox``.

The handler calls the outbox file layer in-process with the same safety
properties as the CLI without ``--force``: metadata only on the wire (ARCH-G2),
the lock taken strictly and a held lock reported as a structured failure that
names the recorded holder, payload-truncated dead letters left dead.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from threading import Thread

import pytest

from gpu_fault.collectors import outbox_file as collector_outbox
from gpu_fault.collectors.outbox_file import OutboxFile
from gpu_fault.collectors.outbox_maintenance import (
    ACTION_LIST,
    ACTION_REQUEUE_DEAD,
    ACTION_STATS,
    OUTBOX_COLLECTORS,
    OutboxMaintenanceRequest,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.operations.collector_outbox import CollectorOutboxRefused
from gpu_fault.node_agent.protocol import NodeActionStatus
from tests.node_agent._support import command, envelope, node_action_executor

SECRET_PAYLOAD = "GPU-SECRET-UUID-0000"


def _record(
    index: int,
    *,
    replayable: bool,
    path: str = "/v1/kernel-logs",
    error: str = "HTTP 422: rejected",
    truncated: bool = False,
) -> dict:
    record = {
        "path": path,
        "payload": {"event_id": f"event-{index}", "gpu_uuid": SECRET_PAYLOAD},
        "replayable": replayable,
        "error": error,
        "failed_at": f"2026-09-10T08:0{index}:00+00:00",
    }
    if truncated:
        record["payload"] = {
            "payload_event_key": f"event-{index}",
            "payload_sha256": "a" * 64,
            "payload_bytes": 999999,
            "payload_excerpt": SECRET_PAYLOAD,
        }
        record["payload_truncated"] = True
    return record


def _seed(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in records),
        encoding="utf-8",
    )


def _agent(tmp_path: Path):
    return node_action_executor(
        tmp_path,
        "outbox-actions.db",
        allowed_operations={WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE},
        collector_outbox_directory=str(tmp_path / "outbox"),
    )


def _run(agent, **parameters):
    values = {"collector": "kernel", "action": ACTION_STATS, "confirm": False}
    values.update(parameters)
    return agent.execute(
        envelope(
            command(
                WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE,
                command_id=f"workflow/step/node-a/{values['action']}",
                parameters=values,
                gpu_uuids=[],
            )
        )
    )


def test_stats_reports_depth_split_and_oldest_failure_without_payloads(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / "outbox" / "kernel.ndjson"
    _seed(
        outbox,
        [
            _record(1, replayable=False),
            _record(2, replayable=True),
            _record(3, replayable=False, truncated=True),
        ],
    )

    result = _run(_agent(tmp_path), action=ACTION_STATS)

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert result.details["collector"] == "kernel"
    assert result.details["action"] == ACTION_STATS
    assert result.details["outbox_path"] == str(outbox)
    stats = result.details["stats"]
    assert (stats["depth"], stats["replayable"], stats["dead"]) == (3, 1, 2)
    assert stats["oldest_failed_at"] == "2026-09-10T08:01:00+00:00"
    assert stats["payload_truncated"] == 1
    assert result.details["lock_holder"] is None, "nobody holds the lock"
    assert SECRET_PAYLOAD not in json.dumps(result.details), (
        "stats must never carry payload bytes (ARCH-G2)"
    )


def test_stats_on_a_missing_outbox_is_an_empty_outbox(tmp_path: Path) -> None:
    result = _run(_agent(tmp_path), action=ACTION_STATS, collector="host")

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert result.details["stats"]["depth"] == 0
    assert result.details["outbox_exists"] is False


def test_list_returns_one_metadata_record_per_entry_and_no_payload(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / "outbox" / "kernel.ndjson"
    _seed(
        outbox,
        [
            _record(1, replayable=False, error="HTTP 422: " + "x" * 300),
            _record(2, replayable=True, path="/v1/other"),
            _record(3, replayable=False, truncated=True),
        ],
    )

    result = _run(_agent(tmp_path), action=ACTION_LIST)

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    records = result.details["records"]
    assert [item["index"] for item in records] == [0, 1, 2]
    assert [item["request_id"] for item in records] == ["event-1", "event-2", "event-3"]
    assert [item["status"] for item in records] == ["dead", "replayable", "dead"]
    assert [item["replayable"] for item in records] == [False, True, False]
    assert [item["payload_truncated"] for item in records] == [False, False, True]
    assert records[1]["path"] == "/v1/other"
    assert records[0]["failed_at"] == "2026-09-10T08:01:00+00:00"
    assert len(records[0]["error"]) <= 120, "the error is a prefix, not the body"
    assert SECRET_PAYLOAD not in json.dumps(result.details), (
        "list must never carry payload bytes (ARCH-G2)"
    )
    for item in records:
        assert "payload" not in item


def test_list_honours_the_path_filter(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox" / "kernel.ndjson"
    _seed(
        outbox,
        [_record(1, replayable=False), _record(2, replayable=True, path="/v1/other")],
    )

    result = _run(_agent(tmp_path), action=ACTION_LIST, path="/v1/other")

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert [item["request_id"] for item in result.details["records"]] == ["event-2"]
    assert result.details["path_filter"] == "/v1/other"


def test_requeue_dead_flips_dead_records_and_skips_truncated_ones(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / "outbox" / "kernel.ndjson"
    _seed(
        outbox,
        [
            _record(1, replayable=False),
            _record(2, replayable=True),
            _record(3, replayable=False, truncated=True),
            _record(4, replayable=False, path="/v1/other"),
        ],
    )

    result = _run(
        _agent(tmp_path),
        action=ACTION_REQUEUE_DEAD,
        confirm=True,
        operator="arn:aws:sts::1:assumed-role/Admin/ops",
        reference="CHG-77",
    )

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert result.details["requeued"] == 2
    assert result.details["skipped_payload_truncated"] == 1
    assert result.details["dead_before"] == 3
    assert result.details["dead_after"] == 1
    written = [json.loads(line) for line in outbox.read_text().splitlines()]
    assert [item["replayable"] for item in written] == [True, True, False, True]
    assert written[0]["error"].startswith(
        "requeued by operator: arn:aws:sts::1:assumed-role/Admin/ops (CHG-77): "
    ), written[0]["error"]
    assert written[0]["error"].endswith("HTTP 422: rejected"), (
        "the previous error survives behind the operator prefix"
    )
    assert written[2]["payload_truncated"] is True, "a truncated record stays dead"
    assert SECRET_PAYLOAD not in json.dumps(result.details)


def test_requeue_dead_honours_the_path_filter(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox" / "kernel.ndjson"
    _seed(
        outbox,
        [_record(1, replayable=False), _record(2, replayable=False, path="/v1/other")],
    )

    result = _run(
        _agent(tmp_path), action=ACTION_REQUEUE_DEAD, confirm=True, path="/v1/other"
    )

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert result.details["requeued"] == 1
    written = [json.loads(line) for line in outbox.read_text().splitlines()]
    assert [item["replayable"] for item in written] == [False, True]


def test_requeue_dead_without_confirmation_is_refused_and_writes_nothing(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / "outbox" / "kernel.ndjson"
    _seed(outbox, [_record(1, replayable=False)])
    before = outbox.read_text()

    result = _run(_agent(tmp_path), action=ACTION_REQUEUE_DEAD, confirm=False)

    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is False
    assert "confirm" in str(result.error)
    assert outbox.read_text() == before


def test_a_held_lock_is_a_structured_refusal_naming_the_recorded_holder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The remote path cannot see whether the collector is stopped, so it never
    rewrites without the lock: the refusal carries the holder line the CLI
    would print, and the step fails without touching the file."""

    monkeypatch.setattr(collector_outbox, "OUTBOX_LOCK_RETRY_SECONDS", 0.02)
    outbox = tmp_path / "outbox" / "kernel.ndjson"
    _seed(outbox, [_record(1, replayable=False)])
    before = outbox.read_text()
    lock_path = OutboxFile(outbox).lock_path
    agent = _agent(tmp_path)
    outcome: list = []

    def requeue() -> None:
        outcome.append(_run(agent, action=ACTION_REQUEUE_DEAD, confirm=True))

    handle = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    worker = Thread(target=requeue, daemon=True)
    try:
        os.write(
            handle,
            json.dumps(
                {"pid": os.getpid(), "role": "collector", "since": "2026-09-10T08:00"}
            ).encode(),
        )
        fcntl.flock(handle, fcntl.LOCK_EX)
        worker.start()
        worker.join(10)
        blocked = worker.is_alive()
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)
        worker.join(10)

    assert not blocked, "the remote requeue must be bounded, not blocking"
    result = outcome[0]
    assert result.status is NodeActionStatus.FAILED, result
    assert result.retryable is False
    assert result.details["lock_unavailable"] is True
    assert str(lock_path) in result.details["lock_path"]
    assert "recorded holder pid" in result.details["lock_holder"], result.details
    assert "(collector)" in result.details["lock_holder"]
    assert "pass --force to work without the lock" not in str(result.error), (
        "the CLI's own --force advice does not apply: the remote path has no such flag"
    )
    assert "node-local" in str(result.error), (
        "a stopped collector's leftover lock is sent to the node-local CLI"
    )
    assert outbox.read_text() == before, "a refused requeue rewrote the outbox"


def test_stats_reports_the_recorded_lock_holder_while_the_lock_is_held(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / "outbox" / "kernel.ndjson"
    _seed(outbox, [_record(1, replayable=False)])
    lock_path = OutboxFile(outbox).lock_path
    handle = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.write(
            handle,
            json.dumps(
                {"pid": os.getpid(), "role": "collector", "since": "2026-09-10T08:00"}
            ).encode(),
        )
        fcntl.flock(handle, fcntl.LOCK_EX)
        result = _run(_agent(tmp_path), action=ACTION_STATS)
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert "recorded holder pid" in str(result.details["lock_holder"])


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        ({"collector": "sqs-hma", "action": ACTION_STATS}, "collector"),
        ({"collector": "kernel", "action": "purge"}, "action"),
        ({"collector": "../etc", "action": ACTION_STATS}, "collector"),
        ({"collector": "kernel"}, "action"),
        ({"collector": "kernel", "action": ACTION_LIST, "path": 7}, "path"),
    ],
)
def test_bad_parameters_fail_the_step_without_touching_the_node(
    tmp_path: Path, parameters: dict, match: str
) -> None:
    agent = _agent(tmp_path)

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE,
                parameters=parameters,
                gpu_uuids=[],
            )
        )
    )

    assert result.status is NodeActionStatus.FAILED
    assert result.retryable is False
    assert match in str(result.error)
    assert not (tmp_path / "outbox").exists(), "a refused request touches no file"


def test_the_request_contract_round_trips_and_names_the_node_collectors() -> None:
    assert set(OUTBOX_COLLECTORS) == {
        "kernel",
        "dcgm",
        "nvidia-smi",
        "host",
        "logs",
        "fabric-manager",
    }, "every node collector unit owns an outbox; the CPU-side collectors do not"
    request = OutboxMaintenanceRequest(
        collector="kernel", action=ACTION_REQUEUE_DEAD, confirm=True, path="/v1/x"
    )
    assert OutboxMaintenanceRequest.from_parameters(request.as_parameters()) == request
    with pytest.raises(ValueError, match="confirm"):
        OutboxMaintenanceRequest(collector="kernel", action=ACTION_REQUEUE_DEAD)
    with pytest.raises(CollectorOutboxRefused):
        raise CollectorOutboxRefused("x", {})
