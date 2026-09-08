"""The sink, not each collector, owns "buffered means delivered" (ARCH-G3).

Only the kernel collector used to read ``exc.buffered and exc.replayable`` as
"the record is safe, move on"; the node log collector rolled its journal cursor
back and re-read the same window every poll, and the Fabric Manager collector
pinned its file offset on the first SXID the outbox had already taken. The sink
now answers with a result whose status the collectors branch on.
"""

from __future__ import annotations

from gpu_fault.collectors.sinks import DeliveryStatus, HttpEventSink, deliver_event

from ._support import (
    NOW,
    BufferingSink,
    CollectorError,
    FabricManagerLogCollector,
    KernelLogCollector,
    NodeLogCollector,
    RecordingSink,
    RejectingSink,
    context,
    json,
    subprocess,
)


def test_deliver_event_maps_a_buffered_replayable_failure_to_buffered() -> None:
    result = deliver_event(BufferingSink(), "/events", {"event_id": "e-1"})

    assert result.status is DeliveryStatus.BUFFERED, result
    assert result.buffered is True, "buffered flag does not follow the status"
    assert result.delivered is False, "a buffered record is not a live delivery"
    assert result.error is not None, "the buffering error is kept for the log"


def test_deliver_event_maps_a_rejection_to_failed_without_raising() -> None:
    result = deliver_event(RejectingSink(), "/events", {"event_id": "e-1"})

    assert result.status is DeliveryStatus.FAILED, result
    assert result.failed is True, "failed flag does not follow the status"
    assert result.error is not None and result.error.status_code == 422, (
        "the rejection is not attached to the result"
    )


def test_deliver_event_returns_the_response_on_success() -> None:
    result = deliver_event(RecordingSink(), "/events", {"event_id": "e-1"})

    assert result.status is DeliveryStatus.DELIVERED, result
    assert result.response == {"accepted": True}, "response body was dropped"


def test_http_sink_deliver_returns_buffered_instead_of_raising(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "gpu_fault.collectors.sinks.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreachable")),
    )
    sink = HttpEventSink(
        "https://control", max_attempts=1, outbox_path=str(tmp_path / "outbox.ndjson")
    )

    result = sink.deliver("/events", {"event_id": "e-1"})

    assert result.status is DeliveryStatus.BUFFERED, result
    assert (tmp_path / "outbox.ndjson").exists(), "the record was not buffered"


def test_node_log_collector_advances_the_cursor_when_the_batch_is_buffered(
    tmp_path, monkeypatch
) -> None:
    from gpu_fault.collectors.logs import node as module

    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/journalctl")
    line = json.dumps(
        {
            "__CURSOR": "c-1",
            "__REALTIME_TIMESTAMP": str(int(NOW.timestamp() * 1_000_000) - 1000),
            "MESSAGE": "machine check hardware error",
            "_TRANSPORT": "journal",
        }
    )

    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(list(command), 0, stdout=line, stderr="")

    state = tmp_path / "state.json"
    sink = BufferingSink()
    collector = NodeLogCollector(
        sink,
        context(),
        node_id="worker-1",
        state_path=str(state),
        now=lambda: NOW,
        runner=runner,
    )

    batch = collector.collect_once()

    assert len(batch.entries) == 1, "the matching entry was not batched"
    assert len(sink.requests) == 1, "the batch was not handed to the sink"
    stored = json.loads(state.read_text())
    assert stored["journal_since"] == NOW.isoformat(), (
        "a buffered batch must move the journal cursor like a delivered one"
    )

    collector.collect_once()

    since = calls[-1][calls[-1].index("--since") + 1]
    assert since == f"@{NOW.timestamp()}", (
        "the next poll re-read the window the outbox already took"
    )


def test_fabric_manager_commits_the_checkpoint_when_the_record_is_buffered(
    tmp_path,
) -> None:
    log = tmp_path / "fabricmanager.log"
    state = tmp_path / "fabric-state.json"
    message = (
        "nvidia-nvswitch3: SXid (PCI:0000:c1:00.0): 12020, Fatal, "
        "Link 46 egress sequence ID error"
    )
    log.write_text(message + "\n")
    stat = log.stat()
    state.write_text(
        json.dumps(
            {
                "journal_cursor": None,
                "files": {
                    str(log): {"device": stat.st_dev, "inode": stat.st_ino, "offset": 0}
                },
            }
        )
    )
    sink = BufferingSink()
    collector = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    stats = collector.collect_once()

    assert len(sink.requests) == 1, "the SXID was not handed to the sink"
    assert stats.delivered == 0, "a buffered record must not count as delivered"
    offset = json.loads(state.read_text())["files"][str(log)]["offset"]
    assert offset == len(message) + 1, (
        "a buffered SXID must advance the file offset like a delivered one"
    )


def test_fabric_manager_still_pins_the_checkpoint_on_a_failed_delivery(
    tmp_path,
) -> None:
    log = tmp_path / "fabricmanager.log"
    state = tmp_path / "fabric-state.json"
    log.write_text(
        "nvidia-nvswitch3: SXid (PCI:0000:c1:00.0): 12020, Fatal, Link 46 error\n"
    )
    stat = log.stat()
    state.write_text(
        json.dumps(
            {
                "journal_cursor": None,
                "files": {
                    str(log): {"device": stat.st_dev, "inode": stat.st_ino, "offset": 0}
                },
            }
        )
    )
    collector = FabricManagerLogCollector(
        RejectingSink(),
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    try:
        collector.collect_once()
    except CollectorError as exc:
        assert exc.status_code == 422, "the sink's error was not propagated"
    else:
        raise AssertionError("a failed delivery must still raise")

    assert json.loads(state.read_text())["files"][str(log)]["offset"] == 0, (
        "a failed delivery must not move the checkpoint"
    )


def test_kernel_collector_buffered_behaviour_is_unchanged() -> None:
    sink = BufferingSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="test-boot", now=lambda: NOW
    )

    stats = collector.collect_lines(
        [
            "3,52011,223456789,-;NVRM: Xid (PCI:0000:59:00): 11, name=first\n",
            "3,52012,223456790,-;NVRM: Xid (PCI:0000:59:00): 11, name=second\n",
        ]
    )

    assert stats.observed == 2, "both kmsg records must be read"
    assert stats.delivered == 0, "buffered records are not delivered"
    assert [item[1]["record_id"] for item in sink.requests] == [
        "kmsg-test-boot-52011",
        "kmsg-test-boot-52012",
    ], "collector stopped after the first buffered record"
