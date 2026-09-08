from __future__ import annotations

from tests._builders import asgi_client, build_context, copy_model

from ._support import (
    NOW,
    BufferingSink,
    CollectorError,
    FabricManagerLogCollector,
    KernelLogCollector,
    NodeLogCollector,
    RecordingSink,
    StopTheLoop,
    context,
    io,
    json,
    logging,
    os,
    pytest,
    subprocess,
    timedelta,
)


def _write_fabric_file_state(state, log, *, offset: int) -> None:
    stat = log.stat()
    state.write_text(
        json.dumps(
            {
                "journal_cursor": None,
                "files": {
                    str(log): {
                        "device": stat.st_dev,
                        "inode": stat.st_ino,
                        "offset": offset,
                    }
                },
            }
        )
    )


def test_kernel_collector_filters_and_uses_stable_kmsg_id() -> None:
    sink = RecordingSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    lines = [
        "6,40,1000,-;ordinary kernel message\n",
        ("3,41,1100,-;NVRM: Xid (PCI:0000:b9:00): 94, pid=1234, name=python\n"),
    ]

    first = collector.collect_lines(lines)
    second = collector.collect_lines(lines)

    assert first.observed == 2
    assert first.skipped == 1
    assert first.delivered == 1
    assert second.duplicates == 1
    assert len(sink.requests) == 1
    path, payload = sink.requests[0]
    assert path == "/v1/collector-events/nvidia-kernel"
    assert payload["record_id"] == "kmsg-boot-123-41"
    assert payload["node_id"] == "worker-1"
    assert payload["observed_at"] == NOW.isoformat()
    assert payload["collected_at"] == NOW.isoformat()
    assert payload["source_monotonic_us"] == 1100
    assert payload["source_boot_id"] == "boot-123"


def test_kernel_collector_uses_monotonic_event_time() -> None:
    sink = RecordingSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector._boot_time = NOW - timedelta(seconds=10)

    collector.collect_lines([("3,41,2000000,-;NVRM: Xid (PCI:0000:b9:00): 94")])

    payload = sink.requests[0][1]
    assert payload["observed_at"] == (NOW - timedelta(seconds=8)).isoformat()
    assert payload["collected_at"] == NOW.isoformat()


def test_kernel_collector_keeps_identical_unsequenced_events() -> None:
    sink = RecordingSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    line = "NVRM: Xid (PCI:0000:b9:00): 74"

    first = collector.collect_lines([line])
    second = collector.collect_lines([line])

    assert first.delivered == 1
    assert second.delivered == 1
    assert len(sink.requests) == 2
    assert sink.requests[0][1]["record_id"] != sink.requests[1][1]["record_id"], (
        "unsequenced events must receive distinct fallback record IDs"
    )


def test_kernel_collector_continues_after_retryable_event_is_buffered() -> None:
    class BufferedThenHealthySink:
        def __init__(self) -> None:
            self.requests = []

        def post(self, path, payload):
            self.requests.append((path, payload))
            if len(self.requests) == 1:
                raise CollectorError(
                    "network unavailable", buffered=True, replayable=True
                )
            return {"accepted": True}

    sink = BufferedThenHealthySink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="test-boot", now=lambda: NOW
    )

    stats = collector.collect_lines(
        [
            "3,52011,223456789,-;NVRM: Xid (PCI:0000:59:00): 11, name=first\n",
            "3,52012,223456790,-;NVRM: Xid (PCI:0000:59:00): 11, name=second\n",
        ]
    )

    assert [item[1]["record_id"] for item in sink.requests] == [
        "kmsg-test-boot-52011",
        "kmsg-test-boot-52012",
    ], "collector stopped reading after the first event was buffered"
    assert stats.observed == 2, "collector did not observe both kmsg records"
    assert stats.delivered == 1, "only the healthy delivery should count as delivered"


def test_kernel_collector_starts_at_live_tail(monkeypatch) -> None:
    class TrackingStream(io.StringIO):
        sought_to_end = False

        def seek(self, offset, whence=0):
            if offset == 0 and whence == os.SEEK_END:
                self.sought_to_end = True
            return super().seek(offset, whence)

    old = "3,41,2000000,-;NVRM: Xid (PCI:0000:b9:00): 94\n"
    stream = TrackingStream(old)
    sink = RecordingSink()

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: stream)

    def stop(_seconds: float) -> None:
        raise KeyboardInterrupt

    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: NOW,
        sleep=stop,
    )

    with pytest.raises(KeyboardInterrupt):
        collector.run()

    assert stream.sought_to_end is True
    assert sink.requests == []


def test_kernel_collector_reopens_after_stream_error(monkeypatch) -> None:
    class BrokenStream(io.StringIO):
        def __iter__(self):
            raise OSError("read failed")

    streams = iter(
        [BrokenStream(), io.StringIO("3,42,2000001,-;NVRM: Xid (PCI:0000:b9:00): 94\n")]
    )
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: next(streams))
    sleeps = 0

    def sleep(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise KeyboardInterrupt

    sink = RecordingSink()
    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: NOW,
        start_at_end=False,
        sleep=sleep,
    )

    with pytest.raises(KeyboardInterrupt):
        collector.run()

    assert len(sink.requests) == 1
    assert sink.requests[0][1]["record_id"] == "kmsg-boot-123-42"


def test_kernel_collector_health_summary_uses_routine_path() -> None:
    from gpu_fault.telemetry import CollectorHealthSummary

    sink = RecordingSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector._send_health_summary(NOW)

    path, payload = sink.requests[0]
    assert path == "/v1/collector-events/collector-health"
    assert payload["collector"] == "NVIDIA_KERNEL"
    assert payload["edge_filter_reasons"] == ["health-summary"]
    CollectorHealthSummary.model_validate(payload)


def test_kernel_live_stream_survives_health_summary_failure(
    monkeypatch, caplog
) -> None:
    class FailingRoutineSink:
        def __init__(self) -> None:
            self.requests = []

        def post(self, path, payload):
            if path == "/v1/collector-events/collector-health":
                raise CollectorError("control plane unavailable")
            self.requests.append((path, payload))
            return {}

    class SelectableStream:
        def __init__(self) -> None:
            self.lines = iter([("3,42,2000001,-;NVRM: Xid (PCI:0000:b9:00): 94\n"), ""])

        def fileno(self) -> int:
            return 42

        def readline(self) -> str:
            return next(self.lines)

    ready = iter([([], [], []), ([42], [], []), ([42], [], [])])
    monkeypatch.setattr(
        "gpu_fault.collectors.logs.kernel.select.select", lambda *_args: next(ready)
    )
    sink = FailingRoutineSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector._next_health_summary = NOW

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.logs.kernel"):
        collector._collect_live_stream(SelectableStream())

    assert [path for path, _payload in sink.requests] == [
        "/v1/collector-events/nvidia-kernel"
    ]
    assert "continuing live stream" in caplog.text


def test_fabric_health_failure_advances_next_schedule() -> None:
    class FailingSink:
        def post(self, *_args, **_kwargs):
            raise RuntimeError("control plane unavailable")

    collector = FabricManagerLogCollector(
        FailingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        journal_enabled=False,
    )
    collector._next_health_summary = NOW

    with pytest.raises(RuntimeError, match="unavailable"):
        collector._maybe_health_summary(NOW)

    assert collector._next_health_summary > NOW


def test_fabric_manager_journal_collector_filters_and_resumes(tmp_path) -> None:
    sink = RecordingSink()
    state = tmp_path / "fabric-manager-state.json"
    calls = []
    journal = "\n".join(
        [
            json.dumps(
                {
                    "__CURSOR": "cursor-1",
                    "__REALTIME_TIMESTAMP": "1753012799000000",
                    "_SYSTEMD_UNIT": "kubelet.service",
                    "MESSAGE": "unrelated",
                }
            ),
            json.dumps(
                {
                    "__CURSOR": "cursor-2",
                    "__REALTIME_TIMESTAMP": "1753012800000000",
                    "_SYSTEMD_UNIT": ("nvidia-fabricmanager.service"),
                    "MESSAGE": "Fabric Manager started",
                }
            ),
            json.dumps(
                {
                    "__CURSOR": "cursor-3",
                    "__REALTIME_TIMESTAMP": "1753012800000000",
                    "SYSLOG_IDENTIFIER": "nvidia-fabricmanager",
                    "MESSAGE": (
                        "nvidia-nvswitch3: "
                        "SXid (PCI:0000:c1:00.0): 12020, "
                        "Fatal, Link 46 egress sequence ID error"
                    ),
                }
            ),
        ]
    )

    def first_runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=journal, stderr="")

    first = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        state_path=str(state),
        now=lambda: NOW,
        runner=first_runner,
    )
    stats = first.collect_once()

    assert stats.observed == 2
    assert stats.skipped == 1
    assert stats.delivered == 1
    assert sink.requests[0][0] == ("/v1/collector-events/fabric-manager")
    assert sink.requests[0][1]["record_id"] == ("fm-journal-cursor-3")
    assert sink.requests[0][1]["source"] == "journal"
    assert json.loads(state.read_text())["journal_cursor"] == ("cursor-3")
    assert "--since" in calls[0]

    def second_runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    second = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        state_path=str(state),
        now=lambda: NOW,
        runner=second_runner,
    )

    assert second.collect_once().delivered == 0
    assert calls[1][calls[1].index("--after-cursor") + 1] == ("cursor-3")


def test_fabric_manager_file_collector_persists_offsets(tmp_path) -> None:
    log = tmp_path / "fabricmanager.log"
    state = tmp_path / "fabric-manager-state.json"
    log.write_text(
        "Fabric Manager started\n"
        "nvidia-nvswitch3: "
        "SXid (PCI:0000:c1:00.0): 28006, Non-fatal, "
        "Link 46 MC TS crumbstore MCTO (First)\n"
    )
    sink = RecordingSink()
    first = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    initial_size = log.stat().st_size
    stats = first.collect_once()
    initial_offset = json.loads(state.read_text())["files"][str(log)]["offset"]
    with log.open("a") as stream:
        stream.write(
            "nvidia-nvswitch3: "
            "SXid (PCI:0000:c1:00.0): 12020, Fatal, "
            "Link 46 egress sequence ID error\n"
        )
    second = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )
    resumed = second.collect_once()

    assert stats.observed == 0
    assert stats.delivered == 0
    assert initial_offset == initial_size
    assert resumed.observed == 1
    assert resumed.delivered == 1
    assert len(sink.requests) == 1
    assert sink.requests[0][1]["source"] == "file"
    assert sink.requests[0][1]["fields"]["path"] == str(log)


def test_fabric_manager_file_cursor_rolls_back_on_delivery_failure(tmp_path) -> None:
    log = tmp_path / "fabricmanager.log"
    state = tmp_path / "fabric-manager-state.json"
    log.write_text(
        "nvidia-nvswitch3: "
        "SXid (PCI:0000:c1:00.0): 12020, Fatal, "
        "Link 46 egress sequence ID error\n"
    )
    _write_fabric_file_state(state, log, offset=0)

    class FailingSink:
        def post(self, *_args, **_kwargs):
            raise CollectorError("control plane unavailable")

    collector = FabricManagerLogCollector(
        FailingSink(),
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    with pytest.raises(CollectorError, match="unavailable"):
        collector.collect_once()

    assert json.loads(state.read_text())["files"][str(log)]["offset"] == 0
    sink = RecordingSink()
    retry = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )
    assert retry.collect_once().delivered == 1


def test_fabric_manager_journal_has_no_line_cap_and_accepts_instance_unit(
    tmp_path,
) -> None:
    calls = []
    journal = json.dumps(
        {
            "__CURSOR": "cursor-instance",
            "__REALTIME_TIMESTAMP": "1786622400000000",
            "_SYSTEMD_UNIT": "nvidia-fabricmanager@0.service",
            "MESSAGE": (
                "nvidia-nvswitch0: "
                "SXid (PCI:0000:ab:00.0): 22013, "
                "Non-fatal, Link 12 SAW_MVB error"
            ),
        }
    )

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=journal, stderr="")

    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        state_path=str(tmp_path / "state.json"),
        runner=runner,
    )

    stats = collector.collect_once()

    assert stats.delivered == 1
    assert "--lines=5000" not in calls[0]
    assert "--lines" not in calls[0]
    assert sink.requests[0][1]["unit"] == ("nvidia-fabricmanager@0.service")


def test_fabric_manager_file_preserves_source_timestamp(tmp_path) -> None:
    log = tmp_path / "fabricmanager.log"
    state = tmp_path / "state.json"
    log.write_text(
        "[2026-08-13T12:34:56Z] nvidia-nvswitch0: "
        "SXid (PCI:0000:ab:00.0): 22013, Non-fatal, "
        "Link 12 SAW_MVB error\n"
    )
    _write_fabric_file_state(state, log, offset=0)
    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    collector.collect_once()

    assert sink.requests[0][1]["observed_at"] == ("2026-08-13T12:34:56+00:00")


def test_fabric_manager_initial_file_baselines_at_eof(tmp_path) -> None:
    log = tmp_path / "fabricmanager.log"
    state = tmp_path / "state.json"
    log.write_text(
        ("x" * 1_000_100) + "\n" + "nvidia-nvswitch0: "
        "SXid (PCI:0000:ab:00.0): 22013, Non-fatal, "
        "Link 12 SAW_MVB error marker=COLLECT011-stale\n"
    )
    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    initial_size = log.stat().st_size
    baseline = collector.collect_once()
    with log.open("a") as stream:
        stream.write(
            "nvidia-nvswitch0: "
            "SXid (PCI:0000:ab:00.0): 22014, Non-fatal, "
            "Link 13 fresh event\n"
        )
    delivered = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    ).collect_once()

    assert baseline.observed == 0
    assert baseline.delivered == 0
    assert json.loads(state.read_text())["files"][str(log)]["offset"] > initial_size
    assert delivered.observed == 1
    assert delivered.delivered == 1
    assert len(sink.requests) == 1
    assert sink.requests[0][1]["message"].endswith("fresh event")


def test_fabric_manager_corrupt_state_does_not_replay_existing_file(tmp_path) -> None:
    log = tmp_path / "fabricmanager.log"
    state = tmp_path / "state.json"
    log.write_text(
        "[2026-08-25T03:48:08Z] nvidia-nvswitch0: "
        "SXid (PCI:0000:ab:00.0): 11001, Fatal, "
        "Link 12 marker=COLLECT011-stale\n"
    )
    state.write_text("{not-json")
    sink = RecordingSink()

    stats = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    ).collect_once()

    assert stats.observed == 0
    assert stats.delivered == 0
    assert sink.requests == []
    assert json.loads(state.read_text())["files"][str(log)]["offset"] == (
        log.stat().st_size
    )


def test_fabric_manager_commits_each_successful_record(tmp_path) -> None:
    log = tmp_path / "fabricmanager.log"
    state = tmp_path / "state.json"
    messages = [
        "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 22013, Non-fatal, Link 12 first",
        "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 22014, Non-fatal, Link 13 second",
    ]
    log.write_text("\n".join(messages) + "\n")
    _write_fabric_file_state(state, log, offset=0)

    class FailSecondSink:
        def __init__(self):
            self.calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                raise CollectorError("second delivery failed")
            return {}

    with pytest.raises(CollectorError, match="second delivery"):
        FabricManagerLogCollector(
            FailSecondSink(),
            context(),
            node_id="worker-1",
            journal_enabled=False,
            log_paths=[str(log)],
            state_path=str(state),
            now=lambda: NOW,
        ).collect_once()

    first_offset = json.loads(state.read_text())["files"][str(log)]["offset"]
    assert first_offset == len(messages[0]) + 1
    retry = RecordingSink()
    stats = FabricManagerLogCollector(
        retry,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    ).collect_once()

    assert stats.delivered == 1
    assert len(retry.requests) == 1
    assert retry.requests[0][1]["message"] == messages[1]


def test_fabric_manager_state_fsyncs_file_and_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    log = tmp_path / "fabricmanager.log"
    log.write_text("Fabric Manager started\n")
    fsync_calls = []
    monkeypatch.setattr(
        "gpu_fault.collectors.logs.fabric_manager.os.fsync", fsync_calls.append
    )
    collector = FabricManagerLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(tmp_path / "state.json"),
        now=lambda: NOW,
    )

    collector.collect_once()

    assert len(fsync_calls) == 2


def test_fabric_manager_state_fsyncs_once_per_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    log = tmp_path / "fabricmanager.log"
    log.write_text(
        "\n".join(
            [
                "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): "
                "22013, Non-fatal, Link 12 first",
                "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): "
                "22014, Non-fatal, Link 13 second",
                "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): "
                "22015, Non-fatal, Link 14 third",
            ]
        )
        + "\n"
    )
    state = tmp_path / "state.json"
    _write_fabric_file_state(state, log, offset=0)
    fsync_calls = []
    monkeypatch.setattr(
        "gpu_fault.collectors.logs.fabric_manager.os.fsync", fsync_calls.append
    )
    sink = RecordingSink()
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

    assert stats.delivered == 3
    assert len(fsync_calls) == 2


def test_fabric_manager_skips_state_write_without_new_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    log = tmp_path / "fabricmanager.log"
    log.write_text("")
    state = tmp_path / "state.json"
    _write_fabric_file_state(state, log, offset=0)
    fsync_calls = []
    monkeypatch.setattr(
        "gpu_fault.collectors.logs.fabric_manager.os.fsync", fsync_calls.append
    )
    collector = FabricManagerLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    assert collector.collect_once().observed == 0
    assert fsync_calls == []


def test_fabric_manager_journal_query_filters_units(tmp_path) -> None:
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    FabricManagerLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        state_path=str(tmp_path / "state.json"),
        now=lambda: NOW,
        runner=runner,
    ).collect_once()

    command = calls[0]
    assert "_SYSTEMD_UNIT=nvidia-fabricmanager.service" in command
    assert "_SYSTEMD_UNIT=nv-fabricmanager.service" in command
    assert "SYSLOG_IDENTIFIER=nvidia-fabricmanager" in command
    assert "SYSLOG_IDENTIFIER=nv-fabricmanager" in command
    assert "_COMM=nvidia-fabricma" in command
    assert "_COMM=nv-fabricmanage" in command
    assert command.count("+") == 5


def test_fabric_manager_journal_recovers_from_stale_cursor(tmp_path) -> None:
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"journal_cursor": "gone", "files": {}}))
    record = json.dumps(
        {
            "__CURSOR": "cursor-new",
            "__REALTIME_TIMESTAMP": "1786622400000000",
            "_SYSTEMD_UNIT": "nvidia-fabricmanager.service",
            "MESSAGE": (
                "nvidia-nvswitch0: "
                "SXid (PCI:0000:ab:00.0): 22013, "
                "Non-fatal, Link 12 SAW_MVB error"
            ),
        }
    )
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        if "--after-cursor" in command:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="Failed to seek to cursor: Invalid argument",
            )
        return subprocess.CompletedProcess(command, 0, stdout=record, stderr="")

    sink = RecordingSink()
    stats = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        state_path=str(state),
        now=lambda: NOW,
        runner=runner,
    ).collect_once()

    assert stats.delivered == 1
    assert "--since" in calls[1]
    assert json.loads(state.read_text())["journal_cursor"] == ("cursor-new")


def test_kernel_collector_only_delivers_official_sxid_summary() -> None:
    sink = RecordingSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )

    stats = collector.collect_lines(
        [
            (
                "3,100,1100,-;nvidia-nvswitch3: "
                "SXid (PCI:0000:c1:00.0): 28006, Non-fatal, "
                "Link 46 MC TS crumbstore MCTO (First)"
            ),
            (
                "3,101,1101,-;nvidia-nvswitch3: "
                "SXid (PCI:0000:c1:00.0): 28006, Severity 0 "
                "Engine instance 46 Sub-engine instance 00"
            ),
            (
                "3,102,1102,-;nvidia-nvswitch3: "
                "SXid (PCI:0000:c1:00.0): 28006, "
                "Data {0x00140004, 0x00100000}"
            ),
        ]
    )

    assert stats.observed == 3
    assert stats.delivered == 1
    assert stats.skipped == 2
    assert sink.requests[0][1]["record_id"] == ("kmsg-boot-123-100")


def test_kernel_collector_runs_official_sxid_end_to_end() -> None:
    sink = RecordingSink()
    collector = KernelLogCollector(
        sink,
        copy_model(context(), product="H200"),
        node_id="worker-1",
        boot_id="boot-sxid-e2e",
        now=lambda: NOW,
    )
    collector.collect_lines(
        [
            (
                "3,200,1200,-;nvidia-nvswitch3: "
                "SXid (PCI:0000:c1:00.0): 28006, Non-fatal, "
                "Link 46 MC TS crumbstore MCTO (First)"
            )
        ]
    )
    path, payload = sink.requests[0]

    async def scenario() -> None:
        async with asgi_client(build_context()) as client:
            response = await client.post(path, json=payload)

        assert response.status_code == 200
        body = response.json()
        sxid = body["normalized"]["sxid_events"][0]
        assert sxid["event_source"] == "KERNEL_LOG"
        assert sxid["sxid"] == 28006
        assert sxid["classification"] == "NON_FATAL"
        assert sxid["switch_id"] == "3"
        assert sxid["pci_bdf"] == "0000:c1:00.0"
        assert sxid["port"] == "46"
        assert body["decisions"][0]["disposition"] == "MONITOR_ONLY"

    import asyncio

    asyncio.run(scenario())


def test_kernel_collector_payload_accepts_official_xid_format() -> None:
    sink = RecordingSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-e2e", now=lambda: NOW
    )
    collector.collect_lines(
        ["3,99,1100,-;NVRM: Xid (0000:03:00): 14, Channel 00000001"]
    )
    path, payload = sink.requests[0]

    async def scenario() -> None:
        async with asgi_client(build_context()) as client:
            response = await client.post(path, json=payload)

        assert response.status_code == 200
        body = response.json()
        xid_event = body["normalized"]["xid_events"][0]
        assert xid_event["xid"] == 14
        assert xid_event["pci_bdf"] == "0000:03:00"
        assert body["decisions"][0]["official_action"] == "IGNORE"
        assert body["decisions"][0]["incident_id"]

    import asyncio

    asyncio.run(scenario())


def test_node_log_collector_chunks_training_log_without_loss(tmp_path) -> None:
    log = tmp_path / "training.log"
    log.write_text("".join(f"line {index}\n" for index in range(5)), encoding="utf-8")
    collector = NodeLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        training_log_paths=[str(log)],
        initial_tail_bytes=1024,
        max_entries_per_batch=2,
        max_batch_bytes=4096,
        max_entry_bytes=512,
        now=lambda: NOW,
    )

    first = collector._training_logs(NOW)
    second = collector._training_logs(NOW)
    third = collector._training_logs(NOW)

    assert [item.message for item in first] == ["line 0", "line 1"]
    assert [item.message for item in second] == ["line 2", "line 3"]
    assert [item.message for item in third] == ["line 4"]


def test_node_log_collector_truncates_oversize_entry(tmp_path) -> None:
    log = tmp_path / "training.log"
    log.write_text(
        "machine check hardware error " + "x" * 5000 + "\n", encoding="utf-8"
    )
    sink = RecordingSink()
    collector = NodeLogCollector(
        sink,
        context(),
        node_id="worker-1",
        training_log_paths=[str(log)],
        initial_tail_bytes=6000,
        max_entries_per_batch=10,
        max_batch_bytes=2048,
        max_entry_bytes=512,
        now=lambda: NOW,
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout="", stderr=""
        ),
    )

    batch = collector.collect_once()

    assert len(batch.entries) == 1
    assert "node-log-entry-truncated" in batch.entries[0].message
    assert len(batch.entries[0].message.encode("utf-8")) <= 512
    assert len(json.dumps(sink.requests[0][1]).encode()) < 4096


def test_fabric_manager_skips_a_file_that_vanishes_between_glob_and_stat(
    tmp_path, monkeypatch
) -> None:
    kept = tmp_path / "fabricmanager.log"
    kept.write_text(
        "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 22013, Non-fatal, Link 12 x\n"
    )
    vanishing = tmp_path / "fabricmanager.1.log"
    vanishing.write_text("old\n")
    state = tmp_path / "state.json"
    _write_fabric_file_state(state, kept, offset=0)
    real_stat = os.stat

    def flaky_stat(path, *args, **kwargs):
        if str(path) == str(vanishing):
            raise FileNotFoundError(str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", flaky_stat)
    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(tmp_path / "fabricmanager*.log")],
        state_path=str(state),
        now=lambda: NOW,
    )

    stats = collector.collect_once()

    assert stats.delivered == 1, "a vanished sibling file stopped the whole round"
    assert len(sink.requests) == 1, "the surviving file's SXID was not delivered"


def test_fabric_manager_bounds_the_tracked_file_table(tmp_path, caplog) -> None:
    state = tmp_path / "state.json"
    stale = {
        f"/var/log/fabricmanager.{index}.log": {
            "device": 1,
            "inode": 1000 + index,
            "offset": 10,
        }
        for index in range(5)
    }
    state.write_text(json.dumps({"journal_cursor": None, "files": stale}))
    live = tmp_path / "fabricmanager.log"
    live.write_text("Fabric Manager started\n")
    collector = FabricManagerLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(live)],
        state_path=str(state),
        now=lambda: NOW,
        max_tracked_files=3,
    )

    with caplog.at_level(
        logging.WARNING, logger="gpu_fault.collectors.logs.fabric_manager"
    ):
        collector.collect_once()

    files = json.loads(state.read_text())["files"]
    assert len(files) <= 3, files
    assert str(live) in files, "the live file was evicted instead of a stale one"
    assert "tracked" in caplog.text.lower(), "the eviction was not logged"


def test_fabric_manager_buffered_health_summary_is_not_a_failed_round(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """A health summary the outbox took must not fail the collection round.

    The summary was posted with a bare ``sink.post`` after the batch loop, so
    the ``CollectorError`` it raises once the outbox has taken the record
    escaped ``collect_once`` and the run loop logged the whole round as
    "Fabric Manager log collection failed" -- although every record committed
    and the summary schedule had already advanced.
    """

    from gpu_fault.collectors.logs import fabric_manager as module

    sink = BufferingSink()
    collector = FabricManagerLogCollector(
        sink, context(), node_id="worker-1", journal_enabled=False, now=lambda: NOW
    )
    collector.now = lambda: NOW + timedelta(seconds=600)
    monkeypatch.setattr(
        module.time, "sleep", lambda _seconds: (_ for _ in ()).throw(StopTheLoop())
    )

    with caplog.at_level(
        logging.ERROR, logger="gpu_fault.collectors.logs.fabric_manager"
    ):
        with pytest.raises(StopTheLoop):
            collector.run()

    assert [path for path, _payload in sink.requests] == [
        "/v1/collector-events/collector-health"
    ], "the health summary was not handed to the sink"
    assert "log collection failed" not in caplog.text, (
        "a buffered health summary was reported as a failed collection round"
    )


# --- Fabric Manager record identity, rotation and file-read guards (task 13) ---


def _fabric_sxid_journal(cursor: str) -> str:
    return json.dumps(
        {
            "__CURSOR": cursor,
            "__REALTIME_TIMESTAMP": "1753012800000000",
            "_SYSTEMD_UNIT": "nvidia-fabricmanager.service",
            "MESSAGE": (
                "nvidia-nvswitch3: SXid (PCI:0000:c1:00.0): 12020, "
                "Fatal, Link 46 egress sequence ID error"
            ),
        }
    )


def test_fabric_manager_file_record_ids_are_node_scoped(tmp_path) -> None:
    """Two nodes at the same path and offset must not mint one record id.

    ``hma.py`` scopes the derived ``event_id`` to the cluster only, so node
    uniqueness is the collector's job. Identically imaged nodes share the
    device and inode of ``/var/log/fabricmanager.log``, so ``dev:ino:offset``
    folded node B's SXID into node A's event -- the P0-38A class that
    ``node.py:_entry_identity`` already fixed for training logs.
    """

    log = tmp_path / "fabricmanager.log"
    log.write_text(
        "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 22013, Non-fatal, Link 12\n"
    )
    record_ids: dict[str, str] = {}
    references: dict[str, str] = {}
    for node_id in ("worker-a", "worker-b"):
        state = tmp_path / f"state-{node_id}.json"
        _write_fabric_file_state(state, log, offset=0)
        sink = RecordingSink()
        FabricManagerLogCollector(
            sink,
            context(),
            node_id=node_id,
            boot_id="boot-shared-by-the-image",
            journal_enabled=False,
            log_paths=[str(log)],
            state_path=str(state),
            now=lambda: NOW,
        ).collect_once()
        record_ids[node_id] = sink.requests[0][1]["record_id"]
        references[node_id] = sink.requests[0][1]["evidence_ref"]

    assert record_ids["worker-a"] != record_ids["worker-b"], (
        "two nodes reading the same file offset shared one record id"
    )
    assert all(value.startswith("fm-file-") for value in record_ids.values()), (
        "the record id prefix that names the source was dropped"
    )
    assert "worker-a" in references["worker-a"], (
        "the evidence reference does not say which node the line came from"
    )


def test_fabric_manager_journal_asks_for_untruncated_fields(tmp_path) -> None:
    """Without ``--all`` journalctl nulls any field over 4096 bytes.

    ``_message_text(None)`` is ``""``, which matches no SXID pattern, so a long
    Fabric Manager line was dropped without a trace on both the cold ``--since``
    query and the resumed ``--after-cursor`` one.
    """

    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(command)
        stdout = "" if "--after-cursor" in command else _fabric_sxid_journal("cursor-1")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    collector = FabricManagerLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        state_path=str(tmp_path / "state.json"),
        now=lambda: NOW,
        runner=runner,
    )

    collector.collect_once()
    collector.collect_once()

    assert len(calls) == 2, f"expected a cold and a resumed journal query: {calls}"
    assert all("--all" in command for command in calls), (
        f"a journal query can still truncate fields over 4096 bytes: {calls}"
    )


def test_fabric_manager_reads_the_tail_of_a_renamed_log(tmp_path) -> None:
    """A rotation must not baseline the old inode at EOF under its new name.

    logrotate renames ``fabricmanager.log`` to ``fabricmanager.1.log``; the glob
    then sees the old inode under a name that has no checkpoint, and baselining
    it at EOF threw away every line written since the last poll -- including the
    fatal SXID that made Fabric Manager rotate in the first place.
    """

    log = tmp_path / "fabricmanager.log"
    log.write_text("Fabric Manager successfully configured\n")
    state = tmp_path / "state.json"
    sink = RecordingSink()

    def build() -> FabricManagerLogCollector:
        return FabricManagerLogCollector(
            sink,
            context(),
            node_id="worker-1",
            journal_enabled=False,
            log_paths=[str(tmp_path / "fabricmanager*.log")],
            state_path=str(state),
            now=lambda: NOW,
        )

    build().collect_once()
    with log.open("a") as stream:
        stream.write(
            "nvidia-nvswitch3: SXid (PCI:0000:c1:00.0): 12020, Fatal, "
            "Link 46 written before the rotation\n"
        )
    log.rename(tmp_path / "fabricmanager.1.log")
    log.write_text("")

    stats = build().collect_once()

    assert stats.delivered == 1, (
        "the tail written between the last poll and the rotation was lost"
    )
    assert sink.requests[0][1]["message"].endswith("written before the rotation"), (
        f"the wrong line was delivered: {sink.requests}"
    )
    assert (
        str(tmp_path / "fabricmanager.1.log")
        in (json.loads(state.read_text())["files"])
    ), "the rotated file kept no checkpoint of its own"


def test_fabric_manager_journal_survives_an_unreadable_configured_file(
    tmp_path, monkeypatch, caplog
) -> None:
    """One 0600 log file must not discard the journal SXIDs of the round.

    ``collect_once`` builds ``[*journal, *files]`` before it delivers anything,
    so a ``PermissionError`` from the file tail threw away the journal records
    of that round as well -- every round, permanently, for a file whose mode
    never changes.
    """

    from pathlib import Path

    log = tmp_path / "fabricmanager.log"
    log.write_text("Fabric Manager successfully configured\n")
    state = tmp_path / "state.json"
    _write_fabric_file_state(state, log, offset=0)
    real_open = Path.open

    def refuse(self, *args, **kwargs):
        if str(self) == str(log):
            raise PermissionError(13, "Permission denied", str(log))
        return real_open(self, *args, **kwargs)

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=_fabric_sxid_journal("cursor-1"), stderr=""
        )

    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
        runner=runner,
    )
    monkeypatch.setattr(Path, "open", refuse)

    with caplog.at_level(
        logging.WARNING, logger="gpu_fault.collectors.logs.fabric_manager"
    ):
        stats = collector.collect_once()

    assert stats.delivered == 1, (
        "an unreadable file starved the journal source of the whole round"
    )
    assert sink.requests[0][1]["source"] == "journal", (
        f"the journal SXID was not the delivered record: {sink.requests}"
    )
    assert "unreadable" in caplog.text.lower(), (
        "the unreadable file was skipped without a warning"
    )


def test_fabric_manager_file_offsets_are_bytes_and_resync_to_a_line_boundary(
    tmp_path, caplog
) -> None:
    """Offsets are byte offsets, and a stored one mid-character re-syncs once.

    The offsets used to be ``TextIOWrapper.tell()`` cookies, so a stored value
    could land inside a multibyte character; ``seek(offset - 1)`` then read a
    replacement character instead of the newline. The read resumes at the next
    line boundary, says so once, and the line that follows is delivered exactly
    once.
    """

    first = "光纤管理器启动 [2026-08-13T12:34:56Z] configured\n"
    second = (
        "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 22013, Non-fatal, "
        "Link 12 SAW_MVB error\n"
    )
    log = tmp_path / "fabricmanager.log"
    log.write_bytes((first + second).encode("utf-8"))
    state = tmp_path / "state.json"
    # Two bytes into the first character: a legacy cookie, not a line boundary.
    _write_fabric_file_state(state, log, offset=2)
    sink = RecordingSink()

    def build() -> FabricManagerLogCollector:
        return FabricManagerLogCollector(
            sink,
            context(),
            node_id="worker-1",
            journal_enabled=False,
            log_paths=[str(log)],
            state_path=str(state),
            now=lambda: NOW,
        )

    with caplog.at_level(
        logging.WARNING, logger="gpu_fault.collectors.logs.fabric_manager"
    ):
        stats = build().collect_once()
    resumed = build().collect_once()

    assert stats.delivered == 1, f"the SXID after the resync was lost: {sink.requests}"
    assert sink.requests[0][1]["fields"]["offset"] == str(len(first.encode("utf-8"))), (
        "the record offset is not the byte offset of the line"
    )
    assert json.loads(state.read_text())["files"][str(log)]["offset"] == (
        log.stat().st_size
    ), "the committed offset is not a byte count"
    assert "line boundary" in caplog.text, (
        "resuming inside a line was not reported once"
    )
    assert resumed.observed == 0, "the same line was read twice"


def test_fabric_manager_keeps_its_offset_at_a_line_the_daemon_is_still_writing(
    tmp_path, caplog
) -> None:
    """A resume boundary inside an unterminated line must not be consumed.

    The resync branch skipped to the next newline and committed
    ``stream.tell()`` even when the "line" it skipped had no newline at all --
    it was the tail of a record the daemon was still writing. The bytes were
    consumed while the line was still incomplete, so the completed line was
    skipped a second time on the next round, and the one-per-collector warning
    never said so again. Only a complete line may advance the checkpoint, which
    is the invariant the main read loop already holds.
    """

    from pathlib import Path

    opening = "Fabric Manager daemon started, build 550.90.07\n"
    fatal = (
        "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 22013, Fatal, Link 12 SAW_MVB error"
    )
    log = tmp_path / "fabricmanager.log"
    log.write_bytes((opening + fatal).encode("utf-8"))
    state = tmp_path / "state.json"
    # A legacy cookie that landed inside the line still being written.
    _write_fabric_file_state(state, log, offset=len(opening.encode("utf-8")) + 10)
    sink = RecordingSink()

    def build() -> FabricManagerLogCollector:
        return FabricManagerLogCollector(
            sink,
            context(),
            node_id="worker-1",
            journal_enabled=False,
            log_paths=[str(log)],
            state_path=str(state),
            now=lambda: NOW,
        )

    with caplog.at_level(
        logging.WARNING, logger="gpu_fault.collectors.logs.fabric_manager"
    ):
        first_round = build().collect_once()
        held = json.loads(state.read_text())["files"][str(log)]["offset"]
        # The daemon finishes the fatal line and logs the next one.
        with Path(log).open("ab") as stream:
            stream.write(
                b"\nnvidia-nvswitch3: SXid (PCI:0000:c1:00.0): 12020, Fatal, "
                b"Link 46 egress sequence ID error\n"
            )
        second_round = build().collect_once()

    assert first_round.delivered == 0, (
        f"an incomplete line was delivered as a record: {sink.requests}"
    )
    assert held == len(opening.encode("utf-8")) + 10, (
        "the checkpoint advanced over a line the daemon was still writing, so "
        f"the completed line is skipped again: {held}"
    )
    assert second_round.delivered == 1, (
        f"the fatal SXID after the completed line was lost: {sink.requests}"
    )
    assert "12020" in sink.requests[0][1]["message"], (
        f"the delivered record is not the fatal SXID: {sink.requests}"
    )
    assert caplog.text.count("line boundary") == 1, (
        f"the skipped line was reported {caplog.text.count('line boundary')} times"
    )


def test_fabric_manager_decodes_a_binary_journal_message(tmp_path) -> None:
    """``journalctl --output=json`` emits MESSAGE as an array of byte values.

    Any field that is not valid UTF-8 arrives as ``[110, 118, ...]``, and
    ``str()`` of that list matches no SXID pattern, so the line was dropped
    without a trace -- the same hole ``node.py:_message_text`` already closed
    for training logs, and reachable here now that ``--all`` stops journalctl
    from nulling long fields.
    """

    text = (
        "nvidia-nvswitch3: SXid (PCI:0000:c1:00.0): 12020, Fatal, "
        "Link 46 egress sequence ID error"
    )
    entry = json.dumps(
        {
            "__CURSOR": "s=binary;i=1",
            "__REALTIME_TIMESTAMP": "1753012800000000",
            "_SYSTEMD_UNIT": "nvidia-fabricmanager.service",
            "MESSAGE": list(text.encode("utf-8")) + [255],
        }
    )
    sink = RecordingSink()
    stats = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        log_paths=[],
        state_path=str(tmp_path / "state.json"),
        now=lambda: NOW,
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=entry + "\n", stderr=""
        ),
    ).collect_once()

    assert stats.delivered == 1, (
        f"a binary MESSAGE field dropped the fatal SXID: {sink.requests}"
    )
    assert text in sink.requests[0][1]["message"], (
        f"the MESSAGE byte array was stringified instead of decoded: {sink.requests}"
    )


def test_fabric_manager_reports_a_stall_inside_an_unfinished_record(
    tmp_path, caplog
) -> None:
    """Holding the offset must not look like a healthy collector.

    Nothing is read from the file until the daemon terminates the line, and a
    daemon killed mid-append never will. The hold was silent, so an operator saw
    a collector reporting success and no SXIDs; it now says so once per file and
    offset instead of every round.
    """

    opening = "Fabric Manager daemon started, build 550.90.07\n"
    torn = "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 22013, Fatal, Link 12 SAW_MVB"
    log = tmp_path / "fabricmanager.log"
    log.write_bytes((opening + torn).encode("utf-8"))
    state = tmp_path / "state.json"
    offset = len(opening.encode("utf-8")) + 10
    _write_fabric_file_state(state, log, offset=offset)
    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink,
        context(),
        node_id="worker-1",
        journal_enabled=False,
        log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    with caplog.at_level(
        logging.WARNING, logger="gpu_fault.collectors.logs.fabric_manager"
    ):
        first = collector.collect_once()
        second = collector.collect_once()

    assert (first.delivered, second.delivered) == (0, 0), (
        f"an unfinished record was delivered: {sink.requests}"
    )
    assert caplog.text.count("has not finished writing") == 1, (
        f"the stalled read was reported {caplog.text.count('has not finished writing')}"
        f" time(s) instead of once: {caplog.text}"
    )
    assert str(log) in caplog.text and str(offset) in caplog.text, (
        f"the warning does not say which file and offset are stuck: {caplog.text}"
    )
