from __future__ import annotations

from ._support import (
    NOW,
    BufferingSink,
    CollectorError,
    HostTelemetryCollector,
    RecordingSink,
    RejectingSink,
    context,
    json,
    pytest,
    rank_liveness_collector,
    rank_liveness_cycle,
    subprocess,
    timedelta,
    write_fake_rank,
)


@pytest.mark.parametrize(("token", "expected"), [("1", True), ("off", False)])
def test_host_edge_filter_switch_reads_every_token(
    monkeypatch: pytest.MonkeyPatch, token: str, expected: bool
) -> None:
    """``GPU_FAULT_HOST_EDGE_FILTER_ENABLED=1`` used to switch the filter off."""

    monkeypatch.setenv("GPU_FAULT_HOST_EDGE_FILTER_ENABLED", token)
    monkeypatch.setenv("GPU_FAULT_RANK_LIVENESS_ENABLED", token)

    collector = HostTelemetryCollector(RecordingSink(), context(), node_id="worker-1")

    assert collector.edge_filter_enabled is expected
    assert collector.rank_liveness_enabled is expected


def test_host_collector_exposes_targeted_rdma_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = HostTelemetryCollector(RecordingSink(), context(), node_id="worker-1")
    expected = [collector._sample("rdma_probe", 1)]
    monkeypatch.setattr(
        collector, "_rdma", lambda observed_at: expected if observed_at == NOW else []
    )

    assert collector.collect_rdma_samples(NOW) == expected


def test_host_collector_disk_latency_tcp_and_rdma_deltas(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sink = RecordingSink()
    collector = HostTelemetryCollector(
        sink, context(), node_id="worker-1", now=lambda: NOW
    )
    snapshots = {
        "/proc/diskstats": [
            "8 0 nvme0n1 10 0 0 20 20 0 0 40 0 50 60\n",
            "8 0 nvme0n1 15 0 0 30 25 0 0 50 0 70 100\n",
        ],
        "/proc/net/snmp": ["Tcp: RetransSegs\nTcp: 7\n", "Tcp: RetransSegs\nTcp: 10\n"],
    }
    reads = {path: iter(values) for path, values in snapshots.items()}
    original_read_text = type(tmp_path).read_text

    def read_text(path, *args, **kwargs):
        key = str(path)
        if key in reads:
            return next(reads[key])
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "read_text", read_text)
    collector._diskstats(NOW)
    collector._tcp(NOW)
    later = NOW.replace(second=15)
    disk = collector._diskstats(later)
    tcp = collector._tcp(later)

    assert next(item.value for item in disk if item.name == "disk_io_await_ms") == 4
    assert tcp[0].name == "tcp_retransmits_delta"
    assert tcp[0].value == 3


def test_host_edge_filter_suppresses_health_and_delivers_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = RecordingSink()
    times = iter(
        [
            NOW,
            NOW + timedelta(seconds=15),
            NOW + timedelta(seconds=30),
            NOW + timedelta(seconds=45),
            NOW + timedelta(seconds=345),
        ]
    )
    values = iter([20.0, 20.0, 99.0, 20.0, 20.0])
    collector = HostTelemetryCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        health_summary_seconds=300,
        history_max_points=3,
    )
    for name in HostTelemetryCollector.CONTRIBUTORS:
        monkeypatch.setattr(collector, name, lambda _observed_at: [])
    monkeypatch.setattr(
        collector,
        "_cpu",
        lambda _observed_at: [
            collector._sample("cpu_usage_percent", next(values), "percent")
        ],
    )

    for _ in range(5):
        collector.collect_once()

    assert len(sink.requests) == 4
    payloads = [payload for _, payload in sink.requests]
    assert payloads[0]["edge_filter_reasons"] == ["baseline"]
    assert payloads[1]["edge_filter_reasons"] == ["threshold:cpu_usage_percent"]
    assert payloads[2]["edge_filter_reasons"] == ["recovered"]
    assert payloads[3]["edge_filter_reasons"] == ["health-summary"]
    assert payloads[3]["context_history"] == []


def test_host_edge_filter_suppresses_persistent_edge_until_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = RecordingSink()
    times = iter(
        [
            NOW,
            NOW + timedelta(seconds=15),
            NOW + timedelta(seconds=30),
            NOW + timedelta(seconds=315),
        ]
    )
    values = iter([20.0, 99.0, 99.0, 99.0])
    collector = HostTelemetryCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        health_summary_seconds=300,
    )
    for name in HostTelemetryCollector.CONTRIBUTORS:
        monkeypatch.setattr(collector, name, lambda _observed_at: [])
    monkeypatch.setattr(
        collector,
        "_cpu",
        lambda _observed_at: [
            collector._sample("cpu_usage_percent", next(values), "percent")
        ],
    )

    for _ in range(4):
        collector.collect_once()

    assert [payload["edge_filter_reasons"] for _, payload in sink.requests] == [
        ["baseline"],
        ["threshold:cpu_usage_percent"],
        ["health-summary"],
    ]


def test_efa_network_counter_is_assigned_to_one_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    device = tmp_path / "rdmap0"
    net = device / "device" / "net" / "eth0"
    net.mkdir(parents=True)
    (device / "device" / "uevent").write_text("DRIVER=efa\n")
    values = iter([10, 15])

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv, 0, stdout=f"rnr_retry_err: {next(values)}\n", stderr=""
        )

    monkeypatch.setattr(
        "gpu_fault.collectors.host.network.shutil.which",
        lambda command: "/usr/sbin/ethtool" if command == "ethtool" else None,
    )
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        infiniband_root=str(tmp_path),
        runner=runner,
    )

    collector._efa_network(NOW)
    samples = collector._efa_network(NOW + timedelta(seconds=15))
    by_name = {item.name: item.value for item in samples}

    assert by_name["efa_rnr_errors_delta"] == 5
    assert by_name["efa_retry_errors_delta"] == 0


def test_host_collector_reports_memory_and_page_cache_percentages(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    original_read_text = type(tmp_path).read_text
    meminfo = """\
MemTotal:       1000000 kB
MemFree:          20000 kB
MemAvailable:     50000 kB
Cached:          900000 kB
SReclaimable:     20000 kB
Shmem:            10000 kB
SwapTotal:            0 kB
SwapFree:             0 kB
"""

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return meminfo
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "read_text", read_text)
    collector = HostTelemetryCollector(RecordingSink(), context(), node_id="worker-1")

    samples = {item.name: item for item in collector._memory(NOW)}

    assert samples["memory_free_percent"].value == 2
    assert samples["memory_available_percent"].value == 5
    assert samples["page_cache_percent"].value == 91
    assert samples["page_cache_bytes"].value == 910_000 * 1024


def test_rank_liveness_reports_checkpoint_writes_as_progress(tmp_path) -> None:
    # A checkpoint streamed to FSx moves ``wchar`` without touching
    # ``write_bytes``; the GPUs stay busy because the collective that
    # follows is already queued.
    collector = rank_liveness_collector(tmp_path, gpu_utilization=99)
    for pid in (1234, 5678):
        write_fake_rank(tmp_path, pid, cpu_ticks=100, write_bytes=0, wchar=0)
    assert rank_liveness_cycle(collector, NOW)["training_rank_process_count"] == 2

    write_fake_rank(
        tmp_path, 1234, cpu_ticks=140, write_bytes=0, wchar=900 * 1024 * 1024
    )
    write_fake_rank(tmp_path, 5678, cpu_ticks=140, write_bytes=0, wchar=0)
    values = rank_liveness_cycle(collector, NOW + timedelta(seconds=15))

    assert values["training_rank_advancing_count"] == 1
    assert values["training_rank_seconds_since_progress"] == 0
    assert values["training_rank_write_chars_delta"] == 900 * 1024 * 1024


def test_rank_liveness_treats_spinning_ranks_as_stalled(tmp_path) -> None:
    # A wedged collective still burns host CPU in the NCCL wait and
    # keeps the SMs busy, so CPU time alone must not count as progress.
    collector = rank_liveness_collector(tmp_path, gpu_utilization=100)
    for pid in (1234, 5678):
        write_fake_rank(tmp_path, pid, cpu_ticks=100, write_bytes=0, wchar=0)
    rank_liveness_cycle(collector, NOW)
    for pid in (1234, 5678):
        write_fake_rank(tmp_path, pid, cpu_ticks=1_600, write_bytes=0, wchar=0)
    values = rank_liveness_cycle(collector, NOW + timedelta(seconds=15))

    assert values["training_rank_process_count"] == 2
    assert values["training_rank_advancing_count"] == 0
    assert values["training_rank_cpu_ticks_delta"] == 3_000
    assert "training_rank_seconds_since_progress" not in values


def test_rank_liveness_drops_counters_of_departed_ranks(tmp_path) -> None:
    collector = rank_liveness_collector(tmp_path, gpu_utilization=50)
    for pid in (1234, 5678):
        write_fake_rank(tmp_path, pid, cpu_ticks=100, write_bytes=0, wchar=0)
    rank_liveness_cycle(collector, NOW)
    assert any(key.startswith("rank/5678/") for key in collector._previous)

    collector.runner = lambda argv, **_kwargs: (
        subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "1234\n"
                if any("compute-apps" in str(i) for i in argv)
                else "GPU-a, 50\n"
            ),
            stderr="",
        )
    )
    rank_liveness_cycle(collector, NOW + timedelta(seconds=15))

    assert not any(key.startswith("rank/5678/") for key in collector._previous)

    # The attempt ends and no GPU process is left. A node keeps this
    # collector running for weeks across many attempts, so the last
    # ranks must not stay in the baseline either.
    collector.runner = lambda argv, **_kwargs: (
        subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "" if any("compute-apps" in str(i) for i in argv) else "GPU-a, 50\n"
            ),
            stderr="",
        )
    )
    assert rank_liveness_cycle(collector, NOW + timedelta(seconds=30)) == {}
    assert not any(key.startswith("rank/") for key in collector._previous)


def test_host_collector_distinguishes_local_and_remote_mounts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    original_read_text = type(tmp_path).read_text
    mountinfo = """\
36 25 259:1 / / rw,relatime - xfs /dev/nvme0n1p1 rw
40 36 0:42 / /fsx rw,relatime - lustre 10.0.0.1@tcp:/fsx rw
"""

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return mountinfo
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "read_text", read_text)

    assert HostTelemetryCollector._filesystem_type("/var") == "xfs"
    assert HostTelemetryCollector._filesystem_type("/fsx/checkpoints") == "lustre"


def test_host_collector_collects_efa_traffic_rate(tmp_path) -> None:
    counters = tmp_path / "rdmap0" / "ports" / "1" / "hw_counters"
    counters.mkdir(parents=True)
    device = counters.parents[2] / "device"
    device.mkdir()
    (device / "uevent").write_text("DRIVER=efa\n")
    port = counters.parent
    (port / "state").write_text("4: ACTIVE\n")
    (port / "phys_state").write_text("5: LinkUp\n")
    for name, value in {
        "rx_bytes": 1_000,
        "tx_bytes": 2_000,
        "rdma_read_bytes": 500,
        "rdma_write_bytes": 700,
        "send_bytes": 800,
        "recv_bytes": 900,
        "rx_pkts": 10,
        "tx_pkts": 20,
    }.items():
        (counters / name).write_text(str(value))

    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        infiniband_root=str(tmp_path),
    )
    collector._rdma(NOW)
    for name, value in {
        "rx_bytes": 2_500,
        "tx_bytes": 5_000,
        "rdma_read_bytes": 900,
        "rdma_write_bytes": 1_200,
        "send_bytes": 1_400,
        "recv_bytes": 1_600,
        "rx_pkts": 25,
        "tx_pkts": 50,
    }.items():
        (counters / name).write_text(str(value))

    samples = collector._rdma(NOW + timedelta(seconds=15))
    by_name = {item.name: item for item in samples}

    assert by_name["efa_rx_bytes_delta"].value == 1_500
    assert by_name["efa_tx_bytes_delta"].value == 3_000
    assert by_name["efa_traffic_bytes_delta"].value == 4_500
    assert by_name["efa_traffic_bytes_per_second"].value == 300


def test_host_collector_treats_efa_counter_reset_as_zero(tmp_path) -> None:
    counters = tmp_path / "rdmap0" / "ports" / "1" / "hw_counters"
    counters.mkdir(parents=True)
    device = counters.parents[2] / "device"
    device.mkdir()
    (device / "uevent").write_text("DRIVER=efa\n")
    (counters / "rx_bytes").write_text("1000")
    (counters / "tx_bytes").write_text("2000")
    collector = HostTelemetryCollector(
        RecordingSink(), context(), node_id="worker-1", infiniband_root=str(tmp_path)
    )
    collector._rdma(NOW)
    (counters / "rx_bytes").write_text("10")
    (counters / "tx_bytes").write_text("20")

    samples = collector._rdma(NOW + timedelta(seconds=15))

    assert (
        next(item.value for item in samples if item.name == "efa_traffic_bytes_delta")
        == 0
    )


def test_efa_inventory_distinguishes_unbound_driver_from_missing_pci(tmp_path) -> None:
    infiniband_root = tmp_path / "infiniband"
    pci_root = tmp_path / "pci"
    for index in range(2):
        pci_device = pci_root / f"0000:0{index}:00.0"
        pci_device.mkdir(parents=True)
        (pci_device / "vendor").write_text("0x1d0f\n")
        (pci_device / "device").write_text("0xefa2\n")
    port = infiniband_root / "efa_0" / "ports" / "1"
    port.mkdir(parents=True)
    device = port.parents[1] / "device"
    device.mkdir()
    (device / "uevent").write_text("DRIVER=efa\n")
    (port / "state").write_text("4: ACTIVE\n")
    (port / "phys_state").write_text("5: LinkUp\n")
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        expected_efa_device_count=2,
        inventory_mismatch_consecutive_samples=1,
        infiniband_root=str(infiniband_root),
        pci_devices_root=str(pci_root),
    )

    by_name = {item.name: item for item in collector._efa_inventory(NOW)}

    assert by_name["efa_inventory_discovered_count"].value == 2
    assert by_name["efa_inventory_driver_bound_count"].value == 1
    assert by_name["efa_inventory_active_count"].value == 1
    assert by_name["efa_inventory_mismatch"].value == 1
    assert by_name["efa_inventory_mismatch"].labels["failure_mode"] == "DRIVER_UNBOUND"


def test_efa_inventory_ignores_non_efa_rdma_devices(tmp_path) -> None:
    for name, driver in (("rdmap0", "efa"), ("mlx5_0", "mlx5_core")):
        port = tmp_path / name / "ports" / "1"
        port.mkdir(parents=True)
        device = port.parents[1] / "device"
        device.mkdir()
        (device / "uevent").write_text(f"DRIVER={driver}\n")
        (port / "state").write_text("4: ACTIVE\n")
        (port / "phys_state").write_text("5: LinkUp\n")
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        expected_efa_device_count=1,
        infiniband_root=str(tmp_path),
    )

    by_name = {item.name: item for item in collector._efa_inventory(NOW)}

    assert by_name["efa_inventory_discovered_count"].value == 1
    assert by_name["efa_inventory_active_count"].value == 1
    assert by_name["efa_inventory_mismatch"].value == 0


def test_rdma_counters_ignore_non_efa_devices(tmp_path) -> None:
    for name, driver in (("rdmap0", "efa"), ("mlx5_0", "mlx5_core")):
        port = tmp_path / name / "ports" / "1"
        counters = port / "counters"
        counters.mkdir(parents=True)
        device = port.parents[1] / "device"
        device.mkdir()
        (device / "uevent").write_text(f"DRIVER={driver}\n")
        (port / "state").write_text("4: ACTIVE\n")
        (port / "phys_state").write_text("5: LinkUp\n")
        (counters / "symbol_error").write_text("1\n")
    collector = HostTelemetryCollector(
        RecordingSink(), context(), node_id="worker-1", infiniband_root=str(tmp_path)
    )

    samples = collector._rdma(NOW)

    assert samples
    assert {item.device for item in samples} == {"rdmap0/1"}


def test_bmc_non_critical_status_is_not_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gpu_fault.collectors.host.system_metrics.shutil.which",
        lambda command: "/usr/bin/ipmitool" if command == "ipmitool" else None,
    )

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=("Temp | 32 | nc\nFan | 0 | cr\nPower | 1 | ok\n"),
            stderr="",
        )

    collector = HostTelemetryCollector(
        RecordingSink(), context(), node_id="worker-1", runner=runner
    )

    samples = collector._bmc(NOW)

    assert len(samples) == 1
    assert samples[0].value == 1


def test_smart_scan_ignores_blank_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "gpu_fault.collectors.host.system_metrics.shutil.which",
        lambda command: "/usr/sbin/smartctl" if command == "smartctl" else None,
    )

    def runner(argv, **_kwargs):
        if argv == ["smartctl", "--scan-open"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout="\n/dev/nvme0 -d nvme\n\n", stderr=""
            )
        return subprocess.CompletedProcess(
            argv, 0, stdout='{"smart_status":{"passed":true}}', stderr=""
        )

    collector = HostTelemetryCollector(
        RecordingSink(), context(), node_id="worker-1", runner=runner
    )

    samples = collector._smart(NOW)

    assert len(samples) == 1
    assert samples[0].device == "/dev/nvme0"
    assert samples[0].value == 0


def test_host_collector_queries_normalized_nvswitch_topology() -> None:
    payload = {
        "ports": [
            {
                "switch_id": "nvidia-nvswitch0",
                "port": 18,
                "link_scope": "access",
                "peer_type": "GPU",
                "gpu_uuid": "GPU-a",
                "fabric_partition": "fabric-a",
            }
        ]
    }

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["topology-query"], returncode=0, stdout=json.dumps(payload), stderr=""
        )

    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        runner=runner,
        nvswitch_topology_command=["topology-query", "--json"],
    )

    samples = collector._nvswitch_topology(NOW)

    assert len(samples) == 1
    assert samples[0].name == "nvswitch_port_topology"
    assert samples[0].device == "nvidia-nvswitch0/18"
    assert samples[0].labels == {
        "trusted": "true",
        "source": "nvswitch-topology-query",
        "switch_id": "nvidia-nvswitch0",
        "port": "18",
        "link_scope": "ACCESS",
        "peer_type": "GPU",
        "gpu_uuid": "GPU-a",
        "fabric_partition": "fabric-a",
    }


def test_host_batch_the_outbox_took_advances_the_edge_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A buffered batch is delivered as far as the edge filter is concerned.

    ``HttpEventSink.post`` writes the record to the durable outbox and *then*
    raises, so the bookkeeping after the post was skipped: the batch stayed at
    "never delivered", every following tick re-satisfied the summary/baseline
    condition, blocked ~47 s in the retry ladder and buffered another 100-200 KB
    batch with a fresh ``batch_id`` until the outbox evicted (ARCH-G3).
    """

    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ())
    sink = BufferingSink()
    times = iter([NOW, NOW + timedelta(seconds=15)])
    collector = HostTelemetryCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        health_summary_seconds=300,
    )

    collector.collect_once()
    collector.collect_once()

    assert [payload["edge_filter_reasons"] for _, payload in sink.requests] == [
        ["baseline"]
    ], "an unchanged tick re-buffered a batch the outbox had already taken"


def test_host_batch_the_control_plane_rejected_is_redelivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected batch went nowhere, so the next tick must try again."""

    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ())
    sink = RejectingSink()
    times = iter([NOW, NOW + timedelta(seconds=15)])
    collector = HostTelemetryCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        health_summary_seconds=300,
    )

    for _ in range(2):
        with pytest.raises(CollectorError, match="rejected"):
            collector.collect_once()

    assert [payload["edge_filter_reasons"] for _, payload in sink.requests] == [
        ["baseline"],
        ["baseline"],
    ], "a rejected batch must not be treated as delivered"


def _smart_runner(devices: list[str], passed: list[bool], calls: list[list[str]]):
    """A ``smartctl`` that records every fork it is asked to make.

    ``devices`` and ``passed`` are read on every call, so a test can hot-add a
    drive or flip a health verdict between ticks.
    """

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        if argv[1:] == ["--scan-open"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="".join(f"{device} -d nvme\n" for device in devices),
                stderr="",
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"smart_status": {"passed": passed[0]}}),
            stderr="",
        )

    return runner


def _smart_collector(
    monkeypatch: pytest.MonkeyPatch, runner, times: list
) -> HostTelemetryCollector:
    monkeypatch.setattr(
        "gpu_fault.collectors.host.system_metrics.shutil.which",
        lambda command: "/usr/sbin/smartctl" if command == "smartctl" else None,
    )
    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ("_smart",))
    stamps = iter(times)
    return HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: next(stamps),
        runner=runner,
    )


def _health_checks(calls: list[list[str]]) -> list[list[str]]:
    return [item for item in calls if "-H" in item]


def test_smart_health_is_read_once_per_cache_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SMART health moves in days; the tick paid for it every 15 s.

    Eight NVMe on a p5 is nine ``smartctl`` forks with 20 s timeouts each, a
    180 s worst case inside a 15 s tick (F-H7).
    """

    calls: list[list[str]] = []
    collector = _smart_collector(
        monkeypatch,
        _smart_runner(["/dev/nvme0", "/dev/nvme1"], [True], calls),
        [NOW, NOW + timedelta(seconds=15)],
    )

    collector.collect_once()
    second = collector.collect_once()

    assert len(_health_checks(calls)) == 2, (
        f"the per-device health check re-forked on the next tick: {calls}"
    )
    assert sorted(item.device for item in second.samples) == [
        "/dev/nvme0",
        "/dev/nvme1",
    ], "a cached verdict must still be reported on every tick"


def test_a_hot_added_drive_is_checked_before_the_cache_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache is keyed on the device set, so a new drive is never invisible."""

    calls: list[list[str]] = []
    devices = ["/dev/nvme0"]
    collector = _smart_collector(
        monkeypatch,
        _smart_runner(devices, [True], calls),
        [NOW, NOW + timedelta(seconds=15)],
    )

    collector.collect_once()
    devices.append("/dev/nvme1")
    second = collector.collect_once()

    assert len(_health_checks(calls)) == 3, (
        f"the added drive was hidden by the cached verdict of the old set: {calls}"
    )
    assert sorted(item.device for item in second.samples) == [
        "/dev/nvme0",
        "/dev/nvme1",
    ], "the hot-added drive reported no health sample"


def test_a_failing_drive_is_reported_within_one_cache_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """300 s is the whole staleness the cache may add to a health *change*."""

    calls: list[list[str]] = []
    passed = [True]
    collector = _smart_collector(
        monkeypatch,
        _smart_runner(["/dev/nvme0"], passed, calls),
        [NOW, NOW + timedelta(seconds=15), NOW + timedelta(seconds=300)],
    )

    collector.collect_once()
    passed[0] = False
    stale = collector.collect_once()
    fresh = collector.collect_once()

    assert [item.value for item in stale.samples] == [0], (
        "the cached verdict is what a tick inside the window reports"
    )
    assert [item.value for item in fresh.samples] == [1], (
        f"the failing drive outlived the 300 s cache bound: {calls}"
    )


def _efa_port(root, name: str = "rdmap0", interface: str = "eth0"):
    """One EFA port with an error counter and byte counters, all quiet."""

    port = root / name / "ports" / "1"
    (port / "counters").mkdir(parents=True)
    (port / "hw_counters").mkdir()
    device = port.parents[1] / "device"
    (device / "net" / interface).mkdir(parents=True)
    (device / "uevent").write_text("DRIVER=efa\n")
    (port / "state").write_text("4: ACTIVE\n")
    (port / "phys_state").write_text("5: LinkUp\n")
    (port / "counters" / "symbol_error").write_text("0\n")
    (port / "hw_counters" / "rx_drops").write_text("0\n")
    (port / "hw_counters" / "rx_bytes").write_text("1000\n")
    (port / "hw_counters" / "tx_bytes").write_text("2000\n")
    return port


def _quiet_interface(root, name: str = "eth0"):
    """One physical interface whose error and drop counters never move."""

    statistics = root / name / "statistics"
    statistics.mkdir(parents=True)
    for counter in ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped"):
        (statistics / counter).write_text("7\n")
    (root / name / "device").mkdir()
    (root / name / "operstate").write_text("up\n")
    return root / name


def _quiet_ethtool_runner(argv, **_kwargs):
    """``ethtool -S`` whose EFA congestion and error counters never move."""

    return subprocess.CompletedProcess(
        argv,
        0,
        stdout=(
            "NIC statistics:\n"
            "     rnr_naks: 3\n"
            "     retry_count: 4\n"
            "     cq_err: 5\n"
            "     pfc_pause_tx: 6\n"
            "     ecn_marked: 7\n"
        ),
        stderr="",
    )


def test_a_quiet_tick_ships_no_zero_valued_per_port_deltas(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """~500 zero samples (~100 KB of JSON) per batch on an idle p5en (F-H8).

    The deltas the consumer reads as latest values still ship at zero: that is
    the only way a fabric validation and the sustained-drop rule can clear. All
    three producers are driven here, because the kept list is a fail-closed
    contract with ``NodeHealthPolicy`` and not a property of one of them.
    """

    monkeypatch.setattr(
        HostTelemetryCollector, "CONTRIBUTORS", ("_network", "_rdma", "_efa_network")
    )
    monkeypatch.setattr(
        "gpu_fault.collectors.host.network.shutil.which",
        lambda command: "/usr/sbin/ethtool" if command == "ethtool" else None,
    )
    infiniband = tmp_path / "infiniband"
    net_class = tmp_path / "net"
    _efa_port(infiniband)
    _quiet_interface(net_class)
    times = iter([NOW, NOW + timedelta(seconds=15)])
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        infiniband_root=str(infiniband),
        net_class_root=str(net_class),
        required_interfaces=["eth0"],
        runner=_quiet_ethtool_runner,
    )

    collector.collect_once()
    quiet = collector.collect_once()

    zero_is_a_reading = {
        "network_errors_delta",
        "network_drops_delta",
        "network_pfc_pause_delta",
        "network_ecn_marks_delta",
        "rdma_errors_delta",
        "efa_rnr_errors_delta",
        "efa_retry_errors_delta",
        "efa_cq_errors_delta",
    }
    names = {item.name for item in quiet.samples}
    assert names == zero_is_a_reading | {
        "network_link_up",
        "network_link_down",
        "rdma_link_down",
        "efa_traffic_bytes_delta",
        "efa_traffic_bytes_per_second",
    }, names
    still_reported = {
        item.name: item.value
        for item in quiet.samples
        if item.name in zero_is_a_reading
    }
    assert still_reported == dict.fromkeys(zero_is_a_reading, 0.0), (
        f"a signal the consumer reads as a latest value cannot clear without a "
        f"zero reading: {still_reported}"
    )


def test_the_cpu_total_excludes_guest_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``guest``/``guest_nice`` are already inside ``user``/``nice`` (F-H9)."""

    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ("_cpu",))
    stats = iter(["cpu 100 0 0 100 0 0 0 0 0 0\n", "cpu 200 0 0 200 0 0 0 0 100 0\n"])
    original_read_text = type(tmp_path).read_text

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/stat":
            return next(stats)
        if str(path) == "/proc/loadavg":
            return "1.00 1.00 1.00 1/100 1\n"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "read_text", read_text)
    times = iter([NOW, NOW + timedelta(seconds=15)])
    collector = HostTelemetryCollector(
        RecordingSink(), context(), node_id="worker-1", now=lambda: next(times)
    )

    collector.collect_once()
    second = collector.collect_once()

    usage = [item.value for item in second.samples if item.name == "cpu_usage_percent"]
    assert usage == [50.0], (
        f"the guest ticks were counted twice in the CPU total: {usage}"
    )


def test_a_new_attempt_does_not_inherit_the_previous_progress_clock(tmp_path) -> None:
    """Back-to-back attempts shared one progress time for a tick (F-H9).

    The clock is the input to the hang decision: a new attempt inheriting a
    stale progress time reports seconds of "no progress" it never had.
    """

    pids = ["1234\n5678\n"]

    def runner(argv, **_kwargs):
        stdout = (
            pids[0]
            if any("compute-apps" in str(item) for item in argv)
            else "GPU-a, 99\nGPU-b, 99\n"
        )
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        proc_root=str(tmp_path),
        runner=runner,
    )
    for pid in (1234, 5678):
        write_fake_rank(tmp_path, pid, cpu_ticks=100, write_bytes=0, wchar=0)
    rank_liveness_cycle(collector, NOW)
    write_fake_rank(
        tmp_path, 1234, cpu_ticks=140, write_bytes=0, wchar=900 * 1024 * 1024
    )
    advanced = rank_liveness_cycle(collector, NOW + timedelta(seconds=15))

    pids[0] = "4321\n8765\n"
    for pid in (4321, 8765):
        write_fake_rank(tmp_path, pid, cpu_ticks=100, write_bytes=0, wchar=0)
    restarted = rank_liveness_cycle(collector, NOW + timedelta(seconds=30))

    assert advanced["training_rank_seconds_since_progress"] == 0, advanced
    assert "training_rank_seconds_since_progress" not in restarted, (
        f"the new attempt inherited the previous attempt's progress time: {restarted}"
    )


def _smart_answer_runner(answers: list[tuple[str, int]], calls: list[list[str]]):
    """A ``smartctl`` whose ``-H`` answer changes from tick to tick.

    ``answers`` is consumed one entry per health check and the last entry
    repeats, so a test can fail a query once and then recover.
    """

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        if argv[1:] == ["--scan-open"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout="/dev/nvme0 -d nvme\n", stderr=""
            )
        stdout, returncode = answers[0] if len(answers) == 1 else answers.pop(0)
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")

    return runner


def test_a_failed_smart_query_is_not_cached_as_a_healthy_drive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drive that starts refusing SMART queries read like a healthy one.

    No verdict meant no sample, and the cache then held that silence for a
    whole period -- so a dying NVMe could vanish from monitoring for 5 minutes.
    """

    calls: list[list[str]] = []
    answers = [
        ("smartctl: unable to open device\n", 2),
        (json.dumps({"smart_status": {"passed": True}}), 0),
    ]
    collector = _smart_collector(
        monkeypatch,
        _smart_answer_runner(answers, calls),
        [NOW, NOW + timedelta(seconds=15), NOW + timedelta(seconds=30)],
    )

    failed = collector.collect_once()
    retried = collector.collect_once()
    cached = collector.collect_once()

    assert [
        (item.name, item.value, item.labels.get("failure_mode"))
        for item in failed.samples
    ] == [("smart_health_failed", 1, "QUERY_FAILED")], (
        f"a refused SMART query must be reported, not read as healthy: {failed.samples}"
    )
    assert len(_health_checks(calls)) == 2, (
        f"the failed query started a 300 s cache window: {calls}"
    )
    assert [item.value for item in retried.samples] == [0], (
        f"the retry's verdict was not reported: {retried.samples}"
    )
    assert [item.value for item in cached.samples] == [0], (
        f"a successful verdict must refresh the cache: {cached.samples}"
    )
    assert len(_health_checks(calls)) == 2, (
        f"the recovered verdict did not restore the cache window: {calls}"
    )


def test_a_drive_without_smart_status_is_not_a_failed_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--scan-open`` lists devices that implement no SMART status at all.

    Those answer cleanly with no verdict. Reporting that as a query failure
    would quarantine every node carrying such a device, so it stays a
    cacheable "nothing to report".
    """

    calls: list[list[str]] = []
    answers = [(json.dumps({"smartctl": {"exit_status": 0}}), 0)]
    collector = _smart_collector(
        monkeypatch,
        _smart_answer_runner(answers, calls),
        [NOW, NOW + timedelta(seconds=15)],
    )

    first = collector.collect_once()
    second = collector.collect_once()

    assert [item.name for item in first.samples] == [], (
        f"a device with no SMART status has no health to report: {first.samples}"
    )
    assert [item.name for item in second.samples] == [], (
        f"the second tick invented a verdict: {second.samples}"
    )
    assert len(_health_checks(calls)) == 1, (
        f"a clean verdict-less answer must stay cacheable: {calls}"
    )
