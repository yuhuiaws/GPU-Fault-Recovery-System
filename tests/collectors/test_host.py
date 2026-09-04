from __future__ import annotations

from ._support import (
    NOW,
    HostTelemetryCollector,
    RecordingSink,
    context,
    json,
    pytest,
    rank_liveness_collector,
    rank_liveness_cycle,
    subprocess,
    timedelta,
    write_fake_rank,
)


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
