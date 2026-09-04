from __future__ import annotations

from ._support import (
    NOW,
    CollectorError,
    HostTelemetryCollector,
    HttpEventSink,
    NodeLogCollector,
    RecordingSink,
    TrainingProgressCollector,
    _LocalControlPlane,
    collectors_cli,
    completed_nvidia_smi,
    context,
    context_from_environment,
    json,
    logging,
    next_stable_phase,
    pytest,
    stable_phase_seconds,
    timedelta,
)


def test_gzip_threshold_is_configurable() -> None:
    with _LocalControlPlane() as server:
        sink = HttpEventSink(server.url, max_attempts=1, gzip_min_bytes=1_048_576)
        sink.post(
            "/v1/collector-events/kernel", {"cluster_id": "c", "blob": "x" * 8192}
        )

        assert server.requests[0][0] is None


def test_periodic_collector_phase_is_restart_stable() -> None:
    inventory_phase = stable_phase_seconds("cluster-a", "node-a", "gpu-inventory", 60)
    assert inventory_phase == stable_phase_seconds(
        "cluster-a", "node-a", "gpu-inventory", 60
    )
    assert 0 <= inventory_phase < 60

    scheduled = next_stable_phase(
        NOW,
        cluster_id="cluster-a",
        node_id="node-a",
        channel="gpu-inventory",
        interval_seconds=60,
    )
    assert scheduled > NOW
    assert int(scheduled.timestamp()) % 60 == inventory_phase


def test_context_prefers_discovered_product_and_validates_config(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "hp-cluster")
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT", "NVIDIA H200 NVL")
    monkeypatch.setenv("GPU_FAULT_CUDA_VERSION", "")
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT_DISCOVERY", "auto")

    def h200_runner(command, **_kwargs):
        if "--query-gpu=driver_version" in command:
            return completed_nvidia_smi("575.57.08\n")
        if command == ["nvidia-smi"]:
            return completed_nvidia_smi("CUDA Version: 12.9")
        return completed_nvidia_smi("0, GPU-a, NVIDIA H200\n")

    discovered = context_from_environment(discover_product=True, runner=h200_runner)
    assert discovered.product == "H200"
    assert discovered.driver_branch == 575
    assert discovered.cuda_version == "12.9"

    def b200_runner(command, **_kwargs):
        if "--query-gpu=driver_version" in command:
            return completed_nvidia_smi("575.57.08\n")
        if command == ["nvidia-smi"]:
            return completed_nvidia_smi("CUDA Version: 12.9")
        return completed_nvidia_smi("0, GPU-a, NVIDIA B200\n")

    with pytest.raises(CollectorError, match="does not match"):
        context_from_environment(discover_product=True, runner=b200_runner)


def test_context_uses_config_only_when_auto_discovery_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "hp-cluster")
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT", "H200")
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT_DISCOVERY", "auto")

    def unavailable(*_args, **_kwargs):
        return completed_nvidia_smi("", returncode=1, stderr="driver unavailable")

    context_value = context_from_environment(discover_product=True, runner=unavailable)
    assert context_value.product == "H200"

    monkeypatch.delenv("GPU_FAULT_GPU_PRODUCT")
    with pytest.raises(CollectorError, match="GPU identity query failed"):
        context_from_environment(discover_product=True, runner=unavailable)

    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT", "H200")
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT_DISCOVERY", "required")
    with pytest.raises(CollectorError, match="GPU identity query failed"):
        context_from_environment(discover_product=True, runner=unavailable)


def test_diskstats_includes_device_mapper_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    snapshots = iter(
        [
            "253 0 dm-0 10 0 0 20 20 0 0 40 0 50 60\n",
            "253 0 dm-0 15 0 0 30 25 0 0 50 0 70 100\n",
        ]
    )
    original_read_text = type(tmp_path).read_text

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/diskstats":
            return next(snapshots)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "read_text", read_text)
    collector = HostTelemetryCollector(RecordingSink(), context(), node_id="worker-1")

    collector._diskstats(NOW)
    samples = collector._diskstats(NOW + timedelta(seconds=15))

    assert {item.device for item in samples} == {"dm-0"}
    assert {item.name for item in samples} == {
        "disk_io_util_percent",
        "disk_io_await_ms",
    }


def test_log_collector_persists_training_file_offset(tmp_path) -> None:
    log = tmp_path / "training.log"
    log.write_text("step 1\n", encoding="utf-8")
    state = tmp_path / "collector-state.json"
    first = NodeLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        training_log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    entries = first._training_logs(NOW)
    assert entries == []
    with log.open("a", encoding="utf-8") as stream:
        stream.write("step 2\n")
    entries = first._training_logs(NOW)
    first._journal_since = NOW
    first._save_state()
    second = NodeLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        training_log_paths=[str(log)],
        state_path=str(state),
        now=lambda: NOW,
    )

    assert len(entries) == 1
    assert entries[0].message == "step 2"
    assert second._journal_since == NOW
    assert second._training_logs(NOW) == []


def test_training_progress_collector_reads_application_file(tmp_path) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "step": 42,
                "samples_per_second": 125.5,
                "loss": 0.75,
                "numerical_error": False,
            }
        ),
        encoding="utf-8",
    )
    sink = RecordingSink()
    collector = TrainingProgressCollector(
        sink,
        cluster_id="cluster-a",
        attempt_id="attempt-a",
        rank=3,
        progress_path=str(progress),
        node_id="node-a",
        now=lambda: NOW,
    )

    heartbeat = collector.collect_once()

    assert heartbeat.step == 42
    assert heartbeat.samples_per_second == 125.5
    assert sink.requests[0][0] == "/v1/training-progress"


def test_collector_cli_configures_root_logging(monkeypatch) -> None:
    """Collector processes must configure root logging themselves.

    Live check on 2026-08-08: the metrics collector journal contained
    zero ``delivered DCGM batch`` lines over a 35 minute window in
    which the control plane recorded 7 deliveries, and its tracebacks
    printed with no timestamp/level/logger name -- the signature of
    ``logging.lastResort``. Root was at WARNING with no handlers
    because nothing in the collector process ever called basicConfig
    (the same defect the control plane had fixed for itself). Every
    ``LOGGER.info`` on the node was silently discarded.
    """

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        root.handlers = []
        monkeypatch.setenv("GPU_FAULT_LOG_LEVEL", "INFO")

        collectors_cli.configure_logging()

        assert root.handlers, "collector left root logger unconfigured"
        assert root.level == logging.INFO
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def test_collector_cli_logging_respects_existing_handlers(monkeypatch) -> None:
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        sentinel = logging.NullHandler()
        root.handlers = [sentinel]
        root.setLevel(logging.CRITICAL)
        monkeypatch.setenv("GPU_FAULT_LOG_LEVEL", "DEBUG")

        collectors_cli.configure_logging()

        assert root.handlers == [sentinel], "an embedder's handler must not be replaced"
        assert root.level == logging.CRITICAL, "an embedder's level must not be raised"
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
