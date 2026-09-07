"""``gpu-fault-collector <command>`` builds the same collector it always did.

Each test drives ``collectors_cli.main()`` end to end -- argv, environment,
subparser, descriptor, factory -- and captures the collector the CLI would have
run. The assertions pin the configuration each subcommand derives from its
arguments and environment, so moving a factory or a table cannot silently drop
an option.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault import collectors_cli
from gpu_fault.collectors import (
    DcgmMetricsCollector,
    FabricManagerLogCollector,
    HostTelemetryCollector,
    KernelLogCollector,
    KubernetesHmaNodeCollector,
    KubernetesNodeResourceCollector,
    NodeLogCollector,
    NvidiaSmiMetricsCollector,
    SqsHmaConsumer,
    TrainingProgressCollector,
)

COMMANDS = (
    "kernel",
    "kubernetes-hma",
    "kubernetes-node-resources",
    "sqs-hma",
    "dcgm",
    "nvidia-smi",
    "host",
    "logs",
    "fabric-manager",
    "training-progress",
)


@pytest.fixture
def node_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT", "H100")
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT_DISCOVERY", "disabled")
    monkeypatch.setenv(
        "GPU_FAULT_COLLECTOR_OUTBOX_PATH", str(tmp_path / "outbox.ndjson")
    )
    monkeypatch.setenv("NODE_NAME", "node-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    for name in (
        "HOSTNAME",
        "GPU_FAULT_FILESYSTEMS",
        "GPU_FAULT_REQUIRED_INTERFACES",
        "GPU_FAULT_TRAINING_LOG_PATHS",
        "GPU_FAULT_FABRIC_MANAGER_LOG_PATHS",
        "GPU_FAULT_HMA_QUEUE_URL",
        "RANK",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _run_cli(
    monkeypatch: pytest.MonkeyPatch, collector_type: type, argv: list[str]
) -> Any:
    built: list[Any] = []

    def capture(self: Any) -> None:
        built.append(self)

    monkeypatch.setattr(collector_type, "run", capture)
    monkeypatch.setattr(sys, "argv", ["gpu-fault-collector", *argv])
    collectors_cli.main()
    (collector,) = built
    return collector


@pytest.mark.parametrize("command", COMMANDS)
def test_parser_registers_every_collector_command(command: str) -> None:
    arguments = collectors_cli.parser().parse_args([command])

    assert arguments.command == command


def test_kernel(monkeypatch: pytest.MonkeyPatch, node_environment: Path) -> None:
    collector = _run_cli(
        monkeypatch,
        KernelLogCollector,
        ["kernel", "--kmsg-path", str(node_environment / "kmsg")],
    )

    assert isinstance(collector, KernelLogCollector), (
        "the CLI must build the collector type its registry row names"
    )
    assert collector.node_id == "node-1"
    assert collector.kmsg_path == str(node_environment / "kmsg")
    assert collector.context.cluster_id == "cluster-a"


def test_kernel_requires_a_node_id(
    monkeypatch: pytest.MonkeyPatch, node_environment: Path
) -> None:
    monkeypatch.delenv("NODE_NAME")

    with pytest.raises(SystemExit, match="node-id"):
        _run_cli(monkeypatch, KernelLogCollector, ["kernel"])


def test_dcgm(monkeypatch: pytest.MonkeyPatch, node_environment: Path) -> None:
    monkeypatch.setenv("GPU_FAULT_DCGM_METRICS_URL", "http://127.0.0.1:9401/metrics")
    monkeypatch.setenv("GPU_FAULT_METRICS_INTERVAL_SECONDS", "7")

    collector = _run_cli(monkeypatch, DcgmMetricsCollector, ["dcgm"])

    assert isinstance(collector, DcgmMetricsCollector), (
        "the CLI must build the collector type its registry row names"
    )
    assert collector.node_id == "node-1"
    assert collector.metrics_url == "http://127.0.0.1:9401/metrics"
    assert collector.interval_seconds == 7.0


def test_nvidia_smi(monkeypatch: pytest.MonkeyPatch, node_environment: Path) -> None:
    monkeypatch.delenv("GPU_FAULT_METRICS_INTERVAL_SECONDS", raising=False)

    collector = _run_cli(
        monkeypatch, NvidiaSmiMetricsCollector, ["nvidia-smi", "--node-id", "override"]
    )

    assert isinstance(collector, NvidiaSmiMetricsCollector), (
        "the CLI must build the collector type its registry row names"
    )
    assert collector.node_id == "override"
    assert collector.interval_seconds == 30.0


def test_host(monkeypatch: pytest.MonkeyPatch, node_environment: Path) -> None:
    monkeypatch.setenv("GPU_FAULT_HOST_INTERVAL_SECONDS", "20")
    monkeypatch.setenv("GPU_FAULT_FILESYSTEMS", "/,/data,")
    monkeypatch.setenv("GPU_FAULT_REQUIRED_INTERFACES", "eth0,eth1")
    monkeypatch.setenv("GPU_FAULT_PCI_DEVICES_ROOT", str(node_environment))
    monkeypatch.setenv("GPU_FAULT_NODE_INSTANCE_TYPE", "p5.48xlarge")
    monkeypatch.setenv("GPU_FAULT_EXPECTED_GPU_COUNT", "8")
    monkeypatch.setenv("GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT", "32")
    monkeypatch.setenv("GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES", "3")

    collector = _run_cli(monkeypatch, HostTelemetryCollector, ["host"])

    assert isinstance(collector, HostTelemetryCollector), (
        "the CLI must build the collector type its registry row names"
    )
    assert collector.node_id == "node-1"
    assert collector.interval_seconds == 20.0
    assert collector.filesystems == ["/", "/data"]
    assert collector.required_interfaces == {"eth0", "eth1"}
    assert collector.pci_devices_root == node_environment
    assert collector.node_instance_type == "p5.48xlarge"
    assert collector.expected_gpu_count == 8
    assert collector.expected_efa_device_count == 32
    assert collector.inventory_mismatch_consecutive_samples == 3


def test_host_defaults(monkeypatch: pytest.MonkeyPatch, node_environment: Path) -> None:
    for name in (
        "GPU_FAULT_HOST_INTERVAL_SECONDS",
        "GPU_FAULT_PCI_DEVICES_ROOT",
        "GPU_FAULT_NODE_INSTANCE_TYPE",
        "GPU_FAULT_EXPECTED_GPU_COUNT",
        "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT",
        "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES",
    ):
        monkeypatch.delenv(name, raising=False)

    collector = _run_cli(monkeypatch, HostTelemetryCollector, ["host"])

    assert collector.interval_seconds == 15.0
    assert collector.filesystems == ["/", "/var", "/tmp"]
    assert collector.required_interfaces == set()
    assert collector.node_instance_type is None
    assert collector.expected_gpu_count is None
    assert collector.expected_efa_device_count is None
    assert collector.inventory_mismatch_consecutive_samples == 2


def test_logs(monkeypatch: pytest.MonkeyPatch, node_environment: Path) -> None:
    monkeypatch.setenv("GPU_FAULT_LOG_INTERVAL_SECONDS", "4")
    monkeypatch.setenv("GPU_FAULT_TRAINING_LOG_PATHS", "/var/log/a.log,,/var/log/b.log")
    monkeypatch.setenv("GPU_FAULT_LOG_STATE_PATH", str(node_environment / "state.json"))

    collector = _run_cli(monkeypatch, NodeLogCollector, ["logs"])

    assert isinstance(collector, NodeLogCollector), (
        "the CLI must build the collector type its registry row names"
    )
    assert collector.node_id == "node-1"
    assert collector.interval_seconds == 4.0
    assert collector.training_log_paths == ["/var/log/a.log", "/var/log/b.log"]
    assert collector.state_path == node_environment / "state.json"


def test_fabric_manager(
    monkeypatch: pytest.MonkeyPatch, node_environment: Path
) -> None:
    monkeypatch.setenv("GPU_FAULT_FABRIC_MANAGER_LOG_INTERVAL_SECONDS", "9")
    monkeypatch.setenv("GPU_FAULT_FABRIC_MANAGER_JOURNAL", "False")
    monkeypatch.setenv("GPU_FAULT_FABRIC_MANAGER_IDENTIFIERS", "fm-a,,fm-b")
    monkeypatch.setenv("GPU_FAULT_FABRIC_MANAGER_LOG_PATHS", "/var/log/fm.log")
    monkeypatch.setenv(
        "GPU_FAULT_FABRIC_MANAGER_STATE_PATH", str(node_environment / "fm.json")
    )

    collector = _run_cli(monkeypatch, FabricManagerLogCollector, ["fabric-manager"])

    assert isinstance(collector, FabricManagerLogCollector), (
        "the CLI must build the collector type its registry row names"
    )
    assert collector.node_id == "node-1"
    assert collector.interval_seconds == 9.0
    assert collector.journal_enabled is False
    assert collector.journal_identifiers == ("fm-a", "fm-b")
    assert collector.log_paths == ["/var/log/fm.log"]
    assert collector.state_path == node_environment / "fm.json"


@pytest.mark.parametrize(("token", "expected"), [("1", True), ("off", False)])
def test_fabric_manager_journal_switch_reads_every_token(
    monkeypatch: pytest.MonkeyPatch, node_environment: Path, token: str, expected: bool
) -> None:
    """``GPU_FAULT_FABRIC_MANAGER_JOURNAL=1`` used to switch the journal off."""

    monkeypatch.setenv("GPU_FAULT_FABRIC_MANAGER_JOURNAL", token)
    monkeypatch.setenv(
        "GPU_FAULT_FABRIC_MANAGER_STATE_PATH", str(node_environment / "fm.json")
    )

    collector = _run_cli(monkeypatch, FabricManagerLogCollector, ["fabric-manager"])

    assert collector.journal_enabled is expected


def test_fabric_manager_defaults(
    monkeypatch: pytest.MonkeyPatch, node_environment: Path
) -> None:
    for name in (
        "GPU_FAULT_FABRIC_MANAGER_LOG_INTERVAL_SECONDS",
        "GPU_FAULT_FABRIC_MANAGER_JOURNAL",
        "GPU_FAULT_FABRIC_MANAGER_IDENTIFIERS",
        "GPU_FAULT_FABRIC_MANAGER_STATE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    collector = _run_cli(monkeypatch, FabricManagerLogCollector, ["fabric-manager"])

    assert collector.interval_seconds == 5.0
    assert collector.journal_enabled is True
    assert collector.journal_identifiers == ("nvidia-fabricmanager", "nv-fabricmanager")
    assert collector.log_paths == []
    assert collector.state_path == Path(
        "/var/lib/gpu-fault/fabric-manager-collector-state.json"
    )


def test_training_progress(
    monkeypatch: pytest.MonkeyPatch, node_environment: Path
) -> None:
    monkeypatch.setenv("GPU_FAULT_ATTEMPT_ID", "attempt-7")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("POD_UID", "pod-uid")
    monkeypatch.setenv("CONTAINER_NAME", "worker")
    monkeypatch.setenv("GPU_FAULT_GPU_UUIDS", "GPU-a,GPU-b,")
    monkeypatch.setenv("GPU_FAULT_TRAINING_PROGRESS_INTERVAL_SECONDS", "2")

    collector = _run_cli(
        monkeypatch,
        TrainingProgressCollector,
        ["training-progress", "--progress-file", str(node_environment / "p.json")],
    )

    assert isinstance(collector, TrainingProgressCollector), (
        "the CLI must build the collector type its registry row names"
    )
    assert collector.cluster_id == "cluster-a"
    assert collector.attempt_id == "attempt-7"
    assert collector.rank == 3
    assert collector.progress_path == node_environment / "p.json"
    assert collector.node_id == "node-1"
    assert collector.pod_uid == "pod-uid"
    assert collector.container_name == "worker"
    assert collector.gpu_uuids == ["GPU-a", "GPU-b"]
    assert collector.interval_seconds == 2.0


def test_training_progress_requires_attempt_and_rank(
    monkeypatch: pytest.MonkeyPatch, node_environment: Path
) -> None:
    monkeypatch.delenv("GPU_FAULT_ATTEMPT_ID", raising=False)

    with pytest.raises(SystemExit, match="attempt-id"):
        _run_cli(monkeypatch, TrainingProgressCollector, ["training-progress"])


def test_kubernetes_hma(
    monkeypatch: pytest.MonkeyPatch, node_environment: Path
) -> None:
    collector = _run_cli(monkeypatch, KubernetesHmaNodeCollector, ["kubernetes-hma"])

    assert isinstance(collector, KubernetesHmaNodeCollector), (
        "the CLI must build the collector type its registry row names"
    )
    assert collector.context.cluster_id == "cluster-a"


def test_kubernetes_node_resources(
    monkeypatch: pytest.MonkeyPatch, node_environment: Path
) -> None:
    monkeypatch.setenv("GPU_FAULT_KUBERNETES_EFA_MISMATCH_SAMPLES", "4")
    monkeypatch.setenv("GPU_FAULT_KUBERNETES_EFA_HEALTH_SUMMARY_SECONDS", "60")
    monkeypatch.setenv("GPU_FAULT_KUBERNETES_NODE_LIST_PAGE_SIZE", "50")

    collector = _run_cli(
        monkeypatch,
        KubernetesNodeResourceCollector,
        ["kubernetes-node-resources", "--interval-seconds", "3"],
    )

    assert isinstance(collector, KubernetesNodeResourceCollector), (
        "the CLI must build the collector type its registry row names"
    )
    assert collector.interval_seconds == 3.0
    assert collector.required_consecutive_samples == 4
    assert collector.health_summary_seconds == 60
    assert collector.list_page_size == 50


def test_sqs_hma(monkeypatch: pytest.MonkeyPatch, node_environment: Path) -> None:
    monkeypatch.setenv(
        "GPU_FAULT_HMA_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/1/hma"
    )
    monkeypatch.delenv("GPU_FAULT_CLUSTER_ID")

    collector = _run_cli(monkeypatch, SqsHmaConsumer, ["sqs-hma"])

    assert isinstance(collector, SqsHmaConsumer), (
        "the CLI must build the collector type its registry row names"
    )
    assert collector.queue_url == "https://sqs.us-east-1.amazonaws.com/1/hma"


def test_sqs_hma_requires_a_queue_url(
    monkeypatch: pytest.MonkeyPatch, node_environment: Path
) -> None:
    with pytest.raises(SystemExit, match="GPU_FAULT_HMA_QUEUE_URL"):
        _run_cli(monkeypatch, SqsHmaConsumer, ["sqs-hma"])


def test_outbox_path_defaults_to_the_command_name(
    monkeypatch: pytest.MonkeyPatch, node_environment: Path
) -> None:
    monkeypatch.delenv("GPU_FAULT_COLLECTOR_OUTBOX_PATH")

    collector = _run_cli(monkeypatch, KernelLogCollector, ["kernel"])

    assert collector.sink.outbox_path == Path("/var/lib/gpu-fault/outbox/kernel.ndjson")
