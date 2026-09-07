"""The collector registry is the one table a new collector is added to.

``COLLECTOR_REGISTRY`` (one row per ``gpu-fault-collector`` subcommand) and
``COLLECTOR_KINDS`` (one row per ``CollectorKind``) replace the seven parallel
tables that used to be edited in lockstep. ``validate_collector_registry()``
runs at import time like the operation, channel and node-action registries, so
a row that names an unknown channel, an unpublished export or leaves a kind
unserved fails the first import instead of the first request.

The expected tables below are written by hand on purpose: comparing a derived
view against the comprehension that produced it proves nothing.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import pytest

from gpu_fault import collector_requirements, collectors_cli, telemetry
from gpu_fault.channel_registry import HOST_TELEMETRY_PATH
from gpu_fault.collector_registry import (
    COLLECTOR_PRODUCER_BY_CHANNEL,
    COLLECTOR_REGISTRY,
    COLLECTOR_SYSTEMD_UNITS,
    CollectorDescriptor,
    CollectorKind,
    collector_registry_with_plugins,
    collector_silent_thresholds,
    validate_collector_registry,
)
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.sinks import EventSink
from gpu_fault.plugins import PluginGroup

EXPECTED_PRODUCERS = {
    CollectorKind.GPU_INVENTORY: "DCGM_METRICS_COLLECTOR",
    CollectorKind.GPU_METRICS: "DCGM_METRICS_COLLECTOR",
    CollectorKind.HOST_TELEMETRY: "HOST_TELEMETRY_COLLECTOR",
    CollectorKind.NODE_LOGS: "NODE_LOG_COLLECTOR",
    CollectorKind.NVIDIA_KERNEL: "KERNEL_LOG_COLLECTOR",
    CollectorKind.FABRIC_MANAGER_LOG: "FABRIC_MANAGER_LOG_COLLECTOR",
    CollectorKind.HMA_NODE: "KUBERNETES_HMA_NODE_COLLECTOR",
    CollectorKind.HMA_CLOUDWATCH: "CLOUDWATCH_HMA_COLLECTOR",
}
EXPECTED_SYSTEMD_UNITS = {
    CollectorKind.NVIDIA_KERNEL: "gpu-fault-kernel-collector",
    CollectorKind.FABRIC_MANAGER_LOG: "gpu-fault-fabric-manager-collector",
    CollectorKind.GPU_INVENTORY: "gpu-fault-metrics-collector",
    CollectorKind.GPU_METRICS: "gpu-fault-metrics-collector",
    CollectorKind.HOST_TELEMETRY: "gpu-fault-host-collector",
    CollectorKind.NODE_LOGS: "gpu-fault-log-collector",
}
EXPECTED_SILENT_DEFAULTS = {
    CollectorKind.GPU_INVENTORY: 180.0,
    CollectorKind.GPU_METRICS: 420.0,
    CollectorKind.HOST_TELEMETRY: 420.0,
    CollectorKind.NVIDIA_KERNEL: 900.0,
    CollectorKind.FABRIC_MANAGER_LOG: 900.0,
    CollectorKind.NODE_LOGS: 900.0,
}
EXPECTED_COMMAND_KINDS = {
    "kernel": (CollectorKind.NVIDIA_KERNEL,),
    "kubernetes-hma": (CollectorKind.HMA_NODE,),
    "kubernetes-node-resources": (CollectorKind.HOST_TELEMETRY,),
    "sqs-hma": (CollectorKind.HMA_CLOUDWATCH,),
    "dcgm": (CollectorKind.GPU_METRICS, CollectorKind.GPU_INVENTORY),
    "nvidia-smi": (CollectorKind.GPU_METRICS, CollectorKind.GPU_INVENTORY),
    "host": (CollectorKind.HOST_TELEMETRY,),
    "logs": (CollectorKind.NODE_LOGS,),
    "fabric-manager": (CollectorKind.FABRIC_MANAGER_LOG,),
    "training-progress": (),
}


class _EntryPoints(list[metadata.EntryPoint]):
    def select(self, *, group: str) -> list[metadata.EntryPoint]:
        return [item for item in self if item.group == group]


class FakePluginCollector:
    """What a third-party wheel would ship: a collector with a ``run``."""

    built: list[FakePluginCollector] = []

    def __init__(
        self, sink: EventSink, context: CollectorContext, greeting: str
    ) -> None:
        self.sink = sink
        self.context = context
        self.greeting = greeting
        self.ran = False

    def run(self) -> None:
        self.ran = True
        FakePluginCollector.built.append(self)


def build_fake_plugin_collector(
    sink: EventSink, context: CollectorContext, arguments: argparse.Namespace
) -> FakePluginCollector:
    return FakePluginCollector(sink, context, greeting=arguments.command)


FAKE_PLUGIN = CollectorDescriptor(
    cli_command="fake-plugin",
    kinds=(),
    channel_paths=(HOST_TELEMETRY_PATH,),
    export_name=None,
    factory="tests.test_collector_registry:build_fake_plugin_collector",
    runs_in="cluster",
)


def _install_plugin(monkeypatch: pytest.MonkeyPatch, *entries: tuple[str, str]) -> None:
    values = _EntryPoints(
        [
            metadata.EntryPoint(name=name, value=value, group=PluginGroup.COLLECTORS)
            for name, value in entries
        ]
    )
    monkeypatch.setattr(metadata, "entry_points", lambda: values)


def test_builtin_registry_validates_and_serves_every_kind() -> None:
    validate_collector_registry()

    assert {
        command: descriptor.kinds for command, descriptor in COLLECTOR_REGISTRY.items()
    } == EXPECTED_COMMAND_KINDS


def test_producers_and_units_are_derived_views_of_the_registry() -> None:
    assert COLLECTOR_PRODUCER_BY_CHANNEL == EXPECTED_PRODUCERS
    assert COLLECTOR_SYSTEMD_UNITS == EXPECTED_SYSTEMD_UNITS
    assert telemetry.COLLECTOR_PRODUCER_BY_CHANNEL is COLLECTOR_PRODUCER_BY_CHANNEL
    assert collector_requirements.COLLECTOR_SYSTEMD_UNITS is COLLECTOR_SYSTEMD_UNITS
    assert telemetry.CollectorKind is CollectorKind


def test_silent_thresholds_default_and_follow_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "GPU_FAULT_GPU_INVENTORY_SILENT_AFTER_SECONDS",
        "GPU_FAULT_GPU_METRICS_SILENT_AFTER_SECONDS",
        "GPU_FAULT_HOST_TELEMETRY_SILENT_AFTER_SECONDS",
        "GPU_FAULT_KERNEL_SILENT_AFTER_SECONDS",
        "GPU_FAULT_FABRIC_MANAGER_SILENT_AFTER_SECONDS",
        "GPU_FAULT_NODE_LOG_SILENT_AFTER_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert collector_silent_thresholds() == EXPECTED_SILENT_DEFAULTS
    assert collector_requirements.collector_silent_thresholds() == (
        EXPECTED_SILENT_DEFAULTS
    )

    monkeypatch.setenv("GPU_FAULT_GPU_INVENTORY_SILENT_AFTER_SECONDS", "45")
    monkeypatch.setenv("GPU_FAULT_NODE_LOG_SILENT_AFTER_SECONDS", "1200.5")

    thresholds = collector_silent_thresholds()

    assert thresholds[CollectorKind.GPU_INVENTORY] == 45.0
    assert thresholds[CollectorKind.NODE_LOGS] == 1200.5
    assert thresholds[CollectorKind.GPU_METRICS] == 420.0


def test_unserved_kind_fails_closed() -> None:
    registry = {
        command: descriptor
        for command, descriptor in COLLECTOR_REGISTRY.items()
        if command != "kernel"
    }

    with pytest.raises(RuntimeError, match="NVIDIA_KERNEL"):
        validate_collector_registry(registry)


def test_unknown_channel_path_fails_closed() -> None:
    registry = dict(COLLECTOR_REGISTRY)
    registry["kernel"] = replace(
        registry["kernel"], channel_paths=("/v1/collector-events/invented-channel",)
    )

    with pytest.raises(RuntimeError, match="invented-channel"):
        validate_collector_registry(registry)


def test_unpublished_export_fails_closed() -> None:
    registry = dict(COLLECTOR_REGISTRY)
    registry["host"] = replace(registry["host"], export_name="InventedCollector")

    with pytest.raises(RuntimeError, match="InventedCollector"):
        validate_collector_registry(registry)


def test_command_key_and_descriptor_must_agree() -> None:
    registry = dict(COLLECTOR_REGISTRY)
    registry["kernel"] = registry["dcgm"]

    with pytest.raises(RuntimeError, match="kernel"):
        validate_collector_registry(registry)


def test_product_discovery_needs_a_node_context() -> None:
    registry = dict(COLLECTOR_REGISTRY)
    registry["sqs-hma"] = replace(registry["sqs-hma"], needs_product_discovery=True)

    with pytest.raises(RuntimeError, match="sqs-hma"):
        validate_collector_registry(registry)


def test_node_command_cannot_straddle_two_systemd_units() -> None:
    registry = dict(COLLECTOR_REGISTRY)
    registry["kernel"] = replace(
        registry["kernel"],
        kinds=(CollectorKind.NVIDIA_KERNEL, CollectorKind.HOST_TELEMETRY),
    )

    with pytest.raises(RuntimeError, match="systemd"):
        validate_collector_registry(registry)


def test_plugin_collector_is_merged_and_built_by_the_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_plugin(
        monkeypatch, ("fake-plugin", "tests.test_collector_registry:FAKE_PLUGIN")
    )
    FakePluginCollector.built.clear()
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT_DISCOVERY", "disabled")
    monkeypatch.setenv(
        "GPU_FAULT_COLLECTOR_OUTBOX_PATH", str(tmp_path / "outbox.ndjson")
    )
    monkeypatch.setattr(sys, "argv", ["gpu-fault-collector", "fake-plugin"])

    registry = collector_registry_with_plugins()
    assert registry["fake-plugin"] is FAKE_PLUGIN
    assert "fake-plugin" not in COLLECTOR_REGISTRY, "plugins never mutate the table"

    collectors_cli.main()

    (collector,) = FakePluginCollector.built
    assert collector.ran, "the plugin collector must have run"
    assert collector.greeting == "fake-plugin"
    assert collector.context.cluster_id == "cluster-a"


def test_plugin_colliding_with_a_builtin_command_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_plugin(
        monkeypatch, ("kernel", "tests.test_collector_registry:FAKE_PLUGIN")
    )

    with pytest.raises(RuntimeError, match="kernel"):
        collector_registry_with_plugins()


def test_plugin_entry_point_name_must_match_its_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_plugin(
        monkeypatch, ("other-name", "tests.test_collector_registry:FAKE_PLUGIN")
    )

    with pytest.raises(RuntimeError, match="other-name"):
        collector_registry_with_plugins()


def test_plugin_must_be_a_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_plugin(
        monkeypatch,
        ("fake-plugin", "tests.test_collector_registry:FakePluginCollector"),
    )

    with pytest.raises(RuntimeError, match="CollectorDescriptor"):
        collector_registry_with_plugins()
