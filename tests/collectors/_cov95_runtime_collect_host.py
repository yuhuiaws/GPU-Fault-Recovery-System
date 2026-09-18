"""Private proc/sys trees and fake device commands for real host collection."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.collectors.host import collector as host_module
from gpu_fault.collectors.host import network, system_metrics
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


@pytest.fixture
def host_case(monkeypatch, tmp_path, isolated_runtime):
    roots = support.host_roots(tmp_path)
    clock = support.Clock()
    sink = support.RecordingSink()
    commands = []
    outputs = {}

    def mapped_path(path):
        raw = str(path)
        if raw == "/proc" or raw.startswith("/proc/"):
            return roots.proc / raw.removeprefix("/proc").lstrip("/")
        return Path(path)

    def runner(argv, **kwargs):
        commands.append((argv, kwargs))
        value = outputs.get(tuple(argv), ("", "", 0))
        if isinstance(value, BaseException):
            raise value
        stdout, stderr, code = value
        return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr=stderr)

    def build(*contributors, **kwargs):
        monkeypatch.setattr(
            host_module.HostTelemetryCollector, "CONTRIBUTORS", contributors
        )
        return host_module.HostTelemetryCollector(
            sink,
            support.collector_context(workload_state="ACTIVE"),
            **{
                "node_id": "node-a",
                "proc_root": str(roots.proc),
                "infiniband_root": str(roots.rdma),
                "net_class_root": str(roots.net),
                "pci_devices_root": str(roots.pci),
                "filesystems": [str(roots.mount)],
                "force_snapshot_path": str(tmp_path / "host.request"),
                "now": clock.now,
                "runner": runner,
                "edge_filter_enabled": False,
                "statvfs": lambda path: os.statvfs_result(
                    (4096, 4096, 100, 60, 50, 0, 0, 0, 0, 255)
                ),
                **kwargs,
            },
        )

    for module in (system_metrics, network):
        monkeypatch.setattr(module, "Path", mapped_path)
    monkeypatch.setattr(system_metrics.shutil, "which", lambda name: None)
    monkeypatch.setattr(system_metrics.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(host_module.os, "sysconf", lambda name: 100)
    return SimpleNamespace(
        build=build,
        roots=roots,
        clock=clock,
        sink=sink,
        commands=commands,
        outputs=outputs,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )


def write_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def values(batch):
    return {(sample.name, sample.device): sample.value for sample in batch.samples}


def enable_tools(case, *names):
    case.monkeypatch.setattr(
        system_metrics.shutil,
        "which",
        lambda name: f"/private/bin/{name}" if name in names else None,
    )
