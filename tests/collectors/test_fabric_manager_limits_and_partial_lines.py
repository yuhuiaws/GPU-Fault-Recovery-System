"""Fabric Manager collector guards that only show up at the edges.

The constructor refuses limits that would disable its own bounds, a node
without ``journalctl`` is not a failed round but an unqueried source, the
journald match list carries one group per value however the identifiers are
spelled, and a file whose last line is still being written delivers the
complete lines before it and parks the checkpoint at the start of the partial
one.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import gpu_fault.collectors.logs.fabric_manager as fabric_manager_module
from gpu_fault.channel_registry import COLLECTOR_HEALTH_PATH
from gpu_fault.collectors import FabricManagerLogCollector
from tests.collectors._support import NOW, RecordingSink, context

SXID = "[Jul 20 2026 11:59:30] [ERROR] [tid 7] SXid (PCI:0000:c1:00.0): 12028, Fatal, "


def _collector(**overrides: Any) -> FabricManagerLogCollector:
    settings: dict[str, Any] = {
        "node_id": "worker-1",
        "now": lambda: NOW,
        "journal_enabled": False,
    }
    settings.update(overrides)
    return FabricManagerLogCollector(RecordingSink(), context(), **settings)


def test_the_tracked_file_limit_must_keep_at_least_one_file() -> None:
    with pytest.raises(ValueError, match="tracked file limit"):
        _collector(max_tracked_files=0)


def test_the_line_age_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max line age"):
        _collector(max_line_age_seconds=0)


def test_the_health_summary_interval_from_the_environment_must_be_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS", "0")
    with pytest.raises(ValueError, match="health summary interval"):
        _collector()


def test_a_node_without_journalctl_queries_nothing_and_delivers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fabric_manager_module.shutil, "which", lambda _name: None)
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink, context(), node_id="worker-1", now=lambda: NOW, runner=runner
    )

    stats = collector.collect_once()

    assert calls == [], "journalctl was invoked although it is not installed"
    assert (stats.observed, stats.delivered, stats.skipped) == (0, 0, 0)
    assert [path for path, _payload in sink.requests] == [COLLECTOR_HEALTH_PATH], (
        "only the health summary may leave a node whose journal cannot be read"
    )


def test_journal_matches_carry_each_value_once_whatever_the_identifier_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fabric_manager_module.shutil, "which", lambda _name: "/usr/bin/journalctl"
    )
    commands: list[list[str]] = []

    def runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    collector = FabricManagerLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        runner=runner,
        journal_identifiers=("nvidia-fabricmanager", "NVIDIA-FabricManager", ""),
    )

    collector.collect_once()

    (command,) = commands
    matches = command[command.index("--since") + 2 :]
    assert matches == [
        "_SYSTEMD_UNIT=nvidia-fabricmanager.service",
        "+",
        "SYSLOG_IDENTIFIER=nvidia-fabricmanager",
        "+",
        "_COMM=nvidia-fabricma",
    ], f"a case-insensitive duplicate identifier doubled the match groups: {matches}"


def test_a_line_still_being_written_waits_for_its_terminator(tmp_path: Path) -> None:
    log = tmp_path / "fabricmanager.log"
    log.write_text("Fabric Manager started\n", encoding="utf-8")
    state = tmp_path / "state.json"
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
    # A new file is baselined at its end; nothing before the checkpoint replays.
    assert collector.collect_once().observed == 0
    baseline = json.loads(state.read_text())["files"][str(log)]["offset"]

    complete = SXID + "Link 46 egress sequence ID error\n"
    with log.open("a", encoding="utf-8") as stream:
        stream.write(complete + SXID + "Link 47 partial")
    stats = collector.collect_once()

    assert (stats.observed, stats.delivered) == (1, 1)
    assert [
        payload["message"]
        for path, payload in sink.requests
        if path != COLLECTOR_HEALTH_PATH
    ] == [complete.rstrip("\n")]
    offset = json.loads(state.read_text())["files"][str(log)]["offset"]
    assert offset == baseline + len(complete.encode("utf-8")), (
        "the checkpoint moved into a line the daemon has not finished writing"
    )

    with log.open("a", encoding="utf-8") as stream:
        stream.write(" completed\n")
    second = collector.collect_once()

    assert second.delivered == 1
    assert sink.requests[-1][1]["message"] == SXID + "Link 47 partial completed"
