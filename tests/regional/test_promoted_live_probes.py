from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.probes import (
    destructive_node_probe,
    ha001_probe,
    ha005_probe,
    ha006_executor,
    net002_executor,
    net003_executor,
    node_host_probe,
)


def _context(key: str, operation: str = "FREEZE_EVIDENCE"):
    return SimpleNamespace(
        idempotency_key=key,
        step=SimpleNamespace(
            operation=SimpleNamespace(value=operation), execution_owner="test-owner"
        ),
        incident=SimpleNamespace(cluster_id="cluster-a", incident_id="incident-a"),
    )


def test_net002_automatic_block_rollback(tmp_path: Path, monkeypatch) -> None:
    block = tmp_path / "block"
    rollback = tmp_path / "rollback.json"
    monkeypatch.setattr(net002_executor, "BLOCK", block)
    monkeypatch.setattr(net002_executor, "ROLLBACK_STATE", rollback)
    monkeypatch.setattr(net002_executor, "BLOCK_ROLLBACK_SECONDS", 0.05)
    block.touch()

    Thread(target=net002_executor.rollback_stale_block, daemon=True).start()
    deadline = time.monotonic() + 2
    while block.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not block.exists(), "automatic rollback left the network block marker"
    # The record is written (atomically) before the marker is lifted, so the
    # moment the marker is gone the record is complete -- no second wait.
    record = json.loads(rollback.read_text())
    assert record["automatic"] is True, record
    assert record["blocked_seconds"] >= 0.05, record
    assert not rollback.with_suffix(".tmp").exists(), (
        "the temporary file was left behind"
    )


def test_net003_ledger_is_exactly_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(net003_executor, "ACTION_STARTED", tmp_path / "started")
    monkeypatch.setattr(net003_executor, "LEDGER", tmp_path / "ledger.json")
    adapter = net003_executor.LedgerAdapter("notification-a")
    context = _context("workflow/0/FREEZE_EVIDENCE")
    monkeypatch.setattr(net003_executor.time, "sleep", lambda _seconds: None)
    # The action gate waits up to 30s for the runner to arm BLOCK; with no live
    # runner the unit test arms it itself, or execute() times out against the
    # ambient /state/block that only happens to exist on a live host.
    block = tmp_path / "block"
    block.write_text("", encoding="utf-8")
    monkeypatch.setattr(net003_executor, "BLOCK", block)
    monkeypatch.setattr(
        net003_executor, "ACTION_GATE_OBSERVED", tmp_path / "action-gate-observed.json"
    )

    first = adapter.execute(context)
    second = adapter.execute(context)

    assert first.details["cached"] is False
    assert second.details["cached"] is True
    assert second.details["physical_count"] == 1
    assert second.details["notification_id"] == "notification-a"


def test_ha001_probe_ledger_replays_three_operations_once(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(ha001_probe, "LEDGER", tmp_path / "ledger.json")
    adapter = ha001_probe.SimulatedAdapter()
    operations = ["FREEZE_EVIDENCE", "STOP_WORKLOADS", "RESTART_WORKLOAD"]

    for index, operation in enumerate(operations):
        context = _context(f"workflow/{index}/{operation}", operation)
        assert adapter.execute(context).details["cached"] is False
        assert adapter.execute(context).details["cached"] is True

    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["physical_count"] == 3
    assert ledger["operations"] == operations


def test_ha005_outbox_status_counts_replayable_records(
    tmp_path: Path, monkeypatch
) -> None:
    outbox = tmp_path / "outbox.ndjson"
    monkeypatch.setattr(ha005_probe, "OUTBOX", outbox)
    outbox.write_text(
        "\n".join(json.dumps({"replayable": value}) for value in (True, False, True))
        + "\n"
    )

    assert ha005_probe.outbox_status() == {"records": 3, "replayable": 2}


class _NotificationRegistry:
    def __init__(self) -> None:
        self.notification_id: str | None = None

    def save_notification_if_absent(self, candidate):
        if self.notification_id is None:
            self.notification_id = candidate.notification_id
            return candidate
        return candidate.model_copy(update={"notification_id": self.notification_id})


def test_ha006_shared_ledger_has_one_winner(tmp_path: Path, monkeypatch) -> None:
    registry = _NotificationRegistry()
    monkeypatch.setattr(ha006_executor, "WINNER", tmp_path / "winner.json")
    pod = ["pod-a"]
    monkeypatch.setattr(socket, "gethostname", lambda: pod[0])
    first = ha006_executor.SharedLedgerAdapter(
        registry, run_id="run-a", sleep_seconds=0
    ).execute(_context("workflow/0/RUN_DCGM_DIAGNOSTIC"))
    pod[0] = "pod-b"
    second = ha006_executor.SharedLedgerAdapter(
        registry, run_id="run-a", sleep_seconds=0
    ).execute(_context("workflow/0/RUN_DCGM_DIAGNOSTIC"))

    assert first.details["cached"] is False
    assert second.details["cached"] is True
    assert (
        first.details["shared_notification_id"]
        == (second.details["shared_notification_id"])
    )
    assert json.loads((tmp_path / "winner.json").read_text())["pod"] == "pod-a"


def test_node_host_probe_reads_fabric_manager_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    import sqlite3

    ledger = tmp_path / "node-actions.db"
    connection = sqlite3.connect(ledger)
    try:
        connection.execute(
            """
            CREATE TABLE results (
                command_id TEXT,
                completed_at TEXT,
                attempt INTEGER,
                state TEXT,
                operation TEXT,
                started_at TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "command-a",
                    "2026-08-31T07:00:02Z",
                    1,
                    "SUCCEEDED",
                    "RESTART_FABRIC_MANAGER",
                    "2026-08-31T07:00:01Z",
                ),
                (
                    "command-b",
                    "2026-08-31T07:00:03Z",
                    1,
                    "SUCCEEDED",
                    "RUN_DCGM_DIAGNOSTIC",
                    "2026-08-31T07:00:01Z",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setattr(node_host_probe, "LEDGER", ledger)

    rows = node_host_probe.ledger_rows()

    assert rows == [
        {
            "command_id": "command-a",
            "completed_at": "2026-08-31T07:00:02Z",
            "attempt": 1,
            "state": "SUCCEEDED",
            "operation": "RESTART_FABRIC_MANAGER",
            "started_at": "2026-08-31T07:00:01Z",
        }
    ]


def test_node_host_probe_timer_snapshot_ignores_countdown_text(monkeypatch) -> None:
    monkeypatch.setattr(
        node_host_probe,
        "run",
        lambda _command: SimpleNamespace(
            stdout=(
                "Mon 2026-08-31 08:00:00 UTC 10min left "
                "gpu-fault-certificate-check.timer "
                "gpu-fault-certificate-check.service\n"
            )
        ),
    )

    assert node_host_probe.gpu_fault_timers() == ["gpu-fault-certificate-check.timer"]


def test_node_host_probe_counts_systemd_unit_transition_once(monkeypatch) -> None:
    lines = [
        {"MESSAGE": 'Started "Nvidia Fabric Manager"'},
        {
            "MESSAGE": (
                "Started nvidia-fabricmanager.service - NVIDIA fabric manager service."
            )
        },
        {"MESSAGE": 'Stopped "Nvidia Fabric Manager"'},
        {
            "MESSAGE": (
                "Stopped nvidia-fabricmanager.service - NVIDIA fabric manager service."
            )
        },
    ]
    monkeypatch.setattr(
        node_host_probe,
        "run",
        lambda _command: SimpleNamespace(
            stdout="\n".join(json.dumps(item) for item in lines)
        ),
    )

    summary = node_host_probe.journal_summary(1.0)

    assert summary["started_count"] == 1
    assert summary["stopped_count"] == 1


def test_destructive_sampler_records_a_timed_out_reading_instead_of_dying(
    monkeypatch,
) -> None:
    """During the reset the sampler exists to observe, `nvidia-smi` blocks and
    the 10s bound trips; the sampler used to die on that exact sample."""

    def hang(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, 10)

    monkeypatch.setattr(destructive_node_probe, "run", hang)

    sample = destructive_node_probe.gpu_sample()

    assert sample["timed_out"] is True
    assert sample["gpu_count"] is None
    assert sample["returncode"] is None


def test_destructive_sampler_start_stops_the_unit_when_no_samples_arrive(
    tmp_path: Path, monkeypatch
) -> None:
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="", returncode=0)

    clock = {"now": 0.0}

    def monotonic() -> float:
        clock["now"] += 31.0
        return clock["now"]

    monkeypatch.setattr(destructive_node_probe, "run", run)
    monkeypatch.setattr(destructive_node_probe, "SAMPLER_DIR", tmp_path)
    monkeypatch.setattr(
        destructive_node_probe,
        "sampler_summary",
        lambda _run_id: {"sample_count": 0, "active": True},
    )
    monkeypatch.setattr(destructive_node_probe.time, "monotonic", monotonic)
    monkeypatch.setattr(destructive_node_probe.time, "sleep", lambda _s: None)
    arguments = SimpleNamespace(
        run_id="run-a",
        probe_script=destructive_node_probe.__file__,
        duration_seconds=600,
        interval_seconds=0.25,
    )

    with pytest.raises(destructive_node_probe.ProbeError, match="unit stopped"):
        destructive_node_probe.start_sampler(arguments)

    unit, _path = destructive_node_probe.sampler_paths("run-a")
    systemd_run = next(command for command in commands if command[0] == "systemd-run")
    assert systemd_run[systemd_run.index("--property=KillMode=control-group") + 1] == (
        sys.executable
    ), systemd_run
    assert "/opt/gpu-fault/venv/bin/python" not in systemd_run
    after_start = commands[commands.index(systemd_run) + 1 :]
    assert ["systemctl", "stop", unit + ".service"] in after_start, after_start
    assert ["systemctl", "reset-failed", unit + ".service"] in after_start


def test_node_host_probe_commands_are_time_bounded(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(node_host_probe.subprocess, "run", fake_run)

    node_host_probe.run(["systemctl", "show", "x"])
    assert seen["timeout"] == 120, seen
    node_host_probe.run(["systemctl", "show", "x"], timeout=30)
    assert seen["timeout"] == 30, seen
