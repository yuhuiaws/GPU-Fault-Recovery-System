"""Host probe commands, file/SQL boundaries and refusal paths, all pre-faked."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.probes import node_host_probe as probe
from tests.regional import _cov95_residual_support as support

residual_isolation = support.residual_isolation


@pytest.mark.parametrize("check", [False, True])
def test_command_wrapper_preserves_timeout_and_checks_failed_exit(monkeypatch, check):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv, 7, stdout="partial", stderr="private failure"
        )

    monkeypatch.setattr(probe.subprocess, "run", run)
    if check:
        with pytest.raises(probe.ProbeError, match="command failed \\(7\\)"):
            probe.run(["systemctl", "show", "fake-unit"], timeout=9)
    else:
        assert (
            probe.run(["systemctl", "show", "fake-unit"], check=False, timeout=9).stdout
            == "partial"
        )
    assert calls[0][1]["timeout"] == 9
    assert calls[0][1]["check"] is False


def test_snapshot_assembles_only_fake_host_observations(monkeypatch, tmp_path, capsys):
    boot = tmp_path / "boot"
    boot.write_text("private-boot\n")
    kmsg = tmp_path / "kmsg"
    kmsg.write_text("")
    ledger = tmp_path / "ledger.db"
    ledger.write_text("placeholder")
    monkeypatch.setattr(probe, "LEDGER", ledger)
    support.redirect_path(
        monkeypatch, probe, {"/proc/sys/kernel/random/boot_id": boot, "/dev/kmsg": kmsg}
    )
    monkeypatch.setattr(probe.os, "access", lambda path, mode: path == "/dev/kmsg")
    commands, queries, closed = [], [], []

    def command(argv, **kwargs):
        commands.append(argv)
        if argv[0] == "nvidia-smi":
            text = (
                "\n00000000:1A:0B.0\n"
                if "pci.bus_id" in argv[1]
                else "malformed\nGPU-a, not-pid, unknown\nGPU-a, 23, python, worker\n"
            )
        elif "show" in argv:
            text = "unparsed\nActiveState=active\nMainPID=23\n"
        elif "list-timers" in argv:
            text = "remaining 1h gpu-fault-b.timer gpu-fault-a.timer\ngpu-fault-b.timer"
        else:
            text = "\n".join(
                [
                    "not-json",
                    json.dumps({"MESSAGE": ""}),
                    json.dumps(
                        {
                            "MESSAGE": "Started nvidia-fabricmanager.service",
                            "__REALTIME_TIMESTAMP": "1",
                        }
                    ),
                    json.dumps(
                        {
                            "MESSAGE": "Stopped nvidia-fabricmanager.service",
                            "_SYSTEMD_INVOCATION_ID": "invoke-a",
                        }
                    ),
                ]
            )
        return subprocess.CompletedProcess(argv, 0, stdout=text, stderr="")

    def execute(query):
        queries.append(query)
        return SimpleNamespace(
            fetchall=lambda: [
                (
                    "command-a",
                    "ended",
                    1,
                    "completed",
                    "RESTART_FABRIC_MANAGER",
                    "started",
                )
            ]
        )

    def connect(database, **kwargs):
        assert database == f"file:{ledger}?mode=ro"
        assert kwargs == {"uri": True}
        return SimpleNamespace(execute=execute, close=lambda: closed.append(True))

    monkeypatch.setattr(probe.subprocess, "run", command)
    monkeypatch.setattr(probe.sqlite3, "connect", connect)
    monkeypatch.setattr(sys, "argv", ["host-probe", "snapshot", "--since-epoch", "1"])
    assert probe.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["boot_id"] == "private-boot"
    assert result["kmsg_exists"] is True and result["kmsg_writable"] is True
    assert result["gpu_pci_bdf"] == "0000:1a:0b"
    assert result["compute_clients"] == [
        {"gpu_uuid": "GPU-a", "pid": "23", "process_name": "python, worker"}
    ]
    assert result["fabric_manager"] == {"ActiveState": "active", "MainPID": "23"}
    assert result["ledger"][0]["command_id"] == "command-a"
    assert result["gpu_fault_timers"] == ["gpu-fault-a.timer", "gpu-fault-b.timer"]
    assert result["journal"]["started_count"] == result["journal"]["stopped_count"] == 1
    assert len(commands) == 5
    assert len(queries) == 1
    assert queries[0].strip().startswith("SELECT "), "ledger access must stay read-only"
    assert closed == [True]


@pytest.mark.parametrize(
    "before,after,code",
    [("active", "active", 0), ("inactive", "active", 0), ("inactive", "failed", 1)],
)
def test_recovery_command_is_conditional_and_verified(
    monkeypatch, before, after, code, capsys
):
    observations = iter([before, after])
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        text = f"ActiveState={next(observations)}\n" if "show" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout=text, stderr="")

    monkeypatch.setattr(probe.subprocess, "run", command)
    monkeypatch.setattr(sys, "argv", ["host-probe", "ensure-fabric-manager-active"])
    assert probe.main() == code
    result = json.loads(capsys.readouterr().out)
    assert sum("restart" in call for call in calls) == int(before != "active")
    if code:
        assert "not active after recovery" in result["error"]
    else:
        assert result["restarted"] is (before != "active")


@pytest.mark.parametrize("mode", ["complete", "short", "write-error"])
def test_xid_write_closes_fake_descriptor_and_never_accepts_incomplete_record(
    monkeypatch, mode, capsys
):
    opened, writes, closed = [], [], []

    def open_device(path, flags):
        opened.append((path, flags))
        return 23

    def write(fd, data):
        writes.append((fd, data))
        if mode == "write-error":
            raise OSError("fake device write failed")
        return len(data) - int(mode == "short")

    arguments = argparse.Namespace(
        marker="marker-a", drill_id="drill-a", pci_bdf="0000:01:02"
    )
    # probe.os is the global os module, which pytest's own tmp_path teardown
    # also uses; keep the fakes to the call under test.
    with monkeypatch.context() as patch:
        patch.setattr(probe.os, "open", open_device)
        patch.setattr(probe.os, "write", write)
        patch.setattr(probe.os, "close", closed.append)
        if mode == "complete":
            probe.write_xid45(arguments)
            result = json.loads(capsys.readouterr().out)
            assert result["bytes_written"] == len(writes[0][1])
        elif mode == "short":
            with pytest.raises(probe.ProbeError, match="short"):
                probe.write_xid45(arguments)
        else:
            with pytest.raises(OSError, match="write failed"):
                probe.write_xid45(arguments)
    assert opened[0][0] == "/dev/kmsg"
    assert closed == [23]
    assert writes[0][1].endswith(b"\n"), (
        "kernel record framing must include a terminator"
    )


@pytest.mark.parametrize(
    "field,value",
    [("marker", "bad marker"), ("drill_id", ""), ("pci_bdf", "0000:01:02.0")],
)
def test_invalid_injection_identity_is_refused_before_open(monkeypatch, field, value):
    arguments = argparse.Namespace(
        marker="marker-a", drill_id="drill-a", pci_bdf="0000:01:02"
    )
    setattr(arguments, field, value)
    with monkeypatch.context() as patch:
        patch.setattr(probe.os, "open", support.forbidden)
        with pytest.raises(probe.ProbeError, match="unsafe"):
            probe.write_xid45(arguments)


def test_unparseable_gpu_identity_and_ledger_read_failure_are_not_absence(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, stdout="not-a-bdf", stderr=""
        ),
    )
    with pytest.raises(probe.ProbeError, match="cannot parse"):
        probe.first_gpu_bdf()
    monkeypatch.setattr(probe, "LEDGER", tmp_path / "absent")
    assert probe.ledger_rows() == []
    path = tmp_path / "ledger"
    path.write_text("placeholder")
    monkeypatch.setattr(probe, "LEDGER", path)
    closed = []

    def fail(query):
        raise OSError("fake ledger unavailable")

    monkeypatch.setattr(
        probe.sqlite3,
        "connect",
        lambda *args, **kwargs: SimpleNamespace(
            execute=fail, close=lambda: closed.append(True)
        ),
    )
    with pytest.raises(OSError, match="ledger unavailable"):
        probe.ledger_rows()
    assert closed == [True]
    assert probe.journal_summary(None)["entry_count"] == 0
