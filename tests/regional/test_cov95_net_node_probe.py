"""NET-001 host protocol exercised against temporary paths and fake syscalls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import net001_node_probe as probe
from tests.regional._cov95_collect_net import (
    FakeSocket,
    forbidden,
    isolate_paths,
    local_socket_module,
    no_external_effects,  # noqa: F401
)


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    paths = isolate_paths(monkeypatch, probe, tmp_path)
    paths("/run").mkdir()
    paths("/dev").mkdir()
    paths("/dev/kmsg").touch()
    outboxes = {
        name: tmp_path / "outboxes" / f"{name}.ndjson" for name in probe.OUTBOXES
    }
    (tmp_path / "outboxes").mkdir()
    monkeypatch.setattr(probe, "OUTBOXES", outboxes)
    calls: list[list[str]] = []
    rules: dict[str, int] = {}
    state = SimpleNamespace(
        calls=calls, rules=rules, timer="active", paths=paths, writes=[], closed=[]
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"text": True, "capture_output": True, "check": False}
        calls.append(command)
        status, stdout = 0, ""
        if command[0] == "/fake/iptables":
            operation = command[1]
            if operation == "-S":
                stdout = "\n".join(
                    f"-A OUTPUT -d {ip} --comment test-net -j REJECT"
                    for ip, count in rules.items()
                    for _ in range(count)
                )
            else:
                ip = command[command.index("-d") + 1]
                count = rules.get(ip, 0)
                if operation == "-C":
                    status = 0 if count else 1
                elif operation == "-I":
                    rules[ip] = count + 1
                elif operation == "-D":
                    rules[ip] = count - 1
                else:
                    raise AssertionError(f"unexpected firewall operation {operation}")
        elif command[:2] == ["systemctl", "show"]:
            stdout = (
                f"ActiveState={state.timer}\nSubState=running\n"
                "NRestarts=0\nMainPID=123\nignored-line\n"
            )
        elif command[0] == "nvidia-smi":
            stdout = "\n00000000:AF:00.0\n"
        return subprocess.CompletedProcess(command, status, stdout, "")

    monkeypatch.setattr(probe.subprocess, "run", run)
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/fake/{name}")
    monkeypatch.setattr(
        probe,
        "socket",
        local_socket_module(
            getaddrinfo=lambda *a, **k: [
                (None, None, None, None, ("192.0.2.2", 443)),
                (None, None, None, None, ("192.0.2.1", 443)),
                (None, None, None, None, ("192.0.2.2", 443)),
            ],
            create_connection=lambda address, timeout: FakeSocket(),
        ),
    )
    monkeypatch.setattr(
        probe,
        "os",
        SimpleNamespace(
            **{
                **vars(os),
                "access": lambda path, mode: os.access(paths(path), mode),
                "open": lambda path, flags: 42,
                "write": lambda fd, data: state.writes.append((fd, data)) or len(data),
                "close": state.closed.append,
            }
        ),
    )
    return state


def invoke(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["net001-probe", *argv])
    return probe.main()


def test_preflight_reads_local_service_outbox_and_network_facts(
    host: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        invoke(
            monkeypatch,
            "preflight",
            "--endpoint-host",
            "control.invalid",
            "--tag-prefix",
            "test-net",
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["endpoint_ipv4"] == ["192.0.2.1", "192.0.2.2"]
    assert result["gpu_pci_bdf"] == "0000:af:00"
    assert result["kmsg_exists"] is True
    assert result["kmsg_writable"] is True
    assert set(result["services"]) == set(probe.SERVICES)
    assert result["connectivity"] == {"192.0.2.1": True, "192.0.2.2": True}
    assert result["outboxes"]["kernel"]["line_count"] == 0
    assert result["outboxes"]["kernel"]["parent_exists"] is True


@pytest.mark.parametrize("missing", ["systemd-run", "nvidia-smi", "iptables"])
def test_preflight_reports_missing_host_tool_before_mutation(
    host: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any, missing: str
) -> None:
    monkeypatch.setattr(
        probe.shutil, "which", lambda name: None if name == missing else f"/fake/{name}"
    )
    assert (
        invoke(
            monkeypatch,
            "preflight",
            "--endpoint-host",
            "control.invalid",
            "--tag-prefix",
            "test-net",
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert missing in result["error"]
    assert not any(command[0] == "systemd-run" for command in host.calls), (
        "read-only preflight must never arm a timer"
    )


@pytest.mark.parametrize("value", ["", "../escape", "bad value", "x" * 129])
def test_invalid_identifiers_are_rejected(host: Any, value: str) -> None:
    with pytest.raises(probe.ToolError, match="unsafe firewall tag"):
        probe.validate_tag(value)
    with pytest.raises(probe.ToolError, match="unsafe test ID"):
        probe.validate_id(value, "test ID")
    assert host.calls == []


def test_outbox_parses_malformed_records_and_matches_payload_shapes(
    host: Any, tmp_path: Path
) -> None:
    path = tmp_path / "records.ndjson"
    rows = [
        "{bad",
        json.dumps(["not-a-record"]),
        json.dumps(
            {
                "path": "/v1/events",
                "error": "offline",
                "replayable": True,
                "payload": {"record_id": "record-a", "marker": "test-a"},
                "failed_at": "2026-09-01T00:00:00Z",
            }
        ),
        json.dumps({"payload": ["test-a"], "replayable": False}),
        json.dumps({"payload": {"marker": "unrelated"}, "replayable": True}),
    ]
    path.write_text("\n".join(rows) + "\n")
    observed = probe.read_outbox(path, ("test-a", "absent"))
    assert observed["line_count"] == 5
    assert observed["malformed_count"] == 2
    assert observed["replayable_count"] == 2
    assert observed["path_counts"] == {"": 2, "/v1/events": 1}
    assert observed["error_counts"] == {"": 2, "offline": 1}
    assert [item["record_id"] for item in observed["matching"]] == ["record-a", None]
    assert observed["matching"][0]["test_ids"] == ["test-a"]


@pytest.mark.parametrize("success", [True, False])
def test_dns_and_connectivity_report_failed_connections(
    host: Any, monkeypatch: pytest.MonkeyPatch, success: bool
) -> None:
    def connect(address: Any, timeout: float) -> FakeSocket:
        assert address == ("192.0.2.1", 443)
        assert timeout == 3
        if not success:
            raise OSError("unreachable")
        return FakeSocket()

    monkeypatch.setattr(probe.socket, "create_connection", connect)
    assert probe.connectivity(["192.0.2.1"]) == {"192.0.2.1": success}
    monkeypatch.setattr(probe.socket, "getaddrinfo", lambda *a, **k: [])
    with pytest.raises(probe.ToolError, match="no IPv4"):
        probe.resolve_ipv4("control.invalid")


@pytest.mark.parametrize("output", ["", "\n", "not-a-pci-device"])
def test_gpu_identity_refuses_unparseable_output(
    host: Any, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(command, 0, output, ""),
    )
    with pytest.raises(probe.ToolError, match="cannot parse"):
        probe.first_gpu_bdf()


@pytest.mark.parametrize("check", [True, False])
def test_command_failure_is_checked_with_diagnostic(
    host: Any, monkeypatch: pytest.MonkeyPatch, check: bool
) -> None:
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(command, 7, "", " denied \n"),
    )
    if check:
        with pytest.raises(probe.ToolError, match=r"failed \(7\).*denied"):
            probe.run(["fake", "value with spaces"])
    else:
        assert probe.run(["fake"], check=False).returncode == 7


def test_arm_block_and_cleanup_are_ordered_and_idempotent(
    host: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    common = ["--tag", "test-net", "--ip", "192.0.2.1", "--ip", "192.0.2.2"]
    assert invoke(monkeypatch, "arm", *common, "--ttl-seconds", "60") == 0
    armed = json.loads(capsys.readouterr().out)
    script = Path(armed["cleanup_script"])
    assert script.is_relative_to(host.paths("/run")), "cleanup file must stay private"
    assert script.stat().st_mode & 0o777 == 0o700
    assert armed["timer"]["ActiveState"] == "active"
    assert not host.rules, "arming a rollback must not yet change firewall rules"
    assert invoke(monkeypatch, "block", *common) == 0
    blocked = json.loads(capsys.readouterr().out)
    assert len(blocked["rules"]) == 2
    assert invoke(monkeypatch, "block", *common) == 0
    capsys.readouterr()
    assert host.rules == {"192.0.2.1": 1, "192.0.2.2": 1}
    host.rules["192.0.2.1"] = 2
    assert invoke(monkeypatch, "cleanup", *common) == 0
    cleaned = json.loads(capsys.readouterr().out)
    assert cleaned["rules"] == []
    assert not script.exists(), "cleanup must remove the owned rollback script"
    operations = [command[0] for command in host.calls]
    assert operations.index("systemd-run") < next(
        index for index, command in enumerate(host.calls) if "-I" in command
    )
    assert invoke(monkeypatch, "cleanup", *common) == 0
    assert json.loads(capsys.readouterr().out)["rules"] == []


@pytest.mark.parametrize("failure", ["no-ip", "existing-rule", "inactive-timer"])
def test_arm_refuses_missing_destination_existing_rules_or_inactive_timer(
    host: Any, failure: str
) -> None:
    if failure == "existing-rule":
        host.rules["192.0.2.1"] = 1
    if failure == "inactive-timer":
        host.timer = "inactive"
    arguments = SimpleNamespace(
        tag="test-net", ip=[] if failure == "no-ip" else ["192.0.2.1"], ttl_seconds=60
    )
    message = {
        "no-ip": "at least one",
        "existing-rule": "already exist",
        "inactive-timer": "not active",
    }[failure]
    with pytest.raises(probe.ToolError, match=message):
        probe.arm(arguments)
    assert not any("-I" in command for command in host.calls), (
        "failed rollback admission must not install a firewall rule"
    )


@pytest.mark.parametrize("tagged", [True, False])
def test_snapshot_optionally_reads_rollback_state(
    host: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any, tagged: bool
) -> None:
    args = ["snapshot", "--test-id", "test-a"]
    if tagged:
        args.extend(["--tag", "test-net"])
    assert invoke(monkeypatch, *args) == 0
    observed = json.loads(capsys.readouterr().out)
    assert bool(observed["timer"]) is tagged
    assert observed["rules"] == []


@pytest.mark.parametrize(
    "failure", [None, OSError("write failed"), subprocess.SubprocessError("failed")]
)
def test_kmsg_protocol_closes_descriptor_and_reports_write_failure(
    host: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any, failure: Any
) -> None:
    opened: list[Any] = []
    monkeypatch.setattr(
        probe.os, "open", lambda path, flags: opened.append((path, flags)) or 42
    )
    if failure is not None:

        def write(fd: int, data: bytes) -> int:
            raise failure

        monkeypatch.setattr(probe.os, "write", write)
    status = invoke(
        monkeypatch,
        "write-kmsg",
        "--test-id",
        "test-a",
        "--drill-id",
        "drill-a",
        "--pci-bdf",
        "0000:af:00",
    )
    result = json.loads(capsys.readouterr().out)
    assert status == (1 if failure is not None else 0)
    assert opened == [("/dev/kmsg", os.O_WRONLY | os.O_CLOEXEC)]
    assert host.closed == [42]
    if failure is None:
        assert host.writes[0][1] == (
            b"<6>gpu-fault NET-001 test_id=test-a drill_id=drill-a "
            b"NVRM: Xid (PCI:0000:af:00): 63, "
            b"monitor-only row remapping acceptance event\n"
        )
        assert result["bytes_written"] == len(host.writes[0][1])
    else:
        assert "failed" in result["error"]


def test_kmsg_rejects_unsafe_bdf_before_open(host: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(probe.os, "open", forbidden)
    with pytest.raises(probe.ToolError, match="unsafe PCI"):
        probe.write_kmsg(
            SimpleNamespace(test_id="test", drill_id="drill", pci_bdf="bad")
        )
