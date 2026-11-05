"""The GF-GENERAL-LIVE isolation probe on the host paths its first suite skips.

``test_general_live_network_isolation.py`` runs the probe over a recording fake
host with ``run``/``emit``/``connectivity``/``resolve_ipv4`` replaced. These
tests keep that fake host for the command paths (a deadman that never arms,
the deadman's own cleanup run, a timer that stays armed, a partial cleanup, a
record the operator must look at) and run the probe's own subprocess, socket
and output boundaries against narrow stand-ins: one ``subprocess.run`` call,
one ``getaddrinfo``, one ``create_connection``, one printed JSON line. The file
also runs as the script the deadman executes.
"""

from __future__ import annotations

import json
import os
import runpy
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_isolation_probe as probe
from tests.regional._general_live_isolation_fakes import POD_IP
from tests.regional.test_general_live_network_isolation import RUN_ID, FakeHost

BLOCK_AGENT = [
    "block",
    "--run-id",
    RUN_ID,
    "--kind",
    "agent-ingress",
    "--deadman-seconds",
    "900",
]
BLOCK_POD = [
    "block",
    "--run-id",
    RUN_ID,
    "--kind",
    "pod-egress",
    "--deadman-seconds",
    "540",
    "--pod-ip",
    POD_IP,
    "--host",
    "cp.example.internal",
    "--port",
    "443",
]


def completed(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout, "")


class EdgeHost(FakeHost):
    """The neighbour suite's fake host plus the host behaviours it never shows."""

    def __init__(self) -> None:
        super().__init__()
        self.arm_fails = False
        self.stuck_timers = False
        self.deadman_pid: str | None = None
        self.junk_properties = False
        self.ss_lines: list[str] | None = None
        self.ip_lines: list[str] | None = None

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        command = list(command)
        name = Path(command[0]).name
        verb = command[1] if len(command) > 1 else ""
        if name == "systemd-run" and self.arm_fails:
            # Accepted, but no timer ever shows up (a refused unit name).
            self.calls.append(command)
            return completed()
        if name == "systemctl" and verb == "stop" and self.stuck_timers:
            self.calls.append(command)
            return completed()
        if (
            name == "systemctl"
            and verb == "show"
            and self.deadman_pid is not None
            and probe.DEADMAN_PREFIX in command[2]
            and command[2].endswith(".service")
        ):
            self.calls.append(command)
            return completed(
                f"ActiveState=active\nSubState=running\nMainPID={self.deadman_pid}"
            )
        if name == "ss" and self.ss_lines is not None:
            self.calls.append(command)
            return completed("\n".join(self.ss_lines))
        if name == "ip" and self.ip_lines is not None:
            self.calls.append(command)
            return completed("\n".join(self.ip_lines))
        result = super().run(command, **kwargs)
        if name == "systemctl" and verb == "show" and self.junk_properties:
            return completed(result.stdout + "\n\nsystemd: a line without a property")
        return result


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> EdgeHost:
    fake = EdgeHost()
    monkeypatch.setattr(probe, "run", fake.run)
    monkeypatch.setattr(probe, "emit", fake.emitted.append)
    monkeypatch.setattr(probe, "connectivity", fake.connectivity)
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/usr/sbin/{name}")
    monkeypatch.setattr(
        probe,
        "resolve_ipv4",
        lambda host: ["203.0.113.10", "203.0.113.11"]
        if host == "cp.example.internal"
        else [host],
    )
    return fake


def recorded_state() -> dict[str, Any]:
    path = probe.state_dir(RUN_ID) / probe.STATE_FILE
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# the probe's own boundaries
# --------------------------------------------------------------------------- #
def test_run_fails_with_the_command_and_its_stderr_only_when_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[list[str], dict[str, Any]]] = []
    codes = [3, 3, 0]

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append((command, kwargs))
        return subprocess.CompletedProcess(command, codes.pop(0), "partial", "e" * 400)

    monkeypatch.setattr(
        probe, "subprocess", SimpleNamespace(run=fake_run, PIPE=subprocess.PIPE)
    )

    with pytest.raises(probe.ProbeError) as raised:
        probe.run(["iptables", "-S", "INPUT"], timeout=7)
    unchecked = probe.run(["iptables", "-S", "INPUT"], check=False)
    passed = probe.run(["ip", "-4", "-o", "addr", "show"])

    assert str(raised.value) == "command failed (3): iptables -S INPUT: " + "e" * 300
    assert (unchecked.returncode, unchecked.stdout) == (3, "partial")
    assert (passed.returncode, passed.stdout) == (0, "partial")
    assert seen[0][1] == {
        "text": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "timeout": 7,
        "check": False,
    }
    assert [kwargs["timeout"] for _command, kwargs in seen[1:]] == [120, 120]


def test_emit_prints_one_compact_sorted_json_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe.emit({"run_id": RUN_ID, "path": Path("/run/gpu-fault"), "clean": True})

    assert capsys.readouterr().out == (
        f'{{"clean":true,"path":"/run/gpu-fault","run_id":"{RUN_ID}"}}\n'
    )


def test_resolve_refuses_unsafe_names_and_looks_up_only_ipv4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[tuple[str, Any, Any]] = []

    def getaddrinfo(host: str, port: Any, family: Any) -> list[Any]:
        asked.append((host, port, family))
        if host == "missing.example.internal":
            raise socket.gaierror(-2, "Name or service not known")
        return [
            (family, socket.SOCK_STREAM, 6, "", (address, 0))
            for address in ("203.0.113.11", "203.0.113.10", "203.0.113.11")
        ]

    monkeypatch.setattr(
        probe,
        "socket",
        SimpleNamespace(
            getaddrinfo=getaddrinfo, gaierror=socket.gaierror, AF_INET=socket.AF_INET
        ),
    )

    with pytest.raises(probe.ProbeError, match="unsafe endpoint host"):
        probe.resolve_ipv4("cp.example.internal;reboot")
    assert probe.resolve_ipv4("203.0.113.7") == ["203.0.113.7"]
    assert probe.resolve_ipv4("cp.example.internal") == [
        "203.0.113.10",
        "203.0.113.11",
    ], "deduplicated and sorted"
    with pytest.raises(probe.ProbeError, match="cannot resolve missing.example"):
        probe.resolve_ipv4("missing.example.internal")
    assert asked == [
        ("cp.example.internal", None, socket.AF_INET),
        ("missing.example.internal", None, socket.AF_INET),
    ], "an address literal is never looked up"


def test_connectivity_is_one_bounded_tcp_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[tuple[tuple[str, int], float]] = []

    class Connection:
        def __enter__(self) -> Connection:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def create_connection(address: tuple[str, int], timeout: float) -> Connection:
        attempts.append((address, timeout))
        if address[1] == 9099:
            return Connection()
        raise ConnectionRefusedError("connection reset by the tagged rule")

    monkeypatch.setattr(
        probe, "socket", SimpleNamespace(create_connection=create_connection)
    )

    assert probe.connectivity("10.0.1.5", 9099) is True
    assert probe.connectivity("10.0.1.5", 9100) is False
    assert attempts == [(("10.0.1.5", 9099), 3), (("10.0.1.5", 9100), 3)]


# --------------------------------------------------------------------------- #
# host readings
# --------------------------------------------------------------------------- #
def test_unit_state_keeps_only_property_lines(host: EdgeHost) -> None:
    host.junk_properties = True

    assert probe.unit_state(probe.AGENT_UNIT) == {
        "ActiveState": "active",
        "SubState": "running",
        "MainPID": "4242",
    }


def test_the_agent_listing_reads_only_its_own_concrete_ipv4_listeners(
    host: EdgeHost,
) -> None:
    host.ss_lines = [
        'LISTEN 0 128 10.0.1.5:9099 0.0.0.0:* users:(("python",pid=4242,fd=7))',
        'LISTEN 0 128 10.0.1.5:9100 0.0.0.0:* users:(("kubelet",pid=777,fd=7))',
        'users:(("python",pid=4242,fd=8))',
        'LISTEN 0 128 [::]:9101 [::]:* users:(("python",pid=4242,fd=9))',
        'LISTEN 0 128 0.0.0.0:9102 0.0.0.0:* users:(("python",pid=4242,fd=10))',
        'LISTEN 0 128 127.0.0.1:9103 0.0.0.0:* users:(("python",pid=4242,fd=11))',
        'LISTEN 0 128 10.0.1.6:* 0.0.0.0:* users:(("python",pid=4242,fd=12))',
    ]

    agent = probe.agent_listen_ports(probe.AGENT_UNIT)

    assert agent == {
        "unit": probe.AGENT_UNIT,
        "active_state": "active",
        "main_pid": 4242,
        # Another process's port and a line without columns are not the
        # Agent's; wildcard and loopback listeners give a port, no address.
        "ports": [9099, 9101, 9102, 9103],
        "addresses": ["10.0.1.5", "10.0.1.6"],
    }


def test_host_addresses_read_only_ipv4_inet_lines(host: EdgeHost) -> None:
    host.ip_lines = [
        "1: lo    inet 127.0.0.1/8 scope host lo",
        "2: eth0    inet6 fe80::1/64 scope link",
        "3: eth1    inet 10.0.1.9/24 brd 10.0.1.255 scope global eth1",
    ]

    assert probe.host_addresses() == ["10.0.1.9", "127.0.0.1"]


# --------------------------------------------------------------------------- #
# block and cleanup on the host
# --------------------------------------------------------------------------- #
def test_a_host_without_iptables_refuses_before_any_command(
    host: EdgeHost, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(probe.shutil, "which", lambda name: None)

    assert probe.main(BLOCK_AGENT) == 1

    error = host.emitted[-1]
    assert error["error_class"] == "ProbeError" and error["command"] == "block"
    assert error["error_site"].startswith("firewall:"), error
    assert capsys.readouterr().err == (
        "ProbeError: iptables is not installed on the host\n"
    )
    assert host.calls == [], "nothing ran on a host without the firewall"


def test_a_deadman_that_does_not_arm_stops_the_block_before_any_rule(
    host: EdgeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    host.arm_fails = True

    assert probe.main(BLOCK_AGENT) == 1

    unit = probe.deadman_unit(RUN_ID, "agent-ingress")
    assert host.emitted[-1]["error_site"].startswith("arm_deadman:"), host.emitted
    assert capsys.readouterr().err == f"ProbeError: deadman {unit} is not armed\n"
    assert host.chains["INPUT"] == [], "no rule without an armed deadman"
    assert not any(call[1:2] == ["-I"] for call in host.calls), host.calls
    entry = recorded_state()["blocks"]["agent-ingress"]
    assert entry["tag"] == probe.tag(RUN_ID, "agent-ingress"), (
        "the intent stays recorded, so a later cleanup finds the run"
    )


def test_the_deadmans_own_cleanup_never_stops_its_own_service(host: EdgeHost) -> None:
    assert probe.main(BLOCK_AGENT) == 0
    host.deadman_pid = str(os.getpid())

    argv = ["cleanup", "--run-id", RUN_ID, "--kind", "agent-ingress", "--strict"]
    assert probe.main(argv) == 0, host.emitted[-1]

    unit = probe.deadman_unit(RUN_ID, "agent-ingress")
    stops = [call for call in host.calls if call[1:2] == ["stop"]]
    assert ["systemctl", "stop", f"{unit}.timer"] in stops, stops
    assert ["systemctl", "stop", f"{unit}.service"] not in stops, (
        "the running cleanup is the deadman's service itself"
    )
    assert host.emitted[-1]["clean"] is True
    assert host.chains["INPUT"] == []


def test_a_timer_that_stays_armed_is_not_clean_and_strict_fails(
    host: EdgeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    assert probe.main(BLOCK_AGENT) == 0
    host.stuck_timers = True
    argv = ["cleanup", "--run-id", RUN_ID, "--kind", "agent-ingress"]

    assert probe.main(argv) == 0

    report = host.emitted[-1]
    item = report["kinds"]["agent-ingress"]
    assert report["clean"] is False, report
    assert (item["deleted"], item["remaining"]) == (1, []), item
    assert item["timer"]["ActiveState"] == "active"

    assert probe.main([*argv, "--strict"]) == 1

    assert host.emitted[-2]["clean"] is False, "the report is emitted first"
    assert host.emitted[-1]["error_site"].startswith("cleanup:"), host.emitted[-1]
    assert capsys.readouterr().err == (
        "ProbeError: isolation cleanup left tagged rules or an armed timer\n"
    )


def test_cleaning_one_kind_keeps_the_record_and_copy_of_the_other(
    host: EdgeHost,
) -> None:
    assert probe.main(BLOCK_AGENT) == 0
    assert probe.main(BLOCK_POD) == 0

    assert probe.main(["cleanup", "--run-id", RUN_ID, "--kind", "agent-ingress"]) == 0

    assert host.emitted[-1]["clean"] is True
    assert list(recorded_state()["blocks"]) == ["pod-egress"]
    assert (probe.state_dir(RUN_ID) / "isolation-probe.py").is_file(), (
        "the pod-egress deadman still needs its copy of the probe"
    )
    assert host.chains["INPUT"] == [] and len(host.chains["FORWARD"]) == 2


def test_a_foreign_file_keeps_the_state_directory_after_the_last_cleanup(
    host: EdgeHost,
) -> None:
    assert probe.main(BLOCK_AGENT) == 0
    foreign = probe.state_dir(RUN_ID) / "operator-notes.txt"
    foreign.write_text("kept", encoding="utf-8")

    assert probe.main(["cleanup", "--run-id", RUN_ID]) == 0

    assert host.emitted[-1]["clean"] is True
    assert foreign.read_text(encoding="utf-8") == "kept", "never removed"
    assert sorted(path.name for path in probe.state_dir(RUN_ID).iterdir()) == [
        "operator-notes.txt"
    ], "the record and the probe copy are gone"


def test_delete_refuses_a_listed_rule_of_another_chain(
    host: EdgeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    rule_tag = probe.tag(RUN_ID, "agent-ingress")
    line = f"-A FORWARD -p tcp --dport 9099 -m comment --comment {rule_tag} -j REJECT"
    monkeypatch.setattr(probe, "tagged_rules", lambda kind, wanted: [line])

    with pytest.raises(probe.ProbeError, match="unexpected rule shape: -A FORWARD"):
        probe.delete_tagged("agent-ingress", rule_tag)

    assert not any(call[1:2] == ["-D"] for call in host.calls), host.calls


@pytest.mark.parametrize(
    "content", ["[]", '{"run_id": "x"}', '{"blocks": []}'], ids=str
)
def test_a_malformed_isolation_record_is_an_error_not_a_reset(
    host: EdgeHost, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    path = probe.state_dir(RUN_ID) / probe.STATE_FILE
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")

    assert probe.main(["status", "--run-id", RUN_ID, "--kind", "agent-ingress"]) == 1

    assert capsys.readouterr().err == "ProbeError: isolation state is malformed\n"
    assert host.emitted[-1]["command"] == "status"
    assert path.read_text(encoding="utf-8") == content, "kept for the operator"


def test_the_probe_file_runs_as_the_script_the_deadman_executes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [probe.__file__, "--help"])

    with pytest.raises(SystemExit) as exited:
        runpy.run_path(probe.__file__, run_name="__main__")

    assert exited.value.code == 0
    usage = capsys.readouterr().out
    assert "Reversible host-level network isolation" in usage, usage
    assert "{survey,block,status,cleanup}" in usage, usage
