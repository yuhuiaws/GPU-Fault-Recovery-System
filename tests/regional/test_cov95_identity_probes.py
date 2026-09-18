from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import auth013_certificate_probe as certificate
from scripts.e2e.regional.probes import auth015_node_probe as custody
from scripts.e2e.regional.probes import cluster_network_probe as network
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [(" active\n", "", "active"), ("", "x" * 100, "x" * 80), ("", "", "")],
)
def test_certificate_service_state_has_a_bounded_fallback(
    monkeypatch: pytest.MonkeyPatch, stdout: str, stderr: str, expected: str
) -> None:
    calls = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 1, stdout, stderr)

    monkeypatch.setattr(certificate.subprocess, "run", run)
    assert certificate.systemctl_state("certificate.timer", "is-active") == expected
    assert calls[0][0] == ["systemctl", "is-active", "certificate.timer"]
    assert calls[0][1]["check"] is False
    assert calls[0][1]["timeout"] == 30


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("armed", [False, True])
def test_certificate_scan_reads_only_the_threshold_and_all_timer_states(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, present: bool, armed: bool
) -> None:
    path = tmp_path / "collector.env"
    if present:
        path.write_text(
            "UNRELATED=value\nGPU_FAULT_CERTIFICATE_MIN_VALIDITY_SECONDS='2592000'\n",
            encoding="ascii",
        )
    monkeypatch.setattr(certificate, "ENV_FILE", path)
    calls = []

    def state(unit: str, *query: str) -> str:
        calls.append((unit, query))
        return {
            "is-enabled": "enabled" if armed else "disabled",
            "is-active": "active" if armed else "inactive",
            "show": "success",
        }[query[0]]

    monkeypatch.setattr(certificate, "systemctl_state", state)
    result = certificate.scan("certificate.timer")
    assert result["env_file_exists"] is present
    assert result["min_validity_seconds"] == (2592000 if present else None)
    assert result["timer_enabled"] is result["timer_active"] is armed
    assert result["last_service_result"] == "success"
    assert calls == [
        ("certificate.timer", ("is-enabled",)),
        ("certificate.timer", ("is-active",)),
        ("certificate.service", ("show", "-p", "Result", "--value")),
    ]


@pytest.mark.parametrize("unit", ["", "../timer", "timer;exit", "-timer"])
def test_certificate_scan_rejects_unsafe_units_before_reads(
    monkeypatch: pytest.MonkeyPatch, unit: str
) -> None:
    calls = []
    monkeypatch.setattr(
        certificate, "systemctl_state", lambda *args: calls.append(args)
    )
    with pytest.raises(certificate.ProbeError, match="unsafe"):
        certificate.scan(unit)
    assert calls == []


@pytest.mark.parametrize(
    "error",
    [
        None,
        OSError("missing"),
        certificate.ProbeError("invalid"),
        subprocess.TimeoutExpired("unit", 30),
    ],
)
def test_certificate_main_reports_success_or_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception | None,
) -> None:
    monkeypatch.setattr(sys, "argv", ["certificate", "--timer", "unit.timer"])

    def scan(unit: str) -> dict[str, Any]:
        assert unit == "unit.timer"
        if error:
            raise error
        return {"timer_active": True}

    monkeypatch.setattr(certificate, "scan", scan)
    assert certificate.main() == int(error is not None)
    result = json.loads(capsys.readouterr().out)
    if error:
        assert result["error"].startswith(type(error).__name__ + ":"), result
    else:
        assert result == {"timer_active": True}


def scan_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[list[Path], Path]:
    roots = [tmp_path / name for name in ("tmp", "environment", "pods")]
    proc = tmp_path / "proc"
    for directory in [*roots, proc]:
        directory.mkdir()
    monkeypatch.setattr(custody, "SCAN_ROOTS", tuple(roots))
    monkeypatch.setattr(custody, "PROC_ROOT", proc)
    return roots, proc


def test_custody_scan_counts_only_nonempty_values_and_matches_process_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roots, proc = scan_roots(monkeypatch, tmp_path)
    synthetic = b"example-custody-value"
    (roots[0] / "empty").write_bytes(b"")
    (roots[1] / "unit.env").write_bytes(
        b"COMMENT\nEMPTY=\nSAFE='example-custody-value'\n"
    )
    process = proc / "123"
    process.mkdir()
    (process / "environ").write_bytes(b"EMPTY=\0IGNORED\0SAFE=" + synthetic + b"\0")
    (proc / "456").mkdir()  # A listed process can exit before environ is read.
    (proc / "self").mkdir()
    result = custody.scan(hashlib.sha256(synthetic).hexdigest())
    assert result["master_matches"] == [
        str(roots[1] / "unit.env") + ":env",
        str(process / "environ") + ":environ",
    ]
    assert result["values_scanned"] == 4
    assert result["host_tmp_exists"] is result["systemd_environment_exists"] is True
    assert synthetic.decode() not in json.dumps(result)


def test_custody_scan_rejects_a_missing_proc_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(custody, "PROC_ROOT", tmp_path / "absent")
    with pytest.raises(custody.ProbeError, match="proc directory"):
        list(custody.process_environments())


def test_custody_scan_does_not_ignore_unreadable_process_environments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _roots, proc = scan_roots(monkeypatch, tmp_path)
    (proc / "123").mkdir()
    (proc / "123" / "environ").mkdir()
    with pytest.raises(custody.ProbeError, match="environment could not be inspected"):
        list(custody.process_environments())


def test_custody_scan_fails_when_directory_walk_loses_visibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scan_roots(monkeypatch, tmp_path)

    def walk(root: Path, *, onerror: Any) -> Any:
        onerror(PermissionError("synthetic denial"))
        return iter(())

    monkeypatch.setattr(custody.os, "walk", walk)
    with pytest.raises(custody.ProbeError, match="directory could not be inspected"):
        list(custody.candidate_files())


@pytest.mark.parametrize("failure", ["oversize", "stat", "read"])
def test_custody_scan_refuses_uninspectable_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    roots, _proc = scan_roots(monkeypatch, tmp_path)
    path = roots[0] / "payload"
    path.write_bytes(b"abcdef")
    if failure == "oversize":
        monkeypatch.setattr(custody, "MAX_SCAN_BYTES", 2)
        message = "bounded scan size"
    else:
        original = Path.stat if failure == "stat" else Path.read_bytes

        def failed_read(self: Path, *args: Any, **kwargs: Any) -> Any:
            if self == path:
                raise PermissionError("synthetic denial")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(
            Path, "stat" if failure == "stat" else "read_bytes", failed_read
        )
        message = "could not be inspected" if failure == "stat" else "could not be read"
    with pytest.raises(custody.ProbeError, match=message):
        custody.scan("a" * 64)


@pytest.mark.parametrize(
    "error", [None, custody.ProbeError("unknown custody"), OSError("unreadable")]
)
def test_custody_main_has_no_pass_for_unproved_scan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception | None,
) -> None:
    monkeypatch.setattr(sys, "argv", ["custody", "--master-sha256", "a" * 64])

    def scan(digest: str) -> dict[str, Any]:
        assert digest == "a" * 64
        if error:
            raise error
        return {"master_matches": [], "values_scanned": 2}

    monkeypatch.setattr(custody, "scan", scan)
    assert custody.main() == int(error is not None)
    result = json.loads(capsys.readouterr().out)
    assert result == (
        {"error": str(error)} if error else {"master_matches": [], "values_scanned": 2}
    )


@pytest.mark.parametrize(
    ("cidrs", "message"),
    [
        (["not-a-cidr"], "invalid"),
        (["0.0.0.0/0"], "default-route"),
        (["::/0"], "default-route"),
        ([], "explicit IPv4"),
        (["2001:db8::/64"], "explicit IPv4"),
    ],
)
def test_network_target_normalization_fails_closed(
    cidrs: list[str], message: str
) -> None:
    with pytest.raises(network.ProbeError, match=message):
        network.normalized_cidrs(cidrs)


@pytest.mark.parametrize("check", [True, False])
def test_network_command_wrapper_does_not_treat_failure_as_success(
    monkeypatch: pytest.MonkeyPatch, check: bool
) -> None:
    calls = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 2, "", "synthetic rejection")

    monkeypatch.setattr(network.subprocess, "run", run)
    if check:
        with pytest.raises(network.ProbeError, match="command failed \\(2\\)"):
            network.run(["iptables", "-S"])
    else:
        assert network.run(["iptables", "-S"], check=False).returncode == 2
    assert calls[0]["check"] is False
    assert calls[0]["timeout"] == 120


class NetworkTransport:
    def __init__(self, failure: str = "") -> None:
        self.failure = failure
        self.commands: list[list[str]] = []
        self.rules: list[list[str]] = []

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[:2] == ["iptables", "-S"]:
            import shlex

            return subprocess.CompletedProcess(
                command, 0, "\n".join(shlex.join(rule) for rule in self.rules), ""
            )
        if command[0] != "iptables":
            assert command[0] in {"systemd-run", "systemctl"}
            return subprocess.CompletedProcess(command, 0, "", "")
        operation, *arguments = command[1:]
        if self.failure == "append" and operation == "-A":
            raise network.ProbeError("synthetic append failure")
        if operation == "-N":
            self.rules.append(["-N", *arguments])
        elif operation == "-A":
            self.rules.append(["-A", *arguments])
        elif operation == "-I":
            if self.failure != "missing-forward" or arguments[0] != "FORWARD":
                self.rules.append(["-A", arguments[0], *arguments[2:]])
        elif operation == "-D":
            rule = ["-A", *arguments]
            if rule in self.rules:
                self.rules.remove(rule)
        elif operation == "-F":
            self.rules = [
                rule for rule in self.rules if rule[:2] != ["-A", arguments[0]]
            ]
        elif operation == "-X":
            self.rules.remove(["-N", *arguments])
        else:
            raise AssertionError(f"unexpected fake operation: {operation}")
        return subprocess.CompletedProcess(command, 0, "", "")


@pytest.mark.parametrize("failure", ["", "append", "missing-forward"])
def test_network_lifecycle_arms_before_reject_and_verifies_both_jump_cleanup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    transport = NetworkTransport(failure)
    monkeypatch.setattr(network, "run", transport.run)
    args = network.parser().parse_args(
        ["block", "--run-id", "unit-cut", "--control-plane-cidr", "10.0.0.5/24"]
    )
    if failure:
        with pytest.raises(network.ProbeError, match="append failure|read back"):
            args.handler(args)
    else:
        args.handler(args)
        blocked = json.loads(capsys.readouterr().out)
        assert blocked["blocked"] is True
        assert blocked["cidrs"] == ["10.0.0.0/24"]
        assert blocked["host_chains"] == ["OUTPUT", "FORWARD"]
        network.status(SimpleNamespace(run_id="unit-cut"))
        status = json.loads(capsys.readouterr().out)
        assert status["blocked"] is status["residual"] is True
        network.unblock(SimpleNamespace(run_id="unit-cut"))
        assert json.loads(capsys.readouterr().out)["residual"] is False
    assert transport.rules == []
    commands = transport.commands
    arm = next(
        index for index, command in enumerate(commands) if command[0] == "systemd-run"
    )
    create = next(
        index
        for index, command in enumerate(commands)
        if command[:2] == ["iptables", "-N"]
    )
    cancel = next(
        index
        for index, command in enumerate(commands)
        if command[:2] == ["systemctl", "stop"]
    )
    delete = next(
        index
        for index, command in enumerate(commands)
        if command[:2] == ["iptables", "-X"]
    )
    assert arm < create < delete < cancel
    assert commands[delete + 1] == ["iptables", "-S"], (
        "cleanup must read back absence before disarming the independent restore"
    )
    assert all(
        command[-1] == "REJECT"
        for command in commands
        if command[:2] == ["iptables", "-A"]
    ), "the bounded ISO006 cut must preserve REJECT semantics"
    assert sum(command[:2] == ["iptables", "-D"] for command in commands) == 2


@pytest.mark.parametrize(
    ("arguments", "success"),
    [
        (["status", "--run-id", "unit-cut"], True),
        (["status", "--run-id", "../unsafe"], False),
        (
            [
                "block",
                "--run-id",
                "unit-cut",
                "--control-plane-cidr",
                "10.0.0.0/24",
                "--restore-seconds",
                "59",
            ],
            False,
        ),
        (
            [
                "block",
                "--run-id",
                "unit-cut",
                "--control-plane-cidr",
                "10.0.0.0/24",
                "--restore-seconds",
                "3601",
            ],
            False,
        ),
    ],
)
def test_network_main_rejects_bad_authorization_shape_without_transport(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    success: bool,
) -> None:
    transport = NetworkTransport()
    monkeypatch.setattr(network, "run", transport.run)
    monkeypatch.setattr(sys, "argv", ["network", *arguments])
    assert network.main() == (0 if success else 1)
    result = json.loads(capsys.readouterr().out)
    if success:
        assert result["residual"] is False
        assert transport.commands == [["iptables", "-S"]]
    else:
        assert result["error"].startswith("ProbeError:"), result
        assert transport.commands == []
