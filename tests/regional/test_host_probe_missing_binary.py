"""A host probe names the node binary or interpreter it is missing.

Live (mixed site, ISO-006 attempt 4): the block was aimed at the executor Pods'
CPU nodes too; those run no Node Agent, so ``/opt/gpu-fault/current/venv/bin/
python`` -- the interpreter the probe wrapper ``chroot``s into -- does not
exist there, and the probe died "host probe returned no JSON (exit 127)" with
nothing naming the gap. The wrapper now picks the interpreter inside the
chroot (the Agent venv, else the node's ``/usr/bin/python3``) and refuses in
the probe's own JSON shape when neither exists; the network probe checks the
binaries it needs before it arms a timer or writes a rule, and the fixture
shows the missing basename -- a fixed token, never a path -- in its refusal.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.e2e.regional import host_probe_fixture as host
from scripts.e2e.regional.probes import cluster_network_probe as network
from tests.regional._host_probe_support import ProbeApi, host_probe


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> ProbeApi:
    value = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", value.run)
    return value


def _exec_shell(api: ProbeApi) -> str:
    for arguments, _kwargs in api.calls:
        if arguments and arguments[0] == "exec" and "host-probe" in arguments:
            return str(arguments[arguments.index("-ceu") + 1])
    raise AssertionError("no exec call recorded")


def test_the_wrapper_chooses_the_interpreter_inside_the_chroot(
    tmp_path: Path, api: ProbeApi
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    api.probe_stdout = json.dumps({"blocked": True})

    probe.execute("status", "--run-id", "unit")

    shell = _exec_shell(api)
    assert "exec chroot /host /bin/sh -c" in shell, "the paths are the node's own"
    assert host.HOST_PROBE_AGENT_PYTHON in shell
    assert host.HOST_PROBE_SYSTEM_PYTHON in shell
    assert shell.index(host.HOST_PROBE_AGENT_PYTHON) < shell.index(
        host.HOST_PROBE_SYSTEM_PYTHON
    ), "the Agent venv wins when the node runs the Agent"
    assert '\\"missing_binary\\":\\"python3\\"' in shell


def test_a_missing_interpreter_or_binary_is_named_in_the_refusal(
    tmp_path: Path, api: ProbeApi
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    api.probe_returncode = 1
    api.probe_stdout = json.dumps(
        {
            "error": "host probe interpreter missing",
            "error_kind": "missing_binary",
            "missing_binary": "python3",
        }
    )

    with pytest.raises(host.HostProbeError) as refused:
        probe.execute("block", "--run-id", "unit")
    assert "missing binary on the node: python3" in str(refused.value)
    assert "output withheld" in str(refused.value)

    api.probe_stdout = json.dumps(
        {
            "error": "MissingBinary: required binary is missing on the node: iptables",
            "error_kind": "missing_binary",
            "missing_binary": "../../etc/shadow",
        }
    )
    with pytest.raises(host.HostProbeError) as unsafe:
        probe.execute("block", "--run-id", "unit")
    assert "missing binary" not in str(unsafe.value), (
        "only a basename-shaped token is ever shown"
    )
    assert "shadow" not in str(unsafe.value)


def test_the_network_probe_refuses_by_name_before_arming_anything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "systemd-run":
            raise network.MissingBinary("systemd-run")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(network, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "network",
            "block",
            "--run-id",
            "unit-cut",
            "--control-plane-cidr",
            "10.0.0.0/24",
        ],
    )

    assert network.main() == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == (
        "MissingBinary: required binary is missing on the node: systemd-run"
    )
    assert payload["error_kind"] == "missing_binary"
    assert payload["missing_binary"] == "systemd-run"
    assert commands == [["iptables", "-S"], ["systemd-run", "--version"]], (
        "read-only probes only: nothing was armed or written"
    )


def test_a_binary_that_vanishes_at_run_time_is_named_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", command[0])

    monkeypatch.setattr(network.subprocess, "run", missing)

    with pytest.raises(network.MissingBinary) as refused:
        network.run(["/usr/sbin/iptables", "-S"])
    assert refused.value.binary == "iptables"
    assert network.REQUIRED_BINARIES == ("iptables", "systemd-run", "systemctl")
