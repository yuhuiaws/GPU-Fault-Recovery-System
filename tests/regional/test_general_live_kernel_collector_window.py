"""The kernel collector's command window of the host source replay probe.

Nothing here runs systemd or a node: the ``FakeHost`` of the source-probe tests
models both collector units, and these tests prove that
``--unit gpu-fault-kernel-collector.service`` (the runner's
``KERNEL_COLLECTOR_SOURCE_REPLAY`` approval) opens a command window there and
nothing else -- the ``nvidia-smi`` driver-version query only, the configured
``GPU_FAULT_DRIVER_BRANCH`` unset for the window, every file/tree/override mode
refused, the kernel collector the only unit restarted, the host collector never
touched -- and that close, the untracked-drop-in path and ``journal-read`` keep
to the same unit.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests.regional.test_general_live_source_probe import (
    KERNEL_PID,
    KERNEL_UNIT,
    MAIN_PID,
    RUN,
    UNIT,
    FakeHost,
)
from tools import host_command_replay as wrapper

VERSION_RULE = {
    "query": "driver_version",
    "action": "exit",
    "exit_code": 1,
    "stderr": "withheld\n",
}


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    fake = FakeHost(tmp_path)
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "gfrs-live")
    monkeypatch.setattr(probe, "DROPIN_ROOT", tmp_path / "run" / "systemd" / "system")
    monkeypatch.setattr(probe, "run", fake.run)
    monkeypatch.setattr(probe, "process_environment", fake.environment_of)
    real = tmp_path / "usr" / "bin" / "nvidia-smi"
    real.parent.mkdir(parents=True)
    real.write_text("#!/bin/sh\nexit 0\n")
    real.chmod(0o755)
    smartctl = tmp_path / "usr" / "bin" / "smartctl"
    smartctl.write_text("#!/bin/sh\nexit 0\n")
    smartctl.chmod(0o755)
    monkeypatch.setattr(probe, "STANDARD_PATH", f"{real.parent}:/bin")
    return fake


def _main(*argv: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = probe.main(list(argv))
    out = capsys.readouterr().out.strip().splitlines()
    return code, json.loads(out[-1])


def _install(name: str, content: bytes, capsys: pytest.CaptureFixture[str]) -> None:
    code, receipt = _main(
        "install-source",
        "--run-id",
        RUN,
        "--name",
        name,
        "--sha256",
        hashlib.sha256(content).hexdigest(),
        "--b64",
        base64.b64encode(content).decode("ascii"),
        capsys=capsys,
    )
    assert code == 0, receipt


def _plan(tool: str, *rules: dict[str, Any]) -> bytes:
    return json.dumps(
        {"tool": tool, "not_after": time.time() + 600, "rules": list(rules)}
    ).encode()


def _wrapper_and_plan(
    capsys: pytest.CaptureFixture[str], tool: str, *rules: dict[str, Any]
) -> None:
    _install(probe.WRAPPER_NAME, Path(wrapper.__file__).read_bytes(), capsys)
    _install(f"plan-{tool}.json", _plan(tool, *rules), capsys)


def test_kernel_collector_window_wraps_the_driver_version_query_only(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    _wrapper_and_plan(capsys, "nvidia-smi", VERSION_RULE)
    code, installed = _main(
        "install-wrapper",
        "--run-id",
        RUN,
        "--tool",
        "nvidia-smi",
        "--unit",
        KERNEL_UNIT,
        capsys=capsys,
    )
    assert code == 0 and installed["unit"] == KERNEL_UNIT, installed
    assert installed["rules"] == 1
    code, opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "command",
        "--unit",
        KERNEL_UNIT,
        "--wrapper",
        "nvidia-smi",
        capsys=capsys,
    )
    assert code == 0, opened
    assert opened["unit"] == KERNEL_UNIT and opened["unset_env"] == [
        "GPU_FAULT_DRIVER_BRANCH"
    ]
    dropin = probe.dropin_path(RUN, KERNEL_UNIT)
    assert Path(opened["dropin"]) == dropin and dropin.parent.name == f"{KERNEL_UNIT}.d"
    assert not probe.dropin_path(RUN, UNIT).exists(), "no host collector drop-in"
    assert dropin.read_text(encoding="utf-8").splitlines() == [
        "[Service]",
        f"Environment=PATH={probe.bin_dir(RUN)}:{probe.STANDARD_PATH}",
        f"ReadWritePaths={probe.audit_dir(RUN)}",
        "UnsetEnvironment=GPU_FAULT_DRIVER_BRANCH",
    ]
    assert host.restarted == [KERNEL_UNIT], "only the kernel collector restarted"
    assert host.dropin_present_at_arm is False, "the deadman is armed first"
    kinds = {item["kind"]: item for item in opened["verification"]["checks"]}
    assert kinds["wrapper"]["ok"] and kinds["wrapper"]["unit_path_first"]
    assert kinds["unset-env"] == {
        "kind": "unset-env",
        "dest": "GPU_FAULT_DRIVER_BRANCH",
        "unit_has": False,
        "ok": True,
    }
    assert opened["verification"]["main_pid"] == KERNEL_PID
    assert "GPU_FAULT_DRIVER_BRANCH" not in host.environments[KERNEL_UNIT], (
        "the configured branch is unset inside the unit for the window"
    )
    assert host.environments[KERNEL_UNIT]["PATH"].split(":")[0] == str(
        probe.bin_dir(RUN)
    )
    assert host.environments[UNIT] == {}, (
        "the host collector's environment is untouched"
    )

    code, status = _main("status", "--run-id", RUN, capsys=capsys)
    assert code == 0 and status["unit_name"] == KERNEL_UNIT
    assert status["unit"]["MainPID"] == KERNEL_PID and status["wrappers"] == [
        "nvidia-smi"
    ]
    code, verified = _main("verify", "--run-id", RUN, capsys=capsys)
    assert code == 0 and verified["ok"] is True and verified["main_pid"] == KERNEL_PID

    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["clean"] is True and closed["unit"] == KERNEL_UNIT
    assert host.restarted == [KERNEL_UNIT, KERNEL_UNIT]
    assert host.environments[KERNEL_UNIT] == {"GPU_FAULT_DRIVER_BRANCH": "580"}, (
        "the configured branch is back once the drop-in is gone"
    )
    assert not dropin.exists() and not host.timers
    assert MAIN_PID not in {call[2] for call in host.calls if call[0] == "nsenter"}, (
        "verification never entered the host collector's namespace"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--mode", "bind", "--unit", KERNEL_UNIT, "--bind", "meminfo:/proc/meminfo"],
        [
            "--mode",
            "command",
            "--unit",
            KERNEL_UNIT,
            "--wrapper",
            "nvidia-smi",
            "--env",
            "GPU_FAULT_EXPECTED_GPU_COUNT=7",
        ],
        [
            "--mode",
            "command",
            "--unit",
            KERNEL_UNIT,
            "--wrapper",
            "nvidia-smi",
            "--inaccessible",
            "/proc/meminfo",
        ],
        ["--mode", "pci-root", "--unit", KERNEL_UNIT],
        ["--mode", "command", "--unit", KERNEL_UNIT, "--wrapper", "smartctl"],
        [
            "--mode",
            "command",
            "--unit",
            probe.NODE_AGENT_UNIT,
            "--wrapper",
            "nvidia-smi",
        ],
    ],
)
def test_kernel_collector_refuses_everything_but_its_command_window(
    host: FakeHost, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    _wrapper_and_plan(capsys, "nvidia-smi", VERSION_RULE)
    _install("meminfo", b"MemTotal: 1 kB\n", capsys)
    code, installed = _main(
        "install-wrapper",
        "--run-id",
        RUN,
        "--tool",
        "nvidia-smi",
        "--unit",
        KERNEL_UNIT,
        capsys=capsys,
    )
    assert code == 0, installed
    code, error = _main("open-window", "--run-id", RUN, *argv, capsys=capsys)
    assert code == 1 and error["error"] == "ProbeError", error
    assert not any(
        probe.dropin_path(RUN, unit).exists() for unit in probe.ALLOWED_UNITS
    ), "a refused window wrote no drop-in for either unit"
    assert host.restarted == [] and not host.timers, "a refused window touched nothing"


def test_kernel_collector_wrapper_takes_driver_version_plans_only(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    _wrapper_and_plan(
        capsys,
        "nvidia-smi",
        VERSION_RULE,
        {
            "query": "utilization",
            "action": "drop_row",
            "uuid": "GPU-11111111-2222-3333-4444-555555555555",
        },
    )
    code, error = _main(
        "install-wrapper",
        "--run-id",
        RUN,
        "--tool",
        "nvidia-smi",
        "--unit",
        KERNEL_UNIT,
        capsys=capsys,
    )
    assert code == 1 and error["error"] == "ProbeError", (
        "a utilization rule may not ride into the kernel collector"
    )
    _install(
        "plan-smartctl.json",
        _plan("smartctl", {"query": "scan", "action": "exit", "exit_code": 2}),
        capsys,
    )
    code, error = _main(
        "install-wrapper",
        "--run-id",
        RUN,
        "--tool",
        "smartctl",
        "--unit",
        KERNEL_UNIT,
        capsys=capsys,
    )
    assert code == 1 and error["error"] == "ProbeError", (
        "smartctl is a host collector source"
    )
    # A wrapper installed for the host collector cannot open on the kernel one.
    _install("plan-nvidia-smi.json", _plan("nvidia-smi", VERSION_RULE), capsys)
    code, installed = _main(
        "install-wrapper", "--run-id", RUN, "--tool", "nvidia-smi", capsys=capsys
    )
    assert code == 0 and installed["unit"] == UNIT
    code, error = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "command",
        "--unit",
        KERNEL_UNIT,
        "--wrapper",
        "nvidia-smi",
        capsys=capsys,
    )
    assert code == 1 and error["error"] == "ProbeError", error
    assert host.restarted == [] and not host.timers
    assert not any(
        probe.dropin_path(RUN, unit).exists() for unit in probe.ALLOWED_UNITS
    ), "a host-unit wrapper never opens a kernel-unit drop-in"


def test_journal_read_is_bounded_to_an_allow_listed_unit(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    host.journal[KERNEL_UNIT] = (
        "nvidia-smi software version discovery unavailable; using configured "
        "driver/CUDA versions\n" + "x" * 1000 + "\n"
    )
    code, journal = _main(
        "journal-read",
        "--run-id",
        RUN,
        "--unit",
        KERNEL_UNIT,
        "--since",
        "2026-10-25T08:00:00+00:00",
        "--tail",
        "5",
        capsys=capsys,
    )
    assert code == 0, journal
    assert (
        journal["unit"] == KERNEL_UNIT and journal["since"] == "2026-10-25 08:00:00 UTC"
    )
    assert journal["lines"][0].startswith("nvidia-smi software version discovery"), (
        journal["lines"]
    )
    assert len(journal["lines"][1]) == probe.MAX_JOURNAL_LINE_CHARS, "lines are bounded"
    assert journal["truncated"] is False and journal["window_open"] is False
    call = next(item for item in host.calls if item[0] == "journalctl")
    assert call == [
        "journalctl",
        "-u",
        KERNEL_UNIT,
        "--since",
        "2026-10-25 08:00:00 UTC",
        "-n",
        "5",
        "--no-pager",
        "-q",
        "-o",
        "cat",
    ]
    for unit, since in (
        (probe.NODE_AGENT_UNIT, "2026-10-25T08:00:00+00:00"),
        ("gpu-fault-metrics-collector.service", "2026-10-25T08:00:00+00:00"),
        (KERNEL_UNIT, "yesterday"),
        (KERNEL_UNIT, "2026-10-25T08:00:00"),
    ):
        code, error = _main(
            "journal-read",
            "--run-id",
            RUN,
            "--unit",
            unit,
            "--since",
            since,
            capsys=capsys,
        )
        assert code == 1 and error["error"] == "ProbeError", (unit, since)


def test_close_window_restores_an_untracked_kernel_dropin(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    dropin = probe.dropin_path(RUN, KERNEL_UNIT)
    dropin.parent.mkdir(parents=True)
    dropin.write_text("[Service]\nUnsetEnvironment=GPU_FAULT_DRIVER_BRANCH\n")
    host.apply_dropin(KERNEL_UNIT)
    assert "GPU_FAULT_DRIVER_BRANCH" not in host.environments[KERNEL_UNIT]
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["untracked_dropin"] is True, closed
    assert closed["unit"] == KERNEL_UNIT and closed["clean"] is True
    assert not dropin.exists() and host.restarted == [KERNEL_UNIT]
    assert host.environments[KERNEL_UNIT] == {"GPU_FAULT_DRIVER_BRANCH": "580"}
    code, snapshot = _main("snapshot", "--run-id", RUN, capsys=capsys)
    assert (
        code == 0 and snapshot["dropin"] is False and snapshot["window_open"] is False
    )
