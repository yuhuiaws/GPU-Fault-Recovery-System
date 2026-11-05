"""The on-node EFA driver fixture probe on the node shapes its contract tests skip.

``test_general_live_efa_probe.py`` proves the two variants on a clean two-
function tree; here the same probe (its CLI ``main`` and public steps, over a
``/sys`` under ``tmp_path`` and a scripted ``systemctl``) meets a PCI tree with
foreign entries and no RDMA class, a node env with comments and other keys,
``systemctl`` output with stray lines and this run's units still loaded, a BDF
that is no device or is bound elsewhere, a trap unit that never started, a
product that never binds back, the deadman releasing from inside its own unit,
a state file of another run -- and the real ``run`` wrapper and the script's
own ``__main__``.
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_efa_probe as probe

BDF_A = "0000:30:00.0"
BDF_B = "0000:31:00.0"
BDF_C = "0000:32:00.0"
RUN = "gl013-x-a1-dead"
LOADED = "LoadState=loaded\nActiveState=active\nSubState=waiting\nMainPID=0\n"


class Systemd:
    """``systemctl`` / ``systemd-run`` as the probe calls them, scripted per unit."""

    def __init__(self, shows: dict[str, str] | None = None, listing: str = "") -> None:
        self.calls: list[list[str]] = []
        self.shows = shows or {}
        self.listing = listing

    def __call__(
        self, command: list[str], *, check: bool = True, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(command))
        stdout = ""
        if command[:2] == ["systemctl", "show"]:
            stdout = self.shows.get(command[2], LOADED)
        elif command[:2] == ["systemctl", "list-units"]:
            stdout = self.listing
        return subprocess.CompletedProcess(command, 0, stdout, "")


def _sysfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    functions: tuple[tuple[str, str | None], ...] = ((BDF_A, "efa"), (BDF_B, "efa")),
    infiniband: bool = True,
    systemd: Systemd | None = None,
) -> Systemd:
    """``functions``: (BDF, bound driver or None); every one an EFA function."""

    sysfs = tmp_path / "sys"
    devices = sysfs / "bus" / "pci" / "devices"
    devices.mkdir(parents=True)
    for index, (bdf, driver) in enumerate(functions):
        device = devices / bdf
        device.mkdir()
        (device / "vendor").write_text("0x1d0f\n")
        (device / "device").write_text("0xefa1\n")
        (device / "driver_override").write_text("(null)\n")
        if driver is not None:
            root = sysfs / "bus" / "pci" / "drivers" / driver
            root.mkdir(parents=True, exist_ok=True)
            (device / "driver").symlink_to(root)
            (root / bdf).symlink_to(device)
        if infiniband:
            rdma = sysfs / "class" / "infiniband" / f"rdmap{index}"
            rdma.mkdir(parents=True)
            (rdma / "device").symlink_to(device)
    (sysfs / "module" / "efa").mkdir(parents=True)
    etc = tmp_path / "etc"
    etc.mkdir()
    (etc / "collector.env").write_text(
        "GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION=true\n"
    )
    (tmp_path / "boot_id").write_text("boot-xyz\n")
    monkeypatch.setattr(probe, "SYSFS", sysfs)
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "efa-driver")
    monkeypatch.setattr(probe, "COLLECTOR_ENV", etc / "collector.env")
    monkeypatch.setattr(probe, "NODE_AGENT_ENV", etc / "node-agent.env")
    monkeypatch.setattr(probe, "BOOT_ID_FILE", tmp_path / "boot_id")
    fake = systemd or Systemd()
    monkeypatch.setattr(probe, "run", fake)
    return fake


def _main(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = probe.main(list(argv))
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _args(**values: Any) -> Any:
    return type("Arguments", (), values)()


# --------------------------------------------------------------------------- #
# The real process wrappers
# --------------------------------------------------------------------------- #
def test_run_names_a_failed_command_by_its_code_and_stderr() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('no such unit'); sys.exit(4)",
    ]
    with pytest.raises(
        probe.ProbeError, match=r"command failed \(4\): .*: no such unit"
    ):
        probe.run(command)
    completed = probe.run(command, check=False)
    assert (completed.returncode, completed.stderr) == (4, "no such unit")
    started = probe.monotonic_clock()
    probe.pause(0.01)
    assert probe.monotonic_clock() - started >= 0.01, "pause sleeps on the real clock"


def test_the_script_answers_help_as_a_standalone_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [probe.__file__, "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(probe.__file__, run_name="__main__")
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    assert "GF-GENERAL-LIVE-013 EFA driver fixture" in out, out
    assert "trap-unbind" in out and "release" in out, out


# --------------------------------------------------------------------------- #
# Read-only facts
# --------------------------------------------------------------------------- #
def test_the_inventory_skips_foreign_entries_and_needs_no_rdma_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sysfs(tmp_path, monkeypatch, infiniband=False)
    devices = probe.pci_devices_root()
    gpu = devices / "0000:53:00.0"
    gpu.mkdir()
    (gpu / "vendor").write_text("0x10de\n")
    (gpu / "device").write_text("0x2335\n")
    (devices / "not-a-bdf").mkdir()

    assert probe.infiniband_map() == {}, "no /sys/class/infiniband: no RDMA names"
    functions = probe.efa_functions()
    assert [item["pci_bdf"] for item in functions] == [BDF_A, BDF_B]
    assert {item["rdma_device"] for item in functions} == {None}
    for path in sorted(devices.iterdir(), reverse=True):
        if path.is_dir() and not path.is_symlink():
            for child in path.iterdir():
                child.unlink()
            path.rmdir()
    devices.rmdir()
    with pytest.raises(probe.ProbeError, match="the PCI device tree is not readable"):
        probe.efa_functions()


def test_node_env_keeps_only_the_products_keys_unquoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sysfs(tmp_path, monkeypatch)
    probe.COLLECTOR_ENV.write_text(
        "# written by the installer\n"
        "GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION='true'\n"
        "a line without an assignment\n"
        "GPU_FAULT_UNRELATED=1\n"
    )
    probe.NODE_AGENT_ENV.write_text('GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT="16"\n')
    assert probe.node_env() == {
        "GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION": "true",
        "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT": "16",
    }


def test_the_summary_lists_this_runs_units_and_ignores_stray_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deadman = probe.unit_name(RUN, "deadman")
    listing = (
        f"{deadman}.timer loaded active waiting deadman\n"
        "\n"
        "unrelated.service loaded active running other\n"
        f"{deadman}.service loaded inactive dead deadman\n"
    )
    shows = {f"{deadman}.timer": "LoadState=loaded\nnoise without a value\n"}
    _sysfs(tmp_path, monkeypatch, systemd=Systemd(shows, listing))
    receipt = probe.summary(RUN, BDF_A)
    assert receipt["run_units"] == [f"{deadman}.service", f"{deadman}.timer"]
    assert receipt["deadman"]["timer"] == {"LoadState": "loaded", "returncode": "0"}
    assert (receipt["bound"], receipt["driver"]) == (True, "efa")
    assert receipt["rdma_device"] == "rdmap0" and receipt["bound_count"] == 2


# --------------------------------------------------------------------------- #
# Refusals before anything is touched
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("bdf", "reason", "message"),
    [
        (BDF_C, "not_a_pci_device", f"{BDF_C} is not a PCI device on this node"),
        (BDF_B, "not_bound_to_efa", f"{BDF_B} is not bound to efa at baseline"),
    ],
)
def test_hold_refuses_a_bdf_that_is_no_device_or_bound_elsewhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bdf: str,
    reason: str,
    message: str,
) -> None:
    fake = _sysfs(
        tmp_path, monkeypatch, functions=((BDF_A, "efa"), (BDF_B, "vfio-pci"))
    )
    code, error = _main(capsys, "hold-unbind", "--run-id", RUN, "--pci-bdf", bdf)
    assert code == 1, error
    assert (error["error_reason"], error["error_message"]) == (reason, message)
    assert fake.calls == [], "refused before the deadman is armed"
    assert not (probe.driver_root() / "unbind").exists(), "nothing was unbound"


def test_trap_refuses_when_its_unit_was_never_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trap = probe.unit_name(RUN, "trap")
    shows = {f"{trap}.service": "LoadState=not-found\nActiveState=inactive\n"}
    _sysfs(tmp_path, monkeypatch, systemd=Systemd(shows))
    code, error = _main(capsys, "trap-unbind", "--run-id", RUN, "--pci-bdf", BDF_A)
    assert code == 1, error
    assert error["error_message"] == f"trap {trap} was not started"
    assert error["error_site"].startswith("_arm_trap:"), error["error_site"]
    assert not (probe.driver_root() / "unbind").exists(), "no unbind without the trap"


def test_an_error_inside_a_helper_is_sited_at_the_calling_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _sysfs(tmp_path, monkeypatch)
    code, error = _main(capsys, "summary", "--run-id", RUN, "--pci-bdf", "0000:zz:00.0")
    assert code == 1 and error["error_message"] == "unsafe PCI BDF", error
    assert error["error_site"].startswith("show_summary:"), (
        "normalize_bdf is a helper frame: the site is the step that called it"
    )


# --------------------------------------------------------------------------- #
# The trap's wait for the product's bind
# --------------------------------------------------------------------------- #
def test_the_trap_gives_up_when_the_product_never_binds_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sysfs(tmp_path, monkeypatch)
    clock = {"now": 0.0}

    def tick() -> float:
        clock["now"] += 10.0
        return clock["now"]

    pauses: list[float] = []
    monkeypatch.setattr(probe, "monotonic_clock", tick)
    monkeypatch.setattr(probe, "pause", pauses.append)
    monkeypatch.setattr(probe, "is_bound", lambda bdf: False)
    probe.trap_run(
        _args(run_id=RUN, pci_bdf=BDF_A, grace_seconds=5, trap_window_seconds=40)
    )
    fired = json.loads((probe.state_dir(RUN) / "trap-fired").read_text())
    assert (fired["fired"], fired["reason"]) == (False, "bind not observed")
    assert pauses == [1, 1], "polled until the window closed, no grace, no unbind"
    assert not (probe.driver_root() / "unbind").exists(), "the trap never fired"


# --------------------------------------------------------------------------- #
# Release: manual, from inside the deadman, and a foreign state
# --------------------------------------------------------------------------- #
def test_a_clean_manual_release_restores_the_hold_and_forgets_its_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _sysfs(tmp_path, monkeypatch)
    code, held = _main(capsys, "hold-unbind", "--run-id", RUN, "--pci-bdf", BDF_A)
    assert code == 0 and held["driver_override"] == probe.HOLD_OVERRIDE, held
    state = probe.state_dir(RUN) / "state.json"
    assert probe.load_state(RUN)["mode"] == "hold"
    code, released = _main(capsys, "release", "--run-id", RUN, "--pci-bdf", BDF_A)
    assert code == 0, released
    assert released["mode"] == "hold" and released["clean"] is True, released
    assert released["override_before"] == probe.HOLD_OVERRIDE
    assert (probe.device_dir(BDF_A) / "driver_override").read_text() == "\n"
    assert not state.exists(), "a clean manual release forgets the run"


def test_the_deadman_release_spares_its_own_unit_and_retries_when_unclean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deadman = probe.unit_name(RUN, "deadman")
    shows = {f"{deadman}.service": f"LoadState=loaded\nMainPID={os.getpid()}\n"}
    fake = _sysfs(tmp_path, monkeypatch, systemd=Systemd(shows))
    probe.save_state(RUN, {"run_id": RUN, "mode": "trap", "pci_bdf": BDF_A})
    code, receipt = _main(
        capsys, "release", "--run-id", RUN, "--pci-bdf", BDF_A, "--automatic"
    )
    assert code == 0 and receipt["clean"] and receipt["automatic"], receipt
    assert ["systemctl", "stop", f"{deadman}.service"] not in fake.calls, (
        "the deadman never stops the unit it runs in"
    )
    assert [
        "systemctl",
        "stop",
        f"{probe.unit_name(RUN, 'trap')}.service",
    ] in fake.calls
    assert (probe.state_dir(RUN) / "state.json").is_file(), "the deadman keeps state"
    (probe.device_dir(BDF_A) / "driver").unlink()
    (probe.driver_root() / BDF_A).unlink()
    code, error = _main(
        capsys, "release", "--run-id", RUN, "--pci-bdf", BDF_A, "--automatic"
    )
    assert code == 1 and error["error_reason"] == "not_restored", error
    assert error["error_message"] == f"deadman could not restore {BDF_A}"


def test_a_state_file_of_another_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _sysfs(tmp_path, monkeypatch)
    probe.save_state(RUN, {"run_id": "gl013-other-a1-beef", "mode": "hold"})
    code, error = _main(capsys, "release", "--run-id", RUN, "--pci-bdf", BDF_A)
    assert code == 1, error
    assert error["error_message"] == "EFA driver fixture state identity mismatch"


def test_an_override_attribute_that_cannot_be_read_is_its_own_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _sysfs(tmp_path, monkeypatch)
    attribute = probe.device_dir(BDF_A) / "driver_override"
    attribute.unlink()
    attribute.mkdir()  # exists, but every read fails (EISDIR here, EACCES live)
    code, error = _main(capsys, "summary", "--run-id", RUN, "--pci-bdf", BDF_A)
    assert code == 1 and error["error_reason"] == "override_unreadable", error
    assert error["error_message"].startswith(
        f"{BDF_A} driver_override is unreadable:"
    ), error["error_message"]
