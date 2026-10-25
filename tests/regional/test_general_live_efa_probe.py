"""Contract tests for the on-node EFA driver fixture probe.

Nothing here runs systemd or touches a real device: ``run`` (systemctl /
systemd-run) is a fake, and ``/sys`` is a small tree under ``tmp_path`` the
module's path constants are pointed at. The tests cover the parsing of the EFA
inventory, the safety preconditions that refuse to touch the node unless
exactly one spare function is safe, the deadman arming that fails closed, and
the release that restores the baseline or fails loudly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_efa_probe as probe

BDF_A = "0000:30:00.0"
BDF_B = "0000:31:00.0"


class FakeRun:
    """systemctl / systemd-run without systemd; records the commands."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, command, *, check=True, timeout=120, input_text=None):
        self.calls.append(list(command))
        text = ""
        if command[:2] == ["systemctl", "show"]:
            # A deadman/trap query: report the timer active and loaded.
            text = "LoadState=loaded\nActiveState=active\nSubState=waiting\nMainPID=0\n"

        class Completed:
            returncode = 0
            stdout = text
            stderr = ""

        return Completed()


def _tree(
    tmp_path: Path,
    monkeypatch,
    *,
    functions=((BDF_A, True, "(none)"), (BDF_B, True, "(none)")),
    module: bool = True,
    allow_remediation: bool = True,
) -> FakeRun:
    sysfs = tmp_path / "sys"
    devices = sysfs / "bus" / "pci" / "devices"
    drivers = sysfs / "bus" / "pci" / "drivers" / "efa"
    ib = sysfs / "class" / "infiniband"
    for path in (devices, drivers, ib):
        path.mkdir(parents=True, exist_ok=True)
    for index, (bdf, bound, override) in enumerate(functions):
        device = devices / bdf
        device.mkdir(parents=True, exist_ok=True)
        (device / "vendor").write_text("0x1d0f\n")
        (device / "device").write_text("0xefa1\n")
        (device / "driver_override").write_text(
            "\n" if override in {"(none)", "", None} else override + "\n"
        )
        if bound:
            (device / "driver").symlink_to(drivers)
            (drivers / bdf).symlink_to(device)
        rdma = ib / f"rdmap{index}"
        rdma.mkdir(parents=True, exist_ok=True)
        (rdma / "device").symlink_to(device)
    if module:
        (sysfs / "module" / "efa").mkdir(parents=True, exist_ok=True)
    env_dir = tmp_path / "etc"
    env_dir.mkdir(exist_ok=True)
    collector_env = env_dir / "collector.env"
    collector_env.write_text(
        "GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION="
        + ("true" if allow_remediation else "false")
        + "\nGPU_FAULT_EXPECTED_EFA_DEVICE_COUNT=16\n"
    )
    boot = tmp_path / "boot_id"
    boot.write_text("boot-xyz\n")
    monkeypatch.setattr(probe, "SYSFS", sysfs)
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "efa-driver")
    monkeypatch.setattr(probe, "COLLECTOR_ENV", collector_env)
    monkeypatch.setattr(probe, "NODE_AGENT_ENV", env_dir / "node-agent.env")
    monkeypatch.setattr(probe, "BOOT_ID_FILE", boot)
    fake = FakeRun()
    monkeypatch.setattr(probe, "run", fake)
    return fake


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0000:30:00.0", "0000:30:00.0"),
        ("00000000:30:00.0", "0000:30:00.0"),
        ("0000:30:00", "0000:30:00.0"),
        ("0000:AB:1F.3", "0000:ab:1f.3"),
    ],
)
def test_normalize_bdf(raw: str, expected: str) -> None:
    assert probe.normalize_bdf(raw) == expected


@pytest.mark.parametrize("bad", ["", "lol", "30:00.0", "0000:30:00.8", "../etc"])
def test_normalize_bdf_rejects_unsafe(bad: str) -> None:
    with pytest.raises(probe.ProbeError):
        probe.normalize_bdf(bad)


def test_bounded_and_run_id() -> None:
    assert probe.bounded(300, 60, 5400, "x") == 300
    with pytest.raises(probe.ProbeError):
        probe.bounded(10, 60, 5400, "x")
    with pytest.raises(probe.ProbeError):
        probe.checked_run_id("../nope")
    assert probe.unit_name("gl013-x-a1-dead", "deadman").startswith("gfrs-"), (
        "transient unit names must carry the gfrs- prefix"
    )


# --------------------------------------------------------------------------- #
# Inventory parsing
# --------------------------------------------------------------------------- #
def test_efa_functions_lists_bound_functions(tmp_path: Path, monkeypatch) -> None:
    _tree(tmp_path, monkeypatch)
    functions = probe.efa_functions()
    assert {item["pci_bdf"] for item in functions} == {BDF_A, BDF_B}
    assert all(item["driver"] == "efa" and item["bound"] for item in functions), (
        functions
    )
    assert all(item["rdma_device"] for item in functions), functions


def test_list_functions_emits_json(tmp_path: Path, monkeypatch, capsys) -> None:
    _tree(tmp_path, monkeypatch)
    probe.list_functions(_ns(run_id="gl013-x-a1-dead"))
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["bound_count"] == 2
    assert payload["module_loaded"] is True


# --------------------------------------------------------------------------- #
# Safety preconditions
# --------------------------------------------------------------------------- #
def test_preconditions_pass_for_one_spare_function(tmp_path: Path, monkeypatch) -> None:
    _tree(tmp_path, monkeypatch)
    facts = probe._preconditions(BDF_A)
    assert facts["baseline_bound_count"] == 2


def test_preconditions_refuse_the_last_bound_function(
    tmp_path: Path, monkeypatch
) -> None:
    _tree(
        tmp_path,
        monkeypatch,
        functions=((BDF_A, True, "(none)"), (BDF_B, False, "(none)")),
    )
    with pytest.raises(probe.ProbeError, match="last one"):
        probe._preconditions(BDF_A)


def test_preconditions_refuse_an_existing_override(tmp_path: Path, monkeypatch) -> None:
    _tree(
        tmp_path,
        monkeypatch,
        functions=((BDF_A, True, "vfio-pci"), (BDF_B, True, "(none)")),
    )
    with pytest.raises(probe.ProbeError, match="driver_override"):
        probe._preconditions(BDF_A)


def test_preconditions_refuse_when_module_missing(tmp_path: Path, monkeypatch) -> None:
    _tree(tmp_path, monkeypatch, module=False)
    with pytest.raises(probe.ProbeError, match="module is not loaded"):
        probe._preconditions(BDF_A)


def test_preconditions_refuse_when_remediation_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    _tree(tmp_path, monkeypatch, allow_remediation=False)
    with pytest.raises(probe.ProbeError, match="REMEDIATE_EFA_DRIVER"):
        probe._preconditions(BDF_A)


def test_preconditions_refuse_a_non_efa_function(tmp_path: Path, monkeypatch) -> None:
    _tree(tmp_path, monkeypatch)
    device = probe.device_dir(BDF_A)
    (device / "vendor").write_text("0x10de\n")  # an NVIDIA GPU, not EFA
    with pytest.raises(probe.ProbeError, match="not an EFA"):
        probe._preconditions(BDF_A)


# --------------------------------------------------------------------------- #
# Deadman fails closed; release restores or fails loudly
# --------------------------------------------------------------------------- #
def test_arm_deadman_requires_an_active_timer(tmp_path: Path, monkeypatch) -> None:
    _tree(tmp_path, monkeypatch)
    # A run that reports the timer inactive must fail closed.
    monkeypatch.setattr(
        probe,
        "run",
        lambda *a, **k: type("C", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    with pytest.raises(probe.ProbeError, match="not armed"):
        probe.arm_deadman("gl013-x-a1-dead", BDF_A, 600)


def test_release_is_clean_when_bound(tmp_path: Path, monkeypatch, capsys) -> None:
    _tree(tmp_path, monkeypatch)
    probe.release(_ns(run_id="gl013-x-a1-dead", pci_bdf=BDF_A, automatic=False))
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["clean"] is True
    assert receipt["bound"] is True
    assert receipt["driver_override"] == "(none)"


def test_release_fails_loudly_when_unbound(tmp_path: Path, monkeypatch, capsys) -> None:
    # The function is left unbound; _bind writes to the (fake) bind file, which
    # does not re-attach it, so the end-state check must refuse.
    _tree(
        tmp_path,
        monkeypatch,
        functions=((BDF_A, False, "(none)"), (BDF_B, True, "(none)")),
    )
    with pytest.raises(probe.ProbeError, match="did not restore"):
        probe.release(_ns(run_id="gl013-x-a1-dead", pci_bdf=BDF_A, automatic=False))
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["clean"] is False


def _ns(**values: Any):
    return type("NS", (), values)()
