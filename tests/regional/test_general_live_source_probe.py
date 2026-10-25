"""Contract tests for the host-collector source replay probe (the shared fixture).

Nothing here runs systemd, nsenter or a node. ``run`` (systemctl / systemd-run
/ nsenter) is replaced by a fake that keeps a tiny model of the host: the one
unit's state, the deadman timer, and -- parsed from the drop-in the probe wrote
-- the unit's private view (binds, trees, inaccessible paths, environment). The
host's own view is this test host. That is enough to prove, for every mode,
that the replacement is installed from the fixture directory only, is visible
in the unit's namespace and not in PID 1's, that the deadman is armed before
the drop-in exists, and that cleanup and the deadman's own command remove
everything that was created.
"""

from __future__ import annotations

import base64
import filecmp
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.collectors.host.observation import efa_inventory_counts
from scripts.e2e.regional.probes import general_live_source_probe as probe
from tools import host_command_replay as wrapper

RUN = "gl007-health-failed-a1-deadbeef"
MAIN_PID = "4242"
UNIT = probe.HOST_UNIT


class FakeHost:
    """systemctl / systemd-run / nsenter as the probe sees them, without systemd."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.calls: list[list[str]] = []
        self.timers: set[str] = set()
        self.restarts = 0
        self.dropin_present_at_arm: bool | None = None
        self.unit_active = True
        self.binds: dict[str, str] = {}
        self.inaccessible: set[str] = set()
        self.environment: dict[str, str] = {}

    # -- the drop-in is what systemd would apply on restart ------------------
    def apply_dropin(self) -> None:
        self.binds, self.inaccessible, self.environment = {}, set(), {}
        path = probe.dropin_path(RUN)
        if not path.is_file():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key == "BindReadOnlyPaths":
                source, _, dest = value.partition(":")
                self.binds[dest] = source
            elif key == "InaccessiblePaths":
                self.inaccessible.add(value.lstrip("-"))
            elif key == "Environment":
                name, _, setting = value.partition("=")
                self.environment[name] = setting

    def unit_view(self, path: str) -> str | None:
        """The path ``path`` resolves to inside the unit, or None if inaccessible."""

        for hidden in self.inaccessible:
            if path == hidden or path.startswith(hidden + "/"):
                return None
        for dest, source in self.binds.items():
            if path == dest:
                return source
            if path.startswith(dest + "/"):
                return source + path[len(dest) :]
        return path

    def nsenter(self, pid: str, command: list[str]) -> subprocess.CompletedProcess[str]:
        target = command[-1]
        real = self.unit_view(target) if pid == MAIN_PID else target
        if real is None:
            return subprocess.CompletedProcess(command, 1, "", "Permission denied")
        try:
            if command[0] == "/usr/bin/cat":
                return subprocess.CompletedProcess(
                    command, 0, Path(real).read_text(), ""
                )
            if command[0] == "/usr/bin/env":
                names = "".join(f"{name}\n" for name in sorted(os.listdir(real)))
                return subprocess.CompletedProcess(command, 0, names, "")
            if command[0] == "/usr/bin/stat":
                return subprocess.CompletedProcess(
                    command, 0, f"{os.stat(real).st_dev}\n", ""
                )
            if command[0] == "/usr/bin/test":
                ok = os.access(real, os.X_OK)
                return subprocess.CompletedProcess(command, 0 if ok else 1, "", "")
        except OSError as error:
            return subprocess.CompletedProcess(command, 1, "", str(error))
        raise AssertionError(f"unexpected nsenter command {command}")

    # -- systemctl / systemd-run ---------------------------------------------
    def run(self, command: list[str], *, check: bool = True, timeout: int = 120) -> Any:
        self.calls.append(list(command))
        if command[0] == "nsenter":
            return self.nsenter(command[2], command[5:])
        if command[0] == "systemd-run":
            unit = next(item for item in command if item.startswith("--unit=")).split(
                "=", 1
            )[1]
            self.dropin_present_at_arm = probe.dropin_path(RUN).exists()
            self.timers.add(f"{unit}.timer")
            return subprocess.CompletedProcess(command, 0, "", "")
        assert command[0] == "systemctl", command
        verb = command[1]
        if verb == "show":
            unit = command[2]
            if unit == UNIT:
                state = "active" if self.unit_active else "failed"
                text = f"LoadState=loaded\nActiveState={state}\nSubState=running\nMainPID={MAIN_PID}\nNRestarts=0\n"
            elif unit.endswith(".timer"):
                state = "active" if unit in self.timers else "inactive"
                text = f"LoadState=loaded\nActiveState={state}\nSubState=waiting\nMainPID=0\nNRestarts=0\n"
            else:
                text = "LoadState=not-found\nActiveState=inactive\nSubState=dead\nMainPID=0\nNRestarts=0\n"
            return subprocess.CompletedProcess(command, 0, text, "")
        if verb == "stop":
            for unit in command[2:]:
                self.timers.discard(unit)
        if verb == "restart":
            assert command[2] == UNIT, (
                f"only the host collector may be restarted: {command}"
            )
            self.restarts += 1
            self.apply_dropin()
        return subprocess.CompletedProcess(command, 0, "", "")

    def environment_of(self, pid: str, keys: tuple[str, ...]) -> dict[str, str]:
        if pid != MAIN_PID:
            return {}
        return {name: value for name, value in self.environment.items() if name in keys}


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    fake = FakeHost(tmp_path)
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "gfrs-live")
    monkeypatch.setattr(probe, "DROPIN_ROOT", tmp_path / "run" / "systemd" / "system")
    monkeypatch.setattr(probe, "run", fake.run)
    monkeypatch.setattr(probe, "process_environment", fake.environment_of)
    return fake


def _main(*argv: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = probe.main(list(argv))
    out = capsys.readouterr().out.strip().splitlines()
    return code, json.loads(out[-1])


def _install(
    name: str, content: bytes, capsys: pytest.CaptureFixture[str]
) -> dict[str, Any]:
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
    return receipt


def _created_outside(tmp_path: Path) -> list[str]:
    """Every file under the test host that is not the probe's own state or drop-in."""

    allowed = (tmp_path / "run" / "gfrs-live", tmp_path / "run" / "systemd")
    return sorted(
        str(path.relative_to(tmp_path))
        for path in tmp_path.rglob("*")
        if path.is_file()
        and not any(str(path).startswith(str(root)) for root in allowed)
        and not str(path.relative_to(tmp_path)).startswith("src")
    )


# --------------------------------------------------------------------------- #
# allow-lists
# --------------------------------------------------------------------------- #
def test_bind_allow_list_names_the_spec_files_only() -> None:
    for dest in (
        "/proc/stat",
        "/proc/meminfo",
        "/proc/driver/nvidia/version",
        "/sys/class/net/gfrs1a2b/statistics/rx_errors",
        "/sys/class/infiniband/rdmap79s0/ports/1/hw_counters/rx_drops",
        "/sys/class/infiniband/rdmap79s0/ports/1/counters/port_rcv_errors",
        "/sys/class/infiniband/rdmap79s0/ports/1/state",
        "/sys/class/infiniband/rdmap79s0/ports/1/phys_state",
    ):
        assert probe.checked_bind_dest(dest) == dest, dest
    for dest in (
        "/etc/passwd",
        "/sys/class/net/eth0/statistics/rx_errors",
        "/sys/class/infiniband/rdmap79s0/ports/1/hw_counters/../../state",
        "/proc/driver/nvidia/gpus",
        "/proc/1/environ",
        "/opt/gpu-fault/current/venv/bin/gpu-fault-node-agent",
    ):
        with pytest.raises(probe.ProbeError, match="not allow-listed"):
            probe.checked_bind_dest(dest)


def test_tree_inaccessible_env_and_unit_allow_lists() -> None:
    assert probe.checked_tree_dest("/sys/class/infiniband") == "/sys/class/infiniband"
    assert probe.checked_tree_dest("/proc/driver/nvidia") == "/proc/driver/nvidia"
    with pytest.raises(probe.ProbeError, match="not allow-listed"):
        probe.checked_tree_dest("/sys/class/net")
    assert probe.checked_inaccessible("/proc/meminfo") == ["/proc/meminfo"]
    with pytest.raises(probe.ProbeError, match="not allow-listed"):
        probe.checked_inaccessible("/etc")
    with pytest.raises(probe.ProbeError, match="not an allow-listed tool"):
        probe.checked_inaccessible("tool:py-spy")
    assert probe.checked_env_override("GPU_FAULT_EXPECTED_GPU_COUNT=7") == (
        "GPU_FAULT_EXPECTED_GPU_COUNT",
        "7",
    )
    assert probe.checked_env_override("GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT=") == (
        "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT",
        "",
    )
    with pytest.raises(probe.ProbeError, match="0..64"):
        probe.checked_env_override("GPU_FAULT_EXPECTED_GPU_COUNT=65")
    with pytest.raises(probe.ProbeError, match="not an overridable"):
        probe.checked_env_override("GPU_FAULT_CLUSTER_TOKEN=x")
    with pytest.raises(probe.ProbeError, match="Node Agent"):
        probe.checked_unit(probe.NODE_AGENT_UNIT)
    assert probe.ALLOWED_UNITS == (UNIT,), "one allow-listed unit: the host collector"
    assert probe.checked_name("plan-smartctl.json") == "plan-smartctl.json"
    with pytest.raises(probe.ProbeError, match="unsafe fixture name"):
        probe.checked_name("../window.json")
    with pytest.raises(probe.ProbeError, match="unsafe run ID"):
        probe.safe_id("Run;rm -rf")
    assert probe.state_dir(RUN) == probe.STATE_ROOT / f"source-{probe.digest(RUN)}", (
        "fixture content lives under /run/gfrs-live/source-<digest>"
    )
    assert str(probe.STATE_ROOT) == "/run/gfrs-live", (
        "the replay root is /run/gfrs-live"
    )


# --------------------------------------------------------------------------- #
# install-source
# --------------------------------------------------------------------------- #
def test_install_source_checks_sha256_and_writes_atomically(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    content = b"MemTotal: 1 kB\n"
    code, error = _main(
        "install-source",
        "--run-id",
        RUN,
        "--name",
        "meminfo",
        "--sha256",
        "0" * 64,
        "--b64",
        base64.b64encode(content).decode("ascii"),
        capsys=capsys,
    )
    assert code == 1 and error["error"] == "ProbeError", error
    assert not probe.fixtures_dir(RUN).exists(), (
        "a rejected source leaves nothing behind"
    )
    receipt = _install("meminfo", content, capsys)
    target = probe.fixtures_dir(RUN) / "meminfo"
    assert (
        target.read_bytes() == content
        and receipt["sha256"] == hashlib.sha256(content).hexdigest()
    )
    inode = target.stat().st_ino
    _install("meminfo", b"MemTotal: 2 kB\n", capsys)
    assert target.read_bytes() == b"MemTotal: 2 kB\n" and not list(
        probe.fixtures_dir(RUN).glob("*.tmp")
    )
    assert target.stat().st_ino != inode, (
        "a re-install replaces the file atomically (new inode)"
    )


# --------------------------------------------------------------------------- #
# bind window: deadman first, one drop-in, visible only inside the unit
# --------------------------------------------------------------------------- #
def test_bind_window_lifecycle(
    host: FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = (
        b"MemTotal:       1000000 kB\nMemFree:        10 kB\nMemAvailable:   20 kB\n"
    )
    _install("meminfo", fixture, capsys)
    code, opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "bind",
        "--bind",
        "meminfo:/proc/meminfo",
        "--restore-seconds",
        "600",
        capsys=capsys,
    )
    assert code == 0, opened
    dropin = probe.dropin_path(RUN)
    assert dropin.parent.name == f"{UNIT}.d", (
        "the drop-in is the host collector's and nobody else's"
    )
    assert dropin.read_text(encoding="utf-8") == (
        f"[Service]\nBindReadOnlyPaths={probe.fixtures_dir(RUN) / 'meminfo'}:/proc/meminfo\n"
    )
    assert host.dropin_present_at_arm is False, (
        "the deadman is armed before the drop-in exists"
    )
    assert host.restarts == 1 and f"{probe.deadman_unit(RUN)}.timer" in host.timers
    armed = next(call for call in host.calls if call[0] == "systemd-run")
    assert armed[-5:] == [
        sys.executable,
        str(probe.state_dir(RUN) / "probe.py"),
        "close-window",
        "--run-id",
        RUN,
    ], armed
    assert "--on-active=600s" in armed and filecmp.cmp(
        probe.state_dir(RUN) / "probe.py", probe.__file__, shallow=False
    ), "the deadman runs an identical copy of the probe after the requested delay"
    assert (
        opened["proof_scope"] == "live-replay" and opened["verification"]["ok"] is True
    ), opened
    check = opened["verification"]["checks"][0]
    assert check["kind"] == "bind" and check["unit_sees_fixture"] is True, check
    assert (
        check["host_sees_fixture"] is False and check["host_on_fixture_device"] is False
    ), "PID 1 still reads the real /proc/meminfo"
    assert _created_outside(tmp_path) == [], (
        "nothing is written outside the run dir and the drop-in"
    )

    code, status = _main("status", "--run-id", RUN, capsys=capsys)
    assert code == 0 and status["fixtures"]["meminfo"].startswith("MemTotal"), status
    code, snap = _main("snapshot", "--run-id", RUN, capsys=capsys)
    assert snap == {
        "run_id": RUN,
        "window_open": True,
        "dropin": True,
        "state_dir": True,
        "deadman": True,
    }, snap

    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["clean"] is True, closed
    assert closed["residue"] == {
        "dropin": False,
        "deadman": False,
        "state_dir": False,
    }, closed
    assert not dropin.exists() and not probe.state_dir(RUN).exists() and not host.timers
    assert host.restarts == 2, (
        "closing restarts the unit once more, without the drop-in"
    )
    assert host.binds == {}, "after the close the unit reads the real file again"
    code, again = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and again["already_closed"] is True and again["clean"] is True, (
        again
    )


def test_open_window_refuses_what_is_not_allow_listed_and_leaves_nothing(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    _install("passwd", b"root:x:0:0\n", capsys)
    code, error = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "bind",
        "--bind",
        "passwd:/etc/passwd",
        capsys=capsys,
    )
    assert code == 1 and error["error"] == "ProbeError", error
    assert (
        not probe.dropin_path(RUN).exists() and not host.timers and host.restarts == 0
    )
    assert not (probe.state_dir(RUN) / "window.json").exists(), (
        "no window state for a refused window"
    )
    code, error = _main("open-window", "--run-id", RUN, "--mode", "bind", capsys=capsys)
    assert code == 1, "bind mode needs a bind"
    code, error = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "mixed",
        "--env",
        "GPU_FAULT_CLUSTER_TOKEN=x",
        capsys=capsys,
    )
    assert code == 1 and not probe.dropin_path(RUN).exists(), error
    code, error = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "bind",
        "--bind",
        "passwd:/proc/meminfo",
        "--restore-seconds",
        "5",
        capsys=capsys,
    )
    assert code == 1 and not host.timers, (
        "a restore window below the floor is refused before arming"
    )


def test_window_that_fails_verification_is_closed_again(
    host: FakeHost, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _install("meminfo", b"MemTotal: 1 kB\n", capsys)
    real_restart = host.run

    def restart_without_applying(command: list[str], **kwargs: Any) -> Any:
        result = real_restart(command, **kwargs)
        if command[:2] == ["systemctl", "restart"]:
            host.binds = {}  # systemd "ignored" the drop-in: the unit sees the real file
        return result

    monkeypatch.setattr(probe, "run", restart_without_applying)
    code, error = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "bind",
        "--bind",
        "meminfo:/proc/meminfo",
        capsys=capsys,
    )
    assert code == 1 and error["error"] == "ProbeError", error
    assert (
        not probe.dropin_path(RUN).exists()
        and not probe.state_dir(RUN).exists()
        and not host.timers
    ), (
        "a window whose replacement is not visible inside the unit is closed, deadman disarmed"
    )


def test_deadman_command_restores_the_baseline(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    _install("stat", b"cpu  1 0 1 1 0 0 0 0 0 0\n", capsys)
    code, _opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "bind",
        "--bind",
        "stat:/proc/stat",
        capsys=capsys,
    )
    assert code == 0
    armed = next(call for call in host.calls if call[0] == "systemd-run")
    deadman_argv = armed[armed.index("close-window") :]
    # The deadman runs the probe copy with exactly this argv; run the same entry point.
    code, closed = _main(*deadman_argv, capsys=capsys)
    assert code == 0 and closed["clean"] is True, closed
    assert not probe.dropin_path(RUN).exists() and not probe.state_dir(RUN).exists(), (
        "the deadman removes the drop-in and the whole fixture directory"
    )
    assert host.binds == {}, "the unit was restarted without the drop-in"


def test_close_window_restores_an_untracked_dropin(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    dropin = probe.dropin_path(RUN)
    dropin.parent.mkdir(parents=True)
    dropin.write_text(
        "[Service]\nBindReadOnlyPaths=/x:/proc/meminfo\n", encoding="utf-8"
    )
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert (
        code == 0 and closed["untracked_dropin"] is True and closed["clean"] is True
    ), closed
    assert not dropin.exists() and host.restarts == 1, (
        "a crash between drop-in and state is still restored"
    )


# --------------------------------------------------------------------------- #
# whole-tree binds: the infiniband and nvidia proc fixtures
# --------------------------------------------------------------------------- #
def _sysfs_infiniband(root: Path, devices: int = 3) -> Path:
    """A /sys/class/infiniband look-alike: symlinks into a devices tree."""

    tree = root / "sys" / "class" / "infiniband"
    tree.mkdir(parents=True)
    drivers = root / "sys" / "bus" / "pci" / "drivers" / "efa"
    drivers.mkdir(parents=True)
    for index in range(devices):
        name = f"rdmap{index}s0"
        device = root / "sys" / "devices" / f"pci{index}" / "infiniband" / name
        (device / "ports" / "1" / "counters").mkdir(parents=True)
        (device / "ports" / "1" / "hw_counters").mkdir(parents=True)
        (device / "ports" / "1" / "state").write_text("4: ACTIVE\n")
        (device / "ports" / "1" / "phys_state").write_text("5: LinkUp\n")
        (device / "ports" / "1" / "counters" / "port_rcv_errors").write_text("0\n")
        (device / "ports" / "1" / "hw_counters" / "rx_drops").write_text("7\n")
        (device / "ports" / "1" / "hw_counters" / "tx_pkts").write_text("100\n")
        pci = root / "sys" / "devices" / f"pci{index}"
        (pci / "net" / f"eth{index}").mkdir(parents=True)
        (pci / "uevent").write_text("DRIVER=efa\nPCI_CLASS=20000\n")
        (pci / "vendor").write_text("0x1d0f\n")
        (pci / "device").write_text("0xefa2\n")
        (pci / "driver").symlink_to(drivers)
        (device / "device").symlink_to(pci)
        (tree / name).symlink_to(device)
    return tree


def test_infiniband_fixture_is_read_by_the_product_as_intended(
    host: FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _sysfs_infiniband(tmp_path / "host", devices=4)
    code, built = _main(
        "build-infiniband-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--drop",
        "rdmap3s0",
        "--duplicate",
        "rdmap9s0=rdmap0s0",
        "--port-state",
        "rdmap1s0:1:1: DOWN",
        "--phys-state",
        "rdmap1s0:1:3: Disabled",
        "--set",
        "rdmap0s0:1:hw_counters/rx_drops=5000",
        "--remove-counter",
        "rdmap2s0:1:hw_counters/tx_pkts",
        capsys=capsys,
    )
    assert code == 0, built
    tree = Path(built["path"])
    assert sorted(os.listdir(tree)) == [
        "rdmap0s0",
        "rdmap1s0",
        "rdmap2s0",
        "rdmap9s0",
    ], "dropped device gone, duplicate present, nothing but devices in the tree root"
    assert (
        built["devices"]["rdmap0s0"]["netdevs"] == ["eth0"]
        and built["devices"]["rdmap0s0"]["driver"] == "efa"
    )
    assert built["devices"]["rdmap9s0"]["duplicate_of"] == "rdmap0s0"
    assert (
        tree / "rdmap0s0" / "ports" / "1" / "hw_counters" / "rx_drops"
    ).read_text() == "5000\n"
    assert not (
        tree / "rdmap2s0" / "ports" / "1" / "hw_counters" / "tx_pkts"
    ).exists(), "the removed counter is gone"
    assert (tree / "rdmap0s0" / "device" / "driver").resolve().name == "efa"
    assert "infiniband-drivers" in str(
        (tree / "rdmap0s0" / "device" / "driver").resolve()
    ), "driver targets live beside the tree, never inside it"
    # The product's own reader over the fixture: 4 bound EFA devices, 3 active.
    pci, bound, active = efa_inventory_counts(tree, None)
    assert (pci, bound, active) == (None, 4, 3), (pci, bound, active)

    code, built = _main(
        "build-infiniband-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--driver",
        "rdmap0s0:none",
        capsys=capsys,
    )
    assert code == 0, built
    with pytest.raises(ValueError, match="missing or ambiguous RDMA driver identity"):
        efa_inventory_counts(Path(built["path"]), None)
    code, built = _main(
        "build-infiniband-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--driver",
        "rdmap0s0:ena",
        capsys=capsys,
    )
    assert code == 0 and efa_inventory_counts(Path(built["path"]), None) == (
        None,
        3,
        3,
    ), "a device bound to another driver is not an EFA device"
    code, built = _main(
        "build-infiniband-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--no-ports",
        "rdmap0s0",
        capsys=capsys,
    )
    assert code == 0, built
    with pytest.raises(ValueError, match="no observable ports"):
        efa_inventory_counts(Path(built["path"]), None)
    for bad in (
        ["--port-state", "rdmap0s0:1:DOWN"],
        ["--set", "rdmap0s0:1:stats/rx_drops=1"],
        ["--set", "rdmap0s0:1:hw_counters/rx_drops=1e9"],
        ["--drop", "../etc"],
        ["--duplicate", "rdmap0s0=rdmap1s0"],
        ["--driver", "rdmap7s0:none"],
    ):
        code, error = _main(
            "build-infiniband-fixture",
            "--run-id",
            RUN,
            "--source",
            str(source),
            *bad,
            capsys=capsys,
        )
        assert code == 1 and error["error"] == "ProbeError", (bad, error)


def test_tree_window_binds_the_whole_tree_and_set_value_steps_a_counter(
    host: FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _sysfs_infiniband(tmp_path / "host", devices=2)
    code, _built = _main(
        "build-infiniband-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        capsys=capsys,
    )
    assert code == 0
    code, opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "tree",
        "--tree",
        "infiniband:/sys/class/infiniband",
        capsys=capsys,
    )
    assert code == 0, opened
    tree = probe.fixtures_dir(RUN) / "infiniband"
    assert probe.dropin_path(RUN).read_text(encoding="utf-8") == (
        f"[Service]\nBindReadOnlyPaths={tree}:/sys/class/infiniband\n"
    )
    check = opened["verification"]["checks"][0]
    assert (
        check["kind"] == "tree"
        and check["unit_sees_fixture"] is True
        and check["ok"] is True
    ), check
    code, stepped = _main(
        "set-value",
        "--run-id",
        RUN,
        "--name",
        "infiniband",
        "--path",
        "rdmap0s0/ports/1/hw_counters/rx_drops",
        "--value",
        "9000",
        capsys=capsys,
    )
    assert code == 0, stepped
    assert (tree / "rdmap0s0/ports/1/hw_counters/rx_drops").read_text() == "9000"
    code, error = _main(
        "set-value",
        "--run-id",
        RUN,
        "--name",
        "infiniband",
        "--path",
        "../window.json",
        "--value",
        "x",
        capsys=capsys,
    )
    assert code == 1 and error["error"] == "ProbeError", (
        "a path that escapes the tree is refused"
    )
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["clean"] is True


def test_nvidia_proc_fixture_inaccessible_and_env_in_one_window(
    host: FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "host" / "proc" / "driver" / "nvidia"
    (source / "gpus" / "0000:53:00.0").mkdir(parents=True)
    (source / "gpus" / "0000:64:00.0").mkdir(parents=True)
    (source / "gpus" / "0000:53:00.0" / "information").write_text("Model: H200\n")
    (source / "version").write_text(
        "NVRM version: NVIDIA UNIX Open Kernel Module 580.65.06\n"
    )
    code, built = _main(
        "build-nvidia-proc-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--omit-version",
        "--gpu-count",
        "1",
        capsys=capsys,
    )
    assert code == 0, built
    tree = Path(built["path"])
    assert built["version"] is None and built["gpus"] == ["0000:53:00.0"]
    assert (
        not (tree / "version").exists()
        and (tree / "gpus" / "0000:53:00.0" / "information").read_text()
        == "Model: H200\n"
    )
    code, built = _main(
        "build-nvidia-proc-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--omit-gpus",
        "--version-text",
        "NVRM version: test",
        capsys=capsys,
    )
    assert (
        code == 0
        and not (Path(built["path"]) / "gpus").exists()
        and (Path(built["path"]) / "version").read_text() == "NVRM version: test\n"
    )
    code, opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "mixed",
        "--tree",
        "nvidia-proc:/proc/driver/nvidia",
        "--inaccessible",
        "/proc/meminfo",
        "--env",
        "GPU_FAULT_EXPECTED_GPU_COUNT=7",
        capsys=capsys,
    )
    assert code == 0, opened
    lines = probe.dropin_path(RUN).read_text(encoding="utf-8").splitlines()
    assert lines == [
        "[Service]",
        f"BindReadOnlyPaths={tree}:/proc/driver/nvidia",
        "InaccessiblePaths=-/proc/meminfo",
        "Environment=GPU_FAULT_EXPECTED_GPU_COUNT=7",
    ], lines
    kinds = {check["kind"]: check for check in opened["verification"]["checks"]}
    assert (
        kinds["inaccessible"]["unit_readable"] is False
        and kinds["inaccessible"]["host_readable"] is True
    ), kinds
    assert (
        kinds["env"]["unit_value"] == "7"
        and kinds["tree"]["ok"] is True
        and opened["verification"]["ok"] is True
    )
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert (
        code == 0
        and closed["clean"] is True
        and host.inaccessible == set()
        and host.environment == {}
    )


# --------------------------------------------------------------------------- #
# command window: the reviewed wrapper first on the unit's PATH
# --------------------------------------------------------------------------- #
def _plan(tool: str, *rules: dict[str, Any]) -> bytes:
    import time

    return json.dumps(
        {"tool": tool, "not_after": time.time() + 600, "rules": list(rules)}
    ).encode()


def test_command_window_installs_a_shim_and_puts_it_first_on_the_units_path(
    host: FakeHost,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "usr" / "sbin" / "smartctl"
    real.parent.mkdir(parents=True)
    real.write_text("#!/bin/sh\nexit 0\n")
    real.chmod(0o755)
    monkeypatch.setattr(probe, "STANDARD_PATH", f"{real.parent}:/usr/bin:/bin")
    _install(probe.WRAPPER_NAME, Path(wrapper.__file__).read_bytes(), capsys)
    code, error = _main(
        "install-wrapper", "--run-id", RUN, "--tool", "smartctl", capsys=capsys
    )
    assert code == 1 and error["error"] == "ProbeError", "no plan yet: refused"
    _install(
        "plan-smartctl.json",
        _plan("smartctl", {"query": "scan", "action": "exit", "exit_code": 2}),
        capsys,
    )
    code, installed = _main(
        "install-wrapper", "--run-id", RUN, "--tool", "smartctl", capsys=capsys
    )
    assert code == 0 and installed["real"] == str(real) and installed["rules"] == 1, (
        installed
    )
    shim = Path(installed["shim"])
    assert shim == probe.bin_dir(RUN) / "smartctl" and os.access(shim, os.X_OK)
    text = shim.read_text(encoding="utf-8")
    assert (
        f"{wrapper.TOOL_ENV}=smartctl" in text
        and f"{wrapper.DIR_ENV}={probe.state_dir(RUN)}" in text
    )
    assert f"-I {probe.fixtures_dir(RUN) / probe.WRAPPER_NAME}" in text, (
        "the shim runs the installed, checked wrapper"
    )
    assert real.read_text() == "#!/bin/sh\nexit 0\n", (
        "the host binary is never replaced"
    )

    code, error = _main(
        "install-wrapper", "--run-id", RUN, "--tool", "py-spy", capsys=capsys
    )
    assert code == 1 and error["error"] == "ProbeError", (
        "py-spy (Node Agent only) is refused by name"
    )
    _install(
        "plan-ethtool.json",
        _plan(
            "ethtool",
            {"query": "stats", "action": "set_json", "field": "x", "value": 1},
        ),
        capsys,
    )
    code, error = _main(
        "install-wrapper", "--run-id", RUN, "--tool", "ethtool", capsys=capsys
    )
    assert code == 1, (
        "a plan outside the per-tool allow-list is refused at install time"
    )
    code, error = _main(
        "install-wrapper", "--run-id", RUN, "--tool", "ipmitool", capsys=capsys
    )
    assert code == 1, "a tool the host lacks needs --allow-missing-real (and a plan)"

    code, opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "command",
        "--wrapper",
        "smartctl",
        capsys=capsys,
    )
    assert code == 0, opened
    lines = probe.dropin_path(RUN).read_text(encoding="utf-8").splitlines()
    assert lines == [
        "[Service]",
        f"Environment=PATH={probe.bin_dir(RUN)}:{probe.STANDARD_PATH}",
        f"ReadWritePaths={probe.audit_dir(RUN)}",
    ], lines
    check = opened["verification"]["checks"][0]
    assert check == {
        "kind": "wrapper",
        "dest": "smartctl",
        "shim_in_unit": True,
        "unit_path_first": True,
        "host_resolves_to": str(real),
        "ok": True,
    }, check
    assert (
        opened["wrappers"]["smartctl"]["plan_sha256"]
        == hashlib.sha256(
            (probe.fixtures_dir(RUN) / "plan-smartctl.json").read_bytes()
        ).hexdigest()
    )
    (probe.audit_dir(RUN) / "smartctl.jsonl").write_text(
        json.dumps({"query": "scan", "applied_rules": [0]}) + "\n", encoding="utf-8"
    )
    (probe.audit_dir(RUN) / "calls.json").write_text(
        json.dumps({"smartctl/0": 1}), encoding="utf-8"
    )
    code, audit = _main(
        "audit-read", "--run-id", RUN, "--tool", "smartctl", capsys=capsys
    )
    assert (
        code == 0
        and audit["entries"] == [{"query": "scan", "applied_rules": [0]}]
        and audit["calls"] == {"smartctl/0": 1}
    )
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["clean"] is True and not shim.exists(), (
        "the shim goes with the run directory"
    )
    assert host.environment == {}, "the unit runs with its own PATH again"


def test_command_window_needs_an_installed_wrapper(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    code, error = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "command",
        "--wrapper",
        "smartctl",
        capsys=capsys,
    )
    assert code == 1 and error["error"] == "ProbeError", error
    assert not probe.dropin_path(RUN).exists() and not host.timers


def test_pci_root_window_keeps_its_phase_one_shape(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    code, built = _main(
        "build-pci-fixture", "--run-id", RUN, "--efa-count", "15", capsys=capsys
    )
    assert code == 0 and built["efa_devices"] == 15
    code, opened = _main(
        "open-window", "--run-id", RUN, "--mode", "pci-root", capsys=capsys
    )
    assert code == 0, opened
    root = probe.fixtures_dir(RUN) / "pci"
    assert (
        probe.dropin_path(RUN).read_text(encoding="utf-8")
        == f"[Service]\nEnvironment={probe.PCI_ENV}={root}\n"
    )
    assert opened["verification"]["checks"] == [
        {"kind": "pci-root", "dest": probe.PCI_ENV, "unit_value": str(root), "ok": True}
    ]
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["clean"] is True


def test_probe_is_standard_library_only_and_python39_grammar() -> None:
    import ast

    for path in (Path(probe.__file__), Path(wrapper.__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
        imported = {
            (
                node.names[0].name
                if isinstance(node, ast.Import)
                else node.module or ""
            ).split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        }
        assert imported <= sys.stdlib_module_names, (
            f"{path.name} imports beyond the stdlib: {imported}"
        )
    assert shutil.which is not None
