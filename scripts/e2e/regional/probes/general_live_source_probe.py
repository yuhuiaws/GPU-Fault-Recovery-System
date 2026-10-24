#!/usr/bin/env python3
"""Host-collector file-source replay for GF-GENERAL-LIVE (001/008/009/013).

Runs on the target GPU node through ``host_probe_fixture.py``. Standard library
only, one JSON object on the last stdout line. It replaces only the raw reading
of one allow-listed source in the *host collector unit's own mount namespace*,
through a ``/run`` systemd drop-in that is removed in cleanup and by a deadman:

* **bind mode** -- ``BindReadOnlyPaths=<fixture>:<source>`` for a procfs/sysfs
  *file* the host collector reads. Only two destinations are allowed: ``/proc/stat``
  (001 invalid_delta: a reviewed static snapshot so the busy delta is <= 0) and a
  host-local ``gfrs*`` test interface's ``statistics/<counter>`` file (008
  statistics_failed: a non-numeric counter; 009 errors_delta: a counter the probe
  steps up between polls). A real interface or a real counter is never bound.
* **pci-root mode** -- ``Environment=GPU_FAULT_PCI_DEVICES_ROOT=<fixture tree>``
  for 013: a reviewed PCI devices tree with one fewer EFA function
  (``pci_missing``) or a malformed vendor file (``unreadable_state`` -> the
  inventory observation is UNAVAILABLE/UNKNOWN, never zero cards).

The Node Agent is never touched; no global mount is made and no real
``/proc``/sysfs file is written. ``proof_scope`` is always ``live-replay``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_ROOT = Path("/run/gpu-fault-source-replay")
DROPIN_ROOT = Path("/run/systemd/system")
HOST_UNIT = "gpu-fault-host-collector.service"
PCI_ENV = "GPU_FAULT_PCI_DEVICES_ROOT"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
#: The only destinations a bind may replace: /proc/stat, or the statistics
#: counter of a host-local gfrs* test interface. Nothing else.
PROC_STAT = "/proc/stat"
TEST_IFACE_STAT = re.compile(
    r"^/sys/class/net/gfrs[0-9a-z]{1,12}/statistics/[a-z_]{1,24}$"
)
EFA_VENDOR = "0x1d0f"
MIN_RESTORE_SECONDS = 60
MAX_RESTORE_SECONDS = 5400


class ProbeError(RuntimeError):
    pass


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str], *, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): {' '.join(command)}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return completed


def safe_id(value: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def digest(run_id: str) -> str:
    return hashlib.sha256(safe_id(run_id).encode()).hexdigest()[:16]


def state_dir(run_id: str) -> Path:
    return STATE_ROOT / digest(run_id)


def dropin_path(run_id: str) -> Path:
    return DROPIN_ROOT / f"{HOST_UNIT}.d" / f"gpu-fault-source-replay-{digest(run_id)}.conf"


def deadman_unit(run_id: str) -> str:
    return f"gpu-fault-source-replay-{digest(run_id)}"


def checked_bind_dest(dest: str) -> str:
    if dest == PROC_STAT or TEST_IFACE_STAT.fullmatch(dest) is not None:
        return dest
    raise ProbeError(f"bind destination {dest!r} is not allow-listed")


def checked_restore_seconds(value: int) -> int:
    if not MIN_RESTORE_SECONDS <= int(value) <= MAX_RESTORE_SECONDS:
        raise ProbeError(f"restore seconds is outside {MIN_RESTORE_SECONDS}..{MAX_RESTORE_SECONDS}")
    return int(value)


def unit_state(unit: str) -> dict[str, str]:
    completed = run(
        ["systemctl", "show", unit, "--property=LoadState", "--property=ActiveState",
         "--property=SubState", "--property=MainPID", "--property=NRestarts"],
        check=False,
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def load_state(run_id: str) -> dict[str, Any]:
    path = state_dir(run_id) / "window.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ProbeError("source replay window identity mismatch")
    return value


def save_state(run_id: str, state: dict[str, Any]) -> None:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "window.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def probe_copy(run_id: str) -> Path:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / "probe.py"
    if not target.is_file():
        shutil.copyfile(Path(__file__).resolve(), target)
        target.chmod(0o700)
    return target


def fixtures_dir(run_id: str) -> Path:
    return state_dir(run_id) / "fixtures"


def checked_name(name: str) -> str:
    if re.fullmatch(r"[a-z][a-z0-9_-]{0,24}", name) is None:
        raise ProbeError("unsafe fixture name")
    return name


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def install_source(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    name = checked_name(arguments.name)
    if SHA256.fullmatch(arguments.sha256) is None:
        raise ProbeError("a sha256 is required for every installed source")
    try:
        content = base64.b64decode(arguments.b64, validate=True)
    except (ValueError, TypeError) as error:
        raise ProbeError(f"source is not valid base64: {error}") from error
    if hashlib.sha256(content).hexdigest() != arguments.sha256:
        raise ProbeError("installed source sha256 does not match")
    directory = fixtures_dir(run_id)
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    target = directory / name
    target.write_bytes(content)
    target.chmod(0o644)
    emit({"run_id": run_id, "name": name, "path": str(target), "bytes": len(content)})


def build_pci_fixture(arguments: argparse.Namespace) -> None:
    """A reviewed /sys/bus/pci/devices tree: non-EFA devices plus EFA functions."""

    run_id = safe_id(arguments.run_id)
    efa_count = int(arguments.efa_count)
    if not 0 <= efa_count <= 32:
        raise ProbeError("efa-count is outside 0..32")
    root = fixtures_dir(run_id) / "pci"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    # A couple of plausible non-EFA devices so the tree is realistic.
    for index, (vendor, device) in enumerate((("0x10de", "0x2330"), ("0x1d0f", "0x8061"))):
        slot = root / f"0000:00:{index:02x}.0"
        slot.mkdir(mode=0o755)
        (slot / "vendor").write_text(vendor + "\n", encoding="ascii")
        (slot / "device").write_text(device + "\n", encoding="ascii")
    for index in range(efa_count):
        slot = root / f"0000:53:{index:02x}.0"
        slot.mkdir(mode=0o755)
        (slot / "vendor").write_text(EFA_VENDOR + "\n", encoding="ascii")
        device_id = "0xzz99" if arguments.malformed and index == 0 else "0xefa2"
        (slot / "device").write_text(device_id + "\n", encoding="ascii")
    emit(
        {
            "run_id": run_id,
            "path": str(root),
            "efa_devices": efa_count,
            "malformed": bool(arguments.malformed),
        }
    )


def set_value(arguments: argparse.Namespace) -> None:
    """Rewrite a bound fixture's backing file (e.g. step up a counter for 009)."""

    run_id = safe_id(arguments.run_id)
    name = checked_name(arguments.name)
    target = fixtures_dir(run_id) / name
    if not target.is_file():
        raise ProbeError("fixture is not installed")
    target.write_text(str(arguments.value), encoding="ascii")
    emit({"run_id": run_id, "name": name, "value": arguments.value})


# --------------------------------------------------------------------------- #
# deadman
# --------------------------------------------------------------------------- #
def arm_deadman(run_id: str, seconds: int) -> str:
    name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service"], check=False)
    run(
        ["systemd-run", f"--unit={name}", f"--on-active={int(seconds)}s",
         "--timer-property=AccuracySec=1s", "--property=Restart=on-failure",
         "--property=RestartSec=30s", "--property=StartLimitIntervalSec=0",
         sys.executable, str(probe_copy(run_id)), "close-window", "--run-id", run_id],
    )
    if unit_state(f"{name}.timer").get("ActiveState") != "active":
        raise ProbeError("source replay deadman timer is not active")
    return name


def disarm_deadman(run_id: str) -> None:
    name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{name}.service"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"], check=False)


# --------------------------------------------------------------------------- #
# window
# --------------------------------------------------------------------------- #
def preflight(arguments: argparse.Namespace) -> None:
    emit({"captured_at": utc_now(), "unit": unit_state(HOST_UNIT)})


def open_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    mode = arguments.mode
    if mode not in {"bind", "pci-root"}:
        raise ProbeError("mode must be bind or pci-root")
    restore_seconds = checked_restore_seconds(arguments.restore_seconds)
    if load_state(run_id):
        raise ProbeError("a source replay window for this run is already open")
    dropin = dropin_path(run_id)
    if dropin.exists():
        raise ProbeError("an untracked source replay drop-in already exists")
    before = unit_state(HOST_UNIT)
    if before.get("ActiveState") != "active":
        raise ProbeError(f"{HOST_UNIT} is not active at baseline")

    lines = ["[Service]"]
    binds: list[dict[str, str]] = []
    if mode == "bind":
        if not arguments.bind:
            raise ProbeError("bind mode needs at least one --bind SRC:DEST")
        for item in arguments.bind:
            src, _, dest = item.partition(":")
            dest = checked_bind_dest(dest)
            source = fixtures_dir(run_id) / checked_name(Path(src).name)
            if not source.is_file():
                raise ProbeError(f"bind source {source} is not installed")
            lines.append(f"BindReadOnlyPaths={source}:{dest}")
            binds.append({"source": str(source), "dest": dest})
    else:
        root = fixtures_dir(run_id) / "pci"
        if not root.is_dir():
            raise ProbeError("build the PCI fixture before opening a pci-root window")
        lines.append(f"Environment={PCI_ENV}={root}")

    probe_copy(run_id)
    state = {
        "run_id": run_id,
        "mode": mode,
        "dropin": str(dropin),
        "binds": binds,
        "before": before,
        "opened_at": utc_now(),
    }
    save_state(run_id, state)
    deadman = arm_deadman(run_id, restore_seconds)
    dropin.parent.mkdir(parents=True, exist_ok=True)
    dropin.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", HOST_UNIT], timeout=120)
    after = unit_state(HOST_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{HOST_UNIT} is not active after the window opened")
    emit(
        {
            **state,
            "after": after,
            "deadman_timer": unit_state(f"{deadman}.timer"),
            "proof_scope": "live-replay",
        }
    )


def apply_value(arguments: argparse.Namespace) -> None:
    """Rewrite a bound counter and restart the unit so it re-reads the new value."""

    run_id = safe_id(arguments.run_id)
    if not load_state(run_id):
        raise ProbeError("no source replay window is open for this run")
    set_value(arguments)
    # The host collector reads the file each poll; a restart is not required,
    # but reported state confirms it stayed active across the counter step.
    emit({"run_id": run_id, "unit": unit_state(HOST_UNIT), "value": arguments.value})


def status(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    if not state:
        raise ProbeError("no source replay window is open for this run")
    fixtures = {}
    directory = fixtures_dir(run_id)
    if directory.is_dir():
        for item in sorted(directory.iterdir()):
            if item.is_file():
                fixtures[item.name] = item.read_text(encoding="utf-8", errors="replace")[:64]
    emit({"run_id": run_id, "mode": state["mode"], "unit": unit_state(HOST_UNIT), "fixtures": fixtures})


def close_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    if not state:
        dropin_path(run_id).unlink(missing_ok=True)
        if state_dir(run_id).is_dir():
            shutil.rmtree(state_dir(run_id))
        emit({"run_id": run_id, "already_closed": True, "clean": True})
        return
    dropin = Path(state["dropin"])
    dropin.unlink(missing_ok=True)
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", HOST_UNIT], timeout=120)
    after = unit_state(HOST_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{HOST_UNIT} is not active after the window closed")
    disarm_deadman(run_id)
    residue = {
        "dropin": dropin.exists(),
        "deadman": unit_state(f"{deadman_unit(run_id)}.timer").get("ActiveState") == "active",
    }
    if state_dir(run_id).is_dir():
        shutil.rmtree(state_dir(run_id))
    emit({"run_id": run_id, "after": after, "residue": residue, "clean": not any(residue.values())})


def snapshot(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    emit(
        {
            "run_id": run_id,
            "window_open": bool(state),
            "dropin": dropin_path(run_id).exists(),
            "deadman": unit_state(f"{deadman_unit(run_id)}.timer").get("ActiveState") == "active",
        }
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    commands.add_parser("preflight").set_defaults(handler=preflight)

    item = commands.add_parser("install-source")
    item.add_argument("--run-id", required=True)
    item.add_argument("--name", required=True)
    item.add_argument("--sha256", required=True)
    item.add_argument("--b64", required=True)
    item.set_defaults(handler=install_source)

    item = commands.add_parser("build-pci-fixture")
    item.add_argument("--run-id", required=True)
    item.add_argument("--efa-count", type=int, required=True)
    item.add_argument("--malformed", action="store_true")
    item.set_defaults(handler=build_pci_fixture)

    item = commands.add_parser("open-window")
    item.add_argument("--run-id", required=True)
    item.add_argument("--mode", choices=("bind", "pci-root"), required=True)
    item.add_argument("--bind", action="append", default=[])
    item.add_argument("--restore-seconds", type=int, default=900)
    item.set_defaults(handler=open_window)

    item = commands.add_parser("set-value")
    item.add_argument("--run-id", required=True)
    item.add_argument("--name", required=True)
    item.add_argument("--value", required=True)
    item.set_defaults(handler=apply_value)

    item = commands.add_parser("status")
    item.add_argument("--run-id", required=True)
    item.set_defaults(handler=status)

    item = commands.add_parser("close-window")
    item.add_argument("--run-id", required=True)
    item.set_defaults(handler=close_window)

    item = commands.add_parser("snapshot")
    item.add_argument("--run-id", required=True)
    item.set_defaults(handler=snapshot)
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        emit({"error": type(exc).__name__, "error_kind": "probe", "command": getattr(arguments, "command", None)})
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
