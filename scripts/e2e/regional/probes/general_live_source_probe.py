#!/usr/bin/env python3
"""Host-collector source replay for GF-GENERAL-LIVE (the shared raw-source fixture).

Runs on the target GPU node through ``host_probe_fixture.py``. Standard library
only, Python 3.9+, one JSON object on the last stdout line. It replaces only the
raw reading of allow-listed sources in the *host collector unit's own private
environment* (``gpu-fault-host-collector.service``), through one ``/run``
systemd drop-in that cleanup and a ``systemd-run --on-active`` deadman remove.
Fixture content lives under ``/run/gfrs-live/source-<digest>/`` (tmpfs) and is
installed base64 + sha256 like ``env_replay_proxy``'s install-source. Modes,
combinable in one window:

* **bind** -- ``BindReadOnlyPaths=<fixture file>:<source file>`` in the unit's
  mount namespace for an allow-listed procfs/sysfs *file*: ``/proc/stat`` (001),
  a host-local ``gfrs*`` test interface's ``statistics/<counter>`` (008/009),
  ``/proc/meminfo`` (002), an EFA port's ``hw_counters``/``counters``/``state``/
  ``phys_state`` file (010/011/013) or ``/proc/driver/nvidia/version`` (032).
* **tree** -- the same for a whole reviewed directory: ``/sys/class/infiniband``
  (013 inventory variants need every device's ports; a single-file bind cannot
  express one missing, extra or inactive device) or ``/proc/driver/nvidia``
  (012/032). ``build-infiniband-fixture`` / ``build-nvidia-proc-fixture`` copy
  the real tree and apply reviewed mutations.
* **command** -- a reviewed pass-through wrapper (``tools/host_command_replay.py``,
  installed with its sha256) first on the unit's PATH for ``smartctl``,
  ``ipmitool``, ``ethtool`` or the host collector's ``nvidia-smi`` queries. It
  forwards to the real binary and rewrites only the plan's allow-listed fields;
  no host binary is replaced. ``py-spy`` is refused (Node Agent only).
* **inaccessible** -- ``InaccessiblePaths=`` for an allow-listed path or tool
  binary inside the unit only (002 read_failure, 007 tool_missing, 012
  proc_unreadable).
* **env** -- an allow-listed expected-count override (012 expectation_wrong).
* **pci-root** -- ``Environment=GPU_FAULT_PCI_DEVICES_ROOT=<fixture tree>`` (013).

``open-window`` arms the deadman *before* writing the drop-in, restarts the one
unit and then **verifies** through ``nsenter`` that every replacement is visible
in the unit's mount namespace / process environment and *not* in PID 1's.
The Node Agent is never touched; no global mount is made and no real
``/proc``/sysfs file is written. ``proof_scope`` is always ``live-replay``.

Two read-only commands serve the variants' oracle: ``inventory`` lists the
node's RDMA tree as the product reads it (driver identity, owned netdevs, port
``state``/``phys_state``, every ``counters``/``hw_counters`` file with its value,
the names ``ethtool -S`` prints per netdev), and ``read-source --path`` shows
one allow-listed source file as the unit sees it next to what PID 1 sees, so a
closed window can prove the real value is read again.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_ROOT = Path("/run/gfrs-live")
DROPIN_ROOT = Path("/run/systemd/system")
HOST_UNIT = "gpu-fault-host-collector.service"
NODE_AGENT_UNIT = "gpu-fault-node-agent.service"
#: The one unit a window may touch. The Node Agent is absent on purpose.
ALLOWED_UNITS = (HOST_UNIT,)
PCI_ENV = "GPU_FAULT_PCI_DEVICES_ROOT"
WRAPPER_NAME = "host_command_replay.py"
STANDARD_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FIXTURE_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,40}(\.(py|json|txt))?$")
MAX_SOURCE_BYTES = 512 * 1024
#: Bind destinations: exact files, or the two sysfs families below.
BIND_FILES = ("/proc/stat", "/proc/meminfo", "/proc/driver/nvidia/version")
TEST_IFACE_STAT = re.compile(
    r"^/sys/class/net/gfrs[0-9a-z]{1,12}/statistics/[a-z_]{1,24}$"
)
EFA_PORT_FILE = re.compile(
    r"^/sys/class/infiniband/[A-Za-z0-9][A-Za-z0-9_.-]{0,31}/ports/[0-9]{1,2}/"
    r"((hw_counters|counters)/[a-z][a-z0-9_]{0,47}|state|phys_state)$"
)
#: Whole-tree bind destinations.
TREE_DESTS = ("/sys/class/infiniband", "/proc/driver/nvidia")
#: Paths a window may make inaccessible inside the unit (plus allow-listed tools).
INACCESSIBLE_PATHS = (
    "/proc/meminfo",
    "/proc/driver/nvidia",
    "/proc/driver/nvidia/gpus",
    "/proc/driver/nvidia/version",
    "/sys/class/infiniband",
)
WRAPPABLE_TOOLS = ("smartctl", "ipmitool", "ethtool", "nvidia-smi")
#: Environment overrides and their value pattern (empty = unset the expectation).
ENV_OVERRIDES = {
    "GPU_FAULT_EXPECTED_GPU_COUNT": re.compile(r"^([0-9]|[1-5][0-9]|6[0-4])?$"),
    "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT": re.compile(r"^([0-9]|[1-5][0-9]|6[0-4])?$"),
}
EFA_VENDOR = "0x1d0f"
MIN_RESTORE_SECONDS = 60
MAX_RESTORE_SECONDS = 5400
MAX_TREE_DEVICES = 64
MAX_AUDIT_LINES = 400
#: Bounds of the read-only ``inventory``: counter files per port group, counter
#: names per ``ethtool -S`` listing (an ENA netdev prints ~45 per queue).
MAX_INVENTORY_COUNTERS = 128
MAX_ETHTOOL_NAMES = 2048
MAX_SOURCE_VALUE_CHARS = 256
NETDEV_ROOT = "/sys/class/net"
NETDEV_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,15}$")
DEVICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")
PORT_NAME = re.compile(r"^[0-9]{1,2}$")
PORT_STATE = re.compile(r"^[0-9]{1,2}: [A-Za-z][A-Za-z_ -]{0,30}$")
COUNTER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
COUNTER_VALUE = re.compile(r"^(-?[0-9]{1,20}|[A-Za-z_-]{1,16})$")
DRIVER_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,15}$")
BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")


class ProbeError(RuntimeError):
    pass


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    command: list[str], *, check: bool = True, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
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
    return STATE_ROOT / f"source-{digest(run_id)}"


def dropin_path(run_id: str) -> Path:
    return (
        DROPIN_ROOT
        / f"{HOST_UNIT}.d"
        / f"gpu-fault-source-replay-{digest(run_id)}.conf"
    )


def deadman_unit(run_id: str) -> str:
    return f"gpu-fault-source-replay-{digest(run_id)}"


def checked_unit(unit: str) -> str:
    if unit == NODE_AGENT_UNIT:
        raise ProbeError("the Node Agent unit is never in a replay allow-list")
    if unit not in ALLOWED_UNITS:
        raise ProbeError(f"{unit!r} is not an allow-listed collector unit")
    return unit


def checked_bind_dest(dest: str) -> str:
    if (
        dest in BIND_FILES
        or TEST_IFACE_STAT.fullmatch(dest) is not None
        or EFA_PORT_FILE.fullmatch(dest) is not None
    ):
        return dest
    raise ProbeError(f"bind destination {dest!r} is not allow-listed")


def checked_tree_dest(dest: str) -> str:
    if dest in TREE_DESTS:
        return dest
    raise ProbeError(f"tree destination {dest!r} is not allow-listed")


def checked_source_path(path: str) -> str:
    """A readable source: an allow-listed bind file, or a file of a tree dest."""

    if ".." in path.split("/") or not path.startswith("/"):
        raise ProbeError(f"source path {path!r} is not absolute and plain")
    try:
        return checked_bind_dest(path)
    except ProbeError:
        if any(path == dest or path.startswith(dest + "/") for dest in TREE_DESTS):
            return path
        raise ProbeError(f"source path {path!r} is not allow-listed") from None


def tool_locations(tool: str) -> list[str]:
    """Every copy of an allow-listed tool in the standard directories."""

    if tool not in WRAPPABLE_TOOLS:
        raise ProbeError(f"{tool!r} is not an allow-listed tool")
    found = []
    for directory in STANDARD_PATH.split(":"):
        candidate = Path(directory) / tool
        if candidate.is_file() and os.access(candidate, os.X_OK):
            found.append(str(candidate))
    return found


def checked_inaccessible(item: str) -> list[str]:
    """``InaccessiblePaths`` entries for an allow-listed path or ``tool:<name>``."""

    if item.startswith("tool:"):
        paths = tool_locations(item[len("tool:") :])
        if not paths:
            raise ProbeError(
                f"{item[5:]} is not installed on the host; nothing to hide"
            )
        return paths
    if item in INACCESSIBLE_PATHS:
        return [item]
    raise ProbeError(f"inaccessible path {item!r} is not allow-listed")


def checked_env_override(pair: str) -> tuple[str, str]:
    name, separator, value = pair.partition("=")
    if not separator or name not in ENV_OVERRIDES:
        raise ProbeError(f"{name!r} is not an overridable collector setting")
    if ENV_OVERRIDES[name].fullmatch(value) is None:
        raise ProbeError(f"{name} accepts only an integer in 0..64 or an empty value")
    return name, value


def checked_restore_seconds(value: int) -> int:
    if not MIN_RESTORE_SECONDS <= int(value) <= MAX_RESTORE_SECONDS:
        raise ProbeError(
            f"restore seconds is outside {MIN_RESTORE_SECONDS}..{MAX_RESTORE_SECONDS}"
        )
    return int(value)


def checked_name(name: str) -> str:
    if FIXTURE_NAME.fullmatch(name) is None:
        raise ProbeError("unsafe fixture name")
    return name


def unit_state(unit: str) -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            unit,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--property=NRestarts",
        ],
        check=False,
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def process_environment(pid: str, keys: tuple[str, ...]) -> dict[str, str]:
    """Allow-listed keys of a running process's real environment."""

    if not pid or not pid.isdigit() or pid == "0":
        return {}
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        key, separator, value = item.partition(b"=")
        name = key.decode("utf-8", "replace")
        if separator and name in keys:
            values[name] = value.decode("utf-8", "replace")
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
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    path = directory / "window.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def probe_copy(run_id: str) -> Path:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    target = directory / "probe.py"
    if not target.is_file():
        shutil.copyfile(Path(__file__).resolve(), target)
        target.chmod(0o700)
    return target


def fixtures_dir(run_id: str) -> Path:
    return state_dir(run_id) / "fixtures"


def bin_dir(run_id: str) -> Path:
    return state_dir(run_id) / "bin"


def audit_dir(run_id: str) -> Path:
    return state_dir(run_id) / "audit"


def write_atomic(path: Path, content: bytes, mode: int) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.chmod(mode)
    os.replace(temporary, path)


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
    if len(content) > MAX_SOURCE_BYTES:
        raise ProbeError("installed source exceeds its size bound")
    if hashlib.sha256(content).hexdigest() != arguments.sha256:
        raise ProbeError("installed source sha256 does not match")
    directory = fixtures_dir(run_id)
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    target = directory / name
    if target.is_dir():
        raise ProbeError("a fixture tree of that name exists")
    # Atomic: a plan re-installed while a window is open is re-read by the
    # wrapper on its next call and must never be seen half-written.
    write_atomic(target, content, 0o644)
    emit(
        {
            "run_id": run_id,
            "name": name,
            "path": str(target),
            "bytes": len(content),
            "sha256": arguments.sha256,
        }
    )


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
    for index, (vendor, device) in enumerate(
        (("0x10de", "0x2330"), ("0x1d0f", "0x8061"))
    ):
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


def _copy_text(source: Path, target: Path) -> bool:
    try:
        content = source.read_bytes()
    except OSError:
        return False
    target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    target.write_bytes(content)
    target.chmod(0o644)
    return True


def _split(value: str, separator: str, count: int, label: str) -> list[str]:
    parts = value.split(separator, count - 1)
    if len(parts) != count:
        raise ProbeError(f"{label} must have {count} '{separator}'-separated parts")
    return parts


def _device_dir(root: Path, device: str) -> Path:
    if DEVICE_NAME.fullmatch(device) is None:
        raise ProbeError(f"unsafe RDMA device name {device!r}")
    path = root / device
    if not path.is_dir():
        raise ProbeError(f"the fixture tree has no device {device!r}")
    return path


def _port_dir(root: Path, device: str, port: str) -> Path:
    if PORT_NAME.fullmatch(port) is None:
        raise ProbeError(f"unsafe port name {port!r}")
    path = _device_dir(root, device) / "ports" / port
    if not path.is_dir():
        raise ProbeError(f"device {device!r} has no port {port!r}")
    return path


def _copy_rdma_device(source: Path, target: Path, drivers: Path) -> dict[str, Any]:
    """One real RDMA device as the product reads it, into the fixture tree."""

    real = source.resolve()
    summary: dict[str, Any] = {
        "ports": {},
        "counters": 0,
        "driver": None,
        "netdevs": [],
    }
    for port in sorted((real / "ports").iterdir()) if (real / "ports").is_dir() else []:
        port_target = target / "ports" / port.name
        port_target.mkdir(mode=0o755, parents=True, exist_ok=True)
        states = {}
        for name in ("state", "phys_state"):
            if _copy_text(port / name, port_target / name):
                states[name] = (port_target / name).read_text(encoding="utf-8").strip()
        for group in ("counters", "hw_counters"):
            group_dir = port / group
            if group_dir.is_dir():
                for counter in sorted(group_dir.iterdir()):
                    if counter.is_file() and _copy_text(
                        counter, port_target / group / counter.name
                    ):
                        summary["counters"] += 1
        summary["ports"][port.name] = states
    device_dir = real / "device"
    (target / "device").mkdir(mode=0o755, parents=True, exist_ok=True)
    driver = device_dir / "driver"
    if driver.is_symlink() or driver.is_dir():
        name = driver.resolve().name
        summary["driver"] = name
        (drivers / name).mkdir(mode=0o755, parents=True, exist_ok=True)
        (target / "device" / "driver").symlink_to(drivers / name)
    for name in ("uevent", "vendor", "device"):
        _copy_text(device_dir / name, target / "device" / name)
    net = device_dir / "net"
    if net.is_dir():
        for netdev in sorted(net.iterdir()):
            (target / "device" / "net" / netdev.name).mkdir(
                mode=0o755, parents=True, exist_ok=True
            )
            summary["netdevs"].append(netdev.name)
    return summary


def _set_driver(device_dir: Path, drivers: Path, mode: str) -> None:
    link = device_dir / "device" / "driver"
    if link.is_symlink() or link.exists():
        link.unlink()
    uevent = device_dir / "device" / "uevent"
    lines = (
        [
            line
            for line in uevent.read_text(encoding="utf-8").splitlines()
            if not line.startswith("DRIVER=")
        ]
        if uevent.is_file()
        else []
    )
    if mode == "ambiguous":
        lines += ["DRIVER=efa", "DRIVER=ena"]
    elif mode != "none":
        if DRIVER_NAME.fullmatch(mode) is None:
            raise ProbeError(f"unsafe driver name {mode!r}")
        (drivers / mode).mkdir(mode=0o755, parents=True, exist_ok=True)
        link.symlink_to(drivers / mode)
        lines.append(f"DRIVER={mode}")
    uevent.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    uevent.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_infiniband_fixture(arguments: argparse.Namespace) -> None:
    """A copy of the node's /sys/class/infiniband with reviewed mutations.

    Mutations, applied in this order: ``--drop DEV`` (device absent),
    ``--duplicate NEW=FROM`` (excess device), ``--port-state DEV:PORT:TEXT`` /
    ``--phys-state DEV:PORT:TEXT`` (link inactive), ``--driver
    DEV:none|ambiguous|NAME`` (unknown or non-EFA identity), ``--set
    DEV:PORT:GROUP/COUNTER=VALUE`` / ``--remove-counter DEV:PORT:GROUP/COUNTER``
    (counter replay, a broken pair), ``--no-ports DEV`` (no observable ports).
    The driver directories the ``device/driver`` symlinks resolve to live beside
    the tree, never inside it, so the product's device walk sees devices only.
    """

    run_id = safe_id(arguments.run_id)
    source = Path(arguments.source)
    if not source.is_dir():
        raise ProbeError(f"{source} is not a directory")
    root = fixtures_dir(run_id) / "infiniband"
    drivers = fixtures_dir(run_id) / "infiniband-drivers"
    for path in (root, drivers):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(mode=0o755, parents=True, exist_ok=True)
    for name in arguments.drop:
        if DEVICE_NAME.fullmatch(name) is None or not (source / name).exists():
            raise ProbeError(f"--drop names no device of {source}: {name!r}")
    devices: dict[str, Any] = {}
    for device in sorted(source.iterdir()):
        if DEVICE_NAME.fullmatch(device.name) is None or device.name in arguments.drop:
            continue
        devices[device.name] = _copy_rdma_device(device, root / device.name, drivers)
    if not devices and not arguments.allow_empty:
        raise ProbeError(
            f"{source} lists no RDMA device; pass --allow-empty to build an empty tree"
        )
    for item in arguments.duplicate:
        new, origin = _split(item, "=", 2, "--duplicate")
        if DEVICE_NAME.fullmatch(new) is None or new in devices:
            raise ProbeError(f"duplicate target {new!r} is unsafe or already present")
        shutil.copytree(_device_dir(root, origin), root / new, symlinks=True)
        devices[new] = {**devices[origin], "duplicate_of": origin}
    if len(devices) > MAX_TREE_DEVICES:
        raise ProbeError(f"the fixture tree holds more than {MAX_TREE_DEVICES} devices")
    mutations: list[str] = []
    for flag, name in (("port_state", "state"), ("phys_state", "phys_state")):
        for item in getattr(arguments, flag):
            device, port, text = _split(item, ":", 3, f"--{flag.replace('_', '-')}")
            if PORT_STATE.fullmatch(text) is None:
                raise ProbeError(f"port state text {text!r} is not allow-listed")
            path = _port_dir(root, device, port) / name
            path.write_text(text + "\n", encoding="utf-8")
            mutations.append(f"{device}/ports/{port}/{name}")
    for item in arguments.driver:
        device, mode = _split(item, ":", 2, "--driver")
        _set_driver(_device_dir(root, device), drivers, mode)
        mutations.append(f"{device}/device/driver")
    for item in arguments.set:
        target, value = _split(item, "=", 2, "--set")
        device, port, counter = _split(target, ":", 3, "--set")
        group, _, name = counter.partition("/")
        if (
            group not in ("counters", "hw_counters")
            or COUNTER_NAME.fullmatch(name) is None
        ):
            raise ProbeError(f"counter {counter!r} is not allow-listed")
        if COUNTER_VALUE.fullmatch(value) is None:
            raise ProbeError(f"counter value {value!r} is not allow-listed")
        path = _port_dir(root, device, port) / group / name
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        path.write_text(value + "\n", encoding="utf-8")
        mutations.append(f"{device}/ports/{port}/{group}/{name}")
    for item in arguments.remove_counter:
        device, port, counter = _split(item, ":", 3, "--remove-counter")
        group, _, name = counter.partition("/")
        if (
            group not in ("counters", "hw_counters")
            or COUNTER_NAME.fullmatch(name) is None
        ):
            raise ProbeError(f"counter {counter!r} is not allow-listed")
        path = _port_dir(root, device, port) / group / name
        if not path.is_file():
            raise ProbeError(f"{device}:{port}:{counter} is not in the tree")
        path.unlink()
        mutations.append(f"{device}/ports/{port}/{group}/{name} (removed)")
    for device in arguments.no_ports:
        ports = _device_dir(root, device) / "ports"
        shutil.rmtree(ports)
        ports.mkdir(mode=0o755)
        mutations.append(f"{device}/ports (emptied)")
    for item in arguments.netdev:
        # 010: the host collector runs ``ethtool -S`` only on the netdevs an
        # EFA function owns (``device/net/*``); on a node whose EFA functions
        # own none, a real physical netdev of the node is attached to one so
        # the ethtool reading can be replayed. The netdev itself is untouched.
        device, iface = _split(item, ":", 2, "--netdev")
        if (
            NETDEV_NAME.fullmatch(iface) is None
            or not (Path(arguments.netdev_root) / iface / "device").exists()
        ):
            raise ProbeError(
                f"--netdev names no physical netdev of {arguments.netdev_root}: {iface!r}"
            )
        (_device_dir(root, device) / "device" / "net" / iface).mkdir(
            mode=0o755, parents=True, exist_ok=True
        )
        devices[device]["netdevs"] = sorted({*devices[device]["netdevs"], iface})
        mutations.append(f"{device}/device/net/{iface} (attached)")
    summary = {
        "run_id": run_id,
        "path": str(root),
        "source": str(source),
        "devices": devices,
        "dropped": list(arguments.drop),
        "mutations": mutations,
    }
    (fixtures_dir(run_id) / "infiniband.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    emit(summary)


def build_nvidia_proc_fixture(arguments: argparse.Namespace) -> None:
    """A copy of /proc/driver/nvidia (version + gpus/<bdf>) with reviewed gaps."""

    run_id = safe_id(arguments.run_id)
    source = Path(arguments.source)
    root = fixtures_dir(run_id) / "nvidia-proc"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    version: str | None = None
    if arguments.version_text is not None:
        if len(arguments.version_text.encode("utf-8")) > 1024:
            raise ProbeError("version text exceeds 1 KiB")
        version = arguments.version_text + "\n"
    elif not arguments.omit_version:
        try:
            version = (source / "version").read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise ProbeError(f"{source}/version is unreadable: {error}") from error
    if version is not None:
        (root / "version").write_text(version, encoding="utf-8")
    gpus: list[str] = []
    if not arguments.omit_gpus:
        gpus_dir = root / "gpus"
        gpus_dir.mkdir(mode=0o755)
        try:
            entries = sorted(
                path.name
                for path in (source / "gpus").iterdir()
                if path.is_dir() and BDF.fullmatch(path.name)
            )
        except OSError as error:
            raise ProbeError(f"{source}/gpus is unreadable: {error}") from error
        if arguments.gpu_count is not None:
            if not 0 <= int(arguments.gpu_count) <= len(entries):
                raise ProbeError("gpu-count is outside 0..<real GPU count>")
            entries = entries[: int(arguments.gpu_count)]
        for name in entries:
            (gpus_dir / name).mkdir(mode=0o755)
            _copy_text(
                source / "gpus" / name / "information", gpus_dir / name / "information"
            )
            gpus.append(name)
    emit(
        {
            "run_id": run_id,
            "path": str(root),
            "source": str(source),
            "version": version.strip()[:120] if version is not None else None,
            "gpus": gpus,
            "omit_gpus": bool(arguments.omit_gpus),
        }
    )


def _text(path: Path, limit: int = MAX_SOURCE_VALUE_CHARS) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()[:limit]
    except OSError:
        return None


def _rdma_driver(device: Path) -> tuple[str | None, list[str]]:
    """``(resolved driver, uevent DRIVER= values)`` the way ``is_efa_device`` reads them."""

    try:
        resolved: str | None = (device / "device" / "driver").resolve(strict=True).name
    except OSError:
        resolved = None
    uevent = _text(device / "device" / "uevent", 4096) or ""
    drivers = [
        line[len("DRIVER=") :]
        for line in uevent.splitlines()
        if line.startswith("DRIVER=")
    ]
    return resolved, drivers


def rdma_inventory(root: Path, netdev_root: Path) -> dict[str, Any]:
    """The node's RDMA devices as the host collector reads them (read-only)."""

    devices: dict[str, Any] = {}
    for device in sorted(root.iterdir()) if root.is_dir() else []:
        if DEVICE_NAME.fullmatch(device.name) is None:
            continue
        driver, uevent_drivers = _rdma_driver(device)
        net = device / "device" / "net"
        ports: dict[str, Any] = {}
        ports_dir = device / "ports"
        for port in sorted(ports_dir.iterdir()) if ports_dir.is_dir() else []:
            if PORT_NAME.fullmatch(port.name) is None:
                continue
            entry: dict[str, Any] = {
                "state": _text(port / "state"),
                "phys_state": _text(port / "phys_state"),
            }
            for group in ("counters", "hw_counters"):
                values: dict[str, str | None] = {}
                group_dir = port / group
                names = sorted(group_dir.iterdir()) if group_dir.is_dir() else []
                for counter in names[:MAX_INVENTORY_COUNTERS]:
                    if counter.is_file() and COUNTER_NAME.fullmatch(counter.name):
                        values[counter.name] = _text(counter, 64)
                entry[group] = values
            ports[port.name] = entry
        devices[device.name] = {
            "driver": driver,
            "uevent_drivers": uevent_drivers,
            "efa": driver == "efa" or (driver is None and uevent_drivers == ["efa"]),
            "netdevs": sorted(item.name for item in net.iterdir())
            if net.is_dir()
            else [],
            "ports": ports,
        }
    physical = (
        sorted(
            item.name
            for item in netdev_root.iterdir()
            if (item / "device").exists() and NETDEV_NAME.fullmatch(item.name)
        )
        if netdev_root.is_dir()
        else []
    )
    return {"root": str(root), "devices": devices, "physical_netdevs": physical}


def ethtool_names(interfaces: list[str]) -> dict[str, Any]:
    """What the real ``ethtool -S`` prints for each netdev: counter names only."""

    path = shutil.which("ethtool", path=STANDARD_PATH)
    result: dict[str, Any] = {"path": path, "statistics": {}}
    if path is None:
        return result
    for interface in interfaces:
        completed = run([path, "-S", interface], check=False, timeout=30)
        names = []
        for line in completed.stdout.splitlines():
            name, separator, value = line.partition(":")
            # ``NIC statistics:`` heads the listing; a counter line has a value.
            if separator and name.strip() and value.strip():
                names.append(name.strip())
        result["statistics"][interface] = {
            "returncode": completed.returncode,
            "counters": names[:MAX_ETHTOOL_NAMES],
            "truncated": len(names) > MAX_ETHTOOL_NAMES,
            "stderr": completed.stderr.strip()[:200],
        }
    return result


def inventory(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    value = rdma_inventory(Path(arguments.source), Path(arguments.netdev_root))
    owned = {
        name
        for device in value["devices"].values()
        if device["efa"]
        for name in device["netdevs"]
    }
    value["ethtool"] = ethtool_names(sorted(owned | set(value["physical_netdevs"])))
    emit({"run_id": run_id, "captured_at": utc_now(), **value})


def view_text(pid: str, path: str) -> dict[str, Any]:
    """What ``pid``'s mount namespace reads at ``path``: the text and its device."""

    read = nsenter(pid, ["/usr/bin/cat", path])
    stat = nsenter(pid, ["/usr/bin/stat", "-L", "-c", "%d", path])
    return {
        "readable": read.returncode == 0,
        "value": read.stdout.strip()[:MAX_SOURCE_VALUE_CHARS]
        if read.returncode == 0
        else None,
        "device": stat.stdout.strip() if stat.returncode == 0 else None,
    }


def read_source(arguments: argparse.Namespace) -> None:
    """One allow-listed source as the unit reads it, next to PID 1's view."""

    run_id = safe_id(arguments.run_id)
    path = checked_source_path(arguments.path)
    pid = unit_state(HOST_UNIT).get("MainPID", "0")
    if not pid.isdigit() or pid == "0":
        raise ProbeError(f"{HOST_UNIT} has no main process to read through")
    unit, host = view_text(pid, path), view_text("1", path)
    emit(
        {
            "run_id": run_id,
            "path": path,
            "main_pid": pid,
            "unit": unit,
            "host": host,
            "same": bool(
                unit["readable"]
                and host["readable"]
                and unit["value"] == host["value"]
                and unit["device"] == host["device"]
            ),
            "window_open": bool(load_state(run_id)),
        }
    )


def set_value(arguments: argparse.Namespace) -> None:
    """Rewrite a fixture's backing file in place (same inode: a bind follows it)."""

    run_id = safe_id(arguments.run_id)
    name = checked_name(arguments.name)
    target = fixtures_dir(run_id) / name
    relative = getattr(arguments, "path", None)
    if relative:
        if not target.is_dir():
            raise ProbeError("fixture tree is not built")
        candidate = (target / relative).resolve()
        if target.resolve() not in candidate.parents:
            raise ProbeError("fixture path escapes the tree")
        target = candidate
    if not target.is_file():
        raise ProbeError("fixture is not installed")
    with open(target, "r+", encoding="utf-8") as handle:
        handle.seek(0)
        handle.truncate()
        handle.write(str(arguments.value))
    emit({"run_id": run_id, "name": name, "path": relative, "value": arguments.value})


# --------------------------------------------------------------------------- #
# command wrappers
# --------------------------------------------------------------------------- #
def load_wrapper_module(run_id: str) -> Any:
    """The installed, sha256-checked wrapper as a module (for its plan validator)."""

    path = fixtures_dir(run_id) / WRAPPER_NAME
    if not path.is_file():
        raise ProbeError(f"install {WRAPPER_NAME} before wrapping a tool")
    spec = importlib.util.spec_from_file_location("gfrs_host_command_replay", path)
    if spec is None or spec.loader is None:
        raise ProbeError("the installed wrapper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrappers_path(run_id: str) -> Path:
    return state_dir(run_id) / "wrappers.json"


def load_wrappers(run_id: str) -> dict[str, Any]:
    path = wrappers_path(run_id)
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def install_wrapper(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    tool = arguments.tool
    module = load_wrapper_module(run_id)
    try:
        module.checked_tool(tool)
        module.checked_unit(HOST_UNIT)
    except ValueError as error:
        raise ProbeError(str(error)) from error
    plan_file = fixtures_dir(run_id) / f"plan-{tool}.json"
    if not plan_file.is_file():
        raise ProbeError(f"install plan-{tool}.json before wrapping {tool}")
    try:
        plan = module.validate_plan(
            tool, json.loads(plan_file.read_text(encoding="utf-8"))
        )
    except ValueError as error:
        raise ProbeError(f"plan-{tool}.json rejected: {error}") from error
    real = shutil.which(tool, path=STANDARD_PATH)
    if real is None and not arguments.allow_missing_real:
        raise ProbeError(
            f"{tool} is not installed on the host; pass --allow-missing-real to "
            "replay it anyway"
        )
    directory = bin_dir(run_id)
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    shim = directory / tool
    script = (
        "#!/bin/sh\n"
        f"{module.TOOL_ENV}={tool} {module.DIR_ENV}={state_dir(run_id)} "
        f'exec {sys.executable} -I {fixtures_dir(run_id) / WRAPPER_NAME} "$@"\n'
    )
    write_atomic(shim, script.encode("utf-8"), 0o755)
    audit_dir(run_id).mkdir(mode=0o755, parents=True, exist_ok=True)
    wrappers = load_wrappers(run_id)
    wrappers[tool] = {
        "real": real,
        "shim": str(shim),
        "plan_sha256": hashlib.sha256(plan_file.read_bytes()).hexdigest(),
        "rules": len(plan["rules"]),
        "not_after": plan["not_after"],
    }
    wrappers_path(run_id).write_text(
        json.dumps(wrappers, sort_keys=True), encoding="utf-8"
    )
    emit({"run_id": run_id, "tool": tool, **wrappers[tool]})


def audit_read(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    tool = arguments.tool
    if tool not in WRAPPABLE_TOOLS:
        raise ProbeError(f"{tool!r} is not an allow-listed tool")
    path = audit_dir(run_id) / f"{tool}.jsonl"
    entries: list[Any] = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = max(1, min(int(arguments.tail), MAX_AUDIT_LINES))
        for line in lines[-tail:]:
            try:
                entries.append(json.loads(line))
            except ValueError:
                entries.append({"unparsable": line[:200]})
    calls_path = audit_dir(run_id) / "calls.json"
    calls = (
        json.loads(calls_path.read_text(encoding="utf-8") or "{}")
        if calls_path.is_file()
        else {}
    )
    emit({"run_id": run_id, "tool": tool, "entries": entries, "calls": calls})


# --------------------------------------------------------------------------- #
# deadman
# --------------------------------------------------------------------------- #
def arm_deadman(run_id: str, seconds: int) -> str:
    name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={name}",
            f"--on-active={int(seconds)}s",
            "--timer-property=AccuracySec=1s",
            "--property=Restart=on-failure",
            "--property=RestartSec=30s",
            "--property=StartLimitIntervalSec=0",
            sys.executable,
            str(probe_copy(run_id)),
            "close-window",
            "--run-id",
            run_id,
        ],
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
# verification: visible inside the unit, invisible to PID 1
# --------------------------------------------------------------------------- #
def nsenter(pid: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    return run(["nsenter", "-t", pid, "-m", "--", *command], check=False, timeout=30)


def view_path(pid: str, path: str) -> dict[str, Any]:
    """What ``pid``'s mount namespace sees at ``path``: readable, sha256, device."""

    read = nsenter(pid, ["/usr/bin/cat", path])
    if read.returncode != 0:
        # Byte order like ``sorted(os.listdir())`` in ``local_view``.
        listing = nsenter(pid, ["/usr/bin/env", "LC_ALL=C", "/usr/bin/ls", "-1", path])
        readable = listing.returncode == 0
        content = listing.stdout if readable else ""
        kind = "dir" if readable else "unreadable"
    else:
        readable, content, kind = True, read.stdout, "file"
    stat = nsenter(pid, ["/usr/bin/stat", "-L", "-c", "%d", path])
    return {
        "readable": readable,
        "kind": kind,
        "sha256": hashlib.sha256(content.encode("utf-8", "replace")).hexdigest(),
        "device": stat.stdout.strip() if stat.returncode == 0 else None,
    }


def local_view(path: Path) -> dict[str, Any]:
    if path.is_dir():
        content = "".join(f"{name}\n" for name in sorted(os.listdir(path)))
    else:
        content = path.read_text(encoding="utf-8", errors="replace")
    return {
        "sha256": hashlib.sha256(content.encode("utf-8", "replace")).hexdigest(),
        "device": str(os.stat(path).st_dev),
    }


def verify_window(run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    pid = unit_state(HOST_UNIT).get("MainPID", "0")
    if not pid.isdigit() or pid == "0":
        raise ProbeError(f"{HOST_UNIT} has no main process to verify against")
    checks: list[dict[str, Any]] = []
    bound = [("bind", item) for item in state.get("binds", [])] + [
        ("tree", item) for item in state.get("trees", [])
    ]
    for kind, bind in bound:
        fixture = local_view(Path(bind["source"]))
        unit = view_path(pid, bind["dest"])
        host = view_path("1", bind["dest"])
        checks.append(
            {
                "kind": kind,
                "dest": bind["dest"],
                "unit_sees_fixture": unit["readable"]
                and unit["sha256"] == fixture["sha256"],
                "unit_on_fixture_device": unit["device"] == fixture["device"],
                "host_sees_fixture": host["sha256"] == fixture["sha256"],
                "host_on_fixture_device": host["device"] == fixture["device"],
                "ok": unit["readable"]
                and unit["sha256"] == fixture["sha256"]
                and host["device"] != unit["device"],
            }
        )
    for path in state.get("inaccessible", []):
        unit = view_path(pid, path)
        host = view_path("1", path)
        checks.append(
            {
                "kind": "inaccessible",
                "dest": path,
                "unit_readable": unit["readable"],
                "host_readable": host["readable"],
                "ok": not unit["readable"] and host["readable"],
            }
        )
    keys = ("PATH", PCI_ENV, *ENV_OVERRIDES)
    environment = process_environment(pid, keys)
    for tool, record in state.get("wrappers", {}).items():
        shim = record["shim"]
        in_unit = nsenter(pid, ["/usr/bin/test", "-x", shim]).returncode == 0
        on_path = environment.get("PATH", "").split(":")[:1] == [str(bin_dir(run_id))]
        host_real = shutil.which(tool, path=STANDARD_PATH)
        checks.append(
            {
                "kind": "wrapper",
                "dest": tool,
                "shim_in_unit": in_unit,
                "unit_path_first": on_path,
                "host_resolves_to": host_real,
                "ok": in_unit and on_path and host_real != shim,
            }
        )
    for name, value in state.get("env", {}).items():
        checks.append(
            {
                "kind": "env",
                "dest": name,
                "unit_value": environment.get(name),
                "ok": environment.get(name) == value,
            }
        )
    if state.get("mode") == "pci-root" or state.get("pci_root"):
        root = str(fixtures_dir(run_id) / "pci")
        checks.append(
            {
                "kind": "pci-root",
                "dest": PCI_ENV,
                "unit_value": environment.get(PCI_ENV),
                "ok": environment.get(PCI_ENV) == root,
            }
        )
    return {"main_pid": pid, "checks": checks, "ok": all(item["ok"] for item in checks)}


# --------------------------------------------------------------------------- #
# window
# --------------------------------------------------------------------------- #
def preflight(arguments: argparse.Namespace) -> None:
    emit(
        {
            "captured_at": utc_now(),
            "unit": unit_state(HOST_UNIT),
            "tools": {tool: tool_locations(tool) for tool in WRAPPABLE_TOOLS},
            "nsenter": shutil.which("nsenter", path=STANDARD_PATH) is not None,
            "infiniband_devices": sorted(
                path.name for path in Path("/sys/class/infiniband").glob("*")
            )
            if Path("/sys/class/infiniband").is_dir()
            else [],
        }
    )


def dropin_lines(
    run_id: str, arguments: argparse.Namespace, state: dict[str, Any]
) -> list[str]:
    """The drop-in body; records every replacement in ``state`` as it goes."""

    lines = ["[Service]"]
    for item in arguments.bind:
        src, _, dest = item.partition(":")
        dest = checked_bind_dest(dest)
        source = fixtures_dir(run_id) / checked_name(Path(src).name)
        if not source.is_file():
            raise ProbeError(f"bind source {source} is not installed")
        lines.append(f"BindReadOnlyPaths={source}:{dest}")
        state["binds"].append({"source": str(source), "dest": dest})
    for item in arguments.tree:
        src, _, dest = item.partition(":")
        dest = checked_tree_dest(dest)
        source = fixtures_dir(run_id) / checked_name(Path(src).name)
        if not source.is_dir():
            raise ProbeError(f"tree source {source} is not built")
        lines.append(f"BindReadOnlyPaths={source}:{dest}")
        state["trees"].append({"source": str(source), "dest": dest})
    if arguments.wrapper:
        wrappers = load_wrappers(run_id)
        for tool in arguments.wrapper:
            if tool not in wrappers or not Path(wrappers[tool]["shim"]).is_file():
                raise ProbeError(
                    f"install-wrapper {tool} before opening a command window"
                )
            state["wrappers"][tool] = wrappers[tool]
        audit_dir(run_id).mkdir(mode=0o755, parents=True, exist_ok=True)
        lines.append(f"Environment=PATH={bin_dir(run_id)}:{STANDARD_PATH}")
        # The wrapper journals and counts calls here; /run is read-only for the
        # unit under ProtectSystem=strict without this.
        lines.append(f"ReadWritePaths={audit_dir(run_id)}")
    for item in arguments.inaccessible:
        for path in checked_inaccessible(item):
            lines.append(f"InaccessiblePaths=-{path}")
            state["inaccessible"].append(path)
    for pair in arguments.env:
        name, value = checked_env_override(pair)
        lines.append(f"Environment={name}={value}")
        state["env"][name] = value
    if arguments.pci_root:
        root = fixtures_dir(run_id) / "pci"
        if not root.is_dir():
            raise ProbeError("build the PCI fixture before opening a pci-root window")
        lines.append(f"Environment={PCI_ENV}={root}")
        state["pci_root"] = True
    if len(lines) == 1:
        raise ProbeError("a window must bind, wrap, hide or override something")
    return lines


def checked_mode(arguments: argparse.Namespace) -> str:
    mode = arguments.mode
    if mode == "bind" and not arguments.bind:
        raise ProbeError("bind mode needs at least one --bind SRC:DEST")
    if mode == "pci-root":
        arguments.pci_root = True
    if mode == "command" and not arguments.wrapper:
        raise ProbeError("command mode needs at least one --wrapper TOOL")
    if mode == "tree" and not arguments.tree:
        raise ProbeError("tree mode needs at least one --tree SRC:DEST")
    return mode


def open_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    mode = checked_mode(arguments)
    restore_seconds = checked_restore_seconds(arguments.restore_seconds)
    if load_state(run_id):
        raise ProbeError("a source replay window for this run is already open")
    dropin = dropin_path(run_id)
    if dropin.exists():
        raise ProbeError("an untracked source replay drop-in already exists")
    before = unit_state(HOST_UNIT)
    if before.get("ActiveState") != "active":
        raise ProbeError(f"{HOST_UNIT} is not active at baseline")
    state: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "unit": checked_unit(HOST_UNIT),
        "dropin": str(dropin),
        "binds": [],
        "trees": [],
        "wrappers": {},
        "inaccessible": [],
        "env": {},
        "pci_root": False,
        "before": before,
        "opened_at": utc_now(),
        "restore_seconds": restore_seconds,
    }
    lines = dropin_lines(run_id, arguments, state)
    probe_copy(run_id)
    save_state(run_id, state)
    # Deadman first: from here on the baseline is restored even if this process,
    # the probe Pod and the control plane all vanish.
    deadman = arm_deadman(run_id, restore_seconds)
    dropin.parent.mkdir(parents=True, exist_ok=True)
    dropin.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", HOST_UNIT], timeout=120)
    after = unit_state(HOST_UNIT)
    if after.get("ActiveState") != "active":
        _close(run_id, state)
        raise ProbeError(f"{HOST_UNIT} is not active after the window opened; closed")
    verification: dict[str, Any] = {"skipped": True}
    if not arguments.skip_verify:
        try:
            verification = verify_window(run_id, state)
        except ProbeError as error:
            _close(run_id, state)
            raise ProbeError(f"window verification failed: {error}; closed") from error
        if not verification["ok"]:
            _close(run_id, state)
            raise ProbeError(
                "the replacement is not visible only inside the unit; closed: "
                + json.dumps(verification["checks"], sort_keys=True)[:600]
            )
    emit(
        {
            **state,
            "after": after,
            "deadman_timer": unit_state(f"{deadman}.timer"),
            "verification": verification,
            "proof_scope": "live-replay",
        }
    )


def apply_value(arguments: argparse.Namespace) -> None:
    """Rewrite a bound fixture value; the unit re-reads the file each poll."""

    run_id = safe_id(arguments.run_id)
    if not load_state(run_id):
        raise ProbeError("no source replay window is open for this run")
    set_value(arguments)
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
                fixtures[item.name] = item.read_text(
                    encoding="utf-8", errors="replace"
                )[:64]
            else:
                fixtures[item.name] = f"<tree: {len(os.listdir(item))} entries>"
    audits = {}
    if audit_dir(run_id).is_dir():
        for item in sorted(audit_dir(run_id).glob("*.jsonl")):
            audits[item.stem] = sum(1 for _ in item.open(encoding="utf-8"))
    emit(
        {
            "run_id": run_id,
            "mode": state["mode"],
            "unit": unit_state(HOST_UNIT),
            "fixtures": fixtures,
            "wrappers": sorted(state.get("wrappers", {})),
            "audit_lines": audits,
        }
    )


def verify(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    if not state:
        raise ProbeError("no source replay window is open for this run")
    emit({"run_id": run_id, **verify_window(run_id, state)})


def _close(run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    dropin = Path(state.get("dropin") or dropin_path(run_id))
    dropin.unlink(missing_ok=True)
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", HOST_UNIT], timeout=120)
    after = unit_state(HOST_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{HOST_UNIT} is not active after the window closed")
    disarm_deadman(run_id)
    residue = {
        "dropin": dropin.exists(),
        "deadman": unit_state(f"{deadman_unit(run_id)}.timer").get("ActiveState")
        == "active",
    }
    if state_dir(run_id).is_dir():
        shutil.rmtree(state_dir(run_id))
    residue["state_dir"] = state_dir(run_id).exists()
    return {"after": after, "residue": residue, "clean": not any(residue.values())}


def close_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    if not state:
        dropin = dropin_path(run_id)
        if dropin.exists():
            # A crash between the drop-in and the state file: still restore.
            emit({"run_id": run_id, "untracked_dropin": True, **_close(run_id, {})})
            return
        if state_dir(run_id).is_dir():
            shutil.rmtree(state_dir(run_id))
        emit({"run_id": run_id, "already_closed": True, "clean": True})
        return
    emit({"run_id": run_id, **_close(run_id, state)})


def snapshot(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    emit(
        {
            "run_id": run_id,
            "window_open": bool(state),
            "dropin": dropin_path(run_id).exists(),
            "state_dir": state_dir(run_id).exists(),
            "deadman": unit_state(f"{deadman_unit(run_id)}.timer").get("ActiveState")
            == "active",
        }
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    def command(name: str, handler: Any) -> argparse.ArgumentParser:
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
        item.set_defaults(handler=handler)
        return item

    commands.add_parser("preflight").set_defaults(handler=preflight)

    item = command("install-source", install_source)
    item.add_argument("--name", required=True)
    item.add_argument("--sha256", required=True)
    item.add_argument("--b64", required=True)

    item = command("build-pci-fixture", build_pci_fixture)
    item.add_argument("--efa-count", type=int, required=True)
    item.add_argument("--malformed", action="store_true")

    item = command("build-infiniband-fixture", build_infiniband_fixture)
    item.add_argument("--source", default="/sys/class/infiniband")
    item.add_argument("--drop", action="append", default=[])
    item.add_argument("--duplicate", action="append", default=[])
    item.add_argument("--port-state", action="append", default=[])
    item.add_argument("--phys-state", action="append", default=[])
    item.add_argument("--driver", action="append", default=[])
    item.add_argument("--set", action="append", default=[])
    item.add_argument("--remove-counter", action="append", default=[])
    item.add_argument("--no-ports", action="append", default=[])
    item.add_argument("--netdev", action="append", default=[])
    item.add_argument("--netdev-root", default=NETDEV_ROOT)
    item.add_argument("--allow-empty", action="store_true")

    item = command("inventory", inventory)
    item.add_argument("--source", default="/sys/class/infiniband")
    item.add_argument("--netdev-root", default=NETDEV_ROOT)

    item = command("read-source", read_source)
    item.add_argument("--path", required=True)

    item = command("build-nvidia-proc-fixture", build_nvidia_proc_fixture)
    item.add_argument("--source", default="/proc/driver/nvidia")
    item.add_argument("--omit-version", action="store_true")
    item.add_argument("--version-text")
    item.add_argument("--gpu-count", type=int)
    item.add_argument("--omit-gpus", action="store_true")

    item = command("install-wrapper", install_wrapper)
    item.add_argument("--tool", required=True)
    item.add_argument("--allow-missing-real", action="store_true")

    item = command("open-window", open_window)
    item.add_argument(
        "--mode",
        choices=("bind", "pci-root", "command", "tree", "mixed"),
        required=True,
    )
    item.add_argument("--bind", action="append", default=[])
    item.add_argument("--tree", action="append", default=[])
    item.add_argument("--wrapper", action="append", default=[])
    item.add_argument("--inaccessible", action="append", default=[])
    item.add_argument("--env", action="append", default=[])
    item.add_argument("--pci-root", action="store_true")
    item.add_argument("--restore-seconds", type=int, default=900)
    item.add_argument("--skip-verify", action="store_true")

    item = command("set-value", apply_value)
    item.add_argument("--name", required=True)
    item.add_argument("--path")
    item.add_argument("--value", required=True)

    item = command("audit-read", audit_read)
    item.add_argument("--tool", required=True)
    item.add_argument("--tail", type=int, default=100)

    command("status", status)
    command("verify", verify)
    command("close-window", close_window)
    command("snapshot", snapshot)
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        emit(
            {
                "error": type(exc).__name__,
                "error_kind": "probe",
                "command": getattr(arguments, "command", None),
            }
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
