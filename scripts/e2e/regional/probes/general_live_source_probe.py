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
  the real tree and apply reviewed mutations; ``build-meminfo-fixture`` copies
  the real ``/proc/meminfo`` and rewrites only the allow-listed memory/swap
  fields (002 ``swap_pressure``) for the ``meminfo`` bind. The builders are
  the second node-side file, ``general_live_source_builders.py``: the
  controller installs it through ``install-source`` (sha256-checked, like the
  command wrapper) before the first build and the probe imports that copy.
* **command** -- a reviewed pass-through wrapper (``tools/host_command_replay.py``,
  installed with its sha256) first on the unit's PATH for ``smartctl``,
  ``ipmitool``, ``ethtool`` or the host collector's ``nvidia-smi`` queries. It
  forwards to the real binary and rewrites only the plan's allow-listed fields;
  no host binary is replaced. ``py-spy`` is refused (Node Agent only). With
  ``--unit gpu-fault-kernel-collector.service`` (the runner's
  ``KERNEL_COLLECTOR_SOURCE_REPLAY`` approval) the same mode -- and only this
  mode -- opens on the kernel collector for its ``nvidia-smi`` driver-version
  discovery query (``UNIT_WRAPPERS``): the unit restarts into the window, its
  ``CollectorContext`` asks the wrapper for the version, and
  ``GPU_FAULT_DRIVER_BRANCH`` is unset in that unit's environment so no
  configured branch stands in (GF-GENERAL-LIVE-032 ``missing_version``).
* **inaccessible** -- an allow-listed procfs *file* or tool binary made
  unopenable inside the unit only (002 read_failure, 007 tool_missing): the
  run's ``unreadable`` node -- a driverless 0:0 character device, mode 0000,
  under the state dir -- is ``BindReadOnlyPaths=`` over it. Not systemd's
  ``InaccessiblePaths=``: the collector units run as root, and root's
  CAP_DAC_OVERRIDE reads the mode-0000 regular node systemd binds there as an
  *empty file* (CAP_DAC_READ_SEARCH lists its directory node as an *empty
  directory*) -- a healthy-looking empty reading, never a read failure, which
  the window verification refused on the first live 002 run. For the same
  reason no directory can be made unreadable for these units: a directory the
  product must fail to read is replayed as a tree without the entry
  (``build-nvidia-proc-fixture --omit-gpus`` for 012 proc_unreadable, so the
  listing of ``gpus`` fails with ENOENT on the unit's own copy).
* **env** -- an allow-listed expected-count override (012 expectation_wrong),
  written to the run's ``env`` file and handed to the unit as
  ``EnvironmentFile=``: the installer puts the real counts into the unit's own
  ``EnvironmentFile=/etc/gpu-fault/collector.env`` and systemd applies every
  EnvironmentFile= after every Environment=, later files winning
  (systemd.exec(5)); a drop-in ``Environment=`` override never reached the
  process (live 2026-10-26). ``UnsetEnvironment=`` is applied last of all, so
  an unset variable is not resurrected by any file.
* **pci-root** -- ``Environment=GPU_FAULT_PCI_DEVICES_ROOT=<fixture tree>`` (013;
  not a key of the installer's environment file).

``open-window`` arms the deadman *before* writing the drop-in, restarts the one
unit and then **verifies** through ``nsenter`` that every replacement is visible
in the unit's mount namespace / process environment and *not* in PID 1's.
``close-window`` (and the deadman) removes the drop-in, reloads until systemd no
longer lists it (``DropInPaths``), restarts the unit and reads the real
configuration back (``restore``: every source as PID 1 sees it, every touched
variable at its recorded baseline). ``await-ticks`` waits on the node for the
collector's own ticks (``WatchdogTimestampMonotonic``) and closes after the
required count, so a bounded exposure follows the ticks the collector took,
not the restart. The Node Agent is never touched; no global mount is made and
no real ``/proc``/sysfs file is written. ``proof_scope`` is always ``live-replay``.

Three read-only commands serve the variants' oracle: ``inventory`` (in the
builders file) lists the node's RDMA tree as the product reads it (driver
identity, owned netdevs, port ``state``/``phys_state``, every ``counters``/
``hw_counters`` file with its value, the names ``ethtool -S`` prints per
netdev), ``read-source --path`` shows one allow-listed source file as the unit
sees it next to what PID 1 sees, so a closed window can prove the real value is
read again, and ``journal-read`` returns the tail of an allow-listed unit's own
journal since a moment (what the collector itself logged about its discovery).
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
import stat
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

STATE_ROOT = Path("/run/gfrs-live")
DROPIN_ROOT = Path("/run/systemd/system")
HOST_UNIT = "gpu-fault-host-collector.service"
KERNEL_UNIT = "gpu-fault-kernel-collector.service"
NODE_AGENT_UNIT = "gpu-fault-node-agent.service"
#: The units a window may touch. The Node Agent is absent on purpose. The host
#: collector carries every mode; the kernel collector carries a *command*
#: window for the queries ``UNIT_WRAPPERS`` lists and nothing else.
ALLOWED_UNITS = (HOST_UNIT, KERNEL_UNIT)
UNIT_WRAPPERS: dict[str, dict[str, tuple[str, ...]]] = {
    KERNEL_UNIT: {"nvidia-smi": ("driver_version",)},
}
#: The kernel collector reads a configured driver branch before it asks
#: ``nvidia-smi`` (``collectors/context.py``; the node installer writes the
#: variable empty by default). It is unset in that unit's environment for the
#: window -- ``UnsetEnvironment=`` applies after ``EnvironmentFile=`` -- so the
#: wrapped query is the only version evidence the unit has.
DRIVER_BRANCH_ENV = "GPU_FAULT_DRIVER_BRANCH"
UNSET_ENV_BY_UNIT: dict[str, tuple[str, ...]] = {KERNEL_UNIT: (DRIVER_BRANCH_ENV,)}
#: ``journal-read`` bounds.
MAX_JOURNAL_LINES = 400
MAX_JOURNAL_LINE_CHARS = 400
PCI_ENV = "GPU_FAULT_PCI_DEVICES_ROOT"
WRAPPER_NAME = "host_command_replay.py"
#: The fixture builders, installed through ``install-source`` like the wrapper.
BUILDERS_NAME = "general_live_source_builders.py"
#: ``sys.modules`` name this probe registers itself under for the builders.
PROBE_ALIAS = "gfrs_general_live_source_probe"
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
#: Files a window may make unopenable inside the unit (plus allow-listed
#: tools). Regular files only -- see the module docstring: for a root unit a
#: directory can only be emptied, never made unreadable, and an empty
#: ``gpus`` or ``infiniband`` is a zero-device reading the product must never
#: be fed by a read-failure variant.
INACCESSIBLE_PATHS = ("/proc/meminfo", "/proc/driver/nvidia/version")
#: The node bound over an inaccessible file: a character device with major 0
#: (no driver behind it: ENXIO) on the state dir's ``nodev`` tmpfs (EACCES),
#: mode 0000 -- unopenable for every uid, like the ``chr`` node systemd itself
#: keeps in /run/systemd/inaccessible for an inaccessible device file.
UNREADABLE_NODE = "unreadable"
WRAPPABLE_TOOLS = ("smartctl", "ipmitool", "ethtool", "nvidia-smi")
#: Environment overrides and their value pattern (empty = unset the expectation).
ENV_OVERRIDES = {
    "GPU_FAULT_EXPECTED_GPU_COUNT": re.compile(r"^([0-9]|[1-5][0-9]|6[0-4])?$"),
    "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT": re.compile(r"^([0-9]|[1-5][0-9]|6[0-4])?$"),
}
EFA_VENDOR = "0x1d0f"
#: ``/proc/meminfo`` fields a meminfo fixture may rewrite: the inputs of the
#: host collector's memory and swap rules (``system_metrics._memory``). Values
#: are kB like the file itself.
MEMINFO_FIELDS = ("MemFree", "MemAvailable", "SwapTotal", "SwapFree")
MEMINFO_FIXTURE = "meminfo"
MAX_MEMINFO_KB = 1 << 50
WINDOW_MODES = ("bind", "pci-root", "command", "tree", "mixed")
#: The deadman's bounds. A bounded exposure (a replayed inactive port confirmed
#: on the collector's required ticks and never read on one more) is closed by
#: ``await-ticks`` from the ticks the collector actually took; its deadman is
#: the fallback (``general_live_source_replay.exposure_fallback_seconds``).
MIN_RESTORE_SECONDS = 20
MAX_RESTORE_SECONDS = 5400
#: The run's ``EnvironmentFile=`` for *env* overrides (see ``dropin_lines``).
ENV_FILE = "env"
#: ``await-ticks``: the host collector pings ``WATCHDOG=1`` after every
#: contributor and once more when a tick's batch is posted (``collectors/host/
#: collector.py`` ``_watchdog_heartbeat`` / ``collect_once``), and systemd
#: records the last ping in the unit's ``WatchdogTimestampMonotonic``. Pings
#: closer than ``TICK_GAP_RATIO`` of the interval (at least
#: ``MIN_TICK_GAP_SECONDS``) belong to one tick; the tick has ended once that
#: gap has passed in silence. The sleep between ticks is the whole interval.
TICK_POLL_SECONDS = 0.5
TICK_GAP_RATIO = 0.4
MIN_TICK_GAP_SECONDS = 5.0
MAX_TICKS = 16
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
    """A refused or failed probe step. ``detail`` is a JSON object the error line
    carries next to the message (the window verification's checks), so the
    controller can keep it as redacted case evidence instead of a bare digest."""

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail


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


def dropin_path(run_id: str, unit: str = HOST_UNIT) -> Path:
    return DROPIN_ROOT / f"{unit}.d" / f"gpu-fault-source-replay-{digest(run_id)}.conf"


def deadman_unit(run_id: str) -> str:
    return f"gpu-fault-source-replay-{digest(run_id)}"


def checked_unit(unit: str) -> str:
    if unit == NODE_AGENT_UNIT:
        raise ProbeError("the Node Agent unit is never in a replay allow-list")
    if unit not in ALLOWED_UNITS:
        raise ProbeError(f"{unit!r} is not an allow-listed collector unit")
    return unit


def checked_unit_wrapper(unit: str, tool: str) -> str:
    """``tool`` may be wrapped on ``unit`` (the kernel collector: nvidia-smi only)."""

    restricted = UNIT_WRAPPERS.get(checked_unit(unit))
    if restricted is not None and tool not in restricted:
        raise ProbeError(
            f"{unit} carries a wrapper for {sorted(restricted)} only, not {tool!r}"
        )
    return tool


def checked_moment(text: str) -> str:
    """An ISO-8601 moment as ``journalctl --since`` reads it (UTC)."""

    try:
        moment = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError as error:
        raise ProbeError(f"not an ISO-8601 moment: {text!r}") from error
    if moment.tzinfo is None:
        raise ProbeError("a journal moment needs a timezone")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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
    """The files the unreadable node is bound over for an allow-listed path or
    ``tool:<name>`` (every copy of the tool on the standard PATH)."""

    if item.startswith("tool:"):
        paths = tool_locations(item[len("tool:") :])
        if not paths:
            raise ProbeError(
                f"{item[5:]} is not installed on the host; nothing to hide"
            )
        return paths
    if item in INACCESSIBLE_PATHS:
        return [item]
    if any(item == dest or item.startswith(dest + "/") for dest in TREE_DESTS):
        raise ProbeError(
            f"{item!r} is a directory: a root unit lists an InaccessiblePaths= "
            "directory as empty (a zero-device reading, not a read failure); "
            "replay a tree fixture without the entry instead"
        )
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
    properties = ("LoadState", "ActiveState", "SubState", "MainPID", "NRestarts")
    completed = run(
        ["systemctl", "show", unit, *(f"--property={key}" for key in properties)],
        check=False,
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def unit_dropins(unit: str) -> dict[str, Any]:
    """The drop-ins systemd has *loaded* for ``unit`` and whether a reload is due.

    A removed drop-in stays in ``DropInPaths`` (``NeedDaemonReload=yes``) until
    ``daemon-reload`` (systemd 255, proven 2026-10-26); the live 013 window left
    exactly that listing behind, so ``_close`` reads this back after its reload.
    """

    completed = run(
        [
            "systemctl",
            "show",
            unit,
            "--property=DropInPaths",
            "--property=NeedDaemonReload",
        ],
        check=False,
    )
    values = {"paths": [], "reload_needed": False}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key == "DropInPaths":
            values["paths"] = value.split()
        elif separator and key == "NeedDaemonReload":
            values["reload_needed"] = value.strip() == "yes"
    return values


def unit_stamps(unit: str) -> dict[str, int]:
    """``ExecMainStartTimestampMonotonic`` / ``WatchdogTimestampMonotonic`` (us)."""

    completed = run(
        [
            "systemctl",
            "show",
            unit,
            "--property=ExecMainStartTimestampMonotonic",
            "--property=WatchdogTimestampMonotonic",
        ],
        check=False,
    )
    values: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and value.strip().isdigit():
            values[key] = int(value)
    return values


def monotonic_us() -> int:
    """CLOCK_MONOTONIC in microseconds, the clock systemd stamps with."""

    return time.monotonic_ns() // 1000


def pause(seconds: float) -> None:
    time.sleep(seconds)


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


def env_file(run_id: str) -> Path:
    return state_dir(run_id) / ENV_FILE


def write_atomic(path: Path, content: bytes, mode: int) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.chmod(mode)
    os.replace(temporary, path)


def unreadable_node_path(run_id: str) -> Path:
    return state_dir(run_id) / UNREADABLE_NODE


def make_unopenable_node(path: Path) -> None:
    """``mknod`` the driverless 0:0 character device nobody can open (CAP_MKNOD)."""

    os.mknod(path, stat.S_IFCHR | 0o000, os.makedev(0, 0))


def unreadable_node(run_id: str) -> Path:
    """This run's unopenable node, created on first use (removed with the state dir)."""

    path = unreadable_node_path(run_id)
    if not path.exists():
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        make_unopenable_node(path)
    return path


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


# --------------------------------------------------------------------------- #
# fixture builders (probes/general_live_source_builders.py, shipped like the wrapper)
# --------------------------------------------------------------------------- #
def load_builders_module(run_id: str) -> Any:
    """The fixture builders: the installed, sha256-checked copy, else the checkout's.

    On the node only the copy the controller installed through
    ``install-source`` exists (a probe is one file; see the module docstring);
    the probe registers itself under ``PROBE_ALIAS`` first so the builders find
    the allow-lists. In a checkout (tests, the controller) the sibling module
    answers when nothing is installed.
    """

    path = fixtures_dir(run_id) / BUILDERS_NAME
    if not path.is_file():
        try:
            return importlib.import_module(
                "scripts.e2e.regional.probes.general_live_source_builders"
            )
        except ImportError:
            raise ProbeError(
                f"install {BUILDERS_NAME} before building a fixture"
            ) from None
    sys.modules.setdefault(PROBE_ALIAS, sys.modules[__name__])
    spec = importlib.util.spec_from_file_location(
        "gfrs_general_live_source_builders", path
    )
    if spec is None or spec.loader is None:
        raise ProbeError("the installed fixture builders cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _builder(name: str) -> Any:
    """The ``build-*`` handler ``name`` of the builders module, loaded per call."""

    def handler(arguments: argparse.Namespace) -> None:
        getattr(load_builders_module(safe_id(arguments.run_id)), name)(arguments)

    return handler


def read_source(arguments: argparse.Namespace) -> None:
    """One allow-listed source as the unit reads it, next to PID 1's view."""

    run_id = safe_id(arguments.run_id)
    path = checked_source_path(arguments.path)
    pid = unit_state(HOST_UNIT).get("MainPID", "0")
    if not pid.isdigit() or pid == "0":
        raise ProbeError(f"{HOST_UNIT} has no main process to read through")
    unit, host = view_path(pid, path), view_path("1", path)
    same = unit["readable"] and host["readable"] and unit["sha256"] == host["sha256"]
    emit(
        {
            "run_id": run_id,
            "path": path,
            "main_pid": pid,
            "unit": unit,
            "host": host,
            "same": bool(same and unit["device"] == host["device"]),
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
    unit = checked_unit(arguments.unit)
    module = load_wrapper_module(run_id)
    try:
        module.checked_tool(tool)
        module.checked_unit(unit)
    except ValueError as error:
        raise ProbeError(str(error)) from error
    checked_unit_wrapper(unit, tool)
    plan_file = fixtures_dir(run_id) / f"plan-{tool}.json"
    if not plan_file.is_file():
        raise ProbeError(f"install plan-{tool}.json before wrapping {tool}")
    # The installed wrapper validates the plan against its own per-tool and
    # per-unit allow-lists: the kernel collector takes driver_version rules only.
    checker = getattr(module, "checked_unit_plan", None)
    if checker is None:
        raise ProbeError("the installed wrapper does not know per-unit plans")
    try:
        plan = checker(unit, tool, json.loads(plan_file.read_text(encoding="utf-8")))
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
        "unit": unit,
        "plan_sha256": hashlib.sha256(plan_file.read_bytes()).hexdigest(),
        "rules": len(plan["rules"]),
        "not_after": plan["not_after"],
    }
    wrappers_path(run_id).write_text(
        json.dumps(wrappers, sort_keys=True), encoding="utf-8"
    )
    emit({"run_id": run_id, "tool": tool, **wrappers[tool]})


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
    """What ``pid``'s mount namespace sees at ``path``: readable, sha256, device, text."""

    read = nsenter(pid, ["/usr/bin/cat", path])
    if read.returncode == 0:
        readable, content, kind = True, read.stdout, "file"
    elif nsenter(pid, ["/usr/bin/test", "-d", path]).returncode == 0:
        # Byte order like ``sorted(os.listdir())`` in ``local_view``. Only a
        # directory is judged by its listing: ``ls`` of an unopenable *file*
        # (the unreadable node bound over it) succeeds on the name alone and
        # would read as accessible (002 ``read_failure``).
        listing = nsenter(pid, ["/usr/bin/env", "LC_ALL=C", "/usr/bin/ls", "-1", path])
        readable = listing.returncode == 0
        content = listing.stdout if readable else ""
        kind = "dir" if readable else "unreadable"
    else:
        readable, content, kind = False, "", "unreadable"
    stat = nsenter(pid, ["/usr/bin/stat", "-L", "-c", "%d", path])
    return {
        "readable": readable,
        "kind": kind,
        "sha256": hashlib.sha256(content.encode("utf-8", "replace")).hexdigest(),
        "device": stat.stdout.strip() if stat.returncode == 0 else None,
        "value": content.strip()[:MAX_SOURCE_VALUE_CHARS] if kind == "file" else None,
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


def window_unit(state: dict[str, Any]) -> str:
    """The unit a window's state names (phase-1 states name none: the host)."""

    return str(state.get("unit") or HOST_UNIT)


def verify_window(run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    unit = window_unit(state)
    pid = unit_state(unit).get("MainPID", "0")
    if not pid.isdigit() or pid == "0":
        raise ProbeError(f"{unit} has no main process to verify against")
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
        # The unit's open() must fail (the unreadable node answers EACCES /
        # ENXIO to root as well) while PID 1 reads the real file or binary.
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
    unset = [str(item) for item in state.get("unset_env") or []]
    keys = ("PATH", PCI_ENV, *ENV_OVERRIDES, *unset)
    environment = process_environment(pid, keys)
    for name in unset:
        checks.append(
            {
                "kind": "unset-env",
                "dest": name,
                "unit_has": name in environment,
                "ok": name not in environment,
            }
        )
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
            "units": {unit: unit_state(unit) for unit in ALLOWED_UNITS},
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
            installed_for = str(wrappers[tool].get("unit") or HOST_UNIT)
            if installed_for != state["unit"]:
                raise ProbeError(
                    f"{tool} was installed for {installed_for}, not {state['unit']}"
                )
            state["wrappers"][tool] = wrappers[tool]
        audit_dir(run_id).mkdir(mode=0o755, parents=True, exist_ok=True)
        lines.append(f"Environment=PATH={bin_dir(run_id)}:{STANDARD_PATH}")
        # The wrapper journals and counts calls here; /run is read-only for the
        # unit under ProtectSystem=strict without this.
        lines.append(f"ReadWritePaths={audit_dir(run_id)}")
        for name in state.get("unset_env") or []:
            lines.append(f"UnsetEnvironment={name}")
    hidden = [
        path for item in arguments.inaccessible for path in checked_inaccessible(item)
    ]
    if hidden:
        # One node per run, created only once every item passed its allow-list.
        node = unreadable_node(run_id)
        for path in hidden:
            lines.append(f"BindReadOnlyPaths={node}:{path}")
            state["inaccessible"].append(path)
    if arguments.env:
        # Through a file, never ``Environment=``: the installer writes the
        # expected counts into the unit's own ``EnvironmentFile=`` and systemd
        # applies every EnvironmentFile= *after* every Environment= -- a drop-in
        # ``Environment=GPU_FAULT_EXPECTED_GPU_COUNT=9`` lost to the unit's
        # ``="8"`` (live 2026-10-26, 012 expectation_wrong). Among files the
        # later-listed wins, and a drop-in is parsed after the unit file
        # (systemd.exec(5) EnvironmentFile=, systemd.unit(5)), so this one is
        # read last. An empty value stays an empty string, as before.
        for pair in arguments.env:
            name, value = checked_env_override(pair)
            state["env"][name] = value
        content = "".join(f"{name}={value}\n" for name, value in state["env"].items())
        state_dir(run_id).mkdir(mode=0o755, parents=True, exist_ok=True)
        write_atomic(env_file(run_id), content.encode("utf-8"), 0o644)
        lines.append(f"EnvironmentFile={env_file(run_id)}")
    if arguments.pci_root:
        root = fixtures_dir(run_id) / "pci"
        # A directory (synthetic or copied root) or the ``--unreadable`` regular
        # file: the product's listing of the latter fails on every tick.
        if not (root.is_dir() or root.is_file()):
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


def checked_unit_window(unit: str, arguments: argparse.Namespace, mode: str) -> str:
    """A window on a unit other than the host collector is a command window only."""

    if unit == HOST_UNIT:
        return unit
    if (
        mode != "command"
        or arguments.bind
        or arguments.tree
        or arguments.inaccessible
        or arguments.env
        or arguments.pci_root
    ):
        raise ProbeError(
            f"{unit} carries a command-output window only (--mode command with "
            "--wrapper); no bind, tree, inaccessible path, count override or PCI root"
        )
    for tool in arguments.wrapper:
        checked_unit_wrapper(unit, tool)
    return unit


def open_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    mode = checked_mode(arguments)
    unit = checked_unit_window(checked_unit(arguments.unit), arguments, mode)
    restore_seconds = checked_restore_seconds(arguments.restore_seconds)
    if load_state(run_id):
        raise ProbeError("a source replay window for this run is already open")
    dropin = dropin_path(run_id, unit)
    if dropin.exists():
        raise ProbeError("an untracked source replay drop-in already exists")
    before = unit_state(unit)
    if before.get("ActiveState") != "active":
        raise ProbeError(f"{unit} is not active at baseline")
    unset = list(UNSET_ENV_BY_UNIT.get(unit, ()))
    state: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "unit": unit,
        "dropin": str(dropin),
        "binds": [],
        "trees": [],
        "wrappers": {},
        "inaccessible": [],
        "env": {},
        "unset_env": unset,
        "pci_root": False,
        "before": before,
        # The unit's real environment for every key a window may touch: what
        # ``_close`` must read back after the restore.
        "baseline_env": process_environment(
            before.get("MainPID", "0"), (PCI_ENV, *ENV_OVERRIDES, *unset)
        ),
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
    run(["systemctl", "restart", unit], timeout=120)
    after = unit_state(unit)
    if after.get("ActiveState") != "active":
        _close(run_id, state)
        raise ProbeError(f"{unit} is not active after the window opened; closed")
    verification: dict[str, Any] = {"skipped": True}
    if not arguments.skip_verify:
        try:
            verification = verify_window(run_id, state)
        except ProbeError as error:
            _close(run_id, state)
            raise ProbeError(f"window verification failed: {error}; closed") from error
        if not verification["ok"]:
            closed = _close(run_id, state)
            raise ProbeError(
                "the replacement is not visible only inside the unit; closed: "
                + json.dumps(verification["checks"], sort_keys=True)[:600],
                detail={"checks": verification["checks"], "close": closed},
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
    state = load_state(run_id)
    if not state:
        raise ProbeError("no source replay window is open for this run")
    set_value(arguments)
    emit(
        {
            "run_id": run_id,
            "unit": unit_state(window_unit(state)),
            "value": arguments.value,
        }
    )


def verify(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    if not state:
        raise ProbeError("no source replay window is open for this run")
    emit({"run_id": run_id, **verify_window(run_id, state)})


def restore_checks(state: dict[str, Any], pid: str) -> list[dict[str, Any]]:
    """The real configuration is read again: every replaced source on the device
    PID 1 reads it from and as readable (a procfs reading never hashes equal
    twice, so content is not compared), every touched variable at its baseline
    value (absent stays absent)."""

    checks: list[dict[str, Any]] = []
    dests = [
        item["dest"] for item in (*state.get("binds", []), *state.get("trees", []))
    ]
    for dest in (*dests, *state.get("inaccessible", [])):
        unit, host = view_path(pid, dest), view_path("1", dest)
        checks.append(
            {
                "kind": "source",
                "dest": dest,
                "unit_readable": unit["readable"],
                "ok": unit["readable"] == host["readable"]
                and unit["device"] == host["device"],
            }
        )
    baseline = state.get("baseline_env")
    if isinstance(baseline, dict):
        keys = (
            PCI_ENV,
            *ENV_OVERRIDES,
            *(str(k) for k in state.get("unset_env") or []),
        )
        environment = process_environment(pid, keys)
        for name in sorted(set(keys) & (set(baseline) | set(state.get("env") or {}))):
            checks.append(
                {
                    "kind": "env",
                    "dest": name,
                    "baseline": baseline.get(name),
                    "unit_value": environment.get(name),
                    "ok": environment.get(name) == baseline.get(name),
                }
            )
    return checks


def _close(run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    unit = window_unit(state)
    dropin = Path(state.get("dropin") or dropin_path(run_id, unit))
    dropin.unlink(missing_ok=True)
    reloads = 0
    while True:
        run(["systemctl", "daemon-reload"], timeout=180)
        reloads += 1
        loaded = unit_dropins(unit)
        listed = str(dropin) in loaded["paths"]
        if not (listed or loaded["reload_needed"]) or reloads >= 2:
            break
    run(["systemctl", "restart", unit], timeout=120)
    after = unit_state(unit)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{unit} is not active after the window closed")
    checks = restore_checks(state, after.get("MainPID", "0"))
    disarm_deadman(run_id)
    residue = {
        "dropin": dropin.exists(),
        "dropin_loaded": listed,
        "deadman": unit_state(f"{deadman_unit(run_id)}.timer").get("ActiveState")
        == "active",
    }
    if state_dir(run_id).is_dir():
        shutil.rmtree(state_dir(run_id))
    residue["state_dir"] = state_dir(run_id).exists()
    return {
        "unit": unit,
        "after": after,
        "closed_at": utc_now(),
        "reloads": reloads,
        "restore": {"checks": checks, "ok": all(item["ok"] for item in checks)},
        "residue": residue,
        "clean": not any(residue.values()),
    }


def close_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    if not state:
        for unit in ALLOWED_UNITS:
            dropin = dropin_path(run_id, unit)
            if dropin.exists():
                # A crash between the drop-in and the state file: still restore.
                emit(
                    {
                        "run_id": run_id,
                        "untracked_dropin": True,
                        **_close(run_id, {"unit": unit, "dropin": str(dropin)}),
                    }
                )
                return
        if state_dir(run_id).is_dir():
            shutil.rmtree(state_dir(run_id))
        emit({"run_id": run_id, "already_closed": True, "clean": True})
        return
    emit({"run_id": run_id, **_close(run_id, state)})


def await_ticks(arguments: argparse.Namespace) -> None:
    """Wait on the node until the unit completed ``--count`` collector ticks since
    the window opened, then (``--close``) restore within the same call.

    The exposure is measured from the ticks the collector actually took, not
    from the restart: after a restart the host collector sleeps a node-specific
    startup phase (``GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS`` plus the
    start-up jitter, up to ~35 s by default) before its first tick, so a fixed
    1.7-interval deadman closed the live 013 ``link_inactive`` window before
    the second confirming tick (2026-10-26). The ticks are read from the
    unit's own ``WatchdogTimestampMonotonic`` (``unit_stamps``), the time of
    the last ``WATCHDOG=1`` the main process sent; pings closer than the gap
    belong to one tick. The deadman armed at ``open-window`` stays the bound
    for a runner that never gets here; ``--timeout`` never exceeds it.
    """

    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    if not state:
        # The deadman got there first: nothing to wait for, nothing to close.
        emit(
            {"run_id": run_id, "already_closed": True, "completed": False, "ticks": []}
        )
        return
    unit = window_unit(state)
    count = max(1, min(int(arguments.count), MAX_TICKS))
    interval = max(1.0, min(float(arguments.interval), 3600.0))
    gap = max(MIN_TICK_GAP_SECONDS, interval * TICK_GAP_RATIO)
    timeout = max(1.0, min(float(arguments.timeout), float(state["restore_seconds"])))
    started, ticks = monotonic_us(), []
    while True:
        stamps = unit_stamps(unit)
        stamp, began = (
            stamps.get("WatchdogTimestampMonotonic", 0),
            stamps.get("ExecMainStartTimestampMonotonic", 0),
        )
        now = monotonic_us()
        if stamp and stamp >= began:
            if not ticks or stamp - ticks[-1]["last"] > gap * 1_000_000:
                ticks.append({"first": stamp, "last": stamp, "pings": 1})
            elif stamp != ticks[-1]["last"]:
                ticks[-1].update(last=stamp, pings=ticks[-1]["pings"] + 1)
        completed = len(ticks) >= count and now - ticks[-1]["last"] >= gap * 1_000_000
        if completed or now - started >= timeout * 1_000_000:
            break
        pause(TICK_POLL_SECONDS)
    at = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "run_id": run_id,
        "unit": unit,
        "count": count,
        "interval_seconds": interval,
        "gap_seconds": gap,
        "completed": completed,
        "waited_seconds": round((now - started) / 1e6, 1),
        "ticks": [
            {
                "pings_seen": tick["pings"],
                "first_ping_at": (
                    at - timedelta(microseconds=now - tick["first"])
                ).isoformat(),
                "last_ping_at": (
                    at - timedelta(microseconds=now - tick["last"])
                ).isoformat(),
            }
            for tick in ticks
        ],
    }
    if arguments.close:
        result["close"] = _close(run_id, state)
    emit(result)


def snapshot(arguments: argparse.Namespace) -> None:
    """What of this run is on the node now; after a reboot the proof it is nothing.

    ``unit_pci_root`` is the running host collector's own ``GPU_FAULT_PCI_
    DEVICES_ROOT`` (None when unset): a fixture path there is residue, a real
    sysfs path or nothing is the product reading the real root again.
    """

    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    unit = unit_state(HOST_UNIT)
    emit(
        {
            "run_id": run_id,
            "window_open": bool(state),
            "dropin": any(dropin_path(run_id, unit).exists() for unit in ALLOWED_UNITS),
            "state_dir": state_dir(run_id).exists(),
            "deadman": unit_state(f"{deadman_unit(run_id)}.timer").get("ActiveState")
            == "active",
            "unit_active_state": unit.get("ActiveState"),
            "unit_pci_root": process_environment(
                unit.get("MainPID", "0"), (PCI_ENV,)
            ).get(PCI_ENV),
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

    def add(item: argparse.ArgumentParser, flags: str, **options: Any) -> None:
        """The space-separated ``flags``, all with the same ``options``."""

        for flag in flags.split():
            item.add_argument(flag, **options)

    many: dict[str, Any] = {"action": "append", "default": []}
    switch: dict[str, Any] = {"action": "store_true"}
    commands.add_parser("preflight").set_defaults(handler=preflight)
    add(
        command("install-source", install_source),
        "--name --sha256 --b64",
        required=True,
    )
    item = command("build-pci-fixture", _builder("build_pci_fixture"))
    item.add_argument("--efa-count", type=int)
    item.add_argument("--source")
    add(item, "--drop", **many)
    add(item, "--malformed --unreadable", **switch)
    item = command("build-infiniband-fixture", _builder("build_infiniband_fixture"))
    item.add_argument("--source", default="/sys/class/infiniband")
    item.add_argument("--netdev-root", default=NETDEV_ROOT)
    add(item, "--drop --duplicate --port-state --phys-state --driver --set", **many)
    add(item, "--remove-counter --no-ports --netdev", **many)
    add(item, "--allow-empty", **switch)
    item = command("inventory", _builder("inventory"))
    item.add_argument("--source", default="/sys/class/infiniband")
    item.add_argument("--netdev-root", default=NETDEV_ROOT)
    command("read-source", read_source).add_argument("--path", required=True)
    item = command("await-ticks", await_ticks)
    item.add_argument("--count", type=int, default=2)
    item.add_argument("--interval", type=float, default=15)
    item.add_argument("--timeout", type=float, default=120)
    add(item, "--close", **switch)
    item = command("build-nvidia-proc-fixture", _builder("build_nvidia_proc_fixture"))
    item.add_argument("--source", default="/proc/driver/nvidia")
    item.add_argument("--version-text")
    item.add_argument("--gpu-count", type=int)
    add(item, "--omit-version --omit-gpus", **switch)
    item = command("build-meminfo-fixture", _builder("build_meminfo_fixture"))
    item.add_argument("--source", default="/proc/meminfo")
    add(item, "--set", **many)
    item = command("install-wrapper", install_wrapper)
    item.add_argument("--tool", required=True)
    item.add_argument("--unit", default=HOST_UNIT)
    add(item, "--allow-missing-real", **switch)
    item = command("open-window", open_window)
    item.add_argument("--mode", choices=WINDOW_MODES, required=True)
    item.add_argument("--unit", default=HOST_UNIT)
    add(item, "--bind --tree --wrapper --inaccessible --env", **many)
    add(item, "--pci-root --skip-verify", **switch)
    item.add_argument("--restore-seconds", type=int, default=900)
    item = command("set-value", apply_value)
    add(item, "--name --value", required=True)
    item.add_argument("--path")
    item = command("audit-read", _builder("audit_read"))
    item.add_argument("--tool", required=True)
    item.add_argument("--tail", type=int, default=100)
    item = command("journal-read", _builder("journal_read"))
    add(item, "--unit --since", required=True)
    item.add_argument("--tail", type=int, default=200)
    for name, handler in (
        ("status", _builder("status")),
        ("verify", verify),
        ("close-window", close_window),
        ("snapshot", snapshot),
    ):
        command(name, handler)
    return value


def exception_site(exc: BaseException) -> str | None:
    """The innermost ``function:line`` of this file on the exception's path."""

    site = None
    frame = exc.__traceback__
    while frame is not None:
        if frame.tb_frame.f_code.co_filename == __file__:
            site = f"{frame.tb_frame.f_code.co_name}:{frame.tb_lineno}"
        frame = frame.tb_next
    return site


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        # The host fixture's *exception text* shows only the tokens it
        # whitelists for ``error_kind: probe``: the class, a digest of the
        # message and the site. The message and the structured detail (a
        # ProbeError's ``detail``: the verification checks) travel in the same
        # line and the fixture keeps them, redacted, as the case's
        # ``probe-error.json`` -- two lanes brute-forced digests on 2026-10-26.
        emit(
            {
                "error": type(exc).__name__,
                "error_kind": "probe",
                "error_class": type(exc).__name__,
                "error_code": hashlib.sha256(str(exc).encode()).hexdigest()[:16],
                "error_site": exception_site(exc),
                "error_message": str(exc)[:2000],
                "error_detail": getattr(exc, "detail", None),
                "command": getattr(arguments, "command", None),
            }
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
