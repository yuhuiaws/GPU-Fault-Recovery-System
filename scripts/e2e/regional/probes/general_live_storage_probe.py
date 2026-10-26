#!/usr/bin/env python3
"""Storage half of the GF-GENERAL-LIVE host probe: test file systems and devices.

Runs on the target node through ``host_probe_fixture.py`` (chroot into the
host, the node's own Python), standard library only, one JSON object on the
last stdout line -- the sibling of ``general_live_node_probe.py``, split off
because one file may not exceed the repository's size limit. It creates only
test-only objects named after a digest of the run id: a loop file system or a
loop + device-mapper device mounted under ``/mnt/gfrs-live/<digest>/<name>``
(the backing file under an allow-listed directory whose file system may not
end above 80 % used), a bounded O_DIRECT writer unit inside that mount, and
the 004 ``probe_error`` fault mapping (``fault_mapping``). Every creation is
recorded in ``storage-state.json`` first and preceded by a ``systemd-run
--on-active`` deadman that undoes it from a copy of this file; ``residue``
names whatever is still present. ``dmsetup`` create/resume/remove block until
the host udevd completes a libdevmapper cookie that never reaches the Pod's
own IPC namespace, so the setup runs as a bounded transient unit and every
other device command joins PID 1's IPC and mount namespaces
(``host_command``); ``CALL_BUDGET_SECONDS`` bounds each subcommand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_ROOT = Path("/run/gpu-fault-acceptance/general")
MOUNT_ROOT = Path("/mnt/gfrs-live")
#: The host root mount namespace's table (the Pod is hostPID).
HOST_MOUNTINFO = Path("/proc/1/mountinfo")
MOUNT_KEYS = ("devno", "mountpoint", "options", "fstype", "source")
SYS_BLOCK = Path("/sys/block")
SYS_MODULE = Path("/sys/module")
#: Modules ``--load-module`` (KERNEL_MODULE_LOAD) may load; ``cleanup`` unloads
#: them again. ``dm-error`` is built into ``dm_mod`` and needs no load.
DELAY_MODULE = "dm_delay"
VFAT_MODULE = "vfat"
LOADABLE_MODULES = (DELAY_MODULE, VFAT_MODULE)
#: File systems a test device may carry. FAT16 is the 004 ``probe_error``
#: carrier: ``statfs`` counts its free clusters from the FAT on the device until
#: one count succeeds (``fault_mapping``); ``errors=continue`` keeps it rw.
MKFS = {
    "ext4": ("mkfs.ext4", "-q", "-F", "-m", "0", "-L"),
    "vfat": ("mkfs.vfat", "-F", "16", "-n"),
}
MOUNT_OPTIONS = {"ext4": (), "vfat": ("-o", "errors=continue,noatime")}
#: The FAT16 carrier's error segment is page-aligned, so a user-space page read
#: of the boot sector or root directory never touches it (no ``Buffer I/O
#: error`` in the kernel log); the FAT must keep this many sectors inside it.
PAGE_SECTORS = os.sysconf("SC_PAGE_SIZE") // 512
MIN_FAT_ERROR_SECTORS = 2 * PAGE_SECTORS
#: Device commands run in PID 1's IPC namespace (udev completes the dmsetup
#: cookie there) and mount namespace (the collector and the writer see it).
HOST_NAMESPACES = ("nsenter", "--target", "1", "--ipc", "--mount", "--")
#: Per-command timeouts (seconds): systemd method calls give up after sd-bus's
#: 25 s; ``dmsetup``/``losetup``/``blockdev`` return in well under a second.
QUICK_SECONDS = 30
#: ``TimeoutStopSec`` of every unit this probe starts (SIGTERM, then SIGKILL
#: after the same timeout), so ``systemctl stop`` returns within ``UNIT_STOP_SECONDS``.
UNIT_TIMEOUT_STOP_SECONDS = 20
UNIT_STOP_SECONDS = 2 * UNIT_TIMEOUT_STOP_SECONDS + 20
MOUNT_SECONDS = 60
UMOUNT_SECONDS = 120
#: How long ``destroy_device`` waits for a mapping's openers after the unmount
#: before removing it (deferred when one stays); polled every ``REMOVE_POLL_SECONDS``.
REMOVE_WAIT_SECONDS = 90
REMOVE_POLL_SECONDS = 3
#: The setup unit's steps and their timeouts; its ``RuntimeMaxSec`` adds
#: ``SETUP_RECORD_SECONDS`` so the worker records a failed step first.
SETUP_STEP_SECONDS = {
    "fallocate": 120,
    "mkfs": 300,
    "mount": MOUNT_SECONDS,
    **dict.fromkeys(("losetup", "blockdev", "dmsetup-create"), QUICK_SECONDS),
}
SETUP_RECORD_SECONDS = 30
SETUP_MAX_SECONDS = sum(SETUP_STEP_SECONDS.values()) + SETUP_RECORD_SECONDS
#: Worst-case wall time of one call per subcommand: ``main`` arms the deadline
#: and ``run`` clamps every command to what is left, so a call ends (with its
#: JSON error) inside the runner's exec timeout; unlisted ones keep per-command timeouts.
CALL_BUDGET_SECONDS = {
    "dm-status": 60,
    "dm-setup": SETUP_MAX_SECONDS - SETUP_RECORD_SECONDS,
    "io-start": 180,
    "io-stop": 120,
    **dict.fromkeys(("dm-create", "dm-recreate", "dm-retarget"), 300),
    **dict.fromkeys(("dm-suspend", "dm-resume"), 180),
    **dict.fromkeys(("dm-destroy", "cleanup"), 600),
}
#: What ``unit_state`` reads back from ``systemctl show``.
UNIT_PROPERTIES = (
    "LoadState ActiveState SubState UnitFileState MainPID NRestarts InvocationID "
    "ExecMainStartTimestamp"
).split()
#: The deadline of the running call (``time.monotonic``), empty when unbudgeted.
DEADLINE: list[float] = []
#: Command helpers every step goes through; an error's site is the caller.
HELPER_FRAMES = frozenset({"run", "clamp", "host_command", "dmsetup", "systemctl"})
RUNNING_STATES = frozenset({"active", "activating", "deactivating", "reloading"})
BACKING_ROOTS = (
    Path("/var/lib/gpu-fault-acceptance"),
    Path("/opt/dlami/nvme/gpu-fault-acceptance"),
)
BACKING_MAX_USED_PERCENT = 80.0
UNIT_PREFIX = "gfrs-"
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
SAFE_NAME = re.compile(r"^[a-z]{1,8}$")
MIN_FS_MIB = 256
MAX_FS_MIB = 4096
MAX_SECONDS = 3600
MIN_DEADMAN_SECONDS = 60
MAX_DEADMAN_SECONDS = 5400
#: The node probe owns ``gfrs-<digest>-deadman-*``; this probe's deadmen are
#: ``gfrs-<digest>-sdeadman-*`` so neither can stop the other's timers.
DEADMAN_PREFIX = "sdeadman"


class ProbeError(RuntimeError):
    pass


class BudgetExhausted(ProbeError):
    """The call used up its ``CALL_BUDGET_SECONDS`` before a command could run."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def clamp(timeout: float) -> float:
    """``timeout`` cut to what is left of the call's budget."""

    if not DEADLINE:
        return timeout
    left = DEADLINE[0] - time.monotonic()
    if left <= 0:
        raise BudgetExhausted("the probe call has used up its time budget")
    return min(float(timeout), left)


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: float = 180,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=clamp(timeout),
        check=False,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): {shlex.join(command)}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return completed


def tool_present(name: str) -> bool:
    return shutil.which(name) is not None


def host_command(
    command: list[str], *, timeout: float = QUICK_SECONDS, **options: Any
) -> subprocess.CompletedProcess[str]:
    """``command`` in PID 1's IPC and mount namespaces (the Pod is hostPID): only
    there does the host udevd's cookie wake ``dmsetup``, and a mount made there is
    the one the collector sees. ``options`` are ``run``'s (``check``, ``input_text``
    -- a multi-segment table goes to ``dmsetup`` on stdin)."""

    if not tool_present("nsenter"):
        raise ProbeError("nsenter is missing: device commands cannot reach the host")
    return run([*HOST_NAMESPACES, *command], timeout=timeout, **options)


def dmsetup(*arguments: str, **options: Any) -> subprocess.CompletedProcess[str]:
    return host_command(["dmsetup", *arguments], **options)


def checked_run_id(value: str) -> str:
    if SAFE_RUN_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def checked_name(value: str) -> str:
    if SAFE_NAME.fullmatch(value) is None:
        raise ProbeError("unsafe resource name")
    return value


def bounded(value: int, low: int, high: int, label: str) -> int:
    number = int(value)
    if not low <= number <= high:
        raise ProbeError(f"{label} is outside {low}..{high}")
    return number


def deadman_seconds(value: int) -> int:
    return bounded(value, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman seconds")


def run_digest(run_id: str) -> str:
    return hashlib.sha256(checked_run_id(run_id).encode()).hexdigest()[:12]


def state_dir(run_id: str) -> Path:
    return STATE_ROOT / run_digest(run_id)


def unit_name(run_id: str, kind: str) -> str:
    return f"{UNIT_PREFIX}{run_digest(run_id)}-{kind}"


def mountpoint(run_id: str, name: str) -> Path:
    return MOUNT_ROOT / run_digest(run_id) / checked_name(name)


def dm_name(run_id: str, name: str) -> str:
    return f"{UNIT_PREFIX}{run_digest(run_id)}-{checked_name(name)}"


#: ``modules`` are the kernel modules this run loaded (recorded before the
#: load), which ``cleanup`` unloads again once no device of the run is left.
EMPTY_STATE: dict[str, Any] = {
    "units": [],
    "filesystems": {},
    "devices": {},
    "modules": [],
}


def load_state(run_id: str) -> dict[str, Any]:
    path = state_dir(run_id) / "storage-state.json"
    if not path.is_file():
        return {"run_id": run_id, **json.loads(json.dumps(EMPTY_STATE))}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ProbeError("storage probe state identity mismatch")
    for key, empty in EMPTY_STATE.items():
        value.setdefault(key, json.loads(json.dumps(empty)))
    return value


def save_state(run_id: str, state: dict[str, Any]) -> None:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "storage-state.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def probe_copy(run_id: str) -> Path:
    """A copy of this file the deadman and the setup unit run, refreshed
    whenever this file differs (a rerun never starts an older copy)."""

    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / "storage-probe.py"
    source = Path(__file__).resolve().read_bytes()
    if not target.is_file() or target.read_bytes() != source:
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(source)
        temporary.chmod(0o700)
        os.replace(temporary, target)
    return target


def systemctl(
    *arguments: str, timeout: int = QUICK_SECONDS
) -> subprocess.CompletedProcess[str]:
    """``systemctl <arguments>``, never raising: unit state is read back instead."""

    return run(["systemctl", *arguments], check=False, timeout=timeout)


def unit_state(unit: str) -> dict[str, str]:
    properties = [f"--property={key}" for key in UNIT_PROPERTIES]
    completed = systemctl("show", unit, *properties)
    pairs = (line.partition("=") for line in completed.stdout.splitlines())
    values = {key: value for key, separator, value in pairs if separator}
    return {**values, "returncode": str(completed.returncode)}


def arm_deadman(run_id: str, kind: str, seconds: int) -> str:
    """Arm ``cleanup --kind <kind>`` after ``seconds``; raise if it is not armed."""

    deadman_seconds(seconds)
    name = unit_name(run_id, f"{DEADMAN_PREFIX}-{kind}")
    systemctl("stop", f"{name}.timer")
    systemctl("reset-failed", f"{name}.service")
    timer = [
        f"--unit={name}",
        f"--on-active={int(seconds)}s",
        "--timer-property=AccuracySec=1s",
    ]
    properties = ("Restart=on-failure", "RestartSec=30s", "StartLimitIntervalSec=0")
    properties += (f"TimeoutStopSec={UNIT_TIMEOUT_STOP_SECONDS}",)
    cleanup = [sys.executable, str(probe_copy(run_id)), "cleanup", "--run-id", run_id]
    unit = ["systemd-run", *timer, *(f"--property={item}" for item in properties)]
    run([*unit, *cleanup, "--kind", kind], timeout=QUICK_SECONDS)
    if unit_state(f"{name}.timer").get("ActiveState") != "active":
        raise ProbeError(f"deadman {name} is not armed")
    return name


def disarm_deadman(run_id: str, kind: str) -> None:
    name = unit_name(run_id, f"{DEADMAN_PREFIX}-{kind}")
    systemctl("stop", f"{name}.timer")
    if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
        systemctl("stop", f"{name}.service", timeout=UNIT_STOP_SECONDS)
    systemctl("reset-failed", f"{name}.service", f"{name}.timer")


def start_unit(
    run_id: str, kind: str, seconds: int, command: list[str], *properties: str
) -> str:
    """A transient unit (host namespaces) that stops itself after ``seconds``."""

    bounded(seconds, 10, MAX_SECONDS, "unit seconds")
    name = unit_name(run_id, kind)
    state = load_state(run_id)
    if name not in state["units"]:
        state["units"].append(name)
        save_state(run_id, state)
    systemctl("stop", f"{name}.service", timeout=UNIT_STOP_SECONDS)
    systemctl("reset-failed", f"{name}.service")
    bounds = (
        f"RuntimeMaxSec={int(seconds)}",
        f"TimeoutStopSec={UNIT_TIMEOUT_STOP_SECONDS}",
    )
    options = [f"--property={item}" for item in (*bounds, *properties)]
    run(
        ["systemd-run", f"--unit={name}", "--collect", *options, *command],
        timeout=QUICK_SECONDS,
    )
    return name


def stop_unit(run_id: str, kind: str) -> dict[str, str]:
    name = unit_name(run_id, kind)
    systemctl("stop", f"{name}.service", timeout=UNIT_STOP_SECONDS)
    systemctl("reset-failed", f"{name}.service")
    return unit_state(f"{name}.service")


def statvfs_reading(path: str) -> dict[str, Any]:
    try:
        stat = os.statvfs(path)
    except OSError as exc:
        return {"path": path, "error": f"{type(exc).__name__}: {exc.errno}"}
    total = stat.f_blocks * stat.f_frsize
    available = stat.f_bavail * stat.f_frsize
    files = stat.f_files
    return {
        "path": path,
        "used_percent": 100 * (total - available) / total if total else None,
        "inodes_used_percent": 100 * (files - stat.f_ffree) / files if files else None,
        "read_only": bool(stat.f_flag & os.ST_RDONLY),
        "total_bytes": total,
        "available_bytes": available,
        "free_inodes": stat.f_ffree,
    }


def mounts(table: Path = Path("/proc/self/mountinfo")) -> list[dict[str, str]]:
    """One entry per ``mountinfo`` line: devno, mountpoint, options, fstype, source."""

    result = []
    for line in table.read_text().splitlines():
        fields = line.split()
        if "-" not in fields:
            continue
        sep = fields.index("-")
        source = fields[sep + 2] if len(fields) > sep + 2 else ""
        devno, mountpoint, options, fstype = (
            fields[2],
            fields[4],
            fields[5],
            fields[sep + 1],
        )
        result.append(
            dict(zip(MOUNT_KEYS, (devno, mountpoint, options, fstype, source)))
        )
    return result


def host_mounts() -> list[dict[str, str]]:
    """The host root mount namespace's mounts (this Pod's own table if unreadable)."""

    try:
        return mounts(HOST_MOUNTINFO)
    except OSError:
        return mounts()


def mounted_on_host(target: str) -> bool:
    return any(item["mountpoint"] == target for item in host_mounts())


def backing_root(value: str, size_bytes: int) -> Path:
    root = Path(value)
    if root not in BACKING_ROOTS or not root.parent.exists():
        raise ProbeError("backing root is not allow-listed or its parent is absent")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    reading = statvfs_reading(str(root))
    total = float(reading.get("total_bytes") or 0)
    available = float(reading.get("available_bytes") or 0)
    if total <= 0 or available <= size_bytes:
        raise ProbeError("backing file system has no room for the test file")
    used_after = 100 * (total - (available - size_bytes)) / total
    if used_after > BACKING_MAX_USED_PERCENT:
        raise ProbeError("backing file system would exceed its usage guard")
    return root


def run_backing_dir(run_id: str, root: Path) -> Path:
    directory = root / f"gfrs-live-{run_digest(run_id)}"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def run_directories(run_id: str) -> list[Path]:
    """The run's backing directory under each allow-listed root and its mount root."""

    digest = run_digest(run_id)
    backing = [root / f"gfrs-live-{digest}" for root in BACKING_ROOTS]
    return [*backing, MOUNT_ROOT / digest]


def prune_run_directories(run_id: str) -> list[str]:
    """Remove the run's empty directories once it holds no file system or device."""

    state = load_state(run_id)
    if state["devices"] or state["filesystems"]:
        return []
    removed: list[str] = []
    for directory in run_directories(run_id):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            removed.append(str(directory))
    return removed


# --------------------------------------------------------------------------- #
# Test file systems and test block devices
# --------------------------------------------------------------------------- #
def loop_attach(path: Path, *, timeout: float = 180) -> str:
    completed = run(["losetup", "--find", "--show", str(path)], timeout=timeout)
    device = completed.stdout.strip()
    if not re.fullmatch(r"/dev/loop\d+", device):
        raise ProbeError("losetup returned an unexpected device")
    return device


def fs_create(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    size_mib = bounded(arguments.size_mib, MIN_FS_MIB, MAX_FS_MIB, "size MiB")
    seconds = deadman_seconds(arguments.deadman_seconds)
    state = load_state(run_id)
    if name in state["filesystems"] or name in state["devices"]:
        raise ProbeError("a test file system of this name already exists")
    root = backing_root(arguments.backing, size_mib * 1024 * 1024)
    deadman = arm_deadman(run_id, f"fs-{name}", seconds)
    backing = run_backing_dir(run_id, root) / f"{name}.img"
    target = mountpoint(run_id, name)
    record = {"backing": str(backing), "mountpoint": str(target), "loop": None}
    state["filesystems"][name] = {**record, "mounted": False}
    save_state(run_id, state)
    run(["fallocate", "-l", f"{size_mib}M", str(backing)])
    loop = loop_attach(backing)
    state["filesystems"][name]["loop"] = loop
    save_state(run_id, state)
    command = ["mkfs.ext4", "-q", "-F", "-m", "0", "-L", f"gfrs-{name}"]
    if arguments.inodes:
        command += ["-N", str(bounded(arguments.inodes, 16, 1_000_000, "inodes"))]
    run([*command, loop], timeout=300)
    target.mkdir(mode=0o755, parents=True, exist_ok=True)
    run(["mount", "-t", "ext4", loop, str(target)])
    state["filesystems"][name]["mounted"] = True
    save_state(run_id, state)
    reading = statvfs_reading(str(target))
    emit({**record, "name": name, "loop": loop, "statvfs": reading, "deadman": deadman})


def checked_filesystem(run_id: str, name: str) -> dict[str, Any]:
    state = load_state(run_id)
    entry = state["filesystems"].get(checked_name(name)) or state["devices"].get(name)
    if entry is None:
        raise ProbeError("no test file system of this name")
    return dict(entry)


def fs_fill(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_filesystem(run_id, arguments.name)
    percent = float(arguments.percent)
    if not 1.0 <= percent <= 100.0:
        raise ProbeError("fill percent is outside 1..100")
    target = Path(entry["mountpoint"])
    fill = target / "gfrs-fill.bin"
    fill.unlink(missing_ok=True)
    os.sync()
    stat = os.statvfs(target)
    total = stat.f_blocks * stat.f_frsize
    used = total - stat.f_bavail * stat.f_frsize
    wanted = int(total * percent / 100) - used
    block = 1 << 20
    wanted = max(0, wanted // block * block)
    if wanted:
        completed = run(["fallocate", "-l", str(wanted), str(fill)], check=False)
        if completed.returncode:
            raise ProbeError("fallocate of the fill file failed")
    os.sync()
    emit({"fill_bytes": wanted, "statvfs": statvfs_reading(str(target))})


def fs_unfill(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_filesystem(run_id, arguments.name)
    target = Path(entry["mountpoint"])
    fill, inode_dir = target / "gfrs-fill.bin", target / "gfrs-inodes"
    removed = int(fill.exists())
    fill.unlink(missing_ok=True)
    if inode_dir.is_dir():
        for item in inode_dir.iterdir():
            item.unlink()
            removed += 1
        inode_dir.rmdir()
    os.sync()
    emit({"removed": removed, "statvfs": statvfs_reading(str(target))})


def fs_inodes(arguments: argparse.Namespace) -> None:
    """Create empty files until the file system has no free inode left."""

    run_id = checked_run_id(arguments.run_id)
    entry = checked_filesystem(run_id, arguments.name)
    target = Path(entry["mountpoint"])
    directory = target / "gfrs-inodes"
    directory.mkdir(exist_ok=True)
    limit = os.statvfs(target).f_files + 16
    created = 0
    exhausted = False
    for index in range(limit):
        try:
            (directory / f"f{index:07d}").touch(exist_ok=False)
        except OSError as exc:
            exhausted = exc.errno == 28
            break
        created += 1
    os.sync()
    reading = statvfs_reading(str(target))
    emit({"created": created, "enospc": exhausted, "statvfs": reading})


def fs_remount(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_filesystem(run_id, arguments.name)
    if arguments.mode not in {"ro", "rw"}:
        raise ProbeError("remount mode must be ro or rw")
    run(["mount", "-o", f"remount,{arguments.mode}", entry["mountpoint"]])
    emit({"mode": arguments.mode, "statvfs": statvfs_reading(entry["mountpoint"])})


def fs_umount(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = checked_filesystem(run_id, name)
    run(["umount", entry["mountpoint"]], timeout=120)
    state = load_state(run_id)
    bucket = "filesystems" if name in state["filesystems"] else "devices"
    state[bucket][name]["mounted"] = False
    save_state(run_id, state)
    path = entry["mountpoint"]
    if arguments.rmdir:
        Path(path).rmdir()
    present, reading = Path(path).exists(), statvfs_reading(path)
    emit({"unmounted": path, "directory_present": present, "statvfs": reading})


def mount_command(fstype: str, source: str, target: str) -> list[str]:
    if fstype not in MKFS:
        raise ProbeError("unsupported test file system type")
    return ["mount", "-t", fstype, *MOUNT_OPTIONS[fstype], source, target]


def fs_mount(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = checked_filesystem(run_id, name)
    source = entry.get("dm_path") or entry["loop"]
    Path(entry["mountpoint"]).mkdir(mode=0o755, parents=True, exist_ok=True)
    run(mount_command(str(entry.get("fstype") or "ext4"), source, entry["mountpoint"]))
    state = load_state(run_id)
    bucket = "filesystems" if name in state["filesystems"] else "devices"
    state[bucket][name]["mounted"] = True
    save_state(run_id, state)
    reading = statvfs_reading(entry["mountpoint"])
    emit({"mounted": entry["mountpoint"], "statvfs": reading})


def destroy_filesystem(run_id: str, name: str) -> dict[str, Any]:
    state = load_state(run_id)
    entry = state["filesystems"].get(name)
    if entry is None:
        return {"name": name, "absent": True}
    target = entry["mountpoint"]
    for flags in ([], ["-l"]):
        if any(item["mountpoint"] == target for item in mounts()):
            run(["umount", *flags, target], check=bool(flags), timeout=120)
    if entry.get("loop"):
        run(["losetup", "-d", entry["loop"]], check=False)
    Path(entry["backing"]).unlink(missing_ok=True)
    if Path(target).is_dir():
        Path(target).rmdir()
    del state["filesystems"][name]
    save_state(run_id, state)
    return {"name": name, "destroyed": True}


def fs_destroy(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    result = destroy_filesystem(run_id, name)
    disarm_deadman(run_id, f"fs-{name}")
    result["directories_removed"] = prune_run_directories(run_id)
    emit(result)


def dm_table(
    sectors: int,
    loop: str,
    delay_ms: int,
    *,
    error: bool = False,
    fat: list[int] | None = None,
) -> str:
    """The data table, the whole device ``error``, or with ``fat`` ``[start, end)``
    only the FAT segment ``error`` (boot sector, root directory and data ``linear``)."""

    if error and fat:
        start, end = fat
        segments = [f"0 {start} linear {loop} 0"] if start else []
        segments.append(f"{start} {end - start} error")
        if end < sectors:
            segments.append(f"{end} {sectors - end} linear {loop} {end}")
        return "\n".join(segments)
    if error:
        return f"0 {sectors} error"
    if delay_ms:
        return f"0 {sectors} delay {loop} 0 {delay_ms}"
    return f"0 {sectors} linear {loop} 0"


def fat_region(device: str) -> list[int]:
    """``[start, end)`` of the FAT16 carrier's FATs in 512-byte sectors (from its
    boot sector), rounded inwards to page boundaries."""

    with open(device, "rb", buffering=0) as handle:
        boot = handle.read(512)
    if len(boot) < 512 or boot[54:62] != b"FAT16   ":
        raise ProbeError("the test device does not carry a FAT16 boot sector")
    # BPB: bytes per sector, reserved sectors, FATs and sectors per FAT.
    bps, _, reserved, fats, _, _, _, per_fat = struct.unpack_from("<HBHBHHBH", boot, 11)
    scale = bps // 512
    start = -(-reserved * scale // PAGE_SECTORS) * PAGE_SECTORS
    end = (reserved + fats * per_fat) * scale // PAGE_SECTORS * PAGE_SECTORS
    if scale < 1 or not reserved or end - start < MIN_FAT_ERROR_SECTORS:
        raise ProbeError("the FAT16 boot sector geometry leaves no error segment")
    return [start, end]


def dm_device_name(dm: str) -> str:
    """``dm-N`` of the mapping (the ``/proc/diskstats`` name the product reports)."""

    for entry in sorted(SYS_BLOCK.glob("dm-*")):
        try:
            named = (entry / "dm" / "name").read_text().strip()
        except OSError:
            continue
        if named == dm:
            return entry.name
    raise ProbeError("the test mapping has no dm-N block device")


def setup_kind(name: str) -> str:
    """The setup unit, ``gfrs-<digest>-io-<name>-setup`` (``residue`` lists it)."""

    return f"io-{checked_name(name)}-setup"


def module_loaded(name: str) -> bool:
    return (SYS_MODULE / name).exists()


def update_device(run_id: str, name: str, **values: Any) -> dict[str, Any]:
    """Merge ``values`` into the device record (written before the next step)."""

    state = load_state(run_id)
    entry = state["devices"].get(name)
    if entry is None:
        raise ProbeError("no test device of this name")
    entry.update(values)
    save_state(run_id, state)
    return dict(entry)


def dm_create(arguments: argparse.Namespace) -> None:
    """Record the test device and start its bounded setup unit (the runner polls
    ``dm-status``); deadman first, record second, so ``cleanup`` undoes whatever
    the setup reached. ``--load-module`` loads a missing module, recorded first."""

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    size_mib = bounded(arguments.size_mib, MIN_FS_MIB, MAX_FS_MIB, "size MiB")
    delay_ms = bounded(arguments.delay_ms, 0, 1000, "delay ms")
    seconds = deadman_seconds(arguments.deadman_seconds)
    fstype = str(arguments.fstype)
    if fstype not in MKFS:
        raise ProbeError("unsupported test file system type")
    wanted = {DELAY_MODULE: bool(delay_ms), VFAT_MODULE: fstype == "vfat"}
    load = [m for m, needed in wanted.items() if needed and not module_loaded(m)]
    if load and not arguments.load_module:
        raise ProbeError(f"{load[0]} is not loaded on this node")
    if not tool_present("nsenter"):
        raise ProbeError("nsenter is missing: device commands cannot reach the host")
    state = load_state(run_id)
    if name in state["devices"] or name in state["filesystems"]:
        raise ProbeError("a test device of this name already exists")
    root = backing_root(arguments.backing, size_mib * 1024 * 1024)
    deadman = arm_deadman(run_id, f"dm-{name}", seconds)
    dm = dm_name(run_id, name)
    unit = unit_name(run_id, setup_kind(name))
    record: dict[str, Any] = {
        "dm": dm,
        "dm_path": f"/dev/mapper/{dm}",
        "setup_unit": unit,
    }
    record["backing"] = str(run_backing_dir(run_id, root) / f"{name}.img")
    record["mountpoint"] = str(mountpoint(run_id, name))
    record.update(loop=None, delay_ms=delay_ms, size_mib=size_mib, fstype=fstype)
    record.update(target="data", mounted=False, phase="setup", step=None)
    state["devices"][name] = record
    # Recorded before the load: it was not loaded, so loaded later is ours.
    state["modules"].extend(m for m in load if m not in state["modules"])
    save_state(run_id, state)
    for module in load:
        run(["modprobe", module], check=False, timeout=QUICK_SECONDS)
        if not module_loaded(module):
            raise ProbeError(f"{module} could not be loaded")
    command = [sys.executable, str(probe_copy(run_id)), "dm-setup", "--run-id", run_id]
    start_unit(run_id, setup_kind(name), SETUP_MAX_SECONDS, [*command, "--name", name])
    report = {"name": name, "setup_max_seconds": SETUP_MAX_SECONDS, "deadman": deadman}
    report.update(module_loaded_by_run=bool(load), modules_loaded_by_run=load)
    emit({**state["devices"][name], **report, "state": unit_state(f"{unit}.service")})


def dm_setup(arguments: argparse.Namespace) -> None:
    """The setup unit's body (host namespaces): back, map, format, mount; a failure
    records the step it reached, error class, message digest and site."""

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = checked_device(run_id, name)
    if entry.get("phase") != "setup":
        raise ProbeError("the test device is not waiting for its setup")
    try:
        ready = build_device(run_id, name, entry)
    except Exception as exc:
        update_device(
            run_id,
            name,
            phase="failed",
            setup_error_class=type(exc).__name__,
            setup_error_code=error_code(exc),
            setup_error_site=exception_site(exc),
        )
        raise
    emit(ready)


def build_device(run_id: str, name: str, entry: dict[str, Any]) -> dict[str, Any]:
    def step(label: str) -> float:
        update_device(run_id, name, step=label)
        return SETUP_STEP_SECONDS[label]

    backing = str(entry["backing"])
    size = f"{int(entry['size_mib'])}M"
    run(["fallocate", "-l", size, backing], timeout=step("fallocate"))
    loop = loop_attach(Path(backing), timeout=step("losetup"))
    update_device(run_id, name, loop=loop)
    blocks = run(["blockdev", "--getsz", loop], timeout=step("blockdev"))
    sectors = int(blocks.stdout.strip())
    update_device(run_id, name, sectors=sectors)
    table = dm_table(sectors, loop, int(entry["delay_ms"]))
    dmsetup("create", entry["dm"], "--table", table, timeout=step("dmsetup-create"))
    device = dm_device_name(entry["dm"])
    update_device(run_id, name, device=device)
    fstype = str(entry.get("fstype") or "ext4")
    # A FAT label is at most 11 upper-case characters.
    label = f"gfrs-{name}" if fstype == "ext4" else f"GFRS{name.upper()}"[:11]
    run([*MKFS[fstype], label, entry["dm_path"]], timeout=step("mkfs"))
    fat = fat_region(entry["dm_path"]) if fstype == "vfat" else None
    update_device(run_id, name, fat_sectors=fat)
    timeout = step("mount")
    Path(entry["mountpoint"]).mkdir(mode=0o755, parents=True, exist_ok=True)
    mount = mount_command(fstype, entry["dm_path"], entry["mountpoint"])
    host_command(mount, timeout=timeout)
    update_device(run_id, name, mounted=True, phase="ready", step=None)
    built = {"name": name, "phase": "ready", "device": device, "loop": loop}
    built.update(fstype=fstype, fat_sectors=fat, delay_ms=entry["delay_ms"])
    return {**built, "dm": entry["dm"], "mountpoint": entry["mountpoint"]}


def dm_status(arguments: argparse.Namespace) -> None:
    """Where the device's setup stands: ``setup``, ``ready`` or ``failed`` (a record
    still at ``setup`` behind a unit that no longer runs: the worker died at the
    step it had reached). No ``error`` key: the fixture would read a failed call."""

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    unit = unit_state(f"{unit_name(run_id, setup_kind(name))}.service")
    entry = checked_device(run_id, name)
    phase = str(entry.get("phase") or "ready")
    lost = phase == "setup" and unit.get("ActiveState") not in RUNNING_STATES
    failed = lost or phase == "failed"
    copied = (
        "step",
        "setup_error_code",
        "setup_error_site",
        "dm",
        "loop",
        "mountpoint",
    )
    status = {key: entry.get(key) for key in (*copied, "delay_ms")}
    status["name"], status["phase"] = name, "failed" if lost else phase
    status["failed_step"] = (entry.get("step") or "start") if failed else None
    status["setup_error_class"] = (
        "SetupUnitEnded" if lost else entry.get("setup_error_class")
    )
    status["device"] = entry.get("device") if phase == "ready" else None
    status["setup_unit"] = {
        k: unit.get(k) for k in ("LoadState", "ActiveState", "SubState")
    }
    emit(status)


def checked_device(run_id: str, name: str) -> dict[str, Any]:
    entry = load_state(run_id)["devices"].get(checked_name(name))
    if entry is None:
        raise ProbeError("no test device of this name")
    return dict(entry)


def ready_device(run_id: str, name: str) -> dict[str, Any]:
    entry = checked_device(run_id, name)
    if entry.get("phase", "ready") != "ready":
        raise ProbeError("the test device has not finished its setup")
    return entry


def dm_suspend(arguments: argparse.Namespace) -> None:
    """Suspend the test device; a deadman resumes it after ``seconds``."""

    run_id = checked_run_id(arguments.run_id)
    entry = ready_device(run_id, arguments.name)
    seconds = bounded(arguments.seconds, MIN_DEADMAN_SECONDS, 180, "suspend seconds")
    deadman = arm_deadman(run_id, f"resume-{checked_name(arguments.name)}", seconds)
    dmsetup("suspend", "--nolockfs", "--noflush", entry["dm"])
    info = dmsetup("info", entry["dm"], check=False).stdout[-400:]
    emit({"suspended": entry["dm"], "deadman": deadman, "info": info})


def dm_resume(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_device(run_id, arguments.name)
    dmsetup("resume", entry["dm"])
    disarm_deadman(run_id, f"resume-{checked_name(arguments.name)}")
    emit({"resumed": entry["dm"]})


def refuse_running_writer(run_id: str, name: str, action: str) -> None:
    if (
        unit_state(f"{unit_name(run_id, f'io-{name}')}.service").get("ActiveState")
        in RUNNING_STATES
    ):
        raise ProbeError(f"the writer still runs; stop it before the {action}")


def dm_recreate(arguments: argparse.Namespace) -> None:
    """Remove and recreate the mapping under the same name (counters restart)."""

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = ready_device(run_id, name)
    refuse_running_writer(run_id, name, "recreate")
    before = dm_device_name(entry["dm"])
    if mounted_on_host(entry["mountpoint"]):
        host_command(["umount", entry["mountpoint"]], timeout=UMOUNT_SECONDS)
    dmsetup("remove", entry["dm"])
    table = dm_table(int(entry["sectors"]), entry["loop"], int(entry["delay_ms"]))
    dmsetup("create", entry["dm"], "--table", table)
    fstype = str(entry.get("fstype") or "ext4")
    mount = mount_command(fstype, entry["dm_path"], entry["mountpoint"])
    host_command(mount, timeout=MOUNT_SECONDS)
    after = dm_device_name(entry["dm"])
    update_device(run_id, name, device=after, mounted=True, target="data")
    emit({"dm": entry["dm"], "device_before": before, "device_after": after})


def mapping_devno(dm: str) -> str:
    """``major:minor`` of the mapping, empty once it is gone."""

    columns = ("info", "-c", "--noheadings", "-o", "major,minor")
    completed = dmsetup(*columns, dm, check=False)
    match = re.fullmatch(r"\s*(\d+):\s*(\d+)\s*", completed.stdout)
    return (
        f"{int(match[1])}:{int(match[2])}" if match and not completed.returncode else ""
    )


def mount_devno(target: str) -> str:
    """``major:minor`` of the file system a path resolves to."""

    dev = os.stat(target).st_dev
    return f"{os.major(dev)}:{os.minor(dev)}"


def direct_read(device: str, sector: int) -> str:
    """One page read at ``sector`` through ``device`` with O_DIRECT -- what the
    table serves, no page cache in between: ``ok`` or the error."""

    fd = os.open(device, os.O_RDONLY | os.O_DIRECT)
    try:
        os.preadv(fd, [mmap.mmap(-1, 512 * PAGE_SECTORS)], sector * 512)
        return "ok"
    except OSError as exc:
        return f"{type(exc).__name__}: {exc.errno}"
    finally:
        os.close(fd)


def fault_mapping(entry: dict[str, Any]) -> dict[str, Any]:
    """The 004 ``probe_error`` injection: a second mapping over the same loop
    device whose FAT segment is ``error``, mounted over the healthy mount.

    A new block device gets a new superblock, so FAT counts its free clusters
    afresh through the fault mapping: EIO on every statvfs, whatever an earlier
    mount cached. The healthy mapping is never touched: a private copy of its
    mount in another mount namespace (the live window probe Pod, started after
    the mount with the default private propagation) keeps its superblock alive,
    and a remount of the SAME device reused that superblock and its cached count
    (the 2026-10-25/26 INCONCLUSIVE). O_DIRECT reads prove the table before the
    mount (boot sector ok, first FAT sector EIO); nothing writes to the error
    segment; ``--target data`` reloads the fault mapping to its data table in place.
    """

    target = str(entry["mountpoint"])
    fault = str(entry.get("fault_dm") or f"{entry['dm']}-fault")
    path = f"/dev/mapper/{fault}"
    if mapping_devno(fault):
        raise ProbeError("the fault mapping already exists; retarget data first")
    fat = [int(value) for value in entry["fat_sectors"]]
    table = dm_table(int(entry["sectors"]), entry["loop"], 0, error=True, fat=fat)
    dmsetup("create", fault, input_text=table + "\n")
    proof = {"boot": direct_read(path, 0), "fat": direct_read(path, fat[0])}
    if proof["boot"] != "ok" or proof["fat"] == "ok":
        raise ProbeError("the fault mapping does not serve the FAT segment as error")
    host_command(mount_command("vfat", path, target), timeout=MOUNT_SECONDS)
    reading, devno = statvfs_reading(target), mount_devno(target)
    result = {"fault_dm": fault, "direct_reads": proof, "statvfs": reading}
    result.update(mount_devno=devno, fault_devno=mapping_devno(fault), diagnosis=None)
    if "error" not in reading:
        result["diagnosis"] = (
            "statfs answered over a mapping whose FAT reads fail: superblock reused"
            if devno == result["fault_devno"]
            else "the mount point does not resolve to the fault mapping"
        )
    return result


def retarget_once(entry: dict[str, Any], *, error: bool) -> dict[str, Any]:
    """Fault or restore the test device; its mount point's statvfs after it:
    ``error`` is ``fault_mapping`` on the FAT16 carrier, else the whole-device
    ``error`` table; ``data`` reloads the fault mapping (else the device) in place."""

    fault = entry.get("fault_dm")
    if error and str(entry.get("fstype") or "ext4") == "vfat":
        return fault_mapping(entry)
    sectors, loop, delay = int(entry["sectors"]), entry["loop"], int(entry["delay_ms"])
    if not error and fault and mapping_devno(str(fault)):
        dm, table = str(fault), dm_table(sectors, loop, 0)
    else:
        dm, table = entry["dm"], dm_table(sectors, loop, delay, error=error)
    dmsetup("suspend", "--nolockfs", "--noflush", dm)
    dmsetup("reload", dm, input_text=table + "\n")
    dmsetup("resume", dm)
    return {"statvfs": statvfs_reading(str(entry["mountpoint"]))}


def dm_retarget(arguments: argparse.Namespace) -> None:
    """Swap the test device between ``data`` and ``error`` (the 004 ``probe_error``
    injection: mount stays, rw in the host table, every ``statvfs`` EIO; ``faulted``
    is the probe's own reading, ``diagnosis`` why it did not fail). ``--delay-ms``
    gives the data table a new delay first (006 ``sustained_latency`` recovery)."""

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = ready_device(run_id, name)
    if arguments.target not in {"data", "error"}:
        raise ProbeError("retarget target must be data or error")
    if arguments.delay_ms is not None:
        delay = bounded(arguments.delay_ms, 0, 1000, "delay ms")
        if delay and not module_loaded(DELAY_MODULE):
            raise ProbeError("dm_delay is not loaded on this node")
        entry = update_device(run_id, name, delay_ms=delay)
    refuse_running_writer(run_id, name, "retarget")
    error = arguments.target == "error"
    result = retarget_once(entry, error=error)
    recorded = {"target": arguments.target}
    if result.get("fault_dm"):
        recorded["fault_dm"] = result["fault_dm"]
    update_device(run_id, name, **recorded)
    report = {"name": name, "dm": entry["dm"], "delay_ms": int(entry["delay_ms"])}
    report.update({"fat_sectors": entry.get("fat_sectors"), "attempts": 1, **recorded})
    report.update(result)
    report["faulted"] = error and "error" in result["statvfs"]
    report["mounted_on_host"] = mounted_on_host(str(entry["mountpoint"]))
    emit(report)


def loop_devices() -> list[dict[str, Any]]:
    if not tool_present("losetup"):
        return []
    listing = run(["losetup", "-J", "-l"], check=False, timeout=QUICK_SECONDS).stdout
    try:
        devices = json.loads(listing or "{}").get("loopdevices") or []
    except json.JSONDecodeError:
        return []
    return [item for item in devices if isinstance(item, dict)]


def backing_loops(backing: str) -> list[str]:
    """Loop devices backed by ``backing`` (also once deleted): the only ones detached."""

    names = {backing, f"{backing} (deleted)"}
    return sorted(
        str(item.get("name"))
        for item in loop_devices()
        if str(item.get("back-file") or "") in names
    )


def open_count(dm: str) -> int:
    """The mapping's openers (``dmsetup info -c -o open``; 0 when it is gone)."""

    completed = dmsetup("info", "-c", "--noheadings", "-o", "open", dm, check=False)
    count = completed.stdout.strip()
    return int(count) if completed.returncode == 0 and count.isdigit() else 0


def unmount_stack(target: str) -> int:
    """Unmount every mount stacked on ``target`` in the host namespace (lazily when
    one is busy); how many were taken down."""

    taken = 0
    while mounted_on_host(target) and taken < 8:
        busy = host_command(["umount", target], check=False, timeout=UMOUNT_SECONDS)
        if busy.returncode:
            host_command(["umount", "-l", target], check=False, timeout=UMOUNT_SECONDS)
        taken += 1
    return taken


def foreign_holders(devnos: set[str]) -> list[dict[str, Any]]:
    """Mount namespaces other than the host's that still mount one of the mappings
    (``/proc/<pid>/mountinfo``; the Pod is hostPID), one per namespace: a container
    started after the mount with private propagation (the live window probe Pod)
    keeps a copy the host unmount never reaches -- and with it the superblock and
    the mapping's opener. ``locked``: owned by another user namespace, so the copy
    cannot be unmounted from outside."""

    host_mnt, host_user = os.readlink("/proc/1/ns/mnt"), os.readlink("/proc/1/ns/user")
    seen: set[str] = set()
    found: list[dict[str, Any]] = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            mnt = os.readlink(proc / "ns" / "mnt")
            if mnt == host_mnt or mnt in seen:
                continue
            seen.add(mnt)
            table = mounts(proc / "mountinfo")
            hits = [item["mountpoint"] for item in table if item["devno"] in devnos]
            if not hits:
                continue
            locked = os.readlink(proc / "ns" / "user") != host_user
            comm = (proc / "comm").read_text().strip()
            found.append(
                dict(pid=int(proc.name), comm=comm, locked=locked, mountpoints=hits)
            )
        except OSError:
            continue
    return found


def evict_holders(devnos: set[str]) -> list[dict[str, Any]]:
    """Lazily unmount the mappings from every other mount namespace; the holders
    with ``evicted`` (a locked copy stays until its namespace ends)."""

    holders = foreign_holders(devnos)
    for holder in holders:
        for path in holder["mountpoints"]:
            nsenter = ["nsenter", "--target", str(holder["pid"]), "--mount", "--"]
            run([*nsenter, "umount", "-l", path], check=False, timeout=UMOUNT_SECONDS)
    left = {holder["pid"] for holder in foreign_holders(devnos)}
    for holder in holders:
        holder["evicted"] = holder["pid"] not in left
    return holders


def destroy_device(run_id: str, name: str) -> dict[str, Any]:
    """Unmount, evict foreign copies, remove the mappings (deferred while a locked
    copy holds one), detach the loop, delete the backing file and the record."""

    state = load_state(run_id)
    entry = state["devices"].get(name)
    if entry is None:
        return {"name": name, "absent": True}
    target = entry["mountpoint"]
    names = [str(dm) for dm in (entry.get("fault_dm"), entry["dm"]) if dm]
    present = [dm for dm in names if tool_present("dmsetup") and mapping_devno(dm)]
    for dm in present:
        dmsetup("resume", dm, check=False)
    if entry.get("target") == "error" and present and not entry.get("fault_dm"):
        # The whole-device error table: data back before the unmount writes.
        retarget_once(entry, error=False)
    devnos = {mapping_devno(dm) for dm in present}
    unmount_stack(target)
    holders = evict_holders(devnos) if present else []
    deferred: list[str] = []
    deadline = time.monotonic() + REMOVE_WAIT_SECONDS
    for dm in present:
        while open_count(dm) and time.monotonic() < deadline:
            time.sleep(REMOVE_POLL_SECONDS)
        if open_count(dm):
            # Removed by the kernel once the last holder closes it.
            dmsetup("remove", "--deferred", dm, check=False)
            deferred.append(dm)
        else:
            dmsetup("remove", "--retry", dm)
    detached = backing_loops(entry["backing"])
    for loop in detached:
        run(["losetup", "-d", loop], check=False, timeout=QUICK_SECONDS)
    Path(entry["backing"]).unlink(missing_ok=True)
    directory_removed = False
    if Path(target).is_dir():
        try:
            Path(target).rmdir()
            directory_removed = True
        except OSError:
            pass  # still a mount point inside a namespace that could not be evicted
    state = load_state(run_id)
    state["devices"].pop(name, None)
    save_state(run_id, state)
    result = {"name": name, "destroyed": True, "detached": detached, "holders": holders}
    return {**result, "deferred": deferred, "directory_removed": directory_removed}


def destroy_dm(run_id: str, name: str) -> dict[str, Any]:
    """Stop the device's setup and writer units, remove it, disarm its deadmen."""

    stop_unit(run_id, setup_kind(name))
    stop_unit(run_id, f"io-{name}")
    result = destroy_device(run_id, name)
    disarm_deadman(run_id, f"resume-{name}")
    disarm_deadman(run_id, f"dm-{name}")
    return result


def unload_modules(run_id: str) -> list[str]:
    """Unload each module this run loaded, once no device of the run is left."""

    state = load_state(run_id)
    if state["devices"] or not state["modules"]:
        return []
    unloaded: list[str] = []
    kept: list[str] = []
    for name in state["modules"]:
        root = SYS_MODULE / str(name)
        if name in LOADABLE_MODULES and root.exists():
            try:
                users = (root / "refcnt").read_text().strip()
            except OSError:
                users = "0"
            if users == "0":
                run(["modprobe", "-r", name], check=False, timeout=QUICK_SECONDS)
        (kept if root.exists() else unloaded).append(str(name))
    state["modules"] = kept
    save_state(run_id, state)
    return unloaded


def dm_destroy(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    result = destroy_dm(run_id, name)
    result["modules_unloaded"] = unload_modules(run_id)
    result["directories_removed"] = prune_run_directories(run_id)
    emit(result)


IO_WRITER = r"""
import mmap, os, sys, threading, time
path, threads, block, seconds, direct, span, pace = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5] == "1", int(sys.argv[6]), float(sys.argv[7])
deadline = time.monotonic() + seconds
flags = os.O_WRONLY | os.O_CREAT | (getattr(os, "O_DIRECT", 0) if direct else 0)
def writer(index):
    buffer = mmap.mmap(-1, block); buffer.write(b"g" * block)
    fd = os.open(f"{path}.{index}", flags, 0o600); offset = 0
    try:
        while time.monotonic() < deadline:
            os.pwrite(fd, buffer, offset); offset = (offset + block) % span
            if not direct: os.fsync(fd)
            if pace: time.sleep(pace)
    finally:
        os.close(fd)
workers = [threading.Thread(target=writer, args=(item,), daemon=True) for item in range(threads)]
for worker in workers: worker.start()
for worker in workers: worker.join()
"""


def io_start(arguments: argparse.Namespace) -> None:
    """A bounded O_DIRECT writer unit on the test mount (host namespaces)."""

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = ready_device(run_id, name)
    if not mounted_on_host(entry["mountpoint"]):
        raise ProbeError("the test device is not mounted on the host")
    threads = bounded(arguments.threads, 1, 64, "threads")
    block_kib = bounded(arguments.block_kib, 4, 1024, "block KiB")
    seconds = bounded(arguments.seconds, 10, 1800, "seconds")
    span = bounded(arguments.span_mib, 1, 64, "span MiB") * 1024 * 1024
    pace_ms = bounded(arguments.pace_ms, 0, 10_000, "pace ms")
    path = str(Path(entry["mountpoint"]) / "gfrs-io")
    numbers = (
        threads,
        block_kib * 1024,
        seconds,
        int(arguments.direct),
        span,
        pace_ms / 1000,
    )
    argv = [sys.executable, "-c", IO_WRITER, path, *(str(value) for value in numbers)]
    unit = start_unit(run_id, f"io-{name}", seconds, argv, "Nice=19")
    report = {
        "unit": unit,
        "threads": threads,
        "block_kib": block_kib,
        "pace_ms": pace_ms,
    }
    emit({**report, "direct": arguments.direct, "state": unit_state(f"{unit}.service")})


def io_stop(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    emit({"stopped": stop_unit(run_id, f"io-{checked_name(arguments.name)}")})


def residue(run_id: str) -> dict[str, Any]:
    """The storage this run still holds on the node (empty lists mean clean)."""

    digest = run_digest(run_id)
    prefix, root, backing = (
        f"{UNIT_PREFIX}{digest}-",
        str(MOUNT_ROOT / digest),
        f"gfrs-live-{digest}",
    )
    own = [
        item["mountpoint"]
        for item in host_mounts()
        if item["mountpoint"].startswith(root)
    ]
    dms: list[str] = []
    if tool_present("dmsetup"):
        # ``ls`` sends no uevent, so it needs no cookie and no host namespace.
        listing = run(["dmsetup", "ls"], check=False, timeout=QUICK_SECONDS).stdout
        dms = [
            line.split()[0] for line in listing.splitlines() if line.startswith(prefix)
        ]
    loops = [
        str(x.get("name"))
        for x in loop_devices()
        if backing in str(x.get("back-file") or "")
    ]
    files = sorted(str(p) for r in BACKING_ROOTS for p in (r / backing).glob("*.img"))
    listed = systemctl("list-units", "--all", "--no-legend", "--plain", f"{prefix}io-*")
    units = [line.split()[0] for line in listed.stdout.splitlines() if line.strip()]
    modules = [name for name in load_state(run_id)["modules"] if module_loaded(name)]
    directories = sorted(str(path) for path in run_directories(run_id) if path.exists())
    found = {"mounts": own, "device_mapper": dms, "loop_devices": loops, "units": units}
    return {
        **found,
        "backing_files": files,
        "modules_loaded_by_run": modules,
        "directories": directories,
    }


def residue_is_clean(value: dict[str, Any]) -> bool:
    return not any(value[key] for key in value)


def cleanup(arguments: argparse.Namespace) -> None:
    """Destroy what ``--kind`` names (``all``, ``fs-<name>``, ``dm-<name>``,
    ``resume-<name>``); the deadman's entry point and the runner's finally."""

    run_id = checked_run_id(arguments.run_id)
    kind = arguments.kind
    if kind != "all" and re.fullmatch(r"(fs|dm|resume)-[a-z]{1,8}", kind) is None:
        raise ProbeError("unsafe cleanup kind")
    actions: list[dict[str, Any]] = []
    state = load_state(run_id)
    devices = set(state["devices"])
    if kind.startswith("dm-"):
        # A device whose record never landed still has its units and deadman.
        devices.add(kind[3:])
    for name in sorted(devices):
        if kind in {"all", f"dm-{name}"}:
            actions.append({"device": destroy_dm(run_id, name)})
        elif (
            kind == f"resume-{name}"
            and name in state["devices"]
            and tool_present("dmsetup")
        ):
            dmsetup("resume", state["devices"][name]["dm"], check=False)
            actions.append({"resumed_device": name})
            disarm_deadman(run_id, kind)
    for name in list(state["filesystems"]):
        if kind in {"all", f"fs-{name}"}:
            actions.append({"filesystem": destroy_filesystem(run_id, name)})
            disarm_deadman(run_id, f"fs-{name}")
    if kind == "all" or kind.startswith("dm-"):
        unloaded = unload_modules(run_id)
        if unloaded:
            actions.append({"modules_unloaded": unloaded})
    pruned = prune_run_directories(run_id)
    if pruned:
        actions.append({"directories_removed": pruned})
    remaining = residue(run_id)
    report = {"run_id": run_id, "kind": kind, "actions": actions, "residue": remaining}
    emit({**report, "clean": residue_is_clean(remaining)})


def show_residue(arguments: argparse.Namespace) -> None:
    value = residue(checked_run_id(arguments.run_id))
    emit({"residue": value, "clean": residue_is_clean(value)})


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    def command(
        name: str, handler: Any, *, named: bool = True
    ) -> argparse.ArgumentParser:
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
        if named:
            item.add_argument("--name", required=True)
        item.set_defaults(handler=handler)
        return item

    command("residue", show_residue, named=False)
    command("cleanup", cleanup, named=False).add_argument("--kind", default="all")
    item = command("fs-create", fs_create)
    item.add_argument("--size-mib", type=int, default=2048)
    item.add_argument("--inodes", type=int, default=0)
    item.add_argument("--backing", required=True)
    item.add_argument("--deadman-seconds", type=int, required=True)
    for name, handler in (
        ("fs-unfill", fs_unfill),
        ("fs-inodes", fs_inodes),
        ("fs-mount", fs_mount),
        ("fs-destroy", fs_destroy),
        ("dm-resume", dm_resume),
        ("dm-recreate", dm_recreate),
        ("dm-destroy", dm_destroy),
        ("dm-setup", dm_setup),
        ("dm-status", dm_status),
        ("io-stop", io_stop),
    ):
        command(name, handler)
    command("fs-fill", fs_fill).add_argument("--percent", type=float, required=True)
    command("fs-remount", fs_remount).add_argument("--mode", required=True)
    command("fs-umount", fs_umount).add_argument("--rmdir", action="store_true")
    item = command("dm-create", dm_create)
    item.add_argument("--size-mib", type=int, default=2048)
    item.add_argument("--delay-ms", type=int, default=0)
    item.add_argument("--backing", required=True)
    item.add_argument("--deadman-seconds", type=int, required=True)
    item.add_argument("--load-module", action="store_true")
    item.add_argument("--fstype", default="ext4", choices=sorted(MKFS))
    item = command("dm-retarget", dm_retarget)
    item.add_argument("--target", required=True, choices=("data", "error"))
    item.add_argument("--delay-ms", type=int, default=None)
    command("dm-suspend", dm_suspend).add_argument("--seconds", type=int, required=True)
    item = command("io-start", io_start)
    item.add_argument("--threads", type=int, default=16)
    item.add_argument("--block-kib", type=int, default=64)
    item.add_argument("--seconds", type=int, required=True)
    item.add_argument("--span-mib", type=int, default=16)
    item.add_argument("--direct", action="store_true")
    item.add_argument("--pace-ms", type=int, default=0)
    return value


def error_code(exc: BaseException) -> str:
    return hashlib.sha256(str(exc).encode()).hexdigest()[:16]


def exception_site(exc: BaseException) -> str | None:
    """The innermost ``function:line`` of this file above the command helpers."""

    site = None
    frame = exc.__traceback__
    while frame is not None:
        code = frame.tb_frame.f_code
        if code.co_filename == __file__ and code.co_name not in HELPER_FRAMES:
            site = f"{code.co_name}:{frame.tb_lineno}"
        frame = frame.tb_next
    return site


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    budget = CALL_BUDGET_SECONDS.get(arguments.command)
    DEADLINE[:] = [time.monotonic() + budget] if budget else []
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        # The host fixture shows only the class, message digest and site.
        error = {
            "error": type(exc).__name__,
            "error_kind": "probe",
            "command": arguments.command,
        }
        error.update(error_class=type(exc).__name__, error_code=error_code(exc))
        emit({**error, "error_site": exception_site(exc)})
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        DEADLINE[:] = []
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
