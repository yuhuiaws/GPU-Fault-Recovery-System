#!/usr/bin/env python3
"""Storage half of the GF-GENERAL-LIVE host probe: test file systems and devices.

Runs on the target node through ``host_probe_fixture.py`` (chroot into the
host, the node's own Python), standard library only, one JSON object on the
last stdout line -- the sibling of ``general_live_node_probe.py``, split off
because one file may not exceed the repository's size limit.

Everything it creates is test-only and named after a digest of the run id:

* a loop file system (``fallocate``-preallocated backing file, ``mkfs.ext4 -m
  0``) or a loop + device-mapper ``linear``/``delay`` device mounted under
  ``/mnt/gfrs-live/<digest>/<name>``; the backing file lives only under an
  allow-listed directory and is refused if its file system would end above
  80 % used, so filling the *test* file system never grows the system disk;
* a bounded O_DIRECT writer unit that writes only files inside that mount;
* the 004 ``probe_error`` faulting mount: the same device carrying FAT16
  (``--fstype vfat``) whose table ``dm-retarget`` swaps to the ``error``
  target built into ``dm_mod`` -- a configured mount whose ``statvfs`` fails
  EIO while it stays mounted -- and back to its data table.

Every creation is recorded in ``storage-state.json`` before it happens and is
preceded by a ``systemd-run --on-active`` deadman that unmounts, detaches and
deletes it from a copy of this file; the run's own backing directory and mount
root go with its last file system or device (``prune_run_directories``), and
``residue`` names any of them still present. No raw disk, real mount, system or
log disk is ever opened.

The device-mapper device lives entirely in the host's namespaces: ``dmsetup``
blocks create/resume/remove until the host udevd completes a libdevmapper
cookie (a SysV semaphore, per IPC namespace), which never reaches the probe
Pod's own IPC namespace -- the 2026-10-25 live ``dm-create`` ``TimeoutExpired``
on both EKS nodes. So the setup runs as a bounded transient unit (``dm-create``
returns at once; the runner polls ``dm-status``) and every other ``dmsetup``,
``mount``, ``umount`` joins PID 1's IPC and mount namespaces (``host_command``).
Each subcommand has a wall-clock budget (``CALL_BUDGET_SECONDS``) that ``run``
clamps every command to, so a call answers before the runner's exec timeout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
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
SYS_BLOCK = Path("/sys/block")
SYS_MODULE = Path("/sys/module")
#: The kernel modules this probe may load (``--load-module``, the
#: KERNEL_MODULE_LOAD approval); unloaded again by ``cleanup``. ``dm-error``
#: itself is a target built into ``dm_mod`` and needs no load.
DELAY_MODULE = "dm_delay"
VFAT_MODULE = "vfat"
LOADABLE_MODULES = (DELAY_MODULE, VFAT_MODULE)
#: File systems a test device may carry. FAT16 is the 004 ``probe_error``
#: carrier: its ``statfs`` reads the FAT on the device at the first call after
#: a mount (ext4 answers from memory), so a fresh mount swapped to ``error``
#: fails EIO; ``errors=continue`` keeps FAT from remounting read-only.
MKFS = {
    "ext4": ("mkfs.ext4", "-q", "-F", "-m", "0", "-L"),
    "vfat": ("mkfs.vfat", "-F", "16", "-n"),
}
MOUNT_OPTIONS = {"ext4": (), "vfat": ("-o", "errors=continue,noatime")}
#: How often ``dm-retarget --target error`` re-mounts and swaps when something
#: read the fresh mount's ``statfs`` first and cached the count.
RETARGET_ATTEMPTS = 3
#: Device commands run in PID 1's IPC namespace (udev completes the dmsetup
#: cookie there) and mount namespace (the collector and the writer see it).
HOST_NAMESPACES = ("nsenter", "--target", "1", "--ipc", "--mount", "--")
#: Per-command timeouts (seconds): systemd method calls give up after sd-bus's
#: 25 s; ``dmsetup``/``losetup``/``blockdev`` return in well under a second.
QUICK_SECONDS = 30
#: ``TimeoutStopSec`` of every unit this probe starts: SIGTERM, then SIGKILL
#: after the same timeout, so a ``systemctl stop`` returns within
#: ``UNIT_STOP_SECONDS``.
UNIT_TIMEOUT_STOP_SECONDS = 20
UNIT_STOP_SECONDS = 2 * UNIT_TIMEOUT_STOP_SECONDS + 20
MOUNT_SECONDS = 60
UMOUNT_SECONDS = 120
#: The setup unit's steps and their timeouts. Its ``RuntimeMaxSec`` adds
#: ``SETUP_RECORD_SECONDS`` so the worker records a failed step before
#: systemd would kill it.
SETUP_STEP_SECONDS = {
    "fallocate": 120,
    "losetup": QUICK_SECONDS,
    "blockdev": QUICK_SECONDS,
    "dmsetup-create": QUICK_SECONDS,
    "mkfs": 300,
    "mount": MOUNT_SECONDS,
}
SETUP_RECORD_SECONDS = 30
SETUP_MAX_SECONDS = sum(SETUP_STEP_SECONDS.values()) + SETUP_RECORD_SECONDS
#: Worst-case wall time of one call (seconds), per subcommand: ``main`` arms
#: the deadline and ``run`` clamps every command to what is left, so a call
#: ends (with its JSON error) inside the runner's exec timeout (budget plus
#: transport margin); ``cleanup``/``dm-destroy`` are cut at theirs and the
#: deadman finishes. Unlisted subcommands keep per-command timeouts only.
CALL_BUDGET_SECONDS = {
    "dm-create": 300,
    "dm-status": 60,
    "dm-setup": SETUP_MAX_SECONDS - SETUP_RECORD_SECONDS,
    "dm-suspend": 180,
    "dm-resume": 180,
    "dm-recreate": 300,
    "dm-retarget": 300,
    "dm-destroy": 600,
    "io-start": 180,
    "io-stop": 120,
    "cleanup": 600,
}
#: What ``unit_state`` reads back from ``systemctl show``.
UNIT_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "NRestarts",
    "InvocationID",
    "ExecMainStartTimestamp",
)
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
    command: list[str], *, check: bool = True, timeout: float = QUICK_SECONDS
) -> subprocess.CompletedProcess[str]:
    """``command`` in PID 1's IPC and mount namespaces (the Pod is hostPID).

    Only there does the host udevd's ``dmsetup udevcomplete`` wake a ``dmsetup``,
    and a mount made there is the one the collector and the writer unit see.
    """

    if not tool_present("nsenter"):
        raise ProbeError("nsenter is missing: device commands cannot reach the host")
    return run([*HOST_NAMESPACES, *command], check=check, timeout=timeout)


def dmsetup(
    *arguments: str, check: bool = True, timeout: float = QUICK_SECONDS
) -> subprocess.CompletedProcess[str]:
    return host_command(["dmsetup", *arguments], check=check, timeout=timeout)


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
    completed = systemctl(
        "show", unit, *(f"--property={key}" for key in UNIT_PROPERTIES)
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    values["returncode"] = str(completed.returncode)
    return values


def arm_deadman(run_id: str, kind: str, seconds: int) -> str:
    """Arm ``cleanup --kind <kind>`` after ``seconds``; raise if it is not armed."""

    bounded(seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman seconds")
    name = unit_name(run_id, f"{DEADMAN_PREFIX}-{kind}")
    systemctl("stop", f"{name}.timer")
    systemctl("reset-failed", f"{name}.service")
    run(
        [
            "systemd-run",
            f"--unit={name}",
            f"--on-active={int(seconds)}s",
            "--timer-property=AccuracySec=1s",
            "--property=Restart=on-failure",
            "--property=RestartSec=30s",
            "--property=StartLimitIntervalSec=0",
            f"--property=TimeoutStopSec={UNIT_TIMEOUT_STOP_SECONDS}",
            sys.executable,
            str(probe_copy(run_id)),
            "cleanup",
            "--run-id",
            run_id,
            "--kind",
            kind,
        ],
        timeout=QUICK_SECONDS,
    )
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
    run_id: str,
    kind: str,
    seconds: int,
    command: list[str],
    *properties: str,
) -> str:
    """A transient unit (host namespaces, PID 1 forks it) that stops itself
    after ``seconds``; a stop is bounded by ``UNIT_TIMEOUT_STOP_SECONDS``."""

    bounded(seconds, 10, MAX_SECONDS, "unit seconds")
    name = unit_name(run_id, kind)
    state = load_state(run_id)
    if name not in state["units"]:
        state["units"].append(name)
        save_state(run_id, state)
    systemctl("stop", f"{name}.service", timeout=UNIT_STOP_SECONDS)
    systemctl("reset-failed", f"{name}.service")
    run(
        [
            "systemd-run",
            f"--unit={name}",
            "--collect",
            f"--property=RuntimeMaxSec={int(seconds)}",
            f"--property=TimeoutStopSec={UNIT_TIMEOUT_STOP_SECONDS}",
            *[f"--property={item}" for item in properties],
            *command,
        ],
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
    return {
        "path": path,
        "used_percent": 100 * (total - available) / total if total else None,
        "inodes_used_percent": (
            100 * (stat.f_files - stat.f_ffree) / stat.f_files if stat.f_files else None
        ),
        "read_only": bool(stat.f_flag & os.ST_RDONLY),
        "total_bytes": total,
        "available_bytes": available,
        "free_inodes": stat.f_ffree,
    }


def mounts(table: Path = Path("/proc/self/mountinfo")) -> list[dict[str, str]]:
    result = []
    for line in table.read_text().splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        result.append(
            {
                "mountpoint": fields[4],
                "fstype": fields[separator + 1],
                "source": fields[separator + 2] if len(fields) > separator + 2 else "",
                "options": fields[5],
            }
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
    if root not in BACKING_ROOTS:
        raise ProbeError("backing root is not allow-listed")
    if not root.parent.exists():
        raise ProbeError("backing root's parent does not exist on this node")
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
    """The run's own directories: its backing directory under each allow-listed
    root and its mount root ``/mnt/gfrs-live/<digest>`` (made on demand)."""

    digest = run_digest(run_id)
    backing = [root / f"gfrs-live-{digest}" for root in BACKING_ROOTS]
    return [*backing, MOUNT_ROOT / digest]


def prune_run_directories(run_id: str) -> list[str]:
    """Remove the run's directories once it holds no test file system or
    device and they are empty (every destroy and ``cleanup`` end here); one
    with content stays and ``residue`` names it."""

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
    seconds = bounded(
        arguments.deadman_seconds,
        MIN_DEADMAN_SECONDS,
        MAX_DEADMAN_SECONDS,
        "deadman seconds",
    )
    state = load_state(run_id)
    if name in state["filesystems"] or name in state["devices"]:
        raise ProbeError("a test file system of this name already exists")
    root = backing_root(arguments.backing, size_mib * 1024 * 1024)
    deadman = arm_deadman(run_id, f"fs-{name}", seconds)
    backing = run_backing_dir(run_id, root) / f"{name}.img"
    target = mountpoint(run_id, name)
    state["filesystems"][name] = {
        "backing": str(backing),
        "mountpoint": str(target),
        "loop": None,
        "mounted": False,
    }
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
    emit(
        {
            "name": name,
            "loop": loop,
            "mountpoint": str(target),
            "backing": str(backing),
            "statvfs": statvfs_reading(str(target)),
            "deadman": deadman,
        }
    )


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
    removed = 0
    fill = target / "gfrs-fill.bin"
    if fill.exists():
        fill.unlink()
        removed += 1
    inode_dir = target / "gfrs-inodes"
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
    emit(
        {
            "created": created,
            "enospc": exhausted,
            "statvfs": statvfs_reading(str(target)),
        }
    )


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
    if arguments.rmdir:
        Path(entry["mountpoint"]).rmdir()
    emit(
        {
            "unmounted": entry["mountpoint"],
            "directory_present": Path(entry["mountpoint"]).exists(),
            "statvfs": statvfs_reading(entry["mountpoint"]),
        }
    )


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
    emit(
        {
            "mounted": entry["mountpoint"],
            "statvfs": statvfs_reading(entry["mountpoint"]),
        }
    )


def destroy_filesystem(run_id: str, name: str) -> dict[str, Any]:
    state = load_state(run_id)
    entry = state["filesystems"].get(name)
    if entry is None:
        return {"name": name, "absent": True}
    target = entry["mountpoint"]
    if any(item["mountpoint"] == target for item in mounts()):
        run(["umount", target], check=False, timeout=120)
        if any(item["mountpoint"] == target for item in mounts()):
            run(["umount", "-l", target], timeout=120)
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


def dm_table(sectors: int, loop: str, delay_ms: int, *, error: bool = False) -> str:
    if error:
        return f"0 {sectors} error"
    if delay_ms:
        return f"0 {sectors} delay {loop} 0 {delay_ms}"
    return f"0 {sectors} linear {loop} 0"


def dm_device_name(dm: str) -> str:
    """``dm-N`` of the mapping ``dm`` (the ``/proc/diskstats`` name the product
    reports), read from the kernel's own ``/sys/block/dm-N/dm/name``."""

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
    """Record the test device and start its bounded setup unit; return at once.

    Deadman first, record second, so ``cleanup --kind dm-<name>`` undoes
    whatever the setup reached. ``--load-module`` (KERNEL_MODULE_LOAD) loads a
    missing ``dm_delay`` or ``vfat``, recorded before the load so ``cleanup``
    unloads it. The runner polls ``dm-status`` until ``ready`` or ``failed``.
    """

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    size_mib = bounded(arguments.size_mib, MIN_FS_MIB, MAX_FS_MIB, "size MiB")
    delay_ms = bounded(arguments.delay_ms, 0, 1000, "delay ms")
    seconds = bounded(
        arguments.deadman_seconds,
        MIN_DEADMAN_SECONDS,
        MAX_DEADMAN_SECONDS,
        "deadman seconds",
    )
    fstype = str(arguments.fstype)
    if fstype not in MKFS:
        raise ProbeError("unsupported test file system type")
    load = [
        module
        for module, needed in (
            (DELAY_MODULE, bool(delay_ms)),
            (VFAT_MODULE, fstype == "vfat"),
        )
        if needed and not module_loaded(module)
    ]
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
    state["devices"][name] = {
        "backing": str(run_backing_dir(run_id, root) / f"{name}.img"),
        "mountpoint": str(mountpoint(run_id, name)),
        "loop": None,
        "dm": dm,
        "dm_path": f"/dev/mapper/{dm}",
        "delay_ms": delay_ms,
        "size_mib": size_mib,
        "fstype": fstype,
        "target": "data",
        "mounted": False,
        "phase": "setup",
        "step": None,
        "setup_unit": unit,
    }
    # Recorded before the load: it was not loaded, so loaded later is ours.
    state["modules"].extend(m for m in load if m not in state["modules"])
    save_state(run_id, state)
    for module in load:
        run(["modprobe", module], check=False, timeout=QUICK_SECONDS)
        if not module_loaded(module):
            raise ProbeError(f"{module} could not be loaded")
    command = [sys.executable, str(probe_copy(run_id)), "dm-setup"]
    start_unit(
        run_id,
        setup_kind(name),
        SETUP_MAX_SECONDS,
        [*command, "--run-id", run_id, "--name", name],
    )
    emit(
        {
            "name": name,
            "dm": dm,
            "phase": "setup",
            "setup_unit": unit,
            "setup_max_seconds": SETUP_MAX_SECONDS,
            "module_loaded_by_run": bool(load),
            "modules_loaded_by_run": load,
            "fstype": fstype,
            "delay_ms": delay_ms,
            "deadman": deadman,
            "state": unit_state(f"{unit}.service"),
        }
    )


def dm_setup(arguments: argparse.Namespace) -> None:
    """The setup unit's body (host namespaces): back, map, format, mount.

    Each step is recorded before it runs; a failure records the step, error
    class, message digest and site, and exits non-zero.
    """

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
    timeout = step("mount")
    Path(entry["mountpoint"]).mkdir(mode=0o755, parents=True, exist_ok=True)
    host_command(
        mount_command(fstype, entry["dm_path"], entry["mountpoint"]), timeout=timeout
    )
    update_device(run_id, name, mounted=True, phase="ready", step=None)
    return {
        "name": name,
        "phase": "ready",
        "device": device,
        "dm": entry["dm"],
        "loop": loop,
        "mountpoint": entry["mountpoint"],
        "fstype": fstype,
        "delay_ms": entry["delay_ms"],
    }


def dm_status(arguments: argparse.Namespace) -> None:
    """Where the device's setup stands: ``setup``, ``ready`` or ``failed``.

    The unit is read before the record: a record still at ``setup`` behind a
    unit that no longer runs means the worker died without a result --
    ``failed`` at the step it had reached. No ``error`` key: the host fixture
    would read the answer as a failed call.
    """

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    unit = unit_state(f"{unit_name(run_id, setup_kind(name))}.service")
    entry = checked_device(run_id, name)
    phase = str(entry.get("phase") or "ready")
    lost = phase == "setup" and unit.get("ActiveState") not in RUNNING_STATES
    failed = lost or phase == "failed"
    emit(
        {
            "name": name,
            "phase": "failed" if lost else phase,
            "step": entry.get("step"),
            "failed_step": (entry.get("step") or "start") if failed else None,
            "setup_error_class": (
                "SetupUnitEnded" if lost else entry.get("setup_error_class")
            ),
            "setup_error_code": entry.get("setup_error_code"),
            "setup_error_site": entry.get("setup_error_site"),
            "device": entry.get("device") if phase == "ready" else None,
            "dm": entry.get("dm"),
            "loop": entry.get("loop"),
            "mountpoint": entry.get("mountpoint"),
            "delay_ms": entry.get("delay_ms"),
            "setup_unit": {
                key: unit.get(key) for key in ("LoadState", "ActiveState", "SubState")
            },
        }
    )


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
    emit(
        {
            "suspended": entry["dm"],
            "deadman": deadman,
            "info": dmsetup("info", entry["dm"], check=False).stdout[-400:],
        }
    )


def dm_resume(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_device(run_id, arguments.name)
    dmsetup("resume", entry["dm"])
    disarm_deadman(run_id, f"resume-{checked_name(arguments.name)}")
    emit({"resumed": entry["dm"]})


def dm_recreate(arguments: argparse.Namespace) -> None:
    """Remove and recreate the mapping under the same name (counters restart);
    the writer must be stopped first or the unmount fails ``EBUSY``."""

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = ready_device(run_id, name)
    writer = unit_state(f"{unit_name(run_id, f'io-{name}')}.service")
    if writer.get("ActiveState") in RUNNING_STATES:
        raise ProbeError("the writer still runs; stop it before the recreate")
    before = dm_device_name(entry["dm"])
    if mounted_on_host(entry["mountpoint"]):
        host_command(["umount", entry["mountpoint"]], timeout=UMOUNT_SECONDS)
    dmsetup("remove", entry["dm"])
    table = dm_table(int(entry["sectors"]), entry["loop"], int(entry["delay_ms"]))
    dmsetup("create", entry["dm"], "--table", table)
    fstype = str(entry.get("fstype") or "ext4")
    host_command(
        mount_command(fstype, entry["dm_path"], entry["mountpoint"]),
        timeout=MOUNT_SECONDS,
    )
    after = dm_device_name(entry["dm"])
    update_device(run_id, name, device=after, mounted=True, target="data")
    emit({"dm": entry["dm"], "device_before": before, "device_after": after})


def retarget_once(entry: dict[str, Any], *, error: bool) -> dict[str, Any]:
    """Load ``error`` or the data table under the mounted device; its statvfs.

    For ``error`` a FAT16 carrier is re-mounted first (no free-cluster count in
    memory) and synced before the swap, so nothing is written to the error
    table: a failed write would log ``Buffer I/O error``, which the node log
    collector's storage rule grades QUARANTINE -- not the fault under test.
    """

    fstype = str(entry.get("fstype") or "ext4")
    target = str(entry["mountpoint"])
    if error and fstype == "vfat":
        if mounted_on_host(target):
            host_command(["umount", target], timeout=UMOUNT_SECONDS)
        host_command(
            mount_command(fstype, entry["dm_path"], target), timeout=MOUNT_SECONDS
        )
        os.sync()
    table = dm_table(
        int(entry["sectors"]), entry["loop"], int(entry["delay_ms"]), error=error
    )
    dmsetup("suspend", "--nolockfs", "--noflush", entry["dm"])
    dmsetup("reload", entry["dm"], "--table", table)
    dmsetup("resume", entry["dm"])
    return statvfs_reading(target)


def dm_retarget(arguments: argparse.Namespace) -> None:
    """Swap the mounted test device between its data table and ``error``.

    ``--target error`` is the 004 ``probe_error`` injection: the mount stays
    (no MOUNT_MISSING), read-write in the host table (no READ_ONLY), and every
    ``statvfs`` fails EIO until ``--target data`` puts the data table back.
    When something read the fresh mount first and cached the count, the swap
    is retried from a new mount; ``faulted`` says whether the probe's own
    reading failed. ``--delay-ms`` gives the data table a new delay first --
    0 takes the injected latency out in place (same dm-N, mount and files),
    the 006 ``sustained_latency`` recovery proof.
    """

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
    writer = unit_state(f"{unit_name(run_id, f'io-{name}')}.service")
    if writer.get("ActiveState") in RUNNING_STATES:
        raise ProbeError("the writer still runs; stop it before the retarget")
    error = arguments.target == "error"
    readings: list[dict[str, Any]] = []
    for _attempt in range(RETARGET_ATTEMPTS if error else 1):
        readings.append(retarget_once(entry, error=error))
        if not error or "error" in readings[-1]:
            break
        retarget_once(entry, error=False)
    update_device(run_id, name, target=arguments.target)
    emit(
        {
            "name": name,
            "dm": entry["dm"],
            "target": arguments.target,
            "delay_ms": int(entry["delay_ms"]),
            "attempts": len(readings),
            "statvfs": readings[-1],
            "faulted": error and "error" in readings[-1],
            "mounted_on_host": mounted_on_host(str(entry["mountpoint"])),
        }
    )


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
    """Loop devices backed by ``backing`` (also once it is deleted): only these
    are ever detached, never a loop device that a recorded name now points at."""

    names = {backing, f"{backing} (deleted)"}
    return sorted(
        str(item.get("name"))
        for item in loop_devices()
        if str(item.get("back-file") or "") in names
    )


def destroy_device(run_id: str, name: str) -> dict[str, Any]:
    state = load_state(run_id)
    entry = state["devices"].get(name)
    if entry is None:
        return {"name": name, "absent": True}
    dm = entry["dm"]
    if tool_present("dmsetup"):
        dmsetup("resume", dm, check=False)
        if (
            entry.get("target") == "error"
            and dmsetup("info", dm, check=False).returncode == 0
        ):
            # Back on the data table before the unmount writes its FAT flags.
            retarget_once(entry, error=False)
    target = entry["mountpoint"]
    if mounted_on_host(target):
        host_command(["umount", target], check=False, timeout=UMOUNT_SECONDS)
        if mounted_on_host(target):
            host_command(["umount", "-l", target], timeout=UMOUNT_SECONDS)
    if tool_present("dmsetup") and dmsetup("info", dm, check=False).returncode == 0:
        dmsetup("remove", "--retry", dm)
    detached = backing_loops(entry["backing"])
    for loop in detached:
        run(["losetup", "-d", loop], check=False, timeout=QUICK_SECONDS)
    Path(entry["backing"]).unlink(missing_ok=True)
    if Path(target).is_dir():
        Path(target).rmdir()
    state = load_state(run_id)
    state["devices"].pop(name, None)
    save_state(run_id, state)
    return {"name": name, "destroyed": True, "detached": detached}


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
    buffer = mmap.mmap(-1, block)
    buffer.write(b"g" * block)
    fd = os.open(f"{path}.{index}", flags, 0o600)
    offset = 0
    try:
        while time.monotonic() < deadline:
            os.pwrite(fd, buffer, offset)
            offset = (offset + block) % span
            if not direct:
                os.fsync(fd)
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
    unit = start_unit(
        run_id,
        f"io-{name}",
        seconds,
        [
            sys.executable,
            "-c",
            IO_WRITER,
            path,
            str(threads),
            str(block_kib * 1024),
            str(seconds),
            "1" if arguments.direct else "0",
            str(span),
            str(pace_ms / 1000),
        ],
        "Nice=19",
    )
    emit(
        {
            "unit": unit,
            "threads": threads,
            "block_kib": block_kib,
            "direct": arguments.direct,
            "pace_ms": pace_ms,
            "state": unit_state(f"{unit}.service"),
        }
    )


def io_stop(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    emit({"stopped": stop_unit(run_id, f"io-{checked_name(arguments.name)}")})


def residue(run_id: str) -> dict[str, Any]:
    """The storage this run still holds on the node (empty lists mean clean)."""

    digest = run_digest(run_id)
    own_mounts = [
        item["mountpoint"]
        for item in host_mounts()
        if item["mountpoint"].startswith(str(MOUNT_ROOT / digest))
    ]
    dms: list[str] = []
    if tool_present("dmsetup"):
        # ``ls`` sends no uevent, so it needs no cookie and no host namespace.
        listing = run(["dmsetup", "ls"], check=False, timeout=QUICK_SECONDS).stdout
        dms = [
            line.split()[0]
            for line in listing.splitlines()
            if line.startswith(f"{UNIT_PREFIX}{digest}-")
        ]
    loops = [
        str(item.get("name"))
        for item in loop_devices()
        if f"gfrs-live-{digest}" in str(item.get("back-file") or "")
    ]
    files = sorted(
        str(path)
        for root in BACKING_ROOTS
        for path in (root / f"gfrs-live-{digest}").glob("*.img")
    )
    listed = systemctl(
        "list-units", "--all", "--no-legend", "--plain", f"{UNIT_PREFIX}{digest}-io-*"
    )
    units = [line.split()[0] for line in listed.stdout.splitlines() if line.strip()]
    modules = [name for name in load_state(run_id)["modules"] if module_loaded(name)]
    return {
        "mounts": own_mounts,
        "device_mapper": dms,
        "loop_devices": loops,
        "backing_files": files,
        "units": units,
        "modules_loaded_by_run": modules,
        "directories": sorted(
            str(path) for path in run_directories(run_id) if path.exists()
        ),
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
    emit(
        {
            "run_id": run_id,
            "kind": kind,
            "actions": actions,
            "residue": remaining,
            "clean": residue_is_clean(remaining),
        }
    )


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
    """The innermost ``function:line`` of this file on the exception's path,
    above the command helpers -- the step that ran the failing command."""

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
        emit(
            {
                "error": type(exc).__name__,
                "error_kind": "probe",
                "error_class": type(exc).__name__,
                "error_code": error_code(exc),
                "error_site": exception_site(exc),
                "command": arguments.command,
            }
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        DEADLINE[:] = []
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
