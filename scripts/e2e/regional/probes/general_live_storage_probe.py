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
* a bounded O_DIRECT writer unit that writes only files inside that mount.

Every creation is recorded in ``storage-state.json`` before it happens and is
preceded by a ``systemd-run --on-active`` deadman that unmounts, detaches and
deletes it from a copy of this file. No raw disk, real mount, system or log
disk is ever opened.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_ROOT = Path("/run/gpu-fault-acceptance/general")
MOUNT_ROOT = Path("/mnt/gfrs-live")
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 180,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): {shlex.join(command)}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return completed


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


EMPTY_STATE: dict[str, Any] = {"units": [], "filesystems": {}, "devices": {}}


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
    """A copy of this file the deadman can run after the probe Pod is gone."""

    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / "storage-probe.py"
    if not target.is_file():
        shutil.copyfile(Path(__file__).resolve(), target)
        target.chmod(0o700)
    return target


def unit_state(unit: str) -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            unit,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=UnitFileState",
            "--property=MainPID",
            "--property=NRestarts",
            "--property=InvocationID",
            "--property=ExecMainStartTimestamp",
        ],
        check=False,
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
            "cleanup",
            "--run-id",
            run_id,
            "--kind",
            kind,
        ]
    )
    if unit_state(f"{name}.timer").get("ActiveState") != "active":
        raise ProbeError(f"deadman {name} is not armed")
    return name


def disarm_deadman(run_id: str, kind: str) -> None:
    name = unit_name(run_id, f"{DEADMAN_PREFIX}-{kind}")
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{name}.service"], check=False)
    run(
        ["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"],
        check=False,
    )


def start_unit(
    run_id: str,
    kind: str,
    seconds: int,
    command: list[str],
    *properties: str,
) -> str:
    """A transient unit that stops itself after ``seconds`` (``RuntimeMaxSec``)."""

    bounded(seconds, 10, MAX_SECONDS, "unit seconds")
    name = unit_name(run_id, kind)
    state = load_state(run_id)
    if name not in state["units"]:
        state["units"].append(name)
        save_state(run_id, state)
    run(["systemctl", "stop", f"{name}.service"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={name}",
            "--collect",
            f"--property=RuntimeMaxSec={int(seconds)}",
            *[f"--property={item}" for item in properties],
            *command,
        ]
    )
    return name


def stop_unit(run_id: str, kind: str) -> dict[str, str]:
    name = unit_name(run_id, kind)
    run(["systemctl", "stop", f"{name}.service"], check=False, timeout=120)
    run(["systemctl", "reset-failed", f"{name}.service"], check=False)
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


def mounts() -> list[dict[str, str]]:
    result = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
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


# --------------------------------------------------------------------------- #
# Test file systems and test block devices
# --------------------------------------------------------------------------- #
def loop_attach(path: Path) -> str:
    completed = run(["losetup", "--find", "--show", str(path)])
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


def fs_mount(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = checked_filesystem(run_id, name)
    source = entry.get("dm_path") or entry["loop"]
    Path(entry["mountpoint"]).mkdir(mode=0o755, parents=True, exist_ok=True)
    run(["mount", "-t", "ext4", source, entry["mountpoint"]])
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
    emit(result)


def dm_table(sectors: int, loop: str, delay_ms: int) -> str:
    if delay_ms:
        return f"0 {sectors} delay {loop} 0 {delay_ms}"
    return f"0 {sectors} linear {loop} 0"


def dm_device_name(dm: str) -> str:
    """``dm-N`` -- the name ``/proc/diskstats`` and the product report."""

    resolved = os.path.realpath(f"/dev/mapper/{dm}")
    return os.path.basename(resolved)


def dm_create(arguments: argparse.Namespace) -> None:
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
    if delay_ms and not Path("/sys/module/dm_delay").exists():
        raise ProbeError("dm_delay is not loaded on this node")
    state = load_state(run_id)
    if name in state["devices"] or name in state["filesystems"]:
        raise ProbeError("a test device of this name already exists")
    root = backing_root(arguments.backing, size_mib * 1024 * 1024)
    deadman = arm_deadman(run_id, f"dm-{name}", seconds)
    backing = run_backing_dir(run_id, root) / f"{name}.img"
    target = mountpoint(run_id, name)
    dm = dm_name(run_id, name)
    state["devices"][name] = {
        "backing": str(backing),
        "mountpoint": str(target),
        "loop": None,
        "dm": dm,
        "dm_path": f"/dev/mapper/{dm}",
        "delay_ms": delay_ms,
        "mounted": False,
    }
    save_state(run_id, state)
    run(["fallocate", "-l", f"{size_mib}M", str(backing)])
    loop = loop_attach(backing)
    state["devices"][name]["loop"] = loop
    save_state(run_id, state)
    sectors = int(run(["blockdev", "--getsz", loop]).stdout.strip())
    run(["dmsetup", "create", dm], input_text=dm_table(sectors, loop, delay_ms) + "\n")
    state["devices"][name]["sectors"] = sectors
    save_state(run_id, state)
    run(
        ["mkfs.ext4", "-q", "-F", "-m", "0", "-L", f"gfrs-{name}", f"/dev/mapper/{dm}"],
        timeout=600,
    )
    target.mkdir(mode=0o755, parents=True, exist_ok=True)
    run(["mount", "-t", "ext4", f"/dev/mapper/{dm}", str(target)])
    state["devices"][name]["mounted"] = True
    save_state(run_id, state)
    emit(
        {
            "name": name,
            "dm": dm,
            "device": dm_device_name(dm),
            "loop": loop,
            "mountpoint": str(target),
            "delay_ms": delay_ms,
            "deadman": deadman,
        }
    )


def checked_device(run_id: str, name: str) -> dict[str, Any]:
    entry = load_state(run_id)["devices"].get(checked_name(name))
    if entry is None:
        raise ProbeError("no test device of this name")
    return dict(entry)


def dm_suspend(arguments: argparse.Namespace) -> None:
    """Suspend the test device; a deadman resumes it after ``seconds``."""

    run_id = checked_run_id(arguments.run_id)
    entry = checked_device(run_id, arguments.name)
    seconds = bounded(arguments.seconds, MIN_DEADMAN_SECONDS, 180, "suspend seconds")
    deadman = arm_deadman(run_id, f"resume-{checked_name(arguments.name)}", seconds)
    run(["dmsetup", "suspend", "--nolockfs", "--noflush", entry["dm"]], timeout=60)
    emit(
        {
            "suspended": entry["dm"],
            "deadman": deadman,
            "info": run(["dmsetup", "info", entry["dm"]], check=False).stdout[-400:],
        }
    )


def dm_resume(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_device(run_id, arguments.name)
    run(["dmsetup", "resume", entry["dm"]], timeout=60)
    disarm_deadman(run_id, f"resume-{checked_name(arguments.name)}")
    emit({"resumed": entry["dm"]})


def dm_recreate(arguments: argparse.Namespace) -> None:
    """Remove and recreate the mapping under the same name: counters restart."""

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = checked_device(run_id, name)
    before = dm_device_name(entry["dm"])
    if entry.get("mounted"):
        run(["umount", entry["mountpoint"]], timeout=120)
    run(["dmsetup", "remove", entry["dm"]], timeout=60)
    run(
        ["dmsetup", "create", entry["dm"]],
        input_text=dm_table(
            int(entry["sectors"]), entry["loop"], int(entry["delay_ms"])
        )
        + "\n",
    )
    run(["mount", "-t", "ext4", entry["dm_path"], entry["mountpoint"]])
    emit(
        {
            "dm": entry["dm"],
            "device_before": before,
            "device_after": dm_device_name(entry["dm"]),
        }
    )


def destroy_device(run_id: str, name: str) -> dict[str, Any]:
    state = load_state(run_id)
    entry = state["devices"].get(name)
    if entry is None:
        return {"name": name, "absent": True}
    if shutil.which("dmsetup"):
        run(["dmsetup", "resume", entry["dm"]], check=False, timeout=60)
    target = entry["mountpoint"]
    if any(item["mountpoint"] == target for item in mounts()):
        run(["umount", target], check=False, timeout=120)
        if any(item["mountpoint"] == target for item in mounts()):
            run(["umount", "-l", target], timeout=120)
    if (
        shutil.which("dmsetup")
        and run(["dmsetup", "info", entry["dm"]], check=False).returncode == 0
    ):
        run(["dmsetup", "remove", "--retry", entry["dm"]], timeout=120)
    if entry.get("loop"):
        run(["losetup", "-d", entry["loop"]], check=False)
    Path(entry["backing"]).unlink(missing_ok=True)
    if Path(target).is_dir():
        Path(target).rmdir()
    del state["devices"][name]
    save_state(run_id, state)
    return {"name": name, "destroyed": True}


def dm_destroy(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    stop_unit(run_id, f"io-{name}")
    result = destroy_device(run_id, name)
    disarm_deadman(run_id, f"dm-{name}")
    disarm_deadman(run_id, f"resume-{name}")
    emit(result)


IO_WRITER = r"""
import mmap, os, sys, threading, time
path, threads, block, seconds, direct, span = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5] == "1", int(sys.argv[6])
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
    finally:
        os.close(fd)
workers = [threading.Thread(target=writer, args=(item,), daemon=True) for item in range(threads)]
for worker in workers: worker.start()
for worker in workers: worker.join()
"""


def io_start(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = checked_device(run_id, name)
    threads = bounded(arguments.threads, 1, 64, "threads")
    block_kib = bounded(arguments.block_kib, 4, 1024, "block KiB")
    seconds = bounded(arguments.seconds, 10, 1800, "seconds")
    span = bounded(arguments.span_mib, 1, 64, "span MiB") * 1024 * 1024
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
        ],
        "Nice=19",
    )
    emit(
        {
            "unit": unit,
            "threads": threads,
            "block_kib": block_kib,
            "direct": arguments.direct,
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
        for item in mounts()
        if item["mountpoint"].startswith(str(MOUNT_ROOT / digest))
    ]
    dms: list[str] = []
    if shutil.which("dmsetup"):
        listing = run(["dmsetup", "ls"], check=False).stdout
        dms = [
            line.split()[0]
            for line in listing.splitlines()
            if line.startswith(f"{UNIT_PREFIX}{digest}-")
        ]
    loops: list[str] = []
    if shutil.which("losetup"):
        listing = run(["losetup", "-J", "-l"], check=False).stdout
        try:
            devices = json.loads(listing or "{}").get("loopdevices") or []
        except json.JSONDecodeError:
            devices = []
        loops = [
            str(item.get("name"))
            for item in devices
            if f"gfrs-live-{digest}" in str(item.get("back-file") or "")
        ]
    files = sorted(
        str(path)
        for root in BACKING_ROOTS
        for path in (root / f"gfrs-live-{digest}").glob("*.img")
    )
    units = [
        line.split()[0]
        for line in run(
            [
                "systemctl",
                "list-units",
                "--all",
                "--no-legend",
                "--plain",
                f"{UNIT_PREFIX}{digest}-io-*",
            ],
            check=False,
        ).stdout.splitlines()
        if line.strip()
    ]
    return {
        "mounts": own_mounts,
        "device_mapper": dms,
        "loop_devices": loops,
        "backing_files": files,
        "units": units,
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
    for name in list(state["devices"]):
        if kind in {"all", f"dm-{name}"}:
            stop_unit(run_id, f"io-{name}")
            actions.append({"device": destroy_device(run_id, name)})
            disarm_deadman(run_id, f"resume-{name}")
            disarm_deadman(run_id, f"dm-{name}")
        elif kind == f"resume-{name}" and shutil.which("dmsetup"):
            run(["dmsetup", "resume", state["devices"][name]["dm"]], check=False)
            actions.append({"resumed_device": name})
            disarm_deadman(run_id, kind)
    for name in list(state["filesystems"]):
        if kind in {"all", f"fs-{name}"}:
            actions.append({"filesystem": destroy_filesystem(run_id, name)})
            disarm_deadman(run_id, f"fs-{name}")
    if kind == "all":
        for root in BACKING_ROOTS:
            directory = root / f"gfrs-live-{run_digest(run_id)}"
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        own = MOUNT_ROOT / run_digest(run_id)
        if own.is_dir() and not any(own.iterdir()):
            own.rmdir()
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

    def command(name: str, handler: Any) -> argparse.ArgumentParser:
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
        item.set_defaults(handler=handler)
        return item

    command("residue", show_residue)
    command("cleanup", cleanup).add_argument("--kind", default="all")
    item = command("fs-create", fs_create)
    item.add_argument("--name", required=True)
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
        ("io-stop", io_stop),
    ):
        command(name, handler).add_argument("--name", required=True)
    item = command("fs-fill", fs_fill)
    item.add_argument("--name", required=True)
    item.add_argument("--percent", type=float, required=True)
    item = command("fs-remount", fs_remount)
    item.add_argument("--name", required=True)
    item.add_argument("--mode", required=True)
    item = command("fs-umount", fs_umount)
    item.add_argument("--name", required=True)
    item.add_argument("--rmdir", action="store_true")
    item = command("dm-create", dm_create)
    item.add_argument("--name", required=True)
    item.add_argument("--size-mib", type=int, default=2048)
    item.add_argument("--delay-ms", type=int, default=0)
    item.add_argument("--backing", required=True)
    item.add_argument("--deadman-seconds", type=int, required=True)
    item = command("dm-suspend", dm_suspend)
    item.add_argument("--name", required=True)
    item.add_argument("--seconds", type=int, required=True)
    item = command("io-start", io_start)
    item.add_argument("--name", required=True)
    item.add_argument("--threads", type=int, default=16)
    item.add_argument("--block-kib", type=int, default=64)
    item.add_argument("--seconds", type=int, required=True)
    item.add_argument("--span-mib", type=int, default=16)
    item.add_argument("--direct", action="store_true")
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
                "error_class": type(exc).__name__,
                "command": arguments.command,
            }
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
