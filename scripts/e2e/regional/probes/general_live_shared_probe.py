#!/usr/bin/env python3
"""Shared-storage half of the GF-GENERAL-LIVE host probe: test NFS client mounts.

Runs on the target node through ``host_probe_fixture.py`` (chroot into the
host, the node's own Python), standard library only, one JSON object on the
last stdout line -- the sibling of ``general_live_storage_probe.py`` for
GF-GENERAL-LIVE-005.

It mounts the test export served by ``shared_export_fixture.py`` (an NFSv4
server Pod) under ``/mnt/gfrs-live/<digest>/<name>`` **in PID 1's mount
namespace** through ``nsenter``: that is the table the host collector's
``/proc/self/mountinfo`` mirrors, where ``_shared_filesystems`` discovers every
``nfs``/``nfs4``/``lustre`` mount by file-system type -- the product has no
shared-filesystem setting to append to. The runner's own readings of the mount
run there too (``host_python``): a ``statvfs`` bounded by a worker thread so a
hung export cannot wedge the probe, a write/fsync/read-back proof, a fill to a
target percentage.

Every mount is recorded in ``shared-state.json`` before it is made and
preceded by a ``systemd-run --on-active`` deadman that unmounts it (``umount``,
then ``-f``, then ``-l``) from a copy of this file. Only an IPv4 literal server
and an export path of the test shape are mounted, always ``soft`` with a short
``timeo`` so a dead export answers EIO instead of parking threads forever, and
nothing outside ``/mnt/gfrs-live/<digest>/`` is touched. ``--load-module``
(the KERNEL_MODULE_LOAD approval) loads a missing ``nfsv4``, recorded first so
``cleanup`` unloads it once no mount of the run is left.
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
HOST_MOUNTINFO = Path("/proc/1/mountinfo")
SYS_MODULE = Path("/sys/module")
#: The NFS client module ``mount -t nfs4`` needs; the only one this probe loads.
NFS_MODULE = "nfsv4"
#: Mount and readings happen in PID 1's mount (and network) namespaces.
HOST_NAMESPACES = ("nsenter", "--target", "1", "--mount", "--net", "--")
#: Soft, short-timeout NFSv4.1 over TCP: a paused or dead export fails a call
#: after about ``timeo`` x ``retrans`` deciseconds instead of hanging forever,
#: and attributes expire in 2 s so a server-side change is seen promptly.
DEFAULT_OPTIONS = (
    "vers=4.1,proto=tcp,port=2049,soft,timeo=100,retrans=2,actimeo=2,"
    "noatime,nosuid,nodev,noexec"
)
QUICK_SECONDS = 30
MOUNT_SECONDS = 90
UMOUNT_SECONDS = 60
UNIT_TIMEOUT_STOP_SECONDS = 20
UNIT_STOP_SECONDS = 2 * UNIT_TIMEOUT_STOP_SECONDS + 20
#: Bounded readings: the statvfs worker's default deadline (the product's own
#: is 5 s) and the ceiling of any single reading subprocess.
STAT_TIMEOUT_SECONDS = 5.0
READING_MAX_SECONDS = 180
UNIT_PREFIX = "gfrs-"
#: The node probe owns ``-deadman-``, the storage probe ``-sdeadman-``.
DEADMAN_PREFIX = "ndeadman"
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
SAFE_NAME = re.compile(r"^[a-z]{1,8}$")
IPV4 = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$"
)
SAFE_EXPORT = re.compile(r"^/[a-z][a-z0-9/_-]{0,40}$")
SAFE_OPTIONS = re.compile(r"^[a-z0-9=.,_-]{1,200}$")
MIN_DEADMAN_SECONDS = 60
MAX_DEADMAN_SECONDS = 5400
MAX_IO_MIB = 64
RUNNING_STATES = frozenset({"active", "activating", "deactivating", "reloading"})


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
    timeout: float = 180,
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


def tool_present(name: str) -> bool:
    return shutil.which(name) is not None


def host_command(
    command: list[str], *, check: bool = True, timeout: float = QUICK_SECONDS
) -> subprocess.CompletedProcess[str]:
    """``command`` in PID 1's mount and network namespaces (the Pod is hostPID)."""

    if not tool_present("nsenter"):
        raise ProbeError("nsenter is missing: mount commands cannot reach the host")
    return run([*HOST_NAMESPACES, *command], check=check, timeout=timeout)


def host_python(snippet: str, *arguments: str, timeout: float) -> dict[str, Any]:
    """Run ``snippet`` with this interpreter in the host namespaces; its JSON.

    A reading that outlives ``timeout`` is reported as ``timed_out`` rather
    than raised: the snippets bound themselves with worker threads, this is the
    backstop for a process stuck in an uninterruptible NFS call.
    """

    timeout = min(float(timeout), float(READING_MAX_SECONDS))
    try:
        completed = host_command(
            [sys.executable, "-I", "-c", snippet, *arguments],
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"timed_out": True, "elapsed": timeout, "error": "TimeoutExpired"}
    lines = completed.stdout.strip().splitlines()
    try:
        value = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        value = None
    if not isinstance(value, dict):
        return {
            "error": f"reading failed ({completed.returncode})",
            "stderr": completed.stderr.strip()[-300:],
        }
    return value


def checked_run_id(value: str) -> str:
    if SAFE_RUN_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def checked_name(value: str) -> str:
    if SAFE_NAME.fullmatch(value) is None:
        raise ProbeError("unsafe mount name")
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


EMPTY_STATE: dict[str, Any] = {"mounts": {}, "modules": []}


def load_state(run_id: str) -> dict[str, Any]:
    path = state_dir(run_id) / "shared-state.json"
    if not path.is_file():
        return {"run_id": run_id, **json.loads(json.dumps(EMPTY_STATE))}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ProbeError("shared probe state identity mismatch")
    for key, empty in EMPTY_STATE.items():
        value.setdefault(key, json.loads(json.dumps(empty)))
    return value


def save_state(run_id: str, state: dict[str, Any]) -> None:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "shared-state.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def probe_copy(run_id: str) -> Path:
    """A copy of this file the deadman runs, refreshed whenever this differs."""

    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / "shared-probe.py"
    source = Path(__file__).resolve().read_bytes()
    if not target.is_file() or target.read_bytes() != source:
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(source)
        temporary.chmod(0o700)
        os.replace(temporary, target)
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
            "--property=MainPID",
        ],
        check=False,
        timeout=QUICK_SECONDS,
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
    run(["systemctl", "stop", f"{name}.timer"], check=False, timeout=QUICK_SECONDS)
    run(
        ["systemctl", "reset-failed", f"{name}.service"],
        check=False,
        timeout=QUICK_SECONDS,
    )
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
    run(["systemctl", "stop", f"{name}.timer"], check=False, timeout=QUICK_SECONDS)
    if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
        run(
            ["systemctl", "stop", f"{name}.service"],
            check=False,
            timeout=UNIT_STOP_SECONDS,
        )
    run(
        ["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"],
        check=False,
        timeout=QUICK_SECONDS,
    )


def mounts(table: Path | None = None) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    try:
        text = (HOST_MOUNTINFO if table is None else table).read_text()
    except OSError:
        return result
    for line in text.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if len(fields) <= separator + 2:
            continue
        result.append(
            {
                "mountpoint": fields[4].replace("\\040", " "),
                "fstype": fields[separator + 1],
                "source": fields[separator + 2],
                "options": fields[5],
            }
        )
    return result


def host_mount_entry(target: str) -> dict[str, str] | None:
    for item in mounts():
        if item["mountpoint"] == target:
            return item
    return None


def module_loaded(name: str) -> bool:
    return (SYS_MODULE / name).exists()


# --------------------------------------------------------------------------- #
# Readings, run in the host namespaces
# --------------------------------------------------------------------------- #
#: ``statvfs`` on a worker thread: ``timed_out`` after the deadline, the error
#: class and errno otherwise, so a hung export is a reading, not a hang.
STATVFS_SNIPPET = r"""
import json, os, queue, sys, threading, time
path, deadline = sys.argv[1], float(sys.argv[2])
answers = queue.SimpleQueue()
def work():
    try:
        answers.put(("ok", os.statvfs(path)))
    except OSError as exc:
        answers.put(("error", f"{type(exc).__name__}: {exc.errno}"))
threading.Thread(target=work, daemon=True).start()
started = time.monotonic()
try:
    kind, value = answers.get(timeout=deadline)
except queue.Empty:
    print(json.dumps({"path": path, "timed_out": True, "elapsed": time.monotonic() - started}), flush=True)
    os._exit(0)  # the parked worker thread must not keep this process alive
elapsed = time.monotonic() - started
if kind == "error":
    print(json.dumps({"path": path, "error": value, "elapsed": elapsed}), flush=True)
else:
    total = value.f_blocks * value.f_frsize
    available = value.f_bavail * value.f_frsize
    print(json.dumps({
        "path": path, "elapsed": elapsed, "total_bytes": total, "available_bytes": available,
        "used_percent": 100 * (total - available) / total if total else None,
        "inodes_total": value.f_files, "read_only": bool(value.f_flag & os.ST_RDONLY),
    }), flush=True)
os._exit(0)
"""

#: Write ``mib`` MiB, fsync, read back, compare: the proof of fresh real I/O.
IO_SNIPPET = r"""
import hashlib, json, os, sys, time
root, mib, keep = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
directory = os.path.join(root, "gfrs-io")
path = os.path.join(directory, f"proof-{os.getpid()}.bin")
block = b"g" * (1 << 20)
started = time.monotonic()
result = {"path": path, "bytes": mib << 20, "ok": False}
try:
    os.makedirs(directory, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    digest = hashlib.sha256()
    try:
        for _ in range(mib):
            os.write(fd, block)
            digest.update(block)
        os.fsync(fd)
    finally:
        os.close(fd)
    check = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            check.update(chunk)
    result["sha256_match"] = check.hexdigest() == digest.hexdigest()
    result["ok"] = bool(result["sha256_match"])
    if not keep:
        os.unlink(path)
except OSError as exc:
    result["error"] = f"{type(exc).__name__}: {exc.errno}"
    result["errno"] = exc.errno
result["elapsed"] = time.monotonic() - started
print(json.dumps(result))
"""

#: Fill the mount to ``percent`` used (statvfs-driven, 1 MiB writes, fsync).
FILL_SNIPPET = r"""
import json, os, sys
root, percent = sys.argv[1], float(sys.argv[2])
path = os.path.join(root, "gfrs-fill.bin")
stat = os.statvfs(root)
total = stat.f_blocks * stat.f_frsize
used = total - stat.f_bavail * stat.f_frsize
wanted = max(0, int(total * percent / 100) - used)
block = b"f" * (1 << 20)
written, enospc, error = 0, False, None
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
try:
    while written + len(block) <= wanted:
        try:
            os.write(fd, block)
            written += len(block)
        except OSError as exc:
            enospc = exc.errno == 28
            error = f"{type(exc).__name__}: {exc.errno}"
            break
    try:
        os.fsync(fd)
    except OSError as exc:
        enospc = enospc or exc.errno == 28
        error = error or f"{type(exc).__name__}: {exc.errno}"
finally:
    os.close(fd)
stat = os.statvfs(root)
total = stat.f_blocks * stat.f_frsize
available = stat.f_bavail * stat.f_frsize
print(json.dumps({
    "path": path, "wanted_bytes": wanted, "fill_bytes": written, "enospc": enospc, "error": error,
    "statvfs": {"total_bytes": total, "available_bytes": available,
                "used_percent": 100 * (total - available) / total if total else None},
}))
"""

#: Remove what the fill and the I/O proof left on the mount.
UNFILL_SNIPPET = r"""
import json, os, shutil, sys
root = sys.argv[1]
removed = 0
fill = os.path.join(root, "gfrs-fill.bin")
if os.path.exists(fill):
    os.unlink(fill)
    removed += 1
proofs = os.path.join(root, "gfrs-io")
if os.path.isdir(proofs):
    removed += len(os.listdir(proofs))
    shutil.rmtree(proofs, ignore_errors=True)
stat = os.statvfs(root)
total = stat.f_blocks * stat.f_frsize
available = stat.f_bavail * stat.f_frsize
print(json.dumps({"removed": removed, "statvfs": {
    "total_bytes": total, "available_bytes": available,
    "used_percent": 100 * (total - available) / total if total else None}}))
"""


def bounded_statvfs(path: str, timeout: float = STAT_TIMEOUT_SECONDS) -> dict[str, Any]:
    reading = host_python(
        STATVFS_SNIPPET, path, str(float(timeout)), timeout=timeout + 15
    )
    reading.setdefault("path", path)
    return reading


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def checked_mount(run_id: str, name: str) -> dict[str, Any]:
    entry = load_state(run_id)["mounts"].get(checked_name(name))
    if entry is None:
        raise ProbeError("no test shared mount of this name")
    return dict(entry)


def nfs_mount(arguments: argparse.Namespace) -> None:
    """Mount ``<server>:<export>`` under the run's mount root in the host namespace.

    Deadman first, record second, mount third; the mount is confirmed in PID
    1's table (type ``nfs4``) and read once with the bounded ``statvfs``.
    """

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    server = str(arguments.server)
    export = str(arguments.export)
    options = str(arguments.options or DEFAULT_OPTIONS)
    if IPV4.fullmatch(server) is None:
        raise ProbeError("the export server must be an IPv4 literal")
    if SAFE_EXPORT.fullmatch(export) is None:
        raise ProbeError("unsafe export path")
    if SAFE_OPTIONS.fullmatch(options) is None or "soft" not in options.split(","):
        raise ProbeError("mount options must be a soft NFS option list")
    seconds = bounded(
        arguments.deadman_seconds,
        MIN_DEADMAN_SECONDS,
        MAX_DEADMAN_SECONDS,
        "deadman seconds",
    )
    for tool in ("nsenter", "mount", "mount.nfs4", "umount", "systemd-run"):
        if not tool_present(tool):
            raise ProbeError(f"{tool} is missing on this node")
    state = load_state(run_id)
    if name in state["mounts"]:
        raise ProbeError("a test shared mount of this name already exists")
    load = not module_loaded(NFS_MODULE)
    if load and not arguments.load_module:
        raise ProbeError(f"{NFS_MODULE} is not loaded on this node")
    target = mountpoint(run_id, name)
    if host_mount_entry(str(target)) is not None:
        raise ProbeError("the mount point is already mounted on the host")
    deadman = arm_deadman(run_id, f"nfs-{name}", seconds)
    state["mounts"][name] = {
        "server": server,
        "export": export,
        "mountpoint": str(target),
        "options": options,
        "mounted": False,
    }
    if load and NFS_MODULE not in state["modules"]:
        # Recorded before the load: it was not loaded, so loaded later is ours.
        state["modules"].append(NFS_MODULE)
    save_state(run_id, state)
    if load:
        run(["modprobe", NFS_MODULE], check=False, timeout=QUICK_SECONDS)
        if not module_loaded(NFS_MODULE):
            raise ProbeError(f"{NFS_MODULE} could not be loaded")
    target.mkdir(mode=0o755, parents=True, exist_ok=True)
    host_command(
        ["mount", "-t", "nfs4", "-o", options, f"{server}:{export}", str(target)],
        timeout=MOUNT_SECONDS,
    )
    entry = host_mount_entry(str(target))
    if entry is None or not entry["fstype"].startswith("nfs"):
        raise ProbeError("the export is not mounted in the host mount table")
    state = load_state(run_id)
    state["mounts"][name]["mounted"] = True
    save_state(run_id, state)
    emit(
        {
            "name": name,
            "mountpoint": str(target),
            "server": server,
            "export": export,
            "host_mount": entry,
            "module_loaded_by_run": load,
            "statvfs": bounded_statvfs(str(target)),
            "deadman": deadman,
        }
    )


def umount_cascade(target: str) -> dict[str, Any]:
    """``umount``, then ``-f``, then ``-l``; the host table decides success."""

    steps: list[dict[str, Any]] = []
    for flags in ((), ("-f",), ("-l",)):
        if host_mount_entry(target) is None:
            break
        completed = host_command(
            ["umount", *flags, target], check=False, timeout=UMOUNT_SECONDS
        )
        steps.append({"flags": list(flags), "returncode": completed.returncode})
    return {"steps": steps, "mounted_after": host_mount_entry(target) is not None}


def nfs_umount(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = checked_mount(run_id, name)
    target = str(entry["mountpoint"])
    result = umount_cascade(target)
    if result["mounted_after"]:
        raise ProbeError("the test shared mount could not be unmounted")
    state = load_state(run_id)
    if arguments.forget:
        state["mounts"].pop(name, None)
        if Path(target).is_dir():
            Path(target).rmdir()
        disarm_deadman(run_id, f"nfs-{name}")
    else:
        state["mounts"][name]["mounted"] = False
    save_state(run_id, state)
    emit(
        {
            "name": name,
            "unmounted": target,
            "forgotten": bool(arguments.forget),
            "directory_present": Path(target).exists(),
            "umount": result,
            "host_mount": host_mount_entry(target),
        }
    )


def nfs_remount(arguments: argparse.Namespace) -> None:
    """Mount a recorded (unmounted) test share again at its mount point."""

    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    entry = checked_mount(run_id, name)
    target = str(entry["mountpoint"])
    if host_mount_entry(target) is None:
        Path(target).mkdir(mode=0o755, parents=True, exist_ok=True)
        host_command(
            [
                "mount",
                "-t",
                "nfs4",
                "-o",
                str(entry["options"]),
                f"{entry['server']}:{entry['export']}",
                target,
            ],
            timeout=MOUNT_SECONDS,
        )
    state = load_state(run_id)
    state["mounts"][name]["mounted"] = True
    save_state(run_id, state)
    emit(
        {
            "name": name,
            "mounted": target,
            "host_mount": host_mount_entry(target),
            "statvfs": bounded_statvfs(target),
        }
    )


def nfs_stat(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_mount(run_id, arguments.name)
    target = str(entry["mountpoint"])
    timeout = float(arguments.timeout)
    if not 1.0 <= timeout <= 120.0:
        raise ProbeError("stat timeout is outside 1..120")
    emit(
        {
            "name": arguments.name,
            "mountpoint": target,
            "host_mount": host_mount_entry(target),
            "statvfs": bounded_statvfs(target, timeout),
        }
    )


def nfs_io(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_mount(run_id, arguments.name)
    target = str(entry["mountpoint"])
    mib = bounded(arguments.mib, 1, MAX_IO_MIB, "I/O MiB")
    reading = host_python(
        IO_SNIPPET,
        target,
        str(mib),
        "1" if arguments.keep else "0",
        timeout=float(arguments.timeout),
    )
    emit({"name": arguments.name, "mountpoint": target, "io": reading})


def nfs_fill(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_mount(run_id, arguments.name)
    percent = float(arguments.percent)
    if not 1.0 <= percent <= 100.0:
        raise ProbeError("fill percent is outside 1..100")
    target = str(entry["mountpoint"])
    reading = host_python(
        FILL_SNIPPET, target, str(percent), timeout=READING_MAX_SECONDS
    )
    emit({"name": arguments.name, "mountpoint": target, **reading})


def nfs_unfill(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_mount(run_id, arguments.name)
    target = str(entry["mountpoint"])
    reading = host_python(UNFILL_SNIPPET, target, timeout=READING_MAX_SECONDS)
    emit({"name": arguments.name, "mountpoint": target, **reading})


def unload_modules(run_id: str) -> list[str]:
    """Unload the module this run loaded once no mount of the run is left."""

    state = load_state(run_id)
    if any(item.get("mounted") for item in state["mounts"].values()):
        return []
    unloaded: list[str] = []
    kept: list[str] = []
    for name in state["modules"]:
        if name == NFS_MODULE and module_loaded(name):
            run(["modprobe", "-r", name], check=False, timeout=QUICK_SECONDS)
        (kept if module_loaded(name) else unloaded).append(str(name))
    state["modules"] = kept
    save_state(run_id, state)
    return unloaded


def residue(run_id: str) -> dict[str, Any]:
    """The shared mounts this run still holds on the node (empty means clean)."""

    digest = run_digest(run_id)
    root = str(MOUNT_ROOT / digest)
    own = [
        item["mountpoint"]
        for item in mounts()
        if item["mountpoint"].startswith(root + "/")
        and item["fstype"].startswith("nfs")
    ]
    timers = [
        line.split()[0]
        for line in run(
            [
                "systemctl",
                "list-units",
                "--all",
                "--no-legend",
                "--plain",
                f"{UNIT_PREFIX}{digest}-{DEADMAN_PREFIX}-*.timer",
            ],
            check=False,
            timeout=QUICK_SECONDS,
        ).stdout.splitlines()
        if line.strip() and " active " in f" {line} "
    ]
    state = load_state(run_id)
    return {
        "mounts": own,
        "deadman_timers": timers,
        "mount_points": [
            str(entry["mountpoint"])
            for entry in state["mounts"].values()
            if Path(str(entry["mountpoint"])).is_dir()
        ],
        "modules_loaded_by_run": [m for m in state["modules"] if module_loaded(m)],
    }


def residue_is_clean(value: dict[str, Any]) -> bool:
    return not any(value[key] for key in value)


def cleanup(arguments: argparse.Namespace) -> None:
    """Unmount what ``--kind`` names (``all`` or ``nfs-<name>``), forget it,
    disarm its deadman, unload the run's module; the deadman's entry point."""

    run_id = checked_run_id(arguments.run_id)
    kind = str(arguments.kind)
    if kind != "all" and re.fullmatch(r"nfs-[a-z]{1,8}", kind) is None:
        raise ProbeError("unsafe cleanup kind")
    actions: list[dict[str, Any]] = []
    state = load_state(run_id)
    names = set(state["mounts"])
    if kind.startswith("nfs-"):
        names.add(kind[4:])
    for name in sorted(names):
        if kind not in {"all", f"nfs-{name}"}:
            continue
        entry = state["mounts"].get(name) or {}
        target = str(entry.get("mountpoint") or mountpoint(run_id, name))
        result = umount_cascade(target)
        if Path(target).is_dir() and not result["mounted_after"]:
            Path(target).rmdir()
        current = load_state(run_id)
        current["mounts"].pop(name, None)
        save_state(run_id, current)
        disarm_deadman(run_id, f"nfs-{name}")
        actions.append({"mount": name, "umount": result})
    unloaded = unload_modules(run_id)
    if unloaded:
        actions.append({"modules_unloaded": unloaded})
    if kind == "all":
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
    item = command("nfs-mount", nfs_mount)
    item.add_argument("--name", required=True)
    item.add_argument("--server", required=True)
    item.add_argument("--export", required=True)
    item.add_argument("--options", default="")
    item.add_argument("--deadman-seconds", type=int, required=True)
    item.add_argument("--load-module", action="store_true")
    item = command("nfs-umount", nfs_umount)
    item.add_argument("--name", required=True)
    item.add_argument("--forget", action="store_true")
    command("nfs-remount", nfs_remount).add_argument("--name", required=True)
    item = command("nfs-stat", nfs_stat)
    item.add_argument("--name", required=True)
    item.add_argument("--timeout", type=float, default=STAT_TIMEOUT_SECONDS)
    item = command("nfs-io", nfs_io)
    item.add_argument("--name", required=True)
    item.add_argument("--mib", type=int, default=4)
    item.add_argument("--keep", action="store_true")
    item.add_argument("--timeout", type=float, default=120.0)
    item = command("nfs-fill", nfs_fill)
    item.add_argument("--name", required=True)
    item.add_argument("--percent", type=float, required=True)
    command("nfs-unfill", nfs_unfill).add_argument("--name", required=True)
    return value


def error_code(exc: BaseException) -> str:
    return hashlib.sha256(str(exc).encode()).hexdigest()[:16]


def exception_site(exc: BaseException) -> str | None:
    site = None
    frame = exc.__traceback__
    while frame is not None:
        code = frame.tb_frame.f_code
        if code.co_filename == __file__ and code.co_name not in {"run", "host_command"}:
            site = f"{code.co_name}:{frame.tb_lineno}"
        frame = frame.tb_next
    return site


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
                "error_code": error_code(exc),
                "error_site": exception_site(exc),
                "command": arguments.command,
            }
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
