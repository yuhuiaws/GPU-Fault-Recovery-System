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
``timeo``, and nothing outside ``/mnt/gfrs-live/<digest>/`` is touched. The
soft option is not what bounds a reading: the live smoke test saw a paused
export fail neither ``statvfs`` within 185 s nor writes within 125 s under
``soft,timeo=100,retrans=2`` (the NFSv4.1 session keeps retrying), so every
reading here runs on a worker thread with its own deadline and reports
``timed_out`` -- like the product's own 5 s bound -- and the option only keeps
an export that is gone for good from parking threads forever. ``--load-module``
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
from collections.abc import Iterable
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
#: Soft, short-timeout NFSv4.1 over TCP. ``soft``/``timeo``/``retrans`` keep a
#: dead export from parking threads forever; they do *not* bound a reading
#: against a paused export (live: no EIO within 185 s) -- the readers' own
#: deadlines do. Attributes expire in 2 s so a server-side change is seen
#: promptly.
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
    """A refused or failed action, reported on the probe's failure line.

    ``reason``: a fixed identifier from this file's closed vocabulary
    (``fill_timeout``, ``fill_unreadable``, ``unfill_timeout``,
    ``unfill_unreadable``, ``unfill_failed``) the host fixture shows beside the
    class, digest and site; ``detail``: JSON the fixture keeps, redacted, in the
    case's ``probe-error.json`` (the reading that was refused).
    """

    def __init__(self, message: str, reason: str = "", detail: Any = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.detail = detail


class MountError(ProbeError):
    """``mount -t nfs4`` failed; ``reason`` is one constant sentence below
    (``error_kind: nfs_mount``'s shown ``message``, not a reason token)."""

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message, reason)


#: The shape the runner lets through for ``error_kind: nfs_mount`` (the same
#: rule as the recovery probe's refusals): no '=', no newline, <= 160 chars.
CONSTANT_MESSAGE = re.compile(r"[A-Za-z0-9 ,;:'()./_-]{1,160}")
#: ``mount.nfs(8)`` stderr fragments -> the one constant sentence the runner
#: shows. The stderr itself (server address, mount point) never leaves the node.
#: Live GF-GENERAL-LIVE-005 attempt 5 stopped six variants on a bare digest; the
#: first entry is what it hid: the server answered the path walk of the export
#: with NFS4ERR_NOENT, which NFS-Ganesha returns for a pseudo path whose export
#: root never initialised or that the client may not see.
MOUNT_FAILURE_REASONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("reason given by server: no such file or directory",),
        "export server answered ENOENT for the export path "
        "(export not initialised or hidden; read the export Pod log)",
    ),
    (
        ("access denied by server", "permission denied"),
        "export server denied access to the export path",
    ),
    (
        ("connection refused",),
        "TCP/2049 connection refused at the export server address",
    ),
    (
        ("connection timed out", "timed out"),
        "TCP/2049 connection to the export server timed out "
        "(security group or network policy)",
    ),
    (
        ("no route to host", "network is unreachable"),
        "no route from the node to the export server address",
    ),
    (
        ("operation not permitted",),
        "mount(2) refused: the probe lacks mount privileges in the host namespace",
    ),
    (
        ("unknown filesystem type", "wrong fs type", "no such device"),
        "nfs4 filesystem type unavailable: nfsv4 module or mount helper missing",
    ),
    (
        ("protocol not supported", "not supported"),
        "NFSv4.1 over TCP is not supported by the export server",
    ),
    (
        ("incorrect mount option", "bad option"),
        "mount options rejected by the kernel NFS client",
    ),
    (
        ("stale file handle",),
        "export server answered a stale file handle for the export path",
    ),
)
MOUNT_FAILURE_UNCLASSIFIED = "mount.nfs4 failed for an unclassified reason"


def classify_mount_failure(stderr: str) -> str:
    """The constant sentence for a ``mount.nfs4`` stderr; never its text."""

    text = " ".join(stderr.casefold().split())
    for needles, sentence in MOUNT_FAILURE_REASONS:
        if any(needle in text for needle in needles):
            return sentence
    return MOUNT_FAILURE_UNCLASSIFIED


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
#: The reading is ``{"fill": {...}, "statvfs": {...}}``: ``fill`` records what
#: was wanted and written and, only when a call failed, the first failure's
#: class (``error``, an identifier), ``errno`` and whether it was ENOSPC. A
#: fill that stops short is a reading the runner grades (INCONCLUSIVE with the
#: reached percentage), not a failed call. Nothing of the snippet's own sits at
#: the top level: ``nfs_fill`` spreads this reading into its result, and the
#: host fixture reads any top-level ``error`` key -- even ``null`` -- as a failed
#: call (live 2026-10-26 the first fill ever reached ended 005 ``capacity_full``
#: on "host probe failed (exit 0); output withheld" for exactly that).
FILL_SNIPPET = r"""
import json, os, sys
root, percent = sys.argv[1], float(sys.argv[2])
block = b"f" * (1 << 20)
fill = {"path": os.path.join(root, "gfrs-fill.bin"), "wanted_bytes": 0, "fill_bytes": 0, "enospc": False}
result = {"fill": fill}
def usage():
    stat = os.statvfs(root)
    total = stat.f_blocks * stat.f_frsize
    available = stat.f_bavail * stat.f_frsize
    return {"total_bytes": total, "available_bytes": available,
            "used_percent": 100 * (total - available) / total if total else None}
def failed(exc):
    fill.setdefault("error", type(exc).__name__)
    fill.setdefault("errno", exc.errno)
    fill["enospc"] = fill["enospc"] or exc.errno == 28
try:
    before = usage()
    fill["wanted_bytes"] = max(
        0, int(before["total_bytes"] * percent / 100) - (before["total_bytes"] - before["available_bytes"]))
    fd = os.open(fill["path"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        while fill["fill_bytes"] + len(block) <= fill["wanted_bytes"]:
            try:
                fill["fill_bytes"] += os.write(fd, block)
            except OSError as exc:
                failed(exc)
                break
        try:
            os.fsync(fd)
        except OSError as exc:
            failed(exc)
    finally:
        os.close(fd)
except OSError as exc:
    failed(exc)
try:
    result["statvfs"] = usage()
except OSError as exc:
    failed(exc)
print(json.dumps(result))
"""

#: Remove what the fill and the I/O proof left on the mount. The reading is
#: ``{"unfill": {"removed": N, ...}, "statvfs": {...}}``; a removal that fails
#: names its class and errno under ``unfill`` and ``nfs_unfill`` turns it into
#: the probe's own failure (the export would stay full).
UNFILL_SNIPPET = r"""
import json, os, shutil, sys
root = sys.argv[1]
unfill = {"removed": 0}
result = {"unfill": unfill}
def failed(exc):
    unfill.setdefault("error", type(exc).__name__)
    unfill.setdefault("errno", exc.errno)
fill = os.path.join(root, "gfrs-fill.bin")
try:
    if os.path.exists(fill):
        os.unlink(fill)
        unfill["removed"] += 1
    proofs = os.path.join(root, "gfrs-io")
    if os.path.isdir(proofs):
        unfill["removed"] += len(os.listdir(proofs))
        shutil.rmtree(proofs)
except OSError as exc:
    failed(exc)
try:
    stat = os.statvfs(root)
    total = stat.f_blocks * stat.f_frsize
    available = stat.f_bavail * stat.f_frsize
    result["statvfs"] = {"total_bytes": total, "available_bytes": available,
                         "used_percent": 100 * (total - available) / total if total else None}
except OSError as exc:
    failed(exc)
print(json.dumps(result))
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
    command = ["mount", "-t", "nfs4", "-o", options, f"{server}:{export}", str(target)]
    completed = host_command(command, check=False, timeout=MOUNT_SECONDS)
    if completed.returncode:
        stderr = completed.stderr.strip()
        raise MountError(
            f"command failed ({completed.returncode}): "
            f"{shlex.join([*HOST_NAMESPACES, *command])}: {stderr[:300]}",
            classify_mount_failure(stderr),
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


def checked_reading(reading: dict[str, Any], what: str) -> dict[str, Any]:
    """A ``host_python`` result that is no reading is this probe's own failure.

    Two shapes: the subprocess bound (``timed_out``) and a snippet that printed
    no JSON (``error``: reading failed). The fill and unfill readings are spread
    into the command's result, so they never carry a top-level ``error`` key of
    their own -- the host fixture reads that key, whatever its value, as a
    failed call. Here the failure is named instead: ``<what>_timeout`` or
    ``<what>_unreadable`` as the reason token, the reading as the detail.
    """

    if reading.get("timed_out"):
        raise ProbeError(
            f"the {what} did not finish within {READING_MAX_SECONDS} s",
            f"{what}_timeout",
            reading,
        )
    if "error" in reading:
        raise ProbeError(
            f"the {what} produced no reading", f"{what}_unreadable", reading
        )
    return reading


def nfs_fill(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_mount(run_id, arguments.name)
    percent = float(arguments.percent)
    if not 1.0 <= percent <= 100.0:
        raise ProbeError("fill percent is outside 1..100")
    target = str(entry["mountpoint"])
    reading = checked_reading(
        host_python(FILL_SNIPPET, target, str(percent), timeout=READING_MAX_SECONDS),
        "fill",
    )
    emit({"name": arguments.name, "mountpoint": target, **reading})


def nfs_unfill(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    entry = checked_mount(run_id, arguments.name)
    target = str(entry["mountpoint"])
    reading = checked_reading(
        host_python(UNFILL_SNIPPET, target, timeout=READING_MAX_SECONDS), "unfill"
    )
    unfill = reading.get("unfill")
    if isinstance(unfill, dict) and "error" in unfill:
        # The fill file (or a proof) is still on the export: the export stays
        # full, so this is the probe's failure, with the errno in the detail.
        raise ProbeError(
            f"the fill could not be removed ({unfill['error']} errno "
            f"{unfill.get('errno')})",
            "unfill_failed",
            reading,
        )
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


def run_root(run_id: str) -> Path:
    """``/mnt/gfrs-live/<digest>``, the run's mount root; its last cleanup removes it."""

    return MOUNT_ROOT / run_digest(run_id)


def _below(path: str, base: str) -> bool:
    return path == base or path.startswith(base.rstrip("/") + "/")


def active_timers(run_id: str) -> list[str]:
    """The run's deadman timers systemd lists active."""

    digest = run_digest(run_id)
    return [
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


def residue(run_id: str, names: Iterable[str] | None = None) -> dict[str, Any]:
    """What this run still holds on the node (every value empty means clean).

    ``names`` scopes the check to those test mounts -- a per-mount cleanup's own
    share: the mount (any NFS mount at or below its mount point), its deadman
    timer, its mount point directory. Without ``names`` the check is the run's:
    every NFS mount under the run's root, every active deadman timer of the run,
    every mount point directory recorded or found under the root, the module
    the run loaded, and the root directory itself (``run_dir``). Live
    2026-10-26 (``healthy_other_mount``, two mounts): the LIFO stack ran
    ``cleanup --kind nfs-other`` first, and the run-scoped check it made listed
    the still-mounted sibling, its timer and the run's module as *that*
    cleanup's residue; the verdict was forced to FAIL on a node the sibling's
    own cleanup then left clean.
    """

    root = run_root(run_id)
    state = load_state(run_id)
    scoped = None if names is None else sorted({checked_name(n) for n in names})
    if scoped is None:
        bases = [f"{root}/"]
        timers = active_timers(run_id)
        points = {str(entry["mountpoint"]) for entry in state["mounts"].values()}
        if root.is_dir():
            points.update(str(path) for path in root.iterdir() if path.is_dir())
    else:
        bases = [str(mountpoint(run_id, name)) for name in scoped]
        wanted = {
            f"{unit_name(run_id, f'{DEADMAN_PREFIX}-nfs-{name}')}.timer"
            for name in scoped
        }
        timers = [timer for timer in active_timers(run_id) if timer in wanted]
        points = set(bases)
    value: dict[str, Any] = {
        "mounts": [
            item["mountpoint"]
            for item in mounts()
            if item["fstype"].startswith("nfs")
            and any(_below(item["mountpoint"], base) for base in bases)
        ],
        "deadman_timers": timers,
        "mount_points": sorted(point for point in points if Path(point).is_dir()),
    }
    if scoped is None:
        value["modules_loaded_by_run"] = [
            name for name in state["modules"] if module_loaded(name)
        ]
        value["run_dir"] = [str(root)] if root.exists() else []
    return value


def residue_is_clean(value: dict[str, Any]) -> bool:
    return not any(value[key] for key in value)


def cleanup(arguments: argparse.Namespace) -> None:
    """Unmount what ``--kind`` names (``all`` or ``nfs-<name>``), forget it and
    disarm its deadman; the deadman's entry point.

    A per-mount cleanup answers for its own mount only (``residue(names=...)``).
    The run-level undo -- unloading the module the run loaded, removing the
    run's root directory -- belongs to the run's *last* cleanup: ``all``, or the
    per-mount cleanup after which the state records no other mount (on the
    harness's LIFO stack the first-mounted share's; a deadman firing after the
    others). The receipt says which this was (``scope``: the mounts in scope,
    ``run_level``, the mounts the run still records) and, for a cleanup that is
    not the last, what it left to the last one (``deferred``).
    """

    run_id = checked_run_id(arguments.run_id)
    kind = str(arguments.kind)
    if kind != "all" and re.fullmatch(r"nfs-[a-z]{1,8}", kind) is None:
        raise ProbeError("unsafe cleanup kind")
    actions: list[dict[str, Any]] = []
    state = load_state(run_id)
    names = set(state["mounts"])
    if kind.startswith("nfs-"):
        names.add(kind[4:])
    scope = sorted(name for name in names if kind in {"all", f"nfs-{name}"})
    for name in scope:
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
    remaining = sorted(load_state(run_id)["mounts"])
    last = kind == "all" or not remaining
    deferred: dict[str, Any] | None = None
    if last:
        unloaded = unload_modules(run_id)
        if unloaded:
            actions.append({"modules_unloaded": unloaded})
        root = run_root(run_id)
        if root.is_dir() and not any(root.iterdir()):
            root.rmdir()
            actions.append({"run_dir_removed": str(root)})
        left = residue(run_id)
    else:
        deferred = {
            "to": "the run's last cleanup",
            "remaining_mounts": remaining,
            "items": [
                "modules_loaded_by_run",
                "run_dir",
                "deadman_timers of the remaining mounts",
            ],
        }
        left = residue(run_id, names=scope)
    emit(
        {
            "run_id": run_id,
            "kind": kind,
            "scope": {
                "mounts": scope,
                "run_level": last,
                "remaining_mounts": remaining,
            },
            "actions": actions,
            "deferred": deferred,
            "residue": left,
            "clean": residue_is_clean(left),
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


def error_fields(exc: BaseException, command: str) -> dict[str, Any]:
    """The failure line: class, digest, site and reason token are what the host
    fixture may show; the message and detail go into the case's redacted
    ``probe-error.json`` (the reading the probe refused, for one)."""

    report: dict[str, Any] = {
        "error": type(exc).__name__,
        "error_kind": "probe",
        "error_class": type(exc).__name__,
        "error_code": error_code(exc),
        "error_site": exception_site(exc),
        "error_message": str(exc)[:2000],
        "error_detail": getattr(exc, "detail", None),
        "command": command,
    }
    if isinstance(exc, MountError):
        # A classified mount failure: the constant sentence may be shown.
        report["error_kind"] = "nfs_mount"
        report["message"] = exc.reason
    elif isinstance(exc, ProbeError) and exc.reason:
        report["error_reason"] = exc.reason
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        emit(error_fields(exc, arguments.command))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
