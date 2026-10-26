#!/usr/bin/env python3
"""Host probe of the CRI Pod-log source fixture (GF-GENERAL-LIVE-017 attribution).

Runs on the target GPU node through ``host_probe_fixture.py`` (a privileged
probe Pod that ``chroot``s into the host and runs this file with the node's own
Python). Standard library only, Python 3.9+, one JSON object on the last stdout
line. It gives the node log collector (``gpu-fault-log-collector.service``,
installed but disabled by default) a reviewed CRI Pod log to read for one
managed training attempt the runner created, and nothing else:

* The collector reads its training logs from ``GPU_FAULT_TRAINING_LOG_PATHS``
  (empty on this site) and binds a file under ``/var/log/pods/<namespace>_<pod>_
  <uid>/<container>/`` to that Pod UID and container name
  (``collectors/logs/node_records.py`` ``pod_identity``). ``open-window`` gives
  that one unit, through a ``/run`` drop-in with a second ``EnvironmentFile``
  (later files override ``/etc/gpu-fault/collector.env``), a glob that matches
  only this run's fixture file ``gfrs-live-<digest>.log*`` inside a container
  log directory, runtime-enables the unit and restarts it. Every other setting
  of the collector stays as installed.
* ``create-log`` creates that fixture file -- empty, 0644 -- in the REAL
  directory kubelet made for the runner's own Pod, so the UID and container name
  the product derives are genuine. Only for a Pod that no longer exists (the
  ``old_pod`` variant) may it create the directory, with the UID that Pod had.
  It never touches kubelet's ``0.log``/rotations.
* ``write`` appends reviewed records in CRI syntax, ``<RFC3339 ns> <stream>
  <P|F> <message>``, stamped with the node clock at the moment of writing: the
  product's event time. ``rotate`` renames the file to ``.1`` and creates an
  empty successor -- the logrotate rename a writer that keeps its descriptor
  survives -- and ``write --target rotated`` continues on that inode.
  ``restart-collector`` restarts the unit so a pending partial record must come
  back from the collector's persisted state.
* ``seed-log`` is ``create-log`` and ``write`` in ONE process, for a directory
  kubelet is going to remove: kubelet's container GC (``ContainerGCPeriod``,
  one minute) deletes every ``/var/log/pods`` directory whose Pod UID it no
  longer knows, so a directory recreated for an old Pod lives an unknown
  remainder of that minute. Two separate probe calls lost that race live
  (HyperPod, 2026-10-26: ``create-log`` succeeded, ``write`` found the file
  gone 6 s later). ``seed-log`` first uses the old Pod's own directory --
  recreated empty, or kubelet's real one if it is still there -- as the canary
  for the GC tick (``--await-gc-seconds``), then recreates directory, file and
  records right after the tick, so the file is there for the whole next period
  and the collector's 10 s poll is sure to visit it. The canary is this run's
  directory under this run's old Pod UID and nothing else.
* ``status`` is the runner's raw-source oracle: the fixture files' inodes and
  sizes and, read-only, what the collector's own state file records about them
  (offset, inode, a pending CRI fragment).

Every refusal is a ``ProbeError`` that names its reason with one token from the
closed vocabulary :data:`REASONS` (``fixture_gone``, ``directory_removed``,
``unit_missing`` ...). The failure line ``main`` prints carries the exception
class, the sha256 digest of the sentence, the ``function:line`` site, that
reason token and the sentence itself: the host fixture shows the first four in
the exception text and keeps the sentence, redacted, as the case's
``probe-error.json`` (``host_probe_fixture.py``). Before this, the line was
``{"error": "ProbeError", "error_kind": "probe"}`` -- a shape the fixture reads
``error_class`` from -- so the live failure read "output withheld" and nothing
else.

Everything is bounded and reversible two ways -- the runner's cleanup and a
``systemd-run --on-active`` deadman that restores the baseline from a copy of
this file even if the runner, the probe Pod and the control plane all vanish.
Nothing here forges a finding, patches a Pod or Node, writes a real Pod log,
touches the Node Agent or any other unit. ``proof_scope`` is always
``live-replay``: the log text is reviewed input; the Pod, the attempt, the
collector and the attribution chain are real.
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
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

STATE_ROOT = Path("/run/gpu-fault-applog-replay")
DROPIN_ROOT = Path("/run/systemd/system")
LOG_UNIT = "gpu-fault-log-collector.service"
POD_LOG_ROOT = Path("/var/log/pods")
#: ``GPU_FAULT_LOG_STATE_PATH`` default of ``collectors/logs/node.py``.
COLLECTOR_STATE_PATH = Path("/var/lib/gpu-fault/log-collector-state.json")
PATHS_ENV = "GPU_FAULT_TRAINING_LOG_PATHS"
INITIAL_TAIL_ENV = "GPU_FAULT_NODE_LOG_INITIAL_TAIL_BYTES"
FILE_PREFIX = "gfrs-live-"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
POD_UID = re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
#: DNS-1123 subdomain (Pod name, namespace) and label (container name).
DNS_NAME = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$")
DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")
CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
MIN_RESTORE_SECONDS = 60
MAX_RESTORE_SECONDS = 5400
MAX_RECORDS = 32
#: One record is carried in every batch until the node recovers, so it is
#: bounded here the way the collector bounds one entry.
MAX_MESSAGE_BYTES = 16 * 1024
MAX_INITIAL_TAIL_BYTES = 1 << 20
STREAMS = ("stdout", "stderr")
FLAGS = ("P", "F")
TARGETS = ("current", "rotated")
#: Where a container's processes live, by cgroup driver and hierarchy; the
#: container id is the only variable part (``kubepods.slice`` is the systemd
#: driver on cgroup v2, ``kubepods`` the cgroupfs driver, ``pids/`` cgroup v1).
CGROUP_PATTERNS = (
    "kubepods.slice/*/cri-containerd-{id}.scope/cgroup.procs",
    "kubepods.slice/*/*/cri-containerd-{id}.scope/cgroup.procs",
    "pids/kubepods.slice/*/cri-containerd-{id}.scope/cgroup.procs",
    "pids/kubepods.slice/*/*/cri-containerd-{id}.scope/cgroup.procs",
    "kubepods/*/{id}/cgroup.procs",
    "kubepods/*/*/{id}/cgroup.procs",
    "pids/kubepods/*/{id}/cgroup.procs",
    "pids/kubepods/*/*/{id}/cgroup.procs",
)
#: ``seed-log``: how long the old Pod's directory may be watched for kubelet's
#: container GC tick, at most. kubelet's ``ContainerGCPeriod`` is one minute,
#: so the runner asks for a little more than one period; nothing is created or
#: observed by the wait except this run's directory.
MAX_GC_WAIT_SECONDS = 180
GC_POLL_SECONDS = 0.5

#: The closed vocabulary of ``ProbeError.reason`` tokens. The host fixture
#: shows the token beside the class, digest and site of a failed call
#: (``host_probe_fixture.py``: ``[a-z][a-z0-9_]{2,40}``) and the runner branches
#: on it -- 017 ``old_pod`` repeats a round whose directory kubelet's GC took
#: (``fixture_gone``, ``directory_removed``) and refuses everything else -- so a
#: new refusal needs a token here before it can be raised.
REASONS = frozenset(
    {
        "input_shape",  # an argument failed its shape check (id, uid, records)
        "command_failed",  # a systemctl/systemd-run call exited non-zero
        "not_approved",  # the window needs ENABLE_NODE_LOG_COLLECTOR
        "window_open",  # an applog window for this run is already open
        "dropin_untracked",  # a replay drop-in exists that no window tracks
        "unit_missing",  # the node log collector unit is not installed
        "unit_inactive",  # the unit did not come back active
        "pod_log_root_missing",  # /var/log/pods does not exist on the node
        "deadman_inactive",  # the restore timer did not arm
        "no_window",  # no applog window is open for this run
        "window_identity",  # the window state belongs to another run
        "container_mismatch",  # the container differs from the window's glob
        "fixture_exists",  # a fixture log already exists for this Pod/container
        "directory_missing",  # the Pod's directory is gone and may not be made
        "directory_removed",  # kubelet's GC took the directory while seeding
        "fixture_gone",  # kubelet's GC took the fixture file before a write
        "no_fixture",  # no fixture log was created for this Pod/container
        "never_rotated",  # write --target rotated before any rotation
        "rotation_exists",  # the rotation target already exists
        "no_process",  # the container's cgroup lists no process
    }
)


class ProbeError(RuntimeError):
    def __init__(self, message: str, reason: str) -> None:
        if reason not in REASONS:
            raise ValueError(f"probe error reason {reason!r} is not in REASONS")
        super().__init__(message)
        #: One token from :data:`REASONS` naming the refusal. The host fixture
        #: shows it in the exception text (the sentence itself only reaches the
        #: redacted ``probe-error.json``) and the runner reads it back as
        #: ``HostProbeError.reason``.
        self.reason = reason


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def run(
    command: list[str], *, check: bool = True, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): {' '.join(command)}: "
            f"{completed.stderr.strip()[:300]}",
            "command_failed",
        )
    return completed


# --------------------------------------------------------------------------- #
# Identifiers, paths, state
# --------------------------------------------------------------------------- #
def safe_id(value: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID", "input_shape")
    return value


def digest(run_id: str) -> str:
    return hashlib.sha256(safe_id(run_id).encode()).hexdigest()[:16]


def state_dir(run_id: str) -> Path:
    return STATE_ROOT / digest(run_id)


def dropin_path(run_id: str) -> Path:
    return (
        DROPIN_ROOT / f"{LOG_UNIT}.d" / f"gpu-fault-applog-replay-{digest(run_id)}.conf"
    )


def env_file_path(run_id: str) -> Path:
    return state_dir(run_id) / "collector.env"


def deadman_unit(run_id: str) -> str:
    return f"gpu-fault-applog-replay-{digest(run_id)}"


def fixture_file_name(run_id: str) -> str:
    return f"{FILE_PREFIX}{digest(run_id)}.log"


def source_glob(container: str, run_id: str) -> str:
    """The one glob the collector reads during the window: this run's files."""

    return f"{POD_LOG_ROOT}/*/{checked_label(container)}/{fixture_file_name(run_id)}*"


def checked_restore_seconds(value: int) -> int:
    if not MIN_RESTORE_SECONDS <= int(value) <= MAX_RESTORE_SECONDS:
        raise ProbeError(
            f"restore seconds is outside {MIN_RESTORE_SECONDS}..{MAX_RESTORE_SECONDS}",
            "input_shape",
        )
    return int(value)


def checked_pod_uid(value: str) -> str:
    if POD_UID.fullmatch(value or "") is None:
        raise ProbeError("a Pod UID must be a lower-case UUID", "input_shape")
    return value


def checked_name(value: str, what: str) -> str:
    if DNS_NAME.fullmatch(value or "") is None:
        raise ProbeError(f"{what} is not a DNS-1123 name", "input_shape")
    return value


def checked_label(value: str) -> str:
    if DNS_LABEL.fullmatch(value or "") is None:
        raise ProbeError("a container name must be a DNS-1123 label", "input_shape")
    return value


def checked_container_id(value: str) -> str:
    value = (value or "").split("://")[-1]
    if CONTAINER_ID.fullmatch(value) is None:
        raise ProbeError("a container id must be 64 hex digits", "input_shape")
    return value


def pod_log_dir(namespace: str, pod_name: str, pod_uid: str) -> Path:
    """kubelet's log directory of one Pod: ``<namespace>_<name>_<uid>``."""

    return POD_LOG_ROOT / (
        f"{checked_name(namespace, 'namespace')}_{checked_name(pod_name, 'Pod name')}"
        f"_{checked_pod_uid(pod_uid)}"
    )


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


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


def load_state(run_id: str) -> dict[str, Any]:
    path = state_dir(run_id) / "window.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ProbeError("applog window identity mismatch", "window_identity")
    return value


def save_state(run_id: str, state: dict[str, Any]) -> None:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "window.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def probe_copy(run_id: str) -> Path:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / "probe.py"
    if not target.is_file():
        shutil.copyfile(Path(__file__).resolve(), target)
        target.chmod(0o700)
    return target


def log_key(pod_uid: str, container: str) -> str:
    return f"{checked_pod_uid(pod_uid)}/{checked_label(container)}"


def open_state(run_id: str) -> dict[str, Any]:
    state = load_state(run_id)
    if not state:
        raise ProbeError("no applog window is open for this run", "no_window")
    return state


# --------------------------------------------------------------------------- #
# Reviewed records and CRI syntax
# --------------------------------------------------------------------------- #
def validate_records(records: Any) -> list[dict[str, str]]:
    """Check a reviewed record set; never mutate it."""

    if not isinstance(records, list) or not records:
        raise ProbeError("reviewed records must be a non-empty list", "input_shape")
    if len(records) > MAX_RECORDS:
        raise ProbeError(
            f"at most {MAX_RECORDS} reviewed records may be written", "input_shape"
        )
    result = []
    for item in records:
        if not isinstance(item, dict):
            raise ProbeError("each reviewed record must be an object", "input_shape")
        message = item.get("message")
        if not isinstance(message, str) or not message:
            raise ProbeError(
                "a reviewed record needs a non-empty message", "input_shape"
            )
        if "\n" in message or "\r" in message:
            raise ProbeError("a reviewed record message is one line", "input_shape")
        if len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ProbeError("reviewed record message is too large", "input_shape")
        stream = str(item.get("stream") or "stdout")
        if stream not in STREAMS:
            raise ProbeError(
                f"stream {stream!r} is not stdout or stderr", "input_shape"
            )
        flag = str(item.get("flag") or "F")
        if flag not in FLAGS:
            raise ProbeError(f"CRI flag {flag!r} is not P or F", "input_shape")
        result.append({"message": message, "stream": stream, "flag": flag})
    return result


def decode_records(sha256: str, b64: str) -> list[dict[str, str]]:
    if SHA256.fullmatch(sha256 or "") is None:
        raise ProbeError(
            "a sha256 is required for the reviewed record set", "input_shape"
        )
    try:
        content = base64.b64decode(b64, validate=True)
    except (ValueError, TypeError) as error:
        raise ProbeError(
            f"records are not valid base64: {error}", "input_shape"
        ) from error
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ProbeError("reviewed record set sha256 does not match", "input_shape")
    return validate_records(json.loads(content.decode("utf-8")))


def cri_stamp(at: datetime) -> str:
    """containerd's RFC 3339 nanosecond stamp (microsecond clock, zero-padded)."""

    return at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def cri_line(record: dict[str, str], at: datetime) -> str:
    return f"{cri_stamp(at)} {record['stream']} {record['flag']} {record['message']}\n"


# --------------------------------------------------------------------------- #
# Container process identity (read-only)
# --------------------------------------------------------------------------- #
def container_init_pid(
    container_id: str,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """The host PID of a container's init process (``NSpid`` ends in 1).

    Read from the container's own cgroup, never guessed from a process name:
    the Pod annotation ``gpu-fault.io/host-pid`` the product attributes logs
    by is the workload's declaration, and this is how the runner learns the
    true value for its own test Pod.
    """

    identifier = checked_container_id(container_id)
    procs_files = [
        path
        for pattern in CGROUP_PATTERNS
        for path in cgroup_root.glob(pattern.format(id=identifier))
    ]
    pids: list[int] = []
    for path in procs_files:
        try:
            pids.extend(
                int(item) for item in path.read_text(encoding="ascii").split() if item
            )
        except OSError:
            continue
    if not pids:
        raise ProbeError("the container's cgroup lists no process", "no_process")
    rows = []
    for pid in sorted(set(pids)):
        rows.append(_process_row(proc_root, pid))
    init = [row for row in rows if row.get("nspid") and row["nspid"][-1] == 1]
    chosen = init[0] if init else rows[0]
    return {
        "container_id": identifier,
        "cgroup_procs": [str(path) for path in procs_files],
        "process_count": len(rows),
        "init_pid": chosen["pid"],
        "init": chosen,
    }


def _process_row(proc_root: Path, pid: int) -> dict[str, Any]:
    row: dict[str, Any] = {"pid": pid}
    try:
        for line in (proc_root / str(pid) / "status").read_text().splitlines():
            if line.startswith("NSpid:"):
                row["nspid"] = [int(item) for item in line.split()[1:]]
            elif line.startswith("Name:"):
                parts = line.split(None, 1)
                row["comm"] = parts[1].strip() if len(parts) > 1 else ""
        stat = (proc_root / str(pid) / "stat").read_text()
        row["start_ticks"] = int(stat[stat.rfind(")") + 2 :].split()[19])
        row["cmdline"] = (
            (proc_root / str(pid) / "cmdline")
            .read_bytes()[:4096]
            .replace(b"\0", b" ")
            .decode("utf-8", errors="replace")
            .strip()
        )
    except (OSError, ValueError, IndexError):
        row.setdefault("error", "process could not be read")
    return row


def real_pod_log_dirs(pod_uid: str) -> list[dict[str, Any]]:
    """kubelet's directories whose name ends in this UID, and their files."""

    found = []
    if not POD_LOG_ROOT.is_dir():
        return found
    for directory in sorted(POD_LOG_ROOT.iterdir()):
        if not directory.is_dir() or not directory.name.endswith(f"_{pod_uid}"):
            continue
        containers = {}
        for child in sorted(directory.iterdir()):
            if child.is_dir():
                containers[child.name] = sorted(item.name for item in child.iterdir())
        found.append({"path": str(directory), "containers": containers})
    return found


# --------------------------------------------------------------------------- #
# The collector's own view (read-only oracle)
# --------------------------------------------------------------------------- #
def collector_state_view(run_id: str) -> dict[str, Any]:
    needle = fixture_file_name(run_id)
    view: dict[str, Any] = {
        "state_path": str(COLLECTOR_STATE_PATH),
        "readable": False,
        "tracked": {},
        "pending": {},
    }
    try:
        value = json.loads(COLLECTOR_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return view
    if not isinstance(value, dict):
        return view
    view["readable"] = True
    files = value.get("training_log_files") or value.get("training_log_offsets") or {}
    if isinstance(files, dict):
        view["tracked"] = {
            key: item for key, item in files.items() if needle in str(key)
        }
    pending = value.get("training_log_pending") or {}
    if isinstance(pending, dict):
        view["pending"] = {
            key: item for key, item in pending.items() if needle in str(key)
        }
    return view


def file_view(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "inode": stat.st_ino,
        "device": stat.st_dev,
        "size": stat.st_size,
    }


# --------------------------------------------------------------------------- #
# Deadman
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
        ]
    )
    if unit_state(f"{name}.timer").get("ActiveState") != "active":
        raise ProbeError(
            "applog replay deadman timer is not active", "deadman_inactive"
        )
    return name


def disarm_deadman(run_id: str) -> None:
    name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{name}.service"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"], check=False)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def pod_log_layout() -> dict[str, int]:
    """What kubelet keeps under the root, read-only: how many directories, how
    many in kubelet's ``<namespace>_<pod>_<uid>`` shape, how many of those hold
    a ``<container>/0.log``. The product binds a CRI log to its Pod through that
    shape (``node_records.pod_identity``) and ``old_pod`` recreates an old Pod's
    directory by it; a node whose kubelet keeps Pod logs elsewhere (a
    ``podLogsDir`` kubelet setting) has no such directory and the runner says so
    before it opens a window there.
    """

    layout = {"directories": 0, "kubelet_shape": 0, "with_container_log": 0}
    if not POD_LOG_ROOT.is_dir():
        return layout
    for directory in POD_LOG_ROOT.iterdir():
        if not directory.is_dir():
            continue
        layout["directories"] += 1
        parts = directory.name.rsplit("_", 2)
        if len(parts) != 3 or POD_UID.fullmatch(parts[2]) is None:
            continue
        layout["kubelet_shape"] += 1
        try:
            children = list(directory.iterdir())
        except OSError:
            continue
        if any(child.is_dir() and (child / "0.log").exists() for child in children):
            layout["with_container_log"] += 1
    return layout


def preflight(arguments: argparse.Namespace) -> None:
    emit(
        {
            "captured_at": utc_now().isoformat(),
            "boot_id": boot_id(),
            "python": sys.version.split()[0],
            "pod_log_root": POD_LOG_ROOT.is_dir(),
            "pod_log_root_path": str(POD_LOG_ROOT),
            "pod_log_layout": pod_log_layout(),
            "collector_state": COLLECTOR_STATE_PATH.is_file(),
            "unit": unit_state(LOG_UNIT),
        }
    )


def _unit_enabled(state: dict[str, str]) -> bool:
    return state.get("UnitFileState") in {"enabled", "enabled-runtime", "static"}


def open_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    container = checked_label(arguments.container)
    restore_seconds = checked_restore_seconds(arguments.restore_seconds)
    initial_tail = int(arguments.initial_tail_bytes)
    if not 0 <= initial_tail <= MAX_INITIAL_TAIL_BYTES:
        raise ProbeError(
            f"initial tail bytes is outside 0..{MAX_INITIAL_TAIL_BYTES}", "input_shape"
        )
    if not arguments.approved:
        raise ProbeError(
            "the node log collector is disabled in production; the CRI Pod-log "
            "window needs the runner's ENABLE_NODE_LOG_COLLECTOR approval",
            "not_approved",
        )
    if load_state(run_id):
        raise ProbeError("an applog window for this run is already open", "window_open")
    dropin = dropin_path(run_id)
    if dropin.exists():
        raise ProbeError(
            "an untracked applog replay drop-in already exists", "dropin_untracked"
        )
    before = unit_state(LOG_UNIT)
    if before.get("LoadState") != "loaded":
        raise ProbeError("the node log collector unit is not installed", "unit_missing")
    if not POD_LOG_ROOT.is_dir():
        raise ProbeError(
            f"{POD_LOG_ROOT} does not exist on this node", "pod_log_root_missing"
        )

    probe_copy(run_id)
    glob = source_glob(container, run_id)
    state: dict[str, Any] = {
        "run_id": run_id,
        "unit": LOG_UNIT,
        "container": container,
        "glob": glob,
        "initial_tail_bytes": initial_tail,
        "dropin": str(dropin),
        "env_file": str(env_file_path(run_id)),
        "before": before,
        "opened_at": utc_now().isoformat(),
        "restore_seconds": restore_seconds,
        "enabled_log_collector": False,
        "boot_id": boot_id(),
        "logs": {},
    }
    save_state(run_id, state)
    deadman = arm_deadman(run_id, restore_seconds)

    lines = [f"{PATHS_ENV}={glob}"]
    if initial_tail:
        lines.append(f"{INITIAL_TAIL_ENV}={initial_tail}")
    env_file = env_file_path(run_id)
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env_file.chmod(0o644)
    dropin.parent.mkdir(parents=True, exist_ok=True)
    # A later EnvironmentFile overrides an earlier one; Environment= would be
    # overridden by /etc/gpu-fault/collector.env, which sets the same key.
    dropin.write_text(f"[Service]\nEnvironmentFile={env_file}\n", encoding="utf-8")

    if not _unit_enabled(before):
        run(["systemctl", "enable", "--runtime", LOG_UNIT], timeout=120)
        state["enabled_log_collector"] = True
        save_state(run_id, state)
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", LOG_UNIT], timeout=120)
    after = unit_state(LOG_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(
            f"{LOG_UNIT} is not active after the window opened", "unit_inactive"
        )
    emit(
        {
            **state,
            "after": after,
            "deadman_timer": unit_state(f"{deadman}.timer"),
            "proof_scope": "live-replay",
        }
    )


def inspect(arguments: argparse.Namespace) -> None:
    """Read-only: the Pod's real log directories and its container's init PID."""

    run_id = safe_id(arguments.run_id)
    pod_uid = checked_pod_uid(arguments.pod_uid)
    value: dict[str, Any] = {
        "run_id": run_id,
        "pod_uid": pod_uid,
        "directories": real_pod_log_dirs(pod_uid),
    }
    if arguments.container_id:
        value["process"] = container_init_pid(arguments.container_id)
    emit(value)


def _window_container(state: dict[str, Any], container: str) -> str:
    container = checked_label(container)
    if container != state["container"]:
        raise ProbeError(
            "the container name differs from the open window's glob",
            "container_mismatch",
        )
    return container


def _recreation_count(state: dict[str, Any], key: str) -> int:
    """How many times this Pod/container's fixture was started over.

    kubelet's GC took the recreated directory and the file with it before the
    collector read it (``old_pod``): the run may try again from scratch, but
    never while a file of the earlier round still exists.
    """

    previous = state["logs"].get(key)
    if previous is None:
        return 0
    gone = [previous["path"], *previous.get("rotated", [])]
    if any(Path(item).exists() for item in gone):
        raise ProbeError(
            "a fixture log already exists for this Pod and container",
            "fixture_exists",
        )
    return int(previous.get("recreated", 0)) + 1


def _directory_missing() -> ProbeError:
    return ProbeError(
        "the Pod's log directory does not exist; only an old Pod's "
        "directory may be created (--allow-create-dir)",
        "directory_missing",
    )


def _materialize(
    directory: Path,
    container_dir: Path,
    path: Path,
    created_dirs: list[str],
    interrupt: Callable[[str], None] | None = None,
) -> None:
    """Make the directory, the container directory and the empty fixture file.

    ``interrupt(stage)`` is a test seam called after each step; production
    passes nothing. A ``FileNotFoundError`` on the way is kubelet's GC taking
    the directory between two steps -- the reason token says so.
    """

    try:
        if not directory.is_dir():
            directory.mkdir(mode=0o755)
            created_dirs.append(str(directory))
        if interrupt is not None:
            interrupt("directory")
        if not container_dir.is_dir():
            container_dir.mkdir(mode=0o755)
            created_dirs.append(str(container_dir))
        if interrupt is not None:
            interrupt("container")
        if path.exists() or Path(str(path) + ".1").exists():
            raise ProbeError("the fixture log file already exists", "fixture_exists")
        with path.open("x", encoding="utf-8"):
            pass
        path.chmod(0o644)
    except FileNotFoundError as error:
        raise ProbeError(
            "the Pod's log directory was removed while the fixture was being "
            "created (kubelet's GC took it)",
            "directory_removed",
        ) from error


def _log_record(
    pod_uid: str,
    container: str,
    directory: Path,
    path: Path,
    created_dirs: list[str],
    recreated: int,
) -> dict[str, Any]:
    return {
        "pod_uid": checked_pod_uid(pod_uid),
        "container": container,
        "directory": str(directory),
        "path": str(path),
        "created_dirs": created_dirs,
        "rotated": [],
        "writes": 0,
        "recreated": recreated,
    }


def _append_records(path: Path, records: list[dict[str, str]]) -> list[str]:
    """Append the reviewed records at the node clock; the CRI stamps written."""

    if not path.is_file():
        raise ProbeError(
            "the fixture log file is gone (kubelet removed the directory?)",
            "fixture_gone",
        )
    stamps = []
    now = utc_now()
    try:
        with path.open("a", encoding="utf-8") as handle:
            for index, record in enumerate(records):
                at = now + timedelta(microseconds=index)
                stamps.append(cri_stamp(at))
                handle.write(cri_line(record, at))
            handle.flush()
            os.fsync(handle.fileno())
    except FileNotFoundError as error:
        raise ProbeError(
            "the fixture log file is gone (kubelet removed the directory?)",
            "fixture_gone",
        ) from error
    return stamps


def create_log(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = open_state(run_id)
    container = _window_container(state, arguments.container)
    key = log_key(arguments.pod_uid, container)
    recreated = _recreation_count(state, key)
    directory = pod_log_dir(arguments.namespace, arguments.pod_name, arguments.pod_uid)
    if not directory.is_dir() and not arguments.allow_create_dir:
        raise _directory_missing()
    created_dirs: list[str] = []
    container_dir = directory / container
    path = container_dir / fixture_file_name(run_id)
    _materialize(directory, container_dir, path, created_dirs)
    state["logs"][key] = _log_record(
        arguments.pod_uid, container, directory, path, created_dirs, recreated
    )
    save_state(run_id, state)
    emit(
        {
            "run_id": run_id,
            "key": key,
            "file": file_view(path),
            "created_dirs": created_dirs,
            "recreated": recreated,
            "kubelet_directory": not created_dirs or created_dirs[0] != str(directory),
            "proof_scope": "live-replay",
        }
    )


def await_directory_gc(
    directory: Path,
    seconds: int,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait for kubelet's container GC to remove ``directory`` (``seconds`` at most).

    The old Pod's own directory is the canary: kubelet's
    ``evictPodLogsDirectories`` judges a ``/var/log/pods`` directory by the Pod
    UID in its name, whether kubelet made it (the real one, if still there) or
    this probe recreated it empty, so its removal marks the GC tick. A
    directory recreated right after lives for the whole next period (one
    minute) instead of an unknown remainder of it, which is what the
    collector's 10 s poll needs. The dictionary says whether the canary was
    created here, whether the tick was seen and after how long.
    """

    canary = False
    if not directory.exists():
        directory.mkdir(mode=0o755)
        canary = True
    started_at = utc_now()
    start = clock()
    observed = False
    while clock() - start < seconds:
        sleep(GC_POLL_SECONDS)
        if not directory.exists():
            observed = True
            break
    return {
        "awaited_seconds": int(seconds),
        "canary": canary,
        "observed": observed,
        "after_seconds": round(clock() - start, 1),
        "started_at": started_at.isoformat(),
    }


def seed_log(
    arguments: argparse.Namespace, *, interrupt: Callable[[str], None] | None = None
) -> None:
    """``create-log`` and ``write`` in one process, after kubelet's GC tick.

    Everything is checked before the node is touched: the reviewed records and
    their digest, the window, the container, an earlier round's leftovers.
    """

    run_id = safe_id(arguments.run_id)
    records = decode_records(arguments.sha256, arguments.b64)
    await_seconds = int(arguments.await_gc_seconds)
    if not 0 <= await_seconds <= MAX_GC_WAIT_SECONDS:
        raise ProbeError(
            f"await-gc-seconds is outside 0..{MAX_GC_WAIT_SECONDS}", "input_shape"
        )
    state = open_state(run_id)
    container = _window_container(state, arguments.container)
    key = log_key(arguments.pod_uid, container)
    recreated = _recreation_count(state, key)
    directory = pod_log_dir(arguments.namespace, arguments.pod_name, arguments.pod_uid)
    if not directory.is_dir() and not arguments.allow_create_dir:
        raise _directory_missing()
    created_dirs: list[str] = []
    gc: dict[str, Any] | None = None
    if await_seconds:
        gc = await_directory_gc(directory, await_seconds)
        if gc["canary"] and not gc["observed"]:
            # The canary stayed: it is this run's directory and is seeded as is.
            created_dirs.append(str(directory))
    container_dir = directory / container
    path = container_dir / fixture_file_name(run_id)
    _materialize(directory, container_dir, path, created_dirs, interrupt)
    stamps = _append_records(path, records)
    record = _log_record(
        arguments.pod_uid, container, directory, path, created_dirs, recreated
    )
    record["writes"] = len(records)
    record["seeded"] = True
    if gc is not None:
        record["gc"] = gc
    state["logs"][key] = record
    save_state(run_id, state)
    emit(
        {
            "run_id": run_id,
            "key": key,
            "file": file_view(path),
            "created_dirs": created_dirs,
            "recreated": recreated,
            "kubelet_directory": not created_dirs or created_dirs[0] != str(directory),
            "gc": gc,
            "target": "current",
            "records": len(records),
            "stamps": stamps,
            "flags": [item["flag"] for item in records],
            "sha256": arguments.sha256,
            "proof_scope": "live-replay",
        }
    )


def _log_entry(state: dict[str, Any], pod_uid: str, container: str) -> dict[str, Any]:
    entry = state["logs"].get(log_key(pod_uid, container))
    if entry is None:
        raise ProbeError(
            "no fixture log was created for this Pod and container", "no_fixture"
        )
    return entry


def write(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = open_state(run_id)
    entry = _log_entry(state, arguments.pod_uid, arguments.container)
    records = decode_records(arguments.sha256, arguments.b64)
    if arguments.target not in TARGETS:
        raise ProbeError(f"target must be one of {TARGETS}", "input_shape")
    if arguments.target == "rotated":
        if not entry["rotated"]:
            raise ProbeError("the fixture log was never rotated", "never_rotated")
        path = Path(entry["rotated"][-1])
    else:
        path = Path(entry["path"])
    stamps = _append_records(path, records)
    entry["writes"] = int(entry.get("writes", 0)) + len(records)
    save_state(run_id, state)
    emit(
        {
            "run_id": run_id,
            "target": arguments.target,
            "file": file_view(path),
            "records": len(records),
            "stamps": stamps,
            "flags": [item["flag"] for item in records],
            "sha256": arguments.sha256,
            "proof_scope": "live-replay",
        }
    )


def rotate(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = open_state(run_id)
    entry = _log_entry(state, arguments.pod_uid, arguments.container)
    path = Path(entry["path"])
    rotated = Path(str(path) + f".{len(entry['rotated']) + 1}")
    if rotated.exists():
        raise ProbeError("the rotation target already exists", "rotation_exists")
    if not path.is_file():
        raise ProbeError(
            "the fixture log file is gone (kubelet removed the directory?)",
            "fixture_gone",
        )
    before = file_view(path)
    os.rename(path, rotated)
    with path.open("x", encoding="utf-8"):
        pass
    path.chmod(0o644)
    entry["rotated"].append(str(rotated))
    save_state(run_id, state)
    emit(
        {
            "run_id": run_id,
            "before": before,
            "rotated": file_view(rotated),
            "current": file_view(path),
            "proof_scope": "live-replay",
        }
    )


def restart_collector(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    open_state(run_id)
    before = unit_state(LOG_UNIT)
    run(["systemctl", "restart", LOG_UNIT], timeout=120)
    after = unit_state(LOG_UNIT)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{LOG_UNIT} is not active after the restart", "unit_inactive")
    emit({"run_id": run_id, "before": before, "after": after})


def status(arguments: argparse.Namespace) -> None:
    """Read back the fixture files and the collector's view of them."""

    run_id = safe_id(arguments.run_id)
    state = open_state(run_id)
    files = {}
    for key, entry in state["logs"].items():
        files[key] = {
            "current": file_view(Path(entry["path"])),
            "rotated": [file_view(Path(item)) for item in entry["rotated"]],
            "directory_exists": Path(entry["directory"]).is_dir(),
            "writes": entry.get("writes", 0),
        }
    emit(
        {
            "run_id": run_id,
            "unit": unit_state(LOG_UNIT),
            "glob": state["glob"],
            "files": files,
            "collector": collector_state_view(run_id),
        }
    )


def close_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    if not state:
        # Nothing tracked: make sure no stray drop-in is left behind.
        dropin_path(run_id).unlink(missing_ok=True)
        if state_dir(run_id).is_dir():
            shutil.rmtree(state_dir(run_id))
        emit({"run_id": run_id, "already_closed": True, "clean": True})
        return
    removed = remove_fixture_files(state)
    Path(state["dropin"]).unlink(missing_ok=True)
    if state.get("enabled_log_collector"):
        run(["systemctl", "disable", "--runtime", LOG_UNIT], check=False, timeout=120)
    run(["systemctl", "daemon-reload"])
    if state.get("enabled_log_collector"):
        run(["systemctl", "stop", LOG_UNIT], check=False, timeout=120)
    else:
        run(["systemctl", "restart", LOG_UNIT], timeout=120)
    after = unit_state(LOG_UNIT)
    if not state.get("enabled_log_collector") and after.get("ActiveState") != "active":
        raise ProbeError(
            f"{LOG_UNIT} is not active after the window closed", "unit_inactive"
        )
    disarm_deadman(run_id)
    residue = window_residue(run_id, state)
    if state_dir(run_id).is_dir():
        shutil.rmtree(state_dir(run_id))
    emit(
        {
            "run_id": run_id,
            "unit": LOG_UNIT,
            "after": after,
            "removed": removed,
            "dropin_removed": not Path(state["dropin"]).exists(),
            "residue": residue,
            "clean": not any(residue.values()),
        }
    )


def remove_fixture_files(state: dict[str, Any]) -> dict[str, list[str]]:
    """Remove exactly what this run created; a directory kubelet took is fine."""

    removed: dict[str, list[str]] = {"files": [], "directories": [], "absent": []}
    for entry in state.get("logs", {}).values():
        for item in [entry["path"], *entry.get("rotated", [])]:
            path = Path(item)
            if path.is_file():
                path.unlink()
                removed["files"].append(item)
            else:
                removed["absent"].append(item)
        for item in reversed(entry.get("created_dirs", [])):
            directory = Path(item)
            if not directory.is_dir():
                removed["absent"].append(item)
                continue
            try:
                directory.rmdir()
                removed["directories"].append(item)
            except OSError:
                # Not ours alone any more: left in place and reported as residue.
                continue
    return removed


def window_residue(run_id: str, state: dict[str, Any]) -> dict[str, bool]:
    files_left = any(
        Path(item).exists()
        for entry in state.get("logs", {}).values()
        for item in [entry["path"], *entry.get("rotated", [])]
    )
    dirs_left = any(
        Path(item).is_dir()
        for entry in state.get("logs", {}).values()
        for item in entry.get("created_dirs", [])
    )
    return {
        "dropin": dropin_path(run_id).exists(),
        "fixture_files": files_left,
        "created_directories": dirs_left,
        "log_collector_enabled_by_run": bool(state.get("enabled_log_collector"))
        and unit_state(LOG_UNIT).get("UnitFileState") == "enabled-runtime",
        "deadman": unit_state(f"{deadman_unit(run_id)}.timer").get("ActiveState")
        == "active",
    }


def snapshot(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    state = load_state(run_id)
    residue = window_residue(run_id, state) if state else {"window": False}
    emit({"run_id": run_id, "window_open": bool(state), "residue": residue})


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    commands.add_parser("preflight").set_defaults(handler=preflight)

    item = commands.add_parser("open-window")
    item.add_argument("--run-id", required=True)
    item.add_argument("--container", required=True)
    item.add_argument("--restore-seconds", type=int, default=900)
    item.add_argument("--initial-tail-bytes", type=int, default=0)
    item.add_argument("--approved", action="store_true")
    item.set_defaults(handler=open_window)

    item = commands.add_parser("inspect")
    item.add_argument("--run-id", required=True)
    item.add_argument("--pod-uid", required=True)
    item.add_argument("--container-id", default="")
    item.set_defaults(handler=inspect)

    item = commands.add_parser("create-log")
    item.add_argument("--run-id", required=True)
    item.add_argument("--pod-uid", required=True)
    item.add_argument("--container", required=True)
    item.add_argument("--namespace", required=True)
    item.add_argument("--pod-name", required=True)
    item.add_argument("--allow-create-dir", action="store_true")
    item.set_defaults(handler=create_log)

    item = commands.add_parser("seed-log")
    item.add_argument("--run-id", required=True)
    item.add_argument("--pod-uid", required=True)
    item.add_argument("--container", required=True)
    item.add_argument("--namespace", required=True)
    item.add_argument("--pod-name", required=True)
    item.add_argument("--sha256", required=True)
    item.add_argument("--b64", required=True)
    item.add_argument("--allow-create-dir", action="store_true")
    item.add_argument("--await-gc-seconds", type=int, default=0)
    item.set_defaults(handler=seed_log)

    item = commands.add_parser("write")
    item.add_argument("--run-id", required=True)
    item.add_argument("--pod-uid", required=True)
    item.add_argument("--container", required=True)
    item.add_argument("--sha256", required=True)
    item.add_argument("--b64", required=True)
    item.add_argument("--target", choices=TARGETS, default="current")
    item.set_defaults(handler=write)

    item = commands.add_parser("rotate")
    item.add_argument("--run-id", required=True)
    item.add_argument("--pod-uid", required=True)
    item.add_argument("--container", required=True)
    item.set_defaults(handler=rotate)

    item = commands.add_parser("restart-collector")
    item.add_argument("--run-id", required=True)
    item.set_defaults(handler=restart_collector)

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


def exception_site(exc: BaseException) -> str | None:
    """The innermost ``function:line`` of this file on the exception's path."""

    site = None
    frame = exc.__traceback__
    while frame is not None:
        if frame.tb_frame.f_code.co_filename == __file__:
            site = f"{frame.tb_frame.f_code.co_name}:{frame.tb_lineno}"
        frame = frame.tb_next
    return site


def error_fields(exc: BaseException, command: str | None) -> dict[str, Any]:
    """The failure line (the shape ``host_probe_fixture.execute`` reads).

    Class, message digest, site and the reason token are what the fixture may
    show in the exception text; the sentence goes into the case's redacted
    ``probe-error.json``. Live 2026-10-26 (017 ``old_pod``, HyperPod) the line
    carried the class under ``error`` only, which the fixture does not read for
    ``error_kind: probe``, and the operator got "output withheld".
    """

    return {
        "error": f"{type(exc).__name__}: {exc}",
        "error_kind": "probe",
        "error_class": type(exc).__name__,
        "error_code": hashlib.sha256(str(exc).encode()).hexdigest()[:16],
        "error_site": exception_site(exc),
        "error_reason": getattr(exc, "reason", "") or None,
        "error_message": str(exc)[:2000],
        "command": command,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        emit(error_fields(exc, getattr(arguments, "command", None)))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
