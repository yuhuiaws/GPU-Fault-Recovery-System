#!/usr/bin/env python3
"""Host probe of GF-GENERAL-LIVE-014: freeze one device-plugin process.

Runs on the target GPU node through ``host_probe_fixture.py`` (a privileged,
``hostPID`` probe Pod that ``chroot``s into the host and runs this file with the
node's own Python). Standard library only, Python 3.9+, one JSON object on the
last stdout line.

It makes a Kubernetes device plugin hang *without* removing its DaemonSet from
the node and *without* touching ``Node.status``: a SIGSTOP of the plugin
container's own process. The frozen plugin keeps its unix socket open, so
kubelet -- whose ``ListAndWatch`` stream to it has no deadline and no keepalive
-- never notices: the Pod stays ``Running`` and allocatable stays full (live
2026-10-27, HyperPod ``gpu_plugin``: ``nvidia.com/gpu`` 8 for four frozen
minutes). What does notice is the product's host collector, which probes every
plugin socket with a bounded ``GetDevicePluginOptions`` each tick
(``collectors/host/device_plugin.py``) and raises
``<gpu|efa>_device_plugin_unresponsive`` after two timed-out ticks -- the fault
the product repairs by deleting and replacing the Pod
(``RESTART_{GPU,EFA}_DEVICE_PLUGIN`` stamped ``plugin_unresponsive``), unlike
the COLLECT-017 fixture, which excludes the DaemonSet by affinity and restores
it before the product acts. A product without that probe (pre-R13h) sees
nothing, which the control-plane fixture reports as INCONCLUSIVE, not FAIL.

Three read-only companions serve the fixture's oracle. ``feature`` says which
product generation this node runs (whether the host collector's installed
package carries the probe modules, the unit's ``GPU_FAULT_DEVICE_PLUGIN_*``
settings); ``probe-log`` lists the host collector's own journal lines about
the probe since a moment (each timed-out tick is one WARNING naming the socket
and the consecutive count); and ``freeze --resume-after-first-timeout`` tails
that journal after the SIGSTOP and SIGCONTs the plugin the instant the *first*
timed-out tick is logged, before the second can confirm the finding (the
``one_sample`` window, decided on the node in well under a second rather than
through the control plane).

Every signalled process must satisfy *both* gates: its cgroup names the plugin
Pod UID the control-plane fixture discovered (``kubepods.../pod<uid>``) *and*
its argv basename is one the fixture allow-listed (``--binary``). The Node
Agent, kubelet, the DCGM host engine and every other process are therefore out
of reach by construction -- a pod-scoped, binary-scoped signal, never a sweep.

Before it signals anything a ``systemd-run --on-active`` deadman is armed that
runs ``cleanup`` from a copy of this file, so the plugin is SIGCONT-resumed even
if the runner, the probe Pod and the control plane all disappear. (If the
product has by then deleted the frozen Pod, the PID is gone and the resume is a
no-op; the DaemonSet's replacement registers on its own.) Nothing here writes
``/proc`` or sysfs, touches a real device, or deletes a Pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_ROOT = Path("/run/gpu-fault-acceptance/device-plugin")
UNIT_PREFIX = "gfrs-dp-"
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
#: A Kubernetes object UID (lower-case hex with dashes).
SAFE_POD_UID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
#: The procfs root every process read goes through; a test points it at a fake
#: tree holding exactly the processes a live Pod ran.
PROC_ROOT = Path("/proc")
MIN_DEADMAN_SECONDS = 60
MAX_DEADMAN_SECONDS = 5400
#: The only process basenames this probe will ever signal. The fixture passes
#: the one it discovered with ``--binary``; the intersection with this list is
#: what is signalled, so a mistaken ``--binary`` can only ever narrow it.
PLUGIN_BINARY_ALLOW_LIST = (
    "nvidia-device-plugin",
    "aws-efa-k8s-device-plugin",
    # The HyperPod EFA plugin's container process: its Pod command/image
    # basename is ``aws-efa-k8s-device-plugin`` but the image entrypoint runs
    # ``efa-k8s-device-plugin`` -- the live refusal record of the HyperPod
    # lane's test node, 2026-10-27 (probe-error.json: pod_processes = pause +
    # efa-k8s-device-plugin, requested_binaries without it), which is why the
    # first efa_plugin runs refused with no_plugin_process. The pod-UID
    # cgroup match in ``plugin_processes`` stays the hard gate.
    "efa-k8s-device-plugin",
    "k8s-device-plugin",
    "device-plugin",
)
RESUME_KIND = "resume-plugin"
#: How long ``freeze`` waits for a SIGSTOP'd process to actually enter the
#: kernel's stopped state before it records ``stopped``. ``os.kill`` only
#: *queues* the signal; the task transitions to ``T`` when it is next
#: scheduled, so a ``/proc/<pid>/stat`` read on the very next line races the
#: transition and sees the old ``S``/``R`` -- the live gpu_plugin freeze
#: recorded ``stopped: false`` for a process that was in fact frozen the whole
#: window (its SIGCONT at cleanup resumed the same pid).
STOP_CONFIRM_SECONDS = 5.0
STOP_POLL_SECONDS = 0.05
#: The product's host collector unit, whose journal carries one WARNING per
#: timed-out ``GetDevicePluginOptions`` tick (``collectors/host/device_plugin.py``
#: ``_device_plugins``) and whose installed package says whether the probe
#: exists on this node at all.
HOST_COLLECTOR_UNIT = "gpu-fault-host-collector.service"
#: The node runtime's venv (``deploy/systemd/gpu-fault-host-collector.service``
#: ``ExecStart=/opt/gpu-fault/current/venv/bin/gpu-fault-collector host``); the
#: unit's own ``ExecStart`` path wins when it names another.
AGENT_VENV = Path("/opt/gpu-fault/current/venv")
AGENT_CURRENT = Path("/opt/gpu-fault/current")
#: The unit's environment file (the node probe's ``collector_env`` source).
COLLECTOR_ENV = Path("/etc/gpu-fault/collector.env")
#: Package files that exist only on a product with the device-plugin liveness
#: probe (b204, R13h): the collector contributor, the gRPC probe and the shared
#: metric/rule names. All three or none; a partial set is reported as it is.
FEATURE_MODULES = {
    "device_plugin": "gpu_fault/collectors/host/device_plugin.py",
    "device_plugin_probe": "gpu_fault/collectors/host/device_plugin_probe.py",
    "device_plugin_liveness": "gpu_fault/device_plugin_liveness.py",
}
DEVICE_PLUGIN_ENV_PREFIX = "GPU_FAULT_DEVICE_PLUGIN_"
#: The host collector's WARNING for one timed-out probe tick (the message of
#: ``LOGGER.warning`` in ``collectors/host/device_plugin.py``: the sockets that
#: timed out, the bound, the consecutive count and the count that opens the
#: finding), and its INFO for a node it does not probe.
PROBE_LINE = re.compile(
    r"device plugin (?P<sockets>\S+) did not answer GetDevicePluginOptions within "
    r"(?P<timeout>[0-9.]+)s \((?P<count>\d+) consecutive; finding after "
    r"(?P<required>\d+)\)"
)
DIRECTORY_ABSENT_LINE = re.compile(
    r"device plugin socket directory (?P<directory>\S+) is absent"
)
JOURNAL_POLL_SECONDS = 0.5
MAX_JOURNAL_ENTRIES = 200
#: ``freeze --resume-after-first-timeout`` never leaves the plugin frozen past
#: this window whether or not a timed-out tick was logged.
MIN_FIRST_TIMEOUT_WINDOW_SECONDS = 5
MAX_FIRST_TIMEOUT_WINDOW_SECONDS = 600


class ProbeError(RuntimeError):
    """A refused or failed step.

    ``reason`` is one token of a closed vocabulary (``[a-z][a-z0-9_]{2,40}``)
    the host fixture shows in its exception text (``; reason
    no_plugin_process``); ``detail`` is a JSON object that travels with the
    message into the case's ``probe-error.json`` -- for the "no allow-listed
    plugin process" refusal it carries the processes the probe *did* find in
    the plugin Pod's cgroup, so the real plugin binary name (unknown for the
    HyperPod EFA plugin until it was seen live) is captured on the next run.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.detail = detail


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            f"command failed ({completed.returncode}): {shlex.join(command)}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return completed


def checked_run_id(value: str) -> str:
    if SAFE_RUN_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def checked_pod_uid(value: str) -> str:
    if SAFE_POD_UID.fullmatch(value) is None:
        raise ProbeError("unsafe pod UID")
    return value


def checked_binaries(values: list[str]) -> list[str]:
    allowed = [item for item in values if item in PLUGIN_BINARY_ALLOW_LIST]
    if not allowed:
        raise ProbeError(
            "no allow-listed plugin binary requested; "
            f"--binary must be one of {list(PLUGIN_BINARY_ALLOW_LIST)}"
        )
    return allowed


#: The shape a Pod-spec-derived command basename must have to be signalled.
SAFE_BINARY = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def checked_pod_binaries(values: list[str]) -> list[str]:
    """Plugin-container command basenames the control-plane fixture read from
    the discovered Pod's own ``spec`` (``--pod-command-binary``).

    These extend the static allow-list with the Pod's *actual* plugin binary,
    which is not guessable across providers (the HyperPod EFA plugin's process
    basename is not one of the four hard-coded names, so a pure allow-list
    refused it live). Only the shape is checked here; the hard safety gate stays
    the pod-UID cgroup match in :func:`plugin_processes`, so a signalled process
    is always inside the one plugin Pod the fixture discovered.
    """

    return [item for item in values if SAFE_BINARY.fullmatch(item)]


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


EMPTY_STATE: dict[str, Any] = {"paused_processes": {}}


def load_state(run_id: str) -> dict[str, Any]:
    path = state_dir(run_id) / "state.json"
    if not path.is_file():
        return {"run_id": run_id, **json.loads(json.dumps(EMPTY_STATE))}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ProbeError("device-plugin probe state identity mismatch")
    for key, empty in EMPTY_STATE.items():
        value.setdefault(key, json.loads(json.dumps(empty)))
    return value


def save_state(run_id: str, state: dict[str, Any]) -> None:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "state.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def probe_copy(run_id: str) -> Path:
    """A copy of this file the deadman can run after the probe Pod is gone."""

    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / "probe.py"
    if not target.is_file():
        shutil.copyfile(Path(__file__).resolve(), target)
        target.chmod(0o700)
    return target


def unit_active_state(unit: str) -> str:
    completed = run(["systemctl", "show", unit, "--property=ActiveState"], check=False)
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key == "ActiveState":
            return value
    return ""


def arm_deadman(run_id: str, kind: str, seconds: int) -> str:
    """Arm ``cleanup --kind <kind>`` after ``seconds``; raise if not armed."""

    bounded(seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman seconds")
    name = unit_name(run_id, f"deadman-{kind}")
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
    if unit_active_state(f"{name}.timer") != "active":
        raise ProbeError(f"deadman {name} is not armed")
    return name


def disarm_deadman(run_id: str, kind: str) -> None:
    name = unit_name(run_id, f"deadman-{kind}")
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    completed = run(
        ["systemctl", "show", f"{name}.service", "--property=MainPID"], check=False
    )
    main_pid = ""
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key == "MainPID":
            main_pid = value
    if main_pid != str(os.getpid()):
        run(["systemctl", "stop", f"{name}.service"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"], check=False)


# --------------------------------------------------------------------------- #
# Process discovery (pod-scoped, binary-scoped)
# --------------------------------------------------------------------------- #
def process_cmdline(pid: int) -> list[str]:
    try:
        raw = (PROC_ROOT / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [item.decode(errors="replace") for item in raw.split(b"\0") if item]


def process_start_ticks(pid: int) -> str | None:
    try:
        text = (PROC_ROOT / str(pid) / "stat").read_text()
    except OSError:
        return None
    fields = text.rsplit(")", 1)[-1].split()
    return fields[19] if len(fields) > 19 else None


def process_state(pid: int, start_ticks: str | None) -> str | None:
    """The scheduler state letter of ``pid`` (``R``/``S``/``D``/``T``/``Z``...),
    or ``None`` when the pid is gone or has been replaced (start ticks differ)."""

    try:
        text = (PROC_ROOT / str(pid) / "stat").read_text()
    except OSError:
        return None
    fields = text.rsplit(")", 1)[-1].split()
    if not fields:
        return None
    if start_ticks is not None and (len(fields) <= 19 or fields[19] != start_ticks):
        return None
    return fields[0]


def process_is_stopped(pid: int, start_ticks: str | None) -> bool:
    return process_state(pid, start_ticks) == "T"


def wait_until_stopped(
    pid: int, start_ticks: str | None, *, deadline_seconds: float = STOP_CONFIRM_SECONDS
) -> bool:
    """Poll until ``pid`` is in the stopped state, bounded.

    ``os.kill(pid, SIGSTOP)`` only queues the signal; the task reaches ``T``
    when it is next scheduled (and is deferred while it sits in an
    uninterruptible ``D`` sleep). Reading the state on the next line races that
    transition, so this polls briefly before the caller records ``stopped`` --
    the fix for the live ``stopped: false`` on an actually-frozen plugin.
    """

    deadline = time.monotonic() + max(0.0, deadline_seconds)
    while True:
        if process_is_stopped(pid, start_ticks):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(STOP_POLL_SECONDS)


def pod_cgroup_processes(pod_uid: str) -> list[dict[str, Any]]:
    """Every process whose cgroup names ``pod_uid`` (allow-listed or not).

    Used only to describe a "no allow-listed plugin process" refusal: the real
    plugin binary basename the node runs is reported so a later run can add it,
    without ever signalling anything off the allow-list.
    """

    found = []
    for entry in PROC_ROOT.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        command = process_cmdline(pid)
        if not command or not process_in_pod(pid, pod_uid):
            continue
        found.append({"pid": pid, "binary": os.path.basename(command[0])})
    return sorted(found, key=lambda item: item["pid"])


def _canonical(value: str) -> str:
    """A cgroup/UID token stripped to compare across cgroup drivers.

    The cgroupfs driver writes ``.../pod<uid>`` with the UID's dashes kept; the
    systemd driver writes ``...-pod<uid_with_underscores>.slice``. Removing
    every dash and underscore makes the two forms comparable without guessing
    the driver.
    """

    return value.replace("-", "").replace("_", "").lower()


def cgroup_names_pod(cgroup_text: str, pod_uid: str) -> bool:
    """Whether a ``/proc/<pid>/cgroup`` body names the plugin Pod UID.

    Driver-agnostic (``kubepods.../pod<uid>`` with dashes, or
    ``...-pod<uid_with_underscores>.slice``) via :func:`_canonical`.
    """

    needle = "pod" + _canonical(pod_uid)
    return any(needle in _canonical(line) for line in cgroup_text.splitlines())


def process_in_pod(pid: int, pod_uid: str) -> bool:
    try:
        text = (PROC_ROOT / str(pid) / "cgroup").read_text()
    except OSError:
        return False
    return cgroup_names_pod(text, pod_uid)


def plugin_processes(pod_uid: str, binaries: list[str]) -> list[dict[str, Any]]:
    """Processes in the plugin Pod whose argv basename is allow-listed."""

    wanted = set(binaries)
    result = []
    for entry in PROC_ROOT.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        command = process_cmdline(pid)
        if not command or os.path.basename(command[0]) not in wanted:
            continue
        if not process_in_pod(pid, pod_uid):
            continue
        result.append(
            {
                "pid": pid,
                "start_ticks": process_start_ticks(pid),
                "binary": os.path.basename(command[0]),
                "argv": command[:12],
            }
        )
    return sorted(result, key=lambda item: item["pid"])


# --------------------------------------------------------------------------- #
# Product generation and the host collector's own probe journal (read-only)
# --------------------------------------------------------------------------- #
def unit_properties(unit: str, names: list[str]) -> dict[str, str]:
    completed = run(
        ["systemctl", "show", unit, f"--property={','.join(names)}"], check=False
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in names:
            values[key] = value
    return values


def exec_start_path(text: str) -> str | None:
    """The executable of a ``systemctl show --property=ExecStart`` value."""

    match = re.search(r"path=(\S+) ;", text or "")
    return match.group(1) if match else None


def installed_modules(venv: Path) -> dict[str, bool]:
    """Which ``FEATURE_MODULES`` the venv's installed ``gpu_fault`` carries."""

    return {
        name: any(venv.glob(f"lib/python3*/site-packages/{relative}"))
        for name, relative in FEATURE_MODULES.items()
    }


def device_plugin_environment(unit_environment: str) -> dict[str, str]:
    """The unit's ``GPU_FAULT_DEVICE_PLUGIN_*`` settings (file, then unit)."""

    values: dict[str, str] = {}
    if COLLECTOR_ENV.is_file():
        for line in COLLECTOR_ENV.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.startswith(DEVICE_PLUGIN_ENV_PREFIX):
                values[key] = value.strip().strip("'\"")
    for token in (unit_environment or "").split():
        key, separator, value = token.partition("=")
        if separator and key.startswith(DEVICE_PLUGIN_ENV_PREFIX):
            values[key] = value.strip("'\"")
    return values


def feature(arguments: argparse.Namespace) -> None:
    """Which product generation this node's host collector is."""

    checked_run_id(arguments.run_id)
    properties = unit_properties(
        HOST_COLLECTOR_UNIT, ["ExecStart", "Environment", "ActiveState"]
    )
    executable = exec_start_path(properties.get("ExecStart", ""))
    venv = Path(executable).parent.parent if executable else AGENT_VENV
    current = None
    if AGENT_CURRENT.is_symlink():
        current = os.readlink(AGENT_CURRENT)
    modules = installed_modules(venv)
    emit(
        {
            "unit": HOST_COLLECTOR_UNIT,
            "active_state": properties.get("ActiveState"),
            "exec_start_path": executable,
            "venv": str(venv),
            "venv_present": venv.is_dir(),
            "current_release": current,
            "modules": modules,
            "liveness_probe": all(modules.values()),
            "device_plugin_env": device_plugin_environment(
                properties.get("Environment", "")
            ),
        }
    )


def collector_journal(since_epoch: float) -> list[dict[str, Any]]:
    """The host collector's device-plugin probe lines since ``since_epoch``."""

    completed = run(
        [
            "journalctl",
            "-u",
            HOST_COLLECTOR_UNIT,
            "--since",
            f"@{since_epoch:.6f}",
            "--output=json",
            "--no-pager",
            "--quiet",
        ],
        check=False,
        timeout=60,
    )
    entries: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = value.get("MESSAGE")
        if isinstance(message, list):
            message = bytes(item for item in message if isinstance(item, int)).decode(
                errors="replace"
            )
        message = str(message or "")
        timed_out = PROBE_LINE.search(message)
        absent = DIRECTORY_ABSENT_LINE.search(message)
        if timed_out is None and absent is None:
            continue
        try:
            realtime = int(value.get("__REALTIME_TIMESTAMP") or 0) / 1_000_000
        except (TypeError, ValueError):
            realtime = 0.0
        entry: dict[str, Any] = {
            "realtime_epoch": realtime,
            "message": message[:400],
        }
        if timed_out is not None:
            entry.update(
                {
                    "kind": "timeout",
                    "sockets": timed_out.group("sockets").split(","),
                    "timeout_seconds": float(timed_out.group("timeout")),
                    "consecutive": int(timed_out.group("count")),
                    "required": int(timed_out.group("required")),
                }
            )
        else:
            entry.update(
                {"kind": "directory_absent", "directory": absent.group("directory")}
            )
        entries.append(entry)
        if len(entries) >= MAX_JOURNAL_ENTRIES:
            break
    return entries


def probe_log(arguments: argparse.Namespace) -> None:
    checked_run_id(arguments.run_id)
    since = float(arguments.since_epoch)
    if since <= 0:
        raise ProbeError("--since-epoch must be a positive epoch time")
    entries = collector_journal(since)
    emit(
        {
            "since_epoch": since,
            "read_at_epoch": time.time(),
            "entries": entries,
            "timeouts": [item for item in entries if item["kind"] == "timeout"],
            "directory_absent": any(
                item["kind"] == "directory_absent" for item in entries
            ),
        }
    )


def await_first_timeout(since_epoch: float, window_seconds: float) -> dict[str, Any]:
    """Watch the host collector's journal until its first timed-out probe tick.

    Returns as soon as a ``PROBE_LINE`` with any consecutive count is logged
    after ``since_epoch`` (``first_timeout``), or at the end of the window with
    ``first_timeout`` None. ``second_timeout_seen`` says whether a second
    consecutive tick had already been logged -- then the finding may have
    opened before the caller could resume.
    """

    deadline = time.monotonic() + max(0.0, window_seconds)
    started = time.monotonic()
    while True:
        timeouts = [
            item for item in collector_journal(since_epoch) if item["kind"] == "timeout"
        ]
        if timeouts or time.monotonic() >= deadline:
            return {
                "first_timeout": timeouts[0] if timeouts else None,
                "second_timeout_seen": any(
                    item["consecutive"] >= 2 for item in timeouts
                ),
                "timeouts": timeouts[:20],
                "waited_seconds": round(time.monotonic() - started, 3),
            }
        time.sleep(JOURNAL_POLL_SECONDS)


# --------------------------------------------------------------------------- #
# freeze / resume
# --------------------------------------------------------------------------- #
def freeze(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    pod_uid = checked_pod_uid(arguments.pod_uid)
    binaries = checked_binaries(list(arguments.binary))
    binaries += [
        item
        for item in checked_pod_binaries(
            list(getattr(arguments, "pod_command_binary", []))
        )
        if item not in binaries
    ]
    seconds = bounded(
        arguments.seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "seconds"
    )
    processes = plugin_processes(pod_uid, binaries)
    if not processes:
        # Name the processes the Pod *does* run (basenames only) so the real
        # plugin binary -- unknown for the HyperPod EFA plugin until it was
        # seen live -- is captured in probe-error.json for the next run,
        # instead of only the message digest (S42's EFA-driver lesson).
        raise ProbeError(
            f"no allow-listed plugin process found in pod {pod_uid}; refusing to signal",
            reason="no_plugin_process",
            detail={
                "pod_uid": pod_uid,
                "requested_binaries": sorted(set(binaries)),
                "pod_processes": pod_cgroup_processes(pod_uid),
            },
        )
    resume_on_first = bool(getattr(arguments, "resume_after_first_timeout", False))
    window = (
        bounded(
            getattr(arguments, "window_seconds", None) or 0,
            MIN_FIRST_TIMEOUT_WINDOW_SECONDS,
            MAX_FIRST_TIMEOUT_WINDOW_SECONDS,
            "window seconds",
        )
        if resume_on_first
        else None
    )
    state = load_state(run_id)
    # Arm the resume deadman before the first SIGSTOP: a crash after this leaves
    # a frozen plugin the deadman resumes on its own.
    deadman = arm_deadman(run_id, RESUME_KIND, seconds)
    frozen_since = time.time()
    stopped = []
    for process in processes:
        state["paused_processes"][str(process["pid"])] = {
            "start_ticks": process["start_ticks"],
            "binary": process["binary"],
            "pod_uid": pod_uid,
            "paused_at": utc_now(),
        }
        save_state(run_id, state)
        os.kill(process["pid"], signal.SIGSTOP)
        # Confirm the task actually reached the stopped state before recording
        # it -- os.kill only queues the signal (live gpu_plugin recorded
        # stopped:false for a process that was frozen the whole window).
        is_stopped = wait_until_stopped(process["pid"], process["start_ticks"])
        stopped.append(
            {
                "pid": process["pid"],
                "binary": process["binary"],
                "stopped": is_stopped,
                "state": process_state(process["pid"], process["start_ticks"]),
            }
        )
    receipt: dict[str, Any] = {
        "run_id": run_id,
        "pod_uid": pod_uid,
        "frozen": stopped,
        "frozen_since_epoch": frozen_since,
        "deadman": deadman,
        "resume_after_seconds": seconds,
    }
    if window is not None and all(item["stopped"] for item in stopped):
        # The one_sample window: SIGCONT the instant the host collector logs
        # its first timed-out tick, before the second tick can confirm the
        # finding; never past the window either way.
        watch = await_first_timeout(frozen_since, window)
        resumed = resume_processes(run_id)
        disarm_deadman(run_id, RESUME_KIND)
        resumed_at = time.time()
        first = watch["first_timeout"]
        receipt.update(
            {
                "window_seconds": window,
                "first_timeout": first,
                "second_timeout_seen": watch["second_timeout_seen"],
                "timeouts": watch["timeouts"],
                "waited_seconds": watch["waited_seconds"],
                "resumed": resumed,
                "resumed_at_epoch": resumed_at,
                "resume_latency_seconds": (
                    round(resumed_at - first["realtime_epoch"], 3)
                    if first and first["realtime_epoch"]
                    else None
                ),
            }
        )
    elif window is not None:
        # A task that never reached the stopped state is not a hang; leave
        # nothing paused behind the receipt the fixture is about to refuse.
        receipt["resumed"] = resume_processes(run_id)
        disarm_deadman(run_id, RESUME_KIND)
    emit(receipt)


def resume_processes(run_id: str) -> list[dict[str, Any]]:
    state = load_state(run_id)
    result = []
    for pid, info in list(state["paused_processes"].items()):
        if process_start_ticks(int(pid)) == info.get("start_ticks"):
            os.kill(int(pid), signal.SIGCONT)
            result.append(
                {
                    "pid": int(pid),
                    "resumed": True,
                    "stopped": process_is_stopped(int(pid), info.get("start_ticks")),
                }
            )
        else:
            # The product deleted the Pod (so kubelet killed the frozen
            # process) or it exited: nothing to resume, and the DaemonSet's
            # replacement is healthy on its own.
            result.append({"pid": int(pid), "resumed": False, "gone": True})
        del state["paused_processes"][pid]
    save_state(run_id, state)
    return result


def resume(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    processes = resume_processes(run_id)
    disarm_deadman(run_id, RESUME_KIND)
    emit({"run_id": run_id, "processes": processes})


def residue(run_id: str) -> dict[str, Any]:
    state = load_state(run_id)
    still_stopped = {}
    for pid, info in state["paused_processes"].items():
        if process_start_ticks(int(pid)) == info.get("start_ticks"):
            still_stopped[pid] = info
    return {"paused_processes": still_stopped}


def residue_is_clean(value: dict[str, Any]) -> bool:
    return not value.get("paused_processes")


def cleanup(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    kind = arguments.kind
    if kind not in {"all", RESUME_KIND}:
        raise ProbeError("unsafe cleanup kind")
    processes = resume_processes(run_id)
    disarm_deadman(run_id, RESUME_KIND)
    remaining = residue(run_id)
    emit(
        {
            "run_id": run_id,
            "kind": kind,
            "processes": processes,
            "residue": remaining,
            "clean": residue_is_clean(remaining),
        }
    )


def survey(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    value = residue(run_id)
    emit({"run_id": run_id, "residue": value, "clean": residue_is_clean(value)})


def show_residue(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    value = residue(run_id)
    emit({"residue": value, "clean": residue_is_clean(value)})


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    def command(name: str, handler: Any) -> argparse.ArgumentParser:
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
        item.set_defaults(handler=handler)
        return item

    command("survey", survey)
    command("residue", show_residue)
    command("feature", feature)
    item = command("probe-log", probe_log)
    item.add_argument("--since-epoch", type=float, required=True)
    item = command("freeze", freeze)
    item.add_argument("--pod-uid", required=True)
    item.add_argument("--binary", action="append", default=[], required=True)
    item.add_argument("--pod-command-binary", action="append", default=[])
    item.add_argument("--seconds", type=int, required=True)
    item.add_argument("--resume-after-first-timeout", action="store_true")
    item.add_argument("--window-seconds", type=int, default=0)
    command("resume", resume)
    item = command("cleanup", cleanup)
    item.add_argument("--kind", default="all")
    return value


#: Trivial frames ``exception_site`` skips so the reported site is the step
#: that refused, not the shared validator/emitter it bubbled through.
HELPER_FRAMES = frozenset(
    {
        "run",
        "emit",
        "utc_now",
        "bounded",
        "checked_run_id",
        "checked_pod_uid",
        "checked_binaries",
        "checked_pod_binaries",
        "load_state",
        "save_state",
        "unit_properties",
        "collector_journal",
    }
)


def exception_site(exc: BaseException) -> str | None:
    site, frame = None, exc.__traceback__
    while frame is not None:
        code = frame.tb_frame.f_code
        if code.co_filename == __file__ and code.co_name not in HELPER_FRAMES:
            site = f"{code.co_name}:{frame.tb_lineno}"
        frame = frame.tb_next
    return site


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        # The host fixture keeps the message and structured detail (redacted)
        # as the case's probe-error.json and shows only whitelisted tokens in
        # its exception text; the device-plugin probe used to emit the class
        # and digest alone, so the live EFA refusal left no probe-error.json
        # and its cause had to be brute-forced from the message digest (S42).
        emit(
            {
                "error": type(exc).__name__,
                "error_kind": "probe",
                "error_class": type(exc).__name__,
                "error_code": hashlib.sha256(str(exc).encode()).hexdigest()[:16],
                "error_site": exception_site(exc),
                "error_reason": getattr(exc, "reason", None),
                "error_message": str(exc)[:2000],
                "error_detail": getattr(exc, "detail", None),
                "command": getattr(arguments, "command", "?"),
            }
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
