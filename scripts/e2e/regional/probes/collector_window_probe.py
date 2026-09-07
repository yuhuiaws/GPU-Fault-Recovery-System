#!/usr/bin/env python3
"""Host probe shared by GF-REGIONAL-COLLECT-018/019/020 and NET-008.

Runs on the target GPU node (chroot into the host from the privileged probe
Pod, ``host_probe_fixture.py``) with the node's own collector venv, so every
Python import below resolves to the *deployed* collector code. Everything it
can change is bounded and reversible:

* **open-window / close-window** -- a systemd drop-in on one allow-listed
  collector unit that appends an ``EnvironmentFile`` the probe wrote, and/or
  puts a ``nvidia-smi`` shadow directory first on ``PATH``. The drop-in, the
  env file and the shadow live under ``/run/gpu-fault-acceptance/<digest>``;
  ``close-window`` removes all three, reloads systemd and restarts the unit.
  A ``systemd-run --on-active`` deadman runs ``close-window`` from a copy of
  this probe if the runner dies. Only three settings exist: a *deliberately
  wrong* cluster token the probe generates itself (never a caller value, so a
  real token cannot travel through a command line), unsetting
  ``GPU_FAULT_EXPECTED_GPU_COUNT``, and the shadow. The Node Agent is not in
  the unit allow-list.
* **shadow modes** -- ``hang:<seconds>`` sleeps past the collector's
  ``nvidia-smi`` timeout and then runs the real binary (unchanged output);
  ``drop-uuid:<uuid>:<calls>`` removes one GPU line from the *first N*
  inventory queries (``--query-gpu=index,uuid,pci.bus_id,name``) and passes
  every other invocation through untouched. The GPU is never touched; only
  what one process reads about it is. ``calls`` is capped below the host
  collector's inventory-mismatch threshold so no REBOOT_NODE can be earned.
* **write-kmsg** -- a user-space write to the real ``/dev/kmsg``: either a
  monitor-only XID 63 (NET-001's line) or an ``NVRM: Xid`` line that carries
  no code. Neither is a hardware fault and both say so in the line.
* **post-rejected-event** -- one ``NvidiaKernelLogEvent`` with an extra field
  the control plane's strict model forbids, sent through the node's own
  configured sink (real token, TLS, endpoint). The ingress answers 202; the
  processor's replay is where the 422 lands -- which is what the case checks.
* **outbox** -- ``gpu-fault-collector outbox`` from the node venv (the shipped
  CLI), metadata only; **seed-outbox-record** appends one replayable record
  that names a channel path the control plane does not serve, so the replay
  dead-letters it. It is written only while the owning unit is stopped by
  ``open-window``'s restart cycle -- never raced against a live rewrite.
* **block / unblock** -- NET-001's tagged ``iptables`` reject of TCP/443 to
  the control plane with a rollback timer.

Every mutating subcommand refuses anything outside its allow-list and every
answer is one JSON object on the last stdout line.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COLLECTOR_ENV = Path("/etc/gpu-fault/collector.env")
VENV_PYTHON = Path("/opt/gpu-fault/current/venv/bin/python")
COLLECTOR_CLI = Path("/opt/gpu-fault/current/venv/bin/gpu-fault-collector")
ACCEPTANCE_ROOT = Path("/run/gpu-fault-acceptance")
OUTBOX_DIRECTORY = Path("/var/lib/gpu-fault/outbox")
ALLOWED_UNITS = (
    "gpu-fault-kernel-collector.service",
    "gpu-fault-host-collector.service",
    "gpu-fault-metrics-collector.service",
    "gpu-fault-fabric-manager-collector.service",
)
OBSERVED_UNITS = ALLOWED_UNITS + ("gpu-fault-node-agent.service",)
OUTBOX_COLLECTORS = ("kernel", "dcgm", "host", "fabric-manager")
#: The only environment names a window may override / unset. The token value
#: is always generated here (``@invalid``); a caller cannot supply one.
WRONG_TOKEN_MARKER = "@invalid"
OVERRIDABLE = {"GPU_FAULT_CONTROL_PLANE_TOKEN": {WRONG_TOKEN_MARKER}}
UNSETTABLE = {"GPU_FAULT_EXPECTED_GPU_COUNT"}
#: Keys ``snapshot`` reads back from collector.env. The token is reported as a
#: presence flag only.
ENV_KEYS = (
    "GPU_FAULT_CLUSTER_ID",
    "GPU_FAULT_NODE_INSTANCE_TYPE",
    "GPU_FAULT_EXPECTED_GPU_COUNT",
    "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES",
    "GPU_FAULT_HOST_INTERVAL_SECONDS",
    "GPU_FAULT_METRICS_INTERVAL_SECONDS",
    "GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS",
    "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS",
    "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS",
    "GPU_FAULT_KERNEL_HEALTH_SUMMARY_SECONDS",
    "GPU_FAULT_METRICS_MODE",
)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
SAFE_BDF = re.compile(r"^0000:[0-9a-f]{2}:[0-9a-f]{2}$")
GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]{8,}$")
SHADOW_MODE = re.compile(r"^(hang):(\d{1,3})$|^(drop-uuid):(GPU-[0-9a-fA-F-]+):(\d)$")
MIN_RESTORE_SECONDS = 60
MAX_RESTORE_SECONDS = 1800
#: ``drop-uuid`` may hide a GPU from at most this many inventory queries. The
#: host collector's ``GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES`` is 2
#: on shipped nodes; one hidden sample can never become a REBOOT_NODE finding.
MAX_DROP_CALLS = 1
#: The only channel path a seeded outbox record may name: one the control plane
#: does not serve, so the replay's 404 is a verdict and the record dead-letters.
RETIRED_CHANNEL_PATH = "/v1/collector-events/retired-acceptance-channel"
XID_MONITOR_ONLY = 63


class ProbeError(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 180,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        env=env,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): "
            f"{shlex.join(command)}: {completed.stderr.strip()}"
        )
    return completed


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def safe_id(value: str, label: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError(f"unsafe {label}")
    return value


def checked_unit(unit: str) -> str:
    if unit not in ALLOWED_UNITS:
        raise ProbeError("unit is not in the collector window allow-list")
    return unit


def checked_restore_seconds(value: int) -> int:
    if not MIN_RESTORE_SECONDS <= int(value) <= MAX_RESTORE_SECONDS:
        raise ProbeError(
            f"restore seconds is outside {MIN_RESTORE_SECONDS}..{MAX_RESTORE_SECONDS}"
        )
    return int(value)


def window_digest(run_id: str) -> str:
    return hashlib.sha256(safe_id(run_id, "run ID").encode()).hexdigest()[:16]


def window_paths(run_id: str) -> dict[str, Path]:
    digest = window_digest(run_id)
    root = ACCEPTANCE_ROOT / digest
    return {
        "root": root,
        "state": root / "window.json",
        "override_env": root / "override.env",
        "shadow_dir": root / "bin",
        "shadow_state": root / "shadow-calls.json",
        "probe_copy": root / "probe.py",
    }


def dropin_name(run_id: str) -> str:
    return f"gpu-fault-acceptance-{window_digest(run_id)}.conf"


def deadman_unit(run_id: str) -> str:
    return f"gpu-fault-collector-window-{window_digest(run_id)}"


def parse_env() -> dict[str, str]:
    if not COLLECTOR_ENV.is_file():
        raise ProbeError("collector env file does not exist")
    values: dict[str, str] = {}
    for line in COLLECTOR_ENV.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value.strip().strip("'\"")
    return values


def env_snapshot() -> dict[str, Any]:
    values = parse_env()
    reported: dict[str, Any] = {key: values.get(key) for key in ENV_KEYS}
    reported["GPU_FAULT_CONTROL_PLANE_TOKEN_present"] = bool(
        values.get("GPU_FAULT_CONTROL_PLANE_TOKEN")
    )
    reported["GPU_FAULT_CONTROL_PLANE_URL_present"] = bool(
        values.get("GPU_FAULT_CONTROL_PLANE_URL")
    )
    return reported


def unit_state(unit: str) -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            unit,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=NRestarts",
            "--property=MainPID",
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


def service_snapshot() -> dict[str, dict[str, str]]:
    return {unit: unit_state(unit) for unit in OBSERVED_UNITS}


def kmsg_stream_identity() -> dict[str, Any]:
    """Which fd the kernel collector holds on /dev/kmsg and where it stands.

    A reopen (ARCH-G4's old behaviour on a delivery failure) shows up as a new
    fd number or a new inode open; a stream that was kept shows the same fd
    with a ``pos`` that only ever grows.
    """

    unit = unit_state("gpu-fault-kernel-collector.service")
    pid = int(unit.get("MainPID") or 0)
    streams = []
    if pid > 0:
        for path in sorted(Path(f"/proc/{pid}/fd").glob("*")):
            try:
                target = os.readlink(path)
            except OSError:
                continue
            if target != "/dev/kmsg":
                continue
            position: int | None = None
            try:
                for line in (
                    Path(f"/proc/{pid}/fdinfo/{path.name}")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ):
                    if line.startswith("pos:"):
                        position = int(line.split(":", 1)[1].strip())
            except (OSError, ValueError):
                position = None
            streams.append({"fd": int(path.name), "pos": position})
    return {
        "pid": pid,
        "invocation_id": unit.get("InvocationID"),
        "kmsg_streams": streams,
    }


def gpu_inventory() -> list[dict[str, str]]:
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader",
        ],
        timeout=60,
    )
    result = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",", 3)]
        if len(values) != 4:
            continue
        result.append(
            {
                "index": values[0],
                "uuid": values[1],
                "pci_bus_id": values[2],
                "name": values[3],
            }
        )
    return result


def outbox_cli(collector: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    if collector not in OUTBOX_COLLECTORS:
        raise ProbeError("outbox collector is not allow-listed")
    if not COLLECTOR_CLI.is_file():
        raise ProbeError("the node has no gpu-fault-collector CLI")
    return run(
        [str(COLLECTOR_CLI), "outbox", "--collector", collector, *arguments],
        check=False,
        timeout=60,
    )


def outbox_stats(collector: str) -> dict[str, Any]:
    completed = outbox_cli(collector, "stats")
    if completed.returncode:
        raise ProbeError(
            f"gpu-fault-collector outbox stats failed: {completed.stderr.strip()}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    value = json.loads(lines[-1]) if lines else {}
    if not isinstance(value, dict):
        raise ProbeError("outbox stats did not print a JSON object")
    return value


def outbox_records(collector: str, marker: str | None) -> list[dict[str, Any]]:
    """Record metadata (never payload bodies) plus whether a marker is inside."""

    path = OUTBOX_DIRECTORY / f"{collector}.ndjson"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            records.append({"malformed": True})
            continue
        if not isinstance(record, dict):
            records.append({"malformed": True})
            continue
        encoded = json.dumps(record.get("payload"), sort_keys=True, default=str)
        records.append(
            {
                "path": record.get("path"),
                "replayable": record.get("replayable"),
                "failed_at": record.get("failed_at"),
                "error": str(record.get("error") or "")[:200],
                "marker_present": marker is not None and marker in encoded,
            }
        )
    return records


def snapshot(arguments: argparse.Namespace) -> None:
    emit(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "collector_env": env_snapshot(),
            "services": service_snapshot(),
            "kmsg_stream": kmsg_stream_identity(),
            "gpu_inventory": gpu_inventory(),
            "outboxes": {
                name: {
                    "stats": outbox_stats(name) if COLLECTOR_CLI.is_file() else None,
                    "records": outbox_records(name, arguments.marker),
                }
                for name in OUTBOX_COLLECTORS
            },
            "windows": sorted(
                str(path.name) for path in ACCEPTANCE_ROOT.glob("*") if path.is_dir()
            )
            if ACCEPTANCE_ROOT.is_dir()
            else [],
        }
    )


# --------------------------------------------------------------------------- #
# nvidia-smi shadow
# --------------------------------------------------------------------------- #
SHADOW_SCRIPT = r'''#!/usr/bin/env python3
"""Acceptance shadow of nvidia-smi; see collector_window_probe.py."""
import fcntl
import json
import os
import subprocess
import sys
import time

MODE = %(mode)r
REAL = %(real)r
STATE = %(state)r
INVENTORY_QUERY = "--query-gpu=index,uuid,pci.bus_id,name"


def _take_call() -> int:
    with open(STATE, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        text = handle.read().strip()
        count = int(text) if text else 0
        handle.seek(0)
        handle.truncate()
        handle.write(str(count + 1))
        return count


argv = sys.argv[1:]
kind, first, second = MODE
if kind == "hang":
    time.sleep(int(first))
    os.execv(REAL, [REAL, *argv])
if INVENTORY_QUERY in argv and _take_call() < int(second):
    completed = subprocess.run([REAL, *argv], capture_output=True, text=True)
    kept = [line for line in completed.stdout.splitlines() if first not in line]
    sys.stdout.write("\n".join(kept) + ("\n" if kept else ""))
    sys.stderr.write(completed.stderr)
    sys.exit(completed.returncode)
os.execv(REAL, [REAL, *argv])
'''


def parse_shadow_mode(value: str) -> tuple[str, str, str]:
    match = SHADOW_MODE.fullmatch(value)
    if match is None:
        raise ProbeError(
            "shadow mode must be hang:<seconds> or drop-uuid:<uuid>:<calls>"
        )
    if match.group(1) == "hang":
        seconds = int(match.group(2))
        if not 1 <= seconds <= 120:
            raise ProbeError("hang seconds must be 1..120")
        return ("hang", str(seconds), "")
    calls = int(match.group(5))
    if not 1 <= calls <= MAX_DROP_CALLS:
        raise ProbeError(
            f"drop-uuid may hide a GPU from at most {MAX_DROP_CALLS} inventory query"
        )
    uuid = match.group(4)
    known = {item["uuid"] for item in gpu_inventory()}
    if uuid not in known:
        raise ProbeError("drop-uuid names a GPU this node does not have")
    return ("drop-uuid", uuid, str(calls))


def write_shadow(paths: dict[str, Path], mode: tuple[str, str, str]) -> str:
    real = shutil.which("nvidia-smi", path="/usr/bin:/usr/local/bin:/bin")
    if real is None:
        raise ProbeError("nvidia-smi is not installed on the host")
    shadow_dir = paths["shadow_dir"]
    shadow_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    script = shadow_dir / "nvidia-smi"
    script.write_text(
        SHADOW_SCRIPT
        % {
            "mode": mode,
            "real": os.path.realpath(real),
            "state": str(paths["shadow_state"]),
        },
        encoding="utf-8",
    )
    script.chmod(0o755)
    paths["shadow_state"].write_text("0", encoding="utf-8")
    return str(shadow_dir)


def dropin_path(unit: str, run_id: str) -> Path:
    return Path("/run/systemd/system") / f"{unit}.d" / dropin_name(run_id)


def open_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    unit = checked_unit(arguments.unit)
    restore_seconds = checked_restore_seconds(arguments.restore_seconds)
    paths = window_paths(run_id)
    if paths["state"].is_file():
        raise ProbeError("a collector window for this run is already open")
    overrides: dict[str, str] = {}
    for pair in arguments.env:
        name, separator, value = pair.partition("=")
        if not separator or name not in OVERRIDABLE:
            raise ProbeError(f"{name!r} is not an overridable collector setting")
        if value not in OVERRIDABLE[name]:
            raise ProbeError(f"{name} accepts only {sorted(OVERRIDABLE[name])}")
        overrides[name] = secrets.token_hex(24)
    unset = []
    for name in arguments.unset:
        if name not in UNSETTABLE:
            raise ProbeError(f"{name!r} may not be unset by a collector window")
        unset.append(name)
    shadow = (
        parse_shadow_mode(arguments.shadow_nvidia_smi)
        if (arguments.shadow_nvidia_smi)
        else None
    )
    if not overrides and not unset and shadow is None:
        raise ProbeError("a window must override, unset or shadow something")
    before = unit_state(unit)
    if before.get("ActiveState") != "active":
        raise ProbeError(f"{unit} is not active at baseline")

    paths["root"].mkdir(mode=0o700, parents=True, exist_ok=True)
    lines = ["[Service]"]
    if overrides:
        paths["override_env"].write_text(
            "".join(f"{name}={value}\n" for name, value in overrides.items()),
            encoding="utf-8",
        )
        paths["override_env"].chmod(0o600)
        lines.append(f"EnvironmentFile=-{paths['override_env']}")
    for name in unset:
        lines.append(f"UnsetEnvironment={name}")
    shadow_dir = None
    if shadow is not None:
        shadow_dir = write_shadow(paths, shadow)
        lines.append(
            f"Environment=PATH={shadow_dir}:/usr/local/sbin:/usr/local/bin:"
            "/usr/sbin:/usr/bin:/sbin:/bin"
        )
    dropin = dropin_path(unit, run_id)
    dropin.parent.mkdir(parents=True, exist_ok=True)
    dropin.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # The deadman closes the window from a copy of this probe: the runner's
    # own copy under /run is replaced on every call and removed at cleanup.
    shutil.copyfile(Path(__file__).resolve(), paths["probe_copy"])
    paths["probe_copy"].chmod(0o700)
    state = {
        "run_id": run_id,
        "unit": unit,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "overrides": sorted(overrides),
        "unset": unset,
        "shadow": list(shadow) if shadow is not None else None,
        "dropin": str(dropin),
        "before": before,
        "deadman_unit": deadman_unit(run_id),
        "restore_seconds": restore_seconds,
    }
    paths["state"].write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    paths["state"].chmod(0o600)
    unit_name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{unit_name}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{unit_name}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={unit_name}",
            f"--on-active={restore_seconds}s",
            "--timer-property=AccuracySec=1s",
            str(VENV_PYTHON),
            str(paths["probe_copy"]),
            "close-window",
            "--run-id",
            run_id,
        ]
    )
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", unit], timeout=120)
    after = unit_state(unit)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{unit} is not active after the window opened")
    emit(
        {
            **state,
            "after": after,
            "shadow_dir": shadow_dir,
            "deadman_timer": unit_state(f"{unit_name}.timer"),
        }
    )


def close_window(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    paths = window_paths(run_id)
    if not paths["state"].is_file():
        raise ProbeError("no collector window is open for this run")
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    unit = checked_unit(str(state["unit"]))
    Path(str(state["dropin"])).unlink(missing_ok=True)
    unit_name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{unit_name}.timer"], check=False)
    run(
        ["systemctl", "reset-failed", f"{unit_name}.service", f"{unit_name}.timer"],
        check=False,
    )
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "restart", unit], timeout=120)
    after = unit_state(unit)
    shutil.rmtree(paths["root"], ignore_errors=True)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{unit} is not active after the window closed")
    emit(
        {
            "run_id": run_id,
            "unit": unit,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "dropin_removed": not Path(str(state["dropin"])).exists(),
            "window_root_removed": not paths["root"].exists(),
            "after": after,
            "deadman_timer": unit_state(f"{unit_name}.timer"),
        }
    )


# --------------------------------------------------------------------------- #
# kmsg injection (user-space, labelled)
# --------------------------------------------------------------------------- #
def checked_bdf(value: str) -> str:
    text = value.strip().lower()
    parts = text.split(":")
    if len(parts) == 3 and len(parts[0]) == 8 and parts[0].startswith("0000"):
        text = ":".join([parts[0][4:], parts[1], parts[2]])
    text = text.rsplit(".", 1)[0] if re.fullmatch(r".*\.[0-7]$", text) else text
    if SAFE_BDF.fullmatch(text) is None:
        raise ProbeError("unsafe PCI BDF")
    return text


def kmsg_line(kind: str, *, marker: str, bdf: str) -> str:
    if kind == "xid63":
        return (
            f"<6>gpu-fault acceptance user-space injection marker={marker} "
            f"NVRM: Xid (PCI:{bdf}): {XID_MONITOR_ONLY}, "
            "monitor-only row remapping acceptance event\n"
        )
    if kind == "unparsed-xid":
        # An Xid token with no code: format drift, not a fault (ARCH-G7).
        return (
            f"<6>gpu-fault acceptance user-space injection marker={marker} "
            f"NVRM: Xid (PCI:{bdf}): , acceptance line without a code\n"
        )
    raise ProbeError("kmsg line kind is not allow-listed")


def write_kmsg(arguments: argparse.Namespace) -> None:
    marker = safe_id(arguments.marker, "marker")
    bdf = checked_bdf(arguments.pci_bdf)
    message = kmsg_line(arguments.kind, marker=marker, bdf=bdf).encode()
    descriptor = os.open("/dev/kmsg", os.O_WRONLY | os.O_CLOEXEC)
    try:
        written = os.write(descriptor, message)
    finally:
        os.close(descriptor)
    emit(
        {
            "kind": arguments.kind,
            "marker": marker,
            "pci_bdf": bdf,
            "bytes_written": written,
            "user_space_injection": True,
        }
    )


# --------------------------------------------------------------------------- #
# Direct sink post of a payload the strict model forbids
# --------------------------------------------------------------------------- #
POST_REJECTED_SCRIPT = r"""
import json
import sys
from datetime import datetime, timezone

from gpu_fault.channel_registry import NVIDIA_KERNEL_PATH
from gpu_fault.collectors import context_from_environment, sink_from_environment
from gpu_fault.collectors.sinks import CollectorError

marker, node_id = sys.argv[1:]
context = context_from_environment()
sink = sink_from_environment()
now = datetime.now(timezone.utc)
payload = {
    "cluster_id": context.cluster_id,
    "node_id": node_id,
    "record_id": f"acceptance-rejected-{marker}",
    "observed_at": now.isoformat(),
    "collected_at": now.isoformat(),
    "message": (
        "gpu-fault acceptance synthetic API injection marker="
        + marker
        + " NVRM: Xid (PCI:0000:00:00): 63, deliberately incompatible payload"
    ),
    "runtime_profile_version": context.runtime_profile_version,
    # The field the control plane's strict model forbids. Its replay is the
    # 4xx the case exists to make visible.
    "acceptance_unknown_field": marker,
}
outcome = {"path": NVIDIA_KERNEL_PATH, "marker": marker, "record_id": payload["record_id"]}
try:
    response = sink.post(NVIDIA_KERNEL_PATH, payload)
except CollectorError as exc:
    outcome["sink_outcome"] = "error"
    outcome["sink_status_code"] = exc.status_code
    outcome["sink_buffered"] = exc.buffered
    outcome["sink_replayable"] = exc.replayable
    outcome["sink_error"] = str(exc)[:300]
else:
    outcome["sink_outcome"] = "accepted"
    outcome["processor_request_id"] = (
        response.get("processor_request_id") if isinstance(response, dict) else None
    )
print(json.dumps(outcome, sort_keys=True, default=str))
"""


def collector_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(parse_env())
    return env


def post_rejected_event(arguments: argparse.Namespace) -> None:
    marker = safe_id(arguments.marker, "marker")
    node_id = safe_id(arguments.node_id, "node ID")
    if not VENV_PYTHON.is_file():
        raise ProbeError("the node has no collector venv")
    completed = run(
        [str(VENV_PYTHON), "-c", POST_REJECTED_SCRIPT, marker, node_id],
        env=collector_environment(),
        timeout=240,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    value = json.loads(lines[-1]) if lines else {}
    emit({**value, "synthetic_api_injection": True})


# --------------------------------------------------------------------------- #
# Outbox (shipped CLI) and a dead-letter seed
# --------------------------------------------------------------------------- #
def outbox_command(arguments: argparse.Namespace) -> None:
    collector = arguments.collector
    if arguments.action == "stats":
        emit({"collector": collector, "stats": outbox_stats(collector)})
        return
    if arguments.action == "list":
        completed = outbox_cli(collector, "list")
        if completed.returncode:
            raise ProbeError(f"outbox list failed: {completed.stderr.strip()}")
        emit(
            {
                "collector": collector,
                "lines": completed.stdout.splitlines(),
                "records": outbox_records(collector, arguments.marker),
            }
        )
        return
    if arguments.action == "requeue-dead":
        refused = outbox_cli(collector, "requeue-dead")
        confirmed = outbox_cli(collector, "requeue-dead", "--yes")
        if confirmed.returncode:
            raise ProbeError(f"requeue-dead failed: {confirmed.stderr.strip()}")
        emit(
            {
                "collector": collector,
                "refused_without_yes": refused.returncode != 0,
                "refusal": refused.stderr.strip()[:200],
                "output": confirmed.stdout.strip(),
                "stats": outbox_stats(collector),
            }
        )
        return
    raise ProbeError("outbox action is not allow-listed")


def seed_outbox_record(arguments: argparse.Namespace) -> None:
    """Append one replayable record naming a channel the control plane retired.

    Written only while the owning unit is inactive: the sink rewrites the
    whole file atomically under its own lock, and an append raced against
    that rewrite could be lost -- which would make a missing dead letter look
    like a delivered one.
    """

    marker = safe_id(arguments.marker, "marker")
    collector = arguments.collector
    if collector not in OUTBOX_COLLECTORS:
        raise ProbeError("outbox collector is not allow-listed")
    unit = {
        "kernel": "gpu-fault-kernel-collector.service",
        "dcgm": "gpu-fault-metrics-collector.service",
        "host": "gpu-fault-host-collector.service",
        "fabric-manager": "gpu-fault-fabric-manager-collector.service",
    }[collector]
    env = parse_env()
    state = unit_state(unit)
    if state.get("ActiveState") == "active":
        raise ProbeError("refusing to seed an outbox while its collector is active")
    record_id = f"acceptance-dead-letter-{marker}"
    record: dict[str, Any] = {
        "path": RETIRED_CHANNEL_PATH,
        "payload": {
            "cluster_id": env.get("GPU_FAULT_CLUSTER_ID"),
            "node_id": os.uname().nodename,
            "record_id": record_id,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "message": f"gpu-fault acceptance dead-letter seed marker={marker}",
        },
        "replayable": True,
        "error": "seeded by acceptance: channel retired on the control plane",
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = OUTBOX_DIRECTORY / f"{collector}.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    emit(
        {
            "collector": collector,
            "path": RETIRED_CHANNEL_PATH,
            "marker": marker,
            "record_id": record_id,
            "stats": outbox_stats(collector),
        }
    )


def purge_outbox_record(arguments: argparse.Namespace) -> None:
    """Remove the records ``seed-outbox-record`` wrote, and only those.

    Same rule as the seed: the owning unit must be inactive. A record is ours
    when it names the retired channel path *and* the seeded record id; a real
    dead letter on the node is never touched.
    """

    marker = safe_id(arguments.marker, "marker")
    collector = arguments.collector
    if collector not in OUTBOX_COLLECTORS:
        raise ProbeError("outbox collector is not allow-listed")
    unit = {
        "kernel": "gpu-fault-kernel-collector.service",
        "dcgm": "gpu-fault-metrics-collector.service",
        "host": "gpu-fault-host-collector.service",
        "fabric-manager": "gpu-fault-fabric-manager-collector.service",
    }[collector]
    if unit_state(unit).get("ActiveState") == "active":
        raise ProbeError("refusing to rewrite an outbox while its collector is active")
    path = OUTBOX_DIRECTORY / f"{collector}.ndjson"
    if not path.exists():
        emit({"collector": collector, "removed": 0, "remaining": 0})
        return
    kept = []
    removed = 0
    record_id = f"acceptance-dead-letter-{marker}"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        payload = record.get("payload") if isinstance(record, dict) else None
        if (
            isinstance(record, dict)
            and record.get("path") == RETIRED_CHANNEL_PATH
            and isinstance(payload, dict)
            and payload.get("record_id") == record_id
        ):
            removed += 1
            continue
        kept.append(line)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    os.replace(temporary, path)
    emit({"collector": collector, "removed": removed, "remaining": len(kept)})


def stop_unit(arguments: argparse.Namespace) -> None:
    """Stop one collector unit with a fail-safe start armed first (DESTR-019)."""

    unit = checked_unit(arguments.unit)
    run_id = safe_id(arguments.run_id, "run ID")
    restore_seconds = checked_restore_seconds(arguments.restore_seconds)
    name = f"gpu-fault-collector-start-{window_digest(run_id)}"
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={name}",
            f"--on-active={restore_seconds}s",
            "--timer-property=AccuracySec=1s",
            "/bin/systemctl",
            "start",
            unit,
        ]
    )
    before = unit_state(unit)
    run(["systemctl", "stop", unit], timeout=120)
    emit({"unit": unit, "before": before, "after": unit_state(unit), "fail_safe": name})


def start_unit(arguments: argparse.Namespace) -> None:
    unit = checked_unit(arguments.unit)
    run_id = safe_id(arguments.run_id, "run ID")
    name = f"gpu-fault-collector-start-{window_digest(run_id)}"
    run(["systemctl", "start", unit], timeout=120)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"], check=False)
    after = unit_state(unit)
    if after.get("ActiveState") != "active":
        raise ProbeError(f"{unit} did not start")
    emit({"unit": unit, "after": after})


# --------------------------------------------------------------------------- #
# Network block (NET-001's tagged reject with a rollback timer)
# --------------------------------------------------------------------------- #
def firewall() -> str:
    path = shutil.which("iptables")
    if path is None:
        raise ProbeError("iptables is not installed on the host")
    return path


def exact_rule(iptables: str, ip: str, tag: str, operation: str) -> list[str]:
    return [
        iptables,
        operation,
        "OUTPUT",
        "-p",
        "tcp",
        "-d",
        ip,
        "--dport",
        "443",
        "-m",
        "comment",
        "--comment",
        tag,
        "-j",
        "REJECT",
        "--reject-with",
        "tcp-reset",
    ]


def rule_present(iptables: str, ip: str, tag: str) -> bool:
    return run(exact_rule(iptables, ip, tag, "-C"), check=False).returncode == 0


def tagged_rules(tag: str) -> list[str]:
    completed = run([firewall(), "-S", "OUTPUT"])
    return [line for line in completed.stdout.splitlines() if tag in line]


def connectivity(ips: list[str]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for ip in ips:
        try:
            with socket.create_connection((ip, 443), timeout=3):
                result[ip] = True
        except OSError:
            result[ip] = False
    return result


def resolve_ipv4(hostname: str) -> list[str]:
    addresses = {
        item[4][0]
        for item in socket.getaddrinfo(
            hostname, 443, family=socket.AF_INET, type=socket.SOCK_STREAM
        )
    }
    if not addresses:
        raise ProbeError(f"no IPv4 addresses resolved for {hostname}")
    return sorted(str(item) for item in addresses)


def checked_tag(value: str) -> str:
    if SAFE_TAG.fullmatch(value) is None:
        raise ProbeError("unsafe firewall tag")
    return value


def block(arguments: argparse.Namespace) -> None:
    tag = checked_tag(arguments.tag)
    ttl = checked_restore_seconds(arguments.ttl_seconds)
    ips = [str(ipaddress.ip_address(value)) for value in arguments.ip]
    if not ips:
        raise ProbeError("at least one firewall destination is required")
    if tagged_rules(tag):
        raise ProbeError("tagged firewall rules already exist")
    iptables = firewall()
    unit = f"{tag}-rollback"
    script = Path(f"/run/{tag}-cleanup.sh")
    lines = ["#!/bin/bash", "set +e"]
    for ip in ips:
        check = shlex.join(exact_rule(iptables, ip, tag, "-C"))
        delete = shlex.join(exact_rule(iptables, ip, tag, "-D"))
        lines.extend(
            [
                f"while {check} >/dev/null 2>&1; do",
                f"  {delete} >/dev/null 2>&1 || break",
                "done",
            ]
        )
    lines.append(f"rm -f {shlex.quote(str(script))}")
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o700)
    run(["systemctl", "stop", f"{unit}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{unit}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={unit}",
            f"--on-active={ttl}s",
            "--timer-property=AccuracySec=1s",
            "/bin/bash",
            str(script),
        ]
    )
    timer = unit_state(f"{unit}.timer")
    if timer.get("ActiveState") != "active":
        raise ProbeError(f"automatic rollback timer is not active: {timer}")
    for ip in ips:
        if not rule_present(iptables, ip, tag):
            run(exact_rule(iptables, ip, tag, "-I"))
    emit(
        {
            "tag": tag,
            "ips": ips,
            "rules": tagged_rules(tag),
            "timer": timer,
            "connectivity": connectivity(ips),
        }
    )


def unblock(arguments: argparse.Namespace) -> None:
    tag = checked_tag(arguments.tag)
    iptables = firewall()
    ips = [str(ipaddress.ip_address(value)) for value in arguments.ip]
    for ip in ips:
        while rule_present(iptables, ip, tag):
            run(exact_rule(iptables, ip, tag, "-D"))
    unit = f"{tag}-rollback"
    run(["systemctl", "stop", f"{unit}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{unit}.service", f"{unit}.timer"], check=False)
    Path(f"/run/{tag}-cleanup.sh").unlink(missing_ok=True)
    emit(
        {
            "tag": tag,
            "rules": tagged_rules(tag),
            "timer": unit_state(f"{unit}.timer"),
            "connectivity": connectivity(ips),
        }
    )


def resolve(arguments: argparse.Namespace) -> None:
    ips = resolve_ipv4(arguments.endpoint_host)
    emit(
        {
            "endpoint_host": arguments.endpoint_host,
            "endpoint_ipv4": ips,
            "connectivity": connectivity(ips),
        }
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    show = commands.add_parser("snapshot")
    show.add_argument("--marker", default=None)
    show.set_defaults(handler=snapshot)

    opened = commands.add_parser("open-window")
    opened.add_argument("--run-id", required=True)
    opened.add_argument("--unit", choices=ALLOWED_UNITS, required=True)
    opened.add_argument("--env", action="append", default=[])
    opened.add_argument("--unset", action="append", default=[])
    opened.add_argument("--shadow-nvidia-smi", default="")
    opened.add_argument("--restore-seconds", type=int, default=600)
    opened.set_defaults(handler=open_window)

    closed = commands.add_parser("close-window")
    closed.add_argument("--run-id", required=True)
    closed.set_defaults(handler=close_window)

    kmsg = commands.add_parser("write-kmsg")
    kmsg.add_argument("--kind", choices=("xid63", "unparsed-xid"), required=True)
    kmsg.add_argument("--marker", required=True)
    kmsg.add_argument("--pci-bdf", required=True)
    kmsg.set_defaults(handler=write_kmsg)

    rejected = commands.add_parser("post-rejected-event")
    rejected.add_argument("--marker", required=True)
    rejected.add_argument("--node-id", required=True)
    rejected.set_defaults(handler=post_rejected_event)

    outbox = commands.add_parser("outbox")
    outbox.add_argument("--collector", choices=OUTBOX_COLLECTORS, required=True)
    outbox.add_argument("--action", choices=("stats", "list", "requeue-dead"))
    outbox.add_argument("--marker", default=None)
    outbox.set_defaults(handler=outbox_command)

    seed = commands.add_parser("seed-outbox-record")
    seed.add_argument("--collector", choices=OUTBOX_COLLECTORS, required=True)
    seed.add_argument("--marker", required=True)
    seed.set_defaults(handler=seed_outbox_record)

    purge = commands.add_parser("purge-outbox-record")
    purge.add_argument("--collector", choices=OUTBOX_COLLECTORS, required=True)
    purge.add_argument("--marker", required=True)
    purge.set_defaults(handler=purge_outbox_record)

    stop = commands.add_parser("stop-unit")
    stop.add_argument("--unit", choices=ALLOWED_UNITS, required=True)
    stop.add_argument("--run-id", required=True)
    stop.add_argument("--restore-seconds", type=int, default=300)
    stop.set_defaults(handler=stop_unit)

    start = commands.add_parser("start-unit")
    start.add_argument("--unit", choices=ALLOWED_UNITS, required=True)
    start.add_argument("--run-id", required=True)
    start.set_defaults(handler=start_unit)

    resolving = commands.add_parser("resolve")
    resolving.add_argument("--endpoint-host", required=True)
    resolving.set_defaults(handler=resolve)

    blocking = commands.add_parser("block")
    blocking.add_argument("--tag", required=True)
    blocking.add_argument("--ttl-seconds", type=int, required=True)
    blocking.add_argument("--ip", action="append", required=True)
    blocking.set_defaults(handler=block)

    unblocking = commands.add_parser("unblock")
    unblocking.add_argument("--tag", required=True)
    unblocking.add_argument("--ip", action="append", required=True)
    unblocking.set_defaults(handler=unblock)
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        emit({"error": f"{type(exc).__name__}: {exc}", "command": arguments.command})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
