#!/usr/bin/env python3
"""Host probe of the GF-GENERAL-LIVE runner: bounded, reversible host load.

Runs on the target GPU node through ``host_probe_fixture.py`` (a privileged
probe Pod that ``chroot``s into the host and runs this file with the node's own
Python). Standard library only, Python 3.9+, one JSON object on the last
stdout line. Test file systems and devices are the sibling
``general_live_storage_probe.py``; network test interfaces are
``general_live_net_probe.py``.

Everything it can change is a *test-only* resource named after a digest of the
run id and recorded in ``/run/gpu-fault-acceptance/general/<digest>/state.json``
*before* it is created:

* transient systemd units ``gfrs-<digest>-<kind>`` (CPU burners, a page-cache
  reader of a *sparse* file, an oracle sampler, a capped memory allocator) --
  every one with ``RuntimeMaxSec`` so it stops itself;
* a SIGSTOP of one allow-listed collector unit, or of the node's DCGM
  exporter process, resumed by a deadman;
* a runtime-only (``/run``) enable of the installed but disabled node log
  collector, behind an explicit approval.

Before any of these a ``systemd-run --on-active`` deadman is armed that runs
``cleanup`` from a copy of this file, so the node returns to its baseline even
if the runner, the probe Pod and the control plane all disappear. Nothing here
writes ``/proc`` or sysfs counters, touches a business process, a real disk,
the system or log disk, or a network interface.
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

STATE_ROOT = Path("/run/gpu-fault-acceptance/general")
COLLECTOR_ENV = Path("/etc/gpu-fault/collector.env")
#: Directories the sparse page-cache file may live in (as the storage probe's
#: backing files). The root file system
#: of a plain EKS node has hundreds of GiB; HyperPod's root is small, so the
#: instance store mount is the second candidate. Never the log disk.
BACKING_ROOTS = (
    Path("/var/lib/gpu-fault-acceptance"),
    Path("/opt/dlami/nvme/gpu-fault-acceptance"),
)
#: A backing file system may not end up more used than this after allocation.
BACKING_MAX_USED_PERCENT = 80.0
UNIT_PREFIX = "gfrs-"
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
SAFE_NAME = re.compile(r"^[a-z]{1,8}$")
SAFE_DEVICE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
MAX_SECONDS = 3600
MIN_DEADMAN_SECONDS = 60
MAX_DEADMAN_SECONDS = 5400
MAX_THREADS_PER_CPU = 4
MAX_PAGE_CACHE_FRACTION = 0.95
MAX_MEMORY_USED_PERCENT = 97.0
#: Units a pause may stop. The Node Agent and the DCGM host engine are not here:
#: the Agent holds the lease the silence cases must keep valid.
PAUSABLE_UNITS = (
    "gpu-fault-host-collector.service",
    "gpu-fault-metrics-collector.service",
    "gpu-fault-kernel-collector.service",
    "gpu-fault-fabric-manager-collector.service",
    "gpu-fault-log-collector.service",
)
LOG_COLLECTOR_UNIT = "gpu-fault-log-collector.service"
PRODUCT_UNITS = (
    *PAUSABLE_UNITS,
    "gpu-fault-node-agent.service",
    "gpu-fault-dcgm-exporter.service",
    "nvidia-dcgm.service",
    "nvidia-fabricmanager.service",
)
ENV_KEYS = (
    "GPU_FAULT_FILESYSTEMS",
    "GPU_FAULT_REQUIRED_INTERFACES",
    "GPU_FAULT_HOST_INTERVAL_SECONDS",
    "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS",
    "GPU_FAULT_HOST_EDGE_FILTER_ENABLED",
    "GPU_FAULT_METRICS_INTERVAL_SECONDS",
    "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS",
    "GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS",
    "GPU_FAULT_ENABLE_NODE_LOG_COLLECTOR",
    "GPU_FAULT_EXPECTED_GPU_COUNT",
    "GPU_FAULT_METRICS_MODE",
    "GPU_FAULT_DCGM_METRICS_URL",
)
TOOLS = (
    "systemd-run",
    "systemctl",
    "losetup",
    "mkfs.ext4",
    "mount",
    "umount",
    "dmsetup",
    "fallocate",
    "blockdev",
    "ip",
    "tc",
    "iptables",
    "smartctl",
    "ipmitool",
    "ethtool",
    "iostat",
    "modprobe",
    "lsblk",
    "nvidia-smi",
    "nsenter",
    "dnf",
    "rpm",
)
MODULES = ("dm_mod", "dm_delay", "loop", "veth", "dummy", "sch_netem", "nfsv4")
#: The exporter process a pause may stop: the dcgm-exporter binary of the
#: product's exporter (DaemonSet Pod or systemd container), never the host
#: engine.
EXPORTER_BINARY = "dcgm-exporter"


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


EMPTY_STATE: dict[str, Any] = {
    "units": [],
    "paused_units": {},
    "paused_processes": {},
    "page_cache_files": [],
    "log_collector": None,
}


def load_state(run_id: str) -> dict[str, Any]:
    path = state_dir(run_id) / "state.json"
    if not path.is_file():
        return {"run_id": run_id, **json.loads(json.dumps(EMPTY_STATE))}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ProbeError("general probe state identity mismatch")
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


def arm_deadman(run_id: str, kind: str, seconds: int, *arguments: str) -> str:
    """Arm ``cleanup --kind <kind>`` after ``seconds``; raise if it is not armed."""

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
            *arguments,
        ]
    )
    if unit_state(f"{name}.timer").get("ActiveState") != "active":
        raise ProbeError(f"deadman {name} is not armed")
    return name


def disarm_deadman(run_id: str, kind: str) -> None:
    name = unit_name(run_id, f"deadman-{kind}")
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


# --------------------------------------------------------------------------- #
# Read-only facts
# --------------------------------------------------------------------------- #
def parse_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if not COLLECTOR_ENV.is_file():
        return values
    for line in COLLECTOR_ENV.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value.strip().strip("'\"")
    return values


def meminfo() -> dict[str, float]:
    values: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, raw = line.partition(":")
        try:
            values[key] = float(raw.strip().split()[0]) * 1024
        except (IndexError, ValueError):
            continue
    return values


def memory_percentages(values: dict[str, float]) -> dict[str, float]:
    """The product's formulas (``collectors/host/system_metrics.py``)."""

    total = values.get("MemTotal", 0.0) or 1.0
    available = values.get("MemAvailable", values.get("MemFree", 0.0))
    page_cache = max(
        0.0,
        values.get("Cached", 0.0)
        + values.get("SReclaimable", 0.0)
        - values.get("Shmem", 0.0),
    )
    return {
        "memory_used_percent": 100 * (total - available) / total,
        "memory_available_percent": 100 * available / total,
        "page_cache_percent": 100 * page_cache / total,
        "swap_total_bytes": values.get("SwapTotal", 0.0),
    }


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


def mount_of(path: Path) -> dict[str, str] | None:
    target = os.path.abspath(str(path))
    best: dict[str, str] | None = None
    for item in mounts():
        point = item["mountpoint"]
        if target == point or target.startswith(point.rstrip("/") + "/"):
            if best is None or len(point) > len(best["mountpoint"]):
                best = item
    return best


def backing_candidates() -> list[dict[str, Any]]:
    result = []
    for root in BACKING_ROOTS:
        probe = root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        mount = mount_of(probe)
        reading = statvfs_reading(str(probe))
        result.append(
            {
                "root": str(root),
                "exists": root.exists(),
                "parent_present": probe != Path("/") or root.parent == Path("/"),
                "nearest": str(probe),
                "mount": mount,
                "statvfs": reading,
            }
        )
    return result


def process_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [item.decode(errors="replace") for item in raw.split(b"\0") if item]


def process_start_ticks(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = text.rsplit(")", 1)[-1].split()
    return fields[19] if len(fields) > 19 else None


def exporter_processes() -> list[dict[str, Any]]:
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        command = process_cmdline(int(entry.name))
        if command and os.path.basename(command[0]) == EXPORTER_BINARY:
            result.append(
                {
                    "pid": int(entry.name),
                    "start_ticks": process_start_ticks(int(entry.name)),
                    "argv": command[:12],
                }
            )
    return sorted(result, key=lambda item: int(item["pid"]))


def module_present(name: str) -> dict[str, bool]:
    loaded = Path(f"/sys/module/{name}").exists()
    available = loaded or (
        run(["modinfo", name], check=False, timeout=30).returncode == 0
        if shutil.which("modinfo")
        else False
    )
    return {"loaded": loaded, "available": available}


def residue(run_id: str) -> dict[str, Any]:
    """What this run still holds on the node (all empty means clean)."""

    digest = run_digest(run_id)
    units = [
        line.split()[0]
        for line in run(
            [
                "systemctl",
                "list-units",
                "--all",
                "--no-legend",
                "--plain",
                f"{UNIT_PREFIX}{digest}-*",
            ],
            check=False,
        ).stdout.splitlines()
        if line.strip()
    ]
    state = load_state(run_id)
    return {
        "units": sorted(
            {item for item in units if "-sdeadman-" not in item and "-io-" not in item}
        ),
        "page_cache_files": [
            item for item in state["page_cache_files"] if Path(item).exists()
        ],
        "paused_units": {
            unit: info
            for unit, info in state["paused_units"].items()
            if unit_is_stopped(unit)
        },
        "paused_processes": {
            str(pid): info
            for pid, info in state["paused_processes"].items()
            if process_is_stopped(int(pid), info.get("start_ticks"))
        },
        "log_collector_enabled_by_run": bool(state.get("log_collector")),
    }


def survey(arguments: argparse.Namespace) -> None:
    env = parse_env()
    memory = meminfo()
    emit(
        {
            "captured_at": utc_now(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "kernel": os.uname().release,
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "memory": {
                "MemTotal": memory.get("MemTotal"),
                "MemAvailable": memory.get("MemAvailable"),
                "SwapTotal": memory.get("SwapTotal"),
                **memory_percentages(memory),
            },
            "tools": {name: shutil.which(name) is not None for name in TOOLS},
            "modules": {name: module_present(name) for name in MODULES},
            "collector_env": {key: env.get(key) for key in ENV_KEYS},
            "units": {unit: unit_state(unit) for unit in PRODUCT_UNITS},
            "backing": backing_candidates(),
            "root_mount": mount_of(Path("/")),
            "exporter_processes": [
                {"pid": item["pid"], "start_ticks": item["start_ticks"]}
                for item in exporter_processes()
            ],
            "residue": residue(arguments.run_id),
        }
    )


# --------------------------------------------------------------------------- #
# Oracle sampler: the runner's own reading of the raw sources
# --------------------------------------------------------------------------- #
def cpu_counters() -> tuple[float, float]:
    fields = [
        float(item)
        for item in Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    ]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return sum(fields[:8]), idle


def diskstats_fields(device: str) -> list[float] | None:
    for line in Path("/proc/diskstats").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 14 and fields[2] == device:
            return [float(item) for item in fields[3:]]
    return None


def netdev_reading(name: str) -> dict[str, Any]:
    root = Path("/sys/class/net") / name
    if not root.exists():
        return {"present": False}
    result: dict[str, Any] = {"present": True}
    try:
        result["operstate"] = (root / "operstate").read_text().strip()
        result["ifindex"] = int((root / "ifindex").read_text().strip())
        for counter in (
            "rx_errors",
            "tx_errors",
            "rx_dropped",
            "tx_dropped",
            "rx_packets",
            "tx_packets",
        ):
            result[counter] = int((root / "statistics" / counter).read_text().strip())
    except (OSError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}"
    return result


def efa_traffic_reading(root: Path = Path("/sys/class/infiniband")) -> dict[str, Any]:
    """The runner's own raw EFA RX/TX byte totals, summed over EFA ports.

    An independent oracle for GF-GENERAL-LIVE-011: a hung collective must be
    confirmed by the EFA counter pair itself going to zero, not by the pause.
    This reads ``/sys/class/infiniband/<dev>/ports/<p>/hw_counters`` directly
    and sums the same pair the host collector's rate uses (``rx_bytes`` and
    ``tx_bytes``, else ``send_bytes`` and ``recv_bytes``) across EFA devices,
    so two readings bracketing the stall show whether traffic actually stopped.
    It never writes a counter.
    """

    ports: list[dict[str, Any]] = []
    rx_total = 0
    tx_total = 0
    complete = True
    for device in sorted(p.name for p in root.glob("*")) if root.exists() else []:
        driver = root / device / "device" / "driver"
        try:
            is_efa = driver.resolve().name == "efa"
        except OSError:
            is_efa = False
        if not is_efa:
            try:
                uevent = (root / device / "device" / "uevent").read_text()
                is_efa = any(
                    line.strip() == "DRIVER=efa" for line in uevent.splitlines()
                )
            except OSError:
                is_efa = False
        if not is_efa:
            continue
        for port in sorted(p.name for p in (root / device / "ports").glob("*")):
            counters = root / device / "ports" / port / "hw_counters"
            names = (
                {item.name for item in counters.iterdir()}
                if counters.exists()
                else set()
            )
            pair = (
                ("rx_bytes", "tx_bytes")
                if {"rx_bytes", "tx_bytes"} <= names
                else ("send_bytes", "recv_bytes")
                if {"send_bytes", "recv_bytes"} <= names
                else None
            )
            if pair is None:
                complete = False
                ports.append({"device": f"{device}/{port}", "pair": None})
                continue
            try:
                first = int((counters / pair[0]).read_text().strip())
                second = int((counters / pair[1]).read_text().strip())
            except (OSError, ValueError):
                complete = False
                ports.append(
                    {"device": f"{device}/{port}", "pair": pair, "error": True}
                )
                continue
            rx_total += first
            tx_total += second
            ports.append(
                {
                    "device": f"{device}/{port}",
                    "pair": pair,
                    pair[0]: first,
                    pair[1]: second,
                }
            )
    return {
        "observed_at": time.time(),
        "rx_bytes_total": rx_total,
        "tx_bytes_total": tx_total,
        "pair_bytes_total": rx_total + tx_total,
        "port_count": len(ports),
        "complete_pairs": complete,
        "ports": ports,
    }


def efa_traffic(arguments: argparse.Namespace) -> None:
    emit(efa_traffic_reading())


def tcp_retransmits() -> int | None:
    lines = Path("/proc/net/snmp").read_text().splitlines()
    for header, values in zip(lines, lines[1:]):
        if header.startswith("Tcp:") and values.startswith("Tcp:"):
            names = header.split()[1:]
            numbers = values.split()[1:]
            if "RetransSegs" in names:
                return int(numbers[names.index("RetransSegs")])
    return None


def oracle_sample(spec: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    record: dict[str, Any] = {"t": now, "at": utc_now()}
    if spec.get("cpu"):
        total, idle = cpu_counters()
        last = previous.get("cpu")
        if last and total > last[0]:
            record["cpu_busy_percent"] = 100 * (
                1 - (idle - last[1]) / (total - last[0])
            )
        previous["cpu"] = (total, idle)
    if spec.get("loadavg"):
        load1 = float(Path("/proc/loadavg").read_text().split()[0])
        record["load1"] = load1
        record["load1_per_cpu"] = load1 / max(1, os.cpu_count() or 1)
    if spec.get("meminfo"):
        record.update(memory_percentages(meminfo()))
    for path in spec.get("statvfs") or []:
        record.setdefault("statvfs", {})[path] = statvfs_reading(path)
    for device in spec.get("diskstats") or []:
        fields = diskstats_fields(device)
        entry: dict[str, Any] = {"present": fields is not None}
        last = previous.get(f"disk/{device}")
        if fields is not None and last is not None:
            elapsed = now - last[0]
            old = last[1]
            ios = (fields[0] - old[0]) + (fields[4] - old[4])
            entry["completed_ios"] = ios
            entry["in_flight"] = fields[8]
            entry["util_percent"] = (
                min(100.0, 100 * (fields[9] - old[9]) / (elapsed * 1000))
                if elapsed > 0
                else None
            )
            # The standard await (iostat r_await/w_await combined): read and
            # write milliseconds over completed reads and writes. The product
            # divides the *weighted* in-flight time instead.
            entry["standard_await_ms"] = (
                ((fields[3] - old[3]) + (fields[7] - old[7])) / ios if ios > 0 else None
            )
            entry["weighted_await_ms"] = (
                (fields[10] - old[10]) / ios if ios > 0 else None
            )
            entry["counters_decreased"] = any(
                new < before for new, before in zip(fields[:11], old[:11])
            )
        if fields is not None:
            previous[f"disk/{device}"] = (now, fields)
        record.setdefault("diskstats", {})[device] = entry
    for name in spec.get("netdev") or []:
        record.setdefault("netdev", {})[name] = netdev_reading(name)
    if spec.get("snmp"):
        record["tcp_retransmits"] = tcp_retransmits()
    return record


def sample_loop(arguments: argparse.Namespace) -> None:
    spec = json.loads(arguments.spec)
    output = Path(arguments.output)
    deadline = time.monotonic() + bounded(
        arguments.seconds, 10, MAX_SECONDS, "sampler seconds"
    )
    previous: dict[str, Any] = {}
    while time.monotonic() < deadline:
        record = oracle_sample(spec, previous)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        time.sleep(max(1.0, float(arguments.interval)))


def oracle_path(run_id: str, name: str) -> Path:
    return state_dir(run_id) / f"oracle-{checked_name(name)}.jsonl"


def oracle_spec(arguments: argparse.Namespace) -> dict[str, Any]:
    for device in arguments.diskstats:
        if SAFE_DEVICE.fullmatch(device) is None:
            raise ProbeError("unsafe block device name")
    for name in arguments.netdev:
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,15}", name) is None:
            raise ProbeError("unsafe interface name")
    for path in arguments.statvfs:
        if not path.startswith("/") or ".." in Path(path).parts:
            raise ProbeError("statvfs path must be absolute")
    return {
        "cpu": arguments.cpu,
        "loadavg": arguments.loadavg,
        "meminfo": arguments.meminfo,
        "statvfs": list(arguments.statvfs),
        "diskstats": list(arguments.diskstats),
        "netdev": list(arguments.netdev),
        "snmp": arguments.snmp,
    }


def oracle_start(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_name(arguments.name)
    spec = oracle_spec(arguments)
    path = oracle_path(run_id, name)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    seconds = bounded(arguments.seconds, 10, MAX_SECONDS, "sampler seconds")
    unit = start_unit(
        run_id,
        f"oracle-{name}",
        seconds + 30,
        [
            sys.executable,
            str(probe_copy(run_id)),
            "sample-loop",
            "--output",
            str(path),
            "--interval",
            str(bounded(arguments.interval, 1, 60, "sampler interval")),
            "--seconds",
            str(seconds),
            "--spec",
            json.dumps(spec, sort_keys=True),
        ],
        "Nice=5",
    )
    emit(
        {
            "unit": unit,
            "path": str(path),
            "spec": spec,
            "state": unit_state(f"{unit}.service"),
        }
    )


def oracle_read(arguments: argparse.Namespace) -> None:
    path = oracle_path(checked_run_id(arguments.run_id), arguments.name)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    offset = max(0, int(arguments.offset))
    selected = lines[offset : offset + 400]
    records = []
    for line in selected:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"malformed": True})
    emit(
        {
            "offset": offset,
            "next_offset": offset + len(selected),
            "total": len(lines),
            "records": records,
        }
    )


def oracle_stop(arguments: argparse.Namespace) -> None:
    emit(
        {
            "stopped": stop_unit(
                checked_run_id(arguments.run_id),
                f"oracle-{checked_name(arguments.name)}",
            )
        }
    )


# --------------------------------------------------------------------------- #
# CPU, page cache and memory load
# --------------------------------------------------------------------------- #
def cpu_start(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    cpus = max(1, os.cpu_count() or 1)
    threads = bounded(arguments.threads, 1, cpus * MAX_THREADS_PER_CPU, "threads")
    seconds = bounded(arguments.seconds, 10, MAX_SECONDS, "seconds")
    deadman = arm_deadman(run_id, "cpu", seconds + 120)
    # SCHED_IDLE busy loops yield to every normal task: /proc/stat reads the
    # host as busy while kubelet, the collectors and the Agent keep running.
    script = f"i=0; while [ $i -lt {threads} ]; do (while :; do :; done) & i=$((i+1)); done; wait"
    unit = start_unit(
        run_id,
        "cpu",
        seconds,
        ["/bin/sh", "-c", script],
        "CPUSchedulingPolicy=idle",
        "Nice=19",
        "KillMode=control-group",
    )
    emit(
        {
            "unit": unit,
            "threads": threads,
            "seconds": seconds,
            "deadman": deadman,
            "state": unit_state(f"{unit}.service"),
        }
    )


def cpu_stop(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    state = stop_unit(run_id, "cpu")
    disarm_deadman(run_id, "cpu")
    emit({"stopped": state})


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


def page_cache_start(arguments: argparse.Namespace) -> None:
    """Fill the page cache from a *sparse* file: no disk space is consumed."""

    run_id = checked_run_id(arguments.run_id)
    fraction = float(arguments.fraction)
    if not 0.1 <= fraction <= MAX_PAGE_CACHE_FRACTION:
        raise ProbeError("page cache fraction is outside 0.1..0.95")
    seconds = bounded(arguments.seconds, 30, MAX_SECONDS, "seconds")
    size = int(meminfo()["MemTotal"] * fraction)
    root = backing_root(arguments.backing, 0)
    deadman = arm_deadman(run_id, "pagecache", seconds + 120)
    path = run_backing_dir(run_id, root) / "page-cache.sparse"
    state = load_state(run_id)
    if str(path) not in state["page_cache_files"]:
        state["page_cache_files"].append(str(path))
        save_state(run_id, state)
    with path.open("wb") as handle:
        handle.truncate(size)
    script = (
        f"cat {shlex.quote(str(path))} > /dev/null; "
        f"while :; do sleep 30; cat {shlex.quote(str(path))} > /dev/null; done"
    )
    unit = start_unit(
        run_id,
        "pagecache",
        seconds,
        ["/bin/sh", "-c", script],
        "Nice=19",
        "IOSchedulingClass=idle",
    )
    emit(
        {
            "unit": unit,
            "file": str(path),
            "apparent_bytes": size,
            "allocated_bytes": path.stat().st_blocks * 512,
            "deadman": deadman,
        }
    )


def page_cache_stop(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    state = stop_unit(run_id, "pagecache")
    removed = remove_page_cache_files(run_id)
    disarm_deadman(run_id, "pagecache")
    emit(
        {"stopped": state, "removed": removed, "memory": memory_percentages(meminfo())}
    )


def remove_page_cache_files(run_id: str) -> list[str]:
    state = load_state(run_id)
    removed = []
    for name in list(state["page_cache_files"]):
        path = Path(name)
        if path.is_file():
            path.unlink()
            removed.append(name)
        state["page_cache_files"].remove(name)
    save_state(run_id, state)
    return removed


MEMORY_ALLOCATOR = r"""
import mmap, os, sys, time
target = float(sys.argv[1]); seconds = float(sys.argv[2]); floor = float(sys.argv[3])
def available_percent():
    values = {}
    for line in open("/proc/meminfo"):
        key, _, raw = line.partition(":")
        values[key] = float(raw.split()[0])
    return 100.0 * values["MemAvailable"] / values["MemTotal"], 100.0 * (values["MemTotal"] - values["MemAvailable"]) / values["MemTotal"]
chunks = []
deadline = time.monotonic() + seconds
step = 1 << 30
while time.monotonic() < deadline:
    available, used = available_percent()
    if available < floor:
        chunks = chunks[: max(0, len(chunks) - 4)]
    elif used < target:
        chunk = mmap.mmap(-1, step, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        for offset in range(0, step, 4096):
            chunk[offset] = 1
        chunks.append(chunk)
        continue
    time.sleep(1)
"""


def memory_start(arguments: argparse.Namespace) -> None:
    """Host memory pressure inside its own capped cgroup (needs approval)."""

    if not arguments.approved:
        raise ProbeError("host memory pressure requires the runner's approval")
    run_id = checked_run_id(arguments.run_id)
    target = float(arguments.target_used_percent)
    if not 50.0 <= target <= MAX_MEMORY_USED_PERCENT:
        raise ProbeError("memory target is outside 50..97 percent")
    seconds = bounded(arguments.seconds, 60, 1800, "seconds")
    total = meminfo()["MemTotal"]
    cap = int(total * (MAX_MEMORY_USED_PERCENT + 0.5) / 100)
    deadman = arm_deadman(run_id, "memory", seconds + 120)
    unit = start_unit(
        run_id,
        "memory",
        seconds,
        [sys.executable, "-c", MEMORY_ALLOCATOR, str(target), str(seconds), "2.5"],
        f"MemoryMax={cap}",
        "MemorySwapMax=0",
        "OOMPolicy=kill",
        "Nice=19",
    )
    emit(
        {
            "unit": unit,
            "target_used_percent": target,
            "memory_max_bytes": cap,
            "deadman": deadman,
        }
    )


def memory_stop(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    state = stop_unit(run_id, "memory")
    disarm_deadman(run_id, "memory")
    emit({"stopped": state, "memory": memory_percentages(meminfo())})


# --------------------------------------------------------------------------- #
# Pausing one collector channel or the exporter (031, 025)
# --------------------------------------------------------------------------- #
def checked_pausable(unit: str) -> str:
    if unit not in PAUSABLE_UNITS:
        raise ProbeError("unit is not in the pause allow-list")
    return unit


def unit_is_stopped(unit: str) -> bool:
    pid = unit_state(unit).get("MainPID") or "0"
    if not pid.isdigit() or pid == "0":
        return False
    return process_is_stopped(int(pid), None)


def process_is_stopped(pid: int, start_ticks: str | None) -> bool:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    fields = text.rsplit(")", 1)[-1].split()
    if start_ticks is not None and (len(fields) <= 19 or fields[19] != start_ticks):
        return False
    return bool(fields) and fields[0] == "T"


def unit_pause(arguments: argparse.Namespace) -> None:
    """SIGSTOP every process of one collector unit; the unit stays active.

    A pause is not a disable: systemd keeps the unit ``active`` and the Agent
    keeps reporting it enabled, which is what the silence scanner must see.
    """

    run_id = checked_run_id(arguments.run_id)
    unit = checked_pausable(arguments.unit)
    seconds = bounded(arguments.seconds, MIN_DEADMAN_SECONDS, 1500, "pause seconds")
    before = unit_state(unit)
    if before.get("ActiveState") != "active":
        raise ProbeError(f"{unit} is not active at baseline")
    kind = f"resume-{unit.split('.')[0].removeprefix('gpu-fault-')}"
    state = load_state(run_id)
    state["paused_units"][unit] = {
        "paused_at": utc_now(),
        "before": before,
        "kind": kind,
    }
    save_state(run_id, state)
    deadman = arm_deadman(run_id, kind, seconds, "--unit", unit)
    run(["systemctl", "kill", "--kill-whom=all", "--signal=SIGSTOP", unit])
    emit(
        {
            "unit": unit,
            "before": before,
            "after": unit_state(unit),
            "stopped": unit_is_stopped(unit),
            "deadman": deadman,
            "resume_after_seconds": seconds,
        }
    )


def resume_unit(run_id: str, unit: str) -> dict[str, Any]:
    run(["systemctl", "kill", "--kill-whom=all", "--signal=SIGCONT", unit], check=False)
    state = load_state(run_id)
    info = state["paused_units"].pop(unit, None)
    save_state(run_id, state)
    if info is not None:
        disarm_deadman(run_id, info["kind"])
    return {"unit": unit, "after": unit_state(unit), "stopped": unit_is_stopped(unit)}


def unit_resume(arguments: argparse.Namespace) -> None:
    emit(
        resume_unit(checked_run_id(arguments.run_id), checked_pausable(arguments.unit))
    )


def exporter_pause(arguments: argparse.Namespace) -> None:
    """SIGSTOP the node's dcgm-exporter process: its scrape port stops answering."""

    run_id = checked_run_id(arguments.run_id)
    seconds = bounded(arguments.seconds, MIN_DEADMAN_SECONDS, 1500, "pause seconds")
    processes = exporter_processes()
    if len(processes) != 1:
        raise ProbeError(
            f"expected exactly one {EXPORTER_BINARY} process, found {len(processes)}"
        )
    target = processes[0]
    state = load_state(run_id)
    state["paused_processes"][str(target["pid"])] = {
        "start_ticks": target["start_ticks"],
        "paused_at": utc_now(),
        "binary": EXPORTER_BINARY,
    }
    save_state(run_id, state)
    deadman = arm_deadman(run_id, "resume-exporter", seconds)
    os.kill(int(target["pid"]), signal.SIGSTOP)
    emit(
        {
            "pid": target["pid"],
            "stopped": process_is_stopped(int(target["pid"]), target["start_ticks"]),
            "deadman": deadman,
            "resume_after_seconds": seconds,
        }
    )


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
            result.append({"pid": int(pid), "resumed": False, "gone": True})
        del state["paused_processes"][pid]
    save_state(run_id, state)
    disarm_deadman(run_id, "resume-exporter")
    return result


def exporter_resume(arguments: argparse.Namespace) -> None:
    emit({"processes": resume_processes(checked_run_id(arguments.run_id))})


def log_collector_enable(arguments: argparse.Namespace) -> None:
    """Runtime-enable and start the disabled node log collector (needs approval).

    ``enable --runtime`` writes the link under ``/run`` only, so a reboot (or
    the deadman) returns the unit to its installed, disabled state.
    """

    if not arguments.approved:
        raise ProbeError(
            "starting the node log collector requires the runner's approval"
        )
    run_id = checked_run_id(arguments.run_id)
    seconds = bounded(
        arguments.seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "seconds"
    )
    before = unit_state(LOG_COLLECTOR_UNIT)
    if before.get("LoadState") != "loaded":
        raise ProbeError("the node log collector unit is not installed")
    if before.get("ActiveState") == "active":
        raise ProbeError("the node log collector is already running")
    state = load_state(run_id)
    state["log_collector"] = {"before": before, "enabled_at": utc_now()}
    save_state(run_id, state)
    deadman = arm_deadman(run_id, "logcollector", seconds)
    run(["systemctl", "enable", "--runtime", "--now", LOG_COLLECTOR_UNIT], timeout=120)
    emit(
        {"before": before, "after": unit_state(LOG_COLLECTOR_UNIT), "deadman": deadman}
    )


def restore_log_collector(run_id: str) -> dict[str, Any]:
    state = load_state(run_id)
    if not state.get("log_collector"):
        return {"restored": False, "reason": "not enabled by this run"}
    run(
        [
            "systemctl",
            "kill",
            "--kill-whom=all",
            "--signal=SIGCONT",
            LOG_COLLECTOR_UNIT,
        ],
        check=False,
    )
    run(
        ["systemctl", "disable", "--runtime", "--now", LOG_COLLECTOR_UNIT],
        check=False,
        timeout=120,
    )
    state["paused_units"].pop(LOG_COLLECTOR_UNIT, None)
    state["log_collector"] = None
    save_state(run_id, state)
    disarm_deadman(run_id, "logcollector")
    return {"restored": True, "after": unit_state(LOG_COLLECTOR_UNIT)}


def log_collector_disable(arguments: argparse.Namespace) -> None:
    emit(restore_log_collector(checked_run_id(arguments.run_id)))


# --------------------------------------------------------------------------- #
# Cleanup (the deadman's entry point, and the runner's finally)
# --------------------------------------------------------------------------- #
def cleanup_kind(run_id: str, kind: str, unit: str | None) -> list[dict[str, Any]]:
    """Undo what ``kind`` names (``all``: everything the run recorded)."""

    everything = kind == "all"
    actions: list[dict[str, Any]] = []
    state = load_state(run_id)
    for load in ("cpu", "memory"):
        if everything or kind == load:
            actions.append({load: stop_unit(run_id, load)})
    if everything or kind == "pagecache":
        actions.append(
            {
                "pagecache": stop_unit(run_id, "pagecache"),
                "removed": remove_page_cache_files(run_id),
            }
        )
    if everything:
        for name in [item for item in state["units"] if "-oracle-" in item]:
            run(["systemctl", "stop", f"{name}.service"], check=False)
    paused = list(state["paused_units"]) if everything else [unit] if unit else []
    for item in paused:
        if item in PAUSABLE_UNITS:
            actions.append({"resumed": resume_unit(run_id, item)})
    if everything or kind == "resume-exporter":
        actions.append({"exporter": resume_processes(run_id)})
    if everything or kind == "logcollector":
        actions.append({"log_collector": restore_log_collector(run_id)})
    if everything:
        for root in BACKING_ROOTS:
            directory = root / f"gfrs-live-{run_digest(run_id)}"
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    return actions


def deadman_kinds(run_id: str) -> list[str]:
    """The deadman kinds this run still has a timer or service for."""

    prefix = unit_name(run_id, "deadman-")
    listing = run(
        ["systemctl", "list-units", "--all", "--no-legend", "--plain", f"{prefix}*"],
        check=False,
    ).stdout
    kinds = set()
    for line in listing.splitlines():
        name = line.split()[0] if line.strip() else ""
        if name.startswith(prefix) and name.endswith((".timer", ".service")):
            kinds.add(name[len(prefix) :].rsplit(".", 1)[0])
    return sorted(kinds)


def cleanup(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    kind = arguments.kind
    if kind != "all" and re.fullmatch(r"[a-z0-9-]{1,40}", kind) is None:
        raise ProbeError("unsafe cleanup kind")
    unit = getattr(arguments, "unit", None)
    if unit is not None:
        checked_pausable(unit)
    actions = cleanup_kind(run_id, kind, unit)
    if kind == "all":
        for item in list(load_state(run_id)["units"]):
            run(["systemctl", "stop", f"{item}.service"], check=False)
        for name in deadman_kinds(run_id):
            disarm_deadman(run_id, name)
    else:
        disarm_deadman(run_id, kind)
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


def residue_is_clean(value: dict[str, Any]) -> bool:
    if any(
        value[key] for key in ("page_cache_files", "paused_units", "paused_processes")
    ):
        return False
    if value.get("log_collector_enabled_by_run"):
        return False
    return not [item for item in value["units"] if not item.endswith(".scope")]


def show_residue(arguments: argparse.Namespace) -> None:
    value = residue(checked_run_id(arguments.run_id))
    emit({"residue": value, "clean": residue_is_clean(value)})


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    def command(
        name: str, handler: Any, *, run_id: bool = True
    ) -> argparse.ArgumentParser:
        item = commands.add_parser(name)
        if run_id:
            item.add_argument("--run-id", required=True)
        item.set_defaults(handler=handler)
        return item

    command("survey", survey)
    command("residue", show_residue)
    item = command("cleanup", cleanup)
    item.add_argument("--kind", default="all")
    item.add_argument("--unit", default=None)

    item = command("sample-loop", sample_loop, run_id=False)
    item.add_argument("--output", required=True)
    item.add_argument("--interval", type=float, default=5)
    item.add_argument("--seconds", type=int, required=True)
    item.add_argument("--spec", required=True)
    item = command("oracle-start", oracle_start)
    item.add_argument("--name", required=True)
    item.add_argument("--interval", type=int, default=5)
    item.add_argument("--seconds", type=int, required=True)
    for flag in ("--cpu", "--loadavg", "--meminfo", "--snmp"):
        item.add_argument(flag, action="store_true")
    for flag in ("--statvfs", "--diskstats", "--netdev"):
        item.add_argument(flag, action="append", default=[])
    item = command("oracle-read", oracle_read)
    item.add_argument("--name", required=True)
    item.add_argument("--offset", type=int, default=0)
    item = command("oracle-stop", oracle_stop)
    item.add_argument("--name", required=True)

    item = command("cpu-start", cpu_start)
    item.add_argument("--threads", type=int, required=True)
    item.add_argument("--seconds", type=int, required=True)
    command("cpu-stop", cpu_stop)
    item = command("page-cache-start", page_cache_start)
    item.add_argument("--fraction", type=float, required=True)
    item.add_argument("--seconds", type=int, required=True)
    item.add_argument("--backing", required=True)
    command("page-cache-stop", page_cache_stop)
    item = command("memory-start", memory_start)
    item.add_argument("--target-used-percent", type=float, required=True)
    item.add_argument("--seconds", type=int, required=True)
    item.add_argument("--approved", action="store_true")
    command("memory-stop", memory_stop)

    item = command("unit-pause", unit_pause)
    item.add_argument("--unit", required=True)
    item.add_argument("--seconds", type=int, required=True)
    command("unit-resume", unit_resume).add_argument("--unit", required=True)
    command("exporter-pause", exporter_pause).add_argument(
        "--seconds", type=int, required=True
    )
    command("exporter-resume", exporter_resume)
    item = command("log-collector-enable", log_collector_enable)
    item.add_argument("--seconds", type=int, required=True)
    item.add_argument("--approved", action="store_true")
    command("log-collector-disable", log_collector_disable)
    command("efa-traffic", efa_traffic)
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
