#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, cast


COLLECTOR_ENV = Path("/etc/gpu-fault/collector.env")
# The node installer publishes the control-plane CA to the collectors as
# SSL_CERT_FILE (install-gpu-fault-collector.sh); the publisher reads that key.
COLLECTOR_CA_ENV_KEY = "SSL_CERT_FILE"
FM_LOG = Path("/var/log/fabricmanager.log")
FM_STATE_CANDIDATES = (
    Path("/var/lib/gpu-fault/fabric-manager-collector-state.json"),
    Path("/var/lib/gpu-fault/fabric-manager-offsets.json"),
    Path("/var/lib/gpu-fault/fabric-manager-state.json"),
)
ALLOWED_XIDS = {13, 31, 46, 48, 54, 62, 63, 78, 109}
ALLOWED_SXIDS = {10003, 11001, 12001, 12020, 19084, 23001, 24007, 99999}
ALLOWED_SERVICES = {
    "gpu-fault-dcgm-collector.service",
    "gpu-fault-fabric-manager-collector.service",
    "gpu-fault-host-collector.service",
    "gpu-fault-kernel-collector.service",
}
ENV_KEYS = {
    "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES",
    "GPU_FAULT_DCGM_EDGE_FILTER_ENABLED",
    "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS",
    # The metrics collector's sampling interval. It is *not* named
    # `GPU_FAULT_DCGM_INTERVAL_SECONDS`: the installer never writes such a key
    # and `collectors_cli` never reads one, so asking for that name silently
    # produced an env snapshot with a hole in it.
    "GPU_FAULT_METRICS_INTERVAL_SECONDS",
    "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT",
    "GPU_FAULT_EXPECTED_GPU_COUNT",
    "GPU_FAULT_HOST_EDGE_FILTER_ENABLED",
    "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS",
    "GPU_FAULT_HOST_INTERVAL_SECONDS",
    "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
INSTANCE_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]{0,63}")
SAFE_BDF = re.compile(r"^0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")


def normalize_bdf(value: str) -> str:
    """The one `0000:bb:dd.f` spelling for every way the node writes a BDF.

    `nvidia-smi --query-gpu=pci.bus_id` prints an eight-digit domain
    (`00000000:59:00.0`), the destructive host probe and the collector's own
    inventory print `0000:59:00` without the function, and sysfs prints
    `0000:5e:00.0`. This probe's actions all validate against SAFE_BDF, so a
    snapshot taken by this very probe used to hand back a value its own
    `append-sxid` rejected as "unsafe PCI BDF" (COLLECT-005, 2026-09-06).
    Canonicalise first; the allowlist stays as strict as before.
    """

    text = value.strip().lower()
    parts = text.split(":")
    if len(parts) == 3 and len(parts[0]) == 8 and parts[0].startswith("0000"):
        text = ":".join([parts[0][4:], parts[1], parts[2]])
    if re.fullmatch(r"0000:[0-9a-f]{2}:[0-9a-f]{2}", text):
        text += ".0"
    if SAFE_BDF.fullmatch(text) is None:
        raise ProbeError("unsafe PCI BDF")
    return text


def normalized_bdf_or_raw(value: str) -> str:
    try:
        return normalize_bdf(value)
    except ProbeError:
        return value.strip().lower()


ACCEPTANCE_STATE = Path("/var/lib/gpu-fault/acceptance")
SYSTEMD_UNIT_DIR = Path("/etc/systemd/system")
HOST_COLLECTOR_UNIT = "gpu-fault-host-collector.service"
BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
PROC_ROOT = Path("/proc")
POWER_CGROUP_ROOT = Path("/sys/fs/cgroup")
POWER_TRANSIENT_UNIT_DIR = Path("/run/systemd/transient")
POWER_SMI = Path("/usr/bin/nvidia-smi")
POWER_TIMEOUT = Path("/usr/bin/timeout")
POWER_LOAD_START_WAIT_SECONDS = 30.0
NODE_LEDGER = Path("/var/lib/gpu-fault/node-actions.db")
FM_RECEIPT_PREFIX = "GPU_FAULT_FM_RECEIPT_V1 "
FM_RECEIPT_COUNTERS = (
    "receipt_seq",
    "round_seq",
    "attempt_seq",
    "attempts_total",
    "delivered_total",
    "buffered_total",
    "failed_total",
    "rounds_completed_total",
    "rounds_failed_total",
    "omitted_receipts_total",
)
FM_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "producer_invocation_id",
        "systemd_invocation_id",
        "pid",
        "stage",
        "outcome",
        "source",
        "record_id_sha256",
        "cluster_id_sha256",
        "node_id_sha256",
        "boot_id_sha256",
        "source_config_sha256",
        "counter_exhausted",
        *FM_RECEIPT_COUNTERS,
    }
)
FM_JOURNAL_LIMIT = 2048
# Unit lines read for the un-anchored tail: the collector's plain log lines
# share the unit, so more than the 16 receipts the anchor needs are read.
FM_JOURNAL_TAIL_LINES = 64
FM_UNIT = "gpu-fault-fabric-manager-collector.service"
SAFE_JOURNAL_CURSOR = re.compile(r"^[A-Za-z0-9_=;.:+-]{1,1024}$")
HEX32 = re.compile(r"^[0-9a-f]{32}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ProbeError(RuntimeError):
    pass


class PowerRecoveryIdentityError(ProbeError):
    """A new boot needs explicit reconciliation, not automatic power writes."""


def run(
    command: list[str], *, check: bool = True, timeout: float = 180
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
            f"command failed ({completed.returncode}): "
            f"{shlex.join(command)}: {completed.stderr.strip()}"
        )
    return completed


def collector_runner(
    argv: list[str],
    *,
    capture_output: bool = False,
    text: bool = False,
    timeout: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """``HostTelemetryCollector.runner`` with the ``BoundedProcessRunner`` call shape.

    The Collector calls ``runner(argv, capture_output=True, text=True,
    timeout=..., check=False)``; ``run`` takes only ``check``/``timeout``, so
    handing it over directly made every isolated sample a collection error.
    ``run`` always captures both pipes as text, so the two flags only assert
    the contract; a ``TimeoutExpired`` propagates unchanged because the
    Collector's nvidia-smi circuit breaker classifies it itself.
    """
    if not (capture_output and text):
        raise ProbeError("isolated collector runner only returns captured text")
    return run(list(argv), check=check, timeout=180 if timeout is None else timeout)


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def safe_id(value: str, label: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError(f"unsafe {label}")
    return value


def parse_env() -> dict[str, str]:
    if not COLLECTOR_ENV.is_file():
        raise ProbeError("collector env file does not exist")
    values = {}
    for line in COLLECTOR_ENV.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key in ENV_KEYS:
            values[key] = value.strip().strip("'\"")
    return values


def service_snapshot() -> dict[str, dict[str, str]]:
    result = {}
    units = sorted(
        ALLOWED_SERVICES
        | {
            "gpu-fault-node-agent.service",
            "kubelet.service",
            "nvidia-fabricmanager.service",
            "nvidia-persistenced.service",
        }
    )
    for unit in units:
        completed = run(
            [
                "systemctl",
                "show",
                unit,
                "--property=LoadState",
                "--property=ActiveState",
                "--property=SubState",
                "--property=MainPID",
                "--property=InvocationID",
            ],
            check=False,
        )
        if completed.returncode:
            continue
        values = {}
        for line in completed.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        if values.get("LoadState") != "not-found":
            result[unit] = values
    return result


def gpu_inventory(*, strict: bool = False) -> list[dict[str, str]]:
    completed = run(
        [
            str(POWER_SMI) if strict else "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader",
        ],
        timeout=10 if strict else 180,
    )
    if strict and completed.returncode:
        raise ProbeError("GPU identity query failed")
    result = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",", 3)]
        if len(values) != 4:
            if strict:
                raise ProbeError("GPU identity query contains an incomplete row")
            continue
        result.append(
            {
                "index": values[0],
                "uuid": values[1],
                "pci_bdf": normalized_bdf_or_raw(values[2]),
                "name": values[3],
            }
        )
    return result


def efa_inventory() -> dict[str, Any]:
    root = Path("/sys/class/infiniband")
    devices = []
    if root.is_dir():
        for device in sorted(root.iterdir()):
            pci_bdf = None
            try:
                pci_bdf = device.joinpath("device").resolve().name.lower()
            except OSError:
                pass
            driver = ""
            try:
                driver = device.joinpath("device/driver").resolve().name
            except OSError:
                pass
            uevent = ""
            try:
                uevent = device.joinpath("device/uevent").read_text()
            except OSError:
                pass
            if driver != "efa" and "DRIVER=efa" not in uevent:
                continue
            active_ports = []
            for port in sorted(device.joinpath("ports").glob("*")):
                try:
                    state = port.joinpath("state").read_text().split(":", 1)[0].strip()
                    phys_path = port.joinpath("phys_state")
                    phys = (
                        phys_path.read_text().split(":", 1)[0].strip()
                        if phys_path.is_file()
                        else "5"
                    )
                except OSError:
                    continue
                if state == "4" and phys == "5":
                    active_ports.append(port.name)
            devices.append(
                {
                    "name": device.name,
                    "pci_bdf": pci_bdf,
                    "driver": driver or "uevent",
                    "active_ports": active_ports,
                    "active": bool(active_ports),
                }
            )
    return {
        "discovered_count": len(devices),
        "active_count": sum(bool(item["active"]) for item in devices),
        "devices": devices,
    }


def kernel_collector_fd() -> dict[str, Any]:
    unit = service_snapshot().get("gpu-fault-kernel-collector.service", {})
    pid = int(unit.get("MainPID") or 0)
    matches = []
    if pid > 0:
        for path in Path(f"/proc/{pid}/fd").glob("*"):
            try:
                target = os.readlink(path)
            except OSError:
                continue
            if target == "/dev/kmsg":
                matches.append(path.name)
    return {"pid": pid, "kmsg_fds": sorted(matches)}


def gpu_power_state(*, strict: bool = False) -> list[dict[str, Any]]:
    """Per-GPU power limits and load, the inputs of the throttle candidate.

    The control plane only calls a power violation a candidate when the draw sits
    at the enforced limit *and* utilization is high, so the acceptance run has to
    be able to read all three from the node rather than assume them.
    """

    completed = run(
        [
            str(POWER_SMI) if strict else "nvidia-smi",
            (
                "--query-gpu=index,uuid,power.draw,power.limit,"
                "power.min_limit,power.default_limit,utilization.gpu"
            ),
            "--format=csv,noheader,nounits",
        ],
        check=False,
        timeout=10 if strict else 180,
    )
    if strict and completed.returncode:
        raise ProbeError("GPU power query failed")
    fields = (
        "index",
        "uuid",
        "power_draw_w",
        "power_limit_w",
        "power_min_limit_w",
        "power_default_limit_w",
        "utilization_percent",
    )
    result = []
    for row in csv.reader(completed.stdout.splitlines(), strict=True):
        values = [item.strip() for item in row]
        if len(values) != len(fields):
            if strict:
                raise ProbeError("GPU power query contains an incomplete row")
            continue
        entry: dict[str, Any] = {}
        for name, value in zip(fields, values, strict=True):
            if name == "uuid":
                entry[name] = value
            elif name == "index":
                entry[name] = int(value)
            else:
                try:
                    entry[name] = float(value)
                except ValueError:
                    entry[name] = None
                if strict and (entry[name] is None or not math.isfinite(entry[name])):
                    raise ProbeError("GPU power query contains an unknown value")
        result.append(entry)
    if strict and not result:
        raise ProbeError("GPU power query is empty")
    return result


def proftester_binary() -> str:
    """The DCGM load generator installed on this node.

    The binary carries the DCGM major version in its name
    (``dcgmproftester13``), so the acceptance run must discover it instead of
    pinning a version that a driver bump will rename.
    """

    candidates = sorted(
        (path for path in Path("/usr/bin").glob("dcgmproftester*") if path.is_file()),
        key=lambda path: path.name,
    )
    if not candidates:
        raise ProbeError("no dcgmproftester binary is installed on this node")
    return str(candidates[-1])


def power_limit_restore_unit(run_id: str) -> str:
    digest = hashlib.sha256(safe_id(run_id, "run ID").encode()).hexdigest()[:16]
    return f"gpu-fault-power-limit-restore-{digest}"


def power_record_path(run_id: str) -> Path:
    return ACCEPTANCE_STATE / f"{power_limit_restore_unit(run_id)}.json"


def power_load_start_path(run_id: str) -> Path:
    return power_record_path(run_id).with_suffix(".load-start.json")


def power_sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def power_read_file(path: Path, *, shared_lock: bool = False) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if shared_lock:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_mode & 0o022
            or info.st_size > 1024 * 1024
        ):
            raise ProbeError("power recovery file is not a trusted regular file")
        return stream.read()


def power_read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(power_read_file(path))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise ProbeError("power recovery record is not an object")
    return payload


def power_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        power_sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def power_create_file(path: Path, contents: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    power_sync_directory(path.parent)


@contextmanager
def power_operation_lock() -> Iterator[None]:
    ACCEPTANCE_STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = ACCEPTANCE_STATE.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o022
    ):
        raise ProbeError("power recovery directory is not trusted")
    descriptor = os.open(
        ACCEPTANCE_STATE / "gpu-power.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
        0o600,
    )
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            raise ProbeError("power operation lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProbeError("another power operation is still executing") from exc
        yield
    finally:
        os.close(descriptor)


def checked_power_state(
    baseline: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    count_text = parse_env().get("GPU_FAULT_EXPECTED_GPU_COUNT", "")
    if not count_text.isdecimal() or int(count_text) < 1:
        raise ProbeError("expected GPU count is unknown")
    state = gpu_power_state(strict=True)
    inventory = gpu_inventory(strict=True)
    count = int(count_text)
    for entries in (state, inventory):
        indexes = [int(item["index"]) for item in entries]
        uuids = [item["uuid"] for item in entries]
        if (
            len(entries) != count
            or set(indexes) != set(range(count))
            or len(set(uuids)) != count
            or any(
                not isinstance(uuid, str)
                or re.fullmatch(
                    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid
                )
                is None
                for uuid in uuids
            )
        ):
            raise ProbeError("GPU inventory is incomplete or has duplicate identities")
    devices = {int(item["index"]): item for item in inventory}
    for item in state:
        device = devices[item["index"]]
        if item["uuid"] != device["uuid"]:
            raise ProbeError("GPU UUID/index binding changed during the power query")
        item["pci_bdf"] = normalize_bdf(device["pci_bdf"])
        if (
            not 0 < item["power_min_limit_w"] < item["power_default_limit_w"]
            or item["power_limit_w"] <= 0
            or item["power_draw_w"] < 0
            or not 0 <= item["utilization_percent"] <= 100
        ):
            raise ProbeError("GPU power limits or load are invalid")
    if len({item["pci_bdf"] for item in state}) != count:
        raise ProbeError("GPU inventory contains duplicate PCI identities")
    state.sort(key=lambda item: item["index"])
    if baseline is not None:
        keys = (
            "index",
            "uuid",
            "pci_bdf",
            "power_min_limit_w",
            "power_default_limit_w",
        )
        if len(state) != len(baseline) or any(
            any(current[key] != original[key] for key in keys)
            for current, original in zip(state, baseline, strict=True)
        ):
            raise ProbeError("GPU identity or power capability differs from baseline")
    return state


def power_recovery_command(record: dict[str, Any], *, load: bool = False) -> list[str]:
    script = power_record_path(record["run_id"]).with_suffix(".py")
    return [
        record["python"],
        "-I",
        "-S",
        "-B",
        str(script),
        "throttle-gpu" if load else "restore-gpu-power-limit",
        "--run-id",
        record["run_id"],
        "--load-only" if load else "--automatic",
    ]


def power_unit_contents(record: dict[str, Any]) -> dict[str, bytes]:
    unit = power_limit_restore_unit(record["run_id"])
    description = f"GPU fault power acceptance {record['run_id']}"
    commands = [
        power_recovery_command(record),
        power_recovery_command(record, load=True),
    ]
    if any(
        re.fullmatch(r"[A-Za-z0-9_./:+-]+", arg) is None
        for cmd in commands
        for arg in cmd
    ):
        raise ProbeError("power recovery command contains unsupported path characters")
    return {
        f"{unit}.service": (
            f"[Unit]\nDescription={description}\nAfter=local-fs.target\n"
            "StartLimitIntervalSec=0\n[Service]\nType=oneshot\n"
            f"ExecStart={' '.join(commands[0])}\n"
            "Restart=on-failure\nRestartSec=5s\nRestartPreventExitStatus=78\n"
            "TimeoutStartSec=300s\n"
            "[Install]\nWantedBy=multi-user.target\n"
        ).encode(),
        f"{unit}.timer": (
            f"[Unit]\nDescription={description}\n[Timer]\n"
            f"OnBootSec={record['restore_deadline_monotonic']:.3f}s\n"
            f"Unit={unit}.service\nAccuracySec=1ms\nRandomizedDelaySec=0\n"
            "[Install]\nWantedBy=timers.target\n"
        ).encode(),
        f"{unit}-load.service": (
            f"[Unit]\nDescription={description}\n[Service]\nType=exec\n"
            f"ExecStart={' '.join(commands[1])}\n"
            f"RuntimeMaxSec={record['load_seconds'] + 5}s\n"
            "TimeoutStopSec=5s\nKillMode=control-group\nSendSIGKILL=yes\n"
            "Restart=no\n"
        ).encode(),
    }


def power_unit_state(name: str) -> dict[str, str]:
    properties = (
        "Id",
        "LoadState",
        "ActiveState",
        "SubState",
        "FragmentPath",
        "DropInPaths",
        "NeedDaemonReload",
        "ExecStart",
        "Type",
        "Restart",
        "RestartUSec",
        "RestartPreventExitStatus",
        "TimeoutStartUSec",
        "RuntimeMaxUSec",
        "TimeoutStopUSec",
        "KillMode",
        "SendSIGKILL",
        "MainPID",
        "ControlPID",
        "ControlGroup",
        "Job",
        "Unit",
        "NextElapseUSecMonotonic",
        "AccuracyUSec",
        "RandomizedDelayUSec",
    )
    result = run(
        ["systemctl", "show", name, "--property=" + ",".join(properties)],
        check=False,
        timeout=10,
    )
    state = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if state.get("Id") != name or (
        result.returncode
        and not (result.returncode in {1, 4} and state.get("LoadState") == "not-found")
    ):
        raise ProbeError("power unit state is unknown")
    if state.get("LoadState") not in {"loaded", "not-found"}:
        raise ProbeError("power unit could not be loaded")
    return state


def power_systemd_seconds(value: str) -> float:
    if value.isdecimal():
        return int(value) / 1_000_000
    units = {
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1,
        "min": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
        "month": 2629800,
        "y": 31557600,
    }
    parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)(us|ms|s|min|h|d|w|month|y)", value)
    if not parts or "".join(number + unit for number, unit in parts) != value.replace(
        " ", ""
    ):
        raise ProbeError("systemd power deadline is unknown")
    seconds = sum(float(number) * units[unit] for number, unit in parts)
    if not math.isfinite(seconds):
        raise ProbeError("systemd power deadline is not finite")
    return seconds


def power_owned_unit(
    record: dict[str, Any], name: str, *, allow_absent: bool = False
) -> dict[str, str]:
    contents = power_unit_contents(record)[name]
    state = power_unit_state(name)
    path = SYSTEMD_UNIT_DIR / name
    if state["LoadState"] == "not-found":
        if allow_absent:
            if not path.exists() and not path.is_symlink():
                return state
            if power_read_file(path) == contents:
                return state
        raise ProbeError("owned power unit is missing")
    if (
        power_read_file(path) != contents
        or state.get("FragmentPath") != str(path)
        or state.get("DropInPaths") != ""
        or state.get("NeedDaemonReload") != "no"
    ):
        raise ProbeError("power unit ownership or configuration changed")
    if name.endswith(".service"):
        command = power_recovery_command(record, load=name.endswith("-load.service"))
        starts = re.findall(r"argv\[\]=(.*?) ;", state.get("ExecStart", ""))
        paths = re.findall(r"\bpath=(\S+) ;", state.get("ExecStart", ""))
        if (
            len(starts) != 1
            or shlex.split(starts[0]) != command
            or paths != [command[0]]
        ):
            raise ProbeError("power unit command differs from the owned command")
    return state


def power_verify_timer(record: dict[str, Any]) -> dict[str, Any]:
    unit = power_limit_restore_unit(record["run_id"])
    timer = power_owned_unit(record, f"{unit}.timer")
    service = power_owned_unit(record, f"{unit}.service")
    load = power_owned_unit(record, f"{unit}-load.service")
    deadline = power_systemd_seconds(timer.get("NextElapseUSecMonotonic", ""))
    if (
        timer.get("ActiveState") != "active"
        or timer.get("SubState") != "waiting"
        or timer.get("Unit") != f"{unit}.service"
        or abs(deadline - record["restore_deadline_monotonic"]) > 0.002
        or power_systemd_seconds(timer.get("AccuracyUSec", "")) > 0.001
        or power_systemd_seconds(timer.get("RandomizedDelayUSec", "")) != 0
        or service.get("Type") != "oneshot"
        or service.get("Restart") != "on-failure"
        or service.get("RestartPreventExitStatus") != "78"
        or power_systemd_seconds(service.get("RestartUSec", "")) != 5
        or power_systemd_seconds(service.get("TimeoutStartUSec", "")) != 300
        or load.get("Type") != "exec"
        or load.get("Restart") != "no"
        or load.get("KillMode") != "control-group"
        or load.get("SendSIGKILL") != "yes"
        or power_systemd_seconds(load.get("TimeoutStopUSec", "")) != 5
        or power_systemd_seconds(load.get("RuntimeMaxUSec", ""))
        != record["load_seconds"] + 5
    ):
        raise ProbeError("owned power timer or load deadline could not be verified")
    power_require_load_time(record)
    return {"armed": True, "unit": f"{unit}.timer", "deadline_monotonic": deadline}


def power_require_load_time(record: dict[str, Any]) -> None:
    if (
        BOOT_ID_FILE.read_text().strip() != record["boot_id"]
        or time.monotonic() + record["load_seconds"] + 10
        >= record["load_hard_deadline_monotonic"]
    ):
        raise ProbeError("GPU load cannot finish before the fixed recovery deadline")


def power_intent_digest(record: dict[str, Any]) -> str:
    immutable = (
        "schema_version",
        "run_id",
        "boot_id",
        "baseline",
        "gpu_index",
        "load_seconds",
        "restore_seconds",
        "restore_deadline_monotonic",
        "load_hard_deadline_monotonic",
        "python",
        "load_binary",
        "probe_sha256",
    )
    body = json.dumps(
        {key: record.get(key) for key in immutable}, sort_keys=True, allow_nan=False
    ).encode()
    return hashlib.sha256(body).hexdigest()


def power_load_record(run_id: str) -> dict[str, Any] | None:
    owner = power_read_json(ACCEPTANCE_STATE / "gpu-power-owner.json")
    if owner is not None and owner.get("run_id") != run_id:
        raise ProbeError("another power operation owns this node")
    record = power_read_json(power_record_path(run_id))
    if record is None:
        if owner is not None:
            raise ProbeError("owned power recovery record is missing")
        return None
    if record.get("boot_id") != BOOT_ID_FILE.read_text().strip():
        raise PowerRecoveryIdentityError(
            "power recovery boot identity changed; manual confirmation is required"
        )
    if (
        record.get("schema_version") != 1
        or record.get("run_id") != run_id
        or record.get("intent_sha256") != power_intent_digest(record)
        or owner is not None
        and owner.get("intent_sha256") != record["intent_sha256"]
        or record.get("phase")
        not in {"PREPARED", "MUTATING", "RESTORING", "RESTORED", "CLEANED"}
        or any(
            type(record.get(key)) is not bool
            for key in ("mutation_started", "load_intended", "load_closed")
        )
        or not record.get("baseline")
        or not record.get("boot_id")
        or any(
            type(record.get(key)) not in {float, int}
            or not math.isfinite(record[key])
            or record[key] <= 0
            for key in (
                "restore_deadline_monotonic",
                "load_hard_deadline_monotonic",
                "load_seconds",
            )
        )
        or record["restore_deadline_monotonic"] - record["load_hard_deadline_monotonic"]
        != 15
    ):
        raise ProbeError("power recovery identity or record is invalid")
    if owner is None and record["mutation_started"] and record["phase"] != "CLEANED":
        raise ProbeError("power mutation lost its node ownership record")
    if record["phase"] != "CLEANED":
        script = power_record_path(run_id).with_suffix(".py")
        if (
            hashlib.sha256(power_read_file(script)).hexdigest()
            != record["probe_sha256"]
        ):
            raise ProbeError("power recovery probe digest changed")
    return record


def power_assert_stopped(state: dict[str, str], name: str) -> None:
    if (
        state.get("ActiveState") not in {"inactive", "failed"}
        or state.get("SubState") not in {"dead", "failed"}
        or state.get("MainPID", "0") != "0"
        or state.get("ControlPID", "0") != "0"
        or state.get("Job") not in {"", "0", "[not set]"}
    ):
        raise ProbeError("owned power process or systemd job is not stopped")
    group = f"/system.slice/{name}"
    if state.get("ControlGroup", "") not in {"", group}:
        raise ProbeError("power load has an unexpected cgroup")
    path = POWER_CGROUP_ROOT / group.lstrip("/")
    try:
        events = dict(
            line.split() for line in (path / "cgroup.events").read_text().splitlines()
        )
    except FileNotFoundError:
        if path.exists():
            raise ProbeError("power cgroup population is unknown") from None
    else:
        if events.get("populated") != "0":
            raise ProbeError("owned power cgroup still contains processes")


def power_stop_load(record: dict[str, Any]) -> None:
    name = power_limit_restore_unit(record["run_id"]) + "-load.service"
    state = power_owned_unit(record, name, allow_absent=True)
    if state["LoadState"] != "not-found":
        run(["systemctl", "stop", name], timeout=20)
        state = power_owned_unit(record, name)
    power_assert_stopped(state, name)


def power_check_limits(
    record: dict[str, Any], state: list[dict[str, Any]], *, restored: bool = False
) -> None:
    for current, original in zip(state, record["baseline"], strict=True):
        allowed = {original["power_limit_w"]}
        if not restored:
            allowed.add(original["power_min_limit_w"])
        if current["power_limit_w"] not in allowed:
            raise ProbeError("GPU power limit drifted outside this operation")


def power_finish_units(record: dict[str, Any]) -> None:
    unit = power_limit_restore_unit(record["run_id"])
    for name in (f"{unit}.timer", f"{unit}.service"):
        state = power_owned_unit(record, name, allow_absent=True)
        if state["LoadState"] != "not-found":
            run(["systemctl", "disable", "--now", name], timeout=30)
            state = power_owned_unit(record, name)
        power_assert_stopped(state, name)
    for name, contents in power_unit_contents(record).items():
        path = SYSTEMD_UNIT_DIR / name
        if path.exists() or path.is_symlink():
            if power_read_file(path) != contents:
                raise ProbeError("refusing to remove an unowned power unit")
            path.unlink()
    power_sync_directory(SYSTEMD_UNIT_DIR)
    run(["systemctl", "daemon-reload"], timeout=20)
    power_verify_removed_units(record)


def power_verify_removed_units(record: dict[str, Any]) -> None:
    for name in power_unit_contents(record):
        state = power_unit_state(name)
        power_assert_stopped(state, name)
        if state["LoadState"] != "not-found":
            raise ProbeError("power unit remains installed after cleanup")


def power_create_load_start(record: dict[str, Any]) -> None:
    path = power_load_start_path(record["run_id"])
    gpu = next(
        item for item in record["baseline"] if item["index"] == record["gpu_index"]
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            # Readers cannot accept the linked receipt until publication is durable.
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            receipt = {
                "run_id": record["run_id"],
                "boot_id": record["boot_id"],
                "intent_sha256": record["intent_sha256"],
                "gpu_index": record["gpu_index"],
                "gpu_uuid": gpu["uuid"],
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "started_monotonic": time.monotonic(),
            }
            stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False).encode())
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
            temporary.unlink()
            power_sync_directory(path.parent)
            power_require_load_time(record)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    # Retain this create-only receipt with the journal, including after cleanup.


def power_wait_load_start(
    record: dict[str, Any], *, requested_at: datetime, requested_monotonic: float
) -> dict[str, Any]:
    latest_start = record["load_hard_deadline_monotonic"] - record["load_seconds"] - 10
    deadline = min(requested_monotonic + POWER_LOAD_START_WAIT_SECONDS, latest_start)
    path = power_load_start_path(record["run_id"])
    while True:
        power_require_load_time(record)
        if time.monotonic() >= deadline:
            raise ProbeError(
                "owned GPU load start receipt did not arrive before deadline"
            )
        try:
            receipt = json.loads(power_read_file(path, shared_lock=True))
        except (FileNotFoundError, BlockingIOError):
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.1, remaining))
            continue
        if not isinstance(receipt, dict) or set(receipt) != {
            "run_id",
            "boot_id",
            "intent_sha256",
            "gpu_index",
            "gpu_uuid",
            "pid",
            "started_at",
            "started_monotonic",
        }:
            raise ProbeError("GPU load start receipt has an invalid shape")
        gpu = next(
            item for item in record["baseline"] if item["index"] == record["gpu_index"]
        )
        if (
            any(
                receipt[key] != record[key]
                for key in ("run_id", "boot_id", "intent_sha256", "gpu_index")
            )
            or type(receipt["gpu_index"]) is not int
            or receipt["gpu_uuid"] != gpu["uuid"]
            or type(receipt["pid"]) is not int
            or receipt["pid"] <= 1
        ):
            raise ProbeError("GPU load start receipt does not match its owned intent")
        started = receipt["started_monotonic"]
        if (
            type(started) not in {float, int}
            or not math.isfinite(started)
            or started <= 0
            or not requested_monotonic <= started <= time.monotonic()
            or started >= latest_start
            or not isinstance(receipt["started_at"], str)
        ):
            raise ProbeError("GPU load start receipt has an invalid timestamp")
        try:
            started_at = datetime.fromisoformat(receipt["started_at"])
        except ValueError as exc:
            raise ProbeError(
                "GPU load start receipt has an invalid UTC timestamp"
            ) from exc
        if (
            started_at.tzinfo != timezone.utc
            or not requested_at <= started_at <= datetime.now(timezone.utc)
        ):
            raise ProbeError("GPU load start receipt has an invalid UTC timestamp")
        load = power_owned_unit(
            record, power_limit_restore_unit(record["run_id"]) + "-load.service"
        )
        if load.get("ActiveState") != "active" or load.get("MainPID") != str(
            receipt["pid"]
        ):
            raise ProbeError("owned GPU load service does not match its start receipt")
        power_require_load_time(record)
        if time.monotonic() >= deadline:
            raise ProbeError(
                "owned GPU load start receipt was not verified before deadline"
            )
        return receipt


def power_load_only(arguments: argparse.Namespace) -> None:
    # The service starts this same, durable probe. Late starts cannot renew its budget.
    record = power_load_record(safe_id(arguments.run_id, "run ID"))
    if record is None or not record["load_intended"] or record["load_closed"]:
        raise ProbeError("GPU load no longer has an owned start intent")
    state = checked_power_state(record["baseline"])
    if any(item["power_limit_w"] != item["power_min_limit_w"] for item in state):
        raise ProbeError("GPU load requires the verified power cap")
    power_require_load_time(record)
    power_create_load_start(record)
    os.execv(
        str(POWER_TIMEOUT),
        [
            str(POWER_TIMEOUT),
            "--signal=TERM",
            "--kill-after=5s",
            f"{record['load_seconds'] + 5}s",
            record["load_binary"],
            "--no-dcgm-validation",
            "-t",
            "1004",
            "-d",
            str(record["load_seconds"]),
            "-i",
            str(record["gpu_index"]),
        ],
    )


def throttle_gpu(arguments: argparse.Namespace) -> None:
    """Cap the complete, default-power GPU set under an owned recovery intent."""
    if getattr(arguments, "load_only", False):
        power_load_only(arguments)
        return
    run_id = safe_id(arguments.run_id, "run ID")
    if (
        type(arguments.load_seconds) is not int
        or type(arguments.restore_seconds) is not int
        or not 60 <= arguments.restore_seconds <= 900
        or not 0 < arguments.load_seconds <= arguments.restore_seconds - 60
    ):
        raise ProbeError("GPU load must end before the deadman restore fires")
    with power_operation_lock():
        owner_path = ACCEPTANCE_STATE / "gpu-power-owner.json"
        if power_read_json(owner_path) is not None:
            raise ProbeError("another power operation owns this node")
        path = power_record_path(run_id)
        if path.exists() or path.is_symlink():
            raise ProbeError("power run ID has already been used; cleanup only")
        receipt_path = power_load_start_path(run_id)
        if receipt_path.exists() or receipt_path.is_symlink():
            raise ProbeError("power load start receipt already exists; cleanup only")
        for directory in (SYSTEMD_UNIT_DIR, POWER_TRANSIENT_UNIT_DIR):
            if (
                next(directory.glob("gpu-fault-power-limit-restore-*"), None)
                is not None
            ):
                raise ProbeError("another power recovery unit already exists")
        baseline = checked_power_state()
        if arguments.gpu_index not in {item["index"] for item in baseline}:
            raise ProbeError("GPU index is not present on this node")
        if any(
            item["power_limit_w"] != item["power_default_limit_w"] for item in baseline
        ):
            raise ProbeError("refusing to overwrite a nondefault GPU power limit")
        if not (POWER_CGROUP_ROOT / "cgroup.controllers").is_file():
            raise ProbeError("verified load cleanup requires cgroup v2")
        binary = proftester_binary()
        for executable in (Path(binary), POWER_SMI, POWER_TIMEOUT):
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ProbeError("power operation executable is unavailable")
        source = Path(__file__).read_bytes()
        deadline = (
            math.ceil((time.monotonic() + arguments.restore_seconds) * 1000) / 1000
        )
        record: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "phase": "PREPARED",
            "boot_id": BOOT_ID_FILE.read_text().strip(),
            "baseline": baseline,
            "mutation_started": False,
            "load_intended": False,
            "load_closed": False,
            "gpu_index": arguments.gpu_index,
            "load_seconds": arguments.load_seconds,
            "restore_seconds": arguments.restore_seconds,
            "restore_deadline_monotonic": deadline,
            "load_hard_deadline_monotonic": deadline - 15,
            "python": str(Path(sys.executable).resolve()),
            "load_binary": binary,
            "probe_sha256": hashlib.sha256(source).hexdigest(),
        }
        if not record["boot_id"]:
            raise ProbeError("power operation has no boot identity")
        record["intent_sha256"] = power_intent_digest(record)
        units = power_unit_contents(record)
        for name in units:
            if (SYSTEMD_UNIT_DIR / name).exists() or (
                SYSTEMD_UNIT_DIR / name
            ).is_symlink():
                raise ProbeError("power recovery unit file already exists")
            if power_unit_state(name)["LoadState"] != "not-found":
                raise ProbeError("power recovery unit already exists")
        power_create_file(path.with_suffix(".py"), source, 0o500)
        power_write_json(path, record)
        power_write_json(
            owner_path, {"run_id": run_id, "intent_sha256": record["intent_sha256"]}
        )
        for name, contents in units.items():
            power_create_file(SYSTEMD_UNIT_DIR / name, contents, 0o644)
        run(["systemctl", "daemon-reload"], timeout=20)
        unit = power_limit_restore_unit(run_id)
        run(["systemctl", "enable", f"{unit}.service"], timeout=20)
        run(["systemctl", "enable", "--now", f"{unit}.timer"], timeout=20)
        timer = power_verify_timer(record)
        record["phase"] = "MUTATING"
        record["mutation_started"] = True
        power_write_json(path, record)
        for original in baseline:
            state = checked_power_state(baseline)
            power_check_limits(record, state)
            power_require_load_time(record)
            run(
                [
                    str(POWER_SMI),
                    "-i",
                    original["uuid"],
                    "-pl",
                    str(original["power_min_limit_w"]),
                ],
                timeout=10,
            )
        state = checked_power_state(baseline)
        if any(item["power_limit_w"] != item["power_min_limit_w"] for item in state):
            raise ProbeError("GPU cap readback does not match the owned target")
        timer = power_verify_timer(record)
        record["load_intended"] = True
        power_write_json(path, record)
        requested_at = datetime.now(timezone.utc)
        requested_monotonic = time.monotonic()
        run(["systemctl", "start", f"{unit}-load.service"], timeout=10)
        load_start = power_wait_load_start(
            record, requested_at=requested_at, requested_monotonic=requested_monotonic
        )
        emit(
            {
                **record,
                "gpu_power": state,
                "timer_proof": timer,
                "timer_armed": True,
                "load_unit": f"{unit}-load.service",
                "restore_unit": f"{unit}.timer",
                "load_unit_active": True,
                "load_start": load_start,
                "cleanup_verified": False,
            }
        )


def restore_gpu_power_limit(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    automatic = getattr(arguments, "automatic", False)
    with power_operation_lock():
        record = power_load_record(run_id)
        if record is None:
            emit(
                {
                    "run_id": run_id,
                    "restored": False,
                    "no_mutation": True,
                    "mutation_started": False,
                    "cleanup_verified": True,
                    "load_stopped": True,
                    "timer_disarmed": True,
                    "gpu_power": [],
                }
            )
            return
        if automatic and time.monotonic() < record["restore_deadline_monotonic"]:
            emit(
                {
                    "run_id": run_id,
                    "restored": False,
                    "cleanup_verified": False,
                    "deferred": True,
                }
            )
            return
        path = power_record_path(run_id)
        if record["phase"] != "CLEANED":
            record["load_closed"] = True
            record["phase"] = "RESTORING"
            power_write_json(path, record)
            power_stop_load(record)
        state: list[dict[str, Any]] = []
        if record["mutation_started"]:
            state = checked_power_state(record["baseline"])
            power_check_limits(record, state, restored=record["phase"] == "CLEANED")
            for current, original in zip(state, record["baseline"], strict=True):
                if current["power_limit_w"] != original["power_limit_w"]:
                    run(
                        [
                            str(POWER_SMI),
                            "-i",
                            original["uuid"],
                            "-pl",
                            str(original["power_limit_w"]),
                        ],
                        timeout=10,
                    )
            state = checked_power_state(record["baseline"])
            power_check_limits(record, state, restored=True)
        if automatic and record["phase"] != "CLEANED":
            record["phase"] = "RESTORED"
            power_write_json(path, record)
            emit(
                {
                    "run_id": run_id,
                    "restored": record["mutation_started"],
                    "no_mutation": not record["mutation_started"],
                    "load_stopped": True,
                    "timer_disarmed": False,
                    "cleanup_verified": False,
                    "cleanup_deferred": True,
                    "gpu_power": state,
                }
            )
            return
        if record["phase"] != "CLEANED":
            record["phase"] = "RESTORED"
            power_write_json(path, record)
            power_finish_units(record)
            record["phase"] = "CLEANED"
            power_write_json(path, record)
        else:
            power_verify_removed_units(record)
        path.with_suffix(".py").unlink(missing_ok=True)
        (ACCEPTANCE_STATE / "gpu-power-owner.json").unlink(missing_ok=True)
        power_sync_directory(ACCEPTANCE_STATE)
        emit(
            {
                "run_id": run_id,
                "restored": record["mutation_started"],
                "no_mutation": not record["mutation_started"],
                "mutation_started": record["mutation_started"],
                "cleanup_verified": True,
                "load_stopped": True,
                "timer_disarmed": True,
                "gpu_power": state,
                "baseline": record["baseline"],
            }
        )


def persistence_mode() -> bool | None:
    completed = run(
        ["nvidia-smi", "--query-gpu=persistence_mode", "--format=csv,noheader"],
        check=False,
    )
    values = {
        line.strip().lower() for line in completed.stdout.splitlines() if line.strip()
    }
    if not values:
        return None
    if values <= {"enabled"}:
        return True
    if values <= {"disabled"}:
        return False
    return None


def file_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "inode": stat.st_ino,
        "device": stat.st_dev,
        "sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() and stat.st_size <= 2 * 1024 * 1024
            else None
        ),
    }


def fabric_manager_cursor() -> dict[str, Any] | None:
    """The Fabric Manager file collector's persisted cursor, decoded.

    COLLECT-005 compared the log's size to itself and called that "the cursor
    advanced". The cursor is the ``files[<path>]`` entry of the collector's
    state file -- device, inode and byte offset -- and only that entry can say
    whether the collector caught up to the line the case appended and kept
    that position across a restart.
    """

    for path in FM_STATE_CANDIDATES:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"path": str(path), "error": "state file is not JSON"}
        files = value.get("files") if isinstance(value, dict) else None
        return {
            "path": str(path),
            "files": files if isinstance(files, dict) else {},
            "journal_cursor": (
                value.get("journal_cursor") if isinstance(value, dict) else None
            ),
        }
    return None


def fm_cursor(_arguments: argparse.Namespace) -> None:
    """The light read COLLECT-005 polls after restarting the FM collector."""

    emit(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "service": service_snapshot().get(FM_UNIT),
            "fabric_manager_log": file_snapshot(FM_LOG),
            "fabric_manager_cursor": fabric_manager_cursor(),
        }
    )


def checked_fm_receipt(value: Any) -> dict[str, Any]:
    """Reject unknown fields before returning any journal content to the runner."""

    if not isinstance(value, dict) or set(value) != FM_RECEIPT_KEYS:
        raise ProbeError("FM receipt fields differ from schema v1")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ProbeError("FM receipt schema version is unsupported")
    if value["counter_exhausted"] is not False:
        raise ProbeError("FM receipt counter capacity is exhausted or unknown")
    if (
        not HEX32.fullmatch(str(value["producer_invocation_id"]))
        or not HEX32.fullmatch(str(value["systemd_invocation_id"]))
        or type(value["pid"]) is not int
        or value["pid"] <= 1
        or any(
            not HEX64.fullmatch(str(value[key]))
            for key in (
                "cluster_id_sha256",
                "node_id_sha256",
                "boot_id_sha256",
                "source_config_sha256",
            )
        )
    ):
        raise ProbeError("FM receipt producer identity is malformed")
    if any(
        type(value[key]) is not int or value[key] < 0 for key in FM_RECEIPT_COUNTERS
    ):
        raise ProbeError("FM receipt counter is malformed")
    if value["receipt_seq"] < 1 or value["round_seq"] < 1:
        raise ProbeError("FM receipt sequence must be positive")
    outcomes = {
        "ATTEMPT": {"STARTED"},
        "COMPLETION": {"DELIVERED", "BUFFERED", "FAILED"},
        "ROUND": {"COMPLETE", "FAILED"},
    }
    stage = value["stage"]
    if (
        not isinstance(stage, str)
        or not isinstance(value["outcome"], str)
        or value["outcome"] not in outcomes.get(stage, set())
    ):
        raise ProbeError("FM receipt stage/outcome is invalid")
    if stage == "ROUND":
        if value["source"] is not None or value["record_id_sha256"] is not None:
            raise ProbeError("FM round receipt contains a record identity")
    elif (
        not isinstance(value["source"], str)
        or value["source"] not in {"file", "journal"}
        or not HEX64.fullmatch(str(value["record_id_sha256"]))
    ):
        raise ProbeError("FM record receipt has no bound source/digest")
    return value


def active_unit_identity(unit: str, what: str) -> tuple[dict[str, Any], str]:
    """The unit's snapshot and MainPID, refused unless it is a running process."""

    before = service_snapshot().get(unit) or {}
    pid = str(before.get("MainPID") or "")
    if (
        before.get("ActiveState") != "active"
        or not pid.isdecimal()
        or int(pid) <= 1
        or not before.get("InvocationID")
    ):
        raise ProbeError(f"{what} process identity is unknown")
    return before, pid


def fm_delivery_evidence(arguments: argparse.Namespace) -> None:
    """Read an inclusive, bounded receipt window without exposing general logs.

    The window is the running boot's only: ``__MONOTONIC_TIMESTAMP`` restarts
    at every boot, so an un-anchored tail spanning a reboot fails the
    journal-order check for hours (live COLLECT-005, 2026-09-21). journald is
    asked for the boot and every row's ``_BOOT_ID`` is checked regardless.
    """

    cursor = arguments.cursor
    if cursor and not SAFE_JOURNAL_CURSOR.fullmatch(cursor):
        raise ProbeError("FM journal cursor is invalid")
    boot_id = BOOT_ID_FILE.read_text().strip().replace("-", "")
    if not HEX32.fullmatch(boot_id):
        raise ProbeError("host boot id is unknown")
    before, _ = active_unit_identity(FM_UNIT, "FM producer")
    if not HEX32.fullmatch(str(before["InvocationID"])):
        raise ProbeError("FM producer process identity is unknown")
    # No ``--grep``: systemd 252 hands ``--grep`` matches back newest-first once
    # ``--lines`` is given and walks *backwards* from ``--cursor`` (live
    # 2026-09-17). The receipt filter is applied here, on the unit's own lines,
    # and the window must arrive in journal order.
    command = [
        "journalctl",
        f"--unit={FM_UNIT}",
        "--output=json",
        "--no-pager",
        "--quiet",
        "--all",
        "--output-fields=MESSAGE,__CURSOR,_SYSTEMD_INVOCATION_ID,_PID,"
        "__MONOTONIC_TIMESTAMP,_BOOT_ID",
        (
            f"--lines=+{FM_JOURNAL_LIMIT + 1}"
            if cursor
            else f"--lines={FM_JOURNAL_TAIL_LINES}"
        ),
    ]
    # An anchor cursor already lies inside this boot (the producer identity is
    # re-checked below); only the un-anchored tail needs journald's boot scope.
    command.append(f"--cursor={cursor}" if cursor else f"--boot={boot_id}")
    completed = run(command, check=False, timeout=60)
    if completed.returncode or len(completed.stdout.encode()) > 4 * 1024 * 1024:
        raise ProbeError("FM journal receipt read failed or exceeded its bound")
    records = []
    unit_lines = 0
    previous_clock = -1
    try:
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["_BOOT_ID"] != boot_id:
                raise ProbeError("FM journal window crosses a boot boundary")
            unit_lines += 1
            message = entry["MESSAGE"]
            if not isinstance(message, str) or FM_RECEIPT_PREFIX not in message:
                continue  # the collector's plain log line: never exported
            if message.count(FM_RECEIPT_PREFIX) != 1:
                raise ProbeError("FM journal entry has no unique receipt payload")
            body = checked_fm_receipt(
                json.loads(message.split(FM_RECEIPT_PREFIX, 1)[1])
            )
            record_cursor = entry["__CURSOR"]
            if (
                not isinstance(record_cursor, str)
                or not SAFE_JOURNAL_CURSOR.fullmatch(record_cursor)
                or entry["_SYSTEMD_INVOCATION_ID"] != body["systemd_invocation_id"]
                or str(entry["_PID"]) != str(body["pid"])
                or not str(entry["__MONOTONIC_TIMESTAMP"]).isdecimal()
            ):
                raise ProbeError("FM receipt is not bound to its journal producer")
            clock = int(entry["__MONOTONIC_TIMESTAMP"])
            if clock < previous_clock:
                raise ProbeError("FM journal window is not in journal order")
            previous_clock = clock
            records.append(
                {"cursor": record_cursor, "monotonic_us": clock, "receipt": body}
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProbeError("FM journal receipt JSON is malformed") from exc
    if cursor and (
        not records or records[0]["cursor"] != cursor or unit_lines > FM_JOURNAL_LIMIT
    ):
        raise ProbeError("FM journal receipt window lost its anchor or was truncated")
    if service_snapshot().get(FM_UNIT) != before:
        raise ProbeError("FM producer changed during the journal read")
    emit(
        {
            "service": before,
            "anchor_cursor": cursor or None,
            "complete": True,
            "captured_monotonic_us": time.monotonic_ns() // 1000,
            "records": records,
        }
    )


def efa_inventory_command(_arguments: argparse.Namespace) -> None:
    """Only the EFA inventory: COLLECT-017 polls it every few seconds.

    A full ``snapshot`` runs nvidia-smi three times and stats every unit; the
    unbind wait needs the sysfs walk alone.
    """

    emit(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "efa_inventory": efa_inventory(),
        }
    )


def gpu_identity(_arguments: argparse.Namespace) -> None:
    from gpu_fault.collectors.gpu.discovery import (
        discover_gpu_product,
        discover_gpu_software_versions,
    )

    product = discover_gpu_product()
    driver, cuda = discover_gpu_software_versions()
    if not cuda:
        raise ProbeError("node CUDA version is unknown")
    emit({"product": product, "driver_branch": driver, "cuda_version": cuda})


def firmware_premise(_arguments: argparse.Namespace) -> None:
    unit = "gpu-fault-node-agent.service"
    before, pid = active_unit_identity(unit, "Node Agent")
    process = PROC_ROOT / pid
    start = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
    environment = {}
    for entry in (process / "environ").read_bytes().split(b"\0"):
        key, separator, value = entry.partition(b"=")
        if separator:
            environment[key.decode()] = value.decode()
    if (
        service_snapshot().get(unit) != before
        or (process / "stat").read_text().rsplit(")", 1)[1].split()[19] != start
    ):
        raise ProbeError("Node Agent changed during firmware premise")
    emit(
        {
            "pid": int(pid),
            "invocation_id": before["InvocationID"],
            "start_ticks": start,
            "allow_disabled": environment.get(
                "GPU_FAULT_NODE_ALLOW_FIRMWARE_UPDATE", ""
            ).lower()
            in {"", "false", "0", "no", "off"},
            "target_present": bool(
                environment.get("GPU_FAULT_TARGET_FIRMWARE_VERSION")
            ),
            "update_command_present": bool(
                environment.get("GPU_FAULT_FIRMWARE_UPDATE_COMMAND")
            ),
            "verify_command_present": bool(
                environment.get("GPU_FAULT_FIRMWARE_VERIFY_COMMAND")
            ),
        }
    )


def inventory_configuration() -> dict[str, Any]:
    """Read only inventory settings from the running host collector and its file."""

    before, pid = active_unit_identity(HOST_COLLECTOR_UNIT, "host collector")
    keys = {
        "GPU_FAULT_EXPECTED_GPU_COUNT",
        "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT",
        "GPU_FAULT_HOST_INTERVAL_SECONDS",
        "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS",
        "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES",
    }
    process = PROC_ROOT / pid
    start = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
    values = {}
    for entry in (process / "environ").read_bytes().split(b"\0"):
        key, separator, value = entry.partition(b"=")
        if separator and key.decode(errors="replace") in keys:
            values[key.decode()] = value.decode()
    if any(not value.isdecimal() for value in values.values()):
        raise ProbeError("host collector inventory configuration is not numeric")
    config_file = file_snapshot(COLLECTOR_ENV)
    file_values = {key: value for key, value in parse_env().items() if key in keys}
    if (
        service_snapshot().get(HOST_COLLECTOR_UNIT) != before
        or (process / "stat").read_text().rsplit(")", 1)[1].split()[19] != start
        or file_snapshot(COLLECTOR_ENV) != config_file
    ):
        raise ProbeError("host collector identity/configuration changed during read")
    return {
        "pid": int(pid),
        "invocation_id": before["InvocationID"],
        "start_ticks": start,
        "running_env": values,
        "file_env": file_values,
        "file": config_file,
    }


def inventory_config(_arguments: argparse.Namespace) -> None:
    emit(inventory_configuration())


def inventory_sampling_identity() -> dict[str, Any]:
    """Same-boot read identity; sampling never owns production configuration."""
    contents = collector_private_file(COLLECTOR_ENV)
    values = collector_env_values(
        contents,
        extra_keys={
            "GPU_FAULT_RUNTIME_PROFILE_VERSION",
            "GPU_FAULT_NODE_INSTANCE_TYPE",
        },
    )
    configuration = inventory_configuration()
    boot = BOOT_ID_FILE.read_text().strip()
    profile = values.get("GPU_FAULT_RUNTIME_PROFILE_VERSION", "")
    # The planner freezes the finding's node_instance_type into the RESTART_NODE
    # plan and VALIDATE_GPU caps the injected expected count at that type's
    # physical GPU count; a sample without it fails validation for ever.
    instance_type = values.get("GPU_FAULT_NODE_INSTANCE_TYPE", "")
    if (
        not boot
        or SAFE_ID.fullmatch(boot) is None
        or not profile
        or SAFE_ID.fullmatch(profile) is None
        or INSTANCE_TYPE.fullmatch(instance_type) is None
        or configuration["file"].get("sha256") != hashlib.sha256(contents).hexdigest()
        or configuration["running_env"] != configuration["file_env"]
    ):
        raise ProbeError("inventory sampling identity is incomplete")
    return {
        "cluster_id": values["GPU_FAULT_CLUSTER_ID"],
        "node_id": values["NODE_NAME"],
        "runtime_profile_version": profile,
        "node_instance_type": instance_type,
        "boot_id": boot,
        "runtime": collector_runtime_identity(),
        "configuration": configuration,
    }


def inventory_receipt_digest(receipt: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class InventorySampleSink:
    """A bounded private sink with no network or outbox implementation."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.batches: list[dict[str, Any]] = []

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path != self.path or len(self.batches) >= 2:
            raise ProbeError("isolated inventory sink rejected an extra delivery")
        self.batches.append(json.loads(json.dumps(payload, allow_nan=False)))
        return {}


def sample_gpu_inventory(arguments: argparse.Namespace) -> None:
    from gpu_fault.channel_registry import HOST_TELEMETRY_PATH
    from gpu_fault.collectors.host.collector import HostTelemetryCollector
    from gpu_fault.collectors.models import CollectorContext

    before = inventory_sampling_identity()
    env = before["configuration"]["file_env"]
    interval = int(env["GPU_FAULT_HOST_INTERVAL_SECONDS"])
    expected = int(env["GPU_FAULT_EXPECTED_GPU_COUNT"])
    if (
        safe_id(arguments.run_id, "inventory sample run") != arguments.run_id
        or arguments.cluster_id != before["cluster_id"]
        or arguments.node_id != before["node_id"]
        or arguments.expected_env_sha256 != before["configuration"]["file"]["sha256"]
        or arguments.expected_boot_id != before["boot_id"]
        or type(arguments.expected_gpu_count) is not int
        or arguments.expected_gpu_count != expected + 1
        or not 1 <= interval <= 300
        or env["GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES"] != "2"
    ):
        raise ProbeError("isolated inventory sampling scope differs from baseline")
    sink = InventorySampleSink(HOST_TELEMETRY_PATH)
    collector = HostTelemetryCollector(
        sink,
        CollectorContext(
            cluster_id=before["cluster_id"],
            runtime_profile_version=before["runtime_profile_version"],
        ),
        node_id=before["node_id"],
        node_instance_type=before["node_instance_type"],
        expected_gpu_count=arguments.expected_gpu_count,
        expected_efa_device_count=int(env["GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT"]),
        interval_seconds=interval,
        inventory_mismatch_consecutive_samples=2,
        edge_filter_enabled=True,
        startup_spread_seconds=1,
        health_summary_seconds=int(env["GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS"]),
        history_max_points=1,
        now=lambda: datetime.now(timezone.utc),
        runner=collector_runner,
    )
    # Exercise the real inventory counter and edge filter without a second
    # production producer or unrelated host-health signals.
    collector.CONTRIBUTORS = ("_gpu_inventory",)
    for index in range(2):
        if index:
            time.sleep(interval)
        batch = collector.collect_once()
        if batch.collection_errors:
            raise ProbeError("isolated inventory collector could not read inventory")
    if inventory_sampling_identity() != before or len(sink.batches) != 2:
        raise ProbeError("inventory sampling changed identity or missed a delivery")
    receipt = {
        "schema_version": 1,
        "kind": "ISOLATED_GPU_INVENTORY",
        "run_id": arguments.run_id,
        "identity": before,
        "interval_seconds": interval,
        "expected_gpu_count": arguments.expected_gpu_count,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "sampler_pid": os.getpid(),
        "publication_performed": False,
        "batches": sink.batches,
    }
    receipt["sha256"] = inventory_receipt_digest(receipt)
    emit(receipt)


def inventory_publication_batches(
    receipt: dict[str, Any], *, run_id: str, expected_sha256: str
) -> list[dict[str, Any]]:
    from gpu_fault.host_health import HostTelemetryBatch

    if (
        receipt.get("schema_version") != 1
        or type(receipt.get("schema_version")) is not int
        or receipt.get("kind") != "ISOLATED_GPU_INVENTORY"
        or receipt.get("run_id") != run_id
        or receipt.get("publication_performed") is not False
        or receipt.get("sha256") != expected_sha256
        or inventory_receipt_digest(receipt) != expected_sha256
        or receipt.get("identity") != inventory_sampling_identity()
    ):
        raise ProbeError("inventory publication is not bound to the completed sample")
    identity = receipt["identity"]
    try:
        sampled = datetime.fromisoformat(receipt["sampled_at"])
    except (KeyError, TypeError, ValueError):
        raise ProbeError("isolated inventory completion clock is invalid") from None
    if sampled.tzinfo is None or not (
        0 <= (datetime.now(timezone.utc) - sampled).total_seconds() <= 300
    ):
        raise ProbeError("isolated inventory sample is not fresh")
    raw_batches = receipt.get("batches")
    if not isinstance(raw_batches, list) or len(raw_batches) != 2:
        raise ProbeError("inventory publication requires exactly two batches")
    try:
        batches = [HostTelemetryBatch.model_validate(value) for value in raw_batches]
    except ValueError:
        raise ProbeError("inventory publication batch is malformed") from None
    expected = int(
        identity["configuration"]["file_env"]["GPU_FAULT_EXPECTED_GPU_COUNT"]
    )
    names = {
        f"gpu_inventory_{suffix}"
        for suffix in (
            "expected_count",
            "active_count",
            "missing_count",
            "excess_count",
            "mismatch",
        )
    }
    for index, batch in enumerate(batches):
        if (
            batch.cluster_id != identity["cluster_id"]
            or batch.node_id != identity["node_id"]
            or batch.runtime_profile_version != identity["runtime_profile_version"]
            or batch.producer != "node"
            or batch.collection_errors
            or batch.affected_workload_ids
            or batch.received_at is not None
            or batch.batch_id
            != f"host-{batch.node_id}-{int(batch.observed_at.timestamp() * 1_000_000)}"
            or len(batch.samples) != len(names)
            or {sample.name for sample in batch.samples} != names
        ):
            raise ProbeError("inventory publication batch scope is invalid")
        mismatch = next(
            sample
            for sample in batch.samples
            if sample.name == "gpu_inventory_mismatch"
        )
        if (
            mismatch.value != index
            or mismatch.device is not None
            or mismatch.labels.get("expected_count") != str(expected + 1)
            or mismatch.labels.get("observed_count") != str(expected)
            or mismatch.labels.get("required_consecutive_samples") != "2"
            or mismatch.labels.get("consecutive_mismatch_samples") != str(index + 1)
            or mismatch.labels.get("node_instance_type")
            != identity["node_instance_type"]
            or (index == 0 and "baseline" not in batch.edge_filter_reasons)
        ):
            raise ProbeError(
                "inventory publication is not the sampled debounce episode"
            )
    elapsed = (batches[1].observed_at - batches[0].observed_at).total_seconds()
    if (
        batches[0].batch_id == batches[1].batch_id
        or not receipt["interval_seconds"] <= elapsed <= receipt["interval_seconds"] * 2
        or sampled < batches[1].observed_at
    ):
        raise ProbeError("inventory publication sample order is invalid")
    return cast(list[dict[str, Any]], raw_batches)


def publish_gpu_inventory(arguments: argparse.Namespace) -> None:
    from urllib.parse import urlsplit

    from gpu_fault.channel_registry import HOST_TELEMETRY_PATH
    from gpu_fault.collectors.sinks import DeliveryStatus, HttpEventSink, deliver_event

    if (
        arguments.confirm != "PUBLISH_GPU_INVENTORY"
        or len(arguments.receipt_json.encode()) > 128 * 1024
    ):
        raise ProbeError("inventory publication is not authorized")
    receipt = json.loads(arguments.receipt_json)
    if not isinstance(receipt, dict):
        raise ProbeError("inventory publication receipt is malformed")
    batches = inventory_publication_batches(
        receipt, run_id=arguments.run_id, expected_sha256=arguments.expected_sha256
    )
    values = collector_env_values(
        collector_private_file(COLLECTOR_ENV),
        extra_keys={
            "GPU_FAULT_CONTROL_PLANE_URL",
            "GPU_FAULT_CONTROL_PLANE_TOKEN",
            COLLECTOR_CA_ENV_KEY,
        },
    )
    url = urlsplit(values.get("GPU_FAULT_CONTROL_PLANE_URL", ""))
    ca_file = values.get(COLLECTOR_CA_ENV_KEY, "")
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or not values.get("GPU_FAULT_CONTROL_PLANE_TOKEN")
        or not ca_file
        or not Path(ca_file).is_file()
    ):
        raise ProbeError("inventory publication requires the configured HTTPS endpoint")
    os.environ["SSL_CERT_FILE"] = ca_file
    sink = HttpEventSink(
        values["GPU_FAULT_CONTROL_PLANE_URL"],
        bearer_token=values["GPU_FAULT_CONTROL_PLANE_TOKEN"],
        timeout_seconds=15,
        max_attempts=1,
        outbox_path=None,
    )
    for batch in batches:
        delivery = deliver_event(sink, HOST_TELEMETRY_PATH, batch)
        if delivery.status is not DeliveryStatus.DELIVERED:
            raise ProbeError("inventory batch acceptance is unresolved")
    emit(
        {
            "run_id": arguments.run_id,
            "sample_sha256": arguments.expected_sha256,
            "publication_performed": True,
            "batch_ids": [batch["batch_id"] for batch in batches],
        }
    )


def reset_audit(_arguments: argparse.Namespace) -> None:
    if not NODE_LEDGER.is_file():
        raise ProbeError("Node Agent reset ledger is missing")
    connection = sqlite3.connect(f"file:{NODE_LEDGER}?mode=ro", uri=True)
    try:
        rows = connection.execute("""
            SELECT command_id, attempt, state, operation, started_at, completed_at,
                   incident_id, workflow_request_id, fencing_token, gpu_uuids,
                   parameters_digest, signature_digest, payload
            FROM results
            WHERE operation IN ('RESET_GPU', 'RESET_ALL_GPUS_NVSWITCHES')
            ORDER BY started_at, command_id, attempt
        """).fetchall()
    finally:
        connection.close()
    fields = (
        "command_id",
        "attempt",
        "state",
        "operation",
        "started_at",
        "completed_at",
        "incident_id",
        "workflow_request_id",
        "fencing_token",
        "gpu_uuids",
        "parameters_digest",
    )
    ledger = []
    for row in rows:
        entry = dict(zip(fields, row[:-2], strict=True))
        entry["signature_digest_present"] = bool(row[-2])
        entry["gpu_uuids"] = (
            json.loads(entry["gpu_uuids"]) if entry["gpu_uuids"] else None
        )
        result = json.loads(row[-1])
        details = result.get("details") or {}
        entry["result"] = {
            key: result.get(key)
            for key in ("command_id", "operation", "status", "attempt")
        }
        entry["result"]["details"] = {
            key: details.get(key)
            for key in (
                "reset_gpu_uuids",
                "verified_no_gpu_clients",
                "reset_attempts",
                "reset_successes",
                "reset_busy_refusals",
                "outcome_unknown",
                "manual_confirmation_required",
                "reset_outcome_unknown",
                "reset_failed",
                "reset_not_attempted",
                "reset_scope",
                "inventory_verified_before",
                "inventory_verified_after",
            )
        }
        ledger.append(entry)
    emit(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "boot_id": BOOT_ID_FILE.read_text().strip(),
            "gpu_inventory": gpu_inventory(),
            "ledger": ledger,
        }
    )


def snapshot(_arguments: argparse.Namespace) -> None:
    emit(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "collector_env": parse_env(),
            # The file's identity, not only the keys parsed out of it: a
            # restore is "the original bytes" only when the digest matches.
            "collector_env_file": file_snapshot(COLLECTOR_ENV),
            "services": service_snapshot(),
            "gpu_inventory": gpu_inventory(),
            "gpu_power": gpu_power_state(),
            "persistence_mode": persistence_mode(),
            "efa_inventory": efa_inventory(),
            "kernel_collector": kernel_collector_fd(),
            "fabric_manager_log": file_snapshot(FM_LOG),
            "fabric_manager_states": [
                value
                for path in FM_STATE_CANDIDATES
                if (value := file_snapshot(path)) is not None
            ],
            "fabric_manager_cursor": fabric_manager_cursor(),
        }
    )


SAFE_POD_UID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def workload_processes(
    pod_uid: str, *, proc: Path = Path("/proc"), excluded_pids: Iterable[int] = ()
) -> list[dict[str, Any]]:
    """Every process whose cgroup names ``pod_uid``, both kubelet spellings.

    kubelet writes the Pod UID into the cgroup path with the dashes either
    kept (``pod<uid>``) or replaced by underscores
    (``kubepods-pod<uid>_.slice``), depending on the cgroup driver; a probe
    that matched only one spelling silently killed nothing on the other. The
    ``pause`` sandbox container shares the Pod's cgroup but is not the training
    process and is left alone -- SIGKILLing it would tear the Pod down as a
    sandbox failure rather than let the trainer exit non-zero.
    """

    if SAFE_POD_UID.fullmatch(pod_uid) is None:
        raise ProbeError("unsafe pod UID")
    spellings = (f"pod{pod_uid}", f"pod{pod_uid.replace('-', '_')}")
    excluded = {int(item) for item in excluded_pids} | {os.getpid(), 1}
    matched: list[dict[str, Any]] = []
    for entry in sorted(
        (item for item in proc.iterdir() if item.name.isdigit()),
        key=lambda item: int(item.name),
    ):
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            cgroup = entry.joinpath("cgroup").read_text(encoding="utf-8")
        except OSError:
            continue
        if not any(spelling in cgroup for spelling in spellings):
            continue
        try:
            comm = entry.joinpath("comm").read_text(encoding="utf-8").strip()
        except OSError:
            comm = ""
        if comm == "pause":
            continue
        matched.append({"pid": pid, "comm": comm})
    return matched


def kill_workload(
    arguments: argparse.Namespace,
    *,
    proc: Path = Path("/proc"),
    kill: Callable[[int, int], None] = os.kill,
) -> None:
    """SIGKILL the training processes of one Pod so the container exits non-zero.

    The Pod object and PyTorchJob are never touched: deleting the Pod reads as
    a user stop and suppresses the passive restart, so the drill has to make
    the container die on its own. Refusing when nothing matched keeps a
    mistyped UID from reading as a silent success.
    """

    pod_uid = arguments.pod_uid
    processes = workload_processes(pod_uid, proc=proc)
    if not processes:
        raise ProbeError(f"no process found in the cgroup of pod {pod_uid}")
    killed: list[dict[str, Any]] = []
    already_exited: list[dict[str, Any]] = []
    for process in processes:
        try:
            kill(process["pid"], signal.SIGKILL)
        except ProcessLookupError:
            already_exited.append(process)
        else:
            killed.append(process)
    emit(
        {
            "pod_uid": pod_uid,
            "signal": "SIGKILL",
            "killed": killed,
            "already_exited": already_exited,
        }
    )


def write_xid(arguments: argparse.Namespace) -> None:
    xid = int(arguments.xid)
    if xid not in ALLOWED_XIDS:
        raise ProbeError("XID is not allowlisted")
    marker = safe_id(arguments.marker, "marker")
    bdf = normalize_bdf(arguments.pci_bdf)
    message = (
        f"<3>NVRM: Xid (PCI:{bdf.rsplit('.', 1)[0]}): {xid}, "
        f"pid={arguments.pid}, name=python, {arguments.message} marker={marker}\n"
    ).encode()
    descriptor = os.open("/dev/kmsg", os.O_WRONLY | os.O_CLOEXEC)
    try:
        written = os.write(descriptor, message)
    finally:
        os.close(descriptor)
    emit({"xid": xid, "marker": marker, "pci_bdf": bdf, "bytes_written": written})


def append_sxid(arguments: argparse.Namespace) -> None:
    sxid = int(arguments.sxid)
    if sxid not in ALLOWED_SXIDS:
        raise ProbeError("SXID is not allowlisted")
    marker = safe_id(arguments.marker, "marker")
    bdf = normalize_bdf(arguments.pci_bdf)
    prefix = (
        f"nvidia-nvswitch{arguments.switch}: "
        if arguments.include_switch
        else "NVSwitch "
    )
    line = (
        f"[{datetime.now(timezone.utc).isoformat()}] [ERROR] "
        f"[tid {arguments.tid}] {prefix}SXid (PCI:{bdf}): {sxid}, "
        f"{arguments.classification}, Link {arguments.port} "
        f"{arguments.message} marker={marker}\n"
    )
    FM_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FM_LOG.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    emit(
        {
            "sxid": sxid,
            "marker": marker,
            "path": str(FM_LOG),
            "size": FM_LOG.stat().st_size,
        }
    )


def restart_service(arguments: argparse.Namespace) -> None:
    service = arguments.service
    if service not in ALLOWED_SERVICES:
        raise ProbeError("collector service is not allowlisted")
    before = service_snapshot().get(service)
    run(["systemctl", "restart", service], timeout=120)
    after = service_snapshot().get(service)
    if not after or after.get("ActiveState") != "active":
        raise ProbeError(f"{service} is not active after restart")
    emit({"service": service, "before": before, "after": after})


def set_persistence_mode(arguments: argparse.Namespace) -> None:
    enabled = arguments.enabled == "true"
    before = run(["nvidia-smi", "-q", "-d", "PERSISTENCE_MODE"]).stdout
    run(["nvidia-smi", "-pm", "1" if enabled else "0"])
    emit(
        {
            "enabled": enabled,
            "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        }
    )


def collector_restore_paths(run_id: str) -> tuple[Path, str]:
    safe = safe_id(run_id, "run ID")
    digest = hashlib.sha256(safe.encode()).hexdigest()[:16]
    return (
        ACCEPTANCE_STATE / f"collector-env-{digest}.backup",
        f"gpu-fault-collector-env-restore-{digest}",
    )


def collector_override_record(backup: Path) -> Path:
    return backup.with_suffix(".override.json")


COLLECTOR_RUNTIME = Path("/opt/gpu-fault/current")
COLLECTOR_FINDMNT = Path("/usr/bin/findmnt")
COLLECTOR_ENV_STATES = {
    "PREPARED",
    "ARMED",
    "MUTATING",
    "ACTIVE",
    "RESTORING",
    "RESTORED",
    "CLEANING",
    "CLEANED",
}
COLLECTOR_ENV_COMMANDS = {
    "override-expected-gpu-count",
    "restore-collector-env",
    "collector-env-start-check",
}
COLLECTOR_GUARD_FAILURE_LIMIT = 16
COLLECTOR_GUARD_FAILURE_FIELDS = {
    "error_code",
    "error_site",
    "command",
    "automatic",
    "phase",
    "recorded_at",
    "run_id",
    "intent_sha256",
}
COLLECTOR_ENV_INTENT_FIELDS = (
    "schema_version",
    "run_id",
    "owner_nonce_sha256",
    "cluster_id",
    "node_id",
    "node_instance_id",
    "boot_id",
    "baseline",
    "override",
    "baseline_sha256",
    "applied_sha256",
    "created_monotonic",
    "created_epoch",
    "restore_seconds",
    "restore_deadline_monotonic",
    "restore_deadline_epoch",
    "python",
    "python_sha256",
    "runtime_path",
    "runtime_device",
    "runtime_inode",
    "probe_sha256",
    "host_unit_path",
    "host_unit_sha256",
)
COLLECTOR_ENV_V3_INTENT_FIELDS = (*COLLECTOR_ENV_INTENT_FIELDS, "runtime_filesystem")


class CollectorEnvGuardError(ProbeError):
    """A collector recovery rejection is not transport loss during a reboot."""


class CollectorEnvBusyError(CollectorEnvGuardError):
    """A live owner keeps the independent recovery service retryable."""


def collector_require_root() -> None:
    if os.geteuid() != 0:
        raise CollectorEnvGuardError("collector env recovery requires root")


def collector_private_file(
    path: Path, *, private: bool = True, max_size: int = 1024 * 1024
) -> bytes:
    collector_require_root()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_mode & (0o077 if private else 0o022)
            or info.st_size > max_size
        ):
            raise CollectorEnvGuardError("collector recovery file is not private")
        contents = stream.read(max_size + 1)
        if len(contents) > max_size:
            raise CollectorEnvGuardError(
                "collector recovery file exceeds its size limit"
            )
        return contents


def collector_json(path: Path) -> dict[str, Any] | None:
    try:
        contents = collector_private_file(path)
    except FileNotFoundError:
        return None
    try:
        value = json.loads(contents)
    except (ValueError, UnicodeError) as exc:
        raise CollectorEnvGuardError("collector recovery JSON is invalid") from exc
    if not isinstance(value, dict):
        raise CollectorEnvGuardError("collector recovery record is not an object")
    return value


@contextmanager
def collector_env_lock() -> Iterator[None]:
    collector_require_root()
    ACCEPTANCE_STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = ACCEPTANCE_STATE.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        raise CollectorEnvGuardError("collector recovery directory is not private")
    power_sync_directory(ACCEPTANCE_STATE.parent)
    descriptor = os.open(
        ACCEPTANCE_STATE / "collector-env.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
        0o600,
    )
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_mode & 0o077
        ):
            raise CollectorEnvGuardError("collector recovery lock is not private")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CollectorEnvBusyError(
                "collector env operation is still running"
            ) from exc
        yield
    finally:
        os.close(descriptor)


def collector_nonce(arguments: argparse.Namespace) -> tuple[str, str]:
    run_id = safe_id(arguments.run_id, "run ID")
    nonce = getattr(arguments, "owner_nonce", "")
    nonce_file = getattr(arguments, "owner_nonce_file", "")
    if nonce_file:
        expected = collector_restore_paths(run_id)[0].with_suffix(".nonce")
        if nonce or str(expected) != nonce_file:
            raise CollectorEnvGuardError("collector recovery capability path differs")
        nonce = collector_private_file(expected).decode("ascii")
    if (
        not isinstance(nonce, str)
        or re.fullmatch(r"(?:[0-9a-f]{32}|[0-9a-f]{64})", nonce) is None
    ):
        raise CollectorEnvGuardError(
            "collector owner nonce must be 32 or 64 hex characters"
        )
    return run_id, nonce


def collector_env_values(
    contents: bytes, *, extra_keys: set[str] | None = None
) -> dict[str, str]:
    identity_keys = {
        "GPU_FAULT_EXPECTED_GPU_COUNT",
        "GPU_FAULT_CLUSTER_ID",
        "NODE_NAME",
        "GPU_FAULT_NODE_INSTANCE_ID",
    }
    keys = identity_keys | (extra_keys or set())
    values: dict[str, str] = {}
    try:
        for line in contents.decode("utf-8").splitlines():
            key, separator, value = line.strip().partition("=")
            if not separator or key not in keys:
                continue
            if key in values:
                raise CollectorEnvGuardError(
                    "collector identity/config key is duplicated"
                )
            parsed = shlex.split(value, comments=True)
            if len(parsed) != 1 or not parsed[0]:
                raise CollectorEnvGuardError(
                    "collector identity/config value is invalid"
                )
            values[key] = parsed[0]
    except (UnicodeError, ValueError) as exc:
        raise CollectorEnvGuardError(
            "collector identity/config parsing failed"
        ) from exc
    if (
        not {"GPU_FAULT_EXPECTED_GPU_COUNT", "GPU_FAULT_CLUSTER_ID", "NODE_NAME"}
        <= values.keys()
    ):
        raise CollectorEnvGuardError("collector node/cluster/count identity is missing")
    for key in identity_keys - {"GPU_FAULT_EXPECTED_GPU_COUNT"}:
        if key in values:
            safe_id(values[key], "collector identity")
    if not values["GPU_FAULT_EXPECTED_GPU_COUNT"].isdecimal():
        raise CollectorEnvGuardError("collector expected count is not an integer")
    return values


def collector_filesystem_path(value: Any) -> Path:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise CollectorEnvGuardError("collector filesystem path is invalid")
    path = Path(value)
    if (
        not path.is_absolute()
        or value.startswith("//")
        or str(path) != value
        or ".." in path.parts
    ):
        raise CollectorEnvGuardError("collector filesystem path is not canonical")
    return path


def collector_filesystem_fields(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "uuid",
        "fstype",
        "target",
        "fsroot",
    }:
        raise CollectorEnvGuardError("collector filesystem identity is incomplete")
    uuid = value["uuid"]
    fstype = value["fstype"]
    if (
        not isinstance(uuid, str)
        or re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid)
        is None
        or not int(uuid.replace("-", ""), 16)
        or not isinstance(fstype, str)
        or re.fullmatch(r"[a-z][a-z0-9._+-]{0,63}", fstype) is None
    ):
        raise CollectorEnvGuardError("collector filesystem UUID or type is invalid")
    return {
        "uuid": uuid.lower(),
        "fstype": fstype,
        "target": str(collector_filesystem_path(value["target"])),
        "fsroot": str(collector_filesystem_path(value["fsroot"])),
    }


def collector_findmnt_tool() -> os.stat_result:
    path = COLLECTOR_FINDMNT
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise CollectorEnvGuardError("collector findmnt path is not trusted")
    for parent in path.parents:
        info = parent.lstat()
        # A root-owned sticky ancestor cannot let another UID replace our child.
        sticky_root = info.st_uid == 0 and info.st_mode & stat.S_ISVTX
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid not in {0, os.geteuid()}
            or info.st_mode & 0o022
            and not sticky_root
        ):
            raise CollectorEnvGuardError("collector findmnt directory is not trusted")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
        or info.st_mode & (0o022 | stat.S_ISUID | stat.S_ISGID)
        or not info.st_mode & 0o111
    ):
        raise CollectorEnvGuardError("collector findmnt executable is not trusted")
    return info


def collector_unique_json_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CollectorEnvGuardError("collector filesystem JSON repeats a field")
        result[key] = value
    return result


def collector_executable_stamp(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def collector_findmnt(runtime: Path) -> dict[str, Any]:
    try:
        expected = collector_executable_stamp(collector_findmnt_tool())
        descriptor = os.open(
            COLLECTOR_FINDMNT, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        try:
            opened = collector_executable_stamp(os.fstat(descriptor))
            executable = Path(f"/proc/self/fd/{descriptor}")
            if (
                opened != expected
                or collector_executable_stamp(executable.stat()) != opened
            ):
                raise CollectorEnvGuardError("collector findmnt executable changed")
            completed = subprocess.run(
                [
                    str(executable),
                    "--kernel",
                    "--json",
                    "--target",
                    str(runtime),
                    "--output",
                    "UUID,FSTYPE,TARGET,FSROOT,MAJ:MIN",
                ],
                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
                pass_fds=(descriptor,),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
            if (
                collector_executable_stamp(os.fstat(descriptor)) != opened
                or collector_executable_stamp(collector_findmnt_tool()) != expected
            ):
                raise CollectorEnvGuardError("collector findmnt executable changed")
        finally:
            os.close(descriptor)
        if completed.returncode or len(completed.stdout) > 65536:
            raise CollectorEnvGuardError("collector filesystem query failed")
        value = json.loads(
            completed.stdout, object_pairs_hook=collector_unique_json_fields
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise CollectorEnvGuardError(
            "collector filesystem query is unavailable"
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"filesystems"}
        or not isinstance(value["filesystems"], list)
        or len(value["filesystems"]) != 1
        or not isinstance(value["filesystems"][0], dict)
        or set(value["filesystems"][0])
        != {"uuid", "fstype", "target", "fsroot", "maj:min"}
    ):
        raise CollectorEnvGuardError("collector filesystem query is ambiguous")
    return dict(value["filesystems"][0])


def collector_runtime_mount(runtime: Path, device: int) -> dict[str, str]:
    try:
        contents = collector_private_file(
            PROC_ROOT / "self/mountinfo", private=False, max_size=4 * 1024 * 1024
        ).decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise CollectorEnvGuardError("collector mount identity is unavailable") from exc
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    escapes = {r"\040": " ", r"\011": "\t", r"\012": "\n", r"\134": "\\"}
    for line in contents.splitlines():
        fields = line.split()
        if " - " not in line:
            raise CollectorEnvGuardError("collector mount identity is malformed")
        separator = fields.index("-")
        if (
            separator < 6
            or len(fields) != separator + 4
            or not fields[0].isdecimal()
            or fields[0] in seen
            or not fields[1].isdecimal()
            or re.fullmatch(r"[0-9]+:[0-9]+", fields[2]) is None
            or any(re.search(r"\\(?!040|011|012|134)", item) for item in fields[3:5])
        ):
            raise CollectorEnvGuardError("collector mount identity is malformed")
        seen.add(fields[0])
        root, target = (
            collector_filesystem_path(
                re.sub(r"\\[0-7]{3}", lambda match: escapes.get(match[0], "\0"), item)
            )
            for item in fields[3:5]
        )
        if runtime.is_relative_to(target):
            candidates.append(
                {
                    "target": str(target),
                    "fsroot": str(root),
                    "fstype": fields[separator + 1],
                    "maj:min": fields[2],
                }
            )
    if not candidates:
        raise CollectorEnvGuardError("collector runtime mount is missing")
    depth = max(len(Path(item["target"]).parts) for item in candidates)
    matches = [item for item in candidates if len(Path(item["target"]).parts) == depth]
    if (
        len(matches) != 1
        or matches[0]["maj:min"] != f"{os.major(device)}:{os.minor(device)}"
    ):
        raise CollectorEnvGuardError("collector runtime mount is ambiguous or changed")
    return matches[0]


def collector_filesystem_identity(runtime: Path, device: int) -> dict[str, str]:
    mount = collector_runtime_mount(runtime, device)
    observed = collector_findmnt(runtime)
    stable = collector_filesystem_fields(
        {key: observed[key] for key in ("uuid", "fstype", "target", "fsroot")}
    )
    target = Path(stable["target"])
    try:
        if (
            {key: observed[key] for key in mount} != mount
            or target.resolve(strict=True) != target
            or target.stat().st_dev != device
            or not target.is_dir()
            or collector_runtime_mount(runtime, device) != mount
        ):
            raise CollectorEnvGuardError(
                "collector filesystem does not bind runtime mount"
            )
    except (OSError, UnicodeError) as exc:
        raise CollectorEnvGuardError(
            "collector filesystem mount identity is unavailable"
        ) from exc
    return stable


def collector_runtime_identity(*, schema_version: int = 2) -> dict[str, Any]:
    if type(schema_version) is not int or schema_version not in {2, 3}:
        raise CollectorEnvGuardError("collector recovery schema is unsupported")
    runtime = COLLECTOR_RUNTIME.resolve(strict=True)
    info = runtime.stat()
    python = (runtime / "venv/bin/python").resolve(strict=True)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o022
        or not os.access(python, os.X_OK)
        or python != Path(sys.executable).resolve(strict=True)
    ):
        raise CollectorEnvGuardError("collector runtime identity is unavailable")
    identity: dict[str, Any] = {
        "runtime_path": str(runtime),
        "runtime_device": info.st_dev,
        "runtime_inode": info.st_ino,
        "python": str(python),
        "python_sha256": hashlib.sha256(
            collector_private_file(python, private=False, max_size=32 * 1024 * 1024)
        ).hexdigest(),
    }
    if schema_version == 3:
        identity["runtime_filesystem"] = collector_filesystem_identity(
            runtime, info.st_dev
        )
        after = runtime.stat()
        if (
            COLLECTOR_RUNTIME.resolve(strict=True) != runtime
            or (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino)
            or (runtime / "venv/bin/python").resolve(strict=True) != python
            or hashlib.sha256(
                collector_private_file(python, private=False, max_size=32 * 1024 * 1024)
            ).hexdigest()
            != identity["python_sha256"]
        ):
            raise CollectorEnvGuardError(
                "collector runtime changed during identity query"
            )
    return identity


def collector_intent_digest(record: dict[str, Any]) -> str:
    version = record["schema_version"]
    if type(version) is not int or version not in {2, 3}:
        raise CollectorEnvGuardError("collector recovery schema is unsupported")
    fields: tuple[str, ...] = COLLECTOR_ENV_INTENT_FIELDS
    if version == 3:
        if (
            collector_filesystem_fields(record["runtime_filesystem"])
            != record["runtime_filesystem"]
        ):
            raise CollectorEnvGuardError(
                "collector filesystem identity is not canonical"
            )
        fields = COLLECTOR_ENV_V3_INTENT_FIELDS
    return hashlib.sha256(
        json.dumps(
            {key: record[key] for key in fields},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def collector_load_record(run_id: str, nonce: str) -> dict[str, Any] | None:
    if ACCEPTANCE_STATE.exists() or ACCEPTANCE_STATE.is_symlink():
        directory = ACCEPTANCE_STATE.lstat()
        if (
            not stat.S_ISDIR(directory.st_mode)
            or directory.st_uid != os.geteuid()
            or directory.st_mode & 0o077
        ):
            raise CollectorEnvGuardError("collector recovery directory is not private")
    nonce_digest = hashlib.sha256(nonce.encode()).hexdigest()
    owner = collector_json(ACCEPTANCE_STATE / "collector-env-owner.json")
    if owner is not None and (
        owner.get("run_id") != run_id or owner.get("owner_nonce_sha256") != nonce_digest
    ):
        raise CollectorEnvGuardError("another collector env operation owns this node")
    backup, _ = collector_restore_paths(run_id)
    record = collector_json(collector_override_record(backup))
    if record is None:
        if owner is not None or any(
            path.exists() or path.is_symlink()
            for path in (
                backup,
                backup.with_suffix(".probe.py"),
                backup.with_suffix(".nonce"),
                backup.with_suffix(".armed.json"),
                SYSTEMD_UNIT_DIR / (collector_restore_paths(run_id)[1] + ".service"),
                SYSTEMD_UNIT_DIR / (collector_restore_paths(run_id)[1] + ".timer"),
            )
        ):
            raise CollectorEnvGuardError("collector recovery intent is missing")
        return None
    try:
        valid = (
            type(record["schema_version"]) is int
            and record["schema_version"] in {2, 3}
            and record["run_id"] == run_id
            and record["owner_nonce_sha256"] == nonce_digest
            and record["state"] in COLLECTOR_ENV_STATES
            and type(record["mutation_started"]) is bool
            and record["intent_sha256"] == collector_intent_digest(record)
            and all(
                isinstance(record[key], str) and HEX64.fullmatch(record[key])
                for key in ("baseline_sha256", "applied_sha256", "intent_sha256")
            )
            and type(record["restore_seconds"]) is int
            and 60 <= record["restore_seconds"] <= 900
            and all(
                type(record[key]) in (int, float) and math.isfinite(record[key])
                for key in (
                    "created_monotonic",
                    "created_epoch",
                    "restore_deadline_monotonic",
                    "restore_deadline_epoch",
                )
            )
            and abs(
                record["restore_deadline_monotonic"]
                - record["created_monotonic"]
                - record["restore_seconds"]
            )
            <= 0.002
            and abs(
                record["restore_deadline_epoch"]
                - record["created_epoch"]
                - record["restore_seconds"]
            )
            <= 0.002
            and bool(record["boot_id"])
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise CollectorEnvGuardError(
            "collector recovery intent or owner does not match"
        )
    if owner is not None and owner.get("intent_sha256") != record["intent_sha256"]:
        raise CollectorEnvGuardError("collector node ownership intent differs")
    if owner is None and record["mutation_started"] and record["state"] != "CLEANED":
        raise CollectorEnvGuardError("collector mutation lost its node owner")
    return record


def collector_guard_failure_path(run_id: str, command: str, error_code: str) -> Path:
    if (
        command not in COLLECTOR_ENV_COMMANDS
        or re.fullmatch(r"[0-9a-f]{16}", error_code) is None
    ):
        raise CollectorEnvGuardError("collector failure receipt identity is invalid")
    backup, _ = collector_restore_paths(run_id)
    return backup.with_suffix(f".guard-{command}-{error_code}.json")


def collector_guard_failure_valid(
    receipt: dict[str, Any], record: dict[str, Any], *, command: str, error_code: str
) -> bool:
    try:
        timestamp = receipt["recorded_at"]
        site = receipt["error_site"]
        return (
            receipt.keys() == COLLECTOR_GUARD_FAILURE_FIELDS
            and receipt["run_id"] == record["run_id"]
            and receipt["intent_sha256"] == record["intent_sha256"]
            and receipt["command"] == command
            and receipt["error_code"] == error_code
            and type(receipt["automatic"]) is bool
            and receipt["phase"] in COLLECTOR_ENV_STATES
            and (
                site is None
                or isinstance(site, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*:[1-9][0-9]{0,8}", site)
                is not None
            )
            and isinstance(timestamp, str)
            and len(timestamp) <= 40
            and datetime.fromisoformat(timestamp).utcoffset()
            == timezone.utc.utcoffset(None)
        )
    except (KeyError, TypeError, ValueError):
        return False


def collector_guard_failure(
    arguments: argparse.Namespace, *, error_code: str, error_site: str | None
) -> dict[str, Any]:
    """Keep bounded first-failure evidence without changing the primary rejection."""
    unavailable = {"status": "OWNERSHIP_UNAVAILABLE"}
    try:
        collector_require_root()
        run_id, nonce = collector_nonce(arguments)
        # Do not create a directory or lock for an unowned/rejected operation.
        if collector_load_record(run_id, nonce) is None:
            return unavailable
    except Exception:
        return unavailable
    try:
        with collector_env_lock():
            try:
                record = collector_load_record(run_id, nonce)
            except Exception:
                return unavailable
            if record is None:
                return unavailable
            path = collector_guard_failure_path(run_id, arguments.command, error_code)
            receipt = {
                "error_code": error_code,
                "error_site": error_site,
                "command": arguments.command,
                "automatic": arguments.command == "collector-env-start-check"
                or getattr(arguments, "automatic", False),
                "phase": record["state"],
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "intent_sha256": record["intent_sha256"],
            }
            if not collector_guard_failure_valid(
                receipt, record, command=arguments.command, error_code=error_code
            ):
                return {"status": "LOGGING_UNAVAILABLE"}
            previous = collector_json(path)
            if previous is not None:
                if not collector_guard_failure_valid(
                    previous, record, command=arguments.command, error_code=error_code
                ):
                    return {"status": "LOGGING_UNAVAILABLE"}
                return {"status": "EXISTS", "receipt": previous}
            backup, _ = collector_restore_paths(run_id)
            for count, _ in enumerate(
                ACCEPTANCE_STATE.glob(f"{backup.stem}.guard-*.json"), start=1
            ):
                if count >= COLLECTOR_GUARD_FAILURE_LIMIT:
                    return {"status": "LIMIT_REACHED"}
            # O_EXCL never replaces even malformed or concurrently created evidence.
            power_create_file(
                path,
                json.dumps(receipt, sort_keys=True, allow_nan=False).encode(),
                0o600,
            )
            if collector_json(path) != receipt:
                return {"status": "LOGGING_UNAVAILABLE"}
            return {"status": "RECORDED", "receipt": receipt}
    except Exception:
        return {"status": "LOGGING_UNAVAILABLE"}


def collector_verify_identity(record: dict[str, Any]) -> tuple[bytes, str]:
    contents = collector_private_file(COLLECTOR_ENV)
    digest = hashlib.sha256(contents).hexdigest()
    values = collector_env_values(contents)
    runtime = collector_runtime_identity(schema_version=record["schema_version"])
    if record["schema_version"] == 3:
        boot_id = BOOT_ID_FILE.read_text().strip()
        if not boot_id or SAFE_ID.fullmatch(boot_id) is None:
            raise CollectorEnvGuardError("collector current boot identity is invalid")
        # Linux can renumber an unchanged NVMe filesystem during a reboot.
        # The original device remains immutable in the intent, and still binds
        # every comparison made in that original boot.
        if boot_id != record["boot_id"]:
            runtime.pop("runtime_device")
    if (
        values["NODE_NAME"] != record["node_id"]
        or values["GPU_FAULT_CLUSTER_ID"] != record["cluster_id"]
        or values.get("GPU_FAULT_NODE_INSTANCE_ID") != record["node_instance_id"]
        or any(record[key] != value for key, value in runtime.items())
    ):
        raise CollectorEnvGuardError(
            "collector node, cluster or runtime identity changed"
        )
    allowed = {record["baseline_sha256"]}
    if record["mutation_started"] and record["state"] not in {
        "RESTORED",
        "CLEANING",
        "CLEANED",
    }:
        allowed.add(record["applied_sha256"])
    if digest not in allowed:
        raise CollectorEnvGuardError("collector env drifted outside the owned digests")
    if record["mutation_started"] and record["state"] != "CLEANED":
        backup, _ = collector_restore_paths(record["run_id"])
        for copied, expected in (
            (backup.with_suffix(".probe.py"), record["probe_sha256"]),
            (backup.with_suffix(".nonce"), record["owner_nonce_sha256"]),
        ):
            if hashlib.sha256(collector_private_file(copied)).hexdigest() != expected:
                raise CollectorEnvGuardError(
                    "collector recovery material identity changed"
                )
    if record["state"] != "CLEANED":
        if (
            hashlib.sha256(power_read_file(Path(record["host_unit_path"]))).hexdigest()
            != record["host_unit_sha256"]
        ):
            raise CollectorEnvGuardError("host collector unit identity changed")
        partial = not record["mutation_started"] or record["state"] == "CLEANING"
        for path, expected in collector_unit_contents(record).items():
            if path.exists() or path.is_symlink():
                if power_read_file(path) != expected:
                    raise CollectorEnvGuardError(
                        "collector recovery resource identity changed"
                    )
            elif not partial:
                raise CollectorEnvGuardError("collector recovery resource is missing")
            if path.suffix != ".conf":
                collector_owned_unit(record, path, absent=partial)
    return contents, digest


def collector_command(
    record: dict[str, Any], *, start_check: bool = False
) -> list[str]:
    backup, _ = collector_restore_paths(record["run_id"])
    command = [
        record["python"],
        "-I",
        "-S",
        "-B",
        str(backup.with_suffix(".probe.py")),
        "collector-env-start-check" if start_check else "restore-collector-env",
        "--run-id",
        record["run_id"],
        "--owner-nonce-file",
        str(backup.with_suffix(".nonce")),
    ]
    if not start_check:
        command.append("--automatic")
    if any(re.fullmatch(r"[A-Za-z0-9_./:+-]+", arg) is None for arg in command):
        raise CollectorEnvGuardError(
            "collector recovery command contains unsafe characters"
        )
    return command


def collector_unit_contents(record: dict[str, Any]) -> dict[Path, bytes]:
    _, unit = collector_restore_paths(record["run_id"])
    return {
        SYSTEMD_UNIT_DIR / f"{unit}.service": (
            "[Unit]\nDescription=Owned collector env recovery\n"
            "After=local-fs.target\n"
            f"Before={HOST_COLLECTOR_UNIT}\nStartLimitIntervalSec=0\n"
            "[Service]\nType=oneshot\n"
            f"ExecStart={' '.join(collector_command(record))}\n"
            "Restart=on-failure\nRestartSec=5s\nRestartPreventExitStatus=78\n"
            "TimeoutStartSec=120s\nTimeoutStopSec=10s\nKillMode=control-group\n"
            "SendSIGKILL=yes\n[Install]\nWantedBy=multi-user.target\n"
        ).encode(),
        SYSTEMD_UNIT_DIR / f"{unit}.timer": (
            "[Unit]\nDescription=Fixed collector env recovery deadline\n[Timer]\n"
            f"OnBootSec={record['restore_deadline_monotonic']:.3f}s\n"
            f"Unit={unit}.service\nAccuracySec=1ms\nRandomizedDelaySec=0\n"
            "[Install]\nWantedBy=timers.target\n"
        ).encode(),
        SYSTEMD_UNIT_DIR
        / f"{HOST_COLLECTOR_UNIT}.d"
        / "90-gpu-fault-collector-env.conf": (
            f"[Unit]\nRequires={unit}.service\nAfter={unit}.service\n"
            "[Service]\n"
            f"ExecStartPre={' '.join(collector_command(record, start_check=True))}\n"
        ).encode(),
    }


def collector_unit_state(name: str) -> dict[str, str]:
    properties = (
        "Id",
        "LoadState",
        "ActiveState",
        "SubState",
        "FragmentPath",
        "DropInPaths",
        "NeedDaemonReload",
        "ExecStart",
        "ExecStartPre",
        "Type",
        "Restart",
        "RestartUSec",
        "RestartPreventExitStatus",
        "TimeoutStartUSec",
        "TimeoutStopUSec",
        "KillMode",
        "SendSIGKILL",
        "MainPID",
        "ControlPID",
        "ControlGroup",
        "Job",
        "Unit",
        "NextElapseUSecMonotonic",
        "AccuracyUSec",
        "RandomizedDelayUSec",
        "UnitFileState",
        "Before",
        "After",
        "Requires",
        "InvocationID",
        "Result",
        "ExecMainStatus",
    )
    result = run(
        ["systemctl", "show", name, "--property=" + ",".join(properties)],
        check=False,
        timeout=10,
    )
    state = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if (
        state.get("Id") != name
        or state.get("LoadState") not in {"loaded", "not-found"}
        or result.returncode
        and not (result.returncode in {1, 4} and state.get("LoadState") == "not-found")
    ):
        raise CollectorEnvGuardError("collector systemd unit state is unknown")
    return state


def collector_exec_matches(value: str, command: list[str]) -> bool:
    starts = re.findall(r"argv\[\]=(.*?) ;", value)
    paths = re.findall(r"\bpath=(\S+) ;", value)
    return (
        len(starts) == 1 and shlex.split(starts[0]) == command and paths == [command[0]]
    )


def collector_host_state(record: dict[str, Any], *, guard: bool) -> dict[str, str]:
    state = collector_unit_state(HOST_COLLECTOR_UNIT)
    _, unit = collector_restore_paths(record["run_id"])
    dropin = next(
        path for path in collector_unit_contents(record) if path.suffix == ".conf"
    )
    if (
        state.get("LoadState") != "loaded"
        or state.get("FragmentPath") != record["host_unit_path"]
        or hashlib.sha256(power_read_file(Path(record["host_unit_path"]))).hexdigest()
        != record["host_unit_sha256"]
        or state.get("NeedDaemonReload") != "no"
        or state.get("DropInPaths") != (str(dropin) if guard else "")
        or (f"{unit}.service" in state.get("Requires", "").split()) != guard
        or guard
        and (
            f"{unit}.service" not in state.get("After", "").split()
            or not collector_exec_matches(
                state.get("ExecStartPre", ""),
                collector_command(record, start_check=True),
            )
        )
    ):
        raise CollectorEnvGuardError("host collector start dependency or unit changed")
    return state


def collector_owned_unit(
    record: dict[str, Any], path: Path, *, absent: bool = False
) -> dict[str, str]:
    state = collector_unit_state(path.name)
    if state["LoadState"] == "not-found" and absent:
        if (path.exists() or path.is_symlink()) and power_read_file(
            path
        ) != collector_unit_contents(record)[path]:
            raise CollectorEnvGuardError(
                "unloaded collector recovery file is not owned"
            )
        power_assert_stopped(state, path.name)
        return state
    if (
        power_read_file(path) != collector_unit_contents(record)[path]
        or state.get("FragmentPath") != str(path)
        or state.get("DropInPaths") != ""
        or state.get("NeedDaemonReload") != "no"
    ):
        raise CollectorEnvGuardError("collector recovery unit ownership changed")
    if path.suffix == ".service" and (
        not collector_exec_matches(
            state.get("ExecStart", ""), collector_command(record)
        )
        or state.get("Type") != "oneshot"
        or state.get("Restart") != "on-failure"
        or state.get("RestartPreventExitStatus") != "78"
        or power_systemd_seconds(state.get("RestartUSec", "")) != 5
        or power_systemd_seconds(state.get("TimeoutStartUSec", "")) != 120
        or power_systemd_seconds(state.get("TimeoutStopUSec", "")) != 10
        or state.get("KillMode") != "control-group"
        or state.get("SendSIGKILL") != "yes"
        or HOST_COLLECTOR_UNIT not in state.get("Before", "").split()
    ):
        raise CollectorEnvGuardError(
            "collector recovery service command or limits changed"
        )
    return state


def collector_verify_armed(record: dict[str, Any]) -> dict[str, Any]:
    backup, unit = collector_restore_paths(record["run_id"])
    for path, contents in collector_unit_contents(record).items():
        if power_read_file(path) != contents:
            raise CollectorEnvGuardError(
                "collector recovery file changed before arming"
            )
    for suffix, target in (
        (".service", "multi-user.target"),
        (".timer", "timers.target"),
    ):
        link = SYSTEMD_UNIT_DIR / f"{target}.wants" / (unit + suffix)
        if (
            not link.is_symlink()
            or link.lstat().st_uid != os.geteuid()
            or link.resolve(strict=True) != SYSTEMD_UNIT_DIR / (unit + suffix)
        ):
            raise CollectorEnvGuardError("collector boot enablement link is not owned")
        power_sync_directory(link.parent)
    power_sync_directory(SYSTEMD_UNIT_DIR)
    service = collector_owned_unit(record, SYSTEMD_UNIT_DIR / f"{unit}.service")
    timer = collector_owned_unit(record, SYSTEMD_UNIT_DIR / f"{unit}.timer")
    if (
        service.get("UnitFileState") != "enabled"
        or timer.get("UnitFileState") != "enabled"
        or timer.get("ActiveState") != "active"
        or timer.get("SubState") != "waiting"
        or timer.get("Unit") != f"{unit}.service"
        or abs(
            power_systemd_seconds(timer.get("NextElapseUSecMonotonic", ""))
            - record["restore_deadline_monotonic"]
        )
        > 0.002
        or power_systemd_seconds(timer.get("AccuracyUSec", "")) > 0.001
        or power_systemd_seconds(timer.get("RandomizedDelayUSec", "")) != 0
        or hashlib.sha256(collector_private_file(backup)).hexdigest()
        != record["baseline_sha256"]
        or hashlib.sha256(
            collector_private_file(backup.with_suffix(".probe.py"))
        ).hexdigest()
        != record["probe_sha256"]
        or hashlib.sha256(
            collector_private_file(backup.with_suffix(".nonce"))
        ).hexdigest()
        != record["owner_nonce_sha256"]
    ):
        raise CollectorEnvGuardError("collector recovery arming proof is incomplete")
    collector_host_state(record, guard=True)
    return {
        "run_id": record["run_id"],
        "intent_sha256": record["intent_sha256"],
        "boot_id": record["boot_id"],
        "timer_armed": True,
        "boot_restore_armed": True,
        "restore_deadline_monotonic": record["restore_deadline_monotonic"],
        "restore_deadline_epoch": record["restore_deadline_epoch"],
    }


def collector_before_deadline(record: dict[str, Any]) -> bool:
    return bool(
        BOOT_ID_FILE.read_text().strip() == record["boot_id"]
        and time.monotonic() < record["restore_deadline_monotonic"]
        and datetime.now(timezone.utc).timestamp() < record["restore_deadline_epoch"]
    )


def collector_check_start(record: dict[str, Any]) -> None:
    _, digest = collector_verify_identity(record)
    boot_id = BOOT_ID_FILE.read_text().strip()
    if record["state"] in {"RESTORED", "CLEANING", "CLEANED"}:
        if (
            digest != record["baseline_sha256"]
            or record.get("restored_boot_id") != boot_id
        ):
            raise CollectorEnvGuardError("collector boot restoration is not proven")
        return
    backup, _ = collector_restore_paths(record["run_id"])
    armed = collector_json(backup.with_suffix(".armed.json"))
    if (
        record["state"] not in {"ARMED", "MUTATING", "ACTIVE"}
        or not collector_before_deadline(record)
        or armed is None
        or armed.get("intent_sha256") != record["intent_sha256"]
        or armed.get("boot_id") != boot_id
        or armed.get("timer_armed") is not True
        or armed.get("boot_restore_armed") is not True
    ):
        raise CollectorEnvGuardError(
            "collector start has no valid recovery authorization"
        )


def collector_env_start_check(arguments: argparse.Namespace) -> None:
    collector_require_root()
    run_id, nonce = collector_nonce(arguments)
    record = collector_load_record(run_id, nonce)
    if record is None:
        raise CollectorEnvGuardError("collector start has no recovery intent")
    collector_check_start(record)
    emit(
        {
            "run_id": run_id,
            "intent_sha256": record["intent_sha256"],
            "start_allowed": True,
        }
    )


def replace_collector_env(contents: bytes, *, expected_sha256: str) -> None:
    original_bytes = collector_private_file(COLLECTOR_ENV)
    original = COLLECTOR_ENV.lstat()
    if hashlib.sha256(original_bytes).hexdigest() != expected_sha256:
        raise CollectorEnvGuardError("collector env compare-and-swap rejected drift")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=COLLECTOR_ENV.parent, prefix=".collector-acceptance-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            current = os.fstat(stream.fileno())
            if (current.st_uid, current.st_gid) != (original.st_uid, original.st_gid):
                os.fchown(stream.fileno(), original.st_uid, original.st_gid)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        current = COLLECTOR_ENV.lstat()
        if (current.st_dev, current.st_ino) != (
            original.st_dev,
            original.st_ino,
        ) or hashlib.sha256(
            collector_private_file(COLLECTOR_ENV)
        ).hexdigest() != expected_sha256:
            raise CollectorEnvGuardError(
                "collector env changed before atomic replacement"
            )
        os.replace(temporary, COLLECTOR_ENV)
        power_sync_directory(COLLECTOR_ENV.parent)
        if collector_private_file(COLLECTOR_ENV) != contents:
            raise CollectorEnvGuardError("collector env replacement readback differs")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def override_expected_gpu_count(arguments: argparse.Namespace) -> None:
    collector_require_root()
    run_id, nonce = collector_nonce(arguments)
    with collector_env_lock():
        backup, unit = collector_restore_paths(run_id)
        path = collector_override_record(backup)
        if collector_json(ACCEPTANCE_STATE / "collector-env-owner.json") is not None:
            raise CollectorEnvGuardError(
                "another collector env operation owns this node"
            )
        for previous in ACCEPTANCE_STATE.glob("collector-env-*.override.json"):
            record = collector_json(previous)
            if previous == path or record is None or record.get("state") != "CLEANED":
                raise CollectorEnvGuardError(
                    "collector env run already exists; cleanup only"
                )
        for directory, pattern in (
            (ACCEPTANCE_STATE, "collector-env-*.backup"),
            (SYSTEMD_UNIT_DIR, "gpu-fault-collector-env-restore-*"),
            (POWER_TRANSIENT_UNIT_DIR, "gpu-fault-collector-env-restore-*"),
        ):
            if next(directory.glob(pattern), None) is not None:
                raise CollectorEnvGuardError(
                    "collector recovery artifacts already exist"
                )
        try:
            original = collector_private_file(COLLECTOR_ENV)
        except FileNotFoundError as exc:
            raise CollectorEnvGuardError("collector env baseline is missing") from exc
        values = collector_env_values(original)
        current = int(values["GPU_FAULT_EXPECTED_GPU_COUNT"])
        target = arguments.value
        if (
            type(target) is not int
            or target != current + 1
            or current <= 0
            or type(arguments.restore_seconds) is not int
            or not 60 <= arguments.restore_seconds <= 900
            or hashlib.sha256(original).hexdigest() != arguments.expected_env_sha256
            or BOOT_ID_FILE.read_text().strip() != arguments.expected_boot_id
            or not arguments.expected_boot_id
            or values["GPU_FAULT_CLUSTER_ID"] != arguments.cluster_id
            or values["NODE_NAME"] != arguments.node_id
        ):
            raise CollectorEnvGuardError(
                "collector baseline, target or deadline binding differs"
            )
        updated, count = re.subn(
            rb"(?m)^GPU_FAULT_EXPECTED_GPU_COUNT=.*$",
            f"GPU_FAULT_EXPECTED_GPU_COUNT={target}".encode(),
            original,
        )
        if count != 1:
            raise CollectorEnvGuardError(
                "collector env has no unique expected GPU count"
            )
        if not (POWER_CGROUP_ROOT / "cgroup.controllers").is_file():
            raise CollectorEnvGuardError(
                "collector recovery stop proof requires cgroup v2"
            )
        host = collector_unit_state(HOST_COLLECTOR_UNIT)
        if (
            host.get("LoadState") != "loaded"
            or host.get("ActiveState") != "active"
            or host.get("SubState") != "running"
            or host.get("DropInPaths") != ""
            or host.get("NeedDaemonReload") != "no"
            or not host.get("FragmentPath")
            or host.get("Job") not in {"", "0", "[not set]"}
        ):
            raise CollectorEnvGuardError("host collector baseline is not stable")
        source = Path(__file__).read_bytes()
        now = time.monotonic()
        epoch = datetime.now(timezone.utc).timestamp()
        record = {
            "schema_version": 3,
            "run_id": run_id,
            "state": "PREPARED",
            "owner_nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
            "cluster_id": arguments.cluster_id,
            "node_id": arguments.node_id,
            "node_instance_id": values.get("GPU_FAULT_NODE_INSTANCE_ID"),
            "boot_id": arguments.expected_boot_id,
            "baseline": current,
            "override": target,
            "baseline_sha256": hashlib.sha256(original).hexdigest(),
            "applied_sha256": hashlib.sha256(updated).hexdigest(),
            "created_monotonic": now,
            "created_epoch": epoch,
            "restore_seconds": arguments.restore_seconds,
            "restore_deadline_monotonic": math.ceil(
                (now + arguments.restore_seconds) * 1000
            )
            / 1000,
            "restore_deadline_epoch": epoch + arguments.restore_seconds,
            "mutation_started": False,
            "probe_sha256": hashlib.sha256(source).hexdigest(),
            "host_unit_path": host["FragmentPath"],
            "host_unit_sha256": hashlib.sha256(
                power_read_file(Path(host["FragmentPath"]))
            ).hexdigest(),
            **collector_runtime_identity(schema_version=3),
        }
        record["intent_sha256"] = collector_intent_digest(record)
        units = collector_unit_contents(record)
        materials = (
            backup,
            path,
            backup.with_suffix(".probe.py"),
            backup.with_suffix(".nonce"),
            backup.with_suffix(".armed.json"),
            *units,
        )
        if any(item.exists() or item.is_symlink() for item in materials):
            raise CollectorEnvGuardError("collector recovery path already exists")
        for item in units:
            if (
                item.suffix != ".conf"
                and collector_unit_state(item.name)["LoadState"] != "not-found"
            ):
                raise CollectorEnvGuardError("collector recovery unit already exists")
        power_write_json(path, record)
        power_write_json(
            ACCEPTANCE_STATE / "collector-env-owner.json",
            {
                key: record[key]
                for key in ("run_id", "owner_nonce_sha256", "intent_sha256")
            },
        )
        power_create_file(backup, original, 0o600)
        power_create_file(backup.with_suffix(".probe.py"), source, 0o500)
        power_create_file(backup.with_suffix(".nonce"), nonce.encode(), 0o600)
        for item, contents in units.items():
            if item.suffix == ".conf":
                item.parent.mkdir(mode=0o755, exist_ok=True)
                info = item.parent.lstat()
                if (
                    not stat.S_ISDIR(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o022
                ):
                    raise CollectorEnvGuardError(
                        "collector drop-in directory is not trusted"
                    )
            power_create_file(item, contents, 0o600)
        run(["systemctl", "daemon-reload"], timeout=10)
        run(["systemctl", "enable", f"{unit}.service"], timeout=10)
        run(["systemctl", "enable", "--now", f"{unit}.timer"], timeout=10)
        proof = collector_verify_armed(record)
        # Bind ARMED to a fresh, successful execution of the copied recovery service.
        before = collector_owned_unit(record, SYSTEMD_UNIT_DIR / f"{unit}.service")
        power_assert_stopped(before, f"{unit}.service")
        run(["systemctl", "start", f"{unit}.service"], timeout=30)
        after = collector_owned_unit(record, SYSTEMD_UNIT_DIR / f"{unit}.service")
        power_assert_stopped(after, f"{unit}.service")
        if (
            not after.get("InvocationID")
            or after["InvocationID"] == before.get("InvocationID")
            or after.get("Result") != "success"
            or after.get("ExecMainStatus") != "0"
        ):
            raise CollectorEnvGuardError(
                "independent collector recovery did not acknowledge ARMED"
            )
        proof["service_invocation_id"] = after["InvocationID"]
        power_create_file(
            backup.with_suffix(".armed.json"),
            json.dumps(proof, sort_keys=True).encode(),
            0o600,
        )
        record["state"] = "ARMED"
        power_write_json(path, record)
        collector_verify_identity(record)
        collector_verify_armed(record)
        if not collector_before_deadline(record):
            raise CollectorEnvGuardError("collector mutation deadline elapsed")
        record["state"] = "MUTATING"
        record["mutation_started"] = True
        power_write_json(path, record)
        replace_collector_env(updated, expected_sha256=record["baseline_sha256"])
        run(["systemctl", "restart", HOST_COLLECTOR_UNIT], timeout=120)
        collector_health(record, guard=True)
        if not collector_before_deadline(record):
            raise CollectorEnvGuardError(
                "collector override activation exceeded its deadline"
            )
        record["state"] = "ACTIVE"
        power_write_json(path, record)
        emit(
            {
                **collector_receipt(record),
                **proof,
                "mutation_started": True,
                "cleanup_verified": False,
            }
        )


def collector_receipt(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "schema_version",
            "run_id",
            "state",
            "cluster_id",
            "node_id",
            "boot_id",
            "mutation_started",
            "intent_sha256",
            "baseline_sha256",
            "applied_sha256",
            "restore_deadline_monotonic",
            "restore_deadline_epoch",
        )
    }


def collector_health(record: dict[str, Any], *, guard: bool) -> None:
    collector_verify_identity(record)
    state = collector_host_state(record, guard=guard)
    if (
        state.get("ActiveState") != "active"
        or state.get("SubState") != "running"
        or not state.get("InvocationID")
        or not state.get("MainPID", "").isdecimal()
        or int(state["MainPID"]) <= 1
        or state.get("Job") not in {"", "0", "[not set]"}
    ):
        raise CollectorEnvGuardError(
            "collector service health or job completion is unknown"
        )


def collector_automatic_restart(record: dict[str, Any]) -> None:
    state = collector_host_state(record, guard=True)
    if (
        state.get("ActiveState") == "active"
        and state.get("SubState") == "running"
        and state.get("MainPID", "").isdecimal()
        and int(state["MainPID"]) > 1
        and state.get("ControlPID") == "0"
        and state.get("Job") in {"", "0", "[not set]"}
    ):
        run(["systemctl", "--no-block", "restart", HOST_COLLECTOR_UNIT], timeout=10)
        return
    if (
        state.get("ActiveState") == "inactive"
        and state.get("SubState") == "dead"
        and state.get("MainPID") == "0"
        and state.get("ControlPID") == "0"
        and BOOT_ID_FILE.read_text().strip() != record["boot_id"]
    ):
        # The recovery service precedes the original boot start. Do not replace it.
        return
    raise CollectorEnvGuardError(
        "collector service state is unknown during automatic restore"
    )


def collector_finish_units(record: dict[str, Any]) -> None:
    units = collector_unit_contents(record)
    dropin = next(path for path in units if path.suffix == ".conf")
    for path, expected in units.items():
        if (path.exists() or path.is_symlink()) and power_read_file(path) != expected:
            raise CollectorEnvGuardError(
                "refusing to remove an unowned collector recovery file"
            )
    for path in units:
        if path.suffix == ".conf":
            continue
        state = collector_owned_unit(record, path, absent=True)
        if path.suffix == ".timer" and state["LoadState"] != "not-found":
            run(["systemctl", "disable", "--now", path.name], timeout=20)
            power_assert_stopped(collector_owned_unit(record, path), path.name)
    # Remove the dependency before stopping its service; Requires= propagates stops.
    dropin.unlink(missing_ok=True)
    if dropin.parent.exists():
        power_sync_directory(dropin.parent)
    run(["systemctl", "daemon-reload"], timeout=10)
    collector_host_state(record, guard=False)
    for path in units:
        if path.suffix == ".conf":
            continue
        state = collector_owned_unit(record, path, absent=True)
        if state["LoadState"] != "not-found":
            run(["systemctl", "disable", "--now", path.name], timeout=20)
            power_assert_stopped(collector_owned_unit(record, path), path.name)
        path.unlink(missing_ok=True)
    power_sync_directory(SYSTEMD_UNIT_DIR)
    run(["systemctl", "daemon-reload"], timeout=10)
    collector_verify_removed(record)


def collector_verify_removed(record: dict[str, Any]) -> None:
    _, unit = collector_restore_paths(record["run_id"])
    for suffix, target in (
        (".service", "multi-user.target"),
        (".timer", "timers.target"),
    ):
        link = SYSTEMD_UNIT_DIR / f"{target}.wants" / (unit + suffix)
        if link.exists() or link.is_symlink():
            raise CollectorEnvGuardError(
                "collector boot enablement link remains installed"
            )
    for path in collector_unit_contents(record):
        if path.exists() or path.is_symlink():
            raise CollectorEnvGuardError(
                "collector recovery unit file remains installed"
            )
        if path.suffix != ".conf":
            state = collector_unit_state(path.name)
            power_assert_stopped(state, path.name)
            if state["LoadState"] != "not-found":
                raise CollectorEnvGuardError("collector recovery unit remains loaded")
    collector_health(record, guard=False)


def collector_remove_materials(record: dict[str, Any]) -> None:
    backup, _ = collector_restore_paths(record["run_id"])
    for path, digest in (
        (backup, record["baseline_sha256"]),
        (backup.with_suffix(".probe.py"), record["probe_sha256"]),
        (backup.with_suffix(".nonce"), record["owner_nonce_sha256"]),
    ):
        if path.exists() or path.is_symlink():
            if hashlib.sha256(collector_private_file(path)).hexdigest() != digest:
                raise CollectorEnvGuardError(
                    "collector recovery material identity changed"
                )
            path.unlink()
    armed = backup.with_suffix(".armed.json")
    proof = collector_json(armed)
    if proof is not None:
        if proof.get("intent_sha256") != record["intent_sha256"]:
            raise CollectorEnvGuardError("collector ARMED receipt identity changed")
        armed.unlink()
    (ACCEPTANCE_STATE / "collector-env-owner.json").unlink(missing_ok=True)
    power_sync_directory(ACCEPTANCE_STATE)


def restore_collector_env(arguments: argparse.Namespace) -> None:
    collector_require_root()
    run_id, nonce = collector_nonce(arguments)
    automatic = getattr(arguments, "automatic", False)
    # The original-boot ARMED execution and start dependencies are read-only.
    # Every actual restoration reloads its authorization under the exclusive lock.
    if automatic:
        record = collector_load_record(run_id, nonce)
        if record is None:
            raise CollectorEnvGuardError(
                "automatic collector recovery intent is missing"
            )
        if record["state"] == "PREPARED" and collector_before_deadline(record):
            collector_verify_identity(record)
            proof = collector_verify_armed(record)
            emit({**proof, "mutation_started": False, "restored": False})
            return
        if (
            record["state"] in {"ARMED", "MUTATING", "ACTIVE"}
            and collector_before_deadline(record)
            or (
                record["state"] in {"CLEANING", "CLEANED"}
                or record["state"] == "RESTORED"
                and record.get("manual_restore") is True
            )
            and record.get("restored_boot_id") == BOOT_ID_FILE.read_text().strip()
        ):
            collector_check_start(record)
            emit(
                {
                    **collector_receipt(record),
                    "restored": record["state"] in {"RESTORED", "CLEANING", "CLEANED"},
                    "deferred": record["state"] in {"ARMED", "MUTATING", "ACTIVE"},
                    "cleanup_verified": False,
                }
            )
            return
    with collector_env_lock():
        record = collector_load_record(run_id, nonce)
        if record is None:
            if automatic:
                raise CollectorEnvGuardError(
                    "automatic collector recovery intent is missing"
                )
            emit(
                {
                    "run_id": run_id,
                    "state": "NOT_STARTED",
                    "mutation_started": False,
                    "no_mutation": True,
                    "restored": False,
                    "cleanup_verified": True,
                    "timer_disarmed": True,
                    "recovery_stopped": True,
                }
            )
            return
        backup, _ = collector_restore_paths(run_id)
        path = collector_override_record(backup)
        current, digest = collector_verify_identity(record)
        if record["state"] != "CLEANED":
            try:
                contents = collector_private_file(backup)
            except FileNotFoundError:
                if record["mutation_started"]:
                    raise CollectorEnvGuardError(
                        "collector backup is missing"
                    ) from None
                contents = current
            if hashlib.sha256(contents).hexdigest() != record["baseline_sha256"]:
                raise CollectorEnvGuardError(
                    "collector backup digest differs from its intent"
                )
            if record["state"] not in {"RESTORED", "CLEANING"}:
                record["state"] = "RESTORING"
                power_write_json(path, record)
                if digest != record["baseline_sha256"]:
                    if not record["mutation_started"]:
                        raise CollectorEnvGuardError(
                            "collector env write has no mutation intent"
                        )
                    replace_collector_env(
                        contents, expected_sha256=record["applied_sha256"]
                    )
                if (
                    hashlib.sha256(collector_private_file(COLLECTOR_ENV)).hexdigest()
                    != record["baseline_sha256"]
                ):
                    raise CollectorEnvGuardError(
                        "collector restore readback differs from baseline"
                    )
                record["state"] = "RESTORED"
                record["restored_boot_id"] = BOOT_ID_FILE.read_text().strip()
                power_write_json(path, record)
            elif record.get("restored_boot_id") != BOOT_ID_FILE.read_text().strip():
                record["restored_boot_id"] = BOOT_ID_FILE.read_text().strip()
                power_write_json(path, record)
            if automatic:
                if record["mutation_started"]:
                    collector_automatic_restart(record)
                emit(
                    {
                        **collector_receipt(record),
                        "restored": True,
                        "cleanup_verified": False,
                        "cleanup_deferred": True,
                    }
                )
                return
            if record["state"] != "CLEANING":
                record["manual_restore"] = True
                power_write_json(path, record)
                if record["mutation_started"]:
                    run(["systemctl", "restart", HOST_COLLECTOR_UNIT], timeout=120)
                    dropin = next(
                        item
                        for item in collector_unit_contents(record)
                        if item.suffix == ".conf"
                    )
                    collector_health(record, guard=dropin.exists())
                record["state"] = "CLEANING"
                power_write_json(path, record)
            collector_finish_units(record)
            record["state"] = "CLEANED"
            power_write_json(path, record)
        else:
            collector_verify_removed(record)
        collector_remove_materials(record)
        emit(
            {
                **collector_receipt(record),
                "restored": True,
                "cleanup_verified": True,
                "timer_disarmed": True,
                "recovery_stopped": True,
                "no_mutation": not record["mutation_started"],
            }
        )


def efa_restore_unit(run_id: str, bdf: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{bdf}".encode()).hexdigest()[:16]
    return f"gpu-fault-efa-restore-{digest}"


def unbind_efa(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    bdf = normalize_bdf(arguments.pci_bdf)
    driver = Path("/sys/bus/pci/drivers/efa")
    if not driver.joinpath(bdf).exists():
        raise ProbeError("EFA BDF is not currently bound")
    unit = efa_restore_unit(run_id, bdf)
    run(
        [
            "systemd-run",
            "--unit",
            unit,
            f"--on-active={arguments.restore_seconds}s",
            "/bin/bash",
            "-ceu",
            f"printf '%s\\n' {shlex.quote(bdf)} > /sys/bus/pci/drivers/efa/bind",
        ]
    )
    driver.joinpath("unbind").write_text(bdf + "\n")
    emit(
        {
            "pci_bdf": bdf,
            "restore_unit": unit + ".timer",
            "restore_seconds": arguments.restore_seconds,
        }
    )


def restore_efa(arguments: argparse.Namespace) -> None:
    """Disarm the bind fail-safe and say who rebound the function.

    COLLECT-017 A passes only when the control plane's REMEDIATE_EFA_DRIVER
    rebound the BDF, so the runner needs to know whether the function was
    already bound when this ran (``already_bound``) and whether the fail-safe
    timer's service had fired and done it instead (``timer_fired``). Both are
    read before anything here changes them.
    """

    run_id = safe_id(arguments.run_id, "run ID")
    bdf = normalize_bdf(arguments.pci_bdf)
    driver = Path("/sys/bus/pci/drivers/efa")
    unit = efa_restore_unit(run_id, bdf)
    already_bound = driver.joinpath(bdf).exists()
    timer_was_active = (
        run(["systemctl", "is-active", unit + ".timer"], check=False).returncode == 0
    )
    started = run(
        ["systemctl", "show", unit + ".service", "-p", "ExecMainStartTimestamp"],
        check=False,
    ).stdout.strip()
    timer_fired = bool(started.split("=", 1)[-1].strip())
    if not already_bound:
        driver.joinpath("bind").write_text(bdf + "\n")
    run(["systemctl", "stop", unit + ".timer"], check=False)
    run(["systemctl", "reset-failed", unit + ".service"], check=False)
    emit(
        {
            "pci_bdf": bdf,
            "bound": driver.joinpath(bdf).exists(),
            "already_bound": already_bound,
            "timer_was_active": timer_was_active,
            "timer_fired": timer_fired,
        }
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    def command(
        name: str, handler: Callable[[argparse.Namespace], None]
    ) -> argparse.ArgumentParser:
        child = commands.add_parser(name)
        child.set_defaults(handler=handler)
        return child

    command("firmware-premise", firmware_premise)
    command("reset-audit", reset_audit)
    command("snapshot", snapshot)
    command("fm-cursor", fm_cursor)
    delivery = command("fm-delivery-evidence", fm_delivery_evidence)
    delivery.add_argument("--cursor", default="")
    command("efa-inventory", efa_inventory_command)
    command("inventory-config", inventory_config)
    sample = command("sample-gpu-inventory", sample_gpu_inventory)
    sample.add_argument("--run-id", required=True)
    sample.add_argument("--cluster-id", required=True)
    sample.add_argument("--node-id", required=True)
    sample.add_argument("--expected-env-sha256", required=True)
    sample.add_argument("--expected-boot-id", required=True)
    sample.add_argument("--expected-gpu-count", type=int, required=True)
    publish = command("publish-gpu-inventory", publish_gpu_inventory)
    publish.add_argument("--run-id", required=True)
    publish.add_argument("--expected-sha256", required=True)
    publish.add_argument("--receipt-json", required=True)
    publish.add_argument("--confirm", required=True)
    command("gpu-identity", gpu_identity)

    xid = command("write-xid", write_xid)
    xid.add_argument("--xid", type=int, choices=sorted(ALLOWED_XIDS), required=True)
    xid.add_argument("--marker", required=True)
    xid.add_argument("--pci-bdf", required=True)
    xid.add_argument("--pid", type=int, default=931000)
    xid.add_argument("--message", default="regional collector acceptance")

    kill = command("kill-workload", kill_workload)
    kill.add_argument("--pod-uid", required=True)

    sxid = command("append-sxid", append_sxid)
    sxid.add_argument("--sxid", type=int, choices=sorted(ALLOWED_SXIDS), required=True)
    sxid.add_argument("--marker", required=True)
    sxid.add_argument("--pci-bdf", required=True)
    sxid.add_argument("--classification", default="Fatal")
    sxid.add_argument("--message", default="regional collector acceptance")
    sxid.add_argument("--switch", type=int, default=0)
    sxid.add_argument("--port", type=int, default=12)
    sxid.add_argument("--tid", type=int, default=990001)
    sxid.add_argument("--include-switch", action=argparse.BooleanOptionalAction)

    restart = command("restart-service", restart_service)
    restart.add_argument("--service", choices=sorted(ALLOWED_SERVICES), required=True)

    persistence = command("set-persistence-mode", set_persistence_mode)
    persistence.add_argument("--enabled", choices=("true", "false"), required=True)

    throttle = command("throttle-gpu", throttle_gpu)
    throttle.add_argument("--run-id", required=True)
    throttle.add_argument("--gpu-index", type=int, default=0)
    throttle.add_argument("--load-seconds", type=int, default=180)
    throttle.add_argument("--restore-seconds", type=int, default=600)
    throttle.add_argument("--load-only", action="store_true", help=argparse.SUPPRESS)

    restore_power = command("restore-gpu-power-limit", restore_gpu_power_limit)
    restore_power.add_argument("--run-id", required=True)
    restore_power.add_argument("--automatic", action="store_true")

    override = command("override-expected-gpu-count", override_expected_gpu_count)
    override.add_argument("--run-id", required=True)
    override.add_argument("--owner-nonce", required=True)
    override.add_argument("--expected-env-sha256", required=True)
    override.add_argument("--expected-boot-id", required=True)
    override.add_argument("--cluster-id", required=True)
    override.add_argument("--node-id", required=True)
    override.add_argument("--value", type=int, required=True)
    override.add_argument("--restore-seconds", type=int, default=600)

    restore_env = command("restore-collector-env", restore_collector_env)
    restore_env.add_argument("--automatic", action="store_true")
    start_check = command("collector-env-start-check", collector_env_start_check)
    for bound in (restore_env, start_check):
        bound.add_argument("--run-id", required=True)
        owner = bound.add_mutually_exclusive_group(required=True)
        owner.add_argument("--owner-nonce", default="")
        owner.add_argument("--owner-nonce-file", default="", help=argparse.SUPPRESS)

    unbind = command("unbind-efa", unbind_efa)
    unbind.add_argument("--run-id", required=True)
    unbind.add_argument("--pci-bdf", required=True)
    unbind.add_argument("--restore-seconds", type=int, default=300)

    restore = command("restore-efa", restore_efa)
    restore.add_argument("--run-id", required=True)
    restore.add_argument("--pci-bdf", required=True)
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


def main() -> int:
    arguments = parser().parse_args()
    try:
        restore_seconds = getattr(arguments, "restore_seconds", 300)
        if not 60 <= restore_seconds <= 900:
            raise ProbeError("restore seconds is outside 60..900")
        arguments.handler(arguments)
    except PowerRecoveryIdentityError as exc:
        emit(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "restored": False,
                "cleanup_verified": False,
                "manual_confirmation_required": True,
            }
        )
        return 78
    except Exception as exc:
        if arguments.command in COLLECTOR_ENV_COMMANDS:
            retryable = isinstance(exc, CollectorEnvBusyError) or not isinstance(
                exc, CollectorEnvGuardError
            )
            error_site = exception_site(exc)
            error_code = hashlib.sha256(str(exc).encode()).hexdigest()[:16]
            failure_receipt = collector_guard_failure(
                arguments, error_code=error_code, error_site=error_site
            )
            emit(
                {
                    "error": (
                        str(exc)
                        if isinstance(exc, CollectorEnvGuardError)
                        else f"collector env recovery failed ({type(exc).__name__})"
                    ),
                    "error_kind": "collector_env_guard",
                    "error_code": error_code,
                    "error_site": error_site,
                    "failure_receipt": failure_receipt,
                    "cleanup_verified": False,
                    "manual_confirmation_required": True,
                    "retryable": retryable,
                }
            )
            return 1 if retryable else 78
        # The host fixture withholds this output; the class, a digest of the
        # message and the site are the message-free trail it may show.
        emit(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "error_kind": "probe",
                "error_class": type(exc).__name__,
                "error_code": hashlib.sha256(str(exc).encode()).hexdigest()[:16],
                "error_site": exception_site(exc),
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
