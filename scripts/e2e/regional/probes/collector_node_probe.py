#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
from typing import Any


COLLECTOR_ENV = Path("/etc/gpu-fault/collector.env")
FM_LOG = Path("/var/log/fabricmanager.log")
FM_STATE_CANDIDATES = (
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


class ProbeError(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 180,
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
        encoding="utf-8",
        errors="replace",
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


def gpu_inventory() -> list[dict[str, str]]:
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader",
        ]
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


def gpu_power_state() -> list[dict[str, Any]]:
    """Per-GPU power limits and load, the inputs of the throttle candidate.

    The control plane only calls a power violation a candidate when the draw sits
    at the enforced limit *and* utilization is high, so the acceptance run has to
    be able to read all three from the node rather than assume them.
    """

    completed = run(
        [
            "nvidia-smi",
            (
                "--query-gpu=index,uuid,power.draw,power.limit,"
                "power.min_limit,power.default_limit,utilization.gpu"
            ),
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
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
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != len(fields):
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
        result.append(entry)
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


def throttle_gpu(arguments: argparse.Namespace) -> None:
    """Lower every GPU's power limit and load one GPU up against that limit.

    This is the only non-destructive way to produce a *real* correlated power
    throttle: the enforced limit drops to the driver's own minimum, so a tensor
    load pins the draw at the limit under full utilization. It makes the card
    cooler than a stock load rather than hotter, and a single ``nvidia-smi -pl``
    restores it -- both a deadman timer and the runner's own cleanup do that.
    """

    run_id = safe_id(arguments.run_id, "run ID")
    state = gpu_power_state()
    if not state:
        raise ProbeError("cannot read GPU power limits")
    indexes = {item["index"] for item in state}
    if arguments.gpu_index not in indexes:
        raise ProbeError("GPU index is not present on this node")
    minimums = [item["power_min_limit_w"] for item in state]
    defaults = {item["power_default_limit_w"] for item in state}
    if None in minimums or None in defaults or len(defaults) != 1:
        raise ProbeError("node does not report a single default GPU power limit")
    target = int(max(float(value) for value in minimums))
    default = int(next(iter(defaults)) or 0)
    if not 0 < target < default:
        raise ProbeError("GPU minimum power limit is not below the default")
    if arguments.load_seconds > arguments.restore_seconds:
        raise ProbeError("GPU load must end before the deadman restore fires")
    unit = power_limit_restore_unit(run_id)
    load_unit = f"{unit}-load"
    run(
        [
            "systemd-run",
            "--unit",
            unit,
            f"--on-active={arguments.restore_seconds}s",
            "/usr/bin/nvidia-smi",
            "-pl",
            str(default),
        ]
    )
    run(["nvidia-smi", "-pl", str(target)])
    run(
        [
            "systemd-run",
            "--unit",
            load_unit,
            "/usr/bin/timeout",
            str(arguments.load_seconds + 60),
            proftester_binary(),
            "--no-dcgm-validation",
            "-t",
            "1004",
            "-d",
            str(arguments.load_seconds),
            "-i",
            str(arguments.gpu_index),
        ]
    )
    emit(
        {
            "run_id": run_id,
            "gpu_index": arguments.gpu_index,
            "power_limit_w": target,
            "default_power_limit_w": default,
            "load_seconds": arguments.load_seconds,
            "load_unit": f"{load_unit}.service",
            "restore_unit": f"{unit}.timer",
            "restore_seconds": arguments.restore_seconds,
            "gpu_power": gpu_power_state(),
        }
    )


def restore_gpu_power_limit(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id, "run ID")
    unit = power_limit_restore_unit(run_id)
    load_unit = f"{unit}-load"
    run(["systemctl", "stop", f"{load_unit}.service"], check=False)
    run(["systemctl", "stop", f"{unit}.timer"], check=False)
    for name in (f"{unit}.service", f"{load_unit}.service"):
        run(["systemctl", "reset-failed", name], check=False)
    state = gpu_power_state()
    defaults = {item["power_default_limit_w"] for item in state}
    if len(defaults) != 1 or None in defaults:
        raise ProbeError("node does not report a single default GPU power limit")
    default = int(next(iter(defaults)) or 0)
    run(["nvidia-smi", "-pl", str(default)])
    after = gpu_power_state()
    if any(item["power_limit_w"] != float(default) for item in after):
        raise ProbeError("GPU power limits are not back at the driver default")
    emit({"run_id": run_id, "default_power_limit_w": default, "gpu_power": after})


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
            "service": service_snapshot().get(
                "gpu-fault-fabric-manager-collector.service"
            ),
            "fabric_manager_log": file_snapshot(FM_LOG),
            "fabric_manager_cursor": fabric_manager_cursor(),
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
    pod_uid: str,
    *,
    proc: Path = Path("/proc"),
    excluded_pids: Iterable[int] = (),
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
    emit(
        {
            "xid": xid,
            "marker": marker,
            "pci_bdf": bdf,
            "bytes_written": written,
        }
    )


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
    """Where the override remembers what it changed, independent of the backup.

    The deadman timer's service only copies the backup back; it does not
    delete it. The one way to lose the backup before ``restore-collector-env``
    runs is a probe that never wrote it, and then the env must still be
    inspected rather than declared restored. This record survives either way.
    """

    return backup.with_suffix(".override.json")


def override_expected_gpu_count(arguments: argparse.Namespace) -> None:
    backup, unit = collector_restore_paths(arguments.run_id)
    values = parse_env()
    current = int(values["GPU_FAULT_EXPECTED_GPU_COUNT"])
    target = int(arguments.value)
    if target != current + 1:
        raise ProbeError("expected GPU override must equal the live value plus one")
    ACCEPTANCE_STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    backup.write_bytes(COLLECTOR_ENV.read_bytes())
    backup.chmod(0o600)
    record = collector_override_record(backup)
    record.write_text(
        json.dumps(
            {
                "key": "GPU_FAULT_EXPECTED_GPU_COUNT",
                "baseline": current,
                "override": target,
                "backup_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    record.chmod(0o600)
    service = SYSTEMD_UNIT_DIR / f"{unit}.service"
    timer = SYSTEMD_UNIT_DIR / f"{unit}.timer"
    service.write_text(
        "\n".join(
            [
                "[Unit]",
                "Description=Restore gpu-fault collector env after acceptance",
                "[Service]",
                "Type=oneshot",
                f"ExecStart=/bin/cp {backup} {COLLECTOR_ENV}",
                f"ExecStart=/bin/systemctl restart {HOST_COLLECTOR_UNIT}",
                # Enabled (not started) below: a oneshot wanted by
                # multi-user.target runs once at the next boot, so the
                # RESTART_NODE this case provokes comes back with the
                # original env. `Persistent=` cannot do this on a monotonic
                # timer, and `OnBootSec=1` on a timer enabled hours after boot
                # is already elapsed, so it fired the restore within a second
                # of the override (COLLECT-004 attempt 1, 2026-09-11).
                "[Install]",
                "WantedBy=multi-user.target",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    timer.write_text(
        "\n".join(
            [
                "[Unit]",
                "Description=Timed gpu-fault collector env restore",
                "[Timer]",
                f"OnActiveSec={arguments.restore_seconds}s",
                f"Unit={unit}.service",
                "[Install]",
                "WantedBy=timers.target",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    text = COLLECTOR_ENV.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(?m)^GPU_FAULT_EXPECTED_GPU_COUNT=.*$",
        f"GPU_FAULT_EXPECTED_GPU_COUNT={target}",
        text,
    )
    if count != 1:
        raise ProbeError("collector env has no unique expected GPU count")
    temporary = COLLECTOR_ENV.with_suffix(".acceptance.tmp")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, COLLECTOR_ENV)
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "enable", f"{unit}.service"])
    run(["systemctl", "enable", "--now", f"{unit}.timer"])
    run(["systemctl", "restart", HOST_COLLECTOR_UNIT])
    emit(
        {
            "run_id": arguments.run_id,
            "baseline": current,
            "override": target,
            "restore_timer": f"{unit}.timer",
            "backup": str(backup),
        }
    )


def restore_collector_env(arguments: argparse.Namespace) -> None:
    """Put the original ``collector.env`` back and say whether that is true.

    ``restored`` used to be an unconditional ``True``: with the backup gone
    (never written, or a probe that died between the two writes) the env kept
    the override and the runner recorded a successful restore. Now the answer
    is read back from the file: the backup's digest when it exists, otherwise
    the override record's baseline value against the live key.
    """

    backup, unit = collector_restore_paths(arguments.run_id)
    record_path = collector_override_record(backup)
    record: dict[str, Any] = {}
    if record_path.is_file():
        try:
            loaded = json.loads(record_path.read_text(encoding="utf-8"))
            record = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            record = {}
    backup_present = backup.is_file()
    if backup_present:
        COLLECTOR_ENV.write_bytes(backup.read_bytes())
        run(["systemctl", "restart", HOST_COLLECTOR_UNIT])
    run(["systemctl", "disable", "--now", f"{unit}.timer"], check=False)
    run(["systemctl", "disable", f"{unit}.service"], check=False)
    for suffix in (".timer", ".service"):
        (SYSTEMD_UNIT_DIR / (unit + suffix)).unlink(missing_ok=True)
    run(["systemctl", "daemon-reload"])
    env_after = parse_env()
    file_after = file_snapshot(COLLECTOR_ENV)
    key = str(record.get("key") or "GPU_FAULT_EXPECTED_GPU_COUNT")
    reason: str | None = None
    if backup_present:
        expected_digest = record.get("backup_sha256")
        restored = not expected_digest or (
            file_after is not None and file_after.get("sha256") == expected_digest
        )
        if not restored:
            reason = "collector.env digest differs from the backup"
    elif record:
        restored = env_after.get(key) == str(record.get("baseline"))
        if not restored:
            reason = f"backup missing and {key} still carries the override"
    else:
        restored = False
        reason = "no backup and no override record for this run ID"
    if restored:
        backup.unlink(missing_ok=True)
        record_path.unlink(missing_ok=True)
    emit(
        {
            "run_id": arguments.run_id,
            "restored": restored,
            "backup_present": backup_present,
            "reason": reason,
            "override_record": record or None,
            "collector_env": env_after,
            "collector_env_file": file_after,
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

    show = commands.add_parser("snapshot")
    show.set_defaults(handler=snapshot)

    cursor = commands.add_parser("fm-cursor")
    cursor.set_defaults(handler=fm_cursor)

    efa = commands.add_parser("efa-inventory")
    efa.set_defaults(handler=efa_inventory_command)

    xid = commands.add_parser("write-xid")
    xid.add_argument("--xid", type=int, choices=sorted(ALLOWED_XIDS), required=True)
    xid.add_argument("--marker", required=True)
    xid.add_argument("--pci-bdf", required=True)
    xid.add_argument("--pid", type=int, default=931000)
    xid.add_argument("--message", default="regional collector acceptance")
    xid.set_defaults(handler=write_xid)

    kill = commands.add_parser("kill-workload")
    kill.add_argument("--pod-uid", required=True)
    kill.set_defaults(handler=kill_workload)

    sxid = commands.add_parser("append-sxid")
    sxid.add_argument("--sxid", type=int, choices=sorted(ALLOWED_SXIDS), required=True)
    sxid.add_argument("--marker", required=True)
    sxid.add_argument("--pci-bdf", required=True)
    sxid.add_argument("--classification", default="Fatal")
    sxid.add_argument("--message", default="regional collector acceptance")
    sxid.add_argument("--switch", type=int, default=0)
    sxid.add_argument("--port", type=int, default=12)
    sxid.add_argument("--tid", type=int, default=990001)
    sxid.add_argument("--include-switch", action=argparse.BooleanOptionalAction)
    sxid.set_defaults(handler=append_sxid)

    restart = commands.add_parser("restart-service")
    restart.add_argument("--service", choices=sorted(ALLOWED_SERVICES), required=True)
    restart.set_defaults(handler=restart_service)

    persistence = commands.add_parser("set-persistence-mode")
    persistence.add_argument("--enabled", choices=("true", "false"), required=True)
    persistence.set_defaults(handler=set_persistence_mode)

    throttle = commands.add_parser("throttle-gpu")
    throttle.add_argument("--run-id", required=True)
    throttle.add_argument("--gpu-index", type=int, default=0)
    throttle.add_argument("--load-seconds", type=int, default=180)
    throttle.add_argument("--restore-seconds", type=int, default=600)
    throttle.set_defaults(handler=throttle_gpu)

    restore_power = commands.add_parser("restore-gpu-power-limit")
    restore_power.add_argument("--run-id", required=True)
    restore_power.set_defaults(handler=restore_gpu_power_limit)

    override = commands.add_parser("override-expected-gpu-count")
    override.add_argument("--run-id", required=True)
    override.add_argument("--value", type=int, required=True)
    override.add_argument("--restore-seconds", type=int, default=600)
    override.set_defaults(handler=override_expected_gpu_count)

    restore_env = commands.add_parser("restore-collector-env")
    restore_env.add_argument("--run-id", required=True)
    restore_env.set_defaults(handler=restore_collector_env)

    unbind = commands.add_parser("unbind-efa")
    unbind.add_argument("--run-id", required=True)
    unbind.add_argument("--pci-bdf", required=True)
    unbind.add_argument("--restore-seconds", type=int, default=300)
    unbind.set_defaults(handler=unbind_efa)

    restore = commands.add_parser("restore-efa")
    restore.add_argument("--run-id", required=True)
    restore.add_argument("--pci-bdf", required=True)
    restore.set_defaults(handler=restore_efa)
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        restore_seconds = getattr(arguments, "restore_seconds", 300)
        if not 60 <= restore_seconds <= 900:
            raise ProbeError("restore seconds is outside 60..900")
        arguments.handler(arguments)
    except Exception as exc:
        emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
