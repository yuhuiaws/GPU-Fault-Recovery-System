#!/usr/bin/env python3
"""Owned, nonrenewable service windows; the copied helper uses only stdlib."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ALLOWED_SERVICES = {"gpu-fault-node-agent.service", "kubelet.service"}
ROOT = Path("/var/lib/gpu-fault-acceptance/service-window")
SYSTEMD = Path("/etc/systemd/system")
PYTHON = Path("/usr/bin/python3")
AGENT_ENV = Path("/etc/gpu-fault/node-agent.env")
CURRENT = Path("/opt/gpu-fault/current")
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
MACHINE_ID = Path("/etc/machine-id")
PRODUCT_ID = Path("/sys/class/dmi/id/product_uuid")
SELF_CGROUP = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")
ACK_SECONDS = 10
ARM_SECONDS = 30
# The independent watcher takes the journal lock once a second for its systemd
# reads; a request landing inside a tick waits this long before it fails closed.
LOCK_POLL_SECONDS = 0.05
LOCK_ATTEMPTS = 300
JOB_SECONDS = 45
RECOVERY_SECONDS = 180
MAX_WINDOW = 900
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
HEX = re.compile(r"[0-9a-f]{64}")
PIN_KEYS = {
    "cluster_id": "GPU_FAULT_NODE_CLUSTER_ID",
    "node": "NODE_NAME",
    "node_uid": "GPU_FAULT_NODE_INSTANCE_ID",
    "artifact_sha256": "GPU_FAULT_NODE_ARTIFACT_SHA256",
    "bundle_sha256": "GPU_FAULT_NODE_INSTALLER_BUNDLE_SHA256",
    "profile_version": "GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION",
}
BINDING_KEYS = {
    "case_id",
    "run_id",
    "owner",
    "cluster_id",
    "node",
    "node_uid",
    "release_id",
    "plan_sha256",
    "helper_sha256",
    "baseline_sha256",
    "host_sha256",
    "boot_id",
    "service",
    "restore_at",
    "expires_at",
    "stop_delay_seconds",
}
UNIT_FIELDS = {
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "FragmentPath",
    "DropInPaths",
    "UnitFileState",
    "Transient",
    "Job",
    "NeedDaemonReload",
    "Description",
}
SERVICE_FIELDS = {
    "MainPID",
    "ControlPID",
    "InvocationID",
    "ExecMainStartTimestampMonotonic",
    "ExecStart",
    "EnvironmentFiles",
    "Environment",
    "User",
    "Group",
    "Type",
    "JobTimeoutUSec",
    "TimeoutStartUSec",
    "TimeoutStopUSec",
    "ControlGroup",
    "KillMode",
    "SendSIGKILL",
}
STATE_FIELDS = {
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "ControlPID",
    "InvocationID",
    "ExecMainStartTimestampMonotonic",
    "Job",
}
SEALED = {"RESTORING", "RESTORED", "CLOSING", "CLOSED", "EXPIRED"}
PHASES = SEALED | {"PREPARING", "INSTALLED", "ARMED", "SCHEDULED", "STOPPING"}


class ProbeError(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise ProbeError(f"unsafe {label}")
    return value


def service_name(value: str) -> str:
    if value not in ALLOWED_SERVICES:
        raise ProbeError("service is not in the warm-spare probe allowlist")
    return value


def binding_key(binding: dict[str, Any]) -> str:
    if not isinstance(binding, dict) or set(binding) != BINDING_KEYS:
        raise ProbeError("durable service binding is incomplete")
    for key in BINDING_KEYS - {"restore_at", "expires_at", "stop_delay_seconds"}:
        safe_id(binding[key], key)
    service_name(binding["service"])
    if binding["case_id"] != "GF-REGIONAL-DESTR-008":
        raise ProbeError("service window is not a DESTR008 fixture")
    for key in ("plan_sha256", "helper_sha256", "baseline_sha256", "host_sha256"):
        if not HEX.fullmatch(binding[key]):
            raise ProbeError("service binding digest is invalid")
    if not re.fullmatch(r"[0-9a-f]{32}", binding["owner"]):
        raise ProbeError("service binding owner is invalid")
    if any(
        type(binding[k]) is not int
        for k in ("restore_at", "expires_at", "stop_delay_seconds")
    ):
        raise ProbeError("service deadline must be an integer")
    if (
        binding["restore_at"] <= 0
        or binding["expires_at"] - binding["restore_at"] != RECOVERY_SECONDS
        or not 0 <= binding["stop_delay_seconds"] <= 120
    ):
        raise ProbeError("service recovery interval is not bounded")
    return digest(binding)


def require_directory(path: Path, *, private: bool = False) -> None:
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & (0o077 if private else 0o022)
    ):
        raise ProbeError("service window directory ownership is invalid")


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def private_read(path: Path) -> dict[str, Any]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
            or info.st_size > 1024 * 1024
        ):
            raise ProbeError("service window record is not private")
        with os.fdopen(fd) as stream:
            fd = -1
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ProbeError("service window record is malformed")
        return value
    finally:
        if fd >= 0:
            os.close(fd)


def write_record(path: Path, value: dict[str, Any]) -> None:
    require_directory(path.parent, private=True)
    if path.exists() or path.is_symlink():
        private_read(path)
    fd, name = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def file_identity(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if info.st_uid != os.geteuid():
        raise ProbeError("service window file owner changed")
    identity: dict[str, Any] = {"dev": info.st_dev, "ino": info.st_ino}
    if stat.S_ISLNK(info.st_mode):
        return {**identity, "link": os.readlink(path)}
    # Writable by the caller's own group is still the caller's privilege (this
    # EKS AMI ships /usr/bin/kubelet as 0775 root:root, live 2026-09-19); any
    # other write bit lets a different principal change the service.
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_mode & 0o002
        or (info.st_mode & 0o020 and info.st_gid != os.getegid())
    ):
        raise ProbeError("service window file is not a protected regular file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ProbeError("service window file changed while opening")
        # The independent units run under the host's /usr/bin/python3 (3.9 on
        # Amazon Linux 2023); hashlib.file_digest only exists from 3.11.
        digest_state = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest_state.update(chunk)
        checksum = digest_state.hexdigest()
        after = os.fstat(stream.fileno())
        if any(
            getattr(after, key) != getattr(opened, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise ProbeError("service window file changed while reading")
    return {
        **identity,
        "sha256": checksum,
        "mtime_ns": info.st_mtime_ns,
        "mode": stat.S_IMODE(info.st_mode),
    }


def matches(path: Path, identity: dict[str, Any]) -> bool:
    try:
        return file_identity(path) == identity
    except FileNotFoundError:
        return False


def publish(source: Path, target: Path, identity: dict[str, Any]) -> None:
    if not matches(source, identity):
        raise ProbeError("service window publication source changed")
    # Never replace an existing destination, including one with identical bytes.
    os.link(source, target, follow_symlinks=False)
    sync_directory(target.parent)


def remove_owned(path: Path, identity: dict[str, Any]) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not matches(path, identity):
        raise ProbeError("service window resource was replaced")
    path.unlink()
    sync_directory(path.parent)


def systemctl(*args: str) -> dict[str, str]:
    result = subprocess.run(
        ["/bin/systemctl", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"},
    )
    values: dict[str, str] = {}
    if args[0] == "show":
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if not separator or key in values:
                raise ProbeError("systemd returned ambiguous properties")
            values[key] = value
    if result.returncode and not (
        args[0] == "show"
        and result.returncode == 4
        and values.get("LoadState") == "not-found"
    ):
        raise ProbeError("service window systemctl failed; output withheld")
    return values


def job_id(unit: dict[str, str]) -> int:
    value = unit["Job"]
    if value in {"", "0", "0 /"}:
        return 0
    match = re.fullmatch(r"([1-9][0-9]*)(?: /org/freedesktop/systemd1/job/\1)?", value)
    if match is None:
        raise ProbeError("systemd job identity is unknown")
    return int(match[1])


IDENTITY_FIELDS = {"Id", "LoadState", "ActiveState", "SubState", "Job"}


def unit_state(name: str) -> dict[str, str]:
    fields = UNIT_FIELDS | (SERVICE_FIELDS if name.endswith(".service") else set())
    value = systemctl("show", name, "--property=" + ",".join(sorted(fields)))
    if not IDENTITY_FIELDS <= value.keys() or value["Id"] != name:
        raise ProbeError("systemd unit identity or properties are incomplete")
    # systemd 252 (Amazon Linux 2023, live 2026-09-19) prints no line for an
    # empty list or exec property -- EnvironmentFiles= of a unit without one,
    # everything of a unit it never loaded -- so an omitted property is empty.
    value = {**{field: "" for field in fields}, **value}
    job_id(value)
    return value


def service_snapshot(service: str) -> dict[str, str]:
    value = unit_state(service_name(service))
    return {key: value[key] for key in STATE_FIELDS}


def running(unit: dict[str, str]) -> bool:
    return (
        unit["LoadState"] == "loaded"
        and unit["ActiveState"] == "active"
        and unit["SubState"] == "running"
        and job_id(unit) == 0
        and unit["MainPID"].isdigit()
        and int(unit["MainPID"]) > 0
        and unit["ControlPID"] == "0"
        and re.fullmatch(r"[0-9a-f]{32}", unit["InvocationID"]) is not None
    )


def exec_start(value: str) -> tuple[Path, list[str]]:
    match = re.fullmatch(
        r"\{ path=([^ ;{}]+) ; argv\[\]=([^{}]+?) ; ignore_errors=no ; [^{}]*\}", value
    )
    if match is None:
        raise ProbeError("service ExecStart is not a single literal command")
    path = Path(match[1])
    argv = shlex.split(match[2])
    if not path.is_absolute() or not argv or argv[0] != str(path):
        raise ProbeError("service executable identity is unknown")
    return path, argv


def boot_id() -> str:
    return safe_id(BOOT_ID.read_text().strip(), "boot identity")


def boot_seconds() -> float:
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def definition(service: str, dropin: Path | None = None) -> dict[str, Any]:
    unit = unit_state(service_name(service))
    if (
        unit["LoadState"] != "loaded"
        or unit["Transient"] != "no"
        or unit["NeedDaemonReload"] != "no"
        or unit["UnitFileState"] not in {"enabled", "disabled", "static"}
    ):
        raise ProbeError("service definition is not stable and loaded")
    fragment = Path(unit["FragmentPath"])
    if not fragment.is_absolute() or fragment.name != service:
        raise ProbeError("service fragment identity is unknown")
    dropins = [Path(p) for p in shlex.split(unit["DropInPaths"])]
    if len(set(dropins)) != len(dropins) or any(not p.is_absolute() for p in dropins):
        raise ProbeError("service drop-in identity is ambiguous")
    executable, argv = exec_start(unit["ExecStart"])
    paths = [
        fragment,
        AGENT_ENV,
        executable,
        PYTHON,
        CURRENT,
        CURRENT / "venv/bin/gpu-fault-node-agent",
    ]
    paths.extend(p for p in dropins if p != dropin)
    environment_files = unit["EnvironmentFiles"]
    pattern = r"(/\S+) \(ignore_errors=(?:yes|no)\)"
    env_paths = re.findall(pattern, environment_files)
    if (
        " ".join(re.findall(r"/\S+ \(ignore_errors=(?:yes|no)\)", environment_files))
        != environment_files
    ):
        raise ProbeError("service environment-file identity is unknown")
    paths.extend(Path(p) for p in env_paths)
    files: dict[str, Any] = {}
    for path in paths:
        files[str(path)] = file_identity(path)
        resolved = path.resolve(strict=True)
        if resolved.is_dir():
            require_directory(resolved)
        else:
            files[str(resolved)] = file_identity(resolved)
    values: dict[str, str] = {}
    for line in AGENT_ENV.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if not words:
            continue
        if len(words) != 1 or "=" not in words[0]:
            raise ProbeError("Node Agent identity assignments are not literal")
        name, value = words[0].split("=", 1)
        if name in values:
            raise ProbeError("Node Agent identity assignments are duplicated")
        values[name] = value
    pins = {key: safe_id(values.get(env, ""), key) for key, env in PIN_KEYS.items()}
    machine = [MACHINE_ID.read_text().strip(), PRODUCT_ID.read_text().strip()]
    if any(v.lower() in {"", "none", "unknown", "uninitialized"} for v in machine):
        raise ProbeError("host incarnation identity is unavailable")
    return {
        "host_sha256": digest(machine),
        "boot_id": boot_id(),
        "pins": pins,
        "files": files,
        "unit_file_state": unit["UnitFileState"],
        "configuration_sha256": digest(
            {
                **{
                    k: unit[k]
                    for k in (
                        "EnvironmentFiles",
                        "Environment",
                        "User",
                        "Group",
                        "Type",
                    )
                },
                "executable": str(executable),
                "argv": argv,
            }
        ),
    }


def capture(service: str) -> dict[str, Any]:
    baseline = definition(service)
    state = service_snapshot(service)
    if not running(state):
        raise ProbeError("service baseline must be active and idle")
    return {
        "service": service,
        "host_sha256": baseline["host_sha256"],
        "boot_id": baseline["boot_id"],
        "pins": baseline["pins"],
        "baseline_sha256": digest({"definition": baseline, "state": state}),
        "state": state,
    }


def cgroup_empty(unit: dict[str, str]) -> bool:
    group = unit["ControlGroup"]
    if not group:
        return True
    if group != f"/system.slice/{unit['Id']}":
        raise ProbeError("owned service cgroup identity is unknown")
    values: dict[str, str] = {}
    for line in (
        (CGROUP_ROOT / group.lstrip("/") / "cgroup.events").read_text().splitlines()
    ):
        fields = line.split()
        if len(fields) != 2 or fields[0] in values:
            raise ProbeError("owned service cgroup read is malformed")
        values[fields[0]] = fields[1]
    if values.get("populated") not in {"0", "1"}:
        raise ProbeError("owned service cgroup population is unknown")
    return values["populated"] == "0"


def idle(unit: dict[str, str]) -> bool:
    if (
        unit["ActiveState"] not in {"inactive", "failed"}
        or job_id(unit)
        or unit["SubState"] not in {"dead", "failed"}
    ):
        return False
    if unit["Id"].endswith(".service"):
        return unit["MainPID"] == unit["ControlPID"] == "0" and cgroup_empty(unit)
    return True


def quiet_report(unit: dict[str, str]) -> dict[str, Any]:
    if not idle(unit):
        raise ProbeError("service window process or job is not quiescent")
    return {
        "unit": unit["Id"],
        "load_state": unit["LoadState"],
        "active_state": unit["ActiveState"],
        "job_id": 0,
        "main_pid": 0,
        "control_pid": 0,
        "cgroup_empty": True,
    }


class ServiceWindow:
    def __init__(self, binding: dict[str, Any]) -> None:
        self.key = binding_key(binding)
        self.binding = binding
        self.service = service_name(binding["service"])
        self.directory = ROOT / self.key
        self.state_path = self.directory / "state.json"
        prefix = f"gpu-fault-destr008-{self.key[:32]}"
        self.units = {
            "stop-service": prefix + "-stop.service",
            "stop-timer": prefix + "-stop.timer",
            "restore-service": prefix + "-restore.service",
        }
        self.dropin = SYSTEMD / f"{self.service}.d" / f"90-{prefix}.conf"
        self.targets = {
            "claim": ROOT / f"{self.service}.claim",
            "helper": self.directory / "helper.py",
            "bound": self.dropin,
            **{key: SYSTEMD / name for key, name in self.units.items()},
        }
        self.record: dict[str, Any] = {}

    @contextmanager
    def locked(self) -> Iterator[None]:
        require_directory(ROOT.parent.parent)
        ROOT.parent.mkdir(mode=0o700, exist_ok=True)
        require_directory(ROOT.parent)
        ROOT.mkdir(mode=0o700, exist_ok=True)
        require_directory(ROOT, private=True)
        fd = os.open(
            ROOT / "lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
            ):
                raise ProbeError("service window lock is not private")
            for attempt in range(LOCK_ATTEMPTS):
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if attempt + 1 == LOCK_ATTEMPTS:
                        raise
                    time.sleep(LOCK_POLL_SECONDS)
            self.directory.mkdir(mode=0o700, exist_ok=True)
            require_directory(self.directory, private=True)
            if self.state_path.exists() or self.state_path.is_symlink():
                self.record = private_read(self.state_path)
                self.validate_record()
            elif any(self.directory.iterdir()):
                raise ProbeError("service window journal is missing beside resources")
            else:
                self.record = {}
            yield
        finally:
            os.close(fd)

    def validate_record(self) -> None:
        if (
            self.record.get("schema_version") != 1
            or self.record.get("binding") != self.binding
            or not isinstance(self.record.get("phase"), str)
            or self.record.get("phase") not in PHASES
            or type(self.record.get("stop_requested")) is not bool
            or type(self.record.get("stop_acknowledged")) is not bool
            or type(self.record.get("start_intent")) is not bool
            or type(self.record.get("installed")) is not bool
            or type(self.record.get("retiring")) is not bool
            or not isinstance(self.record.get("resources"), dict)
            or not isinstance(self.record.get("baseline"), dict)
            or not isinstance(self.record.get("before"), dict)
            or not isinstance(self.record.get("clock"), dict)
        ):
            raise ProbeError("service window journal identity or state is invalid")
        for key in ("restore", "expires"):
            value = self.record["clock"].get(key)
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ProbeError("service window elapsed deadline is invalid")
        if (
            self.record["clock"]["expires"] - self.record["clock"]["restore"]
            != RECOVERY_SECONDS
        ):
            raise ProbeError("service window elapsed interval changed")
        if (
            digest(
                {"definition": self.record["baseline"], "state": self.record["before"]}
            )
            != self.binding["baseline_sha256"]
        ):
            raise ProbeError("service window baseline was rewritten")
        for name, resource in self.record["resources"].items():
            if (
                name not in self.targets
                or not isinstance(resource, dict)
                or set(resource) != {"source", "target", "identity"}
                or resource["source"] != str(self.directory / f"source-{name}")
                or resource["target"] != str(self.targets[name])
                or not isinstance(resource["identity"], dict)
            ):
                raise ProbeError("service window publication receipt is invalid")
        if self.record["stop_requested"] and set(self.record["resources"]) != set(
            self.targets
        ):
            raise ProbeError("stop intent has no complete recovery ownership")
        if self.record["stop_acknowledged"]:
            observed = self.record.get("stop_observation")
            if (
                not self.record["stop_requested"]
                or not isinstance(observed, dict)
                or not STATE_FIELDS <= observed.keys()
                or observed["Id"] != self.service
                or any(not isinstance(observed[k], str) for k in STATE_FIELDS)
            ):
                raise ProbeError("stop acknowledgement has no scoped observation")

    def save(self) -> None:
        write_record(self.state_path, self.record)

    def initialize(self) -> None:
        baseline = definition(self.service)
        before = service_snapshot(self.service)
        if (
            not running(before)
            or digest({"definition": baseline, "state": before})
            != self.binding["baseline_sha256"]
            or baseline["host_sha256"] != self.binding["host_sha256"]
            or baseline["boot_id"] != self.binding["boot_id"]
            or any(
                baseline["pins"][k] != self.binding[k]
                for k in ("cluster_id", "node", "node_uid")
            )
        ):
            raise ProbeError("service baseline or host binding changed")
        now = time.time()
        elapsed = boot_seconds()
        self.record = {
            "schema_version": 1,
            "binding": self.binding,
            "phase": "PREPARING",
            "baseline": baseline,
            "before": before,
            "resources": {},
            "stop_requested": False,
            "stop_acknowledged": False,
            "start_intent": False,
            "installed": False,
            "retiring": False,
            "clock": {
                "restore": elapsed + self.binding["restore_at"] - now,
                "expires": elapsed + self.binding["expires_at"] - now,
            },
        }
        self.save()

    def remaining(self, deadline: str) -> float:
        if boot_id() != self.binding["boot_id"]:
            raise ProbeError("service window host boot changed")
        now, elapsed = time.time(), boot_seconds()
        if not math.isfinite(now) or not math.isfinite(elapsed):
            raise ProbeError("service window clock is unavailable")
        return float(
            min(
                self.binding[f"{deadline}_at"] - now,
                self.record["clock"][deadline] - elapsed,
            )
        )

    def verify_target(self) -> None:
        if definition(self.service, self.dropin) != self.record["baseline"]:
            raise ProbeError("service definition, host or runtime was replaced")

    def resource(self, name: str, content: bytes) -> None:
        source = self.directory / f"source-{name}"
        fd = os.open(
            source, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(self.directory)
        identity = file_identity(source)
        self.record["resources"][name] = {
            "source": str(source),
            "target": str(self.targets[name]),
            "identity": identity,
        }
        self.save()
        publish(source, self.targets[name], identity)

    def verify_resources(self, *, missing_ok: bool = False) -> None:
        for name, target in self.targets.items():
            resource = self.record["resources"].get(name)
            present = target.exists() or target.is_symlink()
            if resource is None:
                if present:
                    raise ProbeError("service window resource has no owner receipt")
                continue
            if not matches(Path(resource["source"]), resource["identity"]):
                raise ProbeError("service window ownership source was replaced")
            if (present or not missing_ok) and not matches(
                target, resource["identity"]
            ):
                raise ProbeError("service window resource was replaced or lost")

    def description(self, role: str) -> str:
        return f"gpu-fault service-window {self.key} {role}"

    def own_state(self, role: str, *, missing_ok: bool = False) -> dict[str, str]:
        name = self.units[role]
        unit = unit_state(name)
        resource = self.record["resources"].get(role)
        present = self.targets[role].exists() or self.targets[role].is_symlink()
        if (
            missing_ok
            and unit["LoadState"] == "not-found"
            and idle(unit)
            and (
                not present
                or (
                    resource is not None
                    and matches(self.targets[role], resource["identity"])
                )
            )
        ):
            return unit
        if (
            resource is None
            or (present and not matches(self.targets[role], resource["identity"]))
            or unit["LoadState"] != "loaded"
            or unit["Transient"] != "no"
            or unit["FragmentPath"] != str(self.targets[role])
            or unit["DropInPaths"]
            or unit["NeedDaemonReload"] != "no"
            or unit["Description"] != self.description(role)
            or (not present and (not missing_ok or not idle(unit)))
        ):
            raise ProbeError("service window systemd ownership is unproven")
        if name.endswith(".service"):
            command = "watch" if role == "restore-service" else "stop-owned"
            executable, argv = exec_start(unit["ExecStart"])
            if (
                executable != PYTHON
                or argv
                != [
                    str(PYTHON),
                    "-I",
                    "-S",
                    "-B",
                    str(self.targets["helper"]),
                    command,
                    "--key",
                    self.key,
                ]
                or unit["KillMode"] != "control-group"
                or unit["SendSIGKILL"] != "yes"
                or unit["EnvironmentFiles"]
                or unit["Environment"]
                or unit["User"] not in {"", "root"}
                or unit["Group"] not in {"", "root"}
            ):
                raise ProbeError("service window process definition changed")
        return unit

    def verify_bounds(self) -> None:
        unit = unit_state(self.service)
        if (
            unit["JobTimeoutUSec"] != "45s"
            or unit["TimeoutStartUSec"] != "30s"
            or unit["TimeoutStopUSec"] != "30s"
        ):
            raise ProbeError("target service job is not bounded")

    def prepare(self) -> dict[str, Any]:
        with self.locked():
            if self.record:
                raise ProbeError("service window already exists; cleanup only")
            now = time.time()
            if (
                not now + JOB_SECONDS
                < self.binding["restore_at"]
                < self.binding["expires_at"]
                <= now + MAX_WINDOW
            ):
                raise ProbeError("service window is expired or too long")
            self.initialize()
            require_directory(SYSTEMD)
            self.verify_resources(missing_ok=True)
            for name in self.units.values():
                state = unit_state(name)
                if state["LoadState"] != "not-found" or not idle(state):
                    raise ProbeError("pre-existing service window unit is not owned")
            self.resource("claim", self.key.encode())
            script = Path(__file__).read_bytes()
            if hashlib.sha256(script).hexdigest() != self.binding["helper_sha256"]:
                raise ProbeError("service window helper source changed")
            self.resource("helper", script)
            self.dropin.parent.mkdir(mode=0o755, exist_ok=True)
            require_directory(self.dropin.parent)
            self.resource(
                "bound",
                b"[Unit]\nJobTimeoutSec=45s\n[Service]\nTimeoutStartSec=30s\nTimeoutStopSec=30s\n",
            )
            for role, command in (
                ("stop-service", "stop-owned"),
                ("restore-service", "watch"),
            ):
                lifetime = 30 if role == "stop-service" else MAX_WINDOW + 30
                self.resource(
                    role,
                    (
                        f"[Unit]\nDescription={self.description(role)}\n"
                        "JobTimeoutSec=30s\n[Service]\nType=exec\n"
                        f"ExecStart={PYTHON} -I -S -B {self.targets['helper']} {command} --key {self.key}\n"
                        f"RuntimeMaxSec={lifetime}s\nTimeoutStartSec=20s\nTimeoutStopSec=5s\n"
                        "KillMode=control-group\nSendSIGKILL=yes\nRestart=no\n"
                        "UMask=0077\nStandardOutput=null\nStandardError=null\n"
                    ).encode(),
                )
            self.resource(
                "stop-timer",
                (
                    f"[Unit]\nDescription={self.description('stop-timer')}\n"
                    "[Timer]\n"
                    f"OnActiveSec={max(1, self.binding['stop_delay_seconds'])}s\n"
                    "AccuracySec=1us\nRandomizedDelaySec=0\n"
                    f"Unit={self.units['stop-service']}\n"
                ).encode(),
            )
            self.record.update(phase="INSTALLED", installed=True)
            self.save()
            systemctl("daemon-reload")
            self.verify_target()
            self.verify_resources()
            self.verify_bounds()
            systemctl(
                "start", "--no-block", "--job-mode=fail", self.units["restore-service"]
            )
            return self.report()

    def independent_identity(self, role: str) -> dict[str, Any]:
        unit = self.own_state(role)
        invocation = os.environ.get("INVOCATION_ID")
        groups = [line.split(":", 2) for line in SELF_CGROUP.read_text().splitlines()]
        if (
            not invocation
            or invocation != unit["InvocationID"]
            or unit["ActiveState"] != "active"
            or unit["SubState"] != "running"
            or unit["MainPID"] != str(os.getpid())
            or not any(
                g == ["0", "", f"/system.slice/{self.units[role]}"] for g in groups
            )
        ):
            raise ProbeError("independent service invocation did not prove ownership")
        return {
            "invocation_id": invocation,
            "pid": os.getpid(),
            "boot_id": boot_id(),
            "at": time.time(),
            "elapsed": boot_seconds(),
        }

    def verify_ack(self, *, scheduling: bool = False) -> None:
        ack = self.record.get("ack")
        if not isinstance(ack, dict) or any(
            type(ack.get(k)) not in {int, float} or not math.isfinite(ack[k])
            for k in ("at", "elapsed")
        ):
            raise ProbeError("independent service arm ACK is missing")
        unit = self.own_state("restore-service")
        if (
            ack.get("boot_id") != self.binding["boot_id"]
            or not 0 <= time.time() - ack["at"] <= ACK_SECONDS
            or not 0 <= boot_seconds() - ack["elapsed"] <= ACK_SECONDS
            or not running(unit)
            or unit["InvocationID"] != ack.get("invocation_id")
            or unit["MainPID"] != str(ack.get("pid"))
            or self.remaining("restore")
            <= (self.binding["stop_delay_seconds"] if scheduling else 0) + JOB_SECONDS
        ):
            raise ProbeError("independent service arm ACK is stale or replaced")

    def schedule_stop(self) -> dict[str, Any]:
        with self.locked():
            if self.record.get("phase") != "ARMED":
                raise ProbeError("stop requires an independently ARMED window")
            self.verify_target()
            self.verify_resources()
            self.verify_bounds()
            self.verify_ack(scheduling=True)
            if service_snapshot(self.service) != self.record["before"]:
                raise ProbeError("service baseline invocation changed before stop")
            self.own_state("stop-timer")
            self.own_state("stop-service")
            self.record.update(
                phase="SCHEDULED",
                stop_not_before=boot_seconds()
                + max(1, self.binding["stop_delay_seconds"]),
            )
            self.save()
            self.verify_ack(scheduling=True)
            systemctl(
                "start", "--no-block", "--job-mode=fail", self.units["stop-timer"]
            )
            return self.report()

    def stop_owned(self) -> dict[str, Any]:
        with self.locked():
            if not self.record:
                raise ProbeError("owned stop journal is missing")
            if self.record["phase"] in SEALED:
                return self.report()
            if self.record["phase"] != "SCHEDULED":
                raise ProbeError("owned stop is not scheduled")
            self.verify_target()
            self.verify_resources()
            self.verify_bounds()
            self.independent_identity("stop-service")
            self.verify_ack()
            if boot_seconds() < self.record["stop_not_before"]:
                raise ProbeError("owned stop fired before its delay")
            if service_snapshot(self.service) != self.record["before"]:
                raise ProbeError("service invocation changed before owned stop")
            self.record.update(phase="STOPPING", stop_requested=True)
            self.save()
            self.verify_ack()
            systemctl("stop", "--no-block", "--job-mode=fail", self.service)
            self.record["stop_observation"] = service_snapshot(self.service)
            self.record["stop_acknowledged"] = True
            self.save()
            return self.report()

    def quiesce(self, role: str) -> dict[str, Any]:
        unit = self.own_state(role, missing_ok=True)
        if not idle(unit):
            if not self.targets[role].exists():
                raise ProbeError("cannot stop a unit without its owned publication")
            systemctl("stop", self.units[role])
            unit = self.own_state(role)
        return quiet_report(unit)

    def wait_target(
        self, unit: dict[str, str], *, start_job: int | None = None
    ) -> dict[str, str]:
        deadline = boot_seconds() + JOB_SECONDS + 5
        while job_id(unit) or unit["ActiveState"] in {"activating", "deactivating"}:
            if start_job is not None and job_id(unit) not in {0, start_job}:
                raise ProbeError("target service job was replaced")
            if boot_seconds() >= deadline or self.remaining("expires") <= 0:
                raise ProbeError("target service job did not become quiescent")
            time.sleep(1)
            self.verify_target()
            unit = service_snapshot(self.service)
        if unit["ControlPID"] != "0":
            raise ProbeError("target service still has a control process")
        return unit

    def restore_locked(self, reason: str) -> None:
        self.verify_target()
        self.verify_resources(
            missing_ok=(
                not self.record["installed"]
                or self.record["retiring"]
                or self.record["phase"] == "CLOSED"
            )
        )
        if self.record["phase"] != "CLOSED":
            self.record.update(
                phase="RESTORING",
                restore_reason=self.record.get("restore_reason", reason),
            )
            self.save()  # Irreversibly fence late timer/stop requests before cancellation.
        quiet = {
            "timer": self.quiesce("stop-timer"),
            "service": self.quiesce("stop-service"),
        }
        if self.record["stop_requested"] and not self.record["stop_acknowledged"]:
            raise ProbeError(
                "stop submission has no durable ACK; recovery is unresolved"
            )
        unit = self.wait_target(service_snapshot(self.service))
        known = {self.record["before"]["InvocationID"]}
        receipt = self.record.get("start_receipt")
        if isinstance(receipt, dict) and receipt.get("invocation_id"):
            known.add(receipt["invocation_id"])
        if running(unit):
            if unit["InvocationID"] not in known:
                raise ProbeError(
                    "active service invocation has another or unknown owner"
                )
        else:
            if (
                not self.record["stop_requested"]
                or self.record["start_intent"]
                or unit["ActiveState"] not in {"inactive", "failed"}
                or unit["SubState"] not in {"dead", "failed"}
                or unit["MainPID"] != "0"
                or unit["ControlPID"] != "0"
                or unit["InvocationID"]
                not in {"", self.record["before"]["InvocationID"]}
            ):
                raise ProbeError("target service is not an owned stopped baseline")
            self.verify_bounds()
            if self.remaining("expires") <= JOB_SECONDS + 25:
                self.record["phase"] = "EXPIRED"
                self.save()
                raise ProbeError("service restoration authority expired")
            self.record["start_intent"] = True
            self.save()
            self.verify_target()
            if self.remaining("expires") <= JOB_SECONDS + 20:
                raise ProbeError("service restoration deadline elapsed during intent")
            if service_snapshot(self.service) != unit:
                raise ProbeError("target service changed before restoration")
            systemctl("start", "--no-block", "--job-mode=fail", self.service)
            accepted = service_snapshot(self.service)
            # A lost start ACK without an invocation receipt is not adoptable.
            self.record["start_receipt"] = {
                "job_id": job_id(accepted),
                "invocation_id": accepted["InvocationID"]
                if accepted["InvocationID"] not in known
                else "",
            }
            self.save()
            unit = self.wait_target(accepted, start_job=job_id(accepted))
            if not running(unit) or unit["InvocationID"] in known:
                raise ProbeError("owned service start did not restore availability")
            receipt = self.record["start_receipt"]
            if (
                receipt["invocation_id"]
                and unit["InvocationID"] != receipt["invocation_id"]
            ):
                raise ProbeError("service invocation changed during restoration")
            receipt["invocation_id"] = unit["InvocationID"]
        self.verify_target()
        if self.record["phase"] != "CLOSED":
            self.record.update(
                phase="RESTORED",
                after=unit,
                stop_quiescence=quiet,
                restored_at=time.time(),
            )
            self.save()

    def restore(self) -> dict[str, Any]:
        with self.locked():
            if not self.record:
                raise ProbeError("service restore requires its durable owner record")
            self.restore_locked("controller")
            return self.report()

    def cleanup(self) -> dict[str, Any]:
        with self.locked():
            if not self.record:
                # A prepare request may have been delayed before acquiring the lock.
                # Fence it only when the original baseline and explicit absence agree.
                if any(p.exists() or p.is_symlink() for p in self.targets.values()):
                    raise ProbeError(
                        "missing journal cannot authorize resource cleanup"
                    )
                for name in self.units.values():
                    unit = unit_state(name)
                    if unit["LoadState"] != "not-found" or not idle(unit):
                        raise ProbeError(
                            "missing journal cannot authorize systemd cleanup"
                        )
                self.initialize()
            self.restore_locked("cleanup")
            was_closed = self.record["phase"] == "CLOSED"
            if not was_closed:
                self.record.update(phase="CLOSING", retiring=True)
                self.save()
            self.record["restore_quiescence"] = self.quiesce("restore-service")
            self.verify_target()
            self.verify_resources(missing_ok=True)
            for name in reversed(tuple(self.record["resources"])):
                if name not in {"claim", "helper"}:
                    resource = self.record["resources"][name]
                    remove_owned(self.targets[name], resource["identity"])
            if not was_closed:
                systemctl("daemon-reload")
            for name in self.units.values():
                unit = unit_state(name)
                if unit["LoadState"] != "not-found" or not idle(unit):
                    raise ProbeError("service window unit retirement is incomplete")
            self.verify_target()
            after = service_snapshot(self.service)
            if (
                not running(after)
                or after["InvocationID"] != self.record["after"]["InvocationID"]
            ):
                raise ProbeError("service restoration proof was lost during cleanup")
            self.record["phase"] = "CLOSED"
            self.save()
            claim = self.record["resources"].get("claim")
            if claim is not None:
                remove_owned(self.targets["claim"], claim["identity"])
            return self.report()

    def tick(self) -> dict[str, Any]:
        with self.locked():
            if not self.record:
                raise ProbeError("independent restore journal is missing")
            if self.record["phase"] in {"RESTORED", "CLOSING", "CLOSED", "EXPIRED"}:
                return self.report()
            self.verify_target()
            self.verify_resources()
            self.verify_bounds()
            self.record["ack"] = self.independent_identity("restore-service")
            if self.record["phase"] == "INSTALLED":
                self.record["phase"] = "ARMED"
            self.save()
            if self.remaining("restore") <= 0 or self.record["phase"] == "RESTORING":
                self.restore_locked("deadline")
            return self.report()

    def watch(self) -> dict[str, Any]:
        deadline = boot_seconds() + MAX_WINDOW + 5
        while boot_seconds() < deadline:
            try:
                report = self.tick()
            except BlockingIOError:
                time.sleep(0.2)
                continue
            if report["phase"] in {"RESTORED", "CLOSING", "CLOSED", "EXPIRED"}:
                return report
            time.sleep(1)
        raise ProbeError("independent service supervision exceeded its bound")

    def status(self) -> dict[str, Any]:
        with self.locked():
            if not self.record:
                raise ProbeError("service status requires a durable owner record")
            self.verify_target()
            self.verify_resources(
                missing_ok=self.record["phase"] in {"CLOSING", "CLOSED"}
            )
            if self.record["phase"] == "ARMED":
                self.verify_ack(scheduling=True)
            return self.report()

    def report(self) -> dict[str, Any]:
        return {
            "binding_sha256": self.key,
            "phase": self.record["phase"],
            "record_kind": "TOMBSTONE"
            if self.record["phase"] == "CLOSED"
            else "RECOVERY_REQUIRED",
            "service": self.service,
            "restore_at": self.binding["restore_at"],
            "expires_at": self.binding["expires_at"],
            "restore_unit": self.units["restore-service"],
            "stop_unit": self.units["stop-timer"],
            "stop_service": self.units["stop-service"],
            "ack": self.record.get("ack"),
            "before": self.record["before"],
            "after": self.record.get("after"),
            "stop_requested": self.record["stop_requested"],
            "stop_acknowledged": self.record["stop_acknowledged"],
            "start_intent": self.record["start_intent"],
            "stop_quiescence": self.record.get("stop_quiescence"),
            "restore_quiescence": self.record.get("restore_quiescence"),
            "restore_reason": self.record.get("restore_reason"),
        }


def bound_window(arguments: argparse.Namespace) -> ServiceWindow:
    value = getattr(arguments, "binding", None)
    if not isinstance(value, str):
        raise ProbeError(
            "durable binding required; service/run names are not ownership"
        )
    return ServiceWindow(json.loads(value))


def snapshot(arguments: argparse.Namespace) -> None:
    emit(capture(service_name(arguments.service)))


def stop_with_failsafe(arguments: argparse.Namespace) -> None:
    emit(bound_window(arguments).schedule_stop())


def restore_service(arguments: argparse.Namespace) -> None:
    emit(bound_window(arguments).restore())


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "command",
        choices=(
            "snapshot",
            "prepare",
            "status",
            "stop-with-failsafe",
            "restore-service",
            "cleanup",
            "watch",
            "stop-owned",
        ),
    )
    value.add_argument("--binding")
    value.add_argument("--key")
    value.add_argument("--service", choices=sorted(ALLOWED_SERVICES))
    # Recognize the old interface, but never treat its names as authority.
    value.add_argument("--run-id")
    value.add_argument("--restore-seconds", type=int, default=180)
    value.add_argument("--stop-delay-seconds", type=int, default=0)
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if not 60 <= arguments.restore_seconds <= 600:
            raise ProbeError("restore seconds is outside 60..600")
        if not 0 <= arguments.stop_delay_seconds <= 120:
            raise ProbeError("stop delay seconds is outside 0..120")
        if arguments.command == "snapshot":
            snapshot(arguments)
            return 0
        if arguments.command in {"watch", "stop-owned"}:
            if (
                arguments.binding
                or not arguments.key
                or not HEX.fullmatch(arguments.key)
            ):
                raise ProbeError("independent service requires its exact journal key")
            binding = private_read(ROOT / arguments.key / "state.json")["binding"]
            window = ServiceWindow(binding)
            if window.key != arguments.key:
                raise ProbeError("independent service journal key changed")
        else:
            if arguments.key:
                raise ProbeError(
                    "controller cannot substitute a journal key for a binding"
                )
            window = bound_window(arguments)
        method = {
            "stop-with-failsafe": "schedule_stop",
            "restore-service": "restore",
            "stop-owned": "stop_owned",
        }.get(arguments.command, arguments.command)
        emit(getattr(window, method)())
        return 0
    except Exception as exc:
        emit({"error": type(exc).__name__, "recovery_required": True})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
