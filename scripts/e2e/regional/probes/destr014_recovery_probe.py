#!/usr/bin/env python3
"""Persistent, stdlib-only DESTR-014 recovery; never resets or reboots a node."""

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

CASE = "GF-REGIONAL-DESTR-014"
ROOT = Path("/var/lib/gpu-fault-acceptance/destr014-recovery")
SYSTEMD = Path("/etc/systemd/system")
AGENT = "gpu-fault-node-agent.service"
ENABLE_LINK = SYSTEMD / "multi-user.target.wants" / AGENT
AGENT_ENV = Path("/etc/gpu-fault/node-agent.env")
PYTHON = Path("/usr/bin/python3")
CURRENT = Path("/opt/gpu-fault/current")
MAX_WINDOW = 7200
RECOVERY_SECONDS = 180
ACK_SECONDS = 45
START_SECONDS = 45
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
    "release_id",
    "plan_sha256",
    "helper_sha256",
    "boot_id",
    "restore_at",
    "expires_at",
    *PIN_KEYS,
}
TERMINAL = {"RESTORED", "EXPIRED", "CLOSING", "CLOSED"}


class RecoveryError(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def binding_key(binding: dict[str, Any]) -> str:
    if set(binding) != BINDING_KEYS or binding["case_id"] != CASE:
        raise RecoveryError("recovery binding is incomplete")
    for key in BINDING_KEYS - {"restore_at", "expires_at"}:
        if not isinstance(binding[key], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", binding[key]
        ):
            raise RecoveryError("recovery binding identity is invalid")
    for key in ("plan_sha256", "helper_sha256", "artifact_sha256", "bundle_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", binding[key]):
            raise RecoveryError("recovery binding digest is invalid")
    for key in ("restore_at", "expires_at"):
        if type(binding[key]) is not int or binding[key] <= 0:
            raise RecoveryError("recovery deadline is invalid")
    if binding["expires_at"] - binding["restore_at"] != RECOVERY_SECONDS:
        raise RecoveryError("recovery interval is not bounded")
    return digest(binding)


def require_private(path: Path, *, directory: bool = False) -> os.stat_result:
    info = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
        or (not directory and info.st_nlink != 1)
    ):
        raise RecoveryError("recovery state is not private")
    return info


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def require_system_directory(path: Path) -> None:
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o022
    ):
        raise RecoveryError("recovery system directory is not owned")


def atomic_state(path: Path, value: dict[str, Any]) -> None:
    """Fsync both the record and its directory; the directory is already owned."""
    require_private(path.parent, directory=True)
    if path.exists() or path.is_symlink():
        require_private(path)
    fd, name = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    temporary = Path(name)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
        ):
            raise RecoveryError("recovery temporary record is not owned")
        with os.fdopen(fd, "w") as handle:
            fd = -1
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def read_record(path: Path) -> dict[str, Any]:
    require_private(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RecoveryError("recovery record is malformed")
    return value


def file_identity(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if info.st_uid != os.geteuid() or (
        not stat.S_ISLNK(info.st_mode) and info.st_mode & 0o022
    ):
        raise RecoveryError("recovery resource is not owned")
    value: dict[str, Any] = {"dev": info.st_dev, "ino": info.st_ino}
    if stat.S_ISLNK(info.st_mode):
        value["link"] = os.readlink(path)
    elif stat.S_ISREG(info.st_mode):
        value.update(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            mtime_ns=info.st_mtime_ns,
            mode=stat.S_IMODE(info.st_mode),
        )
    else:
        raise RecoveryError("recovery resource is not a regular file or link")
    return value


def same_identity(
    actual: dict[str, Any], expected: dict[str, Any], *, lenient: bool = False
) -> bool:
    """Whether a file is the recorded one.

    ``lenient`` drops the device number: a reboot may enumerate the root
    filesystem under another NVMe device number, while inode, content, mtime,
    mode and link target still identify the recorded file on the same disk.
    """
    if actual == expected:
        return True
    if not lenient or not isinstance(expected, dict):
        return False
    return {k: v for k, v in actual.items() if k != "dev"} == {
        k: v for k, v in expected.items() if k != "dev"
    }


def matches(path: Path, identity: dict[str, Any], *, lenient: bool = False) -> bool:
    try:
        return same_identity(file_identity(path), identity, lenient=lenient)
    except FileNotFoundError:
        return False


def remove_owned(
    path: Path, identity: dict[str, Any], *, lenient: bool = False
) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not matches(path, identity, lenient=lenient):
        raise RecoveryError("recovery resource was replaced; refusing removal")
    path.unlink()
    sync_directory(path.parent)


def publish(
    source: Path, target: Path, identity: dict[str, Any], *, lenient: bool = False
) -> None:
    if not matches(source, identity, lenient=lenient):
        raise RecoveryError("recovery source was replaced")
    try:
        os.link(source, target, follow_symlinks=False)
        sync_directory(target.parent)
    except FileExistsError:
        if not matches(target, identity, lenient=lenient):
            raise RecoveryError("recovery destination has another owner") from None


def systemctl(*args: str) -> dict[str, str]:
    result = subprocess.run(
        ["/bin/systemctl", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"},
    )
    if result.returncode:
        raise RecoveryError("recovery systemctl request failed; output withheld")
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def unit_state(unit: str) -> dict[str, str]:
    return systemctl(
        "show",
        unit,
        "--property=LoadState,ActiveState,SubState,UnitFileState,FragmentPath,"
        "DropInPaths,InvocationID,Job,JobTimeoutUSec,TimeoutStartUSec",
    )


def no_job(unit: dict[str, str]) -> bool:
    return unit.get("Job") in {"", "0", "0 /"}


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def runtime_identity(binding: dict[str, Any], own_dropin: Path) -> dict[str, Any]:
    # Do not return env contents: the same file contains the cluster credential.
    values: dict[str, str] = {}
    for line in AGENT_ENV.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if not words:
            continue
        if len(words) != 1 or "=" not in words[0]:
            raise RecoveryError("Node Agent environment is not a literal assignment")
        name, value = words[0].split("=", 1)
        if name in values:
            raise RecoveryError("Node Agent environment has duplicate assignments")
        values[name] = value
    if any(values.get(env) != binding[key] for key, env in PIN_KEYS.items()):
        raise RecoveryError("Node Agent target or release pins changed")
    unit = unit_state(AGENT)
    fragment = Path(unit.get("FragmentPath", ""))
    if unit.get("LoadState") != "loaded" or fragment != SYSTEMD / AGENT:
        raise RecoveryError("Node Agent unit identity is unavailable")
    if not CURRENT.is_symlink() or not CURRENT.resolve().is_dir():
        raise RecoveryError("Node Agent runtime slot identity is unavailable")
    dropins = [Path(item) for item in shlex.split(unit.get("DropInPaths", ""))]
    paths = [fragment, AGENT_ENV, *[p for p in dropins if p != own_dropin]]
    paths.extend(
        [CURRENT.resolve() / "venv/bin/gpu-fault-node-agent", PYTHON.resolve()]
    )
    files = {}
    for path in paths:
        files[str(path)] = file_identity(path)
        resolved = path.resolve()
        files[str(resolved)] = {
            **file_identity(resolved),
            "ctime_ns": resolved.stat().st_ctime_ns,
        }
    files[str(CURRENT)] = file_identity(CURRENT)
    machine = [
        Path("/etc/machine-id").read_bytes().strip(),
        Path("/sys/class/dmi/id/product_uuid").read_bytes().strip(),
    ]
    if any(
        value.lower() in {b"", b"none", b"unknown", b"uninitialized"}
        for value in machine
    ):
        raise RecoveryError("host incarnation evidence is unavailable")
    return {
        "files": files,
        "machine": hashlib.sha256(b"\0".join(machine)).hexdigest(),
    }


class Recovery:
    def __init__(self, binding: dict[str, Any]) -> None:
        self.binding = binding
        self.key = binding_key(binding)
        self.directory = ROOT / self.key
        self.state_path = self.directory / "state.json"
        self.name = f"gpu-fault-destr014-recovery-{self.key[:32]}"
        self.service = self.name + ".service"
        self.timer = self.name + ".timer"
        self.dropin = SYSTEMD / f"{AGENT}.d" / f"90-{self.name}.conf"
        self.record: dict[str, Any] = {}

    @contextmanager
    def locked(self) -> Iterator[None]:
        require_system_directory(ROOT.parent.parent)
        ROOT.parent.mkdir(mode=0o700, exist_ok=True)
        require_system_directory(ROOT.parent)
        ROOT.mkdir(mode=0o700, exist_ok=True)
        require_private(ROOT, directory=True)
        fd = os.open(
            ROOT / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_nlink != 1
            ):
                raise RecoveryError("recovery lock is not owned")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.directory.mkdir(mode=0o700, exist_ok=True)
            require_private(self.directory, directory=True)
            if self.state_path.exists() or self.state_path.is_symlink():
                self.record = read_record(self.state_path)
                if (
                    self.record.get("binding") != self.binding
                    or self.record.get("schema_version") != 1
                ):
                    raise RecoveryError("recovery journal belongs to another run")
                self.validate_record()
            else:
                self.record = {}
            yield
        finally:
            os.close(fd)

    def save(self) -> None:
        atomic_state(self.state_path, self.record)

    def initialize(self) -> None:
        if not self.record:
            if any(self.directory.iterdir()):
                raise RecoveryError("recovery journal is missing beside residual files")
            self.record = {
                "schema_version": 1,
                "binding": self.binding,
                "phase": "PREPARING",
                "resources": [],
                "disable_started": False,
                "start_requested": False,
            }
            self.validate_record()
            self.save()

    def validate_record(self) -> None:
        targets = {
            self.directory / "recovery.py",
            self.dropin,
            SYSTEMD / self.service,
            SYSTEMD / self.timer,
            SYSTEMD / "timers.target.wants" / self.timer,
        }
        resources = self.record.get("resources")
        if (
            self.record.get("phase")
            not in TERMINAL
            | {
                "PREPARING",
                "INSTALLED",
                "ARMED",
                "DISABLING",
                "DISABLED",
                "RESTORING",
            }
            or type(self.record.get("disable_started")) is not bool
            or type(self.record.get("start_requested")) is not bool
            or not isinstance(resources, list)
            or (
                self.record.get("phase") in {"DISABLING", "DISABLED"}
                and self.record.get("disable_started") is not True
            )
        ):
            raise RecoveryError("recovery journal state is invalid")
        seen: set[str] = set()
        for resource in resources:
            if (
                not isinstance(resource, dict)
                or set(resource) != {"source", "target", "identity"}
                or not isinstance(resource["source"], str)
                or Path(resource["source"]).parent != self.directory
                or not isinstance(resource["target"], str)
                or Path(resource["target"]) not in targets
                or resource["target"] in seen
                or not isinstance(resource["identity"], dict)
            ):
                raise RecoveryError("recovery resource journal is invalid")
            seen.add(resource["target"])
        if any(
            str(target) not in seen and (target.exists() or target.is_symlink())
            for target in targets
        ):
            raise RecoveryError("recovery resource has no publication receipt")
        if self.record["disable_started"] and (
            not isinstance(self.record.get("baseline"), dict)
            or not isinstance(self.record.get("enable_link"), dict)
            or len(resources) != 5
        ):
            raise RecoveryError("disabled Agent has no complete recovery journal")

    def resource(self, name: str, target: Path, content: bytes | str) -> None:
        source = self.directory / name
        if isinstance(content, str):
            os.symlink(content, source)
        else:
            fd = os.open(
                source, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        sync_directory(self.directory)
        identity = file_identity(source)
        self.record["resources"].append(
            {"source": str(source), "target": str(target), "identity": identity}
        )
        self.save()  # Intent precedes publication, including a lost link() ACK.
        publish(source, target, identity)

    @property
    def rebooted(self) -> bool:
        """The host runs a boot other than the one the binding was armed on."""
        return boot_id() != self.binding["boot_id"]

    def same(self, path: Path, identity: dict[str, Any]) -> bool:
        return matches(path, identity, lenient=self.rebooted)

    def enable_link_owner(self) -> str:
        """Who owns the Agent enable link now.

        ``saved``: the inode this recovery preserved. ``installer``: the
        standard installer link (absolute target, any inode) recreated on a
        new boot -- the product's node-installer re-enables the Agent when it
        reinstalls after a reboot. ``missing`` or ``foreign`` otherwise; a
        recreated link on the original boot is always foreign.
        """
        try:
            identity = file_identity(ENABLE_LINK)
        except FileNotFoundError:
            return "missing"
        if same_identity(identity, self.record["enable_link"], lenient=self.rebooted):
            return "saved"
        if self.rebooted and identity.get("link") == str(SYSTEMD / AGENT):
            return "installer"
        return "foreign"

    def verify_resources(self) -> None:
        for resource in self.record["resources"]:
            if not self.same(Path(resource["target"]), resource["identity"]):
                raise RecoveryError("persistent recovery resource changed")

    def verify_runtime(self, *, require_claim: bool = True) -> None:
        if require_claim and not self.same(ROOT / "active", self.record["claim"]):
            raise RecoveryError("recovery claim owner changed")
        if self.rebooted:
            # A new boot cannot present the armed runtime: the installer
            # rewrites the unit, its enable link and the venv on the way back
            # and the baseline ctimes died with the old boot (every tick on the
            # node refused here after the product's reboot). The Agent is
            # judged per file where it matters: saved inode or installer link.
            return
        if runtime_identity(self.binding, self.dropin) != self.record["baseline"]:
            raise RecoveryError("Node Agent or host incarnation changed")

    def reclaim_stale_active(self) -> str | None:
        """Release ``active`` when the recovery holding it can never resume.

        A recovery armed on an earlier boot whose units the installer has since
        replaced (every resource target gone) is dead: nothing on this boot can
        ACK, disable or restore for it, yet its claim fenced every later arm as
        "another owner" (DESTR-014 attempt 8 behind the attempt-4 record). The
        record stays as forensics; only the ``active`` link is released. A claim
        of this boot, or one whose units are still installed, keeps the fence:
        that is a live or half-retired recovery and cleanup-only territory.
        """

        active = ROOT / "active"
        try:
            info = active.lstat()
        except FileNotFoundError:
            return None
        for directory in sorted(ROOT.iterdir()):
            if directory == self.directory or not directory.is_dir():
                continue
            try:
                claim = (directory / "claim").lstat()
            except FileNotFoundError:
                continue
            if (claim.st_dev, claim.st_ino) != (info.st_dev, info.st_ino):
                continue
            record = read_record(directory / "state.json")
            binding = record.get("binding") or {}
            if (
                binding.get("boot_id") == boot_id()
                and record.get("phase") not in TERMINAL
            ):
                raise RecoveryError("recovery destination has another owner")
            for resource in record.get("resources") or []:
                target = Path(str(resource.get("target")))
                if target.exists() or target.is_symlink():
                    raise RecoveryError(
                        "stale recovery still owns host resources; "
                        "manual recovery required"
                    )
            active.unlink()
            sync_directory(ROOT)
            return directory.name
        # An active link no record claims was not made by this helper: never take it.
        raise RecoveryError("recovery destination has another owner")

    def prepare(self) -> dict[str, Any]:
        with self.locked():
            if self.record:
                raise RecoveryError("recovery attempt already exists; cleanup only")
            now = time.time()
            if (
                not now + 60
                < self.binding["restore_at"]
                < self.binding["expires_at"]
                <= now + MAX_WINDOW
            ):
                raise RecoveryError("recovery window is expired or too long")
            self.initialize()
            require_system_directory(SYSTEMD)
            require_system_directory(ENABLE_LINK.parent)
            require_system_directory(SYSTEMD / "timers.target.wants")
            claim = self.directory / "claim"
            descriptor = os.open(
                claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(descriptor, "w") as handle:
                handle.write(self.key)
                handle.flush()
                os.fsync(handle.fileno())
            sync_directory(self.directory)
            identity = file_identity(claim)
            self.record["claim"] = identity
            self.save()  # A refused claim still leaves a journal: cleanup only.
            reclaimed = self.reclaim_stale_active()
            if reclaimed is not None:
                self.record["reclaimed_stale_active"] = reclaimed
                self.save()
            publish(claim, ROOT / "active", identity)
            if boot_id() != self.binding["boot_id"]:
                raise RecoveryError("node rebooted before recovery arm")
            before = unit_state(AGENT)
            if (
                before.get("UnitFileState") != "enabled"
                or before.get("ActiveState") != "active"
                or not no_job(before)
            ):
                raise RecoveryError(
                    "Node Agent baseline must be enabled, active and idle"
                )
            # Only the standard installer boot link is supported; aliases could
            # otherwise keep the Agent boot-enabled or expand the restoration.
            links = {
                p
                for root in (SYSTEMD, Path("/run/systemd/system"))
                for p in root.rglob("*")
                if p.is_symlink() and p.resolve() == (SYSTEMD / AGENT).resolve()
            }
            if links != {ENABLE_LINK}:
                raise RecoveryError("Node Agent has an unsupported enable-link layout")
            original = file_identity(ENABLE_LINK)
            if original.get("link") != str(SYSTEMD / AGENT):
                raise RecoveryError("Node Agent enable link is not the installer link")
            self.record["baseline"] = runtime_identity(self.binding, self.dropin)
            self.record["enable_link"] = original
            self.save()
            publish(ENABLE_LINK, self.directory / "agent-boot-link", original)
            script = Path(__file__).read_bytes()
            if hashlib.sha256(script).hexdigest() != self.binding["helper_sha256"]:
                raise RecoveryError("recovery helper source drifted")
            self.resource("helper", self.directory / "recovery.py", script)
            self.dropin.parent.mkdir(mode=0o755, exist_ok=True)
            require_system_directory(self.dropin.parent)
            self.resource(
                "agent-start-bound",
                self.dropin,
                b"[Unit]\nJobTimeoutSec=45s\n[Service]\nTimeoutStartSec=30s\n",
            )
            self.resource(
                "service",
                SYSTEMD / self.service,
                (
                    "[Unit]\nDescription=DESTR-014 bounded Agent recovery\n"
                    "After=local-fs.target\n[Service]\nType=oneshot\n"
                    f"ExecStart={PYTHON} -I -S -B {self.directory}/recovery.py tick --key {self.key}\n"
                    "TimeoutStartSec=120s\nTimeoutStopSec=5s\nKillMode=control-group\n"
                    "UMask=0077\nStandardOutput=null\nStandardError=null\n"
                ).encode(),
            )
            self.resource(
                "timer",
                SYSTEMD / self.timer,
                (
                    "[Unit]\nDescription=DESTR-014 persistent recovery timer\n[Timer]\n"
                    "OnBootSec=5s\nOnCalendar=*-*-* *:*:00,15,30,45 UTC\n"
                    "Persistent=true\nAccuracySec=1s\nRandomizedDelaySec=0\n"
                    f"Unit={self.service}\n[Install]\nWantedBy=timers.target\n"
                ).encode(),
            )
            self.resource(
                "timer-boot-link",
                SYSTEMD / "timers.target.wants" / self.timer,
                str(SYSTEMD / self.timer),
            )
            self.record["phase"] = "INSTALLED"
            self.save()
            systemctl("daemon-reload")
            systemctl("start", self.timer)
            systemctl("start", "--no-block", self.service)
            return self.report()

    def independent_ack(self) -> dict[str, Any]:
        unit = unit_state(self.service)
        invocation = os.environ.get("INVOCATION_ID")
        cgroup = Path("/proc/self/cgroup").read_text()
        if (
            not invocation
            or unit.get("InvocationID") != invocation
            or unit.get("ActiveState") != "activating"
            or unit.get("FragmentPath") != str(SYSTEMD / self.service)
            or f"/{self.service}\n" not in cgroup
        ):
            raise RecoveryError(
                "recovery ACK did not come from its independent service"
            )
        return {"invocation_id": invocation, "boot_id": boot_id(), "at": time.time()}

    def verify_timer(self) -> None:
        self.verify_resources()
        timer = unit_state(self.timer)
        # A timer whose unit is executing right now reports SubState=running,
        # not waiting; the independent tick is that unit, so both states mean
        # the persistent timer is armed (verified on the node: every tick
        # refused itself as "not armed" while the runner waited for the ACK).
        if (
            timer.get("ActiveState") != "active"
            or timer.get("SubState") not in {"waiting", "running"}
            or timer.get("UnitFileState") != "enabled"
            or timer.get("FragmentPath") != str(SYSTEMD / self.timer)
        ):
            raise RecoveryError("persistent recovery timer is not armed")
        agent = unit_state(AGENT)
        if (
            agent.get("JobTimeoutUSec") != "45s"
            or agent.get("TimeoutStartUSec") != "30s"
        ):
            raise RecoveryError("Agent recovery start job is not bounded")

    def report(self) -> dict[str, Any]:
        return {
            "binding_sha256": self.key,
            "phase": self.record.get("phase"),
            "record_kind": (
                "FORENSIC_TOMBSTONE"
                if self.record.get("phase") == "CLOSED"
                else "UNFINISHED_RECOVERY"
            ),
            "ack": self.record.get("ack"),
            "last_tick_error": self.record.get("last_tick_error"),
            "disable_started": self.record.get("disable_started"),
            "restored_at": self.record.get("restored_at"),
            "restore_reason": self.record.get("restore_reason"),
            "restored_by": self.record.get("restored_by"),
            "boot_id_observed": self.record.get("boot_id_observed"),
            "expired_at": self.record.get("expired_at"),
            "retired": bool(self.record.get("retired")),
            "start_requested": self.record.get("start_requested"),
            "reclaimed_stale_active": self.record.get("reclaimed_stale_active"),
            "timer": self.timer,
            "service": self.service,
        }

    def status(self) -> dict[str, Any]:
        with self.locked():
            if not self.record:
                raise RecoveryError("recovery journal is absent")
            if self.record["phase"] not in TERMINAL:
                self.verify_runtime()
                self.verify_timer()
            return self.report()

    def disable(self) -> dict[str, Any]:
        with self.locked():
            ack = self.record.get("ack") or {}
            now = time.time()
            if (
                self.record.get("phase") != "ARMED"
                or ack.get("boot_id") != self.binding["boot_id"]
                or boot_id() != self.binding["boot_id"]
                or not isinstance(ack.get("at"), (float, int))
                or not math.isfinite(ack["at"])
                or not 0 <= now - ack["at"] <= ACK_SECONDS
                or now + 60 >= self.binding["restore_at"]
            ):
                raise RecoveryError("fresh independent recovery arm is required")
            self.verify_runtime()
            self.verify_timer()
            now = time.time()
            if (
                not 0 <= now - ack["at"] <= ACK_SECONDS
                or now + 60 >= self.binding["restore_at"]
                or boot_id() != self.binding["boot_id"]
            ):
                raise RecoveryError(
                    "independent recovery arm expired during verification"
                )
            if not matches(ENABLE_LINK, self.record["enable_link"]):
                raise RecoveryError("Node Agent enable-link owner changed")
            self.record.update(phase="DISABLING", disable_started=True)
            self.save()
            if (
                time.time() + 60 >= self.binding["restore_at"]
                or not 0 <= time.time() - ack["at"] <= ACK_SECONDS
                or boot_id() != self.binding["boot_id"]
            ):
                raise RecoveryError(
                    "independent recovery arm expired while saving intent"
                )
            remove_owned(ENABLE_LINK, self.record["enable_link"])
            systemctl("daemon-reload")
            unit = unit_state(AGENT)
            if (
                unit.get("UnitFileState") != "disabled"
                or unit.get("ActiveState") != "active"
            ):
                raise RecoveryError("Node Agent boot disable was not verified")
            self.record["phase"] = "DISABLED"
            self.save()
            return self.report()

    def expire_locked(self) -> None:
        self.record.update(
            phase="EXPIRED", expired_at=time.time(), boot_id_observed=boot_id()
        )
        self.save()

    def restore_locked(self, reason: str) -> None:
        if self.record["phase"] in TERMINAL:
            return
        if self.record["phase"] != "RESTORING":
            self.record["restore_reason"] = reason  # The first trigger owns it.
        self.record["phase"] = "RESTORING"
        self.save()  # Seal disable before restoring, even if its ACK was lost.
        observed = boot_id()
        restored_by = None
        if self.record["disable_started"]:
            self.verify_runtime()
            self.verify_resources()
            owner = self.enable_link_owner() if self.rebooted else "saved"
            if owner == "foreign":
                raise RecoveryError(
                    "Node Agent enable link has an unknown owner; manual recovery required"
                )
            if owner == "installer":
                self.accept_installer_restore()
                restored_by = "reboot-installer"
            else:
                self.restore_saved_link()
                self.verify_runtime()
                restored_by = "probe"
        self.record.update(
            phase="RESTORED",
            restored_at=time.time(),
            restored_by=restored_by,
            boot_id_observed=observed,
        )
        self.save()

    def accept_installer_restore(self) -> None:
        """A new boot on which the standard installer already re-enabled the
        Agent under a fresh link inode: nothing is left to restore, and that
        link is not this recovery's to touch. Record what was observed."""
        unit = unit_state(AGENT)
        if (
            unit.get("UnitFileState") != "enabled"
            or unit.get("ActiveState") != "active"
            or not no_job(unit)
        ):
            raise RecoveryError(
                "installer-restored Node Agent is not enabled, active and idle yet"
            )
        self.record["enable_link_observed"] = file_identity(ENABLE_LINK)
        self.save()

    def restore_saved_link(self) -> None:
        """Publish the preserved enable-link inode and start the same Agent if
        it is not running; identical on the original boot and on a new one."""
        if time.time() >= self.binding["expires_at"]:
            self.expire_locked()
            raise RecoveryError("recovery window expired; manual recovery required")
        publish(
            self.directory / "agent-boot-link",
            ENABLE_LINK,
            self.record["enable_link"],
            lenient=self.rebooted,
        )
        systemctl("daemon-reload")
        unit = unit_state(AGENT)
        if unit.get("ActiveState") != "active":
            if not self.record["start_requested"]:
                if not no_job(unit):
                    raise RecoveryError("another Node Agent job is in progress")
                self.verify_timer()
                if time.time() + START_SECONDS + 25 >= self.binding["expires_at"]:
                    self.expire_locked()
                    raise RecoveryError("insufficient bounded Agent start window")
                self.record["start_requested"] = True
                self.save()
                if time.time() + START_SECONDS + 20 >= self.binding["expires_at"]:
                    self.expire_locked()
                    raise RecoveryError(
                        "Agent start window expired while saving intent"
                    )
                # The owned drop-in caps even a start request whose ACK is lost.
                systemctl("start", "--no-block", "--job-mode=fail", AGENT)
            deadline = time.monotonic() + START_SECONDS + 5
            while time.monotonic() < deadline:
                unit = unit_state(AGENT)
                if unit.get("ActiveState") == "active" and no_job(unit):
                    break
                time.sleep(1)
        if (
            unit.get("UnitFileState") != "enabled"
            or unit.get("ActiveState") != "active"
            or not no_job(unit)
        ):
            raise RecoveryError("Node Agent recovery is not complete")

    def retire_locked(self, *, independent: bool) -> None:
        if self.record["phase"] not in TERMINAL:
            raise RecoveryError("recovery is not terminal")
        # Check the entire set before touching any member; no name-based cleanup.
        resources = self.record["resources"]
        for resource in resources:
            target = Path(resource["target"])
            if (target.exists() or target.is_symlink()) and not self.same(
                target, resource["identity"]
            ):
                raise RecoveryError("recovery cleanup found a replaced owner")
        names = {
            Path(item["target"]).name
            for item in resources
            if Path(item["target"]).exists() or Path(item["target"]).is_symlink()
        }
        if any(Path(item["target"]).name == self.timer for item in resources):
            timer = unit_state(self.timer)
            if timer.get("ActiveState") not in {"inactive", "failed"} or not no_job(
                timer
            ):
                if timer.get("FragmentPath") != str(SYSTEMD / self.timer):
                    raise RecoveryError("recovery timer owner is unproven")
                systemctl("stop", self.timer)
                timer = unit_state(self.timer)
                if timer.get("ActiveState") not in {"inactive", "failed"} or not no_job(
                    timer
                ):
                    raise RecoveryError("persistent recovery timer has not stopped")
        if not independent and any(
            Path(item["target"]).name == self.service for item in resources
        ):
            service = unit_state(self.service)
            if service.get("ActiveState") not in {"inactive", "failed"} or not no_job(
                service
            ):
                if self.service not in names and (
                    service.get("InvocationID")
                    != (self.record.get("ack") or {}).get("invocation_id")
                    or service.get("FragmentPath") != str(SYSTEMD / self.service)
                ):
                    raise RecoveryError("recovery process owner is unproven")
                systemctl("stop", self.service)
                service = unit_state(self.service)
                if service.get("ActiveState") not in {
                    "inactive",
                    "failed",
                } or not no_job(service):
                    raise RecoveryError("independent recovery process has not stopped")
        if "enable_link" in self.record and not no_job(unit_state(AGENT)):
            raise RecoveryError(
                "Agent start job is pending; recovery artifacts retained"
            )
        for resource in reversed(resources):
            remove_owned(
                Path(resource["target"]), resource["identity"], lenient=self.rebooted
            )
        if names:
            # The drop-in is gone from disk; reload so no later Agent start
            # runs under the retired 45 s bound.
            systemctl("daemon-reload")
        self.record["retired"] = True
        self.save()
        if not independent:
            if "enable_link" in self.record:
                self.verify_runtime(require_claim=self.record["phase"] != "CLOSED")
                agent = unit_state(AGENT)
                owner = self.enable_link_owner()
                if (
                    owner not in {"saved", "installer"}
                    or agent.get("UnitFileState") != "enabled"
                    or agent.get("ActiveState") != "active"
                    or not no_job(agent)
                ):
                    raise RecoveryError("Node Agent restoration proof was lost")
                if owner == "installer" and not self.record.get("restored_by"):
                    # EXPIRED without a restore of its own, then the host came
                    # back with the installer's Agent: that is the proof.
                    self.record.update(
                        restored_by="reboot-installer",
                        enable_link_observed=file_identity(ENABLE_LINK),
                        boot_id_observed=boot_id(),
                    )
            self.record["phase"] = "CLOSED"
            self.save()
            if "claim" in self.record:
                remove_owned(
                    ROOT / "active", self.record["claim"], lenient=self.rebooted
                )

    def restore(self) -> dict[str, Any]:
        with self.locked():
            self.initialize()
            self.restore_locked("controller")
            return self.report()

    def cleanup(self) -> dict[str, Any]:
        with self.locked():
            self.initialize()  # Tombstone also fences an arm request arriving late.
            self.restore_locked("cleanup")
            self.retire_locked(independent=False)
            return self.report()

    def tick(self) -> dict[str, Any]:
        try:
            return self._tick()
        except RecoveryError as exc:
            # The service discards its output; keep the reason where `status`
            # can report it, so a runner that never sees ARMED learns why.
            self.note_tick_error(str(exc))
            raise

    def note_tick_error(self, reason: str) -> None:
        try:
            with self.locked():
                if self.record and self.record.get("phase") not in TERMINAL:
                    self.record["last_tick_error"] = reason
                    self.save()
        except Exception:
            pass

    def _tick(self) -> dict[str, Any]:
        with self.locked():
            if not self.record or self.record["phase"] == "CLOSED":
                return self.report()
            if self.record["phase"] in TERMINAL:
                self.retire_locked(independent=True)
                return self.report()
            # Expiry and the deadline come before any check that only the
            # original boot can pass.
            if time.time() >= self.binding["expires_at"]:
                self.expire_locked()
                self.retire_locked(independent=True)
                return self.report()
            if self.rebooted:
                self._tick_new_boot()
                return self.report()
            self.verify_runtime()
            self.verify_timer()
            self.record["ack"] = self.independent_ack()
            if self.record["phase"] == "INSTALLED":
                self.record["phase"] = "ARMED"
            self.save()
            if time.time() >= self.binding["restore_at"]:
                try:
                    self.restore_locked("deadline")
                finally:
                    if self.record["phase"] in TERMINAL:
                        self.retire_locked(independent=True)
            return self.report()

    def _tick_new_boot(self) -> None:
        """The node rebooted under this recovery: no ACK or disable can follow,
        only restoration. The installer may already have re-enabled the Agent
        (accept it and retire, deadline or not: the 45 s start bound must not
        outlive its purpose); a still-missing link waits for the deadline
        exactly as on the original boot; a foreign link is refused and left
        alone until expiry retires the probe."""
        due = time.time() >= self.binding["restore_at"]
        if (
            self.record["disable_started"]
            and not due
            and self.enable_link_owner() == "missing"
        ):
            return
        try:
            self.restore_locked("deadline" if due else "reboot")
        finally:
            if self.record["phase"] in TERMINAL:
                self.retire_locked(independent=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "status", "disable", "restore", "cleanup", "tick"),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--binding")
    group.add_argument("--key")
    args = parser.parse_args(argv)
    try:
        if args.key:
            if args.command != "tick" or not re.fullmatch(r"[0-9a-f]{64}", args.key):
                raise RecoveryError("only independent tick accepts a journal key")
            binding = read_record(ROOT / args.key / "state.json")["binding"]
            if binding_key(binding) != args.key:
                raise RecoveryError("recovery journal key changed")
        else:
            if args.command == "tick":
                raise RecoveryError("independent tick requires its journal key")
            binding = json.loads(args.binding)
        recovery = Recovery(binding)
        report = getattr(recovery, args.command)()
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "recovery_required": True}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
