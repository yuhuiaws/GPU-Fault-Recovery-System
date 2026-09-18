"""Exercise the deployed FM reader using only nonce-owned files and a local sink."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, NoReturn


CASE_ID = "GF-REGIONAL-COLLECT-022"
PRIVATE_BASE = Path("/var/lib/gpu-fault/acceptance")
NODE_PREFIX = Path("/opt/gpu-fault/current/venv")
BOOT_FILE = Path("/proc/sys/kernel/random/boot_id")
SERVICE = "gpu-fault-fabric-manager-collector.service"
MAX_FILE_BYTES = 128 * 1024
FILES = frozenset(
    {
        "owner.json",
        "progress.json",
        "progress.tmp",
        "active.log",
        "rotated.log",
        "cursor.json",
        "cursor.json.tmp",
    }
)
STEPS = (
    "baseline",
    "append",
    "restart",
    "loss",
    "after-loss",
    "corrupt",
    "after-corrupt",
    "rotate",
    "rotated-restart",
    "truncate",
    "truncated-restart",
)
TAGS = {
    "baseline": [],
    "append": ["A"],
    "restart": [],
    "loss": [],
    "after-loss": ["B"],
    "corrupt": [],
    "after-corrupt": ["C"],
    "rotate": ["R", "N"],
    "rotated-restart": [],
    "truncate": ["T"],
    "truncated-restart": [],
}
TRUNCATED_LOG = "active.log"


class ProbeError(RuntimeError):
    pass


class RootAbsent(FileNotFoundError):
    pass


def checked_nonce(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ProbeError(
            "private cursor nonce must be 32 lowercase hexadecimal characters"
        )
    return value


def check_deadline(expires_at: float) -> None:
    if (
        isinstance(expires_at, bool)
        or not math.isfinite(expires_at)
        or not time.time() < expires_at <= time.time() + 7200
    ):
        raise ProbeError("private cursor deadline is expired or invalid")


def deployed_identity() -> dict[str, str]:
    import gpu_fault
    from gpu_fault.collectors.logs import (
        fabric_manager,
        fabric_manager_cursor,
        fabric_manager_receipts,
    )

    if Path(sys.prefix).resolve() != NODE_PREFIX.resolve() or not NODE_PREFIX.is_dir():
        raise ProbeError("probe is not running in the deployed Node Runtime")
    implementation = {
        module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (fabric_manager, fabric_manager_cursor, fabric_manager_receipts)
    }
    return {
        "python_prefix": str(Path(sys.prefix).resolve()),
        "package_version": str(gpu_fault.__version__),
        "collector_module_sha256": hashlib.sha256(
            json.dumps(implementation, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def service_identity() -> dict[str, str]:
    completed = subprocess.run(
        [
            "systemctl",
            "show",
            SERVICE,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=MainPID",
            "--property=InvocationID",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=10,
    )
    fields = dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
    )
    if (
        completed.returncode
        or fields.get("LoadState") != "loaded"
        or fields.get("ActiveState") != "active"
        or not fields.get("MainPID", "").isdecimal()
        or int(fields["MainPID"]) <= 1
        or not fields.get("InvocationID")
    ):
        raise ProbeError("existing FM collector service identity is unproven")
    return {
        key: fields[key]
        for key in ("LoadState", "ActiveState", "MainPID", "InvocationID")
    }


def process_identity() -> str:
    start = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    return f"{os.getpid()}:{start}"


def directory_fd(path: Path, *, create_leaf: bool = False) -> int:
    if not path.is_absolute():
        raise ProbeError("private base must be absolute")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(path.parts[1:]):
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create_leaf or index != len(path.parts) - 2:
                    raise
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
            raise ProbeError("private base ownership or permissions are unsafe")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class PrivateRoot:
    def __init__(self, base_fd: int, descriptor: int, nonce: str) -> None:
        self.base_fd = base_fd
        self.fd = descriptor
        self.nonce = nonce
        self.name = f"c022-{nonce}"
        self.path = PRIVATE_BASE / self.name
        self.identity = self.metadata(os.fstat(descriptor))

    @staticmethod
    def metadata(value: os.stat_result) -> dict[str, int]:
        return {"device": value.st_dev, "inode": value.st_ino, "size": value.st_size}

    def validate(self) -> None:
        current = os.stat(self.name, dir_fd=self.base_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_uid != os.geteuid()
            or current.st_mode & 0o077
            or (current.st_dev, current.st_ino)
            != (self.identity["device"], self.identity["inode"])
        ):
            raise ProbeError("private root identity changed")
        for name in os.listdir(self.fd):
            self.file_stat(name)

    def file_stat(self, name: str, *, optional: bool = False) -> os.stat_result | None:
        if name not in FILES:
            raise ProbeError("unexpected entry in the private cursor directory")
        try:
            value = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            if optional:
                return None
            raise
        self.check_file(value)
        return value

    @staticmethod
    def check_file(value: os.stat_result) -> None:
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or value.st_uid != os.geteuid()
            or value.st_mode & 0o077
            or value.st_size > MAX_FILE_BYTES
        ):
            raise ProbeError(
                "private cursor file has unsafe type, ownership, links or size"
            )

    def read(self, name: str) -> bytes:
        self.file_stat(name)
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)
        try:
            self.check_file(os.fstat(descriptor))
            data = os.read(descriptor, MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ProbeError("private cursor file exceeds its size bound")
            return data
        finally:
            os.close(descriptor)

    def write(
        self, name: str, data: bytes, *, exclusive: bool = False, append: bool = False
    ) -> None:
        self.validate()
        existing = self.file_stat(name, optional=True)
        if (
            len(data) + (existing.st_size if existing and append else 0)
            > MAX_FILE_BYTES
        ):
            raise ProbeError("private cursor write exceeds its size bound")
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
        if exclusive:
            flags |= os.O_EXCL
        if append:
            flags |= os.O_APPEND
        descriptor = os.open(name, flags, 0o600, dir_fd=self.fd)
        try:
            self.check_file(os.fstat(descriptor))
            if not append:
                os.ftruncate(descriptor, 0)
            if os.write(descriptor, data) != len(data):
                raise ProbeError("private cursor write was incomplete")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def progress(self, value: dict[str, Any]) -> None:
        self.write("progress.tmp", json.dumps(value, sort_keys=True).encode())
        self.file_stat("progress.json", optional=True)
        os.replace(
            "progress.tmp", "progress.json", src_dir_fd=self.fd, dst_dir_fd=self.fd
        )
        os.fsync(self.fd)

    def snapshot(self) -> dict[str, Any]:
        self.validate()
        logs = {
            name: self.metadata(value)
            for name in ("active.log", "rotated.log")
            if (value := self.file_stat(name, optional=True)) is not None
        }
        cursor: dict[str, Any] = {"present": False, "valid": False, "files": {}}
        if self.file_stat("cursor.json", optional=True) is not None:
            contents = self.read("cursor.json")
            cursor.update(present=True, sha256=hashlib.sha256(contents).hexdigest())
            try:
                parsed = json.loads(contents)
            except ValueError:
                pass
            else:
                files = parsed.get("files") if isinstance(parsed, dict) else None
                if isinstance(files, dict):
                    expected = {str(self.path / name) for name in logs}
                    if set(files) - expected:
                        raise ProbeError("cursor refers outside the private log set")
                    cursor.update(valid=True, files=files)
        return {"logs": logs, "cursor": cursor}


def line(nonce: str, tag: str) -> bytes:
    context = "already emitted before copytruncate " if tag == "N" else ""
    return (
        "nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): 99999, Non-fatal, "
        f"{context}private cursor replay marker=c022-{nonce}-{tag}\n"
    ).encode()


@contextmanager
def owned_root(
    nonce: str,
    cluster_id: str,
    node_id: str,
    *,
    create: bool = False,
    owner: dict[str, Any] | None = None,
) -> Iterator[PrivateRoot]:
    checked_nonce(nonce)
    try:
        base_fd = directory_fd(PRIVATE_BASE, create_leaf=create)
    except FileNotFoundError as exc:
        if create:
            raise
        raise RootAbsent from exc
    root_fd: int | None = None
    lock_fd: int | None = None
    try:
        name = f"c022-{nonce}"
        if create:
            os.mkdir(name, mode=0o700, dir_fd=base_fd)
        try:
            root_fd = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=base_fd
            )
        except FileNotFoundError as exc:
            raise RootAbsent from exc
        root = PrivateRoot(base_fd, root_fd, nonce)
        root.validate()
        if create:
            identity = {key: root.identity[key] for key in ("device", "inode")}
            root.write(
                "owner.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "case_id": CASE_ID,
                        "nonce": nonce,
                        "cluster_id": cluster_id,
                        "node_id": node_id,
                        "root_identity": identity,
                        **(owner or {}),
                    },
                    sort_keys=True,
                ).encode(),
                exclusive=True,
            )
        root.file_stat("owner.json")
        lock_fd = os.open("owner.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(root.read("owner.json"))
        if (
            not isinstance(record, dict)
            or type(record.get("schema_version")) is not int
            or any(
                record.get(key) != value
                for key, value in {
                    "schema_version": 1,
                    "case_id": CASE_ID,
                    "nonce": nonce,
                    "cluster_id": cluster_id,
                    "node_id": node_id,
                    "root_identity": {
                        key: root.identity[key] for key in ("device", "inode")
                    },
                }.items()
            )
        ):
            raise ProbeError("private cursor ownership does not match this run")
        yield root
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if root_fd is not None:
            os.close(root_fd)
        os.close(base_fd)


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.health_summaries = 0

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        from gpu_fault.channel_registry import (
            COLLECTOR_HEALTH_PATH,
            FABRIC_MANAGER_PATH,
        )

        if path == COLLECTOR_HEALTH_PATH:
            self.health_summaries += 1
        elif path == FABRIC_MANAGER_PATH and len(self.events) < 16:
            self.events.append(
                {
                    key: payload.get(key)
                    for key in (
                        "cluster_id",
                        "node_id",
                        "record_id",
                        "message",
                        "fields",
                        "source",
                        "evidence_ref",
                    )
                }
            )
        else:
            raise ProbeError("isolated collector attempted an unexpected delivery")
        return {"status": "recorded-locally"}


def refuse_external_command(*args: Any, **kwargs: Any) -> NoReturn:
    raise ProbeError("isolated collector attempted an external command")


def collect_private(
    root: PrivateRoot, cluster_id: str, node_id: str, boot_id: str
) -> dict[str, Any]:
    from gpu_fault.collectors.logs.fabric_manager import FabricManagerLogCollector
    from gpu_fault.collectors.models import CollectorContext

    root.validate()
    sink = RecordingSink()
    collector = FabricManagerLogCollector(
        sink,
        CollectorContext(cluster_id=cluster_id),
        node_id=node_id,
        boot_id=boot_id,
        journal_enabled=False,
        journal_identifiers=(),
        log_paths=[str(root.path / name) for name in ("active.log", "rotated.log")],
        state_path=str(root.path / "cursor.json"),
        runner=refuse_external_command,
    )
    stats = collector.collect_once()
    return {
        "events": sink.events,
        "stats": stats.model_dump(mode="json"),
        "health_summaries": sink.health_summaries,
        "sink": "non-forwarding",
    }


def truncate_private(root: PrivateRoot) -> dict[str, Any]:
    before = root.snapshot()
    metadata = before["logs"].get(TRUNCATED_LOG) or {}
    checkpoint = before["cursor"]["files"].get(str(root.path / TRUNCATED_LOG)) or {}
    progress = json.loads(root.read("progress.json"))
    emitted = [
        event
        for event in (progress.get("receipts", {}).get("rotate", {}).get("events", []))
        if (event.get("fields") or {}).get("path") == str(root.path / TRUNCATED_LOG)
        and (event.get("fields") or {}).get("offset") == "0"
    ]
    data = line(root.nonce, "T")
    if (
        before["cursor"]["valid"] is not True
        or not isinstance(checkpoint, dict)
        or any(checkpoint.get(key) != metadata.get(key) for key in ("device", "inode"))
        or type(checkpoint.get("offset")) is not int
        or checkpoint["offset"] != metadata.get("size")
        or len(data) >= checkpoint["offset"]
        or len(emitted) != 1
        or not emitted[0].get("record_id")
        or not emitted[0].get("evidence_ref")
        or any(
            (emitted[0].get("fields") or {}).get(key) != str(checkpoint.get(key))
            for key in ("device", "inode", "generation")
        )
        or root.read(TRUNCATED_LOG) != line(root.nonce, "N")
        or emitted[0].get("message") != line(root.nonce, "N").decode().rstrip("\n")
    ):
        raise ProbeError("private truncation checkpoint/size premise is unproven")
    # Reuse the byte-zero position that was actually delivered in the rotate phase.
    root.write(TRUNCATED_LOG, b"")
    truncated = root.snapshot()
    root.write(TRUNCATED_LOG, data, append=True)
    return {
        "truncation": {
            "before": before,
            "truncated": truncated,
            "reused_event": emitted[0],
        }
    }


def mutate_private(root: PrivateRoot, step: str) -> dict[str, Any]:
    if step in {"append", "after-loss", "after-corrupt"}:
        root.write("active.log", line(root.nonce, TAGS[step][0]), append=True)
    elif step == "loss":
        root.file_stat("cursor.json")
        os.unlink("cursor.json", dir_fd=root.fd)
    elif step == "corrupt":
        root.write("cursor.json", b'{"files":')
    elif step == "rotate":
        root.write("active.log", line(root.nonce, "R"), append=True)
        if root.file_stat("rotated.log", optional=True) is not None:
            raise ProbeError("rotation destination already exists")
        os.rename("active.log", "rotated.log", src_dir_fd=root.fd, dst_dir_fd=root.fd)
        root.write("active.log", line(root.nonce, "N"), exclusive=True)
    elif step == "truncate":
        return truncate_private(root)
    return {}


def run_action(
    action: str,
    *,
    nonce: str,
    cluster_id: str,
    node_id: str,
    expires_at: float | None = None,
    step: str | None = None,
) -> dict[str, Any]:
    checked_nonce(nonce)
    if not cluster_id or not node_id:
        raise ProbeError("private cursor scope is empty")
    if action == "cleanup":
        return cleanup(nonce=nonce, cluster_id=cluster_id, node_id=node_id)
    if action not in {"init", "step"} or expires_at is None:
        raise ProbeError("private cursor action or deadline is missing")
    check_deadline(expires_at)
    runtime, service = deployed_identity(), service_identity()
    boot_id = BOOT_FILE.read_text().strip()
    if not boot_id:
        raise ProbeError("node boot identity is missing")
    identity = {
        "runtime": runtime,
        "service": service,
        "boot_id": boot_id,
        "expires_at": expires_at,
    }
    check_deadline(expires_at)
    previous_umask = os.umask(0o077)
    try:
        with owned_root(
            nonce, cluster_id, node_id, create=action == "init", owner=identity
        ) as root:
            owner = json.loads(root.read("owner.json"))
            if any(owner.get(key) != value for key, value in identity.items()):
                raise ProbeError("node runtime, service, boot or deadline changed")
            if action == "init":
                root.write("active.log", line(nonce, "H"), exclusive=True)
                root.progress({"next_step": 0, "inflight": None, "receipts": {}})
                return {"initialized": True, **owner}
            if step not in STEPS:
                raise ProbeError("private cursor phase is not allowed")
            progress = json.loads(root.read("progress.json"))
            index = STEPS.index(step)
            if progress.get("inflight") is not None:
                raise ProbeError("prior private phase outcome is unresolved")
            if type(progress.get("next_step")) is not int:
                raise ProbeError("private cursor phase state is malformed")
            if index < progress["next_step"]:
                return dict(progress["receipts"][step])
            if index != progress["next_step"]:
                raise ProbeError("private cursor phases must run in strict order")
            progress["inflight"] = step
            root.progress(progress)
            check_deadline(expires_at)
            mutation = mutate_private(root, step)
            before = root.snapshot()
            captured = collect_private(root, cluster_id, node_id, boot_id)
            after = root.snapshot()
            if service_identity() != service or deployed_identity() != runtime:
                raise ProbeError(
                    "existing service or deployed runtime changed during collection"
                )
            receipt = {
                "schema_version": 1,
                "case_id": CASE_ID,
                "nonce": nonce,
                "cluster_id": cluster_id,
                "node_id": node_id,
                "step": step,
                "boot_id": boot_id,
                "runtime": runtime,
                "service": service,
                "root_identity": owner["root_identity"],
                "process_identity": process_identity(),
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "before": before,
                "after": after,
                **captured,
                **mutation,
            }
            progress["receipts"][step] = receipt
            progress["next_step"] = index + 1
            progress["inflight"] = None
            root.progress(progress)
            return receipt
    finally:
        os.umask(previous_umask)


def cleanup(*, nonce: str, cluster_id: str, node_id: str) -> dict[str, bool]:
    try:
        with owned_root(nonce, cluster_id, node_id) as root:
            root.validate()
            for name in sorted(os.listdir(root.fd)):
                if name != "owner.json":
                    root.file_stat(name)
                    os.unlink(name, dir_fd=root.fd)
            root.validate()
            os.unlink("owner.json", dir_fd=root.fd)
            os.fsync(root.fd)
            os.rmdir(root.name, dir_fd=root.base_fd)
            os.fsync(root.base_fd)
    except RootAbsent:
        pass
    return {"private_root": False, "creation_unresolved": False}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("action", choices=("init", "step", "cleanup"))
    value.add_argument("--nonce", required=True)
    value.add_argument("--cluster-id", required=True)
    value.add_argument("--node-id", required=True)
    value.add_argument("--expires-at", type=float)
    value.add_argument("--step", choices=STEPS)
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        result = run_action(
            arguments.action,
            nonce=arguments.nonce,
            cluster_id=arguments.cluster_id,
            node_id=arguments.node_id,
            expires_at=arguments.expires_at,
            step=arguments.step,
        )
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
