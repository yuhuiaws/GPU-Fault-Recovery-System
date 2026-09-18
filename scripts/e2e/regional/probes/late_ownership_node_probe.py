"""Independent node-side exec and workload-client witness for the live drill."""

from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import os
import select
import secrets
import shutil
import socket
import stat
import struct
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied, process_identity
from scripts.e2e.regional.late_ownership_contract import (
    AcceptanceScope,
    NodeIdentity,
    QuiescenceReceipt,
)
from scripts.e2e.regional.late_ownership_trace import AttachedExecWitness

MAX_MESSAGE_BYTES = 65536
CLEANUP_SECONDS = 180
RPC_ACTIONS = frozenset(
    {"status", "calibrated", "clients", "finish", "exit", "cleanup-mailbox"}
)
CALIBRATION = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name",
    "--format=csv,noheader,nounits",
)


def mailbox_path(scope: AcceptanceScope, node: NodeIdentity) -> Path:
    node_key = hashlib.sha256(node.name.encode()).hexdigest()[:12]
    return Path("/run") / f"gpu-fault-late-{scope.challenge[:20]}" / node_key


def private_directory(directory: Path) -> None:
    metadata = directory.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise BoundaryDenied("node witness mailbox is not privately owned")


def remove_empty_parent(directory: Path) -> None:
    try:
        directory.parent.rmdir()
    except OSError as exc:
        if exc.errno != errno.ENOTEMPTY:
            raise


def daemon_request(
    scope: AcceptanceScope, node: NodeIdentity, action: str, payload: dict[str, Any]
) -> dict[str, Any]:
    directory = mailbox_path(scope, node)
    private_directory(directory.parent)
    private_directory(directory)
    request_id = secrets.token_hex(16)
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
        client.settimeout(30)
        client.connect(str(directory / "control.sock"))
        peer_pid, peer_uid, _ = struct.unpack(
            "3i",
            client.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
            ),
        )
        peer = process_identity(peer_pid)
        if peer_uid != os.geteuid() or peer.boot_id != node.boot_id:
            raise BoundaryDenied("node witness daemon identity is unproven")
        message = json.dumps(
            {
                "scope_sha256": scope.digest(),
                "action": action,
                "payload": payload,
                "request_id": request_id,
            },
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        if len(message) > MAX_MESSAGE_BYTES:
            raise BoundaryDenied("node witness mailbox request is oversized")
        client.sendall(message)
        data, _, flags, _ = client.recvmsg(MAX_MESSAGE_BYTES)
        if flags & socket.MSG_TRUNC:
            raise BoundaryDenied("node witness mailbox response was truncated")
    value = json.loads(data)
    if (
        not isinstance(value, dict)
        or value.get("scope_sha256") != scope.digest()
        or value.get("request_id") != request_id
        or value.get("action") != action
        or "error_kind" in value
    ):
        raise BoundaryDenied("node witness mailbox response is unbound")
    value["peer"] = peer.model_dump(mode="json")
    return value


def cleanup_mailbox(scope: AcceptanceScope, node: NodeIdentity, pod_uid: str) -> None:
    directory = mailbox_path(scope, node)
    try:
        directory.lstat()
    except FileNotFoundError:
        return
    private_directory(directory.parent)
    private_directory(directory)
    owner_path = directory / "owner.json"
    owner_stat = owner_path.lstat()
    if (
        not stat.S_ISREG(owner_stat.st_mode)
        or owner_stat.st_uid != os.geteuid()
        or owner_stat.st_mode & 0o077
    ):
        raise BoundaryDenied("node witness owner receipt is not private")
    owner = json.loads(owner_path.read_text())
    if owner.get("scope_sha256") != scope.digest() or owner.get("pod_uid") != pod_uid:
        raise BoundaryDenied("node witness cleanup does not own this mailbox")
    recorded = owner["producer"]
    try:
        Path(f"/proc/{recorded['pid']}").lstat()
    except FileNotFoundError:
        pass
    else:
        if process_identity(recorded["pid"]).model_dump(mode="json") == recorded:
            raise BoundaryDenied("node witness process is still alive")
    for child in directory.iterdir():
        info = child.lstat()
        if (
            child.name not in {"owner.json", "control.sock"}
            or info.st_uid != os.geteuid()
        ):
            raise BoundaryDenied("node witness cleanup found an unowned file")
        if child.name == "control.sock" and not stat.S_ISSOCK(info.st_mode):
            raise BoundaryDenied("node witness control path was replaced")
    (directory / "control.sock").unlink(missing_ok=True)
    owner_path.unlink()
    directory.rmdir()
    remove_empty_parent(directory)


def matches_pod(cgroups: str, pod_uid: str) -> bool:
    for line in cgroups.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            raise BoundaryDenied("GPU client cgroup identity is malformed")
        for component in Path(fields[2]).parts:
            if component == f"pod{pod_uid}" or (
                component.startswith("kubepods-")
                and component.endswith(f"-pod{pod_uid.replace('-', '_')}.slice")
            ):
                return True
    return False


def observe_clients(
    scope: AcceptanceScope,
    node: NodeIdentity,
    pod_uids: tuple[str, ...],
    *,
    runner: Any = subprocess.run,
    proc: Path = Path("/proc"),
) -> dict[str, Any]:
    result = runner(
        list(CALIBRATION), text=True, capture_output=True, check=True, timeout=15
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    observations = []
    for row in rows:
        if not row:
            continue
        if len(row) != 3 or not row[0].strip() or not row[1].strip().isdigit():
            raise BoundaryDenied("NVIDIA compute-client response is malformed")
        pid = int(row[1].strip())
        before = process_identity(pid, proc=proc)
        groups = (proc / str(pid) / "cgroup").read_text(encoding="ascii")
        if process_identity(pid, proc=proc) != before:
            raise BoundaryDenied("GPU client process identity changed")
        matched = [uid for uid in pod_uids if matches_pod(groups, uid)]
        if matched:
            observations.append(
                {
                    "process": before.model_dump(mode="json"),
                    "gpu_uuid": row[0].strip(),
                    "cgroup_sha256": hashlib.sha256(groups.encode()).hexdigest(),
                    "pod_uids": matched,
                }
            )
    return {
        "scope_sha256": scope.digest(),
        "node": node.model_dump(mode="json"),
        "producer": process_identity(os.getpid()).model_dump(mode="json"),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "source": "nvidia-compute-apps+proc-cgroup/v1",
        "pod_uids": list(pod_uids),
        "observations": observations,
    }


class NodeProbe:
    def __init__(
        self,
        scope: AcceptanceScope,
        node: NodeIdentity,
    ) -> None:
        self.scope = scope
        self.node = node
        self.deadline = time.monotonic() + max(
            0,
            (
                scope.maintenance_end
                + timedelta(seconds=CLEANUP_SECONDS)
                - datetime.now(timezone.utc)
            ).total_seconds(),
        )

    def run_daemon(self) -> None:
        self.scope.check_window(datetime.now(timezone.utc))
        pod_uid = os.environ.get("LATE_OWNERSHIP_WITNESS_POD_UID", "")
        if not pod_uid:
            raise BoundaryDenied("node witness has no downward-API Pod UID")
        directory = mailbox_path(self.scope, self.node)
        directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        private_directory(directory.parent)
        directory.mkdir(mode=0o700)
        owner_path = directory / "owner.json"
        descriptor = os.open(
            owner_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "w") as owner:
            json.dump(
                {
                    "scope_sha256": self.scope.digest(),
                    "pod_uid": pod_uid,
                    "producer": process_identity(os.getpid()).model_dump(mode="json"),
                },
                owner,
            )
        executable = shutil.which("nvidia-smi")
        if executable is None:
            raise BoundaryDenied("physical reset executable is unavailable")
        nvidia_smi = Path(executable).resolve()
        pid = subprocess.run(
            [
                "systemctl",
                "show",
                "--property=MainPID",
                "--value",
                "gpu-fault-node-agent.service",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        if not pid.isdigit():
            raise BoundaryDenied("Node Agent process identity is unavailable")
        tracee = process_identity(int(pid))
        if tracee.boot_id != self.node.boot_id:
            raise BoundaryDenied("Node Agent boot identity changed")
        witness = AttachedExecWitness(
            directory, tracee, deadline=self.deadline, executable=nvidia_smi
        )
        start = None
        end = None
        try:
            witness.start()
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as server:
                server.bind(str(directory / "control.sock"))
                (directory / "control.sock").chmod(0o600)
                server.listen(2)
                while time.monotonic() < self.deadline:
                    if end is None:
                        witness.check()
                    if not select.select(
                        [server], [], [], min(5, self.deadline - time.monotonic())
                    )[0]:
                        continue
                    with server.accept()[0] as client:
                        client.settimeout(30)
                        _, uid, _ = struct.unpack(
                            "3i",
                            client.getsockopt(
                                socket.SOL_SOCKET,
                                socket.SO_PEERCRED,
                                struct.calcsize("3i"),
                            ),
                        )
                        data, _, flags, _ = client.recvmsg(MAX_MESSAGE_BYTES)
                        if uid != os.geteuid() or flags & socket.MSG_TRUNC:
                            raise BoundaryDenied(
                                "node witness mailbox caller is invalid"
                            )
                        request = json.loads(data)
                        if (
                            not isinstance(request, dict)
                            or set(request)
                            != {"scope_sha256", "action", "payload", "request_id"}
                            or request["scope_sha256"] != self.scope.digest()
                            or not isinstance(request["request_id"], str)
                            or len(request["request_id"]) != 32
                            or not isinstance(request["payload"], dict)
                        ):
                            raise BoundaryDenied(
                                "node witness mailbox request is unbound"
                            )
                        action, payload = request["action"], request["payload"]
                        response: dict[str, Any] = {
                            "scope_sha256": self.scope.digest(),
                            "request_id": request["request_id"],
                            "action": action,
                        }
                        if action == "status":
                            response.update(
                                {
                                    "phase": "COMPLETE"
                                    if end is not None
                                    else "ARMED"
                                    if start is not None
                                    else "ATTACHED",
                                    "producer": process_identity(
                                        os.getpid()
                                    ).model_dump(mode="json"),
                                }
                            )
                        elif action == "calibrated" and start is None:
                            start = witness.start_receipt(
                                self.scope,
                                self.node,
                                witness_id=f"{self.scope.run_id}/{self.node.name}",
                                nvidia_smi=nvidia_smi,
                                calibration_argv=CALIBRATION,
                            )
                            response["receipt"] = start.model_dump(mode="json")
                        elif action == "clients" and start is not None and end is None:
                            uids = payload.get("pod_uids")
                            if (
                                not isinstance(uids, list)
                                or not uids
                                or not all(isinstance(uid, str) and uid for uid in uids)
                            ):
                                raise BoundaryDenied(
                                    "client observation has no exact Pod UID set"
                                )
                            response["observation"] = observe_clients(
                                self.scope, self.node, tuple(uids)
                            )
                        elif action == "finish" and start is not None:
                            quiet = QuiescenceReceipt.model_validate_json(
                                json.dumps(payload)
                            )
                            if end is None:
                                end = witness.finish_receipt(
                                    start,
                                    quiet,
                                    nvidia_smi=nvidia_smi,
                                    calibration_argv=CALIBRATION,
                                )
                            elif end.quiescence_sha256 != quiet.digest():
                                raise BoundaryDenied(
                                    "witness finish was replayed for another quiescence"
                                )
                            response["receipt"] = end.model_dump(mode="json")
                        elif action == "exit" and end is not None:
                            response["exited"] = True
                            client.sendall(json.dumps(response).encode())
                            return
                        else:
                            raise BoundaryDenied(
                                "node witness mailbox command is out of order"
                            )
                        encoded = json.dumps(response, separators=(",", ":")).encode()
                        if len(encoded) > MAX_MESSAGE_BYTES:
                            raise BoundaryDenied(
                                "node witness response exceeds its bound"
                            )
                        client.sendall(encoded)
            raise BoundaryDenied(
                "node witness deadline expired without terminal closure"
            )
        finally:
            witness.close()
            (directory / "control.sock").unlink(missing_ok=True)
            owner_path.unlink()
            directory.rmdir()
            remove_empty_parent(directory)


def read_request() -> dict[str, Any]:
    # The pinned source loader may already have prefetched this line as text.
    raw = sys.stdin.readline(MAX_MESSAGE_BYTES + 1)
    if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise BoundaryDenied("node witness input exceeds its byte bound")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise BoundaryDenied("node witness input has duplicate fields")
            value[key] = item
        return value

    def invalid_constant(value: str) -> Any:
        raise BoundaryDenied("node witness input has a nonfinite constant")

    request = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(request, dict):
        raise BoundaryDenied("node witness input is not an object")
    if set(request) == {"scope", "node", "daemon"} and request["daemon"] is True:
        return request
    if (
        set(request) != {"scope", "node", "action", "payload"}
        or not isinstance(request["action"], str)
        or request["action"] not in RPC_ACTIONS
        or not isinstance(request["payload"], dict)
    ):
        raise BoundaryDenied("node witness request mode is invalid")
    return request


def main() -> int:
    try:
        request = read_request()
        scope = AcceptanceScope.model_validate_json(json.dumps(request["scope"]))
        now = datetime.now(timezone.utc)
        daemon = request.get("daemon") is True
        if daemon:
            scope.check_window(now)
        else:
            if (
                not scope.maintenance_start
                <= now
                < scope.maintenance_end + timedelta(seconds=CLEANUP_SECONDS)
            ):
                raise BoundaryDenied("node witness cleanup window ended")
        node = next(node for node in scope.nodes if node.name == request["node"])
        if daemon:
            NodeProbe(scope, node).run_daemon()
        elif request["action"] == "cleanup-mailbox":
            cleanup_mailbox(scope, node, request["payload"]["pod_uid"])
            print(
                json.dumps({"scope_sha256": scope.digest(), "mailbox_absent": True}),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    daemon_request(scope, node, request["action"], request["payload"])
                ),
                flush=True,
            )
        return 0
    except BaseException as exc:
        print(json.dumps({"error_kind": type(exc).__name__}), flush=True)
        return 1
