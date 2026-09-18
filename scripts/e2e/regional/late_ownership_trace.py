"""Independent Linux exec-trace accounting for the acceptance node probes.

Only sanitized exec identities leave this module. Trace gaps, unknown syscall
syntax, unfinished execs, missing exits and uncalibrated captures are refusals,
not evidence of zero actions. The observer must attach before STOP is allowed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import select
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied, process_identity
from scripts.e2e.regional.late_ownership_contract import (
    AcceptanceScope,
    NodeIdentity,
    PhysicalAction,
    ProcessIdentity,
    QuiescenceReceipt,
    WitnessEnd,
    WitnessStart,
)

TRACE_LIMIT = 4 * 1024 * 1024
_LINE = re.compile(r"^(\d+)\s+(\d+)\.(\d{6,9})\s+(.*)$")
_HEX_STRING = r'"(?:\\x[0-9a-f]{2})*"'
_EXEC = re.compile(
    rf"^execve\(({_HEX_STRING}), (\[(?:{_HEX_STRING}(?:, )?)*\]), "
    r"(?:0x[0-9a-f]+|NULL|/\* \d+ vars \*/)(?: /\* \d+ vars \*/)?\) "
    r"= (0|-1(?: [A-Z0-9_]+ \([^)\n]+\))?)$"
)
_EXIT = re.compile(r"^\+\+\+ exited with (\d+) \+\+\+$")
_SIGNALLED = re.compile(
    r"^\+\+\+ killed by (SIG[A-Z0-9]+)(?: \(core dumped\))? \+\+\+$"
)
TRACER_CHILD = Path(__file__).with_name("probes") / "late_ownership_tracer_child.py"


@dataclass(frozen=True)
class ExecEvent:
    pid: int
    started_ns: int
    ended_ns: int
    executable: str
    argv: tuple[str, ...]
    returncode: int


def _decode_literal(text: str) -> str:
    value = ast.literal_eval(text)
    if not isinstance(value, str) or "\x00" in value:
        raise BoundaryDenied("exec trace contains an invalid argument")
    return value


def parse_exec_trace(data: bytes) -> tuple[ExecEvent, ...]:
    """Parse the deliberately fixed ``strace -f -ttt -xx`` exec-only format."""
    if len(data) > TRACE_LIMIT or (data and not data.endswith(b"\n")):
        raise BoundaryDenied("exec trace is oversized or incomplete")
    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeDecodeError:
        raise BoundaryDenied("exec trace is not in the pinned format") from None
    unfinished: dict[int, tuple[int, str]] = {}
    running: dict[int, tuple[int, str, tuple[str, ...]]] = {}
    events: list[ExecEvent] = []
    last_ns: dict[int, int] = {}
    for line in lines:
        match = _LINE.fullmatch(line)
        if match is None:
            raise BoundaryDenied("exec trace contains an unaccounted record")
        pid = int(match[1])
        stamp = int(match[2]) * 1_000_000_000 + int(match[3].ljust(9, "0"))
        if pid <= 0 or stamp < last_ns.get(pid, 0):
            raise BoundaryDenied("exec trace process clock or identity changed")
        last_ns[pid] = stamp
        body = match[4]
        if body.endswith(" <unfinished ...>"):
            if pid in unfinished or not body.startswith("execve("):
                raise BoundaryDenied("exec trace has an ambiguous unfinished syscall")
            unfinished[pid] = (stamp, body.removesuffix(" <unfinished ...>"))
            continue
        if body.startswith("<... execve resumed>"):
            if pid not in unfinished:
                raise BoundaryDenied("exec trace resumed an unknown syscall")
            started, prefix = unfinished.pop(pid)
            body = prefix + body.removeprefix("<... execve resumed>")
        else:
            started = stamp
        call = _EXEC.fullmatch(body)
        if call is not None:
            if pid in running:
                raise BoundaryDenied("exec trace contains an unbounded exec chain")
            executable = _decode_literal(call[1])
            raw_args = ast.literal_eval(call[2])
            args = tuple(_decode_literal(repr(item)) for item in raw_args)
            if not executable.startswith("/") or not args:
                raise BoundaryDenied("exec trace target is not an absolute executable")
            if call[3] != "0":
                events.append(ExecEvent(pid, started, stamp, executable, args, -1))
            else:
                running[pid] = (started, executable, args)
            continue
        exit_match = _EXIT.fullmatch(body)
        signal_match = _SIGNALLED.fullmatch(body)
        if exit_match is None and signal_match is None:
            raise BoundaryDenied("exec trace syscall or output is unsupported")
        if pid in unfinished:
            raise BoundaryDenied("trace process exited during an unfinished exec")
        if pid not in running:
            continue
        code = int(exit_match[1]) if exit_match is not None else -1
        began, executable, args = running.pop(pid)
        events.append(ExecEvent(pid, began, stamp, executable, args, code))
    if unfinished or running:
        raise BoundaryDenied("exec trace has not drained every observed exec")
    return tuple(events)


def physical_actions(
    events: tuple[ExecEvent, ...],
    *,
    nvidia_smi: Path,
    calibration_argv: tuple[str, ...],
) -> tuple[PhysicalAction, ...]:
    """Recognize the physical reset executable, not a model's action counter.

    All other execs except the exact calibration are rejected. The drill owns
    this quiet action interval, so an unexpected tool cannot be waved away as
    background activity or an alternative reset spelling.
    """
    if not nvidia_smi.is_absolute() or not calibration_argv:
        raise BoundaryDenied("exec witness has no pinned executable and calibration")
    if len({(event.pid, event.started_ns, event.ended_ns) for event in events}) != len(
        events
    ):
        raise BoundaryDenied("exec witness contains duplicated physical records")
    calibrations = 0
    actions = []
    for event in events:
        if event.executable != str(nvidia_smi):
            raise BoundaryDenied(
                "unapproved executable entered the observed action interval"
            )
        if event.argv == calibration_argv:
            if event.returncode != 0:
                raise BoundaryDenied("exec witness calibration did not complete")
            calibrations += 1
            continue
        args = event.argv[1:]
        if args in {
            (
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ),
            ("--query-gpu=uuid,index", "--format=csv,noheader,nounits"),
            ("-q", "-x"),
        }:
            continue
        operation: Literal["RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES"]
        if args == ("--gpu-reset",):
            operation = "RESET_ALL_GPUS_NVSWITCHES"
        elif len(args) == 3 and args[:2] == ("--gpu-reset", "-i") and args[2]:
            operation = "RESET_GPU"
        else:
            raise BoundaryDenied("unknown NVIDIA invocation cannot prove no action")
        actions.append(
            PhysicalAction(
                operation=operation,
                argv_sha256=hashlib.sha256(
                    json.dumps(event.argv).encode("ascii")
                ).hexdigest(),
                pid=event.pid,
                started_ns=event.started_ns,
                ended_ns=event.ended_ns,
                returncode=event.returncode,
            )
        )
    if calibrations < 1:
        raise BoundaryDenied("exec witness requires a physical calibration")
    return tuple(actions)


def tracer_pids(tracee: ProcessIdentity) -> set[int]:
    if process_identity(tracee.pid) != tracee:
        raise BoundaryDenied("observed Agent process was replaced")
    tasks = tuple((Path("/proc") / str(tracee.pid) / "task").iterdir())
    if not tasks:
        raise BoundaryDenied("observed Agent has no verifiable threads")
    values: set[int] = set()
    for task in tasks:
        fields = dict(
            line.split(":", 1)
            for line in (task / "status").read_text(encoding="ascii").splitlines()
            if ":" in line
        )
        values.add(int(fields["TracerPid"].strip()))
    return values


class AttachedExecWitness:
    """A bounded tracer of an existing, explicitly pinned Agent process.

    The caller owns the private directory, the tracer child and its cleanup.
    This class never stops, restarts, or signals the tracee. Closing the tracer
    without a complete terminal receipt cannot be promoted to a no-action proof.
    """

    def __init__(
        self,
        directory: Path,
        tracee: ProcessIdentity,
        *,
        deadline: float,
        executable: Path | None = None,
    ) -> None:
        self.directory = directory
        self.tracee = tracee
        self.deadline = deadline
        self.child: subprocess.Popen[bytes] | None = None
        self.raw: BinaryIO | None = None
        self.tracer: ProcessIdentity | None = None
        self.pidfd: int | None = None
        self.armed = False
        self.closed = False
        self.baseline: bytes | None = None
        self.start_released = False
        self.executable = executable

    def start(self) -> None:
        if self.child is not None or self.closed:
            raise BoundaryDenied("exec witness cannot be reused")
        if not math_is_valid_deadline(self.deadline):
            raise BoundaryDenied("exec witness deadline is invalid")
        info = self.directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            raise BoundaryDenied("exec witness directory is not private")
        if tracer_pids(self.tracee) != {0}:
            raise BoundaryDenied("Agent is already traced")
        path = self.directory / "late-ownership-exec.raw"
        descriptor = os.open(
            path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        self.raw = os.fdopen(descriptor, "w+b")
        # Anonymous after open: raw argv never becomes an artifact or survives
        # the last owner. strace opens only this inherited regular-file handle.
        gate_read, gate_write = -1, -1
        try:
            path.unlink()
            gate_read, gate_write = os.pipe2(os.O_CLOEXEC)
            self.child = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(TRACER_CHILD),
                    str(os.getpid()),
                    str(gate_read),
                    str(TRACE_LIMIT + 1),
                    "strace",
                    "-f",
                    "-ttt",
                    "-xx",
                    "-s",
                    "65536",
                    "-e",
                    "trace=execve,execveat",
                    "-e",
                    "signal=none",
                    *(
                        ["-P", str(self.executable)]
                        if self.executable is not None
                        else []
                    ),
                    "-o",
                    f"/proc/self/fd/{descriptor}",
                    "-p",
                    str(self.tracee.pid),
                ],
                pass_fds=(descriptor, gate_read),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                close_fds=True,
            )
            self.pidfd = os.pidfd_open(self.child.pid)
            self.tracer = process_identity(self.child.pid)
            os.write(gate_write, b"G")
            self.start_released = True
            self._await_attachment()
            self.armed = True
        except BaseException:
            if gate_write >= 0:
                os.close(gate_write)
                gate_write = -1
            self.close()
            raise
        finally:
            if gate_read >= 0:
                os.close(gate_read)
            if gate_write >= 0:
                os.close(gate_write)

    def _await_attachment(self) -> None:
        child = self.child
        if child is None or child.stderr is None:
            raise BoundaryDenied("exec witness did not start")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or not select.select([child.stderr], [], [], remaining)[0]:
            raise BoundaryDenied("exec witness attachment timed out")
        diagnostic = os.read(child.stderr.fileno(), 4096)
        if (
            child.poll() is not None
            or len(diagnostic) == 4096
            or not diagnostic.startswith(b"strace: Process ")
            or b" attached" not in diagnostic
            or tracer_pids(self.tracee) != {child.pid}
        ):
            raise BoundaryDenied("exec witness could not prove complete attachment")

    def check(self) -> None:
        child = self.child
        if not self.armed or self.closed or child is None or self.raw is None:
            raise BoundaryDenied("exec witness is not armed")
        if (
            not math_is_valid_deadline(self.deadline)
            or child.poll() is not None
            or process_identity(child.pid) != self.tracer
            or tracer_pids(self.tracee) != {child.pid}
            or os.fstat(self.raw.fileno()).st_size > TRACE_LIMIT
        ):
            raise BoundaryDenied("exec witness continuity was lost")

    def snapshot(self) -> bytes:
        self.check()
        if self.raw is None:
            raise BoundaryDenied("exec witness raw handle is unavailable")
        size = os.fstat(self.raw.fileno()).st_size
        data = os.pread(self.raw.fileno(), size, 0)
        self.check()
        if len(data) != size:
            raise BoundaryDenied("exec witness capture is incomplete")
        return data

    def start_receipt(
        self,
        scope: AcceptanceScope,
        node: NodeIdentity,
        *,
        witness_id: str,
        nvidia_smi: Path,
        calibration_argv: tuple[str, ...],
    ) -> WitnessStart:
        if self.baseline is not None or node not in scope.nodes:
            raise BoundaryDenied("witness calibration was reused or escaped scope")
        raw = self.snapshot()
        events = parse_exec_trace(raw)
        if physical_actions(
            events, nvidia_smi=nvidia_smi, calibration_argv=calibration_argv
        ):
            raise BoundaryDenied("physical action preceded the STOP rendezvous")
        if self.tracee.boot_id != node.boot_id:
            raise BoundaryDenied("witness was attached on the wrong node incarnation")
        self.baseline = raw
        with nvidia_smi.open("rb") as source:
            executable_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        return WitnessStart(
            scope_sha256=scope.digest(),
            executor_uid=scope.executor_uid,
            producer=process_identity(os.getpid()),
            node=node,
            witness_id=witness_id,
            sequence=1,
            calibration_execs=1,
            observer="linux-exec-trace",
            tracee=self.tracee,
            executable_path=str(nvidia_smi),
            executable_sha256=executable_sha256,
        )

    def finish_receipt(
        self,
        start: WitnessStart,
        quiet: QuiescenceReceipt,
        *,
        nvidia_smi: Path,
        calibration_argv: tuple[str, ...],
    ) -> WitnessEnd:
        if (
            self.baseline is None
            or start.producer != process_identity(os.getpid())
            or start.tracee != self.tracee
            or quiet.scope_sha256 != start.scope_sha256
            or quiet.executor_uid != start.executor_uid
            or quiet.open_commands
            or quiet.pending_callbacks
            or not quiet.workflow_terminal
            or not quiet.gate_revoked
        ):
            raise BoundaryDenied("witness closure has no exact quiescence proof")
        raw = self.snapshot()
        if not raw.startswith(self.baseline):
            raise BoundaryDenied("witness trace prefix changed")
        events = parse_exec_trace(raw)
        with nvidia_smi.open("rb") as source:
            executable_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        if (
            start.executable_path != str(nvidia_smi)
            or start.executable_sha256 != executable_sha256
        ):
            raise BoundaryDenied("physical reset executable changed during observation")
        actions = physical_actions(
            events, nvidia_smi=nvidia_smi, calibration_argv=calibration_argv
        )
        self.check()
        receipt = WitnessEnd(
            scope_sha256=start.scope_sha256,
            executor_uid=start.executor_uid,
            producer=start.producer,
            node=start.node,
            witness_id=start.witness_id,
            start_sha256=start.digest(),
            quiescence_sha256=quiet.digest(),
            sequence=start.sequence + 1,
            tracee=self.tracee,
            executable_path=start.executable_path,
            executable_sha256=executable_sha256,
            lost_events=0,
            trace_complete=True,
            trace_sha256=hashlib.sha256(raw).hexdigest(),
            trace_bytes=len(raw),
            exec_events=len(events),
            actions=actions,
        )
        self.close()
        return receipt

    def close(self) -> None:
        self.closed = True
        child = self.child
        try:
            if child is not None:
                if child.poll() is None:
                    if not self.start_released:
                        try:
                            child.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            raise BoundaryDenied(
                                "unreleased tracer child did not exit"
                            ) from None
                        return
                    if process_identity(child.pid) != self.tracer:
                        raise BoundaryDenied("cannot clean up a replaced tracer")
                    if self.pidfd is None:
                        raise BoundaryDenied("tracer cleanup has no pinned pidfd")
                    signal.pidfd_send_signal(self.pidfd, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if self.pidfd is not None:
                        signal.pidfd_send_signal(self.pidfd, signal.SIGKILL)
                        child.wait(timeout=3)
                    raise BoundaryDenied(
                        "exec witness cleanup did not finish"
                    ) from None
        finally:
            if child is not None and child.stderr is not None:
                child.stderr.close()
            if self.raw is not None:
                self.raw.close()
                self.raw = None
            if self.pidfd is not None:
                os.close(self.pidfd)
                self.pidfd = None


def math_is_valid_deadline(deadline: float) -> bool:
    return deadline < float("inf") and deadline > time.monotonic()
