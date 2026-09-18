"""Read-only, continuously supervised Agent exec witness for DESTR-015."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import select
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.e2e.regional.destr015_physical_evidence import (
    CLOCK_MARGIN_NS,
    MAX_CLOCK_DRIFT_PPM,
    ResetIntervalScope,
    evidence_digest,
)
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied, process_identity
from scripts.e2e.regional.late_ownership_trace import (
    AttachedExecWitness,
    parse_exec_trace,
    physical_actions,
)

MAX_MESSAGE_BYTES = 65536
QUERY_ARGS = (
    "--query-compute-apps=gpu_uuid,pid,process_name",
    "--format=csv,noheader,nounits",
)


class ClockEnvelope:
    def __init__(self) -> None:
        self.previous: tuple[int, int] | None = None
        self.minimum: int | None = None
        self.maximum: int | None = None

    def sample(self) -> dict[str, int]:
        before, wall, after = time.monotonic_ns(), time.time_ns(), time.monotonic_ns()
        if after < before or after - before > CLOCK_MARGIN_NS:
            raise BoundaryDenied("physical witness clock sampling was interrupted")
        minimum, maximum = wall - after, wall - before
        if self.previous is not None:
            old_mono, old_wall = self.previous
            elapsed = before - old_mono
            permitted = elapsed * MAX_CLOCK_DRIFT_PPM // 1_000_000 + CLOCK_MARGIN_NS
            if elapsed < 0 or abs((wall - old_wall) - elapsed) > permitted:
                raise BoundaryDenied("physical witness realtime clock jumped")
        self.previous = (before, wall)
        self.minimum = minimum if self.minimum is None else min(self.minimum, minimum)
        self.maximum = maximum if self.maximum is None else max(self.maximum, maximum)
        return {"monotonic_ns": after}


def reset_events(
    raw: bytes, *, executable: Path, gpu_uuid: str
) -> tuple[list[dict[str, Any]], int]:
    events = parse_exec_trace(raw)
    physical_actions(
        events,
        nvidia_smi=executable,
        calibration_argv=("nvidia-smi", *QUERY_ARGS),
    )
    resets: list[dict[str, Any]] = []
    calibration_ends: list[int] = []
    for event in events:
        if event.executable != str(executable):
            raise BoundaryDenied("physical witness observed an unpinned executable")
        args = event.argv[1:]
        if args == QUERY_ARGS and event.returncode == 0:
            calibration_ends.append(event.ended_ns)
        if "--gpu-reset" not in args:
            continue
        if args != ("--gpu-reset", "-i", gpu_uuid):
            raise BoundaryDenied("physical reset escaped its approved GPU")
        resets.append(
            {
                "operation": "RESET_GPU",
                "gpu_uuid": gpu_uuid,
                "pid": event.pid,
                "started_ns": event.started_ns,
                "ended_ns": event.ended_ns,
                "returncode": event.returncode,
            }
        )
    if not resets or not any(
        ended <= min(reset["started_ns"] for reset in resets)
        for ended in calibration_ends
    ):
        raise BoundaryDenied("physical reset trace has no completed query calibration")
    return resets, len(calibration_ends)


def main() -> int:
    initial = sys.stdin.readline(MAX_MESSAGE_BYTES + 1)
    if not initial.endswith("\n") or len(initial.encode()) > MAX_MESSAGE_BYTES:
        raise BoundaryDenied("physical witness arm request is incomplete")
    message = json.loads(initial)
    scope = ResetIntervalScope.model_validate_json(json.dumps(message["payload"]))
    if (
        set(message) != {"kind", "scope_sha256", "payload"}
        or message["kind"] != "arm"
        or message["scope_sha256"] != scope.digest()
        or scope.maintenance_end.tzinfo is None
    ):
        raise BoundaryDenied("physical witness arm request is unbound")
    seconds = (scope.maintenance_end - datetime.now(timezone.utc)).total_seconds()
    if not 0 < seconds <= 7200:
        raise BoundaryDenied("physical witness requires a bounded maintenance window")
    executable_text = shutil.which("nvidia-smi")
    if not executable_text:
        raise BoundaryDenied("physical reset executable is unavailable")
    executable = Path(executable_text).resolve()
    pid_text = subprocess.run(
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
    if not pid_text.isdecimal():
        raise BoundaryDenied("physical witness has no Agent process identity")
    tracee = process_identity(int(pid_text))
    if tracee.boot_id != scope.boot_id:
        raise BoundaryDenied("physical witness Agent boot differs")
    clock = ClockEnvelope()
    sequence = 0

    def emit(kind: str, payload: dict[str, Any]) -> None:
        nonlocal sequence
        sequence += 1
        value = json.dumps(
            {
                "kind": kind,
                "scope_sha256": scope.digest(),
                "sequence": sequence,
                "payload": payload,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(value.encode()) > MAX_MESSAGE_BYTES:
            raise BoundaryDenied("physical witness response is oversized")
        print(value, flush=True)

    with tempfile.TemporaryDirectory(prefix="gpu-fault-reset-witness-") as directory:
        witness = AttachedExecWitness(
            Path(directory),
            tracee,
            deadline=time.monotonic() + seconds + 120,
            executable=executable,
        )
        start = {
            "scope_sha256": scope.digest(),
            "witness_id": secrets.token_hex(24),
            "tracee": tracee.model_dump(mode="json"),
            "producer": process_identity(os.getpid()).model_dump(mode="json"),
            "executable_path": str(executable),
            "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        }
        try:
            witness.start()
            emit("armed", {"start": start, **clock.sample()})
            while True:
                clock.sample()
                witness.check()
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                line = sys.stdin.readline(MAX_MESSAGE_BYTES + 1)
                if not line.endswith("\n") or len(line.encode()) > MAX_MESSAGE_BYTES:
                    raise BoundaryDenied("physical witness controller was lost")
                request = json.loads(line)
                if (
                    set(request) != {"kind", "scope_sha256", "payload"}
                    or request["scope_sha256"] != scope.digest()
                    or request["payload"] != {}
                ):
                    raise BoundaryDenied("physical witness request is unbound")
                if request["kind"] == "clock":
                    emit("clock", clock.sample())
                elif request["kind"] == "abort":
                    witness.close()
                    emit("closed", {"closed": True, "proof_complete": False})
                    return 0
                elif request["kind"] == "finish":
                    raw = witness.snapshot()
                    actions, calibration = reset_events(
                        raw,
                        executable=executable,
                        gpu_uuid=scope.gpu_uuid,
                    )
                    if (
                        hashlib.sha256(executable.read_bytes()).hexdigest()
                        != start["executable_sha256"]
                    ):
                        raise BoundaryDenied("physical reset executable changed")
                    stamp = clock.sample()
                    witness.close()
                    emit(
                        "finished",
                        {
                            **stamp,
                            "end": {
                                **start,
                                "start_sha256": evidence_digest(start),
                                "trace_complete": True,
                                "closed": True,
                                "lost_events": 0,
                                "trace_sha256": hashlib.sha256(raw).hexdigest(),
                                "trace_bytes": len(raw),
                                "calibration_execs": calibration,
                                "actions": actions,
                                "wall_minus_monotonic_min_ns": clock.minimum,
                                "wall_minus_monotonic_max_ns": clock.maximum,
                            },
                        },
                    )
                    return 0
                else:
                    raise BoundaryDenied("unsupported physical witness request")
        finally:
            witness.close()


if __name__ == "__main__":
    raise SystemExit(main())
