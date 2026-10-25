"""Bound parent wait time even when a diagnostic worker is stuck in kernel I/O."""

from __future__ import annotations

import contextlib
import json
import os
import selectors
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

from gpu_fault.node_agent import observation_guard
from gpu_fault.node_agent.observation_limits import OBSERVATION_REQUEST_BUDGET_BYTES

#: ``failure_stage`` of a result the supervisor could not obtain: the worker was
#: never started (the request itself was refused, or the process could not be
#: spawned), or it ran and its output could not be used.
WORKER_NOT_STARTED = "WORKER_NOT_STARTED"
WORKER_OUTPUT = "WORKER_OUTPUT"


class ObservationIncomplete(RuntimeError):
    """A node action whose evidence is incomplete; ``action_details`` is the result.

    The message carries the cause next to the status. Live, a bundle refused
    before its worker started reached the workflow as the bare
    ``ObservationIncomplete: UNAVAILABLE`` with ``error="ValueError"`` -- nobody
    could tell a refused request from a crashed probe.
    """

    def __init__(self, result: dict[str, Any]):
        status = str(result.get("status", "INCOMPLETE"))
        error = result.get("error")
        super().__init__(
            f"{status}: {error}" if isinstance(error, str) and error else status
        )
        self.action_details = result


def request_overflow(request: dict[str, Any], size: int) -> str:
    """Why a request is refused: its size, the budget and what weighs most in it.

    Only command parameters are weighed by name: that is the part of a request
    the control plane composes (a hung bundle's ``hung_triage_decision`` ran to
    259 KB live), and the name is what an operator needs to find the heavy
    field in the workflow record.
    """

    command = request.get("command")
    parameters = command.get("parameters") if isinstance(command, dict) else None
    detail = ""
    if isinstance(parameters, dict) and parameters:
        weights = sorted(
            ((len(json.dumps(value)), str(key)) for key, value in parameters.items()),
            reverse=True,
        )[:3]
        detail = "; largest parameters: " + ", ".join(
            f"{key}={weight} bytes" for weight, key in weights
        )
    return (
        "observation input exceeds budget: "
        f"{size} > {OBSERVATION_REQUEST_BUDGET_BYTES} bytes{detail}"
    )


class ObservationSupervisor:
    def __init__(self) -> None:
        self.slot = threading.BoundedSemaphore(1)
        self.last_result: dict[str, Any] = {}

    def run(
        self,
        request: dict[str, Any],
        timeout: float,
        *,
        release: Callable[[], None] = lambda: None,
    ) -> dict[str, Any]:
        if not self.slot.acquire(blocking=False):
            release()
            return {
                "status": "WORKER_STILL_RUNNING",
                "evidence_complete": False,
                "ranks": [],
            }
        process = None
        abandoned = False
        result: dict[str, Any] = {"status": "COLLECTING", "ranks": []}
        try:
            deadline = time.monotonic() + timeout
            with tempfile.TemporaryFile() as incoming:
                raw = json.dumps(request).encode()
                if len(raw) > OBSERVATION_REQUEST_BUDGET_BYTES:
                    raise ValueError(request_overflow(request, len(raw)))
                incoming.write(raw)
                incoming.seek(0)
                process = subprocess.Popen(
                    [sys.executable, "-m", observation_guard.__name__],
                    stdin=incoming,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env={
                        key: value
                        for key, value in os.environ.items()
                        if not key.startswith("GPU_FAULT_")
                    },
                )
            buffer = bytearray()
            received = 0
            final = False
            output = process.stdout
            if output is None:
                raise RuntimeError("observation output pipe was not created")
            with selectors.DefaultSelector() as selector:
                selector.register(output, selectors.EVENT_READ)
                while time.monotonic() < deadline:
                    for _, _ in selector.select(
                        min(0.1, max(0, deadline - time.monotonic()))
                    ):
                        block = os.read(output.fileno(), 65536)
                        if not block:
                            selector.unregister(output)
                            break
                        buffer.extend(block)
                        received += len(block)
                        if len(buffer) > 1_048_576 or received > 4 * 1_048_576:
                            raise ValueError("observation output exceeds budget")
                        while b"\n" in buffer:
                            line, _, rest = buffer.partition(b"\n")
                            buffer = bytearray(rest)
                            message = json.loads(line)
                            if not isinstance(message, dict) or not isinstance(
                                message.get("result"), dict
                            ):
                                raise ValueError("invalid observation result")
                            result, final = message["result"], message["final"] is True
                    if not selector.get_map():
                        break
                if not final or process.poll() is None:
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(
                            timeout=min(0.1, max(0.001, deadline - time.monotonic()))
                        )
                if not final or process.poll() != 0:
                    result = {
                        **result,
                        "status": "PARTIAL_TIMEOUT"
                        if process.poll() is None
                        else "WORKER_EXITED",
                        "evidence_complete": False,
                        "worker_returncode": process.poll(),
                    }
            self.last_result = result
            return result
        except (OSError, ValueError, KeyError) as exc:
            # The class alone said nothing (live: ``"error": "ValueError"`` for
            # a request the budget refused in 5 ms). Carry the message, which
            # component refused and whether a worker ever ran.
            result.update(
                status="UNAVAILABLE",
                evidence_complete=False,
                error=f"{type(exc).__name__}: {str(exc)[:384]}",
                failed_component="observation_supervisor",
                failure_stage=WORKER_NOT_STARTED if process is None else WORKER_OUTPUT,
                worker_started=process is not None,
            )
            self.last_result = result
            return result
        finally:
            if process is not None:
                if process.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=0.2)
                if process.stdout is not None:
                    process.stdout.close()
                if process.poll() is None:
                    abandoned = True
                    result.update(worker_tree_running=True, evidence_complete=False)

                    def reap() -> None:
                        process.wait()
                        release()
                        self.slot.release()

                    threading.Thread(
                        target=reap, name="observation-reaper", daemon=True
                    ).start()
            if not abandoned:
                release()
                self.slot.release()
