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


class ObservationIncomplete(RuntimeError):
    def __init__(self, result: dict[str, Any]):
        super().__init__(str(result.get("status", "INCOMPLETE")))
        self.action_details = result


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
                if len(raw) > 262144:
                    raise ValueError("observation input exceeds budget")
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
            result.update(
                status="UNAVAILABLE", evidence_complete=False, error=type(exc).__name__
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
