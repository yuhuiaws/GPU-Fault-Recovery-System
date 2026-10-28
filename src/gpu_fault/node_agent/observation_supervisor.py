"""Bound parent wait time even when a diagnostic worker is stuck in kernel I/O."""

from __future__ import annotations

import contextlib
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

from gpu_fault.node_agent import observation_guard
from gpu_fault.node_agent.observation_limits import (
    OBSERVATION_DEADLINE_VARIABLE,
    OBSERVATION_REQUEST_BUDGET_BYTES,
    OBSERVATION_RESULT_BUDGET_BYTES,
    WORKER_NOT_STARTED,
    WORKER_OUTPUT,
)

__all__ = [
    "ObservationIncomplete",
    "ObservationSlotBusy",
    "ObservationSupervisor",
    "SLOT_BUSY",
    "SLOT_WAIT_FRACTION",
    "SLOT_WAIT_MAX_SECONDS",
    "WORKER_NOT_STARTED",
    "WORKER_OUTPUT",
    "request_overflow",
    "slot_wait_for",
]

#: Status of a request the supervisor could not give the node's single worker
#: slot. Kept under its historical name: records, the hand-off mail and the
#: acceptance runner read it.
SLOT_BUSY = "WORKER_STILL_RUNNING"
#: How long a control-plane command waits for the slot before it is refused: a
#: share of its own timeout, at most ``SLOT_WAIT_MAX_SECONDS``. The wait is
#: inside the timeout (the deadline runs from the request's arrival), so the
#: control plane's 30-second per-node wait is never exceeded; a triage waits
#: at most 8.4 s of its 28 and the worker plans within what is left.
SLOT_WAIT_FRACTION = 0.3
SLOT_WAIT_MAX_SECONDS = 10.0


def slot_wait_for(timeout: float) -> float:
    return round(min(SLOT_WAIT_MAX_SECONDS, max(0.0, timeout) * SLOT_WAIT_FRACTION), 3)


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


class ObservationSlotBusy(ObservationIncomplete):
    """The node's one worker slot stayed held for the whole wait; nothing ran.

    Live (GF-GENERAL-LIVE-011 ``fr_temporal_completion``, HyperPod, 2026-10-26
    04:21:31Z), COLLECT_HUNG_TRIAGE reached both nodes 1.45 s into a 30-second
    epoch -- while each Agent's passive communication sample held the slot --
    and both answered ``WORKER_STILL_RUNNING`` within 1.2 s without waiting or
    naming the holder; the control plane recorded both as failed and the
    decision saw no process. This refusal is a retryable failure: the executor
    says so in its result, so a resubmission of the same command runs a new
    attempt instead of replaying the refusal from the ledger.
    """


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
        self._holder_lock = threading.Lock()
        #: Who holds the slot: ``operation``, ``since`` (monotonic), ``process``,
        #: ``preemptible``, ``preempted_by`` and ``abandoned`` (its caller left
        #: at its deadline; the reaper frees the slot when the tree exits).
        self.holder: dict[str, Any] | None = None

    @staticmethod
    def _describe(holder: dict[str, Any]) -> dict[str, Any]:
        return {
            "operation": holder["operation"],
            "held_seconds": round(time.monotonic() - holder["since"], 3),
            "preempted": holder.get("preempted_by") is not None,
            "abandoned": bool(holder.get("abandoned")),
        }

    def _preempt(self, operation: str) -> dict[str, Any] | None:
        """Stop a preemptible holder's worker tree; the holder, described.

        The passive communication sample is the only preemptible activity: it
        runs every 30 s on its own, and one missed epoch costs nothing against
        a control-plane command that would otherwise be refused. A command's
        worker is never stopped for another command.
        """

        with self._holder_lock:
            holder = self.holder
            if holder is None:
                return None
            process = holder.get("process")
            if (
                holder.get("preemptible")
                and holder.get("preempted_by") is None
                and process is not None
                and process.poll() is None
            ):
                holder["preempted_by"] = operation
                with contextlib.suppress(OSError):
                    # The guard is its own session leader: the whole tree.
                    os.killpg(process.pid, signal.SIGTERM)
            return self._describe(holder)

    @staticmethod
    def _refusal(
        waited: float, slot_wait: float, holder: dict[str, Any] | None
    ) -> dict[str, Any]:
        if holder is None:
            held = "an activity that already left"
        else:
            held = f"{holder['operation']} for {holder['held_seconds']:.1f} s"
            if holder["preempted"]:
                held += " (preempted, not yet exited)"
            if holder["abandoned"]:
                held += " (abandoned by its caller at its deadline)"
        return {
            "status": SLOT_BUSY,
            "evidence_complete": False,
            "ranks": [],
            "error": f"observation worker slot held by {held}; waited {waited:.1f} s",
            "failed_component": "observation_supervisor",
            "failure_stage": WORKER_NOT_STARTED,
            "worker_started": False,
            "slot_holder": holder,
            "slot_wait_seconds": round(waited, 3),
            "slot_wait_allowed_seconds": round(slot_wait, 3),
        }

    def run(
        self,
        request: dict[str, Any],
        timeout: float,
        *,
        release: Callable[[], None] = lambda: None,
        preemptible: bool = False,
        slot_wait: float = 0.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + timeout
        operation = str(request.get("operation", "observation"))
        acquired = self.slot.acquire(blocking=False)
        seen: dict[str, Any] | None = None
        if not acquired and slot_wait > 0:
            seen = self._preempt(operation)
            acquired = self.slot.acquire(
                timeout=min(slot_wait, max(0.0, deadline - time.monotonic()))
            )
        waited = time.monotonic() - started if slot_wait > 0 else 0.0
        if not acquired:
            release()
            with self._holder_lock:
                holder = self._describe(self.holder) if self.holder else seen
            refusal = self._refusal(waited, slot_wait, holder)
            self.last_result = refusal
            return refusal
        process = None
        abandoned = False
        record: dict[str, Any] = {
            "operation": operation,
            "since": time.monotonic(),
            "process": None,
            "preemptible": preemptible,
            "preempted_by": None,
            "abandoned": False,
        }
        with self._holder_lock:
            self.holder = record
        result: dict[str, Any] = {"status": "COLLECTING", "ranks": []}
        try:
            with tempfile.TemporaryFile() as incoming:
                raw = json.dumps(request).encode()
                if len(raw) > OBSERVATION_REQUEST_BUDGET_BYTES:
                    raise ValueError(request_overflow(request, len(raw)))
                incoming.write(raw)
                incoming.seek(0)
                environment = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith("GPU_FAULT_")
                }
                # The deadline itself, not the seconds left now: the guard's
                # and the worker's start run on the same clock, and the worker
                # plans the triage from what is left when it plans it.
                environment[OBSERVATION_DEADLINE_VARIABLE] = repr(deadline)
                process = subprocess.Popen(
                    [sys.executable, "-m", observation_guard.__name__],
                    stdin=incoming,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=environment,
                )
            with self._holder_lock:
                record["process"] = process
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
                        if (
                            len(buffer) > OBSERVATION_RESULT_BUDGET_BYTES
                            or received > 4 * OBSERVATION_RESULT_BUDGET_BYTES
                        ):
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
                    left = max(0.001, deadline - time.monotonic())
                    # A final result and the end of the output are a worker
                    # tree on its way out (the guard closes the pipe as it
                    # exits): wait for it up to the deadline. The 0.1 s that
                    # bounds every other wait here turned complete results
                    # into PARTIAL_TIMEOUT with evidence_complete false on a
                    # CPU-starved host, where the exiting process stayed off
                    # the CPU longer than that between closing its stdout and
                    # being reaped (48 busy loops on its CPU, b214).
                    exiting = final and not selector.get_map()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=left if exiting else min(0.1, left))
                if not final or process.poll() != 0:
                    with self._holder_lock:
                        preempted_by = record.get("preempted_by")
                    # The deadline hands back the worker's last checkpoint
                    # (``publish(..., final=False)``): the ranks it found, the
                    # dumps it read and the stage it was in.
                    result = {
                        **result,
                        "status": "PREEMPTED"
                        if preempted_by
                        else "PARTIAL_TIMEOUT"
                        if process.poll() is None
                        else "WORKER_EXITED",
                        "evidence_complete": False,
                        "worker_returncode": process.poll(),
                        "supervisor_timeout_seconds": timeout,
                    }
                    if preempted_by:
                        result["preempted_by"] = preempted_by
            if waited >= 0.01:
                result = {**result, "slot_wait_seconds": round(waited, 3)}
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
            if waited >= 0.01:
                result["slot_wait_seconds"] = round(waited, 3)
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
                    with self._holder_lock:
                        record["abandoned"] = True

                    def reap() -> None:
                        process.wait()
                        with self._holder_lock:
                            if self.holder is record:
                                self.holder = None
                        release()
                        self.slot.release()

                    threading.Thread(
                        target=reap, name="observation-reaper", daemon=True
                    ).start()
            if not abandoned:
                with self._holder_lock:
                    if self.holder is record:
                        self.holder = None
                release()
                self.slot.release()
