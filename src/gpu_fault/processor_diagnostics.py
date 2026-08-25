from __future__ import annotations

import faulthandler
import gc
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock, current_thread, get_native_id


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _InboundReplay:
    request_id: str
    owner_id: str
    path: str
    lane_epoch: int | None
    started: float
    phase: str
    phase_started: float
    thread_name: str
    native_thread_id: int


class ProcessorReplayTracker:
    """Process-local view of replay handlers currently executing."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: dict[str, _InboundReplay] = {}

    def start(
        self,
        request_id: str,
        *,
        owner_id: str,
        path: str,
        lane_epoch: int | None,
        phase: str,
    ) -> None:
        observed = time.monotonic()
        thread = current_thread()
        state = _InboundReplay(
            request_id=request_id,
            owner_id=owner_id,
            path=path,
            lane_epoch=lane_epoch,
            started=observed,
            phase=phase,
            phase_started=observed,
            thread_name=thread.name,
            native_thread_id=get_native_id(),
        )
        with self._lock:
            self._requests[request_id] = state

    def update(self, request_id: str, phase: str) -> None:
        observed = time.monotonic()
        thread = current_thread()
        with self._lock:
            state = self._requests.get(request_id)
            if state is None:
                return
            self._requests[request_id] = replace(
                state,
                phase=phase,
                phase_started=observed,
                thread_name=thread.name,
                native_thread_id=get_native_id(),
            )

    def finish(self, request_id: str) -> None:
        with self._lock:
            self._requests.pop(request_id, None)

    def snapshot(self) -> list[dict]:
        observed = time.monotonic()
        with self._lock:
            states = list(self._requests.values())
        return [
            {
                "request_id": state.request_id,
                "owner_id": state.owner_id,
                "path": state.path,
                "lane_epoch": state.lane_epoch,
                "phase": state.phase,
                "elapsed_seconds": max(0.0, observed - state.started),
                "phase_elapsed_seconds": max(0.0, observed - state.phase_started),
                "thread_name": state.thread_name,
                "native_thread_id": state.native_thread_id,
            }
            for state in states
        ]

    def phases(self) -> dict[str, dict[str, float | int]]:
        observed = time.monotonic()
        with self._lock:
            states = list(self._requests.values())
        phases: dict[str, dict[str, float | int]] = {}
        for state in states:
            phase = phases.setdefault(
                state.phase,
                {"count": 0, "oldest_seconds": 0.0},
            )
            phase["count"] += 1
            phase["oldest_seconds"] = max(
                float(phase["oldest_seconds"]),
                observed - state.phase_started,
            )
        return phases


class ProcessorDiagnosticsPublisher:
    """Shares all uvicorn worker snapshots through the Pod filesystem."""

    def __init__(
        self,
        directory: str,
        snapshot: Callable[[], dict],
        *,
        interval_seconds: float = 1.0,
        stale_seconds: float = 5.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("processor diagnostics interval must be positive")
        if stale_seconds < interval_seconds * 2:
            raise ValueError("processor diagnostics stale window is too short")
        self.directory = Path(directory)
        self.snapshot = snapshot
        self.interval_seconds = interval_seconds
        self.stale_seconds = stale_seconds
        self.path = self.directory / f"{os.getpid()}.json"

    def run(self, stop: Event) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            while not stop.is_set():
                try:
                    self.publish()
                except Exception:
                    LOGGER.exception("processor diagnostics publish failed")
                if stop.wait(self.interval_seconds):
                    break
        finally:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def publish(self) -> None:
        document = {
            "published_at": time.time(),
            **self.snapshot(),
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def read_all(self) -> list[dict]:
        observed = time.time()
        documents = []
        try:
            paths = list(self.directory.glob("*.json"))
        except OSError:
            return documents
        for path in paths:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                age = observed - float(document["published_at"])
                int(document["process"]["pid"])
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
            if 0 <= age <= self.stale_seconds:
                document["snapshot_age_seconds"] = age
                documents.append(document)
        return sorted(
            documents,
            key=lambda item: int(item["process"]["pid"]),
        )


_ACTIVE_REPLAY: ContextVar[tuple[ProcessorReplayTracker, str] | None] = ContextVar(
    "gpu_fault_active_processor_replay", default=None
)


def bind_processor_replay(
    tracker: ProcessorReplayTracker,
    request_id: str,
) -> Token:
    return _ACTIVE_REPLAY.set((tracker, request_id))


def reset_processor_replay(token: Token) -> None:
    _ACTIVE_REPLAY.reset(token)


def report_processor_replay_phase(phase: str) -> None:
    active = _ACTIVE_REPLAY.get()
    if active is not None:
        tracker, request_id = active
        tracker.update(request_id, phase)


def process_runtime_snapshot() -> dict:
    result: dict[str, object] = {
        "pid": os.getpid(),
        "gc_counts": list(gc.get_count()),
    }
    if hasattr(sys, "getallocatedblocks"):
        result["allocated_blocks"] = sys.getallocatedblocks()
    result.update(_read_memory_file("/proc/self/status"))
    result.update(
        _read_memory_file(
            "/proc/self/smaps_rollup",
            prefix="smaps_",
        )
    )
    return result


def _read_memory_file(
    path: str,
    *,
    prefix: str = "",
) -> dict[str, object]:
    allowed = {
        "VmRSS",
        "VmHWM",
        "VmSize",
        "Threads",
        "Rss",
        "Pss",
        "Private_Clean",
        "Private_Dirty",
        "Anonymous",
        "Swap",
    }
    try:
        with open(path, encoding="utf-8") as source:
            lines = source.read().splitlines()
    except OSError:
        return {}
    result: dict[str, object] = {}
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if not separator or key not in allowed:
            continue
        value = raw_value.strip()
        parts = value.split()
        normalized_key = prefix + _snake_case(key)
        if parts and parts[0].isdigit():
            parsed = int(parts[0])
            if len(parts) == 2 and parts[1].lower() == "kb":
                result[normalized_key + "_bytes"] = parsed * 1024
            else:
                result[normalized_key] = parsed
        else:
            result[normalized_key] = value
    return result


def _snake_case(value: str) -> str:
    output = []
    for index, character in enumerate(value):
        if index and character.isupper() and value[index - 1].islower():
            output.append("_")
        output.append(character.lower())
    return "".join(output)


def register_thread_dump_signal(name: str) -> int | None:
    normalized = name.strip().upper()
    if not normalized:
        return None
    if normalized not in {"SIGUSR1", "SIGUSR2"}:
        raise ValueError("processor thread dump signal must be SIGUSR1 or SIGUSR2")
    signum = int(getattr(signal, normalized))
    faulthandler.register(
        signum,
        all_threads=True,
        chain=False,
    )
    return signum


def unregister_thread_dump_signal(signum: int | None) -> None:
    if signum is not None:
        faulthandler.unregister(signum)
