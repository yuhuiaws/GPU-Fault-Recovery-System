"""Node-local admission and bounded reads for optional diagnostic evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Iterator

_READ_SLOTS = threading.BoundedSemaphore(4)


def publish_partial(agent: Any, document: dict[str, Any]) -> None:
    sink = getattr(agent, "partial_evidence_sink", None)
    if callable(sink):
        sink(document)


def bounded_read(read: Callable[[], Any], timeout: float = 1) -> Any:
    # A blocked filesystem read cannot be killed by Python. Keep its slot until
    # it actually exits, so repeated incidents cannot accumulate reader threads.
    if not _READ_SLOTS.acquire(blocking=False):
        raise TimeoutError("diagnostic read slots exhausted")
    result: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def worker() -> None:
        try:
            result.put((True, read()))
        except Exception as exc:
            result.put((False, exc))
        finally:
            _READ_SLOTS.release()

    threading.Thread(target=worker, name="hung-evidence-reader", daemon=True).start()
    try:
        ok, value = result.get(timeout=timeout)
    except Empty as exc:
        raise TimeoutError("diagnostic filesystem read timed out") from exc
    if not ok:
        raise value
    return value


@contextmanager
def collection_lease(agent: Any, command: Any) -> Iterator[str | None]:
    job, attempt = (
        command.parameters.get("job_id"),
        command.parameters.get("attempt_id"),
    )
    if not job or not attempt:
        yield None
        return
    root = Path(agent.diagnostic_output_dir)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(root / "hung-collection.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield "NODE_COLLECTION_BUSY"
            return
        path = root / "hung-collection-budget.json"
        now = time.time()
        try:
            state = json.loads(path.read_text()) if path.exists() else {}
            state = {
                key: stamps
                for key, stamps in state.items()
                if isinstance(stamps, list) and stamps and now - stamps[-1] < 300
            }
        except (OSError, ValueError, TypeError):
            yield "COLLECTION_BUDGET_UNAVAILABLE"
            return
        key = hashlib.sha256(
            json.dumps([job, attempt, command.operation.value]).encode()
        ).hexdigest()
        stamps = [stamp for stamp in state.get(key, []) if now - stamp < 300]
        if len(stamps) >= 2 or len(state) >= 256:
            yield "TASK_COLLECTION_RATE_LIMIT"
            return
        state[key] = [*stamps, now]
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(state))
        temporary.replace(path)
        yield None
    finally:
        os.close(descriptor)
