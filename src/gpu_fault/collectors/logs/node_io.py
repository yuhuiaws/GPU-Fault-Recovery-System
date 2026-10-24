"""One isolated file reader per collector; stalled mounts cannot block journald."""

from __future__ import annotations

import copy
import queue
import threading
from collections.abc import Iterator
from datetime import datetime
from typing import Any


def journal_lines(source: Any, max_bytes: int) -> Iterator[str]:
    """Bound allocation before JSON decoding, including a record without LF."""
    while True:
        line = source.readline(max(1, max_bytes // 4))
        if not line:
            return
        yield line


class TrainingReadWorker:
    def __init__(self, timeout: float = 2.0) -> None:
        if timeout <= 0:
            raise ValueError("training I/O timeout must be positive")
        self.timeout = timeout
        self.results: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.thread: threading.Thread | None = None

    def read(self, reader: Any, at: datetime, budget: Any) -> list[Any]:
        if self.thread is None:
            clone = copy.copy(reader)
            clone.files = copy.deepcopy(reader.files)
            clone.pending = copy.deepcopy(reader.pending)
            errors: list[str] = []
            discards: list[tuple[str, int]] = []
            clone.record_error = errors.append
            clone.record_discard = lambda reason, amount=1: discards.append(
                (reason, amount)
            )

            def run() -> None:
                try:
                    result = clone.read(at, budget=budget)
                    self.results.put((clone, result, errors, discards, None))
                except Exception as error:
                    self.results.put((clone, [], errors, discards, error))

            self.thread = threading.Thread(
                target=run, name="node-log-file-reader", daemon=True
            )
            self.thread.start()
        try:
            clone, entries, errors, discards, error = self.results.get(
                timeout=self.timeout
            )
        except queue.Empty:
            reader.record_error(
                "training log file I/O deadline exceeded; coverage UNKNOWN; "
                "one pending reader retained, no additional worker started"
            )
            return []
        self.thread.join()
        self.thread = None
        if error is not None:
            raise error
        for field in (
            "files",
            "pending",
            "resume_after",
            "deferred_rounds",
            "_resync_warned",
            "_partial_resume_warned",
        ):
            setattr(reader, field, getattr(clone, field))
        for message in errors:
            reader.record_error(message)
        for reason, count in discards:
            reader.record_discard(reason, count)
        return list(entries)
