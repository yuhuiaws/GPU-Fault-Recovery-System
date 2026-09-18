from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from tests.admin.test_api_budget_handoff_model import ProcessModel


@dataclass
class ProcessRecord:
    parent: int
    start: str
    state: str = "S"


class ProcessPath:
    def __init__(self, model, value):
        self.model = model
        self.path = Path(value)
        self.name = self.path.name

    def __truediv__(self, value):
        return ProcessPath(self.model, self.path / value)

    def stat(self):
        pid = int(self.path.parts[2])
        if pid in self.model.errors:
            raise self.model.errors[pid]
        if pid not in self.model.records:
            raise FileNotFoundError("modeled exited process")
        return SimpleNamespace()

    def read_text(self):
        self.stat()
        pid = int(self.path.parts[2])
        record = self.model.records[pid]
        fields = [
            record.state,
            str(record.parent),
            "0",
            "0",
            *(["0"] * 15),
            record.start,
        ]
        return f"{pid} (owned fake CLI) " + " ".join(fields)

    def iterdir(self):
        pid = int(self.path.parts[2])
        return iter([ProcessPath(self.model, self.path / str(pid))])


class RecoveryProcesses(ProcessModel):
    """Virtual procfs and pidfds; signals cannot reach the operating system."""

    def __init__(self):
        super().__init__()
        self.records = {
            1: ProcessRecord(0, "1"),
            100: ProcessRecord(1, "10"),
            101: ProcessRecord(100, "11"),
            102: ProcessRecord(101, "12"),
            103: ProcessRecord(102, "13"),
            999: ProcessRecord(1, "99"),
        }
        self.errors = {}
        self.pid = 102
        self.descriptors = {}
        self.signal_targets = []
        self.os.getpid = lambda: self.pid
        self.os.getppid = lambda: self.records[self.pid].parent

    def path(self, value):
        return (
            ProcessPath(self, value) if str(value).startswith("/proc/") else Path(value)
        )

    def open_pidfd(self, pid):
        assert pid in self.records, "pidfd requested for a non-modeled process"
        descriptor = 10000 + pid
        self.descriptors[descriptor] = pid
        return descriptor

    def close(self, descriptor):
        assert descriptor in self.descriptors, "attempted to close a host descriptor"
        self.descriptors.pop(descriptor)
        self.closed.append(descriptor)

    def send_signal(self, descriptor, signum):
        assert descriptor in self.descriptors, (
            "attempted to signal an unowned descriptor"
        )
        pid = self.descriptors[descriptor]
        self.signals.append(signum)
        self.signal_targets.append(pid)
        self.records[pid].state = {
            signal.SIGSTOP: "T",
            signal.SIGCONT: "S",
            signal.SIGKILL: "Z",
        }[signum]


class Clock:
    def __init__(self):
        self.value = time.monotonic()
        self.advance = 3601
        self.on_sleep = None

    def monotonic(self):
        return self.value

    def sleep(self, _seconds):
        if self.on_sleep is not None:
            callback, self.on_sleep = self.on_sleep, None
            callback()
        self.value += self.advance
