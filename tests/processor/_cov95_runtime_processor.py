"""Cooperative scheduler around the real coordinator and in-memory store."""

from __future__ import annotations

import json
from concurrent.futures import Future
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorPoolSettings,
)
from gpu_fault.store import InMemoryStore
from tests._builders import processor_request
from tests.collectors import _cov95_runtime_collect as collect_support

isolated_runtime = collect_support.isolated_runtime


class Scheduler:
    def __init__(self):
        self.clock = collect_support.Clock()
        self.waits = 0
        self.wait_limit = 2
        self.on_wait = None
        self.on_submit = None
        self.thread_start_error = None
        self.submit_error = None
        self.defer = False
        self.threads = []
        self.pools = []
        self.submitted = []
        self.processor = None

    def event(self):
        return ManualEvent(self)

    def thread(self, **kwargs):
        thread = ManualThread(self, **kwargs)
        self.threads.append(thread)
        return thread

    def executor(self, **kwargs):
        executor = ImmediatePool(self, **kwargs)
        self.pools.append(executor)
        return executor

    def make_processor(self, store=None, **kwargs):
        defaults = {
            "owner_id": "runtime-owner",
            "internal_token": "test-replay",
            "active_consumers": True,
            "pools": ProcessorPoolSettings(
                fault_worker_count=1,
                observation_worker_count=1,
                gpu_telemetry_worker_count=1,
                host_telemetry_worker_count=1,
            ),
            "lease": ProcessorLeaseSettings(poll_seconds=0.01),
        }
        self.processor = ProcessorCoordinator(
            store if store is not None else InMemoryStore(), **(defaults | kwargs)
        )
        return self.processor

    def wait(self, event, timeout):
        if event.flag:
            return True
        self.clock.sleep(timeout or 0)
        self.waits += 1
        if self.on_wait is not None:
            self.on_wait(event, self.waits)
        elif self.waits >= self.wait_limit and self.processor is not None:
            self.processor.stop()
        return event.flag


class ManualEvent:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.flag = False

    def is_set(self):
        return self.flag

    def set(self):
        self.flag = True

    def clear(self):
        self.flag = False

    def wait(self, timeout=None):
        return self.scheduler.wait(self, timeout)


class ManualThread:
    def __init__(self, scheduler, *, target, args=(), name="", daemon=False):
        self.scheduler = scheduler
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.started = False
        self.joined = False

    def start(self):
        if self.scheduler.thread_start_error is not None:
            raise self.scheduler.thread_start_error
        self.started = True

    def run(self):
        return self.target(*self.args)

    def join(self, timeout=None):
        self.joined = True


class ImmediatePool:
    def __init__(self, scheduler, *, max_workers, thread_name_prefix):
        self.scheduler = scheduler
        self.max_workers = max_workers
        self.name = thread_name_prefix
        self.closed = False

    def submit(self, function, *args):
        if self.scheduler.submit_error is not None:
            raise self.scheduler.submit_error
        future = Future()
        self.scheduler.submitted.append((function.__name__, args, future))
        if not self.scheduler.defer:
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)
        if self.scheduler.on_submit is not None:
            self.scheduler.on_submit(future)
        return future

    def shutdown(self, *, wait, cancel_futures):
        self.closed = wait and cancel_futures
        for _, _, future in self.scheduler.submitted:
            if not future.done():
                future.cancel()


@pytest.fixture
def runtime(monkeypatch, isolated_runtime):
    scheduler = Scheduler()
    clock = scheduler.clock

    class WallClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now().astimezone(tz or timezone.utc)

    fake_time = SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
    for module in (
        "gpu_fault.processor.coordinator",
        "gpu_fault.processor.telemetry_spool",
        "gpu_fault.processor.metrics",
        "gpu_fault.processor.lane_runtime",
        "gpu_fault.processor.replay_completion",
    ):
        monkeypatch.setattr(f"{module}.time", fake_time)
        if module != "gpu_fault.processor.metrics":
            monkeypatch.setattr(f"{module}.datetime", WallClock)
    monkeypatch.setattr("gpu_fault.processor.coordinator.Event", scheduler.event)
    monkeypatch.setattr("gpu_fault.processor.coordinator.Thread", scheduler.thread)
    monkeypatch.setattr("gpu_fault.store.memory.processor_leases.datetime", WallClock)
    for module in ("coordinator", "telemetry_spool"):
        monkeypatch.setattr(
            f"gpu_fault.processor.{module}.ThreadPoolExecutor", scheduler.executor
        )
    return scheduler


def enqueue(processor, path, payload=None, *, body=None, **updates):
    item = processor_request(
        path, body=body if body is not None else json.dumps(payload or {}).encode()
    ).model_copy(update={"created_at": collect_support.NOW, **updates})
    processor.store.enqueue_processor_request(item)
    return item
