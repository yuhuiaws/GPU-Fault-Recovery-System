"""Kernel reader boundaries without devices, process signals or delivery threads."""

from __future__ import annotations

import signal
import threading
from types import SimpleNamespace

import pytest

from gpu_fault.collectors.logs import kernel
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


class DeliveryThread:
    def __init__(self, owner, *, target, args=(), name="", daemon=False):
        self.owner = owner
        self.target = target
        self.args = args
        self.alive = False
        self.joined = False

    def start(self):
        self.alive = self.owner.start_alive

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.joined = True
        if not self.owner.survive_join:
            self.alive = False


@pytest.fixture
def kernel_case(monkeypatch, tmp_path, isolated_runtime):
    clock = support.Clock()
    sink = support.RecordingSink()
    uptime = tmp_path / "uptime"
    uptime.write_text("100 0\n")
    state = SimpleNamespace(
        clock=clock,
        sink=sink,
        uptime=uptime,
        threads=[],
        start_alive=True,
        survive_join=False,
        handlers={signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_DFL},
        signal_calls=[],
        main=True,
    )
    main_thread, other_thread = object(), object()

    def thread(**kwargs):
        item = DeliveryThread(state, **kwargs)
        state.threads.append(item)
        return item

    def install(number, handler):
        state.signal_calls.append((number, handler))
        previous = state.handlers[number]
        state.handlers[number] = handler
        return previous

    def reopen(seconds):
        clock.sleep(seconds)
        raise support.StopLoop

    def build(**kwargs):
        return kernel.KernelLogCollector(
            sink,
            support.collector_context(),
            **{
                "node_id": "node-a",
                "boot_id": "private-boot",
                "uptime_path": str(uptime),
                "now": clock.now,
                "sleep": reopen,
                **kwargs,
            },
        )

    monkeypatch.setattr(
        kernel,
        "threading",
        SimpleNamespace(
            Thread=thread,
            Condition=threading.Condition,
            Event=threading.Event,
            main_thread=lambda: main_thread,
            current_thread=lambda: main_thread if state.main else other_thread,
        ),
    )
    monkeypatch.setattr(
        kernel,
        "signal",
        SimpleNamespace(
            SIGTERM=signal.SIGTERM,
            SIGINT=signal.SIGINT,
            SIG_DFL=signal.SIG_DFL,
            SIG_IGN=signal.SIG_IGN,
            getsignal=lambda number: state.handlers[number],
            signal=install,
        ),
    )
    monkeypatch.setattr(kernel, "time", SimpleNamespace(monotonic=clock.monotonic))
    state.build = build
    return state


def line(sequence=1, monotonic=1_000_000):
    return f"6,{sequence},{monotonic},-;NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus\n"
