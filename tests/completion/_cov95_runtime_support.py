from __future__ import annotations

from typing import Any

from gpu_fault.completion_controller import KubernetesCompletionController
from tests.completion._support import NOW, FakeCoreApi, FakeSink


class StopLoop(BaseException):
    pass


def make_controller(core=None, sink=None, **overrides: Any):
    options = {
        "cluster_id": "hp-cluster",
        "now": lambda: NOW,
        "publish_observations": True,
    }
    options.update(overrides)
    return KubernetesCompletionController(
        FakeCoreApi() if core is None else core,
        FakeSink() if sink is None else sink,
        **options,
    )


class ManualTimer:
    def __init__(self, interval, callback):
        self.interval = interval
        self.callback = callback
        self.daemon = False
        self.started = False
        self.cancelled = False
        self.running = False
        self.stuck = False
        self.joins = []

    def start(self):
        self.started = True
        self.running = True

    def cancel(self):
        self.cancelled = True

    def is_alive(self):
        return self.running

    def join(self, timeout=None):
        self.joins.append(timeout)
        self.running = self.stuck

    def fire(self):
        self.callback()
        self.running = False
