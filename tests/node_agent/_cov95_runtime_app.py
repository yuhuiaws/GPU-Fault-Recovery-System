from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Callable

import pytest

from gpu_fault.node_agent import app


class QueuedAction(Future[Any]):
    def __init__(self, call: Callable[..., Any], args: tuple[Any, ...]) -> None:
        super().__init__()
        self.call = call
        self.args = args
        self.callbacks: list[Callable[[Future[Any]], Any]] = []

    def add_done_callback(self, fn: Callable[[Future[Any]], Any]) -> None:
        self.callbacks.append(fn)

    def complete(self) -> None:
        try:
            result = self.call(*self.args)
        except Exception as error:
            self.set_exception(error)
        else:
            self.set_result(result)

    def notify(self) -> None:
        for callback in self.callbacks:
            callback(self)


class ActionPool:
    def __init__(self) -> None:
        self.actions: list[QueuedAction] = []
        self.shutdowns: list[dict[str, bool]] = []

    def submit(self, call: Callable[..., Any], *args: Any) -> QueuedAction:
        action = QueuedAction(call, args)
        self.actions.append(action)
        return action

    def shutdown(self, **kwargs: bool) -> None:
        self.shutdowns.append(kwargs)


@pytest.fixture(name="action_pool")
def action_pool_fixture(monkeypatch: pytest.MonkeyPatch) -> ActionPool:
    pool = ActionPool()
    monkeypatch.setattr(app, "ThreadPoolExecutor", lambda **kwargs: pool)
    return pool
