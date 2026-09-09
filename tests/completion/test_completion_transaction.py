"""Completion path: one store transaction per event and no process lock.

FINAL-建议汇总 F-G2 (3): the event row, the plan (with its incident and
workflow) and the decision are written inside one ``completion_transaction``
keyed by the event, so a crash between them cannot leave a poisoned event row.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    CompletionDecision,
    DecisionStatus,
    RecoveryPlan,
    TerminalEvent,
)
from gpu_fault.store import InMemoryStore
from tests._builders import build_context, copy_model


class RecordingStore(InMemoryStore):
    """Remembers which completion transaction each write happened under."""

    def __init__(self) -> None:
        super().__init__()
        self.open_keys: list[str] = []
        self.writes: list[tuple[str, str | None]] = []

    def _current_key(self) -> str | None:
        return self.open_keys[-1] if self.open_keys else None

    @contextmanager
    def _recorded(self, event_key: str) -> Iterator[None]:
        self.open_keys.append(event_key)
        try:
            with self._lock:
                yield
        finally:
            self.open_keys.pop()

    def completion_transaction(self, event_key: str):
        return self._recorded(event_key)

    def save_event_if_absent(self, event: TerminalEvent) -> bool:
        self.writes.append(("event", self._current_key()))
        return super().save_event_if_absent(event)

    def save_plan(self, plan: RecoveryPlan) -> None:
        self.writes.append(("plan", self._current_key()))
        super().save_plan(plan)

    def save_decision(self, decision: CompletionDecision) -> None:
        self.writes.append(("decision", self._current_key()))
        super().save_decision(decision)


@pytest.fixture
def recording_store() -> RecordingStore:
    return RecordingStore()


@pytest.fixture
def recording_context(recording_store: RecordingStore) -> ApplicationContext:
    return build_context(store=recording_store)


def _no_allocation(event: TerminalEvent) -> TerminalEvent:
    return copy_model(event, allocation=[])


def test_completion_service_holds_no_process_wide_lock(
    context: ApplicationContext,
) -> None:
    """Active-active replicas share nothing in-process; the store serializes."""

    assert not hasattr(context.completion, "_lock"), (
        "CompletionService still carries an in-process RLock"
    )


def test_terminal_writes_all_happen_inside_the_event_transaction(
    recording_context: ApplicationContext,
    recording_store: RecordingStore,
    failed_event: TerminalEvent,
) -> None:
    """Event row, plan and decision are one unit keyed by the event."""

    decision = recording_context.completion.handle_terminal(
        _no_allocation(failed_event)
    )

    assert decision.status is DecisionStatus.PLAN_CREATED
    kinds = {kind for kind, _key in recording_store.writes}
    assert kinds == {"event", "plan", "decision"}, recording_store.writes
    assert all(
        key == failed_event.event_key for _kind, key in recording_store.writes
    ), f"a completion write escaped the event transaction: {recording_store.writes}"
