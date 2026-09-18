from __future__ import annotations

from contextvars import ContextVar
from threading import Event

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import run_ha005_rollout_continuity as ha005


def receipt(request_id: str) -> dict[str, object]:
    return {"request_id": request_id, "status": "COMPLETED", "response_status": 200}


def test_polling_inherits_the_callers_execution_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ContextVar("unit-receipt-context", default="unbound")
    token = context.set("bound")
    observed: list[str] = []
    sampled = Event()

    def accepted() -> list[str]:
        observed.append(context.get())
        sampled.set()
        return ["request-a"]

    monkeypatch.setattr(
        ha005,
        "processor_receipts",
        lambda _ids: {"requests": [receipt("request-a")], "missing": []},
    )
    ledger = ha005.ReceiptLedger(accepted, interval_seconds=0.005)
    try:
        ledger.start()
        context.set("caller-changed")
        assert sampled.wait(5), "the receipt poller did not run"
    finally:
        ledger.stop()
        context.reset(token)
    assert observed and set(observed) == {"bound"}


def test_polling_supervision_loss_is_not_swallowed_or_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled = Event()
    calls: list[list[str]] = []

    def lost(ids: list[str]) -> dict:
        calls.append(ids)
        sampled.set()
        raise ProcessSupervisionLost("unit command completion is unproven")

    monkeypatch.setattr(ha005, "processor_receipts", lost)
    ledger = ha005.ReceiptLedger(lambda: ["request-a"], interval_seconds=0.005)
    ledger.start()
    try:
        assert sampled.wait(5), "the receipt poller did not run"
    finally:
        with pytest.raises(ProcessSupervisionLost, match="completion is unproven"):
            ledger.stop()
    with pytest.raises(ProcessSupervisionLost):
        ledger.wait(["request-a"], timeout_seconds=1)
    assert calls == [["request-a"]], "supervision loss allowed another remote read"


def test_unjoined_receipt_thread_refuses_subsequent_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class UnfinishedThread:
        def __init__(self, *, target, daemon):
            assert callable(target) and daemon is True, (
                "the receipt poller must run its callable on a daemon thread"
            )
            self.ident = None

        def start(self):
            self.ident = 1
            events.append("started")

        def join(self, *, timeout):
            events.append(("join", timeout))

        def is_alive(self):
            return True

    monkeypatch.setattr(ha005.threading, "Thread", UnfinishedThread)
    monkeypatch.setattr(
        ha005,
        "processor_receipts",
        lambda _ids: pytest.fail("an unjoined poller must not permit more reads"),
    )
    ledger = ha005.ReceiptLedger(lambda: [])
    ledger.start()
    with pytest.raises(ProcessSupervisionLost, match="poller did not stop"):
        ledger.stop()
    with pytest.raises(ProcessSupervisionLost):
        ledger.wait(["request-a"], timeout_seconds=1)
    assert events == ["started", ("join", 30)], (
        "an unjoined poller must retain the bounded join and failure latch"
    )


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_invalid_polling_interval_is_rejected(interval: float) -> None:
    with pytest.raises(ha005.CaseError, match="finite and positive"):
        ha005.ReceiptLedger(lambda: [], interval_seconds=interval)


def test_a_stopped_ledger_cannot_restart_a_polling_thread() -> None:
    ledger = ha005.ReceiptLedger(lambda: [])
    ledger.stop()
    with pytest.raises(ha005.CaseError, match="cannot be started twice"):
        ledger.start()
