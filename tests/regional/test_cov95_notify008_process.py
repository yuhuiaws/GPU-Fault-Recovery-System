from __future__ import annotations

import signal
from contextlib import closing

import pytest

from scripts.e2e.regional.probes import notify008_process as module
from scripts.e2e.regional.probes.notify008_protocol import (
    ProbeError,
    Variant,
    variant_errors,
)
from scripts.e2e.regional.probes.notify008_provider import read_receipts
from tests.regional._cov95_notify008_support import RUN_ID, SQLiteSimulator
from tests.regional._cov95_notify008_support import provider_fixture as provider_fixture


@pytest.mark.parametrize("kind", ["gpu-reset", "workload-restart"])
@pytest.mark.parametrize(
    "crash", ["before-provider", "accepted-before-commit", "committed-before-ack"]
)
def test_real_inline_worker_loss_and_reconstruction_with_sqlite_simulator(
    tmp_path, kind, crash
):
    variant = Variant("inline", crash, kind)
    result = module.exercise(
        SQLiteSimulator(tmp_path / "unit.db"),
        tmp_path,
        RUN_ID,
        selected=(variant,),
        seconds=60,
    )

    assert result["children_reaped"] is True, (
        "every owned child must exit before return"
    )
    assert result["provider"] == "SIMULATED", "local acceptance is not SNS/SES evidence"
    assert "validation_scope" not in result and "postgres_major" not in result, (
        "the process mechanism must not label a SQLite simulation as regional PostgreSQL"
    )
    (row,) = result["variants"]
    assert variant_errors(row, run_id=RUN_ID, variant=variant) == [], (
        "the process crash, durable Store result and independent receipt must agree"
    )
    assert row["crash_exitcode"] == -signal.SIGKILL, (
        "the old runtime must be killed, not merely represented by a raised exception"
    )
    assert row["exactly_once_proven"] is False, (
        "one durable notification identity cannot establish external exactly-once delivery"
    )
    assert len(row["receipts"]) == (2 if crash == "accepted-before-commit" else 1), (
        "the uncommitted acceptance window must expose a second provider receipt"
    )


@pytest.mark.parametrize(
    "crash", ["before-provider", "accepted-before-commit", "committed-before-ack"]
)
def test_real_outbox_process_loss_obeys_actual_lease_before_reclaim(tmp_path, crash):
    variant = Variant("outbox", crash, "gpu-reset")
    result = module.exercise(
        SQLiteSimulator(tmp_path / "unit.db"),
        tmp_path,
        RUN_ID,
        selected=(variant,),
        seconds=80,
    )

    (row,) = result["variants"]
    assert variant_errors(row, run_id=RUN_ID, variant=variant) == [], (
        "outbox recovery must honor the real lease and retain provider ambiguity"
    )
    assert row["early_retry_added_acceptances"] == 0, (
        "a new process cannot send while the previous delivery lease remains valid"
    )
    assert row["replay_added_acceptances"] == 0, (
        "a committed result must suppress the next explicit replay"
    )
    assert result["children_reaped"] is True, (
        "the actual worker processes must be reaped"
    )


def test_worker_self_deadline_terminates_a_paused_runtime_without_supervisor_kill(
    provider, tmp_path
):
    factory = SQLiteSimulator(tmp_path / "unit.db")
    variant = Variant("inline", "before-provider", "gpu-reset")
    with closing(factory()) as store:
        store.save_notification_if_absent(module.notification(RUN_ID, variant))
    processes = module.Processes(module.Budget(20))
    try:
        child, pipe = processes.start(
            module.worker_entry, (factory, RUN_ID, variant, provider.port, True, 3.0)
        )
        processes.receive(child, pipe, "ready")
        pipe.send("dispatch")
        processes.receive(child, pipe, "before-provider")
        child.join(6)
        assert child.exitcode == -signal.SIGALRM, (
            "the worker must stop at its independent wall-clock deadline"
        )
        with closing(factory()) as observer:
            assert (
                observer.get_notification_result(variant.notification_id(RUN_ID))
                is None
            ), (
                "watchdog termination must not be caught and committed as a delivery result"
            )
        assert read_receipts(provider.ledger, run_id=RUN_ID) == [], (
            "the paused pre-provider worker must never cross the acceptance boundary"
        )
    finally:
        processes.close()


@pytest.mark.parametrize(
    "seconds", [0, -1, 601, True, float("inf"), float("nan"), "10"]
)
def test_process_budget_refuses_unbounded_or_untyped_lifetimes(seconds):
    with pytest.raises(ProbeError, match="budget"):
        module.Budget(seconds)


def test_process_budget_expiry_is_observed_before_another_wait(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    budget = module.Budget(5)
    assert budget.remaining() == 5, "the process must receive only its finite budget"
    clock[0] = 15.0
    with pytest.raises(ProbeError, match="expired"):
        budget.remaining()
