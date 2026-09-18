from __future__ import annotations

import os
from contextlib import closing
from types import SimpleNamespace

import pytest

from gpu_fault.models import NotificationResult, NotificationStatus
from scripts.e2e.regional.probes import notify008_process as module
from scripts.e2e.regional.probes.notify008_protocol import ProbeError, Variant
from tests.regional._cov95_notify008_support import RUN_ID, SQLiteSimulator


def test_controlled_crash_and_cleanup_refuse_a_child_without_observed_exit():
    events = []
    child = SimpleNamespace(
        pid=77,
        exitcode=None,
        is_alive=lambda: True,
        kill=lambda: events.append("kill"),
        terminate=lambda: events.append("terminate"),
        join=lambda seconds: events.append(("join", seconds)),
    )
    processes = module.Processes(module.Budget(10))
    with pytest.raises(ProbeError, match="SIGKILL was not observed"):
        processes.crash(child)
    processes.children.append(child)
    processes.pipes.append(SimpleNamespace(close=lambda: events.append("pipe-close")))
    with pytest.raises(ProbeError, match="could not be reaped"):
        processes.close()
    assert events.count("kill") == 2 and events[-1] == "pipe-close", (
        "unproven termination must remain failure while owned IPC still closes"
    )


def test_process_mechanism_refuses_non_posix_before_opening_store_or_children(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt", getpid=os.getpid))

    def forbidden():
        pytest.fail("unsupported process platform opened the Store")

    with pytest.raises(ProbeError, match="POSIX"):
        module.exercise(forbidden, tmp_path, RUN_ID)
    assert list(tmp_path.iterdir()) == [], (
        "platform refusal must happen before provider or database work"
    )


@pytest.mark.parametrize("port", [None, 0, 65536, True, "32800"])
def test_supervisor_rejects_provider_readiness_without_an_owned_loopback_port(
    tmp_path, monkeypatch, port
):
    actions = []

    class ProviderOnly:
        def __init__(self, budget):
            self.budget = budget

        def start(self, target, args):
            actions.append("start")
            assert target is module.provider_entry, (
                "no runtime child may start before the provider is ready"
            )
            return SimpleNamespace(pid=123), object()

        def receive(self, process, pipe, stage):
            return {"pid": process.pid, "port": port}

        def close(self):
            actions.append("close")

    monkeypatch.setattr(module, "Processes", ProviderOnly)
    with pytest.raises(ProbeError, match="provider did not become ready"):
        module.exercise(
            lambda: pytest.fail("provider failure reached Store"), tmp_path, RUN_ID
        )
    assert actions == ["start", "close"], (
        "invalid provider readiness must still close the owned child supervisor"
    )


def test_missing_independent_receipt_aborts_an_actual_worker_before_crash(
    tmp_path, monkeypatch
):
    variant = Variant("inline", "accepted-before-commit", "gpu-reset")
    read = module.read_receipts
    observed = []

    def unavailable(*args, **kwargs):
        observed.extend(read(*args, **kwargs))
        return []

    monkeypatch.setattr(module, "read_receipts", unavailable)
    with pytest.raises(ProbeError, match="independent acceptance was not observed"):
        module.exercise(
            SQLiteSimulator(tmp_path / "unit.db"),
            tmp_path,
            RUN_ID,
            selected=(variant,),
            seconds=30,
        )
    assert len(observed) == 1, "the unit observer must lose a real independent receipt"
    assert len(read(tmp_path / "provider.jsonl", run_id=RUN_ID)) == 1, (
        "refused evidence must not dispatch a replacement or manufacture acceptance"
    )


def test_concurrent_durable_write_during_actual_loss_is_not_accepted_as_crash_evidence(
    tmp_path, monkeypatch
):
    variant = Variant("inline", "before-provider", "gpu-reset")
    factory = SQLiteSimulator(tmp_path / "unit.db")
    crash = module.Processes.crash

    def foreign_write(processes, child):
        crash(processes, child)
        with closing(factory()) as store:
            store.save_notification_result(
                NotificationResult(
                    notification_id=variant.notification_id(RUN_ID),
                    status=NotificationStatus.SENT,
                    provider_message_id="simulated-" + "1" * 32,
                )
            )

    monkeypatch.setattr(module.Processes, "crash", foreign_write)
    with pytest.raises(
        ProbeError, match="result changed during controlled process loss"
    ):
        module.exercise(factory, tmp_path, RUN_ID, selected=(variant,), seconds=30)
    assert module.read_receipts(tmp_path / "provider.jsonl", run_id=RUN_ID) == [], (
        "the supervisor must reject the unexplained commit before dispatching a replacement"
    )


def test_premature_outbox_result_is_rejected_before_waiting_for_lease_expiry(
    tmp_path, monkeypatch
):
    variant = Variant("outbox", "accepted-before-commit", "gpu-reset")
    factory = SQLiteSimulator(tmp_path / "unit.db")
    receive = module.Processes.receive

    def result_before_lease(processes, child, pipe, stage):
        value = receive(processes, child, pipe, stage)
        if stage == "dispatched":
            with closing(factory()) as store:
                store.save_notification_result(
                    NotificationResult(
                        notification_id=variant.notification_id(RUN_ID),
                        status=NotificationStatus.SENT,
                        provider_message_id="simulated-" + "1" * 32,
                    )
                )
        return value

    monkeypatch.setattr(module.Processes, "receive", result_before_lease)
    monkeypatch.setattr(
        module,
        "wait_for_lease",
        lambda *args: pytest.fail("contradiction reached lease wait"),
    )
    with pytest.raises(ProbeError, match="before the existing outbox lease expired"):
        module.exercise(factory, tmp_path, RUN_ID, selected=(variant,), seconds=30)
    assert len(module.read_receipts(tmp_path / "provider.jsonl", run_id=RUN_ID)) == 1, (
        "the real local provider must have accepted only the killed worker's attempt"
    )


def test_unexplained_receipt_on_committed_replay_prevents_success(
    tmp_path, monkeypatch
):
    variant = Variant("inline", "accepted-before-commit", "gpu-reset")
    read = module.read_receipts
    observations = []

    def changed_history(*args, **kwargs):
        receipts = read(*args, **kwargs)
        observations.append(receipts)
        return [*receipts, receipts[-1]] if len(observations) == 4 else receipts

    monkeypatch.setattr(module, "read_receipts", changed_history)
    with pytest.raises(ProbeError, match="observations disagree"):
        module.exercise(
            SQLiteSimulator(tmp_path / "unit.db"),
            tmp_path,
            RUN_ID,
            selected=(variant,),
            seconds=30,
        )
    assert len(read(tmp_path / "provider.jsonl", run_id=RUN_ID)) == 2, (
        "the corrupted observation must be rejected without changing actual provider history"
    )
