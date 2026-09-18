from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.models import NotificationDelivery
from scripts.e2e.regional.probes import notify008_process as module
from scripts.e2e.regional.probes.notify008_protocol import ProbeError, Variant
from tests.regional._cov95_notify008_support import RUN_ID, SQLiteSimulator


class Pipe:
    def __init__(self, replies=(), *, ready=True):
        self.replies = list(replies)
        self.ready = ready
        self.sent = []
        self.closed = False

    def poll(self, timeout):
        return self.ready

    def recv(self):
        return self.replies.pop(0)

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


class Process:
    def __init__(self, *, pid=77, alive=False, code=0, resist=False, error=False):
        self.pid, self.alive, self.exitcode = pid, alive, code
        self.resist, self.error = resist, error
        self.events = []

    def start(self):
        self.events.append("start")
        if self.error:
            raise OSError("local start acknowledgement lost")

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.events.append("terminate")
        if self.error:
            raise OSError("local termination failure")
        if not self.resist:
            self.alive = False

    def kill(self):
        self.events.append("kill")
        self.alive = False
        self.exitcode = -9

    def join(self, timeout):
        self.events.append("join")

    def close(self):
        self.events.append("close")


@pytest.mark.parametrize("mode", ["inline", "outbox"])
def test_worker_command_loop_unit_uses_real_service_and_store_with_fake_transport(
    tmp_path, monkeypatch, mode
):
    factory = SQLiteSimulator(tmp_path / "worker.db")
    variant = Variant(mode, "before-provider", "gpu-reset")
    with closing(factory()) as store:
        store.save_notification_if_absent(module.notification(RUN_ID, variant))
    pipe = Pipe(["dispatch", "stop"])
    alarms = []
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(module.signal, "setitimer", lambda *args: alarms.append(args))
    monkeypatch.setattr(
        module,
        "accept_notification",
        lambda *args: {"message_id": "simulated-" + "a" * 32},
    )

    module.worker(factory, RUN_ID, variant, 1, pipe, False, 10)

    assert [item["stage"] for item in pipe.sent] == ["ready", "dispatched"], (
        "the worker must acknowledge readiness before an actual service dispatch"
    )
    assert pipe.closed is True and alarms[-1][1] == 0, (
        "normal worker shutdown must release its IPC endpoint and alarm"
    )
    with closing(factory()) as observer:
        assert (
            observer.get_notification_result(
                variant.notification_id(RUN_ID)
            ).status.value
            == "SENT"
        ), "the unit command loop must execute the real notification service write"


@pytest.mark.parametrize("failure", ["command", "timeout", "store"])
def test_worker_unit_failures_are_reported_without_sending_sensitive_exception_text(
    monkeypatch, tmp_path, failure
):
    pipe = Pipe(["unknown"], ready=failure != "timeout")
    factory = SQLiteSimulator(tmp_path / "worker.db")
    if failure == "store":

        def factory():
            raise OSError("detail-not-for-report")

    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(module.signal, "setitimer", lambda *args: None)

    with pytest.raises(SystemExit) as raised:
        module.worker(
            factory,
            RUN_ID,
            Variant("inline", "before-provider", "gpu-reset"),
            1,
            pipe,
            False,
            10,
        )

    assert raised.value.code == 2, "a failed command loop must not exit successfully"
    assert pipe.sent[-1]["stage"] == "failed" and pipe.closed is True, (
        "worker failure must be visible before its IPC endpoint closes"
    )
    assert "detail-not-for-report" not in str(pipe.sent), (
        "transport exception details must not be copied into evidence"
    )


def test_controlled_boundary_unit_stops_after_emitting_its_exact_stage(monkeypatch):
    class Stopped(BaseException):
        pass

    pipe = Pipe()
    variant = Variant("inline", "before-provider", "gpu-reset")
    notifier = module.ControlledNotifier(RUN_ID, variant, 1, pipe, True)

    def stopped():
        raise Stopped

    monkeypatch.setattr(module.signal, "pause", stopped)
    with pytest.raises(Stopped):
        notifier.send(module.notification(RUN_ID, variant))
    assert [item["stage"] for item in pipe.sent] == ["before-provider"], (
        "the unit boundary must precede provider invocation"
    )
    with pytest.raises(ProbeError, match="unowned"):
        notifier.send(
            module.notification(RUN_ID, variant).model_copy(
                update={"drill_id": "foreign"}
            )
        )


@pytest.mark.parametrize(
    "reply",
    [
        None,
        {"pid": 77, "stage": "failed"},
        {"pid": True, "stage": "ready"},
        {"pid": 78, "stage": "ready"},
    ],
)
def test_process_acknowledgement_requires_owned_pid_and_expected_stage(reply):
    processes = module.Processes(module.Budget(5))
    with pytest.raises(ProbeError, match="acknowledgement"):
        processes.receive(Process(), Pipe([reply]), "ready")


def test_process_acknowledgement_wait_is_bounded():
    processes = module.Processes(module.Budget(5))
    with pytest.raises(ProbeError, match="timed out"):
        processes.receive(Process(), Pipe(ready=False), "ready")


def test_process_start_is_journaled_before_a_lost_start_ack():
    processes = module.Processes(module.Budget(5))
    process = Process(error=True)
    parent, child = Pipe(), Pipe()
    processes.context = SimpleNamespace(
        Pipe=lambda: (parent, child), Process=lambda **kwargs: process
    )
    with pytest.raises(OSError, match="acknowledgement"):
        processes.start(lambda *args: None, ())
    assert processes.children == [process] and child.closed, (
        "a partially started child must remain tracked for cleanup"
    )
    process.error = False
    processes.close()
    assert "close" in process.events and parent.closed, (
        "lost ACK must not leak the child handle"
    )


def test_owned_process_cleanup_handles_unstarted_and_term_resistant_children():
    processes = module.Processes(module.Budget(5))
    unstarted = Process(pid=None)
    resistant = Process(alive=True, resist=True)
    pipe = Pipe()
    processes.children = [unstarted, resistant]
    processes.pipes = [pipe]
    processes.close()
    assert unstarted.events == ["close"], (
        "unstarted process must not be joined or signalled"
    )
    assert (
        "terminate" in resistant.events
        and "kill" in resistant.events
        and "close" in resistant.events
    ), "cleanup must escalate only for its owned unresponsive child"
    assert pipe.closed, "cleanup must close the parent's pipe"


def test_process_cleanup_failure_is_not_reported_as_reaped():
    processes = module.Processes(module.Budget(5))
    processes.children = [Process(alive=True, error=True)]
    with pytest.raises(ProbeError, match="reaped"):
        processes.close()


def test_crash_and_finish_require_observed_termination():
    processes = module.Processes(module.Budget(5))
    with pytest.raises(ProbeError, match="exited before"):
        processes.crash(Process())
    process = Process(alive=True)
    processes.crash(process)
    assert process.exitcode == -9, "controlled crash must observe SIGKILL termination"
    with pytest.raises(ProbeError, match="terminate cleanly"):
        processes.finish(Process(code=2), Pipe())


@pytest.mark.parametrize("lease", [None, "missing", "expired", "too-long", "budget"])
def test_takeover_wait_refuses_missing_or_out_of_budget_leases(lease):
    now = datetime.now(UTC)
    row = NotificationDelivery(
        notification_id="test",
        available_at=now,
        created_at=now,
        updated_at=now,
        lease_expires_at=now
        + timedelta(
            seconds=60 if lease == "too-long" else -1 if lease == "expired" else 30
        ),
    )
    if lease == "missing":
        row = row.model_copy(update={"lease_expires_at": None})
    store = SimpleNamespace(
        get_notification_delivery=lambda key: None if lease is None else row
    )
    with pytest.raises(ProbeError):
        module.wait_for_lease(
            store, "test", module.Budget(10 if lease == "budget" else 60)
        )
