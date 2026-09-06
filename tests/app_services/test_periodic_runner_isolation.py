"""One failing periodic job, or one store error taking a task lease, must not
kill the other five jobs for the life of the process.

FINAL-建议汇总 F-F1 (P0-75A, P1-75B). ``PeriodicServiceRunner.run()`` had
exactly one unprotected statement in its loop body -- the store write inside
``_owns_task_lease`` -- and every ``_run_*`` was called from the same thread.
One exception there ended the thread, silently, and with it training health,
spare health, identity, cleanup, archive and collector-silence detection.
"""

from __future__ import annotations

from threading import Event
from types import SimpleNamespace

from gpu_fault.app.periodic_services import PeriodicServiceRunner


def _runner(store) -> PeriodicServiceRunner:
    processor = SimpleNamespace(
        is_healthy=lambda: True,
        active_consumers=True,
        is_leader=lambda: True,
        owner_id="pod-a:1",
    )
    return PeriodicServiceRunner(
        context=SimpleNamespace(store=store),
        processor=processor,
        stop=Event(),
        identity_registries=[],
        ingest_node_health_findings=lambda *args, **kwargs: None,
        notify_silent_collectors=lambda *args, **kwargs: None,
    )


class _BrokenLeaseStore:
    def __init__(self) -> None:
        self.calls = 0

    def acquire_periodic_task_lease(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("could not connect to server")


def test_a_store_error_while_taking_a_task_lease_is_counted_not_fatal():
    store = _BrokenLeaseStore()
    runner = _runner(store)

    assert runner._due("cleanup", 1000.0, 1.0) is False
    assert store.calls == 1
    assert runner.metrics_snapshot()["periodic_lease_errors_total"] == 1


def test_one_failing_job_does_not_stop_the_others_in_the_same_tick(monkeypatch):
    runner = _runner(_BrokenLeaseStore())
    ran: list[str] = []

    def boom(now: float) -> None:
        raise RuntimeError("training scan exploded")

    monkeypatch.setattr(runner, "_run_training", boom)
    for name in ("spare", "identity", "cleanup", "archive", "silence"):
        monkeypatch.setattr(
            runner, f"_run_{name}", lambda now, name=name: ran.append(name)
        )

    runner.run_all_due(1000.0)

    assert ran == ["spare", "identity", "cleanup", "archive", "silence"]
    assert runner.metrics_snapshot()["periodic_job_errors_total"] == {"training": 1}
