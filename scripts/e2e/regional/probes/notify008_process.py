"""Actual notification workers, independent acceptance, and owned-process loss."""

from __future__ import annotations

import math
import multiprocessing
import os
import signal
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from gpu_fault.models import (
    AdvisoryNotification,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from scripts.e2e.regional.probes.notify008_protocol import (
    CRASH_EXIT,
    LEASE_SECONDS,
    ProbeError,
    Variant,
    run_identity,
    variant_errors,
    variants,
)
from scripts.e2e.regional.probes.notify008_provider import (
    accept_notification,
    read_receipts,
    serve_provider,
)

StoreFactory = Callable[[], Any]


class Budget:
    def __init__(self, seconds: float) -> None:
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or not 0 < seconds <= 600
        ):
            raise ProbeError("invalid process budget")
        self.until = time.monotonic() + seconds

    def remaining(self) -> float:
        remaining = self.until - time.monotonic()
        if remaining <= 0:
            raise ProbeError("isolated process budget expired")
        return remaining


def notification(run_id: str, variant: Variant) -> AdvisoryNotification:
    notification_id = variant.notification_id(run_id)
    return AdvisoryNotification(
        notification_id=notification_id,
        deduplication_key=notification_id,
        cluster_name="notify008-isolated",
        incident_id=f"{run_id}/isolated-incident",
        drill_id=run_id,
        category=variant.kind.upper(),
        subject=f"[DRILL:{run_id}] SIMULATED provider {variant.kind}",
        body_text="Isolated notification commit-window acceptance. No recovery action.",
        support_case_draft="",
    )


class ControlledNotifier:
    def __init__(
        self, run_id: str, variant: Variant, port: int, pipe: Any, crash: bool
    ):
        self.run_id, self.variant, self.port = run_id, variant, port
        self.pipe, self.crash = pipe, crash

    def boundary(self, name: str) -> None:
        if self.crash and self.variant.crash == name:
            self.pipe.send({"stage": name, "pid": os.getpid()})
            while True:
                signal.pause()

    def send(self, value: AdvisoryNotification) -> NotificationResult:
        if (
            value.notification_id != self.variant.notification_id(self.run_id)
            or value.drill_id != self.run_id
        ):
            raise ProbeError("runtime attempted an unowned notification")
        self.boundary("before-provider")
        receipt = accept_notification(self.port, self.run_id, value.notification_id)
        self.boundary("accepted-before-commit")
        return NotificationResult(
            notification_id=value.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id=receipt["message_id"],
        )


def worker(
    factory: StoreFactory,
    run_id: str,
    variant: Variant,
    port: int,
    pipe: Any,
    crash: bool,
    seconds: float,
) -> None:
    try:
        budget = Budget(seconds)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.setitimer(signal.ITIMER_REAL, budget.remaining())
        with closing(factory()) as store:
            notifier = ControlledNotifier(run_id, variant, port, pipe, crash)
            service = AdvisoryNotificationService(
                store,
                notifier,
                async_delivery=variant.mode == "outbox",
                deliver_drills=True,
                deliver_backlog=True,
                ttl_seconds=0,
            )
            pipe.send({"stage": "ready", "pid": os.getpid()})
            while pipe.poll(budget.remaining()):
                command = pipe.recv()
                if command == "stop":
                    return
                if command != "dispatch":
                    raise ProbeError("unknown worker command")
                if variant.mode == "outbox":
                    service.dispatch_outbox(
                        f"{run_id}/worker/{os.getpid()}",
                        limit=1,
                        lease_seconds=LEASE_SECONDS,
                    )
                else:
                    service.send(variant.notification_id(run_id))
                notifier.boundary("committed-before-ack")
                pipe.send({"stage": "dispatched", "pid": os.getpid()})
            raise ProbeError("worker supervision timed out")
    except Exception as exc:
        pipe.send({"stage": "failed", "pid": os.getpid(), "error": type(exc).__name__})
        raise SystemExit(2) from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        pipe.close()


class Processes:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.context = multiprocessing.get_context("spawn")
        self.children: list[Any] = []
        self.pipes: list[Any] = []

    def start(
        self, target: Callable[..., None], arguments: tuple[Any, ...]
    ) -> tuple[Any, Any]:
        parent, child = self.context.Pipe()
        process = self.context.Process(target=target, args=(*arguments, child))
        self.pipes.append(parent)
        self.children.append(process)
        try:
            process.start()
        finally:
            child.close()
        return process, parent

    def receive(self, process: Any, pipe: Any, stage: str | None) -> dict[str, Any]:
        if not pipe.poll(min(20, self.budget.remaining())):
            raise ProbeError("owned process acknowledgement timed out")
        value = pipe.recv()
        if (
            not isinstance(value, dict)
            or type(value.get("pid")) is not int
            or value["pid"] != process.pid
            or (stage is not None and value.get("stage") != stage)
        ):
            raise ProbeError("owned process acknowledgement identity or stage differs")
        return value

    def crash(self, process: Any) -> None:
        if not process.is_alive():
            raise ProbeError("worker exited before the controlled crash")
        process.kill()
        process.join(min(5, self.budget.remaining()))
        if process.is_alive() or process.exitcode != CRASH_EXIT:
            raise ProbeError("controlled worker SIGKILL was not observed")

    def finish(self, process: Any, pipe: Any) -> None:
        pipe.send("stop")
        process.join(min(10, self.budget.remaining()))
        if process.is_alive() or process.exitcode != 0:
            raise ProbeError("replacement process did not terminate cleanly")

    def close(self) -> None:
        failed = []
        for process in reversed(self.children):
            try:
                if process.pid is None:
                    process.close()
                    continue
                if process.is_alive():
                    process.terminate()
                process.join(3)
                if process.is_alive():
                    process.kill()
                    process.join(3)
                if process.is_alive():
                    failed.append(process.pid)
                else:
                    process.close()
            except Exception:
                failed.append("unreaped")
        for pipe in self.pipes:
            pipe.close()
        if failed:
            raise ProbeError("owned processes could not be reaped")


def provider_entry(path: Path, run_id: str, seconds: float, pipe: Any) -> None:
    serve_provider(path, run_id, pipe, seconds)


def worker_entry(
    factory: StoreFactory,
    run_id: str,
    variant: Variant,
    port: int,
    crash: bool,
    seconds: float,
    pipe: Any,
) -> None:
    worker(factory, run_id, variant, port, pipe, crash, seconds)


def wait_for_lease(store: Any, notification_id: str, budget: Budget) -> None:
    delivery = store.get_notification_delivery(notification_id)
    if delivery is None or delivery.lease_expires_at is None:
        raise ProbeError("uncommitted outbox delivery has no observed lease")
    delay = (delivery.lease_expires_at - datetime.now(UTC)).total_seconds() + 0.05
    if not 0 < delay <= LEASE_SECONDS + 1 or delay + 5 >= budget.remaining():
        raise ProbeError("outbox lease is no longer within the bounded takeover window")
    time.sleep(delay)


def exercise_variant(
    factory: StoreFactory,
    run_id: str,
    variant: Variant,
    processes: Processes,
    port: int,
    provider_pid: int,
    ledger: Path,
) -> dict[str, Any]:
    notification_id = variant.notification_id(run_id)
    with closing(factory()) as observer:
        observer.save_notification_if_absent(notification(run_id, variant))
        first, first_pipe = processes.start(
            worker_entry,
            (factory, run_id, variant, port, True, processes.budget.remaining()),
        )
        processes.receive(first, first_pipe, "ready")
        first_pipe.send("dispatch")
        processes.receive(first, first_pipe, variant.crash)
        before_receipts = read_receipts(
            ledger, run_id=run_id, notification_id=notification_id
        )
        before_result = observer.get_notification_result(notification_id)
        if len(before_receipts) != variant.first_acceptances:
            raise ProbeError(
                "independent acceptance was not observed at the crash boundary"
            )
        processes.crash(first)
        result_after_loss = observer.get_notification_result(notification_id)
        if result_after_loss != before_result:
            raise ProbeError("durable result changed during controlled process loss")

        replacement, replacement_pipe = processes.start(
            worker_entry,
            (factory, run_id, variant, port, False, processes.budget.remaining()),
        )
        processes.receive(replacement, replacement_pipe, "ready")
        replacement_pipe.send("dispatch")
        processes.receive(replacement, replacement_pipe, "dispatched")
        first_retry = read_receipts(
            ledger, run_id=run_id, notification_id=notification_id
        )
        early_added = 0
        if variant.mode == "outbox" and before_result is None:
            early_added = len(first_retry) - len(before_receipts)
            if (
                early_added
                or observer.get_notification_result(notification_id) is not None
            ):
                raise ProbeError(
                    "replacement dispatched before the existing outbox lease expired"
                )
            wait_for_lease(observer, notification_id, processes.budget)
            replacement_pipe.send("dispatch")
            processes.receive(replacement, replacement_pipe, "dispatched")
        final = observer.get_notification_result(notification_id)
        accepted = read_receipts(ledger, run_id=run_id, notification_id=notification_id)
        replacement_pipe.send("dispatch")
        processes.receive(replacement, replacement_pipe, "dispatched")
        replay = read_receipts(ledger, run_id=run_id, notification_id=notification_id)
        processes.finish(replacement, replacement_pipe)
        row = {
            "variant": variant.key,
            "notification_id": notification_id,
            "crash_exitcode": first.exitcode,
            "before_result": before_result.status.value if before_result else None,
            "final_result": final.status.value if final else None,
            "provider_message_id": final.provider_message_id if final else None,
            "stored_notifications": sum(
                item.deduplication_key == notification_id
                for item in observer.list_notifications()
            ),
            "replay_added_acceptances": len(replay) - len(accepted),
            "early_retry_added_acceptances": early_added,
            "exactly_once_proven": False,
            "duplicate_observed": len(accepted) > 1,
            "supervisor_pid": os.getpid(),
            "provider_pid": provider_pid,
            "first_pid": first.pid,
            "replacement_pid": replacement.pid,
            "before_receipts": before_receipts,
            "receipts": accepted,
        }
        if variant_errors(row, run_id=run_id, variant=variant):
            raise ProbeError("independent process/Store/provider observations disagree")
        return row


def exercise(
    factory: StoreFactory,
    root: Path,
    run_id: str,
    *,
    selected: tuple[Variant, ...] | None = None,
    seconds: float = 480,
) -> dict[str, Any]:
    run_identity(run_id)
    if os.name != "posix":
        raise ProbeError("NOTIFY008 requires owned POSIX process isolation")
    budget = Budget(seconds)
    processes = Processes(budget)
    ledger = root / "provider.jsonl"
    rows = []
    try:
        provider, pipe = processes.start(
            provider_entry, (ledger, run_id, budget.remaining())
        )
        ready = processes.receive(provider, pipe, None)
        port = ready.get("port")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ProbeError("independent provider did not become ready")
        for variant in variants() if selected is None else selected:
            rows.append(
                exercise_variant(
                    factory, run_id, variant, processes, port, provider.pid, ledger
                )
            )
    finally:
        processes.close()
    return {"provider": "SIMULATED", "variants": rows, "children_reaped": True}
