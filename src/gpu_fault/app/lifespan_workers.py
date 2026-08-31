from __future__ import annotations

import logging
import os
import random
from threading import Event, Thread
import time
from typing import Any

from gpu_fault.app.periodic_services import PeriodicServiceRunner


LOGGER = logging.getLogger(__name__)


def start_regional_registry_worker(runtime: Any, stop: Event) -> Thread | None:
    if runtime is None:
        return None
    worker = Thread(
        target=runtime.run,
        args=(stop,),
        name="gpu-fault-regional-registry",
        daemon=True,
    )
    worker.start()
    return worker


def start_processor_threads(
    *,
    context,
    processor,
    diagnostics_publisher,
    stop: Event,
    identity_registries: list,
    ingest_node_health_findings,
    notify_silent_collectors,
) -> tuple[Thread, list[Thread]]:
    diagnostics_worker = Thread(
        target=diagnostics_publisher.run,
        args=(stop,),
        name="gpu-fault-processor-diagnostics",
        daemon=True,
    )
    diagnostics_worker.start()

    def services_are_active() -> bool:
        return processor.is_healthy() and (
            processor.active_consumers or processor.is_leader()
        )

    def run_dispatch() -> None:
        next_dispatch = 0.0
        while not stop.wait(0.05):
            if not context.dispatcher.config.enabled or not services_are_active():
                continue
            now = time.monotonic()
            if not (now >= next_dispatch or context.dispatcher.consume_wake()):
                continue
            try:
                context.dispatcher.run_once()
            except Exception:
                LOGGER.exception("workflow dispatch cycle failed")
            next_dispatch = now + context.dispatcher.config.poll_interval_seconds

    def run_xid_correlation() -> None:
        next_xid = 0.0
        while not stop.wait(0.05):
            if not services_are_active():
                continue
            now = time.monotonic()
            if now < next_xid:
                continue
            try:
                context.xid_correlation.run_once()
            except Exception:
                LOGGER.exception("XID correlation cycle failed")
            next_xid = now + context.xid_correlation.poll_interval_seconds

    periodic_runner = PeriodicServiceRunner(
        context=context,
        processor=processor,
        stop=stop,
        identity_registries=identity_registries,
        ingest_node_health_findings=ingest_node_health_findings,
        notify_silent_collectors=notify_silent_collectors,
    )
    threads = [
        Thread(
            target=processor.run_processor,
            name="gpu-fault-processor-inbox",
            daemon=True,
        ),
        Thread(
            target=periodic_runner.run,
            name="gpu-fault-periodic-services",
            daemon=True,
        ),
        Thread(
            target=run_dispatch,
            name="gpu-fault-workflow-dispatcher",
            daemon=True,
        ),
        Thread(
            target=run_xid_correlation,
            name="gpu-fault-xid-correlation",
            daemon=True,
        ),
    ]
    if hasattr(context.store, "listen_processor_queue_notifications"):
        threads.append(
            Thread(
                target=processor.run_queue_notifications,
                name="gpu-fault-processor-notifications",
                daemon=True,
            )
        )
    if processor.telemetry_spool_enabled:
        threads.append(
            Thread(
                target=processor.run_telemetry_spool,
                name="gpu-fault-telemetry-spool",
                daemon=True,
            )
        )
        if hasattr(
            context.store,
            "listen_telemetry_spool_notifications",
        ):
            threads.append(
                Thread(
                    target=processor.run_telemetry_spool_notifications,
                    name="gpu-fault-telemetry-spool-notifications",
                    daemon=True,
                )
            )
    if not processor.active_consumers:
        threads.insert(
            0,
            Thread(
                target=processor.run_leadership,
                name="gpu-fault-processor-leadership",
                daemon=True,
            ),
        )
    for thread in threads:
        thread.start()
    return diagnostics_worker, threads


def start_spool_threads(context, processor) -> list[Thread]:
    threads = []
    if hasattr(
        context.store,
        "listen_telemetry_spool_notifications",
    ):
        threads.append(
            Thread(
                target=processor.run_telemetry_spool_notifications,
                name="gpu-fault-telemetry-spool-notifications",
                daemon=True,
            )
        )
    threads.append(
        Thread(
            target=processor.run_telemetry_spool,
            name="gpu-fault-telemetry-spool",
            daemon=True,
        )
    )
    for thread in threads:
        thread.start()
    return threads


def start_nonprocessor_workers(
    *,
    context,
    stop: Event,
    identity_registries: list,
    ingest_node_health_findings,
) -> tuple[
    Thread,
    Thread | None,
    Thread | None,
    Thread | None,
]:
    xid_worker = Thread(
        target=context.xid_correlation.run_forever,
        name="gpu-fault-xid-correlation",
        daemon=True,
    )
    xid_worker.start()
    training_worker = _start_training_worker(context, stop, ingest_node_health_findings)
    spare_worker = _start_spare_worker(context, stop)
    identity_worker = _start_identity_worker(identity_registries, stop)
    return xid_worker, training_worker, spare_worker, identity_worker


def _start_training_worker(context, stop, ingest):
    if os.getenv("GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR", "false").lower() != "true":
        return None
    interval = float(os.getenv("GPU_FAULT_TRAINING_HEALTH_SCAN_SECONDS", "15"))

    def monitor() -> None:
        while not stop.wait(interval):
            try:
                result = context.training_health.scan_all()
                if result.findings:
                    ingest("training-health-monitor", result.findings)
            except Exception:
                LOGGER.exception("training health scan failed")

    worker = Thread(
        target=monitor,
        name="gpu-fault-training-health-monitor",
        daemon=True,
    )
    worker.start()
    return worker


def _start_spare_worker(context, stop):
    if context.spare_health_controller is None:
        return None
    interval = float(os.getenv("GPU_FAULT_SPARE_HEALTH_SCAN_SECONDS", "30"))

    def monitor() -> None:
        while not stop.is_set():
            try:
                context.spare_health_controller.scan()
            except Exception:
                LOGGER.exception("HyperPod spare health scan failed")
            if stop.wait(interval):
                break

    worker = Thread(
        target=monitor,
        name="gpu-fault-spare-health-monitor",
        daemon=True,
    )
    worker.start()
    return worker


def _start_identity_worker(registries, stop):
    if not registries:
        return None
    interval = float(os.getenv("GPU_FAULT_HYPERPOD_IDENTITY_REFRESH_SECONDS", "20"))

    def refresh() -> None:
        while not stop.is_set():
            try:
                for registry in registries:
                    registry.refresh()
            except Exception:
                LOGGER.exception("HyperPod identity refresh failed")
            if stop.wait(interval):
                break

    worker = Thread(
        target=refresh,
        name="gpu-fault-hyperpod-identity-refresh",
        daemon=True,
    )
    worker.start()
    return worker


def start_notification_worker(
    *,
    context,
    stop: Event,
    owner: str,
    throttle_delay,
) -> Thread:
    interval = float(os.getenv("GPU_FAULT_NOTIFICATION_POLL_SECONDS", "2"))
    cap = float(os.getenv("GPU_FAULT_NOTIFICATION_THROTTLE_MAX_SECONDS", "300"))

    def dispatch() -> None:
        current_delay = 0.0
        while not stop.is_set():
            report = None
            try:
                report = context.advisory_notifications.dispatch_outbox(
                    owner,
                    limit=int(os.getenv("GPU_FAULT_NOTIFICATION_BATCH_SIZE", "25")),
                    lease_seconds=int(
                        os.getenv(
                            "GPU_FAULT_NOTIFICATION_LEASE_SECONDS",
                            "120",
                        )
                    ),
                    max_attempts=int(
                        os.getenv("GPU_FAULT_NOTIFICATION_MAX_ATTEMPTS", "8")
                    ),
                )
            except Exception:
                LOGGER.exception("notification dispatch cycle failed")
            delay = interval
            if report is not None and report.throttled:
                current_delay = throttle_delay(
                    current_delay,
                    interval=interval,
                    cap=cap,
                )
                delay = current_delay * random.uniform(0.8, 1.2)
            else:
                current_delay = 0.0
            if stop.wait(delay):
                break

    worker = Thread(
        target=dispatch,
        name="gpu-fault-notification-dispatcher",
        daemon=True,
    )
    worker.start()
    return worker
