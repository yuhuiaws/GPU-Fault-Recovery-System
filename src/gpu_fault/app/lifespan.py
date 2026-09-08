from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, Callable

from gpu_fault.app.runtime import EventLoopLag
from gpu_fault.app.lifespan_workers import (
    start_nonprocessor_workers,
    start_notification_worker,
    start_process_counters_worker,
    start_processor_threads,
    start_regional_registry_worker,
    start_spool_threads,
)
from gpu_fault.lifecycle import ShutdownCoordinator


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LifespanDependencies:
    context: Any
    processor: Any | None
    processor_diagnostics_publisher: Any
    background_services_enabled: bool
    spool_only_services_enabled: bool
    ingest_node_health_findings: Callable
    notify_silent_collectors: Callable
    pod_process_owner: Callable[[], str | None]
    notification_throttle_delay: Callable
    register_thread_dump_signal: Callable
    unregister_thread_dump_signal: Callable
    lifespan_shutdown_max_seconds: float
    store_io: Any
    decode_io: Any
    fault_store_io: Any
    evidence_store_io: Any
    fault_decode_io: Any
    telemetry_spool_store_io: Any
    processor_admission_batcher: Any
    fault_admission_batcher: Any
    evidence_admission_batcher: Any
    telemetry_spool_batcher: Any
    event_loop_lag: EventLoopLag
    collector_metrics_snapshot: Any
    regional_registry_runtime: Any | None


def create_lifespan(dependencies: LifespanDependencies):
    @asynccontextmanager
    async def lifespan(_):
        ctx = dependencies.context
        processor = dependencies.processor
        processor_diagnostics_publisher = dependencies.processor_diagnostics_publisher
        background_services_enabled = dependencies.background_services_enabled
        spool_only_services_enabled = dependencies.spool_only_services_enabled
        ingest_node_health_findings = dependencies.ingest_node_health_findings
        notify_silent_collectors = dependencies.notify_silent_collectors
        _pod_process_owner = dependencies.pod_process_owner
        notification_throttle_delay = dependencies.notification_throttle_delay
        register_thread_dump_signal = dependencies.register_thread_dump_signal
        unregister_thread_dump_signal = dependencies.unregister_thread_dump_signal
        lifespan_shutdown_max_seconds = dependencies.lifespan_shutdown_max_seconds
        store_io = dependencies.store_io
        decode_io = dependencies.decode_io
        fault_store_io = dependencies.fault_store_io
        evidence_store_io = dependencies.evidence_store_io
        fault_decode_io = dependencies.fault_decode_io
        telemetry_spool_store_io = dependencies.telemetry_spool_store_io
        processor_admission_batcher = dependencies.processor_admission_batcher
        fault_admission_batcher = dependencies.fault_admission_batcher
        evidence_admission_batcher = dependencies.evidence_admission_batcher
        telemetry_spool_batcher = dependencies.telemetry_spool_batcher
        event_loop_lag = dependencies.event_loop_lag
        collector_metrics_snapshot = dependencies.collector_metrics_snapshot

        async def monitor_event_loop_lag() -> None:
            interval = 0.5
            loop = asyncio.get_running_loop()
            expected = loop.time() + interval
            while True:
                await asyncio.sleep(interval)
                observed = loop.time()
                lag = max(0.0, observed - expected)
                event_loop_lag.observe(lag)
                expected = observed + interval

        event_loop_lag_task = asyncio.create_task(
            monitor_event_loop_lag(),
            name="gpu-fault-event-loop-lag",
        )
        worker = None
        xid_correlation_worker = None
        training_worker = None
        spare_worker = None
        identity_worker = None
        notification_worker = None
        training_stop = Event()
        registry_worker = start_regional_registry_worker(
            dependencies.regional_registry_runtime,
            training_stop,
        )
        collector_metrics_worker = Thread(
            target=collector_metrics_snapshot.run,
            args=(training_stop,),
            name="gpu-fault-collector-metrics-snapshot",
            daemon=True,
        )
        collector_metrics_worker.start()
        process_counters_worker = start_process_counters_worker(ctx, training_stop)
        processor_threads: list[Thread] = []
        diagnostics_worker = None
        identity_registries = [
            *(
                [ctx.hyperpod_identity_registry]
                if ctx.hyperpod_identity_registry is not None
                else []
            ),
            *ctx.hyperpod_identity_registries,
        ]
        if processor is not None and background_services_enabled:
            diagnostics_worker, processor_threads = start_processor_threads(
                context=ctx,
                processor=processor,
                diagnostics_publisher=processor_diagnostics_publisher,
                stop=training_stop,
                identity_registries=identity_registries,
                ingest_node_health_findings=ingest_node_health_findings,
                notify_silent_collectors=notify_silent_collectors,
            )
        elif processor is not None and spool_only_services_enabled:
            processor_threads = start_spool_threads(ctx, processor)
        elif background_services_enabled and ctx.dispatcher.config.enabled:
            worker = Thread(
                target=ctx.dispatcher.run_forever,
                name="gpu-fault-workflow-dispatcher",
                daemon=True,
            )
            worker.start()
        if background_services_enabled and processor is None:
            (
                xid_correlation_worker,
                training_worker,
                spare_worker,
                identity_worker,
            ) = start_nonprocessor_workers(
                context=ctx,
                stop=training_stop,
                identity_registries=identity_registries,
                ingest_node_health_findings=ingest_node_health_findings,
            )
        # Logged again here: an injected context is built before
        # create_app configures logging, so the constructor's line can
        # be lost, and this is the point where the dispatcher decision
        # actually takes effect.
        LOGGER.info(
            "notification delivery: %s",
            ctx.advisory_notifications.describe_delivery_mode(),
        )
        if (
            background_services_enabled
            and ctx.advisory_notifications.async_delivery
            and ctx.advisory_notifications.dispatcher_enabled
        ):
            notification_worker = start_notification_worker(
                context=ctx,
                stop=training_stop,
                owner=_pod_process_owner() or f"local:{os.getpid()}",
                throttle_delay=notification_throttle_delay,
            )
        thread_dump_signal = register_thread_dump_signal(
            (
                os.getenv(
                    "GPU_FAULT_PROCESSOR_THREAD_DUMP_SIGNAL",
                    "",
                )
                if processor is not None and background_services_enabled
                else ""
            )
        )
        if thread_dump_signal is not None:
            LOGGER.info(
                "processor thread dumps enabled pid=%s signal=%s",
                os.getpid(),
                thread_dump_signal,
            )
        try:
            yield
        finally:
            shutdown = ShutdownCoordinator(lifespan_shutdown_max_seconds)

            event_loop_lag_task.cancel()
            try:
                await event_loop_lag_task
            except asyncio.CancelledError:
                pass
            training_stop.set()
            if processor is not None and processor_threads:
                processor.stop()
                for thread in processor_threads:
                    shutdown.join(thread, thread.name)
                if shutdown.failures:
                    abandon = processor.abandon_in_flight()
                    LOGGER.error(
                        "processor shutdown deadline exceeded "
                        "threads=%s in_flight=%s released=%s failed=%s",
                        shutdown.failures,
                        abandon["in_flight"],
                        abandon["released"],
                        abandon["failed"],
                    )
            elif background_services_enabled:
                ctx.xid_correlation.stop()
            shutdown.join(training_worker, "training health worker")
            shutdown.join(spare_worker, "spare health worker")
            shutdown.join(identity_worker, "identity refresh worker")
            shutdown.join(notification_worker, "notification dispatcher")
            shutdown.join(collector_metrics_worker, "collector metrics snapshot")
            shutdown.join(process_counters_worker, "process counters publisher")
            shutdown.join(registry_worker, "regional registry watcher")
            shutdown.join(
                diagnostics_worker,
                "processor diagnostics publisher",
            )
            if worker is not None:
                ctx.dispatcher.stop()
                shutdown.join(worker, "workflow dispatcher")
            if xid_correlation_worker is not None:
                shutdown.join(
                    xid_correlation_worker,
                    "XID correlation worker",
                )
            await processor_admission_batcher.close()
            await fault_admission_batcher.close()
            await evidence_admission_batcher.close()
            await telemetry_spool_batcher.close()
            store_io.close()
            decode_io.close()
            fault_store_io.close()
            evidence_store_io.close()
            fault_decode_io.close()
            telemetry_spool_store_io.close()
            unregister_thread_dump_signal(thread_dump_signal)
            shutdown.raise_if_failed()

    return lifespan
