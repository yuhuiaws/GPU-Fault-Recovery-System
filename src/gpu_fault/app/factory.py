from __future__ import annotations

import asyncio
import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from gpu_fault.app.admission import (
    _ProcessorAdmissionBatcher as _ProcessorAdmissionBatcher,
)
from gpu_fault.app.admission import (
    _StripedAdmissionScope as _StripedAdmissionScope,
)
from gpu_fault.app.admission_runtime import AdmissionRuntimeFactory
from gpu_fault.app.authorization import (
    ExplicitAuthorizationRegistry,
    iter_api_routes,
    validate_direct_client_identity_environment,
)
from gpu_fault.app.cluster_binding import (
    payload_cluster_ids as payload_cluster_ids,
)
from gpu_fault.app.collector_metrics import CollectorMetricsSnapshot
from gpu_fault.app.collector_silence import (
    notify_silent_collectors,
)
from gpu_fault.app.context import (
    ApplicationContext,
    default_simulated_profile,
)
from gpu_fault.app.identity import (
    pod_process_owner as _pod_process_owner,
)
from gpu_fault.app.ingest import (
    FaultIngestionService,
    NodeHealthIngestionService,
    TelemetryContextService,
    TelemetryIngestionService,
)
from gpu_fault.app.lifespan import (
    LifespanDependencies,
    create_lifespan,
)
from gpu_fault.app.metric_scan_cache import MetricScanCache
from gpu_fault.app.metrics import (
    get_app_runtime,
    process_local_metric_lines,
)
from gpu_fault.app.metrics import (
    router as metrics_router,
)
from gpu_fault.app.middleware.auth import (
    RegionalAuthDependencies,
    install_regional_authorization,
)
from gpu_fault.app.middleware.backpressure import (
    IngressBackpressureDependencies,
    install_ingress_backpressure,
    install_request_deadline,
)
from gpu_fault.app.middleware.dispatch import (
    ProcessorDispatchDependencies,
    install_processor_dispatch,
)
from gpu_fault.app.processor_factory import ProcessorFactory
from gpu_fault.app.remote_command_wakeups import RemoteCommandWakeupHub
from gpu_fault.app.routes.admin import (
    AdminRouterDependencies,
    get_admin_dependencies,
)
from gpu_fault.app.routes.admin import (
    router as admin_router,
)
from gpu_fault.app.routes.collector_events import (
    CollectorRouterDependencies,
    get_collector_dependencies,
)
from gpu_fault.app.routes.collector_events import (
    router as collector_router,
)
from gpu_fault.app.routes.completion import (
    CompletionRouterDependencies,
    get_completion_dependencies,
)
from gpu_fault.app.routes.completion import (
    router as completion_router,
)
from gpu_fault.app.routes.configuration import (
    ConfigurationRouterDependencies,
    get_configuration_dependencies,
)
from gpu_fault.app.routes.configuration import (
    router as configuration_router,
)
from gpu_fault.app.routes.fleet import (
    FleetRouterDependencies,
    get_fleet_dependencies,
)
from gpu_fault.app.routes.fleet import (
    router as fleet_router,
)
from gpu_fault.app.routes.gpu_events import (
    GpuEventRouterDependencies,
    get_gpu_event_dependencies,
)
from gpu_fault.app.routes.gpu_events import (
    router as gpu_event_router,
)
from gpu_fault.app.routes.incidents import (
    IncidentRouterDependencies,
    get_incident_dependencies,
)
from gpu_fault.app.routes.incidents import (
    router as incident_router,
)
from gpu_fault.app.routes.processor import (
    ProcessorRouterDependencies,
    get_processor_dependencies,
)
from gpu_fault.app.routes.processor import (
    router as processor_router,
)
from gpu_fault.app.routes.regional import (
    RegionalRouterDependencies,
    get_regional_dependencies,
)
from gpu_fault.app.routes.regional import (
    router as regional_router,
)
from gpu_fault.app.routes.regional_registry import (
    router as regional_registry_router,
)
from gpu_fault.app.routes.telemetry import (
    TelemetryRouterDependencies,
    get_telemetry_dependencies,
)
from gpu_fault.app.routes.telemetry import (
    router as telemetry_router,
)
from gpu_fault.app.routes.workflows import (
    WorkflowRouterDependencies,
    get_workflow_dependencies,
)
from gpu_fault.app.routes.workflows import (
    router as workflow_router,
)
from gpu_fault.app.runtime import (
    AppRuntime,
    EventLoopLag,
)
from gpu_fault.capabilities import ProfileValidationError
from gpu_fault.channel_registry import (
    COLLECTOR_EVENT_PREFIX,
    TRAINING_PROGRESS_PATH,
    WORKLOAD_OBSERVATIONS_PATH,
    channel_for_path,
    is_fault_path,
    validate_collector_routes,
)
from gpu_fault.execution import (
    WorkflowExecutionError,
)
from gpu_fault.lifecycle import (
    required_processor_shutdown_seconds,
    validate_lifespan_shutdown_budget,
)
from gpu_fault.logging_setup import configure_logging
from gpu_fault.notification_service import (
    AdvisoryNotApplicableError,
)
from gpu_fault.planner import UnsupportedPlanError
from gpu_fault.processor import (
    processor_internal_token_valid,
)
from gpu_fault.processor_diagnostics import (
    ProcessorDiagnosticsPublisher,
    ProcessorReplayTracker,
    process_runtime_snapshot,
    register_thread_dump_signal,
    unregister_thread_dump_signal,
)
from gpu_fault.regional import TOKEN_SLOT_RETIRING
from gpu_fault.regional_compatibility import (
    RegionalExecutorCompatibilityPolicy,
)
from gpu_fault.regional_registry_runtime import RegionalRegistryRuntime
from gpu_fault.store import (
    EfaTrafficAdminConflict,
    NotFoundError,
    WorkflowLeaseError,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ApplicationContext",
    "create_app",
    "default_simulated_profile",
    "run",
]


def notification_throttle_delay(
    previous: float, *, interval: float, cap: float
) -> float:
    """How long to hold off after the provider refused a send.

    Doubles per consecutive throttled cycle from twice the poll interval,
    capped. The first step has to be longer than the poll interval or the
    backoff is not one: the dispatcher releases a throttled batch without
    charging it an attempt, so the loop would otherwise re-offer the same
    notification at exactly the rate it was refused at.
    """

    return min(max(previous * 2, interval * 2), cap)


def _configure_service_runtime(
    ctx,
    processor,
    processor_exit_grace_seconds,
    processor_replay_tracker,
    node_health_ingestion,
    store_io,
    decode_io,
    fault_store_io,
    evidence_store_io,
    fault_decode_io,
    telemetry_spool_store_io,
    processor_admission_batcher,
    fault_admission_batcher,
    evidence_admission_batcher,
    telemetry_spool_batcher,
):
    service_role = os.getenv("GPU_FAULT_SERVICE_ROLE", "all").strip().lower()
    if service_role not in {
        "all",
        "ingress",
        "worker",
        "spool-worker",
    }:
        raise ValueError(
            "GPU_FAULT_SERVICE_ROLE must be all, ingress, worker, or spool-worker"
        )
    regional_auth_registry = (
        RegionalRegistryRuntime.bootstrap(
            ctx.store,
            member_id=(_pod_process_owner() or f"local:{service_role}:{os.getpid()}"),
            service_role=service_role,
            release_id=os.getenv("GPU_FAULT_RELEASE_ID", "local"),
            poll_seconds=float(os.getenv("GPU_FAULT_REGISTRY_POLL_SECONDS", "1")),
            stale_seconds=float(os.getenv("GPU_FAULT_REGISTRY_STALE_SECONDS", "10")),
            # A Pod that starts during an Aurora writer failover must wait it
            # out, not crash into a restart loop that needs Aurora again.
            retry_budget_seconds=float(
                os.getenv("GPU_FAULT_STARTUP_STORE_RETRY_SECONDS", "120")
            ),
            secret_config_sha256=ctx.regional_registry_secret_sha256,
        )
        if ctx.regional_mode
        else None
    )
    ctx.bind_regional_registry_runtime(regional_auth_registry)
    ctx.regional_registry_runtime = regional_auth_registry
    background_services_enabled = service_role in {
        "all",
        "worker",
    }
    spool_only_services_enabled = service_role == "spool-worker"
    if spool_only_services_enabled and (
        processor is None or not processor.telemetry_spool_enabled
    ):
        raise ValueError(
            "spool-worker role requires queued processor mode and "
            "GPU_FAULT_TELEMETRY_SPOOL=true"
        )
    processor_services_enabled = processor is not None and (
        background_services_enabled or spool_only_services_enabled
    )
    required_shutdown_seconds = (
        required_processor_shutdown_seconds(
            processor.request_max_execution_seconds,
            processor_exit_grace_seconds,
        )
        if processor_services_enabled
        else 20
    )
    lifespan_shutdown_max_seconds = float(
        os.getenv(
            "GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS",
            str(required_shutdown_seconds),
        )
    )
    validate_lifespan_shutdown_budget(
        lifespan_shutdown_max_seconds,
        required_shutdown_seconds,
    )

    def local_processor_diagnostics() -> dict:
        process = process_runtime_snapshot()
        process["thread_dump_signal"] = os.getenv(
            "GPU_FAULT_PROCESSOR_THREAD_DUMP_SIGNAL",
            "",
        )
        return {
            "process": process,
            "local_owner_id": (processor.owner_id if processor is not None else None),
            "in_flight_requests": (
                processor.in_flight_snapshot() if processor is not None else []
            ),
            "inbound_replay_requests": (processor_replay_tracker.snapshot()),
            "processor_runtime": (
                processor.metrics_snapshot() if processor is not None else None
            ),
        }

    processor_diagnostics_publisher = ProcessorDiagnosticsPublisher(
        os.getenv(
            "GPU_FAULT_PROCESSOR_DIAGNOSTICS_DIR",
            "/tmp/gpu-fault-processor-diagnostics",
        ),
        local_processor_diagnostics,
        interval_seconds=float(
            os.getenv(
                "GPU_FAULT_PROCESSOR_DIAGNOSTICS_INTERVAL_SECONDS",
                "2",
            )
        ),
        stale_seconds=float(
            os.getenv(
                "GPU_FAULT_PROCESSOR_DIAGNOSTICS_STALE_SECONDS",
                "10",
            )
        ),
    )
    ingress_fault_concurrency = int(
        os.getenv("GPU_FAULT_INGRESS_FAULT_CONCURRENCY", "256")
    )
    ingress_normal_concurrency = int(
        os.getenv("GPU_FAULT_INGRESS_NORMAL_CONCURRENCY", "1000")
    )
    ingress_fault_wait_seconds = float(
        os.getenv("GPU_FAULT_INGRESS_FAULT_WAIT_SECONDS", "30")
    )
    ingress_normal_wait_seconds = float(
        os.getenv("GPU_FAULT_INGRESS_NORMAL_WAIT_SECONDS", "2")
    )
    if (
        ingress_fault_concurrency <= 0
        or ingress_normal_concurrency <= 0
        or ingress_fault_wait_seconds <= 0
        or ingress_normal_wait_seconds <= 0
    ):
        raise ValueError("ingress concurrency and wait limits must be positive")
    ingress_fault_semaphore = asyncio.Semaphore(ingress_fault_concurrency)
    ingress_normal_semaphore = asyncio.Semaphore(ingress_normal_concurrency)
    ingress_backpressure_rejections = {
        "fault": 0,
        "normal": 0,
    }
    event_loop_lag = EventLoopLag()
    collector_metrics_snapshot = CollectorMetricsSnapshot(
        ctx,
        owner_id=(_pod_process_owner() or f"collector-metrics:{os.getpid()}"),
        enabled=service_role in {"all", "ingress"},
    )
    lifespan = create_lifespan(
        LifespanDependencies(
            context=ctx,
            processor=processor,
            processor_diagnostics_publisher=(processor_diagnostics_publisher),
            background_services_enabled=background_services_enabled,
            spool_only_services_enabled=spool_only_services_enabled,
            ingest_node_health_findings=(node_health_ingestion.ingest),
            notify_silent_collectors=notify_silent_collectors,
            pod_process_owner=_pod_process_owner,
            notification_throttle_delay=(notification_throttle_delay),
            register_thread_dump_signal=(register_thread_dump_signal),
            unregister_thread_dump_signal=(unregister_thread_dump_signal),
            lifespan_shutdown_max_seconds=(lifespan_shutdown_max_seconds),
            store_io=store_io,
            decode_io=decode_io,
            fault_store_io=fault_store_io,
            evidence_store_io=evidence_store_io,
            fault_decode_io=fault_decode_io,
            telemetry_spool_store_io=telemetry_spool_store_io,
            processor_admission_batcher=processor_admission_batcher,
            fault_admission_batcher=fault_admission_batcher,
            evidence_admission_batcher=evidence_admission_batcher,
            telemetry_spool_batcher=telemetry_spool_batcher,
            event_loop_lag=event_loop_lag,
            collector_metrics_snapshot=collector_metrics_snapshot,
            regional_registry_runtime=regional_auth_registry,
            process_metrics_render=process_local_metric_lines,
        )
    )
    # Long-poll claim wakeups (``remote_command_wakeups.py``): one lazily
    # started LISTEN thread per process, closed when the lifespan exits.
    remote_command_wakeups = RemoteCommandWakeupHub(ctx.store)
    lifespan = remote_command_wakeups.wrap_lifespan(lifespan)
    return (
        regional_auth_registry,
        service_role,
        background_services_enabled,
        processor_diagnostics_publisher,
        ingress_fault_semaphore,
        ingress_normal_semaphore,
        ingress_fault_wait_seconds,
        ingress_normal_wait_seconds,
        ingress_backpressure_rejections,
        event_loop_lag,
        collector_metrics_snapshot,
        lifespan,
        local_processor_diagnostics,
        remote_command_wakeups,
    )


def _install_regional_auth(
    app,
    ctx,
    regional_auth_registry,
    processor_replay_authorized,
    decode_io,
    decode_json_body,
    processor_max_request_bytes,
    *,
    fault_decode_io=None,
    is_fault_path=None,
    admission=None,
):
    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)

    def authenticate_regional_cluster(cluster_id, authorization):
        if regional_auth_registry is None or not regional_auth_registry.is_ready():
            raise HTTPException(
                status_code=503,
                detail="regional registry snapshot is not current",
                headers={"Retry-After": "2"},
            )
        if not cluster_id:
            raise HTTPException(
                status_code=401, detail="X-GPU-Fault-Cluster-ID is required"
            )
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="regional cluster bearer token is required"
            )
        registration = regional_auth_registry.get(cluster_id)
        if registration is None:
            raise HTTPException(
                status_code=403, detail="regional cluster is not registered"
            )
        token = authorization.removeprefix("Bearer ").strip()
        slot = registration.matched_token_slot(token)
        if slot is None:
            raise HTTPException(
                status_code=403, detail="regional cluster authentication failed"
            )
        if slot == TOKEN_SLOT_RETIRING:
            # This is how an operator learns the rotation is unfinished. As long
            # as it appears, at least one executor still holds the old token and
            # dropping the retiring digest would lock that cluster out.
            LOGGER.warning(
                "regional cluster %s authenticated with the retiring token; "
                "rotation window closes at %s",
                registration.cluster_id,
                registration.token_rotation_expires_at,
            )
        return registration

    install_regional_authorization(
        app,
        RegionalAuthDependencies(
            context=ctx,
            replay_authorized=processor_replay_authorized,
            authorization_bucket=registry.effective,
            route_exists=registry.route_exists,
            authenticate_cluster=authenticate_regional_cluster,
            decode_io=decode_io,
            decode_json_body=decode_json_body,
            payload_cluster_ids=payload_cluster_ids,
            processor_max_request_bytes=processor_max_request_bytes,
            # A-3 / A-4 / A-8 / E-3 (control-plane review 2026-09-08)
            fault_decode_io=fault_decode_io,
            is_fault_path=is_fault_path,
            dispatch_state=admission.dispatch_state if admission else None,
            decode_rejections=admission.decode_rejections if admission else None,
            retry_after_seconds=admission.retry_after_seconds if admission else 2,
        ),
    )

    @app.exception_handler(NotFoundError)
    async def not_found_handler(_, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404, content={"detail": f"resource not found: {exc.args[0]}"}
        )

    @app.exception_handler(ProfileValidationError)
    @app.exception_handler(UnsupportedPlanError)
    @app.exception_handler(AdvisoryNotApplicableError)
    @app.exception_handler(EfaTrafficAdminConflict)
    @app.exception_handler(WorkflowExecutionError)
    @app.exception_handler(WorkflowLeaseError)
    async def conflict_handler(_, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    return registry


def _install_core_routes(
    app,
    ctx,
    processor,
    processor_mode,
    service_role,
    store_io,
    decode_io,
    fault_store_io,
    evidence_store_io,
    fault_decode_io,
    telemetry_spool_store_io,
    processor_replay_tracker,
    processor_diagnostics_publisher,
    local_processor_diagnostics,
    regional_auth_registry,
    fault_ingestion,
    telemetry_ingestion,
    admission,
    background_services_enabled,
    ingress_backpressure_rejections,
    event_loop_lag,
    collector_metrics_snapshot,
):
    processor_admission_batcher = admission.admission_batcher
    fault_admission_batcher = admission.fault_batcher
    evidence_admission_batcher = admission.evidence_batcher
    telemetry_spool_batcher = admission.spool_batcher
    telemetry_spool_enabled = admission.spool_enabled
    processor_queue_bypass_enabled = admission.queue_bypass_enabled
    processor_queue_bypass_paths = admission.queue_bypass_paths
    processor_admission_rejections = admission.admission_rejections
    processor_admission_rejections_by_path = admission.admission_rejections_by_path
    processor_queue_bypasses_by_path = admission.queue_bypasses_by_path
    telemetry_spool_admitted_by_path = admission.spool_admitted_by_path
    telemetry_spool_rejections = admission.spool_rejections
    dispatch_state = admission.dispatch_state
    processor_max_queue_depth = admission.max_queue_depth
    processor_max_cluster_queue_depth = admission.max_cluster_queue_depth
    processor_fault_reserved_queue_depth = admission.fault_reserved_depth
    processor_fault_reserved_cluster_depth = admission.fault_reserved_cluster_depth
    processor_max_request_bytes = admission.max_request_bytes
    processor_global_admission_guard = admission.global_admission_guard
    app.dependency_overrides[get_regional_dependencies] = lambda: (
        RegionalRouterDependencies(
            context=ctx,
            store_io=store_io,
            auth_registry=regional_auth_registry,
            max_unclaimed_seconds=float(
                os.getenv(
                    "GPU_FAULT_EXECUTOR_READINESS_MAX_UNCLAIMED_SECONDS",
                    "300",
                )
            ),
            max_claim_age_seconds=float(
                os.getenv(
                    "GPU_FAULT_EXECUTOR_READINESS_MAX_CLAIM_AGE_SECONDS",
                    "300",
                )
            ),
            executor_compatibility=(
                RegionalExecutorCompatibilityPolicy.from_mapping(os.environ)
            ),
            remote_command_wakeups=app.state.remote_command_wakeups,
        )
    )
    app.include_router(regional_router)
    app.include_router(regional_registry_router)

    app.dependency_overrides[get_admin_dependencies] = lambda: (
        AdminRouterDependencies(
            context=ctx,
            processor=processor,
            processor_mode=processor_mode,
            service_role=service_role,
            environment=os.environ,
            regional_registry_runtime=regional_auth_registry,
        )
    )
    app.include_router(admin_router)

    app_runtime = AppRuntime(
        context=ctx,
        processor=processor,
        processor_replay_tracker=processor_replay_tracker,
        store_io=store_io,
        decode_io=decode_io,
        fault_store_io=fault_store_io,
        evidence_store_io=evidence_store_io,
        fault_decode_io=fault_decode_io,
        telemetry_spool_store_io=telemetry_spool_store_io,
        processor_admission_batcher=processor_admission_batcher,
        fault_admission_batcher=fault_admission_batcher,
        evidence_admission_batcher=evidence_admission_batcher,
        telemetry_spool_batcher=telemetry_spool_batcher,
        background_services_enabled=background_services_enabled,
        telemetry_spool_enabled=telemetry_spool_enabled,
        processor_queue_bypass_enabled=(processor_queue_bypass_enabled),
        processor_queue_bypass_paths=processor_queue_bypass_paths,
        processor_admission_rejections=(processor_admission_rejections),
        processor_admission_rejections_by_path=(processor_admission_rejections_by_path),
        processor_queue_bypasses_by_path=(processor_queue_bypasses_by_path),
        telemetry_spool_admitted_by_path=(telemetry_spool_admitted_by_path),
        telemetry_spool_rejections=telemetry_spool_rejections,
        ingress_backpressure_rejections=(ingress_backpressure_rejections),
        event_loop_lag_snapshot=event_loop_lag.snapshot,
        dispatch_state=dispatch_state,
        collector_metrics_snapshot=collector_metrics_snapshot,
        metric_scan_cache=MetricScanCache(ctx.store),
        decode_rejections=admission.decode_rejections,
    )
    app.dependency_overrides[get_app_runtime] = lambda: app_runtime
    app.include_router(metrics_router)
    app.state.runtime = app_runtime

    app.dependency_overrides[get_processor_dependencies] = lambda: (
        ProcessorRouterDependencies(
            context=ctx,
            processor=processor,
            processor_mode=processor_mode,
            store_io=store_io,
            telemetry_spool_store_io=telemetry_spool_store_io,
            evidence_store_io=evidence_store_io,
            diagnostics=local_processor_diagnostics,
            diagnostics_publisher=processor_diagnostics_publisher,
            max_queue_depth=processor_max_queue_depth,
            max_cluster_queue_depth=processor_max_cluster_queue_depth,
            fault_reserved_queue_depth=(processor_fault_reserved_queue_depth),
            fault_reserved_cluster_depth=(processor_fault_reserved_cluster_depth),
            max_request_bytes=processor_max_request_bytes,
            global_admission_guard=processor_global_admission_guard,
        )
    )
    app.include_router(processor_router)

    app.dependency_overrides[get_fleet_dependencies] = lambda: (
        FleetRouterDependencies(
            context=ctx,
            store_io=store_io,
        )
    )
    app.include_router(fleet_router)

    app.dependency_overrides[get_configuration_dependencies] = lambda: (
        ConfigurationRouterDependencies(
            context=ctx,
            store_io=store_io,
        )
    )
    app.include_router(configuration_router)

    app.dependency_overrides[get_gpu_event_dependencies] = lambda: (
        GpuEventRouterDependencies(
            context=ctx,
            store_io=store_io,
            ingest_xid=fault_ingestion.ingest_xid,
            ingest_sxid=fault_ingestion.ingest_sxid,
            enrich_sxid_scope=fault_ingestion._enrich_sxid_scope,
        )
    )
    app.include_router(gpu_event_router)

    if processor is not None:
        processor.telemetry_spool_replay_handler = (
            telemetry_ingestion._ingest_processor_telemetry_batch_core
        )


def _install_domain_routes(
    app,
    ctx,
    processor,
    store_io,
    fault_ingestion,
    telemetry_context,
    telemetry_ingestion,
    node_health_ingestion,
    processor_replay_authorized,
):
    app.dependency_overrides[get_collector_dependencies] = lambda: (
        CollectorRouterDependencies(
            context=ctx,
            store_io=store_io,
            processor=processor,
            replay_authorized=processor_replay_authorized,
            enrich_workload_context=(telemetry_context._enrich_workload_context),
            enrich_sxid_scope=fault_ingestion._enrich_sxid_scope,
            resolve_kernel_xid_gpu_uuid=(fault_ingestion._resolve_kernel_xid_gpu_uuid),
            record_collector_status=(telemetry_context._record_collector_status),
            capture_evidence=telemetry_context._capture_evidence,
            ingest_xid=fault_ingestion.ingest_xid,
            ingest_sxid=fault_ingestion.ingest_sxid,
            ingest_gpu_inventory_batch=(telemetry_context._ingest_gpu_inventory_batch),
            persist_gpu_metrics=(telemetry_ingestion._persist_gpu_metrics),
            finish_gpu_metrics=(telemetry_ingestion._finish_gpu_metrics),
            ingest_node_health_findings=(node_health_ingestion.ingest),
            persist_host_telemetry=(telemetry_ingestion._persist_host_telemetry),
            finish_host_telemetry=(telemetry_ingestion._finish_host_telemetry),
            ingest_telemetry_batch=(
                telemetry_ingestion._ingest_processor_telemetry_batch_core
            ),
            ingest_unresolved_signals=fault_ingestion.ingest_unresolved_signals,
        )
    )
    app.include_router(collector_router)

    app.dependency_overrides[get_telemetry_dependencies] = lambda: (
        TelemetryRouterDependencies(
            context=ctx,
            store_io=store_io,
            ingest_node_health_findings=(node_health_ingestion.ingest),
        )
    )
    app.include_router(telemetry_router)

    app.dependency_overrides[get_incident_dependencies] = lambda: (
        IncidentRouterDependencies(
            context=ctx,
            store_io=store_io,
            notification_owner=(_pod_process_owner() or f"manual:{os.getpid()}"),
            notification_lease_seconds=int(
                os.getenv("GPU_FAULT_NOTIFICATION_LEASE_SECONDS", "120")
            ),
            notification_max_attempts=int(
                os.getenv("GPU_FAULT_NOTIFICATION_MAX_ATTEMPTS", "8")
            ),
        )
    )
    app.include_router(incident_router)

    app.dependency_overrides[get_workflow_dependencies] = lambda: (
        WorkflowRouterDependencies(
            context=ctx,
            store_io=store_io,
        )
    )
    app.include_router(workflow_router)

    app.dependency_overrides[get_completion_dependencies] = lambda: (
        CompletionRouterDependencies(
            context=ctx,
            store_io=store_io,
        )
    )
    app.include_router(completion_router)


def _request_budgets() -> tuple[float, float, float]:
    request = float(os.getenv("GPU_FAULT_REQUEST_BUDGET_SECONDS", "15"))
    fault = float(os.getenv("GPU_FAULT_FAULT_REQUEST_BUDGET_SECONDS", "30"))
    telemetry = float(os.getenv("GPU_FAULT_TELEMETRY_REQUEST_BUDGET_SECONDS", "30"))
    if telemetry <= 0:
        raise ValueError("GPU_FAULT_TELEMETRY_REQUEST_BUDGET_SECONDS must be positive")
    return request, fault, telemetry


def create_app(context: ApplicationContext | None = None) -> FastAPI:
    # Before ApplicationContext, whose construction already logs.
    configure_logging()
    validate_direct_client_identity_environment()
    ctx = context or ApplicationContext.from_environment()
    fault_ingestion = FaultIngestionService(ctx)
    # Exposed so /metrics can render fault_ingestion.unresolved_signal_totals.
    ctx.fault_ingestion = fault_ingestion
    telemetry_context = TelemetryContextService(ctx)
    node_health_ingestion = NodeHealthIngestionService(ctx)
    telemetry_ingestion = TelemetryIngestionService(
        ctx,
        telemetry_context,
        node_health_ingestion,
        fault_ingestion,
    )
    processor_replay_tracker = ProcessorReplayTracker()
    processor_mode = os.getenv("GPU_FAULT_PROCESSOR_MODE", "direct").strip().lower()
    processor_exit_grace_seconds = float(
        os.getenv("GPU_FAULT_PROCESSOR_EXIT_GRACE_SECONDS", "5")
    )
    processor = ProcessorFactory(
        ctx, mode=processor_mode, exit_grace_seconds=processor_exit_grace_seconds
    ).build()
    admission = AdmissionRuntimeFactory(ctx, processor).build()
    processor_max_queue_depth = admission.max_queue_depth
    processor_max_cluster_queue_depth = admission.max_cluster_queue_depth
    processor_fault_reserved_queue_depth = admission.fault_reserved_depth
    processor_fault_reserved_cluster_depth = admission.fault_reserved_cluster_depth
    processor_max_request_bytes = admission.max_request_bytes
    processor_global_admission_guard = admission.global_admission_guard
    processor_admission_rejections = admission.admission_rejections
    processor_admission_rejections_by_path = admission.admission_rejections_by_path
    dispatch_state = admission.dispatch_state
    processor_queue_bypass_paths = admission.queue_bypass_paths
    processor_queue_bypass_enabled = admission.queue_bypass_enabled
    processor_queue_bypasses_by_path = admission.queue_bypasses_by_path
    telemetry_spool_enabled = admission.spool_enabled
    telemetry_spool_max_item_bytes = admission.spool_max_item_bytes
    telemetry_spool_rejections = admission.spool_rejections
    telemetry_spool_admitted_by_path = admission.spool_admitted_by_path
    store_io = admission.store_io
    decode_io = admission.decode_io
    fault_store_io = admission.fault_store_io
    evidence_store_io = admission.evidence_store_io
    fault_decode_io = admission.fault_decode_io
    telemetry_spool_store_io = admission.spool_store_io
    processor_admission_batcher = admission.admission_batcher
    fault_admission_batcher = admission.fault_batcher
    evidence_admission_batcher = admission.evidence_batcher
    telemetry_spool_batcher = admission.spool_batcher
    decode_json_body = admission.decode_json_body

    (
        regional_auth_registry,
        service_role,
        background_services_enabled,
        processor_diagnostics_publisher,
        ingress_fault_semaphore,
        ingress_normal_semaphore,
        ingress_fault_wait_seconds,
        ingress_normal_wait_seconds,
        ingress_backpressure_rejections,
        event_loop_lag,
        collector_metrics_snapshot,
        lifespan,
        local_processor_diagnostics,
        remote_command_wakeups,
    ) = _configure_service_runtime(
        ctx,
        processor,
        processor_exit_grace_seconds,
        processor_replay_tracker,
        node_health_ingestion,
        store_io,
        decode_io,
        fault_store_io,
        evidence_store_io,
        fault_decode_io,
        telemetry_spool_store_io,
        processor_admission_batcher,
        fault_admission_batcher,
        evidence_admission_batcher,
        telemetry_spool_batcher,
    )

    app = FastAPI(
        title="GPU Fault Control Plane",
        version="0.10.0",
        description=(
            "Portable completion, NVIDIA policy, and recovery-plan API. "
            f"Workflow executor mode: {ctx.executor_mode}."
        ),
        lifespan=lifespan,
    )
    app.state.remote_command_wakeups = remote_command_wakeups
    app.state.context = ctx
    app.state.processor = processor
    app.state.processor_replay_tracker = processor_replay_tracker
    app.state.processor_diagnostics_publisher = processor_diagnostics_publisher
    app.state.store_io = store_io
    app.state.fault_store_io = fault_store_io
    app.state.evidence_store_io = evidence_store_io
    app.state.fault_decode_io = fault_decode_io
    app.state.telemetry_spool_store_io = telemetry_spool_store_io
    # Exposed so a test can stall one scope and read the per-scope
    # series back out of /metrics, which is the only way to attribute a
    # stall to a cluster when scrapes land on a random uvicorn process.
    app.state.processor_admission_batcher = processor_admission_batcher
    app.state.fault_admission_batcher = fault_admission_batcher
    app.state.evidence_admission_batcher = evidence_admission_batcher
    app.state.telemetry_spool_batcher = telemetry_spool_batcher
    app.state.regional_registry_runtime = regional_auth_registry
    processor_paths = (
        "/v1/runtime-profiles",
        "/v1/installation-resources",
        "/v1/markers",
        "/v1/gpu-events/",
        "/v1/provider-events/",
        "/v1/collector-events/",
        "/v1/efa-traffic/admin-actions",
        TRAINING_PROGRESS_PATH,
        "/v1/training-health/",
        "/v1/incidents/",
        "/v1/advisory-notifications/",
        "/v1/workflows/",
        WORKLOAD_OBSERVATIONS_PATH,
        # Also covers ATTEMPT_COVERAGE_PATH, the watcher's coverage heartbeat.
        "/v1/attempts/",
        "/v1/recovery-plans/",
        "/v1/fleet/deployments",
        "/v1/fleet/agents/",
    )
    loopback_clients = frozenset({"127.0.0.1", "::1", "localhost"})

    def processor_replay_authorized(request: Request) -> bool:
        client_host = request.client.host if request.client else None
        return bool(
            processor is not None
            and client_host in loopback_clients
            and processor_internal_token_valid(
                request.headers.get("X-GPU-Fault-Processor-Replay"),
                processor.internal_token,
            )
        )

    def requires_processor(request: Request) -> bool:
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return False
        path = request.url.path
        if path == "/v1/fleet/agents/heartbeat":
            return False
        return path.startswith(processor_paths)

    def returns_processor_receipt(path: str) -> bool:
        channel = channel_for_path(path)
        return bool(channel and channel.receipt) or path.startswith("/v1/attempts/")

    (
        request_budget_seconds,
        fault_request_budget_seconds,
        telemetry_request_budget_seconds,
    ) = _request_budgets()
    install_processor_dispatch(
        app,
        ProcessorDispatchDependencies(
            context=ctx,
            processor=processor,
            state=dispatch_state,
            store_io=store_io,
            decode_io=decode_io,
            fault_store_io=fault_store_io,
            fault_decode_io=fault_decode_io,
            processor_admission_batcher=processor_admission_batcher,
            fault_admission_batcher=fault_admission_batcher,
            evidence_admission_batcher=evidence_admission_batcher,
            telemetry_spool_batcher=telemetry_spool_batcher,
            processor_replay_tracker=processor_replay_tracker,
            requires_processor=requires_processor,
            replay_authorized=processor_replay_authorized,
            returns_processor_receipt=returns_processor_receipt,
            is_fault_ingress_path=is_fault_path,
            decode_rejections=admission.decode_rejections,
            decode_json_body=decode_json_body,
            processor_max_queue_depth=processor_max_queue_depth,
            processor_max_cluster_queue_depth=(processor_max_cluster_queue_depth),
            processor_fault_reserved_queue_depth=(processor_fault_reserved_queue_depth),
            processor_fault_reserved_cluster_depth=(
                processor_fault_reserved_cluster_depth
            ),
            processor_global_admission_guard=(processor_global_admission_guard),
            processor_max_request_bytes=processor_max_request_bytes,
            processor_retry_after_seconds=admission.retry_after_seconds,
            processor_response_timeout_seconds=admission.response_timeout_seconds,
            processor_queue_bypass_enabled=(processor_queue_bypass_enabled),
            processor_queue_bypass_paths=processor_queue_bypass_paths,
            processor_admission_rejections=(processor_admission_rejections),
            processor_admission_rejections_by_path=(
                processor_admission_rejections_by_path
            ),
            processor_queue_bypasses_by_path=(processor_queue_bypasses_by_path),
            telemetry_spool_enabled=telemetry_spool_enabled,
            telemetry_spool_max_item_bytes=(telemetry_spool_max_item_bytes),
            telemetry_spool_rejections=telemetry_spool_rejections,
            telemetry_spool_admitted_by_path=(telemetry_spool_admitted_by_path),
            telemetry_request_budget_seconds=(telemetry_request_budget_seconds),
        ),
    )

    _install_core_routes(
        app,
        ctx,
        processor,
        processor_mode,
        service_role,
        store_io,
        decode_io,
        fault_store_io,
        evidence_store_io,
        fault_decode_io,
        telemetry_spool_store_io,
        processor_replay_tracker,
        processor_diagnostics_publisher,
        local_processor_diagnostics,
        regional_auth_registry,
        fault_ingestion,
        telemetry_ingestion,
        admission,
        background_services_enabled,
        ingress_backpressure_rejections,
        event_loop_lag,
        collector_metrics_snapshot,
    )

    _install_domain_routes(
        app,
        ctx,
        processor,
        store_io,
        fault_ingestion,
        telemetry_context,
        telemetry_ingestion,
        node_health_ingestion,
        processor_replay_authorized,
    )
    authorization_registry = _install_regional_auth(
        app,
        ctx,
        regional_auth_registry,
        processor_replay_authorized,
        decode_io,
        decode_json_body,
        processor_max_request_bytes,
        fault_decode_io=fault_decode_io,
        is_fault_path=is_fault_path,
        admission=admission,
    )

    install_ingress_backpressure(
        app,
        IngressBackpressureDependencies(
            service_role=service_role,
            requires_processor=requires_processor,
            replay_authorized=processor_replay_authorized,
            is_fault_path=is_fault_path,
            fault_semaphore=ingress_fault_semaphore,
            normal_semaphore=ingress_normal_semaphore,
            fault_wait_seconds=ingress_fault_wait_seconds,
            normal_wait_seconds=ingress_normal_wait_seconds,
            rejections=ingress_backpressure_rejections,
        ),
    )
    install_request_deadline(
        app,
        request_budget_seconds=request_budget_seconds,
        fault_request_budget_seconds=fault_request_budget_seconds,
        is_fault_path=is_fault_path,
    )

    authorization_inventory = dict(authorization_registry.inventory)
    collector_route_paths = {
        route.path
        for route in iter_api_routes(app.routes)
        if route.path.startswith(COLLECTOR_EVENT_PREFIX)
    }
    validate_collector_routes(collector_route_paths)
    app.state.regional_authorization_inventory = authorization_inventory
    app.state.service_role = service_role
    app.state.regional_authorization_bucket = authorization_registry.declared

    return app


def run() -> None:
    import uvicorn

    uvicorn.run(
        "gpu_fault.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=8080,
        reload=False,
    )


if __name__ == "__main__":
    run()
