from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app.ingest.node_health import NodeHealthIngestionService
from gpu_fault.app.routes import configuration, processor, telemetry
from tests._builders import build_context
from tests.regional._cov95_runtime_routes import EXECUTION_TOKEN, HEADERS, StoreIo


class ApiIo(StoreIo):
    workers = 1
    max_in_flight = 1
    in_flight = 0
    rejected_total = 0


@dataclass
class Api:
    context: ApplicationContext
    client: TestClient
    io: ApiIo


@contextmanager
def full_app(context: ApplicationContext) -> Iterator[tuple[FastAPI, TestClient]]:
    app = create_app(context)
    client = TestClient(app)
    try:
        yield app, client
    finally:
        client.close()
        for name in (
            "store_io",
            "decode_io",
            "fault_store_io",
            "evidence_store_io",
            "fault_decode_io",
            "telemetry_spool_store_io",
        ):
            getattr(app.state.runtime, name).close()


@pytest.fixture(name="api")
def api_fixture() -> Iterator[Api]:
    context = build_context()
    context.execution_token = EXECUTION_TOKEN
    io = ApiIo()
    app = FastAPI()
    app.include_router(configuration.router)
    app.include_router(telemetry.router)
    app.include_router(processor.router)
    configuration_dependencies = configuration.ConfigurationRouterDependencies(
        context=context, store_io=io
    )
    telemetry_dependencies = telemetry.TelemetryRouterDependencies(
        context=context,
        store_io=io,
        ingest_node_health_findings=NodeHealthIngestionService(context).ingest,
    )
    processor_dependencies = processor.ProcessorRouterDependencies(
        context=context,
        processor=None,
        processor_mode="disabled",
        store_io=io,
        telemetry_spool_store_io=io,
        evidence_store_io=io,
        diagnostics=lambda: {},
        diagnostics_publisher=SimpleNamespace(read_all=lambda: []),
        max_queue_depth=100,
        max_cluster_queue_depth=10,
        fault_reserved_queue_depth=10,
        fault_reserved_cluster_depth=1,
        max_request_bytes=1024,
        global_admission_guard=100,
    )
    app.dependency_overrides[configuration.get_configuration_dependencies] = (
        lambda: configuration_dependencies
    )
    app.dependency_overrides[telemetry.get_telemetry_dependencies] = (
        lambda: telemetry_dependencies
    )
    app.dependency_overrides[processor.get_processor_dependencies] = (
        lambda: processor_dependencies
    )
    with TestClient(app, headers=HEADERS) as client:
        yield Api(context, client, io)
