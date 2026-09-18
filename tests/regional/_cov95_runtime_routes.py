"""Exercise router handlers with real models and an offline Store I/O boundary."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gpu_fault.app import ApplicationContext
from gpu_fault.app.routes import fleet, regional
from gpu_fault.regional_compatibility import RegionalExecutorCompatibilityPolicy
from tests._builders import build_context
from tests.app_services.test_periodic_registry_heartbeat import CLUSTER_A
from tests.fleet._support import heartbeat, registry, signed

EXECUTION_TOKEN = "synthetic-runtime-execution-token"
HEADERS = {
    "X-GPU-Fault-Cluster-ID": "cluster-a",
    "X-GPU-Fault-Execution-Token": EXECUTION_TOKEN,
}


class StoreIo:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.error: Exception | None = None

    async def run(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(function.__name__)
        if self.error is not None:
            raise self.error
        return function(*args, **kwargs)


@dataclass
class Routes:
    context: ApplicationContext
    client: TestClient
    io: StoreIo


@pytest.fixture(name="routes")
def routes_fixture() -> Iterator[Routes]:
    fleet_registry = registry()
    context = build_context(store=fleet_registry.store)
    context.fleet_registry = fleet_registry
    context.regional_mode = True
    context.execution_token = EXECUTION_TOKEN
    context.store.save_regional_cluster(CLUSTER_A)
    for node in ("node-a", "node-b"):
        fleet_registry.register(signed(heartbeat(node)))
    io = StoreIo()
    app = FastAPI()
    app.include_router(fleet.router)
    app.include_router(regional.router)
    fleet_dependencies = fleet.FleetRouterDependencies(context=context, store_io=io)
    regional_dependencies = regional.RegionalRouterDependencies(
        context=context,
        store_io=io,
        auth_registry={"cluster-a": CLUSTER_A},
        max_unclaimed_seconds=30,
        max_claim_age_seconds=30,
        executor_compatibility=RegionalExecutorCompatibilityPolicy.from_mapping({}),
    )
    app.dependency_overrides[fleet.get_fleet_dependencies] = lambda: fleet_dependencies
    app.dependency_overrides[regional.get_regional_dependencies] = (
        lambda: regional_dependencies
    )
    with TestClient(app) as client:
        yield Routes(context, client, io)
