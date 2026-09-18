from __future__ import annotations

import ipaddress
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app.context import (
    register_spare_reservation_release,
    spare_reservation_release_hook,
)
from gpu_fault.fleet import FleetRegistry
from gpu_fault.fleet_endpoint import endpoint_networks_for_cluster
from gpu_fault.models import WorkflowStatus
from gpu_fault.regional import RegionalRegistryRevision
from gpu_fault.regional_registry_runtime import RegionalRegistryRuntime
from tests._builders import build_context, workflow_request
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime
from tests.regional._regional_support import TOKEN_A, registration


@pytest.mark.parametrize("change", ["same", "cidrs", "removed"])
def test_bound_context_uses_current_registry_for_preexisting_cluster_cidrs(
    change: str,
) -> None:
    now = datetime.now(timezone.utc)
    context = build_context()
    context.regional_mode = True
    original = registration("cluster-a", TOKEN_A)
    context.store.save_regional_cluster(original)
    initial = (ipaddress.ip_network("10.0.0.0/16"),)
    context.fleet_registry = FleetRegistry(
        context.store,
        "m" * 32,
        endpoint_allowed_networks_by_cluster={"cluster-a": initial},
    )
    runtime = RegionalRegistryRuntime.bootstrap(
        context.store,
        member_id="unit-member",
        service_role="ingress",
        release_id="unit-release",
        poll_seconds=1,
        stale_seconds=5,
        now=lambda: now,
    )
    context.bind_regional_registry_runtime(runtime)
    current = runtime.snapshot()
    changed = (
        original.model_copy(update={"agent_endpoint_allowed_cidrs": ["10.77.0.0/16"]})
        if change == "cidrs"
        else original
    )
    context.store.publish_regional_registry_revision(
        RegionalRegistryRevision.build(
            generation=current.generation + 1,
            registrations=[] if change == "removed" else [changed],
            previous_generation=current.generation,
            required_member_ids=[],
            reason="unit policy revision",
            created_at=now,
        ),
        expected_generation=current.generation,
    )
    runtime.refresh_once(raise_on_failure=True)
    networks = context.fleet_registry.endpoint_allowed_networks_by_cluster
    if change == "removed":
        with pytest.raises(ValueError, match="unavailable for cluster cluster-a"):
            endpoint_networks_for_cluster("cluster-a", networks, initial)
    else:
        expected = (
            (ipaddress.ip_network("10.77.0.0/16"),) if change == "cidrs" else initial
        )
        assert endpoint_networks_for_cluster("cluster-a", networks, initial) == expected


@pytest.mark.parametrize("failure", [False, True])
def test_registry_bound_observers_cache_one_instance_and_refuse_unknown_state(
    failure: bool,
) -> None:
    context = build_context()
    context.regional_mode = True
    context.fleet_registry = FleetRegistry(context.store, "m" * 32)
    existing = object()
    observers = SimpleNamespace(observers={"existing": existing})
    context.regional_managed_observer = observers
    registrations = {"cluster-a": registration("cluster-a", TOKEN_A)}
    builds = []

    def factory(value: Any) -> object:
        builds.append(value.cluster_id)
        return object()

    def snapshot() -> dict[str, Any]:
        if failure:
            raise RuntimeError("synthetic registry snapshot unavailable")
        return registrations

    context.regional_observer_factory = factory
    context.bind_regional_registry_runtime(SimpleNamespace(snapshot=snapshot))
    sentinel = object()
    assert observers.observers.get("existing") is existing
    assert observers.observers.get("missing", sentinel) is sentinel
    first = observers.observers.get("cluster-a", sentinel)
    second = observers.observers.get("cluster-a", sentinel)
    assert first is second
    assert builds == ([] if failure else ["cluster-a"])
    if failure:
        assert first is sentinel
        with pytest.raises(ValueError, match="unavailable"):
            endpoint_networks_for_cluster(
                "cluster-a",
                context.fleet_registry.endpoint_allowed_networks_by_cluster,
                (ipaddress.ip_network("10.0.0.0/8"),),
            )


def test_concurrent_observer_resolution_builds_one_shared_instance() -> None:
    context = build_context()
    context.regional_mode = True
    context.regional_managed_observer = SimpleNamespace(observers={})
    registered = registration("cluster-a", TOKEN_A)
    entered = Event()
    release = Event()
    started = Barrier(5)
    builds = []

    def factory(value: Any) -> object:
        builds.append(value.cluster_id)
        entered.set()
        assert release.wait(5), "test did not release the observer factory"
        return object()

    context.regional_observer_factory = factory
    context.bind_regional_registry_runtime(
        SimpleNamespace(snapshot=lambda: {"cluster-a": registered})
    )
    observers = context.regional_managed_observer.observers

    def read() -> Any:
        started.wait(5)
        return observers.get("cluster-a")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(read) for _ in range(4)]
        started.wait(5)
        assert entered.wait(5), "observer factory never started"
        release.set()
        results = [future.result(timeout=5) for future in futures]
    assert builds == ["cluster-a"]
    assert all(value is results[0] for value in results), (
        "cluster observers were built more than once"
    )


@pytest.mark.parametrize("regional", [False, True])
def test_absent_registry_runtime_does_not_install_a_resolver(regional: bool) -> None:
    context = build_context()
    context.regional_mode = regional
    context.fleet_registry = FleetRegistry(context.store, "m" * 32)
    before = context.fleet_registry.endpoint_allowed_networks_by_cluster
    context.bind_regional_registry_runtime(None)
    assert context.fleet_registry.endpoint_allowed_networks_by_cluster is before


@pytest.mark.parametrize(
    "status",
    [WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.SUPERSEDED],
)
@pytest.mark.parametrize("released", [[], ["node-b", "node-a"]])
def test_spare_release_hook_preserves_successful_failover_and_releases_failed_claims(
    status: WorkflowStatus, released: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    calls = []
    adapter = SimpleNamespace(
        release_spare_reservations=lambda *args: calls.append(args) or released
    )
    workflow = workflow_request("unit-workflow", "unit-incident", status=status)
    hook = spare_reservation_release_hook(adapter)
    hook(workflow, object(), [])
    if status is WorkflowStatus.SUCCEEDED:
        assert calls == []
    else:
        assert calls == [(workflow, workflow.incident_id)]
        assert ("released warm spare reservations" in caplog.text) is bool(released)


def test_spare_hooks_are_registered_only_for_capable_adapters() -> None:
    callbacks = []
    executor = SimpleNamespace(on_terminal=callbacks)
    adapter = SimpleNamespace(release_spare_reservations=lambda *args: [])
    assert register_spare_reservation_release(executor, [object(), adapter]) == 1
    assert len(callbacks) == 1
