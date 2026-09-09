"""Liveness must not depend on Aurora (architecture review H1).

Every control-plane Deployment used ``/healthz`` for both probes, and
``/healthz`` answers 503 whenever the regional registry runtime has not
refreshed from Aurora within its stale window. A 30-60 s writer failover
therefore restarted every control-plane Pod, and the restarted process
needed Aurora again to bootstrap. ``/livez`` asserts only process-local
liveness; readiness keeps ``/healthz``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.app import create_app
from gpu_fault.app.lifespan_workers import start_processor_threads
from gpu_fault.regional_registry_runtime import RegionalRegistryRuntime
from gpu_fault.store import InMemoryStore
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import TOKEN_A, registration

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "deploy/control-plane/regional/generated"
DEPLOYMENTS = (
    "gpu-fault-api-ha-ingress.yaml",
    "gpu-fault-control-worker.yaml",
    "gpu-fault-telemetry-spool-worker.yaml",
)
NOW = datetime(2026, 9, 7, tzinfo=timezone.utc)


def _regional_context():
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    return context


def test_livez_answers_200_while_registry_runtime_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(_regional_context())
    monkeypatch.setattr(app.state.regional_registry_runtime, "is_ready", lambda: False)

    async def scenario() -> tuple[int, dict, int]:
        async with asgi_client(app) as client:
            live = await client.get("/livez")
            ready = await client.get("/healthz")
        return live.status_code, live.json(), ready.status_code

    live_status, live_body, ready_status = asyncio.run(scenario())

    assert live_status == 200, "liveness followed the Aurora-backed registry"
    assert live_body["status"] == "alive", live_body
    assert ready_status == 503, "readiness must still fail closed on a stale registry"


def test_livez_is_a_public_route() -> None:
    app = create_app(_regional_context())

    assert app.state.regional_authorization_bucket("/livez") == "public", (
        "/livez must be reachable by the kubelet without credentials"
    )


def test_livez_fails_when_the_spool_consumer_thread_exited() -> None:
    from gpu_fault.app.routes.admin import AdminRouterDependencies, livez

    dependencies = AdminRouterDependencies(
        context=SimpleNamespace(),
        processor=SimpleNamespace(spool_consumer_running=False),
        processor_mode="queued",
        service_role="spool-worker",
        environment={},
        regional_registry_runtime=None,
    )

    response = asyncio.run(livez(dependencies))

    assert response.status_code == 503, "a dead spool consumer is a dead process"


@pytest.mark.parametrize("manifest", DEPLOYMENTS)
def test_generated_deployments_probe_livez_for_liveness_and_healthz_for_readiness(
    manifest: str,
) -> None:
    documents = [
        item
        for item in yaml.safe_load_all((GENERATED / manifest).read_text("utf-8"))
        if item and item.get("kind") == "Deployment"
    ]
    assert len(documents) == 1, f"{manifest} must render exactly one Deployment"
    container = documents[0]["spec"]["template"]["spec"]["containers"][0]

    assert container["livenessProbe"]["httpGet"]["path"] == "/livez", (
        f"{manifest} liveness still depends on the Aurora-backed /healthz"
    )
    assert container["readinessProbe"]["httpGet"]["path"] == "/healthz", (
        f"{manifest} readiness must keep the registry-aware /healthz"
    )


def test_regional_patch_stale_window_covers_an_aurora_writer_failover() -> None:
    patch = yaml.safe_load(
        (
            ROOT / "deploy/control-plane/regional/regional-control-plane-patch.yaml"
        ).read_text("utf-8")
    )
    container = patch["spec"]["template"]["spec"]["containers"][0]
    values = {item["name"]: item.get("value") for item in container["env"]}

    assert float(values["GPU_FAULT_REGISTRY_STALE_SECONDS"]) >= 60, (
        "the registry stale window is shorter than an Aurora writer failover"
    )


class _FlakyStore(InMemoryStore):
    """Aurora that is unreachable for the first ``failures`` head reads."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.attempts = 0

    def get_regional_registry_head(self):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise ConnectionError("writer failover in progress")
        return super().get_regional_registry_head()


def test_bootstrap_retries_with_backoff_while_the_store_is_unavailable() -> None:
    store = _FlakyStore(failures=3)
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    clock = [NOW]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] = clock[0] + timedelta(seconds=seconds)

    runtime = RegionalRegistryRuntime.bootstrap(
        store,
        member_id="pod-a",
        service_role="ingress",
        release_id="release-a",
        poll_seconds=1,
        stale_seconds=5,
        now=lambda: clock[0],
        retry_budget_seconds=120,
        sleep=sleep,
    )

    assert runtime.is_ready(), runtime.status()
    assert store.attempts > 3, "bootstrap gave up on the first store failure"
    assert sleeps == sorted(sleeps), "backoff must grow, not shrink"
    assert len(sleeps) >= 3, "each failed attempt must wait before retrying"


def test_bootstrap_fails_once_the_retry_budget_is_exhausted() -> None:
    store = _FlakyStore(failures=10_000)
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    clock = [NOW]

    def sleep(seconds: float) -> None:
        clock[0] = clock[0] + timedelta(seconds=seconds)

    with pytest.raises(ConnectionError, match="writer failover"):
        RegionalRegistryRuntime.bootstrap(
            store,
            member_id="pod-a",
            service_role="ingress",
            release_id="release-a",
            poll_seconds=1,
            stale_seconds=5,
            now=lambda: clock[0],
            retry_budget_seconds=30,
            sleep=sleep,
        )
    assert store.attempts < 100, "bootstrap must stop retrying after the budget"


def test_dispatch_thread_survives_an_exception_from_the_health_check() -> None:
    """One raise in the readiness guard must not kill the dispatcher thread."""

    stop = Event()
    health_calls = {"count": 0}
    dispatched = Event()

    def is_healthy() -> bool:
        health_calls["count"] += 1
        if health_calls["count"] == 1:
            raise RuntimeError("health latch unavailable")
        return True

    processor = SimpleNamespace(
        is_healthy=is_healthy,
        active_consumers=True,
        is_leader=lambda: True,
        run_processor=lambda: stop.wait(),
        run_leadership=lambda: stop.wait(),
        telemetry_spool_enabled=False,
    )
    dispatcher = SimpleNamespace(
        config=SimpleNamespace(enabled=True, poll_interval_seconds=0.01),
        consume_wake=lambda: False,
        run_once=lambda: dispatched.set(),
    )
    context = SimpleNamespace(
        dispatcher=dispatcher,
        xid_correlation=SimpleNamespace(
            run_once=lambda: None, poll_interval_seconds=60
        ),
        store=SimpleNamespace(),
        periodic_runner=None,
    )
    diagnostics = SimpleNamespace(run=lambda _stop: None)

    class _Runner:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self) -> None:
            stop.wait()

    import gpu_fault.app.lifespan_workers as workers

    original = workers.PeriodicServiceRunner
    workers.PeriodicServiceRunner = _Runner  # type: ignore[misc,assignment]
    try:
        start_processor_threads(
            context=context,
            processor=processor,
            diagnostics_publisher=diagnostics,
            stop=stop,
            identity_registries=[],
            ingest_node_health_findings=lambda *_args: None,
            notify_silent_collectors=lambda *_args: None,
        )
        assert dispatched.wait(5), (
            "dispatcher thread died after the health check raised"
        )
    finally:
        stop.set()
        workers.PeriodicServiceRunner = original  # type: ignore[misc]


# --- B-6: the worker role's consumer loop is a liveness criterion --------------


def _worker_livez(processor) -> Any:
    from gpu_fault.app.routes.admin import AdminRouterDependencies, livez

    return asyncio.run(
        livez(
            AdminRouterDependencies(
                context=SimpleNamespace(),
                processor=processor,
                processor_mode="active-active",
                service_role="worker",
                environment={"GPU_FAULT_SERVICE_ROLE": "worker"},
                regional_registry_runtime=None,
            )
        )
    )


def _consumer(*, running: bool, age: float | None, live: bool) -> SimpleNamespace:
    return SimpleNamespace(
        active_consumers=True,
        spool_consumer_running=True,
        processor_consumer_running=running,
        consumer_last_cycle_age_seconds=age,
        consumer_is_live=lambda *, max_cycle_age_seconds: live,
        processor_notification_fallback_seconds=5.0,
    )


def test_livez_fails_when_the_processor_consumer_thread_exited() -> None:
    """``gpu-fault-processor-inbox`` died: Pod stayed Ready, claimed nothing."""

    response = _worker_livez(_consumer(running=False, age=12.0, live=False))

    assert response.status_code == 503
    assert "processor-consumer" in response.body.decode()


def test_livez_fails_when_the_processor_consumer_is_wedged() -> None:
    response = _worker_livez(_consumer(running=True, age=600.0, live=False))

    assert response.status_code == 503


def test_livez_gives_the_consumer_a_start_up_grace() -> None:
    """Before the first cycle there is nothing to be stale about."""

    payload = _worker_livez(_consumer(running=False, age=None, live=False))

    assert payload["status"] == "alive", payload


def test_livez_is_alive_while_the_consumer_cycles() -> None:
    payload = _worker_livez(_consumer(running=True, age=0.5, live=True))

    assert payload["status"] == "alive", payload
    assert payload["dead_threads"] == []


def test_livez_ignores_the_consumer_outside_the_worker_role() -> None:
    from gpu_fault.app.routes.admin import AdminRouterDependencies, livez

    payload = asyncio.run(
        livez(
            AdminRouterDependencies(
                context=SimpleNamespace(),
                processor=_consumer(running=False, age=12.0, live=False),
                processor_mode="active-active",
                service_role="ingress",
                environment={},
                regional_registry_runtime=None,
            )
        )
    )

    assert payload["status"] == "alive", payload
