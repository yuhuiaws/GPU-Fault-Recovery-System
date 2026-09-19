"""Hot pin reload through the application.

The runtime a test binds on the context is the one the app serves: executor
admission and the mint-time batching gate read it per request, ``/healthz``
and ``/v1/version`` report it, the fleet registry's policy follows it, and the
app lifespan runs its poller. No Pod restart is involved anywhere below.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from gpu_fault import module_digest
from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.fleet_pins import FleetPinRuntime
from gpu_fault.regional import RegionalRemoteWorkflowAdapter
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from tests._builders import asgi_client, build_context
from tests.fleet._support import ARTIFACT, registry
from tests.regional._batching_support import (
    ALL_OWNERS,
    chain_state,
    reset_chain,
    step_context,
)
from tests.regional._fleet_pin_support import (
    CONFIG_MAP,
    EXECUTOR_ARTIFACT,
    NAMESPACE,
    NEW_ARTIFACT,
    NEW_EXECUTOR_ARTIFACT,
    FakeConfigMapReader,
    config_map_data,
    never_read,
    startup_environment,
)
from tests.regional._regional_support import NOW, TOKEN_A, registration

EXECUTION_TOKEN = "x" * 32
EXECUTION_HEADERS = {"X-GPU-Fault-Execution-Token": EXECUTION_TOKEN}
CLUSTER_HEADERS = {
    "Authorization": f"Bearer {TOKEN_A}",
    "X-GPU-Fault-Cluster-ID": "cluster-a",
}
READINESS = "/v1/regional/executors/readiness"
FLEET_PIN_STATUS_KEYS = {
    "source",
    "config_map",
    "content_sha256",
    "resource_version",
    "observed_at",
    "generation",
    "error",
}


def _context(reader=never_read, *, poll_seconds: float = 1.0) -> ApplicationContext:
    context = build_context(execution_token=EXECUTION_TOKEN)
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.fleet_pin_runtime = FleetPinRuntime(
        startup_environment(),
        config_map=CONFIG_MAP,
        namespace=NAMESPACE,
        poll_seconds=poll_seconds,
        reader=reader,
        now=lambda: NOW,
    )
    return context


def _probe(artifact: str) -> dict[str, object]:
    return {
        "executor_id": "executor-b",
        "executor_protocol_version": CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        "executor_artifact_sha256": artifact,
        "execution_owners": ["gpu-fault-kubernetes-adapter"],
        "last_successful_claim_age_seconds": 1.0,
    }


# --- Task 2: executor admission and the mint-time gate --------------------------


def test_executor_admission_follows_the_pin_window_without_a_restart() -> None:
    context = _context()
    runtime = context.fleet_pin_runtime
    app = create_app(context)
    assert app.state.fleet_pin_runtime is runtime, (
        "the app must serve the bound runtime"
    )

    async def scenario():
        async with asgi_client(app) as client:
            refused = await client.post(
                READINESS, headers=CLUSTER_HEADERS, json=_probe(NEW_EXECUTOR_ARTIFACT)
            )
            # Stage: the target joins the compatible sets, required stays.
            runtime.apply(
                config_map_data(
                    **{
                        "compatible-regional-executor-artifact-sha256s": (
                            NEW_EXECUTOR_ARTIFACT
                        ),
                        "compatible-regional-executor-compatibility-digests": (
                            NEW_EXECUTOR_ARTIFACT
                        ),
                    }
                ),
                resource_version="20",
            )
            widened = await client.post(
                READINESS, headers=CLUSTER_HEADERS, json=_probe(NEW_EXECUTOR_ARTIFACT)
            )
            # Finalize: required becomes the target, the compatible sets empty.
            runtime.apply(
                config_map_data(
                    **{
                        "required-regional-executor-artifact-sha256": (
                            NEW_EXECUTOR_ARTIFACT
                        ),
                        "compatible-regional-executor-artifact-sha256s": "",
                        "compatible-regional-executor-compatibility-digests": "",
                    }
                ),
                resource_version="21",
            )
            finalized = await client.post(
                READINESS, headers=CLUSTER_HEADERS, json=_probe(NEW_EXECUTOR_ARTIFACT)
            )
            previous = await client.post(
                READINESS, headers=CLUSTER_HEADERS, json=_probe(EXECUTOR_ARTIFACT)
            )
            return refused, widened, finalized, previous

    refused, widened, finalized, previous = asyncio.run(scenario())

    assert refused.status_code == 503, refused.text
    assert any("artifact mismatch" in reason for reason in refused.json()["reasons"]), (
        refused.json()
    )
    assert widened.status_code == 200, widened.text
    assert finalized.status_code == 200, finalized.text
    assert previous.status_code == 503, (
        "after finalize the previous executor artifact must be refused again"
    )


def test_the_mint_time_batching_gate_reads_the_served_window() -> None:
    context = _context()
    runtime = context.fleet_pin_runtime
    store = context.store
    # The provider ``_base_adapters`` hands the adapter: read at mint time.
    adapter = RegionalRemoteWorkflowAdapter(
        store, owners=ALL_OWNERS, step_batching=context.step_batching_policy
    )

    runtime.apply(
        config_map_data(**{"compatible-regional-executor-protocol-versions": "2"}),
        resource_version="30",
    )
    assert adapter.step_batching.active is False, adapter.step_batching
    incident, workflow = chain_state(store, reset_chain(), request_id="workflow-staged")
    adapter.execute(step_context(workflow, incident, 1))
    (staged,) = [
        command
        for command in store.list_remote_commands()
        if command.workflow_request_id == "workflow-staged"
    ]
    assert staged.batched_steps == [], (
        "a protocol-2 executor may still claim, so one command per step"
    )

    runtime.apply(config_map_data(), resource_version="31")
    assert adapter.step_batching.active is True, adapter.step_batching
    incident, workflow = chain_state(store, reset_chain(), request_id="workflow-final")
    adapter.execute(step_context(workflow, incident, 1))
    (final,) = [
        command
        for command in store.list_remote_commands()
        if command.workflow_request_id == "workflow-final"
    ]
    assert [step.step_index for step in final.batched_steps] == [2, 3, 4], (
        final.batched_steps
    )


# --- Task 3: routes -------------------------------------------------------------


def test_healthz_reports_fleet_pins_and_stays_200_when_the_reader_fails() -> None:
    reader = FakeConfigMapReader(config_map_data(), resource_version="9")
    context = _context(reader)
    runtime = context.fleet_pin_runtime
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            initial = await client.get("/healthz")
            reader.error = RuntimeError(
                'configmaps "gpu-fault-release-metadata" is forbidden'
            )
            runtime.refresh_once()
            degraded = await client.get("/healthz")
            return initial, degraded

    initial, degraded = asyncio.run(scenario())

    assert initial.status_code == 200, initial.text
    pins = initial.json()["fleet_pins"]
    assert set(pins) == FLEET_PIN_STATUS_KEYS, pins
    assert (pins["source"], pins["generation"], pins["error"]) == (
        "environment",
        1,
        None,
    ), pins
    assert pins["config_map"] == CONFIG_MAP, pins
    assert degraded.status_code == 200, "pins never gate readiness"
    served = degraded.json()["fleet_pins"]
    assert served["error"] == (
        'RuntimeError: configmaps "gpu-fault-release-metadata" is forbidden'
    ), served
    assert served["content_sha256"] == pins["content_sha256"], (
        "the last snapshot must still be served"
    )


def test_healthz_503_branch_still_carries_fleet_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The convergence wait reads the payload of a 503 too."""

    app = create_app(_context())
    monkeypatch.setattr(app.state.regional_registry_runtime, "is_ready", lambda: False)

    async def scenario():
        async with asgi_client(app) as client:
            return await client.get("/healthz")

    response = asyncio.run(scenario())

    assert response.status_code == 503, response.text
    assert set(response.json()["fleet_pins"]) == FLEET_PIN_STATUS_KEYS, response.json()


def test_version_reports_the_snapshot_pins_and_keeps_the_other_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GPU_FAULT_SERVICE_ROLE", raising=False)
    context = _context()
    runtime = context.fleet_pin_runtime
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            before = await client.get("/v1/version", headers=EXECUTION_HEADERS)
            runtime.apply(
                config_map_data(
                    **{
                        "required-regional-executor-artifact-sha256": (
                            NEW_EXECUTOR_ARTIFACT
                        ),
                        "compatible-regional-executor-artifact-sha256s": (
                            EXECUTOR_ARTIFACT
                        ),
                        "required-agent-artifact-sha256": NEW_ARTIFACT,
                        "compatible-agent-artifact-sha256s": ARTIFACT,
                        "compatible-agent-protocol-versions": "2",
                    }
                ),
                resource_version="12",
            )
            after = await client.get("/v1/version", headers=EXECUTION_HEADERS)
            return before, after

    before, after = asyncio.run(scenario())

    assert before.status_code == 200, before.text
    first = before.json()
    assert first["required_regional_executor_artifact_sha256"] == EXECUTOR_ARTIFACT
    assert first["required_agent_artifact_sha256"] == ARTIFACT, first
    assert first["fleet_pins"]["source"] == "environment", first["fleet_pins"]
    body = after.json()
    assert body["required_regional_executor_artifact_sha256"] == NEW_EXECUTOR_ARTIFACT
    assert body["compatible_regional_executor_artifact_sha256s"] == [EXECUTOR_ARTIFACT]
    assert body["required_agent_artifact_sha256"] == NEW_ARTIFACT, body
    assert body["compatible_agent_artifact_sha256s"] == [ARTIFACT], body
    assert body["compatible_agent_protocol_versions"] == [2], body
    assert body["fleet_pins"]["generation"] == 2, body["fleet_pins"]
    assert body["fleet_pins"]["source"] == "configmap", body["fleet_pins"]
    assert body["fleet_pins"]["resource_version"] == "12", body["fleet_pins"]
    # Everything that is not a pin keeps its source: the process, not the runtime.
    assert body["module_digest"] == module_digest(), body
    assert body["version"] == first["version"], body
    assert body["deployment_mode"] == "regional", body
    assert body["service_role"] == "combined", (
        "service_role comes from the process environment, not the pin snapshot"
    )


# --- Task 5: lifecycle wiring ---------------------------------------------------


def test_create_app_builds_a_runtime_from_the_environment_when_none_is_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "GPU_FAULT_FLEET_PIN_CONFIGMAP",
        "GPU_FAULT_FLEET_PIN_POLL_SECONDS",
        "GPU_FAULT_NAMESPACE",
    ):
        monkeypatch.delenv(name, raising=False)
    context = build_context()
    assert context.fleet_registry is None, "this context runs without the registry"

    app = create_app(context)

    runtime = app.state.fleet_pin_runtime
    assert runtime is context.fleet_pin_runtime, "one runtime per process"
    assert runtime.status()["source"] == "environment", runtime.status()
    assert runtime.config_map == "gpu-fault-release-metadata", runtime.config_map
    assert runtime.namespace == "gpu-fault-system", runtime.namespace


def test_create_app_binds_the_fleet_registry_to_the_runtime() -> None:
    fleet = registry(now=lambda: NOW)
    context = build_context(fleet_registry=fleet)
    context.fleet_pin_runtime = FleetPinRuntime(
        startup_environment(),
        config_map=CONFIG_MAP,
        namespace=NAMESPACE,
        poll_seconds=1,
        reader=never_read,
        now=lambda: NOW,
    )

    create_app(context)
    context.fleet_pin_runtime.apply(
        config_map_data(**{"compatible-agent-artifact-sha256s": NEW_ARTIFACT})
    )

    assert NEW_ARTIFACT in fleet.policy.compatible_artifact_sha256s, fleet.policy
    assert fleet.policy.required_agent_version == "0.9.0", (
        "the non-pin fields of the start-up policy must survive the rebind"
    )


def test_the_app_lifespan_runs_the_pin_poller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "direct")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    reader = FakeConfigMapReader(
        config_map_data(
            **{"compatible-regional-executor-artifact-sha256s": NEW_EXECUTOR_ARTIFACT}
        ),
        resource_version="3",
    )
    context = _context(reader, poll_seconds=0.01)
    runtime = context.fleet_pin_runtime
    app = create_app(context)
    assert reader.calls == 0, "nothing may poll before the lifespan starts"

    async def scenario():
        async with app.router.lifespan_context(app):
            deadline = time.monotonic() + 5
            while runtime.snapshot().generation < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            return runtime.snapshot()

    running = asyncio.run(scenario())

    assert running.generation == 2, runtime.status()
    assert running.source == "configmap", running
    calls = reader.calls
    time.sleep(0.05)
    assert reader.calls == calls, "the poller must stop with the lifespan"
