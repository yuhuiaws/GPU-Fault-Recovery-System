from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import SyntheticNodeReplacementRequest
from gpu_fault.models import WorkflowOperation
from tests._builders import asgi_client, attempt_observation, container_observation
from tests.regional._regional_support import TOKEN_A, registration
from tests.store.test_activation_inhibition_claims import command

PATH = "/v1/admin/test/node-replacement"
EXECUTION = "test-execution-credential-" + "x" * 32
HEADERS = {"X-GPU-Fault-Execution-Token": EXECUTION}


def payload(**changes: Any) -> dict[str, Any]:
    return {
        "event_id": "inhibited-replacement",
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_profile_version": "simulated-v1",
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "affected_workload_ids": ["training/pytorchjob/job-a"],
        "gpu_uuids": ["GPU-a"],
        "reason": "hermetic inhibition test",
        **changes,
    }


def replacements(context, workflow_id):
    return [
        step
        for step in context.store.get_workflow(workflow_id).official_steps
        if step.operation is WorkflowOperation.REPLACE_NODE
    ]


def seed_attempt(context: ApplicationContext) -> None:
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "attempt-a",
            datetime.now(timezone.utc),
            containers=[
                container_observation(
                    "pod-a", "worker-a", 0, "node-a", gpu_uuids=["GPU-a"]
                ),
                container_observation(
                    "pod-b", "worker-b", 1, "node-b", gpu_uuids=["GPU-b"]
                ),
            ],
            workload_ids=["training/pytorchjob/job-a"],
        )
    )


@pytest.mark.parametrize("value", [False, None, 0, 1, "true", "false", {}, []])
def test_request_rejects_every_present_nonliteral_true(value: Any) -> None:
    with pytest.raises(ValidationError):
        SyntheticNodeReplacementRequest.model_validate(
            payload(activation_forbidden=value)
        )


def test_request_inhibition_is_opt_in_frozen_and_omits_the_absent_default() -> None:
    ordinary = SyntheticNodeReplacementRequest.model_validate(payload())
    assert "activation_forbidden" not in ordinary.model_dump(mode="json")
    inhibited = SyntheticNodeReplacementRequest.model_validate(
        payload(activation_forbidden=True)
    )
    assert inhibited.model_dump(mode="json")["activation_forbidden"] is True
    with pytest.raises(ValidationError, match="frozen"):
        inhibited.activation_forbidden = None


@pytest.mark.parametrize("marker", [False, None, 1, "true"])
def test_route_rejects_malformed_inhibition_before_persisting_a_finding(
    monkeypatch: pytest.MonkeyPatch, marker: Any
) -> None:
    monkeypatch.setenv("GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS", "true")
    context = ApplicationContext(execution_token=EXECUTION)

    async def run() -> None:
        async with asgi_client(context) as client:
            result = await client.post(
                PATH, json=payload(activation_forbidden=marker), headers=HEADERS
            )
            assert result.status_code == 422
            assert context.store.get_incident_by_event("inhibited-replacement") is None

    asyncio.run(run())


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("inhibited", [False, True])
def test_real_route_and_family_builder_preserve_inhibition(
    monkeypatch: pytest.MonkeyPatch, grouped: bool, inhibited: bool
) -> None:
    monkeypatch.setenv("GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS", "true")
    context = ApplicationContext(execution_token=EXECUTION)
    if grouped:
        seed_attempt(context)
    body = payload(**({"activation_forbidden": True} if inhibited else {}))

    async def run() -> None:
        async with asgi_client(context) as client:
            response = await client.post(PATH, json=body, headers=HEADERS)
            assert response.status_code == 200, response.text
            steps = replacements(context, response.json()["workflow_request_ids"][0])
            assert steps, "synthetic replacement must produce REPLACE_NODE steps"
            assert all(
                step.parameters
                == {
                    "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY",
                    **({"activation_forbidden": True} if inhibited else {}),
                }
                for step in steps
            ), "replacement parameters must retain strategy and inhibition"
            if inhibited:
                incident = context.store.get_incident(
                    response.json()["incident_ids"][0]
                )
                assert incident.policy_source == "SITE_SYNTHETIC_REPLACEMENT_TEST"

    asyncio.run(run())


@pytest.mark.parametrize("first_inhibited", [False, True])
def test_reposting_event_cannot_change_activation_authority(
    monkeypatch: pytest.MonkeyPatch, first_inhibited: bool
) -> None:
    monkeypatch.setenv("GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS", "true")
    context = ApplicationContext(execution_token=EXECUTION)
    ordinary = payload()
    guarded = {**ordinary, "activation_forbidden": True}

    async def run() -> None:
        async with asgi_client(context) as client:
            first = await client.post(
                PATH, json=guarded if first_inhibited else ordinary, headers=HEADERS
            )
            assert first.status_code == 200
            workflow_id = first.json()["workflow_request_ids"][0]
            second = await client.post(
                PATH, json=ordinary if first_inhibited else guarded, headers=HEADERS
            )
            assert second.status_code == 409
            assert all(
                ("activation_forbidden" in step.parameters) is first_inhibited
                for step in replacements(context, workflow_id)
            ), "event replay must preserve the original activation authority"

    asyncio.run(run())


def test_mixed_authority_findings_do_not_recompile_one_anothers_pending_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS", "true")
    context = ApplicationContext(execution_token=EXECUTION)
    seed_attempt(context)

    async def run() -> None:
        async with asgi_client(context) as client:
            first = await client.post(
                PATH, json=payload(activation_forbidden=True), headers=HEADERS
            )
            assert first.status_code == 200
            initial_id = first.json()["workflow_request_ids"][0]
            second = await client.post(
                PATH,
                json=payload(
                    event_id="ordinary-second", node_id="node-b", gpu_uuids=["GPU-b"]
                ),
                headers=HEADERS,
            )
            assert second.status_code == 200, second.text
            assert second.json()["workflow_request_ids"][0] != initial_id
            assert all(
                step.parameters["activation_forbidden"] is True
                for step in replacements(context, initial_id)
            ), "ordinary findings must not relax an inhibited workflow"

    asyncio.run(run())


def test_inhibition_never_grants_cluster_token_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS", "true")
    context = ApplicationContext(execution_token=EXECUTION)
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))

    async def run() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                PATH,
                json=payload(activation_forbidden=True),
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )
            assert response.status_code == 403
            assert context.store.get_incident_by_event("inhibited-replacement") is None

    asyncio.run(run())


@pytest.mark.parametrize("batched", [False, True])
def test_claim_route_filters_old_but_globally_compatible_executor(
    monkeypatch: pytest.MonkeyPatch, batched: bool
) -> None:
    monkeypatch.setenv("GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION", "4")
    monkeypatch.setenv(
        "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS", "1,2,3"
    )
    context = ApplicationContext()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.ensure_remote_command(command("a-inhibited", True, batched=batched))
    context.store.ensure_remote_command(command("b-ordinary"))
    headers = {
        "Authorization": f"Bearer {TOKEN_A}",
        "X-GPU-Fault-Cluster-ID": "cluster-a",
    }

    async def run() -> None:
        async with asgi_client(context) as client:
            old = await client.post(
                "/v1/regional/executors/claim",
                headers=headers,
                json={
                    "executor_id": "old",
                    "executor_protocol_version": 3,
                    "max_commands": 5,
                },
            )
            assert old.status_code == 200, old.text
            assert [item["command_id"] for item in old.json()["commands"]] == [
                "b-ordinary"
            ]
            current = await client.post(
                "/v1/regional/executors/claim",
                headers=headers,
                json={
                    "executor_id": "current",
                    "executor_protocol_version": 4,
                    "max_commands": 5,
                },
            )
            assert current.status_code == 200, current.text
            assert [item["command_id"] for item in current.json()["commands"]] == [
                "a-inhibited"
            ]

    asyncio.run(run())
