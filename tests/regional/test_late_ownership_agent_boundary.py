from __future__ import annotations

import io
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from gpu_fault.adapters import HyperPodLifecycleStepAdapter, NodeActionWorkflowAdapter
from gpu_fault.adapters.kubernetes.stop_ownership import stop_ownership_scope
from gpu_fault.adapters.node_action import transport
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.node_agent import create_node_agent_app
from gpu_fault.node_agent.late_ownership import current_ownership_challenge
from tests.node_agent._support import (
    SECRET,
    FakeRunner,
    no_device_clients,
    node_action_executor,
    result_params,
)
from tests.regional._late_ownership_runtime import stopped_runtime


class LocalRegistry:
    policy = None

    def __init__(self):
        self.store = self

    def readiness(self, cluster_id, nodes):
        return SimpleNamespace(
            ready=True,
            nodes=[SimpleNamespace(node_id=node, generation=1) for node in nodes],
        )

    def endpoint(self, cluster_id, node_id):
        return "http://node-local", 1

    def get_agent(self, cluster_id, node_id):
        return SimpleNamespace(node_action_key_version=1)

    def now(self):
        return datetime.now(timezone.utc)


def transport_bridge(client):
    def send(request, **kwargs):
        url = urlsplit(request.full_url)
        response = client.request(
            request.get_method(),
            url.path + ("?" + url.query if url.query else ""),
            content=request.data,
            headers=dict(request.header_items()),
        )
        if response.status_code >= 400:
            assert HTTPError is transport.urllib_error.HTTPError, (
                "the test and transport must share the standard HTTP error class"
            )
            raise HTTPError(
                request.full_url,
                response.status_code,
                "local test response",
                {},
                io.BytesIO(response.content),
            )
        return io.BytesIO(response.content)

    return send


@pytest.mark.parametrize("drift", ["none", "owner", "late-sibling"])
def test_product_checks_again_after_agent_queue_before_physical_reset(
    tmp_path, monkeypatch, drift
):
    state, _kube, validator, context = stopped_runtime()
    step = context.step.model_copy(update={"gpu_uuids": ["GPU-a"]})
    context = replace(context, step=step)
    hardware = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "final-boundary.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        now=None,
        runner=hardware,
        require_final_ownership=True,
        agent_generation=1,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        device_client_samples=1,
    )
    observed = []

    def at_final_boundary(current, stop):
        challenge = current_ownership_challenge()
        if challenge is None:
            return
        assert challenge.boundary == "AGENT_PRE_SPAWN"
        assert not any("--gpu-reset" in call for call in hardware.commands), (
            "the native challenge must precede every physical reset"
        )
        assert agent.ownership_gate.challenge(challenge.command_id) == challenge
        observed.append(challenge)
        if drift == "owner":
            state.job["metadata"]["uid"] = "replaced-after-queue"
        elif drift == "late-sibling":
            state.pods.append(state.pod("node-b", "late-after-queue"))

    validator.before_recheck = at_final_boundary
    adapter = NodeActionWorkflowAdapter({}, SECRET, registry=LocalRegistry())
    with TestClient(create_node_agent_app(agent)) as client:
        monkeypatch.setattr(transport, "urlopen", transport_bridge(client))
        with stop_ownership_scope(validator):
            for _ in range(100):
                outcome = adapter.execute(context)
                if outcome.status is not WorkflowStepStatus.WAITING:
                    break
            else:
                pytest.fail(
                    "the owned Agent did not finish its one-shot boundary: "
                    f"state={outcome.details}; probes={len(hardware.commands)}; "
                    f"observed={len(observed)}"
                )
        assert len(observed) == 1
        command_id = observed[0].command_id
        for _ in range(100):
            response = client.get(
                "/v1/node-actions/result", params=result_params(command_id)
            )
            assert response.status_code == 200
            result = response.json()
            if result["state"] != "PENDING":
                break
        else:
            pytest.fail(
                "the denied/authorized Agent callback did not drain: "
                f"outcome={outcome.status}; details={outcome.details}; "
                f"error={outcome.error}; probes={len(hardware.commands)}"
            )
        if drift == "none":
            assert outcome.status is WorkflowStepStatus.SUCCEEDED
            assert result["state"] == "SUCCEEDED"
            assert len(result["result"]["details"]["physical_ownership_checks"]) == 1
            assert [call for call in hardware.commands if "--gpu-reset" in call] == [
                ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"]
            ]
        else:
            assert outcome.status is WorkflowStepStatus.FAILED
            assert outcome.details["agent_queue_ownership_checked"] is True
            assert outcome.details["manual_confirmation_required"] is True
            assert result["state"] == "FAILED"
            assert result["result"]["retryable"] is False
            assert result["result"]["details"]["manual_confirmation_required"] is True
            assert not any("--gpu-reset" in call for call in hardware.commands), (
                "post-queue ownership refusal must prevent all physical resets"
            )


class NoProvider:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        self.calls.append(name)
        raise AssertionError("an ownership refusal reached a provider")


@pytest.mark.parametrize(
    "operation", [WorkflowOperation.RESTART_NODE, WorkflowOperation.REPLACE_NODE]
)
@pytest.mark.parametrize("drift", ["owner", "late-sibling"])
def test_direct_provider_and_spare_adapters_cannot_bypass_ownership_refusal(
    operation, drift
):
    state, kube, validator, context = stopped_runtime()
    if drift == "owner":
        state.job["metadata"]["uid"] = "replacement"
    else:
        state.pods.append(state.pod("node-b", "late"))
    provider = NoProvider()
    spares = NoProvider()
    adapter = HyperPodLifecycleStepAdapter(
        provider, kubernetes_adapter=kube, spare_coordinator=spares
    )
    context = replace(
        context,
        step=context.step.model_copy(
            update={
                "operation": operation,
                "execution_owner": adapter.owner,
                "parameters": {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
            }
        ),
        request=context.request.model_copy(
            update={"confirm_cluster_name": "cluster-local"}
        ),
    )
    with stop_ownership_scope(validator):
        result = adapter.execute(context)
    assert result.status is WorkflowStepStatus.FAILED
    assert result.details["safety_rejection"] is True
    assert result.details["manual_confirmation_required"] is True
    assert provider.calls == []
    assert spares.calls == []
