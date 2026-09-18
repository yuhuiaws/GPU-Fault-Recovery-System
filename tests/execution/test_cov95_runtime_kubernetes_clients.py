from __future__ import annotations

import builtins

import pytest
from kubernetes import client, config

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.adapters.kubernetes.primitives import build_kubernetes_clients
from tests.execution.test_restart_safety import UnusedApi


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ({"workload_log_tail_lines": 0}, "tail lines must be positive"),
        ({"workload_log_max_bytes": 1024}, "max bytes must be at least 4096"),
        ({"workload_log_s3_max_bytes": 4096}, "S3 max bytes must cover the tail"),
    ],
)
def test_invalid_log_bounds_are_rejected_with_fully_injected_clients(
    values, reason
) -> None:
    with pytest.raises(ValueError, match=reason):
        KubernetesWorkflowAdapter(
            core_api=UnusedApi(),
            batch_api=UnusedApi(),
            custom_api=UnusedApi(),
            **values,
        )


@pytest.mark.parametrize("explicit", [None, 3, (2, 4)])
def test_kubernetes_clients_share_default_timeout_without_overwriting_call_bounds(
    explicit, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def call_api(self, *args, **kwargs):
        calls.append((args, kwargs))
        return {"metadata": {"name": "node-a"}}

    monkeypatch.setattr(client.ApiClient, "call_api", call_api)
    core, batch, custom = build_kubernetes_clients(
        client.Configuration(), request_timeout_seconds=7
    )
    try:
        assert core.api_client is batch.api_client is custom.api_client, (
            "all Kubernetes APIs must share the bounded client"
        )
        result = core.read_node("node-a", _request_timeout=explicit)
        assert result == {"metadata": {"name": "node-a"}}, result
        assert len(calls) == 1, calls
        assert calls[0][1]["_request_timeout"] == (
            7 if explicit is None else explicit
        ), calls
        assert calls[0][0][1] == "GET", calls
    finally:
        core.api_client.close()


@pytest.mark.parametrize("fallback", [False, True])
def test_adapter_client_initialization_uses_only_the_selected_fake_config_loader(
    fallback: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaders = []
    configuration = client.Configuration()

    def incluster():
        loaders.append("incluster")
        if fallback:
            raise config.ConfigException("unit outside cluster")

    def kubeconfig():
        loaders.append("kubeconfig")

    monkeypatch.setattr(config, "load_incluster_config", incluster)
    monkeypatch.setattr(config, "load_kube_config", kubeconfig)
    monkeypatch.setattr(
        client.Configuration, "get_default_copy", classmethod(lambda cls: configuration)
    )
    injected = UnusedApi()
    adapter = KubernetesWorkflowAdapter(core_api=injected, request_timeout_seconds=9)
    try:
        assert loaders == (
            ["incluster", "kubeconfig"] if fallback else ["incluster"]
        ), loaders
        assert adapter.core is not injected, (
            "a partial client set must not mix configurations"
        )
        assert (
            adapter.core.api_client
            is adapter.batch.api_client
            is adapter.custom.api_client
        ), adapter
        assert adapter.core.api_client.configuration is configuration, adapter
        assert adapter.request_timeout_seconds == 9, adapter
    finally:
        adapter.core.api_client.close()


def test_missing_kubernetes_dependency_is_a_named_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "kubernetes":
            raise ImportError("unit optional dependency absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(
        RuntimeError, match=r"install gpu-fault-control-plane\[collectors\]"
    ):
        KubernetesWorkflowAdapter(request_timeout_seconds=9)
