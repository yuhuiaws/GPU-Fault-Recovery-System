"""``executor_from_environment`` picks the provider from ``GPU_FAULT_CLUSTER_PROVIDER``.

On ``eks-managed`` the SageMaker-backed ``HyperPodLifecycleAdapter`` is never
constructed; the step adapter is the same class with owner
``gpu-fault-eks-adapter`` over an ``EksManagedNodeProvider``, the warm-spare
coordinator is built for the provider (a Kubernetes-level handover), and the
node-key sync selects by ``GPU_FAULT_NODE_SELECTOR``.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.cluster_executor.bootstrap import (
    _node_key_sync_from_environment,
    executor_from_environment,
)
from gpu_fault.cluster_executor.regional_client import ClusterExecutorError
from gpu_fault.eks_node_evidence import EksNodeEvidence
from gpu_fault.eks_provider import EksManagedNodeProvider
from gpu_fault.hyperpod_spares import HyperPodSpareCoordinator
from gpu_fault.node_provider import ClusterProvider
from tests.eks_provider._support import (
    CLUSTER_ID,
    SELECTOR,
    FakeCore,
    FakeEc2,
    FakeEks,
    default_instances,
    default_nodes,
)


def _environment(monkeypatch: pytest.MonkeyPatch, *, provider: str | None) -> None:
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control-plane.example")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", "token")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", CLUSTER_ID)
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", "true")
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER", CLUSTER_ID)
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CONFIRM_CLUSTER", CLUSTER_ID)
    monkeypatch.setenv("GPU_FAULT_NODE_SELECTOR", SELECTOR)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.delenv("GPU_FAULT_HYPERPOD_OWNER", raising=False)
    monkeypatch.delenv("GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER", raising=False)
    if provider is None:
        monkeypatch.delenv("GPU_FAULT_CLUSTER_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("GPU_FAULT_CLUSTER_PROVIDER", provider)
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.bootstrap.missing_aws_credentials", lambda: None
    )
    monkeypatch.setattr(
        EksManagedNodeProvider,
        "_create_client",
        staticmethod(
            lambda service, region_name: (
                FakeEc2(default_instances()) if service == "ec2" else FakeEks()
            )
        ),
    )


def _fake_kubernetes_adapter(monkeypatch: pytest.MonkeyPatch) -> FakeCore:
    core = FakeCore(default_nodes())

    class FakeKubernetesAdapter(KubernetesWorkflowAdapter):
        owner = "gpu-fault-kubernetes-adapter"

        def __init__(self, **_kwargs: Any) -> None:
            self.core = core

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.bootstrap.KubernetesWorkflowAdapter",
        FakeKubernetesAdapter,
    )
    return core


def _capture_step_adapter(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class FakeStepAdapter:
        def __init__(self, lifecycle: Any, **kwargs: Any) -> None:
            self.owner = kwargs["owner"]
            captured["lifecycle"] = lifecycle
            captured.update(kwargs)

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.bootstrap.HyperPodLifecycleStepAdapter",
        FakeStepAdapter,
    )
    return captured


def test_eks_managed_executor_never_builds_the_sagemaker_adapter(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _environment(monkeypatch, provider="eks-managed")
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER", "true")
    core = _fake_kubernetes_adapter(monkeypatch)
    captured = _capture_step_adapter(monkeypatch)
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.bootstrap.HyperPodLifecycleAdapter",
        lambda *_args, **_kwargs: pytest.fail(
            "the SageMaker HyperPod adapter must not be constructed on eks-managed"
        ),
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.cluster_executor"):
        executor = executor_from_environment()

    lifecycle = captured["lifecycle"]
    assert isinstance(lifecycle, EksManagedNodeProvider), type(lifecycle)
    assert lifecycle.core is core, "nodes come from the executor's Kubernetes client"
    assert lifecycle.config.cluster_id == CLUSTER_ID
    assert lifecycle.config.node_selector == SELECTOR
    assert lifecycle.store is not None, (
        "submission idempotency is held by the control plane"
    )
    assert captured["owner"] == "gpu-fault-eks-adapter"
    coordinator = captured["spare_coordinator"]
    assert isinstance(coordinator, HyperPodSpareCoordinator), (
        "warm-spare failover is a Kubernetes-level handover on plain EKS too"
    )
    assert coordinator.provider is ClusterProvider.EKS_MANAGED
    assert coordinator.lifecycle is lifecycle, "the coordinator reads the EKS provider"
    assert coordinator.core is core, "spares are read through the executor's client"
    assert "warm-spare failover is not available" not in caplog.text
    assert executor.confirm_cluster_name == CLUSTER_ID
    assert executor.node_key_sync is not None
    assert executor.node_key_sync.node_selector == SELECTOR
    assert [adapter.owner for adapter in executor.adapters] == [
        "gpu-fault-kubernetes-adapter",
        "gpu-fault-eks-adapter",
    ]


def test_eks_managed_spare_failover_requires_remote_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same start-up guard as HyperPod's, worded for the provider."""

    _environment(monkeypatch, provider="eks-managed")
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER", "true")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE", "false")
    _fake_kubernetes_adapter(monkeypatch)
    _capture_step_adapter(monkeypatch)

    with pytest.raises(ClusterExecutorError, match="regional EKS spare failover"):
        executor_from_environment()


def test_eks_managed_owner_can_still_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch, provider="eks-managed")
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_OWNER", "site-eks-owner")
    _fake_kubernetes_adapter(monkeypatch)
    captured = _capture_step_adapter(monkeypatch)

    executor_from_environment()

    assert captured["owner"] == "site-eks-owner"


def test_absent_provider_keeps_the_hyperpod_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch, provider=None)
    monkeypatch.delenv("GPU_FAULT_NODE_SELECTOR", raising=False)
    _fake_kubernetes_adapter(monkeypatch)
    captured = _capture_step_adapter(monkeypatch)
    built: list[Any] = []

    class FakeLifecycle:
        def __init__(self, config: Any, *, store: Any = None) -> None:
            built.append(config)
            self.store = store

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.bootstrap.HyperPodLifecycleAdapter", FakeLifecycle
    )

    executor = executor_from_environment()

    assert isinstance(captured["lifecycle"], FakeLifecycle), captured["lifecycle"]
    assert built[0].cluster_name == CLUSTER_ID
    assert captured["owner"] == "gpu-fault-hyperpod-adapter"
    assert executor.node_key_sync is not None
    assert executor.node_key_sync.node_selector == (
        f"sagemaker.amazonaws.com/cluster-name={CLUSTER_ID}"
    )


def test_unknown_provider_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    _environment(monkeypatch, provider="eks")
    _fake_kubernetes_adapter(monkeypatch)
    _capture_step_adapter(monkeypatch)

    with pytest.raises(ValueError, match="GPU_FAULT_CLUSTER_PROVIDER must be one of"):
        executor_from_environment()


def test_eks_managed_requires_the_node_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch, provider="eks-managed")
    monkeypatch.delenv("GPU_FAULT_NODE_SELECTOR", raising=False)
    _fake_kubernetes_adapter(monkeypatch)
    _capture_step_adapter(monkeypatch)

    with pytest.raises(ValueError, match="GPU_FAULT_NODE_SELECTOR is required"):
        executor_from_environment()


def test_eks_managed_still_requires_the_confirmation_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch, provider="eks-managed")
    monkeypatch.delenv("GPU_FAULT_HYPERPOD_CONFIRM_CLUSTER", raising=False)
    _fake_kubernetes_adapter(monkeypatch)

    with pytest.raises(ClusterExecutorError, match="required when the EKS adapter"):
        executor_from_environment()


def test_node_key_sync_is_disabled_without_any_scope(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("GPU_FAULT_CLUSTER_PROVIDER", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_SELECTOR", raising=False)
    monkeypatch.delenv("GPU_FAULT_HYPERPOD_CLUSTER", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_KEY_SYNC_SECONDS", raising=False)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.cluster_executor"):
        sync = _node_key_sync_from_environment(object(), object())  # type: ignore[arg-type]

    assert sync is None
    assert "GPU_FAULT_NODE_SELECTOR and GPU_FAULT_HYPERPOD_CLUSTER are unset" in (
        caplog.text
    )


def test_an_eks_managed_sync_gets_the_ec2_attester_of_its_replacement_nodes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    owned = "gpu-fault.io/control-plane=us-west-2-hp-cluster-0a1b2c3d"
    monkeypatch.setenv("GPU_FAULT_CLUSTER_PROVIDER", "eks-managed")
    monkeypatch.setenv("GPU_FAULT_NODE_SELECTOR", owned)
    monkeypatch.setenv("GPU_FAULT_EKS_CLUSTER_NAME", "gpu-eks-cluster")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.delenv("GPU_FAULT_NODE_KEY_SYNC_SECONDS", raising=False)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.cluster_executor"):
        sync = _node_key_sync_from_environment(object(), object())  # type: ignore[arg-type]

    assert sync is not None
    assert isinstance(sync.attester, EksNodeEvidence), type(sync.attester)
    assert sync.attester.eks_cluster_name == "gpu-eks-cluster"
    assert sync.attester.region_name == "us-west-2"
    assert "no Node Action key" not in caplog.text


@pytest.mark.parametrize(
    ("selector", "eks_cluster", "reason"),
    [
        (
            "gpu-fault.io/control-plane=us-west-2-hp-cluster-0a1b2c3d",
            "",
            "GPU_FAULT_EKS_CLUSTER_NAME",
        ),
        ("nvidia.com/gpu.present=true", "gpu-eks-cluster", "node-scope --converge"),
    ],
)
def test_an_eks_managed_sync_that_cannot_attest_says_so_and_keeps_running(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    selector: str,
    eks_cluster: str,
    reason: str,
) -> None:
    monkeypatch.setenv("GPU_FAULT_CLUSTER_PROVIDER", "eks-managed")
    monkeypatch.setenv("GPU_FAULT_NODE_SELECTOR", selector)
    monkeypatch.setenv("GPU_FAULT_EKS_CLUSTER_NAME", eks_cluster)
    monkeypatch.delenv("GPU_FAULT_NODE_KEY_SYNC_SECONDS", raising=False)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.cluster_executor"):
        sync = _node_key_sync_from_environment(object(), object())  # type: ignore[arg-type]

    assert sync is not None, "nodes the control plane knows are still keyed"
    assert sync.attester is None
    assert "get no Node Action key until the next deploy" in caplog.text
    assert reason in caplog.text


def test_a_hyperpod_sync_has_no_attester(monkeypatch: pytest.MonkeyPatch) -> None:
    """SageMaker identities place HyperPod nodes on the control plane itself."""

    monkeypatch.delenv("GPU_FAULT_CLUSTER_PROVIDER", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_SELECTOR", raising=False)
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER", "hp-cluster-a")
    monkeypatch.delenv("GPU_FAULT_NODE_KEY_SYNC_SECONDS", raising=False)

    sync = _node_key_sync_from_environment(object(), object())  # type: ignore[arg-type]

    assert sync is not None
    assert sync.attester is None
