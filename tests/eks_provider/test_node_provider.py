"""The provider contract the runtime reads: enum, env, selector, providerID, mapping.

``GPU_FAULT_CLUSTER_PROVIDER`` absent means HyperPod and every default stays
what it was; ``eks-managed`` switches the node selector requirement, the EFA
device plugin name, the managed-recovery owners, the completion watcher's
default environment and the ``RESTART_VM`` mapping text.
"""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from gpu_fault import completion_controller
from gpu_fault.cluster_executor.node_key_sync import NodeActionKeySync
from gpu_fault.models import Environment, RecoveryAction
from gpu_fault.node_agent.common import DEFAULT_QUIESCE_CONTAINERS
from gpu_fault.node_agent.quiesce import GpuServiceQuiesceManager
from gpu_fault.node_provider import (
    EKS_NODEGROUP_LABEL,
    EKS_REBOOT_CONFIRMATION_SOURCE,
    HYPERPOD_INSTANCE_GROUP_LABEL,
    HYPERPOD_REBOOT_CONFIRMATION_SOURCE,
    REBOOT_CONFIRMATION_SOURCES,
    ClusterProvider,
    cluster_provider,
    efa_device_plugin_name,
    instance_group_label,
    node_selector,
    parse_provider_id,
    restart_vm_effective_action,
)
from gpu_fault.settings import ControlPlaneSettings
from tests.fleet.test_settings import active_values
from tests.node_agent._cov95_installer_support import make_reconciler
from tests.node_agent._support import ServiceRunner


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ClusterProvider.HYPERPOD),
        ("", ClusterProvider.HYPERPOD),
        ("  ", ClusterProvider.HYPERPOD),
        ("hyperpod", ClusterProvider.HYPERPOD),
        ("EKS-Managed", ClusterProvider.EKS_MANAGED),
        ("eks-managed", ClusterProvider.EKS_MANAGED),
    ],
)
def test_cluster_provider_defaults_to_hyperpod(
    value: str | None, expected: ClusterProvider
) -> None:
    environ = {} if value is None else {"GPU_FAULT_CLUSTER_PROVIDER": value}

    assert cluster_provider(environ) is expected


def test_cluster_provider_refuses_unknown_values() -> None:
    with pytest.raises(ValueError, match="hyperpod, eks-managed; got 'eks'"):
        cluster_provider({"GPU_FAULT_CLUSTER_PROVIDER": "eks"})


def test_node_selector_defaults_to_the_hyperpod_cluster_label() -> None:
    assert node_selector({"GPU_FAULT_HYPERPOD_CLUSTER": "hp-a"}) == (
        "sagemaker.amazonaws.com/cluster-name=hp-a"
    )
    assert node_selector({}) is None
    assert (
        node_selector(
            {
                "GPU_FAULT_HYPERPOD_CLUSTER": "hp-a",
                "GPU_FAULT_NODE_SELECTOR": " gpu=true ",
            }
        )
        == "gpu=true"
    ), "an explicit selector wins on every provider"


def test_node_selector_is_required_for_eks_managed() -> None:
    environ = {
        "GPU_FAULT_CLUSTER_PROVIDER": "eks-managed",
        "GPU_FAULT_HYPERPOD_CLUSTER": "gpu-a",
    }

    with pytest.raises(ValueError, match="GPU_FAULT_NODE_SELECTOR is required"):
        node_selector(environ)
    assert (
        node_selector(
            {**environ, "GPU_FAULT_NODE_SELECTOR": "nvidia.com/gpu.present=true"}
        )
        == "nvidia.com/gpu.present=true"
    )


def test_instance_group_label_per_provider() -> None:
    assert (
        instance_group_label(ClusterProvider.HYPERPOD) == HYPERPOD_INSTANCE_GROUP_LABEL
    )
    assert instance_group_label(ClusterProvider.EKS_MANAGED) == EKS_NODEGROUP_LABEL
    assert EKS_NODEGROUP_LABEL == "eks.amazonaws.com/nodegroup"


def test_efa_device_plugin_name_follows_the_provider_unless_overridden() -> None:
    assert (
        efa_device_plugin_name({}) == "hyperpod-dependencies-aws-efa-k8s-device-plugin"
    )
    assert (
        efa_device_plugin_name({"GPU_FAULT_CLUSTER_PROVIDER": "eks-managed"})
        == "aws-efa-k8s-device-plugin"
    )
    assert (
        efa_device_plugin_name(
            {"GPU_FAULT_EFA_DEVICE_PLUGIN_NAME": "custom-efa-plugin"}
        )
        == "custom-efa-plugin"
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("aws:///us-west-2a/i-0a000001", ("us-west-2a", "i-0a000001")),
        ("aws:///us-west-2b/i-abcdef12", ("us-west-2b", "i-abcdef12")),
        ("aws:///us-west-2a/hp-cluster-i-0a000001", (None, "i-0a000001")),
        ("aws:///us-west-2a/cluster-i-spare", (None, "i-spare")),
        ("aws:///us-west-2a/opaque-provider", None),
        ("gce://project/zone/instance", None),
        ("", None),
        (None, None),
        (42, None),
    ],
)
def test_parse_provider_id(
    value: object, expected: tuple[str | None, str] | None
) -> None:
    assert parse_provider_id(value) == expected


def test_reboot_confirmation_sources_keep_the_hyperpod_string() -> None:
    assert (
        HYPERPOD_REBOOT_CONFIRMATION_SOURCE
        == "hyperpod-running-and-new-agent-incarnation"
    )
    assert EKS_REBOOT_CONFIRMATION_SOURCE == "eks-running-and-new-agent-incarnation"
    assert REBOOT_CONFIRMATION_SOURCES == {
        HYPERPOD_REBOOT_CONFIRMATION_SOURCE,
        EKS_REBOOT_CONFIRMATION_SOURCE,
    }


@pytest.mark.parametrize(
    "environment,action,fragment",
    [
        (
            Environment.HYPERPOD_EKS,
            RecoveryAction.REBOOT_NODE,
            "BatchRebootClusterNodes",
        ),
        (
            Environment.HYPERPOD_SLURM,
            RecoveryAction.REBOOT_NODE,
            "BatchRebootClusterNodes",
        ),
        (Environment.EKS, RecoveryAction.REBOOT_NODE, "EC2 RebootInstances"),
        (Environment.KUBERNETES, None, None),
    ],
)
def test_restart_vm_maps_to_reboot_node_through_each_provider(
    environment: Environment, action: RecoveryAction | None, fragment: str | None
) -> None:
    effective, reason = restart_vm_effective_action(environment)

    assert effective is action
    if fragment is None:
        assert reason is None
    else:
        assert reason is not None and fragment in reason
        assert "not EC2 stop/start" in reason


def test_control_plane_settings_route_the_eks_adapter_remotely() -> None:
    values = active_values()
    values["GPU_FAULT_DEPLOYMENT_MODE"] = "regional"
    values["GPU_FAULT_ENABLE_AGENT_REGISTRY"] = "true"
    values["GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256"] = "a" * 64
    values["GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST"] = "b" * 64
    values["GPU_FAULT_REGIONAL_CLUSTERS_JSON"] = json.dumps(
        [{"cluster_id": "gpu-a", "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"]}]
    )

    settings = ControlPlaneSettings.from_mapping(values)

    assert {"gpu-fault-hyperpod-adapter", "gpu-fault-eks-adapter"} <= (
        settings.remote_execution_owners
    )
    assert settings.managed_recovery_owners == {
        "hyperpod-managed-node-recovery",
        "hyperpod-managed-job-recovery",
    }, "the HyperPod default is unchanged when the provider is absent"


def test_control_plane_settings_drop_hyperpod_managed_owners_for_eks_managed() -> None:
    values = active_values()
    values["GPU_FAULT_CLUSTER_PROVIDER"] = "eks-managed"

    settings = ControlPlaneSettings.from_mapping(values)

    assert settings.managed_recovery_owners == frozenset()
    explicit = ControlPlaneSettings.from_mapping(
        {**values, "GPU_FAULT_MANAGED_RECOVERY_OWNERS": "custom-owner"}
    )
    assert explicit.managed_recovery_owners == {"custom-owner"}


def test_completion_watcher_default_environment_follows_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GPU_FAULT_CLUSTER_PROVIDER", raising=False)
    assert (
        completion_controller.default_watcher_environment() is Environment.HYPERPOD_EKS
    )
    monkeypatch.setenv("GPU_FAULT_CLUSTER_PROVIDER", "eks-managed")
    assert completion_controller.default_watcher_environment() is Environment.EKS


def test_node_key_sync_takes_a_selector_or_the_legacy_hyperpod_name() -> None:
    legacy = NodeActionKeySync(
        object(), object(), hyperpod_cluster="hp-a", namespace="gpu-fault-system"
    )
    assert legacy.node_selector == "sagemaker.amazonaws.com/cluster-name=hp-a"

    explicit = NodeActionKeySync(
        object(),
        object(),
        namespace="gpu-fault-system",
        node_selector="nvidia.com/gpu.present=true",
    )
    assert explicit.node_selector == "nvidia.com/gpu.present=true"
    assert explicit.hyperpod_cluster == ""
    with pytest.raises(ValueError, match="node_selector or hyperpod_cluster"):
        NodeActionKeySync(object(), object(), namespace="gpu-fault-system")


def test_node_key_sync_lists_nodes_by_the_selector() -> None:
    class Core:
        def __init__(self) -> None:
            self.selectors: list[str] = []

        def list_node(self, label_selector: str) -> dict[str, object]:
            self.selectors.append(label_selector)
            # A Node as the API returns it: the sync checks the UID and the
            # selector's labels the way the deploy host's helper does.
            node = {
                "name": "ip-10-0-33-205.ec2.internal",
                "uid": "uid-1",
                "labels": {"node-type": "gpu"},
            }
            return {"items": [{"metadata": node}]}

    core = Core()
    sync = NodeActionKeySync(
        object(), core, namespace="gpu-fault-system", node_selector="node-type=gpu"
    )

    assert sync.labelled_nodes() == ["ip-10-0-33-205.ec2.internal"]
    assert core.selectors == ["node-type=gpu"]


def test_node_installer_reconciler_selector_defaults_to_the_hyperpod_label() -> None:
    default = make_reconciler()
    assert default.node_selector == "sagemaker.amazonaws.com/cluster-name=hp-cluster-a"

    scoped = make_reconciler(node_selector="nvidia.com/gpu.present=true")
    assert scoped.node_selector == "nvidia.com/gpu.present=true"
    blank = make_reconciler(node_selector="  ")
    assert blank.node_selector == "sagemaker.amazonaws.com/cluster-name=hp-cluster-a"


def test_quiesce_skips_a_container_selector_that_matches_nothing(
    tmp_path: Path,
) -> None:
    """A plain EKS node has no HyperPod health-monitoring-agent to stop."""

    assert "aws-hyperpod/health-monitoring-agent" in DEFAULT_QUIESCE_CONTAINERS

    class Runner(ServiceRunner):
        def __init__(self) -> None:
            super().__init__(active={"kubelet"})
            self.task_state = "RUNNING"

        def __call__(self, value, *, check=False, **options):
            if value[:4] == ["ctr", "-n", "k8s.io", "containers"]:
                self.commands.append(value)
                selector = value[-1]
                stdout = (
                    "plugin-container\n"
                    if "nvidia-device-plugin-ctr" in selector
                    else ""
                )
                return CompletedProcess(value, 0, stdout=stdout, stderr="")
            if value[:5] == ["ctr", "-n", "k8s.io", "tasks", "list"]:
                self.commands.append(value)
                return CompletedProcess(
                    value,
                    0,
                    stdout=f"TASK PID STATUS\nplugin-container 12 {self.task_state}\n",
                    stderr="",
                )
            if value[:5] == ["ctr", "-n", "k8s.io", "tasks", "kill"]:
                self.commands.append(value)
                self.task_state = "STOPPED"
                return CompletedProcess(value, 0, stdout="", stderr="")
            return super().__call__(value, check=check, **options)

    runner = Runner()
    manager = GpuServiceQuiesceManager(
        state_dir=str(tmp_path / "quiesce"),
        services=("kubelet",),
        containers=DEFAULT_QUIESCE_CONTAINERS,
        settle_seconds=0,
        restore_settle_seconds=0,
        proc_root=str(tmp_path),
        runner=runner,
        sleeper=lambda seconds: None,
        boot_id_reader=lambda: "boot",
    )

    receipt = manager.quiesce(incident_id="incident-eks", workflow_request_id="wf-eks")

    assert receipt["stopped_containers"] == [
        {
            "selector": "kube-system/nvidia-device-plugin-ctr",
            "container_id": "plugin-container",
        }
    ], "the absent HMA selector contributes no target and raises nothing"
    queried = [
        command[-1]
        for command in runner.commands
        if command[:4] == ["ctr", "-n", "k8s.io", "containers"]
    ]
    assert any("aws-hyperpod" in selector for selector in queried), (
        "the HMA selector is still asked for, so a HyperPod node keeps stopping it"
    )
    killed = [
        command[-1] for command in runner.commands if command[3:5] == ["tasks", "kill"]
    ]
    assert killed == ["plugin-container"], "only the container that exists is signalled"
