"""One control plane, two Runtime Profiles: owners resolve per profile version.

The coordinator loads ``store.get_profile(event.runtime_profile_version)`` for
every event and ``WorkflowBuilder.owner`` reads the OWN/DELEGATE owner from
that profile, so an eks-managed cluster's events (whose Agents carry
``eks-v1``) are owned by the EKS adapter while a HyperPod cluster's (on
``hyperpod-v1``) stay with the HyperPod adapter, in the same store.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.cluster_scope import PROVIDER_NODE_REBOOT_OWNERS
from gpu_fault.models import CapabilityName, RuntimeProfile
from gpu_fault.orchestration.workflow_builder import WorkflowBuilder
from gpu_fault.store import InMemoryStore

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "hyperpod-v1": ROOT / "config/runtime-profile.regional-hyperpod-safe.example.yaml",
    "eks-v1": ROOT / "config/runtime-profile.regional-eks-safe.example.yaml",
}


def _registered_store() -> InMemoryStore:
    store = InMemoryStore()
    for version, path in PROFILES.items():
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["cluster_id"] = "gpu-a" if version == "hyperpod-v1" else "gpu-eks"
        document["profile_version"] = version
        store.save_profile(
            compile_runtime_profile(RuntimeProfile.model_validate(document))
        )
    return store


def test_node_reboot_owner_follows_the_event_profile_version() -> None:
    store = _registered_store()
    owner = WorkflowBuilder.owner

    hyperpod_profile = store.get_profile("hyperpod-v1")
    eks_profile = store.get_profile("eks-v1")
    assert (
        owner(None, hyperpod_profile, CapabilityName.NODE_REBOOT)
        == (PROVIDER_NODE_REBOOT_OWNERS["hyperpod"])
    ), "a HyperPod cluster's events (hyperpod-v1) are rebooted by the HyperPod adapter"
    assert (
        owner(None, eks_profile, CapabilityName.NODE_REBOOT)
        == (PROVIDER_NODE_REBOOT_OWNERS["eks-managed"])
    ), "a plain EKS cluster's events (eks-v1) are rebooted by the EKS adapter"
    assert owner(None, eks_profile, CapabilityName.NODE_REPLACE) is None, (
        "nodeReplace is OBSERVE on eks-v1: no executable owner, so it fails closed"
    )
    assert (
        owner(None, hyperpod_profile, CapabilityName.NODE_REPLACE)
        == (PROVIDER_NODE_REBOOT_OWNERS["hyperpod"])
    ), "the HyperPod profile keeps its warm-spare replace owner"
    assert hyperpod_profile.environment.value == "hyperpod-eks", "HyperPod environment"
    assert eks_profile.environment.value == "eks", "plain EKS environment"
