from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gpu_fault.capabilities import ProfileValidationError, compile_runtime_profile
from gpu_fault.models import (
    CapabilityClaim,
    CapabilityMode,
    CapabilityName,
    Environment,
    ObservedCapability,
    RuntimeProfile,
)

ROOT = Path(__file__).parents[2]
PROFILE_EXAMPLES = tuple(
    sorted((ROOT / "config").glob("runtime-profile.*.example.yaml"))
)


def test_environment_members_are_exactly_the_ones_decisions_branch_on() -> None:
    """An environment nobody branches on takes the default path silently.

    The recovery-ownership and RESTART_VM mappings ask which environment a profile
    declares; a member outside that set would still load a profile and still
    configure a watcher, and then behave like plain Kubernetes. So the member list
    is pinned here, and an unsupported value fails at parse time instead.
    """

    assert {member.value for member in Environment} == {
        "eks",
        "hyperpod-eks",
        "hyperpod-slurm",
        "kubernetes",
    }
    with pytest.raises(ValueError, match="slurm"):
        Environment("slurm")


def test_rejects_multiple_destructive_writers() -> None:
    profile = RuntimeProfile(
        cluster_id="cluster-a",
        environment=Environment.EKS,
        profile_version="bad-v1",
        claims=[
            CapabilityClaim(
                capability=CapabilityName.NODE_REPLACE,
                mode=CapabilityMode.OWN,
                owner="custom",
                adapter="ec2",
            ),
            CapabilityClaim(
                capability=CapabilityName.NODE_REPLACE,
                mode=CapabilityMode.DELEGATE,
                owner="eks-auto-repair",
                adapter="eks",
            ),
        ],
        observed=[],
    )

    with pytest.raises(ProfileValidationError, match="multiple active writers"):
        compile_runtime_profile(profile)


def test_rejects_augment_for_destructive_capability() -> None:
    profile = RuntimeProfile(
        cluster_id="cluster-a",
        environment=Environment.EKS,
        profile_version="bad-v2",
        claims=[
            CapabilityClaim(
                capability=CapabilityName.NODE_REBOOT,
                mode=CapabilityMode.AUGMENT,
                owner="two-writers",
                adapter="eks",
            )
        ],
        observed=[],
    )

    with pytest.raises(ProfileValidationError, match="cannot use AUGMENT"):
        compile_runtime_profile(profile)


def test_unavailable_owner_is_reduced_to_observe() -> None:
    profile = RuntimeProfile(
        cluster_id="cluster-a",
        environment=Environment.HYPERPOD_EKS,
        profile_version="hp-v1",
        claims=[
            CapabilityClaim(
                capability=CapabilityName.NODE_REPLACE,
                mode=CapabilityMode.DELEGATE,
                owner="hyperpod",
                adapter="hyperpod",
            )
        ],
        observed=[
            ObservedCapability(
                capability=CapabilityName.NODE_REPLACE,
                owner="hyperpod",
                available=False,
            )
        ],
    )

    effective = compile_runtime_profile(profile)

    assert effective.capabilities[0].mode is CapabilityMode.OBSERVE
    assert "unavailable" in effective.warnings[0]


def test_regional_hyperpod_safe_profile_matches_enabled_adapters() -> None:
    payload = yaml.safe_load(
        (
            ROOT / "config/runtime-profile.regional-hyperpod-safe.example.yaml"
        ).read_text()
    )
    assert payload["cluster_id"] == "REPLACE_WITH_CLUSTER_ID"
    profile = RuntimeProfile.model_validate(payload)

    effective = compile_runtime_profile(profile)

    assert effective.profile_version == "hyperpod-v1"
    assert effective.warnings == []
    executable = {
        item.capability: item.owner
        for item in effective.capabilities
        if item.mode in {CapabilityMode.OWN, CapabilityMode.DELEGATE}
    }
    assert executable[CapabilityName.WORKLOAD_RESTART] == (
        "gpu-fault-kubernetes-adapter"
    )
    # Regional production installs the signed Node Agent before registering
    # the Profile, so catalog actions that require one-GPU reset must have an
    # executable owner. The remaining unapproved node capabilities stay
    # observation-only.
    for capability in (
        CapabilityName.DIAGNOSTIC_BUNDLE_CAPTURE,
        CapabilityName.GPU_RESET,
        CapabilityName.FABRIC_MANAGER_RESTART,
        CapabilityName.FABRIC_RESET,
    ):
        assert executable[capability] == "gpu-fault-node-agent"
    assert CapabilityName.NVLINK_DIAGNOSTICS not in executable
    assert CapabilityName.MEMORY_DIAGNOSTICS not in executable
    assert CapabilityName.DRIVER_REMEDIATION not in executable
    assert CapabilityName.SOFTWARE_FIRMWARE_UPDATE not in executable
    # Node lifecycle is the one mutation class this profile does own: the
    # provider performs it, so it needs no agent on the failed node.
    # GPU_FAULT_ALLOW_HYPERPOD_REBOOT defaults to true in deploy.sh, and
    # the warm-spare failover path drives REPLACE_NODE through the same
    # adapter, so both must resolve to an executor.
    assert executable[CapabilityName.NODE_REBOOT] == ("gpu-fault-hyperpod-adapter")
    assert executable[CapabilityName.NODE_REPLACE] == ("gpu-fault-hyperpod-adapter")


@pytest.mark.parametrize("path", PROFILE_EXAMPLES, ids=lambda path: path.name)
def test_runtime_profile_examples_are_direct_registration_payloads(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    profile = RuntimeProfile.model_validate(payload)
    effective = compile_runtime_profile(profile)

    assert profile.cluster_id
    assert profile.profile_version
    assert effective.warnings == []


def test_config_has_no_multi_profile_payload() -> None:
    assert PROFILE_EXAMPLES
    assert not list((ROOT / "config").glob("runtime-profiles*.yaml"))
    assert [path.name for path in PROFILE_EXAMPLES] == [
        "runtime-profile.regional-hyperpod-safe.example.yaml"
    ]
