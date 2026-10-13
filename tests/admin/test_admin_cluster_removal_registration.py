"""remove-cluster of the site's Runtime Profile registration cluster.

``spec.runtimeProfile.registrationClusterId`` names the cluster whose provider
the site base profile family follows (``bootstrap_discovery.regional_profile_for``)
and the anchor a new profile version registers under (the control plane
accepts only a registered cluster). remove-cluster kept it when it removed that
very cluster while others remained, and every later deploy refused: "registration
cluster gpu-a is not among the managed GPU clusters; the site Runtime Profile
base cannot be re-derived". The anchor now moves, with the cluster's removal, to
a remaining member of the same provider; without one the removal is refused
before anything changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.bootstrap_discovery import (
    HYPERPOD_PROFILE,
    regional_profile_for,
    registration_cluster_provider,
)
from gpu_fault.admin.cluster_removal import (
    RemoveClusterRequest,
    remove_cluster,
    resolve_removal_cluster_id,
)
from gpu_fault.admin.cluster_removal_state import (
    REGISTRATION_REANCHOR,
    canonical_digest,
    registration_reanchor,
    site_documents,
)
from gpu_fault.admin.site import load_site
from tests.admin._cluster_removal_support import EKS_ARN, HP_ARN, RemovalScenario

ACCOUNT = "123456789012"
REGION = "us-east-1"


def _arn(cluster_id: str) -> str:
    return f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{cluster_id}"


def _add_members(
    scenario: RemovalScenario, *members: tuple[str, str], own_profile: str = ""
) -> None:
    """Add GPU clusters beside ``gpu-a`` (the registration cluster).

    ``members`` are ``(cluster_id, provider)``; each gets its own token file,
    which remove-cluster requires of a preserved cluster. ``own_profile`` names
    a member that is pinned to its own Runtime Profile version.
    """

    document = yaml.safe_load(scenario.path.read_text(encoding="utf-8"))
    first = document["spec"]["clusters"][0]
    secure = Path(first["tokenFile"]).parent
    for cluster_id, provider in members:
        token = secure / f"{cluster_id}.token"
        token.write_text("u" * 64, encoding="utf-8")
        token.chmod(0o600)
        entry = {
            key: value for key, value in first.items() if key != "hyperpodClusterName"
        }
        entry.update(
            {
                "clusterId": cluster_id,
                "context": cluster_id,
                "eksClusterArn": _arn(cluster_id),
                "tokenFile": str(token),
            }
        )
        if provider == "eks-managed":
            entry["provider"] = "eks-managed"
            entry["nodeSelector"] = "gpu-fault.io/control-plane=test-site"
        else:
            entry["hyperpodClusterName"] = f"hp-{cluster_id}"
        if cluster_id == own_profile:
            entry["runtimeProfile"] = {
                "source": "config/profile.yaml",
                "version": "hyperpod-v2",
            }
        document["spec"]["clusters"].append(entry)
    scenario.path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def _registry_with_members(
    scenario: RemovalScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durable registry the fake reports: ``gpu-a`` at the scenario's
    lifecycle, every other member of the site ACTIVE."""

    def snapshot(_site: Any) -> dict[str, Any]:
        members = [
            str(item["clusterId"])
            for item in yaml.safe_load(scenario.path.read_text(encoding="utf-8"))[
                "spec"
            ]["clusters"]
            if item["clusterId"] != "gpu-a"
        ]
        states = dict.fromkeys(members, "ACTIVE")
        if scenario.lifecycle is not None:
            states["gpu-a"] = scenario.lifecycle
        return {
            "registry_generation": 1,
            "registry_content_sha256": "a" * 64,
            "live_release_identity_sha256": scenario.release_identity,
            "registry_cluster_states": states,
        }

    monkeypatch.setattr(removal, "membership_runtime_snapshot", snapshot)


@pytest.fixture
def scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RemovalScenario:
    value = RemovalScenario(tmp_path, monkeypatch)
    _registry_with_members(value, monkeypatch)
    return value


def _site_document(scenario: RemovalScenario) -> dict[str, Any]:
    return yaml.safe_load(scenario.path.read_text(encoding="utf-8"))


def _registration(scenario: RemovalScenario) -> str:
    return str(
        _site_document(scenario)["spec"]["runtimeProfile"]["registrationClusterId"]
    )


def _identity(entry: dict[str, Any]) -> ClusterIdentity:
    """The identity the next deploy discovers for a remaining site entry."""

    plain = entry.get("provider") == "eks-managed"
    name = "" if plain else str(entry["hyperpodClusterName"])
    return ClusterIdentity(
        input_arn=str(entry["eksClusterArn"]),
        role="gpu",
        region=REGION,
        account_id=ACCOUNT,
        hyperpod_arn=""
        if plain
        else f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/{name}",
        hyperpod_name=name,
        eks_arn=str(entry["eksClusterArn"]),
        eks_name=str(entry["clusterId"]),
        vpc_id="vpc-gpu",
        subnet_ids=(),
        node_recovery="" if plain else "None",
        context=str(entry["context"]),
        provider="eks-managed" if plain else "hyperpod",
    )


def test_removing_the_registration_cluster_moves_the_anchor_to_a_member_of_its_provider(
    scenario: RemovalScenario,
) -> None:
    # The plain EKS member sorts first: only the provider may decide.
    _add_members(scenario, ("eks-first", "eks-managed"), ("gpu-b", "hyperpod"))

    result = remove_cluster(scenario.request())

    assert result["phase"] == "COMPLETED", result
    assert result[REGISTRATION_REANCHOR] == "gpu-b", (
        "the removal result names the cluster the site anchor moved to"
    )
    assert _registration(scenario) == "gpu-b", (
        "the site anchor must move to the remaining HyperPod cluster, never to the "
        "plain EKS one: the site base profile follows the anchor's provider"
    )
    assert scenario.state()[REGISTRATION_REANCHOR] == "gpu-b", (
        "the decision is journaled with the attempt"
    )
    remaining = _site_document(scenario)["spec"]["clusters"]
    assert [item["clusterId"] for item in remaining] == ["eks-first", "gpu-b"], (
        "only the removed cluster leaves the site"
    )
    identities = [_identity(item) for item in remaining]
    assert registration_cluster_provider(scenario.path.parent, identities) == (
        "hyperpod"
    ), "the next deploy reads the base family from the moved anchor"
    source, version = regional_profile_for(
        Path("/repo"), identities, state_dir=scenario.path.parent
    )
    assert (source, version) == (f"/repo/{HYPERPOD_PROFILE[0]}", "hyperpod-v1"), (
        "the next deploy keeps the HyperPod base family; it used to refuse here"
    )


def test_a_member_on_the_site_profile_is_preferred_to_one_pinned_to_its_own(
    scenario: RemovalScenario,
) -> None:
    _add_members(
        scenario, ("gpu-b", "hyperpod"), ("gpu-c", "hyperpod"), own_profile="gpu-b"
    )

    assert registration_reanchor(load_site(scenario.path), "gpu-a") == "gpu-c", (
        "the anchor names a member that runs the site profile when there is one"
    )


def test_removing_the_registration_cluster_without_a_member_of_its_provider_is_refused(
    scenario: RemovalScenario,
) -> None:
    _add_members(scenario, ("gpu-e", "eks-managed"))
    before = scenario.path.read_bytes()

    with pytest.raises(BootstrapError) as raised:
        remove_cluster(scenario.request())

    message = str(raised.value)
    assert "registration cluster" in message and "(hyperpod)" in message, message
    assert "remaining: gpu-e" in message, "the refusal names what remains"
    assert "Join another hyperpod cluster first, then remove gpu-a" in message, (
        "the refusal names the remedy: " + message
    )
    assert message.endswith("Nothing was changed"), message
    assert scenario.calls == [], "the refusal precedes every removal step"
    assert scenario.path.read_bytes() == before, "site.yaml is untouched"
    assert not (scenario.path.parent / "remove-cluster/gpu-a").exists(), (
        "no removal journal is written for a refused removal"
    )


@pytest.mark.parametrize(
    "members,cluster_id",
    [
        pytest.param((("gpu-b", "hyperpod"),), "gpu-b", id="another-member"),
        pytest.param((("gpu-e", "eks-managed"),), "gpu-e", id="other-provider"),
        pytest.param((), "gpu-a", id="last-cluster"),
    ],
)
def test_other_removals_keep_the_anchor(
    scenario: RemovalScenario, members: tuple[tuple[str, str], ...], cluster_id: str
) -> None:
    _add_members(scenario, *members)
    site = load_site(scenario.path)

    assert registration_reanchor(site, cluster_id) is None, (
        "only the registration cluster leaving with members left moves the anchor; "
        "the last cluster keeps it (an empty site has no member to name)"
    )


@pytest.mark.parametrize("arn", [EKS_ARN, HP_ARN])
def test_a_partial_site_commit_resumes_with_the_moved_anchor(
    scenario: RemovalScenario, arn: str
) -> None:
    _add_members(scenario, ("gpu-b", "hyperpod"))
    scenario.failure = "bootstrap"
    with pytest.raises(BootstrapError, match="bootstrap"):
        remove_cluster(scenario.request(arn))
    assert _registration(scenario) == "gpu-b", "the site commit moves the anchor"
    assert "SITE_UPDATED" not in scenario.state()["completed_steps"]

    site = load_site(scenario.path)
    resolved = resolve_removal_cluster_id(
        site, arn, discover=lambda _arn: (EKS_ARN, "hp-gpu-a")
    )
    scenario.failure = None
    result = remove_cluster(
        RemoveClusterRequest(site, resolved, "REMOVE_GPU_CLUSTER", arn)
    )

    assert result["phase"] == "COMPLETED", result
    assert _registration(scenario) == "gpu-b", "the resumed commit keeps the anchor"
    assert scenario.calls.count("bootstrap") == 2, "only the failed step reruns"


def test_a_journal_started_without_the_decision_finishes_as_it_began(
    scenario: RemovalScenario,
) -> None:
    """A removal journaled by an older release bound the site without the
    anchor rule; its resume must write exactly the site it bound."""

    _add_members(scenario, ("gpu-b", "hyperpod"))
    scenario.failure = "drain"
    with pytest.raises(BootstrapError, match="drain"):
        remove_cluster(scenario.request())
    path = scenario.path.parent / "remove-cluster/gpu-a/state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    site = load_site(scenario.path)
    state.pop(REGISTRATION_REANCHOR)
    state["remaining_site_sha256"] = site_documents(site, "gpu-a")[1]
    released = dict(site.release_config)
    released["clusters"] = [
        item
        for item in site.release_config["clusters"]
        if item["cluster_id"] != "gpu-a"
    ]
    state["remaining_release_sha256"] = canonical_digest(released)
    path.write_text(json.dumps(state), encoding="utf-8")
    scenario.failure = None

    result = remove_cluster(scenario.request())

    assert result["phase"] == "COMPLETED", result
    assert REGISTRATION_REANCHOR not in result, "no decision was journaled"
    assert _registration(scenario) == "gpu-a", (
        "a journal without the decision keeps the anchor it bound"
    )
