"""A release with Agent waves over a site where one GPU cluster has no node.

The populated cluster rolls its Agents in waves as always. The cluster without
nodes -- parked, or unavailable -- has no node to preflight, wave or wait for:
its node work is skipped and recorded, and only its steady-state Reconciler
rolls -- with the release's own node runtime, so a node that joins later
installs the current release, never the previous one. Its rollback restores
that Reconciler alone, its resume asks nothing of nodes it never had, and the
rollback's Agent pins come from the clusters that had live Agents -- or, with
no node on any cluster, from the previous release's own ``-config-core``.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_parked_cluster as PARKED
from gpu_fault_release import regional_release_node_preflight as NODE_PREFLIGHT
from gpu_fault_release import regional_release_node_runtime_rollout as NODE_RUNTIME
from gpu_fault_release import regional_release_resume_validation as RESUME
from gpu_fault_release import regional_release_rollback_target as ROLLBACK_TARGET
from gpu_fault_release import regional_release_validation as VALIDATION
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from gpu_fault_release.regional_release_images import NodeDependencyTarget
from gpu_fault_release.regional_release_joined_nodes import (
    captured_node_ids,
    installing_newcomers,
)
from gpu_fault_release.regional_release_rollback_context import (
    rollback_controller_config_for,
)
from tests.regional._cov95_release_support import (
    OLD_IMAGE,
    ResourceRelease,
    deployment,
    previous_snapshot,
)
from tests.regional._parked_cluster_support import (
    EKS,
    GROUP,
    HYPERPOD,
    OWNERSHIP,
    SITE,
    ParkedSite,
    node,
)

BUNDLE = "b" * 64
TEMPLATE = "t" * 64


class WaveSite(ParkedSite):
    """The parked site with the Reconciler deploy recorded instead of run."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config.upgrade_max_unavailable = 1
        self.reconciler_calls: list[tuple[str, dict[str, Any]]] = []

    def _deploy_reconciler(self, target: Any, **kwargs: Any) -> tuple[str, str]:
        self.reconciler_calls.append((target.cluster_id, kwargs))
        return BUNDLE, TEMPLATE


def _roll(site: WaveSite, phase: str = "upgrade", **extra: Any) -> tuple[str, str]:
    return NODE_RUNTIME.roll_node_runtime(
        site,
        site.eks,
        phase=phase,
        wheel_cm="candidate-wheel",
        bundle_cm="candidate-bundle",
        artifact_sha=site.node_wheel_sha,
        config_digest="d" * 64,
        **{"candidate_preflight_completed": True, **extra},
    )


def test_an_agent_wave_rolls_only_the_steady_reconciler_of_a_parked_cluster(
    capsys: pytest.CaptureFixture[str],
) -> None:
    site = WaveSite()

    identity = _roll(site)

    assert identity == (BUNDLE, TEMPLATE)
    ((cluster_id, call),) = site.reconciler_calls
    assert cluster_id == EKS
    assert call["allowed_node_names"] is None, "the steady state, never a paused wave"
    assert call["zero_nodes"]["parked"] is True, "the script gets the proof"
    # The current release's node runtime: what a node that joins later installs.
    assert call["artifact_sha"] == site.node_wheel_sha
    assert call["wheel_cm"] == "candidate-wheel"
    assert call["bundle_cm"] == "candidate-bundle"
    assert call["node_dependency_target"] is NodeDependencyTarget.CANDIDATE
    entry = site.state["zero_node_clusters"][EKS]
    assert entry["state"] == "parked" and entry["phase"] == "upgrade"
    assert entry["skipped"] == [
        "agent-snapshot",
        "node-preflight-batch",
        "fleet-rollout",
        "agent-waves",
        "agent-wait",
    ]
    assert entry["reconciler"] == {"bundle_sha256": BUNDLE, "template_sha256": TEMPLATE}
    line = next(
        item
        for item in capsys.readouterr().err.splitlines()
        if item.startswith("zero-node-gpu-cluster ")
    )
    for field in (
        f"cluster={EKS}",
        "state=parked",
        "phase=upgrade",
        "skipped=agent-snapshot,node-preflight-batch,fleet-rollout,agent-waves,"
        "agent-wait",
        "reconciler=steady-state",
        f"reconciler_bundle={BUNDLE[:12]}",
    ):
        assert field in line, line


def test_a_cluster_that_lost_its_nodes_rolls_only_its_reconciler_as_unavailable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An outage beside a populated cluster: the release proceeds, the skip is
    recorded as ``unavailable`` with the reason, and the nodes that come back
    install the current release."""

    site = WaveSite(desired=2)

    assert _roll(site) == (BUNDLE, TEMPLATE)

    ((_cluster, call),) = site.reconciler_calls
    assert call["allowed_node_names"] is None
    assert call["zero_nodes"]["parked"] is False
    assert call["artifact_sha"] == site.node_wheel_sha, "the current runtime"
    entry = site.state["zero_node_clusters"][EKS]
    assert entry["state"] == "unavailable", entry
    assert entry["reason"].startswith(f"node group {GROUP} wants 2 node(s)"), entry
    line = next(
        item
        for item in capsys.readouterr().err.splitlines()
        if item.startswith("zero-node-gpu-cluster ")
    )
    assert "state=unavailable" in line and 'reason="node group ' in line, line


def test_a_cluster_with_a_node_is_never_rolled_as_one_without() -> None:
    site = WaveSite(desired=2)
    site.documents[(EKS, "nodes", "-l " + OWNERSHIP)] = {
        "items": [node("ip-10-0-1-10", {"gpu-fault.io/control-plane": SITE})]
    }

    assert PARKED.zero_nodes_now(site, site.eks) is None


def test_the_node_preflight_of_a_parked_cluster_runs_the_script_without_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = WaveSite()
    environments: list[dict[str, str]] = []
    provider_run = site.provider.run

    def run(arguments: list[str], **kwargs: Any) -> str:
        if str(arguments[0]).endswith("deploy-node-installer-reconciler.sh"):
            environments.append(dict(kwargs["env"]))
            return '{"node_count":0,"status":"PASSED","zero_node_cluster":true}'
        return provider_run(arguments, **kwargs)

    site.provider.run = run  # type: ignore[method-assign]
    monkeypatch.setattr(
        NODE_PREFLIGHT, "build_reconciler_environment", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        NODE_PREFLIGHT, "reconciler_template_identity", lambda *_a: TEMPLATE
    )
    monkeypatch.setattr(NODE_PREFLIGHT, "pin_installer_version", lambda *_a: None)

    NODE_RUNTIME.preflight_node_runtime(
        site,
        site.eks,
        phase="upgrade",
        wheel_cm="candidate-wheel",
        bundle_cm="candidate-bundle",
        artifact_sha=site.node_wheel_sha,
        config_digest="d" * 64,
    )

    (environment,) = environments
    assert environment["GPU_FAULT_RECONCILER_PREFLIGHT_ONLY"] == "true"
    assert environment["GPU_FAULT_ZERO_NODE_CLUSTER"] == "true"
    render = json.loads(environment["GPU_FAULT_ZERO_NODE_RENDER_NODE"])
    labels = render["metadata"]["labels"]
    assert labels["node.kubernetes.io/instance-type"] == "p5en.48xlarge"
    assert labels["eks.amazonaws.com/nodegroup"] == GROUP


def test_a_cluster_of_an_unknown_gpu_type_fails_its_reconciler_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No declared instance type the installer knows: no template to render,
    so the cluster's reconciler step fails before the script runs."""

    site = WaveSite(desired=2)
    site.provider.nodegroups[GROUP]["instanceTypes"] = ["x9.huge"]
    scripts: list[list[str]] = []
    provider_run = site.provider.run

    def run(arguments: list[str], **kwargs: Any) -> str:
        if str(arguments[0]).endswith("deploy-node-installer-reconciler.sh"):
            scripts.append(arguments)
        return provider_run(arguments, **kwargs)

    site.provider.run = run  # type: ignore[method-assign]
    monkeypatch.setattr(
        NODE_PREFLIGHT, "build_reconciler_environment", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        NODE_PREFLIGHT, "reconciler_template_identity", lambda *_a: TEMPLATE
    )
    monkeypatch.setattr(NODE_PREFLIGHT, "pin_installer_version", lambda *_a: None)

    with pytest.raises(ReleaseError, match="declare no GPU instance type"):
        _roll(site, candidate_preflight_completed=False)
    assert scripts == [] and site.reconciler_calls == [], "nothing ran or rolled"


def test_a_node_that_joins_the_parked_cluster_is_a_newcomer_of_the_release() -> None:
    """Captured with no node, the cluster's joiners are newcomers: the verify
    waits out their install by the steady-state Reconciler -- the one rolled
    with this release's runtime above -- instead of failing on them."""

    proof = WaveSite().state  # an empty state before the capture
    assert proof == {}
    state = {
        "previous": {
            "agent_identities": {
                EKS: PARKED.zero_node_identity(
                    {
                        "parked": True,
                        "provider": "eks-managed",
                        "reason": "r",
                        "groups": [],
                    }
                )
            }
        }
    }
    release = type("Release", (), {"state": state})()
    joining = node("ip-10-0-1-10", {"gpu-fault.io/control-plane": "x"})
    joining["metadata"]["annotations"]["gpu-fault.io/installer-state"] = "Running"

    assert captured_node_ids(state, EKS) == frozenset()
    assert installing_newcomers(release, EKS, [joining]) == {"ip-10-0-1-10": "Running"}


# --- rollback ----------------------------------------------------------------------


def _previous(site: ParkedSite) -> dict[str, Any]:
    hyperpod = {
        "node_ids": ["node-a"],
        "agent_protocol_version": 3,
        "agent_version": "0.10.0",
        "policy_version": "policy-v1",
        "artifact_sha256": "1" * 64,
        "compatibility_digest": "1" * 64,
        "runtime_profile_version": "hyperpod-v1",
        "config_digest": "7" * 64,
        "node_action_key_version": 2,
    }
    proof = PARKED.without_nodes(site, site.eks)
    assert proof is not None, "the fixture's EKS cluster has no node"
    return {
        "metadata": {"required-agent-protocol-version": "3"},
        "cluster_runtime_profiles": {},
        "clusters": {
            target.cluster_id: {
                "reconciler_wheel": "previous-wheel",
                "bundle": "previous-bundle",
                "template": "previous-template",
            }
            for target in site.config.clusters
        },
        "agent_identities": {HYPERPOD: hyperpod, EKS: PARKED.zero_node_identity(proof)},
    }


class RollbackSite(WaveSite):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.node_runtime_calls: list[dict[str, Any]] = []

    def _roll_node_runtime(self, target: Any, **kwargs: Any) -> tuple[str, str]:
        self.node_runtime_calls.append({"cluster_id": target.cluster_id, **kwargs})
        return BUNDLE, TEMPLATE


def test_the_rollback_of_a_parked_cluster_restores_its_reconciler_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = RollbackSite()
    previous = _previous(site)
    monkeypatch.setattr(
        ROLLBACK_TARGET, "validate_rollback_node_template", lambda *_a, **_k: None
    )

    ROLLBACK_TARGET.rollback_target(
        site,
        site.eks,
        previous=previous,
        artifact="1" * 64,
        config_digest="7" * 64,
        runtime_profile_version="hyperpod-v1",
        executor_artifact="2" * 64,
        executor_compatibility="2" * 64,
        node_compatibility="1" * 64,
        runtime_image="registry.example/runtime@sha256:" + "a" * 64,
        node_installer_image="registry.example/installer@sha256:" + "c" * 64,
        components=frozenset({ReleaseComponent.AGENT}),
    )

    (call,) = site.node_runtime_calls
    assert call["phase"] == "rollback"
    assert call["agent_identity"] is None, "no Agent identity was ever recorded"
    assert call["allow_legacy_identity"] is False
    assert (
        call["wheel_cm"] == "previous-wheel" and call["bundle_cm"] == "previous-bundle"
    )


def test_the_rollback_of_a_parked_cluster_verifies_no_agent_and_its_pins_come_from_live_agents() -> (
    None
):
    site = RollbackSite()
    previous = _previous(site)

    VALIDATION.validate_agent_rollback_target(site, previous, site.eks)
    pins = rollback_controller_config_for(site, previous, "hyperpod-v1")

    assert pins is not None
    assert pins["GPU_FAULT_REQUIRED_AGENT_VERSION"] == "0.10.0"
    assert pins["GPU_FAULT_REQUIRED_POLICY_VERSION"] == "policy-v1"


def _zero_node(state: str) -> dict[str, Any]:
    """The explicit entry a snapshot records for a cluster without nodes."""

    return PARKED.zero_node_identity(
        {
            "parked": state == "parked",
            "provider": "hyperpod",
            "reason": f"{state} for the test",
            "groups": [],
        }
    )


#: The Agent pins every CPU role's ``-config-core`` carried when the release
#: started, as the capture records them (``cpu_role_config_maps``).
CAPTURED_PINS = {
    "GPU_FAULT_REQUIRED_AGENT_VERSION": "0.9.0",
    "GPU_FAULT_REQUIRED_POLICY_VERSION": "policy-v0",
    "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "hyperpod-v1",
}


def _captured_config_maps(**overrides: dict[str, str]) -> dict[str, dict[str, str]]:
    maps = {
        f"{deployment}-config-core": {
            "GPU_FAULT_PROCESSOR_WORKERS": "24",
            **CAPTURED_PINS,
        }
        for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
    }
    for name, data in overrides.items():
        maps[name] = data
    return maps


def _without_any_node(site: RollbackSite, **fields: Any) -> dict[str, Any]:
    """The previous state of a release that started with no node anywhere:
    gpu-a unavailable, gpu-eks parked."""

    return {
        **_previous(site),
        "agent_identities": {
            HYPERPOD: _zero_node("unavailable"),
            EKS: _zero_node("parked"),
        },
        **fields,
    }


def test_with_no_node_on_any_cluster_the_rollback_stages_the_captured_pins(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every cluster the release covers recorded an explicit zero-node entry,
    parked or unavailable: no Agent ran to report an identity. The pins staged
    are the previous release's own, from the ``-config-core`` the capture
    recorded, and the rollback says so, naming the clusters and the source."""

    site = RollbackSite()
    previous = _without_any_node(site, cpu_role_config_maps=_captured_config_maps())

    assert rollback_controller_config_for(site, previous, "hyperpod-v1") == (
        CAPTURED_PINS
    )
    line = next(
        item
        for item in capsys.readouterr().err.splitlines()
        if item.startswith("rollback-agent-pins ")
    )
    for field in (
        "source=previous-config-core",
        "reason=no-cluster-had-a-node",
        "clusters=" + ",".join(sorted([HYPERPOD, EKS])),
    ):
        assert field in line, line


@pytest.mark.parametrize(
    "captured",
    [
        pytest.param({}, id="no-cpu-state-captured"),
        pytest.param(
            _captured_config_maps(
                **{
                    f"{inventory.CPU_RUNTIME_DEPLOYMENTS[0]}-config-core": {
                        **CAPTURED_PINS,
                        "GPU_FAULT_REQUIRED_POLICY_VERSION": "policy-v2",
                    }
                }
            ),
            id="roles-that-disagree",
        ),
        pytest.param(
            _captured_config_maps(
                **{
                    f"{inventory.CPU_RUNTIME_DEPLOYMENTS[-1]}-config-core": {
                        "GPU_FAULT_PROCESSOR_WORKERS": "24"
                    }
                }
            ),
            id="a-role-without-the-pins",
        ),
    ],
)
def test_with_no_node_and_no_captured_pins_the_rollback_stages_none(
    capsys: pytest.CaptureFixture[str], captured: dict[str, dict[str, str]]
) -> None:
    """A release without a CPU stage captures no ``-config-core`` and changes
    none; roles that disagree name no single pin. Nothing is staged then: the
    CPU restore puts the previous ``-config-core`` back as it is."""

    site = RollbackSite()
    previous = _without_any_node(site, cpu_role_config_maps=captured)

    assert rollback_controller_config_for(site, previous, "hyperpod-v1") == {}
    line = next(
        item
        for item in capsys.readouterr().err.splitlines()
        if item.startswith("rollback-agent-pins ")
    )
    assert "source=none" in line and "reason=no-cluster-had-a-node" in line, line


@pytest.mark.parametrize(
    ("identities", "refusal"),
    [
        pytest.param({}, "snapshot is empty", id="no-entry"),
        pytest.param(
            {EKS: _zero_node("parked")},
            "snapshot is empty",
            id="a-cluster-without-an-entry",
        ),
        pytest.param(
            {EKS: _zero_node("parked"), HYPERPOD: {}},
            "agent_version differs across clusters",
            id="an-empty-identity-beside-a-zero-node-entry",
        ),
        pytest.param(
            {
                EKS: _zero_node("parked"),
                HYPERPOD: {"node_ids": [], PARKED.ZERO_NODES_IDENTITY_KEY: {}},
            },
            "snapshot is empty",
            id="a-zero-node-entry-in-no-known-state",
        ),
    ],
)
def test_a_snapshot_short_of_an_entry_per_cluster_still_refuses_the_rollback_pins(
    identities: dict[str, Any], refusal: str
) -> None:
    """Only an explicit zero-node entry for every cluster means no Agent ran;
    anything less is a snapshot that cannot say which pins to restore."""

    site = RollbackSite()
    previous = {**_previous(site), "agent_identities": identities}

    with pytest.raises(ReleaseError, match=refusal):
        rollback_controller_config_for(site, previous, "hyperpod-v1")


def test_the_zero_node_rollback_runs_the_parked_path_of_the_node_runtime() -> None:
    """The rolled-back Reconciler is the previous one, its template the
    previous ConfigMap; still no node to wave."""

    site = WaveSite()
    site.state = {"previous": _previous(site)}

    identity = _roll(
        site,
        phase="rollback",
        steady_template_config_map="previous-template",
        steady_runtime_image="registry.example/runtime@sha256:" + "a" * 64,
    )

    assert identity == (BUNDLE, TEMPLATE)
    ((_cluster, call),) = site.reconciler_calls
    assert call["template_config_map"] == "previous-template"
    assert call["node_dependency_target"] is NodeDependencyTarget.PREVIOUS
    assert call["allowed_node_names"] is None


def test_a_rollback_whose_provider_read_fails_still_uses_the_recorded_park() -> None:
    """The rollback of a release that started on a parked cluster is not held
    up by an EKS read that fails now: the snapshot's proof and an empty node
    list are enough to know there is nothing to wave."""

    site = WaveSite()
    site.state = {"previous": _previous(site)}

    def broken(_arguments: list[str], **_kwargs: Any) -> str:
        raise RuntimeError("ThrottlingException")

    site.provider.run = broken  # type: ignore[method-assign]

    assert _roll(site, phase="rollback") == (BUNDLE, TEMPLATE)


# --- resume ------------------------------------------------------------------------


def _checkpoint(release: ResourceRelease, states: dict[str, str]) -> dict[str, Any]:
    return {
        "cluster_ids": [target.cluster_id for target in release.config.clusters],
        "cluster_registry_digest": release.cluster_registry_digest,
        "cluster_attempts": {
            cluster_id: {"state": state} for cluster_id, state in states.items()
        },
        "completed_phases": [],
    }


@pytest.mark.parametrize(
    "parked_state", ["CONVERGED", "PENDING"], ids=["parked-done", "parked-not-started"]
)
def test_a_resume_after_an_interrupted_wave_asks_nothing_of_the_parked_cluster(
    parked_state: str,
) -> None:
    """The HyperPod cluster's waves converged, the release stopped before or
    after the parked cluster: the resume validates the populated cluster's
    Agents and only the parked cluster's Reconciler."""

    release = ResourceRelease(("gpu-a", "gpu-eks"))
    if parked_state == "PENDING":
        # Not started: the parked cluster still runs the previous release.
        reconciler = deployment(
            inventory.GPU_RECONCILER_DEPLOYMENT,
            image=OLD_IMAGE,
            wheel="previous-wheel",
            container="reconciler",
        )
        release.documents[
            ("gpu-eks", "deployment", inventory.GPU_RECONCILER_DEPLOYMENT)
        ] = reconciler
    previous = previous_snapshot(release)
    previous["agent_identities"]["gpu-eks"] = PARKED.zero_node_identity(
        {"parked": True, "provider": "eks-managed", "reason": "r", "groups": []}
    )

    RESUME.validate_resume_checkpoint(
        release,
        loaded=_checkpoint(release, {"gpu-a": "CONVERGED", "gpu-eks": parked_state}),
        previous=copy.deepcopy(previous),
        plan=ReleaseExecutionPlan(nodes=(ReleaseComponent.AGENT,)),
    )

    asked = {item["cluster_id"] for item in release.heartbeat_checks}
    assert "gpu-eks" not in asked, "no Agent heartbeat asked of the parked cluster"
    if parked_state == "CONVERGED":
        assert asked == {"gpu-a"}


class LostNodesRelease(ResourceRelease):
    """``gpu-eks`` lists no node now: its fleet node names refuse, as
    ``target_node_names`` answers an empty node set."""

    def _target_node_names(self, target: Any) -> tuple[str, ...]:
        if target.cluster_id == "gpu-eks":
            raise ReleaseError("gpu-eks has no GPU nodes for fleet rollout")
        return self.node_names


def test_a_resume_after_a_cluster_lost_every_node_mid_release_asks_nothing_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot recorded ``gpu-eks``'s node; it left and the cluster
    selects none now. Its departure proven, the resume treats the cluster as
    one without nodes -- its Reconciler checked, no Agent asked -- instead of
    refusing on the empty node list."""

    release = LostNodesRelease(("gpu-a", "gpu-eks"))
    release.documents[("gpu-eks", "nodes", "")] = {"items": []}
    previous = previous_snapshot(release)
    recorded = tuple(previous["agent_identities"]["gpu-eks"]["node_ids"])
    assert recorded, "the snapshot recorded the node that left"
    proofs: list[tuple[str, tuple[str, ...]]] = []

    def departed(_release: Any, target: Any, node_names: Any, **kwargs: Any) -> Any:
        proofs.append((target.cluster_id, tuple(kwargs["present_node_names"])))
        return frozenset(node_names)

    monkeypatch.setattr(RESUME, "departed_node_ids", departed)

    RESUME.validate_resume_checkpoint(
        release,
        loaded=_checkpoint(release, {"gpu-a": "CONVERGED", "gpu-eks": "CONVERGED"}),
        previous=previous,
        plan=ReleaseExecutionPlan(nodes=(ReleaseComponent.AGENT,)),
    )

    assert proofs == [("gpu-eks", ())], "the departure is proven, not presumed"
    assert {item["cluster_id"] for item in release.heartbeat_checks} == {"gpu-a"}


# --- a FULL release over the parked cluster, then its rollback ----------------------

FULL_GPU = (
    ReleaseComponent.DCGM,
    ReleaseComponent.OBSERVABILITY,
    ReleaseComponent.EXECUTOR,
    ReleaseComponent.WATCHER,
    ReleaseComponent.COLLECTOR,
    ReleaseComponent.RECONCILER,
    ReleaseComponent.AGENT,
)


class StageSite(WaveSite):
    """Records every cluster-level apply; rolls the node runtime for real."""

    executor_wheel_cm = "candidate-wheel"
    bundle_cm = "candidate-bundle"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.applied: list[str] = []

    def _apply_gpu_dcgm_exporter(self, target: Any, **_kwargs: Any) -> None:
        self.applied.append(f"{target.cluster_id}:dcgm")

    def _apply_gpu_adot_collector(self, target: Any) -> None:
        self.applied.append(f"{target.cluster_id}:adot")

    def _apply_gpu_deployments(
        self, target: Any, _wheel: str, *, deployment_names: Any, **_kwargs: Any
    ) -> None:
        self.applied.extend(
            f"{target.cluster_id}:{name}" for name in sorted(deployment_names)
        )

    def _roll_node_runtime(self, target: Any, **kwargs: Any) -> tuple[str, str]:
        return NODE_RUNTIME.roll_node_runtime(self, target, **kwargs)


def test_a_full_release_rolls_the_parked_clusters_components_and_skips_its_nodes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from gpu_fault_release.regional_release_diff import diff_from_changed
    from gpu_fault_release.regional_release_gpu_rollout import upgrade_gpu_target

    site = StageSite()
    progress: list[tuple[tuple[str, ...], str]] = []

    upgrade_gpu_target(
        site,
        site.eks,
        diff_from_changed({"node_runtime_wheel", "executor_wheel", "dcgm"}),
        ReleaseExecutionPlan(nodes=FULL_GPU),
        progress=lambda components, status, _detail: progress.append(
            (tuple(item.value for item in components), status)
        ),
        candidate_preflighted=True,
    )

    assert sorted(site.applied) == sorted(
        [
            f"{EKS}:dcgm",
            f"{EKS}:adot",
            f"{EKS}:gpu-fault-cluster-executor",
            f"{EKS}:gpu-fault-completion-watcher",
            f"{EKS}:gpu-fault-kubernetes-node-resource-collector",
        ]
    ), "every cluster-level component rolls"
    ((_cluster, call),) = site.reconciler_calls
    assert call["allowed_node_names"] is None and call["zero_nodes"]["parked"] is True
    assert (("reconciler", "agent"), "COMPLETED") in progress
    assert "skipped=agent-snapshot" in capsys.readouterr().err


class RollbackStageSite(StageSite):
    def _config_map_sha(self, _kubectl: Any, _name: str, _key: str) -> str:
        return "2" * 64


def test_a_full_rollback_of_the_parked_cluster_restores_its_components_and_reconciler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = RollbackStageSite()
    previous = _previous(site)
    previous["clusters"][EKS].update(
        {
            "wheel": "previous-wheel",
            "dcgm_image": "registry.example/dcgm@sha256:" + "d" * 64,
        }
    )
    site.state = {"previous": previous}
    monkeypatch.setattr(
        ROLLBACK_TARGET, "validate_rollback_node_template", lambda *_a, **_k: None
    )
    preflights: list[tuple[str, str, bool, str | None]] = []
    monkeypatch.setattr(
        NODE_RUNTIME,
        "ensure_node_candidate_preflight",
        lambda _release, target, candidate: preflights.append(
            (
                target.cluster_id,
                candidate.phase,
                candidate.zero_nodes is not None,
                candidate.template_config_map,
            )
        ),
    )

    ROLLBACK_TARGET.rollback_target(
        site,
        site.eks,
        previous=previous,
        artifact="1" * 64,
        config_digest="7" * 64,
        runtime_profile_version="hyperpod-v1",
        executor_artifact="2" * 64,
        executor_compatibility="2" * 64,
        node_compatibility="1" * 64,
        runtime_image="registry.example/runtime@sha256:" + "a" * 64,
        node_installer_image="registry.example/installer@sha256:" + "c" * 64,
        components=frozenset(FULL_GPU) - {ReleaseComponent.OBSERVABILITY},
    )

    assert preflights == [(EKS, "rollback", True, "previous-template")], (
        "the pinned previous template is admitted in its zero-node mode: no "
        "node batch, no render node"
    )
    assert f"{EKS}:dcgm" in site.applied
    assert f"{EKS}:gpu-fault-cluster-executor" in site.applied
    ((_cluster, call),) = site.reconciler_calls
    assert call["node_dependency_target"] is NodeDependencyTarget.PREVIOUS
    assert call["wheel_cm"] == "previous-wheel" and call["zero_nodes"] is not None
    assert site.state["zero_node_clusters"][EKS]["phase"] == "rollback"


def test_a_resume_of_a_full_release_validates_the_parked_clusters_reconciler_only() -> (
    None
):
    release = ResourceRelease(("gpu-a", "gpu-eks"))
    previous = previous_snapshot(release)
    previous["agent_identities"]["gpu-eks"] = PARKED.zero_node_identity(
        {"parked": True, "provider": "eks-managed", "reason": "r", "groups": []}
    )

    RESUME.validate_resume_checkpoint(
        release,
        loaded=_checkpoint(release, {"gpu-a": "CONVERGED", "gpu-eks": "CONVERGED"}),
        previous=previous,
        plan=ReleaseExecutionPlan(nodes=tuple(RESUME.GPU_COMPONENTS)),
    )

    assert {item["cluster_id"] for item in release.heartbeat_checks} == {"gpu-a"}
