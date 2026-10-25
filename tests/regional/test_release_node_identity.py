"""The installer template identity a cluster's nodes carry, and its diff guard.

A plain EKS cluster's DCGM host engine mode is part of the node installer
reconciler's node identity but of no heartbeat field, so the release folds it
into the installer template identity the reconciler stamps, the Agents report
and the fleet compares (``regional_release_node_identity``). Only ``host``
folds: HyperPod, the embedded engine and every cluster rendered before the mode
existed keep the template source digest. The diff plans the Agent waves for a
release that would re-deploy the reconciler over installed nodes it would
re-install (``node_installer_identity_drift``); the end-to-end waves are
``tests/regional/test_release_node_config_waves.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.cluster_scope import node_selector
from gpu_fault.node_installer_reconciler import (
    INSTALLER_ARTIFACT_ANNOTATION,
    INSTALLER_BUNDLE_ANNOTATION,
    INSTALLER_DIGEST_ANNOTATION,
    INSTALLER_NODE_UID_ANNOTATION,
    INSTALLER_STATE_ANNOTATION,
    INSTALLER_TEMPLATE_ANNOTATION,
)
from gpu_fault.node_installer_rendering import INSTALLER_DCGM_HOSTENGINE_ANNOTATION
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_node_identity as IDENTITY
from gpu_fault_release import regional_release_resume_validation as RESUME
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release.regional_deployment_inventory import GPU_RECONCILER_DEPLOYMENT
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import (
    ReleaseChangeKind,
    ReleaseComponent,
    ReleaseExecutionPlan,
)
from gpu_fault_release.regional_release_gpu_rollout import agents_converged
from gpu_fault_release.regional_release_images import NodeDependencyTarget
from gpu_fault_release.regional_release_reporting import build_release_status
from tests.regional.test_regional_release_eks_managed import _hyperpod_cluster
from tests.regional.test_release_dcgm_hostengine import HOST_ENTRY, _eks, _release

SOURCE = "4" * 64
FOLDED = hashlib.sha256(f"{SOURCE}:dcgm-hostengine:host".encode()).hexdigest()


def _target(provider: str = "eks-managed", mode: str | None = None) -> Any:
    fields: dict[str, Any] = {"cluster_id": "gpu-a", "provider": provider}
    if provider == "hyperpod":
        fields["hyperpod_cluster_name"] = "hp-gpu-a"
    if mode is not None:
        fields.update(
            dcgm_hostengine=mode,
            dcgm_hostengine_source="node-group-label" if mode == "host" else "default",
        )
    return SimpleNamespace(**fields)


# --- the identity ---------------------------------------------------------------


@pytest.mark.parametrize("mode", [None, "", "embedded"])
def test_only_the_host_engine_folds_into_the_template_identity(mode) -> None:
    assert IDENTITY.installer_template_identity(SOURCE, mode) == SOURCE, (
        "HyperPod, embedded and mode-less renders keep the source digest"
    )
    assert IDENTITY.installer_template_identity(SOURCE, "host") == FOLDED, (
        "host folds the way the reconciler names its Jobs"
    )
    assert FOLDED != SOURCE


def test_the_identity_fails_closed_on_what_it_cannot_fold() -> None:
    with pytest.raises(ValueError, match="host or embedded"):
        IDENTITY.installer_template_identity(SOURCE, "hosted")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        IDENTITY.installer_template_identity("not-a-digest", "host")
    assert IDENTITY.installer_template_identity("legacy-label", "embedded") == (
        "legacy-label"
    ), "a value nothing folds passes through as before"


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (_target("hyperpod"), SOURCE),
        (_target(), SOURCE),
        (_target(mode="embedded"), SOURCE),
        (_target(mode="host"), FOLDED),
    ],
    ids=["hyperpod", "eks-default", "eks-embedded", "eks-host"],
)
def test_the_candidate_identity_is_the_clusters(target, expected) -> None:
    release = SimpleNamespace(node_template_sha=SOURCE)
    assert IDENTITY.candidate_installer_template_identity(release, target) == expected
    assert (
        IDENTITY.candidate_installer_template_identity(release, target, SOURCE)
        == expected
    ), "the release's source digest names the candidate template"
    assert (
        IDENTITY.candidate_installer_template_identity(release, target, "9" * 64)
        == "9" * 64
    ), "an identity already resolved (the paused reconciler's) passes through"
    for phase_target in NodeDependencyTarget:
        resolved = IDENTITY.reconciler_template_identity(
            release, target, None, phase_target
        )
        if phase_target is NodeDependencyTarget.CANDIDATE:
            assert resolved == expected
        else:
            assert resolved is None, "a rollback reads its captured identity"


def _annotations(template: str, mode: str | None) -> dict[str, str]:
    annotations = {INSTALLER_TEMPLATE_ANNOTATION: template}
    if mode is not None:
        annotations[INSTALLER_DCGM_HOSTENGINE_ANNOTATION] = mode
    return annotations


@pytest.mark.parametrize(
    ("installed", "mode", "expected", "matches"),
    [
        (SOURCE, None, SOURCE, True),
        (FOLDED, "host", FOLDED, True),
        (FOLDED, "host", SOURCE, True),
        (SOURCE, "host", SOURCE, True),
        (SOURCE, "host", FOLDED, False),
        (FOLDED, None, SOURCE, False),
        (SOURCE, None, FOLDED, False),
    ],
    ids=[
        "source",
        "identity",
        "source-of-a-host-install",
        "unfolded-host-install-by-source",
        "unfolded-host-install-by-identity",
        "folded-label-without-mode",
        "embedded-install-by-host-identity",
    ],
)
def test_a_template_annotation_names_its_identity_or_its_source(
    installed, mode, expected, matches
) -> None:
    assert (
        IDENTITY.installed_template_matches(_annotations(installed, mode), expected)
        is matches
    )


def test_convergence_by_the_recorded_source_accepts_a_host_install() -> None:
    """The node-key activation waits by the release state's template source
    digest, the one value it holds: a host-mode node re-installed by the
    steady-state reconciler carries the folded identity of that source."""

    target = SimpleNamespace(provider="eks-managed", node_selector="site=a")
    node = {
        "metadata": {
            "name": "node-a",
            "uid": "uid-a",
            "labels": {"site": "a"},
            "annotations": {
                INSTALLER_STATE_ANNOTATION: "Succeeded",
                INSTALLER_ARTIFACT_ANNOTATION: "a" * 64,
                INSTALLER_TEMPLATE_ANNOTATION: FOLDED,
                INSTALLER_DCGM_HOSTENGINE_ANNOTATION: "host",
                INSTALLER_NODE_UID_ANNOTATION: "uid-a",
            },
        }
    }
    assert agents_converged([node], target, "a" * 64, template_sha=SOURCE), (
        "the source digest of the template a host node was installed from"
    )
    assert agents_converged(
        [node], target, "a" * 64, template_sha=FOLDED, dcgm_hostengine="host"
    ), "the forward wave's identity"
    assert not agents_converged(
        [node], target, "a" * 64, template_sha=SOURCE, dcgm_hostengine="embedded"
    ), "a wait that names the mode still holds the node to it"


def test_status_names_the_identity_each_clusters_agents_report(tmp_path: Path) -> None:
    release = _release(
        tmp_path, _hyperpod_cluster(tmp_path), _eks(tmp_path, **HOST_ENTRY)
    )
    hyperpod, eks = release.config.clusters
    source = release.node_template_sha

    clusters = build_release_status(release)["clusters"]

    assert clusters[hyperpod.cluster_id]["installer_template_sha256"] == source
    assert clusters[eks.cluster_id]["installer_template_sha256"] == (
        IDENTITY.installer_template_identity(source, "host")
    ), "a host-mode cluster's Agents report the folded identity, not the source"


# --- the diff guard ---------------------------------------------------------------


class NodeRunner:
    """Answers ``get nodes`` per kubectl context; records every node read."""

    dry_run = False

    def __init__(self, nodes: dict[str, list[dict[str, Any]]]) -> None:
        self.nodes = nodes
        self.reads: list[list[str]] = []

    def run(self, arguments: list[str], **_options: Any) -> str:
        if "nodes" not in arguments:
            return ""
        self.reads.append(list(arguments))
        context = arguments[arguments.index("--context") + 1]
        return json.dumps({"items": self.nodes.get(context, [])})


def _installed(
    release: Any, target: Any, name: str, *, template: str, mode: str | None = None
) -> dict[str, Any]:
    key, _, value = node_selector(target).partition("=")
    annotations = {
        INSTALLER_STATE_ANNOTATION: "Succeeded",
        INSTALLER_ARTIFACT_ANNOTATION: release.node_wheel_sha,
        INSTALLER_DIGEST_ANNOTATION: release.config.agent_config_digest,
        INSTALLER_BUNDLE_ANNOTATION: release.bundle_sha,
        INSTALLER_TEMPLATE_ANNOTATION: template,
        INSTALLER_NODE_UID_ANNOTATION: f"uid-{name}",
    }
    if mode is not None:
        annotations[INSTALLER_DCGM_HOSTENGINE_ANNOTATION] = mode
    return {
        "metadata": {
            "name": name,
            "uid": f"uid-{name}",
            "labels": {key: value},
            "annotations": annotations,
        }
    }


@pytest.fixture
def host_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A HyperPod and a host-mode EKS cluster, committed, every probe quiet."""

    release = _release(
        tmp_path, _hyperpod_cluster(tmp_path), _eks(tmp_path, **HOST_ENTRY)
    )
    release.config = replace(
        release.config,
        delivery_component_digests={
            **release.config.delivery_component_digests,
            "cpu": "c" * 64,
        },
    )
    hyperpod, eks = release.config.clusters
    STATE.save_state(release, "complete")
    state = dict(release.state)
    for probe in ("_observability_drift", "_aurora_refresh_drift"):
        monkeypatch.setattr(release, probe, lambda: False)
    for probe in ("cpu_pod_profile_drift", "node_scope_drift", "dcgm_hostengine_drift"):
        monkeypatch.setattr(DIFF, probe, lambda _release: {})
    source = release.node_template_sha
    return SimpleNamespace(
        release=release,
        state=state,
        hyperpod=hyperpod,
        eks=eks,
        current={
            hyperpod.context: [
                _installed(release, hyperpod, "hp-node", template=source)
            ],
            eks.context: [
                _installed(
                    release,
                    eks,
                    name,
                    template=IDENTITY.installer_template_identity(source, "host"),
                    mode="host",
                )
                for name in ("eks-node-a", "eks-node-b")
            ],
        },
    )


def _reconciler_only(state: dict[str, Any]) -> dict[str, Any]:
    """The recorded state of a site whose executor image is about to change."""

    return {**state, "executor_image": "registry.example/old@sha256:" + "0" * 64}


def test_a_reconciler_only_release_over_current_nodes_keeps_its_plan(
    host_site: SimpleNamespace,
) -> None:
    runner = NodeRunner(host_site.current)
    host_site.release.runner = runner

    diff = DIFF.classify_release(host_site.release, _reconciler_only(host_site.state))

    assert IDENTITY.NODE_INSTALLER_IDENTITY_DRIFT_INPUT not in diff.changed
    plan = DIFF.build_execution_plan(diff)
    assert plan.has(ReleaseComponent.RECONCILER), "the executor image rolls it"
    assert not plan.has(ReleaseComponent.AGENT), "no node is re-installed"
    assert runner.reads, "the guard looked at the installed nodes"


def test_unfolded_host_agents_turn_a_reconciler_only_release_into_agent_waves(
    host_site: SimpleNamespace,
) -> None:
    """The first release after the fold on a host-mode cluster: its Agents still
    carry the source digest, and the reconciler about to be re-deployed with
    ``*`` would re-install every one of them at once."""

    source = host_site.release.node_template_sha
    nodes = dict(host_site.current)
    nodes[host_site.eks.context] = [
        _installed(host_site.release, host_site.eks, name, template=source, mode="host")
        for name in ("eks-node-a", "eks-node-b")
    ]
    host_site.release.runner = NodeRunner(nodes)

    assert IDENTITY.node_installer_identity_drift(host_site.release) == {
        host_site.eks.cluster_id: ["eks-node-a", "eks-node-b"]
    }, "only the host-mode cluster's Agents differ"
    diff = DIFF.classify_release(host_site.release, _reconciler_only(host_site.state))

    assert IDENTITY.NODE_INSTALLER_IDENTITY_DRIFT_INPUT in diff.changed
    assert diff.kind is ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    plan = DIFF.build_execution_plan(diff)
    assert plan.has(ReleaseComponent.RECONCILER), "the reconciler still rolls"
    assert plan.has(ReleaseComponent.AGENT), (
        "the nodes it would re-install go through the canary and the wave gates"
    )


@pytest.mark.parametrize(
    ("drift", "stale"),
    [
        ("template", True),
        ("artifact", True),
        ("bundle", True),
        ("config", True),
        ("uid", False),
        ("not-installed", False),
    ],
)
def test_the_guard_judges_installed_nodes_by_the_identity_their_agents_report(
    host_site: SimpleNamespace, drift: str, stale: bool
) -> None:
    """Only what the fleet compares: a re-created Node object (another UID) or
    an install the reconciler retries holds no Agent of this identity that a
    wave would have to protect."""

    (node, _other) = host_site.current[host_site.eks.context]
    annotations = node["metadata"]["annotations"]
    if drift == "template":
        annotations[INSTALLER_TEMPLATE_ANNOTATION] = host_site.release.node_template_sha
    elif drift == "artifact":
        annotations[INSTALLER_ARTIFACT_ANNOTATION] = "9" * 64
    elif drift == "bundle":
        annotations[INSTALLER_BUNDLE_ANNOTATION] = "9" * 64
    elif drift == "config":
        annotations[INSTALLER_DIGEST_ANNOTATION] = "9" * 64
    elif drift == "uid":
        annotations[INSTALLER_NODE_UID_ANNOTATION] = "uid-of-a-replaced-node"
    else:
        annotations[INSTALLER_STATE_ANNOTATION] = "Failed"
        annotations[INSTALLER_ARTIFACT_ANNOTATION] = "9" * 64
    host_site.release.runner = NodeRunner(host_site.current)

    found = IDENTITY.node_installer_identity_drift(host_site.release)

    assert found == ({host_site.eks.cluster_id: ["eks-node-a"]} if stale else {})


def test_a_plan_that_rolls_the_agents_never_asks(host_site: SimpleNamespace) -> None:
    runner = NodeRunner({})
    host_site.release.runner = runner
    stale = {**_reconciler_only(host_site.state), "node_wheel_sha256": "0" * 64}

    plan = DIFF.build_execution_plan(DIFF.classify_release(host_site.release, stale))

    assert plan.has(ReleaseComponent.AGENT), "a new Node Agent artifact rolls waves"
    assert runner.reads == [], "the Agent waves re-install every stale node anyway"


def test_the_guard_never_reads_a_cluster_for_a_dry_run(
    host_site: SimpleNamespace,
) -> None:
    runner = NodeRunner(host_site.current)
    runner.dry_run = True
    host_site.release.runner = runner

    assert IDENTITY.node_installer_identity_drift(host_site.release) == {}
    assert runner.reads == [], "a dry run probes nothing"


# --- resume ---------------------------------------------------------------------


def _resume(reconciler_template: str) -> tuple[SimpleNamespace, list[Any]]:
    """A host-mode cluster the interrupted release had already converged."""

    target = SimpleNamespace(
        cluster_id="gpu-a",
        context="gpu-a",
        provider="eks-managed",
        node_selector="site=a",
        **HOST_ENTRY,
    )
    agent, executor, image = "a" * 64, "b" * 64, "runtime@sha256:" + "d" * 64
    metadata = {
        "required-agent-artifact-sha256": agent,
        "required-agent-compatibility-digest": agent,
        "required-agent-protocol-version": "3",
        "required-agent-config-digest": "c" * 64,
        "required-regional-executor-artifact-sha256": executor,
        "required-regional-executor-compatibility-digest": executor,
        "required-regional-executor-protocol-version": "2",
    }
    node = {
        "metadata": {
            "name": "node-a",
            "uid": "uid-a",
            "labels": {"site": "a"},
            "annotations": {
                INSTALLER_STATE_ANNOTATION: "Succeeded",
                INSTALLER_ARTIFACT_ANNOTATION: agent,
                INSTALLER_DIGEST_ANNOTATION: "c" * 64,
                INSTALLER_BUNDLE_ANNOTATION: "e" * 64,
                INSTALLER_TEMPLATE_ANNOTATION: FOLDED,
                INSTALLER_DCGM_HOSTENGINE_ANNOTATION: "host",
                INSTALLER_NODE_UID_ANNOTATION: "uid-a",
            },
        }
    }
    reconciler = {
        "spec": {
            "replicas": 1,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "reconciler",
                            "image": image,
                            "env": [
                                {
                                    "name": "GPU_FAULT_INSTALLER_BUNDLE_SHA256",
                                    "value": "e" * 64,
                                },
                                {
                                    "name": "GPU_FAULT_INSTALLER_TEMPLATE_SHA256",
                                    "value": reconciler_template,
                                },
                            ],
                        }
                    ]
                }
            },
        },
        "status": {"readyReplicas": 1},
    }
    heartbeat_templates: list[Any] = []

    def get_json(arguments: list[str]) -> dict[str, Any]:
        if GPU_RECONCILER_DEPLOYMENT in arguments:
            return reconciler
        assert "nodes" in arguments, f"unexpected read {arguments}"
        return {"items": [node]}

    def heartbeats(_target: Any, **options: Any) -> bool:
        heartbeat_templates.append(options["template_sha"])
        return True

    release = SimpleNamespace(
        config=SimpleNamespace(
            clusters=(target,),
            namespace="gpu-fault-system",
            agent_protocol_version=3,
            executor_protocol_version=2,
            component_digests={"node_runtime": agent, "executor": executor},
            agent_config_digest="c" * 64,
            runtime_profile_version="profile-a",
        ),
        cluster_registry_digest="registry-a",
        node_wheel_sha=agent,
        executor_wheel_sha=executor,
        bundle_sha="e" * 64,
        node_template_sha=SOURCE,
        executor_wheel_cm="wheel-cm",
        executor_image=image,
        _config_map_data=lambda _name: dict(metadata),
        _target_node_names=lambda _target: ("node-a",),
        _aurora_refresh_drift=lambda: False,
        _deployment_wheel=lambda *_args: "wheel-cm",
        _gpu=lambda _target, *arguments: list(arguments),
        _get_json=get_json,
        _agent_heartbeats_converged=heartbeats,
    )
    checkpoint = SimpleNamespace(
        release=release,
        previous={
            "metadata": dict(metadata),
            "runtime_image": image,
            "agent_identities": {"gpu-a": {"node_ids": ["node-a"]}},
        },
        loaded={
            "cluster_ids": ["gpu-a"],
            "cluster_registry_digest": "registry-a",
            "cluster_attempts": {
                "gpu-a": {"state": "CONVERGED", "attempt_generation": 1}
            },
            "completed_phases": [],
        },
        plan=ReleaseExecutionPlan(
            nodes=(
                ReleaseComponent.RECONCILER,
                ReleaseComponent.AGENT,
                ReleaseComponent.VERIFY,
            )
        ),
    )
    return checkpoint, heartbeat_templates


def test_a_resume_holds_a_host_cluster_to_its_folded_identity() -> None:
    checkpoint, heartbeat_templates = _resume(FOLDED)

    RESUME.validate_resume_checkpoint(
        checkpoint.release,
        loaded=checkpoint.loaded,
        previous=checkpoint.previous,
        plan=checkpoint.plan,
    )

    assert heartbeat_templates == [FOLDED], (
        "the converged Agents are held to the identity they report"
    )


def test_a_resume_refuses_a_reconciler_without_the_folded_identity() -> None:
    checkpoint, _heartbeat_templates = _resume(SOURCE)

    with pytest.raises(ReleaseError, match="resume Reconciler identity drifted"):
        RESUME.validate_resume_checkpoint(
            checkpoint.release,
            loaded=checkpoint.loaded,
            previous=checkpoint.previous,
            plan=checkpoint.plan,
        )
