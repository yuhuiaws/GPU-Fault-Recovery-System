from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_node_runtime_rollout as RUNTIME
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import (
    ReleaseChangeKind,
    ReleaseComponent,
    ReleaseDiff,
    ReleaseExecutionPlan,
)
from gpu_fault_release.regional_release_gpu_rollout import upgrade_gpu_target
from gpu_fault_release.regional_release_images import (
    NodeDependencyTarget,
    previous_node_dependency_environment,
)
from gpu_fault_release.regional_release_rendering import build_reconciler_environment
from gpu_fault_release.regional_release_rollback_target import rollback_target
from tests.deploy.test_installer_template_identity import template_job

ARTIFACT = "1" * 64
CONFIG = "2" * 64
BUNDLE = "3" * 64
TEMPLATE = "4" * 64
PREVIOUS_DEPENDENCY = {
    "reference": "registry.example/node-dependencies@sha256:" + "a" * 64,
    "wheelhouse_sha256": "b" * 64,
}
CANDIDATE_DEPENDENCY = {
    "reference": "registry.example/node-dependencies@sha256:" + "c" * 64,
    "wheelhouse_sha256": "d" * 64,
}
PREVIOUS_EXECUTOR = "registry.example/previous-executor@sha256:" + "e" * 64
CANDIDATE_EXECUTOR = "registry.example/candidate-executor@sha256:" + "f" * 64
INSTALLER = "registry.example/installer@sha256:" + "5" * 64


@pytest.fixture
def node_release(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    environments: list[dict[str, str]] = []
    operations: list[str] = []
    job = template_job(offline=False)
    job["spec"]["template"]["spec"]["volumes"][-1]["configMap"]["name"] = (
        "shared-bundle"
    )
    from gpu_fault.node_installer_rendering import configure_node_dependencies

    configure_node_dependencies(
        job, PREVIOUS_DEPENDENCY["reference"], PREVIOUS_DEPENDENCY["wheelhouse_sha256"]
    )
    template_document = {"data": {"job.yaml": yaml.safe_dump(job)}}
    previous = {
        "release_manifest_schema_version": 4,
        "node_dependencies": dict(PREVIOUS_DEPENDENCY),
        "metadata": {
            "required-agent-protocol-version": 3,
            "required-node-action-key-version": 2,
        },
        "clusters": {
            "gpu-a": {
                "reconciler_wheel": "previous-wheel",
                "bundle": "shared-bundle",
                "bundle_sha256": BUNDLE,
                "template": "previous-template",
                "template_sha256": TEMPLATE,
                "template_content_sha256": hashlib.sha256(
                    template_document["data"]["job.yaml"].encode()
                ).hexdigest(),
            }
        },
        "agent_identities": {
            "gpu-a": {
                "agent_protocol_version": 3,
                "agent_version": "0.10.0",
                "artifact_sha256": ARTIFACT,
                "compatibility_digest": ARTIFACT,
                "installer_bundle_sha256": BUNDLE,
                "installer_template_sha256": TEMPLATE,
                "policy_version": "test-policy",
                "runtime_profile_version": "previous-profile",
                "config_digest": CONFIG,
                "node_action_key_version": 2,
                "node_ids": ["node-a"],
            }
        },
    }
    node = {
        "metadata": {
            "name": "node-a",
            "uid": "uid-a",
            "labels": {"topology.kubernetes.io/zone": "zone-a"},
            "annotations": {
                "gpu-fault.io/installer-artifact-sha256": ARTIFACT,
                "gpu-fault.io/installer-config-digest": CONFIG,
                "gpu-fault.io/installer-node-uid": "uid-a",
                "gpu-fault.io/installer-state": "Succeeded",
            },
        },
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }

    def run(arguments, *, env, **kwargs):
        assert Path(arguments[0]).name == "deploy-node-installer-reconciler.sh"
        environments.append(dict(env))
        return (
            json.dumps({"status": "PASSED", "node_count": 1})
            if kwargs.get("capture")
            else ""
        )

    def fleet_command(operation, _payload):
        operations.append(operation)
        if operation == "create":
            return {
                "status": "PENDING",
                "waves": [["node-a"]],
                "nodes": [{"node_id": "node-a", "status": "PENDING"}],
            }
        if operation == "next-wave":
            return {"node_ids": ["node-a"]}
        if operation == "get":
            return {"status": "SUCCEEDED"}
        assert operation in {"normalize-records", "terminalize-cluster-rollouts"}
        return {}

    release = SimpleNamespace(
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            executor_wheel=Path("executor.whl"),
            agent_config_digest=CONFIG,
            component_digests={"node_runtime": ARTIFACT},
            runtime_profile_version="candidate-profile",
            release_manifest_schema_version=4,
            release_delivery_identity={
                "images": {"node_dependencies": dict(CANDIDATE_DEPENDENCY)}
            },
            upgrade_max_unavailable=1,
            rollback_max_unavailable=1,
        ),
        runner=SimpleNamespace(dry_run=False, run=run),
        state={"previous": previous},
        release_id="candidate",
        bundle_cm="shared-bundle",
        bundle_sha=BUNDLE,
        node_template_sha=TEMPLATE,
        node_wheel_sha=ARTIFACT,
        executor_wheel_cm="candidate-wheel",
        executor_image=CANDIDATE_EXECUTOR,
        node_installer_image=INSTALLER,
        _target_node_names=lambda _target: ("node-a",),
        _get_json=lambda arguments: (
            template_document
            if arguments[-1] == "previous-template"
            else {"items": [node]}
        ),
        _gpu=lambda _target, *arguments: list(arguments),
        _settle_installer_jobs=lambda _target: operations.append("settle"),
        _fleet_deployment_id=lambda *_args, **_kwargs: "test-deployment",
        _fleet_command=fleet_command,
        _wait_agents=lambda *_args, **_kwargs: None,
        _deploy_reconciler=lambda target, **kwargs: FLEET.deploy_reconciler(
            release, target, **kwargs
        ),
        _roll_node_runtime=lambda target, **kwargs: RUNTIME.roll_node_runtime(
            release, target, **kwargs
        ),
    )
    target = SimpleNamespace(
        cluster_id="gpu-a",
        context="gpu-a-context",
        hyperpod_cluster_name="hp-gpu-a",
        fleet_master_file=None,
    )
    monkeypatch.setattr(
        FLEET, "wait_deployment_rollout", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        FLEET, "reconciler_container_env", lambda *_args: environments[-1]
    )
    monkeypatch.setattr(
        FLEET, "hand_wave_to_reconciler", lambda *_args, **_kwargs: None
    )
    for module in (FLEET, RUNTIME):
        monkeypatch.setattr(
            module,
            "ensure_rollout_wave_safe",
            lambda *_args, **_kwargs: {"minimum_lease_remaining_seconds": 120},
        )
    return SimpleNamespace(
        release=release,
        previous=previous,
        target=target,
        environments=environments,
        operations=operations,
        template_document=template_document,
    )


def legacy_template(node_release):
    text = yaml.safe_dump(template_job(offline=False))
    node_release.template_document["data"]["job.yaml"] = text
    node_release.previous["clusters"]["gpu-a"]["template_content_sha256"] = (
        hashlib.sha256(text.encode()).hexdigest()
    )


def rollback(node_release, component=ReleaseComponent.AGENT):
    rollback_target(
        node_release.release,
        node_release.target,
        previous=node_release.previous,
        artifact=ARTIFACT,
        config_digest=CONFIG,
        runtime_profile_version="previous-profile",
        executor_artifact=ARTIFACT,
        executor_compatibility=ARTIFACT,
        node_compatibility=ARTIFACT,
        runtime_image=PREVIOUS_EXECUTOR,
        node_installer_image=INSTALLER,
        components=frozenset({component}),
    )


@pytest.mark.parametrize("current_version,previous_version", [(4, 4), (4, 3), (3, 4)])
@pytest.mark.parametrize("same_bundle", [True, False])
def test_rollback_image_target_reaches_preflight_waves_and_steady_restore(
    node_release, monkeypatch, current_version, previous_version, same_bundle
):
    release = node_release.release
    release.config.release_manifest_schema_version = current_version
    node_release.previous["release_manifest_schema_version"] = previous_version
    if current_version == 3:
        release.config.release_delivery_identity = {"images": {}}
    if previous_version == 3:
        node_release.previous.pop("node_dependencies")
        legacy_template(node_release)
    if not same_bundle:
        release.bundle_cm = "candidate-bundle"
        release.bundle_sha = "6" * 64
    monkeypatch.setenv(
        "GPU_FAULT_NODE_DEPENDENCY_IMAGE", CANDIDATE_DEPENDENCY["reference"]
    )
    monkeypatch.setenv(
        "GPU_FAULT_NODE_WHEELHOUSE_SHA256", CANDIDATE_DEPENDENCY["wheelhouse_sha256"]
    )

    rollback(node_release)

    environments = node_release.environments
    assert len(environments) == 4
    assert environments[0]["GPU_FAULT_RECONCILER_PREFLIGHT_ONLY"] == "true"
    assert environments[0]["GPU_FAULT_REQUIRE_ROLLBACK_SLOT"] == "true"
    assert [env["GPU_FAULT_INSTALLER_ALLOWED_NODES"] for env in environments] == [
        "node-a",
        "",
        "node-a",
        "*",
    ]
    assert [env["GPU_FAULT_RUNTIME_IMAGE"] for env in environments] == [
        CANDIDATE_EXECUTOR,
        CANDIDATE_EXECUTOR,
        CANDIDATE_EXECUTOR,
        PREVIOUS_EXECUTOR,
    ]
    expected = PREVIOUS_DEPENDENCY if previous_version == 4 else {}
    assert {
        (
            env["GPU_FAULT_NODE_DEPENDENCY_IMAGE"],
            env["GPU_FAULT_NODE_WHEELHOUSE_SHA256"],
        )
        for env in environments
    } == {(expected.get("reference", ""), expected.get("wheelhouse_sha256", ""))}


@pytest.mark.parametrize("phase", ["bootstrap", "join", "upgrade", "rollback"])
def test_standalone_preflight_uses_explicit_phase(node_release, phase):
    release = node_release.release
    RUNTIME.preflight_node_runtime(
        release,
        node_release.target,
        phase=phase,
        wheel_cm="test-wheel",
        bundle_cm=release.bundle_cm,
        artifact_sha=ARTIFACT,
        config_digest=CONFIG,
        bundle_sha256=BUNDLE,
    )

    assert node_release.operations == []
    assert len(node_release.environments) == 1
    expected = PREVIOUS_DEPENDENCY if phase == "rollback" else CANDIDATE_DEPENDENCY
    environment = node_release.environments[0]
    assert environment["GPU_FAULT_NODE_DEPENDENCY_IMAGE"] == expected["reference"]
    assert (
        environment["GPU_FAULT_NODE_WHEELHOUSE_SHA256"] == expected["wheelhouse_sha256"]
    )


@pytest.mark.parametrize(
    "component", [ReleaseComponent.AGENT, ReleaseComponent.RECONCILER]
)
@pytest.mark.parametrize("current_version", [3, 4])
def test_forward_deployment_selects_candidate_with_same_bundle(
    node_release, component, current_version
):
    release = node_release.release
    release.config.release_manifest_schema_version = current_version
    if current_version == 3:
        release.config.release_delivery_identity = {"images": {}}
    release.state["phase"] = "rollback-data-restored"
    upgrade_gpu_target(
        release,
        node_release.target,
        ReleaseDiff(ReleaseChangeKind.DATA_PLANE_COMPATIBLE, frozenset()),
        ReleaseExecutionPlan(nodes=(component,)),
    )

    assert len(node_release.environments) == (
        4 if component == ReleaseComponent.AGENT else 1
    )
    expected = CANDIDATE_DEPENDENCY if current_version == 4 else {}
    assert {
        (
            env["GPU_FAULT_NODE_DEPENDENCY_IMAGE"],
            env["GPU_FAULT_NODE_WHEELHOUSE_SHA256"],
        )
        for env in node_release.environments
    } == {(expected.get("reference", ""), expected.get("wheelhouse_sha256", ""))}


def test_legacy_fast_restore_keeps_previous_offline_absence(node_release):
    node_release.previous["release_manifest_schema_version"] = 3
    node_release.previous.pop("node_dependencies")
    legacy_template(node_release)
    identity = node_release.previous["agent_identities"]["gpu-a"]
    identity["installer_bundle_sha256"] = None
    identity["installer_template_sha256"] = None

    rollback(node_release)

    assert node_release.operations == ["settle", "settle"]
    assert len(node_release.environments) == 3
    assert node_release.environments[-1]["GPU_FAULT_RUNTIME_IMAGE"] == PREVIOUS_EXECUTOR
    assert all(
        env["GPU_FAULT_NODE_DEPENDENCY_IMAGE"] == ""
        and env["GPU_FAULT_NODE_WHEELHOUSE_SHA256"] == ""
        for env in node_release.environments
    ), "legacy rollback inherited candidate offline dependency identity"


@pytest.mark.parametrize("current_version", [3, 4])
@pytest.mark.parametrize(
    "component", [ReleaseComponent.AGENT, ReleaseComponent.RECONCILER]
)
@pytest.mark.parametrize(
    "identity",
    [
        None,
        {},
        [],
        {
            **PREVIOUS_DEPENDENCY,
            "reference": "registry.example/node-dependencies:latest",
        },
        {**PREVIOUS_DEPENDENCY, "wheelhouse_sha256": "invalid"},
    ],
)
def test_missing_or_invalid_previous_identity_fails_before_installer_jobs(
    node_release, current_version, component, identity
):
    node_release.release.config.release_manifest_schema_version = current_version
    node_release.previous["node_dependencies"] = identity

    with pytest.raises(ReleaseError, match="offline node dependency identity"):
        rollback(node_release, component)

    assert node_release.environments == []
    assert node_release.operations == []


@pytest.mark.parametrize(
    "cluster",
    [
        None,
        {},
        {"bundle": "wrong-bundle", "bundle_sha256": BUNDLE},
        {"bundle": "shared-bundle"},
        {"bundle": "shared-bundle", "bundle_sha256": "9" * 64},
    ],
)
def test_mismatched_previous_bundle_binding_cannot_select_candidate(
    node_release, cluster
):
    node_release.previous["clusters"] = {"gpu-a": cluster}

    with pytest.raises(ReleaseError, match="rollback bundle"):
        FLEET.deploy_reconciler(
            node_release.release,
            node_release.target,
            wheel_cm="previous-wheel",
            bundle_cm="shared-bundle",
            bundle_sha256=BUNDLE,
            artifact_sha=ARTIFACT,
            config_digest=CONFIG,
            node_dependency_target=NodeDependencyTarget.PREVIOUS,
        )

    assert node_release.environments == []
    assert node_release.operations == []


@pytest.mark.parametrize("previous", [None, {}, []])
def test_rollback_requires_captured_previous_snapshot(node_release, previous):
    node_release.release.state["previous"] = previous
    with pytest.raises(ReleaseError, match="captured previous release identity"):
        rollback(node_release, ReleaseComponent.RECONCILER)
    assert node_release.operations == []


@pytest.mark.parametrize("version", [0, False, "", "invalid", 5])
def test_unknown_previous_schema_cannot_be_treated_as_legacy(node_release, version):
    node_release.previous["release_manifest_schema_version"] = version
    node_release.previous.pop("node_dependencies")
    with pytest.raises(ReleaseError, match="previous release manifest schema version"):
        rollback(node_release, ReleaseComponent.RECONCILER)
    assert node_release.environments == []
    assert node_release.operations == []


@pytest.mark.parametrize("version", [None, 1, 2, 3])
def test_legacy_absence_does_not_require_new_bundle_identity(version):
    assert previous_node_dependency_environment(
        {
            "release_manifest_schema_version": version,
            "clusters": {"gpu-a": {"bundle": "legacy-bundle"}},
        },
        cluster_id="gpu-a",
        bundle_cm="legacy-bundle",
        bundle_sha256=None,
    ) == {"GPU_FAULT_NODE_DEPENDENCY_IMAGE": "", "GPU_FAULT_NODE_WHEELHOUSE_SHA256": ""}


def test_dry_run_rollback_uses_previous_identity(node_release):
    node_release.release.runner.dry_run = True
    rollback(node_release)
    assert len(node_release.environments) == 1
    assert (
        node_release.environments[0]["GPU_FAULT_NODE_DEPENDENCY_IMAGE"]
        == PREVIOUS_DEPENDENCY["reference"]
    )


@pytest.mark.parametrize(
    "bundle_cm,bundle_sha", [("other-bundle", BUNDLE), ("shared-bundle", "9" * 64)]
)
def test_candidate_image_cannot_be_paired_with_another_bundle(
    node_release, bundle_cm, bundle_sha
):
    with pytest.raises(ReleaseError, match="candidate bundle disagrees"):
        FLEET.deploy_reconciler(
            node_release.release,
            node_release.target,
            wheel_cm="test-wheel",
            bundle_cm=bundle_cm,
            bundle_sha256=bundle_sha,
            artifact_sha=ARTIFACT,
            config_digest=CONFIG,
        )
    assert node_release.environments == []
    assert node_release.operations == []


def test_split_rollback_requires_explicit_bundle_hash(node_release):
    with pytest.raises(ReleaseError, match="rollback bundle disagrees"):
        RUNTIME.preflight_node_runtime(
            node_release.release,
            node_release.target,
            phase="rollback",
            wheel_cm="previous-wheel",
            bundle_cm="shared-bundle",
            artifact_sha=ARTIFACT,
            config_digest=CONFIG,
        )
    assert node_release.environments == []
    assert node_release.operations == []


def test_candidate_missing_identity_never_uses_previous_image(node_release):
    node_release.release.config.release_delivery_identity = {"images": {}}
    with pytest.raises(ReleaseError, match="offline node dependency identity"):
        RUNTIME.preflight_node_runtime(
            node_release.release,
            node_release.target,
            phase="upgrade",
            wheel_cm="test-wheel",
            bundle_cm="shared-bundle",
            artifact_sha=ARTIFACT,
            config_digest=CONFIG,
        )
    assert node_release.environments == []
    assert node_release.operations == []


def test_unknown_phase_does_not_guess_an_image_target(node_release):
    with pytest.raises(ReleaseError, match="known rollout phase"):
        RUNTIME.preflight_node_runtime(
            node_release.release,
            node_release.target,
            phase="unknown",
            wheel_cm="test-wheel",
            bundle_cm="shared-bundle",
            artifact_sha=ARTIFACT,
            config_digest=CONFIG,
        )
    assert node_release.environments == []


def test_unknown_image_target_fails_closed(node_release):
    with pytest.raises(ReleaseError, match="image target is invalid"):
        build_reconciler_environment(
            node_release.release,
            node_release.target,
            wheel_cm="test-wheel",
            bundle_cm="shared-bundle",
            artifact_sha=ARTIFACT,
            config_digest=CONFIG,
            node_dependency_target="guess",
        )


@pytest.mark.parametrize(
    "component", [ReleaseComponent.AGENT, ReleaseComponent.RECONCILER]
)
def test_rollback_restores_the_captured_content_pin(node_release, component):
    expected = node_release.previous["clusters"]["gpu-a"]["template_content_sha256"]
    rollback(node_release, component)
    final = node_release.environments[-1]
    assert final["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"] == "previous-template"
    assert final["GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"] == expected
    assert final["GPU_FAULT_INSTALLER_TEMPLATE_SHA256"] == TEMPLATE
    assert expected != TEMPLATE, "source identity was substituted for the content pin"


@pytest.mark.parametrize(
    "component", [ReleaseComponent.AGENT, ReleaseComponent.RECONCILER]
)
@pytest.mark.parametrize("version", [3, 4])
def test_rollback_requires_independently_captured_content_pin(
    node_release, component, version
):
    node_release.previous["release_manifest_schema_version"] = version
    node_release.previous["clusters"]["gpu-a"].pop("template_content_sha256")
    with pytest.raises(ReleaseError, match="captured trusted content pin"):
        rollback(node_release, component)
    assert node_release.environments == [], "rollback reached an installer sink"
    assert node_release.operations == [], "rollback started a node mutation"


@pytest.mark.parametrize(
    "component", [ReleaseComponent.AGENT, ReleaseComponent.RECONCILER]
)
def test_mutable_previous_template_drift_is_rejected_before_rollback(
    node_release, component
):
    text = node_release.template_document["data"]["job.yaml"]
    node_release.template_document["data"]["job.yaml"] = text.replace(
        "previous-template-program", "tampered-template-program"
    )
    with pytest.raises(ReleaseError, match="does not match"):
        rollback(node_release, component)
    assert node_release.environments == [], "drift reached preflight or restoration"
    assert node_release.operations == [], "drift was detected after a node mutation"


@pytest.mark.parametrize(
    "component", [ReleaseComponent.AGENT, ReleaseComponent.RECONCILER]
)
def test_pinned_previous_job_must_use_previous_dependency_image(
    node_release, component
):
    text = node_release.template_document["data"]["job.yaml"].replace(
        PREVIOUS_DEPENDENCY["reference"], CANDIDATE_DEPENDENCY["reference"]
    )
    node_release.template_document["data"]["job.yaml"] = text
    node_release.previous["clusters"]["gpu-a"]["template_content_sha256"] = (
        hashlib.sha256(text.encode()).hexdigest()
    )
    with pytest.raises(ReleaseError, match="dependency image does not match"):
        rollback(node_release, component)
    assert node_release.environments == []
    assert node_release.operations == []


def test_managed_rendering_clears_inherited_template_overrides(
    node_release, monkeypatch
):
    for name, value in (
        ("GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP", "inherited-template"),
        ("GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256", "8" * 64),
        ("GPU_FAULT_INSTALLER_TEMPLATE_PATH", "/untrusted/job.yaml"),
    ):
        monkeypatch.setenv(name, value)
    environment = build_reconciler_environment(
        node_release.release,
        node_release.target,
        wheel_cm="test-wheel",
        bundle_cm="shared-bundle",
        artifact_sha=ARTIFACT,
        config_digest=CONFIG,
        template_config_map=None,
    )
    assert environment["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"] == ""
    assert environment["GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"] == ""
    assert environment["GPU_FAULT_INSTALLER_TEMPLATE_PATH"] == ""


def test_explicit_override_cannot_inherit_its_content_pin(node_release, monkeypatch):
    monkeypatch.setenv("GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256", "8" * 64)
    with pytest.raises(ReleaseError, match="requires a trusted content pin"):
        RUNTIME.preflight_node_runtime(
            node_release.release,
            node_release.target,
            phase="upgrade",
            wheel_cm="test-wheel",
            bundle_cm="shared-bundle",
            artifact_sha=ARTIFACT,
            config_digest=CONFIG,
            template_config_map="explicit-template",
        )
    assert node_release.environments == []


def test_explicit_content_identity_reaches_preflight_and_rollout(node_release):
    RUNTIME.roll_node_runtime(
        node_release.release,
        node_release.target,
        phase="upgrade",
        wheel_cm="test-wheel",
        bundle_cm="shared-bundle",
        artifact_sha=ARTIFACT,
        config_digest=CONFIG,
        template_config_map="explicit-template",
        template_content_sha256="8" * 64,
    )
    assert len(node_release.environments) == 4
    assert {
        (
            env["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"],
            env["GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"],
        )
        for env in node_release.environments
    } == {("explicit-template", "8" * 64)}


def test_rollback_default_template_cannot_skip_the_previous_trust_anchor(node_release):
    node_release.previous["clusters"]["gpu-a"].pop("template_content_sha256")
    with pytest.raises(ReleaseError, match="captured trusted content pin"):
        RUNTIME.preflight_node_runtime(
            node_release.release,
            node_release.target,
            phase="rollback",
            wheel_cm="previous-wheel",
            bundle_cm="shared-bundle",
            artifact_sha=ARTIFACT,
            config_digest=CONFIG,
            bundle_sha256=BUNDLE,
            template_config_map=None,
        )
    assert node_release.environments == []
    assert node_release.operations == []
