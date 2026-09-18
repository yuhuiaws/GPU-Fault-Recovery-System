from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_resume_validation as resume
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseComponent as Component
from gpu_fault_release.regional_release_diff import ReleaseExecutionPlan
from tests.regional._cov95_release_support import (
    ResourceRelease,
    mixed_probe_result,
    previous_snapshot,
)


def checkpoint(release: ResourceRelease, state: str = "CONVERGED") -> dict[str, Any]:
    return {
        "cluster_ids": [target.cluster_id for target in release.config.clusters],
        "cluster_registry_digest": release.cluster_registry_digest,
        "cluster_attempts": {
            target.cluster_id: {"state": state} for target in release.config.clusters
        },
        "completed_phases": [],
    }


def validate(
    release: ResourceRelease,
    loaded: dict[str, Any],
    *components: Component,
    previous: dict[str, Any] | None = None,
) -> None:
    resume.validate_resume_checkpoint(
        release,
        loaded=loaded,
        previous=previous_snapshot(release) if previous is None else previous,
        plan=ReleaseExecutionPlan(nodes=components),
    )


@pytest.mark.parametrize(
    "components",
    [
        (Component.VERIFY,),
        (Component.EXECUTOR,),
        (Component.WATCHER,),
        (Component.COLLECTOR,),
        (Component.RECONCILER,),
        (Component.AGENT,),
        (Component.DCGM,),
        (Component.ENDPOINT,),
        tuple(resume.GPU_COMPONENTS),
    ],
)
def test_converged_checkpoint_rechecks_exact_planned_components(
    components: tuple[Component, ...],
) -> None:
    release = ResourceRelease()
    validate(release, checkpoint(release), *components)
    assert release.endpoint_checks == (
        ["gpu-a"] if Component.ENDPOINT in components else []
    )
    assert bool(release.heartbeat_checks) is (Component.AGENT in components)
    assert release.runner.calls == [], "resume checks must not mutate or run verifiers"


@pytest.mark.parametrize(
    "change,problem",
    [
        ({"cluster_ids": ["gpu-b"]}, "membership drifted"),
        ({"cluster_registry_digest": "foreign"}, "identity digest drifted"),
        ({"failed_cluster_ids": ["gpu-a"]}, "categories overlap"),
        ({"paused_cluster_ids": ["gpu-b"]}, "unknown cluster"),
    ],
)
def test_checkpoint_membership_and_category_errors_block_all_resource_reads(
    change: dict[str, Any], problem: str
) -> None:
    release = ResourceRelease()
    loaded = {**checkpoint(release), **change}
    with pytest.raises(ReleaseError, match=problem):
        validate(release, loaded, Component.EXECUTOR)
    assert release.reads == []


@pytest.mark.parametrize(
    "identity,problem", [(None, "identity is missing"), ({}, "node set is empty")]
)
def test_gpu_resume_needs_previous_node_identity(identity: Any, problem: str) -> None:
    release = ResourceRelease()
    previous = previous_snapshot(release)
    previous["agent_identities"]["gpu-a"] = identity
    with pytest.raises(ReleaseError, match=problem):
        validate(release, checkpoint(release), Component.AGENT, previous=previous)
    validate(release, checkpoint(release), Component.CPU_FINALIZE, previous=previous)


@pytest.mark.parametrize(
    "field", ["required-agent-artifact-sha256", "compatible-agent-artifact-sha256s"]
)
def test_finalized_checkpoint_rejects_required_or_compatible_pin_drift(
    field: str,
) -> None:
    release = ResourceRelease()
    previous = previous_snapshot(release)
    release.metadata[field] = "foreign-pin"
    loaded = checkpoint(release)
    loaded["completed_phases"] = ["cpu-finalized"]
    with pytest.raises(ReleaseError, match="finalized pin drifted"):
        validate(release, loaded, Component.VERIFY, previous=previous)
    assert len(release.reads) == 1


def test_finalized_checkpoint_accepts_complete_pin_promotion() -> None:
    release = ResourceRelease()
    loaded = checkpoint(release)
    loaded["completed_phases"] = ["cpu-finalized"]
    validate(release, loaded, Component.EXECUTOR)


@pytest.mark.parametrize(
    "fault", ["wheel", "scaled-down", "not-ready", "image", "missing-container"]
)
def test_converged_deployment_drift_fails_before_heartbeats(fault: str) -> None:
    release = ResourceRelease()
    item = release.documents[("gpu-a", "deployment", inventory.GPU_EXECUTOR_DEPLOYMENT)]
    pod = item["spec"]["template"]["spec"]
    if fault == "wheel":
        pod["volumes"][0]["configMap"]["name"] = "foreign-wheel"
    elif fault == "scaled-down":
        item["spec"]["replicas"] = 0
    elif fault == "not-ready":
        item["status"]["readyReplicas"] = 0
    elif fault == "image":
        pod["containers"][0]["image"] = "foreign-image"
    else:
        pod["containers"] = []
    with pytest.raises(
        ReleaseError, match="wheel drifted|not fully Ready|image drifted"
    ):
        validate(release, checkpoint(release), Component.EXECUTOR, Component.AGENT)
    assert release.heartbeat_checks == []


@pytest.mark.parametrize(
    "field",
    ["GPU_FAULT_INSTALLER_BUNDLE_SHA256", "GPU_FAULT_INSTALLER_TEMPLATE_SHA256"],
)
def test_reconciler_snapshot_checks_both_bundle_and_template(field: str) -> None:
    release = ResourceRelease()
    pod = release.documents[
        ("gpu-a", "deployment", inventory.GPU_RECONCILER_DEPLOYMENT)
    ]["spec"]["template"]["spec"]
    for entry in pod["containers"][0]["env"]:
        if entry["name"] == field:
            entry["value"] = "foreign"
    with pytest.raises(ReleaseError, match="Reconciler identity drifted"):
        validate(release, checkpoint(release), Component.RECONCILER)


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("dcgm", "DCGM image drifted"),
        ("annotations", "node annotations drifted"),
        ("heartbeat", "heartbeats drifted"),
        ("read", "transport unavailable"),
    ],
)
def test_converged_candidate_rejects_drift_and_transport_failure(
    fault: str, problem: str
) -> None:
    release = ResourceRelease()
    if fault == "dcgm":
        release.documents[("gpu-a", "daemonset", "gpu-fault-dcgm-exporter")]["spec"][
            "template"
        ]["spec"]["containers"][0]["image"] = "foreign"
    elif fault == "annotations":
        release.documents[("gpu-a", "nodes", "")]["items"][0]["metadata"][
            "annotations"
        ]["gpu-fault.io/installer-node-uid"] = "old-node"
    elif fault == "heartbeat":
        release.heartbeat_ready = False
    else:
        release.documents[
            ("gpu-a", "deployment", inventory.GPU_EXECUTOR_DEPLOYMENT)
        ] = ReleaseError("transport unavailable")
    with pytest.raises(ReleaseError, match=problem):
        validate(release, checkpoint(release), *tuple(resume.GPU_COMPONENTS))


@pytest.mark.parametrize("state", ["PENDING", "FAILED", "PAUSED", "unknown"])
def test_unmodified_cluster_is_checked_against_previous_release(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    release = ResourceRelease()
    calls = []

    def rollback(
        release_arg: Any,
        previous: dict[str, Any],
        image: str,
        target: Any,
        **kwargs: Any,
    ) -> None:
        calls.append((target.cluster_id, image, kwargs))

    monkeypatch.setattr(resume, "validate_gpu_rollback_target", rollback)
    validate(release, checkpoint(release, state), Component.EXECUTOR)
    assert calls == [
        (
            "gpu-a",
            previous_snapshot(release)["runtime_image"],
            {"components": frozenset({Component.EXECUTOR}), "run_verifier": False},
        )
    ]
    assert release.reads == [("cpu", "configmap", "gpu-fault-release-metadata")]


@pytest.mark.parametrize("state", ["PAUSED", "FAILED", "IN_PROGRESS"])
@pytest.mark.parametrize("agent_status", ["STARTED", "FAILED"])
def test_partial_checkpoint_validates_completed_previous_and_inflight_components(
    monkeypatch: pytest.MonkeyPatch, state: str, agent_status: str
) -> None:
    release = ResourceRelease()
    loaded = checkpoint(release, state)
    loaded["component_progress"] = {
        "clusters": {
            "gpu-a": {
                Component.EXECUTOR.value: {"status": "COMPLETED"},
                Component.AGENT.value: {"status": agent_status},
                Component.CPU_STAGE.value: {"status": "COMPLETED"},
                "unknown-component": {"status": "COMPLETED"},
                "malformed": "COMPLETED",
            }
        }
    }
    previous = previous_snapshot(release)
    previous["agent_identities"]["gpu-a"]["installer_bundle_sha256"] = None
    previous["agent_identities"]["gpu-a"]["installer_template_sha256"] = None
    previous["agent_identities"]["gpu-a"]["compatibility_digest"] = None
    calls = []
    probes = []

    def rollback(
        _release: Any, _previous: Any, _image: str, target: Any, **kwargs: Any
    ) -> None:
        calls.append((target.cluster_id, kwargs))

    def probe(release_arg: Any, **kwargs: Any) -> str:
        probes.append(json.loads(kwargs["input_text"]))
        return json.dumps(mixed_probe_result(release_arg))

    monkeypatch.setattr(resume, "validate_gpu_rollback_target", rollback)
    monkeypatch.setattr(resume, "exec_cpu_ingress_probe", probe)
    validate(
        release,
        loaded,
        Component.EXECUTOR,
        Component.WATCHER,
        Component.AGENT,
        previous=previous,
    )
    assert calls == [
        ("gpu-a", {"components": frozenset({Component.WATCHER}), "run_verifier": False})
    ]
    assert probes[0]["node_ids"] == ["node-a"]
    assert probes[0]["deployment_id"] == "fleet-gpu-a"
    assert probes[0]["allowed_identities"][0]["bundle"] is None
    assert probes[0]["allowed_identities"][0]["compatibility"] == release.node_wheel_sha
    assert probes[0]["allowed_identities"][1]["artifact"] == release.node_wheel_sha
    assert release.heartbeat_checks == [], (
        "an in-flight Agent is checked by the mixed-state probe"
    )


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("blocked-agent", "outside the transaction compatibility window"),
        ("missing-deployment", "FleetDeployment is missing"),
        ("foreign-cluster", "contract drifted"),
        ("foreign-nodes", "contract drifted"),
        ("foreign-artifact", "contract drifted"),
        ("foreign-config", "contract drifted"),
        ("foreign-bundle", "contract drifted"),
        ("foreign-template", "contract drifted"),
        ("unknown-status", "invalid node states"),
        ("installing-paused", "non-terminal wave"),
        ("failed-paused", "non-terminal wave"),
    ],
)
def test_mixed_checkpoint_rejects_unknown_or_unbound_observations(
    monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    release = ResourceRelease()
    loaded = checkpoint(release, "PAUSED")
    loaded["component_progress"] = {
        "clusters": {"gpu-a": {Component.AGENT.value: {"status": "STARTED"}}}
    }
    response = mixed_probe_result(release)
    if fault == "blocked-agent":
        response["agent_blocker_count"] = 1
    elif fault == "missing-deployment":
        response["deployment"] = None
    elif fault == "foreign-cluster":
        response["deployment"]["cluster_id"] = "other"
    elif fault == "foreign-nodes":
        response["deployment"]["nodes"][0]["node_id"] = "other"
    elif fault.startswith("foreign-"):
        field = {
            "foreign-artifact": "desired_artifact_sha256",
            "foreign-config": "desired_config_digest",
            "foreign-bundle": "desired_bundle_sha256",
            "foreign-template": "desired_template_sha256",
        }[fault]
        response["deployment"][field] = "other"
    else:
        response["deployment"]["nodes"][0]["status"] = {
            "unknown-status": "UNKNOWN",
            "installing-paused": "INSTALLING",
            "failed-paused": "FAILED",
        }[fault]
    monkeypatch.setattr(
        resume, "exec_cpu_ingress_probe", lambda *_args, **_kwargs: json.dumps(response)
    )
    before = copy.deepcopy(loaded)
    with pytest.raises(ReleaseError, match=problem):
        validate(release, loaded, Component.AGENT)
    assert loaded == before, "rejected live evidence must not rewrite the checkpoint"


def test_only_completed_component_in_partial_cluster_needs_no_previous_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = ResourceRelease()
    loaded = checkpoint(release, "FAILED")
    loaded["component_progress"] = {
        "clusters": {"gpu-a": {Component.EXECUTOR.value: {"status": "COMPLETED"}}}
    }

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("no incomplete or previous component remains to probe")

    monkeypatch.setattr(resume, "validate_gpu_rollback_target", unexpected)
    monkeypatch.setattr(resume, "exec_cpu_ingress_probe", unexpected)
    validate(release, loaded, Component.EXECUTOR)
    assert ("gpu-a", "deployment", inventory.GPU_EXECUTOR_DEPLOYMENT) in release.reads
