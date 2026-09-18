from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from gpu_fault_release import regional_release_images as images
from gpu_fault_release import regional_release_legacy as legacy
from gpu_fault_release import regional_release_workflow_safety as workflow
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._cov95_release_support import NEW_IMAGE, OLD_IMAGE, ResourceRelease


@pytest.mark.parametrize("capture_gpu", [False, True])
def test_split_image_capture_preserves_separate_executor_identity(
    capture_gpu: bool,
) -> None:
    state = {
        "release_manifest_schema_version": 4,
        "runtime_image": OLD_IMAGE,
        "executor_image": NEW_IMAGE,
    }
    result = images.capture_previous_image_identity(
        state,
        cpu_images={"cpu": OLD_IMAGE},
        executor_images={"gpu-a": NEW_IMAGE} if capture_gpu else {},
        capture_gpu=capture_gpu,
    )
    assert result == (OLD_IMAGE, OLD_IMAGE, NEW_IMAGE)


@pytest.mark.parametrize(
    "state,cpu,executor,problem",
    [
        ({}, None, OLD_IMAGE, "cannot capture previous"),
        (
            {"adopted_live_runtime_image": NEW_IMAGE, "runtime_image": "mutable:tag"},
            NEW_IMAGE,
            NEW_IMAGE,
            "no immutable rollback",
        ),
        (
            {"adopted_live_runtime_image": NEW_IMAGE, "runtime_image": OLD_IMAGE},
            NEW_IMAGE,
            OLD_IMAGE,
            "Executor image drifted",
        ),
        (
            {"release_manifest_schema_version": 3},
            OLD_IMAGE,
            NEW_IMAGE,
            "CPU and Executor runtime images disagree",
        ),
        (
            {
                "release_manifest_schema_version": 4,
                "runtime_image": NEW_IMAGE,
                "executor_image": NEW_IMAGE,
            },
            OLD_IMAGE,
            NEW_IMAGE,
            "split runtime images drifted",
        ),
    ],
)
def test_image_capture_rejects_missing_or_drifted_runtime_identity(
    state: dict[str, Any], cpu: str | None, executor: str, problem: str
) -> None:
    with pytest.raises(ReleaseError, match=problem):
        images.capture_previous_image_identity(
            state,
            cpu_images={"cpu": cpu},
            executor_images={"gpu-a": executor},
            capture_gpu=True,
        )


def template_snapshot() -> dict[str, Any]:
    return {
        "clusters": {
            "gpu-a": {
                "template": "previous-template",
                "template_content_sha256": "a" * 64,
                "template_sha256": "b" * 64,
                "bundle_sha256": "c" * 64,
            }
        }
    }


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("nonobject", "no captured previous"),
        ("cluster", "no captured cluster binding"),
        ("template", "no captured cluster binding"),
        ("content", "no captured trusted content pin"),
        ("source", "no captured source identity"),
        ("bundle", "no captured bundle identity"),
    ],
)
def test_rollback_template_environment_requires_every_captured_identity(
    fault: str, problem: str
) -> None:
    previous: Any = template_snapshot()
    if fault == "nonobject":
        previous = None
    elif fault == "cluster":
        previous["clusters"] = {}
    else:
        field = {
            "template": "template",
            "content": "template_content_sha256",
            "source": "template_sha256",
            "bundle": "bundle_sha256",
        }[fault]
        previous["clusters"]["gpu-a"][field] = ""
    with pytest.raises(ReleaseError, match=problem):
        images.previous_node_template_environment(
            previous,
            cluster_id="gpu-a",
            template_config_map="previous-template",
            template_sha256=None,
        )


def test_valid_template_environment_uses_captured_content_not_source_digest() -> None:
    result = images.previous_node_template_environment(
        template_snapshot(),
        cluster_id="gpu-a",
        template_config_map=None,
        template_sha256="b" * 64,
    )
    assert result["GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"] == "a" * 64
    assert result["GPU_FAULT_INSTALLER_TEMPLATE_SHA256"] == "b" * 64
    assert result["GPU_FAULT_INSTALLER_BUNDLE_SHA256"] == "c" * 64
    with pytest.raises(ReleaseError, match="must be an object"):
        images.node_dependency_pin([])


def agent_identity() -> dict[str, Any]:
    return {
        "agent_protocol_version": 3,
        "node_action_key_version": 2,
        "agent_version": "previous-agent",
        "policy_version": "previous-policy",
        "artifact_sha256": "a" * 64,
        "compatibility_digest": "b" * 64,
        "config_digest": "c" * 64,
        "runtime_profile_version": "previous-profile",
        "node_ids": ["node-a"],
        "installer_bundle_sha256": None,
        "installer_template_sha256": None,
    }


@pytest.mark.parametrize(
    "field,value,problem",
    [
        ("agent_protocol_version", 4, "disagrees with pins"),
        ("installer_bundle_sha256", "d" * 64, "optional identity is incomplete"),
        ("agent_version", "", "version or policy is missing"),
        ("policy_version", "", "version or policy is missing"),
        ("node_ids", [], "node set is empty"),
    ],
)
def test_legacy_rollback_identity_does_not_fill_missing_or_conflicting_facts(
    field: str, value: Any, problem: str
) -> None:
    identity = agent_identity()
    identity[field] = value
    with pytest.raises(ReleaseError, match=problem):
        legacy.validate_rollback_agent_identity(
            "gpu-a",
            identity,
            metadata={
                "required-agent-protocol-version": "3",
                "required-node-action-key-version": "2",
            },
            artifact="a" * 64,
            compatibility="b" * 64,
            config_digest="c" * 64,
            runtime_profile_version="previous-profile",
        )


def test_legacy_controller_and_request_keep_only_explicit_identity_fields() -> None:
    with pytest.raises(ReleaseError, match="snapshot is empty"):
        legacy.rollback_controller_config({})
    request = {"desired_agent_protocol_version": 3}
    legacy.apply_fleet_request_identity(
        request, {"agent_protocol_version": None, "agent_version": "previous-agent"}
    )
    assert request == {
        "desired_agent_protocol_version": 3,
        "desired_agent_version": "previous-agent",
    }
    identity = agent_identity()
    before = copy.deepcopy(identity)
    config = legacy.rollback_controller_config({"gpu-a": identity})
    assert config["GPU_FAULT_REQUIRED_POLICY_VERSION"] == "previous-policy"
    assert identity == before


@pytest.mark.parametrize(
    "raw,problem",
    [
        ("{", "invalid evidence"),
        ("[]", "non-object evidence"),
        ('{"blocker_count":1}', "active destructive workflows"),
    ],
)
def test_workflow_safety_refuses_bad_evidence_or_active_blockers(
    monkeypatch: pytest.MonkeyPatch, raw: str, problem: str
) -> None:
    monkeypatch.setattr(
        workflow, "exec_cpu_ingress_probe", lambda *_args, **_kwargs: raw
    )
    with pytest.raises(ReleaseError, match=problem):
        workflow.workflow_safety_snapshot(ResourceRelease())
    monkeypatch.setattr(
        workflow,
        "exec_cpu_ingress_probe",
        lambda *_args, **_kwargs: json.dumps({"blocker_count": 0}),
    )
    assert workflow.workflow_safety_snapshot(ResourceRelease()) == {"blocker_count": 0}
