from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.node_installer_rendering import configure_node_dependencies
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseComponent
from gpu_fault_release.regional_release_images import previous_executor_image
from gpu_fault_release.regional_release_rendering import build_reconciler_environment
from gpu_fault_release.regional_release_rollback_target import rollback_target
from tests.deploy.test_installer_template_identity import template_job


def test_legacy_snapshot_explicitly_reuses_shared_image() -> None:
    image = "registry.example/old@sha256:" + "a" * 64
    assert previous_executor_image({"runtime_image": image}) == image


def test_split_snapshot_selects_previous_executor_not_cpu() -> None:
    cpu = "registry.example/cpu@sha256:" + "a" * 64
    executor = "registry.example/executor@sha256:" + "b" * 64
    assert (
        previous_executor_image(
            {
                "release_manifest_schema_version": 4,
                "runtime_image": cpu,
                "executor_image": executor,
            }
        )
        == executor
    )


@pytest.mark.parametrize("executor", [None, "", "registry.example/runtime:latest"])
def test_split_snapshot_missing_executor_never_guesses_cpu_or_candidate(
    executor: str | None,
) -> None:
    with pytest.raises(ReleaseError, match="Executor image"):
        previous_executor_image(
            {
                "release_manifest_schema_version": 4,
                "runtime_image": "registry.example/cpu@sha256:" + "a" * 64,
                "executor_image": executor,
            }
        )


@pytest.mark.parametrize("same_wheelhouse", [True, False])
def test_reconciler_rollback_selects_previous_image_for_unchanged_bundle(
    same_wheelhouse: bool,
) -> None:
    previous_dependency = {
        "reference": "registry.example/node-dependencies@sha256:" + "a" * 64,
        "wheelhouse_sha256": "b" * 64,
    }
    candidate_dependency = {
        "reference": "registry.example/node-dependencies@sha256:" + "c" * 64,
        "wheelhouse_sha256": (
            previous_dependency["wheelhouse_sha256"] if same_wheelhouse else "d" * 64
        ),
    }
    bundle_sha = "e" * 64
    previous_installer = "registry.example/installer@sha256:" + "7" * 64
    job = template_job(offline=False)
    pod = job["spec"]["template"]["spec"]
    pod["containers"][0]["image"] = previous_installer
    pod["volumes"][-1]["configMap"]["name"] = "shared-bundle"
    configure_node_dependencies(
        job, previous_dependency["reference"], previous_dependency["wheelhouse_sha256"]
    )
    template_text = yaml.safe_dump(job)
    content_pin = hashlib.sha256(template_text.encode()).hexdigest()
    previous: dict[str, Any] = {
        "release_manifest_schema_version": 4,
        "node_dependencies": previous_dependency,
        "clusters": {
            "gpu-a": {
                "reconciler_wheel": "previous-wheel",
                "bundle": "shared-bundle",
                "bundle_sha256": bundle_sha,
                "template": "previous-template",
                "template_sha256": "f" * 64,
                "template_content_sha256": content_pin,
            }
        },
    }
    environments: list[dict[str, str]] = []
    reads: list[list[str]] = []

    def read_template(arguments: list[str]) -> dict[str, Any]:
        assert arguments == [
            "-n",
            "gpu-fault-system",
            "get",
            "configmap",
            "previous-template",
        ]
        assert environments == [], "the trusted Job must be checked before deployment"
        reads.append(arguments)
        return {"data": {"job.yaml": template_text}}

    release: Any = SimpleNamespace(
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            executor_wheel=Path("executor.whl"),
            upgrade_max_unavailable=1,
            release_manifest_schema_version=4,
            release_delivery_identity={
                "images": {"node_dependencies": candidate_dependency}
            },
        ),
        bundle_cm="shared-bundle",
        bundle_sha=bundle_sha,
        runner=SimpleNamespace(dry_run=False),
        state={"previous": previous},
        _gpu=lambda _target, *arguments: list(arguments),
        _get_json=read_template,
        _deploy_reconciler=lambda target, **kwargs: environments.append(
            build_reconciler_environment(release, target, **kwargs)
        ),
    )
    target: Any = SimpleNamespace(
        cluster_id="gpu-a",
        context="gpu-a-context",
        hyperpod_cluster_name="hp-gpu-a",
        fleet_master_file=None,
    )
    rollback_target(
        release,
        target,
        previous=previous,
        artifact="1" * 64,
        config_digest="2" * 64,
        runtime_profile_version="previous-profile",
        executor_artifact="3" * 64,
        executor_compatibility="4" * 64,
        node_compatibility="5" * 64,
        runtime_image="registry.example/executor@sha256:" + "6" * 64,
        node_installer_image=previous_installer,
        components=frozenset({ReleaseComponent.RECONCILER}),
    )

    assert len(reads) == 1
    assert len(environments) == 1
    assert (
        environments[0]["GPU_FAULT_NODE_DEPENDENCY_IMAGE"]
        == previous_dependency["reference"]
    )
    assert (
        environments[0]["GPU_FAULT_NODE_WHEELHOUSE_SHA256"]
        == previous_dependency["wheelhouse_sha256"]
    )
    assert environments[0]["GPU_FAULT_RUNTIME_IMAGE"] == (
        "registry.example/executor@sha256:" + "6" * 64
    )
    assert environments[0]["GPU_FAULT_NODE_INSTALLER_IMAGE"] == previous_installer
    assert environments[0]["GPU_FAULT_INSTALLER_CONFIG_MAP"] == "shared-bundle"
    assert environments[0]["GPU_FAULT_INSTALLER_BUNDLE_SHA256"] == bundle_sha
    assert environments[0]["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"] == (
        "previous-template"
    )
    assert environments[0]["GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"] == content_pin
    assert environments[0]["GPU_FAULT_INSTALLER_TEMPLATE_SHA256"] == "f" * 64


def test_rollback_hands_the_previous_pin_window_to_the_executor_rollout() -> None:
    """``rollback_target`` restores the previous executor under the previous
    release's pin window; without a window in the snapshot the live check runs
    as before."""

    applied: list[dict[str, Any]] = []
    release: Any = SimpleNamespace(
        config=SimpleNamespace(
            namespace="gpu-fault-system", executor_wheel=Path("executor.whl")
        ),
        _gpu=lambda _target, *arguments: list(arguments),
        _apply_gpu_deployments=lambda target, wheel, **kwargs: applied.append(
            {"target": target.cluster_id, "wheel": wheel, **kwargs}
        ),
    )
    target: Any = SimpleNamespace(cluster_id="gpu-a", context="gpu-a-context")
    window = {
        "required-regional-executor-artifact-sha256": "3" * 64,
        "required-regional-executor-compatibility-digest": "4" * 64,
        "required-regional-executor-protocol-version": "4",
    }
    for metadata in (window, None):
        previous: dict[str, Any] = {
            "clusters": {"gpu-a": {"wheel": "previous-wheel", "wheel_key": "e.whl"}}
        }
        if metadata is not None:
            previous["metadata"] = metadata
        rollback_target(
            release,
            target,
            previous=previous,
            artifact="1" * 64,
            config_digest="2" * 64,
            runtime_profile_version="previous-profile",
            executor_artifact="3" * 64,
            executor_compatibility="4" * 64,
            node_compatibility="5" * 64,
            runtime_image="registry.example/executor@sha256:" + "6" * 64,
            node_installer_image="registry.example/installer@sha256:" + "7" * 64,
            components=frozenset({ReleaseComponent.EXECUTOR}),
        )
    assert [item["pin_metadata"] for item in applied] == [window, None]
    assert all(item["executor_artifact_sha"] == "3" * 64 for item in applied), (
        "the rollback must restore the previous executor artifact"
    )
    assert all(item["wheel"] == "previous-wheel" for item in applied), (
        "the rollback must restore the previous executor wheel"
    )
