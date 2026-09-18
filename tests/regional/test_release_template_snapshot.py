from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_state import capture_gpu_cluster_snapshot
from tests.deploy.test_installer_template_identity import (
    DEPENDENCY_IMAGE,
    INVENTORY_SHA,
    template_job,
)

BUNDLE_SHA = "4" * 64
SOURCE_SHA = "5" * 64
RUNTIME_IMAGE = "registry.example/runtime@sha256:" + "6" * 64


@pytest.fixture
def snapshot_source() -> SimpleNamespace:
    text = yaml.safe_dump(template_job())
    content_pin = hashlib.sha256(text.encode()).hexdigest()
    deployment: dict[str, Any] = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "reconciler",
                            "image": RUNTIME_IMAGE,
                            "env": [
                                {"name": name, "value": value}
                                for name, value in {
                                    "GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP": "old-template",
                                    "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256": content_pin,
                                    "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": SOURCE_SHA,
                                    "GPU_FAULT_INSTALLER_BUNDLE_SHA256": BUNDLE_SHA,
                                }.items()
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "installer-template",
                            "configMap": {"name": "old-template"},
                        }
                    ],
                }
            }
        }
    }
    source = SimpleNamespace(
        deployment=deployment,
        template={"data": {"job.yaml": text}},
        content_pin=content_pin,
        bundle_sha=BUNDLE_SHA,
        reads=[],
    )

    def read(arguments: list[str]) -> dict[str, Any]:
        source.reads.append(arguments)
        kind, name = arguments[-2:]
        if kind == "configmap":
            assert name == "old-template", "snapshot selected a different ConfigMap"
            return copy.deepcopy(source.template)
        if kind == "deployment":
            return copy.deepcopy(source.deployment)
        assert kind == "daemonset"
        return {
            "spec": {"template": {"spec": {"containers": [{"image": RUNTIME_IMAGE}]}}}
        }

    source.release = SimpleNamespace(
        config=SimpleNamespace(
            namespace="gpu-fault-system", bundle=Path("bundle.tar.gz")
        ),
        state={
            "release_manifest_schema_version": 4,
            "node_dependencies": {
                "reference": DEPENDENCY_IMAGE,
                "wheelhouse_sha256": INVENTORY_SHA,
            },
        },
        _gpu=lambda _target, *arguments: list(arguments),
        _get_json=read,
        _deployment_wheel=lambda *_args: "old-wheel",
        _config_map_binary_key=lambda *_args: "artifact",
        _config_map_sha=lambda *_args: source.bundle_sha,
    )
    source.target = SimpleNamespace(cluster_id="test-cluster")
    return source


def test_snapshot_captures_the_prior_reconciler_content_pin(
    snapshot_source: SimpleNamespace,
) -> None:
    snapshot, _images, _installer = capture_gpu_cluster_snapshot(
        snapshot_source.release, snapshot_source.target
    )
    assert snapshot["template_content_sha256"] == snapshot_source.content_pin
    assert snapshot["template_sha256"] == SOURCE_SHA
    assert snapshot["template_content_sha256"] != snapshot["template_sha256"]
    assert snapshot["bundle"] == "old-bundle"
    assert snapshot["bundle_sha256"] == BUNDLE_SHA


def test_snapshot_rejects_configmap_drift_instead_of_blessing_current_bytes(
    snapshot_source: SimpleNamespace,
) -> None:
    snapshot_source.template["data"]["job.yaml"] += "\n# mutated after deployment\n"
    with pytest.raises(ReleaseError, match="does not match"):
        capture_gpu_cluster_snapshot(snapshot_source.release, snapshot_source.target)


def test_snapshot_cannot_take_the_pin_from_a_sidecar(
    snapshot_source: SimpleNamespace,
) -> None:
    text = snapshot_source.template["data"]["job.yaml"] + "\n# drift\n"
    snapshot_source.template["data"]["job.yaml"] = text
    containers = snapshot_source.deployment["spec"]["template"]["spec"]["containers"]
    containers.insert(
        0,
        {
            "name": "unrelated-container",
            "env": [
                {
                    "name": "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256",
                    "value": hashlib.sha256(text.encode()).hexdigest(),
                }
            ],
        },
    )
    with pytest.raises(ReleaseError, match="does not match"):
        capture_gpu_cluster_snapshot(snapshot_source.release, snapshot_source.target)


@pytest.mark.parametrize("version", [3, 4])
def test_snapshot_fails_closed_when_prior_reconciler_has_no_content_pin(
    snapshot_source: SimpleNamespace, version: int
) -> None:
    snapshot_source.release.state["release_manifest_schema_version"] = version
    container = snapshot_source.deployment["spec"]["template"]["spec"]["containers"][0]
    container["env"] = [
        item
        for item in container["env"]
        if item["name"] != "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"
    ]
    with pytest.raises(
        ReleaseError, match="GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256 is required"
    ):
        capture_gpu_cluster_snapshot(snapshot_source.release, snapshot_source.target)


def test_snapshot_does_not_fabricate_a_legacy_source_identity(
    snapshot_source: SimpleNamespace,
) -> None:
    container = snapshot_source.deployment["spec"]["template"]["spec"]["containers"][0]
    container["env"] = [
        item
        for item in container["env"]
        if item["name"] != "GPU_FAULT_INSTALLER_TEMPLATE_SHA256"
    ]
    snapshot, _images, _installer = capture_gpu_cluster_snapshot(
        snapshot_source.release, snapshot_source.target
    )
    assert snapshot["template_sha256"] is None
    assert snapshot["template_content_sha256"] == snapshot_source.content_pin


@pytest.mark.parametrize("field", ["reference", "wheelhouse_sha256"])
def test_snapshot_compares_the_actual_job_to_previous_release_dependencies(
    snapshot_source: SimpleNamespace, field: str
) -> None:
    snapshot_source.release.state["node_dependencies"][field] = (
        "registry.example/candidate@sha256:" + "7" * 64
        if field == "reference"
        else "8" * 64
    )
    with pytest.raises(ReleaseError, match="node dependency .* does not match"):
        capture_gpu_cluster_snapshot(snapshot_source.release, snapshot_source.target)


def test_snapshot_rejects_a_template_reference_that_disagrees_with_the_volume(
    snapshot_source: SimpleNamespace,
) -> None:
    container = snapshot_source.deployment["spec"]["template"]["spec"]["containers"][0]
    reference = next(
        item
        for item in container["env"]
        if item["name"] == "GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"
    )
    reference["value"] = "another-template"
    with pytest.raises(ReleaseError, match="reference disagrees with its volume"):
        capture_gpu_cluster_snapshot(snapshot_source.release, snapshot_source.target)
    assert len(snapshot_source.reads) == 1, "an unbound template was fetched"


def test_snapshot_rejects_bundle_content_drift(
    snapshot_source: SimpleNamespace,
) -> None:
    snapshot_source.bundle_sha = "9" * 64
    with pytest.raises(
        ReleaseError, match="bundle disagrees with Reconciler trusted pin"
    ):
        capture_gpu_cluster_snapshot(snapshot_source.release, snapshot_source.target)


@pytest.mark.parametrize("version", [False, 0, "", 5])
def test_unknown_schema_cannot_downgrade_dependency_identity(
    snapshot_source: SimpleNamespace, version: object
) -> None:
    snapshot_source.release.state["release_manifest_schema_version"] = version
    with pytest.raises(ReleaseError, match="previous release manifest schema version"):
        capture_gpu_cluster_snapshot(snapshot_source.release, snapshot_source.target)


def test_snapshot_after_rollback_uses_restored_dependency_identity(
    snapshot_source: SimpleNamespace,
) -> None:
    previous = copy.deepcopy(snapshot_source.release.state)
    snapshot_source.release.state.update(
        phase="rolled-back",
        rollback_result={"status": "PASSED"},
        previous=previous,
        node_dependencies={
            "reference": "registry.example/failed-candidate@sha256:" + "8" * 64,
            "wheelhouse_sha256": "9" * 64,
        },
    )
    snapshot, _images, _installer = capture_gpu_cluster_snapshot(
        snapshot_source.release, snapshot_source.target
    )
    assert snapshot["template_content_sha256"] == snapshot_source.content_pin
