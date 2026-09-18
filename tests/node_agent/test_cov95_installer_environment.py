from __future__ import annotations

import copy
import json

import pytest

from gpu_fault.container_env_snapshot import (
    ContainerEnvSnapshotError,
    container_env_differences,
    load_container_env_snapshot,
    normalised_container_env,
    pod_container_env,
    validate_container_env_snapshot,
)


def environment():
    return {
        "env": [
            {"name": "Z", "value": "last"},
            {
                "name": "A",
                "valueFrom": {"configMapKeyRef": {"name": "local", "key": "value"}},
            },
        ],
        "envFrom": [
            {"configMapRef": {"name": "base"}},
            {"secretRef": {"name": "local-ref"}},
        ],
    }


@pytest.mark.parametrize(
    "snapshot,message",
    [
        (None, "non-empty Deployment"),
        ({}, "non-empty Deployment"),
        ({1: {"app": {}}}, "Deployment entries"),
        ({"deployment": []}, "Deployment entries"),
        ({"deployment": {}}, "lists no containers"),
        ({"deployment": {1: {"env": [], "envFrom": []}}}, "exactly env"),
        ({"deployment": {"app": []}}, "exactly env"),
        ({"deployment": {"app": {"env": []}}}, "exactly env"),
        ({"deployment": {"app": {"env": {}, "envFrom": []}}}, "exactly env"),
        ({"deployment": {"app": {"env": [], "envFrom": {}}}}, "exactly env"),
    ],
)
def test_environment_snapshot_rejects_unknown_container_shape(snapshot, message):
    with pytest.raises(ContainerEnvSnapshotError, match=message):
        validate_container_env_snapshot(snapshot)


@pytest.mark.parametrize(
    "entry,message",
    [
        (None, "without a name"),
        ({"name": None, "value": "x"}, "without a name"),
        ({"name": "A"}, "exactly one"),
        ({"name": "A", "value": "x", "valueFrom": {}}, "exactly one"),
        (
            {"name": "GPU_FAULT_TEST_TOKEN", "value": "unit-placeholder"},
            "sensitive env",
        ),
    ],
)
def test_environment_snapshot_refuses_ambiguous_or_literal_sensitive_values(
    entry, message
):
    snapshot = {"deployment": {"app": {"env": [entry], "envFrom": []}}}
    with pytest.raises(ContainerEnvSnapshotError, match=message):
        validate_container_env_snapshot(snapshot)


def test_environment_from_sources_must_be_objects() -> None:
    with pytest.raises(ContainerEnvSnapshotError, match="malformed envFrom"):
        validate_container_env_snapshot(
            {"deployment": {"app": {"env": [], "envFrom": ["unknown"]}}}
        )


def test_file_loader_validates_parsed_data_and_reports_read_errors(tmp_path) -> None:
    snapshot = {"gpu-fault-api-ha": {"app": environment()}}
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    assert load_container_env_snapshot(path) == snapshot
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ContainerEnvSnapshotError, match="cannot read"):
        load_container_env_snapshot(path)
    with pytest.raises(ContainerEnvSnapshotError, match="cannot read"):
        load_container_env_snapshot(tmp_path / "missing.json")


def test_capture_includes_init_containers_and_keeps_reference_order() -> None:
    spec = environment()
    deployment = {
        "spec": {
            "template": {
                "spec": {
                    "initContainers": [{"name": "init"}],
                    "containers": [{"name": "app", **spec}],
                }
            }
        }
    }
    captured = pod_container_env(deployment)
    assert captured == {"init": {"env": [], "envFrom": []}, "app": spec}
    normalized = normalised_container_env(spec)
    assert [item["name"] for item in normalized["env"]] == ["A", "Z"]
    assert normalized["envFrom"] == spec["envFrom"]
    assert [item["name"] for item in spec["env"]] == ["Z", "A"]
    assert pod_container_env({}) == {}


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("missing-deployment", "snapshot names Deployment"),
        ("extra-deployment", "not in the snapshot"),
        ("missing-container", "has no live container"),
        ("extra-container", "container extra is not in the snapshot"),
        ("missing-env", "env Z is in the snapshot but not live"),
        ("extra-env", "env B is live but not in the snapshot"),
        ("changed-env", "env Z differs from the snapshot"),
        ("source-order", "envFrom[0] differs"),
    ],
)
def test_snapshot_comparison_reports_both_direction_and_content_drift(
    mutation, message
):
    snapshot = {"gpu-fault-api-ha": {"app": environment()}}
    actual = copy.deepcopy(snapshot)
    spec = actual["gpu-fault-api-ha"]["app"]
    if mutation == "missing-deployment":
        actual.clear()
    elif mutation == "extra-deployment":
        actual["extra"] = {}
    elif mutation == "missing-container":
        actual["gpu-fault-api-ha"].clear()
    elif mutation == "extra-container":
        actual["gpu-fault-api-ha"]["extra"] = {"env": [], "envFrom": []}
    elif mutation == "missing-env":
        spec["env"].pop(0)
    elif mutation == "extra-env":
        spec["env"].append({"name": "B", "value": "new"})
    elif mutation == "changed-env":
        spec["env"][0]["value"] = "changed"
    else:
        spec["envFrom"].reverse()
    problems = container_env_differences(snapshot, actual, actual_label="live")
    assert len(problems) == 1
    assert message in problems[0]


def test_env_order_is_ignored_but_duplicate_entries_remain_visible() -> None:
    snapshot = {"gpu-fault-api-ha": {"app": environment()}}
    actual = copy.deepcopy(snapshot)
    actual["gpu-fault-api-ha"]["app"]["env"].reverse()
    assert container_env_differences(snapshot, actual, actual_label="live") == []
    actual["gpu-fault-api-ha"]["app"]["env"].append({"name": "Z", "value": "last"})
    assert container_env_differences(snapshot, actual, actual_label="live") == [
        "gpu-fault-api-ha/app env Z is live but not in the snapshot"
    ]
