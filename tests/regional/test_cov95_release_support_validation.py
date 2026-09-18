from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault_release import regional_manifest_snapshot as manifests
from gpu_fault_release import regional_runtime_profile as profiles
from gpu_fault_release import regional_schema_change as schema
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._cov95_release_support import ResourceRelease, json_response


@pytest.mark.parametrize("source", ["[]", "null", "example"])
def test_runtime_profile_rendering_and_policy_digest_require_mapping(
    tmp_path: Path, source: str
) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(source)
    release = ResourceRelease()
    release.config.runtime_profile_source = path
    with pytest.raises(ReleaseError, match="one mapping"):
        profiles.render_runtime_profile_payload(release.config)
    with pytest.raises(ReleaseError, match="one mapping"):
        profiles.runtime_profile_policy_digest(path)


def test_profile_digest_ignores_binding_and_list_order_but_not_policy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profile.yaml"
    document = {
        "cluster_id": "a",
        "profile_version": "v1",
        "claims": [{"name": "b"}, {"name": "a"}],
        "environment": "hyperpod-eks",
    }
    path.write_text(yaml.safe_dump(document))
    first = profiles.runtime_profile_policy_digest(path)
    document.update(cluster_id="b", profile_version="v2")
    document["claims"].reverse()
    path.write_text(yaml.safe_dump(document))
    assert profiles.runtime_profile_policy_digest(path) == first
    document["environment"] = "other"
    path.write_text(yaml.safe_dump(document))
    assert profiles.runtime_profile_policy_digest(path) != first


@pytest.mark.parametrize(
    "desired,existing,problem",
    [
        (None, None, "no desired profile"),
        ({"warnings": ["capability unavailable"]}, None, "unavailable OWN/DELEGATE"),
        ({}, [], "invalid existing data"),
        (
            {"profile": "desired"},
            {"profile": "other"},
            "differs from the declared policy",
        ),
        ({}, None, "not registered"),
    ],
)
def test_profile_verification_rejects_invalid_or_drifted_compilation(
    monkeypatch: pytest.MonkeyPatch, desired: Any, existing: Any, problem: str
) -> None:
    monkeypatch.setattr(
        profiles,
        "inspect_runtime_profile",
        lambda _release: {"desired": desired, "existing": existing},
    )
    with pytest.raises(ReleaseError, match=problem):
        profiles.verify_runtime_profile(ResourceRelease())


def test_profile_registration_cannot_replace_existing_content_or_accept_wrong_ack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = ResourceRelease()
    path = tmp_path / "profile.yaml"
    path.write_text("environment: hyperpod-eks\n")
    release.config.runtime_profile_source = path
    release.config.runtime_profile_registration_cluster_id = "gpu-a"
    desired = {"profile": "desired", "warnings": []}
    inspection = {"desired": desired, "existing": desired}
    monkeypatch.setattr(
        profiles, "inspect_runtime_profile", lambda _release: copy.deepcopy(inspection)
    )
    profiles.verify_runtime_profile(release)
    profiles.ensure_runtime_profile(release)
    inspection["existing"] = {"profile": "other"}
    with pytest.raises(ReleaseError, match="choose a new profile version"):
        profiles.ensure_runtime_profile(release)
    inspection["existing"] = None
    posted = []

    def register(_release: Any, **kwargs: Any) -> str:
        posted.append(json.loads(kwargs["input_text"]))
        return json.dumps({"wrong": "ack"})

    monkeypatch.setattr(profiles, "exec_cpu_ingress_command", register)
    with pytest.raises(ReleaseError, match="differs from the validated payload"):
        profiles.ensure_runtime_profile(release)
    assert posted[0]["cluster_id"] == "gpu-a"
    assert posted[0]["profile_version"] == "candidate-profile"
    release.runner.dry_run = True
    profiles.ensure_runtime_profile(release)
    assert len(posted) == 1


@pytest.mark.parametrize(
    "text,problem",
    [
        ("null\n---\n[]\n", "declares no objects"),
        ("kind: Service\nmetadata: {name: service}\n", "unidentifiable object"),
        ("apiVersion: v1\nkind: Service\n", "unidentifiable object"),
    ],
)
def test_manifest_capture_rejects_undeclared_identity(text: str, problem: str) -> None:
    with pytest.raises(ReleaseError, match=problem):
        manifests.declared_manifest_objects(text, label="candidate")


@pytest.mark.parametrize(
    "value,problem", [([], "not a Kubernetes object"), ({}, "has no metadata")]
)
def test_restore_object_must_have_metadata(value: Any, problem: str) -> None:
    with pytest.raises(ReleaseError, match=problem):
        manifests.restorable_object(value, "owned", label="previous")


def test_restorable_object_strips_only_server_metadata_and_empty_annotations() -> None:
    value = {
        "kind": "Deployment",
        "metadata": {
            "name": "owned",
            "uid": "uid-a",
            "labels": {"keep": "yes"},
            "annotations": {"deployment.kubernetes.io/revision": "3"},
        },
        "spec": {"replicas": 2},
        "status": {"readyReplicas": 2},
    }
    before = copy.deepcopy(value)
    assert manifests.restorable_object(value, "owned", label="previous") == {
        "kind": "Deployment",
        "metadata": {"name": "owned", "labels": {"keep": "yes"}},
        "spec": {"replicas": 2},
    }
    assert value == before


@pytest.mark.parametrize(
    "absent", [[None], [{}], [{"name": "owned"}], [{"resource": "service"}]]
)
def test_snapshot_absence_entries_require_resource_and_name(absent: list[Any]) -> None:
    with pytest.raises(ReleaseError, match="snapshot is invalid"):
        manifests.snapshot_parts(
            {"namespace": "example", "objects": [], "absent": absent}, label="snapshot"
        )


def test_restored_deployment_without_name_is_not_restarted() -> None:
    release = ResourceRelease()
    with pytest.raises(ReleaseError, match="has no name"):
        manifests.restart_snapshot_deployments(
            release,
            "example",
            [None, {"kind": "Service"}, {"kind": "Deployment"}],
            timeout="5s",
        )
    assert release.runner.calls == []


def test_schema_acceptance_uses_explicit_mode_and_preserves_recorded_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        schema.requested_acceptance_mode({schema.ACCEPT_SCHEMA_CHANGE_ENV: "1"})
        == "snapshot"
    )
    release = ResourceRelease()
    release.config.auto_rollback = True
    release.config.database_schema_version = 17
    monkeypatch.setenv(schema.ACCEPT_SCHEMA_CHANGE_ENV, "snapshot")
    result = schema.resolve_acceptance(
        release, changed={"database_schema"}, resume=True
    )
    assert result["mode"] == "snapshot"
    assert result["database_schema_version"] == 17
    assert "no database snapshot was taken" in schema.fail_forward_reason(result)
    assert "snapshot example" in schema.fail_forward_reason(
        {**result, "snapshot_id": "example"}
    )


@pytest.mark.parametrize(
    "value",
    [{}, {"postgres-url": ""}, {"postgres-url": base64.b64encode(b"\xff").decode()}],
)
def test_schema_snapshot_cannot_infer_database_identity_from_missing_or_invalid_input(
    value: dict[str, Any],
) -> None:
    release = ResourceRelease()
    release.config.health.aurora_cluster_id = None
    release.config.database_schema_version = 17
    release.documents[("cpu", "secret", "gpu-fault-aurora")] = {"data": value}
    assert schema.aurora_cluster_identifier(release) is None
    with pytest.raises(
        ReleaseError, match="cannot take the pre-schema Aurora snapshot"
    ):
        schema.ensure_schema_change_snapshot(release, {"mode": "snapshot"})
    assert release.runner.calls == []


def test_database_identity_read_failure_is_not_a_snapshot_creation_authorization() -> (
    None
):
    release = ResourceRelease()
    release.config.health.aurora_cluster_id = None
    release.documents[("cpu", "secret", "gpu-fault-aurora")] = ReleaseError(
        "cannot read"
    )
    assert schema.aurora_cluster_identifier(release) is None
    assert release.runner.calls == []


@pytest.mark.parametrize("status", ["failed", "deleted", "deleting"])
def test_failed_database_snapshot_stops_before_schema_mutation(status: str) -> None:
    release = ResourceRelease()
    release.config.health.aurora_cluster_id = "aurora-example"
    release.config.database_schema_version = 17
    identifier = schema.snapshot_identifier(release)
    release.runner.handler = json_response(
        {
            "DBClusterSnapshots": [
                {"DBClusterSnapshotIdentifier": "other", "Status": "available"},
                {"DBClusterSnapshotIdentifier": identifier, "Status": status},
            ]
        }
    )
    with pytest.raises(ReleaseError, match=f"is {status}"):
        schema.ensure_schema_change_snapshot(release, {"mode": "snapshot"})
    assert len(release.runner.calls) == 1
    assert release.runner.calls[0][0][1:3] == ["rds", "describe-db-cluster-snapshots"]


def test_recorded_available_snapshot_is_returned_without_any_transport() -> None:
    release = ResourceRelease()
    accepted = {
        "mode": "snapshot",
        "snapshot_id": "example",
        "snapshot_status": "available",
    }
    assert schema.ensure_schema_change_snapshot(release, accepted) == accepted
    assert release.runner.calls == []
