from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from gpu_fault.admin import config
from tests.admin.test_admin_config import (
    RELEASE_IDENTITY,
    SITE_IDENTITY,
    legacy_capacity_record,
)


@pytest.mark.parametrize(
    "section",
    [
        config.RemediationCapacity(max_active_region=0),
        config.RemediationCapacity(
            max_active_region=1,
            max_active_per_cluster=2,
            max_active_per_resource_class=1,
        ),
        config.RemediationCapacity(
            max_active_region=1,
            max_active_per_cluster=1,
            max_active_per_resource_class=2,
        ),
        config.TelemetrySpoolCapacity(replicas=-1),
        config.TelemetrySpoolCapacity(enabled=True, replicas=0),
        config.CapacityConfig(control_worker_replicas=0),
        config.AuroraCapacityConfig(min_acu=0),
        config.AuroraCapacityConfig(max_acu=257),
        config.ProcessorConfig(max_queue_depth=1023),
        config.ProcessorConfig(max_queue_depth=1024, max_cluster_queue_depth=2048),
        config.ProcessorConfig(retry_backoff_seconds=2, retry_backoff_max_seconds=1),
        config.ProcessorConfig(completed_retention_seconds=60),
        config.WorkflowConfig(poll_interval_seconds=0),
        config.WorkflowConfig(dispatcher_workers=0),
        config.NotificationDeliveryConfig(batch_size=0),
        config.NotificationDeliveryConfig(max_attempts=0),
        config.EvidenceConfig(retention_hours=0),
        config.EvidenceConfig(max_records_per_node=99),
    ],
)
def test_each_configuration_section_rejects_out_of_bounds_values(section):
    with pytest.raises(config.AdminConfigError):
        section.validate()


def test_structural_queue_floor_is_required_even_for_recorded_state():
    value = replace(
        config.AdminConfig(),
        processor=config.ProcessorConfig(max_cluster_queue_depth=128),
    )
    with pytest.raises(config.AdminConfigError, match="cannot hold one fault-priority"):
        value.validate(enforce_capacity_evidence=False)


@pytest.mark.parametrize("document", [[], {"schema_version": 2}])
def test_recorded_admin_config_requires_mapping_and_supported_schema(document):
    with pytest.raises(config.AdminConfigError, match="mapping|schema_version"):
        config.AdminConfig.from_mapping(document)


def test_role_payload_rejects_unknown_control_plane_role():
    with pytest.raises(config.AdminConfigError, match="unknown control-plane role"):
        config.AdminConfig().role_payload("gpu")


def begin(root, **changes):
    defaults = config.AdminConfig()
    desired = replace(
        defaults, workflow=replace(defaults.workflow, poll_interval_seconds=6.0)
    )
    options = {
        "site_identity": SITE_IDENTITY,
        "release_identity": RELEASE_IDENTITY,
        "desired": desired,
        "source": "example-config",
        "approver_identity": "example-operator",
        "reference": "CHG-EXAMPLE",
        "started_at": datetime(2026, 9, 12, tzinfo=UTC),
        **changes,
    }
    return config.begin_admin_config_apply(root, **options)


@pytest.mark.parametrize(
    "field,value",
    [("site_name", ""), ("aws_region", " padded "), ("cpu_eks_arn", None)],
)
def test_apply_requires_complete_site_identity_before_state_write(
    tmp_path, field, value
):
    identity = {**SITE_IDENTITY, field: value}
    with pytest.raises(config.AdminConfigError, match="site_identity"):
        begin(tmp_path, site_identity=identity)
    assert not config.admin_config_pending_path(tmp_path).exists(), (
        "invalid site identity created a pending apply"
    )


@pytest.mark.parametrize(
    "field,value",
    [("release_id", ""), ("manifest_sha256", "invalid"), ("staging_only", "false")],
)
def test_apply_requires_complete_release_identity_before_state_write(
    tmp_path, field, value
):
    with pytest.raises(config.AdminConfigError, match="release_identity"):
        begin(tmp_path, release_identity={**RELEASE_IDENTITY, field: value})
    assert not config.admin_config_pending_path(tmp_path).exists(), (
        "invalid release identity created a pending apply"
    )


@pytest.mark.parametrize(
    "kind", ["invalid-json", "not-object", "schema", "config", "roles"]
)
def test_desired_config_refuses_corrupt_record(tmp_path, kind):
    path = config.persist_desired_admin_config(
        tmp_path, config=config.AdminConfig(), source="example"
    )
    if kind in {"invalid-json", "not-object"}:
        path.write_text("invalid" if kind == "invalid-json" else "[]")
    else:
        record = json.loads(path.read_text())
        field = {
            "schema": "schema_version",
            "config": "config",
            "roles": "role_sha256",
        }[kind]
        record[field] = 2 if kind == "schema" else []
        path.write_text(json.dumps(record))
    with pytest.raises(config.AdminConfigError, match="invalid|object|mapping|digests"):
        config.load_desired_admin_config(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("history", "invalid"),
        ("history", "20260912T000000Z-" + "f" * 64),
        ("reference", "invalid reference"),
        ("approver_identity", ""),
        ("started_at", None),
        ("started_at", "invalid"),
        ("started_at", "2026-09-12T00:00:00"),
    ],
)
def test_pending_apply_rejects_inconsistent_checkpoint_fields(tmp_path, field, value):
    begin(tmp_path)
    path = config.admin_config_pending_path(tmp_path)
    document = json.loads(path.read_text())
    document[field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(
        config.AdminConfigError, match="pending admin config apply record is invalid"
    ):
        config.load_pending_admin_config_apply(tmp_path)


def test_pending_apply_rejects_missing_or_extra_fields(tmp_path):
    begin(tmp_path)
    path = config.admin_config_pending_path(tmp_path)
    document = json.loads(path.read_text())
    document["untrusted"] = True
    path.write_text(json.dumps(document))
    with pytest.raises(config.AdminConfigError, match="schema"):
        config.load_pending_admin_config_apply(tmp_path)


def test_wrong_target_cannot_complete_an_apply(tmp_path):
    apply = begin(tmp_path)
    with pytest.raises(config.AdminConfigError, match="changed while"):
        config.complete_admin_config_apply(
            tmp_path, config_sha256="0" * 64, release_id="release-a", success=True
        )
    assert (
        config.load_pending_admin_config_apply(tmp_path)["desired_config_sha256"]
        == apply.desired.sha256()
    )


def test_missing_pending_apply_cannot_be_completed(tmp_path):
    with pytest.raises(config.AdminConfigError, match="changed while"):
        config.complete_admin_config_apply(
            tmp_path, config_sha256="0" * 64, release_id="release-a", success=True
        )
    assert not (tmp_path / "admin-config").exists(), (
        "missing pending apply created configuration state"
    )


@pytest.mark.parametrize("history", ["../outside", "invalid", "20260912T000000Z-bad"])
def test_history_path_rejects_noncanonical_identity(tmp_path, history):
    with pytest.raises(config.AdminConfigError, match="history id"):
        config.admin_config_history_path(tmp_path, history)


def test_legacy_migration_preserves_audit_without_requiring_source_label(tmp_path):
    path = config.admin_config_desired_path(tmp_path)
    path.parent.mkdir()
    record = legacy_capacity_record(config.AdminConfig())
    record["source"] = None
    record["approver_identity"] = "example-operator"
    path.write_text(json.dumps(record))
    value = config.load_desired_admin_config(tmp_path, migrate_legacy=True)
    migrated = json.loads(path.read_text())
    assert migrated["source"] == "legacy-capacity-migration"
    assert migrated["approver_identity"] == "example-operator"
    assert migrated["reference"] == record["reference"]
    assert migrated["config_sha256"] == value.sha256()


def test_new_target_does_not_overwrite_previous_failed_result(tmp_path):
    first = begin(tmp_path)
    result_path = config.complete_admin_config_apply(
        tmp_path,
        config_sha256=first.desired.sha256(),
        release_id="release-a",
        success=False,
        error="example failure",
    )
    original = result_path.read_bytes()
    second = begin(
        tmp_path,
        desired=replace(
            first.desired, workflow=config.WorkflowConfig(poll_interval_seconds=7.0)
        ),
    )
    assert not second.resumed, "different target reused the failed apply attempt"
    assert second.history != first.history
    assert result_path.read_bytes() == original
    assert json.loads(original)["status"] == "FAILED"
