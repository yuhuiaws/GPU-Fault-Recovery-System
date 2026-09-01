from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin_config import (
    AdminConfigError,
    admin_config_approval_path,
    admin_config_desired_path,
    admin_config_history_path,
    admin_config_plan_path,
    complete_admin_config_apply,
    create_admin_config_plan,
    default_admin_config,
    load_desired_admin_config,
    prepare_admin_config_apply,
    preset_admin_config,
)
from gpu_fault.admin_config_file import (
    admin_config_file_path,
    initialize_desired_admin_config,
    load_admin_config_file,
)

ROOT = Path(__file__).resolve().parents[2]
SITE_IDENTITY = {
    "site_name": "test-site",
    "aws_region": "us-east-1",
    "cpu_eks_arn": "arn:aws:eks:us-east-1:123456789012:cluster/control",
}
RELEASE_IDENTITY = {
    "release_id": "release-a",
    "manifest_sha256": "a" * 64,
    "staging_only": False,
}


def _config_file(tmp_path: Path, capacity: dict[str, object]) -> Path:
    path = tmp_path / "admin-config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                "spec": {"capacity": capacity},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def test_capacity_presets_are_coherent_and_bounded() -> None:
    disabled = preset_admin_config("32-disabled")
    enabled = preset_admin_config("50-enabled")

    assert disabled.capacity.remediation.max_active_region == 128
    assert disabled.capacity.telemetry_spool.enabled is False
    assert disabled.capacity.telemetry_spool.replicas == 0
    assert enabled.capacity.remediation.max_active_region == 200
    assert enabled.capacity.telemetry_spool.enabled is True
    assert enabled.capacity.telemetry_spool.replicas == 3
    assert disabled.role_sha256()["worker"] != enabled.role_sha256()["worker"]
    assert disabled.role_sha256()["ingress"] != enabled.role_sha256()["ingress"]


def test_admin_config_file_merges_omitted_fields_with_current_state(
    tmp_path: Path,
) -> None:
    current = preset_admin_config("32-disabled")
    path = _config_file(tmp_path, {"telemetrySpool": {"enabled": True, "replicas": 3}})

    desired = load_admin_config_file(path, base=current)

    assert desired.capacity.remediation == current.capacity.remediation
    assert desired.capacity.control_worker_replicas == 6
    assert desired.capacity.telemetry_spool.enabled is True
    assert desired.capacity.telemetry_spool.replicas == 3


def test_admin_config_rejects_unsafe_or_incoherent_values(tmp_path: Path) -> None:
    path = _config_file(tmp_path, {"telemetrySpool": {"enabled": False, "replicas": 3}})
    with pytest.raises(AdminConfigError, match="disabled telemetry spool"):
        load_admin_config_file(path)

    path = _config_file(tmp_path, {"remediation": {"maxActivePerNode": 1}})
    with pytest.raises(AdminConfigError, match="unknown fields"):
        load_admin_config_file(path)

    path = _config_file(tmp_path, {"controlWorkerReplicas": 8})
    with pytest.raises(AdminConfigError, match="connection ceiling"):
        load_admin_config_file(path)


def test_admin_config_file_must_be_private_and_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    path = _config_file(tmp_path, {"unknownCapacity": 1})
    with pytest.raises(AdminConfigError, match="unknown fields"):
        load_admin_config_file(path)

    path = _config_file(tmp_path, {})
    path.chmod(0o644)
    with pytest.raises(AdminConfigError, match="group/other"):
        load_admin_config_file(path)


def test_plan_apply_persists_audited_desired_config(tmp_path: Path) -> None:
    initialize_desired_admin_config(tmp_path)
    desired = preset_admin_config("32-disabled")
    plan = create_admin_config_plan(
        tmp_path,
        site_identity=SITE_IDENTITY,
        release_identity=RELEASE_IDENTITY,
        desired=desired,
        source="preset:32-disabled",
    )
    approved_at = datetime(2026, 9, 1, 1, 0, tzinfo=UTC)

    prepared = prepare_admin_config_apply(
        tmp_path,
        expected_plan_sha256=str(plan["plan_sha256"]),
        reference="CHG-12345",
        current_release_identity=RELEASE_IDENTITY,
        approved_at=approved_at,
    )

    assert prepared.no_op is False
    assert load_desired_admin_config(tmp_path) == desired
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert admin_config_desired_path(tmp_path).stat().st_mode & 0o777 == 0o600
    assert admin_config_approval_path(tmp_path).is_file(), (
        "admin config approval was not persisted"
    )
    history = admin_config_history_path(tmp_path, str(plan["plan_sha256"]))
    assert json.loads((history / "plan.json").read_text()) == plan

    result = complete_admin_config_apply(
        tmp_path,
        expected_plan_sha256=str(plan["plan_sha256"]),
        release_id="release-a",
        success=True,
        completed_at=datetime(2026, 9, 1, 1, 5, tzinfo=UTC),
    )

    assert json.loads(result.read_text())["status"] == "APPLIED"
    assert not admin_config_plan_path(tmp_path).exists(), (
        "successful apply left the active plan behind"
    )
    assert not admin_config_approval_path(tmp_path).exists(), (
        "successful apply left the active approval behind"
    )


def test_failed_apply_keeps_plan_and_supports_idempotent_resume(tmp_path: Path) -> None:
    initialize_desired_admin_config(tmp_path)
    desired = preset_admin_config("50-disabled")
    plan = create_admin_config_plan(
        tmp_path,
        site_identity=SITE_IDENTITY,
        release_identity=RELEASE_IDENTITY,
        desired=desired,
        source="preset:50-disabled",
    )
    first = prepare_admin_config_apply(
        tmp_path,
        expected_plan_sha256=str(plan["plan_sha256"]),
        reference="MW-2026-09-01",
        current_release_identity=RELEASE_IDENTITY,
    )
    repeated = prepare_admin_config_apply(
        tmp_path,
        expected_plan_sha256=str(plan["plan_sha256"]),
        reference="MW-2026-09-01",
        current_release_identity=RELEASE_IDENTITY,
    )
    assert repeated.config == first.config

    complete_admin_config_apply(
        tmp_path,
        expected_plan_sha256=str(plan["plan_sha256"]),
        release_id="release-a",
        success=False,
        error="verification failed",
    )

    assert admin_config_plan_path(tmp_path).is_file(), (
        "failed apply did not retain its retryable plan"
    )
    assert admin_config_approval_path(tmp_path).is_file(), (
        "failed apply did not retain its approval"
    )
    assert (
        prepare_admin_config_apply(
            tmp_path,
            expected_plan_sha256=str(plan["plan_sha256"]),
            reference="MW-2026-09-01",
            current_release_identity=RELEASE_IDENTITY,
        ).config
        == desired
    )


def test_apply_rejects_release_drift_before_persisting_desired_config(
    tmp_path: Path,
) -> None:
    initialize_desired_admin_config(tmp_path)
    desired = preset_admin_config("32-disabled")
    plan = create_admin_config_plan(
        tmp_path,
        site_identity=SITE_IDENTITY,
        release_identity=RELEASE_IDENTITY,
        desired=desired,
        source="preset:32-disabled",
    )

    with pytest.raises(AdminConfigError, match="signed release differs"):
        prepare_admin_config_apply(
            tmp_path,
            expected_plan_sha256=str(plan["plan_sha256"]),
            reference="CHG-12345",
            current_release_identity={**RELEASE_IDENTITY, "manifest_sha256": "b" * 64},
        )

    assert load_desired_admin_config(tmp_path) == default_admin_config()
    assert not admin_config_approval_path(tmp_path).exists(), (
        "release drift persisted an approval before applying desired config"
    )


def test_new_plan_supersedes_an_active_approval(tmp_path: Path) -> None:
    initialize_desired_admin_config(tmp_path)
    first_plan = create_admin_config_plan(
        tmp_path,
        site_identity=SITE_IDENTITY,
        release_identity=RELEASE_IDENTITY,
        desired=preset_admin_config("32-disabled"),
        source="preset:32-disabled",
    )
    prepare_admin_config_apply(
        tmp_path,
        expected_plan_sha256=str(first_plan["plan_sha256"]),
        reference="CHG-12345",
        current_release_identity=RELEASE_IDENTITY,
    )

    replacement = create_admin_config_plan(
        tmp_path,
        site_identity=SITE_IDENTITY,
        release_identity=RELEASE_IDENTITY,
        desired=preset_admin_config("50-disabled"),
        source="preset:50-disabled",
    )

    history = admin_config_history_path(tmp_path, str(first_plan["plan_sha256"]))
    superseded = json.loads((history / "superseded.json").read_text())
    assert superseded["replacement_plan_sha256"] == replacement["plan_sha256"]
    assert not admin_config_approval_path(tmp_path).exists(), (
        "replacement plan left the superseded approval active"
    )


def test_tampered_desired_config_fails_closed(tmp_path: Path) -> None:
    initialize_desired_admin_config(tmp_path)
    path = admin_config_desired_path(tmp_path)
    record = json.loads(path.read_text())
    record["config"]["capacity"]["control_worker_replicas"] = 7
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(AdminConfigError, match="digest does not match"):
        load_desired_admin_config(tmp_path)


def test_default_admin_config_is_stable() -> None:
    config = default_admin_config()

    assert config.capacity.control_worker_replicas == 6
    assert config.capacity.remediation.max_active_region == 20
    assert config.capacity.telemetry_spool.enabled is False


def test_release_admin_config_template_matches_defaults() -> None:
    template = ROOT / "config/admin-config.example.yaml"

    config = load_admin_config_file(template, require_private=False)

    assert config == default_admin_config()


def test_release_template_contains_only_eighteen_admin_fields() -> None:
    document = yaml.safe_load(
        (ROOT / "config/admin-config.example.yaml").read_text(encoding="utf-8")
    )

    def leaves(value: object) -> int:
        if not isinstance(value, dict):
            return 1
        return sum(leaves(item) for item in value.values())

    assert leaves(document["spec"]) == 18
    remediation = document["spec"]["capacity"]["remediation"]
    assert "maxActivePerNode" not in remediation
    assert "maxActivePerFailureDomain" not in remediation


def test_common_admin_tuning_changes_every_cpu_role_digest(tmp_path: Path) -> None:
    path = tmp_path / "admin-config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                "spec": {
                    "processor": {
                        "maxQueueDepth": 131072,
                        "maxClusterQueueDepth": 2048,
                        "retryBackoffMaxSeconds": 45,
                        "completedRetentionSeconds": 900,
                    },
                    "workflow": {"pollIntervalSeconds": 10, "dispatcherWorkers": 12},
                    "notificationDelivery": {"batchSize": 50, "maxAttempts": 10},
                    "evidence": {"retentionHours": 48, "maxRecordsPerNode": 20000},
                },
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    current = default_admin_config()

    desired = load_admin_config_file(path, base=current)

    assert desired.processor.max_queue_depth == 131072
    assert desired.processor.max_cluster_queue_depth == 2048
    assert desired.workflow.dispatcher_workers == 12
    assert desired.notification_delivery.batch_size == 50
    assert desired.evidence.retention_hours == 48
    for role, digest in desired.role_sha256().items():
        assert digest != current.role_sha256()[role], (
            f"{role} digest ignored common administrator tuning"
        )


def test_initialization_materializes_private_editable_config(tmp_path: Path) -> None:
    initialized = initialize_desired_admin_config(tmp_path)
    editable = admin_config_file_path(tmp_path)

    assert editable.is_file(), "initialization did not create admin-config.yaml"
    assert editable.stat().st_mode & 0o777 == 0o600
    assert load_admin_config_file(editable) == initialized
