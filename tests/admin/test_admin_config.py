from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigError,
    AuroraCapacityConfig,
    CapacityConfig,
    ProcessorConfig,
    admin_config_approval_path,
    admin_config_desired_path,
    admin_config_history_path,
    admin_config_plan_path,
    aurora_min_acu_floor,
    canonical_sha256,
    complete_admin_config_apply,
    create_admin_config_plan,
    default_admin_config,
    load_desired_admin_config,
    prepare_admin_config_apply,
)
from gpu_fault.admin.config_file import (
    admin_config_file_path,
    initialize_desired_admin_config,
    load_admin_config_file,
)
from gpu_fault.admin.config_parser import AdminConfigParseError, boolean_field
from gpu_fault.admin.config_patch import apply_capacity_patch, preset_admin_config
from gpu_fault.admin.site import SiteConfigError

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


def test_capacity_presets_model_their_topology_and_its_aurora_floor() -> None:
    # 32 x 256 and 50 x 256 nodes (perf plan section 2); the 50-cluster run
    # needed Min ACU ~124-128 pre-provisioned to reach zero 503 (section 13.3),
    # so the preset must carry that floor instead of the 8 ACU default.
    thirty_two = preset_admin_config("32-enabled")
    fifty = preset_admin_config("50-disabled")

    assert thirty_two.capacity.largest_cluster_node_count == 256
    assert thirty_two.capacity.managed_node_count == 32 * 256
    assert thirty_two.aurora == AuroraCapacityConfig(min_acu=82.0, max_acu=128.0)
    assert fifty.capacity.largest_cluster_node_count == 256
    assert fifty.capacity.managed_node_count == 50 * 256
    assert fifty.aurora == AuroraCapacityConfig(min_acu=128.0, max_acu=128.0)
    for name in ("32-disabled", "32-enabled", "50-disabled", "50-enabled"):
        preset_admin_config(name).validate()


def test_aurora_min_acu_floor_rounds_up_to_half_acu() -> None:
    # Perf plan section 13.3: 50 x 256 = 12,800 nodes needed ~124-128 ACU.
    assert aurora_min_acu_floor(12800) == 128.0
    assert aurora_min_acu_floor(512) == 5.5
    assert aurora_min_acu_floor(2048) == 20.5
    assert aurora_min_acu_floor(8192) == 82.0
    assert aurora_min_acu_floor(100) == 1.0
    assert aurora_min_acu_floor(1) == 0.5
    assert aurora_min_acu_floor(101) == 1.5


def test_node_count_bounds_and_ordering_are_validated() -> None:
    with pytest.raises(AdminConfigError, match="largestClusterNodeCount"):
        CapacityConfig(largest_cluster_node_count=0).validate()
    with pytest.raises(AdminConfigError, match="largestClusterNodeCount"):
        CapacityConfig(
            largest_cluster_node_count=4097, managed_node_count=8192
        ).validate()
    with pytest.raises(AdminConfigError, match="managedNodeCount"):
        CapacityConfig(
            largest_cluster_node_count=512, managed_node_count=511
        ).validate()
    with pytest.raises(AdminConfigError, match="managedNodeCount"):
        CapacityConfig(
            largest_cluster_node_count=512, managed_node_count=65537
        ).validate()
    CapacityConfig(largest_cluster_node_count=512, managed_node_count=512).validate()


def test_cluster_queue_depth_must_hold_four_waves_of_the_largest_cluster() -> None:
    # Perf plan section 13.4: a 1000-node cluster at depth 1024 produced 243
    # HTTP 429; depth 4096 produced 0. Four requests per node is the measured
    # ratio.
    config = AdminConfig(
        capacity=CapacityConfig(
            largest_cluster_node_count=1000, managed_node_count=1000
        ),
        aurora=AuroraCapacityConfig(min_acu=10.0, max_acu=32.0),
        processor=ProcessorConfig(max_cluster_queue_depth=1024),
    )
    with pytest.raises(AdminConfigError) as error:
        config.validate()
    message = str(error.value)
    assert "spec.processor.maxClusterQueueDepth" in message
    assert "spec.capacity.largestClusterNodeCount" in message
    replace(config, processor=ProcessorConfig(max_cluster_queue_depth=4000)).validate()


def test_aurora_min_acu_must_cover_the_managed_node_count() -> None:
    config = AdminConfig(
        capacity=CapacityConfig(largest_cluster_node_count=512, managed_node_count=2048)
    )
    with pytest.raises(AdminConfigError) as error:
        config.validate()
    message = str(error.value)
    assert "spec.aurora.minAcu" in message
    assert "spec.capacity.managedNodeCount" in message
    assert "20.5" in message
    replace(config, aurora=AuroraCapacityConfig(min_acu=20.5, max_acu=32.0)).validate()


def test_fault_reserved_depths_derive_from_node_count_and_queue_depth() -> None:
    config = default_admin_config()

    assert config.fault_reserved_cluster_depth() == 512
    assert config.fault_reserved_queue_depth() == 65536 // 8
    wide = replace(config, processor=ProcessorConfig(max_cluster_queue_depth=65536))
    assert wide.fault_reserved_cluster_depth() == 65536 // 8
    tall = AdminConfig(
        capacity=CapacityConfig(
            largest_cluster_node_count=1000, managed_node_count=1000
        ),
        processor=ProcessorConfig(max_cluster_queue_depth=4096),
    )
    assert tall.fault_reserved_cluster_depth() == 1000
    assert (
        tall.fault_reserved_cluster_depth() >= tall.capacity.largest_cluster_node_count
    )


def test_legacy_persisted_config_without_node_counts_derives_them_from_depth() -> None:
    # A desired.json written before node counts existed carries depth 1024 and
    # no node counts; it must load as a 256-node cluster, not as the new 512
    # default that depth 1024 cannot hold.
    raw = default_admin_config().as_dict()
    capacity = raw["capacity"]
    assert isinstance(capacity, dict), "capacity must serialise as a mapping"
    del capacity["largest_cluster_node_count"]
    del capacity["managed_node_count"]
    processor = raw["processor"]
    assert isinstance(processor, dict), "processor must serialise as a mapping"
    processor["max_cluster_queue_depth"] = 1024

    config = AdminConfig.from_mapping(raw)

    assert config.capacity.largest_cluster_node_count == 256
    assert config.capacity.managed_node_count == 256
    assert config.processor.max_cluster_queue_depth == 1024
    assert AdminConfig.from_mapping({}).capacity == CapacityConfig()


def test_recorded_state_below_the_aurora_floor_loads_but_cannot_be_planned() -> None:
    # A site that predates the floor really runs 0.5/8 ACU (the hidden legacy
    # baseline). Reading that fact must not fail, or the administrator could
    # never load the state to plan the fix; authoring a new desired config
    # from it must.
    raw = default_admin_config().as_dict()
    raw["aurora"] = {"min_acu": 0.5, "max_acu": 8.0}

    recorded = AdminConfig.from_mapping(raw)

    assert recorded.aurora == AuroraCapacityConfig(min_acu=0.5, max_acu=8.0)
    with pytest.raises(AdminConfigError, match="spec.aurora.minAcu"):
        recorded.validate()
    with pytest.raises(AdminConfigError, match="spec.aurora.minAcu"):
        apply_capacity_patch(recorded, {"controlWorkerReplicas": 5})


def test_capacity_patch_accepts_node_counts_and_preset_adopts_its_aurora_floor(
    tmp_path: Path,
) -> None:
    # 4000 nodes need 40 ACU (section 13.3 ratio), so the base has to carry it
    # before the topology can be declared.
    generous = replace(
        default_admin_config(),
        aurora=AuroraCapacityConfig(min_acu=100.0, max_acu=200.0),
    )
    patched = apply_capacity_patch(
        generous, {"largestClusterNodeCount": 1000, "managedNodeCount": 4000}
    )
    assert patched.capacity.largest_cluster_node_count == 1000
    assert patched.capacity.managed_node_count == 4000
    with pytest.raises(AdminConfigError, match="spec.aurora.minAcu"):
        apply_capacity_patch(default_admin_config(), {"managedNodeCount": 8192})
    with pytest.raises(AdminConfigError, match="managedNodeCount"):
        apply_capacity_patch(default_admin_config(), {"largestClusterNodeCount": 1024})

    # A preset names a topology; choosing it raises Aurora to that topology's
    # floor when the current value is below it, and never lowers it.
    from_default = apply_capacity_patch(
        default_admin_config(), {"preset": "32-disabled"}
    )
    assert from_default.aurora == AuroraCapacityConfig(min_acu=82.0, max_acu=128.0)
    assert apply_capacity_patch(generous, {"preset": "32-disabled"}).aurora == (
        generous.aurora
    )
    assert apply_capacity_patch(from_default, {"preset": "default"}).aurora == (
        from_default.aurora
    )
    path = _config_file(
        tmp_path, {"largestClusterNodeCount": 600, "managedNodeCount": 1200}
    )
    loaded = load_admin_config_file(path, base=generous)
    assert loaded.capacity.largest_cluster_node_count == 600
    assert loaded.capacity.managed_node_count == 1200
    with pytest.raises(AdminConfigError, match="spec.aurora.minAcu"):
        load_admin_config_file(path)

    # One file may raise the topology and its Aurora floor together; the
    # capacity half must not be judged before the Aurora half is read.
    together = tmp_path / "together.yaml"
    together.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                "spec": {
                    "capacity": {"managedNodeCount": 2048},
                    "aurora": {"minAcu": 20.5, "maxAcu": 32},
                },
            }
        ),
        encoding="utf-8",
    )
    together.chmod(0o600)
    assert load_admin_config_file(together).capacity.managed_node_count == 2048


def test_node_counts_change_every_role_digest() -> None:
    # The derived fault reserve is rendered into every role's environment, so
    # a node count change that leaves all digests alone would never roll.
    current = default_admin_config()
    desired = apply_capacity_patch(
        current, {"largestClusterNodeCount": 600, "managedNodeCount": 600}
    )
    for role, digest in desired.role_sha256().items():
        assert digest != current.role_sha256()[role], (
            f"{role} digest ignored the node count change"
        )


def _pre_node_count_role_sha256(config: AdminConfig) -> dict[str, str]:
    """Role digests exactly as the release before node counts computed them."""
    capacity = config.capacity
    common = {
        "processor": config.processor.as_dict(),
        "workflow": config.workflow.as_dict(),
        "notification_delivery": config.notification_delivery.as_dict(),
        "evidence": config.evidence.as_dict(),
    }
    payloads = {
        "ingress": {
            **common,
            "telemetry_spool_enabled": capacity.telemetry_spool.enabled,
        },
        "worker": {
            **common,
            "control_worker_replicas": capacity.control_worker_replicas,
            "remediation": capacity.remediation.as_dict(),
        },
        "spool": {**common, "telemetry_spool": capacity.telemetry_spool.as_dict()},
    }
    return {role: canonical_sha256(payload) for role, payload in payloads.items()}


def test_legacy_full_admin_config_without_node_counts_is_verified_and_migrated(
    tmp_path: Path,
) -> None:
    # The shape every site persisted between the Aurora fields and the node
    # counts: all twenty fields, depth 1024, digests over exactly that content.
    expected = replace(
        default_admin_config(),
        capacity=CapacityConfig(largest_cluster_node_count=256, managed_node_count=256),
        processor=ProcessorConfig(max_cluster_queue_depth=1024),
    )
    raw = expected.as_dict()
    capacity = raw["capacity"]
    assert isinstance(capacity, dict), "capacity must serialise as a mapping"
    del capacity["largest_cluster_node_count"]
    del capacity["managed_node_count"]
    record = {
        "schema_version": 1,
        "config": raw,
        "config_sha256": canonical_sha256(raw),
        "role_sha256": _pre_node_count_role_sha256(expected),
        "source": "approved-plan:pre-node-counts",
        "plan_sha256": "b" * 64,
        "reference": "CHG-PRE-NODE-COUNTS",
        "updated_at": "2026-09-01T00:00:00+00:00",
    }
    path = admin_config_desired_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")

    assert load_desired_admin_config(tmp_path) == expected
    initialize_desired_admin_config(tmp_path)
    migrated = json.loads(path.read_text(encoding="utf-8"))

    assert migrated["config"] == expected.as_dict()
    assert migrated["config_sha256"] == expected.sha256()
    assert migrated["role_sha256"] == expected.role_sha256()
    assert migrated["source"] == (
        "legacy-admin-config-migration:approved-plan:pre-node-counts"
    )
    assert migrated["plan_sha256"] == "b" * 64
    assert load_desired_admin_config(tmp_path) == expected

    record["config"]["capacity"]["control_worker_replicas"] = 7
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(AdminConfigError, match="digest does not match"):
        load_desired_admin_config(tmp_path)


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
    assert desired.aurora == current.aurora


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

    config = default_admin_config()
    with pytest.raises(AdminConfigError, match="0.5 ACU increments"):
        AuroraCapacityConfig(min_acu=8.25, max_acu=32).validate()
    with pytest.raises(AdminConfigError, match="must not exceed"):
        AuroraCapacityConfig(min_acu=32, max_acu=8).validate()
    config.validate()


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
    assert load_desired_admin_config(tmp_path) == default_admin_config()
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


def _legacy_capacity_record(config) -> dict[str, object]:
    capacity = config.capacity
    content = {"schema_version": 1, "capacity": capacity.as_dict()}
    return {
        "schema_version": 1,
        "config": content,
        "config_sha256": canonical_sha256(content),
        "role_sha256": {
            "ingress": canonical_sha256(
                {"telemetry_spool_enabled": capacity.telemetry_spool.enabled}
            ),
            "worker": canonical_sha256(
                {
                    "control_worker_replicas": capacity.control_worker_replicas,
                    "remediation": capacity.remediation.as_dict(),
                }
            ),
            "spool": canonical_sha256(capacity.telemetry_spool.as_dict()),
        },
        "source": "approved-plan:legacy",
        "plan_sha256": "a" * 64,
        "reference": "CHG-LEGACY",
        "updated_at": "2026-08-31T00:00:00+00:00",
    }


def test_legacy_capacity_only_desired_config_is_verified_and_migrated(
    tmp_path: Path,
) -> None:
    desired = replace(
        preset_admin_config("32-disabled"),
        aurora=AuroraCapacityConfig(min_acu=0.5, max_acu=8.0),
    )
    path = admin_config_desired_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_legacy_capacity_record(desired)), encoding="utf-8")

    assert load_desired_admin_config(tmp_path) == desired
    initialize_desired_admin_config(tmp_path)
    migrated = json.loads(path.read_text(encoding="utf-8"))

    assert migrated["config"] == desired.as_dict()
    assert migrated["config_sha256"] == desired.sha256()
    assert migrated["role_sha256"] == desired.role_sha256()
    assert migrated["source"] == ("legacy-capacity-migration:approved-plan:legacy")
    assert migrated["plan_sha256"] == "a" * 64
    assert migrated["reference"] == "CHG-LEGACY"


def test_legacy_full_admin_config_preserves_hidden_aurora_baseline(
    tmp_path: Path,
) -> None:
    desired = replace(
        preset_admin_config("32-disabled"),
        aurora=AuroraCapacityConfig(min_acu=0.5, max_acu=8.0),
    )
    raw = desired.as_dict()
    raw.pop("aurora")
    record = {
        "schema_version": 1,
        "config": raw,
        "config_sha256": canonical_sha256(raw),
        "role_sha256": desired.role_sha256(),
        "source": "release-defaults",
        "updated_at": "2026-09-01T00:00:00+00:00",
    }
    path = admin_config_desired_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")

    assert load_desired_admin_config(tmp_path) == desired
    initialize_desired_admin_config(tmp_path)
    migrated = json.loads(path.read_text(encoding="utf-8"))

    assert migrated["config"]["aurora"] == {"min_acu": 0.5, "max_acu": 8.0}
    assert migrated["source"] == ("legacy-admin-config-migration:release-defaults")


def test_tampered_legacy_capacity_only_desired_config_fails_closed(
    tmp_path: Path,
) -> None:
    record = _legacy_capacity_record(preset_admin_config("32-disabled"))
    record["config"]["capacity"]["control_worker_replicas"] = 7
    path = admin_config_desired_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(AdminConfigError, match="digest does not match"):
        load_desired_admin_config(tmp_path, migrate_legacy=True)


def test_default_admin_config_is_stable() -> None:
    config = default_admin_config()

    assert config.capacity.control_worker_replicas == 6
    assert config.capacity.remediation.max_active_region == 20
    assert config.capacity.telemetry_spool.enabled is False
    # The product requirement is N clusters of 500+ nodes; one 512-node
    # cluster is the smallest topology that satisfies it.
    assert config.capacity.largest_cluster_node_count == 512
    assert config.capacity.managed_node_count == 512
    # Perf plan section 13.4: depth 1024 rejected 243 requests from one
    # 1000-node cluster, depth 4096 rejected none.
    assert config.processor.max_cluster_queue_depth == 4096
    assert config.processor.max_queue_depth == 65536
    assert config.aurora == AuroraCapacityConfig(min_acu=8.0, max_acu=32.0)
    assert config.aurora.min_acu >= aurora_min_acu_floor(
        config.capacity.managed_node_count
    )


def test_release_admin_config_template_matches_defaults() -> None:
    template = ROOT / "config/admin-config.example.yaml"

    config = load_admin_config_file(template, require_private=False)

    assert config == default_admin_config()


def test_release_template_contains_exactly_twenty_two_admin_fields() -> None:
    document = yaml.safe_load(
        (ROOT / "config/admin-config.example.yaml").read_text(encoding="utf-8")
    )

    def leaves(value: object) -> int:
        if not isinstance(value, dict):
            return 1
        return sum(leaves(item) for item in value.values())

    assert leaves(document["spec"]) == 22
    assert document["spec"]["aurora"] == {"minAcu": 8, "maxAcu": 32}
    assert document["spec"]["capacity"]["largestClusterNodeCount"] == 512
    assert document["spec"]["capacity"]["managedNodeCount"] == 512
    assert document["spec"]["processor"]["maxClusterQueueDepth"] == 4096
    remediation = document["spec"]["capacity"]["remediation"]
    assert "maxActivePerNode" not in remediation
    assert "maxActivePerFailureDomain" not in remediation


def test_admin_config_file_updates_aurora_without_changing_capacity(
    tmp_path: Path,
) -> None:
    current = preset_admin_config("32-enabled")
    path = tmp_path / "admin-config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                # 32 x 256 nodes need at least 82 ACU (section 13.3), so the
                # explicit value has to sit above the preset floor.
                "spec": {"aurora": {"minAcu": 96, "maxAcu": 160}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)

    desired = load_admin_config_file(path, base=current)

    assert desired.aurora == AuroraCapacityConfig(min_acu=96.0, max_acu=160.0)
    assert desired.capacity == current.capacity
    assert desired.role_sha256() == current.role_sha256()


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


@pytest.mark.parametrize(
    "error", [AdminConfigParseError, AdminConfigError, SiteConfigError]
)
def test_yaml_boolean_field_is_one_rule_raising_the_callers_error(
    error: type[ValueError],
) -> None:
    """Three byte-identical copies used to live in three admin modules."""

    assert boolean_field(None, "spec.flag", default=True, error=error) is True
    assert boolean_field(False, "spec.flag", default=True, error=error) is False
    with pytest.raises(error, match="spec.flag must be a boolean"):
        boolean_field("yes", "spec.flag", default=True, error=error)
    with pytest.raises(error, match="spec.flag must be a boolean"):
        boolean_field(1, "spec.flag", default=True, error=error)
