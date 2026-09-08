from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin.config import (
    MAX_ACTIVE_PER_FAILURE_DOMAIN,
    MAX_ACTIVE_PER_NODE,
    AdminConfig,
    AdminConfigError,
    AuroraCapacityConfig,
    CapacityConfig,
    ProcessorConfig,
    RemediationCapacity,
    admin_config_desired_path,
    admin_config_history_path,
    admin_config_pending_path,
    admin_config_spec,
    aurora_min_acu_floor,
    begin_admin_config_apply,
    canonical_sha256,
    complete_admin_config_apply,
    default_admin_config,
    default_admin_config_reference,
    load_desired_admin_config,
    load_pending_admin_config_apply,
    matching_pending_admin_config_apply,
    upgrade_legacy_record,
)
from gpu_fault.admin.config_file import (
    admin_config_file_path,
    initialize_desired_admin_config,
    load_admin_config_file,
)
from gpu_fault.admin.config_parser import (
    AdminConfigParseError,
    boolean_field,
    camel_case,
)
from gpu_fault.admin.config_patch import apply_patch, preset_admin_config
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
APPROVER = "arn:aws:sts::123456789012:assumed-role/Admin/alice"
STARTED = datetime(2026, 9, 1, 1, 0, tzinfo=UTC)


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


def _begin(tmp_path: Path, desired: AdminConfig, **overrides):
    arguments = {
        "site_identity": SITE_IDENTITY,
        "release_identity": RELEASE_IDENTITY,
        "desired": desired,
        "source": "file:/secure/admin-config.yaml",
        "approver_identity": APPROVER,
        "reference": "CHG-12345",
        "started_at": STARTED,
    }
    arguments.update(overrides)
    return begin_admin_config_apply(tmp_path, **arguments)


# ---------------------------------------------------------------------------
# The schema: one set of dataclasses, two spellings.
# ---------------------------------------------------------------------------


def test_default_as_dict_is_the_persisted_digest_contract() -> None:
    """Every live desired.json digests exactly this tree; a key change here
    would refuse every site on its next read. The two safety constants stay
    in the record although they are no longer fields."""

    expected = {
        "schema_version": 1,
        "capacity": {
            "control_worker_replicas": 6,
            "largest_cluster_node_count": 512,
            "managed_node_count": 512,
            "telemetry_spool": {"enabled": False, "replicas": 0},
            "remediation": {
                "max_active_region": 20,
                "max_active_per_cluster": 5,
                "max_active_per_resource_class": 2,
                "max_active_per_node": 1,
                "max_active_per_failure_domain": 1,
            },
        },
        "aurora": {"min_acu": 8.0, "max_acu": 32.0},
        "processor": {
            "max_queue_depth": 65536,
            "max_cluster_queue_depth": 4096,
            "retry_after_seconds": 2,
            "retry_backoff_seconds": 1,
            "retry_backoff_max_seconds": 30,
            "completed_retention_seconds": 600,
        },
        "workflow": {"poll_interval_seconds": 5.0, "dispatcher_workers": 8},
        "notification_delivery": {"batch_size": 25, "max_attempts": 8},
        "evidence": {"retention_hours": 24, "max_records_per_node": 10000},
    }

    config = default_admin_config()

    assert config.as_dict() == expected
    assert config.sha256() == canonical_sha256(expected)
    assert AdminConfig.from_mapping(expected) == config


def test_constant_budgets_are_readable_but_not_fields() -> None:
    remediation = RemediationCapacity()

    assert remediation.max_active_per_node == MAX_ACTIVE_PER_NODE == 1
    assert remediation.max_active_per_failure_domain == MAX_ACTIVE_PER_FAILURE_DOMAIN
    with pytest.raises(TypeError):
        RemediationCapacity(max_active_per_node=1)  # type: ignore[call-arg]
    # Old YAML may still spell them, at the constant value only.
    accepted = apply_patch(
        default_admin_config(), {"capacity": {"remediation": {"maxActivePerNode": 1}}}
    )
    assert accepted == default_admin_config()
    with pytest.raises(
        AdminConfigError, match="maxActivePerFailureDomain is fixed at 1"
    ):
        apply_patch(
            default_admin_config(),
            {"capacity": {"remediation": {"maxActivePerFailureDomain": 2}}},
        )
    # The persisted spelling behaves the same way.
    raw = default_admin_config().as_dict()
    raw["capacity"]["remediation"]["max_active_per_node"] = 2
    with pytest.raises(AdminConfigError, match="max_active_per_node is fixed at 1"):
        AdminConfig.from_mapping(raw)


def test_yaml_spec_is_the_record_in_camel_case_without_constants() -> None:
    def leaves(value: object, prefix: str = "") -> set[str]:
        if not isinstance(value, dict):
            return {prefix}
        return set().union(
            *(
                leaves(item, f"{prefix}.{key}" if prefix else key)
                for key, item in value.items()
            )
        )

    config = preset_admin_config("32-enabled")
    record = {k: v for k, v in config.as_dict().items() if k != "schema_version"}
    spec = admin_config_spec(config)

    camel_record = {
        ".".join(camel_case(part) for part in leaf.split("."))
        for leaf in leaves(record)
    }
    constants = {
        "capacity.remediation.maxActivePerNode",
        "capacity.remediation.maxActivePerFailureDomain",
    }
    assert leaves(spec) == camel_record - constants
    assert len(leaves(spec)) == 22
    assert camel_case("notification_delivery") == "notificationDelivery"
    assert camel_case("min_acu") == "minAcu"
    # Round trip: the spec read back on top of the defaults is the config.
    assert apply_patch(default_admin_config(), spec) == config


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


# ---------------------------------------------------------------------------
# Legacy records: read as what the site really ran, digested as they were.
# ---------------------------------------------------------------------------


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


def test_upgrade_legacy_record_fills_only_what_the_record_predates() -> None:
    upgraded = upgrade_legacy_record(
        {"capacity": {"largest_cluster_node_count": 300}, "processor": {}}
    )
    assert upgraded["aurora"] == {"min_acu": 0.5, "max_acu": 8.0}
    assert upgraded["capacity"] == {
        "largest_cluster_node_count": 300,
        "managed_node_count": 300,
    }
    complete = default_admin_config().as_dict()
    assert upgrade_legacy_record(complete) == complete
    # A malformed section is left alone for the reader to reject by name.
    with pytest.raises(
        AdminConfigError, match="admin config.capacity must be a mapping"
    ):
        AdminConfig.from_mapping({"capacity": ["x"]})


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
        apply_patch(recorded, {"capacity": {"controlWorkerReplicas": 5}})


def test_patch_accepts_node_counts_and_a_preset_adopts_its_aurora_floor(
    tmp_path: Path,
) -> None:
    # 4000 nodes need 40 ACU (section 13.3 ratio), so the base has to carry it
    # before the topology can be declared.
    generous = replace(
        default_admin_config(),
        aurora=AuroraCapacityConfig(min_acu=100.0, max_acu=200.0),
    )
    patched = apply_patch(
        generous,
        {"capacity": {"largestClusterNodeCount": 1000, "managedNodeCount": 4000}},
    )
    assert patched.capacity.largest_cluster_node_count == 1000
    assert patched.capacity.managed_node_count == 4000
    with pytest.raises(AdminConfigError, match="spec.aurora.minAcu"):
        apply_patch(default_admin_config(), {"capacity": {"managedNodeCount": 8192}})
    with pytest.raises(AdminConfigError, match="managedNodeCount"):
        apply_patch(
            default_admin_config(), {"capacity": {"largestClusterNodeCount": 1024}}
        )

    # A preset names a topology; choosing it raises Aurora to that topology's
    # floor when the current value is below it, and never lowers it.
    from_default = apply_patch(
        default_admin_config(), {"capacity": {"preset": "32-disabled"}}
    )
    assert from_default.aurora == AuroraCapacityConfig(min_acu=82.0, max_acu=128.0)
    assert apply_patch(generous, {"capacity": {"preset": "32-disabled"}}).aurora == (
        generous.aurora
    )
    assert apply_patch(from_default, {"capacity": {"preset": "default"}}).aurora == (
        from_default.aurora
    )
    # Explicit fields in the same document override the preset's values.
    tuned = apply_patch(
        default_admin_config(),
        {"capacity": {"preset": "32-disabled", "controlWorkerReplicas": 5}},
    )
    assert tuned.capacity.control_worker_replicas == 5
    assert tuned.capacity.remediation.max_active_region == 128
    with pytest.raises(AdminConfigError, match="unknown capacity preset"):
        apply_patch(default_admin_config(), {"capacity": {"preset": "64-enabled"}})

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
    desired = apply_patch(
        current, {"capacity": {"largestClusterNodeCount": 600, "managedNodeCount": 600}}
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

    path = _config_file(tmp_path, {"remediation": {"maxActivePerNode": 2}})
    with pytest.raises(AdminConfigError, match="maxActivePerNode is fixed at 1"):
        load_admin_config_file(path)

    path = _config_file(tmp_path, {"controlWorkerReplicas": 8})
    with pytest.raises(AdminConfigError, match="connection ceiling"):
        load_admin_config_file(path)

    path = _config_file(tmp_path, {"controlWorkerReplicas": "6"})
    with pytest.raises(
        AdminConfigError, match="spec.capacity.controlWorkerReplicas must be an integer"
    ):
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


# ---------------------------------------------------------------------------
# The apply record: pending.json and history/<started>-<sha>/.
# ---------------------------------------------------------------------------


def test_begin_records_the_attempt_and_complete_closes_it(tmp_path: Path) -> None:
    initialize_desired_admin_config(tmp_path)
    before_record = json.loads(admin_config_desired_path(tmp_path).read_text())
    desired = preset_admin_config("32-disabled")

    apply = _begin(tmp_path, desired)

    assert apply.no_op is False
    assert apply.resumed is False
    assert apply.affected_roles == ["ingress", "spool", "worker"]
    assert apply.aurora_changed is True
    assert apply.history == f"20260901T010000Z-{desired.sha256()}"
    pending = json.loads(admin_config_pending_path(tmp_path).read_text())
    assert pending == apply.record
    assert set(pending) == {
        "schema_version",
        "history",
        "started_at",
        "source",
        "reference",
        "approver_identity",
        "site_identity",
        "release_identity",
        "before_config",
        "before_config_sha256",
        "desired_config",
        "desired_config_sha256",
        "affected_roles",
        "aurora_changed",
        "changes",
    }
    assert pending["approver_identity"] == APPROVER
    assert pending["reference"] == "CHG-12345"
    assert pending["before_config_sha256"] == default_admin_config().sha256()
    assert pending["desired_config_sha256"] == desired.sha256()
    history = admin_config_history_path(tmp_path, apply.history)
    assert json.loads((history / "before.json").read_text()) == before_record
    assert not (history / "result.json").exists(), "begin already wrote a result"
    live = json.loads(admin_config_desired_path(tmp_path).read_text())
    assert load_desired_admin_config(tmp_path) == desired
    assert live["source"] == f"pending-apply:{apply.history}"
    assert live["approver_identity"] == APPROVER
    assert live["reference"] == "CHG-12345"
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert admin_config_desired_path(tmp_path).stat().st_mode & 0o777 == 0o600

    result_path = complete_admin_config_apply(
        tmp_path,
        config_sha256=desired.sha256(),
        release_id="release-a",
        success=True,
        details={"aurora": {"modified": True}},
        completed_at=STARTED + timedelta(minutes=5),
    )

    assert result_path == history / "result.json"
    result = json.loads(result_path.read_text())
    assert result["status"] == "APPLIED"
    assert result["approver_identity"] == APPROVER
    assert result["release_id"] == "release-a"
    assert result["config_sha256"] == desired.sha256()
    assert result["details"] == {"aurora": {"modified": True}}
    assert json.loads((history / "after.json").read_text()) == live
    assert not admin_config_pending_path(tmp_path).exists(), (
        "successful apply left pending.json behind"
    )
    assert load_desired_admin_config(tmp_path) == desired


def test_failed_apply_restores_before_and_the_rerun_is_a_new_attempt(
    tmp_path: Path,
) -> None:
    initialize_desired_admin_config(tmp_path)
    desired = preset_admin_config("50-disabled")
    first = _begin(tmp_path, desired)

    complete_admin_config_apply(
        tmp_path,
        config_sha256=desired.sha256(),
        release_id="release-a",
        success=False,
        error="verification failed",
    )

    assert load_desired_admin_config(tmp_path) == default_admin_config()
    restored = json.loads(admin_config_desired_path(tmp_path).read_text())
    assert restored["source"] == f"rollback-after-failed-apply:{first.history}"
    first_result = json.loads(
        (admin_config_history_path(tmp_path, first.history) / "result.json").read_text()
    )
    assert first_result["status"] == "FAILED"
    assert first_result["error"] == "verification failed"
    assert admin_config_pending_path(tmp_path).is_file(), (
        "failed apply did not keep pending.json for the resume"
    )

    second = _begin(
        tmp_path,
        desired,
        started_at=STARTED + timedelta(hours=1),
        reference="CHG-12345-RETRY",
        approver_identity="arn:aws:sts::123456789012:assumed-role/Admin/bob",
    )

    assert second.resumed is True
    assert second.history == f"20260901T020000Z-{desired.sha256()}"
    assert second.reference == "CHG-12345-RETRY"
    assert load_desired_admin_config(tmp_path) == desired
    second_before = json.loads(
        (
            admin_config_history_path(tmp_path, second.history) / "before.json"
        ).read_text()
    )
    assert second_before == restored
    complete_admin_config_apply(
        tmp_path, config_sha256=desired.sha256(), release_id="release-a", success=True
    )
    assert not admin_config_pending_path(tmp_path).exists(), (
        "successful retry left pending.json behind"
    )
    assert (
        admin_config_history_path(tmp_path, first.history) / "result.json"
    ).is_file(), "the retry erased the failed attempt's record"


def test_interrupted_apply_resumes_in_the_same_attempt(tmp_path: Path) -> None:
    """A crash between begin and complete: desired.json already names the
    target, no result was written, and the rerun continues that attempt."""

    initialize_desired_admin_config(tmp_path)
    desired = preset_admin_config("32-enabled")
    first = _begin(tmp_path, desired)
    pending_before = admin_config_pending_path(tmp_path).read_bytes()

    resumed = _begin(tmp_path, desired, started_at=STARTED + timedelta(hours=1))

    assert resumed.resumed is True
    assert resumed.history == first.history
    assert resumed.record == first.record
    assert admin_config_pending_path(tmp_path).read_bytes() == pending_before
    assert (
        matching_pending_admin_config_apply(
            tmp_path,
            site_identity=SITE_IDENTITY,
            release_identity=RELEASE_IDENTITY,
            desired=desired,
        )
        == first.record
    )
    assert (
        matching_pending_admin_config_apply(
            tmp_path,
            site_identity=SITE_IDENTITY,
            release_identity={**RELEASE_IDENTITY, "manifest_sha256": "b" * 64},
            desired=desired,
        )
        is None
    ), "a different signed release matched the pending apply"
    assert (
        matching_pending_admin_config_apply(
            tmp_path,
            site_identity=SITE_IDENTITY,
            release_identity=RELEASE_IDENTITY,
            desired=preset_admin_config("50-enabled"),
        )
        is None
    ), "a different target matched the pending apply"


def test_a_different_target_supersedes_an_unfinished_pending_apply(
    tmp_path: Path,
) -> None:
    initialize_desired_admin_config(tmp_path)
    first = _begin(tmp_path, preset_admin_config("32-disabled"))
    # The crashed attempt was never completed, so desired.json still names
    # its target; the new apply starts from there.
    replacement = preset_admin_config("50-disabled")

    second = _begin(tmp_path, replacement, started_at=STARTED + timedelta(hours=1))

    first_result = json.loads(
        (admin_config_history_path(tmp_path, first.history) / "result.json").read_text()
    )
    assert first_result["status"] == "SUPERSEDED"
    assert first_result["release_id"] is None
    assert second.resumed is False
    assert second.before == preset_admin_config("32-disabled")
    assert load_pending_admin_config_apply(tmp_path) == second.record


def test_resume_refuses_a_desired_json_that_moved_underneath(tmp_path: Path) -> None:
    initialize_desired_admin_config(tmp_path)
    desired = preset_admin_config("32-disabled")
    _begin(tmp_path, desired)
    path = admin_config_desired_path(tmp_path)
    foreign = preset_admin_config("50-enabled")
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": foreign.as_dict(),
                "config_sha256": foreign.sha256(),
                "role_sha256": foreign.role_sha256(),
                "source": "by-hand",
                "updated_at": "2026-09-01T02:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AdminConfigError, match="changed after the interrupted apply"):
        _begin(tmp_path, desired)


def test_no_op_apply_writes_nothing(tmp_path: Path) -> None:
    initialize_desired_admin_config(tmp_path)
    before = admin_config_desired_path(tmp_path).read_bytes()

    apply = _begin(tmp_path, default_admin_config())

    assert apply.no_op is True
    assert apply.record["changes"] == []
    assert apply.affected_roles == []
    assert not admin_config_pending_path(tmp_path).exists(), (
        "a no-op wrote pending.json"
    )
    assert not (tmp_path / "admin-config/history").exists(), "a no-op wrote history"
    assert admin_config_desired_path(tmp_path).read_bytes() == before


def test_reference_defaults_to_approver_and_time_and_approver_is_required(
    tmp_path: Path,
) -> None:
    initialize_desired_admin_config(tmp_path)
    desired = preset_admin_config("32-disabled")

    apply = _begin(tmp_path, desired, reference=None)

    assert apply.reference == f"{APPROVER}:20260901T010000Z"
    assert apply.reference == default_admin_config_reference(APPROVER, STARTED)
    # A user@host fallback identity still yields a well-formed reference.
    assert default_admin_config_reference("alice@deploy-host", STARTED) == (
        "alice-deploy-host:20260901T010000Z"
    )
    assert len(default_admin_config_reference("x" * 300, STARTED)) == 128
    with pytest.raises(AdminConfigError, match="--reference must be"):
        _begin(tmp_path, desired, reference="bad reference with spaces")
    with pytest.raises(AdminConfigError, match="approver identity must be non-empty"):
        _begin(tmp_path, desired, approver_identity="  ")


def test_tampered_pending_record_fails_closed_and_names_the_next_command(
    tmp_path: Path,
) -> None:
    initialize_desired_admin_config(tmp_path)
    desired = preset_admin_config("32-disabled")
    _begin(tmp_path, desired)
    path = admin_config_pending_path(tmp_path)
    record = json.loads(path.read_text())
    record["affected_roles"] = ["worker"]
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(AdminConfigError) as error:
        load_pending_admin_config_apply(tmp_path)

    message = str(error.value)
    assert "affected_roles does not match" in message
    assert f"gpu-fault-admin config --state-dir {tmp_path}" in message
    assert "plan" not in message.replace("pending admin config apply", "")
    with pytest.raises(AdminConfigError, match="rerun gpu-fault-admin config"):
        complete_admin_config_apply(
            tmp_path, config_sha256="c" * 64, release_id="release-a", success=True
        )


def test_tampered_desired_config_fails_closed(tmp_path: Path) -> None:
    initialize_desired_admin_config(tmp_path)
    path = admin_config_desired_path(tmp_path)
    record = json.loads(path.read_text())
    record["config"]["capacity"]["control_worker_replicas"] = 7
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(AdminConfigError, match="digest does not match"):
        load_desired_admin_config(tmp_path)


def legacy_capacity_record(config: AdminConfig) -> dict[str, object]:
    """The first release's desired.json: capacity only, capacity-only role digests."""

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
    path.write_text(json.dumps(legacy_capacity_record(desired)), encoding="utf-8")

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
    record = legacy_capacity_record(preset_admin_config("32-disabled"))
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
    document = yaml.safe_load(editable.read_text(encoding="utf-8"))
    assert document["spec"] == admin_config_spec(initialized)


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
