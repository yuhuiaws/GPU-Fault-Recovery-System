from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import pytest

from gpu_fault.admin import resource_registry as admin_resource_registry
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_commit import registry_delta
from gpu_fault.admin.resource_registry import (
    LegacyInstallationRegistryMissing,
    build_installation_snapshot,
    fetch_installation_resource_registry,
    find_bootstrap_state,
    load_installation_resource_snapshot,
    write_installation_resource_snapshot,
)
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)
from tests.admin.test_admin_site import site_file


def _bootstrap_state() -> dict:
    return {
        "site_id": "test-site",
        "resources": {
            "release_repositories": {
                "runtime": {
                    "repository_name": "gpu-fault/runtime-test",
                    "repository_arn": (
                        "arn:aws:ecr:us-east-1:123456789012:"
                        "repository/gpu-fault/runtime-test"
                    ),
                    "repository_uri": (
                        "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
                        "gpu-fault/runtime-test"
                    ),
                    "ownership": "CREATED",
                    "purpose": "runtime",
                },
                "cache": {
                    "repository_name": "gpu-fault/runtime-cache-test",
                    "repository_arn": (
                        "arn:aws:ecr:us-east-1:123456789012:"
                        "repository/gpu-fault/runtime-cache-test"
                    ),
                    "repository_uri": (
                        "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
                        "gpu-fault/runtime-cache-test"
                    ),
                    "ownership": "CREATED",
                    "purpose": "build-cache",
                },
            },
            "nlb_network": {
                "security_group": "sg-created",
                "security_group_ownership": "CREATED",
                "vpc_id": "vpc-cpu",
                "public_subnets": ["subnet-created", "subnet-shared"],
                "subnet_resources": [
                    {
                        "subnet_id": "subnet-created",
                        "availability_zone": "us-east-1a",
                        "ownership": "CREATED",
                        "vpc_id": "vpc-cpu",
                        "route_table_id": "rtb-created",
                        "route_table_association_id": "rtbassoc-created",
                    },
                    {
                        "subnet_id": "subnet-shared",
                        "availability_zone": "us-east-1b",
                        "ownership": "REUSED",
                        "vpc_id": "vpc-cpu",
                    },
                ],
                "internet_gateway": {
                    "internet_gateway_id": "igw-created",
                    "ownership": "CREATED",
                    "vpc_id": "vpc-cpu",
                },
            },
            "pki": {
                "hosted_zone_id": "ZTEST123",
                "zone_name": "test-site.gpu-fault.internal",
                "zone_ownership": "CREATED",
                "hostname": "api.test-site.gpu-fault.internal",
                "certificate_arn": (
                    "arn:aws:acm:us-east-1:123456789012:certificate/test"
                ),
                "certificate_ownership": "CREATED",
                "pki_secret_id": "gpu-fault/test-site/regional-pki",
                "pki_secret_ownership": "CREATED",
                "vpc_associations": [],
            },
            "monitoring_resources": {
                "workspace_id": "ws-created",
                "workspace_ownership": "CREATED",
                "sns_topic_arn": ("arn:aws:sns:us-east-1:123456789012:test-alerts"),
                "sns_topic_ownership": "CREATED",
                "email_subscription_arn": (
                    "arn:aws:sns:us-east-1:123456789012:test-alerts:email-sub"
                ),
                "email_subscription_status": "CONFIRMED",
                "email_subscription_endpoint": "ops@example.com",
                "sns_topic_generation": "a" * 32,
            },
            "adot_writer_role:gpu-a": {
                "role_arn": (
                    "arn:aws:iam::123456789012:role/gpu-fault-test-gpu-a-adot-writer"
                ),
                "ownership": "CREATED",
                "inline_policy_name": "GPUFaultDataplaneAmpWriter",
                "oidc_provider_arn": (
                    "arn:aws:iam::123456789012:oidc-provider/oidc.eks/id/A"
                ),
                "oidc_provider_ownership": "EXTERNAL",
                "cluster_name": "gpu-a",
            },
            "control_plane_role": {
                "role_arn": ("arn:aws:iam::123456789012:role/gpu-fault-control"),
                "ownership": "CREATED",
                "inline_policy_name": "GPUFaultRegionalObserve",
                "association_id": "a-control",
                "association_ownership": "CREATED",
                "cluster_name": "control",
                "namespace": "gpu-fault-system",
                "service_account": "gpu-fault-control-plane",
            },
            "email_notifications": {
                "identity": "ops@example.com",
                "identity_arn": (
                    "arn:aws:ses:us-east-1:123456789012:identity/ops@example.com"
                ),
                "identity_ownership": "EXTERNAL",
                "verified": True,
                "production_access_enabled": False,
            },
            "load_balancer_controller": {"reused": True},
            "pod_identity_agent": {
                "addon_name": "eks-pod-identity-agent",
                "cluster_name": "control",
                "ownership": "REUSED",
            },
            "aurora": {
                "cluster_id": "gpu-fault-test-aurora",
                "cluster_ownership": "CREATED",
                "instance_ids": [
                    "gpu-fault-test-aurora-writer",
                    "gpu-fault-test-aurora-reader",
                ],
                "subnet_group": "gpu-fault-test-aurora",
                "subnet_group_ownership": "CREATED",
                "security_group": "sg-aurora",
                "security_group_ownership": "CREATED",
                "parameter_group": "gpu-fault-test-aurora-pg",
                "parameter_group_ownership": "CREATED",
                "master_secret_arn": (
                    "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds-master"
                ),
            },
        },
    }


def test_registry_records_ownership_dependencies_and_delete_policy(tmp_path) -> None:
    site = load_site(site_file(tmp_path))
    snapshot = build_installation_snapshot(
        site,
        _bootstrap_state(),
        {
            "nlb": {
                "arn": (
                    "arn:aws:elasticloadbalancing:us-east-1:123456789012:"
                    "loadbalancer/net/test/123"
                ),
                "dns_name": "test.elb.amazonaws.com",
                "listeners": [
                    "arn:aws:elasticloadbalancing:us-east-1:123456789012:"
                    "listener/net/test/123/456"
                ],
                "target_groups": [
                    "arn:aws:elasticloadbalancing:us-east-1:123456789012:"
                    "targetgroup/test/789"
                ],
            }
        },
    )
    by_key = {resource.resource_key: resource for resource in snapshot.resources}

    assert by_key["aws/network/public-subnet/1"].delete_policy is (
        InstallationResourceDeletePolicy.DELETE
    )
    assert by_key["aws/network/public-subnet/2"].delete_policy is (
        InstallationResourceDeletePolicy.PRESERVE
    )
    assert by_key["cluster/gpu-a/eks"].ownership is (
        InstallationResourceOwnership.EXTERNAL
    )
    assert by_key["kubernetes/lbc/helm-release"].delete_policy is (
        InstallationResourceDeletePolicy.DELETE
    )
    # The cluster depends on all three: they are deleted only after it is gone.
    assert by_key["aws/aurora/cluster"].dependencies == [
        "aws/aurora/parameter-group",
        "aws/aurora/security-group",
        "aws/aurora/subnet-group",
    ]
    parameter_group = by_key["aws/aurora/parameter-group"]
    assert parameter_group.resource_type == "rds_cluster_parameter_group"
    assert parameter_group.resource_id == "gpu-fault-test-aurora-pg"
    assert parameter_group.ownership is InstallationResourceOwnership.CREATED
    assert parameter_group.delete_policy is InstallationResourceDeletePolicy.DELETE
    assert parameter_group.dependencies == []
    assert by_key["aws/ecr/runtime"].delete_policy is (
        InstallationResourceDeletePolicy.DELETE
    )
    adot_writer = by_key["aws/iam/adot-writer/gpu-a/role"]
    assert adot_writer.resource_type == "iam_role"
    assert adot_writer.resource_id == "gpu-fault-test-gpu-a-adot-writer"
    assert adot_writer.delete_policy is InstallationResourceDeletePolicy.DELETE, (
        "uninstall would leave the data-plane ADOT writer role behind"
    )
    assert adot_writer.attributes["inline_policy_name"] == "GPUFaultDataplaneAmpWriter"
    assert "aws/iam/adot-writer/gpu-a/oidc-provider" not in by_key, (
        "the executor task owns the OIDC provider row; a second row would make "
        "uninstall delete it twice"
    )
    assert by_key["aws/ecr/cache"].attributes["purpose"] == "build-cache"
    assert (
        by_key["aws/ses/administrator-email-identity"].delete_policy
        is InstallationResourceDeletePolicy.PRESERVE
    )
    email_subscription = by_key["aws/sns/email-subscription/af3c82544f648b38"]
    assert email_subscription.delete_policy is (InstallationResourceDeletePolicy.DETACH)
    assert email_subscription.attributes["status"] == "CONFIRMED"
    assert snapshot.source_sha256 == snapshot.digest()
    assert all(
        resource.ownership is not InstallationResourceOwnership.REUSED
        for resource in snapshot.resources
    )
    assert all(
        "password" not in key.lower()
        and "token" not in key.lower()
        and "secret" not in key.lower()
        for resource in snapshot.resources
        for key in resource.attributes
    )


LEGACY_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/test-alerts"


def _legacy_queue_state() -> dict:
    """A state written while bootstrap still created the alerts SQS queue."""

    state = _bootstrap_state()
    state["resources"]["monitoring_resources"].update(
        {
            "sqs_queue_url": LEGACY_QUEUE_URL,
            "sqs_queue_arn": "arn:aws:sqs:us-east-1:123456789012:test-alerts",
            "sqs_queue_ownership": "CREATED",
            "queue_subscription_arn": (
                "arn:aws:sns:us-east-1:123456789012:test-alerts:sub"
            ),
            "queue_subscription_ownership": "CREATED",
        }
    )
    return state


def test_a_fresh_bootstrap_registers_no_alerts_queue(tmp_path) -> None:
    """Nothing consumed the alerts queue, so bootstrap no longer creates it and
    the registry must not invent a row uninstall would then fail to find."""

    site = load_site(site_file(tmp_path))
    snapshot = build_installation_snapshot(site, _bootstrap_state(), {})
    by_key = {resource.resource_key: resource for resource in snapshot.resources}

    queue_rows = sorted(
        key
        for key, resource in by_key.items()
        if key.startswith("aws/sqs/") or resource.resource_type.startswith("sqs_")
    )
    assert queue_rows == [], f"fresh bootstrap registered queue rows: {queue_rows}"
    assert "aws/sns/queue-subscription" not in by_key
    assert "aws/sns/topic" in by_key
    assert "aws/sns/email-subscription/af3c82544f648b38" in by_key


def test_a_legacy_queue_state_keeps_its_rows_so_uninstall_deletes_them(
    tmp_path,
) -> None:
    """Sites bootstrapped before the queue was dropped still own one; the
    registry keeps both rows from the old state so uninstall removes it."""

    site = load_site(site_file(tmp_path))
    snapshot = build_installation_snapshot(site, _legacy_queue_state(), {})
    by_key = {resource.resource_key: resource for resource in snapshot.resources}

    queue = by_key["aws/sqs/queue"]
    assert queue.resource_type == "sqs_queue"
    assert queue.resource_id == LEGACY_QUEUE_URL
    assert queue.delete_policy is InstallationResourceDeletePolicy.DELETE
    binding = by_key["aws/sqs/topic-policy-binding"]
    assert binding.resource_type == "sqs_policy_binding"
    assert binding.delete_policy is InstallationResourceDeletePolicy.DETACH
    assert binding.attributes["topic_arn"] == (
        "arn:aws:sns:us-east-1:123456789012:test-alerts"
    )
    assert by_key["aws/sns/queue-subscription"].dependencies == [
        "aws/sns/topic",
        "aws/sqs/queue",
    ]


def test_a_state_written_before_parameter_groups_registers_no_group(tmp_path) -> None:
    """A cluster bootstrapped before the diagnostics group existed sits on the
    engine default group; there is nothing of ours to delete, and inventing a
    name would make uninstall try to delete a group it never created."""

    site = load_site(site_file(tmp_path))
    state = _bootstrap_state()
    del state["resources"]["aurora"]["parameter_group"]
    del state["resources"]["aurora"]["parameter_group_ownership"]

    snapshot = build_installation_snapshot(site, state, {})
    by_key = {resource.resource_key: resource for resource in snapshot.resources}

    assert "aws/aurora/parameter-group" not in by_key
    assert by_key["aws/aurora/cluster"].dependencies == [
        "aws/aurora/security-group",
        "aws/aurora/subnet-group",
    ]


def test_the_master_secret_is_registered_from_the_readiness_task_too(tmp_path) -> None:
    """Since the instance wait left the ``aurora`` task, a fresh bootstrap
    records the master Secret ARN under ``aurora_ready``; uninstall must still
    see ``aws/aurora/managed-master`` exactly as it did from the old shape."""

    site = load_site(site_file(tmp_path))
    state = _bootstrap_state()
    old_shape = build_installation_snapshot(site, state, {})
    arn = state["resources"]["aurora"].pop("master_secret_arn")
    state["resources"]["aurora_ready"] = {
        "master_secret_arn": arn,
        "master_secret_kms_key_arn": "",
    }

    new_shape = build_installation_snapshot(site, state, {})

    # Everything but the two ``record()`` timestamps must read the same.
    timestamps = {"created_at", "updated_at"}
    old_master = {r.resource_key: r for r in old_shape.resources}[
        "aws/aurora/managed-master"
    ].model_dump(exclude=timestamps)
    new_master = {r.resource_key: r for r in new_shape.resources}[
        "aws/aurora/managed-master"
    ].model_dump(exclude=timestamps)
    assert new_master == old_master, "the registry's view of the master secret changed"


def test_registry_snapshot_digest_detects_tampering(tmp_path) -> None:
    site = load_site(site_file(tmp_path))
    snapshot = build_installation_snapshot(site, _bootstrap_state(), {})
    path = write_installation_resource_snapshot(
        site, snapshot, path=tmp_path / "registry.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["resources"][0]["status"] = "FAILED"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Exception, match="digest mismatch"):
        load_installation_resource_snapshot(path)


def test_bootstrap_state_is_found_by_site_id(tmp_path) -> None:
    path = site_file(tmp_path)
    site = load_site(path)
    state = {
        "schema_version": 2,
        "site_id": "test-site",
        "resources": {},
        "completed_tasks": [],
    }
    (path.parent / "bootstrap-state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    assert find_bootstrap_state(site) == state


def test_old_control_plane_api_is_detected_before_authorization(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))
    monkeypatch.setattr(admin_resource_registry, "_cpu_pod", lambda _site: "api-pod")
    monkeypatch.setattr(
        admin_resource_registry.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 44, stdout="", stderr="GPU_FAULT_LEGACY_REGISTRY_API"
        ),
    )

    with pytest.raises(LegacyInstallationRegistryMissing):
        fetch_installation_resource_registry(site)


def test_registry_sync_retries_store_io_503(tmp_path, monkeypatch) -> None:
    site = load_site(site_file(tmp_path))
    snapshot = build_installation_snapshot(site, _bootstrap_state(), {})
    results = iter(
        [
            subprocess.CompletedProcess(
                ["kubectl"], 1, stdout="", stderr="HTTP Error 503"
            ),
            subprocess.CompletedProcess(["kubectl"], 0, stdout="[]", stderr=""),
        ]
    )
    sleeps = []
    monkeypatch.setattr(admin_resource_registry, "_cpu_pod", lambda _site: "api-pod")
    monkeypatch.setattr(
        admin_resource_registry.subprocess,
        "run",
        lambda *_args, **_kwargs: next(results),
    )
    monkeypatch.setattr(
        admin_resource_registry.time, "sleep", lambda value: sleeps.append(value)
    )

    admin_resource_registry.sync_installation_resource_snapshot(site, snapshot)

    assert sleeps == [2]


def test_registry_sync_falls_back_to_direct_aurora_upsert(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))
    snapshot = build_installation_snapshot(site, _bootstrap_state(), {})
    attempts = []
    direct = []
    monkeypatch.setattr(admin_resource_registry, "_cpu_pod", lambda _site: "api-pod")

    def unavailable(*_args, **_kwargs):
        attempts.append(True)
        return subprocess.CompletedProcess(
            ["kubectl"], 1, stdout="", stderr="HTTP Error 503"
        )

    monkeypatch.setattr(admin_resource_registry.subprocess, "run", unavailable)
    monkeypatch.setattr(admin_resource_registry.time, "sleep", lambda _value: None)
    monkeypatch.setattr(
        admin_resource_registry,
        "sync_installation_resource_snapshot_direct",
        lambda _site, value: direct.append(value),
    )

    admin_resource_registry.sync_installation_resource_snapshot(site, snapshot)

    assert len(attempts) == 4
    assert direct == [snapshot]


def _row(key: str, resource_id: str, *, arn: str | None = None) -> InstallationResource:
    now = datetime.now(timezone.utc)
    return InstallationResource(
        site_id="test-site",
        resource_key=key,
        resource_type="gpu_eks" if key.endswith("/eks") else "iam_role",
        resource_id=resource_id,
        resource_arn=arn,
        region="us-east-1",
        account_id="123456789012",
        ownership=InstallationResourceOwnership.EXTERNAL,
        delete_policy=InstallationResourceDeletePolicy.PRESERVE,
        created_at=now,
        updated_at=now,
    )


def _before(*rows: InstallationResource) -> InstallationResourceSnapshot:
    snapshot = InstallationResourceSnapshot(site_id="test-site", resources=list(rows))
    return snapshot.model_copy(update={"source_sha256": snapshot.digest()})


def test_join_registry_delta_is_exactly_the_joined_resources() -> None:
    """The write that reaches Aurora is the join's own rows; the after snapshot
    is the DISCOVERED snapshot with those rows applied."""

    before = _before(_row("aws/nlb", "gpu-fault-nlb"), _row("cluster/gpu-a/eks", "a"))
    joined = [
        _row("cluster/gpu-b/eks", "gpu-b", arn="arn:aws:eks:::cluster/gpu-b"),
        _row("aws/iam/executor/gpu-b/role", "gpu-b-executor"),
    ]

    delta, merged = registry_delta(before, joined, site_id="test-site")

    assert [item.resource_key for item in delta.resources] == sorted(
        item.resource_key for item in joined
    )
    assert delta.resources == sorted(joined, key=lambda item: item.resource_key)
    assert [item.resource_key for item in merged.resources] == [
        "aws/iam/executor/gpu-b/role",
        "aws/nlb",
        "cluster/gpu-a/eks",
        "cluster/gpu-b/eks",
    ]
    assert delta.source_sha256 == delta.digest(), "the delta snapshot is unsealed"
    assert merged.source_sha256 == merged.digest(), "the merged snapshot is unsealed"


def test_join_registry_delta_is_idempotent_over_its_own_rows() -> None:
    """A resumed attempt whose before snapshot already carries the same rows
    (an earlier life of the same cluster) is not a conflict."""

    row = _row("cluster/gpu-b/eks", "gpu-b", arn="arn:aws:eks:::cluster/gpu-b")
    delta, merged = registry_delta(_before(row), [row], site_id="test-site")

    assert delta.resources == [row]
    assert merged.resources == [row]


def test_join_registry_delta_refuses_a_key_naming_another_resource() -> None:
    before = _before(_row("cluster/gpu-b/eks", "gpu-b-old"))

    with pytest.raises(BootstrapError, match="cluster/gpu-b/eks"):
        registry_delta(
            before, [_row("cluster/gpu-b/eks", "gpu-b")], site_id="test-site"
        )
    with pytest.raises(BootstrapError, match="belongs to site"):
        registry_delta(before, [], site_id="other-site")
