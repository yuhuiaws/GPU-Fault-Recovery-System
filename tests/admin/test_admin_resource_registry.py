from __future__ import annotations

import json
import subprocess

import pytest

from gpu_fault import admin_resource_registry
from gpu_fault.admin_resource_registry import (
    LegacyInstallationRegistryMissing,
    build_installation_snapshot,
    fetch_installation_resource_registry,
    find_bootstrap_state,
    load_installation_resource_snapshot,
    write_installation_resource_snapshot,
)
from gpu_fault.admin_site import load_site
from gpu_fault.installation_resources import (
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
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
                "sqs_queue_url": (
                    "https://sqs.us-east-1.amazonaws.com/123456789012/test-alerts"
                ),
                "sqs_queue_arn": ("arn:aws:sqs:us-east-1:123456789012:test-alerts"),
                "sqs_queue_ownership": "CREATED",
                "queue_subscription_arn": (
                    "arn:aws:sns:us-east-1:123456789012:test-alerts:sub"
                ),
                "queue_subscription_ownership": "CREATED",
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
    assert by_key["aws/aurora/cluster"].dependencies == [
        "aws/aurora/security-group",
        "aws/aurora/subnet-group",
    ]
    assert by_key["aws/ecr/runtime"].delete_policy is (
        InstallationResourceDeletePolicy.DELETE
    )
    assert by_key["aws/ecr/cache"].attributes["purpose"] == "build-cache"
    assert (
        by_key["aws/ses/administrator-email-identity"].delete_policy
        is InstallationResourceDeletePolicy.PRESERVE
    )
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
