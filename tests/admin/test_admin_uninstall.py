from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from gpu_fault import admin_aws_cleanup, admin_aws_commands, admin_uninstall
from gpu_fault.admin_aws_cleanup import ordered_aurora_instances
from gpu_fault.admin_bootstrap_common import BootstrapError
from gpu_fault.admin_site import load_site
from gpu_fault.admin_uninstall import (
    UninstallRequest,
    _delete_aurora_last,
    _delete_non_aurora_resources,
    _effective_policy,
    _kubectl_prefix,
    _load_or_export_registry,
    _terminal_resource,
)
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)
from tests.admin.test_admin_site import site_file


def _resource(
    key: str,
    resource_type: str,
    resource_id: str,
    *,
    policy: InstallationResourceDeletePolicy = (
        InstallationResourceDeletePolicy.DELETE
    ),
) -> InstallationResource:
    now = datetime.now(timezone.utc)
    ownership = (
        InstallationResourceOwnership.CREATED
        if policy is not InstallationResourceDeletePolicy.PRESERVE
        else InstallationResourceOwnership.EXTERNAL
    )
    return InstallationResource(
        site_id="test-site",
        resource_key=key,
        resource_type=resource_type,
        resource_id=resource_id,
        region="us-east-1",
        account_id="123456789012",
        ownership=ownership,
        delete_policy=policy,
        created_at=now,
        updated_at=now,
    )


def test_aws_verification_does_not_treat_access_denied_as_absent(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))

    def denied(arguments, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(
            arguments, 254, stdout="", stderr="AccessDenied: not authorized"
        )

    monkeypatch.setattr(admin_aws_commands.subprocess, "run", denied)
    cleaner = admin_aws_cleanup.ResourceCleaner(site)

    with pytest.raises(BootstrapError, match="verification failed"):
        cleaner.exists(_resource("aws/nlb/security-group", "security_group", "sg-test"))


def test_external_ses_identity_is_supported_and_probed(tmp_path, monkeypatch) -> None:
    site = load_site(site_file(tmp_path))
    calls: list[list[str]] = []

    def available(arguments, **kwargs):
        del kwargs
        calls.append(list(arguments))
        return subprocess.CompletedProcess(
            arguments, 0, stdout='{"IdentityType":"EMAIL_ADDRESS"}', stderr=""
        )

    monkeypatch.setattr(admin_aws_commands.subprocess, "run", available)
    cleaner = admin_aws_cleanup.ResourceCleaner(site)
    identity = _resource(
        "aws/ses/administrator-email-identity",
        "ses_email_identity",
        "ops@example.com",
        policy=InstallationResourceDeletePolicy.PRESERVE,
    )

    cleaner.validate_supported([identity])

    assert cleaner.exists(identity) is True
    assert calls == [
        [
            "aws",
            "sesv2",
            "get-email-identity",
            "--region",
            "us-east-1",
            "--email-identity",
            "ops@example.com",
        ]
    ]


def test_last_sqs_topic_binding_clears_the_policy_attribute(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))
    calls: list[list[str]] = []
    topic_arn = "arn:aws:sns:us-east-1:123456789012:test"
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sqs:SendMessage",
                "Condition": {"ArnEquals": {"aws:SourceArn": topic_arn}},
            }
        ],
    }
    queue_reads = iter(
        ({"Attributes": {"Policy": json.dumps(policy)}}, {"Attributes": {}})
    )

    def sqs(arguments, **kwargs):
        del kwargs
        calls.append(list(arguments))
        if "get-queue-attributes" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, stdout=json.dumps(next(queue_reads)), stderr=""
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(admin_aws_commands.subprocess, "run", sqs)
    cleaner = admin_aws_cleanup.ResourceCleaner(site)
    binding = _resource(
        "aws/sqs/topic-policy-binding",
        "sqs_policy_binding",
        "https://sqs.us-east-1.amazonaws.com/123456789012/test",
        policy=InstallationResourceDeletePolicy.DETACH,
    ).model_copy(update={"attributes": {"topic_arn": topic_arn}})

    cleaner.delete(binding)

    set_call = next(call for call in calls if "set-queue-attributes" in call)
    attributes = json.loads(set_call[set_call.index("--attributes") + 1])
    assert attributes == {"Policy": ""}


def test_gpu_cleanup_verification_uses_site_kubeconfig_environment(tmp_path) -> None:
    site = load_site(site_file(tmp_path))
    site = replace(
        site, environment={**site.environment, "KUBECONFIG": "/secure/gpu.kubeconfig"}
    )

    assert _kubectl_prefix(site, plane="gpu", context="gpu-a") == [
        "kubectl",
        "--kubeconfig",
        "/secure/gpu.kubeconfig",
        "--context",
        "gpu-a",
    ], "GPU cleanup verification ignored the dedicated kubeconfig"


def test_aurora_cleanup_runs_after_non_aurora_resources(tmp_path) -> None:
    site = load_site(site_file(tmp_path))
    nlb = _resource("aws/nlb", "nlb", "test-nlb")
    aurora = _resource("aws/aurora/cluster", "aurora_cluster", "test-aurora")
    instance = _resource(
        "aws/aurora/instance/writer", "aurora_instance", "test-aurora-writer"
    )
    snapshot = InstallationResourceSnapshot(
        site_id="test-site", resources=[nlb, aurora, instance]
    )
    calls: list[str] = []

    class Cleaner:
        def delete(self, resource):
            calls.append(resource.resource_key)

        def delete_aurora(
            self, cluster, *, final_snapshot_policy, final_snapshot_identifier
        ):
            del final_snapshot_policy, final_snapshot_identifier
            calls.append(cluster.resource_key)
            return None

        def wait_absent(self, resource, *, timeout_seconds=900):
            del resource, timeout_seconds

    cleaner = Cleaner()
    state_path = tmp_path / "uninstall-state.json"
    state = {"final_snapshot_identifier": "test-final", "phase": "STARTED"}
    _delete_non_aurora_resources(
        cleaner, snapshot, cpu_disposition="keep", state_path=state_path, state=state
    )
    request = UninstallRequest(
        site=site,
        cpu_disposition="keep",
        confirmation="UNINSTALL_GPU_FAULT",
        final_snapshot_policy="skip",
    )
    _delete_aurora_last(cleaner, snapshot, request, state)

    assert calls == ["aws/nlb", "aws/aurora/cluster"]


def test_aurora_instances_are_deleted_readers_before_writer() -> None:
    database = {
        "DBClusterMembers": [
            {"DBInstanceIdentifier": "writer", "IsClusterWriter": True},
            {"DBInstanceIdentifier": "reader-b", "IsClusterWriter": False},
            {"DBInstanceIdentifier": "reader-a", "IsClusterWriter": False},
        ]
    }
    instances = [
        {"DBInstanceIdentifier": "writer"},
        {"DBInstanceIdentifier": "reader-b"},
        {"DBInstanceIdentifier": "reader-a"},
    ]

    ordered = ordered_aurora_instances(database, instances)

    assert [item["DBInstanceIdentifier"] for item in ordered] == [
        "reader-a",
        "reader-b",
        "writer",
    ]


def test_legacy_reused_solution_resource_is_adopted_for_deletion() -> None:
    resource = _resource(
        "aws/sns/topic",
        "sns_topic",
        "arn:aws:sns:us-east-1:123456789012:test",
        policy=InstallationResourceDeletePolicy.PRESERVE,
    ).model_copy(update={"ownership": InstallationResourceOwnership.REUSED})

    policy = _effective_policy(resource, cpu_disposition="keep")
    terminal = _terminal_resource(
        resource, policy=policy, now=datetime.now(timezone.utc)
    )

    assert policy is InstallationResourceDeletePolicy.DELETE
    assert terminal.ownership is InstallationResourceOwnership.CREATED
    assert terminal.delete_policy is InstallationResourceDeletePolicy.DELETE


def test_cpu_bound_addon_is_removed_with_cpu_cluster() -> None:
    addon = _resource(
        "aws/eks/pod-identity-agent",
        "eks_addon",
        "eks-pod-identity-agent",
        policy=InstallationResourceDeletePolicy.PRESERVE,
    )

    assert (
        _effective_policy(addon, cpu_disposition="keep")
        is InstallationResourceDeletePolicy.PRESERVE
    )
    assert (
        _effective_policy(addon, cpu_disposition="delete")
        is InstallationResourceDeletePolicy.DELETE
    )


def test_legacy_online_registry_is_backfilled_without_manual_cleanup(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))
    snapshot = InstallationResourceSnapshot(
        site_id="test-site", resources=[_resource("aws/nlb", "nlb", "test-nlb")]
    )
    direct_syncs = []

    def missing(*_args, **_kwargs):
        raise admin_uninstall.LegacyInstallationRegistryMissing("legacy")

    monkeypatch.setattr(
        admin_uninstall, "fetch_installation_resource_registry", missing
    )
    monkeypatch.setattr(
        admin_uninstall, "build_legacy_installation_snapshot", lambda _site: snapshot
    )
    monkeypatch.setattr(
        admin_uninstall,
        "sync_installation_resource_snapshot_direct",
        lambda _site, value: direct_syncs.append(value),
    )
    monkeypatch.setattr(
        admin_uninstall,
        "write_installation_resource_snapshot",
        lambda _site, _value, *, path=None: path,
    )
    request = UninstallRequest(
        site=site, cpu_disposition="keep", confirmation="UNINSTALL_GPU_FAULT"
    )

    exported = _load_or_export_registry(request, tmp_path / "uninstall")

    assert exported == snapshot
    assert len(direct_syncs) == 1
    assert direct_syncs[0].resources[0].status is (
        admin_uninstall.InstallationResourceStatus.DELETE_PENDING
    )
