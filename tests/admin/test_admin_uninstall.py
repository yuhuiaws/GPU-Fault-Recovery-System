from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu_fault.admin import aws_cleanup as admin_aws_cleanup
from gpu_fault.admin import aws_commands as admin_aws_commands
from gpu_fault.admin import uninstall as admin_uninstall
from gpu_fault.admin.aws_cleanup import ordered_aurora_instances
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.grafana import CREATED_TAG_KEY, CREATED_TAG_VALUE
from gpu_fault.admin.site import load_site
from gpu_fault.admin.uninstall import (
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
from tests.admin.test_uninstall_lifecycle import STATE, Harness


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

    monkeypatch.setattr(admin_aws_commands, "bounded_command", denied)
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

    monkeypatch.setattr(admin_aws_commands, "bounded_command", available)
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


def test_created_ecr_repository_is_deleted_and_verified(tmp_path, monkeypatch) -> None:
    site = load_site(site_file(tmp_path))
    calls: list[list[str]] = []
    state = {"deleted": False}

    def ecr(arguments, **kwargs):
        del kwargs
        calls.append(list(arguments))
        if "get-caller-identity" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, stdout='{"Account":"123456789012"}', stderr=""
            )
        if "list-tags-for-resource" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout='{"tags":{"gpu-fault:site-id":"test-site"}}',
                stderr="",
            )
        if "delete-repository" in arguments:
            state["deleted"] = True
        if "describe-repositories" in arguments and state["deleted"]:
            return subprocess.CompletedProcess(
                arguments, 254, stdout="", stderr="RepositoryNotFoundException"
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="{}", stderr="")

    monkeypatch.setattr(admin_aws_commands, "bounded_command", ecr)
    cleaner = admin_aws_cleanup.ResourceCleaner(site)
    repository = _resource(
        "aws/ecr/runtime", "ecr_repository", "gpu-fault/runtime-test"
    )

    cleaner.validate_supported([repository])
    cleaner.delete(repository)

    delete = next(call for call in calls if "delete-repository" in call)
    assert "--force" in delete
    assert delete[delete.index("--repository-name") + 1] == ("gpu-fault/runtime-test")


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
    queue = {"policy": json.dumps(policy)}

    def sqs(arguments, **kwargs):
        del kwargs
        calls.append(list(arguments))
        if "get-caller-identity" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, stdout='{"Account":"123456789012"}', stderr=""
            )
        if "get-queue-attributes" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=json.dumps({"Attributes": {"Policy": queue["policy"]}}),
                stderr="",
            )
        if "set-queue-attributes" in arguments:
            queue["policy"] = json.loads(
                arguments[arguments.index("--attributes") + 1]
            )["Policy"]
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(admin_aws_commands, "bounded_command", sqs)
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


def test_legacy_alerts_queue_rows_are_still_deleted_by_uninstall(
    tmp_path, monkeypatch
) -> None:
    """Bootstrap stopped creating the alerts queue, but sites installed before
    that still own one. A registry snapshot carrying the old ``sqs_queue`` and
    ``sqs_policy_binding`` rows must detach the binding first and then delete
    the queue, exactly as before."""

    site = load_site(site_file(tmp_path))
    topic_arn = "arn:aws:sns:us-east-1:123456789012:test-alerts"
    queue_url = "https://sqs.us-east-1.amazonaws.com/123456789012/test-alerts"
    queue = {
        "deleted": False,
        "policy": json.dumps(
            {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "sqs:SendMessage",
                        "Condition": {"ArnEquals": {"aws:SourceArn": topic_arn}},
                    }
                ]
            }
        ),
    }
    calls: list[list[str]] = []

    def sqs(arguments, **kwargs):
        del kwargs
        calls.append(list(arguments))
        if "get-caller-identity" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, stdout='{"Account":"123456789012"}', stderr=""
            )
        if "sqs" not in arguments:
            raise AssertionError(arguments)
        if queue["deleted"]:
            return subprocess.CompletedProcess(
                arguments,
                254,
                stdout="",
                stderr="An error occurred (AWS.SimpleQueueService.NonExistentQueue)",
            )
        if "get-queue-attributes" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=json.dumps({"Attributes": {"Policy": queue["policy"]}}),
                stderr="",
            )
        if "list-queue-tags" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout='{"Tags":{"gpu-fault:site-id":"test-site"}}',
                stderr="",
            )
        if "set-queue-attributes" in arguments:
            attributes = json.loads(arguments[arguments.index("--attributes") + 1])
            queue["policy"] = attributes["Policy"]
        elif "delete-queue" in arguments:
            queue["deleted"] = True
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(admin_aws_commands, "bounded_command", sqs)
    cleaner = admin_aws_cleanup.ResourceCleaner(site)
    binding = _resource(
        "aws/sqs/topic-policy-binding",
        "sqs_policy_binding",
        queue_url,
        policy=InstallationResourceDeletePolicy.DETACH,
    ).model_copy(update={"attributes": {"topic_arn": topic_arn}})
    snapshot = InstallationResourceSnapshot(
        site_id="test-site",
        resources=[_resource("aws/sqs/queue", "sqs_queue", queue_url), binding],
    )

    _delete_non_aurora_resources(
        cleaner,
        snapshot,
        cpu_disposition="keep",
        state_path=tmp_path / "uninstall-state.json",
        state={"phase": "STARTED"},
    )

    operations = [call[2] for call in calls]
    assert "set-queue-attributes" in operations, "binding was never detached"
    assert "delete-queue" in operations, "legacy queue was never deleted"
    assert operations.index("set-queue-attributes") < operations.index(
        "delete-queue"
    ), "queue deleted before its policy binding was detached"
    assert queue["deleted"] is True


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
    """The cluster parameter group is an Aurora-phase resource: RDS refuses to
    delete a group a cluster still uses, so it goes after the cluster, never in
    the non-Aurora pass."""

    site = load_site(site_file(tmp_path))
    nlb = _resource("aws/nlb", "nlb", "test-nlb")
    aurora = _resource("aws/aurora/cluster", "aurora_cluster", "test-aurora")
    instance = _resource(
        "aws/aurora/instance/writer", "aurora_instance", "test-aurora-writer"
    )
    parameter_group = _resource(
        "aws/aurora/parameter-group", "rds_cluster_parameter_group", "test-aurora-pg"
    )
    snapshot = InstallationResourceSnapshot(
        site_id="test-site", resources=[nlb, parameter_group, aurora, instance]
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
    state["phase"] = "READY_TO_DELETE_AURORA"
    request = UninstallRequest(
        site=site,
        cpu_disposition="keep",
        confirmation="UNINSTALL_GPU_FAULT",
        reset_database=True,
    )
    _delete_aurora_last(cleaner, snapshot, request, state)

    assert calls == ["aws/nlb", "aws/aurora/cluster", "aws/aurora/parameter-group"]


def test_the_cluster_parameter_group_is_deleted_by_name_and_absence_is_fine(
    tmp_path, monkeypatch
) -> None:
    """One ``delete-db-cluster-parameter-group`` in the site's region, then the
    probe confirms it is gone. A group that is already absent (a re-run after a
    half-finished uninstall) is success, not an error."""

    site = load_site(site_file(tmp_path))
    calls: list[list[str]] = []
    state = {"deleted": False}

    def rds(arguments, **kwargs):
        del kwargs
        calls.append(list(arguments))
        if "get-caller-identity" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, stdout='{"Account":"123456789012"}', stderr=""
            )
        if "list-tags-for-resource" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout='{"TagList":{"gpu-fault:site-id":"test-site"}}',
                stderr="",
            )
        if "delete-db-cluster-parameter-group" in arguments:
            state["deleted"] = True
        if "describe-db-cluster-parameter-groups" in arguments and state["deleted"]:
            return subprocess.CompletedProcess(
                arguments,
                254,
                stdout="",
                stderr="An error occurred (DBParameterGroupNotFound) ...",
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="{}", stderr="")

    monkeypatch.setattr(admin_aws_commands, "bounded_command", rds)
    cleaner = admin_aws_cleanup.ResourceCleaner(site)
    group = _resource(
        "aws/aurora/parameter-group", "rds_cluster_parameter_group", "test-aurora-pg"
    )

    cleaner.validate_supported([group])
    assert admin_aws_cleanup.is_aurora_resource(group), (
        "the parameter group belongs to the Aurora deletion phase"
    )
    cleaner.delete(group)

    deletion = next(
        call for call in calls if "delete-db-cluster-parameter-group" in call
    )
    assert deletion == [
        "aws",
        "rds",
        "delete-db-cluster-parameter-group",
        "--region",
        "us-east-1",
        "--db-cluster-parameter-group-name",
        "test-aurora-pg",
    ]
    assert calls.index(deletion) > next(
        index for index, call in enumerate(calls) if "list-tags-for-resource" in call
    )

    def already_gone(arguments, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(
            arguments,
            254,
            stdout="",
            stderr="An error occurred (DBParameterGroupNotFound) ...",
        )

    monkeypatch.setattr(admin_aws_commands, "bounded_command", already_gone)

    cleaner.delete(group)


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


def test_created_grafana_workspace_is_deleted_and_absence_is_fine(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))
    calls: list[list[str]] = []
    state = {"workspace_deleted": False, "account_deleted": False}

    def grafana(arguments, **kwargs):
        del kwargs
        calls.append(list(arguments))
        if "get-caller-identity" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, stdout='{"Account":"123456789012"}', stderr=""
            )
        if "list-tags-for-resource" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=json.dumps(
                    {
                        "tags": {
                            "gpu-fault:site-id": "test-site",
                            CREATED_TAG_KEY: CREATED_TAG_VALUE,
                        }
                    }
                ),
                stderr="",
            )
        if "list-workspace-service-accounts" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=json.dumps(
                    {
                        "serviceAccounts": []
                        if state["account_deleted"]
                        else [{"id": "9", "name": "gpu-fault-test"}]
                    }
                ),
                stderr="",
            )
        if "delete-workspace-service-account" in arguments:
            state["account_deleted"] = True
        if "delete-workspace" in arguments:
            state["workspace_deleted"] = True
        if "describe-workspace" in arguments and state["workspace_deleted"]:
            return subprocess.CompletedProcess(
                arguments,
                254,
                stdout="",
                stderr="An error occurred (ResourceNotFoundException) ...",
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="{}", stderr="")

    monkeypatch.setattr(admin_aws_commands, "bounded_command", grafana)
    cleaner = admin_aws_cleanup.ResourceCleaner(site)
    workspace = _resource("aws/grafana/workspace", "grafana_workspace", "g-created01")
    account = _resource(
        "aws/grafana/service-account", "grafana_service_account", "9"
    ).model_copy(
        update={"attributes": {"workspace_id": "g-created01", "name": "gpu-fault-test"}}
    )

    cleaner.validate_supported([workspace, account])
    cleaner.delete(account)
    cleaner.delete(workspace)

    account_delete = next(
        call for call in calls if "delete-workspace-service-account" in call
    )
    assert account_delete[1] == "grafana"
    assert account_delete[account_delete.index("--workspace-id") + 1] == "g-created01"
    assert account_delete[account_delete.index("--service-account-id") + 1] == "9"
    delete = next(
        call
        for call in calls
        if "delete-workspace" in call and "delete-workspace-service-account" not in call
    )
    assert delete[1] == "grafana", "the Grafana workspace was deleted through amp"
    assert delete[delete.index("--workspace-id") + 1] == "g-created01"
    assert delete[delete.index("--region") + 1] == "us-east-1"


def test_the_hyperpod_owned_grafana_workspace_is_preserved_by_uninstall() -> None:
    """``g-5b81a13d97`` was created by the HyperPod observability component; deploy
    only tags it, so the registry records it as EXTERNAL and uninstall must not
    adopt it the way it adopts REUSED solution resources."""

    workspace = _resource(
        "aws/grafana/workspace",
        "grafana_workspace",
        "g-5b81a13d97",
        policy=InstallationResourceDeletePolicy.PRESERVE,
    )

    policy = _effective_policy(workspace, cpu_disposition="delete")
    terminal = _terminal_resource(
        workspace, policy=policy, now=datetime.now(timezone.utc)
    )

    assert workspace.ownership is InstallationResourceOwnership.EXTERNAL
    assert policy is InstallationResourceDeletePolicy.PRESERVE
    assert terminal.status is admin_uninstall.InstallationResourceStatus.PRESERVED


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
    assert terminal.immutable_identity() == resource.immutable_identity()
    assert terminal.status is admin_uninstall.InstallationResourceStatus.DELETED


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
        site_id="test-site",
        resources=[
            _resource("aws/nlb", "nlb", "test-nlb"),
            _resource("cluster/cpu-eks", "cpu_eks", "control").model_copy(
                update={"resource_arn": site.release_config["cpu_eks_arn"]}
            ),
            _resource("cluster/cpu-hyperpod", "cpu_hyperpod", "control"),
            _resource("cluster/gpu-a/eks", "gpu_eks", "gpu-a").model_copy(
                update={
                    "resource_arn": site.release_config["clusters"][0][
                        "eks_cluster_arn"
                    ]
                }
            ),
            _resource("cluster/gpu-a/hyperpod", "gpu_hyperpod", "hp-gpu-a"),
            _resource("aws/aurora/cluster", "aurora_cluster", "gpu-fault-aurora"),
        ],
    )
    snapshot = snapshot.model_copy(update={"source_sha256": snapshot.digest()})
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


def _aurora_stack() -> list[InstallationResource]:
    return [
        _resource("aws/aurora/cluster", "aurora_cluster", "test-aurora"),
        _resource(
            "aws/aurora/instance/writer", "aurora_instance", "test-aurora-writer"
        ),
        _resource("aws/aurora/secret", "rds_managed_secret", "test-aurora-secret"),
        _resource(
            "aws/aurora/subnet-group", "rds_db_subnet_group", "test-aurora-subnets"
        ),
        _resource("aws/aurora/security-group", "security_group", "sg-aurora"),
        _resource(
            "aws/aurora/parameter-group", "rds_cluster_parameter_group", "test-pg"
        ),
    ]


def test_keep_mode_is_a_reinstall_and_preserves_the_aurora_stack() -> None:
    """``--cpu-cluster keep`` keeps the site's incident, workflow and registry
    records: every Aurora-phase resource stays, whatever the registry's own
    delete policy says, and even for a legacy REUSED record."""

    for resource in _aurora_stack():
        for ownership in (
            InstallationResourceOwnership.CREATED,
            InstallationResourceOwnership.REUSED,
        ):
            candidate = resource.model_copy(update={"ownership": ownership})
            assert (
                _effective_policy(candidate, cpu_disposition="keep")
                is InstallationResourceDeletePolicy.PRESERVE
            ), f"{candidate.resource_key} ({ownership.value}) must be preserved"


def test_reset_database_deletes_the_aurora_stack_in_keep_mode() -> None:
    for resource in _aurora_stack():
        assert (
            _effective_policy(resource, cpu_disposition="keep", reset_database=True)
            is InstallationResourceDeletePolicy.DELETE
        ), f"{resource.resource_key} must be deleted by --reset-database"


def test_delete_mode_still_deletes_the_aurora_stack() -> None:
    for resource in _aurora_stack():
        assert (
            _effective_policy(resource, cpu_disposition="delete")
            is InstallationResourceDeletePolicy.DELETE
        ), f"{resource.resource_key} must be deleted on retirement"


def test_keep_mode_never_calls_delete_db_cluster(tmp_path) -> None:
    site = load_site(site_file(tmp_path))
    snapshot = InstallationResourceSnapshot(
        site_id="test-site",
        resources=[_resource("aws/nlb", "nlb", "x"), *_aurora_stack()],
    )
    calls: list[str] = []

    class Cleaner:
        def delete(self, resource):
            calls.append(f"delete:{resource.resource_key}")

        def delete_aurora(self, cluster, **_kwargs):
            calls.append(f"delete-db-cluster:{cluster.resource_id}")
            return None

        def wait_absent(self, resource, *, timeout_seconds=900):
            del resource, timeout_seconds

    request = UninstallRequest(
        site=site, cpu_disposition="keep", confirmation="UNINSTALL_GPU_FAULT"
    )
    state = {
        "final_snapshot_identifier": "test-final",
        "phase": "READY_TO_DELETE_AURORA",
    }

    retained = _delete_aurora_last(Cleaner(), snapshot, request, state)

    assert retained is None, "no final snapshot exists when nothing was deleted"
    assert calls == [], "a reinstall must not touch the Aurora stack"


def test_reset_database_runs_the_aurora_phase(tmp_path) -> None:
    site = load_site(site_file(tmp_path))
    snapshot = InstallationResourceSnapshot(
        site_id="test-site", resources=_aurora_stack()
    )
    calls: list[str] = []

    class Cleaner:
        def delete(self, resource):
            calls.append(f"delete:{resource.resource_key}")

        def delete_aurora(self, cluster, *, final_snapshot_policy, **_kwargs):
            calls.append(
                f"delete-db-cluster:{cluster.resource_id}:{final_snapshot_policy}"
            )
            return "test-final"

        def wait_absent(self, resource, *, timeout_seconds=900):
            del resource, timeout_seconds

    request = UninstallRequest(
        site=site,
        cpu_disposition="keep",
        confirmation="UNINSTALL_GPU_FAULT",
        reset_database=True,
    )
    state = {
        "final_snapshot_identifier": "test-final",
        "phase": "READY_TO_DELETE_AURORA",
    }

    retained = _delete_aurora_last(Cleaner(), snapshot, request, state)

    assert retained is not None, "the retained final snapshot must be reported"
    assert retained.resource_id == "test-final", "snapshot id comes from the cleaner"
    assert calls[0] == "delete-db-cluster:test-aurora:retain", (
        "the cluster goes first, with the default retain policy"
    )
    assert "delete:aws/aurora/parameter-group" in calls, (
        "the parameter group is deleted after the cluster"
    )


@pytest.mark.parametrize(
    ("disposition", "overrides", "message"),
    (
        ("keep", {"final_snapshot_policy": "skip"}, "--cpu-cluster delete"),
        ("delete", {"reset_database": True}, "--cpu-cluster keep"),
    ),
)
def test_uninstall_request_refuses_contradictory_flags(
    tmp_path, disposition, overrides, message
) -> None:
    site = load_site(site_file(tmp_path))

    with pytest.raises(BootstrapError, match=message):
        UninstallRequest(
            site=site, cpu_disposition=disposition, confirmation="x", **overrides
        )


@pytest.mark.parametrize("site_tag", ["test-site", "another-site"])
def test_legacy_effective_delete_requires_live_ownership_and_keeps_registry_identity(
    tmp_path, monkeypatch, site_tag
) -> None:
    site = load_site(site_file(tmp_path))
    resource = _resource(
        "aws/sns/topic",
        "sns_topic",
        "arn:aws:sns:us-east-1:123456789012:test",
        policy=InstallationResourceDeletePolicy.PRESERVE,
    ).model_copy(update={"ownership": InstallationResourceOwnership.REUSED})
    snapshot = InstallationResourceSnapshot(site_id="test-site", resources=[resource])
    calls = []
    state = {"deleted": False}

    def aws(arguments, **_kwargs):
        calls.append(arguments)
        if "get-caller-identity" in arguments:
            output = {"Account": "123456789012"}
        elif "list-tags-for-resource" in arguments:
            output = {"Tags": {"gpu-fault:site-id": site_tag}}
        elif "delete-topic" in arguments:
            state["deleted"] = True
            output = {}
        elif "get-topic-attributes" in arguments and state["deleted"]:
            return subprocess.CompletedProcess(
                arguments, 254, stdout="", stderr="An error occurred (NotFound)"
            )
        else:
            output = {"Attributes": {}}
        return subprocess.CompletedProcess(
            arguments, 0, stdout=json.dumps(output), stderr=""
        )

    monkeypatch.setattr(admin_aws_commands, "bounded_command", aws)
    cleaner = admin_aws_cleanup.ResourceCleaner(site)
    arguments = dict(
        cpu_disposition="keep",
        state_path=tmp_path / "state.json",
        state={"phase": "STARTED"},
    )
    if site_tag == "test-site":
        _delete_non_aurora_resources(cleaner, snapshot, **arguments)
        assert state["deleted"]
    else:
        with pytest.raises(BootstrapError):
            _delete_non_aurora_resources(cleaner, snapshot, **arguments)
        assert not state["deleted"]
    assert resource.ownership is InstallationResourceOwnership.REUSED
    assert resource.delete_policy is InstallationResourceDeletePolicy.PRESERVE
    assert any("list-tags-for-resource" in call for call in calls), (
        "legacy resource deletion must check live ownership tags"
    )


def test_a_failed_kubernetes_cleanup_record_is_resumed_without_rebinding(
    tmp_path, monkeypatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    request = harness.request()
    path = harness.site.source.parent / "uninstall/kubernetes-cleanup.json"
    harness.fail_cleanup = BootstrapError("injected cleanup interruption")
    run = harness.run
    failed_documents = []
    resumed_paths = []

    def cleanup(arguments, **keywords):
        if Path(arguments[0]).name != "prepare-clean-redeploy.sh":
            return run(arguments, **keywords)
        state_file = Path(arguments[arguments.index("--state-file") + 1])
        assert state_file == path
        if not failed_documents:
            try:
                return run(arguments, **keywords)
            except BootstrapError:
                document = STATE.read_state(path)
                STATE.record_resource(
                    document,
                    resource_scope="gpu:gpu-a",
                    context="gpu-a",
                    kind="namespace",
                    name="gpu-fault-system",
                    previous="original-namespace",
                )
                for phase in (
                    "PREFLIGHT",
                    "CLUSTERS_DRAINING",
                    "GPU_DATA_PLANE_SOURCES_STOPPED",
                ):
                    STATE.transition(
                        document, phase=phase, status="COMPLETED", message="fixture"
                    )
                STATE.transition(
                    document,
                    phase="QUEUES_DRAINED",
                    status="FAILED",
                    message="injected cleanup interruption",
                )
                STATE.atomic_write(path, document)
                failed_documents.append(STATE.read_state(path))
                raise
        document = STATE.read_state(path)
        assert document == failed_documents[0], (
            "uninstall rebound or replaced the failed cleanup journal before retry"
        )
        resumed_paths.append(state_file)
        harness.events.append("cleanup")
        completed = set(STATE.completed_phases(document))
        for phase in STATE.required_phases(document):
            if phase not in completed:
                STATE.transition(
                    document, phase=phase, status="COMPLETED", message="fixture"
                )
        STATE.transition(
            document, phase="CLEANUP_COMPLETED", status="COMPLETED", message="fixture"
        )
        STATE.atomic_write(path, document)
        return ""

    monkeypatch.setattr(harness, "run", cleanup)
    with pytest.raises(BootstrapError, match="injected cleanup interruption"):
        admin_uninstall.uninstall(request, runner=harness)
    assert harness.state()["phase"] == "REGISTRY_EXPORTED"
    assert harness.events == ["cleanup", "cleanup-owned"]
    original = failed_documents[0]
    assert original["phase"] == "QUEUES_DRAINED"
    assert original["status"] == "FAILED"
    assert STATE.completed_phases(original) == [
        "PREFLIGHT",
        "CLUSTERS_DRAINING",
        "GPU_DATA_PLANE_SOURCES_STOPPED",
    ]

    admin_uninstall.uninstall(request, runner=harness)

    result = STATE.read_state(path)
    assert harness.state()["phase"] == "COMPLETED"
    assert result["original_resources"] == original["original_resources"]
    assert result["run_id"] == original["run_id"]
    assert result["config_sha256"] == original["config_sha256"]
    assert result["phase_order"] == original["phase_order"]
    assert result["history"][: len(original["history"])] == original["history"]
    assert resumed_paths == [path]
    assert list(path.parent.glob("kubernetes-cleanup*.json")) == [path]
    assert harness.exports == 1, "retry exported a new registry instead of resuming"
    assert harness.events.count("cleanup") == 2
    assert harness.events.index("delete:aws/nlb") > harness.events.index(
        "cleanup-owned"
    )


def test_a_reset_reinstall_may_skip_the_final_aurora_snapshot(tmp_path) -> None:
    """``keep`` alone keeps Aurora, so a skipped snapshot is meaningless and
    refused; ``keep --reset-database`` deletes it like a retirement does and
    may skip the audit snapshot (the live uninstall of 2026-09-12 spent most
    of its Aurora phase on snapshots nobody asked for)."""

    site = load_site(site_file(tmp_path))
    with pytest.raises(BootstrapError, match="only valid with --cpu-cluster delete"):
        admin_uninstall.UninstallRequest(
            site=site,
            cpu_disposition="keep",
            confirmation="UNINSTALL_GPU_FAULT",
            final_snapshot_policy="skip",
        )
    request = admin_uninstall.UninstallRequest(
        site=site,
        cpu_disposition="keep",
        confirmation="UNINSTALL_GPU_FAULT",
        final_snapshot_policy="skip",
        reset_database=True,
    )
    assert request.final_snapshot_policy == "skip"


def test_certificate_deletion_waits_for_acm_to_release_the_listener(
    tmp_path, monkeypatch
) -> None:
    """Live 2026-09-13: the NLB was already gone, but ACM still answered
    ResourceInUseException for its listener ten minutes later and the uninstall
    stopped at the certificate. That one refusal is retried until it clears."""

    site = load_site(site_file(tmp_path))
    deletes: list[int] = [0]

    def acm(arguments, **kwargs):
        del kwargs
        if "get-caller-identity" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, '{"Account":"123456789012"}', ""
            )
        if "list-tags-for-certificate" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                '{"Tags":[{"Key":"gpu-fault:site-id","Value":"test-site"}]}',
                "",
            )
        if "delete-certificate" in arguments:
            deletes[0] += 1
            if deletes[0] <= 2:
                return subprocess.CompletedProcess(
                    arguments,
                    254,
                    stdout="",
                    stderr="An error occurred (ResourceInUseException) ... is in use.",
                )
            return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")
        if "describe-certificate" in arguments:
            if deletes[0] < 3:
                return subprocess.CompletedProcess(
                    arguments, 0, '{"Certificate":{}}', ""
                )
            return subprocess.CompletedProcess(
                arguments, 254, stdout="", stderr="ResourceNotFoundException"
            )
        raise AssertionError(f"unexpected aws call: {arguments}")

    monkeypatch.setattr(admin_aws_commands, "bounded_command", acm)
    monkeypatch.setattr(admin_aws_commands.time, "sleep", lambda _seconds: None)
    cleaner = admin_aws_cleanup.ResourceCleaner(site)
    certificate = _resource(
        "aws/acm/certificate",
        "acm_certificate",
        "arn:aws:acm:us-east-1:123456789012:certificate/test",
    )

    cleaner.delete(certificate)

    assert deletes[0] == 3, (
        "two in-use refusals were retried, the third delete went through"
    )


def test_certificate_deletion_still_raises_on_any_other_failure(
    tmp_path, monkeypatch
) -> None:
    site = load_site(site_file(tmp_path))

    def acm(arguments, **kwargs):
        del kwargs
        if "get-caller-identity" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, '{"Account":"123456789012"}', ""
            )
        if "list-tags-for-certificate" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                '{"Tags":[{"Key":"gpu-fault:site-id","Value":"test-site"}]}',
                "",
            )
        if "describe-certificate" in arguments:
            return subprocess.CompletedProcess(arguments, 0, '{"Certificate":{}}', "")
        assert "delete-certificate" in arguments
        return subprocess.CompletedProcess(
            arguments, 254, stdout="", stderr="AccessDeniedException: no"
        )

    monkeypatch.setattr(admin_aws_commands, "bounded_command", acm)
    cleaner = admin_aws_cleanup.ResourceCleaner(site)

    with pytest.raises(BootstrapError, match="AccessDeniedException"):
        cleaner.delete(
            _resource(
                "aws/acm/certificate",
                "acm_certificate",
                "arn:aws:acm:us-east-1:123456789012:certificate/test",
            )
        )


def test_a_fresh_session_over_a_prior_sessions_working_files_is_refused(
    tmp_path, monkeypatch
) -> None:
    """A prior uninstall's working files (``kubernetes-cleanup.json`` still
    CLEANUP_COMPLETED, a registry snapshot of the retired install) must never be
    trusted by a later session of a since-redeployed site: doing so skipped the
    real cleanup and then failed verification against resources the redeploy
    recreated (live 2026-09-15 ``installed resource still exists:
    cpu:cpu:deployment/gpu-fault-api-ha``). A completed uninstall is retired
    whole (``retire_completed_site`` moves the ``uninstall/`` directory into the
    ``retired-*`` archive on the next deploy), so a fresh session starts in an
    empty directory; working files without their ``state.json`` journal are an
    inconsistent transaction and the uninstall refuses before it reads them."""

    harness = Harness(tmp_path, monkeypatch)
    request = harness.request()
    uninstall_dir = harness.site.source.parent / "uninstall"
    uninstall_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    cleanup_path = uninstall_dir / "kubernetes-cleanup.json"
    stale = {"phase": "CLEANUP_COMPLETED", "status": "COMPLETED", "session": "PRIOR"}
    cleanup_path.write_text(json.dumps(stale), encoding="utf-8")
    before_path = uninstall_dir / "installation-resources-before.json"
    before_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        admin_uninstall,
        "_run_cleanup",
        lambda *_args, **_kwargs: pytest.fail("cleanup ran over a stale session"),
    )

    with pytest.raises(BootstrapError, match="lack their original journal"):
        admin_uninstall.uninstall(request, runner=harness)

    assert not (uninstall_dir / "state.json").exists(), (
        "the refused session must not open a journal beside the stale files"
    )
    assert json.loads(cleanup_path.read_text(encoding="utf-8")) == stale, (
        "the stale record is evidence for the operator and must stay untouched"
    )
    assert harness.exports == 0, "the stale registry snapshot must not be re-read"
