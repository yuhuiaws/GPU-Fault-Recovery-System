"""Uninstall after remove-cluster leaves the removed cluster's rows as settled.

Batch 79 noted a possible gap, confirmed here (batch 91): ``remove-cluster``
deletes the removed cluster's IAM roles and records the rest of its rows as
DETACHED -- the EKS/HyperPod clusters it preserves and a CREATED executor OIDC
provider, which it leaves in place -- before ``site.yaml`` drops the cluster.
``uninstall`` ignored those statuses and processed the rows again: the OIDC
row (``attributes.cluster_name`` = the removed EKS cluster) reached the
cleaner's scope check, "cleanup resource refers to a cluster outside the site",
and the uninstall stopped at its first AWS deletion on every rerun; a removed
cluster the customer later deleted stopped it at "GPU cluster was not
preserved". The rows a completed removal settled are now carried as they are.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import aws_commands
from gpu_fault.admin.aws_cleanup_ownership import CleanupOwnership
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal_resources import (
    removal_settled_keys,
    remove_target_resources,
)
from gpu_fault.admin.oidc_provider_release import release_oidc_provider
from gpu_fault.admin.resource_registry import load_installation_resource_snapshot
from gpu_fault.admin.site import load_site
from gpu_fault.admin.uninstall import uninstall
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceSnapshot,
)
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
from gpu_fault.installation_resources import InstallationResourceOwnership as Ownership
from gpu_fault.installation_resources import InstallationResourceStatus as Status
from tests.admin._aws_cleanup_support import TAGS, Aws
from tests.admin.test_uninstall_lifecycle import Harness

ACCOUNT = "123456789012"
EKS_B = f"arn:aws:eks:us-east-1:{ACCOUNT}:cluster/gpu-b"
OIDC_B = f"arn:aws:iam::{ACCOUNT}:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/B"
ROLE_B = "aws/iam/executor/gpu-b/role"
PROVIDER_B = "aws/iam/executor/gpu-b/oidc-provider"
REMOVED_B = {"cluster/gpu-b/eks", "cluster/gpu-b/hyperpod", ROLE_B, PROVIDER_B}


def _row(
    key: str,
    kind: str,
    identifier: str,
    ownership: Ownership,
    policy: Policy,
    **updates: object,
) -> InstallationResource:
    return InstallationResource(
        site_id="test-site",
        resource_key=key,
        resource_type=kind,
        resource_id=identifier,
        region="us-east-1",
        account_id=ACCOUNT,
        ownership=ownership,
        delete_policy=policy,
    ).model_copy(update=updates)


def _joined_b() -> list[InstallationResource]:
    """gpu-b's rows as join-cluster records them, with a site-created provider."""

    return [
        _row(
            "cluster/gpu-b/eks",
            "gpu_eks",
            "gpu-b",
            Ownership.EXTERNAL,
            Policy.PRESERVE,
            resource_arn=EKS_B,
        ),
        _row(
            "cluster/gpu-b/hyperpod",
            "gpu_hyperpod",
            "hp-gpu-b",
            Ownership.EXTERNAL,
            Policy.PRESERVE,
            resource_arn=f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:cluster/hpgpub",
            dependencies=["cluster/gpu-b/eks"],
        ),
        _row(
            ROLE_B,
            "iam_role",
            "gpu-fault-test-site-hp-gpu-b-executor",
            Ownership.CREATED,
            Policy.DELETE,
            resource_arn=(
                f"arn:aws:iam::{ACCOUNT}:role/gpu-fault-test-site-hp-gpu-b-executor"
            ),
            attributes={"inline_policy_name": "GPUFaultRegionalExecutor"},
        ),
        _row(
            PROVIDER_B,
            "iam_oidc_provider",
            OIDC_B,
            Ownership.CREATED,
            Policy.DETACH,
            resource_arn=OIDC_B,
            attributes={"cluster_name": "gpu-b"},
        ),
    ]


def _sealed(resources: list[InstallationResource]) -> InstallationResourceSnapshot:
    value = InstallationResourceSnapshot(site_id="test-site", resources=resources)
    return value.model_copy(update={"source_sha256": value.digest()})


class _RemovalCleaner:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def validate_supported(self, _resources: object) -> None:
        pass

    def delete(self, resource: InstallationResource) -> None:
        self.deleted.append(resource.resource_key)

    def release_oidc_provider(
        self, resource: InstallationResource, **_options: object
    ) -> dict[str, object]:
        # The removal found a customer IRSA role trusting the provider: kept.
        return {
            "resource_key": resource.resource_key,
            "provider_arn": resource.resource_arn,
            "outcome": "kept",
            "trusting_roles": ["customer-irsa"],
        }


def _removed_b(harness: Harness, tmp_path: Path) -> list[InstallationResource]:
    """Run remove-cluster's registry step for gpu-b on the two-cluster site."""

    document = yaml.safe_load(harness.site.source.read_text(encoding="utf-8"))
    cluster = copy.deepcopy(document["spec"]["clusters"][0])
    cluster.update(
        {
            "clusterId": "gpu-b",
            "context": "gpu-b",
            "hyperpodClusterName": "hp-gpu-b",
            "eksClusterArn": EKS_B,
            "executorIrsaRoleArn": (
                f"arn:aws:iam::{ACCOUNT}:role/gpu-fault-test-site-hp-gpu-b-executor"
            ),
        }
    )
    document["spec"]["clusters"].append(cluster)
    two_clusters = tmp_path / "two-clusters" / "site.yaml"
    two_clusters.parent.mkdir()
    two_clusters.write_text(yaml.safe_dump(document, sort_keys=False))
    two_clusters.chmod(0o600)
    cleaner = _RemovalCleaner()
    removed, evidence = remove_target_resources(
        load_site(two_clusters),
        "gpu-b",
        _sealed([*harness.snapshot.resources, *_joined_b()]),
        cleaner_factory=lambda _site: cleaner,
    )
    assert cleaner.deleted == [ROLE_B], "remove-cluster deletes the roles only"
    assert evidence == {
        "deleted": [ROLE_B],
        "detached": sorted(REMOVED_B - {ROLE_B}),
        "oidc_providers": [
            {
                "resource_key": PROVIDER_B,
                "provider_arn": OIDC_B,
                "outcome": "kept",
                "trusting_roles": ["customer-irsa"],
            }
        ],
    }
    return [item for item in removed.resources if item.resource_key in REMOVED_B]


def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    trusted_by: tuple[str, ...] = ("customer-irsa",),
) -> Harness:
    """The uninstall harness over a registry that still holds gpu-b's rows.

    Its fake cleaner applies the real cleaner's scope check first to every row
    bound to a cluster -- the step that refused the removed cluster's provider
    (the harness's own fixture rows are not shaped for the rest of it) -- and
    releases a left provider with the real double proof over IAM fakes in which
    ``trusted_by`` still trust it.
    """

    harness = Harness(tmp_path, monkeypatch)
    removed = _removed_b(harness, tmp_path)
    harness.snapshot = _sealed([*harness.snapshot.resources, *removed])
    # The role is gone; the provider was detached, not deleted; the customer
    # keeps the EKS/HyperPod clusters.
    harness.existing = {item.resource_key for item in harness.snapshot.resources} - {
        ROLE_B
    }
    fake_delete = harness.delete
    scope = CleanupOwnership(harness.site)

    def delete(resource: InstallationResource) -> None:
        if "cluster_name" in resource.attributes:
            scope.validate_scope(resource)
        fake_delete(resource)

    monkeypatch.setattr(harness, "delete", delete)
    trust = {"Statement": [{"Principal": {"Federated": OIDC_B}}]}

    def deleted(_arguments: list[str]) -> dict[str, object]:
        harness.existing.discard(PROVIDER_B)
        return {}

    aws = Aws(
        {
            ("iam", "list-open-id-connect-provider-tags"): {"Tags": TAGS},
            ("iam", "list-roles"): {
                "Roles": [
                    {"RoleName": name, "AssumeRolePolicyDocument": trust}
                    for name in trusted_by
                ]
            },
            ("iam", "delete-open-id-connect-provider"): deleted,
        }
    )
    monkeypatch.setattr(aws_commands, "bounded_command", aws)

    def release(resource: InstallationResource, **options: Any) -> dict[str, Any]:
        return release_oidc_provider(
            resource,
            ownership=CleanupOwnership(harness.site),
            exists=harness.exists,
            wait_absent=harness.wait_absent,
            **options,
        )

    monkeypatch.setattr(harness, "release_oidc_provider", release, raising=False)
    return harness


def _final(harness: Harness) -> dict[str, Status]:
    final = load_installation_resource_snapshot(
        harness.site.source.parent / "uninstall" / "installation-resources-final.json"
    )
    return {item.resource_key: item.status for item in final.resources}


@pytest.mark.parametrize("customer_deleted_cluster", [False, True])
def test_uninstall_after_removing_a_cluster_with_a_created_oidc_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    customer_deleted_cluster: bool,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    if customer_deleted_cluster:
        harness.existing -= {"cluster/gpu-b/eks", "cluster/gpu-b/hyperpod"}

    result = uninstall(harness.request(), runner=harness)

    (verdict,) = result["oidc_provider_releases"]
    assert (verdict["outcome"], verdict["trusting_roles"]) == (
        "kept",
        ["customer-irsa"],
    )
    assert harness.state()["oidc_provider_releases"] == [verdict], "journaled"
    finding = capsys.readouterr().err
    assert f"finding: OIDC provider {OIDC_B}" in finding
    assert f"--open-id-connect-provider-arn {OIDC_B}" in finding

    assert harness.state()["phase"] == "COMPLETED"
    assert result["registry_entries_settled_by_removal"] == sorted(REMOVED_B)
    assert not [event for event in harness.events if "gpu-b" in event], (
        "uninstall settled a removed cluster's row a second time"
    )
    assert PROVIDER_B in harness.existing, "a still-trusted provider stays"
    statuses = _final(harness)
    assert {key: statuses[key] for key in REMOVED_B} == {
        "cluster/gpu-b/eks": Status.PRESERVED,
        "cluster/gpu-b/hyperpod": Status.PRESERVED,
        ROLE_B: Status.DELETED,
        PROVIDER_B: Status.DETACHED,
    }
    assert result["gpu_cluster_records_verified"] == 2, "only gpu-a is probed"


def test_uninstall_deletes_a_left_provider_once_nothing_trusts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, trusted_by=())

    result = uninstall(harness.request(), runner=harness)

    (verdict,) = result["oidc_provider_releases"]
    assert verdict["outcome"] == "deleted" and verdict["site_tagged"] is True
    assert PROVIDER_B not in harness.existing, "the provider was not deleted"
    assert _final(harness)[PROVIDER_B] is Status.DETACHED


def test_a_removed_clusters_row_that_is_not_settled_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a terminal row is settled: an ACTIVE row of a cluster the site no
    longer lists (a removal that never reached its registry update, a stale
    write) keeps failing closed."""

    harness = _harness(tmp_path, monkeypatch)
    harness.snapshot = _sealed(
        [
            item.model_copy(update={"status": Status.ACTIVE})
            if item.resource_key == PROVIDER_B
            else item
            for item in harness.snapshot.resources
        ]
    )

    with pytest.raises(BootstrapError, match="refers to a cluster outside the site"):
        uninstall(harness.request(), runner=harness)
    assert harness.state()["phase"] == "NON_AURORA_DELETE_IN_PROGRESS"
    assert "delete:aurora" not in harness.events


def test_settled_rows_are_the_terminal_rows_of_clusters_the_site_no_longer_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    member = _row(
        "aws/iam/executor/gpu-a/oidc-provider",
        "iam_oidc_provider",
        OIDC_B.replace("/B", "/A"),
        Ownership.CREATED,
        Policy.DETACH,
        status=Status.DETACHED,
        attributes={"cluster_name": "gpu-a"},
    )
    unrelated = _row(
        "aws/iam/control-plane/role",
        "iam_role",
        "control-plane",
        Ownership.CREATED,
        Policy.DELETE,
        status=Status.DELETED,
    )
    snapshot = _sealed([*harness.snapshot.resources, member, unrelated])

    assert removal_settled_keys(harness.site, snapshot) == frozenset(REMOVED_B)
