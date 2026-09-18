from __future__ import annotations

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.installation_resources import (
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
)
from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_native as adapter
from scripts.e2e.regional import boot032_observe as observe
from tests.regional._cov95_boot032_membership import cross_region_world
from tests.regional._cov95_boot032_world import World


def add_resource(world, site, *, kind, identifier):
    original = world.snapshots[site.metadata_name]
    target = site == world.target
    row = original.resources[-1].model_copy(
        update={
            "resource_key": "aws/shared-identity-control",
            "resource_type": kind,
            "resource_id": "shared-name",
            "resource_arn": identifier,
            "ownership": InstallationResourceOwnership.CREATED
            if target
            else InstallationResourceOwnership.EXTERNAL,
            "delete_policy": InstallationResourceDeletePolicy.DELETE
            if target
            else InstallationResourceDeletePolicy.PRESERVE,
        }
    )
    snapshot = InstallationResourceSnapshot(
        site_id=original.site_id, resources=[*original.resources, row]
    )
    world.snapshots[site.metadata_name] = snapshot.model_copy(
        update={"source_sha256": snapshot.digest()}
    )


def test_same_global_iam_arn_across_two_legal_regions_blocks_uninstall(
    tmp_path, monkeypatch
):
    world = cross_region_world(tmp_path, monkeypatch)
    shared = "arn:aws:iam::123456789012:role/shared-accepted-role"
    for site in (world.target, world.protected):
        add_resource(world, site, kind="iam_role", identifier=shared)
    with pytest.raises(contract.UninstallCaseError, match="overlaps"):
        adapter.NativeBackend(world.settings).initial()
    assert not world.settings.native_dir.exists(), (
        "global IAM role referenced by the accepted site must never become a sacrificial deletion target"
    )


def test_same_regional_resource_name_in_distinct_regions_remains_disjoint(
    tmp_path, monkeypatch
):
    world = cross_region_world(tmp_path, monkeypatch)
    for site in (world.target, world.protected):
        region = site.release_config["aws_region"]
        add_resource(
            world,
            site,
            kind="ecr_repository",
            identifier=f"arn:aws:ecr:{region}:123456789012:repository/shared-name",
        )
    binding = adapter.NativeBackend(world.settings).initial()
    assert binding["inputs"]["target"]["clusters"][0]["region"] == "us-east-1", (
        "target must retain its explicit regional identity"
    )
    assert binding["inputs"]["protected"]["clusters"][0]["region"] == "us-west-2", (
        "different valid regional ARNs must not be confused with a shared global resource"
    )


@pytest.mark.parametrize(
    ("kind", "identifier"),
    [
        ("iam_role", None),
        ("iam_role", "not-an-arn"),
        ("iam_role", "arn:aws:iam:us-east-1:123456789012:role/invalid-regional-iam"),
        ("ecr_repository", "arn:aws:ecr::123456789012:repository/unknown-region"),
        ("ecr_repository", "arn:aws:ecr:us-east-1::repository/unknown-account"),
    ],
)
def test_unknown_resource_scope_is_refused_not_inferred_from_site(
    tmp_path, monkeypatch, kind, identifier
):
    world = cross_region_world(tmp_path, monkeypatch)
    row = (
        world.snapshots[world.target.metadata_name]
        .resources[-1]
        .model_copy(
            update={
                "resource_type": kind,
                "resource_id": "name-only",
                "resource_arn": identifier,
            }
        )
    )
    with pytest.raises((contract.UninstallCaseError, BootstrapError)):
        observe.resource_identity(row, world.target)


def test_canonical_arn_identity_preserves_every_global_identity_component(
    tmp_path, monkeypatch
):
    world = cross_region_world(tmp_path, monkeypatch)
    template = world.snapshots[world.target.metadata_name].resources[-1]
    arns = [
        "arn:aws:iam::123456789012:role/owned/path",
        "arn:aws-us-gov:iam::123456789012:role/owned/path",
        "arn:aws:iam::111122223333:role/owned/path",
        "arn:aws:iam::123456789012:role/another/path",
        "arn:aws:route53:::hostedzone/zone-id",
    ]
    identities = {
        observe.resource_identity(
            template.model_copy(update={"resource_arn": value}), world.target
        )
        for value in arns
    }
    assert len(identities) == len(arns), (
        "partition, service, account and complete ARN resource must all remain significant"
    )
    row = template.model_copy(update={"resource_arn": arns[0]})
    assert observe.resource_identity(row, world.target) == observe.resource_identity(
        row.model_copy(update={"region": "us-west-2"}), world.protected
    ), "a global ARN's identity must not inherit either site's Region"


def test_shared_global_hosted_zone_id_is_not_regionalized(tmp_path, monkeypatch):
    world = cross_region_world(tmp_path, monkeypatch)
    for site in (world.target, world.protected):
        add_resource(world, site, kind="route53_zone", identifier=None)
    with pytest.raises(contract.UninstallCaseError, match="overlaps"):
        adapter.NativeBackend(world.settings).initial()
    assert not world.settings.native_dir.exists(), (
        "a global hosted zone cannot be isolated by changing site Region"
    )


def test_equal_addon_names_in_independent_cpu_clusters_do_not_alias(
    tmp_path, monkeypatch
):
    world = World(tmp_path, monkeypatch)
    for site in (world.target, world.protected):
        original = world.snapshots[site.metadata_name]
        rows = [
            row.model_copy(
                update={
                    "resource_id": "eks-pod-identity-agent",
                    "attributes": {
                        "cluster_name": contract.cluster_specs(site)[0]["eks_name"]
                    },
                }
            )
            if row.resource_type == "eks_addon"
            else row
            for row in original.resources
        ]
        snapshot = InstallationResourceSnapshot(
            site_id=original.site_id, resources=rows
        )
        world.snapshots[site.metadata_name] = snapshot.model_copy(
            update={"source_sha256": snapshot.digest()}
        )
    binding = adapter.NativeBackend(world.settings).initial()
    assert (
        binding["inputs"]["target"]["clusters"][0]["eks_arn"]
        != binding["inputs"]["protected"]["clusters"][0]["eks_arn"]
    ), "the standard addon name must be scoped to its actual CPU EKS cluster"


@pytest.mark.parametrize(
    ("kind", "attributes"),
    [
        ("route53_record", {"hosted_zone_id": "ZSHARED", "record_type": "CNAME"}),
        (
            "route53_vpc_association",
            {
                "hosted_zone_id": "ZSHARED",
                "vpc_id": "vpc-owned",
                "vpc_region": "us-east-1",
            },
        ),
    ],
)
def test_global_dns_binding_identity_uses_its_own_scope_not_site_region(
    tmp_path, monkeypatch, kind, attributes
):
    world = cross_region_world(tmp_path, monkeypatch)
    row = (
        world.snapshots[world.target.metadata_name]
        .resources[-1]
        .model_copy(
            update={
                "resource_type": kind,
                "resource_id": "control.example.invalid",
                "attributes": attributes,
            }
        )
    )
    assert observe.resource_identity(row, world.target) == observe.resource_identity(
        row.model_copy(update={"region": "us-west-2"}), world.protected
    ), (
        "a global DNS relationship cannot acquire a new identity from its referring site Region"
    )
    with pytest.raises(contract.UninstallCaseError, match="scope"):
        observe.resource_identity(
            row.model_copy(update={"attributes": {}}), world.target
        )


def test_child_resource_scope_is_required_and_workspace_specific(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    row = (
        world.snapshots[world.target.metadata_name]
        .resources[-1]
        .model_copy(
            update={
                "resource_type": "grafana_service_account",
                "resource_id": "1",
                "attributes": {"workspace_id": "workspace-one"},
            }
        )
    )
    other = row.model_copy(update={"attributes": {"workspace_id": "workspace-two"}})
    assert observe.resource_identity(row, world.target) != observe.resource_identity(
        other, world.target
    ), "workspace-local service-account IDs must not alias different workspaces"
    with pytest.raises(contract.UninstallCaseError, match="scope"):
        observe.resource_identity(
            row.model_copy(update={"attributes": {}}), world.target
        )
