from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin import cluster_removal_network as network
from gpu_fault.admin import resource_registry as registry
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.cluster_join_commit import registry_delta
from gpu_fault.admin.cluster_removal_resources import (
    association_to_detach,
    deletion_waves,
)
from gpu_fault.admin.resource_records import record
from gpu_fault.admin.resource_registry_dns import (
    reconcile_vpc_association_resource,
    recorded_zone_vpc_ownership,
    vpc_association_entry,
    vpc_association_resource,
    zone_vpc_associations,
)
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceSnapshot,
)
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
from gpu_fault.installation_resources import InstallationResourceOwnership as Ownership
from gpu_fault.installation_resources import InstallationResourceStatus as Status
from gpu_fault.store import InMemoryStore, NotFoundError
from tests.admin._cluster_removal_support import RemovalScenario
from tests.admin.test_admin_site import site_file

REGION = "us-east-1"
ACCOUNT = "123456789012"
ZONE = "Z123"
CPU_VPC = "vpc-cpu"
GPU_VPC = "vpc-gpu"
KEY = f"aws/route53/vpc-association/{REGION}/{GPU_VPC}"


def identity(name: str, vpc_id: str, *, role: str = "gpu") -> ClusterIdentity:
    return ClusterIdentity(
        input_arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{name}",
        role=role,
        region=REGION,
        account_id=ACCOUNT,
        hyperpod_arn=f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hp-{name}",
        hyperpod_name=f"hp-{name}",
        eks_arn=f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{name}",
        eks_name=name,
        vpc_id=vpc_id,
        subnet_ids=("subnet-test",),
        node_recovery="None",
        context=name,
    )


def entry(vpc_id: str = GPU_VPC, **updates: Any) -> dict[str, Any]:
    values = {
        "vpc_id": vpc_id,
        "vpc_region": REGION,
        "cluster_ids": ["gpu-a"],
        "ownership": "CREATED",
        "native": vpc_id == CPU_VPC,
    }
    return vpc_association_entry(**{**values, **updates})


def association(**updates: Any) -> InstallationResource:
    return vpc_association_resource(
        **{
            "site_id": "test-site",
            "hosted_zone_id": ZONE,
            "vpc_id": GPU_VPC,
            "vpc_region": REGION,
            "region": REGION,
            "account_id": ACCOUNT,
            "ownership": Ownership.CREATED,
            **updates,
        }
    )


def sealed(*resources: InstallationResource) -> InstallationResourceSnapshot:
    value = InstallationResourceSnapshot(
        site_id=resources[0].site_id if resources else "test-site",
        resources=list(resources),
    )
    return value.model_copy(update={"source_sha256": value.digest()})


def bootstrap(*associations: dict[str, Any]) -> dict[str, Any]:
    return {
        "site_id": "test-site",
        "resources": {
            "nlb_network": {"vpc_id": CPU_VPC},
            "pki": {
                "hosted_zone_id": ZONE,
                "zone_ownership": "CREATED",
                "vpc_associations": list(associations),
            },
        },
    }


def dns_rows(snapshot: InstallationResourceSnapshot) -> list[InstallationResource]:
    return [
        item
        for item in snapshot.resources
        if item.resource_type == "route53_vpc_association"
    ]


@pytest.fixture
def site(tmp_path):
    value = load_site(site_file(tmp_path))
    value.release_config["dns"] = {"hosted_zone_id": ZONE}
    return value


def test_zone_entries_deduplicate_region_vpc_and_mark_native_cpu() -> None:
    cpu = identity("cpu", CPU_VPC, role="cpu")
    gpu = [
        identity("z", GPU_VPC),
        identity("a", GPU_VPC),
        identity("shared-cpu", CPU_VPC),
        replace(identity("other-region", CPU_VPC), region="us-west-2"),
    ]
    rows = zone_vpc_associations(cpu, gpu)

    assert [(item["vpc_region"], item["vpc_id"]) for item in rows] == [
        (REGION, CPU_VPC),
        (REGION, GPU_VPC),
        ("us-west-2", CPU_VPC),
    ]
    assert [item["native"] for item in rows] == [True, False, False]
    assert rows[0]["cluster_ids"] == ["hp-shared-cpu"]
    assert rows[1]["cluster_ids"] == ["hp-a", "hp-z"]
    assert {item["ownership"] for item in rows} == {"EXTERNAL"}
    assert rows[1]["resource_key"] == KEY
    assert zone_vpc_associations(cpu, list(reversed(gpu))) == rows
    after = zone_vpc_associations(cpu, [gpu[0]])
    assert after[1]["resource_key"] == rows[1]["resource_key"]


def test_zone_ownership_requires_a_complete_explicit_map() -> None:
    cpu, gpu = identity("cpu", CPU_VPC), identity("gpu", GPU_VPC)
    with pytest.raises(BootstrapError, match="incomplete"):
        zone_vpc_associations(
            cpu, [gpu], ownership_by_vpc={(REGION, CPU_VPC): "CREATED"}
        )
    rows = zone_vpc_associations(
        cpu,
        [gpu],
        ownership_by_vpc={(REGION, CPU_VPC): "CREATED", (REGION, GPU_VPC): "REUSED"},
    )
    assert [item["ownership"] for item in rows] == ["CREATED", "REUSED"]


def test_zone_rejects_conflicting_cluster_identities() -> None:
    gpu = identity("gpu", GPU_VPC)
    with pytest.raises(BootstrapError, match="ambiguous"):
        zone_vpc_associations(
            identity("cpu", CPU_VPC), [gpu, replace(gpu, vpc_id="vpc-other")]
        )


@pytest.mark.parametrize("ownership", [None, "", "created", "UNKNOWN", 7, True])
def test_entries_never_default_unknown_ownership_to_created(ownership) -> None:
    with pytest.raises(BootstrapError, match="ownership"):
        entry(ownership=ownership)


@pytest.mark.parametrize(
    "updates",
    [
        {"vpc_id": ""},
        {"vpc_region": None},
        {"vpc_id": "vpc/a"},
        {"vpc_region": "us-east-1:other"},
        {"cluster_ids": "gpu-a"},
        {"cluster_ids": [""]},
        {"native": "false"},
    ],
)
def test_entries_reject_malformed_identity_or_membership(updates) -> None:
    with pytest.raises(BootstrapError):
        entry(**updates)


def test_recorded_ownership_uses_resolved_cpu_not_list_position() -> None:
    cpu = identity("cpu", CPU_VPC)
    gpu = {"vpc_id": GPU_VPC, "vpc_region": REGION, "ownership": "CREATED"}
    native = {"vpc_id": CPU_VPC, "vpc_region": REGION, "ownership": "EXTERNAL"}
    pki = {"hosted_zone_id": ZONE, "vpc_associations": [gpu, native, gpu]}
    before = copy.deepcopy(pki)

    assert recorded_zone_vpc_ownership(pki, hosted_zone_id=ZONE, cpu=cpu) == {
        (REGION, GPU_VPC): "CREATED",
        (REGION, CPU_VPC): "EXTERNAL",
    }
    assert pki == before
    assert recorded_zone_vpc_ownership(None, hosted_zone_id=ZONE, cpu=cpu) == {}
    assert recorded_zone_vpc_ownership({}, hosted_zone_id=ZONE, cpu=cpu) == {}


@pytest.mark.parametrize(
    "pki",
    [
        {"hosted_zone_id": "ZOTHER"},
        {"hosted_zone_id": ZONE, "vpc_associations": {}},
        {"hosted_zone_id": ZONE, "vpc_associations": [None]},
        {"hosted_zone_id": ZONE, "vpc_associations": [{"vpc_id": GPU_VPC}]},
        {
            "hosted_zone_id": ZONE,
            "vpc_associations": [
                entry(ownership="CREATED"),
                entry(ownership="EXTERNAL"),
            ],
        },
        {"hosted_zone_id": ZONE, "vpc_associations": [entry(native=True)]},
        {"hosted_zone_id": ZONE, "vpc_associations": [entry(CPU_VPC, native=False)]},
        {"hosted_zone_id": ZONE, "vpc_associations": [{**entry(), "native": 1}]},
        {
            "hosted_zone_id": ZONE,
            "vpc_associations": [
                entry(),
                {**entry(), "resource_key": "aws/route53/vpc-association/old"},
            ],
        },
    ],
)
def test_recorded_ownership_rejects_conflicting_or_incomplete_proof(pki) -> None:
    with pytest.raises(BootstrapError):
        recorded_zone_vpc_ownership(
            pki, hosted_zone_id=ZONE, cpu=identity("cpu", CPU_VPC)
        )


@pytest.mark.parametrize("zone_ownership", ["CREATED", "EXTERNAL"])
@pytest.mark.parametrize("ownership", list(Ownership))
def test_snapshot_registers_gpu_associations_for_both_zone_policies(
    site, zone_ownership, ownership
) -> None:
    state = bootstrap(entry(CPU_VPC), entry(ownership=ownership.value))
    state["resources"]["pki"]["zone_ownership"] = zone_ownership
    snapshot = registry.build_installation_snapshot(site, state)
    rows = dns_rows(snapshot)

    assert len(rows) == 1
    assert rows[0].resource_key == KEY
    assert rows[0].resource_id == f"{ZONE}:{REGION}:{GPU_VPC}"
    assert rows[0].ownership is ownership
    assert rows[0].delete_policy is (
        Policy.DETACH if ownership is Ownership.CREATED else Policy.PRESERVE
    )
    assert rows[0].dependencies == ["aws/route53/zone"]
    if ownership is Ownership.CREATED:
        zone = next(
            item for item in snapshot.resources if item.resource_type == "route53_zone"
        )
        assert deletion_waves([zone, rows[0]]) == [[rows[0]], [zone]]


def test_snapshot_deduplicates_exact_entries_but_not_same_vpc_in_another_region(
    site,
) -> None:
    state = bootstrap(
        entry(CPU_VPC),
        entry(),
        entry(),
        entry(CPU_VPC, vpc_region="us-west-2", native=False),
    )
    rows = dns_rows(registry.build_installation_snapshot(site, state))
    assert {item.resource_id for item in rows} == {
        f"{ZONE}:{REGION}:{GPU_VPC}",
        f"{ZONE}:us-west-2:{CPU_VPC}",
    }


def test_legacy_owned_gpu_association_gets_a_physical_key_not_a_cluster_guess(
    site,
) -> None:
    state = bootstrap({"vpc_id": GPU_VPC, "vpc_region": REGION, "ownership": "CREATED"})
    state["joined_clusters"] = {"unrelated": {"vpc_id": GPU_VPC}}
    assert (
        dns_rows(registry.build_installation_snapshot(site, state))[0].resource_key
        == KEY
    )


@pytest.mark.parametrize(
    "change", ["site", "zone", "cpu", "ownership", "region", "duplicate", "native"]
)
def test_snapshot_refuses_ambiguous_legacy_or_conflicting_checkpoint(
    site, change
) -> None:
    state = bootstrap(entry())
    pki = state["resources"]["pki"]
    if change == "site":
        state["site_id"] = "foreign"
    elif change == "zone":
        pki["hosted_zone_id"] = "ZOTHER"
    elif change == "cpu":
        state["resources"].pop("nlb_network")
        pki["vpc_associations"][0].pop("native")
    elif change == "ownership":
        pki["vpc_associations"][0].pop("ownership")
    elif change == "region":
        pki["vpc_associations"][0].pop("vpc_region")
    elif change == "duplicate":
        pki["vpc_associations"].append(entry(ownership="EXTERNAL"))
    elif change == "native":
        pki["vpc_associations"][0]["native"] = True
    with pytest.raises(BootstrapError):
        registry.build_installation_snapshot(site, state)


def test_explicit_native_identity_works_without_an_nlb_checkpoint(site) -> None:
    state = bootstrap(entry(), entry(CPU_VPC))
    state["resources"].pop("nlb_network")
    assert [
        item.resource_key
        for item in dns_rows(registry.build_installation_snapshot(site, state))
    ] == [KEY]


@pytest.mark.parametrize(
    "key",
    ["aws/route53/vpc-association/2", "aws/route53/vpc-association/removed-cluster"],
)
@pytest.mark.parametrize("status", [Status.ACTIVE, Status.DETACHED])
def test_snapshot_preserves_existing_physical_identity_key_and_status(
    site, key, status
) -> None:
    previous = association(resource_key=key).model_copy(
        update={
            "status": status,
            "attributes": {**association().attributes, "audit_origin": "bootstrap"},
        }
    )
    store = InMemoryStore()
    store.save_installation_resource(previous)
    state = bootstrap(entry(cluster_ids=["new-first-cluster"]), entry(CPU_VPC))
    result = registry.build_installation_snapshot(
        site, state, existing=sealed(previous)
    )
    current = dns_rows(result)[0]
    store.save_installation_resource(current)

    assert current.immutable_identity() == previous.immutable_identity()
    assert current.status is status
    assert current.created_at == previous.created_at
    assert current.attributes == previous.attributes
    assert store.get_installation_resource("test-site", key) == current
    with pytest.raises(NotFoundError):
        store.get_installation_resource("test-site", KEY)


def test_existing_explicit_ownership_can_be_preserved_without_relabeling_legacy_state(
    site,
) -> None:
    previous = association(
        ownership=Ownership.EXTERNAL, resource_key="aws/route53/vpc-association/2"
    )
    state = bootstrap({"vpc_id": GPU_VPC, "vpc_region": REGION})
    before = copy.deepcopy(state)
    current = dns_rows(
        registry.build_installation_snapshot(site, state, existing=sealed(previous))
    )[0]
    assert current.immutable_identity() == previous.immutable_identity()
    assert state == before


@pytest.mark.parametrize(
    "updates",
    [
        {"region": "us-west-2"},
        {"account_id": "111122223333"},
        {"provider": "foreign"},
        {"dependencies": []},
        {"resource_id": f"ZOTHER:{REGION}:{GPU_VPC}"},
        {
            "attributes": {
                "hosted_zone_id": ZONE,
                "vpc_region": REGION,
                "vpc_id": "vpc-other",
            }
        },
        {"delete_policy": Policy.DELETE},
    ],
)
def test_existing_association_identity_conflicts_never_authorize_a_rewrite(
    site, updates
) -> None:
    previous = association().model_copy(update=updates)
    with pytest.raises(BootstrapError):
        registry.build_installation_snapshot(
            site, bootstrap(entry()), existing=sealed(previous)
        )


def test_two_immutable_keys_for_one_physical_association_require_reconciliation(
    site,
) -> None:
    previous = association(resource_key="aws/route53/vpc-association/2")
    with pytest.raises(BootstrapError, match="ambiguous"):
        registry.build_installation_snapshot(
            site, bootstrap(entry()), existing=sealed(previous, association())
        )


def test_snapshot_and_join_never_change_existing_external_ownership(site) -> None:
    previous = association(ownership=Ownership.EXTERNAL)
    with pytest.raises(BootstrapError, match="ownership"):
        registry.build_installation_snapshot(
            site, bootstrap(entry()), existing=sealed(previous)
        )
    with pytest.raises(BootstrapError, match="ownership"):
        registry_delta(sealed(previous), [association()], site_id="test-site")


def test_join_revives_a_legacy_key_without_creating_a_second_row() -> None:
    previous = association(
        resource_key="aws/route53/vpc-association/old-cluster"
    ).model_copy(update={"status": Status.DETACHED})
    delta, merged = registry_delta(
        sealed(previous), [association()], site_id="test-site"
    )
    assert len(merged.resources) == 1
    assert delta.resources[0].immutable_identity() == previous.immutable_identity()
    assert delta.resources[0].status is Status.ACTIVE


def test_conflicting_requested_key_is_not_hidden_by_another_physical_match() -> None:
    previous = association(resource_key="aws/route53/vpc-association/old-cluster")
    collision = association(vpc_id="vpc-other", resource_key=KEY)
    with pytest.raises(BootstrapError, match="already names"):
        reconcile_vpc_association_resource(association(), [previous, collision])


def test_native_cpu_row_is_preserved_or_refused_never_renamed_for_a_gpu(site) -> None:
    state = bootstrap(entry(CPU_VPC, cluster_ids=["gpu-a"]))
    previous = association(
        vpc_id=CPU_VPC,
        ownership=Ownership.EXTERNAL,
        resource_key="aws/route53/vpc-association/1",
    )
    assert dns_rows(
        registry.build_installation_snapshot(site, state, existing=sealed(previous))
    ) == [previous]
    unsafe = association(vpc_id=CPU_VPC, resource_key=previous.resource_key)
    with pytest.raises(BootstrapError, match="native association has a deletion row"):
        registry.build_installation_snapshot(site, state, existing=sealed(unsafe))


def test_snapshot_rejects_another_installations_registry(site) -> None:
    previous = association(site_id="test-site:installation:previous")
    with pytest.raises(BootstrapError, match="another site"):
        registry.build_installation_snapshot(
            site, bootstrap(entry()), existing=sealed(previous)
        )


def test_registry_sync_fetches_existing_keys_before_build_and_keeps_their_identity(
    site, monkeypatch
) -> None:
    previous = association(resource_key="aws/route53/vpc-association/2")
    events = []
    written = []
    monkeypatch.setattr(
        registry,
        "find_bootstrap_state",
        lambda _site: bootstrap(entry(), entry(CPU_VPC)),
    )
    monkeypatch.setattr(
        registry,
        "fetch_installation_resource_registry",
        lambda _site, **kwargs: events.append(("fetch", kwargs["allow_empty"]))
        or sealed(previous),
    )
    monkeypatch.setattr(
        registry,
        "discover_runtime_resources",
        lambda _site: events.append("runtime") or {},
    )
    monkeypatch.setattr(
        registry,
        "sync_installation_resource_snapshot",
        lambda _site, value: events.append("sync") or written.append(value),
    )

    output = registry.sync_installation_resource_registry(site)

    assert events == [("fetch", True), "runtime", "sync"]
    assert dns_rows(written[0])[0].immutable_identity() == previous.immutable_identity()
    assert registry.load_installation_resource_snapshot(output) == written[0]


def test_registry_sync_never_treats_failed_fetch_as_an_empty_registry(
    site, monkeypatch
) -> None:
    monkeypatch.setattr(
        registry, "find_bootstrap_state", lambda _site: bootstrap(entry())
    )

    def unavailable(_site, **_kwargs):
        raise BootstrapError("registry read unavailable")

    monkeypatch.setattr(registry, "fetch_installation_resource_registry", unavailable)
    monkeypatch.setattr(
        registry,
        "discover_runtime_resources",
        lambda _site: pytest.fail("continued after failed registry read"),
    )
    with pytest.raises(BootstrapError, match="unavailable"):
        registry.sync_installation_resource_registry(site)


@pytest.mark.parametrize("shared_with", ["cpu", "gpu"])
def test_removal_preserves_native_and_shared_association_rows(
    site, shared_with
) -> None:
    previous = association()
    result = association_to_detach(
        site,
        sealed(previous),
        target_network={"vpc_id": GPU_VPC},
        remaining_networks=[{"vpc_id": GPU_VPC}] if shared_with == "gpu" else [],
        cpu_vpc_id=GPU_VPC if shared_with == "cpu" else CPU_VPC,
    )
    assert result is None
    assert previous.status is Status.ACTIVE


def test_removal_shared_identity_includes_region_not_only_vpc_id(site) -> None:
    previous = association(resource_key="aws/route53/vpc-association/former-member")
    result = association_to_detach(
        site,
        sealed(previous),
        target_network={"vpc_id": GPU_VPC},
        remaining_networks=[{"vpc_id": GPU_VPC, "vpc_region": "us-west-2"}],
        cpu_vpc_id=CPU_VPC,
    )
    assert result == previous


@pytest.mark.parametrize("ownership", [Ownership.EXTERNAL, Ownership.REUSED])
def test_removal_preserves_explicit_non_created_associations(site, ownership) -> None:
    assert (
        association_to_detach(
            site,
            sealed(association(ownership=ownership)),
            target_network={"vpc_id": GPU_VPC},
            remaining_networks=[],
            cpu_vpc_id=CPU_VPC,
        )
        is None
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"ownership": Ownership.EXTERNAL},
        {"status": Status.DETACHED},
        {"account_id": "111122223333"},
        {"attributes": {}},
    ],
)
def test_removal_rejects_inconsistent_dns_authorization(site, updates) -> None:
    previous = association().model_copy(update=updates)
    with pytest.raises(BootstrapError):
        association_to_detach(
            site,
            sealed(previous),
            target_network={"vpc_id": GPU_VPC},
            remaining_networks=[],
            cpu_vpc_id=CPU_VPC,
        )


def test_network_helper_does_not_detach_without_explicit_dns_authorization(
    site, monkeypatch
) -> None:
    monkeypatch.setattr(
        network,
        "disassociate_vpc_from_hosted_zone",
        lambda **_kwargs: pytest.fail("unapproved DNS detach"),
    )
    result = network.detach_network(
        removal.RemoveClusterRequest(site, "gpu-a", "REMOVE_GPU_CLUSTER"),
        target_network={"vpc_id": GPU_VPC, "nat_eips": []},
        remaining_networks=[],
        cpu_vpc_id=CPU_VPC,
    )
    assert result["detached_vpc_id"] is None


def removal_scenario(tmp_path, monkeypatch, ownership=Ownership.CREATED):
    scenario = RemovalScenario(tmp_path, monkeypatch)
    document = yaml.safe_load(scenario.path.read_text())
    document["spec"]["dns"] = {"hostedZoneId": ZONE, "hostname": "api.test.internal"}
    scenario.path.write_text(yaml.safe_dump(document, sort_keys=False))
    previous = association(
        ownership=ownership, resource_key="aws/route53/vpc-association/old-cluster"
    )
    zone = record(
        site_id="test-site",
        resource_key="aws/route53/zone",
        resource_type="route53_zone",
        resource_id=ZONE,
        region=REGION,
        account_id=ACCOUNT,
        ownership=Ownership.CREATED,
        delete_policy=Policy.DELETE,
    )
    scenario.registry = sealed(
        *(
            item
            for item in scenario.registry.resources
            if item.resource_type != "route53_vpc_association"
        ),
        previous,
        zone,
    )
    return scenario, previous


@pytest.mark.parametrize("ownership", [Ownership.CREATED, Ownership.EXTERNAL])
def test_remove_controller_uses_dns_policy_and_keeps_original_registry_key(
    tmp_path, monkeypatch, ownership
) -> None:
    scenario, previous = removal_scenario(tmp_path, monkeypatch, ownership)
    detached = []

    def detach(_request, **kwargs):
        detached.append(kwargs["detach_dns"])
        return {
            "detached_vpc_id": GPU_VPC if kwargs["detach_dns"] else None,
            "revoked_nat_eips": [],
            "route53_change_id": "/change/test" if kwargs["detach_dns"] else None,
            "route53_change_status": "INSYNC" if kwargs["detach_dns"] else None,
        }

    monkeypatch.setattr(removal, "_detach_network", detach)
    result = removal.remove_cluster(scenario.request(), runner=object())
    assert result["phase"] == "COMPLETED"
    assert detached == [ownership is Ownership.CREATED]
    before = Path(scenario.state()["evidence"]["DISCOVERED"]["registry_snapshot"])
    saved = registry.load_installation_resource_snapshot(
        before.with_name("installation-resources-after.json")
    )
    row = dns_rows(saved)[0]
    assert row.immutable_identity() == previous.immutable_identity()
    assert row.status is (
        Status.DETACHED if ownership is Ownership.CREATED else Status.ACTIVE
    )
    assert row.resource_key == previous.resource_key


@pytest.mark.parametrize(
    "conflict",
    ["missing", "preserved-dependent", "target-dependent", "missing-zone", "duplicate"],
)
def test_remove_refuses_unproven_or_dependent_dns_before_draining(
    tmp_path, monkeypatch, conflict
) -> None:
    scenario, previous = removal_scenario(tmp_path, monkeypatch)
    rows = list(scenario.registry.resources)
    if conflict == "missing":
        rows.remove(previous)
    elif conflict == "missing-zone":
        rows = [item for item in rows if item.resource_type != "route53_zone"]
    elif conflict == "duplicate":
        rows.append(association())
    elif conflict == "target-dependent":
        rows = [
            item.model_copy(update={"dependencies": [previous.resource_key]})
            if item.resource_key == "aws/iam/executor/gpu-a/role"
            else item
            for item in rows
        ]
    else:
        rows.append(
            record(
                site_id="test-site",
                resource_key="cluster/retained/dns",
                resource_type="external-consumer",
                resource_id="consumer",
                ownership=Ownership.EXTERNAL,
                delete_policy=Policy.PRESERVE,
                dependencies=[previous.resource_key],
            )
        )
    scenario.registry = sealed(*rows)
    with pytest.raises(BootstrapError):
        removal.remove_cluster(scenario.request(), runner=object())
    assert scenario.calls == ["snapshot"]
