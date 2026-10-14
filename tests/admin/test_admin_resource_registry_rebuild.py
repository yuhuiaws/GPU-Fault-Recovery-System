"""A deploy or ``config`` rebuild of the installation registry converges on the
rows the registry already records (batch 79).

Every deploy and ``config`` since the first plain-EKS trial printed
"installation resource registry not synchronized ... ValueError", and the
processor replayed ``/v1/installation-resources/sync`` with "installation
resource identity cannot change" until its retry horizon, although no resource
had been replaced:

* a bootstrap rediscovery listed the owned public subnets in another order, so
  the position-numbered ``aws/network/public-*/<n>`` rows swapped resource IDs;
* ``join-cluster`` records ``cluster/<id>/hyperpod`` with the HyperPod ARN it
  described, and the deploy builder, which has no ARN for it, rebuilt the row
  with ``None``.

The store guard is unchanged: these tests save every rebuilt row through it.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.admin import resource_registry as registry
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.resource_registry_network import registered_slots
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceSnapshot,
)
from gpu_fault.store import InMemoryStore
from tests.admin.test_admin_site import site_file

NLB_ARN = (
    "arn:aws:elasticloadbalancing:us-east-1:123456789012:loadbalancer/net/regional/1"
)
LISTENERS = [NLB_ARN.replace("loadbalancer", "listener") + f"/{n}" for n in (1, 2)]
TARGET_GROUPS = [
    f"arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/k8s-{n}/abc"
    for n in ("a", "b")
]
# What ``cluster_join_commit._joined_resources`` records for the fixture's
# HyperPod cluster: the deploy builder's row plus the ARN the join described.
HYPERPOD_KEY = "cluster/gpu-a/hyperpod"
HYPERPOD_ARN = "arn:aws:sagemaker:us-east-1:123456789012:cluster/hpgpua0001"


def _subnet(subnet_id: str, zone: str) -> dict[str, str]:
    return {
        "subnet_id": subnet_id,
        "availability_zone": zone,
        "ownership": "CREATED",
        "vpc_id": "vpc-cpu",
        "route_table_id": "rtb-" + subnet_id.removeprefix("subnet-"),
        "route_table_association_id": "rtbassoc-" + subnet_id.removeprefix("subnet-"),
    }


SUBNET_A = _subnet("subnet-a", "us-east-1a")
SUBNET_B = _subnet("subnet-b", "us-east-1b")
SUBNET_C = _subnet("subnet-c", "us-east-1a")


def _state(*subnets: dict[str, str]) -> dict[str, Any]:
    return {
        "site_id": "test-site",
        "resources": {
            "nlb_network": {
                "vpc_id": "vpc-cpu",
                "security_group": "sg-nlb",
                "security_group_ownership": "CREATED",
                "subnet_resources": [dict(item) for item in subnets],
            }
        },
    }


def _runtime(
    listeners: list[str] = LISTENERS, target_groups: list[str] = TARGET_GROUPS
) -> dict[str, Any]:
    return {
        "nlb": {
            "arn": NLB_ARN,
            "dns_name": "regional.elb.example",
            "listeners": list(listeners),
            "target_groups": list(target_groups),
        }
    }


def _sealed(resources: list[InstallationResource]) -> InstallationResourceSnapshot:
    snapshot = InstallationResourceSnapshot(site_id="test-site", resources=resources)
    return snapshot.model_copy(update={"source_sha256": snapshot.digest()})


def _identities(snapshot: InstallationResourceSnapshot) -> dict[str, tuple]:
    return {item.resource_key: item.immutable_identity() for item in snapshot.resources}


def _store(snapshot: InstallationResourceSnapshot) -> InMemoryStore:
    store = InMemoryStore()
    for item in snapshot.resources:
        store.save_installation_resource(item)
    return store


def _joined(snapshot: InstallationResourceSnapshot) -> InstallationResourceSnapshot:
    """The same registry after the HyperPod cluster was joined, not deployed."""

    return _sealed(
        [
            item.model_copy(update={"resource_arn": HYPERPOD_ARN})
            if item.resource_key == HYPERPOD_KEY
            else item
            for item in snapshot.resources
        ]
    )


@pytest.fixture
def site(tmp_path):
    return load_site(site_file(tmp_path))


def test_a_rediscovered_subnet_order_rebuilds_the_registered_rows(site) -> None:
    first = registry.build_installation_snapshot(
        site, _state(SUBNET_A, SUBNET_B), _runtime()
    )
    by_key = {item.resource_key: item for item in first.resources}
    assert by_key["aws/network/public-subnet/1"].resource_id == "subnet-a", (
        "a fresh registry is no longer numbered in list order"
    )
    assert by_key["aws/network/public-route-table/2"].resource_id == "rtb-b"

    rebuilt = registry.build_installation_snapshot(
        site, _state(SUBNET_B, SUBNET_A), _runtime(), existing=first
    )

    assert _identities(rebuilt) == _identities(first)
    store = _store(first)
    for item in rebuilt.resources:
        assert store.save_installation_resource(item) == item


def test_listener_and_target_group_numbers_follow_the_registry(site) -> None:
    first = registry.build_installation_snapshot(
        site, _state(SUBNET_A, SUBNET_B), _runtime()
    )

    rebuilt = registry.build_installation_snapshot(
        site,
        _state(SUBNET_A, SUBNET_B),
        _runtime(LISTENERS[::-1], TARGET_GROUPS[::-1]),
        existing=first,
    )

    assert _identities(rebuilt) == _identities(first)
    by_key = {item.resource_key: item for item in rebuilt.resources}
    assert by_key["aws/nlb/listener/1"].resource_id == LISTENERS[0]
    assert by_key["aws/nlb/target-group/2"].resource_arn == TARGET_GROUPS[1]


def test_registered_slots_keep_numbers_and_fill_the_lowest_free_one(site) -> None:
    prefix = "aws/network/public-subnet/"
    first = {
        item.resource_key: item
        for item in registry.build_installation_snapshot(
            site, _state(SUBNET_A, SUBNET_B), _runtime()
        ).resources
    }
    existing = [
        first[prefix + "1"],
        first[prefix + "2"].model_copy(
            update={"resource_key": prefix + "3", "resource_id": "subnet-c"}
        ),
        first["aws/nlb"],
    ]

    assert registered_slots(prefix, ["subnet-x", "subnet-y"], []) == [1, 2]
    assert registered_slots(prefix, ["subnet-d", "subnet-c", "subnet-a"], existing) == [
        2,
        3,
        1,
    ]
    assert registered_slots(prefix, ["subnet-a", "subnet-a"], existing) == [1, 2], (
        "a duplicate must not share a key with the registered row"
    )
    twice = [
        *existing,
        first[prefix + "1"].model_copy(update={"resource_key": prefix + "9"}),
    ]
    with pytest.raises(BootstrapError, match="twice"):
        registered_slots(prefix, ["subnet-a"], twice)


def test_a_replaced_subnet_gets_a_new_number_and_the_nlb_change_is_named(
    site, monkeypatch
) -> None:
    first = registry.build_installation_snapshot(
        site, _state(SUBNET_A, SUBNET_B), _runtime()
    )

    rebuilt = registry.build_installation_snapshot(
        site, _state(SUBNET_C, SUBNET_B), _runtime(), existing=first
    )

    by_key = {item.resource_key: item for item in rebuilt.resources}
    assert by_key["aws/network/public-subnet/3"].resource_id == "subnet-c"
    assert by_key["aws/network/public-route-table/3"].resource_id == "rtb-c"
    assert by_key["aws/network/public-subnet/2"].resource_id == "subnet-b"
    assert "aws/network/public-subnet/1" not in by_key, (
        "the replaced subnet's row must stay as recorded, not be overwritten"
    )
    _sync_events(site, monkeypatch, state=_state(SUBNET_C, SUBNET_B), existing=first)
    with pytest.raises(BootstrapError, match=r"aws/nlb \(dependencies\)") as refused:
        registry.sync_installation_resource_registry(site)
    assert "aws/network/public-subnet" not in str(refused.value)


def test_a_joined_hyperpod_arn_survives_a_deploy_rebuild(site) -> None:
    deployed = registry.build_installation_snapshot(
        site, _state(SUBNET_A, SUBNET_B), _runtime()
    )
    joined = _joined(deployed)
    store = _store(joined)
    stale = {item.resource_key: item for item in deployed.resources}[HYPERPOD_KEY]
    with pytest.raises(ValueError, match="identity cannot change"):
        store.save_installation_resource(stale)

    rebuilt = registry.build_installation_snapshot(
        site, _state(SUBNET_A, SUBNET_B), _runtime(), existing=joined
    )

    assert _identities(rebuilt) == _identities(joined)
    assert {item.resource_key: item for item in rebuilt.resources}[
        HYPERPOD_KEY
    ].resource_arn == HYPERPOD_ARN
    for item in rebuilt.resources:
        assert store.save_installation_resource(item) == item


def test_a_row_recorded_without_an_arn_keeps_it_through_a_rebuild(site) -> None:
    deployed = registry.build_installation_snapshot(
        site, _state(SUBNET_A, SUBNET_B), _runtime()
    )
    legacy = _sealed(
        [
            item.model_copy(update={"resource_arn": None})
            if item.resource_key == "cluster/gpu-a/eks"
            else item
            for item in deployed.resources
        ]
    )

    rebuilt = registry.build_installation_snapshot(
        site, _state(SUBNET_A, SUBNET_B), _runtime(), existing=legacy
    )

    assert _identities(rebuilt) == _identities(legacy)


def _sync_events(
    site, monkeypatch, *, state: dict[str, Any], existing: InstallationResourceSnapshot
) -> list[Any]:
    events: list[Any] = []
    monkeypatch.setattr(registry, "find_bootstrap_state", lambda _site: state)
    monkeypatch.setattr(
        registry,
        "fetch_installation_resource_registry",
        lambda _site, **kwargs: events.append(("fetch", kwargs["allow_empty"]))
        or existing,
    )
    monkeypatch.setattr(
        registry,
        "discover_runtime_resources",
        lambda _site: events.append("runtime") or _runtime(),
    )
    monkeypatch.setattr(
        registry,
        "sync_installation_resource_snapshot",
        lambda _site, value: events.append(("sync", value)),
    )
    return events


def test_the_deploy_sync_reads_the_registry_without_route53_associations(
    site, monkeypatch
) -> None:
    """The live failure: a joined HyperPod row, rebuilt after a fetch that the
    old sync skipped unless the bootstrap recorded Route53 associations."""

    deployed = registry.build_installation_snapshot(
        site, _state(SUBNET_A, SUBNET_B), _runtime()
    )
    joined = _joined(deployed)
    events = _sync_events(
        site, monkeypatch, state=_state(SUBNET_B, SUBNET_A), existing=joined
    )

    output = registry.sync_installation_resource_registry(site)

    assert events[:2] == [("fetch", True), "runtime"]
    assert [event[0] for event in events[2:]] == ["sync"]
    synced = events[2][1]
    assert _identities(synced) == _identities(joined)
    assert registry.load_installation_resource_snapshot(output) == synced
    store = _store(joined)
    for item in synced.resources:
        assert store.save_installation_resource(item) == item


def test_a_changed_identity_is_refused_by_key_and_field_before_any_write(
    site, monkeypatch
) -> None:
    deployed = registry.build_installation_snapshot(
        site, _state(SUBNET_A, SUBNET_B), _runtime()
    )
    other_arn = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-other"
    drifted = _sealed(
        [
            item.model_copy(update={"resource_arn": other_arn})
            if item.resource_key == "cluster/gpu-a/eks"
            else item
            for item in deployed.resources
        ]
    )
    events = _sync_events(
        site, monkeypatch, state=_state(SUBNET_A, SUBNET_B), existing=drifted
    )

    with pytest.raises(BootstrapError) as refused:
        registry.sync_installation_resource_registry(site)

    assert str(refused.value) == (
        "installation registry rebuild would change recorded identities: "
        "cluster/gpu-a/eks (resource_arn) -- " + registry.REPLACEMENT_REMEDY
    ), "the refusal names key, field and the one supported transition"
    assert not [event for event in events if event[0] == "sync"], (
        "a refused rebuild must not reach the registry"
    )
