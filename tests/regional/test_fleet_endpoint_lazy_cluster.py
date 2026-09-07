"""Online-joined clusters must not need a Pod restart (architecture review H4).

The fleet registry's per-cluster agent-endpoint allow-list was a dict built
once at start-up from the store, so a cluster joined online was refused
(409 on heartbeat) until every CPU Pod restarted. It now resolves unseen
clusters from the registry runtime snapshot on first sight, and still fails
closed for a cluster the snapshot does not know.
"""

from __future__ import annotations

import ipaddress
from datetime import datetime, timezone

import pytest

from gpu_fault.fleet import FleetRegistry
from gpu_fault.fleet_endpoint import (
    ClusterEndpointNetworks,
    endpoint_networks_for_cluster,
)
from gpu_fault.regional import RegionalRegistryRevision
from gpu_fault.regional_registry_runtime import RegionalRegistryRuntime
from gpu_fault.store import InMemoryStore
from tests._builders import build_context
from tests.regional._regional_support import TOKEN_A, TOKEN_B, registration

NOW = datetime(2026, 9, 7, tzinfo=timezone.utc)
SECRET = "s" * 32


def _runtime(store: InMemoryStore) -> RegionalRegistryRuntime:
    return RegionalRegistryRuntime.bootstrap(
        store,
        member_id="pod-a",
        service_role="ingress",
        release_id="release-a",
        poll_seconds=1,
        stale_seconds=5,
        now=lambda: NOW,
    )


def _join(store: InMemoryStore, runtime: RegionalRegistryRuntime, cluster_id: str):
    current = runtime.snapshot()
    store.publish_regional_registry_revision(
        RegionalRegistryRevision.build(
            generation=current.generation + 1,
            registrations=[
                *current.registrations.values(),
                registration(cluster_id, TOKEN_B).model_copy(
                    update={"agent_endpoint_allowed_cidrs": ["10.77.0.0/16"]}
                ),
            ],
            previous_generation=current.generation,
            required_member_ids=[],
            reason=f"join {cluster_id}",
            created_at=NOW,
        ),
        expected_generation=current.generation,
    )
    runtime.refresh_once(raise_on_failure=True)


def test_cluster_joined_after_app_creation_resolves_its_endpoint_networks() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.fleet_registry = FleetRegistry(
        context.store,
        SECRET,
        endpoint_allowed_networks_by_cluster={
            "cluster-a": (ipaddress.ip_network("10.0.0.0/16"),)
        },
    )
    runtime = _runtime(context.store)
    context.bind_regional_registry_runtime(runtime)

    _join(context.store, runtime, "cluster-b")

    networks = endpoint_networks_for_cluster(
        "cluster-b", context.fleet_registry.endpoint_allowed_networks_by_cluster, ()
    )
    assert networks == (ipaddress.ip_network("10.77.0.0/16"),), (
        "a cluster the registry runtime knows must be honoured without a restart"
    )
    assert endpoint_networks_for_cluster(
        "cluster-a", context.fleet_registry.endpoint_allowed_networks_by_cluster, ()
    ) == (ipaddress.ip_network("10.0.0.0/16"),), "the start-up clusters still work"


def test_unknown_cluster_still_fails_closed() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.fleet_registry = FleetRegistry(context.store, SECRET)
    context.bind_regional_registry_runtime(_runtime(context.store))

    with pytest.raises(ValueError, match="unavailable for cluster ghost"):
        endpoint_networks_for_cluster(
            "ghost",
            context.fleet_registry.endpoint_allowed_networks_by_cluster,
            (ipaddress.ip_network("10.0.0.0/8"),),
        )


def test_registry_backed_networks_never_fall_back_to_the_global_list() -> None:
    """Empty at start-up is not "unconfigured" once a registry backs the map."""

    networks = ClusterEndpointNetworks(
        {}, resolver=lambda cluster_id: () if cluster_id == "cluster-z" else None
    )

    assert (
        endpoint_networks_for_cluster(
            "cluster-z", networks, (ipaddress.ip_network("10.0.0.0/8"),)
        )
        == ()
    ), "the cluster's own (empty) allow-list wins over the global fallback"
    with pytest.raises(ValueError, match="unavailable for cluster ghost"):
        endpoint_networks_for_cluster(
            "ghost", networks, (ipaddress.ip_network("10.0.0.0/8"),)
        )
