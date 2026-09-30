"""``POST /v1/regional/node-action-keys``: keys for replacement nodes.

``deploy``/``join-cluster`` key the nodes present at that time; a node HyperPod
replaced afterwards (live 2026-09-30: ``hyperpod-i-00000000000000002``) had no
entry in ``gpu-fault-node-action-keys``, so every installer Job for it failed on
the missing Secret key and the node never got an Agent. The executor asks this
route for the missing names. What the route must hold:

* cluster-token bucket, and the payload ``cluster_id`` is bound to the
  authenticated cluster (the middleware refuses a foreign one);
* only node ids the control plane already knows from this cluster's own
  data-plane evidence are keyed -- anything else is one 404 for the request;
* no fleet master, no keys (503), never a crash;
* the value is exactly what the control plane signs with
  (``derive_node_action_secret`` for a node without a rotated key), and it is
  never logged.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import httpx
import pytest

from gpu_fault.app import create_app
from gpu_fault.app.authorization import ExplicitAuthorizationRegistry
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSource,
)
from gpu_fault.managed_recovery import HyperPodNodeIdentity
from gpu_fault.node_action_keys import derive_node_action_secret
from gpu_fault.regional import (
    NODE_ACTION_KEY_REQUEST_MAX_NODES,
    RemoteNodeActionKeyRequest,
)
from tests._builders import asgi_client, build_context
from tests.fleet._support import SECRET, heartbeat, registry, signed
from tests.regional._regional_support import NOW, TOKEN_A, TOKEN_B, registration

PATH = "/v1/regional/node-action-keys"
HEADERS_A = {
    "Authorization": f"Bearer {TOKEN_A}",
    "X-GPU-Fault-Cluster-ID": "cluster-a",
}
HEADERS_B = {
    "Authorization": f"Bearer {TOKEN_B}",
    "X-GPU-Fault-Cluster-ID": "cluster-b",
}
REPLACEMENT = "hyperpod-i-00000000000000002"


def _context(*, with_registry: bool = True):
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    if with_registry:
        fleet_registry = registry(store=context.store)
        fleet_registry.register(signed(heartbeat("node-a")))
        context.fleet_registry = fleet_registry
    return context


def _post(context, headers: dict[str, str], payload: dict) -> httpx.Response:
    async def scenario() -> httpx.Response:
        async with asgi_client(context) as client:
            return await client.post(PATH, headers=headers, json=payload)

    return asyncio.run(scenario())


def _inventory(cluster_id: str, node_id: str) -> GpuInventorySnapshot:
    return GpuInventorySnapshot(
        cluster_id=cluster_id,
        node_id=node_id,
        observed_at=NOW,
        source=GpuMetricSource.NVIDIA_SMI,
        source_boot_id="boot-replacement",
        devices=[
            GpuInventoryDevice(
                gpu_index=0,
                gpu_uuid="GPU-replacement-0",
                pci_bdf="0000:59:00.0",
                product="H100",
            )
        ],
    )


def test_the_route_is_in_the_cluster_token_bucket() -> None:
    app = create_app(_context())
    declared = ExplicitAuthorizationRegistry()
    declared.load(app.routes)

    assert declared.declared(PATH, "POST") == "cluster-token"


def test_a_foreign_payload_cluster_is_refused_by_the_binding_check() -> None:
    context = _context()
    context.store.save_gpu_inventory_snapshot(_inventory("cluster-b", REPLACEMENT))

    response = _post(
        context, HEADERS_A, {"cluster_id": "cluster-b", "node_ids": [REPLACEMENT]}
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "authenticated cluster does not match all payload cluster_id values"
    )


def test_a_node_nothing_in_the_cluster_has_reported_is_not_keyed() -> None:
    context = _context()
    # Known to cluster-b only: another cluster's evidence never keys a name here.
    context.store.save_gpu_inventory_snapshot(_inventory("cluster-b", REPLACEMENT))

    response = _post(
        context,
        HEADERS_A,
        {"cluster_id": "cluster-a", "node_ids": ["node-a", REPLACEMENT]},
    )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail == (
        "one or more node_ids are not known to the control plane for the "
        "authenticated cluster"
    )
    # The refusal covers the whole request: the known node's key is not leaked
    # alongside the refusal either.
    assert "keys" not in response.json()


def test_without_a_fleet_registry_the_route_answers_503_not_a_crash() -> None:
    context = _context(with_registry=False)
    context.store.save_gpu_inventory_snapshot(_inventory("cluster-a", REPLACEMENT))

    response = _post(
        context, HEADERS_A, {"cluster_id": "cluster-a", "node_ids": [REPLACEMENT]}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "agent registry is disabled"


def test_a_missing_fleet_master_answers_503() -> None:
    context = _context()
    context.store.save_gpu_inventory_snapshot(_inventory("cluster-a", REPLACEMENT))
    # ``FleetRegistry`` refuses a short secret at construction, so the
    # unavailable-master state is reached by emptying it on the live registry.
    context.fleet_registry.secret = ""

    response = _post(
        context, HEADERS_A, {"cluster_id": "cluster-a", "node_ids": [REPLACEMENT]}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "node action fleet secret is unavailable"


def test_known_nodes_get_the_key_the_control_plane_signs_with(caplog) -> None:
    context = _context()
    # Three kinds of evidence, one node each: a heartbeating Agent (node-a), a
    # GPU inventory snapshot for the replacement node (the DCGM path reports it
    # long before any Agent exists), and a HyperPod identity naming a third.
    context.store.save_gpu_inventory_snapshot(_inventory("cluster-a", REPLACEMENT))
    context.store.save_hyperpod_node_identity(
        HyperPodNodeIdentity(
            cluster_name="hp-cluster-a",
            node_logical_id="worker-3",
            instance_id="i-3",
            kubernetes_node_name="hyperpod-i-3",
            status="Running",
            aliases=["worker-3", "i-3", "hyperpod-i-3"],
            retired_aliases=["hyperpod-i-retired"],
            observed_at=NOW,
        )
    )
    node_ids = ["node-a", REPLACEMENT, "hyperpod-i-3"]

    with caplog.at_level(logging.INFO, logger="gpu_fault.app.routes.regional"):
        response = _post(
            context, HEADERS_A, {"cluster_id": "cluster-a", "node_ids": node_ids}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cluster_id"] == "cluster-a"
    assert set(body["keys"]) == set(node_ids)
    for node_id in node_ids:
        expected = derive_node_action_secret(SECRET, "cluster-a", node_id)
        assert body["keys"][node_id] == expected
        # Cluster-scoped: the same name in another cluster derives differently.
        assert body["keys"][node_id] != derive_node_action_secret(
            SECRET, "cluster-b", node_id
        )
    # The audit line names the cluster and the nodes, never a key value.
    audit = [
        record
        for record in caplog.records
        if "issued node action keys" in record.getMessage()
    ]
    assert len(audit) == 1
    assert "cluster-a" in audit[0].getMessage()
    assert all(node_id in audit[0].getMessage() for node_id in node_ids), (
        "the audit line must name every node id"
    )
    assert not any(
        value in record.getMessage()
        for record in caplog.records
        for value in body["keys"].values()
    ), "a derived key leaked into the audit line"


def test_a_retired_hyperpod_alias_is_not_evidence() -> None:
    """A replaced node's old name must not be re-keyed off its identity record."""

    context = _context()
    context.store.save_hyperpod_node_identity(
        HyperPodNodeIdentity(
            cluster_name="hp-cluster-a",
            node_logical_id="worker-3",
            kubernetes_node_name="hyperpod-i-3",
            status="Running",
            aliases=["hyperpod-i-3"],
            retired_aliases=["hyperpod-i-retired"],
            observed_at=NOW,
        )
    )

    response = _post(
        context,
        HEADERS_A,
        {"cluster_id": "cluster-a", "node_ids": ["hyperpod-i-retired"]},
    )

    assert response.status_code == 404


def test_a_rotated_key_in_the_cpu_mirror_wins_over_derivation() -> None:
    """The route answers what the control plane signs with, not blindly HMAC."""

    context = _context()
    context.store.save_gpu_inventory_snapshot(_inventory("cluster-a", REPLACEMENT))
    rotated = "r" * 64
    context.fleet_registry.node_secrets[REPLACEMENT] = rotated

    response = _post(
        context, HEADERS_A, {"cluster_id": "cluster-a", "node_ids": [REPLACEMENT]}
    )

    assert response.status_code == 200
    assert response.json()["keys"] == {REPLACEMENT: rotated}


def test_the_other_cluster_derives_its_own_scope() -> None:
    context = _context()
    context.store.save_gpu_inventory_snapshot(
        _inventory("cluster-b", REPLACEMENT).model_copy(
            update={"observed_at": NOW + timedelta(seconds=1)}
        )
    )

    response = _post(
        context, HEADERS_B, {"cluster_id": "cluster-b", "node_ids": [REPLACEMENT]}
    )

    assert response.status_code == 200
    assert response.json()["keys"] == {
        REPLACEMENT: derive_node_action_secret(SECRET, "cluster-b", REPLACEMENT)
    }


@pytest.mark.parametrize(
    "node_ids",
    [
        [],
        ["node-a"] * 2,
        ["bad node"],
        ["../etc"],
        [f"node-{index}" for index in range(NODE_ACTION_KEY_REQUEST_MAX_NODES + 1)],
    ],
)
def test_the_request_is_bounded_and_node_ids_are_validated(node_ids) -> None:
    with pytest.raises(ValueError):
        RemoteNodeActionKeyRequest(cluster_id="cluster-a", node_ids=node_ids)


def test_a_malformed_body_is_a_validation_error_not_a_key() -> None:
    context = _context()

    response = _post(
        context, HEADERS_A, {"cluster_id": "cluster-a", "node_ids": ["bad node"]}
    )

    assert response.status_code == 422
