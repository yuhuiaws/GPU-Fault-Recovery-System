"""``POST /v1/regional/node-action-keys`` for a plain EKS node that joined late.

Live: a spot interruption replaced a GPU node with a new instance of the same
ownership-labelled managed node group. The executor's node-key sync
asked for the new name every minute and got 404, because nothing on the control
plane places a fresh ``eks-managed`` node -- no Agent record, no telemetry (both
come from what the installer puts on the node) and no SageMaker identity -- so
the installer Pod stayed in ``FailedMount`` until the next deploy. The executor
now attests the node with what EC2 says about the instance behind it. What the
route must hold:

* an attestation keys a node only when it binds to the cluster's registration:
  eks-managed, the registered EKS cluster's tag, a running instance named like
  the Node, an address in the registered node subnets;
* a name another cluster reports, or one the CPU node-key mirror holds a
  different key for, fails closed (409) -- the deploy host's own rule;
* a HyperPod cluster never admits a statement (SageMaker places its nodes), so
  a retired alias cannot be re-keyed that way;
* one refusal withholds every key of the request; values are never logged.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest

from gpu_fault.fleet_compatibility import NODE_ACTION_KEY_VERSION_DERIVED
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSource,
)
from gpu_fault.managed_recovery import HyperPodNodeIdentity
from gpu_fault.node_action_keys import (
    derive_node_action_secret,
    resolve_node_action_secret,
)
from gpu_fault.node_key_attestation import (
    AttestedNodeActionKeyRequest,
    EksNodeAttestation,
)
from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalClusterRegistration,
    cluster_token_sha256,
)
from tests._builders import asgi_client, build_context
from tests.fleet._support import SECRET, registry
from tests.regional._regional_support import NOW, TOKEN_A, registration

PATH = "/v1/regional/node-action-keys"
EKS = "gpu-eks-cluster"
TOKEN = "e" * 32
HEADERS = {"Authorization": f"Bearer {TOKEN}", "X-GPU-Fault-Cluster-ID": EKS}
HEADERS_HYPERPOD = {
    "Authorization": f"Bearer {TOKEN_A}",
    "X-GPU-Fault-Cluster-ID": "cluster-a",
}
NODE = "ip-10-0-36-21.us-west-2.compute.internal"
KNOWN = "ip-10-0-33-205.us-west-2.compute.internal"
NODEGROUP = "gpu-p5en-efa-spot-node-group"


def _eks_registration(
    cluster_id: str = EKS, token: str = TOKEN, **overrides: Any
) -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id=cluster_id,
        region="us-west-2",
        eks_cluster_arn=f"arn:aws:eks:us-west-2:123456789012:cluster/{cluster_id}",
        provider="eks-managed",
        token_sha256=cluster_token_sha256(token),
        allowed_namespaces=["training", "gpu-fault-system"],
        agent_endpoint_allowed_cidrs=["10.0.32.0/19"],
        **overrides,
    )


def _context():
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(_eks_registration())
    # A HyperPod cluster on the same site (the live topology).
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.fleet_registry = registry(store=context.store)
    return context


def _attestation(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "node_id": NODE,
        "node_uid": "5d7c0a3e-4b1f-4c2a-9a37-7d1f0c9e2b11",
        "instance_id": "i-0b000021",
        "instance_state": "running",
        "private_dns_name": NODE,
        "private_ip": "10.0.36.21",
        "eks_cluster_name": EKS,
        "nodegroup": NODEGROUP,
    }
    values.update(overrides)
    return values


def _payload(*attestations: dict[str, Any], node_ids: list[str] | None = None):
    return {
        "cluster_id": EKS,
        "node_ids": node_ids or [item["node_id"] for item in attestations],
        "nodes": list(attestations),
    }


def _post(
    context, payload: dict[str, Any], headers: dict[str, str] = HEADERS
) -> httpx.Response:
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
        source_boot_id="boot-a",
        devices=[
            GpuInventoryDevice(
                gpu_index=0, gpu_uuid="GPU-0", pci_bdf="0000:59:00.0", product="H200"
            )
        ],
    )


def test_a_fresh_eks_node_without_an_attestation_is_still_unknown() -> None:
    """The live refusal: no evidence of any kind exists before the first install."""

    response = _post(_context(), {"cluster_id": EKS, "node_ids": [NODE]})

    assert response.status_code == 404, response.text
    assert "not known to the control plane" in response.json()["detail"]


def test_an_attested_eks_node_gets_the_key_the_control_plane_signs_with(
    caplog: pytest.LogCaptureFixture,
) -> None:
    context = _context()

    with caplog.at_level(logging.INFO, logger="gpu_fault.app.routes.regional"):
        response = _post(context, _payload(_attestation()))

    assert response.status_code == 200, response.text
    keys = response.json()["keys"]
    assert keys == {NODE: derive_node_action_secret(SECRET, EKS, NODE)}
    # Exactly what the control plane verifies the new Agent's heartbeats with.
    assert keys[NODE] == resolve_node_action_secret(
        SECRET,
        context.fleet_registry.node_secrets,
        EKS,
        NODE,
        NODE_ACTION_KEY_VERSION_DERIVED,
    )
    messages = [record.getMessage() for record in caplog.records]
    admitted = [message for message in messages if "EC2 identity" in message]
    assert len(admitted) == 1, messages
    for fragment in (EKS, NODE, "i-0b000021", NODEGROUP, "5d7c0a3e"):
        assert fragment in admitted[0], (fragment, admitted[0])
    assert any("issued node action keys" in message for message in messages), (
        "the issue audit line is missing"
    )
    assert not any(keys[NODE] in message for message in messages), (
        "a derived key leaked into a log line"
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"eks_cluster_name": "another-eks"}, "not the registered one"),
        ({"instance_state": "stopped"}, "not running"),
        (
            {"private_dns_name": "ip-10-0-36-22.us-west-2.compute.internal"},
            "private DNS name",
        ),
        ({"private_ip": "10.0.36.22"}, "encodes another address"),
        (
            {
                "node_id": "ip-10-1-0-5.us-west-2.compute.internal",
                "private_dns_name": "ip-10-1-0-5.us-west-2.compute.internal",
                "private_ip": "10.1.0.5",
            },
            "outside the cluster's registered node subnets",
        ),
        (
            {
                "node_id": "i-0b000021.us-west-2.compute.internal",
                "private_dns_name": "i-0b000021.us-west-2.compute.internal",
                "private_ip": "169.254.10.10",
            },
            "outside the cluster's registered node subnets",
        ),
    ],
)
def test_an_attestation_that_does_not_bind_to_the_registration_keys_nothing(
    overrides: dict[str, Any], reason: str
) -> None:
    response = _post(_context(), _payload(_attestation(**overrides)))

    assert response.status_code == 403, response.text
    assert reason in response.json()["detail"]
    assert "keys" not in response.json()


def test_a_hyperpod_cluster_never_admits_an_attestation() -> None:
    """SageMaker places a HyperPod node; a statement cannot re-key a retired alias."""

    context = _context()
    context.store.save_hyperpod_node_identity(
        HyperPodNodeIdentity(
            cluster_name="hp-cluster-a",
            node_logical_id="worker-3",
            kubernetes_node_name="hyperpod-i-00000000000000001",
            status="Running",
            aliases=["hyperpod-i-00000000000000001"],
            retired_aliases=["hyperpod-i-00000000000000002"],
            observed_at=NOW,
        )
    )
    retired = "hyperpod-i-00000000000000002"
    payload = {
        "cluster_id": "cluster-a",
        "node_ids": [retired],
        "nodes": [
            _attestation(
                node_id=retired, private_dns_name=retired, eks_cluster_name="cluster-a"
            )
        ],
    }

    response = _post(context, payload, HEADERS_HYPERPOD)

    assert response.status_code == 403, response.text
    assert "SageMaker identity" in response.json()["detail"]


def test_a_name_another_cluster_already_reports_fails_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    context = _context()
    context.store.save_gpu_inventory_snapshot(_inventory("cluster-a", NODE))

    with caplog.at_level(logging.WARNING, logger="gpu_fault.app.routes.regional"):
        response = _post(context, _payload(_attestation()))

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert "already known to another cluster" in detail
    assert "exclusive cluster ownership requires reconciliation" in detail
    assert "cluster-a" not in detail, "another cluster's records stay on the CPU"
    assert any(
        "already known to cluster cluster-a" in record.getMessage()
        for record in caplog.records
    ), "the operator's log names the cluster that holds the name"


def test_a_removed_cluster_s_leftovers_do_not_block_the_name() -> None:
    context = _context()
    context.store.save_regional_cluster(
        _eks_registration(
            "gpu-eks-old",
            "o" * 32,
            enabled=False,
            lifecycle_state=RegionalClusterLifecycle.REVOKED,
        )
    )
    context.store.save_gpu_inventory_snapshot(_inventory("gpu-eks-old", NODE))

    response = _post(context, _payload(_attestation()))

    assert response.status_code == 200, response.text


def test_a_mirror_key_that_is_not_this_cluster_s_fails_closed() -> None:
    """The deploy host's CPU NodeName key conflict, enforced at sync time."""

    context = _context()
    context.fleet_registry.node_secrets[NODE] = "r" * 64

    response = _post(context, _payload(_attestation()))

    assert response.status_code == 409, response.text
    assert "CPU NodeName key conflict" in response.json()["detail"]
    assert "keys" not in response.json()


def test_a_mirror_that_already_holds_this_cluster_s_key_is_no_conflict() -> None:
    context = _context()
    derived = derive_node_action_secret(SECRET, EKS, NODE)
    context.fleet_registry.node_secrets[NODE] = derived

    response = _post(context, _payload(_attestation()))

    assert response.status_code == 200, response.text
    assert response.json()["keys"] == {NODE: derived}


def test_a_known_node_and_an_attested_one_are_keyed_together() -> None:
    context = _context()
    context.store.save_gpu_inventory_snapshot(_inventory(EKS, KNOWN))

    response = _post(context, _payload(_attestation(), node_ids=[KNOWN, NODE]))

    assert response.status_code == 200, response.text
    assert set(response.json()["keys"]) == {KNOWN, NODE}


def test_one_refused_attestation_withholds_every_key_of_the_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    context = _context()
    context.store.save_gpu_inventory_snapshot(_inventory(EKS, KNOWN))
    second = "ip-10-0-36-30.us-west-2.compute.internal"
    payload = _payload(
        _attestation(),
        _attestation(
            node_id=second,
            private_dns_name=second,
            private_ip="10.0.36.30",
            instance_state="terminated",
        ),
        node_ids=[KNOWN, NODE, second],
    )

    with caplog.at_level(logging.INFO, logger="gpu_fault.app.routes.regional"):
        refused = _post(context, payload)

    assert refused.status_code == 403, refused.text
    assert "keys" not in refused.json()
    assert not [
        record for record in caplog.records if "EC2 identity" in record.getMessage()
    ], "no admission is audited for a request that issued nothing"

    response = _post(
        context,
        _payload(_attestation(instance_state="terminated"), node_ids=[KNOWN, NODE]),
    )

    assert response.status_code == 403, response.text
    assert "keys" not in response.json()


@pytest.mark.parametrize(
    "payload",
    [
        {"cluster_id": EKS, "node_ids": [KNOWN], "nodes": [_attestation()]},
        {"cluster_id": EKS, "node_ids": [NODE], "nodes": [_attestation()] * 2},
        {"cluster_id": EKS, "node_ids": [NODE], "nodes": [_attestation(node_uid="")]},
        {
            "cluster_id": EKS,
            "node_ids": [NODE],
            "nodes": [_attestation(instance_id="instance-1")],
        },
        {
            "cluster_id": EKS,
            "node_ids": [NODE],
            "nodes": [_attestation(private_ip="not-an-address")],
        },
    ],
)
def test_a_malformed_attestation_is_a_validation_error(payload) -> None:
    with pytest.raises(ValueError):
        AttestedNodeActionKeyRequest.model_validate(payload)

    response = _post(_context(), payload)

    assert response.status_code == 422, response.text


def test_the_attestation_carries_no_field_for_key_material() -> None:
    assert set(EksNodeAttestation.model_fields) == {
        "node_id",
        "node_uid",
        "instance_id",
        "instance_state",
        "private_dns_name",
        "private_ip",
        "eks_cluster_name",
        "nodegroup",
    }
