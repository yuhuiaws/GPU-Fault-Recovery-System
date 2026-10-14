"""A plain-EKS spot replacement installs without a deploy, end to end.

Live: a spot interruption replaced a GPU node of the site's ownership-labelled
managed node group with a new instance. The installer Pod for the new
node stayed in ``FailedMount ... references non-existent secret key`` because
the executor's node-key sync was refused with 404 every minute: the control
plane had nothing that placed a never-installed eks node in the cluster.

This drives the real pieces against each other -- the control plane's route
(through the real ``RegionalExecutorClient`` over ASGI), the executor's sync
with the EC2 attester, the installer reconciler, the Agent registry and the
executor's node-action key view -- with fakes only for Kubernetes and AWS:

* before: the plain ask is refused and the reconciler's Job references a key
  the Secret does not hold (what the kubelet reports as FailedMount);
* after one sync pass: the key is in the Secret, the Job's mount resolves, the
  new Agent's signed heartbeat registers, the executor signs actions for the
  node without a restart, and the departed node's key is left to deploy;
* a node name another cluster reports fails closed;
* rotate-token's node inventory is the sync's: every node it re-installs has
  a key.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request

import httpx
import pytest

from gpu_fault.adapters.node_action.adapter import NodeActionWorkflowAdapter
from gpu_fault.app import create_app
from gpu_fault.cluster_executor import ClusterExecutorError, NodeActionKeySync
from gpu_fault.cluster_executor.regional_client import RegionalExecutorClient
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from gpu_fault.eks_node_evidence import EksNodeEvidence
from gpu_fault.fleet import SignedAgentHeartbeat, sign_agent_heartbeat
from gpu_fault.fleet_compatibility import NODE_ACTION_KEY_VERSION_DERIVED
from gpu_fault.gpu_metrics import (
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricSource,
)
from gpu_fault.node_action_keys import (
    derive_node_action_secret,
    resolve_node_action_secret,
)
from gpu_fault.node_installer_reconciler import (
    INSTALLER_STATE_ANNOTATION,
    NodeInstallerReconciler,
)
from gpu_fault.regional import RegionalClusterRegistration, cluster_token_sha256
from gpu_fault_release.regional_release_config import ClusterTarget
from gpu_fault_release.regional_release_fleet_rollout import target_node_names
from tests._builders import asgi_client, build_context
from tests.eks_provider._support import (
    EKS_CLUSTER,
    NODEGROUP,
    FakeEc2,
    FakeEks,
    ec2_instance,
    gpu_node,
)
from tests.fleet._support import SECRET, heartbeat, registry
from tests.node_agent.test_node_installer_reconciler import (
    ARTIFACT,
    BUNDLE,
    TEMPLATE,
    BatchApi,
    job,
    template,
)
from tests.regional._regional_support import NOW, TOKEN_A
from tests.regional._regional_support import registration as hyperpod_registration

EKS = EKS_CLUSTER
TOKEN = "e" * 32
NAMESPACE = "gpu-fault-system"
SECRET_NAME = "gpu-fault-node-action-keys"
SITE = "us-west-2-hp-cluster-0a1b2c3d"
SELECTOR = f"{OWNERSHIP_LABEL}={SITE}"
INSTALLED = "ip-10-0-40-17.us-west-2.compute.internal"
DEPARTED = "ip-10-0-33-205.us-west-2.compute.internal"
NEW = "ip-10-0-36-21.us-west-2.compute.internal"
NODES = {INSTALLED: ("i-0b000017", "10.0.40.17"), NEW: ("i-0b000021", "10.0.36.21")}


def _b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


class Cluster:
    """The GPU cluster's Kubernetes API: selector-filtered Nodes and the key Secret."""

    def __init__(self, keys: dict[str, str]) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        for name, (instance_id, address) in NODES.items():
            node = gpu_node(name, instance_id, labels={OWNERSHIP_LABEL: SITE})
            node["status"]["addresses"] = [{"type": "InternalIP", "address": address}]
            self.nodes[name] = node
        self.secret = {
            "metadata": {"name": SECRET_NAME, "resourceVersion": "41"},
            "data": {name: _b64(value) for name, value in keys.items()},
        }
        self.patches: list[dict[str, Any]] = []

    def list_node(self, label_selector: str | None = None, **_: Any) -> dict:
        assert label_selector == SELECTOR
        return {"items": list(self.nodes.values())}

    def patch_node(self, name: str, body: dict[str, Any], **_: Any) -> None:
        annotations = self.nodes[name]["metadata"].setdefault("annotations", {})
        annotations.update(body["metadata"]["annotations"])

    def read_namespaced_secret(self, name: str, namespace: str) -> dict[str, Any]:
        assert (name, namespace) == (SECRET_NAME, NAMESPACE)
        return self.secret

    def patch_namespaced_secret(
        self, name: str, namespace: str, body: dict[str, Any]
    ) -> None:
        assert (
            body["metadata"]["resourceVersion"]
            == self.secret["metadata"]["resourceVersion"]
        )
        self.patches.append(body)
        self.secret["data"].update(body["data"])

    def mounts(self, job: dict[str, Any]) -> str | None:
        """What the kubelet mounts for the Job's node-secret volume, or None."""

        volume = next(
            item
            for item in job["spec"]["template"]["spec"]["volumes"]
            if item["name"] == "node-secret"
        )
        assert volume["secret"]["secretName"] == SECRET_NAME
        (item,) = volume["secret"]["items"]
        encoded = self.secret["data"].get(item["key"])
        return None if encoded is None else base64.b64decode(encoded).decode()


class AsgiExecutorClient(RegionalExecutorClient):
    """The real wire client; only the socket is the control plane's ASGI app."""

    def __init__(self, app: Any) -> None:
        super().__init__("http://control-plane.test", EKS, TOKEN)
        self.app = app

    def _send(self, request: Request, *, timeout_seconds: float | None = None) -> bytes:
        async def exchange() -> httpx.Response:
            async with asgi_client(self.app) as client:
                return await client.request(
                    request.get_method(),
                    request.full_url.removeprefix("http://control-plane.test"),
                    headers=dict(request.header_items()),
                    content=request.data,
                )

        response = asyncio.run(exchange())
        if response.status_code >= 400:
            raise ClusterExecutorError(
                "regional control plane rejected request "
                f"({response.status_code}): {response.text}",
                status_code=response.status_code,
            )
        return response.content


def _control_plane():
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(
        RegionalClusterRegistration(
            cluster_id=EKS,
            region="us-west-2",
            eks_cluster_arn=f"arn:aws:eks:us-west-2:123456789012:cluster/{EKS}",
            provider="eks-managed",
            token_sha256=cluster_token_sha256(TOKEN),
            allowed_namespaces=["training", NAMESPACE],
            agent_endpoint_allowed_cidrs=["10.0.32.0/19"],
        )
    )
    context.store.save_regional_cluster(hyperpod_registration("cluster-a", TOKEN_A))
    context.fleet_registry = registry(store=context.store)
    return context, create_app(context)


def _sync(app: Any, cluster: Cluster) -> NodeActionKeySync:
    instances = []
    for name, (instance_id, address) in NODES.items():
        instance = ec2_instance(instance_id, private_dns=name)
        instance["PrivateIpAddress"] = address
        instances.append(instance)
    evidence = EksNodeEvidence(
        eks_cluster_name=EKS,
        node_selector=SELECTOR,
        ec2_client=FakeEc2(instances),
        eks_client=FakeEks(
            nodegroups={
                NODEGROUP: {"status": "ACTIVE", "labels": {OWNERSHIP_LABEL: SITE}}
            }
        ),
    )
    return NodeActionKeySync(
        AsgiExecutorClient(app),
        cluster,
        node_selector=SELECTOR,
        namespace=NAMESPACE,
        attester=evidence,
    )


def _reconciler(cluster: Cluster, batch: BatchApi) -> NodeInstallerReconciler:
    return NodeInstallerReconciler(
        cluster,
        batch,
        namespace=NAMESPACE,
        cluster_name=EKS,
        node_selector=SELECTOR,
        version="0.10.0",
        config_digest="config-sha",
        artifact_sha256=ARTIFACT,
        bundle_sha256=BUNDLE,
        template_sha256=TEMPLATE,
        job_template=template(),
        dcgm_metrics_url_template="http://{node_ip}:9400/metrics",
    )


def _installed_key() -> str:
    return derive_node_action_secret(SECRET, EKS, INSTALLED)


def test_a_spot_replacement_gets_its_key_and_an_agent_without_a_deploy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context, app = _control_plane()
    cluster = Cluster({INSTALLED: _installed_key(), DEPARTED: "d" * 64})
    batch = BatchApi()
    _reconciler(cluster, batch).reconcile_once()
    # max-unavailable 1: the first pass installs the first node in name order,
    # which is the replacement.
    jobs = {
        body["spec"]["template"]["spec"]["nodeName"]: body for _, body in batch.created
    }
    assert list(jobs) == [NEW]
    assert cluster.mounts(jobs[NEW]) is None, "the live FailedMount: no key yet"
    with pytest.raises(ClusterExecutorError, match="404"):
        AsgiExecutorClient(app).node_action_keys([NEW])

    assert _sync(app, cluster).run() == [NEW]

    key = derive_node_action_secret(SECRET, EKS, NEW)
    assert cluster.mounts(jobs[NEW]) == key, "the waiting installer Pod now mounts"
    assert base64.b64decode(cluster.secret["data"][DEPARTED]).decode() == "d" * 64, (
        "the departed node's key is the next deploy's to prune, not the sync's"
    )
    # The installed Agent signs its heartbeat with the mounted key; the control
    # plane verifies it with what it derives, so the node registers.
    beat = heartbeat(NEW).model_copy(
        update={
            "cluster_id": EKS,
            "node_action_key_version": NODE_ACTION_KEY_VERSION_DERIVED,
        }
    )
    record = context.fleet_registry.register(
        SignedAgentHeartbeat(heartbeat=beat, signature=sign_agent_heartbeat(beat, key))
    )
    assert (record.cluster_id, record.node_id) == (EKS, NEW)
    # The executor signs node actions with the kubelet's projection of the
    # patched Secret, read again on the miss -- no restart.
    directory = tmp_path / "node-action-keys"
    directory.mkdir()
    for name, encoded in cluster.secret["data"].items():
        (directory / name).write_text(base64.b64decode(encoded).decode())
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_KEYS_DIR", str(directory))
    monkeypatch.delenv("GPU_FAULT_NODE_ACTION_KEYS_JSON", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_ACTION_SECRET", raising=False)
    monkeypatch.setenv(
        "GPU_FAULT_NODE_AGENT_ENDPOINTS", json.dumps({NEW: f"https://{NEW}:9099"})
    )
    adapter = NodeActionWorkflowAdapter.from_environment()
    assert (
        resolve_node_action_secret(
            adapter.secret,
            adapter.node_secrets,
            EKS,
            NEW,
            NODE_ACTION_KEY_VERSION_DERIVED,
        )
        == key
    )


def test_a_replacement_whose_name_another_cluster_reports_fails_closed() -> None:
    context, app = _control_plane()
    context.store.save_gpu_inventory_snapshot(
        GpuInventorySnapshot(
            cluster_id="cluster-a",
            node_id=NEW,
            observed_at=NOW,
            source=GpuMetricSource.NVIDIA_SMI,
            source_boot_id="boot-a",
            devices=[
                GpuInventoryDevice(
                    gpu_index=0,
                    gpu_uuid="GPU-0",
                    pci_bdf="0000:59:00.0",
                    product="H200",
                )
            ],
        )
    )
    cluster = Cluster({INSTALLED: _installed_key()})
    sync = _sync(app, cluster)

    with pytest.raises(ClusterExecutorError, match="already known to another cluster"):
        sync.run()

    assert cluster.patches == []
    assert NEW not in cluster.secret["data"]


def test_rotate_token_re_installs_exactly_the_nodes_the_sync_keys() -> None:
    """The rotation's waves come from the same selector; every node has a key."""

    _context, app = _control_plane()
    cluster = Cluster({INSTALLED: _installed_key()})
    _sync(app, cluster).run()

    class Release:
        def _gpu(self, target: ClusterTarget, *arguments: str) -> list[str]:
            return ["kubectl", "--context", target.context, *arguments]

        def _get_json(self, command: list[str]) -> dict[str, Any]:
            assert command[-2:] == ["-l", SELECTOR]
            return {"items": list(cluster.nodes.values())}

    target = ClusterTarget(
        cluster_id=EKS,
        context="gpu-context-a",
        executor_irsa_role_arn="arn:aws:iam::123456789012:role/executor",
        region="us-west-2",
        hyperpod_cluster_name="",
        eks_cluster_arn=f"arn:aws:eks:us-west-2:123456789012:cluster/{EKS}",
        provider="eks-managed",
        node_selector=SELECTOR,
    )

    rolled = target_node_names(Release(), target)

    assert rolled == tuple(sorted(NODES)), "the replacement is in the rotation"
    assert all(name in cluster.secret["data"] for name in rolled), (
        "every node the rotation re-installs mounts its own key"
    )


def test_an_install_that_failed_before_the_key_restarts_on_the_next_pass() -> None:
    """No manual Job delete and no backoff: the key's arrival replays the node."""

    _context, app = _control_plane()
    cluster = Cluster({INSTALLED: _installed_key()})
    batch = BatchApi()
    reconciler = _reconciler(cluster, batch)

    def jobs_for_new() -> list[dict[str, Any]]:
        return [
            body
            for _, body in batch.created
            if body["spec"]["template"]["spec"]["nodeName"] == NEW
        ]

    reconciler.reconcile_once()
    (first,) = jobs_for_new()
    assert cluster.mounts(first) is None, "created while the key was missing"
    # FailedMount until activeDeadlineSeconds: the Job fails, the node is parked.
    # (The fake answers a Job by a name fragment; the full name is one.)
    name = first["metadata"]["name"]
    batch.jobs[name] = job(condition="Failed", failed_at=datetime.now(UTC))
    reconciler.reconcile_once()
    reconciler.reconcile_once()
    annotations = cluster.nodes[NEW]["metadata"]["annotations"]
    assert annotations[INSTALLER_STATE_ANNOTATION] == "Failed", annotations
    assert len(jobs_for_new()) == 1, "without the key the backoff holds, as before"
    assert not [deleted for deleted in batch.deleted if deleted[0] == name]

    assert _sync(app, cluster).run() == [NEW]
    result = reconciler.reconcile_once()

    assert result["created"] >= 1, result
    assert [deleted[0] for deleted in batch.deleted if deleted[0] == name] == [name], (
        "the failed Job was replaced, not left behind"
    )
    replacement = jobs_for_new()[-1]
    assert len(jobs_for_new()) == 2, "replaced in the first pass after the key landed"
    assert replacement["metadata"]["name"] == name
    assert cluster.mounts(replacement) == derive_node_action_secret(SECRET, EKS, NEW)
    assert cluster.nodes[NEW]["metadata"]["annotations"][
        INSTALLER_STATE_ANNOTATION
    ] == ("Installing")
