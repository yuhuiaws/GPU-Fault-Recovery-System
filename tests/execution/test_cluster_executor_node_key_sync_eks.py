"""The node-key sync keys a plain-EKS replacement node without a deploy.

Live: a spot interruption replaced a GPU node of the site's managed node group
with a new instance. The sync asked the control plane for the new
name every 60 s and got 404 (nothing there places a never-installed eks node),
so the installer Pod stayed in FailedMount. Now a 404 for an ``eks-managed``
cluster is followed by one request per node carrying the EC2 identity
``EksNodeEvidence`` read for it. What the sync must hold on that path:

* the node set is the deploy host's: valid Node names, a UID each, no repeats,
  every node matching the selector -- an inventory that breaks any of these is
  not trusted at all; a node being deleted is left alone;
* a node outside the site's node groups is never asked for;
* one refused node keys nothing for itself and does not hold back the others;
  a request without a verdict writes nothing and the next pass retries;
* the departed node's key is never rewritten or deleted (deploy prunes it).
"""

from __future__ import annotations

import base64
import copy
from typing import Any

import pytest

from gpu_fault.cluster_executor import ClusterExecutorError, NodeActionKeySync
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from gpu_fault.eks_node_evidence import EksNodeEvidence
from gpu_fault.node_installer_reconciler import (
    INSTALLER_REASON_ANNOTATION,
    INSTALLER_STATE_ANNOTATION,
    NODE_KEY_ADDED_REASON,
)
from gpu_fault.node_key_attestation import EksNodeAttestation
from tests.eks_provider._support import (
    EKS_CLUSTER,
    NODEGROUP,
    FakeCore,
    FakeEc2,
    FakeEks,
    ec2_instance,
    gpu_node,
)

CLUSTER = "gpu-eks-cluster"
NAMESPACE = "gpu-fault-system"
SECRET = "gpu-fault-node-action-keys"
SITE = "us-west-2-hp-cluster-0a1b2c3d"
SELECTOR = f"{OWNERSHIP_LABEL}={SITE}"
DEPARTED = "ip-10-0-33-205.us-west-2.compute.internal"
NEW = "ip-10-0-36-21.us-west-2.compute.internal"
SIBLING = "ip-10-0-40-17.us-west-2.compute.internal"
KEY_OLD = "o" * 64
KEY_NEW = "n" * 64
NOT_KNOWN = ClusterExecutorError(
    "regional control plane rejected request (404): not known", status_code=404
)


def _b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def _node(name: str, instance_id: str, address: str, **overrides: Any) -> dict:
    labels = overrides.pop("labels", {OWNERSHIP_LABEL: SITE})
    node = gpu_node(name, instance_id, labels=labels, **overrides)
    node["status"]["addresses"] = [{"type": "InternalIP", "address": address}]
    return node


def _instance(instance_id: str, name: str, address: str, **overrides: Any) -> dict:
    instance = ec2_instance(instance_id, private_dns=name, **overrides)
    instance["PrivateIpAddress"] = address
    return instance


class KeyedCore(FakeCore):
    """Selector-filtering nodes plus the one Secret the sync reads and patches."""

    def __init__(self, nodes: list[dict], keys: dict[str, str]) -> None:
        super().__init__(nodes)
        self.secret = {
            "metadata": {"name": SECRET, "resourceVersion": "41"},
            "data": {name: _b64(value) for name, value in keys.items()},
        }
        self.patches: list[dict[str, Any]] = []

    def read_namespaced_secret(self, name: str, namespace: str) -> dict[str, Any]:
        assert (name, namespace) == (SECRET, NAMESPACE)
        return self.secret

    def patch_namespaced_secret(
        self, name: str, namespace: str, body: dict[str, Any]
    ) -> None:
        assert (name, namespace) == (SECRET, NAMESPACE)
        self.patches.append(body)
        self.secret["data"].update(body["data"])


class AttestingClient:
    """A control plane that knows only ``known`` by evidence (404 for the rest
    unless attested) and answers attested nodes per ``verdicts``."""

    cluster_id = CLUSTER

    def __init__(
        self,
        verdicts: dict[str, Exception | str] | None = None,
        *,
        known: frozenset[str] = frozenset(),
    ) -> None:
        self.verdicts = verdicts or {}
        self.known = known
        self.requests: list[tuple[list[str], list[EksNodeAttestation] | None]] = []

    def node_action_keys(
        self, node_ids: list[str], attestations: list[EksNodeAttestation] | None = None
    ) -> dict[str, str]:
        self.requests.append((list(node_ids), attestations))
        if attestations is None:
            if not set(node_ids) <= self.known:
                raise NOT_KNOWN
            return dict.fromkeys(node_ids, KEY_NEW)
        assert [item.node_id for item in attestations] == node_ids
        verdict = self.verdicts.get(node_ids[0], KEY_NEW)
        if isinstance(verdict, Exception):
            raise verdict
        return {node_ids[0]: verdict}


def _world(nodes: list[dict], instances: list[dict]) -> tuple[KeyedCore, Any]:
    core = KeyedCore(nodes, {DEPARTED: KEY_OLD})
    evidence = EksNodeEvidence(
        eks_cluster_name=EKS_CLUSTER,
        node_selector=SELECTOR,
        ec2_client=FakeEc2(instances),
        eks_client=FakeEks(
            nodegroups={
                NODEGROUP: {"status": "ACTIVE", "labels": {OWNERSHIP_LABEL: SITE}}
            }
        ),
    )
    return core, evidence


def _sync(client: Any, core: KeyedCore, evidence: Any) -> NodeActionKeySync:
    return NodeActionKeySync(
        client, core, node_selector=SELECTOR, namespace=NAMESPACE, attester=evidence
    )


def test_a_replacement_the_control_plane_does_not_know_is_attested_and_keyed() -> None:
    core, evidence = _world(
        [_node(NEW, "i-0b000021", "10.0.36.21")],
        [_instance("i-0b000021", NEW, "10.0.36.21")],
    )
    client = AttestingClient()

    added = _sync(client, core, evidence).run()

    assert added == [NEW]
    assert [request[0] for request in client.requests] == [[NEW], [NEW]]
    assert client.requests[0][1] is None, "the first ask is the plain one"
    attestation = client.requests[1][1][0]
    assert (attestation.instance_id, attestation.private_ip) == (
        "i-0b000021",
        "10.0.36.21",
    )
    assert core.patches == [
        {"metadata": {"resourceVersion": "41"}, "data": {NEW: _b64(KEY_NEW)}}
    ], "one fenced patch adding only the new node"
    assert core.secret["data"][DEPARTED] == _b64(KEY_OLD), (
        "the departed node's key is neither rewritten nor deleted"
    )
    assert core.list_selectors == [SELECTOR]


def test_a_node_outside_the_site_scope_is_never_asked_for() -> None:
    unlabelled = _node(SIBLING, "i-0b000022", "10.0.40.17", labels={})
    core, evidence = _world(
        [unlabelled], [_instance("i-0b000022", SIBLING, "10.0.40.17")]
    )
    client = AttestingClient()

    assert _sync(client, core, evidence).run() == []

    assert client.requests == [], "a node the selector does not match is not asked for"
    assert core.patches == []


def test_a_self_labelled_node_of_a_foreign_node_group_gets_nothing() -> None:
    """A kubelet can label its own Node; it cannot change its node group's labels."""

    foreign = _node(SIBLING, "i-0b000022", "10.0.40.17", nodegroup="cpu-node-group-1")
    core, evidence = _world(
        [foreign],
        [_instance("i-0b000022", SIBLING, "10.0.40.17", nodegroup="cpu-node-group-1")],
    )
    client = AttestingClient()

    with pytest.raises(ClusterExecutorError, match="does not declare"):
        _sync(client, core, evidence).run()

    assert [request[1] for request in client.requests] == [None, None], (
        "a node that does not bind is never attested, only asked for plainly"
    )
    assert core.patches == []


def test_a_node_known_by_other_evidence_is_keyed_even_unattested() -> None:
    """A batch refused for its unknown member still keys the known one."""

    # The known node's instance is tagged for another node group: no statement.
    core, evidence = _world(
        [
            _node(NEW, "i-0b000021", "10.0.36.21"),
            _node(SIBLING, "i-0b000022", "10.0.40.17"),
        ],
        [
            _instance("i-0b000021", NEW, "10.0.36.21", nodegroup="other"),
            _instance("i-0b000022", SIBLING, "10.0.40.17"),
        ],
    )
    client = AttestingClient(known=frozenset({NEW}))

    added = _sync(client, core, evidence).run()

    assert added == [NEW, SIBLING]
    assert [(names, bool(attested)) for names, attested in client.requests] == [
        ([NEW, SIBLING], False),
        ([NEW], False),
        ([SIBLING], True),
    ]


def test_one_refused_node_does_not_hold_back_its_sibling() -> None:
    core, evidence = _world(
        [
            _node(NEW, "i-0b000021", "10.0.36.21"),
            _node(SIBLING, "i-0b000022", "10.0.40.17"),
        ],
        [
            _instance("i-0b000021", NEW, "10.0.36.21"),
            _instance("i-0b000022", SIBLING, "10.0.40.17"),
        ],
    )
    conflict = ClusterExecutorError(
        "regional control plane rejected request (409): CPU NodeName key conflict",
        status_code=409,
    )
    client = AttestingClient({NEW: conflict})
    sync = _sync(client, core, evidence)

    with pytest.raises(ClusterExecutorError, match="CPU NodeName key conflict"):
        sync.run()

    assert core.patches == [
        {"metadata": {"resourceVersion": "41"}, "data": {SIBLING: _b64(KEY_NEW)}}
    ]
    assert NEW not in core.secret["data"]
    assert (sync.added_total, sync.failures_total) == (1, 1)


def test_an_attested_request_without_a_verdict_writes_nothing() -> None:
    core, evidence = _world(
        [_node(NEW, "i-0b000021", "10.0.36.21")],
        [_instance("i-0b000021", NEW, "10.0.36.21")],
    )
    client = AttestingClient(
        {NEW: ClusterExecutorError("regional control plane request failed: timeout")}
    )

    with pytest.raises(ClusterExecutorError, match="timeout"):
        _sync(client, core, evidence).run()

    assert core.patches == []


def test_a_short_attested_answer_is_refused_before_any_patch() -> None:
    core, evidence = _world(
        [_node(NEW, "i-0b000021", "10.0.36.21")],
        [_instance("i-0b000021", NEW, "10.0.36.21")],
    )

    with pytest.raises(ClusterExecutorError, match="malformed"):
        _sync(AttestingClient({NEW: "short"}), core, evidence).run()

    assert core.patches == []


def test_without_an_attester_a_404_is_still_the_whole_answer() -> None:
    """An eks executor that could not build its attester keeps the old behaviour."""

    core, _evidence = _world(
        [_node(NEW, "i-0b000021", "10.0.36.21")],
        [_instance("i-0b000021", NEW, "10.0.36.21")],
    )
    client = AttestingClient()

    with pytest.raises(ClusterExecutorError, match="404"):
        _sync(client, core, None).run()

    assert [request[1] for request in client.requests] == [None]
    assert core.patches == []


def _inventory_breakage(kind: str) -> list[dict]:
    first = _node(NEW, "i-0b000021", "10.0.36.21")
    second = _node(SIBLING, "i-0b000022", "10.0.40.17")
    if kind == "repeated-uid":
        second["metadata"]["uid"] = first["metadata"]["uid"]
    elif kind == "missing-uid":
        second["metadata"]["uid"] = ""
    elif kind == "invalid-name":
        second["metadata"]["name"] = "Bad_Node"
    elif kind == "outside-selector":
        second["metadata"]["labels"] = {}
    return [first, second]


@pytest.mark.parametrize(
    "kind", ["repeated-uid", "missing-uid", "invalid-name", "outside-selector"]
)
def test_an_inventory_the_deploy_host_would_refuse_requests_nothing(kind) -> None:
    class UnfilteredCore(KeyedCore):
        """An API answer the selector did not filter, as a proxy or cache might."""

        def list_node(self, label_selector: str | None = None, **_: Any) -> dict:
            self.list_selectors.append(label_selector)
            return {"items": list(self.nodes.values())}

    core = UnfilteredCore(_inventory_breakage(kind), {DEPARTED: KEY_OLD})
    client = AttestingClient()

    with pytest.raises(ClusterExecutorError, match="invalid or ambiguous"):
        _sync(client, core, None).run()

    assert client.requests == []
    assert core.patches == []


def test_a_repeated_name_is_refused_like_the_deploy_host_refuses_it() -> None:
    class RepeatingCore(KeyedCore):
        def list_node(self, label_selector: str | None = None, **_: Any) -> dict:
            node = next(iter(self.nodes.values()))
            return {
                "items": [node, dict(node, metadata=dict(node["metadata"], uid="x"))]
            }

    core = RepeatingCore([_node(NEW, "i-0b000021", "10.0.36.21")], {DEPARTED: KEY_OLD})

    with pytest.raises(ClusterExecutorError, match="invalid or ambiguous"):
        _sync(AttestingClient(), core, None).run()


def test_a_node_being_deleted_is_not_keyed() -> None:
    leaving = _node(NEW, "i-0b000021", "10.0.36.21")
    leaving["metadata"]["deletionTimestamp"] = "2026-10-14T01:33:00Z"
    core, evidence = _world([leaving], [_instance("i-0b000021", NEW, "10.0.36.21")])
    client = AttestingClient()

    assert _sync(client, core, evidence).run() == []
    assert client.requests == []


class VersionedCore(KeyedCore):
    """Nodes as the API server keeps them: listed as copies, patched under a
    resourceVersion precondition, ``on_secret_patch`` runs between the two."""

    def __init__(self, nodes: list[dict], keys: dict[str, str]) -> None:
        super().__init__(nodes, keys)
        self.node_patches: list[tuple[str, dict[str, Any]]] = []
        self.on_secret_patch: Any = None

    def list_node(self, label_selector: str | None = None, **_: Any) -> dict:
        return copy.deepcopy(super().list_node(label_selector))

    def read_node(self, name: str, **_: Any) -> dict[str, Any]:
        return copy.deepcopy(super().read_node(name))

    def patch_namespaced_secret(
        self, name: str, namespace: str, body: dict[str, Any]
    ) -> None:
        super().patch_namespaced_secret(name, namespace, body)
        if self.on_secret_patch is not None:
            self.on_secret_patch(self)

    def patch_node(self, name: str, body: dict[str, Any], **_: Any) -> dict:
        self.node_patches.append((name, copy.deepcopy(body)))
        metadata = self.nodes[name]["metadata"]
        if body["metadata"].get("resourceVersion") not in {
            None,
            metadata["resourceVersion"],
        }:
            error = RuntimeError("the object has been modified")
            error.status = 409  # type: ignore[attr-defined]
            raise error
        metadata["resourceVersion"] = str(int(metadata["resourceVersion"]) + 1)
        return super().patch_node(name, body)


REPLAY = {
    INSTALLER_STATE_ANNOTATION: "Retrying",
    INSTALLER_REASON_ANNOTATION: NODE_KEY_ADDED_REASON,
}


def _installer_world(states: dict[str, str]) -> tuple[VersionedCore, Any]:
    addresses = {
        NEW: ("i-0b000021", "10.0.36.21"),
        SIBLING: ("i-0b000022", "10.0.40.17"),
    }
    nodes, instances = [], []
    for name, state in states.items():
        instance_id, address = addresses[name]
        item = _node(name, instance_id, address)
        item["metadata"]["annotations"] = {INSTALLER_STATE_ANNOTATION: state}
        nodes.append(item)
        instances.append(_instance(instance_id, name, address))
    _core, evidence = _world(nodes, instances)
    return VersionedCore(nodes, {DEPARTED: KEY_OLD}), evidence


def test_a_keyed_node_whose_install_failed_is_handed_back_to_the_reconciler() -> None:
    """Its Job could not mount the missing key; the reconciler replaces it now."""

    core, evidence = _installer_world({NEW: "Failed", SIBLING: "Installing"})

    added = _sync(AttestingClient(), core, evidence).run()

    assert added == [NEW, SIBLING]
    assert core.node_patches == [
        (NEW, {"metadata": {"resourceVersion": "7", "annotations": REPLAY}})
    ], "a Job still running mounts the key by itself; only the failed one is replayed"
    assert core.nodes[NEW]["metadata"]["annotations"] == REPLAY


def test_a_node_the_reconciler_moved_on_meanwhile_is_not_replayed() -> None:
    """A Retrying written over Installing would replay an install about to succeed."""

    core, evidence = _installer_world({NEW: "Failed"})

    def reconciler_retried(cluster: VersionedCore) -> None:
        metadata = cluster.nodes[NEW]["metadata"]
        metadata["annotations"] = {INSTALLER_STATE_ANNOTATION: "Installing"}
        metadata["resourceVersion"] = "8"

    core.on_secret_patch = reconciler_retried

    assert _sync(AttestingClient(), core, evidence).run() == [NEW]

    assert [name for name, _body in core.node_patches] == [NEW], (
        "one fenced attempt, refused; the re-read shows the node moved on"
    )
    assert core.nodes[NEW]["metadata"]["annotations"] == {
        INSTALLER_STATE_ANNOTATION: "Installing"
    }


def test_an_unrelated_node_update_is_re_read_and_the_replay_still_lands() -> None:
    core, evidence = _installer_world({NEW: "Failed"})

    def status_update(cluster: VersionedCore) -> None:
        cluster.nodes[NEW]["metadata"]["resourceVersion"] = "9"

    core.on_secret_patch = status_update

    assert _sync(AttestingClient(), core, evidence).run() == [NEW]

    assert [body["metadata"]["resourceVersion"] for _n, body in core.node_patches] == [
        "7",
        "9",
    ]
    assert core.nodes[NEW]["metadata"]["annotations"] == REPLAY


def test_a_node_refused_its_key_is_never_replayed() -> None:
    core, evidence = _installer_world({NEW: "Failed"})
    conflict = ClusterExecutorError(
        "regional control plane rejected request (409): CPU NodeName key conflict",
        status_code=409,
    )

    with pytest.raises(ClusterExecutorError, match="key conflict"):
        _sync(AttestingClient({NEW: conflict}), core, evidence).run()

    assert core.node_patches == []
