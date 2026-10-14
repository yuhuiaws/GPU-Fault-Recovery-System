"""Keep the GPU node-key Secret complete for the nodes this cluster has.

``gpu-fault-admin deploy``/``join-cluster`` write one node-scoped Node Action
key per node into ``gpu-fault-node-action-keys`` for the nodes present at that
time. A node HyperPod replaces later (spot reclaim, hardware failure) arrives
with no key: every installer Job for it fails on the missing Secret key after
the full 840 s deadline, forever, and the node never gets an Agent (live
2026-09-30). The reconciler deliberately holds no ``secrets`` grant, so the
executor -- which already has the cluster token and the node-key Secret
mounted -- asks the control plane for the missing names and patches only
those keys in.

Invariants, each of them a test:

* only keys for nodes the cluster's node selector matches are requested
  (``GPU_FAULT_NODE_SELECTOR``; for HyperPod the cluster-name label), the
  same selection the deploy host's provisioning helper uses, checked the way
  it checks it: valid Node names, one UID each, no repeats, every node
  matching the selector -- or nothing is requested at all;
* only *missing* keys are written, in one merge patch fenced by the Secret's
  ``resourceVersion``, so an existing (possibly rotated) key is never
  overwritten and a concurrent writer makes the patch fail instead of losing;
* nothing is deleted and the Secret is never created here;
* a refusal keys nothing for the nodes it covers, and a malformed answer or a
  request without a verdict means no patch at all;
* logs carry node names and counts, never key material.

A plain EKS node that was never installed is unknown to the control plane
(404): nothing there places it before its first install. On an
``eks-managed`` cluster such a refusal is followed by one request per node
carrying the EC2 identity ``attester`` read for it
(``gpu_fault/eks_node_evidence.py``); a node it cannot attest is not
asked for, and one refused node does not hold back the others.

A node whose installer Job already failed while its key was missing is then
handed back to the reconciler (``installer-state=Retrying`` with
``installer-reason=NodeActionKeyAdded``), which replaces the failed Job on its
next pass instead of after a backoff of up to an hour. A Job still running
needs nothing -- the kubelet mounts the key once it exists.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Callable, Protocol, Sequence

from gpu_fault.cluster_executor.regional_client import (
    ClusterExecutorError,
    RegionalExecutorClient,
)
from gpu_fault.cluster_scope import node_selector as scope_node_selector
from gpu_fault.cluster_scope import selector_matches
from gpu_fault.node_installer_reconciler import (
    INSTALLER_REASON_ANNOTATION,
    INSTALLER_STATE_ANNOTATION,
    NODE_KEY_ADDED_REASON,
)
from gpu_fault.node_key_attestation import NODE_NAME_PATTERN, EksNodeAttestation
from gpu_fault.node_provider import HYPERPOD_CLUSTER_LABEL as HYPERPOD_CLUSTER_LABEL

# The pre-split module's name: operators filter on ``gpu_fault.cluster_executor``.
LOGGER = logging.getLogger("gpu_fault.cluster_executor")

#: How often the executor compares the labelled nodes with the Secret's keys.
#: One node LIST and one Secret GET per cycle; a replacement node is keyed
#: within about this long of joining the cluster.
NODE_KEY_SYNC_INTERVAL_SECONDS = 60.0
NODE_ACTION_KEYS_SECRET_NAME = "gpu-fault-node-action-keys"
#: The control plane bounds one request to this many node ids.
NODE_KEY_REQUEST_BATCH = 64
#: ``derive_node_action_secret`` returns 64 hex characters; anything shorter is
#: not a key this fleet would sign with.
MIN_NODE_KEY_LENGTH = 32


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _api_status(error: Exception) -> int | None:
    status = getattr(error, "status", None)
    return status if isinstance(status, int) else None


class NodeKeyAttester(Protocol):
    """What the sync needs to attest nodes the control plane does not know."""

    def attest(
        self, nodes: Sequence[Any]
    ) -> tuple[dict[str, EksNodeAttestation], dict[str, str]]: ...


class NodeActionKeySync:
    """One reconciliation of the node-key Secret against the labelled nodes."""

    def __init__(
        self,
        client: RegionalExecutorClient,
        core: Any,
        *,
        hyperpod_cluster: str = "",
        namespace: str,
        node_selector: str | None = None,
        secret_name: str = NODE_ACTION_KEYS_SECRET_NAME,
        interval_seconds: float = NODE_KEY_SYNC_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        attester: NodeKeyAttester | None = None,
    ) -> None:
        # Either an explicit selector (any provider) or the HyperPod cluster
        # name, which implies the cluster-name label selector.
        selector = (node_selector or "").strip()
        if not selector and not hyperpod_cluster.strip():
            raise ValueError("node_selector or hyperpod_cluster must be given")
        if not namespace.strip():
            raise ValueError("namespace must not be empty")
        if not secret_name.strip():
            raise ValueError("secret_name must not be empty")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.client = client
        self.core = core
        self.hyperpod_cluster = hyperpod_cluster.strip()
        # The one selector reading every component shares
        # (``gpu_fault.cluster_scope.node_selector``): the explicit selector, or
        # the legacy HyperPod cluster-name label.
        self._node_selector = scope_node_selector(
            None,
            provider="hyperpod",
            hyperpod_cluster_name=self.hyperpod_cluster,
            selector=selector,
        )
        self.namespace = namespace.strip()
        self.secret_name = secret_name.strip()
        self.interval_seconds = interval_seconds
        self.clock = clock
        # ``eks-managed`` only: states a node's EC2 identity when the control
        # plane has no evidence of it; None everywhere else.
        self.attester = attester
        self.added_total = 0
        self.failures_total = 0
        self._next_due: float | None = None

    @property
    def node_selector(self) -> str:
        return self._node_selector

    def due(self) -> bool:
        return self._next_due is None or self.clock() >= self._next_due

    def labelled_nodes(self) -> list[str]:
        """Names of the Kubernetes nodes this cluster's node selector matches."""

        return list(self._inventory())

    def _inventory(self) -> dict[str, Any]:
        """The selected nodes by name, checked the way the deploy host checks them.

        A name that is not a Kubernetes Node name, a missing UID, a repeated
        name or UID, or a node the selector does not match means the list
        cannot be trusted and nothing is requested from it (the provisioning
        helper's "node identity is invalid or ambiguous"). A node being
        deleted is left out: it is leaving.
        """

        response = self.core.list_node(label_selector=self.node_selector)
        nodes: dict[str, Any] = {}
        names: set[str] = set()
        uids: set[str] = set()
        for item in _value(response, "items", []) or []:
            metadata = _value(item, "metadata", {}) or {}
            name = str(_value(metadata, "name", "") or "")
            uid = str(_value(metadata, "uid", "") or "")
            labels = _value(metadata, "labels", {}) or {}
            if (
                NODE_NAME_PATTERN.fullmatch(name) is None
                or not uid
                or name in names
                or uid in uids
                or not selector_matches(dict(labels), self.node_selector)
            ):
                raise ClusterExecutorError(
                    "the selected node inventory is invalid or ambiguous; no node "
                    "action key is requested from it"
                )
            names.add(name)
            uids.add(uid)
            if not (
                _value(metadata, "deletion_timestamp")
                or _value(metadata, "deletionTimestamp")
            ):
                nodes[name] = item
        return dict(sorted(nodes.items()))

    def _read_secret(self) -> tuple[str, frozenset[str]] | None:
        """The Secret's resourceVersion and key *names*; None when it is absent.

        Only the names are read. The values are the fleet's per-node signing
        keys and nothing in this process needs them.
        """

        try:
            secret = self.core.read_namespaced_secret(self.secret_name, self.namespace)
        except Exception as exc:
            if _api_status(exc) == 404:
                return None
            raise
        metadata = _value(secret, "metadata", {}) or {}
        version = _value(metadata, "resource_version") or _value(
            metadata, "resourceVersion"
        )
        if not version:
            raise ClusterExecutorError(
                f"Secret {self.namespace}/{self.secret_name} has no resourceVersion"
            )
        data = _value(secret, "data", None) or {}
        return str(version), frozenset(str(key) for key in data)

    def run(self) -> list[str]:
        """Add the keys the labelled nodes lack; returns the node names added.

        Raises on any refusal, transport failure or malformed answer. A
        transport failure or a malformed answer writes nothing, because the
        patch is built only after every requested key has been validated; a
        refusal of some attested nodes still patches the keys granted for the
        others before it raises. The executor's wrapper turns that into a log
        line and the next cycle tries again.
        """

        self._next_due = self.clock() + self.interval_seconds
        try:
            return self._run()
        except Exception:
            self.failures_total += 1
            raise

    def _run(self) -> list[str]:
        nodes = self._inventory()
        current = self._read_secret()
        if current is None:
            # The deploy host creates the Secret with the deploy-time node set
            # and owns its identity (UID) checks; creating it here would give
            # the provisioning helper a Secret it never wrote.
            raise ClusterExecutorError(
                f"Secret {self.namespace}/{self.secret_name} is absent; the "
                "deploy host creates it, this sync only adds missing node keys"
            )
        resource_version, existing = current
        missing = [name for name in nodes if name not in existing]
        if not missing:
            return []
        encoded: dict[str, str] = {}
        refused: dict[str, str] = {}
        for start in range(0, len(missing), NODE_KEY_REQUEST_BATCH):
            chunk = missing[start : start + NODE_KEY_REQUEST_BATCH]
            keys = self._request(chunk, nodes, refused)
            for node_name, value in keys.items():
                if len(value) < MIN_NODE_KEY_LENGTH or value != value.strip():
                    raise ClusterExecutorError(
                        f"node action key for {node_name} is malformed"
                    )
                encoded[node_name] = base64.b64encode(value.encode("utf-8")).decode(
                    "ascii"
                )
        added = self._patch(resource_version, encoded)
        for name in added:
            self._request_replay(nodes[name])
        if refused:
            raise ClusterExecutorError(
                f"no node action key for {len(refused)} node(s): "
                + "; ".join(f"{name}: {reason}" for name, reason in refused.items())
            )
        return added

    def _request(
        self, chunk: list[str], nodes: dict[str, Any], refused: dict[str, str]
    ) -> dict[str, str]:
        """The keys the control plane grants for ``chunk``; refusals go to ``refused``.

        The plain request first, exactly what every control plane answers. A
        404 on an ``eks-managed`` cluster -- a node nothing there knows yet --
        is followed by one request per node: with the EC2 identity the
        attester states for it, or plain when it states none (a node the
        control plane knows by other evidence is still keyed). Every other
        refusal of the first request, and any request without a verdict,
        raises with nothing written.
        """

        try:
            return self._checked(chunk, self.client.node_action_keys(chunk))
        except ClusterExecutorError as exc:
            if exc.status_code != 404 or self.attester is None:
                raise
        attestations, reasons = self.attester.attest([nodes[name] for name in chunk])
        keys: dict[str, str] = {}
        for name in chunk:
            attestation = attestations.get(name)
            try:
                answer = self.client.node_action_keys(
                    [name], attestations=[attestation] if attestation else None
                )
            except ClusterExecutorError as exc:
                if exc.status_code is None:
                    raise
                # The attester's reason says more than the control plane's 404.
                refused[name] = reasons.get(name) or str(exc)
                continue
            keys.update(self._checked([name], answer))
        return keys

    @staticmethod
    def _checked(requested: list[str], keys: dict[str, str]) -> dict[str, str]:
        if set(keys) != set(requested):
            raise ClusterExecutorError(
                "control plane answered a different node set than requested"
            )
        return keys

    def _request_replay(self, node: Any) -> None:
        """Ask the reconciler to replace a just-keyed node's failed installer Job.

        Only a node whose state still reads ``Failed`` -- its Job failed while
        the key was missing, so it never ran -- and only through a patch fenced
        by the Node's ``resourceVersion``: the reconciler may have moved the
        node on since it was listed, and a ``Retrying`` written over an
        ``Installing`` would replay an install that is about to succeed. One
        re-read after a conflict; anything else is a WARNING and the normal
        backoff, never a failed sync.
        """

        metadata = _value(node, "metadata", {}) or {}
        name = str(_value(metadata, "name", "") or "")
        for _attempt in range(2):
            annotations = _value(metadata, "annotations", {}) or {}
            version = _value(metadata, "resource_version") or _value(
                metadata, "resourceVersion"
            )
            if annotations.get(INSTALLER_STATE_ANNOTATION) != "Failed" or not version:
                return
            try:
                self.core.patch_node(
                    name,
                    {
                        "metadata": {
                            "resourceVersion": str(version),
                            "annotations": {
                                INSTALLER_STATE_ANNOTATION: "Retrying",
                                INSTALLER_REASON_ANNOTATION: NODE_KEY_ADDED_REASON,
                            },
                        }
                    },
                )
            except Exception as exc:
                if _api_status(exc) != 409:
                    LOGGER.warning(
                        "could not ask for a new installer Job for node %s: %s",
                        name,
                        type(exc).__name__,
                    )
                    return
                try:
                    metadata = _value(self.core.read_node(name), "metadata", {}) or {}
                except Exception as read_error:
                    LOGGER.warning(
                        "could not re-read node %s: %s",
                        name,
                        type(read_error).__name__,
                    )
                    return
                continue
            LOGGER.warning(
                "node %s has its action key now; asked the reconciler to replace "
                "its failed installer Job",
                name,
            )
            return

    def _patch(self, resource_version: str, encoded: dict[str, str]) -> list[str]:
        if not encoded:
            return []
        # A JSON merge patch adds exactly these keys and touches no other; the
        # resourceVersion precondition makes the apiserver refuse it (409) if
        # anything -- the deploy host, the sibling replica -- wrote the Secret
        # since it was read, so a concurrent rotation is never overwritten.
        self.core.patch_namespaced_secret(
            self.secret_name,
            self.namespace,
            {"metadata": {"resourceVersion": resource_version}, "data": encoded},
        )
        added = sorted(encoded)
        self.added_total += len(added)
        LOGGER.warning(
            "added %d node action key(s) to Secret %s/%s for replacement node(s) %s",
            len(added),
            self.namespace,
            self.secret_name,
            ",".join(added),
        )
        return added
