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

* only keys for nodes that carry this cluster's HyperPod label are requested,
  the same selection the deploy host's provisioning helper uses;
* only *missing* keys are written, in one merge patch fenced by the Secret's
  ``resourceVersion``, so an existing (possibly rotated) key is never
  overwritten and a concurrent writer makes the patch fail instead of losing;
* nothing is deleted and the Secret is never created here;
* any refusal or malformed answer from the control plane means no patch;
* logs carry node names and counts, never key material.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Callable

from gpu_fault.cluster_executor.regional_client import (
    ClusterExecutorError,
    RegionalExecutorClient,
)

# The pre-split module's name: operators filter on ``gpu_fault.cluster_executor``.
LOGGER = logging.getLogger("gpu_fault.cluster_executor")

#: How often the executor compares the labelled nodes with the Secret's keys.
#: One node LIST and one Secret GET per cycle; a replacement node is keyed
#: within about this long of joining the cluster.
NODE_KEY_SYNC_INTERVAL_SECONDS = 60.0
NODE_ACTION_KEYS_SECRET_NAME = "gpu-fault-node-action-keys"
#: The label the deploy host's provisioning helper selects nodes by.
HYPERPOD_CLUSTER_LABEL = "sagemaker.amazonaws.com/cluster-name"
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


class NodeActionKeySync:
    """One reconciliation of the node-key Secret against the labelled nodes."""

    def __init__(
        self,
        client: RegionalExecutorClient,
        core: Any,
        *,
        hyperpod_cluster: str,
        namespace: str,
        secret_name: str = NODE_ACTION_KEYS_SECRET_NAME,
        interval_seconds: float = NODE_KEY_SYNC_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not hyperpod_cluster.strip():
            raise ValueError("hyperpod_cluster must not be empty")
        if not namespace.strip():
            raise ValueError("namespace must not be empty")
        if not secret_name.strip():
            raise ValueError("secret_name must not be empty")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.client = client
        self.core = core
        self.hyperpod_cluster = hyperpod_cluster.strip()
        self.namespace = namespace.strip()
        self.secret_name = secret_name.strip()
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.added_total = 0
        self.failures_total = 0
        self._next_due: float | None = None

    @property
    def node_selector(self) -> str:
        return f"{HYPERPOD_CLUSTER_LABEL}={self.hyperpod_cluster}"

    def due(self) -> bool:
        return self._next_due is None or self.clock() >= self._next_due

    def labelled_nodes(self) -> list[str]:
        """Names of the Kubernetes nodes carrying this cluster's HyperPod label."""

        response = self.core.list_node(label_selector=self.node_selector)
        names: set[str] = set()
        for item in _value(response, "items", []) or []:
            name = _value(_value(item, "metadata", {}) or {}, "name")
            if name:
                names.add(str(name))
        return sorted(names)

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

        Raises on any refusal, transport failure or malformed answer -- with
        nothing written, because the patch is built only after every requested
        key has been validated. The executor's wrapper turns that into a log
        line and the next cycle tries again.
        """

        self._next_due = self.clock() + self.interval_seconds
        try:
            return self._run()
        except Exception:
            self.failures_total += 1
            raise

    def _run(self) -> list[str]:
        nodes = self.labelled_nodes()
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
        for start in range(0, len(missing), NODE_KEY_REQUEST_BATCH):
            chunk = missing[start : start + NODE_KEY_REQUEST_BATCH]
            keys = self.client.node_action_keys(chunk)
            if set(keys) != set(chunk):
                raise ClusterExecutorError(
                    "control plane answered a different node set than requested"
                )
            for node_name, value in keys.items():
                if len(value) < MIN_NODE_KEY_LENGTH or value != value.strip():
                    raise ClusterExecutorError(
                        f"node action key for {node_name} is malformed"
                    )
                encoded[node_name] = base64.b64encode(value.encode("utf-8")).decode(
                    "ascii"
                )
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
