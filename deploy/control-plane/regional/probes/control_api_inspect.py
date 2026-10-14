"""Read the whole control API surface `status` reports on, in one exec.

Request: two environment variables, because this probe asks a question per
cluster and the answer has to name the clusters it asked about --
``ADMIN_CLUSTER_IDS_JSON`` (a JSON list of cluster ids) and
``ADMIN_EXPECTED_NODES_JSON`` (a JSON object of cluster id to node id list).
Response: one JSON object on stdout with ``healthz``, ``version``, ``registry``,
``remote_commands``, ``collector_thresholds`` (the silence threshold in seconds
the control plane applies to each collector kind), ``departed_node_cleanup_
enabled`` and a ``clusters`` entry per cluster id, which carries the cluster's
departed-node ledger (``departed_node_cleanup``: one line per departed node,
CLEANED / PENDING / REFUSED with its blockers and the manual commands; None
from a control plane older than the cleanup).

Six endpoints in one exec rather than six execs: every call is a loopback
request inside the Pod, so the expensive part is the ``kubectl exec`` around
them, and a status that answered each question in its own exec would describe
the fleet at six different moments.

The ``ADMIN_`` prefix on those names is deliberate, and pinned by a test: the
Pod's own environment carries the execution token under the runtime's prefix, so
an input injected under the same prefix would be indistinguishable from a
variable the runtime is entitled to read.
"""

import json
import os
import urllib.request
from typing import Any

from gpu_fault.app import ApplicationContext


def get(path: str) -> Any:
    request = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        headers={
            "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def post(path: str, body: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def collector_thresholds() -> dict[str, float]:
    try:
        from gpu_fault.collector_kinds import collector_silent_thresholds
    except ImportError:
        # A control plane older than the engine, as a rollback's probe may find.
        return {}
    return {
        kind.value: seconds for kind, seconds in collector_silent_thresholds().items()
    }


# The ledger fields ``status`` reads; the evidence stays in the store.
DEPARTED_NODE_FIELDS = {
    "node_id",
    "provider",
    "status",
    "reference",
    "instance_id",
    "absent_since",
    "blockers",
    "blocker_classes",
    "admin_commands",
    "actions",
    "first_seen_at",
    "updated_at",
    "cleaned_at",
}


def departed_node_cleanup(store: Any, cluster_id: str) -> list[Any] | None:
    reader = getattr(store, "list_departed_node_cleanups", None)
    if reader is None:
        # A control plane older than the engine, as a rollback's probe may find.
        return None
    return [
        record.model_dump(mode="json", include=DEPARTED_NODE_FIELDS)
        for record in reader(cluster_id)
    ]


def departed_node_cleanup_enabled() -> bool | None:
    try:
        from gpu_fault.departed_node_cleanup import cleanup_enabled
    except ImportError:
        return None
    return cleanup_enabled()


def main() -> None:
    clusters = json.loads(os.environ["ADMIN_CLUSTER_IDS_JSON"])
    expected_nodes = json.loads(os.environ["ADMIN_EXPECTED_NODES_JSON"])
    store = ApplicationContext.from_environment().store
    result: dict[str, Any] = {
        "healthz": get("/healthz"),
        "version": get("/v1/version"),
        "registry": get("/v1/regional/clusters"),
        "collector_thresholds": collector_thresholds(),
        "clusters": {},
        "remote_commands": store.remote_command_stats(),
        "departed_node_cleanup_enabled": departed_node_cleanup_enabled(),
    }
    for cluster_id in clusters:
        agents = get("/v1/fleet/agents?cluster_id=" + cluster_id)
        node_ids = sorted(expected_nodes[cluster_id])
        result["clusters"][cluster_id] = {
            "expected_node_ids": node_ids,
            "agents": agents,
            # An empty node list is not an empty readiness question, it is no
            # question at all: posting it would ask the control plane whether zero
            # nodes are ready and get back `True`.
            "fleet_readiness": (
                post(
                    "/v1/fleet/readiness",
                    {"cluster_id": cluster_id, "node_ids": node_ids},
                )
                if node_ids
                else None
            ),
            "collector_readiness": get("/v1/collector-readiness/" + cluster_id),
            "departed_node_cleanup": departed_node_cleanup(store, cluster_id),
        }
    print(json.dumps(result, separators=(",", ":")))


main()
