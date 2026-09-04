"""Read the whole control API surface `status` reports on, in one exec.

Request: two environment variables, because this probe asks a question per
cluster and the answer has to name the clusters it asked about --
``ADMIN_CLUSTER_IDS_JSON`` (a JSON list of cluster ids) and
``ADMIN_EXPECTED_NODES_JSON`` (a JSON object of cluster id to node id list).
Response: one JSON object on stdout with ``healthz``, ``version``, ``registry``,
``remote_commands`` and a ``clusters`` entry per cluster id.

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


def main() -> None:
    clusters = json.loads(os.environ["ADMIN_CLUSTER_IDS_JSON"])
    expected_nodes = json.loads(os.environ["ADMIN_EXPECTED_NODES_JSON"])
    result: dict[str, Any] = {
        "healthz": get("/healthz"),
        "version": get("/v1/version"),
        "registry": get("/v1/regional/clusters"),
        "clusters": {},
        "remote_commands": (
            ApplicationContext.from_environment().store.remote_command_stats()
        ),
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
        }
    print(json.dumps(result, separators=(",", ":")))


main()
