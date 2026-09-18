"""Authenticated fleet-read isolation, using each GPU Pod's actual credential."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote, urlencode

from scripts.e2e.regional.identity_acceptance_common import (
    ClusterTarget,
    IdentityAcceptanceError,
    IdentitySite,
)

FLEET_BASELINE_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
store = ApplicationContext.from_environment().store
print(json.dumps({
    "clusters": {
        cluster: sorted(agent.node_id for agent in store.list_agents(cluster))
        for cluster in sys.argv[1:]
    }
}))
"""

FLEET_SCOPE_PROBE = r"""
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
requests = json.loads(sys.argv[1])
context = ssl.create_default_context(
    cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
)
rows = {}
for name, path in requests.items():
    request = urllib.request.Request(
        os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/") + path,
        method="GET",
        headers={
            "Authorization": "Bearer " + os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
            "X-GPU-Fault-Cluster-ID": os.environ["GPU_FAULT_CLUSTER_ID"],
        },
    )
    try:
        with urllib.request.urlopen(request, context=context, timeout=20) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        with error:
            status, raw = error.code, error.read()
    body = json.loads(raw)
    if status == 200:
        items = body if isinstance(body, list) else [body]
        rows[name] = {
            "status": status,
            "agents": [
                {"cluster_id": item.get("cluster_id"), "node_id": item.get("node_id")}
                for item in items
            ],
        }
    else:
        rows[name] = {"status": status, "detail": body.get("detail")}
print(json.dumps({"cluster_id": os.environ["GPU_FAULT_CLUSTER_ID"], "results": rows}))
"""


def fleet_scope_requests(
    cluster_id: str, baseline: dict[str, list[str]]
) -> dict[str, str]:
    if (
        len(baseline) < 2
        or cluster_id not in baseline
        or any(
            not isinstance(nodes, list)
            or not nodes
            or any(not isinstance(node, str) or not node for node in nodes)
            or len(set(nodes)) != len(nodes)
            for nodes in baseline.values()
        )
    ):
        raise IdentityAcceptanceError(
            "fleet isolation needs populated local and peer baselines"
        )
    result = {
        "local-list": "/v1/fleet/agents",
        "local-query": "/v1/fleet/agents?" + urlencode({"cluster_id": cluster_id}),
        "local-node": (
            "/v1/fleet/agents/"
            + quote(cluster_id, safe="")
            + "/"
            + quote(baseline[cluster_id][0], safe="")
        ),
    }
    for index, peer in enumerate(sorted(set(baseline) - {cluster_id})):
        result[f"peer-{index}-query"] = "/v1/fleet/agents?" + urlencode(
            {"cluster_id": peer}
        )
        result[f"peer-{index}-node"] = (
            "/v1/fleet/agents/"
            + quote(peer, safe="")
            + "/"
            + quote(baseline[peer][0], safe="")
        )
    return result


def fleet_scope_errors(
    document: dict[str, Any], *, cluster_id: str, baseline: dict[str, list[str]]
) -> list[str]:
    expected = fleet_scope_requests(cluster_id, baseline)
    rows = document.get("results")
    if (
        document.get("cluster_id") != cluster_id
        or not isinstance(rows, dict)
        or set(rows) != set(expected)
    ):
        return ["authenticated fleet request inventory is incomplete or foreign"]
    errors = []
    for name, row in rows.items():
        if not isinstance(row, dict):
            errors.append(f"{name}: malformed response")
            continue
        if name.startswith("peer-"):
            detail = (
                "authenticated cluster cannot read agents for another cluster"
                if name.endswith("-query")
                else "authenticated cluster cannot read an agent from another cluster"
            )
            if (
                type(row.get("status")) is not int
                or row["status"] != 403
                or row.get("detail") != detail
            ):
                errors.append(f"{name}: wrong authenticated scope denial")
            continue
        expected_nodes = (
            [baseline[cluster_id][0]] if name == "local-node" else baseline[cluster_id]
        )
        agents = row.get("agents")
        if (
            type(row.get("status")) is not int
            or row["status"] != 200
            or not isinstance(agents, list)
            or len(agents) != len(expected_nodes)
            or any(
                not isinstance(agent, dict) or agent.get("cluster_id") != cluster_id
                for agent in agents
            )
            or sorted(agent.get("node_id", "") for agent in agents)
            != sorted(expected_nodes)
        ):
            errors.append(
                f"{name}: local fleet listing is missing, incomplete or foreign"
            )
    return errors


def authenticated_fleet_isolation(
    site: IdentitySite, target: ClusterTarget
) -> dict[str, Any]:
    clusters = sorted(site.targets)
    baseline = site.api_pod_json(FLEET_BASELINE_PROBE, *clusters).get("clusters")
    if not isinstance(baseline, dict) or set(baseline) != set(clusters):
        raise IdentityAcceptanceError("fleet isolation baseline is incomplete")
    requests = fleet_scope_requests(target.cluster_id, baseline)
    observed = site.pod_json(
        "gpu",
        target,
        site.any_executor_pod(target),
        FLEET_SCOPE_PROBE,
        json.dumps(requests),
        timeout=300,
    )
    after = site.api_pod_json(FLEET_BASELINE_PROBE, *clusters).get("clusters")
    errors = fleet_scope_errors(
        observed, cluster_id=target.cluster_id, baseline=baseline
    )
    if after != baseline:
        errors.append("fleet membership changed during the authenticated read window")
    return {"results": observed, "errors": errors, "passed": not errors}
