"""PostgreSQL connection budget of the CPU control-plane roles.

Pooled connections per process plus every unpooled listener a role installs,
against the server's ``max_connections``; read from the live Deployments and
ConfigMaps by the capacity harness (CAP-003).
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Protocol

from gpu_fault.postgres_capacity import PostgresPoolCapacity
from scripts.e2e.regional.capacity_acceptance_common import CapError
from scripts.e2e.regional.regional_live_fixture import component_python


class BudgetHost(Protocol):
    def kubectl_json(self, *args: str) -> Any: ...

    def kubectl(
        self, *args: str, input_text: str | None = None, **options: Any
    ) -> subprocess.CompletedProcess[str]: ...


def unpooled_connection_budget(deployment: str) -> dict[str, int]:
    """Upper bound for installed listeners, including lazy claim wakeups."""
    roles = {
        "gpu-fault-api-ha": "ingress",
        "gpu-fault-control-worker": "worker",
        "gpu-fault-telemetry-spool-worker": "spool-worker",
    }
    if deployment not in roles:
        raise CapError("unrecognized CPU role in connection budget")
    return PostgresPoolCapacity.listener_connections(
        roles[deployment],
        queued_processor=True,
        spool_enabled=True,
        workflow_dispatcher_enabled=True,
        regional=True,
    )


def connection_budget(host: BudgetHost) -> dict[str, Any]:
    deployments = host.kubectl_json("get", "deployments", "-o", "json")
    configmaps = host.kubectl_json(
        "get",
        "configmap",
        "gpu-fault-api-ha-config-postgres",
        "gpu-fault-control-worker-config-postgres",
        "gpu-fault-telemetry-spool-worker-config-postgres",
        "-o",
        "json",
    )
    pools = {
        item["metadata"]["name"]: int(
            item.get("data", {}).get("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "0")
        )
        for item in configmaps["items"]
    }
    rows = []
    total = 0
    config_by_deployment = {
        "gpu-fault-api-ha": "gpu-fault-api-ha-config-postgres",
        "gpu-fault-control-worker": "gpu-fault-control-worker-config-postgres",
        "gpu-fault-telemetry-spool-worker": (
            "gpu-fault-telemetry-spool-worker-config-postgres"
        ),
    }
    if set(pools) != set(config_by_deployment.values()) or any(
        pool < 1 for pool in pools.values()
    ):
        raise CapError("PostgreSQL pool configuration is incomplete")
    for item in deployments["items"]:
        name = item["metadata"]["name"]
        if name not in config_by_deployment:
            continue
        replicas = item["spec"].get("replicas")
        if type(replicas) is not int or replicas < 0:
            raise CapError("capacity role replica count is missing or invalid")
        container = item["spec"]["template"]["spec"]["containers"][0]
        args = " ".join(container["args"])
        match = re.search(r"--workers\s+(\d+)", args)
        processes = int(match.group(1)) if match else 1
        pool = pools[config_by_deployment[name]]
        overrides = [
            entry
            for entry in container.get("env", [])
            if entry.get("name") == "GPU_FAULT_POSTGRES_POOL_MAX_SIZE"
        ]
        if overrides:
            if len(overrides) != 1 or not str(overrides[0].get("value", "")).isdigit():
                raise CapError("effective PostgreSQL pool size is unknown")
            pool = int(overrides[0]["value"])
        if pool < 1 or processes < 1:
            raise CapError("capacity process or pool size is invalid")
        listeners = unpooled_connection_budget(name)
        unpooled = sum(listeners.values())
        maximum = replicas * processes * (pool + unpooled)
        rows.append(
            {
                "deployment": name,
                "replicas": replicas,
                "processes_per_pod": processes,
                "pool_max_size": pool,
                "unpooled_per_process": unpooled,
                "unpooled_by_consumer": listeners,
                "pooled_connections": replicas * processes * pool,
                "unpooled_connections": replicas * processes * unpooled,
                "theoretical_connections": maximum,
            }
        )
        total += maximum
    if len(rows) != len(config_by_deployment) or {
        row["deployment"] for row in rows
    } != set(config_by_deployment):
        raise CapError("capacity connection budget is missing a CPU role")
    worker = host.kubectl_json(
        "get", "pods", "-l", "app=gpu-fault-control-worker", "-o", "json"
    )
    # Exclude legacy capacity probes that used the production app label;
    # only a production worker has the production database credential.
    pod = next(
        item["metadata"]["name"]
        for item in worker["items"]
        if item.get("status", {}).get("phase") == "Running"
        and not item.get("metadata", {}).get("deletionTimestamp")
        and not item.get("metadata", {})
        .get("labels", {})
        .get("gpu-fault.io/capacity-probe")
    )
    script = r"""
import os,psycopg
from pathlib import Path
path = os.environ.get("GPU_FAULT_STORE_URL_FILE", "").strip()
url = Path(path).read_text().strip() if path else os.environ["GPU_FAULT_STORE_URL"]
if not url:
    raise RuntimeError("database credential reference is empty")
with psycopg.connect(url, connect_timeout=10) as c:
    with c.cursor() as cur:
        cur.execute("select current_setting('max_connections')::int")
        print(cur.fetchone()[0])
"""
    max_connections = int(
        host.kubectl(
            "exec",
            "-i",
            pod,
            "--",
            component_python("cpu"),
            "-",
            input_text=script,
        ).stdout.strip()
    )
    if max_connections <= 0:
        raise CapError("PostgreSQL max_connections is invalid")
    return {
        "roles": rows,
        "theoretical_total": total,
        "max_connections": max_connections,
        "budget_ratio": total / max_connections,
        "budget_scope": (
            "all installed role listeners, including lazy claim listeners; "
            "disabled features may use fewer connections"
        ),
        "excluded_consumers": [
            "transient schema/administrator/probe connections",
            "other applications on the same Aurora server",
            "temporary surge Pods during a rollout",
        ],
    }
