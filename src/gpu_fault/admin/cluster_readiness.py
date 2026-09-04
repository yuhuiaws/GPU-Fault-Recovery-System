from __future__ import annotations

import json
import subprocess
import time
from typing import Any, cast

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite

COLLECTOR_READINESS_SCRIPT = r"""
import json
import os
import urllib.parse
import urllib.request

cluster_id = urllib.parse.quote(os.environ["CLUSTER_ID"], safe="")
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/collector-readiness/" + cluster_id,
    headers={
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
with urllib.request.urlopen(request, timeout=15) as response:
    print(json.dumps(json.load(response), separators=(",", ":")))
"""


def _collector_readiness_report(
    site: RenderedSite,
    cluster_id: str,
) -> dict[str, Any]:
    kubectl = [
        "kubectl",
        "--kubeconfig",
        str(site.release_config["cpu_kubeconfig"]),
        "-n",
        str(site.release_config["namespace"]),
    ]
    pod = subprocess.run(
        [
            *kubectl,
            "get",
            "pod",
            "-l",
            "app=gpu-fault-api-ha",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        text=True,
        capture_output=True,
    )
    if pod.returncode or not pod.stdout.strip():
        raise BootstrapError("cannot find a Running CPU ingress Pod")
    result = subprocess.run(
        [
            *kubectl,
            "exec",
            pod.stdout.strip(),
            "--",
            "env",
            f"CLUSTER_ID={cluster_id}",
            "python",
            "-c",
            COLLECTOR_READINESS_SCRIPT,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise BootstrapError(
            "collector readiness query failed: " + result.stderr.strip()
        )
    return cast(dict[str, Any], json.loads(result.stdout))


def wait_collector_readiness(
    site: RenderedSite,
    cluster_id: str,
    *,
    timeout_seconds: float = 600,
    interval_seconds: float = 10,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            last = _collector_readiness_report(site, cluster_id)
            last_error = None
            if last.get("ready") is True:
                return last
        except (BootstrapError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(interval_seconds)
    detail = last_error or json.dumps(last or {}, sort_keys=True)
    raise BootstrapError(
        f"{cluster_id} collectors did not become ready within "
        f"{timeout_seconds:g}s: {detail}"
    )
