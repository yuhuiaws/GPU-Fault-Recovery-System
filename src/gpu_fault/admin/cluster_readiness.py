from __future__ import annotations

import json
import math
import subprocess
import time
from typing import Any, cast

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.execution import (
    deadline_scope,
    remaining_timeout,
    run_command,
)
from gpu_fault.admin.site import RenderedSite, effective_environment
from gpu_fault_release.regional_deployment_inventory import CPU_INGRESS_DEPLOYMENT

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
    pod = run_command(
        [
            *kubectl,
            "get",
            "pod",
            "-l",
            f"app={CPU_INGRESS_DEPLOYMENT}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        environment=effective_environment(site),
        timeout_seconds=30,
    )
    if pod.returncode or not pod.stdout.strip():
        raise BootstrapError("cannot find a Running CPU ingress Pod")
    result = run_command(
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
        environment=effective_environment(site),
        timeout_seconds=30,
    )
    if result.returncode:
        raise BootstrapError(
            "collector readiness query failed: "
            + diagnostic_text(result.stderr.strip())
        )
    report = json.loads(result.stdout)
    if not isinstance(report, dict) or report.get("cluster_id") != cluster_id:
        raise BootstrapError("collector readiness response cluster identity mismatch")
    if not isinstance(report.get("ready"), bool) or not isinstance(
        report.get("nodes"), list
    ):
        raise BootstrapError("collector readiness response is malformed")
    if report["ready"] and (
        not report["nodes"]
        or any(
            not isinstance(node, dict) or node.get("ready") is not True
            for node in report["nodes"]
        )
    ):
        raise BootstrapError("collector readiness response has no ready node evidence")
    return cast(dict[str, Any], report)


def wait_collector_readiness(
    site: RenderedSite,
    cluster_id: str,
    *,
    timeout_seconds: float = 600,
    interval_seconds: float = 10,
) -> dict[str, Any]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("collector readiness timeout must be finite and positive")
    if not math.isfinite(interval_seconds) or interval_seconds < 0:
        raise ValueError("collector readiness interval must be finite and nonnegative")
    last: dict[str, Any] | None = None
    last_error: str | None = None
    try:
        with deadline_scope("join collector readiness", timeout_seconds) as deadline:
            while True:
                remaining_timeout(timeout_seconds)
                try:
                    last = _collector_readiness_report(site, cluster_id)
                    last_error = None
                    remaining_timeout(timeout_seconds)
                    if last.get("ready") is True:
                        return last
                except (BootstrapError, json.JSONDecodeError) as exc:
                    last_error = str(exc)
                except subprocess.TimeoutExpired:
                    last_error = "collector readiness request exceeded its time budget"
                time.sleep(min(interval_seconds, deadline.remaining()))
    except (TimeoutError, subprocess.TimeoutExpired):
        pass
    detail = last_error or json.dumps(last or {}, sort_keys=True)
    raise BootstrapError(
        f"{cluster_id} collectors did not become ready within "
        f"{timeout_seconds:g}s: {diagnostic_text(detail)}"
    )
