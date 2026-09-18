"""Keep HA SQL probes on the same credential source as running Store pools."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from scripts.e2e.regional.regional_pod_inventory import ready_pod_records
from scripts.e2e.regional.regional_live_fixture import component_python


def store_probe_script(script: str) -> str:
    # This executes only inside the CPU probe process. Never return its DSN.
    preamble = """
import os
from gpu_fault.store.postgres.pool import StoreCredentials
os.environ["GPU_FAULT_STORE_URL"] = StoreCredentials(
    os.environ["GPU_FAULT_STORE_URL"],
    path=os.environ.get("GPU_FAULT_STORE_URL_FILE"),
).conninfo()
"""
    return preamble + "\n" + script


def cpu_store_probe(
    control: Callable[..., str], script: str, *arguments: str
) -> dict[str, Any]:
    pods = ready_pod_records(
        json.loads(control("get", "pod", "-l", "app=gpu-fault-api-ha", "-o", "json"))
    )
    if not pods:
        raise RuntimeError("HA Store probe requires a Ready CPU ingress Pod")
    output = control(
        "exec",
        "-i",
        pods[0]["name"],
        "--",
        component_python("cpu"),
        "-",
        *arguments,
        stdin=store_probe_script(script).encode(),
        timeout=180,
    )
    value = json.loads(output.splitlines()[-1])
    if not isinstance(value, dict):
        raise ValueError("HA Store probe response is not an object")
    return value
