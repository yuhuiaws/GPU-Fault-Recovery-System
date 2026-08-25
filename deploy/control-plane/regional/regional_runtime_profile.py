from __future__ import annotations

import json
from typing import Any

import regional_deployment_inventory as inventory
import yaml
from regional_release_config import ReleaseConfig, ReleaseError


INSPECT_SCRIPT = """
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import RuntimeProfile
from gpu_fault.store.shared.errors import NotFoundError

payload = json.load(sys.stdin)
desired = compile_runtime_profile(
    RuntimeProfile.model_validate(payload)
)
try:
    existing = ApplicationContext.from_environment().store.get_profile(
        desired.profile_version
    )
except NotFoundError:
    existing = None
print(json.dumps({
    "desired": desired.model_dump(mode="json"),
    "existing": (
        existing.model_dump(mode="json")
        if existing is not None else None
    ),
}, separators=(",", ":")))
"""

REGISTER_SCRIPT = """
import json
import os
import sys
import urllib.request

payload = sys.stdin.buffer.read()
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/runtime-profiles",
    data=payload,
    method="POST",
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Execution-Token": (
            os.environ["GPU_FAULT_EXECUTION_TOKEN"]
        ),
    },
)
with urllib.request.urlopen(request, timeout=15) as response:
    print(json.dumps(json.load(response), separators=(",", ":")))
"""


def render_runtime_profile_payload(config: ReleaseConfig) -> dict[str, Any]:
    document = yaml.safe_load(config.runtime_profile_source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ReleaseError("runtime profile source must contain one mapping")
    payload = dict(document)
    payload["cluster_id"] = config.runtime_profile_registration_cluster_id
    payload["profile_version"] = config.runtime_profile_version
    return payload


def ensure_runtime_profile(release: Any) -> None:
    runner = release.runner
    config = release.config
    if runner.dry_run:
        return
    pod = runner.run(
        release._cpu(
            "-n",
            config.namespace,
            "get",
            "pod",
            "-l",
            f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ),
        capture=True,
    )
    if not pod:
        raise ReleaseError("cannot register the Runtime Profile: no CPU ingress Pod")
    payload = render_runtime_profile_payload(config)
    payload_text = json.dumps(payload, separators=(",", ":"))
    inspection = json.loads(
        runner.run(
            release._cpu(
                "-n",
                config.namespace,
                "exec",
                "-i",
                pod,
                "--",
                "python",
                "-c",
                INSPECT_SCRIPT,
            ),
            input_text=payload_text,
            capture=True,
            sensitive=True,
        )
    )
    desired = inspection.get("desired")
    existing = inspection.get("existing")
    if not isinstance(desired, dict):
        raise ReleaseError("Runtime Profile validation returned no desired profile")
    warnings = desired.get("warnings") or []
    if warnings:
        raise ReleaseError(
            "Runtime Profile has unavailable OWN/DELEGATE capabilities: "
            + "; ".join(str(item) for item in warnings)
        )
    if existing is not None:
        if existing != desired:
            raise ReleaseError(
                "Runtime Profile "
                f"{config.runtime_profile_version} already exists "
                "with different content; choose a new profile version "
                "instead of overwriting a live policy"
            )
        return
    registered = json.loads(
        runner.run(
            release._cpu(
                "-n",
                config.namespace,
                "exec",
                "-i",
                pod,
                "--",
                "python",
                "-c",
                REGISTER_SCRIPT,
            ),
            input_text=payload_text,
            capture=True,
            sensitive=True,
        )
    )
    if registered != desired:
        raise ReleaseError(
            "registered Runtime Profile differs from the validated payload"
        )
