"""The failure-domain map reaches the executor by exactly the path the admin renders.

`gpu-fault-admin failure-domain-map` writes one ConfigMap; the control-worker
mounts it and points `GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP` at the file. If
either side drifts the domain budget tier goes silently inert again, which is
the defect this wiring exists to close.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from gpu_fault.admin.failure_domain_map import (
    FAILURE_DOMAIN_CONFIGMAP,
    FAILURE_DOMAIN_MOUNT_DIR,
)
from gpu_fault.execution.remediation_budget import FAILURE_DOMAIN_MAP_ENV
from gpu_fault.failure_domains import FAILURE_DOMAIN_LABELS
from gpu_fault_release import regional_release_fleet_rollout as FLEET_MODULE

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "deploy/control-plane/regional/generated"


def _worker() -> dict:
    return yaml.safe_load(
        (GENERATED / "gpu-fault-control-worker.yaml").read_text(encoding="utf-8")
    )


def test_control_worker_mounts_the_optional_failure_domain_map() -> None:
    pod = _worker()["spec"]["template"]["spec"]
    (api,) = pod["containers"]

    env = {item["name"]: item for item in api["env"]}
    assert env[FAILURE_DOMAIN_MAP_ENV]["valueFrom"]["configMapKeyRef"] == {
        "name": FAILURE_DOMAIN_CONFIGMAP,
        "key": "map-path",
        "optional": True,
    }
    mount = next(
        item for item in api["volumeMounts"] if item["name"] == "failure-domain-map"
    )
    assert mount == {
        "name": "failure-domain-map",
        "mountPath": FAILURE_DOMAIN_MOUNT_DIR,
        "readOnly": True,
    }
    volume = next(
        item for item in pod["volumes"] if item["name"] == "failure-domain-map"
    )
    assert volume["configMap"] == {"name": FAILURE_DOMAIN_CONFIGMAP, "optional": True}


def test_fleet_rollout_and_remediation_budget_share_one_failure_domain_priority() -> (
    None
):
    # Two readers of "what is a failure domain" must not be able to disagree.
    assert FLEET_MODULE.FAILURE_DOMAIN_LABELS is FAILURE_DOMAIN_LABELS
