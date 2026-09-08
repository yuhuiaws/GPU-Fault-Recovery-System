from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
QUICK_VALIDATION_EVIDENCE_ENV = "GPU_FAULT_QUICK_VALIDATION_EVIDENCE"
QUICK_VALIDATION_EVIDENCE_FILE = "quick-validation.json"
QUICK_VALIDATION_MAX_AGE_SECONDS = 600


def quick_validation_evidence_path(state_dir: Path) -> Path:
    """The one place a deploy's quick-validation evidence lives.

    `state_dir` is the managed state directory -- the one holding `site.yaml`.
    The release driver finalizes the evidence here and the admin `status`
    command reads it from here; both derive the path from this function, so the
    two can no longer drift apart (they did: the driver kept the file under its
    per-release directory, `status` looked at the root, and every report re-ran
    every probe).
    """

    return state_dir.expanduser().resolve() / QUICK_VALIDATION_EVIDENCE_FILE


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def quick_validation_evidence(
    release: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    raw = os.getenv(QUICK_VALIDATION_EVIDENCE_ENV, "").strip()
    if not raw:
        return None, None
    path = Path(raw).expanduser().resolve()
    try:
        if not path.is_file() or path.stat().st_mode & 0o077:
            raise ValueError("evidence file is missing or not private")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("evidence schema is invalid")
        expected_site = {
            "site_name": release.config.site_name,
            "aws_region": release.config.aws_region,
            "cpu_eks_arn": release.config.cpu_eks_arn,
            "cluster_ids": sorted(
                target.cluster_id for target in release.config.clusters
            ),
        }
        age = int(time.time()) - int(value["verified_at_epoch"])
        checks = value.get("checks")
        expected_state = str(value.get("release_state_sha256") or "")
        allowed_checks = {
            "control_plane_role_split",
            "runtime_component_identity",
            *(
                f"data_plane_executor:{target.cluster_id}"
                for target in release.config.clusters
            ),
        }
        if (
            value.get("release_id") != release.release_id
            or value.get("release_delivery_sha256")
            != release.config.release_delivery_sha256
            or value.get("site_identity") != expected_site
            or not isinstance(checks, list)
            or any(not isinstance(item, str) for item in checks)
            or not set(checks).issubset(allowed_checks)
            or age < 0
            or age > QUICK_VALIDATION_MAX_AGE_SECONDS
            or len(expected_state) != 64
        ):
            raise ValueError("evidence identity or freshness does not match")
        live_state = release._load_state()
        if canonical_sha256(live_state) != expected_state:
            raise ValueError("live release state changed after quick validation")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, str(exc)
    return value, None


def read_only_verifier_details(
    release: Any,
    *,
    reused_checks: set[str] | None = None,
) -> dict[str, object]:
    reused = reused_checks or set()
    control_output: object
    if "control_plane_role_split" in reused:
        control_output = {"reused": True}
    else:
        control_output = release.runner.run(
            [
                "python3",
                str(
                    ROOT
                    / "deploy/control-plane/tools/verify_control_plane_role_split.py"
                ),
            ],
            env={
                **os.environ,
                "KUBECONFIG": release.config.cpu_kubeconfig,
                "GPU_FAULT_NAMESPACE": release.config.namespace,
                "GPU_FAULT_RUNTIME_IMAGE": release.runtime_image,
            },
            capture=True,
        )
    gpu_outputs = {}
    for target in release.config.clusters:
        check = f"data_plane_executor:{target.cluster_id}"
        if check in reused:
            gpu_outputs[target.cluster_id] = {"reused": True}
        else:
            gpu_outputs[target.cluster_id] = release.runner.run(
                [
                    "python3",
                    str(ROOT / "deploy/dataplane/tools/verify_dataplane_executor.py"),
                ],
                env={
                    **os.environ,
                    "GPU_FAULT_NAMESPACE": release.config.namespace,
                    "GPU_FAULT_KUBE_CONTEXT": target.context,
                    "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": (
                        release.config.cpu_kubeconfig
                    ),
                    "GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP": release.executor_wheel_cm,
                },
                capture=True,
            )
    return {"cpu": control_output, "clusters": gpu_outputs}
