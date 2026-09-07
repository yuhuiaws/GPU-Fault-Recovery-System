from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]
from gpu_fault_release.regional_release_config import ReleaseConfig, ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import (
    CONTROL_PLANE_PYTHON,
    exec_cpu_ingress_command,
    exec_cpu_ingress_probe,
)


def render_runtime_profile_payload(config: ReleaseConfig) -> dict[str, Any]:
    document = yaml.safe_load(config.runtime_profile_source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ReleaseError("runtime profile source must contain one mapping")
    payload = dict(document)
    payload["cluster_id"] = config.runtime_profile_registration_cluster_id
    payload["profile_version"] = config.runtime_profile_version
    return payload


def runtime_profile_policy_digest(path: Path) -> str:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ReleaseError("runtime profile source must contain one mapping")
    policy = {
        key: value
        for key, value in document.items()
        if key not in {"cluster_id", "profile_version"}
    }
    for field in ("claims", "observed"):
        values = policy.get(field)
        if isinstance(values, list):
            policy[field] = sorted(
                values,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
    return hashlib.sha256(
        json.dumps(
            policy,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def inspect_runtime_profile(release: Any) -> dict[str, Any]:
    config = release.config
    payload = render_runtime_profile_payload(config)
    payload_text = json.dumps(payload, separators=(",", ":"))
    return json.loads(
        exec_cpu_ingress_probe(
            release,
            script=probe_source("runtime_profile_inspect"),
            failure="Runtime Profile inspection",
            input_text=payload_text,
            sensitive=True,
        )
    )


def _validated_inspection(
    release: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    inspection = inspect_runtime_profile(release)
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
    if existing is not None and not isinstance(existing, dict):
        raise ReleaseError("Runtime Profile inspection returned invalid existing data")
    return desired, existing


def verify_runtime_profile(release: Any) -> None:
    desired, existing = _validated_inspection(release)
    if existing is None:
        raise ReleaseError(
            f"Runtime Profile {release.config.runtime_profile_version} is not registered"
        )
    if existing != desired:
        raise ReleaseError(
            "Runtime Profile "
            f"{release.config.runtime_profile_version} differs from the declared policy"
        )


def ensure_runtime_profile(release: Any) -> None:
    runner = release.runner
    config = release.config
    if runner.dry_run:
        return
    desired, existing = _validated_inspection(release)
    if existing is not None:
        if existing != desired:
            raise ReleaseError(
                "Runtime Profile "
                f"{config.runtime_profile_version} already exists "
                "with different content; choose a new profile version "
                "instead of overwriting a live policy"
            )
        return
    payload_text = json.dumps(
        render_runtime_profile_payload(config),
        separators=(",", ":"),
    )
    registered = json.loads(
        exec_cpu_ingress_command(
            release,
            arguments=(
                CONTROL_PLANE_PYTHON,
                "-c",
                probe_source("runtime_profile_register"),
            ),
            failure="Runtime Profile registration",
            input_text=payload_text,
            sensitive=True,
        )
    )
    if registered != desired:
        raise ReleaseError(
            "registered Runtime Profile differs from the validated payload"
        )
