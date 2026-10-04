from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault_release.regional_release_config import ReleaseConfig, ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_profile_model import (
    ProfileModelError,
    RuntimeProfileSpec,
    cluster_runtime_profiles,
    runtime_profile_policy_sha256,
    runtime_profiles,
    site_runtime_profile,
)
from gpu_fault_release.regional_release_runtime_identity import (
    CONTROL_PLANE_PYTHON,
    exec_cpu_ingress_command,
    exec_cpu_ingress_probe,
)


def render_runtime_profile_payload(
    config: ReleaseConfig, spec: RuntimeProfileSpec | None = None
) -> dict[str, Any]:
    """The registration payload of one profile: the site's, or a cluster's own."""

    profile = spec or site_runtime_profile(config)
    if profile.source is None:
        raise ReleaseError("runtime profile source is not configured")
    document = yaml.safe_load(profile.source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ReleaseError("runtime profile source must contain one mapping")
    payload = dict(document)
    payload["cluster_id"] = profile.registration_cluster_id
    payload["profile_version"] = profile.version
    return payload


def runtime_profile_policy_digest(path: Path) -> str:
    try:
        return runtime_profile_policy_sha256(path)
    except ProfileModelError as exc:
        raise ReleaseError(str(exc)) from exc


def inspect_runtime_profile(
    release: Any, spec: RuntimeProfileSpec | None = None
) -> dict[str, Any]:
    config = release.config
    payload = render_runtime_profile_payload(config, spec)
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
    release: Any, spec: RuntimeProfileSpec | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    # The site profile keeps the historical one-argument call; only a
    # cluster's own profile passes its spec.
    inspection = (
        inspect_runtime_profile(release)
        if spec is None
        else inspect_runtime_profile(release, spec)
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
    if existing is not None and not isinstance(existing, dict):
        raise ReleaseError("Runtime Profile inspection returned invalid existing data")
    return desired, existing


def verify_runtime_profile(release: Any) -> None:
    """Every profile version in use is registered with the declared policy."""

    for index, spec in enumerate(runtime_profiles(release.config)):
        desired, existing = _validated_inspection(release, spec if index else None)
        if existing is None:
            raise ReleaseError(f"Runtime Profile {spec.version} is not registered")
        if existing != desired:
            raise ReleaseError(
                f"Runtime Profile {spec.version} differs from the declared policy"
            )


def verify_runtime_profile_check(release: Any) -> tuple[str, dict[str, Any]]:
    """``verify_runtime_profile`` as a status check: (summary, details)."""

    verify_runtime_profile(release)
    return (
        "declared Runtime Profile is registered without drift",
        {
            "version": release.config.runtime_profile_version,
            "cluster_overrides": cluster_runtime_profiles(release.config),
        },
    )


def ensure_runtime_profile(release: Any) -> None:
    """Register every profile version in use: the site's, then each cluster's own."""

    runner = release.runner
    config = release.config
    if runner.dry_run:
        return
    for index, spec in enumerate(runtime_profiles(config)):
        own = spec if index else None
        desired, existing = _validated_inspection(release, own)
        if existing is not None:
            if existing != desired:
                raise ReleaseError(
                    "Runtime Profile "
                    f"{spec.version} already exists "
                    "with different content; choose a new profile version "
                    "instead of overwriting a live policy"
                )
            continue
        _register_profile(release, spec, own, desired)


def _register_profile(
    release: Any,
    spec: RuntimeProfileSpec,
    own: RuntimeProfileSpec | None,
    desired: dict[str, Any],
) -> None:
    """Register one profile; a failed POST is re-inspected and retried once.

    Registration is idempotent -- the API stores the compiled profile under
    its version and the engine compares content before and after -- so a POST
    that failed or timed out (live 2026-10-04: the eks-v1 registration of a
    joining cluster got no answer within the probe's budget) is checked for
    having landed and otherwise sent once more, instead of failing the join on
    one unanswered request.
    """

    payload_text = json.dumps(
        render_runtime_profile_payload(release.config, own), separators=(",", ":")
    )
    error: ReleaseError | None = None
    for attempt in range(2):
        if attempt:
            _desired, existing = _validated_inspection(release, own)
            if existing == desired:
                return
            if existing is not None:
                raise ReleaseError(
                    f"Runtime Profile {spec.version} already exists with different "
                    "content; choose a new profile version instead of overwriting "
                    "a live policy"
                )
        try:
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
        except ReleaseError as exc:
            error = exc
            continue
        if registered != desired:
            raise ReleaseError(
                "registered Runtime Profile differs from the validated payload"
            )
        return
    assert error is not None
    raise ReleaseError(
        f"Runtime Profile {spec.version} registration failed twice: {error}"
    ) from error
