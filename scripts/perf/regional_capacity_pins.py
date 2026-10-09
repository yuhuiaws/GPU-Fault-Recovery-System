"""Runtime Profile pins the synthetic perf clusters inherit from their provider.

``FleetRegistry.cluster_policy`` holds a cluster's Agents to the pins of its
own regional registration (``runtime_profile_version`` and
``agent_config_digest``) when it carries them, and to the site policy
(``GPU_FAULT_REQUIRED_*``) otherwise. The perf seed copies its synthetic
Agents from a live Agent of the synthetic clusters' provider, so they report
that provider's Runtime Profile and config digest. A synthetic registration
without the same pins is held by the fleet compatibility preflight before its
first destructive step on a site whose base family is the other provider
(``runtime profile mismatch: expected eks-v1, got hyperpod-v1; runtime profile
drift is PIN_AHEAD_OF_FLEET: no live agent in perf-cap-000 runs the pinned
value`` -- 96 of 128 integrated workflows PENDING for the whole window). The
synthetic clusters of provider P therefore register with exactly the pins the
production registration of P carries -- none when P is the site's base family
-- and registration refuses when no production registration of P exists.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

#: Mirrors ``seed_regional_action_workflows.SYNTHETIC_PROVIDER_ENV``: the seed
#: runs standalone on the api Pod (``python3 -`` over stdin) and cannot import
#: this module, so the name is declared in both places.
SYNTHETIC_PROVIDER_ENV = "ACTION_SYNTHETIC_PROVIDER"
PIN_FIELDS = ("runtime_profile_version", "agent_config_digest")


def synthetic_registration_provider() -> str:
    """The provider the synthetic clusters register as (default hyperpod)."""

    from gpu_fault.cluster_scope import cluster_provider

    try:
        return cluster_provider(os.getenv(SYNTHETIC_PROVIDER_ENV) or None).value
    except ValueError as exc:
        raise RuntimeError(
            f"{SYNTHETIC_PROVIDER_ENV} is not a cluster provider: {exc}"
        ) from exc


def registration_provider(entry: Mapping[str, Any]) -> str:
    """The provider a registry entry is read as (absent means the default)."""

    from gpu_fault.cluster_scope import cluster_provider

    try:
        return cluster_provider(entry.get("provider") or None).value
    except ValueError as exc:
        raise RuntimeError(
            f"production registration {entry.get('cluster_id')!r} has an unknown "
            f"provider: {exc}"
        ) from exc


def inherited_pins(
    production: Iterable[Mapping[str, Any]], provider: str
) -> dict[str, Any]:
    """The pins a synthetic cluster of ``provider`` registers with.

    ``{"provider", "pins", "source_cluster_ids"}``: ``pins`` holds the
    ``runtime_profile_version`` / ``agent_config_digest`` the production
    registrations of ``provider`` carry (empty when that family inherits the
    site pins). No production registration of ``provider``, or registrations
    of it that disagree on their pins, refuse: there is no unambiguous value a
    joined cluster of that family carries.
    """

    entries = [dict(entry) for entry in production]
    matching = [entry for entry in entries if registration_provider(entry) == provider]
    if not matching:
        seen = sorted({registration_provider(entry) for entry in entries}) or ["none"]
        raise RuntimeError(
            f"synthetic clusters register as {provider}, but no production "
            f"registration of provider {provider} exists to inherit Runtime "
            f"Profile pins from (production registrations: {', '.join(seen)})"
        )
    by_pins: dict[tuple[str, ...], list[str]] = {}
    for entry in matching:
        key = tuple(str(entry.get(field) or "") for field in PIN_FIELDS)
        by_pins.setdefault(key, []).append(str(entry.get("cluster_id")))
    if len(by_pins) != 1:
        described = "; ".join(
            f"{', '.join(sorted(cluster_ids))}: "
            + ", ".join(
                f"{field}={value or '<none>'}"
                for field, value in zip(PIN_FIELDS, key, strict=True)
            )
            for key, cluster_ids in sorted(by_pins.items())
        )
        raise RuntimeError(
            f"production registrations of provider {provider} carry different "
            "Runtime Profile pins, so the synthetic clusters cannot inherit one "
            f"unambiguously: {described}"
        )
    ((key, cluster_ids),) = by_pins.items()
    pins = {field: value for field, value in zip(PIN_FIELDS, key, strict=True) if value}
    return {
        "provider": provider,
        "pins": pins,
        "source_cluster_ids": sorted(cluster_ids),
    }
