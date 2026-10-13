"""The Runtime Profile a release pins a cluster to, and its policy digest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

__all__ = [
    "RuntimeProfileSpec",
    "cluster_profiles_digest",
    "cluster_runtime_profiles",
    "runtime_profile_policy_sha256",
    "runtime_profile_version_for",
    "runtime_profiles",
    "site_runtime_profile",
    "target_agent_config_digest",
    "target_runtime_profile",
    "target_runtime_profile_version",
]


class ProfileModelError(ValueError):
    pass


def runtime_profile_policy_sha256(path: Path) -> str:
    """Digest of a Runtime Profile's policy content, independent of its
    ``cluster_id``/``profile_version`` labels and of claim order."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ProfileModelError("runtime profile source must contain one mapping")
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
                    item, sort_keys=True, separators=(",", ":")
                ),
            )
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RuntimeProfileSpec:
    """One Runtime Profile a site registers: the site's, or a cluster's own.

    ``registration_cluster_id`` is the registration anchor the control plane
    stores with the profile (``runtime_profile.registration_cluster_id`` for
    the site profile, a cluster running the override for an override; see
    ``runtime_profiles``). ``version`` is the pin the cluster's Agents,
    Executor and installer carry.
    """

    #: ``None`` only for a config that declares no profile file at all.
    source: Path | None
    template_source: Path | None
    version: str
    registration_cluster_id: str
    agent_config_digest: str


# The accessors below read a release config and a cluster target by their
# attributes (``runtime_profile_version``, ``agent_config_digest`` and the
# target's optional ``runtime_profile_*`` override), so every release-engine
# caller resolves "this cluster's pin, else the site's" the same way.


def target_runtime_profile_version(config: Any, target: Any) -> str:
    """The version ``target``'s Agents, Executor and installer are pinned to."""

    return str(
        getattr(target, "runtime_profile_version", "") or config.runtime_profile_version
    )


def target_agent_config_digest(config: Any, target: Any) -> str:
    """The Agent configuration digest computed for ``target``'s version."""

    return str(getattr(target, "agent_config_digest", "") or config.agent_config_digest)


def site_runtime_profile(config: Any) -> RuntimeProfileSpec:
    """The site profile every cluster follows unless it overrides it."""

    source = getattr(config, "runtime_profile_source", None)
    return RuntimeProfileSpec(
        source=source,
        template_source=getattr(config, "runtime_profile_template_source", None)
        or source,
        version=str(config.runtime_profile_version),
        registration_cluster_id=str(
            getattr(config, "runtime_profile_registration_cluster_id", "")
        ),
        agent_config_digest=str(getattr(config, "agent_config_digest", "")),
    )


def _overrides(config: Any) -> list[Any]:
    return [
        target
        for target in getattr(config, "clusters", ())
        if getattr(target, "overrides_runtime_profile", False)
    ]


def target_runtime_profile(config: Any, target: Any) -> RuntimeProfileSpec:
    """The profile ``target`` is pinned to: its own, or the site's."""

    if not getattr(target, "overrides_runtime_profile", False):
        return site_runtime_profile(config)
    source = target.runtime_profile_source
    return RuntimeProfileSpec(
        source=source,
        template_source=target.runtime_profile_template_source or source,
        version=target.runtime_profile_version,
        registration_cluster_id=target.cluster_id,
        agent_config_digest=target.agent_config_digest,
    )


def runtime_profiles(
    config: Any, *, anchor: str | None = None
) -> tuple[RuntimeProfileSpec, ...]:
    """Every distinct profile version in use: the site's first, then each
    cluster override once, in cluster order.

    A cluster version is anchored on its first member in cluster order, or on
    ``anchor`` when that cluster runs it: a ``join-cluster`` release registers
    under the cluster it joins, the one member it has just put into the
    registry (a batch sibling listed first may not be there yet). The anchor
    is a label, not policy -- join, batch join, deploy and remove-cluster all
    reorder the members -- so drift is judged without it (``profile_policy``).
    """

    site = site_runtime_profile(config)
    profiles = {site.version: site}
    for target in _overrides(config):
        spec = target_runtime_profile(config, target)
        profiles.setdefault(spec.version, spec)
        if target.cluster_id == anchor and spec.version != site.version:
            profiles[spec.version] = replace(
                profiles[spec.version], registration_cluster_id=anchor
            )
    return tuple(profiles.values())


def runtime_profile_version_for(config: Any, cluster_id: str) -> str:
    """The effective version of the named cluster; the site's when unknown."""

    for target in getattr(config, "clusters", ()):
        if target.cluster_id == cluster_id:
            return target_runtime_profile_version(config, target)
    return str(config.runtime_profile_version)


def cluster_runtime_profiles(config: Any) -> dict[str, dict[str, str]]:
    """``{cluster_id: {version, agent_config_digest, policy_sha256}}`` for the
    clusters that override the site profile; empty on a single-profile site."""

    overrides: dict[str, dict[str, str]] = {}
    for target in _overrides(config):
        spec = target_runtime_profile(config, target)
        if spec.source is None:
            raise ProfileModelError(
                f"{target.cluster_id} runtime profile has no source"
            )
        overrides[str(target.cluster_id)] = {
            "version": spec.version,
            "agent_config_digest": spec.agent_config_digest,
            "policy_sha256": runtime_profile_policy_sha256(spec.source),
        }
    return overrides


def cluster_profiles_digest(config: Any) -> str | None:
    """One digest over every cluster's own Runtime Profile (version, Agent
    config digest, policy), or ``None`` on a single-profile site so recorded
    release states written before per-cluster profiles still compare equal."""

    overrides = cluster_runtime_profiles(config)
    if not overrides:
        return None
    return hashlib.sha256(
        json.dumps(overrides, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
