"""The Runtime Profile a release pins a cluster to, and its policy digest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped,unused-ignore]

__all__ = ["RuntimeProfileSpec", "runtime_profile_policy_sha256"]


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
    the site profile, the cluster itself for an override). ``version`` is the
    pin the cluster's Agents, Executor and installer carry.
    """

    source: Path
    template_source: Path
    version: str
    registration_cluster_id: str
    agent_config_digest: str
