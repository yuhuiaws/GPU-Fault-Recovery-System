"""``spec.clusters[].runtimeProfile``: a GPU cluster's own Runtime Profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gpu_fault.admin.site_fields import (
    IDENTIFIER_PATTERN,
    SiteConfigError,
    mapping,
    optional_text,
    required_text,
)
from gpu_fault.digests import SHA256_PATTERN

__all__ = ["ClusterRuntimeProfileSiteConfig", "cluster_runtime_profile_version"]


def cluster_runtime_profile_version(config: Mapping[str, Any], cluster_id: str) -> str:
    """The Runtime Profile version ``cluster_id``'s Agents and executor run.

    The cluster's own ``runtime_profile`` override when the site declares one,
    else the site profile -- the same rule the custody binding and the release
    pins follow. Anything the control plane later compiles against a workflow's
    ``runtime_profile_version`` (step owners, escalation rungs, provider
    correlation) reads that profile, so an operator submission for a cluster
    must name the cluster's version, not the site's (live 2026-10-05: a
    validated restore on an eks-managed cluster was stamped ``hyperpod-v1``).
    """

    entry: Mapping[str, Any] = next(
        (
            item
            for item in config.get("clusters") or ()
            if str(item.get("cluster_id") or "") == cluster_id
        ),
        {},
    )
    override = entry.get("runtime_profile") or {}
    version = (
        str(override.get("version") or "") if isinstance(override, Mapping) else ""
    )
    return version or str(config["runtime_profile"]["version"])


@dataclass(frozen=True)
class ClusterRuntimeProfileSiteConfig:
    """A GPU cluster's own Runtime Profile, overriding the site's.

    Needed when the site's profile cannot own this cluster's ``nodeReboot``
    (a plain EKS cluster joining a HyperPod site, or the reverse): the
    cluster's Agents, Executor, Watcher and installer are pinned to this
    version and the cluster's registration tells the control plane so.
    ``agentConfigDigest`` is the Agent config digest computed for this
    version; the deploy/join fill it in, like ``spec.release.agentConfigDigest``.
    """

    source: str
    template_source: str
    version: str
    agent_config_digest: str | None = None

    @classmethod
    def from_value(cls, value: object, path: str) -> ClusterRuntimeProfileSiteConfig:
        data = mapping(
            value,
            path,
            allowed={"source", "templateSource", "version", "agentConfigDigest"},
        )
        source = required_text(data.get("source"), f"{path}.source")
        version = required_text(data.get("version"), f"{path}.version")
        if not IDENTIFIER_PATTERN.fullmatch(version):
            raise SiteConfigError(f"{path}.version contains unsupported characters")
        digest = optional_text(
            data.get("agentConfigDigest"), f"{path}.agentConfigDigest"
        )
        if digest is not None and not SHA256_PATTERN.fullmatch(digest):
            raise SiteConfigError(f"{path}.agentConfigDigest must be a SHA-256")
        return cls(
            source=source,
            template_source=required_text(
                data.get("templateSource", source), f"{path}.templateSource"
            ),
            version=version,
            agent_config_digest=digest,
        )
