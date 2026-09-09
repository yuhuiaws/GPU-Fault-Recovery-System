"""Render and ship the node failure-domain map during the control-plane apply.

The remediation budget's ``domain:`` tier only works for nodes the executor
can place in a failure domain, and the control plane holds no node topology of
its own. The release engine does: every upgrade and bootstrap already lists
each managed GPU cluster's nodes for the fleet rollout. So the same apply step
renders ``gpu-fault-failure-domain-map`` from those labels, refuses to ship a
document the worker would refuse to load, applies it, and hands the apply
script a content digest to stamp on the control-worker pod template -- the
worker rolls when the map changed and only then, through the same
pod-template-annotation mechanism the role ConfigMaps use.

A site whose nodes carry none of the configured labels gets an empty map for
every cluster, which the executor loads as "no domain tier": the ConfigMap's
presence is then a no-op, exactly like its absence.
"""

from __future__ import annotations

import json
from typing import Any

from gpu_fault.failure_domains import (
    failure_domain_configmap,
    failure_domain_map,
    failure_domain_map_sha256,
    validate_failure_domain_manifest,
)
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_gpu_rollout import gpu_node_items

FAILURE_DOMAIN_MAP_SHA256_ENV = "GPU_FAULT_FAILURE_DOMAIN_MAP_SHA256"


def render_failure_domain_map(release: Any) -> dict[str, Any]:
    """The ConfigMap for every managed cluster's nodes, validated, not applied."""

    mapping: dict[str, dict[str, str]] = {}
    for target in release.config.clusters:
        mapping.update(
            failure_domain_map(
                target.cluster_id,
                gpu_node_items(release, target),
                label_keys=release.config.failure_domain_labels,
            )
        )
    manifest = failure_domain_configmap(mapping, namespace=release.config.namespace)
    try:
        validate_failure_domain_manifest(manifest)
    except ValueError as exc:
        raise ReleaseError(f"failure-domain map render is invalid: {exc}") from exc
    return manifest


def apply_failure_domain_map(release: Any) -> str:
    """Render, validate and ``kubectl apply`` the map; return its content digest.

    A dry run lists no nodes and applies nothing, so it returns an empty digest
    and the apply script stamps no annotation.
    """

    if release.runner.dry_run:
        return ""
    manifest = render_failure_domain_map(release)
    release.runner.run(
        release._cpu("apply", "-f", "-"),
        input_text=json.dumps(manifest, sort_keys=True),
    )
    return failure_domain_map_sha256(manifest)
