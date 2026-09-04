from __future__ import annotations

import json
import time
from typing import Any

from regional_release_config import (
    ClusterLocalReleaseError,
    ClusterTarget,
    ReleaseError,
)
from regional_release_gpu_rollout import agents_converged, gpu_node_items
from regional_release_probes import probe_source
from regional_release_runtime_identity import exec_cpu_ingress_probe

WAIT_AGENTS_POLL_SECONDS = 5
WAIT_AGENTS_MAX_POLL_SECONDS = 15
WAIT_AGENTS_POLL_BACKOFF = 1.5


def wait_agents(
    release: Any,
    target: ClusterTarget,
    artifact_sha: str,
    *,
    bundle_sha: str | None = None,
    template_sha: str | None = None,
    config_digest: str | None = None,
    runtime_profile_version: str | None = None,
    node_names: tuple[str, ...] = (),
    timeout_seconds: int = 900,
    legacy_identity: bool = False,
    agent_identity: dict[str, Any] | None = None,
) -> None:
    expected_bundle = None
    expected_template = None
    if not legacy_identity:
        expected_bundle = (
            bundle_sha
            if bundle_sha is not None
            else (
                release.bundle_sha
                if release.config.release_manifest_schema_version >= 3
                else None
            )
        )
        expected_template = (
            template_sha
            if template_sha is not None
            else (
                release.node_template_sha
                if release.config.release_manifest_schema_version >= 3
                else None
            )
        )
    expected_config = config_digest or release.config.agent_config_digest
    expected_profile = runtime_profile_version or release.config.runtime_profile_version
    expected_nodes = frozenset(node_names) if node_names else None
    deadline = time.monotonic() + timeout_seconds
    poll_seconds = float(WAIT_AGENTS_POLL_SECONDS)
    while time.monotonic() < deadline:
        nodes = gpu_node_items(release, target, fresh=True)
        selected_nodes = [
            item
            for item in nodes
            if (
                item.get("metadata", {})
                .get("labels", {})
                .get("sagemaker.amazonaws.com/cluster-name")
                == target.hyperpod_cluster_name
                and (
                    expected_nodes is None
                    or str(item.get("metadata", {}).get("name") or "") in expected_nodes
                )
            )
        ]
        failed_nodes = sorted(
            str(item.get("metadata", {}).get("name") or "")
            for item in selected_nodes
            if (
                (item.get("metadata", {}).get("annotations") or {}).get(
                    "gpu-fault.io/installer-state"
                )
                == "Failed"
            )
        )
        if failed_nodes:
            raise ClusterLocalReleaseError(
                f"{target.cluster_id} installer failed on: " + ", ".join(failed_nodes)
            )
        if agents_converged(
            nodes,
            target,
            artifact_sha,
            bundle_sha=expected_bundle,
            template_sha=expected_template,
            config_digest=expected_config,
            require_node_uid=True,
            node_names=expected_nodes,
        ) and release._agent_heartbeats_converged(
            target,
            node_count=len(selected_nodes),
            node_names=tuple(
                sorted(
                    str(item.get("metadata", {}).get("name") or "")
                    for item in selected_nodes
                )
            ),
            artifact_sha=artifact_sha,
            config_digest=expected_config,
            runtime_profile_version=expected_profile,
            bundle_sha=expected_bundle,
            template_sha=expected_template,
            agent_identity=agent_identity,
        ):
            return
        if release.runner.dry_run:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_seconds, remaining))
        # Installer waves take minutes, so back off after the fast early polls
        # instead of re-listing every node every 5s for the whole window.
        poll_seconds = min(
            WAIT_AGENTS_MAX_POLL_SECONDS,
            poll_seconds * WAIT_AGENTS_POLL_BACKOFF,
        )
    raise ReleaseError(f"{target.cluster_id} agents did not converge")


def agent_heartbeats_converged(
    release: Any,
    target: ClusterTarget,
    *,
    node_count: int,
    node_names: tuple[str, ...],
    artifact_sha: str,
    config_digest: str,
    runtime_profile_version: str,
    bundle_sha: str | None,
    template_sha: str | None,
    agent_identity: dict[str, Any] | None = None,
) -> bool:
    if release.runner.dry_run:
        return True
    raw = exec_cpu_ingress_probe(
        release,
        script=probe_source("agent_convergence"),
        failure="agent heartbeat convergence",
        input_text=json.dumps(
            {
                "cluster_id": target.cluster_id,
                "node_names": list(node_names),
                "artifact": artifact_sha,
                "config": config_digest,
                "profile": runtime_profile_version,
                "bundle": bundle_sha,
                "template": template_sha,
                "protocol": (agent_identity or {}).get("agent_protocol_version"),
                "version": (agent_identity or {}).get("agent_version"),
                "compatibility": (agent_identity or {}).get("compatibility_digest"),
                "policy": (agent_identity or {}).get("policy_version"),
                "key_version": (agent_identity or {}).get("node_action_key_version"),
            }
        ),
    )
    result = json.loads(raw)
    return (
        node_count > 0
        and int(result.get("active", 0)) == node_count
        and int(result.get("aligned", 0)) == node_count
    )
