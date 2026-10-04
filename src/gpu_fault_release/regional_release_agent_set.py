"""The Agent-set invariant a release checks before it mutates anything.

Every cluster's ACTIVE, lease-valid Agents must be exactly its GPU nodes; the
rollback snapshot then has one identity per cluster to restore to. The check
is read-only, re-read once across a heartbeat gap, and names what differs.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from gpu_fault.cluster_scope import is_eks_managed
from gpu_fault_release import regional_release_state as state
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import exec_cpu_ingress_probe

__all__ = ["AGENT_SET_RECHECK_SECONDS", "capture_agent_identities_matching_nodes"]

#: One Agent heartbeat period plus slack: the probe keeps an Agent only while
#: its lease (``observed_at + max_heartbeat_age``) is ahead of the Pod's clock,
#: so a single late heartbeat can drop a healthy node for a few seconds.
AGENT_SET_RECHECK_SECONDS = 35.0


#: Monkeypatchable wait between the two reads.
SLEEP = time.sleep


def _agent_set_mismatch(
    cluster_id: str,
    expected_nodes: set[str],
    identity: dict[str, Any],
    kind: str,
    records: dict[str, dict[str, Any]],
) -> str:
    captured = set(identity["node_ids"])
    missing = sorted(expected_nodes - captured)
    extra = sorted(captured - expected_nodes)
    detail = []
    for node in [*missing, *extra]:
        record = records.get(node)
        if record is None:
            detail.append(f"{node}: no ACTIVE Agent with a live lease")
        else:
            detail.append(
                f"{node}: lifecycle={record.get('lifecycle_state')} "
                f"lease_expires_at={record.get('lease_expires_at')} "
                f"last_seen_at={record.get('last_seen_at')}"
            )
    return (
        f"{cluster_id} active Agent set does not match {kind} nodes: "
        f"nodes without an active Agent={missing or '[]'}, "
        f"Agents on nodes outside the cluster={extra or '[]'}; " + "; ".join(detail)
    )


def agent_lease_facts(release: Any) -> dict[str, dict[str, dict[str, Any]]]:
    """``{cluster_id: {node_id: lifecycle/lease/last_seen}}`` from the same
    probe, for the Agent-set diagnosis; never part of the recorded snapshot."""

    records = json.loads(
        exec_cpu_ingress_probe(
            release,
            script=probe_source("active_agent_identities"),
            failure="Agent lease facts",
        )
    )
    facts: dict[str, dict[str, dict[str, Any]]] = {}
    for item in records:
        facts.setdefault(str(item.get("cluster_id")), {})[str(item.get("node_id"))] = {
            key: item.get(key)
            for key in ("lifecycle_state", "lease_expires_at", "last_seen_at")
        }
    return facts


def _lease_facts(release: Any, cluster_id: str) -> dict[str, dict[str, Any]]:
    """Best effort: the refusal stands without them, it only says less."""

    try:
        return agent_lease_facts(release).get(cluster_id, {})
    except (ReleaseError, ValueError, OSError) as error:
        print(f"Agent lease facts unavailable: {error}", file=sys.stderr)
        return {}


def capture_agent_identities_matching_nodes(release: Any) -> dict[str, dict[str, Any]]:
    """Every cluster's ACTIVE Agents must be exactly its GPU nodes.

    Fail-closed: a release must not start over a node without a live Agent or
    an Agent the cluster no longer has. The sets are re-read once after a
    heartbeat period when they differ (live 2026-10-04: the mixed-site settle
    deploy refused with a bare "does not match" while every HyperPod Agent was
    ACTIVE and ready minutes later), and the refusal names the nodes and the
    Agent records behind the difference.
    """

    capture_identities = getattr(release, "_capture_agent_identities", None)
    problem = ""
    for attempt in range(2):
        agent_identities = (
            capture_identities()
            if capture_identities is not None
            else state.capture_agent_identities(release)
        )
        problem = ""
        for target, expected_nodes in state._map_clusters(
            release,
            lambda target: set(release._target_node_names(target)),
            targets=release.config.clusters,
            failure="active Agent set capture",
        ):
            captured_nodes = set(agent_identities[target.cluster_id]["node_ids"])
            if expected_nodes != captured_nodes:
                kind = "GPU" if is_eks_managed(target) else "HyperPod"
                facts = _lease_facts(release, target.cluster_id)
                problem = _agent_set_mismatch(
                    target.cluster_id,
                    expected_nodes,
                    agent_identities[target.cluster_id],
                    kind,
                    facts,
                )
                break
        if not problem:
            return agent_identities
        if attempt == 0:
            print(
                f"active Agent set re-read in {AGENT_SET_RECHECK_SECONDS:.0f}s: {problem}",
                file=sys.stderr,
                flush=True,
            )
            SLEEP(AGENT_SET_RECHECK_SECONDS)
    raise ReleaseError(problem)
