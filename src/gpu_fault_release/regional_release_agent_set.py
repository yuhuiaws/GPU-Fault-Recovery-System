"""The Agent-set invariant a release checks before it mutates anything.

Every cluster's ACTIVE, lease-valid Agents must be exactly its GPU nodes; the
rollback snapshot then has one identity per cluster to restore to. The check
is read-only, re-read once across a heartbeat gap, and names what differs.

The ``sync-state`` of a ``join-cluster`` / ``remove-cluster`` is scoped to the
verb's target (``regional_verify_scope``): it commits the release state for
the site the verb left, which records no Agent set (``previous`` is cleared),
so another cluster's Agents not matching its nodes -- a node a spot reclaim
just took, its replacement not installed yet -- is that cluster's finding,
logged as a ``not-gating`` line, never a refusal (which, before activation,
rolled a join back). The joining cluster's set still gates, and every deploy
(whose rollback snapshot the identities are) judges every cluster as before.

A cluster parked at zero GPU nodes (``regional_parked_cluster``) has no Agent
and no node: its empty set matches. The capture records it as an explicit
entry without an identity (``zero_node_identity``), which every reader of the
snapshot -- the CPU roll's heartbeat barrier, the rollback, the resume, a
newcomer's join -- takes for "nothing to roll, nothing to restore".
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable

from gpu_fault.cluster_scope import is_eks_managed
from gpu_fault_release import regional_parked_cluster as parked_cluster
from gpu_fault_release import regional_release_state as state
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_legacy import AGENT_IDENTITY_FIELDS
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import exec_cpu_ingress_probe
from gpu_fault_release.regional_verify_scope import gates, narrate_not_gating

__all__ = [
    "AGENT_SET_RECHECK_SECONDS",
    "agent_identities",
    "capture_agent_identities_matching_nodes",
]
#: The check name of another cluster's Agent-set finding (``not-gating`` lines).
CHECK = "agent_set"

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


def agent_identities(
    release: Any,
    records: list[dict[str, Any]],
    *,
    parked: Callable[[Any], bool] | None = None,
) -> dict[str, dict[str, Any]]:
    """Each cluster's one Agent identity (``AGENT_IDENTITY_FIELDS``) and its
    node ids, from the ``active_agent_identities`` probe's ``records``.

    A cluster without an ACTIVE, lease-valid Agent, or whose Agents differ,
    refuses -- unless it is another cluster of a scoped ``sync-state``: then
    the problem is logged and the cluster left out -- or a gating cluster
    without any Agent that ``parked`` accepts (it may refuse by itself): then
    the cluster is left out.
    """

    result: dict[str, dict[str, Any]] = {}
    for target in release.config.clusters:
        selected = [
            item for item in records if item.get("cluster_id") == target.cluster_id
        ]
        identities = {
            tuple(item.get(field) for field in AGENT_IDENTITY_FIELDS)
            for item in selected
        }
        if (
            not selected
            and parked is not None
            and gates(release, target.cluster_id)
            and parked(target)
        ):
            continue
        if len(identities) != 1:
            problem = (
                "active Agent identities differ"
                if selected
                else "has no active Agent identity to capture"
            )
            if gates(release, target.cluster_id):
                raise ReleaseError(f"{target.cluster_id} {problem}")
            narrate_not_gating(release, CHECK, target.cluster_id, problem)
            continue
        identity = dict(zip(AGENT_IDENTITY_FIELDS, identities.pop(), strict=True))
        identity["node_ids"] = sorted(str(item["node_id"]) for item in selected)
        result[target.cluster_id] = identity
    return result


def _other_clusters(
    release: Any, identities: dict[str, dict[str, Any]], targets: list[Any]
) -> None:
    """Log another cluster's Agent set that does not match its nodes (scoped
    ``sync-state`` only); an unreadable node list is logged the same way."""

    for target in targets:
        identity = identities.get(target.cluster_id)
        if identity is None:
            continue  # already logged by ``agent_identities``
        try:
            expected = set(release._target_node_names(target))
        except Exception as error:  # noqa: BLE001 - another cluster's, logged
            narrate_not_gating(
                release, CHECK, target.cluster_id, f"node list unreadable: {error}"
            )
            continue
        if expected != set(identity["node_ids"]):
            kind = "GPU" if is_eks_managed(target) else "HyperPod"
            narrate_not_gating(
                release,
                CHECK,
                target.cluster_id,
                _agent_set_mismatch(target.cluster_id, expected, identity, kind, {}),
            )


def capture_agent_identities_matching_nodes(release: Any) -> dict[str, dict[str, Any]]:
    """Every cluster's ACTIVE Agents must be exactly its GPU nodes.

    Fail-closed: a release must not start over a node without a live Agent or
    an Agent the cluster no longer has. The sets are re-read once after a
    heartbeat period when they differ (live 2026-10-04: the mixed-site settle
    deploy refused with a bare "does not match" while every HyperPod Agent was
    ACTIVE and ready minutes later), and the refusal names the nodes and the
    Agent records behind the difference.

    A gating cluster without any Agent whose selector selects no node --
    parked or unavailable (``regional_parked_cluster.without_nodes``) -- is
    recorded as an explicit entry with no identity and no node. Live Agents on
    nodes the selector no longer selects still refuse: their set is not the
    cluster's (empty) node set.
    """

    capture_identities = getattr(release, "_capture_agent_identities", None)
    # A scoped sync-state judges the target's set; another's is only logged.
    gating = [
        item for item in release.config.clusters if gates(release, item.cluster_id)
    ]
    others = [item for item in release.config.clusters if item not in gating]
    zero_nodes: dict[str, dict[str, Any]] = {}

    def accepted_zero_nodes(target: Any) -> bool:
        proof = parked_cluster.without_nodes(release, target)
        if proof is None:
            return False
        if target.cluster_id not in zero_nodes:
            parked_cluster.narrate(
                target.cluster_id,
                state=parked_cluster.state_of(proof),
                step=CHECK,
                recorded="no Agent identity, no node",
                reason=str(proof.get("reason")),
            )
        zero_nodes[target.cluster_id] = proof
        return True

    problem = ""
    for attempt in range(2):
        identities = (
            capture_identities()
            if capture_identities is not None
            else state.capture_agent_identities(release, parked=accepted_zero_nodes)
        )
        problem = ""
        for target, expected_nodes in state._map_clusters(
            release,
            lambda target: set(release._target_node_names(target)),
            targets=[item for item in gating if item.cluster_id not in zero_nodes],
            failure="active Agent set capture",
        ):
            captured_nodes = set(identities[target.cluster_id]["node_ids"])
            if expected_nodes != captured_nodes:
                kind = "GPU" if is_eks_managed(target) else "HyperPod"
                facts = _lease_facts(release, target.cluster_id)
                problem = _agent_set_mismatch(
                    target.cluster_id,
                    expected_nodes,
                    identities[target.cluster_id],
                    kind,
                    facts,
                )
                break
        if not problem:
            _other_clusters(release, identities, others)
            return {
                **identities,
                **{
                    cluster_id: parked_cluster.zero_node_identity(proof)
                    for cluster_id, proof in zero_nodes.items()
                },
            }
        if attempt == 0:
            print(
                f"active Agent set re-read in {AGENT_SET_RECHECK_SECONDS:.0f}s: {problem}",
                file=sys.stderr,
                flush=True,
            )
            SLEEP(AGENT_SET_RECHECK_SECONDS)
    raise ReleaseError(problem)
