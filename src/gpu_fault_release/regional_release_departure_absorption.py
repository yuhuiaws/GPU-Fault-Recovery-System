"""The forward path absorbs a node that provably left, the way rollback does.

A release records each cluster's Agent node set when it starts and fixes the
fleet deployment's node list when the cluster's roll starts. A node the
provider reclaims after either moment broke the forward path in three places:
the candidate CPU heartbeat barrier wanted a refreshed heartbeat from it (a
plain ``ReleaseError``, which the upgrade answers with its automatic rollback);
a later wave's safety gate found its Agent's lease lapsing and timed out; and
the wave holding it waited out the 900 s convergence timeout for a Node
Kubernetes no longer has, while the fleet deployment kept the node
``INSTALLING`` -- an ``IN_PROGRESS`` record that fences destructive remediation
for the whole cluster.

The rule here is the rollback verification's: a node is set aside only on the
shared departure proof (``regional_release_departed_nodes.departure_proof`` --
gone from Kubernetes, gone at the provider, Agent ``REVOKED`` / absent / lease
lapsed), the proof is recorded under ``departed_nodes`` in the release state
and narrated as ``departed-node-excluded``, and an absence it cannot prove
fails closed with the proof's reason. Only an Agent that fell *silent* is ever
judged: a blocker that says the Agent is alive (a pin that is not aligned) is
the gate's own failure, and so is a silent Agent on a Node that is still
``Ready``.

The proof is not decidable the moment a node goes: the Node object lingers
``NotReady`` until the node lifecycle controller deletes it, EC2 reports
``shutting-down`` before ``terminated``, HyperPod still lists a node it is
replacing, and an Agent lease runs ``max_heartbeat_age_seconds`` (90 s) past
the last heartbeat. Failing at the first look would turn a routine spot reclaim
into a rollback that then needs the same proof. So a node that looks gone (no
Node, or a Node that is NotReady or being deleted) is re-judged every
``DEPARTURE_RECHECK_SECONDS`` for at most ``DEPARTURE_DECISION_SECONDS`` --
the product's own ten-minute departure guard -- and refused with the proof's
last reason once that has passed. Waiting cannot make an unsafe decision: the
exclusion still needs the whole proof, and a node that comes back simply
heartbeats again.

A departed node leaves the fleet deployment through the probe's
``exclude-departed`` operation (``fleet_deployment_command``): it is taken out
of the record's nodes and waves while ``PENDING`` or ``INSTALLING``, so the
record ends ``SUCCEEDED`` over the nodes that remain, never ``FAILED`` for a
node that left; a node that converged before it left stays ``READY``. The
evidence row names the deployment it was taken out of.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from gpu_fault_release.regional_release_config import (
    ClusterLocalReleaseError,
    ClusterTarget,
)
from gpu_fault_release.regional_release_departed_nodes import (
    departure_proof,
    record_departed_nodes,
    recorded_departed_nodes,
)
from gpu_fault_release.regional_release_errors import WaveAgentSafetyError
from gpu_fault_release.regional_release_gpu_rollout import gpu_node_items
from gpu_fault_release.regional_release_narration import narrate_step

#: How long a node that looks gone may stay undecided: the product's
#: ``DEPARTED_AGENT_GUARD_AGE``, its own horizon for "silent means gone".
DEPARTURE_DECISION_SECONDS = 600.0
#: The proof reads AWS and execs into the control plane: at most this often.
DEPARTURE_RECHECK_SECONDS = 30.0
BARRIER_PHASE = "cpu-heartbeat-barrier"
WAVE_PHASE = "fleet-wave"
#: The fleet probe operation that takes departed nodes out of a rollout record.
EXCLUDE_DEPARTED_OPERATION = "exclude-departed"
#: Barrier and gate blocker reasons that say the Agent fell silent.
SILENT_AGENT_REASONS = frozenset(
    {"missing", "heartbeat-not-refreshed", "lease-missing", "lease-margin"}
)
PRESENT_REASON = (
    "the Node is still in Kubernetes ({state}); a departure is proven only "
    "once Kubernetes no longer lists it"
)


def silent_agent(reason: object) -> bool:
    """A blocker reason a departed node produces (``lifecycle-REVOKED`` too)."""

    text = str(reason or "")
    return text in SILENT_AGENT_REASONS or text.startswith("lifecycle-")


def node_leaving_state(item: Mapping[str, Any]) -> str | None:
    """``NotReady`` / ``being deleted`` for a Node on its way out, else None."""

    if (item.get("metadata") or {}).get("deletionTimestamp"):
        return "being deleted"
    ready = any(
        isinstance(condition, dict)
        and condition.get("type") == "Ready"
        and condition.get("status") == "True"
        for condition in (item.get("status") or {}).get("conditions") or []
    )
    return None if ready else "NotReady"


@dataclass(frozen=True)
class DepartureVerdict:
    """One look at the nodes a gate, a barrier or a wait could not account for."""

    #: Proven departed (now or earlier), evidence recorded.
    departed: frozenset[str]
    #: On a Ready Node: not a departure, whatever the Agent says.
    alive: tuple[str, ...]
    #: Look gone, not provable yet, still inside the decision window.
    pending: tuple[str, ...]
    #: Look gone, unproven past the decision window: ``node -> reason``.
    refused: Mapping[str, str]

    def refusal(self, cluster_id: str, what: str) -> str:
        return (
            f"{cluster_id} {what} names nodes missing from Kubernetes that are not "
            "provably departed: "
            + "; ".join(
                f"{node}: {reason}" for node, reason in sorted(self.refused.items())
            )
        )


@dataclass
class DepartureWatch:
    """One cluster's departure decisions across the polls of a gate or a wait.

    Remembers when each node first looked gone, when it was last put to the
    proof, the evidence once proven and the reason while not, so the proof is
    read at most every ``DEPARTURE_RECHECK_SECONDS`` and a node is refused
    once it has looked gone for ``DEPARTURE_DECISION_SECONDS``.
    """

    release: Any
    target: ClusterTarget
    phase: str
    #: The clock the windows are measured on; ``time.monotonic`` when None.
    monotonic: Callable[[], float] | None = None
    first_seen: dict[str, float] = field(default_factory=dict)
    checked_at: dict[str, float] = field(default_factory=dict)
    departed: dict[str, dict[str, Any]] = field(default_factory=dict)
    unproven: dict[str, str] = field(default_factory=dict)

    @property
    def departed_ids(self) -> frozenset[str]:
        return frozenset(self.departed)

    def decide(self, node_ids: Collection[str]) -> DepartureVerdict:
        """Judge ``node_ids`` against a fresh Node list and, when due, the proof."""

        now = self.monotonic() if self.monotonic is not None else time.monotonic()
        asked = sorted({str(node) for node in node_ids if str(node)})
        open_nodes = [node for node in asked if node not in self.departed]
        items = {
            str((item.get("metadata") or {}).get("name") or ""): item
            for item in gpu_node_items(self.release, self.target, fresh=True)
        }
        alive: list[str] = []
        lingering: dict[str, str] = {}
        absent: list[str] = []
        for node in open_nodes:
            item = items.get(node)
            if item is None:
                absent.append(node)
                continue
            state = node_leaving_state(item)
            if state is None:
                alive.append(node)
            else:
                lingering[node] = state
        for node in [*absent, *lingering]:
            self.first_seen.setdefault(node, now)
        for node, state in lingering.items():
            self.unproven[node] = PRESENT_REASON.format(state=state)
        due = [
            node
            for node in absent
            if now - self.checked_at.get(node, float("-inf"))
            >= DEPARTURE_RECHECK_SECONDS
        ]
        if due:
            self._prove(due, present=frozenset(items))
            for node in due:
                self.checked_at[node] = now
        pending = [node for node in [*absent, *lingering] if node not in self.departed]
        refused = {
            node: self.unproven.get(node, "not judged")
            for node in pending
            if now - self.first_seen[node] >= DEPARTURE_DECISION_SECONDS
        }
        return DepartureVerdict(
            departed=frozenset(node for node in asked if node in self.departed),
            alive=tuple(alive),
            pending=tuple(node for node in pending if node not in refused),
            refused=refused,
        )

    def _prove(self, nodes: list[str], *, present: frozenset[str]) -> None:
        proof = departure_proof(
            self.release, self.target, nodes, present_node_names=present
        )
        rows = [{**row, "excluded_at": self.phase} for row in proof.departed]
        record_departed_nodes(self.release, self.target, rows, phase=self.phase)
        for row in rows:
            node = str(row["node_id"])
            self.departed[node] = dict(row)
            self.unproven.pop(node, None)
        for line in proof.unproven:
            node, _, reason = line.partition(": ")
            self.unproven[node] = reason or line


# -- the candidate CPU heartbeat barrier ------------------------------------


def without_recorded_departures(
    release: Any, expected_by_cluster: Mapping[str, list[str]]
) -> dict[str, list[str]]:
    """``expected_by_cluster`` minus nodes this release already proved departed.

    The proof was written by an earlier gate of the same release; a node is
    left out only while Kubernetes still has no Node of that name, so a name
    that came back is expected again.
    """

    state = getattr(release, "state", None)
    targets = {
        target.cluster_id: target
        for target in getattr(getattr(release, "config", None), "clusters", ())
    }
    result = {
        cluster_id: list(nodes) for cluster_id, nodes in expected_by_cluster.items()
    }
    for cluster_id, nodes in expected_by_cluster.items():
        recorded = set(recorded_departed_nodes(state, cluster_id)) & set(nodes)
        target = targets.get(cluster_id)
        if not recorded or target is None:
            continue
        present = {
            str((item.get("metadata") or {}).get("name") or "")
            for item in gpu_node_items(release, target, fresh=True)
        }
        gone = sorted(recorded - present)
        if not gone:
            continue
        result[cluster_id] = [node for node in nodes if node not in gone]
        narrate_step(
            "departed-node-already-excluded",
            cluster=cluster_id,
            phase=BARRIER_PHASE,
            nodes=",".join(gone),
        )
    return {cluster_id: nodes for cluster_id, nodes in result.items() if nodes}


def settle_departed_barrier_nodes(
    release: Any,
    expected_by_cluster: Mapping[str, list[str]],
    barrier: Callable[[dict[str, list[str]]], dict[str, Any]],
    *,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Run the in-Pod barrier; absorb the blocked nodes that provably left.

    Returns the barrier's evidence: ``PASSED`` once every blocker of a run is a
    proven departure (the rest of that run's nodes refreshed in the same
    window) or a later run passes over the narrowed set; otherwise the last
    ``FAILED`` evidence, with ``departure_unproven`` when the refusal is an
    unproven absence. Anything the departure rule does not cover -- more
    blockers than the probe names, an Agent that is alive, a silent Agent on
    a Ready Node, a cluster outside the release -- returns the failure as is.
    """

    expected = {
        cluster_id: sorted(nodes) for cluster_id, nodes in expected_by_cluster.items()
    }
    result = barrier(expected)
    watches: dict[str, DepartureWatch] = {}
    while result.get("status") != "PASSED":
        targets = {
            target.cluster_id: target
            for target in getattr(release.config, "clusters", ())
        }
        blocked = _barrier_blockers(result)
        if blocked is None or any(cluster not in targets for cluster in blocked):
            return result
        pending = False
        for cluster_id, nodes in blocked.items():
            watch = watches.setdefault(
                cluster_id,
                DepartureWatch(
                    release, targets[cluster_id], BARRIER_PHASE, monotonic=monotonic
                ),
            )
            verdict = watch.decide(nodes)
            if verdict.alive:
                return result
            if verdict.refused:
                return {
                    **result,
                    "departure_unproven": verdict.refusal(
                        cluster_id, "heartbeat barrier"
                    ),
                }
            pending = pending or bool(verdict.pending)
        departed = {
            cluster_id: sorted(watch.departed_ids)
            for cluster_id, watch in watches.items()
            if watch.departed_ids
        }
        if not pending:
            return {
                "status": "PASSED",
                "expected_count": result.get("expected_count"),
                "refreshed_count": result.get("refreshed_count"),
                "departed_excluded": departed,
            }
        (sleep or time.sleep)(DEPARTURE_RECHECK_SECONDS)
        expected = {
            cluster_id: [
                node for node in nodes if node not in set(departed.get(cluster_id, ()))
            ]
            for cluster_id, nodes in expected.items()
        }
        expected = {
            cluster_id: nodes for cluster_id, nodes in expected.items() if nodes
        }
        result = barrier(expected) if expected else {"status": "PASSED"}
        if result.get("status") == "PASSED" and departed:
            result = {**result, "departed_excluded": departed}
    return result


def _barrier_blockers(result: Mapping[str, Any]) -> dict[str, list[str]] | None:
    """``cluster -> silent nodes`` when the barrier named every blocker, else None."""

    blockers = result.get("blockers")
    if not isinstance(blockers, list) or not blockers:
        return None
    try:
        count = int(result.get("blocker_count") or 0)
    except (TypeError, ValueError):
        return None
    if count != len(blockers):
        return None
    blocked: dict[str, list[str]] = {}
    for item in blockers:
        if not isinstance(item, dict) or not silent_agent(item.get("reason")):
            return None
        cluster_id = str(item.get("cluster_id") or "")
        node = str(item.get("node_id") or "")
        if not cluster_id or not node:
            return None
        blocked.setdefault(cluster_id, []).append(node)
    return blocked


def barrier_failure_details(result: Mapping[str, Any]) -> dict[str, Any]:
    """The barrier diagnostics the refusal prints, the departure facts included."""

    details = {
        "expected_count": result.get("expected_count"),
        "refreshed_count": result.get("refreshed_count"),
        "blocker_count": result.get("blocker_count"),
        "blockers": result.get("blockers"),
    }
    for key in ("departure_unproven", "departed_excluded"):
        if result.get(key):
            details[key] = result[key]
    return details


# -- the fleet waves -------------------------------------------------------


class WaveDepartures:
    """A cluster's departures during its node rollout: gate, waits, record.

    Inert unless ``enabled`` (the forward ``upgrade`` phase): every method then
    passes straight through, so a rollback, a join and a bootstrap keep their
    behaviour exactly.
    """

    def __init__(self, release: Any, target: ClusterTarget, *, enabled: bool) -> None:
        self.release = release
        self.target = target
        self.watch = DepartureWatch(release, target, WAVE_PHASE) if enabled else None

    @classmethod
    def for_phase(
        cls, release: Any, target: ClusterTarget, phase: str
    ) -> WaveDepartures:
        return cls(release, target, enabled=phase == "upgrade")

    @property
    def departed_ids(self) -> frozenset[str]:
        return self.watch.departed_ids if self.watch is not None else frozenset()

    def wait_options(self) -> dict[str, Any]:
        """The convergence wait's ``departed`` hook, when this rollout has one."""

        return {"departed": self} if self.watch is not None else {}

    def __call__(self, absent: frozenset[str]) -> frozenset[str]:
        """The wait's hook: which of the named nodes Kubernetes lacks are gone."""

        assert self.watch is not None, "the hook is handed out only when enabled"
        verdict = self.watch.decide(absent)
        if verdict.refused:
            raise ClusterLocalReleaseError(
                verdict.refusal(self.target.cluster_id, "fleet wave")
            )
        return verdict.departed

    def wave_safety(
        self,
        gate: Callable[..., dict[str, Any]],
        *,
        wave: tuple[str, ...],
        safety_node_names: set[str],
    ) -> dict[str, Any]:
        """The wave's safety gate; a blocker that provably left leaves the set.

        The gate's own window is the wait: a node still undecided after it
        sends the gate round again until the decision window runs out.
        """

        while True:
            try:
                return gate(
                    self.release,
                    self.target,
                    wave=wave,
                    node_names=tuple(sorted(safety_node_names)),
                )
            except WaveAgentSafetyError as error:
                if self.watch is None:
                    raise
                blocked = [
                    str(item.get("node_id") or "")
                    for item in error.blockers
                    if isinstance(item, dict) and silent_agent(item.get("reason"))
                ]
                if (
                    error.blocker_count != len(error.blockers)
                    or len(blocked) != len(error.blockers)
                    or not all(blocked)
                ):
                    raise
                verdict = self.watch.decide(blocked)
                if verdict.alive:
                    raise
                if verdict.refused:
                    raise ClusterLocalReleaseError(
                        f"{error}; "
                        + verdict.refusal(self.target.cluster_id, "fleet wave gate")
                    ) from error
                safety_node_names.difference_update(verdict.departed)

    def reconcile_record(
        self, deployment: dict[str, Any], deployment_id: str
    ) -> dict[str, Any]:
        """Take the proven departures out of the fleet deployment record."""

        if self.watch is None or not self.watch.departed:
            return deployment
        statuses = {
            str(item.get("node_id") or ""): str(item.get("status") or "")
            for item in deployment.get("nodes") or []
            if isinstance(item, dict)
        }
        drop = sorted(
            node
            for node in self.watch.departed
            if statuses.get(node) in {"PENDING", "INSTALLING"}
        )
        if not drop:
            return deployment
        result = self.release._fleet_command(
            EXCLUDE_DEPARTED_OPERATION,
            {"deployment_id": deployment_id, "node_ids": drop},
        )
        updated = result.get("deployment") if isinstance(result, dict) else None
        accounted = {
            str(node)
            for key in ("excluded", "absent")
            for node in (result.get(key) if isinstance(result, dict) else None) or []
        }
        if not isinstance(updated, dict) or accounted != set(drop):
            raise ClusterLocalReleaseError(
                f"{self.target.cluster_id} fleet deployment {deployment_id} did not "
                f"exclude the departed nodes {drop}: {json.dumps(result, sort_keys=True)}"
            )
        rows = [
            {
                **self.watch.departed[node],
                "fleet_deployment": {
                    "deployment_id": deployment_id,
                    "status_before": statuses[node],
                },
            }
            for node in drop
        ]
        for row in rows:
            self.watch.departed[str(row["node_id"])] = row
        record_departed_nodes(
            self.release, self.target, rows, phase=WAVE_PHASE, narrate=False
        )
        narrate_step(
            "fleet-deployment-node-excluded",
            cluster=self.target.cluster_id,
            deployment=deployment_id,
            nodes=",".join(drop),
            status=str(updated.get("status") or ""),
        )
        return updated
