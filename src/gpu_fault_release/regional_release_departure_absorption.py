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

Two more shapes of the same departure are absorbed by the waves: a node already
on its way out (NotReady or being deleted) when the cluster's roll starts is
waited for the same way and the roll is planned without it once it is proven
gone, and a destructive workflow the product opened for the departed node --
never dispatched, so harmless on an Agent a wave restarts, but younger than the
product's ten-minute dispatch guard the gate sets such records aside after --
holds the wave until it is old enough, instead of refusing it outright.
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
    ReleaseError,
)
from gpu_fault_release.regional_release_departed_nodes import (
    departure_proof,
    record_departed_nodes,
    recorded_departed_nodes,
)
from gpu_fault_release.regional_release_errors import (
    WaveAgentSafetyError,
    WaveWorkflowSafetyError,
)
from gpu_fault_release.regional_release_gpu_rollout import (
    NODE_INVENTORY,
    gpu_node_items,
)
from gpu_fault_release.regional_release_narration import narrate_step

#: How long a node that looks gone may stay undecided: the product's
#: ``DEPARTED_AGENT_GUARD_AGE``, its own horizon for "silent means gone".
DEPARTURE_DECISION_SECONDS = 600.0
#: The proof reads AWS and execs into the control plane: at most this often.
DEPARTURE_RECHECK_SECONDS = 30.0
#: Two looks at a node this far apart did not see one absence: it was back in
#: between (nothing asked about it), so its decision window starts again. Longer
#: than any gap between two looks of one absence (a gate or barrier window plus
#: a re-check).
DEPARTURE_CONTINUITY_SECONDS = 180.0
#: The wave probe's never-dispatched destructive records still inside the guard.
YOUNG_NEVER_DISPATCHED_KEY = "never_dispatched_young_destructive_workflows"
#: The probe's ``NEVER_DISPATCHED_GUARD`` (the product's dispatch guard): a
#: never-dispatched record this old is set aside on the departure proof.
NEVER_DISPATCHED_GUARD_SECONDS = 600.0
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


def leaving_node_names(
    release: Any, target: ClusterTarget, node_names: Collection[str]
) -> tuple[str, ...]:
    """The nodes of ``node_names`` on their way out, from the roll's own read.

    Only the Nodes the roll's inventory scope already holds are looked at: this
    never adds a read, and a caller that listed names without the scope's
    Nodes gets nothing to judge.
    """

    scope = NODE_INVENTORY.get()
    items = (
        scope[1].get(target.cluster_id)
        if scope is not None and scope[0] is release
        else None
    )
    wanted = set(node_names)
    return tuple(
        sorted(
            name
            for item in items or ()
            if (name := str((item.get("metadata") or {}).get("name") or "")) in wanted
            and node_leaving_state(item) is not None
        )
    )


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
    #: A fresh read of the cluster's Nodes; ``gpu_node_items(fresh=True)`` when
    #: None (a verify, inside its report snapshot, reads past the cache).
    read_nodes: Callable[[], list[dict[str, Any]]] | None = None
    first_seen: dict[str, float] = field(default_factory=dict)
    last_gone: dict[str, float] = field(default_factory=dict)
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
        listed = (
            self.read_nodes()
            if self.read_nodes is not None
            else gpu_node_items(self.release, self.target, fresh=True)
        )
        items = {
            str((item.get("metadata") or {}).get("name") or ""): item for item in listed
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
        for node in alive:
            self._forget(node)
        for node in [*absent, *lingering]:
            if now - self.last_gone.get(node, now) > DEPARTURE_CONTINUITY_SECONDS:
                self._forget(node)
            self.first_seen.setdefault(node, now)
            self.last_gone[node] = now
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

    def _forget(self, node: str) -> None:
        for book in (self.first_seen, self.last_gone, self.checked_at, self.unproven):
            book.pop(node, None)

    def _prove(self, nodes: list[str], *, present: frozenset[str]) -> None:
        try:
            proof = departure_proof(
                self.release, self.target, nodes, present_node_names=present
            )
        except ReleaseError as error:
            # A read that failed decides nothing: the nodes stay undecided and
            # are refused with this reason once the window has passed.
            for node in nodes:
                self.unproven[node] = f"departure proof unavailable: {error}"
            return
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
        sends the gate round again until the decision window runs out. A young
        never-dispatched destructive record of departed nodes holds the gate the
        same bounded way (``_hold_for_departed_workflows``).
        """

        held_since: list[float] = []
        while True:
            try:
                return gate(
                    self.release,
                    self.target,
                    wave=wave,
                    node_names=tuple(sorted(safety_node_names)),
                )
            except WaveWorkflowSafetyError as error:
                if self.watch is None:
                    raise
                self._hold_for_departed_workflows(error, held_since)
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

    def _hold_for_departed_workflows(
        self, error: WaveWorkflowSafetyError, held_since: list[float]
    ) -> None:
        """Return to re-run the gate, or raise: young records of departed nodes.

        The product opens an incident -- and a destructive workflow no Agent was
        ever handed -- for a node that left; the gate sets such a record aside
        on the departure proof once it is past the product's ten-minute dispatch
        guard (``departed_workflow_exclusions``). A wave that absorbed the node
        reaches the next gate sooner than that. Every record the gate kept must
        be one of those, of this cluster, naming only nodes the watch proves (or
        is still proving) departed; then the gate is asked again, for at most
        the decision window. Anything else is the gate's own refusal, as before.
        """

        assert self.watch is not None, "held only when enabled"
        young = {
            str(item.get("request_id") or ""): item
            for item in error.snapshot.get(YOUNG_NEVER_DISPATCHED_KEY) or []
            if isinstance(item, dict)
        }
        remaining = [str(request_id) for request_id in error.remaining]
        records = [young.get(request_id) for request_id in remaining]
        nodes = sorted(
            {
                str(node)
                for record in records
                if record is not None
                for node in record.get("node_ids") or []
                if str(node)
            }
        )
        if (
            not remaining
            or error.remaining_count != len(remaining)
            or any(
                record is None
                or str(record.get("cluster_id") or "") != self.target.cluster_id
                or not record.get("node_ids")
                for record in records
            )
        ):
            raise error
        verdict = self.watch.decide(nodes)
        if verdict.alive:
            raise error
        if verdict.refused:
            raise ClusterLocalReleaseError(
                f"{error}; "
                + verdict.refusal(self.target.cluster_id, "fleet wave gate")
            ) from error
        now = time.monotonic()
        if not held_since:
            held_since.append(now)
            narrate_step(
                "departed-node-workflows-held",
                cluster=self.target.cluster_id,
                workflows=",".join(remaining),
                nodes=",".join(nodes),
            )
        if (
            now - held_since[0]
            > NEVER_DISPATCHED_GUARD_SECONDS + DEPARTURE_RECHECK_SECONDS
        ):
            raise ClusterLocalReleaseError(
                f"{error}; the never-dispatched records of the departed nodes "
                f"{','.join(nodes)} did not pass the product's dispatch guard within "
                f"{NEVER_DISPATCHED_GUARD_SECONDS:.0f}s"
            ) from error
        time.sleep(DEPARTURE_RECHECK_SECONDS)

    def settle_leaving(self, leaving: Collection[str]) -> frozenset[str]:
        """Nodes on their way out when the cluster's roll starts: which left.

        Each is judged until it is proven departed, is Ready again, or its
        decision window runs out; the caller plans the roll without the
        departed ones, and the pre-mutation gate judges the rest as before.
        """

        if self.watch is None or not leaving:
            return frozenset()
        while True:
            verdict = self.watch.decide(leaving)
            if verdict.refused or not verdict.pending:
                return verdict.departed
            time.sleep(DEPARTURE_RECHECK_SECONDS)

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
        try:
            result = self.release._fleet_command(
                EXCLUDE_DEPARTED_OPERATION,
                {"deployment_id": deployment_id, "node_ids": drop},
            )
        except ReleaseError as error:
            # The record stays as it was; the cluster pauses like a wave that
            # did not converge, never a rollback of every converged cluster.
            raise ClusterLocalReleaseError(
                f"{self.target.cluster_id} fleet deployment {deployment_id} did not "
                f"exclude the departed nodes {drop}: {error}"
            ) from error
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
