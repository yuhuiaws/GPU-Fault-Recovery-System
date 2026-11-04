"""What a release's own node churn excuses in its ``verify``, and nothing else.

The driver verifies a release right after its deploy and rolls back on any
FAIL, so a node the release itself saw leave or arrive must not read as a
broken release:

* a Ready node outside the node set the release captured whose installer has
  not succeeded yet -- capacity that joined while the release ran, which the
  steady-state Reconciler is installing -- is left out of the installer check,
  the Agent coverage and the fleet readiness question, and named in a ``WARN``
  (``joined_nodes_installing``);
* an Agent of a node this release proved departed (``departed_nodes`` in the
  release state) that has no Kubernetes Node any more is not coverage, not a
  Runtime Profile to compare and not a collector to wait for, until the
  departed-node cleanup retires its record; so is the Agent of a node the
  release never got to judge (it left before its cluster rolled, or after the
  last wave, or -- for a membership verb -- while the verb ran) once the same
  read-only departure proof holds for it.

That proof is not decidable the moment a node goes (live 2026-11-04: five
minutes after an EC2 Spot interruption the Node was still listed, NotReady,
and its Agent ACTIVE): so, in a cluster whose findings gate this verify, an
Agent whose Node is gone or no longer Ready is judged the way the forward path
judges it (``regional_release_departure_absorption.DepartureWatch``): re-read
every ``DEPARTURE_RECHECK_SECONDS`` for at most ``DEPARTURE_DECISION_SECONDS``,
the product's ten-minute departure guard, excused with its evidence
(``release_churn.departure_evidence``) once Kubernetes no longer lists it, the
provider no longer has it and its lease has lapsed, and otherwise counted --
the coverage refusal then names each node and why (``departure_unproven``). A
Node that is Ready again, or a Kubernetes Node without an Agent, is never
excused. In a cluster a scoped verify does not gate on
(``regional_verify_scope``) the proof is taken once and nothing waits.

``status`` judges the fleet as it is and gets none of this: a node still
installing is a real ``installer is not Succeeded`` there, as before. And a
newcomer the installer did finish is judged like every node -- installed with
any artifact but the release's, it is ``artifact drift``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded, remaining_timeout
from gpu_fault.cluster_scope import node_selector, selector_matches
from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_departed_nodes import (
    departure_proof,
    recorded_departed_nodes,
)
from gpu_fault_release.regional_release_departure_absorption import (
    DEPARTURE_DECISION_SECONDS,
    DEPARTURE_RECHECK_SECONDS,
    PRESENT_REASON,
    DepartureWatch,
)
from gpu_fault_release.regional_release_gpu_rollout import gpu_node_command
from gpu_fault_release.regional_release_joined_nodes import installing_newcomers
from gpu_fault_release.regional_release_narration import narrate_step
from gpu_fault_release.regional_verify_scope import gates

#: Per-cluster key of the control API report, written in ``verify`` only.
RELEASE_CHURN_KEY = "release_churn"
#: The phase the verify's departure evidence and log lines carry.
VERIFY_PHASE = "verify"
ALIVE_REASON = (
    "the Node is Ready again (no departure); the report counted it while it "
    "was not Ready, so verify again"
)
# The clock and the pause of the departure wait (tests replace them).
_monotonic = time.monotonic
_sleep = time.sleep


@dataclass
class _Departures:
    """One cluster's verdict on its Agents without a Ready Node."""

    proven: set[str] = field(default_factory=set)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    unproven: dict[str, str] = field(default_factory=dict)
    #: Still undecided after the first look: the departure wait's nodes.
    undecided: list[str] = field(default_factory=list)


def verify_joining(
    release: Any,
    cluster_id: str,
    ready_items: Sequence[Mapping[str, Any]],
    *,
    verify: bool,
) -> dict[str, str]:
    """``node -> installer state`` of the newcomers a ``verify`` leaves out."""

    return installing_newcomers(release, cluster_id, ready_items) if verify else {}


def annotate_release_churn(
    report: dict[str, Any],
    release: Any,
    joining_by_cluster: Mapping[str, Mapping[str, str]],
) -> None:
    """Write what this release's churn excuses into the report, per cluster."""

    clusters = report.get("clusters")
    state = getattr(release, "state", None)
    targets = {target.cluster_id: target for target in release.config.clusters}
    judged: dict[str, tuple[dict[str, Any], set[str], dict[str, str], _Departures]]
    judged = {}
    for cluster_id, cluster in clusters.items() if isinstance(clusters, dict) else ():
        if not isinstance(cluster, dict):
            continue
        kubernetes = {str(item) for item in cluster.get("kubernetes_node_ids") or []}
        recorded = set(
            recorded_departed_nodes(
                state if isinstance(state, Mapping) else None, cluster_id
            )
        )
        joining = dict(sorted((joining_by_cluster.get(cluster_id) or {}).items()))
        departures = _first_look(
            release,
            targets.get(cluster_id),
            cluster,
            kubernetes=kubernetes,
            excused=(recorded - kubernetes) | set(joining),
        )
        judged[cluster_id] = (cluster, recorded - kubernetes, joining, departures)
    _wait_out_departures(
        release,
        {
            cluster_id: (targets[cluster_id], departures)
            for cluster_id, (*_, departures) in judged.items()
            if departures.undecided
            and cluster_id in targets
            and gates(release, cluster_id)
        },
    )
    for cluster, recorded_gone, joining, departures in judged.values():
        departed = sorted(recorded_gone | departures.proven)
        if departed or joining or departures.unproven:
            cluster[RELEASE_CHURN_KEY] = {
                "departed": departed,
                "proven_by_verify": sorted(departures.proven),
                "joined_installing": joining,
                **(
                    {"departure_evidence": departures.evidence}
                    if departures.evidence
                    else {}
                ),
                **(
                    {"departure_unproven": dict(sorted(departures.unproven.items()))}
                    if departures.unproven
                    else {}
                ),
            }


def _first_look(
    release: Any,
    target: Any,
    cluster: Mapping[str, Any],
    *,
    kubernetes: set[str],
    excused: set[str],
) -> _Departures:
    """Counted Agents without a Ready Node: the proof taken once, from the report.

    An Agent without a Node is put to the departure proof against the report's
    own Node list; one whose Node is listed but not Ready cannot be proven gone
    yet. Both stay undecided for the wait when the proof does not hold.
    """

    expected = {str(item) for item in cluster.get("expected_node_ids") or []}
    candidates = (
        {
            str(item.get("node_id"))
            for item in cluster.get("agents") or []
            if isinstance(item, dict) and item.get("lifecycle_state") != "REVOKED"
        }
        - expected
        - excused
    )
    result = _Departures(
        unproven={
            node: PRESENT_REASON.format(state="not Ready")
            for node in candidates & kubernetes
        }
    )
    absent = sorted(candidates - kubernetes)
    if absent and target is not None:
        try:
            proof = departure_proof(
                release, target, absent, present_node_names=kubernetes
            )
        except ReleaseError as error:
            # Not observable is not proof: the nodes stay undecided.
            result.unproven.update(
                {node: f"departure proof unavailable: {error}" for node in absent}
            )
        else:
            result.proven = set(proof.departed_node_ids)
            result.evidence = [dict(row) for row in proof.departed]
            for line in proof.unproven:
                node, _, reason = line.partition(": ")
                result.unproven[node] = reason or line
    result.undecided = sorted(set(result.unproven) - result.proven)
    return result


def _fresh_nodes(release: Any, target: Any) -> list[dict[str, Any]]:
    """The cluster's Nodes under its selector, read past the report's snapshot.

    The verify reads every object once inside its snapshot; the departure wait
    must see the Node list change, so it asks the API server each time.
    """

    raw = release.runner.run(
        [*gpu_node_command(release, target), "-o", "json"], capture=True
    )
    items = (json.loads(raw) if raw else {}).get("items") or []
    selector = node_selector(target)
    return [
        item
        for item in items
        if isinstance(item, dict)
        and selector_matches((item.get("metadata") or {}).get("labels"), selector)
    ]


def _capped(seconds: float) -> float:
    try:
        return remaining_timeout(seconds)
    except DeploymentDeadlineExceeded:
        return 0.0


def _wait_out_departures(
    release: Any, waiting: Mapping[str, tuple[Any, _Departures]]
) -> None:
    """Re-judge the undecided Agents of the gating clusters until each is decided.

    Bounded by the product's ten-minute departure guard (and the verify's own
    deadline); a node that is Ready again or refused past the guard ends the
    wait at once -- the verify fails on it either way.
    """

    if not waiting:
        return
    watches = {
        cluster_id: DepartureWatch(
            release,
            target,
            VERIFY_PHASE,
            monotonic=_monotonic,
            read_nodes=partial(_fresh_nodes, release, target),
        )
        for cluster_id, (target, _departures) in waiting.items()
    }
    narrate_step(
        "departure-undecided",
        phase=VERIFY_PHASE,
        nodes=",".join(
            f"{cluster_id}/{node}"
            for cluster_id, (_target, departures) in sorted(waiting.items())
            for node in departures.undecided
        ),
        guard=f"{DEPARTURE_DECISION_SECONDS:.0f}s",
    )
    budget = _capped(DEPARTURE_DECISION_SECONDS + DEPARTURE_RECHECK_SECONDS)
    started = _monotonic()
    while _monotonic() - started < budget:
        _sleep(DEPARTURE_RECHECK_SECONDS)
        final = False
        for cluster_id, (_target, departures) in waiting.items():
            if not departures.undecided:
                continue
            watch = watches[cluster_id]
            try:
                verdict = watch.decide(departures.undecided)
            except ReleaseError as error:
                # A read that failed decides nothing; the next look asks again.
                for node in departures.undecided:
                    departures.unproven[node] = f"departure undecided: {error}"
                continue
            for node in verdict.departed:
                departures.proven.add(node)
                departures.evidence.append(dict(watch.departed[node]))
                departures.unproven.pop(node, None)
            for node in verdict.alive:
                departures.unproven[node] = ALIVE_REASON
            departures.unproven.update(verdict.refused)
            for node in verdict.pending:
                departures.unproven[node] = watch.unproven.get(node, "not judged")
            departures.undecided = list(verdict.pending)
            final = final or bool(verdict.alive or verdict.refused)
        if final or not any(item.undecided for _t, item in waiting.values()):
            return


def churn_departed(cluster: Mapping[str, Any]) -> frozenset[str]:
    churn = cluster.get(RELEASE_CHURN_KEY)
    return frozenset(
        str(node)
        for node in (churn.get("departed") if isinstance(churn, dict) else ()) or ()
    )


def departure_unproven(cluster: Mapping[str, Any]) -> dict[str, str]:
    """``node -> reason`` for the Agents whose departure this verify did not prove."""

    churn = cluster.get(RELEASE_CHURN_KEY)
    unproven = churn.get("departure_unproven") if isinstance(churn, dict) else None
    return {
        str(node): str(reason)
        for node, reason in (unproven if isinstance(unproven, dict) else {}).items()
    }


def churn_excused(cluster: Mapping[str, Any]) -> frozenset[str]:
    """The nodes whose Agents this verify does not count, for one cluster."""

    churn = cluster.get(RELEASE_CHURN_KEY)
    joining = churn.get("joined_installing") if isinstance(churn, dict) else None
    return churn_departed(cluster) | frozenset(
        str(node) for node in (joining if isinstance(joining, dict) else {})
    )


def with_joining(
    value: CheckValue, cluster_id: str, joining: Mapping[str, str]
) -> CheckValue:
    """The ``gpu_cluster`` verdict, a ``WARN`` naming the newcomers left out."""

    if not joining:
        return value
    details = dict(value.details or {})
    details["joined_nodes_installing"] = dict(sorted(joining.items()))
    shown = ", ".join(f"{node} ({state})" for node, state in sorted(joining.items()))
    return replace(
        value,
        summary=(
            f"{value.summary}; {len(joining)} node(s) that joined during the release "
            f"are still being installed by the steady-state Reconciler: {shown}"
        ),
        details=details,
        status="WARN",
    )


def with_churn(value: CheckValue, report: Mapping[str, Any]) -> CheckValue:
    """The ``control_api`` verdict, a ``WARN`` when this release's churn was excused."""

    clusters = report.get("clusters")
    parts = []
    for cluster_id, cluster in sorted(
        (clusters if isinstance(clusters, dict) else {}).items()
    ):
        churn = cluster.get(RELEASE_CHURN_KEY) if isinstance(cluster, dict) else None
        if not isinstance(churn, dict):
            continue
        if churn.get("departed"):
            parts.append(
                f"{cluster_id} departed (proven by this release): "
                + ",".join(churn["departed"])
            )
        if churn.get("joined_installing"):
            parts.append(
                f"{cluster_id} joined, still installing: "
                + ",".join(sorted(churn["joined_installing"]))
            )
    if not parts:
        return value
    return replace(
        value,
        summary=f"{value.summary}; not counted: " + "; ".join(parts),
        status="WARN",
    )
