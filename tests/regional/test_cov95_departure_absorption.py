"""The departure absorption rules the release simulations leave unexercised.

``regional_release_departure_absorption`` sets a node aside only on the shared
departure proof and fails closed on everything else. The churn and barrier
simulations drive its happy paths over a whole release; these cases pin the
refusals and the narrow shapes around them directly on the module's public
surface: a barrier failure that does not name its blockers is returned as it
came, a re-run barrier that passes carries the departures proved on the way, a
wave gate whose blockers are not all silent (or whose node is Ready) refuses
as before, an absence that stays unproven past the decision window is refused
with the proof's reason, a young never-dispatched record of a departed node
holds the gate only as long as the product's dispatch guard, and a fleet
record that does not account for every dropped node pauses the cluster.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_departure_absorption as ABSORPTION
from gpu_fault_release.regional_release_config import (
    ClusterLocalReleaseError,
    ClusterTarget,
    ReleaseError,
)
from gpu_fault_release.regional_release_departed_nodes import DepartureProof
from gpu_fault_release.regional_release_errors import (
    WaveAgentSafetyError,
    WaveWorkflowSafetyError,
)
from tests.regional._cov95_release_support import Clock

CLUSTER = "gpu-a"
LIVE = "node-live"
GONE = "node-gone"
SLOW = "node-slow"
TARGET = ClusterTarget(
    cluster_id=CLUSTER,
    context=CLUSTER,
    executor_irsa_role_arn="arn:aws:iam::123456789012:role/example",
    region="us-east-1",
    hyperpod_cluster_name="hp-" + CLUSTER,
    eks_cluster_arn=f"arn:aws:eks:us-east-1:123456789012:cluster/{CLUSTER}",
)
UNPROVEN = "the instance is still listed"


def node(name: str, *, ready: bool = True, deleting: bool = False) -> dict[str, Any]:
    item: dict[str, Any] = {
        "metadata": {"name": name},
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}]
        },
    }
    if deleting:
        item["metadata"]["deletionTimestamp"] = "2026-11-05T04:00:00Z"
    return item


class Release:
    """The release collaborator: its state, its clusters, its fleet probe."""

    def __init__(self) -> None:
        self.release_id = "release-cov95"
        self.state: dict[str, Any] = {}
        self.config = SimpleNamespace(clusters=(TARGET,))
        self.fleet_commands: list[tuple[str, dict[str, Any]]] = []
        self.fleet_answer: Any = None

    def _fleet_command(self, operation: str, payload: dict[str, Any]) -> Any:
        self.fleet_commands.append((operation, payload))
        return self.fleet_answer


class Site:
    """The Node list, the departure proof and the engine clock of one cluster."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        listed: dict[str, dict[str, Any]] | None = None,
        provable: tuple[str, ...] = (GONE,),
    ) -> None:
        self.clock = Clock()
        self.release = Release()
        self.listed = listed if listed is not None else {LIVE: node(LIVE)}
        self.provable = set(provable)
        self.proofs: list[list[str]] = []
        monkeypatch.setattr(ABSORPTION, "gpu_node_items", self.nodes)
        monkeypatch.setattr(ABSORPTION, "departure_proof", self.proof)
        monkeypatch.setattr(ABSORPTION, "time", self.clock)
        monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)

    def nodes(
        self, release: Any, target: ClusterTarget, *, fresh: bool
    ) -> list[dict[str, Any]]:
        assert (release, target, fresh) == (self.release, TARGET, True), (
            "the watch reads its own cluster's Nodes past the cache"
        )
        return list(self.listed.values())

    def proof(
        self,
        release: Any,
        target: ClusterTarget,
        nodes: list[str],
        *,
        present_node_names: frozenset[str],
    ) -> DepartureProof:
        assert not set(nodes) & present_node_names, "only absent nodes are proved"
        self.proofs.append(sorted(nodes))
        return DepartureProof(
            departed=tuple(
                {"node_id": name, "provider": "hyperpod", "agent": None}
                for name in nodes
                if name in self.provable
            ),
            unproven=tuple(
                f"{name}: {UNPROVEN}" for name in nodes if name not in self.provable
            ),
        )

    def departures(self, phase: str = "upgrade") -> ABSORPTION.WaveDepartures:
        return ABSORPTION.WaveDepartures.for_phase(self.release, TARGET, phase)

    def recorded(self) -> list[str]:
        entry = self.release.state["departed_nodes"][CLUSTER]
        return [row["node_id"] for row in entry["nodes"]]


def silent(
    *names: str, reason: str = "heartbeat-not-refreshed"
) -> list[dict[str, Any]]:
    return [
        {"cluster_id": CLUSTER, "node_id": name, "reason": reason} for name in names
    ]


# --- Node state -----------------------------------------------------------------


def test_a_node_with_a_deletion_timestamp_is_leaving_even_while_ready() -> None:
    assert ABSORPTION.node_leaving_state(node(GONE, deleting=True)) == "being deleted"
    assert ABSORPTION.node_leaving_state(node(GONE, ready=False)) == "NotReady"
    assert ABSORPTION.node_leaving_state(node(LIVE)) is None


# --- the candidate CPU heartbeat barrier ----------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        {"status": "FAILED", "blocker_count": 0, "blockers": []},
        {"status": "FAILED", "blocker_count": "two", "blockers": silent(GONE)},
        {
            "status": "FAILED",
            "blocker_count": 1,
            "blockers": [{"cluster_id": CLUSTER, "node_id": "", "reason": "missing"}],
        },
    ],
    ids=["no-blockers", "unreadable-count", "unnamed-node"],
)
def test_a_barrier_failure_that_does_not_name_every_blocker_is_returned_as_is(
    monkeypatch: pytest.MonkeyPatch, failure: dict[str, Any]
) -> None:
    site = Site(monkeypatch)
    requests: list[dict[str, list[str]]] = []

    def barrier(expected: dict[str, list[str]]) -> dict[str, Any]:
        requests.append(expected)
        return failure

    result = ABSORPTION.settle_departed_barrier_nodes(
        site.release,
        {CLUSTER: [LIVE, GONE]},
        barrier,
        sleep=site.clock.sleep,
        monotonic=site.clock.monotonic,
    )

    assert result is failure, "a failure the departure rule cannot read stays as is"
    assert requests == [{CLUSTER: [GONE, LIVE]}]
    assert site.proofs == [], "nothing was put to the departure proof"
    assert "departed_nodes" not in site.release.state
    assert site.clock.sleeps == []


def test_a_rerun_barrier_that_passes_carries_the_departures_proved_before_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GONE is proved at the first look; SLOW is still undecided, so the barrier
    runs again over the narrowed set and SLOW refreshes in that run."""

    site = Site(monkeypatch)
    requests: list[dict[str, list[str]]] = []

    def barrier(expected: dict[str, list[str]]) -> dict[str, Any]:
        requests.append(expected)
        if len(requests) == 1:
            return {
                "status": "FAILED",
                "expected_count": 3,
                "refreshed_count": 1,
                "blocker_count": 2,
                "blockers": silent(GONE, SLOW),
            }
        return {"status": "PASSED", "expected_count": 2, "refreshed_count": 2}

    result = ABSORPTION.settle_departed_barrier_nodes(
        site.release,
        {CLUSTER: [SLOW, LIVE, GONE]},
        barrier,
        sleep=site.clock.sleep,
        monotonic=site.clock.monotonic,
    )

    assert result == {
        "status": "PASSED",
        "expected_count": 2,
        "refreshed_count": 2,
        "departed_excluded": {CLUSTER: [GONE]},
    }
    assert requests == [{CLUSTER: [GONE, LIVE, SLOW]}, {CLUSTER: [LIVE, SLOW]}], (
        "the re-run leaves out only the node proven departed"
    )
    assert site.clock.sleeps == [ABSORPTION.DEPARTURE_RECHECK_SECONDS]
    assert site.proofs == [[GONE, SLOW]]
    assert site.recorded() == [GONE]
    row = site.release.state["departed_nodes"][CLUSTER]["nodes"][0]
    assert row["excluded_at"] == ABSORPTION.BARRIER_PHASE
    assert ABSORPTION.barrier_failure_details(result)["departed_excluded"] == {
        CLUSTER: [GONE]
    }


# --- the fleet wave gate ----------------------------------------------------------


def workflow_error(
    snapshot: dict[str, Any], remaining: list[str], count: int | None = None
) -> WaveWorkflowSafetyError:
    return WaveWorkflowSafetyError(
        "active destructive workflows block the wave",
        snapshot=snapshot,
        remaining=remaining,
        remaining_count=len(remaining) if count is None else count,
    )


def young(request_id: str, *nodes: str, cluster: str = CLUSTER) -> dict[str, Any]:
    return {"request_id": request_id, "cluster_id": cluster, "node_ids": list(nodes)}


def test_a_disabled_rollout_reraises_the_workflow_gate_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = Site(monkeypatch)
    departures = site.departures("rollback")
    refusal = workflow_error({}, ["request-1"])
    calls: list[tuple[str, ...]] = []

    def gate(*_args: Any, node_names: tuple[str, ...], **_kwargs: Any) -> Any:
        calls.append(node_names)
        raise refusal

    with pytest.raises(WaveWorkflowSafetyError) as raised:
        departures.wave_safety(gate, wave=(LIVE,), safety_node_names={GONE, LIVE})

    assert raised.value is refusal
    assert calls == [(GONE, LIVE)], "a rollback asks the gate once, as before"
    assert departures.wait_options() == {} and departures.departed_ids == frozenset()
    assert departures.settle_leaving({GONE}) == frozenset()
    assert site.proofs == []


@pytest.mark.parametrize(
    ("blockers", "count"),
    [
        (silent(GONE), 2),
        (silent(GONE) + silent(LIVE, reason="pin-not-aligned"), 2),
        (silent(""), 1),
    ],
    ids=["more-blockers-than-named", "an-alive-agent", "unnamed-node"],
)
def test_a_wave_gate_refusal_outside_the_departure_rule_is_reraised(
    monkeypatch: pytest.MonkeyPatch, blockers: list[dict[str, Any]], count: int
) -> None:
    site = Site(monkeypatch)
    refusal = WaveAgentSafetyError(
        "unhealthy Agents outside the wave", blockers=blockers, blocker_count=count
    )

    def gate(*_args: Any, **_kwargs: Any) -> Any:
        raise refusal

    with pytest.raises(WaveAgentSafetyError) as raised:
        site.departures().wave_safety(
            gate, wave=(LIVE,), safety_node_names={GONE, LIVE}
        )

    assert raised.value is refusal
    assert site.proofs == [], "the proof is never read for such a refusal"


def test_a_silent_agent_on_a_ready_node_is_the_gates_own_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = Site(monkeypatch, listed={LIVE: node(LIVE), GONE: node(GONE)})
    refusal = WaveAgentSafetyError(
        "unhealthy Agents outside the wave", blockers=silent(GONE), blocker_count=1
    )

    def gate(*_args: Any, **_kwargs: Any) -> Any:
        raise refusal

    with pytest.raises(WaveAgentSafetyError) as raised:
        site.departures().wave_safety(gate, wave=(LIVE,), safety_node_names={GONE})

    assert raised.value is refusal
    assert site.proofs == [], "a Ready Node is not a departure, whatever the Agent says"


def test_an_absence_unproven_past_the_decision_window_refuses_the_wave_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = Site(monkeypatch, provable=())
    calls = 0

    def gate(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        site.clock.sleep(120)  # the gate spends its own window
        raise WaveAgentSafetyError(
            "unhealthy Agents outside the wave", blockers=silent(GONE), blocker_count=1
        )

    with pytest.raises(ClusterLocalReleaseError) as raised:
        site.departures().wave_safety(
            gate, wave=(LIVE,), safety_node_names={GONE, LIVE}
        )

    message = str(raised.value)
    assert message.startswith("unhealthy Agents outside the wave; "), message
    assert (
        f"{CLUSTER} fleet wave gate names nodes missing from Kubernetes that are "
        f"not provably departed: {GONE}: {UNPROVEN}"
    ) in message
    assert isinstance(raised.value.__cause__, WaveAgentSafetyError), (
        "the refusal chains the gate's own error"
    )
    assert calls == 6, "asked again until the 600 s decision window ran out"
    assert site.proofs == [[GONE]] * 6, "re-proved on every look past the recheck"
    assert "departed_nodes" not in site.release.state


def test_a_proven_departure_leaves_the_safety_set_and_the_gate_is_asked_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = Site(monkeypatch)
    asked: list[tuple[str, ...]] = []

    def gate(*_args: Any, node_names: tuple[str, ...], **_kwargs: Any) -> Any:
        asked.append(node_names)
        if GONE in node_names:
            raise WaveAgentSafetyError(
                "unhealthy Agents", blockers=silent(GONE), blocker_count=1
            )
        return {"status": "SAFE"}

    departures = site.departures()
    safety = {GONE, LIVE}

    assert departures.wave_safety(gate, wave=(LIVE,), safety_node_names=safety) == {
        "status": "SAFE"
    }
    assert asked == [(GONE, LIVE), (LIVE,)]
    assert safety == {LIVE}
    assert departures.departed_ids == frozenset({GONE})
    assert site.recorded() == [GONE]


@pytest.mark.parametrize(
    ("snapshot", "remaining", "count"),
    [
        ({}, [], 0),
        ({ABSORPTION.YOUNG_NEVER_DISPATCHED_KEY: [young("r-1", GONE)]}, ["r-1"], 2),
        ({ABSORPTION.YOUNG_NEVER_DISPATCHED_KEY: [young("r-1", GONE)]}, ["r-2"], 1),
        (
            {
                ABSORPTION.YOUNG_NEVER_DISPATCHED_KEY: [
                    young("r-1", GONE, cluster="gpu-b")
                ]
            },
            ["r-1"],
            1,
        ),
        ({ABSORPTION.YOUNG_NEVER_DISPATCHED_KEY: [young("r-1")]}, ["r-1"], 1),
    ],
    ids=[
        "nothing-remaining",
        "more-remaining-than-named",
        "not-a-young-record",
        "another-cluster",
        "no-node",
    ],
)
def test_a_workflow_refusal_that_is_not_a_young_departed_record_is_reraised(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: dict[str, Any],
    remaining: list[str],
    count: int,
) -> None:
    site = Site(monkeypatch)
    refusal = workflow_error(snapshot, remaining, count)

    def gate(*_args: Any, **_kwargs: Any) -> Any:
        raise refusal

    with pytest.raises(WaveWorkflowSafetyError) as raised:
        site.departures().wave_safety(gate, wave=(LIVE,), safety_node_names={LIVE})

    assert raised.value is refusal
    assert site.proofs == [] and site.clock.sleeps == [], "nothing was held"


def test_a_young_record_of_a_ready_node_is_the_gates_own_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = Site(monkeypatch, listed={LIVE: node(LIVE), GONE: node(GONE)})
    refusal = workflow_error(
        {ABSORPTION.YOUNG_NEVER_DISPATCHED_KEY: [young("r-1", GONE)]}, ["r-1"]
    )

    def gate(*_args: Any, **_kwargs: Any) -> Any:
        raise refusal

    with pytest.raises(WaveWorkflowSafetyError) as raised:
        site.departures().wave_safety(gate, wave=(LIVE,), safety_node_names={LIVE})

    assert raised.value is refusal
    assert site.clock.sleeps == [], "a record of a live node is never waited for"


def test_a_young_record_of_an_unprovable_absence_is_refused_after_the_window(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site = Site(monkeypatch, provable=())
    refusal = workflow_error(
        {ABSORPTION.YOUNG_NEVER_DISPATCHED_KEY: [young("r-1", GONE)]}, ["r-1"]
    )

    def gate(*_args: Any, **_kwargs: Any) -> Any:
        raise refusal

    with pytest.raises(ClusterLocalReleaseError) as raised:
        site.departures().wave_safety(gate, wave=(LIVE,), safety_node_names={LIVE})

    assert raised.value.__cause__ is refusal
    assert "fleet wave gate names nodes missing from Kubernetes" in str(raised.value)
    assert f"{GONE}: {UNPROVEN}" in str(raised.value)
    window = ABSORPTION.DEPARTURE_DECISION_SECONDS
    assert sum(site.clock.sleeps) == window, "held one recheck at a time, no longer"
    assert "departed-node-workflows-held" in capsys.readouterr().err


def test_a_departed_nodes_record_that_never_ages_past_the_guard_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site = Site(monkeypatch)
    refusal = workflow_error(
        {ABSORPTION.YOUNG_NEVER_DISPATCHED_KEY: [young("r-1", GONE)]}, ["r-1"]
    )
    gates = 0

    def gate(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal gates
        gates += 1
        raise refusal

    with pytest.raises(ClusterLocalReleaseError) as raised:
        site.departures().wave_safety(gate, wave=(LIVE,), safety_node_names={LIVE})

    message = str(raised.value)
    assert (
        f"the never-dispatched records of the departed nodes {GONE} did not pass "
        "the product's dispatch guard within 600s"
    ) in message, message
    assert raised.value.__cause__ is refusal
    guard = ABSORPTION.NEVER_DISPATCHED_GUARD_SECONDS
    assert sum(site.clock.sleeps) > guard, "the hold outlasted the dispatch guard"
    assert gates == len(site.clock.sleeps) + 1
    assert site.recorded() == [GONE], "the departure itself stayed proven"
    held = capsys.readouterr().err
    assert held.count("departed-node-workflows-held") == 1, "narrated once"
    assert "workflows=r-1" in held and f"nodes={GONE}" in held


# --- the fleet deployment record -------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [{"deployment": {"status": "IN_PROGRESS"}, "excluded": []}, "accepted"],
    ids=["node-not-accounted", "no-deployment"],
)
def test_a_record_that_does_not_account_for_every_dropped_node_pauses_the_cluster(
    monkeypatch: pytest.MonkeyPatch, answer: Any
) -> None:
    site = Site(monkeypatch)
    departures = site.departures()
    assert departures(frozenset({GONE})) == frozenset({GONE})
    site.release.fleet_answer = answer
    deployment = {
        "nodes": [
            {"node_id": GONE, "status": "INSTALLING"},
            {"node_id": LIVE, "status": "READY"},
        ]
    }

    with pytest.raises(ClusterLocalReleaseError) as raised:
        departures.reconcile_record(deployment, "deploy-1")

    assert (
        f"{CLUSTER} fleet deployment deploy-1 did not exclude the departed nodes "
        f"['{GONE}']"
    ) in str(raised.value)
    assert site.release.fleet_commands == [
        (
            ABSORPTION.EXCLUDE_DEPARTED_OPERATION,
            {"deployment_id": "deploy-1", "node_ids": [GONE]},
        )
    ]
    row = site.release.state["departed_nodes"][CLUSTER]["nodes"][0]
    assert "fleet_deployment" not in row, "nothing is recorded for a refused exclusion"
    assert isinstance(raised.value, ReleaseError), (
        "a cluster-local pause, never a bare exception"
    )
