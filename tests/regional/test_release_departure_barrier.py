"""The candidate CPU heartbeat barrier over a node that left after the capture.

The release records its Agent node set before it mutates anything and the
in-Pod barrier (``agent_heartbeat_barrier``) wants a refreshed heartbeat from
every node of it. A node the provider reclaimed in between never heartbeats
again: the barrier timed out and raised a plain ``ReleaseError``, which the
upgrade answers with its automatic rollback. The barrier now absorbs a node the
shared departure proof (``regional_release_departed_nodes``) shows gone, records
the evidence, waits a bounded time for a proof that is not decidable yet, and
still fails closed on an absence it cannot prove.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from gpu_fault_release import regional_release_departed_nodes as DEPARTED
from gpu_fault_release import regional_release_departure_absorption as ABSORPTION
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._cov95_release_support import Clock, ResourceRelease

LIVE = ("hyperpod-i-0000000000a1", "hyperpod-i-0000000000a2")
GONE = "hyperpod-i-0000000000a3"
GONE_INSTANCE = GONE.removeprefix("hyperpod-")
PAST = "2026-10-31T09:00:00+00:00"
EXPECTED = {"gpu-a": {"node_ids": [*LIVE, GONE]}}


@dataclass
class Reclaim:
    """How the reclaimed node looks to each source, by seconds into the barrier.

    ``None`` is "never": the Node stays NotReady, the instance stays listed,
    the lease stays live for the whole case.
    """

    node_gone_at: float | None = 0.0
    delisted_at: float | None = 0.0
    lease_lapses_at: float | None = 0.0
    #: Barrier blocker reason per node; the others refresh.
    blocked: dict[str, str] = field(
        default_factory=lambda: {GONE: "heartbeat-not-refreshed"}
    )


class Site:
    """The release double, the in-Pod barrier, the Agent facts and HyperPod."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, reclaim: Reclaim) -> None:
        self.reclaim = reclaim
        self.clock = Clock()
        self.release = ResourceRelease(("gpu-a",))
        self.release.runner.handler = self.aws
        self.barriers: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []
        self.inventory_reads = 0
        self.refresh()
        monkeypatch.setattr(FLEET, "exec_cpu_ingress_probe", self.barrier)
        monkeypatch.setattr(DEPARTED, "exec_cpu_ingress_probe", self.lifecycle)
        monkeypatch.setattr(ABSORPTION, "time", self)
        monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)

    # The engine's clock: the Node list follows the reclaim as time passes.
    def monotonic(self) -> float:
        return self.clock.monotonic()

    def sleep(self, seconds: float) -> None:
        self.clock.sleep(seconds)
        self.refresh()

    def at(self, moment: float | None) -> bool:
        return moment is not None and self.clock.elapsed >= moment

    def refresh(self) -> None:
        names = [*LIVE] if self.at(self.reclaim.node_gone_at) else [*LIVE, GONE]
        self.release.documents[("gpu-a", "nodes", "")] = {
            "items": [
                {
                    "metadata": {
                        "name": name,
                        "uid": f"{name}-uid",
                        "labels": {"sagemaker.amazonaws.com/cluster-name": "hp-gpu-a"},
                    },
                    "status": {
                        "conditions": [
                            {
                                "type": "Ready",
                                "status": "False" if name == GONE else "True",
                            }
                        ]
                    },
                }
                for name in names
            ]
        }

    def aws(self, arguments: list[str], _kwargs: dict[str, Any]) -> str:
        assert arguments[:3] == ["aws", "sagemaker", "list-cluster-nodes"], arguments
        self.inventory_reads += 1
        listed = [*(name[9:] for name in LIVE)]
        if not self.at(self.reclaim.delisted_at):
            listed.append(GONE_INSTANCE)
        return json.dumps(
            {"ClusterNodeSummaries": [{"InstanceId": item} for item in listed]}
        )

    def barrier(self, _release: Any, **options: Any) -> str:
        request = json.loads(options["input_text"])
        self.barriers.append(request)
        expected = sorted(
            node for nodes in request["expected_by_cluster"].values() for node in nodes
        )
        blocked = {
            node: reason
            for node, reason in self.reclaim.blocked.items()
            if node in expected
        }
        # The in-Pod barrier spends its own window before it answers.
        self.clock.sleep(request["timeout_seconds"] if blocked else 1)
        self.refresh()
        return json.dumps(
            {
                "status": "FAILED" if blocked else "PASSED",
                "expected_count": len(expected),
                "refreshed_count": len(expected) - len(blocked),
                **(
                    {
                        "blocker_count": len(blocked),
                        "blockers": [
                            {"cluster_id": "gpu-a", "node_id": node, "reason": reason}
                            for node, reason in sorted(blocked.items())
                        ],
                    }
                    if blocked
                    else {}
                ),
            }
        )

    def lifecycle(self, _release: Any, **options: Any) -> str:
        request = json.loads(options["input_text"])
        self.facts.append(request)
        live = not self.at(self.reclaim.lease_lapses_at)
        return json.dumps(
            {
                "read_at": PAST,
                "agents": {
                    GONE: {
                        "lifecycle_state": "ACTIVE",
                        "lease_expires_at": PAST,
                        "last_seen_at": PAST,
                        "lease_live": live,
                        "node_instance_id": GONE_INSTANCE,
                    }
                },
            }
        )


def wait(site: Site, **options: Any) -> None:
    FLEET.wait_candidate_cpu_agent_heartbeats(site.release, EXPECTED, **options)


def test_a_node_reclaimed_after_the_capture_is_excluded_on_the_proof(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Before: ``candidate CPU did not observe refreshed Agent heartbeats`` --
    a global failure, the trigger of the automatic rollback."""

    site = Site(monkeypatch, Reclaim())

    wait(site)

    assert site.facts == [{"cluster_id": "gpu-a", "node_ids": [GONE]}]
    assert len(site.barriers) == 1, (
        "the run that named the departed node already proved every other node"
    )
    entry = site.release.state["departed_nodes"]["gpu-a"]
    assert (entry["release_id"], entry["phase"]) == (
        site.release.release_id,
        "cpu-heartbeat-barrier",
    )
    (row,) = entry["nodes"]
    assert (row["node_id"], row["provider"], row["instance_id"]) == (
        GONE,
        "hyperpod",
        GONE_INSTANCE,
    )
    assert row["excluded_at"] == "cpu-heartbeat-barrier"
    assert row["provider_inventory"]["listed"] is False
    assert row["agent"]["lifecycle_state"] == "ACTIVE"
    assert f"--retire-departed-agents --node {GONE}" in row["retire_hint"]
    log = capsys.readouterr().err
    assert (
        "departed-node-excluded" in log
        and f"phase=cpu-heartbeat-barrier nodes={GONE}" in log
    ), log


def test_a_reclaim_not_yet_decidable_is_waited_for_within_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Node lingers NotReady, HyperPod still lists the instance and the
    lease runs on: the barrier re-runs over the same set until the proof
    holds, then passes with the node set aside."""

    site = Site(
        monkeypatch, Reclaim(node_gone_at=150, delisted_at=200, lease_lapses_at=180)
    )

    wait(site)

    assert site.release.state["departed_nodes"]["gpu-a"]["nodes"][0]["node_id"] == GONE
    assert len(site.barriers) >= 3, site.barriers
    assert all(
        request["expected_by_cluster"] == {"gpu-a": sorted([*LIVE, GONE])}
        for request in site.barriers
    ), "the node stays expected until it is proven gone"
    assert site.clock.elapsed < ABSORPTION.DEPARTURE_DECISION_SECONDS + 200


@pytest.mark.parametrize(
    "reclaim,reason",
    [
        (
            Reclaim(delisted_at=None),
            f"{GONE}: HyperPod cluster hp-gpu-a still lists instance {GONE_INSTANCE}",
        ),
        (
            Reclaim(lease_lapses_at=None),
            f"{GONE}: Agent record is ACTIVE with a live lease",
        ),
        (
            Reclaim(node_gone_at=None),
            f"{GONE}: the Node is still in Kubernetes (NotReady)",
        ),
    ],
)
def test_an_absence_it_cannot_prove_fails_closed_with_the_reason(
    monkeypatch: pytest.MonkeyPatch, reclaim: Reclaim, reason: str
) -> None:
    site = Site(monkeypatch, reclaim)

    with pytest.raises(ReleaseError) as failure:
        wait(site)

    message = str(failure.value)
    assert message.startswith(
        "candidate CPU did not observe refreshed Agent heartbeats: "
    ), message
    details = json.loads(message.split(": ", 1)[1])
    assert details["departure_unproven"].startswith(
        "gpu-a heartbeat barrier names nodes missing from Kubernetes that are not "
        "provably departed: "
    ), details
    assert reason in details["departure_unproven"], details
    assert details["blockers"][0]["node_id"] == GONE
    assert "departed_nodes" not in site.release.state
    assert site.clock.elapsed >= ABSORPTION.DEPARTURE_DECISION_SECONDS


def test_a_silent_agent_on_a_ready_node_fails_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a departure: no proof is read and nothing waits."""

    site = Site(monkeypatch, Reclaim(blocked={LIVE[0]: "heartbeat-not-refreshed"}))

    with pytest.raises(ReleaseError, match="did not observe refreshed") as failure:
        wait(site)

    assert "departure_unproven" not in str(failure.value)
    assert (site.facts, site.inventory_reads, len(site.barriers)) == ([], 0, 1)


@pytest.mark.parametrize(
    "blocked",
    [
        {GONE: "heartbeat-not-refreshed", LIVE[0]: "pin-not-aligned"},
        {GONE: "pin-not-aligned"},
    ],
)
def test_a_live_agent_among_the_blockers_fails_as_before(
    monkeypatch: pytest.MonkeyPatch, blocked: dict[str, str]
) -> None:
    site = Site(monkeypatch, Reclaim(blocked=blocked))

    with pytest.raises(ReleaseError, match="window stays open"):
        wait(
            site,
            required_identity={
                "artifact_sha256": "a" * 64,
                "compatibility_digest": "b" * 64,
                "agent_protocol_version": "3",
                "config_digest": "c" * 64,
            },
        )

    assert site.facts == [], "an Agent that reports a pin is alive"


def test_a_barrier_that_cannot_name_every_blocker_fails_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = Site(monkeypatch, Reclaim())
    original = site.barrier

    def capped(release: Any, **options: Any) -> str:
        result = json.loads(original(release, **options))
        result["blocker_count"] = 101
        return json.dumps(result)

    monkeypatch.setattr(FLEET, "exec_cpu_ingress_probe", capped)

    with pytest.raises(ReleaseError, match="did not observe refreshed"):
        wait(site)

    assert site.facts == []


def test_a_node_this_release_already_proved_gone_is_not_waited_for(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fleet waves proved it; the CPU finalize barrier does not ask again."""

    site = Site(monkeypatch, Reclaim())
    DEPARTED.record_departed_nodes(
        site.release,
        site.release.config.clusters[0],
        [{"node_id": GONE, "agent": None}],
        phase="fleet-wave",
        narrate=False,
    )

    wait(site)

    assert site.barriers[0]["expected_by_cluster"] == {"gpu-a": sorted(LIVE)}
    assert site.facts == [], "no second proof"
    assert "departed-node-already-excluded" in capsys.readouterr().err


def test_a_recorded_departure_whose_name_is_back_is_expected_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = Site(monkeypatch, Reclaim(node_gone_at=None, blocked={}))
    DEPARTED.record_departed_nodes(
        site.release,
        site.release.config.clusters[0],
        [{"node_id": GONE, "agent": None}],
        phase="fleet-wave",
        narrate=False,
    )

    wait(site)

    assert site.barriers[0]["expected_by_cluster"] == {"gpu-a": sorted([*LIVE, GONE])}
