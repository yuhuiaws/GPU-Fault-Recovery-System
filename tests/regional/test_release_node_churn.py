"""A node that leaves or joins while a release rolls its cluster.

The fleet deployment's node list is fixed when the cluster's roll starts and
the release's Agent node set when the release starts, so capacity that changes
in between used to end the release: a reclaimed node never converged in its
wave (900 s, then a paused cluster) or blocked the next wave's safety gate,
and a node that joined was waited for in the steady-state wait it could not
finish, or failed the pre-mutation gate, or the verify. These cases drive the
real wave loop, safety gate, convergence wait and departure proof over a
simulated HyperPod cluster (``_node_churn_site``).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_agent_convergence as CONVERGENCE
from gpu_fault_release import regional_release_departed_nodes as DEPARTED
from gpu_fault_release import regional_release_departure_absorption as ABSORPTION
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_joined_nodes as JOINED
from gpu_fault_release import regional_release_node_runtime_rollout as RUNTIME
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional import test_release_node_image_targets as IMAGE_TARGETS
from tests.regional._node_churn_site import (
    HYPERPOD_CLUSTER,
    OLD_TEMPLATE,
    ChurnSite,
    SiteClock,
    node_name,
)
from tests.regional.test_release_node_config_waves import BUNDLE_FILE
from tests.regional.test_release_node_image_targets import ARTIFACT, CONFIG

node_release = IMAGE_TARGETS.node_release
#: Taken before any fixture patches it: these cases run the real gate.
REAL_GATE = FLEET.ensure_rollout_wave_safe
NODES = [node_name(index) for index in range(1, 5)]
NEWCOMER = node_name(5)


@pytest.fixture
def churn(node_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The node image target release, wired to a ``ChurnSite``."""

    release = node_release.release
    release.config.upgrade_max_unavailable = 0
    release.config.bundle = Path(BUNDLE_FILE)
    target = SimpleNamespace(
        cluster_id="gpu-a",
        context="gpu-a-context",
        hyperpod_cluster_name=HYPERPOD_CLUSTER,
        fleet_master_file=None,
        region="us-west-2",
    )
    release.config.clusters = (target,)
    node_release.previous["agent_identities"]["gpu-a"]["node_ids"] = sorted(NODES)
    site = ChurnSite(target, NODES, template=OLD_TEMPLATE)
    clock = SiteClock(site)

    def run(arguments: list[str], *, env: dict[str, str] | None = None, **_: Any):
        if "patch" in arguments:
            return ""
        if arguments[:1] == ["aws"]:
            return site.aws(arguments)
        assert Path(arguments[0]).name == "deploy-node-installer-reconciler.sh", (
            f"unexpected command {arguments}"
        )
        assert env is not None, "the reconciler script runs with an environment"
        node_release.environments.append(dict(env))
        site.deploy(env)
        return ""

    def exec_probe(_release: Any, *, script: str, **options: Any) -> str:
        raw = options.get("input_text")
        return site.answer(script, json.loads(raw) if raw else None)

    monkeypatch.setattr(release.runner, "run", run)
    for name, value in {
        "_get_json": lambda arguments: site.get_json(
            arguments, node_release.template_document
        ),
        "_fleet_deployment_id": lambda target, **options: FLEET.fleet_deployment_id(
            release, target, **options
        ),
        "_target_node_names": lambda target: FLEET.target_node_names(release, target),
        "_fleet_command": site.fleet_command,
        "_wait_agents": lambda target, artifact, **options: CONVERGENCE.wait_agents(
            release, target, artifact, **options
        ),
        "_agent_heartbeats_converged": lambda target, **options: (
            CONVERGENCE.agent_heartbeats_converged(release, target, **options)
        ),
    }.items():
        monkeypatch.setattr(release, name, value, raising=False)
    for module in (FLEET, RUNTIME):
        monkeypatch.setattr(module, "ensure_rollout_wave_safe", REAL_GATE)
    monkeypatch.setattr(RUNTIME, "product_isolated_nodes", lambda *_args: [])
    for module in (FLEET, CONVERGENCE, DEPARTED):
        monkeypatch.setattr(module, "exec_cpu_ingress_probe", exec_probe)
    for module in (FLEET, CONVERGENCE, ABSORPTION, JOINED):
        monkeypatch.setattr(module, "time", clock)
    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    return SimpleNamespace(release=release, target=target, site=site, clock=clock)


def roll(churn: Any) -> tuple[str, str]:
    return RUNTIME.roll_node_runtime(
        churn.release,
        churn.target,
        phase="upgrade",
        wheel_cm="candidate-wheel",
        bundle_cm=churn.release.bundle_cm,
        artifact_sha=ARTIFACT,
        config_digest=CONFIG,
        candidate_preflight_completed=True,
    )


def deployed_node_ids(churn: Any) -> list[str]:
    return sorted(item.node_id for item in churn.site.deployment().nodes)


def wave_waits(churn: Any) -> list[str]:
    """The allowed-node sets the release handed the reconciler, in order."""

    return [item.allowed for item in churn.site.deploys]


def departed_row(churn: Any, node: str) -> dict[str, Any]:
    entry = churn.release.state["departed_nodes"]["gpu-a"]
    assert entry["release_id"] == churn.release.release_id
    return next(row for row in entry["nodes"] if row["node_id"] == node)


def joined_rows(churn: Any) -> dict[str, dict[str, Any]]:
    entry = churn.release.state["joined_nodes"]["gpu-a"]
    assert entry["release_id"] == churn.release.release_id
    return {row["node_id"]: row for row in entry["nodes"]}


def test_without_churn_every_node_rolls_through_the_waves(churn: Any) -> None:
    roll(churn)

    deployment = churn.site.deployment()
    assert deployment.status.value == "SUCCEEDED", deployment
    assert deployed_node_ids(churn) == sorted(NODES)
    assert wave_waits(churn) == ["", NODES[0], ",".join(sorted(NODES[1:])), "*"], (
        "paused, the canary, the rest, then the steady state"
    )
    assert "departed_nodes" not in churn.release.state
    assert "joined_nodes" not in churn.release.state
    assert "exclude-departed" not in churn.site.operations


# -- a node that leaves -------------------------------------------------------


def test_a_node_reclaimed_inside_its_wave_is_excluded_on_the_proof(
    churn: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Wave 2 holds three nodes; the provider takes one back as the wave starts.

    Before: the two others converged, the wave wait returned, the record kept
    the reclaimed node INSTALLING, the loop leased it again alone and the wait
    for a node Kubernetes no longer has ran out its 900 s -- a paused cluster
    and a fleet deployment left IN_PROGRESS, fencing destructive remediation.
    """

    gone = NODES[2]
    wave_two = ",".join(sorted(NODES[1:]))
    churn.site.hooks.append((f"deploy:{wave_two}", lambda: churn.site.reclaim(gone)))

    roll(churn)

    deployment = churn.site.deployment()
    assert deployment.status.value == "SUCCEEDED", deployment
    assert deployed_node_ids(churn) == sorted(set(NODES) - {gone}), (
        "the departed node left the record, not FAILED in it"
    )
    assert all(gone not in wave for wave in deployment.waves), deployment.waves
    assert all(item.status.value == "READY" for item in deployment.nodes), (
        deployment.nodes
    )
    assert wave_waits(churn).count(gone) == 0, "the departed node was never re-leased"
    row = departed_row(churn, gone)
    assert (row["provider"], row["instance_id"]) == ("hyperpod", gone[9:])
    assert row["excluded_at"] == "fleet-wave"
    assert row["kubernetes"]["present"] is False
    assert row["provider_inventory"]["listed"] is False
    assert row["agent"]["lease_live"] is False
    assert row["fleet_deployment"] == {
        "deployment_id": deployment.deployment_id,
        "status_before": "INSTALLING",
    }
    # The proof needs the Agent lease to lapse (90 s after its last heartbeat).
    assert 90 <= churn.clock.elapsed < 200, churn.clock.elapsed
    log = capsys.readouterr().err
    assert log.count("departed-node-excluded") == 1, log
    assert f"phase=fleet-wave nodes={gone}" in log
    assert "fleet-deployment-node-excluded" in log, log
    assert "status=SUCCEEDED" in log, log


def test_a_canary_reclaimed_before_its_install_keeps_a_one_node_canary(
    churn: Any,
) -> None:
    gone = NODES[0]
    churn.site.hooks.append((f"deploy:{gone}", lambda: churn.site.reclaim(gone)))

    roll(churn)

    assert churn.site.deployment().status.value == "SUCCEEDED"
    assert gone not in deployed_node_ids(churn)
    assert churn.clock.elapsed < 600, (
        f"the release waited {churn.clock.elapsed:.0f}s for a node that left"
    )
    assert wave_waits(churn) == [
        "",
        gone,
        NODES[1],
        ",".join(sorted(NODES[2:])),
        "*",
    ], "with nothing converged yet, the next node goes alone first"
    assert departed_row(churn, gone)["fleet_deployment"]["status_before"] == (
        "INSTALLING"
    )


def test_a_rolled_node_reclaimed_before_the_next_wave_stops_blocking_its_gate(
    churn: Any,
) -> None:
    """The canary converged, then left: wave 2's gate sees its lease lapse.

    Before: ``fleet wave Agent safety did not converge`` after the gate's 60 s
    -- a paused cluster.
    """

    gone = NODES[0]

    def reclaim_then_wait() -> None:
        # After wave 1 converged, a minute before wave 2's gate reads the
        # leases: the silent Agent no longer has the gate's 40 s margin.
        churn.site.reclaim(gone)
        churn.clock.sleep(55)

    churn.site.hooks.append(("get#1", reclaim_then_wait))

    roll(churn)

    deployment = churn.site.deployment()
    assert deployment.status.value == "SUCCEEDED", deployment
    statuses = {item.node_id: item.status.value for item in deployment.nodes}
    assert statuses.get(gone) == "READY", (
        "a node that converged before it left stays READY"
    )
    row = departed_row(churn, gone)
    assert row["excluded_at"] == "fleet-wave"
    assert "fleet_deployment" not in row, "a READY node is not taken out"


def test_a_reclaim_the_proof_cannot_decide_still_fails_closed(churn: Any) -> None:
    """HyperPod keeps listing the instance: the absence is not proof."""

    gone = NODES[2]
    wave_two = ",".join(sorted(NODES[1:]))

    def reclaim_but_listed() -> None:
        churn.site.reclaim(gone)
        churn.site.listed.add(gone.removeprefix("hyperpod-"))

    churn.site.hooks.append((f"deploy:{wave_two}", reclaim_but_listed))

    with pytest.raises(ReleaseError, match="not provably departed") as failure:
        roll(churn)

    assert str(failure.value) == (
        f"gpu-a fleet wave names nodes missing from Kubernetes that are not "
        f"provably departed: {gone}: HyperPod cluster {HYPERPOD_CLUSTER} still "
        f"lists instance {gone[9:]}"
    )
    assert ABSORPTION.DEPARTURE_DECISION_SECONDS <= churn.clock.elapsed < 700
    assert "departed_nodes" not in churn.release.state
    statuses = {
        item.node_id: item.status.value for item in churn.site.deployment().nodes
    }
    assert statuses[gone] == "INSTALLING", "an unproven node is never taken out"


# -- a node that joins ------------------------------------------------------


def test_a_node_that_joins_mid_rollout_is_installed_at_finalize(
    churn: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    churn.site.hooks.append(
        (f"deploy:{NODES[0]}", lambda: churn.site.join(NEWCOMER, installed=None))
    )

    roll(churn)

    assert churn.site.deployment().status.value == "SUCCEEDED"
    assert NEWCOMER not in deployed_node_ids(churn), "a newcomer is in no wave"
    assert all(NEWCOMER not in item for item in wave_waits(churn)[:-1]), (
        "no wave handed the newcomer to the reconciler"
    )
    annotations = churn.site.nodes.items[NEWCOMER]["metadata"]["annotations"]
    assert annotations["gpu-fault.io/installer-state"] == "Succeeded"
    row = joined_rows(churn)[NEWCOMER]
    assert (row["outcome"], row["installer_state"]) == ("installed", "Succeeded")
    assert row["joined_at"], row
    assert "joined-nodes" in capsys.readouterr().err


def test_a_newcomer_slower_than_the_grace_is_reported_not_failed(
    churn: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live, a new node's installer Job starts minutes after it joins.

    Before: the steady-state wait waited for every selected node, the newcomer
    included, and paused the cluster after 900 s.
    """

    def join_slow() -> None:
        churn.site.join(NEWCOMER, installed=None)
        churn.site.stuck.add(NEWCOMER)

    churn.site.hooks.append((f"deploy:{NODES[0]}", join_slow))
    started = churn.clock.elapsed

    roll(churn)

    row = joined_rows(churn)[NEWCOMER]
    assert (row["outcome"], row["installer_state"]) == ("pending", "Installing")
    elapsed = churn.clock.elapsed - started
    assert JOINED.NEWCOMER_GRACE_SECONDS <= elapsed < 400, (
        f"the grace bounds the wait for the newcomer: {elapsed:.0f}s"
    )
    log = capsys.readouterr().err
    assert f"pending={NEWCOMER}:Installing" in log, log


def test_a_newcomer_without_an_agent_at_roll_start_is_deferred(churn: Any) -> None:
    """Joined after the release captured its node set, before this cluster rolled.

    Before: in the fleet deployment and the pre-mutation gate, whose Agent
    check found no Agent for it and paused the cluster after 60 s.
    """

    churn.site.join(NEWCOMER, installed=None)

    roll(churn)

    assert NEWCOMER not in deployed_node_ids(churn)
    annotations = churn.site.nodes.items[NEWCOMER]["metadata"]["annotations"]
    assert annotations["gpu-fault.io/installer-state"] == "Succeeded"
    assert joined_rows(churn)[NEWCOMER]["outcome"] == "installed"
    assert churn.release.state["joined_nodes"]["gpu-a"]["phase"] == "finalize"


def test_a_newcomer_mid_install_at_roll_start_does_not_trigger_a_rollback(
    churn: Any,
) -> None:
    """Before: ``installer-active`` in the pre-mutation gate -- a global
    ReleaseError, which the upgrade answers with its automatic rollback."""

    churn.site.join(NEWCOMER, installed=None)
    churn.site.nodes.items[NEWCOMER]["metadata"]["annotations"][
        "gpu-fault.io/installer-state"
    ] = "Installing"

    roll(churn)

    assert NEWCOMER not in deployed_node_ids(churn)
    assert joined_rows(churn)[NEWCOMER]["outcome"] == "installed"


def test_a_newcomer_installed_before_the_roll_is_rolled_with_the_fleet(
    churn: Any,
) -> None:
    """Joined after the capture, installed by the previous reconciler before
    this cluster rolled: an Agent on the old identity, capacity like any node."""

    churn.site.join(NEWCOMER, installed=OLD_TEMPLATE)

    roll(churn)

    assert NEWCOMER in deployed_node_ids(churn), "it is in a wave"
    assert churn.site.deployment().status.value == "SUCCEEDED"
    assert "joined_nodes" not in churn.release.state


def test_a_newcomer_on_the_old_identity_is_held_to_the_planned_wait(churn: Any) -> None:
    """Joined during the waves with the previous release's installation: the
    steady-state Reconciler re-installs it and the release waits for that."""

    churn.site.hooks.append(
        (
            f"deploy:{NODES[0]}",
            lambda: churn.site.join(NEWCOMER, installed=OLD_TEMPLATE),
        )
    )

    roll(churn)

    assert joined_rows(churn)[NEWCOMER]["outcome"] == "installed"
    assert churn.site.agents[NEWCOMER] != OLD_TEMPLATE, "re-installed"


def test_a_newcomer_left_on_the_old_identity_is_caught(churn: Any) -> None:
    """The Reconciler never reaches it: drift, never waved past as pending."""

    def join_old_and_unreached() -> None:
        churn.site.join(NEWCOMER, installed=OLD_TEMPLATE)
        churn.site.held.add(NEWCOMER)

    churn.site.hooks.append((f"deploy:{NODES[0]}", join_old_and_unreached))

    with pytest.raises(ReleaseError, match="agents did not converge"):
        roll(churn)

    assert "joined_nodes" not in churn.release.state


def test_a_newcomer_that_leaves_again_is_recorded_as_left(churn: Any) -> None:
    def join_slow() -> None:
        churn.site.join(NEWCOMER, installed=None)
        churn.site.stuck.add(NEWCOMER)

    churn.site.hooks.append((f"deploy:{NODES[0]}", join_slow))
    churn.site.hooks.append(("deploy:*", lambda: churn.site.reclaim(NEWCOMER)))

    roll(churn)

    row = joined_rows(churn)[NEWCOMER]
    assert (row["outcome"], row["installer_state"]) == ("left", None)


def test_departures_are_inert_outside_the_forward_upgrade() -> None:
    """Rollback, join and bootstrap keep their behaviour exactly."""

    def gate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ABSORPTION.WaveAgentSafetyError(
            "gpu-a fleet wave Agent safety did not converge: {}",
            blockers=[{"node_id": "n", "reason": "lease-margin"}],
            blocker_count=1,
        )

    for phase in ("rollback", "join", "bootstrap"):
        departures = ABSORPTION.WaveDepartures.for_phase(
            SimpleNamespace(), SimpleNamespace(cluster_id="gpu-a"), phase
        )
        assert departures.wait_options() == {}
        assert departures.departed_ids == frozenset()
        deployment = {"nodes": [{"node_id": "n", "status": "INSTALLING"}]}
        assert departures.reconcile_record(deployment, "d") is deployment
        with pytest.raises(ABSORPTION.WaveAgentSafetyError):
            departures.wave_safety(gate, wave=(), safety_node_names={"n"})
