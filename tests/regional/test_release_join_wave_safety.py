"""A join's wave safety protects only nodes that already hold a live agent.

And when no node holds one, there is nothing for a canary wave to protect
either: the join installs the whole fleet in one wave.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.failure_domains import UNKNOWN_FAILURE_DOMAIN
from gpu_fault.fleet_deployment import FleetDeploymentRequest, deployment_waves
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_node_runtime_rollout as RUNTIME
from gpu_fault_release import rollout as MODULE
from tests.regional._release_orchestrator_support import config_file

NODES = ("node-a", "node-b", "node-c")
IDENTITY = ("b" * 64, "e" * 64)


def _release(live: dict[str, list[str]] | None) -> Any:
    def probe(_release: Any, **_keywords: Any) -> str:
        return json.dumps(live if live is not None else {})

    return SimpleNamespace(runner=SimpleNamespace(dry_run=False)), probe


def test_live_agent_node_names_keeps_only_active_leased_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, probe = _release({"gpu-a": ["node-b", "node-zzz"], "gpu-x": ["node-a"]})
    monkeypatch.setattr(FLEET, "exec_cpu_ingress_probe", probe)

    live = FLEET.live_agent_node_names(
        release, SimpleNamespace(cluster_id="gpu-a"), NODES
    )

    assert live == ("node-b",), "another cluster's node or an unknown name is not ours"


def test_live_agent_node_names_accepts_a_cluster_with_no_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first join of a cluster has nothing live; that is not an error."""

    release, probe = _release({"gpu-x": ["node-a"]})
    monkeypatch.setattr(FLEET, "exec_cpu_ingress_probe", probe)

    assert (
        FLEET.live_agent_node_names(release, SimpleNamespace(cluster_id="gpu-a"), NODES)
        == ()
    ), "a cluster absent from the inventory holds no live agent"


def _roll(
    tmp_path, monkeypatch: pytest.MonkeyPatch, *, phase: str, live: tuple[str, ...]
) -> tuple[Any, dict[str, Any]]:
    """Run ``roll_node_runtime`` with every seam stubbed; return what it handed on."""

    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    target = config.clusters[0]
    seen: dict[str, Any] = {"deploys": [], "live_probes": 0}
    monkeypatch.setattr(release, "_target_node_names", lambda _target: NODES)
    monkeypatch.setattr(
        RUNTIME,
        "target_node_failure_domains",
        lambda *_args: {name: "zone-a" for name in NODES},
    )
    monkeypatch.setattr(
        RUNTIME, "ensure_pre_node_mutation_barrier", lambda *_args, **_kwargs: None
    )

    def deploy_reconciler(_target: Any, **kwargs: Any) -> tuple[str, str]:
        seen["deploys"].append(kwargs)
        return IDENTITY

    monkeypatch.setattr(release, "_deploy_reconciler", deploy_reconciler)
    monkeypatch.setattr(
        RUNTIME, "finish_legacy_node_runtime_rollback", lambda *_a, **_k: None
    )

    def create(_release: Any, _target: Any, candidate: Any, policy: Any, *_a, **_k):
        seen["candidate"] = candidate
        seen["policy"] = policy
        return "dep-1", {"status": "PLANNED"}

    monkeypatch.setattr(RUNTIME, "_create_fleet_rollout", create)

    def live_names(_release: Any, _target: Any, names: tuple[str, ...]):
        seen["live_probes"] += 1
        return tuple(name for name in names if name in live)

    monkeypatch.setattr(RUNTIME, "live_agent_node_names", live_names)
    monkeypatch.setattr(
        RUNTIME,
        "run_fleet_waves",
        lambda _release, _target, context, _deployment: seen.__setitem__(
            "context", context
        ),
    )

    def finalize(_release: Any, _target: Any, candidate: Any, *_a, **_k):
        seen["finalize_candidate"] = candidate
        return IDENTITY

    monkeypatch.setattr(RUNTIME, "_finalize_node_runtime", finalize)

    RUNTIME.roll_node_runtime(
        release,
        target,
        phase=phase,
        wheel_cm=release.executor_wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.node_wheel_sha,
        config_digest=config.agent_config_digest,
        candidate_preflight_completed=True,
    )
    return release, seen


@pytest.mark.parametrize(
    ("phase", "expected"),
    [("join", ("node-b",)), ("bootstrap", ("node-b",)), ("upgrade", NODES)],
)
def test_the_wave_safety_node_set_depends_on_the_phase(
    tmp_path, monkeypatch: pytest.MonkeyPatch, phase: str, expected: tuple[str, ...]
) -> None:
    """Live 2026-09-12: remove-cluster left agent records whose leases had
    expired hours earlier, and the re-join's first wave blocked on the three
    nodes outside it ("lease-margin"). A join must not require nodes that hold
    no live agent to stay available; an upgrade keeps protecting every node."""

    _unused, seen = _roll(tmp_path, monkeypatch, phase=phase, live=("node-b",))

    assert seen["context"].node_names == expected, (
        "the wave safety gate was given the wrong node set for this phase"
    )


def test_a_join_with_no_live_agent_plans_one_wave_for_the_whole_fleet(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live 2026-09-12 (join, 4 nodes, no failure-domain labels): a one-node
    canary wave and then the rest, each wave ~20s of safety polling and
    hand-off before the install. With no live agent anywhere there is no
    capacity for a wave to take away, so N waves are exactly as safe as one."""

    release, seen = _roll(tmp_path, monkeypatch, phase="join", live=())

    policy = seen["policy"]
    assert policy == FLEET.NodeRolloutPolicy(
        max_unavailable=3,
        first_wave_max_unavailable=3,
        max_unavailable_per_failure_domain=3,
    ), "the fleet record is created with every node in the first wave"
    assert seen["deploys"][0]["max_unavailable"] == 3, (
        "the paused Reconciler is deployed with the same concurrency"
    )
    assert seen["context"].max_unavailable == 3, (
        "the wave hand-off patches the same concurrency"
    )
    assert seen["context"].node_names == (), "nothing outside the wave to protect"
    assert seen["finalize_candidate"].max_unavailable == 1, (
        "the steady-state Reconciler keeps the site's conservative concurrency"
    )
    plan = release.state[RUNTIME.FLEET_WAVE_PLANS_STATE_KEY]["gpu-a"]
    assert plan == {
        "release_id": release.release_id,
        "phase": "join",
        "reason": RUNTIME.JOIN_SINGLE_WAVE_REASON,
        "node_count": 3,
        "waves": 1,
        "max_unavailable": 3,
        "first_wave_max_unavailable": 3,
    }, "the state says why this join rolled no canary wave"
    narration = capsys.readouterr().err
    assert "fleet-wave-plan" in narration, "the operator is told the plan"
    assert "reason=no-live-agents" in narration
    assert "waves=1" in narration


def test_a_join_with_a_live_agent_keeps_the_canary_policy(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, seen = _roll(tmp_path, monkeypatch, phase="join", live=("node-b",))

    assert seen["policy"].first_wave_max_unavailable == 1
    assert seen["policy"] == FLEET.node_rollout_policy(
        release, {name: "zone-a" for name in NODES}, phase="join"
    ), "a live agent on any node means the ordinary wave policy applies"
    assert RUNTIME.FLEET_WAVE_PLANS_STATE_KEY not in release.state


def test_an_upgrade_never_consults_live_agents_and_keeps_its_policy(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, seen = _roll(tmp_path, monkeypatch, phase="upgrade", live=())

    assert seen["live_probes"] == 0, "an upgrade protects every node regardless"
    assert seen["policy"].first_wave_max_unavailable == 1
    assert seen["policy"] == FLEET.node_rollout_policy(
        release, {name: "zone-a" for name in NODES}, phase="upgrade"
    )
    assert RUNTIME.FLEET_WAVE_PLANS_STATE_KEY not in release.state


def test_the_single_wave_policy_respects_the_installer_concurrency_ceiling() -> None:
    """The ceiling caps installer Jobs, not availability; it holds in a join."""

    assert RUNTIME.join_single_wave_policy(40) == FLEET.NodeRolloutPolicy(
        max_unavailable=FLEET.MAX_UPGRADE_UNAVAILABLE,
        first_wave_max_unavailable=FLEET.MAX_UPGRADE_UNAVAILABLE,
        max_unavailable_per_failure_domain=FLEET.MAX_UPGRADE_UNAVAILABLE,
    )
    assert RUNTIME.join_single_wave_policy(1) == FLEET.NodeRolloutPolicy(1, 1, 1)
    assert RUNTIME.join_single_wave_policy(0) == FLEET.NodeRolloutPolicy(1, 1, 1)


def test_the_single_wave_policy_plans_one_wave_even_for_unknown_domains() -> None:
    """The control plane's planner, fed the policy, yields exactly one wave --
    including for the unlabelled fleet the live join ran on, which the
    ordinary policy would split into one wave per node."""

    domains = {name: UNKNOWN_FAILURE_DOMAIN for name in NODES}

    def request(policy: FLEET.NodeRolloutPolicy) -> FleetDeploymentRequest:
        return FleetDeploymentRequest(
            cluster_id="gpu-a",
            node_ids=list(NODES),
            desired_agent_version="0.10.0",
            desired_artifact_sha256="a" * 64,
            desired_policy_version="catalog-a",
            desired_runtime_profile_version="hyperpod-v1",
            desired_config_digest="c" * 64,
            max_unavailable=policy.max_unavailable,
            first_wave_max_unavailable=policy.first_wave_max_unavailable,
            max_unavailable_per_failure_domain=(
                policy.max_unavailable_per_failure_domain
            ),
            node_failure_domains=domains,
        )

    single = deployment_waves(request(RUNTIME.join_single_wave_policy(len(NODES))))
    ordinary = deployment_waves(
        request(
            FLEET.node_rollout_policy(
                SimpleNamespace(config=SimpleNamespace(upgrade_max_unavailable=0)),
                domains,
                phase="join",
            )
        )
    )

    assert single == [list(NODES)]
    assert ordinary == [[name] for name in NODES], (
        "the contrast the single wave saves: one safety round trip per node"
    )


def test_the_wave_plan_record_keeps_only_this_release(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    release.state[RUNTIME.FLEET_WAVE_PLANS_STATE_KEY] = {
        "gpu-old": {"release_id": "someone-else", "waves": 1},
        "gpu-z": {"release_id": release.release_id, "waves": 1},
    }

    RUNTIME.record_join_wave_plan(
        release,
        config.clusters[0],
        node_count=4,
        policy=RUNTIME.join_single_wave_policy(4),
    )

    assert set(release.state[RUNTIME.FLEET_WAVE_PLANS_STATE_KEY]) == {"gpu-a", "gpu-z"}


def test_a_first_bootstrap_plans_one_wave_like_a_join(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live 2026-09-12 trace: with an empty store every node is "missing" to the
    safety probe, so a bootstrap's one-node canary wave could never converge on
    a cluster larger than one node. Nothing is live, so nothing needs rationing."""

    release, seen = _roll(tmp_path, monkeypatch, phase="bootstrap", live=())

    assert seen["context"].node_names == ()
    assert seen["context"].max_unavailable == len(NODES)
    plan = release.state[RUNTIME.FLEET_WAVE_PLANS_STATE_KEY]["gpu-a"]
    assert plan["phase"] == "bootstrap" and plan["waves"] == 1
