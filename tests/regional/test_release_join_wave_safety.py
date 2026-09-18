"""Join lease checks exclude uninstalled nodes without widening node budgets."""

from __future__ import annotations

import json
from dataclasses import replace
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
    release, probe = _release({"gpu-a": ["node-b"], "gpu-x": ["node-a"]})
    monkeypatch.setattr(FLEET, "exec_cpu_ingress_probe", probe)

    live = FLEET.live_agent_node_names(
        release, SimpleNamespace(cluster_id="gpu-a"), NODES
    )

    assert live == ("node-b",), "another cluster's node is not ours"


@pytest.mark.parametrize("listed", [["node-unknown"], ["node-b", "node-b"], {}, None])
def test_live_agent_inventory_rejects_unknown_or_malformed_members(
    monkeypatch: pytest.MonkeyPatch, listed
) -> None:
    release, probe = _release({"gpu-a": listed})
    monkeypatch.setattr(FLEET, "exec_cpu_ingress_probe", probe)
    with pytest.raises(FLEET.ReleaseError, match="Agent inventory is invalid"):
        FLEET.live_agent_node_names(release, SimpleNamespace(cluster_id="gpu-a"), NODES)


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
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: str,
    live: tuple[str, ...],
    nodes: tuple[str, ...] = NODES,
    domain: str = "zone-a",
    configuration: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run ``roll_node_runtime`` with every seam stubbed; return what it handed on."""

    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    config = replace(config, **(configuration or {}))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    target = config.clusters[0]
    seen: dict[str, Any] = {"deploys": [], "live_probes": 0}
    monkeypatch.setattr(release, "_target_node_names", lambda _target: nodes)
    monkeypatch.setattr(
        RUNTIME,
        "target_node_failure_domains",
        lambda *_args: {name: domain for name in nodes},
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


def test_a_join_with_no_live_agent_preserves_the_configured_canary_budget(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No Agent lease is not authorization to widen the installer window."""

    release, seen = _roll(tmp_path, monkeypatch, phase="join", live=())

    policy = seen["policy"]
    assert policy == FLEET.NodeRolloutPolicy(
        max_unavailable=1,
        first_wave_max_unavailable=1,
        max_unavailable_per_failure_domain=1,
    ), "the fleet record keeps the configured budget"
    assert seen["deploys"][0]["max_unavailable"] == 1, (
        "the paused Reconciler is deployed with the same concurrency"
    )
    assert seen["context"].max_unavailable == 1, (
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
        "max_unavailable": 1,
        "first_wave_max_unavailable": 1,
        "max_unavailable_per_failure_domain": 1,
    }, "the state records the budget without inventing Agent availability"
    narration = capsys.readouterr().err
    assert "fleet-wave-plan" in narration, "the operator is told the plan"
    assert "reason=no-live-agents" in narration
    assert "max_unavailable=1" in narration


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


@pytest.mark.parametrize("phase", ["join", "bootstrap"])
def test_join_without_agents_keeps_the_site_installer_budget(
    tmp_path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    _, seen = _roll(
        tmp_path,
        monkeypatch,
        phase=phase,
        live=(),
        nodes=tuple(f"node-{index}" for index in range(512)),
        configuration={
            "upgrade_max_unavailable": 32,
            "upgrade_max_parallel_clusters": 8,
        },
    )
    assert seen["policy"].max_unavailable == 8
    assert seen["policy"].first_wave_max_unavailable == 1
    assert seen["deploys"][0]["max_unavailable"] == 8


def test_a_join_without_agents_keeps_unknown_failure_domains_serial(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    _, seen = _roll(
        tmp_path,
        monkeypatch,
        phase="join",
        live=(),
        domain=UNKNOWN_FAILURE_DOMAIN,
        configuration={"upgrade_max_unavailable": 0},
    )

    assert deployment_waves(request(seen["policy"])) == [[name] for name in NODES]


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
        policy=FLEET.NodeRolloutPolicy(4, 1, 2),
    )

    assert set(release.state[RUNTIME.FLEET_WAVE_PLANS_STATE_KEY]) == {"gpu-a", "gpu-z"}


def test_a_first_bootstrap_excludes_uninstalled_agents_but_keeps_the_canary(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The missing-Agent issue is independent of installer safety budgets."""

    release, seen = _roll(tmp_path, monkeypatch, phase="bootstrap", live=())

    assert seen["context"].node_names == ()
    assert seen["context"].max_unavailable == 1
    plan = release.state[RUNTIME.FLEET_WAVE_PLANS_STATE_KEY]["gpu-a"]
    assert plan["phase"] == "bootstrap" and plan["first_wave_max_unavailable"] == 1


def test_join_protects_newly_converged_agents_on_the_next_wave(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, seen = _roll(tmp_path, monkeypatch, phase="join", live=())
    waves = iter((("node-a",), ("node-b",)))
    current: list[tuple[str, ...]] = []
    gates: list[tuple[str, ...]] = []

    def next_wave(_deployment):
        wave = next(waves)
        current.append(wave)
        return wave

    def fleet(operation, _payload):
        if operation == "next-wave":
            return {"node_ids": current[-1]}
        return {"status": "SUCCEEDED" if len(current) == 2 else "IN_PROGRESS"}

    monkeypatch.setattr(FLEET, "next_deployment_wave", next_wave)
    monkeypatch.setattr(
        FLEET,
        "ensure_rollout_wave_safe",
        lambda *_args, **kwargs: gates.append(kwargs["node_names"]),
    )
    monkeypatch.setattr(
        FLEET, "wave_lease_margin_holds", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(FLEET, "hand_wave_to_reconciler", lambda *_args: IDENTITY)
    monkeypatch.setattr(release, "_fleet_command", fleet)
    monkeypatch.setattr(release, "_wait_agents", lambda *_args, **_kwargs: None)
    FLEET.run_fleet_waves(
        release, release.config.clusters[0], seen["context"], {"status": "PLANNED"}
    )
    assert gates == [(), ("node-a",)]
