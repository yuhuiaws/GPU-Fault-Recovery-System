"""A join's wave safety protects only nodes that already hold a live agent."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_node_runtime_rollout as RUNTIME
from gpu_fault_release import rollout as MODULE
from tests.regional._release_orchestrator_support import config_file

NODES = ("node-a", "node-b", "node-c")


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

    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    target = config.clusters[0]
    monkeypatch.setattr(release, "_target_node_names", lambda _target: NODES)
    monkeypatch.setattr(
        RUNTIME,
        "target_node_failure_domains",
        lambda *_args: {name: "zone-a" for name in NODES},
    )
    monkeypatch.setattr(
        RUNTIME, "ensure_pre_node_mutation_barrier", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        release, "_deploy_reconciler", lambda *_args, **_kwargs: ("b" * 64, "e" * 64)
    )
    monkeypatch.setattr(
        RUNTIME, "finish_legacy_node_runtime_rollback", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        RUNTIME,
        "_create_fleet_rollout",
        lambda *_a, **_k: ("dep-1", {"status": "PLANNED"}),
    )
    monkeypatch.setattr(
        RUNTIME,
        "live_agent_node_names",
        lambda _release, _target, names: tuple(n for n in names if n == "node-b"),
    )
    contexts: list[Any] = []
    monkeypatch.setattr(
        RUNTIME,
        "run_fleet_waves",
        lambda _release, _target, context, _deployment: contexts.append(context),
    )
    monkeypatch.setattr(
        RUNTIME, "_finalize_node_runtime", lambda *_a, **_k: ("b" * 64, "e" * 64)
    )

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

    assert [context.node_names for context in contexts] == [expected], (
        "the wave safety gate was given the wrong node set for this phase"
    )
