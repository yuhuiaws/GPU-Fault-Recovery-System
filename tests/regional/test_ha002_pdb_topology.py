"""HA-002: PDB re-check before the second Eviction, derived topology, per-step cleanup."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.e2e.regional import run_ha002_pdb_topology as ha002


def test_second_eviction_is_refused_when_the_pdb_reopened() -> None:
    open_budget = {"name": "gpu-fault-api-ha-pdb", "disruptions_allowed": 1}
    with pytest.raises(ha002.CaseError, match="allows 1 disruption"):
        ha002.require_pdb_still_blocked(
            "gpu-fault-api-ha-pdb", snapshot=lambda _n: open_budget
        )

    closed = {"name": "gpu-fault-api-ha-pdb", "disruptions_allowed": 0}
    assert (
        ha002.require_pdb_still_blocked(
            "gpu-fault-api-ha-pdb", snapshot=lambda _n: closed
        )
        == closed
    )


def test_balanced_distribution_follows_replicas_not_three_nodes() -> None:
    assert ha002.balanced_distribution({"a": 1, "b": 1, "c": 1}, 3) is True, (
        "one replica per node over three nodes is balanced"
    )
    assert ha002.balanced_distribution({"a": 2, "b": 2, "c": 2}, 6) is True, (
        "two replicas per node is balanced when replicas exceed nodes"
    )
    assert ha002.balanced_distribution({"a": 1, "b": 1, "c": 1, "d": 1}, 4) is True, (
        "four nodes with one replica each is balanced; three is not a magic number"
    )
    assert ha002.balanced_distribution({"a": 2, "b": 1, "c": 1}, 4) is True, (
        "a one-replica spread is the best four replicas can do on three nodes"
    )
    assert not ha002.balanced_distribution({"a": 3, "b": 1, "c": 1}, 5), (
        "a two-replica gap between nodes is skewed"
    )
    assert not ha002.balanced_distribution({"a": 1, "b": 1}, 3), (
        "a missing replica is not balanced"
    )
    assert ha002.balanced_distribution({}, 0) is True, (
        "zero replicas on zero nodes is trivially balanced"
    )


def test_capacity_assessment_is_derived_from_the_cordon_observation() -> None:
    constrained = ha002.capacity_assessment(
        cpu_nodes=3, ingress_replicas=3, ingress_ready_during_cordon=2
    )
    assert constrained["topology_constrained"] is True
    assert constrained["shortfall"] == 1
    assert "add a CPU node" in constrained["recommendation"]
    assert "2 of 3" in constrained["recommendation"]

    roomy = ha002.capacity_assessment(
        cpu_nodes=4, ingress_replicas=3, ingress_ready_during_cordon=3
    )
    assert roomy["topology_constrained"] is False
    assert roomy["shortfall"] == 0
    assert roomy["recommendation"].startswith("no additional CPU node required"), roomy[
        "recommendation"
    ]

    odd = ha002.capacity_assessment(
        cpu_nodes=5, ingress_replicas=3, ingress_ready_during_cordon=2
    )
    assert odd["topology_constrained"] is False
    assert "inspect scheduling events" in odd["recommendation"]


def test_schedulable_cpu_nodes_excludes_cordoned_and_tainted() -> None:
    nodes = {
        "items": [
            {
                "metadata": {"name": "a"},
                "spec": {},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            },
            {
                "metadata": {"name": "b"},
                "spec": {"unschedulable": True},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            },
            {
                "metadata": {"name": "c"},
                "spec": {"taints": [{"key": "x"}]},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            },
            {
                "metadata": {"name": "d"},
                "spec": {},
                "status": {"conditions": [{"type": "Ready", "status": "False"}]},
            },
        ]
    }
    assert ha002.schedulable_cpu_nodes(nodes) == ["a"]


def test_cleanup_runs_every_step_even_when_uncordon_fails(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def run(argv: list[str], **_kwargs: object):
        calls.append(" ".join(argv[-2:]))
        raise ha002.CaseError("kubectl timed out")

    def gpu(*args: str, **_kwargs: object) -> str:
        calls.append("gpu " + " ".join(args[:2]))
        return ""

    def cpu(*args: str, **_kwargs: object) -> str:
        calls.append("cpu " + " ".join(args[:2]))
        return ""

    monkeypatch.setattr(ha002.COMMON, "run", run)
    monkeypatch.setattr(ha002.COMMON, "gpu", gpu)
    monkeypatch.setattr(ha002.COMMON, "cpu", cpu)
    monkeypatch.setattr(
        ha002.COMMON,
        "control_sample",
        lambda include_queue: {"queue": {"depth": 0}, "spool_ready": 0},
    )
    monkeypatch.setattr(
        ha002.COMMON, "probe_resources", lambda: {"count": 0, "resources": {}}
    )
    monkeypatch.setattr(
        ha002,
        "current_node",
        lambda name: {"name": name, "unschedulable": False, "taints": []},
    )
    stopped = []
    monkeypatch.setattr(ha002, "stop_watchdog", lambda p, h: stopped.append(True))
    context = {
        "node_name": "node-a",
        "spool_replicas": 0,
        "baseline_node": {"taints": []},
    }
    state = {
        "cordoned": True,
        "watchdog": None,
        "watchdog_handle": None,
        "probe_created": True,
    }
    result: dict = {}

    errors = ha002.cleanup_case(tmp_path, context, state, result)

    assert errors == ["uncordon: CaseError: kubectl timed out"], errors
    assert stopped == [True], "the watchdog step must run after a failed uncordon"
    assert any(call.startswith("gpu delete") for call in calls), (
        f"the GPU-side delete must still run after a failed uncordon: {calls}"
    )
    assert any(call.startswith("cpu rollout") for call in calls), (
        f"the CPU-side rollout must still run after a failed uncordon: {calls}"
    )
    assert "postflight" in result


def test_execute_context_requires_the_derived_plan_fields(tmp_path: Path) -> None:
    with pytest.raises(ha002.CaseError, match="re-plan"):
        ha002.execute_context({"target_node": {"name": "n"}}, tmp_path / "plan.json")


def test_wait_recovery_uses_declared_replicas(monkeypatch) -> None:
    samples = iter(
        [
            {
                "ingress_ready": 3,
                "worker_ready": 5,
                "endpoint_ready": 3,
                "spool_ready": 0,
                "queue": {"depth": 0},
            },
            {
                "ingress_ready": 4,
                "worker_ready": 8,
                "endpoint_ready": 4,
                "spool_ready": 2,
                "queue": {"depth": 0},
            },
        ]
    )
    monkeypatch.setattr(
        ha002.COMMON, "control_sample", lambda include_queue: next(samples)
    )
    monkeypatch.setattr(ha002.time, "sleep", lambda _s: None)

    timeline = ha002.wait_recovery(
        replicas={
            ha002.COMMON.INGRESS_APP: 4,
            ha002.COMMON.WORKER_APP: 8,
            ha002.COMMON.SPOOL_APP: 2,
        },
        timeout_seconds=5,
    )

    assert len(timeline) == 2, "a 3/6 sample must not satisfy a 4/8 deployment"


def test_ha002_main_wraps_signal_abort_like_ha001() -> None:
    import inspect

    assert "install_abort_signals" in dir(ha002)
    assert "run_case_main" in dir(ha002)
    owner = inspect.getmodule(ha002.run_case_main).__name__
    assert owner.endswith("regional_live_fixture"), (
        f"run_case_main must be the shared fixture entrypoint, not a copy: {owner}"
    )
