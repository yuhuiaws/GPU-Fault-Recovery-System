"""``--retire-departed-agents``: the inventory shapes it refuses and the plan
verdicts at the edges.

The control-plane inventory must carry a list of Agent mappings and a
timezone-aware clock, or nothing is planned. An Agent outside the retireable
lifecycles or with a naive ``last_seen_at`` is reported ineligible with the
reason. Two Agents of one cluster share one Kubernetes node read. A blank
``--node`` is refused before any read, and an inventory that moves between the
plan and the apply refuses the apply by naming what moved.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin import workflow_reconcile_departed_agents as departed
from gpu_fault.admin.bootstrap_common import BootstrapError

NODE_A = "hyperpod-i-00000000000000001"
NODE_B = "hyperpod-i-00000000000000002"
EVALUATED_AT = "2026-10-01T06:00:00+00:00"
REFERENCE = "CHG-2026-10-01"


def _site(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "cpu_eks_arn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu-control",
            "clusters": [
                {
                    "cluster_id": "gpu-a",
                    "context": "gpu-a-context",
                    "hyperpod_cluster_name": "gpu-a-hyperpod",
                    "region": "us-west-2",
                }
            ],
        },
        environment={"KUBECONFIG": str(tmp_path / "gpu.kubeconfig")},
        source_sha256="a" * 64,
    )


def _agent(node_id: str = NODE_A, **overrides: object) -> dict[str, object]:
    instance = node_id.removeprefix("hyperpod-")
    entry: dict[str, object] = {
        "cluster_id": "gpu-a",
        "node_id": node_id,
        "lifecycle_state": "ACTIVE",
        "generation": 4,
        "node_instance_id": instance,
        "agent_incarnation_id": f"inc-{instance}",
        "transition_id": None,
        "first_seen_at": "2026-09-30T10:00:00+00:00",
        "last_seen_at": "2026-10-01T05:00:00+00:00",
        "lease_expires_at": "2026-10-01T05:01:00+00:00",
        "agent_version": "2026.9.10",
        "runtime_profile_version": "hyperpod-v1",
    }
    entry.update(overrides)
    return entry


class Commands:
    """``run_command`` double: an empty GPU node list and an empty HyperPod inventory."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **_options: object) -> SimpleNamespace:
        self.calls.append(list(command))
        if command[0] == "kubectl":
            return SimpleNamespace(
                returncode=0, stdout=json.dumps({"items": []}), stderr=""
            )
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"ClusterNodeSummaries": []}), stderr=""
        )

    @property
    def kubectl_calls(self) -> list[list[str]]:
        return [call for call in self.calls if call[0] == "kubectl"]


def _inventory(agents: list[dict[str, object]], **overrides: object) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": "agent-inventory",
        "evaluated_at": EVALUATED_AT,
        "agents": agents,
    }
    result.update(overrides)
    return result


def _pod(inventories: list[dict[str, Any]], calls: list[dict[str, Any]]):
    def run(_site: object, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        if payload["mode"] == "list-agents":
            return inventories.pop(0)
        return {
            "mode": "retire-departed-agents-apply",
            "retired_agents": [],
            "failed_agents": [],
            "failures": {},
        }

    return run


@pytest.mark.parametrize(
    ("inventory", "problem"),
    [
        (
            {**_inventory([]), "agents": "not-a-list"},
            "inventory from the control plane is invalid",
        ),
        (_inventory(["bare"]), "inventory from the control plane is invalid"),
        (_inventory([], evaluated_at=None), "has no clock"),
        (
            _inventory([], evaluated_at="2026-10-01T06:00:00"),
            "clock is not timezone-aware",
        ),
    ],
    ids=["agents-not-list", "agent-not-mapping", "no-clock", "naive-clock"],
)
def test_an_unusable_inventory_plans_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory: dict[str, Any],
    problem: str,
) -> None:
    commands = Commands()
    monkeypatch.setattr(departed, "run_pod_script", _pod([inventory], []))
    monkeypatch.setattr(reconcile, "run_command", commands)

    with pytest.raises(BootstrapError, match=problem):
        departed.run_retire_departed_agents(_site(tmp_path), tmp_path, dry_run=True)
    assert commands.calls == [], "no cluster is read for an inventory that is refused"


def test_lifecycle_and_naive_heartbeat_are_reported_as_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents = [
        _agent(NODE_A, lifecycle_state="UNKNOWN_STATE"),
        _agent(NODE_B, last_seen_at="2026-10-01T05:00:00"),
    ]
    commands = Commands()
    monkeypatch.setattr(departed, "run_pod_script", _pod([_inventory(agents)], []))
    monkeypatch.setattr(reconcile, "run_command", commands)

    plan = departed.run_retire_departed_agents(_site(tmp_path), tmp_path, dry_run=True)

    by_node = {item["node_id"]: item for item in plan["items"]}
    assert by_node[NODE_A]["eligible"] is False
    assert "agent lifecycle state is UNKNOWN_STATE" in by_node[NODE_A]["reasons"]
    assert by_node[NODE_B]["eligible"] is False
    assert (
        "agent last_seen_at is missing or not timezone-aware"
        in (by_node[NODE_B]["reasons"])
    )
    assert len(commands.kubectl_calls) == 1, "one cluster, one node read for both"


def test_a_blank_node_is_refused_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(departed, "run_pod_script", _pod([], calls))

    with pytest.raises(BootstrapError, match="--node must not be blank"):
        departed.run_retire_departed_agents(
            _site(tmp_path), tmp_path, node_ids=(NODE_A, "  "), dry_run=True
        )
    assert calls == []


def test_an_agent_that_vanishes_between_plan_and_apply_refuses_the_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        departed,
        "run_pod_script",
        _pod([_inventory([_agent(NODE_A)]), _inventory([])], calls),
    )
    monkeypatch.setattr(reconcile, "run_command", Commands())

    with pytest.raises(BootstrapError, match="plan changed before apply") as info:
        departed.run_retire_departed_agents(
            _site(tmp_path), tmp_path, reference=REFERENCE
        )

    message = str(info.value)
    assert f"gpu-a/{NODE_A}: no longer in the plan" in message
    assert f"{NODE_A}: newly in the plan" in message
    assert [call["mode"] for call in calls] == ["list-agents", "list-agents"], (
        "the apply never reached the Pod"
    )
    assert not (tmp_path / departed.HISTORY_PATH).exists(), "nothing was archived"
