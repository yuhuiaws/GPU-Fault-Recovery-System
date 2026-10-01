"""``workflow-reconcile --retire-departed-agents`` and the Agent of a reclaimed node.

HyperPod terminated a spot GPU node; the node left Kubernetes and the HyperPod
node list, but its fleet Agent record stayed ``ACTIVE`` for good -- the Agent
process died with the instance and nothing else revokes it. The release
verification then read one Agent more than there were nodes and rolled every
deploy back.

The flag plans with the never-dispatched evidence (absent from Kubernetes
*and* from HyperPod, read on the deploy host), guards on the heartbeat age,
refuses on drift between plan and apply, and retires through the registry's
own drain/revoke path in the Pod with the evidence on the transition.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin import workflow_reconcile_departed_agents as departed
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import SiteConfigError
from gpu_fault.fleet import (
    AgentLifecycleState,
    AgentRecord,
    AgentTransitionRequest,
    FleetRegistry,
)
from tests.admin.conftest import TEST_OPERATOR_ARN

NODE = "hyperpod-i-00000000000000001"
INSTANCE = "i-00000000000000001"
LIVE_NODE = "hyperpod-i-00000000000000002"
LIVE_INSTANCE = "i-00000000000000002"
EVALUATED_AT = "2026-10-01T07:00:00+00:00"
REFERENCE = "CHG-2026-1001-02"


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


def _agent(node_id: str = NODE, **overrides: object) -> dict[str, object]:
    """The Pod's inventory entry for one Agent record."""

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
    """``run_command`` double: the GPU node list and the HyperPod inventory."""

    def __init__(
        self,
        *,
        instances: tuple[str, ...] = (LIVE_INSTANCE,),
        kubernetes_nodes: tuple[str, ...] = (LIVE_NODE,),
    ) -> None:
        self.instances = instances
        self.kubernetes_nodes = kubernetes_nodes
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **_options: object) -> SimpleNamespace:
        self.calls.append(list(command))
        if command[0] == "kubectl":
            assert command[-4:] == ["get", "nodes", "-o", "json"], command
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {"name": name, "annotations": {}},
                                "spec": {"unschedulable": False, "taints": []},
                            }
                            for name in self.kubernetes_nodes
                        ]
                    }
                ),
                stderr="",
            )
        assert command[:3] == ["aws", "sagemaker", "list-cluster-nodes"], command
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "ClusterNodeSummaries": [
                        {"InstanceId": instance} for instance in self.instances
                    ]
                }
            ),
            stderr="",
        )

    @property
    def aws_calls(self) -> list[list[str]]:
        return [call for call in self.calls if call[0] == "aws"]


def _pod(agents: list[dict[str, object]], calls: list[dict[str, Any]]):
    def run(_site: object, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        if payload["mode"] == "list-agents":
            return {
                "mode": "agent-inventory",
                "evaluated_at": EVALUATED_AT,
                "agents": agents,
            }
        return {
            "mode": "retire-departed-agents-apply",
            "retired_agents": [
                {
                    "agent": f"{item['cluster_id']}/{item['node_id']}",
                    "generation": item["generation"] + 1,
                    "transition_id": payload["transition_id"],
                    "lifecycle_state": "REVOKED",
                }
                for item in payload["items"]
            ],
            "failed_agents": [],
            "failures": {},
        }

    return run


def _dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commands: Commands,
    *,
    agents: list[dict[str, object]] | None = None,
    **selectors: Any,
) -> dict[str, Any]:
    monkeypatch.setattr(departed, "run_pod_script", _pod(agents or [_agent()], []))
    monkeypatch.setattr(reconcile, "run_command", commands)
    return departed.run_retire_departed_agents(
        _site(tmp_path), tmp_path, dry_run=True, **selectors
    )


def _item(plan: dict[str, Any], node_id: str = NODE) -> dict[str, Any]:
    [item] = [item for item in plan["items"] if item["node_id"] == node_id]
    return item


# ------------------------------------------------------------- the discovery


def test_the_agent_of_a_departed_node_is_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands()

    plan = _dry_run(tmp_path, monkeypatch, commands)

    item = _item(plan)
    assert item["eligible"] is True, item["reasons"]
    assert item["reasons"] == []
    assert item["terminalization"] == departed.RETIRE_DEPARTED_AGENTS
    assert item["agent"] == f"gpu-a/{NODE}"
    assert item["generation"] == 4 and item["lifecycle_state"] == "ACTIVE"
    evidence = item["node_evidence"]
    assert evidence["absent_from_kubernetes"] is True
    assert evidence["absent_from_provider"] is True
    assert evidence["instance_id"] == INSTANCE
    assert evidence["blockers"] == []
    assert evidence["sources"] == {
        "kubernetes": {"consulted": True, "present": False},
        "hyperpod": {
            "consulted": True,
            "cluster_name": "gpu-a-hyperpod",
            "region": "us-west-2",
            "listed": False,
        },
    }
    read_at = item["evidence_read_at"]
    assert sorted(read_at) == ["hyperpod", "kubernetes"]
    for value in read_at.values():
        assert datetime.fromisoformat(value).tzinfo is not None, value
    assert plan["discovery"] == {
        "agents": 1,
        "revoked": 0,
        "candidates": 1,
        "eligible": 1,
    }
    assert plan["dry_run"] is True
    assert len(commands.aws_calls) == 1, "one provider read per cluster per plan"


def test_an_agent_whose_node_is_still_in_kubernetes_is_not_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Present -- Ready or NotReady, cordoned or clean -- is not departed."""

    commands = Commands(kubernetes_nodes=(NODE, LIVE_NODE))

    plan = _dry_run(tmp_path, monkeypatch, commands)

    item = _item(plan)
    assert item["eligible"] is False
    assert item["reasons"] == [f"{NODE}: {departed.KUBERNETES_PRESENT_REASON}"]
    assert item["node_evidence"]["absent_from_kubernetes"] is False
    assert item["node_evidence"]["sources"] == {
        "kubernetes": {"consulted": True, "present": True}
    }
    assert commands.aws_calls == [], "a node Kubernetes has is never looked up"


def test_an_agent_whose_instance_hyperpod_still_lists_is_not_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands(instances=(INSTANCE, LIVE_INSTANCE))

    plan = _dry_run(tmp_path, monkeypatch, commands)

    item = _item(plan)
    assert item["eligible"] is False
    assert item["node_evidence"]["absent_from_kubernetes"] is True
    assert item["node_evidence"]["absent_from_provider"] is False
    assert item["node_evidence"]["sources"]["hyperpod"]["listed"] is True
    assert item["reasons"] == [
        f"{NODE}: node is missing from Kubernetes but HyperPod still lists "
        f"instance {INSTANCE}"
    ]


def test_a_heartbeat_younger_than_the_guard_keeps_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gone from both sources three minutes after its last heartbeat is a node
    mid-replacement as far as the fleet knows, not a departed one."""

    recent = (datetime.fromisoformat(EVALUATED_AT) - timedelta(minutes=3)).isoformat()

    plan = _dry_run(
        tmp_path, monkeypatch, Commands(), agents=[_agent(last_seen_at=recent)]
    )

    item = _item(plan)
    assert item["eligible"] is False
    assert item["reasons"] == [departed.GUARD_REASON]
    assert item["node_evidence"]["absent_from_provider"] is True, "the node did go"


def test_a_revoked_record_is_not_a_candidate_and_a_named_one_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents = [_agent(lifecycle_state="REVOKED"), _agent(LIVE_NODE)]

    plan = _dry_run(tmp_path, monkeypatch, Commands(), agents=agents)
    named = _dry_run(tmp_path, monkeypatch, Commands(), agents=agents, node_ids=(NODE,))

    assert [item["node_id"] for item in plan["items"]] == [LIVE_NODE]
    assert plan["discovery"] == {
        "agents": 2,
        "revoked": 1,
        "candidates": 1,
        "eligible": 0,
    }
    assert _item(plan, LIVE_NODE)["reasons"] == [
        f"{LIVE_NODE}: {departed.KUBERNETES_PRESENT_REASON}"
    ]
    [item] = named["items"]
    assert item == {
        "agent": NODE,
        "node_id": NODE,
        "terminalization": departed.RETIRE_DEPARTED_AGENTS,
        "eligible": False,
        "reasons": [departed.NO_RECORD_REASON],
    }


def test_an_agent_of_a_cluster_outside_the_site_is_reported_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = Commands()

    plan = _dry_run(
        tmp_path, monkeypatch, commands, agents=[_agent(cluster_id="gpu-gone")]
    )

    item = _item(plan)
    assert item["eligible"] is False
    assert item["reasons"] == ["cluster gpu-gone is not in the managed site"]
    assert item["node_evidence"] is None
    assert commands.calls == [], "no kubeconfig for it, so nothing is read"


def test_a_named_ineligible_agent_refuses_the_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(departed, "run_pod_script", _pod([_agent()], calls))
    monkeypatch.setattr(reconcile, "run_command", Commands(instances=(INSTANCE,)))

    with pytest.raises(BootstrapError, match="refuses ineligible Agents") as refused:
        departed.run_retire_departed_agents(
            _site(tmp_path), tmp_path, node_ids=(NODE,), reference=REFERENCE
        )

    assert "HyperPod still lists" in str(refused.value)
    assert [call["mode"] for call in calls] == ["list-agents"], "nothing was applied"


def test_an_apply_without_a_reference_is_refused_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(departed, "run_pod_script", _pod([_agent()], calls))

    with pytest.raises(BootstrapError, match="requires --reference"):
        departed.run_retire_departed_agents(_site(tmp_path), tmp_path)

    assert calls == []


# ------------------------------------------------------------------ the digest


def test_the_read_times_are_outside_the_digest_and_the_verdicts_inside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _dry_run(tmp_path, monkeypatch, Commands())
    second = _dry_run(tmp_path, monkeypatch, Commands())
    listed = _dry_run(tmp_path, monkeypatch, Commands(instances=(INSTANCE,)))

    assert first["plan_sha256"] == second["plan_sha256"], (
        "a re-plan a moment later is not drift"
    )
    assert first["plan_sha256"] != listed["plan_sha256"], (
        "the provider's verdict binds the approval"
    )


def test_evidence_that_changes_between_plan_and_apply_refuses_the_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The instance reappears at HyperPod between the plan and the re-plan."""

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(departed, "run_pod_script", _pod([_agent()], calls))
    answers = iter([Commands(), Commands(instances=(INSTANCE,))])
    current: list[Commands] = []

    def run_command(command: list[str], **options: object) -> SimpleNamespace:
        if command[0] == "kubectl":
            current.append(next(answers))
        return current[-1](command, **options)

    monkeypatch.setattr(reconcile, "run_command", run_command)

    with pytest.raises(BootstrapError, match="plan changed before apply") as refused:
        departed.run_retire_departed_agents(
            _site(tmp_path), tmp_path, node_ids=(NODE,), reference=REFERENCE
        )

    message = str(refused.value)
    assert f"gpu-a/{NODE}: eligible True -> False" in message
    assert f"gpu-a/{NODE}: node_evidence" in message
    assert [call["mode"] for call in calls] == ["list-agents", "list-agents"], (
        "nothing was applied"
    )
    assert not (tmp_path / departed.HISTORY_PATH).exists(), "a refusal archives nothing"


def test_a_record_that_moved_between_plan_and_apply_refuses_the_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A heartbeat lands between the plan and the re-plan: the generation and
    ``last_seen_at`` the approval bound are no longer the record's."""

    inventories = iter([[_agent()], [_agent(generation=5, last_seen_at=EVALUATED_AT)]])
    calls: list[dict[str, Any]] = []

    def run(_site: object, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return {
            "mode": "agent-inventory",
            "evaluated_at": EVALUATED_AT,
            "agents": next(inventories),
        }

    monkeypatch.setattr(departed, "run_pod_script", run)
    monkeypatch.setattr(reconcile, "run_command", Commands())

    with pytest.raises(BootstrapError, match="plan changed before apply") as refused:
        departed.run_retire_departed_agents(
            _site(tmp_path), tmp_path, reference=REFERENCE
        )

    assert f"gpu-a/{NODE}: generation 4 -> 5" in str(refused.value)
    assert [call["mode"] for call in calls] == ["list-agents", "list-agents"]


# ------------------------------------------------------------------- the apply


def test_the_apply_sends_the_bound_keys_and_the_evidence_to_the_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(departed, "run_pod_script", _pod([_agent()], calls))
    monkeypatch.setattr(reconcile, "run_command", Commands())

    result = departed.run_retire_departed_agents(
        _site(tmp_path), tmp_path, reference=REFERENCE
    )

    assert [call["mode"] for call in calls] == [
        "list-agents",
        "list-agents",
        "retire-departed-agents",
    ]
    assert calls[0]["cluster_ids"] == ["gpu-a"]
    apply = calls[-1]
    [entry] = apply["items"]
    assert entry["cluster_id"] == "gpu-a" and entry["node_id"] == NODE
    assert entry["generation"] == 4
    assert entry["lifecycle_state"] == "ACTIVE"
    assert entry["last_seen_at"] == "2026-10-01T05:00:00+00:00"
    evidence = entry["departed_node_evidence"]
    assert evidence["node_id"] == NODE
    assert evidence["instance_id"] == INSTANCE
    assert evidence["absent_from_kubernetes"] is True
    assert evidence["absent_from_provider"] is True
    assert evidence["sources"]["hyperpod"]["cluster_name"] == "gpu-a-hyperpod"
    assert sorted(evidence["read_at"]) == ["hyperpod", "kubernetes"]
    assert apply["reference"] == REFERENCE
    assert apply["actor"] == TEST_OPERATOR_ARN
    assert apply["admin_plan_sha256"] == result["plan_sha256"]
    assert apply["transition_id"] == (
        f"workflow-reconcile/{REFERENCE}/{result['plan_sha256'][:16]}"
    )
    assert result["retired_agents"] == [
        {
            "agent": f"gpu-a/{NODE}",
            "generation": 5,
            "transition_id": apply["transition_id"],
            "lifecycle_state": "REVOKED",
        }
    ]
    assert result["failed_agents"] == []
    assert result["ineligible"] == {}
    assert result["dry_run"] is False
    archive = tmp_path / departed.HISTORY_PATH / str(result["plan_sha256"])
    plan = json.loads((archive / "plan.json").read_text(encoding="utf-8"))
    applied = json.loads((archive / "applied.json").read_text(encoding="utf-8"))
    assert plan["items"][0]["node_evidence"]["sources"]["hyperpod"]["listed"] is False
    assert plan["items"][0]["evidence_read_at"], "the archive keeps the read times"
    assert applied["actor"] == TEST_OPERATOR_ARN


def test_a_discovery_with_nothing_eligible_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(departed, "run_pod_script", _pod([_agent(LIVE_NODE)], calls))
    monkeypatch.setattr(reconcile, "run_command", Commands())

    result = departed.run_retire_departed_agents(
        _site(tmp_path), tmp_path, reference=REFERENCE
    )

    assert [call["mode"] for call in calls] == ["list-agents"]
    assert result["retired_agents"] == []
    assert list(result["ineligible"]) == [f"gpu-a/{LIVE_NODE}"]


# ----------------------------------------------------------- the in-Pod script


def _record(node_id: str, now: datetime, **overrides: Any) -> AgentRecord:
    values: dict[str, Any] = {
        "cluster_id": "gpu-a",
        "node_id": node_id,
        "endpoint": "https://10.0.0.7:9099",
        "agent_version": "2026.9.10",
        "artifact_sha256": "a" * 64,
        "policy_version": "610",
        "runtime_profile_version": "hyperpod-v1",
        "config_digest": "b" * 64,
        "allowed_operations": [],
        "node_instance_id": node_id.removeprefix("hyperpod-"),
        "agent_incarnation_id": f"inc-{node_id}",
        "first_seen_at": now - timedelta(days=1),
        "last_seen_at": now - timedelta(hours=1),
        "lease_expires_at": now - timedelta(minutes=59),
        "generation": 4,
    }
    values.update(overrides)
    return AgentRecord(**values)


def _fleet(now: datetime) -> tuple[Any, FleetRegistry]:
    from tests._builders import build_store

    store = build_store()
    registry = FleetRegistry(store, "s" * 32)
    store.save_agent(_record(NODE, now))
    store.save_agent(_record("hyperpod-i-0a000003", now))
    store.save_agent(
        _record("hyperpod-i-0a000004", now, last_seen_at=now - timedelta(minutes=2))
    )
    store.save_agent(_record("hyperpod-i-0a000005", now))
    drained = _record("hyperpod-i-0a000006", now)
    store.save_agent(drained)
    registry.drain_agent(
        "gpu-a",
        drained.node_id,
        AgentTransitionRequest(
            expected_generation=4, transition_id="earlier/drain", reason="earlier"
        ),
    )
    return store, registry


def _exec_script(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    store: Any,
    registry: FleetRegistry | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    from gpu_fault.app import ApplicationContext

    context = ApplicationContext(store=store, fleet_registry=registry)
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    exec(compile(departed.AGENT_SCRIPT, "<retire-departed-agents>", "exec"), {})
    return json.loads(capsys.readouterr().out)


def _evidence(node_id: str, *, absent_from_provider: bool = True) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "instance_id": node_id.removeprefix("hyperpod-"),
        "provider": "hyperpod",
        "absent_from_kubernetes": True,
        "absent_from_provider": absent_from_provider,
        "sources": {
            "kubernetes": {"consulted": True, "present": False},
            "hyperpod": {
                "consulted": True,
                "cluster_name": "gpu-a-hyperpod",
                "region": "us-west-2",
                "listed": not absent_from_provider,
            },
        },
        "read_at": {
            "kubernetes": "2026-10-01T07:00:00+00:00",
            "hyperpod": "2026-10-01T07:00:01+00:00",
        },
    }


def test_the_inventory_lists_every_record_with_its_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    now = datetime.now(timezone.utc)
    store, registry = _fleet(now)

    result = _exec_script(
        monkeypatch,
        capsys,
        store,
        registry,
        {"mode": "list-agents", "cluster_ids": ["gpu-a"]},
    )

    assert result["mode"] == "agent-inventory"
    assert datetime.fromisoformat(result["evaluated_at"]).tzinfo is not None
    assert [item["node_id"] for item in result["agents"]] == [
        NODE,
        "hyperpod-i-0a000003",
        "hyperpod-i-0a000004",
        "hyperpod-i-0a000005",
        "hyperpod-i-0a000006",
    ]
    first = result["agents"][0]
    assert set(first) == set(departed.RECORD_FIELDS)
    assert first["lifecycle_state"] == "ACTIVE" and first["generation"] == 4
    assert result["agents"][-1]["lifecycle_state"] == "DRAINING"
    assert result["agents"][-1]["transition_id"] == "earlier/drain"


def test_the_script_retires_through_drain_and_revoke_with_the_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The compare-and-set on generation, lifecycle and ``last_seen_at``, the
    guard re-checked on the Pod's clock, the evidence re-checked, the audit
    line on the transition, and every other item reported, not written."""

    now = datetime.now(timezone.utc)
    store, registry = _fleet(now)
    transition_id = f"workflow-reconcile/{REFERENCE}/0123456789abcdef"

    def entry(node_id: str, **overrides: Any) -> dict[str, Any]:
        record = store.get_agent("gpu-a", node_id)
        values: dict[str, Any] = {
            "cluster_id": "gpu-a",
            "node_id": node_id,
            "generation": record.generation,
            "lifecycle_state": record.lifecycle_state.value,
            "last_seen_at": record.last_seen_at.isoformat(),
            "departed_node_evidence": _evidence(node_id),
        }
        values.update(overrides)
        return values

    result = _exec_script(
        monkeypatch,
        capsys,
        store,
        registry,
        {
            "mode": "retire-departed-agents",
            "items": [
                entry(NODE),
                entry("hyperpod-i-0a000003", generation=3),
                entry("hyperpod-i-0a000004"),
                entry(
                    "hyperpod-i-0a000005",
                    departed_node_evidence=_evidence(
                        "hyperpod-i-0a000005", absent_from_provider=False
                    ),
                ),
                entry("hyperpod-i-0a000006"),
            ],
            "reference": REFERENCE,
            "actor": TEST_OPERATOR_ARN,
            "admin_plan_sha256": "d" * 64,
            "transition_id": transition_id,
        },
    )

    assert [item["agent"] for item in result["retired_agents"]] == [
        f"gpu-a/{NODE}",
        "gpu-a/hyperpod-i-0a000006",
    ]
    assert result["failed_agents"] == [
        "gpu-a/hyperpod-i-0a000003",
        "gpu-a/hyperpod-i-0a000004",
        "gpu-a/hyperpod-i-0a000005",
    ]
    assert (
        "generation changed: expected 3, found 4"
        in (result["failures"]["gpu-a/hyperpod-i-0a000003"])
    )
    assert departed.GUARD_REASON in (result["failures"]["gpu-a/hyperpod-i-0a000004"])
    assert (
        "does not prove the node gone"
        in (result["failures"]["gpu-a/hyperpod-i-0a000005"])
    )

    retired = store.get_agent("gpu-a", NODE)
    assert retired.lifecycle_state is AgentLifecycleState.REVOKED
    assert retired.generation == 5, "the drain bumped the generation once"
    assert retired.transition_id == transition_id
    assert retired.retired_incarnation_ids == [f"inc-{NODE}"]
    assert retired.lease_expires_at is not None and retired.lease_expires_at >= now
    reason = retired.transition_reason or ""
    assert reason.startswith(f"operator reconciliation {REFERENCE}: retired Agent"), (
        reason
    )
    for fragment in (
        NODE,
        f"instance {INSTANCE}",
        "HyperPod cluster gpu-a-hyperpod in us-west-2",
        "kubernetes read 2026-10-01T07:00:00+00:00",
        "hyperpod read 2026-10-01T07:00:01+00:00",
        f"actor {TEST_OPERATOR_ARN}",
        f"admin plan {'d' * 64}",
    ):
        assert fragment in reason, reason

    completed = store.get_agent("gpu-a", "hyperpod-i-0a000006")
    assert completed.lifecycle_state is AgentLifecycleState.REVOKED
    assert completed.transition_id == "earlier/drain", (
        "a stuck drain is finished under the transition that started it"
    )
    [_first, second] = result["retired_agents"]
    assert second["transition_id"] == "earlier/drain"

    for node_id in (
        "hyperpod-i-0a000003",
        "hyperpod-i-0a000004",
        "hyperpod-i-0a000005",
    ):
        untouched = store.get_agent("gpu-a", node_id)
        assert untouched.lifecycle_state is AgentLifecycleState.ACTIVE
        assert untouched.generation == 4


def test_the_script_accepts_the_inventory_spelling_of_the_heartbeat(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plan carries ``last_seen_at`` as the inventory serialized it (JSON
    ``Z`` suffix) while the record yields ``+00:00``; the compare-and-set must
    compare the instant, not the spelling (live defect: every apply failed)."""

    now = datetime.now(timezone.utc)
    store, registry = _fleet(now)
    record = store.get_agent("gpu-a", NODE)
    spelled = record.last_seen_at.isoformat().replace("+00:00", "Z")
    assert spelled.endswith("Z"), "the test must exercise the JSON spelling"

    result = _exec_script(
        monkeypatch,
        capsys,
        store,
        registry,
        {
            "mode": "retire-departed-agents",
            "items": [
                {
                    "cluster_id": "gpu-a",
                    "node_id": NODE,
                    "generation": record.generation,
                    "lifecycle_state": record.lifecycle_state.value,
                    "last_seen_at": spelled,
                    "departed_node_evidence": _evidence(NODE),
                }
            ],
            "reference": REFERENCE,
            "actor": TEST_OPERATOR_ARN,
            "admin_plan_sha256": "d" * 64,
            "transition_id": f"workflow-reconcile/{REFERENCE}/0123456789abcdef",
        },
    )

    assert result["failed_agents"] == [], result.get("failures")
    assert [item["agent"] for item in result["retired_agents"]] == [f"gpu-a/{NODE}"]
    assert store.get_agent("gpu-a", NODE).lifecycle_state is AgentLifecycleState.REVOKED


def test_the_script_refuses_without_a_registry_and_an_unknown_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    now = datetime.now(timezone.utc)
    store, _registry = _fleet(now)

    with pytest.raises(ValueError, match="agent registry is not enabled"):
        _exec_script(
            monkeypatch,
            capsys,
            store,
            None,
            {"mode": "retire-departed-agents", "items": []},
        )
    with pytest.raises(ValueError, match="unsupported departed-agent mode"):
        _exec_script(monkeypatch, capsys, store, None, {"mode": "retire"})
    assert store.get_agent("gpu-a", NODE).lifecycle_state is AgentLifecycleState.ACTIVE


# --------------------------------------------------------------------- the CLI


def _state_dir(tmp_path: Path) -> Path:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text("placeholder\n", encoding="utf-8")
    return state_dir


def test_the_flag_rides_workflow_reconcile_and_reaches_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(site: object, state_dir: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append({"site": site, "state_dir": state_dir, **kwargs})
        return {
            "mode": "retire-departed-agents-apply",
            "retired_agents": [],
            "failed_agents": ["gpu-a/" + NODE],
            "failures": {"gpu-a/" + NODE: "ValueError: agent generation changed"},
        }

    monkeypatch.setattr(cli, "load_site", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(departed, "run_retire_departed_agents", fake_run)
    state_dir = _state_dir(tmp_path)

    exit_code = cli.run(
        cli.parser().parse_args(
            [
                "workflow-reconcile",
                "--state-dir",
                str(state_dir),
                "--retire-departed-agents",
                "--node",
                NODE,
                "--reference",
                REFERENCE,
            ]
        )
    )

    assert exit_code == 1, "a failed item exits non-zero"
    (call,) = calls
    assert call["state_dir"] == state_dir.resolve()
    assert call["node_ids"] == (NODE,)
    assert call["reference"] == REFERENCE
    assert call["dry_run"] is False
    printed = json.loads(capsys.readouterr().out)
    assert printed["failed_agents"] == ["gpu-a/" + NODE]


@pytest.mark.parametrize(
    "other",
    [
        ["--workflow-id", "workflow-a"],
        ["--incident-id", "incident-a"],
        ["--max-items", "3"],
        ["--close-escalated"],
        ["--close-incident", "incident-a"],
    ],
)
def test_the_flag_refuses_the_selectors_of_the_other_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other: list[str]
) -> None:
    monkeypatch.setattr(cli, "load_site", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        departed,
        "run_retire_departed_agents",
        lambda *_args, **_kwargs: pytest.fail("the runner must not be reached"),
    )

    with pytest.raises(SiteConfigError, match="takes --node, --reference"):
        cli.run(
            cli.parser().parse_args(
                [
                    "workflow-reconcile",
                    "--state-dir",
                    str(_state_dir(tmp_path)),
                    "--retire-departed-agents",
                    "--dry-run",
                    *other,
                ]
            )
        )
