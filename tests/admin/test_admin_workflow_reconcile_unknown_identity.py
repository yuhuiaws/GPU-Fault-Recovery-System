"""``--retire-departed-agents --node N --accept-unknown-provider-identity``.

The administrator's exit for a node whose provider identity is unknown -- an
eks-managed Agent without an EC2 instance id, a HyperPod node name without an
instance, or no Agent record at all (live: DESTR-020's alias on
gpu-eks-cluster, which the departed-node cleanup could only refuse). These run
the CLI against the real in-Pod script on an in-memory control plane, with
``kubectl`` answered by a double: the plan, its refusals, and the apply that
retires the Agent (if any) and records the acceptance the cleanup then ends
the rest on.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin import workflow_reconcile_departed_agents as departed
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.departed_nodes import DepartedNodeCleanupStatus
from gpu_fault.fleet import AgentLifecycleState, FleetRegistry
from gpu_fault.models import IncidentState
from tests._builders import build_store
from tests.admin.conftest import TEST_OPERATOR_ARN
from tests.departed_nodes._support import (
    EKS_CLUSTER,
    EKS_PRESENT,
    agent,
    ago,
    blocked_never_changed,
    registration,
    world,
)

ALIAS = "acceptance-alias-0a0b0c0d0e0f"
REFERENCE = "CHG-2026-1014-07"
REASON = "DESTR-020 alias: never a Kubernetes node"


def _site(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "cpu_eks_arn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu-control",
            "clusters": [
                {
                    "cluster_id": EKS_CLUSTER,
                    "context": "gpu-eks-context",
                    "provider": "eks-managed",
                    "region": "us-west-2",
                }
            ],
        },
        environment={"KUBECONFIG": str(tmp_path / "gpu.kubeconfig")},
        source_sha256="a" * 64,
    )


class ControlPlane:
    """The CPU Pod: the real ``AGENT_SCRIPT`` on an in-memory store."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.store = build_store()
        self.registry = FleetRegistry(self.store, "s" * 32)
        self.world = world(self.store)
        self.store.save_regional_cluster(
            registration(EKS_CLUSTER, provider="eks-managed")
        )
        self.payloads: list[dict[str, Any]] = []
        monkeypatch.setattr(departed, "run_pod_script", self.run)

    def run(self, _site: object, payload: dict[str, Any]) -> dict[str, Any]:
        from gpu_fault.app import ApplicationContext

        self.payloads.append(payload)
        context = ApplicationContext(store=self.store, fleet_registry=self.registry)
        output = io.StringIO()
        with self.monkeypatch.context() as scoped:
            scoped.setattr(
                ApplicationContext, "from_environment", classmethod(lambda cls: context)
            )
            scoped.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
            scoped.setattr(sys, "stdout", output)
            exec(compile(departed.AGENT_SCRIPT, "<pod>", "exec"), {})
        return dict(json.loads(output.getvalue()))


def _kubectl(monkeypatch: pytest.MonkeyPatch, *nodes: str) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command: list[str], **_options: object) -> SimpleNamespace:
        calls.append(list(command))
        assert command[0] == "kubectl", "an unknown identity is never put to AWS"
        items = [
            {
                "metadata": {"name": name, "annotations": {}},
                "spec": {"unschedulable": False, "taints": []},
            }
            for name in nodes
        ]
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"items": items}), stderr=""
        )

    monkeypatch.setattr(reconcile, "run_command", run)
    return calls


def _alias(plane: ControlPlane, *, with_agent: bool) -> None:
    """DESTR-020's residue, and the ledger line the cleanup wrote for it."""

    if with_agent:
        plane.store.save_agent(
            agent(EKS_CLUSTER, ALIAS, instance_id=None, last_seen=ago(minutes=40))
        )
    blocked_never_changed(plane.store, EKS_CLUSTER, ALIAS, suffix="alias")
    plane.world.reports_since(EKS_CLUSTER, [EKS_PRESENT], minutes=15)
    [line] = plane.world.cleanup.run_once()
    assert line.blocker_classes == ["provider_identity_unknown"], line.blockers


def _run(tmp_path: Path, **values: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "node_ids": (ALIAS,),
        "accept_unknown_provider": True,
        "reason": REASON,
        "reference": REFERENCE,
        "dry_run": False,
        **values,
    }
    return departed.run_retire_departed_agents(_site(tmp_path), tmp_path, **options)


@pytest.mark.parametrize("with_agent", [True, False])
def test_the_alias_is_accepted_and_the_cleanup_ends_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_agent: bool
) -> None:
    plane = ControlPlane(monkeypatch)
    _alias(plane, with_agent=with_agent)
    kubectl = _kubectl(monkeypatch, EKS_PRESENT)

    result = _run(tmp_path)
    [line] = plane.world.cleanup.run_once()

    assert result["failed_agents"] == [], result["failures"]
    assert result["accepted_identities"] == [f"{EKS_CLUSTER}/{ALIAS}"]
    [acceptance] = plane.store.list_departed_node_identity_acceptances(EKS_CLUSTER)
    assert acceptance.reason == REASON and acceptance.change_reference == REFERENCE
    assert acceptance.actor == TEST_OPERATOR_ARN
    assert acceptance.kubernetes["never_reported"] is True
    if with_agent:
        revoked = plane.store.get_agent(EKS_CLUSTER, ALIAS)
        assert revoked.lifecycle_state is AgentLifecycleState.REVOKED
        assert revoked.transition_id == result["retired_agents"][0]["transition_id"]
        assert "provider identity is unknown" in str(revoked.transition_reason)
        assert REASON in str(revoked.transition_reason)
    else:
        assert result["retired_agents"] == [], "no Agent record to retire"
    assert line.status is DepartedNodeCleanupStatus.CLEANED, line.blockers
    assert plane.store.get_incident("incident-alias").state is IncidentState.RECOVERED
    assert kubectl, "Kubernetes is read through the site's GPU kubeconfig"
    archive = tmp_path / departed.HISTORY_PATH / result["plan_sha256"]
    assert (archive / "plan.json").is_file() and (archive / "applied.json").is_file()


def test_the_dry_run_shows_the_verdict_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plane = ControlPlane(monkeypatch)
    _alias(plane, with_agent=True)
    _kubectl(monkeypatch, EKS_PRESENT)

    plan = _run(tmp_path, dry_run=True, reason=None, reference=None)

    [item] = plan["items"]
    assert plan["mode"] == "accept-unknown-provider-identity-plan"
    assert item["eligible"] is True, item["reasons"]
    assert item["retire"] is True
    assert item["kubernetes_reports"] == {"open": True, "never_reported": True}
    assert "kubernetes_reports_read" not in departed.plan_digest_items([item])[0]
    assert plane.store.list_departed_node_identity_acceptances() == []
    assert plane.store.get_agent(EKS_CLUSTER, ALIAS).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )


def _refusal(tmp_path: Path, **values: Any) -> str:
    with pytest.raises(BootstrapError) as refused:
        _run(tmp_path, **values)
    return str(refused.value)


def test_a_node_that_left_the_reports_an_hour_ago_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plane = ControlPlane(monkeypatch)
    plane.store.save_agent(
        agent(EKS_CLUSTER, ALIAS, instance_id=None, last_seen=ago(hours=2))
    )
    for minute in range(70, 60, -1):
        plane.world.report(EKS_CLUSTER, [ALIAS, EKS_PRESENT], at=ago(minutes=minute))
    plane.world.reports_since(EKS_CLUSTER, [EKS_PRESENT], minutes=60)
    _kubectl(monkeypatch, EKS_PRESENT)

    message = _refusal(tmp_path)

    assert "accepted 24 hours after that" in message, message
    assert plane.store.list_departed_node_identity_acceptances() == []


@pytest.mark.parametrize(
    ("instance_id", "last_seen", "kubernetes", "match"),
    [
        pytest.param(
            "i-0e000009",
            {"hours": 1},
            (EKS_PRESENT,),
            "provider identity is known",
            id="known-identity",
        ),
        pytest.param(
            None,
            {"minutes": 2},
            (EKS_PRESENT,),
            "younger than the 10-minute departure guard",
            id="live-agent",
        ),
        pytest.param(
            None,
            {"hours": 1},
            (EKS_PRESENT, ALIAS),
            "still in Kubernetes",
            id="node-in-kubernetes",
        ),
    ],
)
def test_each_unmet_condition_refuses_the_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instance_id: str | None,
    last_seen: dict[str, float],
    kubernetes: tuple[str, ...],
    match: str,
) -> None:
    plane = ControlPlane(monkeypatch)
    plane.store.save_agent(
        agent(
            EKS_CLUSTER,
            ALIAS,
            instance_id=instance_id,
            last_seen=ago(**last_seen),
            lease_seconds=30,
        )
    )
    plane.world.reports_since(EKS_CLUSTER, [EKS_PRESENT], minutes=15)
    _kubectl(monkeypatch, *kubernetes)

    message = _refusal(tmp_path)

    assert match in message, message
    assert plane.store.get_agent(EKS_CLUSTER, ALIAS).lifecycle_state is (
        AgentLifecycleState.ACTIVE
    )


def test_a_node_nothing_names_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plane = ControlPlane(monkeypatch)
    plane.world.reports_since(EKS_CLUSTER, [EKS_PRESENT], minutes=15)
    _kubectl(monkeypatch, EKS_PRESENT)

    assert "no Agent record and no departed-node ledger line" in _refusal(tmp_path)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        pytest.param({"node_ids": ()}, "needs --node", id="no-node"),
        pytest.param({"reason": "  "}, "requires --reason", id="no-reason"),
        pytest.param({"reason": "x" * 513}, "at most 512", id="long-reason"),
    ],
)
def test_the_inputs_are_refused_before_the_control_plane_is_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values: dict[str, Any], match: str
) -> None:
    monkeypatch.setattr(
        departed, "run_pod_script", lambda *_a, **_k: pytest.fail("unreachable")
    )

    assert match in _refusal(tmp_path, **values)


def test_the_flag_rides_workflow_reconcile_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake(site: object, state_dir: Path, **values: Any) -> dict[str, Any]:
        seen.update(values)
        return {"failed_agents": []}

    monkeypatch.setattr(departed, "run_retire_departed_agents", fake)
    arguments = cli.parser().parse_args(
        [
            "workflow-reconcile",
            "--state-dir",
            str(tmp_path),
            "--retire-departed-agents",
            "--node",
            ALIAS,
            "--accept-unknown-provider-identity",
            "--reason",
            REASON,
            "--reference",
            REFERENCE,
        ]
    )

    departed.retire_departed_agents_command(
        arguments, site=_site(tmp_path), state_dir=tmp_path
    )

    assert seen["accept_unknown_provider"] is True
    assert seen["reason"] == REASON
    assert seen["node_ids"] == (ALIAS,)


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        pytest.param(
            ["--accept-unknown-provider-identity", "--node", ALIAS],
            "taken by --retire-departed-agents only",
            id="accept-without-retire",
        ),
        pytest.param(
            ["--retire-departed-agents", "--node", ALIAS, "--reason", REASON],
            "--reason without --accept-unknown-provider-identity",
            id="reason-without-accept",
        ),
    ],
)
def test_a_misplaced_flag_is_refused_not_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], match: str
) -> None:
    monkeypatch.setattr(
        departed, "run_pod_script", lambda *_a, **_k: pytest.fail("unreachable")
    )
    arguments = cli.parser().parse_args(
        ["workflow-reconcile", "--state-dir", str(tmp_path), *flags]
    )

    assert departed.selected(arguments), "workflow-reconcile routes it here"
    with pytest.raises(BootstrapError, match=match):
        departed.retire_departed_agents_command(
            arguments, site=_site(tmp_path), state_dir=tmp_path
        )
