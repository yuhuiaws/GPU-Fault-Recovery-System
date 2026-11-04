"""A join of cluster B while cluster A has lost a node: the join completes.

Live 2026-11-04 (join-cluster of a HyperPod cluster, 12:09-12:32Z): an EC2
Spot interruption took a plain EKS GPU node at 12:15Z. The join's final
``verify`` failed on ``control_api: gpu-eks-cluster Agent coverage drift``
(the departed node's Agent still ACTIVE, inside the product's departure
guard), ``gpu-fault-admin`` reported ``regional verify failed`` and rolled the
HyperPod join back (``fail-cluster``, ``rollback-cluster``).

Here the join runs end to end over the scenario fakes, and its ``verify`` is
the real release engine, in process, over a fake fleet of both clusters:
cluster ``gpu-a`` holds an ACTIVE Agent for a node Kubernetes no longer lists
and whose departure is not provable yet. The engine's verify scoped to the
joining cluster reports that and gates on the shared control plane and the
joining cluster only, so the join is not rolled back and its result names the
finding. Only modules that existed before the fix are imported, so the same
file shows the old behaviour on the old tree: the join fails and rolls back.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_release_departure_absorption as ABSORPTION
from gpu_fault_release import regional_release_verify_churn as CHURN
from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_release_departed_nodes import DepartureProof
from tests.admin._cov95_join_support import JoinScenario
from tests.regional.test_regional_admin_checks import healthy_control_api_report
from tests.regional.test_release_churn_verify import OTHER_CHECKS
from tests.regional.test_release_verify_departure_wait import (
    engine_verify,
    two_cluster_release,
)

OTHER = "gpu-a"
JOINING = "hp-gpu-b"
OTHER_NODE = "node-a"
DEPARTED = "ip-10-0-38-176.us-west-2.compute.internal"
JOINING_NODE = "hyperpod-i-0000000000b1"


def _node(name: str) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "labels": {}, "annotations": {}},
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }


def _probe(_release: Any, *, arguments: tuple[str, ...], **_kwargs: Any) -> str:
    """The control API: gpu-a still holds the departed node's ACTIVE Agent."""

    expected = next(
        json.loads(item.split("=", 1)[1])
        for item in arguments
        if item.startswith("ADMIN_EXPECTED_NODES_JSON=")
    )
    agents = {OTHER: [OTHER_NODE, DEPARTED], JOINING: [JOINING_NODE]}
    report = healthy_control_api_report()
    report["registry"] = [{"cluster_id": cluster} for cluster in agents]
    report["clusters"] = {
        cluster: {
            "expected_node_ids": expected[cluster],
            "agents": [
                {
                    "node_id": name,
                    "lifecycle_state": "ACTIVE",
                    "runtime_profile_version": "hyperpod-v1",
                }
                for name in names
            ],
            "fleet_readiness": {"ready": True},
            "collector_readiness": {"ready": True},
        }
        for cluster, names in agents.items()
    }
    return json.dumps(report)


def _unprovable(_release: Any, _target: Any, nodes: list[str], **_kw: Any) -> Any:
    return DepartureProof(
        (),
        tuple(
            f"{node}: Agent record is ACTIVE with a live lease (expires soon)"
            for node in nodes
        ),
    )


def _engine_fleet(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The engine's verify over both clusters; every check but control_api passes."""

    for name in (*OTHER_CHECKS, "_check_gpu_cluster"):
        monkeypatch.setattr(
            CHECKS, name, lambda *_args, name=name, **_kwargs: CheckValue(name)
        )
    nodes = {OTHER: [_node(OTHER_NODE)], JOINING: [_node(JOINING_NODE)]}
    monkeypatch.setattr(
        CHECKS,
        "selected_nodes",
        lambda _release, target: ("selector", {"items": nodes[target.cluster_id]}),
    )
    monkeypatch.setattr(CHECKS, "exec_cpu_ingress", _probe)
    for module in (CHURN, ABSORPTION):
        monkeypatch.setattr(module, "departure_proof", _unprovable)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(CHURN, "_monotonic", lambda: clock.now, raising=False)
    monkeypatch.setattr(
        CHURN,
        "_sleep",
        lambda seconds: setattr(clock, "now", clock.now + seconds),
        raising=False,
    )
    monkeypatch.setattr(
        CHURN,
        "_fresh_nodes",
        lambda _release, target: nodes[target.cluster_id],
        raising=False,
    )
    return two_cluster_release(OTHER, JOINING)


@pytest.fixture
def scenario(tmp_path, monkeypatch) -> tuple[JoinScenario, list[list[str]], list[str]]:
    scenario = JoinScenario(tmp_path, monkeypatch, suffixes=("b",), rollback=True)
    release = _engine_fleet(monkeypatch)
    verifies: list[list[str]] = []
    rolled_back: list[str] = []

    def driver(arguments: list[str], **options: Any) -> Any:
        if arguments[1] != "verify":
            return scenario.driver(arguments, **options)
        scenario.driver(arguments, **options)  # the scenario's own bookkeeping
        verifies.append(list(arguments[1:]))
        code, report = engine_verify(release, *arguments[4:])
        return subprocess.CompletedProcess(arguments, code, json.dumps(report), "")

    monkeypatch.setattr(join, "run_driver", driver)
    monkeypatch.setattr(
        join, "_rollback", lambda *_args, **_kwargs: rolled_back.append("rolled back")
    )
    return scenario, verifies, rolled_back


def test_a_join_is_not_rolled_back_by_another_clusters_departed_node(scenario):
    """Before (ad048f18): ``regional verify failed`` and the join rolled back."""

    joins, verifies, rolled_back = scenario

    result = joins.join()

    assert rolled_back == [], "another cluster's node departure rolled the join back"
    assert result["phase"] == "COMPLETED", result
    assert verifies == [
        ["verify", "--config", verifies[0][2], "--joining-cluster-id", JOINING]
    ], "the join's verify is scoped to the joining cluster"
    findings = result["not_gating_findings"]
    assert [(item["cluster_id"], item["check"]) for item in findings] == [
        (OTHER, "control_api")
    ]
    assert findings[0]["finding"].startswith(f"{OTHER} Agent coverage drift"), findings
    evidence = joins.state()["evidence"]["VERIFIED"]
    assert evidence["not_gating_findings"] == findings, "kept with the VERIFIED proof"
    record = json.loads(
        (joins.path.parent / "membership-verify" / "findings.json").read_text()
    )
    assert (record["verb"], record["cluster_id"]) == ("join-cluster", JOINING)
    assert record["not_gating_findings"] == findings, "the record status prints"


def test_the_joining_clusters_own_failure_still_rolls_the_join_back(
    scenario, monkeypatch
):
    """Only the scope changed: a finding on the joining cluster still fails it."""

    joins, _verifies, rolled_back = scenario

    def joining_cluster_down(_release: Any, target: Any, **_kw: Any) -> CheckValue:
        if target.cluster_id == JOINING:
            raise BootstrapError(f"{JOINING} TLS probe did not return ok")
        return CheckValue(f"{target.cluster_id} data plane is healthy")

    monkeypatch.setattr(CHECKS, "_check_gpu_cluster", joining_cluster_down)

    with pytest.raises(BootstrapError, match="regional verify failed"):
        joins.join()

    assert rolled_back == ["rolled back"]
    assert joins.state()["failure"]["after_step"] == "COLLECTORS_READY"
