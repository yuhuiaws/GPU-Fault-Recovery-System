"""The verify's departure wait: an Agent whose Node is going, where coverage gates.

The verify judged an Agent without a Ready Node once: absent from Kubernetes
and provable at that moment, it was excused; anything else -- a Node still
listed NotReady, EC2 still ``shutting-down``, a lease not lapsed yet -- failed
coverage at once, and a deploy rolled back on it (live 2026-11-04 the Node of a
spot-reclaimed EKS GPU node was still listed five minutes after the
interruption). Now, in a cluster whose findings gate the verify -- the target
of a join, every cluster of a ``deploy`` -- such an Agent is re-judged every
30 s for at most the product's ten-minute departure guard: excused with its
evidence once Kubernetes no longer lists it and the proof holds, counted (and
named with the reason) once the guard has passed, and never excused while its
Node is Ready again.

Only modules that existed before the change are imported here (the join's
scope is asked for on the engine's command line), so this file also runs on
the old tree, where these departures fail the verify.
"""

from __future__ import annotations

import contextlib
import io
import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded
from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_release_departure_absorption as ABSORPTION
from gpu_fault_release import regional_release_verify_churn as CHURN
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_check_value import CheckValue
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_departed_nodes import DepartureProof
from tests.regional.test_regional_admin_checks import (
    control_api_release,
    healthy_control_api_report,
)
from tests.regional.test_release_churn_verify import OTHER_CHECKS

#: The guard the wait is bounded by (``DEPARTURE_DECISION_SECONDS``).
GUARD_SECONDS = 600.0
PRESENT = "the Node is still in Kubernetes"

ARTIFACT = "a" * 64
EKS = "gpu-eks-cluster"
HP = "hp-cluster-join"
EKS_NODE = "ip-10-0-44-18.us-west-2.compute.internal"
EKS_GONE = "ip-10-0-38-176.us-west-2.compute.internal"
HP_NODE = "hyperpod-i-0000000000b1"
HP_GOING = "hyperpod-i-0000000000b2"


def node(name: str, *, ready: bool = True) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "labels": {},
            "annotations": {
                "gpu-fault.io/installer-state": "Succeeded",
                "gpu-fault.io/installer-artifact-sha256": ARTIFACT,
            },
        },
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}]
        },
    }


class ControlPlane:
    """The control API probe: per cluster the Agents it holds, healthy otherwise;
    ``stale`` names the clusters whose collectors are not ready."""

    def __init__(
        self, agents: dict[str, list[str]], *, stale: frozenset[str] = frozenset()
    ) -> None:
        self.agents = agents
        self.stale = stale
        self.asked: dict[str, list[str]] = {}
        self.registry = list(agents)

    def __call__(self, _release: Any, *, arguments: tuple[str, ...], **_: Any) -> str:
        self.asked = next(
            json.loads(item.split("=", 1)[1])
            for item in arguments
            if item.startswith("ADMIN_EXPECTED_NODES_JSON=")
        )
        report = healthy_control_api_report()
        report["registry"] = [{"cluster_id": cluster} for cluster in self.registry]
        report["clusters"] = {
            cluster: {
                "expected_node_ids": self.asked[cluster],
                "agents": [
                    {
                        "node_id": name,
                        "lifecycle_state": "ACTIVE",
                        "runtime_profile_version": "hyperpod-v1",
                    }
                    for name in names
                ],
                "fleet_readiness": {"ready": True},
                "collector_readiness": {
                    "ready": cluster not in self.stale,
                    "nodes": [
                        {"node_id": name, "ready": cluster not in self.stale}
                        for name in names
                    ],
                },
            }
            for cluster, names in self.agents.items()
        }
        return json.dumps(report)


def two_cluster_release(*cluster_ids: str) -> SimpleNamespace:
    """The engine's release for a site of ``cluster_ids`` (EKS and HP)."""

    release = control_api_release()
    release.config.site_name = "test-site"
    release.config.clusters = [
        SimpleNamespace(cluster_id=cluster_id)
        for cluster_id in cluster_ids or (EKS, HP)
    ]
    release.release_id = "candidate"
    release._apply_health_baseline = lambda _state: None
    return release


def health(
    monkeypatch: pytest.MonkeyPatch,
    release: Any,
    *,
    nodes: dict[str, list[dict[str, Any]]],
    plane: ControlPlane,
    gpu_cluster: Any = None,
) -> dict[str, Any]:
    """The verify report with every other check passing by fiat."""

    for name in OTHER_CHECKS:
        monkeypatch.setattr(
            CHECKS, name, lambda *_args, name=name, **_kwargs: CheckValue(name)
        )
    monkeypatch.setattr(
        CHECKS,
        "_check_gpu_cluster",
        gpu_cluster
        or (lambda _release, target, **_kw: CheckValue(f"{target.cluster_id} ok")),
    )
    monkeypatch.setattr(
        CHECKS,
        "selected_nodes",
        lambda _release, target: ("selector", {"items": nodes[target.cluster_id]}),
    )
    monkeypatch.setattr(CHECKS, "exec_cpu_ingress", plane)
    return CHECKS.build_health_report(release, mode="verify")


def by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in report["checks"]}


def unprovable(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """No departure proof holds (the Agent's lease is still live)."""

    asked: list[list[str]] = []

    def proof(_release: Any, _target: Any, nodes: list[str], **_kw: Any) -> Any:
        asked.append(list(nodes))
        return DepartureProof(
            (),
            tuple(
                f"{item}: Agent record is ACTIVE with a live lease" for item in nodes
            ),
        )

    monkeypatch.setattr(CHURN, "departure_proof", proof)
    monkeypatch.setattr(ABSORPTION, "departure_proof", proof)
    return asked


def proves_gone(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The departure proof holds for every node Kubernetes no longer lists."""

    asked: list[list[str]] = []

    def proof(
        _release: Any, _target: Any, nodes: list[str], *, present_node_names: Any = None
    ) -> Any:
        asked.append(list(nodes))
        present = set(present_node_names or ())
        rows = tuple(
            {
                "node_id": item,
                "provider": "eks-managed",
                "instance_id": "i-00000000000000001",
                "kubernetes": {"present": False},
                "provider_inventory": {"source": "ec2", "state": "terminated"},
                "agent": {"lifecycle_state": "ACTIVE", "lease_live": False},
                "retire_hint": None,
            }
            for item in nodes
            if item not in present
        )
        return DepartureProof(rows, ())

    monkeypatch.setattr(CHURN, "departure_proof", proof)
    monkeypatch.setattr(ABSORPTION, "departure_proof", proof)
    return asked


def departure_clock(
    monkeypatch: pytest.MonkeyPatch, reads: list[list[dict[str, Any]]]
) -> SimpleNamespace:
    """The verify's departure wait on a fake clock; ``reads`` are the fresh
    Node lists it sees in turn, the last one repeating."""

    clock = SimpleNamespace(now=0.0, reads=0)

    def sleep(seconds: float) -> None:
        clock.now += seconds

    def fresh(_release: Any, _target: Any) -> list[dict[str, Any]]:
        clock.reads += 1
        return reads[min(clock.reads, len(reads)) - 1]

    monkeypatch.setattr(CHURN, "_monotonic", lambda: clock.now, raising=False)
    monkeypatch.setattr(CHURN, "_sleep", sleep, raising=False)
    monkeypatch.setattr(CHURN, "_fresh_nodes", fresh, raising=False)
    return clock


def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_seconds: float) -> None:
        raise AssertionError("the verify waited")

    monkeypatch.setattr(CHURN, "_sleep", refuse, raising=False)


def engine_verify(release: Any, *scope: str) -> tuple[int, dict[str, Any]]:
    """The engine's ``verify`` mode, as the driver runs it, with ``scope``
    (``--joining-cluster-id <id>`` for a join's)."""

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = ROLLOUT._dispatch_mode(
            release,
            ROLLOUT.parse_arguments(["verify", "--config", "site.json", *scope]),
        )
    return code, json.loads(output.getvalue())


def verified(
    monkeypatch: pytest.MonkeyPatch,
    release: Any,
    *scope: str,
    nodes: dict[str, list[dict[str, Any]]],
    plane: Any,
) -> tuple[int, dict[str, Any]]:
    """``engine_verify`` with every check but ``control_api`` passing by fiat."""

    for name in (*OTHER_CHECKS, "_check_gpu_cluster"):
        monkeypatch.setattr(
            CHECKS, name, lambda *_args, name=name, **_kwargs: CheckValue(name)
        )
    monkeypatch.setattr(
        CHECKS,
        "selected_nodes",
        lambda _release, target: ("selector", {"items": nodes[target.cluster_id]}),
    )
    monkeypatch.setattr(CHECKS, "exec_cpu_ingress", plane)
    return engine_verify(release, *scope)


def test_a_departing_node_on_the_target_is_excused_once_proven_with_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before: the NotReady Node still listed at the report failed coverage at
    once; now the verify re-reads it and excuses it once it is proven gone."""

    asked = proves_gone(monkeypatch)
    listed = [node(HP_NODE), node(HP_GOING, ready=False)]
    clock = departure_clock(monkeypatch, [listed, [node(HP_NODE)]])
    plane = ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE, HP_GOING]})

    code, report = verified(
        monkeypatch,
        two_cluster_release(),
        "--joining-cluster-id",
        HP,
        nodes={EKS: [node(EKS_NODE)], HP: listed},
        plane=plane,
    )

    record = by_name(report)["control_api"]
    assert (code, report["healthy"]) == (0, True), record
    assert record["status"] == "WARN", record
    assert f"{HP} departed (proven by this release): {HP_GOING}" in record["summary"]
    churn = record["details"]["clusters"][HP]["release_churn"]
    assert churn["proven_by_verify"] == [HP_GOING]
    assert [row["node_id"] for row in churn["departure_evidence"]] == [HP_GOING]
    assert churn["departure_evidence"][0]["provider_inventory"]["state"] == "terminated"
    assert "departure_unproven" not in churn
    assert asked == [[HP_GOING]], "proven once Kubernetes no longer listed it"
    assert 0 < clock.now < GUARD_SECONDS


def test_a_node_still_listed_past_the_guard_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proves_gone(monkeypatch)
    listed = [node(HP_NODE), node(HP_GOING, ready=False)]
    clock = departure_clock(monkeypatch, [listed])
    plane = ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE, HP_GOING]})

    code, report = verified(
        monkeypatch,
        two_cluster_release(),
        "--joining-cluster-id",
        HP,
        nodes={EKS: [node(EKS_NODE)], HP: listed},
        plane=plane,
    )

    record = by_name(report)["control_api"]
    assert (code, record["status"]) == (1, "FAIL"), record
    assert (
        f"departure not proven: {HP_GOING}: the Node is still in Kubernetes (NotReady)"
        in record["summary"]
    ), record
    assert clock.now >= GUARD_SECONDS


def test_a_node_that_is_ready_again_is_not_excused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proves_gone(monkeypatch)
    listed = [node(HP_NODE), node(HP_GOING, ready=False)]
    clock = departure_clock(monkeypatch, [[node(HP_NODE), node(HP_GOING)]])
    plane = ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE, HP_GOING]})

    code, report = verified(
        monkeypatch,
        two_cluster_release(),
        "--joining-cluster-id",
        HP,
        nodes={EKS: [node(EKS_NODE)], HP: listed},
        plane=plane,
    )

    record = by_name(report)["control_api"]
    assert (code, record["status"]) == (1, "FAIL"), record
    assert (
        f"departure not proven: {HP_GOING}: the Node is Ready again"
        in (record["summary"])
    ), record
    assert clock.reads == 1, "the wait ends at the first look that decides"


def test_a_deploy_verify_waits_out_a_departing_node_on_any_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deploy forward path's verify (no scope, every cluster gates): the
    live shape -- NotReady and listed at the report -- no longer rolls back."""

    proves_gone(monkeypatch)
    listed = [node(EKS_NODE), node(EKS_GONE, ready=False)]
    departure_clock(monkeypatch, [listed, listed, [node(EKS_NODE)]])
    plane = ControlPlane({EKS: [EKS_NODE, EKS_GONE], HP: [HP_NODE]})

    code, report = verified(
        monkeypatch,
        two_cluster_release(),
        nodes={EKS: listed, HP: [node(HP_NODE)]},
        plane=plane,
    )

    record = by_name(report)["control_api"]
    assert (code, report["healthy"]) == (0, True), record
    assert f"{EKS} departed (proven by this release): {EKS_GONE}" in record["summary"]
    assert "cluster_scope" not in report


def clock_only(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds: float) -> None:
        clock.now += seconds

    monkeypatch.setattr(CHURN, "_monotonic", lambda: clock.now, raising=False)
    monkeypatch.setattr(CHURN, "_sleep", sleep, raising=False)
    return clock


def test_the_wait_reads_the_api_server_past_the_report_and_rechecks_the_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report's Node list is a snapshot; the wait asks the API server each
    look, and keeps only the Nodes the cluster's selector really selects."""

    proves_gone(monkeypatch)
    clock_only(monkeypatch)
    selector = "sagemaker.amazonaws.com/cluster-name=hp-join"
    labels = {"sagemaker.amazonaws.com/cluster-name": "hp-join"}
    stranger = {
        **node(HP_GOING, ready=False),
        "metadata": {"name": HP_GOING, "labels": {"other": "cluster"}},
    }
    answers = [
        [
            {**node(HP_NODE), "metadata": {"name": HP_NODE, "labels": labels}},
            {
                **node(HP_GOING, ready=False),
                "metadata": {"name": HP_GOING, "labels": labels},
            },
        ],
        [{**node(HP_NODE), "metadata": {"name": HP_NODE, "labels": labels}}, stranger],
    ]
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class Runner:
        def run(self, arguments: list[str], **options: Any) -> str:
            calls.append((list(arguments), options))
            return json.dumps({"items": answers[min(len(calls), 2) - 1]})

    release = two_cluster_release()
    release.config.clusters[1] = SimpleNamespace(
        cluster_id=HP, hyperpod_cluster_name="hp-join"
    )
    release.runner = Runner()
    release._gpu = lambda target, *arguments: ["kubectl", "--context", HP, *arguments]
    listed = [node(HP_NODE), node(HP_GOING, ready=False)]

    code, report = verified(
        monkeypatch,
        release,
        "--joining-cluster-id",
        HP,
        nodes={EKS: [node(EKS_NODE)], HP: listed},
        plane=ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE, HP_GOING]}),
    )

    assert code == 0, report
    assert [arguments for arguments, _options in calls] == [
        ["kubectl", "--context", HP, "get", "nodes", "-l", selector, "-o", "json"]
    ] * 2
    assert all(options.get("capture") is True for _arguments, options in calls), calls
    churn = by_name(report)["control_api"]["details"]["clusters"][HP]["release_churn"]
    assert churn["proven_by_verify"] == [HP_GOING], (
        "a Node outside the selector is gone"
    )


def test_a_proof_unavailable_at_the_first_look_is_taken_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[list[str]] = []
    gone = proves_gone(monkeypatch)
    proof = CHURN.departure_proof

    def throttled_once(*args: Any, **kwargs: Any) -> Any:
        attempts.append(list(args[2]))
        if len(attempts) == 1:
            raise ReleaseError("aws ec2 describe-instances: ThrottlingException")
        return proof(*args, **kwargs)

    monkeypatch.setattr(CHURN, "departure_proof", throttled_once)
    monkeypatch.setattr(ABSORPTION, "departure_proof", throttled_once)
    departure_clock(monkeypatch, [[node(HP_NODE)]])

    code, report = verified(
        monkeypatch,
        two_cluster_release(),
        "--joining-cluster-id",
        HP,
        nodes={EKS: [node(EKS_NODE)], HP: [node(HP_NODE)]},
        plane=ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE, HP_GOING]}),
    )

    assert code == 0, report
    assert attempts == [[HP_GOING], [HP_GOING]], "throttled, then proven"
    assert gone == [[HP_GOING]]


def test_a_node_list_that_cannot_be_read_decides_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proves_gone(monkeypatch)
    clock = clock_only(monkeypatch)

    def unreadable(_release: Any, _target: Any) -> list[dict[str, Any]]:
        raise ReleaseError("kubectl get nodes: Unable to connect to the server")

    monkeypatch.setattr(CHURN, "_fresh_nodes", unreadable, raising=False)
    listed = [node(HP_NODE), node(HP_GOING, ready=False)]

    code, report = verified(
        monkeypatch,
        two_cluster_release(),
        "--joining-cluster-id",
        HP,
        nodes={EKS: [node(EKS_NODE)], HP: listed},
        plane=ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE, HP_GOING]}),
    )

    record = by_name(report)["control_api"]
    assert (code, record["status"]) == (1, "FAIL"), record
    assert (
        f"{HP_GOING}: departure undecided: kubectl get nodes: Unable to connect"
        in (record["summary"])
    ), record
    assert clock.now >= GUARD_SECONDS


def test_a_verify_out_of_time_does_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    def exhausted(_seconds: float) -> float:
        raise DeploymentDeadlineExceeded("release/verify exceeded its deadline")

    proves_gone(monkeypatch)
    no_wait(monkeypatch)
    monkeypatch.setattr(CHURN, "remaining_timeout", exhausted, raising=False)
    listed = [node(HP_NODE), node(HP_GOING, ready=False)]

    code, report = verified(
        monkeypatch,
        two_cluster_release(),
        "--joining-cluster-id",
        HP,
        nodes={EKS: [node(EKS_NODE)], HP: listed},
        plane=ControlPlane({EKS: [EKS_NODE], HP: [HP_NODE, HP_GOING]}),
    )

    record = by_name(report)["control_api"]
    assert (code, record["status"]) == (1, "FAIL"), record
    assert f"departure not proven: {HP_GOING}: {PRESENT}" in record["summary"]
