"""The surge steering tool: plans the deletion order of a surge-first roll
from the preflight's snapshots, annotates the old Pods, re-steers from where
the new Pods actually landed and explains an expired rollout wait.

The kubectl command is a recording fake; the model is the real one."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "deploy/control-plane/tools"
MODULE = lazy_script_module(TOOLS / "control_plane_surge_steering.py")
GI = 1024**3
COST = "controller.kubernetes.io/pod-deletion-cost"
NODES = ("hyperpod-cpu-a", "hyperpod-cpu-b", "hyperpod-cpu-c")
OLD_HASH, NEW_HASH = "7c9d5b8f4", "55767f6fbd"
OLD = {
    NODES[0]: f"gpu-fault-api-ha-{OLD_HASH}-llr52",
    NODES[1]: f"gpu-fault-api-ha-{OLD_HASH}-876dt",
    NODES[2]: f"gpu-fault-api-ha-{OLD_HASH}-ss92d",
}
DEPLOYMENT = "gpu-fault-api-ha"


def _node(name: str) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "labels": {"kubernetes.io/hostname": name}},
        "spec": {},
        "status": {
            "allocatable": {"cpu": "15890m", "memory": "30Gi"},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }


def _pod(
    name: str,
    node: str | None,
    cpu: str,
    memory: str,
    app: str = "other",
    *,
    template_hash: str = "",
    annotations: dict[str, str] | None = None,
    phase: str = "Running",
    conditions: list[dict[str, str]] | None = None,
    terminating: bool = False,
) -> dict[str, Any]:
    labels = {"app": app}
    if template_hash:
        labels["pod-template-hash"] = template_hash
    metadata: dict[str, Any] = {"name": name, "labels": labels}
    if annotations:
        metadata["annotations"] = annotations
    if terminating:
        metadata["deletionTimestamp"] = "2000-01-01T00:00:00Z"
    spec: dict[str, Any] = {
        "containers": [
            {"name": "api", "resources": {"requests": {"cpu": cpu, "memory": memory}}}
        ]
    }
    if node is not None:
        spec["nodeName"] = node
    status: dict[str, Any] = {"phase": phase}
    if conditions is not None:
        status["conditions"] = conditions
    return {"metadata": metadata, "spec": spec, "status": status}


def _deployment(
    *, replicas: int = 3, max_unavailable: int = 0, stamp: str = "new"
) -> dict[str, Any]:
    return {
        "kind": "Deployment",
        "metadata": {"name": DEPLOYMENT, "namespace": "gpu-fault-system"},
        "spec": {
            "replicas": replicas,
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {"maxUnavailable": max_unavailable, "maxSurge": 1},
            },
            "template": {
                "metadata": {
                    "annotations": {"gpu-fault.io/release-wheel-sha256": stamp}
                },
                "spec": {
                    "topologySpreadConstraints": [
                        {
                            "maxSkew": 1,
                            "topologyKey": "kubernetes.io/hostname",
                            "whenUnsatisfiable": "DoNotSchedule",
                            "matchLabelKeys": ["pod-template-hash"],
                        }
                    ],
                    "containers": [
                        {
                            "name": "api",
                            "image": "python:3.12-slim",
                            "resources": {"requests": {"cpu": "4", "memory": "4Gi"}},
                        }
                    ],
                },
            },
        },
    }


def _cluster() -> tuple[dict[str, Any], dict[str, Any]]:
    """Free 6.02 / 3.67 / 3.62 cores: one node with the 4-core slot."""

    pods = [
        _pod(OLD[node], node, "4", "4Gi", DEPLOYMENT, template_hash=OLD_HASH)
        for node in NODES
    ]
    for node, used in zip(NODES, ("5870m", "8220m", "8270m"), strict=True):
        pods.append(_pod(f"filler-{node[-6:]}", node, used, "10Gi"))
    return {"items": [_node(name) for name in NODES]}, {"items": pods}


class FakeKubectl:
    """Answers the reads from canned documents; records every call."""

    def __init__(self, **documents: Any) -> None:
        self.documents = documents
        self.calls: list[list[str]] = []
        self.fail_annotate: set[str] = set()

    def __call__(self, arguments: list[str]) -> str:
        self.calls.append(arguments)
        if arguments[0] == "annotate":
            if arguments[2] in self.fail_annotate:
                raise MODULE.SteeringError(f'pods "{arguments[2]}" not found')
            return f"pod/{arguments[2]} annotated\n"
        kind = arguments[1]
        key = "all_pods" if kind == "pods" and "-A" in arguments else kind
        return json.dumps(self.documents[key])

    def annotations(self) -> list[tuple[str, str]]:
        return [
            (call[2], call[3].split("=", 1)[1])
            for call in self.calls
            if call[0] == "annotate"
        ]


def _plan(
    run: FakeKubectl | None, *, live_stamp: str = "old", **overrides: Any
) -> dict[str, Any]:
    nodes, pods = _cluster()
    return MODULE.plan(
        nodes=nodes,
        pods=pods,
        live_deployments={"items": [_deployment(stamp=live_stamp)]},
        desired=_deployment(**overrides),
        run=run,
    )


def test_plan_annotates_the_old_pods_in_the_steered_order() -> None:
    kubectl = FakeKubectl()
    state = _plan(kubectl)

    assert state["active"] is True
    assert state["surge_nodes"] == [NODES[0]] == state["planned_surge_nodes"]
    assert state["progress_guaranteed"] is True
    assert kubectl.annotations() == [
        (OLD[NODES[2]], "100"),
        (OLD[NODES[1]], "200"),
        (OLD[NODES[0]], "300"),
    ], "lowest cost is deleted first; the Pod sharing the surge node goes last"
    assert all(
        call[3].startswith(COST + "=") and call[4] == "--overwrite"
        for call in kubectl.calls
    ), kubectl.calls
    assert state["applied_costs"] == {
        OLD[NODES[2]]: 100,
        OLD[NODES[1]]: 200,
        OLD[NODES[0]]: 300,
    }
    assert [pod["name"] for pod in state["old_pods"]] == [OLD[node] for node in NODES]


def test_plan_without_a_kubectl_command_only_models() -> None:
    state = _plan(None)
    assert state["active"] is True and state["applied_costs"] == {}


def test_plan_skips_tiers_that_do_not_roll_surge_first() -> None:
    kubectl = FakeKubectl()
    state = _plan(kubectl, max_unavailable=1)
    assert state["active"] is False
    assert state["reason"] == "not a surge-first roll (maxUnavailable 1, maxSurge 1)"
    assert kubectl.calls == []


def test_plan_skips_a_template_that_does_not_roll() -> None:
    kubectl = FakeKubectl()
    state = _plan(kubectl, live_stamp="new")
    assert state["active"] is False
    assert state["reason"] == "the template does not roll"
    assert kubectl.calls == []


def test_plan_skips_a_deployment_without_old_pods() -> None:
    nodes, pods = _cluster()
    pods["items"] = [
        pod for pod in pods["items"] if pod["metadata"]["labels"]["app"] != DEPLOYMENT
    ]
    state = MODULE.plan(
        nodes=nodes,
        pods=pods,
        live_deployments={"items": [_deployment(stamp="old")]},
        desired=_deployment(),
        run=FakeKubectl(),
    )
    assert state["active"] is False
    assert "no old Pod" in state["reason"]


def test_plan_skips_a_pod_the_controller_already_took() -> None:
    kubectl = FakeKubectl()
    kubectl.fail_annotate = {OLD[NODES[1]]}
    state = _plan(kubectl)
    assert sorted(state["applied_costs"]) == sorted([OLD[NODES[2]], OLD[NODES[0]]])


def _replicasets() -> dict[str, Any]:
    def replicaset(template_hash: str, revision: str) -> dict[str, Any]:
        return {
            "metadata": {
                "name": f"{DEPLOYMENT}-{template_hash}",
                "labels": {"app": DEPLOYMENT, "pod-template-hash": template_hash},
                "annotations": {"deployment.kubernetes.io/revision": revision},
            },
            "spec": {"template": _deployment()["spec"]["template"]},
        }

    return {"items": [replicaset(OLD_HASH, "7"), replicaset(NEW_HASH, "8")]}


def _live_pods(*, surge_node: str, costs: dict[str, int]) -> dict[str, Any]:
    pods = [
        _pod(
            OLD[node],
            node,
            "4",
            "4Gi",
            DEPLOYMENT,
            template_hash=OLD_HASH,
            annotations={COST: str(costs[OLD[node]])},
        )
        for node in NODES
    ]
    pods.append(
        _pod(
            f"{DEPLOYMENT}-{NEW_HASH}-sf7g8",
            surge_node,
            "4",
            "4Gi",
            DEPLOYMENT,
            template_hash=NEW_HASH,
        )
    )
    return {"items": pods}


def test_reconcile_resteers_when_the_surge_pod_landed_elsewhere() -> None:
    planned = FakeKubectl()
    state = _plan(planned)
    kubectl = FakeKubectl(
        pods=_live_pods(surge_node=NODES[1], costs=state["applied_costs"]),
        replicasets=_replicasets(),
    )

    outcome = MODULE.reconcile(state, kubectl)

    assert outcome["unexpected"] == [NODES[1]]
    assert state["surge_nodes"] == [NODES[1]]
    assert [entry["pod"] for entry in state["deletion_order"]] == [
        OLD[NODES[2]],
        OLD[NODES[0]],
        OLD[NODES[1]],
    ], "the old Pod sharing the actual surge node now goes last"
    assert kubectl.annotations() == [(OLD[NODES[0]], "200"), (OLD[NODES[1]], "300")], (
        "only the Pods whose cost changed are annotated again"
    )
    assert outcome["roll"]["progress_guaranteed"] is True

    again = FakeKubectl(
        pods=_live_pods(surge_node=NODES[1], costs=state["applied_costs"]),
        replicasets=_replicasets(),
    )
    assert MODULE.reconcile(state, again)["changed"] == []
    assert [call[0] for call in again.calls] == ["get", "get"], (
        "a settled order writes nothing"
    )


def test_reconcile_confirms_the_predicted_surge_node_without_writing() -> None:
    state = _plan(FakeKubectl())
    kubectl = FakeKubectl(
        pods=_live_pods(surge_node=NODES[0], costs=state["applied_costs"]),
        replicasets=_replicasets(),
    )
    outcome = MODULE.reconcile(state, kubectl)
    assert outcome["changed"] == [] and outcome["unexpected"] == []
    assert not any(call[0] == "annotate" for call in kubectl.calls), kubectl.calls


def test_reconcile_waits_for_the_surge_pod_to_be_scheduled() -> None:
    state = _plan(FakeKubectl())
    pods = _live_pods(surge_node=NODES[0], costs=state["applied_costs"])
    del pods["items"][-1]["spec"]["nodeName"]
    pods["items"][-1]["status"]["phase"] = "Pending"
    kubectl = FakeKubectl(pods=pods, replicasets=_replicasets())
    assert MODULE.reconcile(state, kubectl) == {
        "changed": [],
        "note": "no new Pod is scheduled yet",
    }


def test_reconcile_credits_the_room_of_old_pods_already_gone() -> None:
    """After the first deletion and the second new Pod: two old Pods remain,
    the gone one's cores are back on its node, and the order for the rest
    still ends with the Pod co-located with a new Pod."""

    state = _plan(FakeKubectl())
    pods = _live_pods(surge_node=NODES[0], costs=state["applied_costs"])
    pods["items"] = [
        pod for pod in pods["items"] if pod["metadata"]["name"] != OLD[NODES[2]]
    ]
    pods["items"].append(
        _pod(
            f"{DEPLOYMENT}-{NEW_HASH}-8t4p5",
            NODES[2],
            "4",
            "4Gi",
            DEPLOYMENT,
            template_hash=NEW_HASH,
        )
    )
    kubectl = FakeKubectl(pods=pods, replicasets=_replicasets())

    outcome = MODULE.reconcile(state, kubectl)

    assert outcome["roll"]["progress_guaranteed"] is True
    assert [entry["pod"] for entry in state["deletion_order"]] == [
        OLD[NODES[1]],
        OLD[NODES[0]],
    ]
    assert kubectl.annotations() == [(OLD[NODES[1]], "100"), (OLD[NODES[0]], "200")]


def test_diagnose_names_the_scheduler_verdict_and_every_node_s_room() -> None:
    nodes, all_pods = _cluster()
    pending = _pod(
        f"{DEPLOYMENT}-{NEW_HASH}-8t4p5",
        None,
        "4",
        "4Gi",
        DEPLOYMENT,
        template_hash=NEW_HASH,
        phase="Pending",
        conditions=[
            {
                "type": "PodScheduled",
                "status": "False",
                "reason": "Unschedulable",
                "message": "0/3 nodes are available: 1 node(s) didn't match pod topology "
                "spread constraints, 2 Insufficient cpu.",
            }
        ],
    )
    surge = _pod(
        f"{DEPLOYMENT}-{NEW_HASH}-sf7g8",
        NODES[0],
        "4",
        "4Gi",
        DEPLOYMENT,
        template_hash=NEW_HASH,
    )
    old = [
        _pod(
            OLD[node],
            node,
            "4",
            "4Gi",
            DEPLOYMENT,
            template_hash=OLD_HASH,
            annotations={COST: cost},
        )
        for node, cost in zip(NODES[1:], ("200", "100"), strict=True)
    ]
    gone = _pod(
        OLD[NODES[0]],
        NODES[0],
        "4",
        "4Gi",
        DEPLOYMENT,
        template_hash=OLD_HASH,
        terminating=True,
    )
    deployment_pods = {"items": [pending, surge, *old, gone]}
    all_pods["items"] = [
        pod
        for pod in all_pods["items"]
        if pod["metadata"]["labels"]["app"] != DEPLOYMENT
    ] + deployment_pods["items"]
    kubectl = FakeKubectl(
        pods=deployment_pods, all_pods=all_pods, replicasets=_replicasets(), nodes=nodes
    )

    lines = MODULE.diagnose(DEPLOYMENT, kubectl)

    assert lines[0] == (
        f"new Pod {DEPLOYMENT}-{NEW_HASH}-8t4p5 is Pending (Unschedulable): 0/3 nodes are "
        "available: 1 node(s) didn't match pod topology spread constraints, 2 Insufficient cpu."
    )
    assert lines[1] == (
        f"old Pod {OLD[NODES[0]]} is terminating on {NODES[0]} and holds its requests until it is gone"
    )
    assert lines[2].startswith(f"node {NODES[0]}: 2.0 cores / "), lines[2]
    assert f"1 new Pod(s) [{DEPLOYMENT}-{NEW_HASH}-sf7g8]; 0 old Pod(s) []" in lines[2]
    assert lines[3].startswith(f"node {NODES[1]}: 3.7 cores / "), lines[3]
    assert (
        f"0 new Pod(s) []; 1 old Pod(s) [{OLD[NODES[1]]} (deletion cost 200)]"
        in lines[3]
    )
    assert lines[-1] == (
        "a new Pod needs 4 cores / 4Gi on a node that hosts no new Pod of this template "
        "(hostname spread maxSkew 1); free such a slot on a second node or add a CPU node"
    )
    assert [call[:2] for call in kubectl.calls] == [
        ["get", "pods"],
        ["get", "replicasets"],
        ["get", "nodes"],
        ["get", "pods"],
    ], "diagnosis only reads"


def test_newest_template_follows_the_deployment_revision() -> None:
    template_hash, template = MODULE.newest_template(_replicasets()["items"])
    assert template_hash == NEW_HASH
    assert template is not None and "spec" in template
    assert MODULE.newest_template([]) == ("", None)


def test_kubectl_runner_turns_a_failure_into_a_steering_error() -> None:
    run = MODULE.kubectl_runner(["sh", "-c", "echo oops >&2; exit 3", "sh"])
    with pytest.raises(MODULE.SteeringError, match="failed: oops"):
        run(["get", "pods"])
    echo = MODULE.kubectl_runner(["echo"])
    assert echo(["a", "b"]) == "a b\n"


def _write(tmp_path: Path, name: str, document: Any) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def test_cli_plan_writes_the_state_and_reports_the_order(
    tmp_path: Path, capsys
) -> None:
    nodes, pods = _cluster()
    state_path = tmp_path / "state.json"
    arguments = [
        "plan",
        "--nodes",
        _write(tmp_path, "nodes.json", nodes),
        "--pods",
        _write(tmp_path, "pods.json", pods),
        "--live-deployments",
        _write(tmp_path, "live.json", {"items": [_deployment(stamp="old")]}),
        "--deployment-document",
        _write(tmp_path, "desired.json", _deployment()),
        "--state",
        str(state_path),
    ]
    assert MODULE.main(arguments) == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["active"] is True and state["applied_costs"] == {}
    err = capsys.readouterr().err
    assert f"surge steering: {DEPLOYMENT}: surge Pod expected on {NODES[0]}" in err
    assert (
        f"{OLD[NODES[0]]} on {NODES[0]} (cost 300, co-located with a new Pod, last)"
        in err
    )

    (tmp_path / "desired.json").write_text(json.dumps(_deployment(max_unavailable=1)))
    assert MODULE.main(arguments) == 0
    assert json.loads(state_path.read_text(encoding="utf-8"))["active"] is False
    assert "nothing to steer (not a surge-first roll" in capsys.readouterr().err

    assert MODULE.main(["reconcile", "--state", str(state_path), "--", "false"]) == 0, (
        "an inactive state reconciles without touching kubectl"
    )
    with pytest.raises(SystemExit):
        MODULE.main(["reconcile", "--state", str(state_path)])
    with pytest.raises(SystemExit):
        MODULE.main(["diagnose", "--deployment", DEPLOYMENT])


def test_cli_reports_a_failing_kubectl_instead_of_a_traceback(
    tmp_path: Path, capsys
) -> None:
    assert MODULE.main(["diagnose", "--deployment", DEPLOYMENT, "--", "false"]) == 1
    assert (
        "surge steering: diagnose failed: kubectl get pods" in capsys.readouterr().err
    )
