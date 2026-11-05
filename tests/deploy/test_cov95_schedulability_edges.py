"""The schedulability preflight's narrow shapes (cov95).

``tests/deploy/test_control_plane_schedulability.py`` models today's 3 x
16-core node group end to end. These cases pin the rules around it: which
tolerations make a tainted node eligible, the defaults of a strategy without
``maxSurge`` and of a spread that is not a hostname DoNotSchedule one, the
shortfall that is the spread rather than the cores, the transient surge slots
of a roll that is not surge-first, the surge-first replay with no old Pod or
more old Pods than the exhaustive search takes, an unknown mode, and the
command line without ``--evidence`` and run as a script.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.deploy.test_control_plane_schedulability import (
    MODULE,
    NODES,
    ROOT,
    _deployment,
    _node,
    _pod,
    _rendered,
    _todays_cluster,
)

SCRIPT = ROOT / "deploy/control-plane/tools/control_plane_schedulability.py"
GI = 1024**3
SPOOL = "gpu-fault-telemetry-spool-worker"
WORKER = "gpu-fault-control-worker"
INGRESS = "gpu-fault-api-ha"


def names(nodes: dict[str, Any], template_spec: dict[str, Any]) -> list[str]:
    return [
        node["name"]
        for node in MODULE.eligible_nodes(nodes, {"items": []}, template_spec)
    ]


def test_a_tainted_node_is_eligible_only_for_a_toleration_that_matches() -> None:
    tainted = _node("ip-10-0-9-9")
    tainted["spec"]["taints"] = [
        {"key": "dedicated", "value": "control-plane", "effect": "NoSchedule"}
    ]
    nodes = {"items": [tainted, _node(NODES[0])]}
    near_misses = [
        {
            "key": "dedicated",
            "operator": "Equal",
            "value": "control-plane",
            "effect": "NoExecute",
        },
        {"key": "nvidia.com/gpu", "operator": "Exists"},
        {"key": "dedicated", "operator": "Equal", "value": "batch"},
    ]
    exact = {"key": "dedicated", "value": "control-plane", "effect": "NoSchedule"}

    assert names(nodes, {"tolerations": near_misses}) == [NODES[0]], (
        "another effect, another key and another value tolerate nothing"
    )
    assert names(nodes, {"tolerations": [*near_misses, exact]}) == [
        "ip-10-0-9-9",
        NODES[0],
    ]
    assert names(nodes, {"tolerations": [{"operator": "Exists"}]}) == [
        "ip-10-0-9-9",
        NODES[0],
    ], "a key-less Exists tolerates every taint"


def test_a_strategy_without_max_surge_and_a_zone_spread_take_the_defaults() -> None:
    desired = _deployment(INGRESS, 3, "4", "4Gi", max_unavailable=0, stamp="new")
    del desired["spec"]["strategy"]["rollingUpdate"]["maxSurge"]
    desired["spec"]["template"]["spec"]["topologySpreadConstraints"] = [
        {
            "maxSkew": 1,
            "topologyKey": "topology.kubernetes.io/zone",
            "whenUnsatisfiable": "DoNotSchedule",
        },
        {
            "maxSkew": 2,
            "topologyKey": "kubernetes.io/hostname",
            "whenUnsatisfiable": "ScheduleAnyway",
        },
    ]
    live = _deployment(INGRESS, 3, "4", "4Gi", max_unavailable=0, stamp="old")

    demand = MODULE.step_demand(desired, live)

    assert (demand["max_surge"], demand["max_unavailable"]) == (1, 0)
    assert demand["surge_pods_needing_a_free_slot"] == 1
    assert demand["max_skew"] is None, "only a hostname DoNotSchedule spread binds"


def _crowded(free_node: str = NODES[0]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Three nodes; only ``free_node`` has room, the others 0.89 cores."""

    nodes = {"items": [_node(name) for name in NODES]}
    pods = {
        "items": [
            _pod(name, "15", "4Gi", f"filler-{name}")
            for name in NODES
            if name != free_node
        ]
    }
    return nodes, pods


def test_added_replicas_the_spread_forbids_are_refused_as_the_spread() -> None:
    nodes, pods = _crowded()
    rendered = {
        "spool": _deployment(SPOOL, 2, "1", "1Gi", max_unavailable=1, stamp="a")
    }
    live = {"items": [_deployment(SPOOL, 0, "1", "1Gi", max_unavailable=1, stamp="a")]}

    with pytest.raises(MODULE.SchedulabilityRefused) as refused:
        MODULE.plan_steps(
            nodes=nodes,
            pods=pods,
            live_deployments=live,
            rendered=rendered,
            order=["spool"],
        )

    message = str(refused.value)
    assert (
        f"step 1 ({SPOOL} roll) cannot schedule 2 added replica(s): Pod 2 of 2 "
        "needs 1 cores / 1Gi; the nodes with room would break the hostname spread"
    ) in message, message


def test_a_roll_that_is_not_surge_first_places_its_surge_pods_transiently() -> None:
    nodes = {"items": [_node(name) for name in NODES]}
    desired = _deployment(WORKER, 3, "3", "4Gi", max_unavailable=1, stamp="new")
    desired["spec"]["strategy"]["rollingUpdate"]["maxSurge"] = 3
    live = {
        "items": [_deployment(WORKER, 3, "3", "4Gi", max_unavailable=1, stamp="old")]
    }

    evidence = MODULE.plan_steps(
        nodes=nodes,
        pods={"items": []},
        live_deployments=live,
        rendered={"worker": desired},
        order=["worker"],
    )

    (step,) = evidence["steps"]
    assert step["surge_pods_needing_a_free_slot"] == 2
    assert len(set(step["surge_placements"])) == 2, "the spread splits the surge Pods"
    assert step["added_placements"] == []
    before = {
        node["name"]: node["free_cpu_millicores"] for node in evidence["nodes_before"]
    }
    after = {
        node["name"]: node["free_cpu_millicores"] for node in evidence["nodes_after"]
    }
    assert after == before, "a surge slot is given back once the old Pod goes"


def test_a_surge_pod_without_room_stops_the_step_before_its_added_replicas() -> None:
    nodes, pods = _crowded(free_node="")
    desired = _deployment(INGRESS, 4, "4", "4Gi", max_unavailable=1, stamp="new")
    desired["spec"]["strategy"]["rollingUpdate"]["maxSurge"] = 2
    live = {
        "items": [_deployment(INGRESS, 3, "4", "4Gi", max_unavailable=1, stamp="old")]
    }

    with pytest.raises(MODULE.SchedulabilityRefused) as refused:
        MODULE.plan_steps(
            nodes=nodes,
            pods=pods,
            live_deployments=live,
            rendered={"ingress": desired},
            order=["ingress"],
        )

    message = str(refused.value)
    assert (
        f"step 1 ({INGRESS} roll) cannot schedule one surge Pod (maxUnavailable 1, "
        "maxSurge 2): Pod 1 of 1 needs 4 cores / 4Gi; no node has room"
    ) in message, message


def test_a_surge_first_roll_without_old_pods_owes_no_deletion() -> None:
    states = MODULE.eligible_nodes(
        {"items": [_node("n1"), _node("n2")]}, {"items": []}, {}
    )

    result = MODULE.simulate_surge_roll(
        node_states=states,
        old_pods=[],
        cpu=4000,
        memory=4 * GI,
        replicas=1,
        max_surge=1,
        max_skew=1,
    )

    assert result["progress_guaranteed"] is True and result["stall"] is None
    assert result["surge_nodes"] == ["n1"]
    assert result["deletion_order"] == []
    assert result["free_after"] == {
        "n1": [15890 - 4000, 30 * GI - 4 * GI],
        "n2": [15890, 30 * GI],
    }


def test_more_old_pods_than_the_exhaustive_search_replay_the_controllers_choice() -> (
    None
):
    """Eight old Pods on a node the live surge Pod filled: only the controller's
    own first deletion is replayed, and it already leaves no node."""

    node = _node("n1")
    node["status"]["allocatable"] = {"cpu": "8", "memory": "16Gi"}
    pods = {"items": [_pod("n1", "1", "1Gi", WORKER) for _ in range(8)]}
    for index, pod in enumerate(pods["items"]):
        pod["metadata"]["name"] = f"old-{index}"
    states = MODULE.eligible_nodes({"items": [node]}, pods, {})
    old = MODULE.deployment_pods(pods, WORKER, {"n1"})
    assert len(old) == MODULE.EXHAUSTIVE_KILL_ORDER_PODS + 1

    result = MODULE.simulate_surge_roll(
        node_states=states,
        old_pods=old,
        cpu=2000,
        memory=GI,
        replicas=8,
        max_surge=1,
        max_skew=None,
        surge_nodes=["n1"],
    )

    unsteered = result["without_steering"]
    assert unsteered["progress_guaranteed"] is False
    assert unsteered["stalling_order"] == ["old-0"]
    assert unsteered["stall"].startswith(
        "after old-0 on n1 is deleted, new Pod 2 of 8 has no node: n1 has "
    ), unsteered["stall"]
    assert result["progress_guaranteed"] is False
    assert result["stall"].startswith(
        "new Pod 2 of 8 (2 cores / 1Gi) has no node whichever old Pod is deleted first"
    ), result["stall"]


def test_an_unknown_mode_is_a_programming_error() -> None:
    nodes, pods, live = _todays_cluster()

    with pytest.raises(ValueError, match="unknown schedulability mode 'both'"):
        MODULE.plan_roll(
            nodes=nodes,
            pods=pods,
            live_deployments=live,
            rendered=_rendered(),
            mode="both",
        )


def _inputs(tmp_path: Path, rendered: dict[str, dict[str, Any]]) -> list[str]:
    nodes, pods, live = _todays_cluster()
    arguments = []
    for flag, document in (
        ("--nodes", nodes),
        ("--pods", pods),
        ("--live-deployments", live),
    ):
        path = tmp_path / f"{flag[2:]}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        arguments += [flag, str(path)]
    for role, document in rendered.items():
        path = tmp_path / f"rendered-{role}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        arguments += ["--rendered", f"{role}={path}"]
    return arguments


def test_the_command_line_without_evidence_prints_the_plan_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = _inputs(tmp_path, _rendered())
    before = sorted(path.name for path in tmp_path.iterdir())

    assert MODULE.main(arguments) == 0

    printed = capsys.readouterr().err
    prefix = "control-plane schedulability: "
    assert printed.startswith(prefix), printed
    evidence = json.loads(printed[len(prefix) :])
    assert (evidence["verdict"], evidence["order"]) == (
        "fits",
        ["ingress", "worker", "spool"],
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == before, (
        "no evidence file without --evidence"
    )


def test_the_script_exits_with_the_refusal_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = _inputs(tmp_path, _rendered(spool_replicas=15))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *arguments])

    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert caught.value.code == 2
    assert "cannot schedule 15 added replica(s)" in capsys.readouterr().err
