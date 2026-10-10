"""The CPU roll's schedulability preflight, on the 3 x 16-core node group that
timed out: ingress first fits, the old consumer-first order does not."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE = lazy_script_module(
    ROOT / "deploy/control-plane/tools/control_plane_schedulability.py"
)
GI = 1024**3
NODES = ("ip-10-0-1-23", "ip-10-0-2-11", "ip-10-0-3-7")
CANONICAL = ["ingress", "worker", "spool"]
STAMP = "gpu-fault.io/release-wheel-sha256"


def _node(name: str) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "labels": {"kubernetes.io/hostname": name}},
        "spec": {},
        "status": {
            "allocatable": {"cpu": "15890m", "memory": "30Gi"},
            "conditions": [{"type": "Ready", "status": "True"}],
        },
    }


def _pod(node: str, cpu: str, memory: str, app: str = "other") -> dict[str, Any]:
    return {
        "metadata": {"name": f"{app}-{node}", "labels": {"app": app}},
        "spec": {
            "nodeName": node,
            "containers": [
                {"name": "c", "resources": {"requests": {"cpu": cpu, "memory": memory}}}
            ],
        },
        "status": {"phase": "Running"},
    }


def _deployment(
    name: str, replicas: int, cpu: str, memory: str, *, max_unavailable: int, stamp: str
) -> dict[str, Any]:
    return {
        "kind": "Deployment",
        "metadata": {"name": name},
        "spec": {
            "replicas": replicas,
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {"maxUnavailable": max_unavailable, "maxSurge": 1},
            },
            "template": {
                "metadata": {"annotations": {STAMP: stamp}},
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
                            "name": "main",
                            "image": "python:3.12-slim",
                            "resources": {"requests": {"cpu": cpu, "memory": memory}},
                        }
                    ],
                },
            },
        },
    }


def _todays_cluster(
    *, workers: int = 6, spool_replicas: int = 0
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """3 x ml.c5.4xlarge: api-ha 3 x 4 cores, control-worker 6 x 3 cores, the
    HyperPod operator 1, kueue 0.5 and system Pods -- 34.4 of 48 cores
    requested, 4.1-4.8 free per node."""

    nodes = {"items": [_node(name) for name in NODES]}
    pods: list[dict[str, Any]] = []
    for index, name in enumerate(NODES):
        pods.append(_pod(name, "4", "4Gi", "gpu-fault-api-ha"))
        for slot in range(2):
            if index * 2 + slot < workers:
                pods.append(_pod(name, "3", "4Gi", "gpu-fault-control-worker"))
    pods.append(_pod(NODES[0], "1", "1Gi", "hyperpod-operator"))
    pods.append(_pod(NODES[1], "500m", "512Mi", "kueue"))
    pods.append(_pod(NODES[0], "790m", "600Mi", "system"))
    pods.append(_pod(NODES[1], "590m", "600Mi", "system"))
    pods.append(_pod(NODES[2], "1490m", "600Mi", "system"))
    for index in range(spool_replicas):
        pods.append(
            _pod(NODES[index % 3], "1", "2Gi", "gpu-fault-telemetry-spool-worker")
        )
    live = {
        "items": [
            _deployment(
                "gpu-fault-api-ha", 3, "4", "4Gi", max_unavailable=0, stamp="old"
            ),
            _deployment(
                "gpu-fault-control-worker",
                workers,
                "3",
                "4Gi",
                max_unavailable=1,
                stamp="old",
            ),
            _deployment(
                "gpu-fault-telemetry-spool-worker",
                spool_replicas,
                "1",
                "2Gi",
                max_unavailable=1,
                stamp="old",
            ),
        ]
    }
    return nodes, {"items": pods}, live


def _rendered(
    *, workers: int = 6, spool_replicas: int = 3, stamp: str = "new"
) -> dict[str, dict[str, Any]]:
    return {
        "ingress": _deployment(
            "gpu-fault-api-ha", 3, "4", "4Gi", max_unavailable=0, stamp=stamp
        ),
        "worker": _deployment(
            "gpu-fault-control-worker",
            workers,
            "3",
            "4Gi",
            max_unavailable=1,
            stamp=stamp,
        ),
        "spool": _deployment(
            "gpu-fault-telemetry-spool-worker",
            spool_replicas,
            "1",
            "2Gi",
            max_unavailable=1,
            stamp=stamp,
        ),
    }


def test_the_canonical_order_rolls_ingress_first() -> None:
    assert MODULE.CANONICAL_ORDER == ("ingress", "worker", "spool")
    assert list(MODULE.ROLE_DEPLOYMENTS) == list(MODULE.CANONICAL_ORDER)


def test_free_cores_match_the_live_node_group() -> None:
    nodes, pods, _ = _todays_cluster()
    states = MODULE.eligible_nodes(nodes, pods, {})
    free = {node["name"]: node["free_cpu_millicores"] for node in states}
    assert free == {"ip-10-0-1-23": 4100, "ip-10-0-2-11": 4800, "ip-10-0-3-7": 4400}
    assert sum(node["requested_cpu_millicores"] for node in states) == 34_370


def test_ingress_first_fits_a_spool_enable_on_todays_nodes() -> None:
    """Spool 0 -> 3 with the worker count unchanged: the canonical order fits."""

    nodes, pods, live = _todays_cluster()
    evidence = MODULE.plan_roll(
        nodes=nodes, pods=pods, live_deployments=live, rendered=_rendered()
    )
    assert evidence["verdict"] == "fits"
    assert evidence["order"] == CANONICAL
    assert evidence["attempts_refused"] == []
    steps = {step["role"]: step for step in evidence["steps"]}
    assert steps["ingress"]["surge_pods_needing_a_free_slot"] == 1
    assert steps["ingress"]["surge_placements"] == ["ip-10-0-2-11"], (
        "the surge Pod goes to the node with the most room"
    )
    assert steps["worker"]["surge_pods_needing_a_free_slot"] == 0, (
        "maxUnavailable 1 frees a slot before the surge Pod needs one"
    )
    assert steps["spool"]["added_replicas"] == 3
    assert sorted(steps["spool"]["added_placements"]) == sorted(NODES), (
        "the hostname spread puts one added replica on each node"
    )
    after = {
        node["name"]: node["free_cpu_millicores"] for node in evidence["nodes_after"]
    }
    assert after == {"ip-10-0-1-23": 3100, "ip-10-0-2-11": 3800, "ip-10-0-3-7": 3400}


def test_the_consumer_first_order_is_refused_naming_the_missing_cores() -> None:
    """Spool replicas first, then the ingress surge Pod: today's timeout,
    caught by the model when it is asked to keep that order."""

    nodes, pods, live = _todays_cluster()
    with pytest.raises(MODULE.SchedulabilityRefused) as info:
        MODULE.plan_steps(
            nodes=nodes,
            pods=pods,
            live_deployments=live,
            rendered=_rendered(),
            order=["spool", "worker", "ingress"],
        )
    message = str(info.value)
    assert "step 3 (gpu-fault-api-ha roll) cannot schedule one surge Pod" in message
    assert "needs 4 cores / 4Gi" in message
    assert "0.2 more free cores on one node would fit it" in message
    assert "ip-10-0-2-11 3.8 cores" in message and "15.89 cores allocatable" in message
    assert "Add CPU node capacity or lower the preset" in message


def test_the_32_enabled_preset_fits_and_records_the_worker_reduction() -> None:
    """32-enabled: workers 6 -> 5, spool 0 -> 3, every role's template rolls."""

    nodes, pods, live = _todays_cluster()
    evidence = MODULE.plan_roll(
        nodes=nodes, pods=pods, live_deployments=live, rendered=_rendered(workers=5)
    )
    assert evidence["order"] == CANONICAL
    worker = evidence["steps"][1]
    assert worker["removed_replicas"] == 1 and worker["added_replicas"] == 0
    after = {
        node["name"]: node["free_cpu_millicores"] for node in evidence["nodes_after"]
    }
    assert sum(after.values()) == 4100 + 4800 + 4400 + 3000 - 3000


def test_room_freeing_roles_go_first_when_the_canonical_order_cannot_fit() -> None:
    """The rollback of an enable: spool 3 -> 0 frees the cores the ingress
    surge Pod needs; the planner records why it left the canonical order."""

    nodes, pods, live = _todays_cluster(spool_replicas=3)
    evidence = MODULE.plan_roll(
        nodes=nodes,
        pods=pods,
        live_deployments=live,
        rendered=_rendered(spool_replicas=0),
    )
    assert evidence["verdict"] == "fits"
    assert evidence["order"] == ["spool", "ingress", "worker"]
    assert len(evidence["attempts_refused"]) == 1
    assert evidence["attempts_refused"][0]["order"] == CANONICAL
    assert (
        "step 1 (gpu-fault-api-ha roll) cannot schedule one surge Pod"
        in (evidence["attempts_refused"][0]["refused"])
    )
    spool = evidence["steps"][0]
    assert spool["removed_replicas"] == 3
    assert evidence["steps"][1]["surge_placements"] == ["ip-10-0-2-11"]


def test_a_fixed_order_is_modelled_as_given_and_refused_as_a_whole() -> None:
    nodes, pods, live = _todays_cluster(spool_replicas=3)
    with pytest.raises(MODULE.SchedulabilityRefused) as info:
        MODULE.plan_roll(
            nodes=nodes,
            pods=pods,
            live_deployments=live,
            rendered=_rendered(spool_replicas=0),
            fixed_order=CANONICAL,
        )
    assert "step 1 (gpu-fault-api-ha roll)" in str(info.value)
    assert "also refused" not in str(info.value), "a fixed order has no alternative"


def test_warn_mode_returns_the_canonical_order_with_every_shortfall() -> None:
    nodes, pods, live = _todays_cluster()
    for name in NODES:
        pods["items"].append(_pod(name, "3", "1Gi", "filler"))
    evidence = MODULE.plan_roll(
        nodes=nodes, pods=pods, live_deployments=live, rendered=_rendered(), mode="warn"
    )
    assert evidence["verdict"] == "warning"
    assert evidence["order"] == CANONICAL
    assert [attempt["order"] for attempt in evidence["attempts_refused"]] == [CANONICAL]
    assert "proceeding because the caller asked to warn only" in evidence["warning"]
    with pytest.raises(MODULE.SchedulabilityRefused) as info:
        MODULE.plan_roll(
            nodes=nodes, pods=pods, live_deployments=live, rendered=_rendered()
        )
    assert "needs 4 cores / 4Gi" in str(info.value)


def test_a_larger_preset_is_refused_at_the_spool_step_with_the_shortfall() -> None:
    nodes, pods, live = _todays_cluster()
    with pytest.raises(MODULE.SchedulabilityRefused) as info:
        MODULE.plan_roll(
            nodes=nodes,
            pods=pods,
            live_deployments=live,
            rendered=_rendered(spool_replicas=15),
        )
    message = str(info.value)
    assert (
        "step 3 (gpu-fault-telemetry-spool-worker roll) cannot schedule 15 added replica(s)"
        in message
    )
    assert "would break the hostname spread" in message or "more free cores" in message


def test_an_unchanged_template_needs_no_surge_slot_so_a_resume_is_not_refused() -> None:
    nodes, pods, live = _todays_cluster(spool_replicas=3)
    rendered = _rendered(stamp="old")
    # Fill the headroom so a modelled surge would not fit.
    for name in NODES:
        pods["items"].append(_pod(name, "3", "1Gi", "filler"))
    evidence = MODULE.plan_roll(
        nodes=nodes, pods=pods, live_deployments=live, rendered=rendered
    )
    assert evidence["verdict"] == "fits"
    assert all(step["template_rolls"] is False for step in evidence["steps"]), (
        "a template equal to the live one does not roll"
    )
    assert all(
        step["surge_pods_needing_a_free_slot"] == 0 for step in evidence["steps"]
    ), "no roll, no surge slot"
    assert all(step["added_replicas"] == 0 for step in evidence["steps"]), (
        "the replica counts already match"
    )


def test_template_rolls_on_image_request_or_stamp_changes_only() -> None:
    live = _deployment(
        "gpu-fault-api-ha", 3, "4", "4Gi", max_unavailable=0, stamp="old"
    )
    same = _deployment(
        "gpu-fault-api-ha", 3, "4", "4Gi", max_unavailable=0, stamp="old"
    )
    assert MODULE.template_rolls(same, live) is False
    same["spec"]["template"]["metadata"]["annotations"]["unrelated/note"] = "x"
    assert MODULE.template_rolls(same, live) is False
    assert MODULE.template_rolls(same, None) is True
    image = json.loads(json.dumps(same))
    image["spec"]["template"]["spec"]["containers"][0]["image"] = "python:3.13-slim"
    assert MODULE.template_rolls(image, live) is True
    request = json.loads(json.dumps(same))
    request["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"][
        "cpu"
    ] = "5"
    assert MODULE.template_rolls(request, live) is True
    renamed = json.loads(json.dumps(same))
    renamed["spec"]["template"]["spec"]["containers"][0]["name"] = "other"
    assert MODULE.template_rolls(renamed, live) is True


def test_a_greenfield_creation_places_added_replicas_without_a_surge() -> None:
    nodes, pods, _ = _todays_cluster()
    pods["items"] = [
        pod
        for pod in pods["items"]
        if not pod["metadata"]["labels"]["app"].startswith("gpu-fault-")
    ]
    evidence = MODULE.plan_roll(
        nodes=nodes,
        pods=pods,
        live_deployments={"items": []},
        rendered=_rendered(workers=6),
    )
    assert [step["surge_pods_needing_a_free_slot"] for step in evidence["steps"]] == [
        0,
        0,
        0,
    ]
    assert [step["added_replicas"] for step in evidence["steps"]] == [3, 6, 3]
    assert [step["template_rolls"] for step in evidence["steps"]] == [
        False,
        False,
        False,
    ]


@pytest.mark.parametrize(
    ("value", "millicores"),
    [("4", 4000), ("500m", 500), ("0.5", 500), ("15890m", 15890), ("", 0)],
)
def test_cpu_quantities(value: str, millicores: int) -> None:
    assert MODULE.parse_cpu_millicores(value) == millicores


@pytest.mark.parametrize(
    ("value", "size"),
    [
        ("4Gi", 4 * GI),
        ("512Mi", 512 * 1024**2),
        ("1000000Ki", 1_024_000_000),
        ("1e9", 10**9),
        ("2G", 2 * 10**9),
        ("100", 100),
        ("1500m", 2),
        ("", 0),
    ],
)
def test_memory_quantities(value: str, size: int) -> None:
    assert MODULE.parse_memory_bytes(value) == size


def test_pod_requests_follow_the_scheduler_rule() -> None:
    spec = {
        "containers": [
            {"resources": {"requests": {"cpu": "1", "memory": "1Gi"}}},
            {"resources": {"requests": {"cpu": "500m"}}},
        ],
        "initContainers": [
            {"resources": {"requests": {"cpu": "2", "memory": "512Mi"}}}
        ],
        "overhead": {"cpu": "100m", "memory": "64Mi"},
    }
    assert MODULE.pod_requests(spec) == (2100, GI + 64 * 1024**2)


def test_percentage_surge_and_memory_shortfall_are_reported() -> None:
    nodes, pods, live = _todays_cluster()
    rendered = _rendered()
    rendered["ingress"]["spec"]["strategy"]["rollingUpdate"] = {
        "maxSurge": "34%",
        "maxUnavailable": "0%",
    }
    rendered["ingress"]["spec"]["template"]["spec"]["containers"][0]["resources"][
        "requests"
    ]["memory"] = "40Gi"
    with pytest.raises(MODULE.SchedulabilityRefused) as info:
        MODULE.plan_roll(
            nodes=nodes, pods=pods, live_deployments=live, rendered=rendered
        )
    message = str(info.value)
    assert "maxUnavailable 0, maxSurge 2" in message
    assert "more free memory on one node would fit it" in message


def test_tainted_and_unready_nodes_are_not_eligible() -> None:
    nodes, pods, _ = _todays_cluster()
    nodes["items"][0]["spec"]["taints"] = [
        {"key": "dedicated", "value": "gpu", "effect": "NoSchedule"}
    ]
    nodes["items"][1]["status"]["conditions"] = [{"type": "Ready", "status": "False"}]
    nodes["items"][2]["spec"]["unschedulable"] = True
    assert MODULE.eligible_nodes(nodes, pods, {}) == []
    tolerant = {
        "tolerations": [
            {
                "key": "dedicated",
                "operator": "Equal",
                "value": "gpu",
                "effect": "NoSchedule",
            }
        ]
    }
    assert [node["name"] for node in MODULE.eligible_nodes(nodes, pods, tolerant)] == [
        NODES[0]
    ]
    exists = {"tolerations": [{"operator": "Exists"}]}
    assert [node["name"] for node in MODULE.eligible_nodes(nodes, pods, exists)] == [
        NODES[0]
    ]
    selector = {"nodeSelector": {"kubernetes.io/hostname": "elsewhere"}}
    assert MODULE.eligible_nodes(nodes, pods, selector) == []
    with pytest.raises(
        MODULE.SchedulabilityRefused, match="no Ready, schedulable node"
    ):
        MODULE.plan_roll(
            nodes=nodes, pods=pods, live_deployments={"items": []}, rendered=_rendered()
        )


def _write_inputs(tmp_path: Path, nodes, pods, live, rendered) -> list[str]:
    (tmp_path / "nodes.json").write_text(json.dumps(nodes))
    (tmp_path / "pods.json").write_text(json.dumps(pods))
    (tmp_path / "live.json").write_text(json.dumps(live))
    arguments = [
        "--nodes",
        str(tmp_path / "nodes.json"),
        "--pods",
        str(tmp_path / "pods.json"),
        "--live-deployments",
        str(tmp_path / "live.json"),
        "--evidence",
        str(tmp_path / "evidence.json"),
    ]
    for role, document in rendered.items():
        (tmp_path / f"{role}.json").write_text(json.dumps(document))
        arguments.extend(["--rendered", f"{role}={tmp_path / f'{role}.json'}"])
    return arguments


def test_cli_writes_evidence_and_refuses_with_exit_2(tmp_path: Path, capsys) -> None:
    nodes, pods, live = _todays_cluster()
    arguments = _write_inputs(tmp_path, nodes, pods, live, _rendered())

    assert MODULE.main(arguments) == 0
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["order"] == CANONICAL
    assert [step["deployment"] for step in evidence["steps"]] == [
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    ]
    assert "control-plane schedulability: {" in capsys.readouterr().err

    (tmp_path / "spool.json").write_text(
        json.dumps(_rendered(spool_replicas=15)["spool"])
    )
    assert MODULE.main(arguments) == 2
    assert "cannot schedule 15 added replica(s)" in capsys.readouterr().err

    assert MODULE.main([*arguments, "--mode", "warn"]) == 0
    err = capsys.readouterr().err
    assert "proceeding because the caller asked to warn only" in err
    assert json.loads((tmp_path / "evidence.json").read_text())["verdict"] == "warning"


def test_cli_fixed_order_must_name_every_rendered_role(tmp_path: Path) -> None:
    nodes, pods, live = _todays_cluster()
    arguments = _write_inputs(tmp_path, nodes, pods, live, _rendered())
    with pytest.raises(SystemExit):
        MODULE.main([*arguments, "--fixed-order", "ingress,worker"])
    assert MODULE.main([*arguments, "--fixed-order", "ingress,worker,spool"]) == 0
    with pytest.raises(SystemExit):
        MODULE.main([*arguments, "--rendered", "bogus=x.json"])
    with pytest.raises(SystemExit):
        MODULE.main(arguments[:-6])


# ---------------------------------------------------------------------------
# The surge-first roll's deletion order (batch 49).
#
# Live control plane, three CPU nodes, before a `config` roll: free CPU
# 6.02 / 3.67 / 3.62 cores -- exactly one node with the 4-core api-ha slot. The
# surge Pod lands there; the ReplicaSet controller then deletes the old Pod
# sharing that node first (more related Pods on the node), whose cores sit on
# the one node the per-template hostname spread forbids for the next new Pod,
# while the other two nodes keep their 4-core old Pods: the roll stalls until
# the rollout timeout and the release rolls back.
# ---------------------------------------------------------------------------
LIVE_NODES = ("hyperpod-cpu-a", "hyperpod-cpu-b", "hyperpod-cpu-c")
OLD_HASH = "7c9d5b8f4"
OLD_API_PODS = {
    LIVE_NODES[0]: f"gpu-fault-api-ha-{OLD_HASH}-llr52",
    LIVE_NODES[1]: f"gpu-fault-api-ha-{OLD_HASH}-876dt",
    LIVE_NODES[2]: f"gpu-fault-api-ha-{OLD_HASH}-ss92d",
}


def _named_pod(
    name: str, node: str, cpu: str, memory: str, app: str, template_hash: str = ""
) -> dict[str, Any]:
    pod = _pod(node, cpu, memory, app)
    pod["metadata"]["name"] = name
    if template_hash:
        pod["metadata"]["labels"]["pod-template-hash"] = template_hash
    return pod


def _live_node_group() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """api-ha 3 x 4 cores (one per node), control-worker 1/2/2 x 3 cores,
    spool-worker 1 core per node and the system share: free 6.02 / 3.67 / 3.62
    cores, 14.9 / 12.2 / 12.5Gi."""

    nodes = {"items": [_node(name) for name in LIVE_NODES]}
    pods: list[dict[str, Any]] = [
        _named_pod(OLD_API_PODS[node], node, "4", "4Gi", "gpu-fault-api-ha", OLD_HASH)
        for node in LIVE_NODES
    ]
    for node, workers, system in (
        (LIVE_NODES[0], 1, ("1870m", "4Gi")),
        (LIVE_NODES[1], 2, ("1220m", "4Gi")),
        (LIVE_NODES[2], 2, ("1270m", "4Gi")),
    ):
        for index in range(workers):
            pods.append(
                _named_pod(
                    f"gpu-fault-control-worker-{node[-6:]}-{index}",
                    node,
                    "3",
                    "4Gi",
                    "gpu-fault-control-worker",
                )
            )
        pods.append(
            _named_pod(
                f"gpu-fault-telemetry-spool-worker-{node[-6:]}",
                node,
                "1",
                "2Gi",
                "gpu-fault-telemetry-spool-worker",
            )
        )
        pods.append(_named_pod(f"system-{node[-6:]}", node, *system, "system"))
    live = {
        "items": [
            _deployment(
                "gpu-fault-api-ha", 3, "4", "4Gi", max_unavailable=0, stamp="old"
            ),
            _deployment(
                "gpu-fault-control-worker",
                5,
                "3",
                "4Gi",
                max_unavailable=1,
                stamp="old",
            ),
            _deployment(
                "gpu-fault-telemetry-spool-worker",
                3,
                "1",
                "2Gi",
                max_unavailable=1,
                stamp="old",
            ),
        ]
    }
    return nodes, {"items": pods}, live


def test_the_live_node_group_has_exactly_one_api_ha_slot() -> None:
    nodes, pods, _ = _live_node_group()
    free = {
        node["name"]: node["free_cpu_millicores"]
        for node in MODULE.eligible_nodes(nodes, pods, {})
    }
    assert free == dict(zip(LIVE_NODES, (6020, 3670, 3620), strict=True))


def test_the_controller_s_own_order_stalls_and_the_steered_order_progresses() -> None:
    """The canonical deadlock: surge on the 6.02 node, the co-located old Pod
    deleted first, new Pod 2 of 3 without a node. Steered, the two old Pods on
    the other nodes go first and the co-located one last; the end state equals
    the start (the roll moves no request between nodes)."""

    nodes, pods, live = _live_node_group()
    evidence = MODULE.plan_roll(
        nodes=nodes, pods=pods, live_deployments=live, rendered=_rendered(workers=5)
    )
    assert evidence["verdict"] == "fits"
    ingress = evidence["steps"][0]
    steering = ingress["steering"]
    assert ingress["surge_placements"] == [LIVE_NODES[0]]
    assert steering["nodes_with_a_free_slot"] == [LIVE_NODES[0]], (
        "exactly one free slot: the case that needs steering, not a refusal"
    )
    assert steering["surge_nodes"] == [LIVE_NODES[0]]
    assert steering["progress_guaranteed"] is True
    assert [entry["pod"] for entry in steering["deletion_order"]] == [
        OLD_API_PODS[LIVE_NODES[2]],
        OLD_API_PODS[LIVE_NODES[1]],
        OLD_API_PODS[LIVE_NODES[0]],
    ], "the old Pods on the other nodes go first, the co-located one last"
    assert [entry["cost"] for entry in steering["deletion_order"]] == [100, 200, 300]
    assert [entry.get("frees_node_for") for entry in steering["deletion_order"]] == [
        LIVE_NODES[2],
        LIVE_NODES[1],
        None,
    ], "each early deletion frees the node the next new Pod lands on"
    unsteered = steering["without_steering"]
    assert unsteered["progress_guaranteed"] is False
    assert unsteered["stalling_order"] == [OLD_API_PODS[LIVE_NODES[0]]], (
        "the controller deletes the old Pod co-located with the surge Pod first"
    )
    assert "new Pod 2 of 3 has no node" in unsteered["stall"]
    assert (
        f"{LIVE_NODES[0]} already hosts 1 new Pod(s) and the hostname spread "
        "(maxSkew 1) forbids another" in unsteered["stall"]
    )
    assert f"{LIVE_NODES[1]} has 3.7 cores" in unsteered["stall"]
    assert "0.33 more cores would fit it" in unsteered["stall"]
    before = {
        node["name"]: node["free_cpu_millicores"] for node in evidence["nodes_before"]
    }
    assert evidence["steps"][1]["free_cpu_millicores_before"] == before, (
        "a request-neutral roll leaves every node's headroom as it was"
    )


def test_the_surge_pod_on_an_unexpected_node_resteers_the_order() -> None:
    """The scheduler is not bound to the model's prediction: told where the
    surge Pod actually landed, the model puts that node's old Pod last."""

    nodes, pods, _ = _live_node_group()
    node_states = MODULE.eligible_nodes(nodes, pods, {})
    old_pods = MODULE.deployment_pods(pods, "gpu-fault-api-ha", set(LIVE_NODES))
    roll = MODULE.simulate_surge_roll(
        node_states=node_states,
        old_pods=old_pods,
        cpu=4000,
        memory=4 * GI,
        replicas=3,
        max_surge=1,
        max_skew=1,
        surge_nodes=[LIVE_NODES[1]],
    )
    assert roll["progress_guaranteed"] is True
    assert [entry["pod"] for entry in roll["deletion_order"]][-1] == (
        OLD_API_PODS[LIVE_NODES[1]]
    )
    assert roll["deletion_order"][-1]["cost"] == 300


def test_a_steered_roll_that_cannot_progress_is_refused_naming_the_node() -> None:
    """Two old Pods on one node, one on another, none on the third: new Pod 3
    of 3 can only go to the empty node, which is one core short. No deletion
    order helps; the refusal names the node, the cores and what to do."""

    nodes = {"items": [_node(name) for name in ("node-a", "node-b", "node-c")]}
    pods = {
        "items": [
            _named_pod("api-a1", "node-a", "4", "4Gi", "gpu-fault-api-ha", OLD_HASH),
            _named_pod("api-a2", "node-a", "4", "4Gi", "gpu-fault-api-ha", OLD_HASH),
            _named_pod("api-b", "node-b", "4", "4Gi", "gpu-fault-api-ha", OLD_HASH),
            _pod("node-a", "3390m", "1Gi", "filler"),
            _pod("node-b", "10890m", "1Gi", "filler"),
            _pod("node-c", "12890m", "1Gi", "filler"),
        ]
    }
    live = {
        "items": [
            _deployment(
                "gpu-fault-api-ha", 3, "4", "4Gi", max_unavailable=0, stamp="old"
            )
        ]
    }
    rendered = {"ingress": _rendered()["ingress"]}
    with pytest.raises(MODULE.SchedulabilityRefused) as info:
        MODULE.plan_roll(
            nodes=nodes, pods=pods, live_deployments=live, rendered=rendered
        )
    message = str(info.value)
    assert (
        "step 1 (gpu-fault-api-ha roll) cannot complete its surge-first roll" in message
    )
    assert "even with the deletion order steered" in message
    assert "new Pod 3 of 3 (4 cores / 4Gi) has no node whichever old Pod" in message
    assert "node-c has 3.0 cores" in message and "1 more core would fit it" in message
    assert "node-a already hosts 1 new Pod(s)" in message
    assert "free one 4-core / 4Gi slot on a second node" in message
    assert "or add a CPU node before applying" in message

    warned = MODULE.plan_roll(
        nodes=nodes, pods=pods, live_deployments=live, rendered=rendered, mode="warn"
    )
    assert warned["verdict"] == "warning"
    assert "even with the deletion order steered" in warned["warning"]


def test_only_surge_first_rolls_carry_a_steering_plan() -> None:
    nodes, pods, live = _live_node_group()
    evidence = MODULE.plan_roll(
        nodes=nodes, pods=pods, live_deployments=live, rendered=_rendered(workers=5)
    )
    steps = {step["role"]: step for step in evidence["steps"]}
    assert "steering" in steps["ingress"]
    assert "steering" not in steps["worker"], (
        "maxUnavailable 1 frees a slot before the surge Pod needs one"
    )
    assert "steering" not in steps["spool"]


def test_a_missing_surge_slot_keeps_the_shortfall_wording() -> None:
    nodes, pods, live = _live_node_group()
    for name in LIVE_NODES:
        pods["items"].append(_pod(name, "3", "1Gi", "filler"))
    with pytest.raises(MODULE.SchedulabilityRefused) as info:
        MODULE.plan_roll(
            nodes=nodes, pods=pods, live_deployments=live, rendered=_rendered(workers=5)
        )
    message = str(info.value)
    assert "cannot schedule one surge Pod (maxUnavailable 0, maxSurge 1)" in message
    assert "Pod 1 of 1 needs 4 cores / 4Gi; no node has room" in message
    assert "0.98 more free cores on one node would fit it" in message


def test_cli_evidence_records_the_steering_plan(tmp_path: Path) -> None:
    nodes, pods, live = _live_node_group()
    arguments = _write_inputs(tmp_path, nodes, pods, live, _rendered(workers=5))
    assert MODULE.main(arguments) == 0
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    steering = evidence["steps"][0]["steering"]
    assert steering["surge_nodes"] == [LIVE_NODES[0]]
    assert steering["progress_guaranteed"] is True
    assert steering["without_steering"]["progress_guaranteed"] is False
    assert [entry["cost"] for entry in steering["deletion_order"]] == [100, 200, 300]
    assert "free_after" not in steering, "the evidence carries placements, not scratch"
