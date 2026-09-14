#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

if __package__:
    from . import run_ha001_control_plane_failover as COMMON
    from .acceptance_runner_common import write_json_atomic
    from .acceptance_scope import current_acceptance_scope
    from .live_driver_guard import (
        add_live_arguments,
        applied_site_profile,
        details_sha256,
        install_site_profile,
    )
    from .live_driver_guard import (
        authorize_execution as guard_authorize_execution,
    )
    from .regional_live_fixture import (
        install_abort_signals,
        run_case_main,
    )
else:
    import run_ha001_control_plane_failover as COMMON
    from acceptance_runner_common import write_json_atomic
    from acceptance_scope import current_acceptance_scope
    from live_driver_guard import (
        add_live_arguments,
        applied_site_profile,
        details_sha256,
        install_site_profile,
    )
    from live_driver_guard import (
        authorize_execution as guard_authorize_execution,
    )
    from regional_live_fixture import (
        install_abort_signals,
        run_case_main,
    )

CASE_ID = "GF-REGIONAL-HA-002"
CONFIRMATION = "HA002_CORDON_AND_EVICT"
WATCHDOG_SECONDS = 1200
OBSERVATION_SECONDS = 60
ROLE_APPS = {
    "ingress": COMMON.INGRESS_APP,
    "worker": COMMON.WORKER_APP,
    "spool": COMMON.SPOOL_APP,
}
PDB_NAMES = {
    "ingress": "gpu-fault-api-ha-pdb",
    "worker": "gpu-fault-control-worker-pdb",
    "spool": "gpu-fault-telemetry-spool-worker-pdb",
}


class CaseError(RuntimeError):
    pass


def current_node(name: str) -> dict[str, Any]:
    value = json.loads(
        COMMON.run(
            [
                "kubectl",
                "--kubeconfig",
                str(COMMON.CPU_KUBECONFIG),
                "get",
                "node",
                name,
                "-o",
                "json",
            ]
        ).stdout
    )
    return {
        "name": value["metadata"]["name"],
        "uid": value["metadata"]["uid"],
        "unschedulable": value["spec"].get("unschedulable", False),
        "taints": value["spec"].get("taints", []),
        "ready": next(
            (
                item["status"]
                for item in value["status"]["conditions"]
                if item["type"] == "Ready"
            ),
            None,
        ),
    }


def pod_by_name(name: str) -> dict[str, Any] | None:
    result = COMMON.cpu("get", "pod", name, "-o", "json", check=False)
    if not result.strip():
        return None
    value = json.loads(result)
    statuses = value.get("status", {}).get("containerStatuses", [])
    return {
        "name": value["metadata"]["name"],
        "uid": value["metadata"]["uid"],
        "node": value["spec"].get("nodeName"),
        "app": value["metadata"].get("labels", {}).get("app"),
        "ready": bool(statuses) and all(bool(item.get("ready")) for item in statuses),
        "phase": value.get("status", {}).get("phase"),
    }


def eviction(pod_name: str) -> subprocess.CompletedProcess[str]:
    manifest = {
        "apiVersion": "policy/v1",
        "kind": "Eviction",
        "metadata": {"name": pod_name, "namespace": COMMON.NAMESPACE},
    }
    return COMMON.run(
        [
            "kubectl",
            "--kubeconfig",
            str(COMMON.CPU_KUBECONFIG),
            "create",
            "--raw",
            (f"/api/v1/namespaces/{COMMON.NAMESPACE}/pods/{pod_name}/eviction"),
            "-f",
            "-",
        ],
        stdin=json.dumps(manifest),
        check=False,
        timeout=60,
    )


def pdb_snapshot(name: str) -> dict[str, Any]:
    value = json.loads(COMMON.cpu("get", "pdb", name, "-o", "json"))
    return {
        "name": name,
        "current_healthy": value.get("status", {}).get("currentHealthy", 0),
        "desired_healthy": value.get("status", {}).get("desiredHealthy", 0),
        "disruptions_allowed": value.get("status", {}).get(
            "disruptionsAllowed",
            0,
        ),
    }


def wait_pdb_block(name: str, timeout_seconds: int = 30) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = pdb_snapshot(name)
        if int(last["disruptions_allowed"]) == 0:
            return last
        time.sleep(1)
    raise CaseError(f"PDB did not enter blocked state: {last}")


def require_pdb_still_blocked(
    name: str,
    *,
    snapshot: Callable[[str], dict[str, Any]] = pdb_snapshot,
) -> dict[str, Any]:
    """Re-read the PDB immediately before the second Eviction, and abort if it opened.

    Between ``wait_pdb_block`` and the second Eviction the replacement for the
    first evicted Pod can become Ready, which restores ``disruptionsAllowed``
    to 1. Issuing the second Eviction then removes a *second* Pod of the same
    role for real -- the case would be taking the role below its floor while
    reporting a PDB rejection it never received. The read happens right before
    the call and the case stops if the budget is open again.
    """

    current = snapshot(name)
    if int(current.get("disruptions_allowed", 0)) > 0:
        raise CaseError(
            f"{name} allows {current['disruptions_allowed']} disruption(s) again; "
            "not issuing the second Eviction because it would succeed"
        )
    return current


def require_eviction_success(
    result: subprocess.CompletedProcess[str], pod: str
) -> dict[str, Any]:
    if result.returncode != 0:
        raise CaseError(f"Eviction failed for {pod}: {result.stderr.strip()}")
    return {
        "pod": pod,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
    }


def require_pdb_rejection(
    result: subprocess.CompletedProcess[str], pod: str
) -> dict[str, Any]:
    combined = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode == 0:
        raise CaseError(f"second Eviction unexpectedly succeeded for {pod}")
    if "disruption budget" not in combined.lower():
        raise CaseError(f"second Eviction was not rejected by PDB: {combined}")
    return {
        "pod": pod,
        "returncode": result.returncode,
        "reason": "would violate PodDisruptionBudget",
    }


def start_uncordon_watchdog(
    case_dir: Path, node: str
) -> tuple[subprocess.Popen[str], Any]:
    log_path = case_dir / "uncordon-watchdog.log"
    handle = log_path.open("w")
    os.chmod(log_path, 0o600)
    command = (
        f"sleep {WATCHDOG_SECONDS}; exec kubectl "
        f"--kubeconfig {COMMON.CPU_KUBECONFIG} uncordon {node}"
    )
    process = subprocess.Popen(
        ["/bin/bash", "-c", command],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    write_json_atomic(
        case_dir / "uncordon-watchdog.json",
        {
            "pid": process.pid,
            "node": node,
            "delay_seconds": WATCHDOG_SECONDS,
        },
    )
    return process, handle


def stop_watchdog(process: subprocess.Popen[str] | None, handle: Any) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    if handle is not None:
        handle.close()


def wait_recovery(
    *,
    replicas: dict[str, int],
    timeout_seconds: int = 300,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    timeline = []
    while time.monotonic() < deadline:
        sample = COMMON.control_sample(include_queue=True)
        timeline.append(sample)
        if (
            sample["ingress_ready"] == replicas[COMMON.INGRESS_APP]
            and sample["worker_ready"] == replicas[COMMON.WORKER_APP]
            and sample["endpoint_ready"] == replicas[COMMON.INGRESS_APP]
            and sample["spool_ready"] == replicas[COMMON.SPOOL_APP]
            and int(sample["queue"]["depth"]) <= 5
        ):
            return timeline
        time.sleep(2)
    raise CaseError(f"control plane did not recover: {timeline[-1:]}")


def topology_snapshot() -> dict[str, Any]:
    return {
        "ingress": COMMON.ready_pods(COMMON.INGRESS_APP),
        "workers": COMMON.ready_pods(COMMON.WORKER_APP),
        "spool_workers": COMMON.ready_pods(COMMON.SPOOL_APP),
    }


def schedulable_cpu_nodes(nodes: dict[str, Any]) -> list[str]:
    """Ready, uncordoned, untainted node names from a ``kubectl get nodes`` document."""

    result = []
    for node in nodes.get("items", []):
        ready = next(
            (
                value["status"]
                for value in node.get("status", {}).get("conditions", [])
                if value.get("type") == "Ready"
            ),
            None,
        )
        if (
            ready == "True"
            and not node.get("spec", {}).get("unschedulable", False)
            and not node.get("spec", {}).get("taints", [])
        ):
            result.append(str(node["metadata"]["name"]))
    return sorted(result)


def build_plan(run_dir: Path, attempt: int) -> dict[str, Any]:
    nodes = json.loads(
        COMMON.run(
            [
                "kubectl",
                "--kubeconfig",
                str(COMMON.CPU_KUBECONFIG),
                "get",
                "nodes",
                "-o",
                "json",
            ]
        ).stdout
    )
    pods = json.loads(COMMON.cpu("get", "pods", "-o", "json"))
    active_by_node: dict[str, list[dict[str, Any]]] = {}
    for item in pods.get("items", []):
        if item.get("status", {}).get("phase") not in {"Pending", "Running"}:
            continue
        node = item.get("spec", {}).get("nodeName")
        if not node:
            continue
        statuses = item.get("status", {}).get("containerStatuses", [])
        active_by_node.setdefault(node, []).append(
            {
                "name": item["metadata"]["name"],
                "uid": item["metadata"]["uid"],
                "app": item["metadata"].get("labels", {}).get("app"),
                "ready": bool(statuses)
                and all(bool(status.get("ready")) for status in statuses),
            }
        )
    topology = COMMON.deployment_and_pdb_snapshot()
    replicas = COMMON.declared_replicas(topology)
    spool_replicas = replicas[COMMON.SPOOL_APP]
    if spool_replicas == 1:
        raise CaseError(
            "enabled spool-worker needs at least two replicas to verify PDB rejection"
        )
    cpu_nodes = schedulable_cpu_nodes(nodes)
    candidates = []
    for node in nodes.get("items", []):
        name = node["metadata"]["name"]
        if name not in cpu_nodes:
            continue
        node_pods = active_by_node.get(name, [])
        ingress = [
            item
            for item in node_pods
            if item["app"] == COMMON.INGRESS_APP and item["ready"]
        ]
        workers = [
            item
            for item in node_pods
            if item["app"] == COMMON.WORKER_APP and item["ready"]
        ]
        spool_workers = [
            item
            for item in node_pods
            if item["app"] == COMMON.SPOOL_APP and item["ready"]
        ]
        extras = [
            item for item in node_pods if item["app"] not in set(ROLE_APPS.values())
        ]
        if (
            len(ingress) == 1
            and len(workers) >= 2
            and (spool_replicas == 0 or spool_workers)
            and not extras
        ):
            candidates.append(
                {
                    "node": {
                        "name": name,
                        "uid": node["metadata"]["uid"],
                        "ready": True,
                        "unschedulable": False,
                        "taints": [],
                    },
                    "ingress": ingress,
                    "workers": sorted(workers, key=lambda value: value["name"]),
                    "spool_workers": sorted(
                        spool_workers,
                        key=lambda value: value["name"],
                    ),
                }
            )
    if not candidates:
        raise CaseError(
            "no clean CPU node carries one ingress and at least two workers"
        )
    selected = sorted(candidates, key=lambda item: item["node"]["name"])[0]
    limits = {
        app: COMMON.failure_window_limit(topology["deployments"][app])
        for app in (COMMON.INGRESS_APP, COMMON.WORKER_APP)
    }
    scope = current_acceptance_scope()
    details = {
        "risk": "live-control-plane-pdb-eviction",
        "mutation": (
            "cordon one CPU node and issue Eviction API calls against its "
            "same-role Pods to prove the PodDisruptionBudget rejects the "
            "second same-role disruption; no Deployment spec change; the "
            "node is uncordoned in a finally block"
        ),
        "target_node": selected["node"]["name"],
        "target_pods": {
            "ingress": [selected["ingress"][0]["name"]],
            "workers": [item["name"] for item in selected["workers"][:2]],
            "spool_worker": [item["name"] for item in selected["spool_workers"][:1]],
        },
        "failure_window_limits": {
            app: item["limit_seconds"] for app, item in limits.items()
        },
        "maintenance_window_required_at_execute": True,
    }
    plan = {
        "schema_version": 2,
        "case_id": CASE_ID,
        "attempt": attempt,
        "confirmation": CONFIRMATION,
        "environment": COMMON.environment_values(),
        "site_profile": applied_site_profile(),
        **scope.plan_fields(),
        "details": details,
        "details_sha256": details_sha256(details),
        "mutation_performed": False,
        "region": COMMON.AWS_REGION,
        "maintenance_window_required_at_execute": True,
        "target_node": selected["node"],
        "target_pods": {
            "ingress": [selected["ingress"][0]["name"]],
            "workers": [item["name"] for item in selected["workers"][:2]],
            "spool_worker": [item["name"] for item in selected["spool_workers"][:1]],
        },
        "replicas": replicas,
        "cpu_nodes": cpu_nodes,
        "failure_window_limits": limits,
        "baseline": {
            "queue": COMMON.queue_stats(),
            "topology": topology,
            "spool_worker_replicas": spool_replicas,
        },
        "stop_conditions": [
            "NLB/API becomes unavailable",
            f"ingress Ready count falls below {replicas[COMMON.INGRESS_APP] - 1}",
            f"worker Ready count falls below {replicas[COMMON.WORKER_APP] - 1}",
            "an Eviction affects a Pod outside the named role",
            "the PDB allows a disruption again before the second same-role Eviction",
            "the second same-role Eviction is not rejected by the PDB",
            "processor queue exceeds the predeclared threshold",
            "uncordon or declared replica recovery fails",
        ],
        "rollback": {
            "always_uncordon_in_finally": True,
            "detached_uncordon_watchdog_seconds": WATCHDOG_SECONDS,
            "preserve_preexisting_taints": True,
            "wait_for_declared_replicas": True,
        },
    }
    path = run_dir / "cases" / CASE_ID / "plan.json"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_json_atomic(path, plan)
    return plan


def execute_context(plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    if "replicas" not in plan or "failure_window_limits" not in plan:
        raise CaseError("plan predates the derived replica fields; re-plan")
    node_name = str(plan["target_node"]["name"])
    baseline_node = current_node(node_name)
    if (
        baseline_node["uid"] != plan["target_node"].get("uid", baseline_node["uid"])
        or baseline_node["unschedulable"]
        or baseline_node["taints"] != plan["target_node"]["taints"]
        or baseline_node["ready"] != "True"
    ):
        raise CaseError(f"target node drifted: {baseline_node}")
    replicas = {app: int(value) for app, value in plan["replicas"].items()}
    ingress_target = str(plan["target_pods"]["ingress"][0])
    worker_targets = [str(value) for value in plan["target_pods"]["workers"]]
    spool_targets = [
        str(value) for value in plan["target_pods"].get("spool_worker", [])
    ]
    ingress_peers = [
        item["name"]
        for item in COMMON.ready_pods(COMMON.INGRESS_APP)
        if item["name"] != ingress_target
    ]
    if len(ingress_peers) != replicas[COMMON.INGRESS_APP] - 1:
        raise CaseError("could not select a second ingress for PDB rejection")
    planned_pods = [
        (ingress_target, COMMON.INGRESS_APP),
        *[(name, COMMON.WORKER_APP) for name in worker_targets],
        *[(name, COMMON.SPOOL_APP) for name in spool_targets],
    ]
    for name, app in planned_pods:
        pod = pod_by_name(name)
        if (
            pod is None
            or pod["app"] != app
            or pod["node"] != node_name
            or not pod["ready"]
        ):
            raise CaseError(f"planned Pod drifted: {name}: {pod}")
    queue_baseline = plan["baseline"].get("queue")
    if not isinstance(queue_baseline, dict):
        queue_baseline = COMMON.queue_stats()
        plan["baseline"]["queue"] = queue_baseline
        write_json_atomic(plan_path, plan)
    return {
        "plan": plan,
        "node_name": node_name,
        "baseline_node": baseline_node,
        "ingress_target": ingress_target,
        "ingress_peer": str(ingress_peers[0]),
        "worker_targets": worker_targets,
        "spool_targets": spool_targets,
        "replicas": replicas,
        "minimum_ready": COMMON.minimum_ready_from_replicas(replicas),
        "failure_window_limit_seconds": max(
            float(item["limit_seconds"])
            for item in plan["failure_window_limits"].values()
        ),
        "spool_replicas": replicas[COMMON.SPOOL_APP],
        "cpu_nodes": [str(value) for value in plan.get("cpu_nodes", [])],
        "baseline_depth": int(queue_baseline["depth"]),
    }


def _observe(
    context: dict[str, Any], label: str, duration: int, **kwargs: Any
) -> dict[str, Any]:
    return COMMON.observe_phase(
        label,
        duration,
        context["baseline_depth"],
        minimum_ready=kwargs.pop("minimum_ready", context["minimum_ready"]),
        max_failure_window_seconds=context["failure_window_limit_seconds"],
        **kwargs,
    )


def _evict_role(
    context: dict[str, Any],
    case_dir: Path,
    *,
    role: str,
    first: str,
    second: str,
    minimum_ready: dict[str, int | None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Evict ``first``, wait for the PDB to close, re-check it, then try ``second``."""

    first_result = require_eviction_success(eviction(first), first)
    blocked = wait_pdb_block(PDB_NAMES[role])
    recheck = require_pdb_still_blocked(PDB_NAMES[role])
    second_result = require_pdb_rejection(eviction(second), second)
    phase = _observe(
        context,
        f"ha002-{role}",
        OBSERVATION_SECONDS,
        **({"minimum_ready": minimum_ready} if minimum_ready is not None else {}),
    )
    write_json_atomic(case_dir / f"{role}-timeline.json", phase)
    return (
        first_result,
        {**blocked, "recheck_before_second": recheck},
        second_result,
        phase,
    )


def _run_disruptions(
    case_dir: Path, context: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    COMMON.create_probe()
    state["probe_created"] = True
    COMMON.wait_probe_file("/state/stats.json", 60)
    baseline_phase = _observe(context, "ha002-baseline", 10)
    write_json_atomic(case_dir / "baseline-timeline.json", baseline_phase)
    roles_before = COMMON.role_snapshot()
    write_json_atomic(case_dir / "roles-before.json", roles_before)
    role_errors = COMMON.validate_roles(roles_before, context["replicas"])
    if role_errors:
        raise CaseError(f"role baseline failed: {role_errors}")

    watchdog, handle = start_uncordon_watchdog(case_dir, context["node_name"])
    state.update({"watchdog": watchdog, "watchdog_handle": handle})
    COMMON.log(f"cordoning CPU node {context['node_name']}")
    COMMON.run(
        [
            "kubectl",
            "--kubeconfig",
            str(COMMON.CPU_KUBECONFIG),
            "cordon",
            context["node_name"],
        ]
    )
    state["cordoned"] = True
    cordoned_state = current_node(context["node_name"])
    if not cordoned_state["unschedulable"]:
        raise CaseError("node did not become unschedulable")
    write_json_atomic(case_dir / "cordoned-node.json", cordoned_state)

    ingress_first, ingress_pdb_blocked, ingress_second, ingress_phase = _evict_role(
        context,
        case_dir,
        role="ingress",
        first=context["ingress_target"],
        second=context["ingress_peer"],
    )
    worker_first, worker_second = context["worker_targets"]
    (
        worker_first_result,
        worker_pdb_blocked,
        worker_second_result,
        worker_phase,
    ) = _evict_role(
        context,
        case_dir,
        role="worker",
        first=worker_first,
        second=worker_second,
    )
    spool_result, spool_phase = _run_spool_disruption(case_dir, context)

    COMMON.log(f"uncordoning CPU node {context['node_name']}")
    COMMON.run(
        [
            "kubectl",
            "--kubeconfig",
            str(COMMON.CPU_KUBECONFIG),
            "uncordon",
            context["node_name"],
        ]
    )
    state["cordoned"] = False
    recovery = wait_recovery(replicas=context["replicas"])
    write_json_atomic(case_dir / "recovery-timeline.json", {"entries": recovery})
    return _ha002_result(
        context,
        cordoned_state,
        ingress_first,
        ingress_pdb_blocked,
        ingress_second,
        ingress_phase,
        worker_first_result,
        worker_pdb_blocked,
        worker_second_result,
        worker_phase,
        spool_result,
        spool_phase,
        roles_before,
    )


def _run_spool_disruption(
    case_dir: Path, context: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result: dict[str, Any] = {
        "replicas": context["spool_replicas"],
        "status": "NOT_APPLICABLE",
    }
    if not context["spool_targets"]:
        return result, None
    spool_target = context["spool_targets"][0]
    spool_peers = [
        item["name"]
        for item in COMMON.ready_pods(COMMON.SPOOL_APP)
        if item["name"] != spool_target
    ]
    if not spool_peers:
        raise CaseError("no second spool-worker Pod for PDB rejection")
    first, blocked, second, phase = _evict_role(
        context,
        case_dir,
        role="spool",
        first=spool_target,
        second=str(spool_peers[0]),
        minimum_ready={
            **context["minimum_ready"],
            "spool": context["spool_replicas"] - 1,
        },
    )
    return {
        "replicas": context["spool_replicas"],
        "status": "TESTED",
        "first_eviction": first,
        "pdb_blocked": blocked,
        "second_eviction": second,
        "summary": phase["summary"],
    }, phase


def _distribution(items: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        result[item["node"]] = result.get(item["node"], 0) + 1
    return result


def balanced_distribution(distribution: dict[str, int], replicas: int) -> bool:
    """Whether ``replicas`` Pods sit across nodes with maxSkew 1, as the spread demands.

    The old check compared against ``[1, 1, 1]``/``[2, 2, 2]`` -- three nodes
    and 3/6 replicas assumed. The property the topology spread actually
    guarantees is that the per-node counts differ by at most one and add up to
    the declared replicas, whatever both numbers are.
    """

    counts = list(distribution.values())
    if sum(counts) != replicas:
        return False
    if not counts:
        return replicas == 0
    return max(counts) - min(counts) <= 1


def capacity_assessment(
    *,
    cpu_nodes: int,
    ingress_replicas: int,
    ingress_ready_during_cordon: int,
    max_skew: int = 1,
) -> dict[str, Any]:
    """A capacity recommendation derived from what the cordon showed.

    Under ``DoNotSchedule`` with ``maxSkew=1`` the replacement for an evicted
    ingress Pod can only land on a node that keeps the skew within one; with
    the node under maintenance cordoned that leaves ``cpu_nodes - 1``
    candidates. The recommendation is made only when the observation confirms
    the shortfall (the role did not return to its declared count while the
    node was cordoned) and the arithmetic explains it.
    """

    schedulable_during_maintenance = max(0, cpu_nodes - 1)
    shortfall = ingress_replicas - ingress_ready_during_cordon
    constrained = ingress_replicas > schedulable_during_maintenance * max_skew
    if shortfall > 0 and constrained:
        recommendation = (
            f"add a CPU node: {cpu_nodes} nodes leave {schedulable_during_maintenance} "
            f"schedulable during one-node maintenance, so only "
            f"{ingress_ready_during_cordon} of {ingress_replicas} ingress replicas "
            f"were Ready while the node was cordoned (maxSkew={max_skew})"
        )
    elif shortfall > 0:
        recommendation = (
            f"ingress stayed at {ingress_ready_during_cordon} of {ingress_replicas} "
            "while cordoned although node capacity allows the spread; inspect "
            "scheduling events before adding capacity"
        )
    else:
        recommendation = (
            f"no additional CPU node required: all {ingress_replicas} ingress "
            "replicas were Ready while one node was cordoned"
        )
    return {
        "cpu_nodes": cpu_nodes,
        "schedulable_during_maintenance": schedulable_during_maintenance,
        "ingress_replicas": ingress_replicas,
        "ingress_ready_during_cordon": ingress_ready_during_cordon,
        "shortfall": shortfall,
        "topology_constrained": constrained,
        "recommendation": recommendation,
    }


def _ha002_result(
    context: dict[str, Any],
    cordoned_state: dict[str, Any],
    ingress_first: dict[str, Any],
    ingress_pdb_blocked: dict[str, Any],
    ingress_second: dict[str, Any],
    ingress_phase: dict[str, Any],
    worker_first_result: dict[str, Any],
    worker_pdb_blocked: dict[str, Any],
    worker_second_result: dict[str, Any],
    worker_phase: dict[str, Any],
    spool_result: dict[str, Any],
    spool_phase: dict[str, Any] | None,
    roles_before: dict[str, Any],
) -> dict[str, Any]:
    replicas = context["replicas"]
    minimum_ready = context["minimum_ready"]
    roles_after = COMMON.role_snapshot()
    topology = topology_snapshot()
    final_probe = COMMON.read_probe()
    errors = [*COMMON.validate_roles(roles_after, replicas)]
    if ingress_phase["summary"]["min_ingress_ready"] < int(
        minimum_ready["ingress"] or 0
    ):
        errors.append(f"ingress Ready fell below {minimum_ready['ingress']}")
    if worker_phase["summary"]["min_worker_ready"] < int(minimum_ready["worker"] or 0):
        errors.append(f"worker Ready fell below {minimum_ready['worker']}")
    if spool_phase is not None and (
        spool_phase["summary"]["min_spool_ready"] < context["spool_replicas"] - 1
    ):
        errors.append("spool-worker Ready fell below its PDB limit")
    if ingress_phase["summary"]["min_endpoint_ready"] < int(
        minimum_ready["ingress"] or 0
    ):
        errors.append(f"NLB endpoint count fell below {minimum_ready['ingress']}")
    measured_window = float(final_probe.get("max_failure_window_seconds", 0))
    if measured_window > context["failure_window_limit_seconds"]:
        errors.append(
            "probe failure window exceeded the derived limit "
            f"{context['failure_window_limit_seconds']}s"
        )
    node_final = current_node(context["node_name"])
    if node_final["unschedulable"] or (
        node_final["taints"] != context["baseline_node"]["taints"]
    ):
        errors.append("node scheduling state was not restored")
    ingress_distribution = _distribution(topology["ingress"])
    worker_distribution = _distribution(topology["workers"])
    spool_distribution = _distribution(topology["spool_workers"])
    if not balanced_distribution(ingress_distribution, replicas[COMMON.INGRESS_APP]):
        errors.append("ingress topology did not return to a balanced spread")
    if not balanced_distribution(worker_distribution, replicas[COMMON.WORKER_APP]):
        errors.append("worker topology did not return to a balanced spread")
    last_cordoned_sample = (
        ingress_phase["samples"][-1] if ingress_phase["samples"] else {}
    )
    return {
        "case_id": CASE_ID,
        "attempt": context["attempt"],
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "plan": context["plan"],
        "replicas": replicas,
        "baseline_node": context["baseline_node"],
        "cordoned_node": cordoned_state,
        "ingress": {
            "first_eviction": ingress_first,
            "pdb_blocked": ingress_pdb_blocked,
            "second_eviction": ingress_second,
            "summary": ingress_phase["summary"],
        },
        "worker": {
            "first_eviction": worker_first_result,
            "pdb_blocked": worker_pdb_blocked,
            "second_eviction": worker_second_result,
            "summary": worker_phase["summary"],
        },
        "spool_worker": spool_result,
        "roles_before": roles_before,
        "roles_after": roles_after,
        "topology": {
            "ingress_by_node": ingress_distribution,
            "worker_by_node": worker_distribution,
            "spool_by_node": spool_distribution,
        },
        "failure_window": {
            "measured_max_seconds": measured_window,
            "limit_seconds": context["failure_window_limit_seconds"],
            "limits": context["plan"]["failure_window_limits"],
        },
        "final_probe": final_probe,
        "node_final": node_final,
        "capacity_assessment": capacity_assessment(
            cpu_nodes=len(context["cpu_nodes"]),
            ingress_replicas=replicas[COMMON.INGRESS_APP],
            ingress_ready_during_cordon=int(
                last_cordoned_sample.get("ingress_ready", 0)
            ),
        ),
    }


def cleanup_case(
    case_dir: Path,
    context: dict[str, Any],
    state: dict[str, Any],
    result: dict[str, Any],
) -> list[str]:
    """Every cleanup step on its own; a failing step never skips the next one."""

    errors: list[str] = []

    def attempt(label: str, action: Callable[[], Any]) -> None:
        try:
            action()
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    if state["cordoned"]:
        attempt(
            "uncordon",
            lambda: COMMON.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    str(COMMON.CPU_KUBECONFIG),
                    "uncordon",
                    context["node_name"],
                ],
                check=False,
            ),
        )
    attempt(
        "stop watchdog",
        lambda: stop_watchdog(state["watchdog"], state["watchdog_handle"]),
    )
    if state["probe_created"]:
        attempt(
            "stop probe",
            lambda: COMMON.gpu(
                "exec",
                COMMON.PROBE_POD,
                "--",
                "touch",
                "/state/stop",
                check=False,
            ),
        )
    attempt(
        "delete probe pod",
        lambda: COMMON.gpu(
            "delete", "pod", COMMON.PROBE_POD, "--ignore-not-found", check=False
        ),
    )
    attempt(
        "delete probe configmap",
        lambda: COMMON.gpu(
            "delete",
            "configmap",
            COMMON.CONFIGMAP,
            "--ignore-not-found",
            check=False,
        ),
    )
    deployments = [COMMON.INGRESS_APP, COMMON.WORKER_APP]
    if context["spool_replicas"]:
        deployments.append(COMMON.SPOOL_APP)
    for deployment in deployments:
        attempt(
            f"rollout status {deployment}",
            functools.partial(
                COMMON.cpu,
                "rollout",
                "status",
                f"deployment/{deployment}",
                "--timeout=600s",
                timeout=700,
            ),
        )

    def postflight() -> None:
        availability = COMMON.control_sample(include_queue=True)
        record = {
            "node": current_node(context["node_name"]),
            "probe_resources": COMMON.probe_resources(),
            "queue": availability["queue"],
            "spool_ready": availability["spool_ready"],
        }
        result["postflight"] = record
        write_json_atomic(case_dir / "postflight.json", record)
        if record["node"]["unschedulable"]:
            raise CaseError("node remains cordoned")
        if record["node"]["taints"] != context["baseline_node"]["taints"]:
            raise CaseError("node taints changed")
        if record["probe_resources"]["count"] != 0:
            raise CaseError("probe resources remain")
        if record["spool_ready"] != context["spool_replicas"]:
            raise CaseError("spool-worker replicas did not recover")

    attempt("postflight", postflight)
    return errors


def execute(run_dir: Path, attempt: int, confirmation: str) -> int:
    if confirmation != CONFIRMATION:
        raise CaseError(f"confirmation must be exactly {CONFIRMATION}")
    case_dir = run_dir / "cases" / CASE_ID
    plan_path = case_dir / "plan.json"
    if not plan_path.is_file():
        raise CaseError("HA-002 plan is missing")
    context = execute_context(json.loads(plan_path.read_text()), plan_path)
    context["attempt"] = attempt
    result: dict[str, Any] = {"case_id": CASE_ID, "attempt": attempt, "verdict": "FAIL"}
    state: dict[str, Any] = {
        "probe_created": False,
        "watchdog": None,
        "watchdog_handle": None,
        "cordoned": False,
    }
    try:
        result = _run_disruptions(case_dir, context, state)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_errors = cleanup_case(case_dir, context, state, result)
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
    install_site_profile()
    parser = argparse.ArgumentParser()
    add_live_arguments(parser, confirmation=CONFIRMATION)
    parser.add_argument("--cpu-kubeconfig", default="")
    parser.add_argument("--gpu-kubeconfig", default="")
    parser.add_argument("--gpu-context", default="")
    parser.add_argument("--namespace", default="gpu-fault-system")
    parser.add_argument("--region", default="")
    args = parser.parse_args()
    os.umask(0o077)
    COMMON.configure(args)
    install_abort_signals()
    if not args.execute:
        plan = build_plan(args.run_dir, args.attempt)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    deadline = guard_authorize_execution(
        args,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=COMMON.environment_values(),
    )
    plan_path = args.run_dir / "cases" / CASE_ID / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["maintenance_window_end"] = deadline.isoformat()
    write_json_atomic(plan_path, plan)
    return execute(args.run_dir, args.attempt, args.confirm)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
