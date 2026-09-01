#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

if __package__:
    from .acceptance_scope import current_acceptance_scope, scoped_case_evidence
    from .live_driver_guard import (
        add_live_arguments,
    )
    from .live_driver_guard import (
        authorize_execution as guard_authorize_execution,
    )
else:
    from acceptance_scope import current_acceptance_scope, scoped_case_evidence
    from live_driver_guard import (
        add_live_arguments,
    )
    from live_driver_guard import (
        authorize_execution as guard_authorize_execution,
    )

HERE = Path(__file__).resolve().parent
COMMON_SPEC = importlib.util.spec_from_file_location(
    "ha001_common",
    HERE / "run_ha001_control_plane_failover.py",
)
if COMMON_SPEC is None or COMMON_SPEC.loader is None:
    raise RuntimeError("could not load HA-001 common helpers")
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
COMMON_SPEC.loader.exec_module(COMMON)

CASE_ID = "GF-REGIONAL-HA-002"
CONFIRMATION = "HA002_CORDON_AND_EVICT"
WATCHDOG_SECONDS = 1200
OBSERVATION_SECONDS = 60


class CaseError(RuntimeError):
    pass


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(scoped_case_evidence(value), indent=2, sort_keys=True) + "\n"
    )
    path.chmod(0o600)


def current_node(name: str) -> dict:
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


def pod_by_name(name: str) -> dict | None:
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


def pdb_snapshot(name: str) -> dict:
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


def wait_pdb_block(name: str, timeout_seconds: int = 30) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = {}
    while time.monotonic() < deadline:
        last = pdb_snapshot(name)
        if int(last["disruptions_allowed"]) == 0:
            return last
        time.sleep(1)
    raise CaseError(f"PDB did not enter blocked state: {last}")


def require_eviction_success(
    result: subprocess.CompletedProcess[str], pod: str
) -> dict:
    if result.returncode != 0:
        raise CaseError(f"Eviction failed for {pod}: {result.stderr.strip()}")
    return {
        "pod": pod,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
    }


def require_pdb_rejection(result: subprocess.CompletedProcess[str], pod: str) -> dict:
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
) -> tuple[subprocess.Popen[str], object]:
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
    write_json(
        case_dir / "uncordon-watchdog.json",
        {
            "pid": process.pid,
            "node": node,
            "delay_seconds": WATCHDOG_SECONDS,
        },
    )
    return process, handle


def stop_watchdog(process: subprocess.Popen[str] | None, handle: object | None) -> None:
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
    expected_spool_ready: int,
    timeout_seconds: int = 300,
) -> list[dict]:
    deadline = time.monotonic() + timeout_seconds
    timeline = []
    while time.monotonic() < deadline:
        sample = COMMON.control_sample(include_queue=True)
        timeline.append(sample)
        if (
            sample["ingress_ready"] == 3
            and sample["worker_ready"] == 6
            and sample["endpoint_ready"] == 3
            and sample["spool_ready"] == expected_spool_ready
            and int(sample["queue"]["depth"]) <= 5
        ):
            return timeline
        time.sleep(2)
    raise CaseError(f"control plane did not recover: {timeline[-1:]}")


def topology_snapshot() -> dict:
    return {
        "ingress": COMMON.ready_pods("gpu-fault-api-ha"),
        "workers": COMMON.ready_pods("gpu-fault-control-worker"),
        "spool_workers": COMMON.ready_pods("gpu-fault-telemetry-spool-worker"),
    }


def build_plan(run_dir: Path, attempt: int) -> dict:
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
    active_by_node: dict[str, list[dict]] = {}
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
    spool = topology["deployments"]["gpu-fault-telemetry-spool-worker"]
    spool_replicas = int(spool["replicas"])
    if spool_replicas == 1:
        raise CaseError(
            "enabled spool-worker needs at least two replicas to verify PDB rejection"
        )
    candidates = []
    for node in nodes.get("items", []):
        name = node["metadata"]["name"]
        ready = next(
            (
                value["status"]
                for value in node.get("status", {}).get("conditions", [])
                if value.get("type") == "Ready"
            ),
            None,
        )
        node_pods = active_by_node.get(name, [])
        ingress = [
            item
            for item in node_pods
            if item["app"] == "gpu-fault-api-ha" and item["ready"]
        ]
        workers = [
            item
            for item in node_pods
            if item["app"] == "gpu-fault-control-worker" and item["ready"]
        ]
        spool_workers = [
            item
            for item in node_pods
            if item["app"] == "gpu-fault-telemetry-spool-worker" and item["ready"]
        ]
        extras = [
            item
            for item in node_pods
            if item["app"]
            not in {
                "gpu-fault-api-ha",
                "gpu-fault-control-worker",
                "gpu-fault-telemetry-spool-worker",
            }
        ]
        if (
            ready == "True"
            and not node.get("spec", {}).get("unschedulable", False)
            and not node.get("spec", {}).get("taints", [])
            and len(ingress) == 1
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
    scope = current_acceptance_scope()
    plan = {
        "schema_version": 2,
        "case_id": CASE_ID,
        "attempt": attempt,
        "confirmation": CONFIRMATION,
        "environment": COMMON.environment_values(),
        **scope.plan_fields(),
        "mutation_performed": False,
        "region": COMMON.AWS_REGION,
        "maintenance_window_required_at_execute": True,
        "target_node": selected["node"],
        "target_pods": {
            "ingress": [selected["ingress"][0]["name"]],
            "workers": [item["name"] for item in selected["workers"][:2]],
            "spool_worker": [item["name"] for item in selected["spool_workers"][:1]],
        },
        "baseline": {
            "queue": COMMON.queue_stats(),
            "topology": topology,
            "spool_worker_replicas": spool["replicas"],
        },
        "stop_conditions": [
            "NLB/API becomes unavailable",
            "ingress Ready count falls below 2",
            "worker Ready count falls below 5",
            "an Eviction affects a Pod outside the named role",
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
    write_json(path, plan)
    return plan


def _execute_context(plan: dict, plan_path: Path) -> dict:
    node_name = str(plan["target_node"]["name"])
    baseline_node = current_node(node_name)
    if (
        baseline_node["uid"] != plan["target_node"].get("uid", baseline_node["uid"])
        or baseline_node["unschedulable"]
        or baseline_node["taints"] != plan["target_node"]["taints"]
        or baseline_node["ready"] != "True"
    ):
        raise CaseError(f"target node drifted: {baseline_node}")
    ingress_target = str(plan["target_pods"]["ingress"][0])
    worker_targets = [str(value) for value in plan["target_pods"]["workers"]]
    spool_targets = [
        str(value) for value in plan["target_pods"].get("spool_worker", [])
    ]
    ingress_peers = [
        item["name"]
        for item in COMMON.ready_pods("gpu-fault-api-ha")
        if item["name"] != ingress_target
    ]
    if len(ingress_peers) != 2:
        raise CaseError("could not select a second ingress for PDB rejection")
    planned_pods = [
        (ingress_target, "gpu-fault-api-ha"),
        *[(name, "gpu-fault-control-worker") for name in worker_targets],
        *[(name, "gpu-fault-telemetry-spool-worker") for name in spool_targets],
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
        write_json(plan_path, plan)
    return {
        "plan": plan,
        "node_name": node_name,
        "baseline_node": baseline_node,
        "ingress_target": ingress_target,
        "ingress_peer": str(ingress_peers[0]),
        "worker_targets": worker_targets,
        "spool_targets": spool_targets,
        "spool_replicas": int(plan["baseline"]["spool_worker_replicas"]),
        "baseline_depth": int(queue_baseline["depth"]),
    }


def _run_disruptions(case_dir: Path, context: dict, state: dict) -> dict:
    COMMON.create_probe()
    state["probe_created"] = True
    COMMON.wait_probe_file("/state/stats.json", 60)
    baseline_phase = COMMON.observe_phase(
        "ha002-baseline", 10, context["baseline_depth"]
    )
    write_json(case_dir / "baseline-timeline.json", baseline_phase)
    roles_before = COMMON.role_snapshot()
    write_json(case_dir / "roles-before.json", roles_before)
    role_errors = COMMON.validate_roles(roles_before)
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
    write_json(case_dir / "cordoned-node.json", cordoned_state)

    ingress_first = require_eviction_success(
        eviction(context["ingress_target"]),
        context["ingress_target"],
    )
    ingress_pdb_blocked = wait_pdb_block("gpu-fault-api-ha-pdb")
    ingress_second = require_pdb_rejection(
        eviction(context["ingress_peer"]),
        context["ingress_peer"],
    )
    ingress_phase = COMMON.observe_phase(
        "ha002-ingress",
        OBSERVATION_SECONDS,
        context["baseline_depth"],
    )
    write_json(case_dir / "ingress-timeline.json", ingress_phase)

    worker_first, worker_second = context["worker_targets"]
    worker_first_result = require_eviction_success(eviction(worker_first), worker_first)
    worker_pdb_blocked = wait_pdb_block("gpu-fault-control-worker-pdb")
    worker_second_result = require_pdb_rejection(eviction(worker_second), worker_second)
    worker_phase = COMMON.observe_phase(
        "ha002-worker",
        OBSERVATION_SECONDS,
        context["baseline_depth"],
    )
    write_json(case_dir / "worker-timeline.json", worker_phase)
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
    recovery = wait_recovery(expected_spool_ready=context["spool_replicas"])
    write_json(case_dir / "recovery-timeline.json", recovery)
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


def _run_spool_disruption(case_dir: Path, context: dict) -> tuple[dict, dict | None]:
    result = {"replicas": context["spool_replicas"], "status": "NOT_APPLICABLE"}
    if not context["spool_targets"]:
        return result, None
    spool_target = context["spool_targets"][0]
    spool_peers = [
        item["name"]
        for item in COMMON.ready_pods("gpu-fault-telemetry-spool-worker")
        if item["name"] != spool_target
    ]
    if not spool_peers:
        raise CaseError("no second spool-worker Pod for PDB rejection")
    first = require_eviction_success(eviction(spool_target), spool_target)
    blocked = wait_pdb_block("gpu-fault-telemetry-spool-worker-pdb")
    second = require_pdb_rejection(eviction(str(spool_peers[0])), str(spool_peers[0]))
    phase = COMMON.observe_phase(
        "ha002-spool",
        OBSERVATION_SECONDS,
        context["baseline_depth"],
        minimum_spool_ready=context["spool_replicas"] - 1,
    )
    write_json(case_dir / "spool-timeline.json", phase)
    return {
        "replicas": context["spool_replicas"],
        "status": "TESTED",
        "first_eviction": first,
        "pdb_blocked": blocked,
        "second_eviction": second,
        "summary": phase["summary"],
    }, phase


def _distribution(items: list[dict]) -> dict:
    result = {}
    for item in items:
        result[item["node"]] = result.get(item["node"], 0) + 1
    return result


def _ha002_result(
    context: dict,
    cordoned_state: dict,
    ingress_first: dict,
    ingress_pdb_blocked: dict,
    ingress_second: dict,
    ingress_phase: dict,
    worker_first_result: dict,
    worker_pdb_blocked: dict,
    worker_second_result: dict,
    worker_phase: dict,
    spool_result: dict,
    spool_phase: dict | None,
    roles_before: dict,
) -> dict:
    roles_after = COMMON.role_snapshot()
    topology = topology_snapshot()
    final_probe = COMMON.read_probe()
    errors = [*COMMON.validate_roles(roles_after)]
    if ingress_phase["summary"]["min_ingress_ready"] < 2:
        errors.append("ingress Ready fell below two")
    if worker_phase["summary"]["min_worker_ready"] < 5:
        errors.append("worker Ready fell below five")
    if spool_phase is not None and (
        spool_phase["summary"]["min_spool_ready"] < context["spool_replicas"] - 1
    ):
        errors.append("spool-worker Ready fell below its PDB limit")
    if ingress_phase["summary"]["min_endpoint_ready"] < 2:
        errors.append("NLB endpoint count fell below two")
    if final_probe.get("max_failure_window_seconds", 0) > 55:
        errors.append("probe failure window exceeded 55 seconds")
    node_final = current_node(context["node_name"])
    if node_final["unschedulable"] or (
        node_final["taints"] != context["baseline_node"]["taints"]
    ):
        errors.append("node scheduling state was not restored")
    ingress_distribution = _distribution(topology["ingress"])
    worker_distribution = _distribution(topology["workers"])
    spool_distribution = _distribution(topology["spool_workers"])
    if sorted(ingress_distribution.values()) != [1, 1, 1]:
        errors.append("ingress topology did not return to one Pod per node")
    if sorted(worker_distribution.values()) != [2, 2, 2]:
        errors.append("worker topology did not return to two Pods per node")
    return {
        "case_id": CASE_ID,
        "attempt": context["attempt"],
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "plan": context["plan"],
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
        "final_probe": final_probe,
        "node_final": node_final,
        "capacity_recommendation": (
            "A fourth CPU node is required to keep all three ingress "
            "replicas schedulable during one-node maintenance under "
            "DoNotSchedule/maxSkew=1."
        ),
    }


def _cleanup_ha002(case_dir: Path, context: dict, state: dict, result: dict) -> None:
    if state["cordoned"]:
        COMMON.run(
            [
                "kubectl",
                "--kubeconfig",
                str(COMMON.CPU_KUBECONFIG),
                "uncordon",
                context["node_name"],
            ],
            check=False,
        )
    stop_watchdog(state["watchdog"], state["watchdog_handle"])
    if state["probe_created"]:
        COMMON.gpu(
            "exec",
            COMMON.PROBE_POD,
            "--",
            "touch",
            "/state/stop",
            check=False,
        )
    COMMON.gpu("delete", "pod", COMMON.PROBE_POD, "--ignore-not-found", check=False)
    COMMON.gpu(
        "delete",
        "configmap",
        COMMON.CONFIGMAP,
        "--ignore-not-found",
        check=False,
    )
    try:
        for deployment in ("gpu-fault-api-ha", "gpu-fault-control-worker"):
            COMMON.cpu(
                "rollout",
                "status",
                f"deployment/{deployment}",
                "--timeout=600s",
                timeout=700,
            )
        if context["spool_replicas"]:
            COMMON.cpu(
                "rollout",
                "status",
                "deployment/gpu-fault-telemetry-spool-worker",
                "--timeout=600s",
                timeout=700,
            )
        availability = COMMON.control_sample(include_queue=True)
        postflight = {
            "node": current_node(context["node_name"]),
            "probe_resources": COMMON.probe_resources(),
            "queue": availability["queue"],
            "spool_ready": availability["spool_ready"],
        }
        result["postflight"] = postflight
        write_json(case_dir / "postflight.json", postflight)
        if postflight["node"]["unschedulable"]:
            raise CaseError("node remains cordoned")
        if postflight["node"]["taints"] != context["baseline_node"]["taints"]:
            raise CaseError("node taints changed")
        if postflight["probe_resources"]["count"] != 0:
            raise CaseError("probe resources remain")
        if postflight["spool_ready"] != context["spool_replicas"]:
            raise CaseError("spool-worker replicas did not recover")
    except Exception as exc:
        result["postflight_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"


def execute(run_dir: Path, attempt: int, confirmation: str) -> int:
    if confirmation != CONFIRMATION:
        raise CaseError(f"confirmation must be exactly {CONFIRMATION}")
    case_dir = run_dir / "cases" / CASE_ID
    plan_path = case_dir / "plan.json"
    if not plan_path.is_file():
        raise CaseError("HA-002 plan is missing")
    context = _execute_context(json.loads(plan_path.read_text()), plan_path)
    context["attempt"] = attempt
    result: dict = {"case_id": CASE_ID, "attempt": attempt, "verdict": "FAIL"}
    state = {
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
        _cleanup_ha002(case_dir, context, state, result)
    write_json(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
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
    write_json(plan_path, plan)
    return execute(args.run_dir, args.attempt, args.confirm)


if __name__ == "__main__":
    raise SystemExit(main())
