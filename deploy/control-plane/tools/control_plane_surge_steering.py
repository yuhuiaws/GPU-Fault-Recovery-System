"""Steer which old Pod a surge-first CPU roll deletes first, and say why a
rollout stalled.

``gpu-fault-api-ha`` rolls with ``maxUnavailable`` 0 / ``maxSurge`` 1 under a
hostname topology spread counted per ``pod-template-hash``. The ReplicaSet
controller picks the old Pod to delete after the surge Pod is Ready by its own
ranking -- "more related Pods on the node first" -- and the surge Pod counts,
so the old Pod sharing the surge Pod's node goes first. When exactly one node
had the free 4-core slot, that freed the cores on the one node the spread
forbids for the next new Pod while the other nodes kept their 4-core old Pods:
the roll deadlocked until the rollout timeout and the release rolled back. The
``controller.kubernetes.io/pod-deletion-cost`` annotation is ranked ahead of
the node rule, so writing the order the schedulability model chose onto the
old Pods makes the roll deterministic.

Three verbs, all reading the model in ``control_plane_schedulability``:

``plan``
    Before the new template is applied: from the preflight's node and Pod
    snapshots and the final Deployment document, decide whether the roll is
    surge-first at all, compute the surge node and the deletion order, write
    the state file the later verbs read and annotate the old Pods.
``reconcile``
    While the rollout runs (every poll of the rollout wait): read the
    Deployment's Pods and ReplicaSets, find where the new Pods actually
    landed, recompute the order from there and re-annotate what changed -- the
    scheduler is not bound to the model's prediction.
``diagnose``
    When the rollout wait expires: print the pending new Pod's scheduling
    message, every node's free CPU/memory with the old and new Pods it hosts,
    and what to free. Keeps the wait fail-closed; improves the evidence.

The kubectl command (``kubectl [--kubeconfig ...] -n <namespace>``) follows
``--``; the verbs run it themselves. Nothing here prints Secrets or
kubeconfig contents: only Pod, ReplicaSet and Node metadata and requests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from control_plane_schedulability import (
    DELETION_COST_ANNOTATION,
    GIBI,
    MILLI,
    deployment_pods,
    eligible_nodes,
    pod_requests,
    simulate_surge_roll,
    step_demand,
)

Kubectl = Callable[[list[str]], str]
REVISION_ANNOTATION = "deployment.kubernetes.io/revision"


class SteeringError(RuntimeError):
    """A kubectl read or write the steering needed failed."""


def kubectl_runner(command: list[str]) -> Kubectl:
    def run(arguments: list[str]) -> str:
        result = subprocess.run(
            [*command, *arguments], text=True, capture_output=True, check=False
        )
        if result.returncode:
            detail = " ".join(result.stderr.split())[:300]
            raise SteeringError(f"kubectl {' '.join(arguments[:3])} failed: {detail}")
        return result.stdout

    return run


def _items(document: str) -> list[dict[str, Any]]:
    loaded = json.loads(document) if document.strip() else {}
    items = loaded.get("items") if isinstance(loaded, dict) else None
    return list(items) if isinstance(items, list) else []


def newest_template(
    replicasets: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    """The ``pod-template-hash`` and template of the ReplicaSet with the
    highest Deployment revision -- the new template once the apply happened."""

    best: tuple[int, str, dict[str, Any] | None] = (-1, "", None)
    for replicaset in replicasets:
        metadata = replicaset.get("metadata") or {}
        try:
            revision = int(
                (metadata.get("annotations") or {}).get(REVISION_ANNOTATION, -1)
            )
        except ValueError:
            revision = -1
        template_hash = (metadata.get("labels") or {}).get("pod-template-hash", "")
        if revision > best[0]:
            best = (
                revision,
                template_hash,
                (replicaset.get("spec") or {}).get("template") or {},
            )
    return best[1], best[2]


def _cost(pod: dict[str, Any]) -> int | None:
    value = ((pod.get("metadata") or {}).get("annotations") or {}).get(
        DELETION_COST_ANNOTATION
    )
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _active(pod: dict[str, Any]) -> bool:
    return not (pod.get("metadata") or {}).get("deletionTimestamp") and (
        pod.get("status") or {}
    ).get("phase") not in ("Succeeded", "Failed")


def _format_order(order: list[dict[str, Any]], surge_nodes: list[str]) -> str:
    parts = []
    for entry in order:
        note = ""
        if entry["node"] in surge_nodes and "frees_node_for" not in entry:
            note = ", co-located with a new Pod, last"
        parts.append(f"{entry['pod']} on {entry['node']} (cost {entry['cost']}{note})")
    return " -> ".join(parts)


def plan(
    *,
    nodes: dict[str, Any],
    pods: dict[str, Any],
    live_deployments: dict[str, Any],
    desired: dict[str, Any],
    run: Kubectl | None,
) -> dict[str, Any]:
    """Decide whether the roll needs steering, model it and annotate the old
    Pods (when ``run`` is given). Returns the state the later verbs read."""

    name = str(desired["metadata"]["name"])
    live = next(
        (
            item
            for item in live_deployments.get("items", [])
            if item["metadata"]["name"] == name
        ),
        None,
    )
    demand = step_demand(desired, live)
    state: dict[str, Any] = {"deployment": name, "active": False}
    if not demand["template_rolls"]:
        state["reason"] = "the template does not roll"
        return state
    if demand["max_unavailable"] > 0 or demand["max_surge"] < 1:
        state["reason"] = (
            f"not a surge-first roll (maxUnavailable {demand['max_unavailable']}, "
            f"maxSurge {demand['max_surge']})"
        )
        return state
    template_spec = ((desired.get("spec") or {}).get("template") or {}).get(
        "spec"
    ) or {}
    node_states = eligible_nodes(nodes, pods, template_spec)
    old_pods = deployment_pods(pods, name, {node["name"] for node in node_states})
    if not old_pods:
        state["reason"] = "no old Pod holds room on an eligible node"
        return state
    request = demand["pod_request"]
    roll = simulate_surge_roll(
        node_states=node_states,
        old_pods=old_pods,
        cpu=request["cpu_millicores"],
        memory=request["memory_bytes"],
        replicas=demand["desired_replicas"],
        max_surge=demand["max_surge"],
        max_skew=demand["max_skew"],
    )
    state.update(
        {
            "active": True,
            "request": request,
            "replicas": demand["desired_replicas"],
            "max_surge": demand["max_surge"],
            "max_skew": demand["max_skew"],
            "nodes": [
                {
                    "name": node["name"],
                    "free_cpu_millicores": node["free_cpu_millicores"],
                    "free_memory_bytes": node["free_memory_bytes"],
                }
                for node in node_states
            ],
            "old_pods": old_pods,
            "planned_surge_nodes": roll["surge_nodes"],
            "surge_nodes": roll["surge_nodes"],
            "deletion_order": roll["deletion_order"],
            "progress_guaranteed": roll["progress_guaranteed"],
            "stall": roll["stall"],
            "applied_costs": {},
        }
    )
    if run is not None:
        apply_costs(state, run)
    return state


def apply_costs(state: dict[str, Any], run: Kubectl) -> list[str]:
    """Annotate the old Pods whose cost differs from the last one written; a
    Pod that vanished meanwhile is skipped (the controller took it)."""

    changed: list[str] = []
    applied: dict[str, int] = state.setdefault("applied_costs", {})
    for entry in state["deletion_order"]:
        pod, cost = entry["pod"], entry["cost"]
        if applied.get(pod) == cost:
            continue
        try:
            run(
                [
                    "annotate",
                    "pod",
                    pod,
                    f"{DELETION_COST_ANNOTATION}={cost}",
                    "--overwrite",
                ]
            )
        except SteeringError as exc:
            print(f"surge steering: {state['deployment']}: {exc}", file=sys.stderr)
            continue
        applied[pod] = cost
        changed.append(pod)
    return changed


def reconcile(state: dict[str, Any], run: Kubectl) -> dict[str, Any]:
    """Re-derive the deletion order from where the new Pods actually are."""

    name = state["deployment"]
    pods = _items(run(["get", "pods", "-l", f"app={name}", "-o", "json"]))
    replicasets = _items(run(["get", "replicasets", "-l", f"app={name}", "-o", "json"]))
    new_hash, _ = newest_template(replicasets)
    new_nodes = sorted(
        str((pod.get("spec") or {}).get("nodeName"))
        for pod in pods
        if _active(pod)
        and (pod.get("spec") or {}).get("nodeName")
        and ((pod.get("metadata") or {}).get("labels") or {}).get("pod-template-hash")
        == new_hash
    )
    if not new_nodes:
        return {"changed": [], "note": "no new Pod is scheduled yet"}
    node_names = {node["name"] for node in state["nodes"]}
    old_pods = [
        pod
        for pod in deployment_pods({"items": pods}, name, node_names)
        if pod["template_hash"] != new_hash
    ]
    # Old Pods gone since the plan gave their room back; the model debits the
    # new Pods itself from the planned surge nodes.
    live_names = {pod["name"] for pod in old_pods}
    node_states = [dict(node) for node in state["nodes"]]
    by_name = {node["name"]: node for node in node_states}
    for pod in state["old_pods"]:
        if pod["name"] not in live_names and pod["node"] in by_name:
            by_name[pod["node"]]["free_cpu_millicores"] += pod["cpu_millicores"]
            by_name[pod["node"]]["free_memory_bytes"] += pod["memory_bytes"]
    roll = simulate_surge_roll(
        node_states=node_states,
        old_pods=old_pods,
        cpu=state["request"]["cpu_millicores"],
        memory=state["request"]["memory_bytes"],
        replicas=state["replicas"],
        max_surge=state["max_surge"],
        max_skew=state["max_skew"],
        surge_nodes=new_nodes,
    )
    unexpected = [
        node for node in new_nodes if node not in state["planned_surge_nodes"]
    ]
    state["surge_nodes"] = new_nodes
    state["deletion_order"] = roll["deletion_order"]
    state["progress_guaranteed"] = roll["progress_guaranteed"]
    state["stall"] = roll["stall"]
    changed = apply_costs(state, run)
    return {"changed": changed, "unexpected": unexpected, "roll": roll}


def diagnose(name: str, run: Kubectl) -> list[str]:
    """Why the rollout of ``name`` has not finished: the pending new Pods'
    scheduler verdicts and each node's room and Pods of this Deployment."""

    pods = _items(run(["get", "pods", "-l", f"app={name}", "-o", "json"]))
    replicasets = _items(run(["get", "replicasets", "-l", f"app={name}", "-o", "json"]))
    nodes = json.loads(run(["get", "nodes", "-o", "json"]))
    all_pods = json.loads(run(["get", "pods", "-A", "-o", "json"]))
    new_hash, template = newest_template(replicasets)
    template_spec = (template or {}).get("spec") or {}
    cpu, memory = pod_requests(template_spec)
    max_skew = next(
        (
            int(constraint.get("maxSkew", 1))
            for constraint in template_spec.get("topologySpreadConstraints") or []
            if constraint.get("topologyKey") == "kubernetes.io/hostname"
            and constraint.get("whenUnsatisfiable", "DoNotSchedule") == "DoNotSchedule"
        ),
        None,
    )
    lines: list[str] = []
    for pod in pods:
        metadata = pod.get("metadata") or {}
        labels = metadata.get("labels") or {}
        status = pod.get("status") or {}
        is_new = labels.get("pod-template-hash") == new_hash
        if is_new and not (pod.get("spec") or {}).get("nodeName"):
            scheduled: dict[str, Any] = next(
                (
                    condition
                    for condition in status.get("conditions") or []
                    if condition.get("type") == "PodScheduled"
                ),
                {},
            )
            lines.append(
                f"new Pod {metadata.get('name')} is {status.get('phase', 'Pending')} "
                f"({scheduled.get('reason', 'unscheduled')}): "
                f"{scheduled.get('message', 'no scheduler verdict recorded yet')}"
            )
        elif metadata.get("deletionTimestamp"):
            lines.append(
                f"{'new' if is_new else 'old'} Pod {metadata.get('name')} is "
                f"terminating on {(pod.get('spec') or {}).get('nodeName')} and holds "
                "its requests until it is gone"
            )
    for node in eligible_nodes(nodes, all_pods, template_spec):
        on_node = [
            pod
            for pod in pods
            if (pod.get("spec") or {}).get("nodeName") == node["name"] and _active(pod)
        ]
        new = [
            pod["metadata"]["name"]
            for pod in on_node
            if pod["metadata"].get("labels", {}).get("pod-template-hash") == new_hash
        ]
        old = [
            f"{pod['metadata']['name']} (deletion cost {_cost(pod)})"
            for pod in on_node
            if pod["metadata"].get("labels", {}).get("pod-template-hash") != new_hash
        ]
        lines.append(
            f"node {node['name']}: {node['free_cpu_millicores'] / MILLI:.1f} cores / "
            f"{node['free_memory_bytes'] / GIBI:.1f}Gi free; {len(new)} new Pod(s) "
            f"[{', '.join(new)}]; {len(old)} old Pod(s) [{', '.join(old)}]"
        )
    spread = (
        f" on a node that hosts no new Pod of this template (hostname spread "
        f"maxSkew {max_skew})"
        if max_skew is not None
        else ""
    )
    lines.append(
        f"a new Pod needs {cpu / MILLI:g} cores / {memory / GIBI:g}Gi{spread}; free "
        "such a slot on a second node or add a CPU node"
    )
    return lines


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _command(arguments: list[str]) -> list[str]:
    return arguments[1:] if arguments[:1] == ["--"] else arguments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    verbs = parser.add_subparsers(dest="verb", required=True)
    planner = verbs.add_parser("plan")
    planner.add_argument("--nodes", required=True, type=Path)
    planner.add_argument("--pods", required=True, type=Path)
    planner.add_argument("--live-deployments", required=True, type=Path)
    planner.add_argument("--deployment-document", required=True, type=Path)
    planner.add_argument("--state", required=True, type=Path)
    planner.add_argument("command", nargs=argparse.REMAINDER)
    reconciler = verbs.add_parser("reconcile")
    reconciler.add_argument("--state", required=True, type=Path)
    reconciler.add_argument("command", nargs=argparse.REMAINDER)
    diagnoser = verbs.add_parser("diagnose")
    diagnoser.add_argument("--deployment", required=True)
    diagnoser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    command = _command(arguments.command)
    try:
        if arguments.verb == "plan":
            state = plan(
                nodes=_read_json(arguments.nodes),
                pods=_read_json(arguments.pods),
                live_deployments=_read_json(arguments.live_deployments),
                desired=_read_json(arguments.deployment_document),
                run=kubectl_runner(command) if command else None,
            )
            _write_state(arguments.state, state)
            name = state["deployment"]
            if not state["active"]:
                print(
                    f"surge steering: {name}: nothing to steer ({state['reason']})",
                    file=sys.stderr,
                )
                return 0
            print(
                f"surge steering: {name}: surge Pod expected on "
                f"{', '.join(state['surge_nodes'])}; old Pods deleted in the order "
                f"{_format_order(state['deletion_order'], state['surge_nodes'])}",
                file=sys.stderr,
            )
            if not state["progress_guaranteed"]:
                print(
                    f"surge steering: {name}: even this order is not guaranteed to "
                    f"progress: {state['stall']}",
                    file=sys.stderr,
                )
            return 0
        if arguments.verb == "reconcile":
            if not command:
                parser.error("reconcile needs the kubectl command after --")
            state = _read_json(arguments.state)
            if not state.get("active"):
                return 0
            outcome = reconcile(state, kubectl_runner(command))
            _write_state(arguments.state, state)
            name = state["deployment"]
            if outcome["changed"]:
                where = ", ".join(state["surge_nodes"])
                planned = ", ".join(state["planned_surge_nodes"])
                landed = (
                    f"new Pod(s) on {where} instead of the planned {planned}"
                    if outcome["unexpected"]
                    else f"new Pod(s) on {where}"
                )
                print(
                    f"surge steering: {name}: {landed}; deletion order now "
                    f"{_format_order(state['deletion_order'], state['surge_nodes'])}",
                    file=sys.stderr,
                )
            if outcome.get("roll") and not outcome["roll"]["progress_guaranteed"]:
                print(
                    f"surge steering: {name}: the roll cannot progress from here: "
                    f"{outcome['roll']['stall']}",
                    file=sys.stderr,
                )
            return 0
        if not command:
            parser.error("diagnose needs the kubectl command after --")
        for line in diagnose(arguments.deployment, kubectl_runner(command)):
            print(f"rollout stalled: {arguments.deployment}: {line}", file=sys.stderr)
        return 0
    except (SteeringError, OSError, ValueError, KeyError) as exc:
        print(f"surge steering: {arguments.verb} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
