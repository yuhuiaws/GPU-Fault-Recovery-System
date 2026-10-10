"""Schedulability preflight for the CPU control-plane roll.

The ingress Deployment rolls surge-first (``maxUnavailable`` 0, ``maxSurge``
1), so its roll needs one free slot of its own size on some node; the
consumer tier's replica additions take headroom for good. Applying the spool
worker and the control worker first and ingress last turned a perf preset into
a seven-minute ``rollout status`` timeout and a rollback on a 3 x 16-core node
group that had ~4.1-4.8 free cores per node: the spool replicas took the only
4-core slot before the ingress surge Pod asked for it (``FailedScheduling ...
0/3 nodes are available: 3 Insufficient cpu``). This preflight models the roll
one Deployment at a time against the live nodes' allocatable CPU/memory minus
the requests of every non-terminal Pod, honouring the Deployment's hostname
topology spread, and refuses fail-closed naming the missing cores and the node
sizes instead of waiting for the timeout.

The order it hands back is the apply order: ingress first (the surge slot),
then the control worker, then the spool worker. Only when that order cannot
fit does it try the roles that free room (replica reductions, unchanged
templates) ahead of the rest -- a rollback shrinking the consumer tier must
not wait on an ingress surge slot the forward roll's additions took. The
rollback context runs the preflight in ``warn`` mode: a roll that still cannot
fit is logged with its shortfall and left to the rollout waits; a rollback is
never refused by a model.

A free surge slot is necessary, not sufficient. The surge-first roll replaces
the old Pods one at a time: the controller deletes an old Pod once the new one
is Ready and creates the next new Pod, which must land on a node that both has
the room and hosts no new Pod yet (the hostname spread with ``maxSkew`` 1 is
counted per ``pod-template-hash``). Which old Pod goes is deterministic. The
ReplicaSet controller ranks them: unassigned, Pending, not Ready, lower
``pod-deletion-cost``, MORE RELATED PODS ON THE SAME NODE (related = Pods of
any ReplicaSet of the Deployment, so the surge Pod counts), ready for less
time, more restarts, newer -- after the surge Pod lands, the old Pod sharing
its node is always deleted first. On a 3-node group where exactly one node had
the 4-core slot (every enabled -> disabled preset change rolls api-ha while the
spool workers still occupy CPU; the enable that succeeded rolled it with the
spool at 0 and two free slots) that freed the cores on the one node the spread
forbids, while the other two nodes kept their 4-core old Pods: the next new
Pod stayed Pending (``1 node(s) didn't match pod topology spread constraints,
2 Insufficient cpu``) until the rollout timeout and the release rolled back,
three times out of three. ``simulate_surge_roll`` replays that sequence: the
steered order (an old Pod whose node the next new Pod may use first, the Pods
co-located with new Pods last) and the controller's own order, every other
kill order searched as well. The plan records both, names the nodes with a
free slot, and fits only when the steered order is guaranteed to progress --
exactly one free slot is the steered case, not a refusal. The apply writes
the order onto the old Pods as ``controller.kubernetes.io/pod-deletion-cost``
(ranked ahead of the node rule) before the new template is applied and
re-steers from where the new Pods actually land while the roll runs.

Pure functions over three JSON documents (``kubectl get nodes -o json``,
``kubectl get pods -A -o json`` and the live Deployments) plus the final
rendered Deployment of every selected role; the apply script fetches the
documents and passes file paths, so the arithmetic is testable without a
cluster.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

#: Preferred apply order: ingress first (its surge-first roll needs the free
#: slot), then the consumer tier one Deployment at a time.
CANONICAL_ORDER = ("ingress", "worker", "spool")
ROLE_DEPLOYMENTS = {
    "ingress": "gpu-fault-api-ha",
    "worker": "gpu-fault-control-worker",
    "spool": "gpu-fault-telemetry-spool-worker",
}
MODES = ("refuse", "warn")
HOSTNAME_KEY = "kubernetes.io/hostname"
MILLI = 1000
GIBI = 1024**3
_MEMORY_SUFFIXES = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "Pi": 1024**5,
    "Ei": 1024**6,
    "k": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
    "P": 1000**5,
    "E": 1000**6,
}
# The apply stamps every digest that changes what runs on the Pod template
# under these names; a resume whose rendered template already carries the live
# values rolls nothing and needs no surge slot.
_ROLL_ANNOTATION_PREFIXES = ("gpu-fault.io/", "kubectl.kubernetes.io/restartedAt")
#: The ReplicaSet controller deletes the Pods with the lowest value of this
#: annotation first (rule 4 of its ranking, ahead of "more related Pods on the
#: node first" and readiness age); beta and on by default since Kubernetes 1.22.
DELETION_COST_ANNOTATION = "controller.kubernetes.io/pod-deletion-cost"
DELETION_COST_STEP = 100
#: Beyond this many old Pods the kill orders are not searched exhaustively;
#: only the controller's own preference is replayed. The surge-first tier has
#: 3 replicas (6 orders), the bound keeps a misconfigured tier from taking
#: minutes in a preflight.
EXHAUSTIVE_KILL_ORDER_PODS = 7


class SchedulabilityRefused(RuntimeError):
    """The roll cannot be scheduled; the message names what is missing."""


def parse_cpu_millicores(value: object) -> int:
    text = str(value).strip()
    if not text:
        return 0
    if text.endswith("m"):
        return int(round(float(text[:-1])))
    return int(round(float(text) * MILLI))


def parse_memory_bytes(value: object) -> int:
    text = str(value).strip()
    if not text:
        return 0
    for suffix, factor in sorted(
        _MEMORY_SUFFIXES.items(), key=lambda item: -len(item[0])
    ):
        if text.endswith(suffix):
            return int(round(float(text[: -len(suffix)]) * factor))
    if text.endswith("m"):  # milli-bytes exist in the quantity grammar
        return int(round(float(text[:-1]) / MILLI))
    return int(round(float(text)))


def pod_requests(spec: dict[str, Any]) -> tuple[int, int]:
    """Effective requests of one Pod spec: the sum over containers, or a
    larger init container, plus the Pod overhead (the scheduler's rule)."""

    def resource(container: dict[str, Any], name: str) -> str:
        return str(
            ((container.get("resources") or {}).get("requests") or {}).get(name, "0")
        )

    cpu = sum(
        parse_cpu_millicores(resource(c, "cpu")) for c in spec.get("containers", [])
    )
    memory = sum(
        parse_memory_bytes(resource(c, "memory")) for c in spec.get("containers", [])
    )
    for init in spec.get("initContainers", []):
        cpu = max(cpu, parse_cpu_millicores(resource(init, "cpu")))
        memory = max(memory, parse_memory_bytes(resource(init, "memory")))
    overhead = spec.get("overhead") or {}
    cpu += parse_cpu_millicores(overhead.get("cpu", "0"))
    memory += parse_memory_bytes(overhead.get("memory", "0"))
    return cpu, memory


def _tolerates(spec: dict[str, Any], taint: dict[str, Any]) -> bool:
    for toleration in spec.get("tolerations") or []:
        if toleration.get("effect") not in (None, "", taint.get("effect")):
            continue
        operator = toleration.get("operator", "Equal")
        if operator == "Exists":
            if toleration.get("key") in (None, "", taint.get("key")):
                return True
        elif toleration.get("key") == taint.get("key") and toleration.get(
            "value", ""
        ) == taint.get("value", ""):
            return True
    return False


def eligible_nodes(
    nodes: dict[str, Any], pods: dict[str, Any], template_spec: dict[str, Any]
) -> list[dict[str, Any]]:
    """Ready, schedulable nodes the template may land on, with their free
    CPU/memory after every non-terminal Pod's requests."""

    requested: dict[str, list[int]] = {}
    for pod in pods.get("items", []):
        node_name = (pod.get("spec") or {}).get("nodeName")
        phase = (pod.get("status") or {}).get("phase")
        if not node_name or phase in ("Succeeded", "Failed"):
            continue
        cpu, memory = pod_requests(pod.get("spec") or {})
        totals = requested.setdefault(node_name, [0, 0])
        totals[0] += cpu
        totals[1] += memory
    result: list[dict[str, Any]] = []
    selector = template_spec.get("nodeSelector") or {}
    for node in nodes.get("items", []):
        metadata = node.get("metadata") or {}
        spec = node.get("spec") or {}
        status = node.get("status") or {}
        labels = metadata.get("labels") or {}
        if spec.get("unschedulable"):
            continue
        if any(labels.get(key) != value for key, value in selector.items()):
            continue
        conditions = {
            c.get("type"): c.get("status") for c in status.get("conditions", [])
        }
        if conditions.get("Ready") != "True":
            continue
        if any(
            taint.get("effect") in ("NoSchedule", "NoExecute")
            and not _tolerates(template_spec, taint)
            for taint in spec.get("taints") or []
        ):
            continue
        allocatable = status.get("allocatable") or {}
        cpu_allocatable = parse_cpu_millicores(allocatable.get("cpu", "0"))
        memory_allocatable = parse_memory_bytes(allocatable.get("memory", "0"))
        cpu_requested, memory_requested = requested.get(
            metadata.get("name", ""), [0, 0]
        )
        result.append(
            {
                "name": metadata.get("name", ""),
                "hostname": labels.get(HOSTNAME_KEY, metadata.get("name", "")),
                "allocatable_cpu_millicores": cpu_allocatable,
                "allocatable_memory_bytes": memory_allocatable,
                "requested_cpu_millicores": cpu_requested,
                "requested_memory_bytes": memory_requested,
                "free_cpu_millicores": cpu_allocatable - cpu_requested,
                "free_memory_bytes": memory_allocatable - memory_requested,
            }
        )
    return result


def _surge_count(strategy: dict[str, Any], replicas: int) -> tuple[int, int]:
    rolling = (strategy or {}).get("rollingUpdate") or {}

    def count(value: object, default: int) -> int:
        if value is None:
            return default
        text = str(value)
        if text.endswith("%"):
            return math.ceil(replicas * float(text[:-1]) / 100)
        return int(text)

    return count(rolling.get("maxSurge"), 1), count(rolling.get("maxUnavailable"), 1)


def template_rolls(desired: dict[str, Any], live: dict[str, Any] | None) -> bool:
    """Whether applying ``desired`` changes the live Pod template.

    Best effort on the fields the apply stamps and the renderer sets: a
    missing live Deployment, a different image or request, or a different
    ``gpu-fault.io/*`` / ``restartedAt`` template annotation means a roll.
    """

    if live is None:
        return True
    desired_template = (desired.get("spec") or {}).get("template") or {}
    live_template = (live.get("spec") or {}).get("template") or {}
    desired_annotations = (desired_template.get("metadata") or {}).get(
        "annotations"
    ) or {}
    live_annotations = (live_template.get("metadata") or {}).get("annotations") or {}
    for key, value in desired_annotations.items():
        if (
            key.startswith(_ROLL_ANNOTATION_PREFIXES)
            and live_annotations.get(key) != value
        ):
            return True
    desired_containers = {
        c.get("name"): c
        for c in (desired_template.get("spec") or {}).get("containers", [])
    }
    live_containers = {
        c.get("name"): c
        for c in (live_template.get("spec") or {}).get("containers", [])
    }
    if set(desired_containers) != set(live_containers):
        return True
    for name, container in desired_containers.items():
        other = live_containers[name]
        if container.get("image") != other.get("image"):
            return True
        if (container.get("resources") or {}).get("requests") != (
            other.get("resources") or {}
        ).get("requests"):
            return True
    return False


def _spread_constraint(template_spec: dict[str, Any]) -> int | None:
    for constraint in template_spec.get("topologySpreadConstraints") or []:
        if (
            constraint.get("topologyKey") == HOSTNAME_KEY
            and constraint.get("whenUnsatisfiable", "DoNotSchedule") == "DoNotSchedule"
        ):
            return int(constraint.get("maxSkew", 1))
    return None


def _shortfall(
    pods_to_place: int, placed: int, cpu: int, memory: int, nodes: list[dict[str, Any]]
) -> str:
    best_cpu = max((node["free_cpu_millicores"] for node in nodes), default=0)
    best_memory = max((node["free_memory_bytes"] for node in nodes), default=0)
    short_cpu = max(0, cpu - best_cpu)
    short_memory = max(0, memory - best_memory)
    need = f"Pod {placed + 1} of {pods_to_place} needs {cpu / MILLI:g} cores / {memory / GIBI:g}Gi; "
    if not (short_cpu or short_memory):
        return need + "the nodes with room would break the hostname spread"
    parts = []
    if short_cpu:
        parts.append(f"{short_cpu / MILLI:g} more free cores")
    if short_memory:
        parts.append(f"{short_memory / GIBI:g}Gi more free memory")
    return need + f"no node has room ({' and '.join(parts)} on one node would fit it)"


def _place(
    pods_to_place: int,
    cpu: int,
    memory: int,
    nodes: list[dict[str, Any]],
    *,
    max_skew: int | None,
    permanent: bool,
) -> tuple[list[str], str | None]:
    """Place ``pods_to_place`` Pods of one new template on ``nodes`` (mutated
    when ``permanent``), one at a time onto the node with the fewest Pods of
    this template that still has room, so a hostname spread of ``max_skew``
    holds. Returns the chosen nodes and, on failure, why."""

    counts = {node["name"]: 0 for node in nodes}
    placed: list[str] = []
    for _ in range(pods_to_place):
        candidates = sorted(
            (
                node
                for node in nodes
                if node["free_cpu_millicores"] >= cpu
                and node["free_memory_bytes"] >= memory
            ),
            key=lambda node: (counts[node["name"]], -node["free_cpu_millicores"]),
        )
        if max_skew is not None and candidates:
            lowest = min(counts.values())
            candidates = [
                node for node in candidates if counts[node["name"]] - lowest < max_skew
            ]
        if not candidates:
            return placed, _shortfall(pods_to_place, len(placed), cpu, memory, nodes)
        node = candidates[0]
        counts[node["name"]] += 1
        placed.append(node["name"])
        if permanent:
            node["free_cpu_millicores"] -= cpu
            node["free_memory_bytes"] -= memory
    return placed, None


def _node_sizes(nodes: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{node['name']} {node['free_cpu_millicores'] / MILLI:.1f} cores / "
        f"{node['free_memory_bytes'] / GIBI:.1f}Gi free of "
        f"{node['allocatable_cpu_millicores'] / MILLI:.2f} cores allocatable"
        for node in nodes
    )


def step_demand(desired: dict[str, Any], live: dict[str, Any] | None) -> dict[str, Any]:
    """What one role's roll asks of the nodes: a transient surge slot when the
    template rolls surge-first, and the replicas the change adds for good."""

    template = (desired.get("spec") or {}).get("template") or {}
    spec = template.get("spec") or {}
    cpu, memory = pod_requests(spec)
    desired_replicas = int((desired.get("spec") or {}).get("replicas", 1))
    live_replicas = int((live.get("spec") or {}).get("replicas", 0)) if live else 0
    max_surge, max_unavailable = _surge_count(
        (desired.get("spec") or {}).get("strategy") or {}, desired_replicas
    )
    rolls = template_rolls(desired, live) and live_replicas > 0
    return {
        "pod_request": {"cpu_millicores": cpu, "memory_bytes": memory},
        "live_replicas": live_replicas,
        "desired_replicas": desired_replicas,
        "template_rolls": rolls,
        "max_surge": max_surge,
        "max_unavailable": max_unavailable,
        "surge_pods_needing_a_free_slot": max(0, max_surge - max_unavailable)
        if rolls
        else 0,
        "added_replicas": max(0, desired_replicas - live_replicas),
        "removed_replicas": max(0, live_replicas - desired_replicas),
        "max_skew": _spread_constraint(spec),
    }


def deployment_pods(
    pods: dict[str, Any], name: str, node_names: set[str]
) -> list[dict[str, Any]]:
    """The Deployment's Pods that hold room on one of ``node_names``: active,
    not already terminating, bound to a node -- with their own requests (what
    their deletion gives back) and the template hash that tells old from new."""

    result: list[dict[str, Any]] = []
    for pod in pods.get("items", []):
        metadata = pod.get("metadata") or {}
        labels = metadata.get("labels") or {}
        spec = pod.get("spec") or {}
        node = spec.get("nodeName")
        if (
            labels.get("app") != name
            or metadata.get("deletionTimestamp")
            or (pod.get("status") or {}).get("phase") in ("Succeeded", "Failed")
            or node not in node_names
        ):
            continue
        cpu, memory = pod_requests(spec)
        result.append(
            {
                "name": metadata.get("name", ""),
                "node": node,
                "cpu_millicores": cpu,
                "memory_bytes": memory,
                "template_hash": labels.get("pod-template-hash", ""),
            }
        )
    return sorted(result, key=lambda pod: (pod["node"], pod["name"]))


def _node_for_new_pod(
    free: dict[str, list[int]],
    new_counts: dict[str, int],
    cpu: int,
    memory: int,
    max_skew: int | None,
) -> str | None:
    """Where the scheduler can put the next new Pod: room for it and, under
    the hostname spread, no more new Pods than the emptiest node plus the
    skew; the emptiest-of-this-template, roomiest node when several qualify."""

    lowest = min(new_counts.values(), default=0)
    candidates = [
        node
        for node, (free_cpu, free_memory) in free.items()
        if free_cpu >= cpu
        and free_memory >= memory
        and (max_skew is None or new_counts[node] - lowest < max_skew)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda node: (new_counts[node], -free[node][0], node))


def _no_node_reason(
    free: dict[str, list[int]],
    new_counts: dict[str, int],
    cpu: int,
    memory: int,
    max_skew: int | None,
) -> str:
    lowest = min(new_counts.values(), default=0)
    parts = []
    for node in sorted(free):
        free_cpu, free_memory = free[node]
        if max_skew is not None and new_counts[node] - lowest >= max_skew:
            parts.append(
                f"{node} already hosts {new_counts[node]} new Pod(s) and the "
                f"hostname spread (maxSkew {max_skew}) forbids another"
            )
            continue
        short = []
        if free_cpu < cpu:
            missing = (cpu - free_cpu) / MILLI
            short.append(f"{missing:g} more {'core' if missing == 1 else 'cores'}")
        if free_memory < memory:
            short.append(f"{(memory - free_memory) / GIBI:g}Gi more memory")
        parts.append(
            f"{node} has {free_cpu / MILLI:.1f} cores / {free_memory / GIBI:.1f}Gi "
            f"free ({' and '.join(short)} would fit it)"
        )
    return "; ".join(parts)


def _released(free: dict[str, list[int]], pod: dict[str, Any]) -> dict[str, list[int]]:
    result = {node: list(values) for node, values in free.items()}
    result[pod["node"]][0] += pod["cpu_millicores"]
    result[pod["node"]][1] += pod["memory_bytes"]
    return result


def _occupy(
    free: dict[str, list[int]],
    new_counts: dict[str, int],
    node: str,
    cpu: int,
    memory: int,
) -> None:
    free.setdefault(node, [0, 0])
    free[node][0] -= cpu
    free[node][1] -= memory
    new_counts[node] = new_counts.get(node, 0) + 1


def _every_kill_order_progresses(
    free: dict[str, list[int]],
    new_counts: dict[str, int],
    remaining: list[dict[str, Any]],
    *,
    to_place: int,
    placed: int,
    replicas: int,
    cpu: int,
    memory: int,
    max_skew: int | None,
) -> tuple[list[str], str | None]:
    """Replay the kill orders of the unsteered controller: its own
    deterministic choice first (the Pod on the node with the most related
    Pods, the new Pods counted), then every other order. Returns the first
    stalling order -- the controller's own when that one stalls -- and why;
    ([], None) when every order completes."""

    if to_place == 0 or not remaining:
        return [], None
    related = {node: count for node, count in new_counts.items()}
    for pod in remaining:
        related[pod["node"]] = related.get(pod["node"], 0) + 1
    ranked = sorted(remaining, key=lambda pod: (-related[pod["node"]], pod["name"]))
    if len(remaining) > EXHAUSTIVE_KILL_ORDER_PODS:
        ranked = ranked[:1]
    for pod in ranked:
        trial_free = _released(free, pod)
        node = _node_for_new_pod(trial_free, new_counts, cpu, memory, max_skew)
        if node is None:
            return [pod["name"]], (
                f"after {pod['name']} on {pod['node']} is deleted, new Pod "
                f"{placed + 1} of {replicas} has no node: "
                + _no_node_reason(trial_free, new_counts, cpu, memory, max_skew)
            )
        trial_counts = dict(new_counts)
        _occupy(trial_free, trial_counts, node, cpu, memory)
        order, stall = _every_kill_order_progresses(
            trial_free,
            trial_counts,
            [other for other in remaining if other is not pod],
            to_place=to_place - 1,
            placed=placed + 1,
            replicas=replicas,
            cpu=cpu,
            memory=memory,
            max_skew=max_skew,
        )
        if stall is not None:
            return [pod["name"], *order], stall
    return [], None


def simulate_surge_roll(
    *,
    node_states: list[dict[str, Any]],
    old_pods: list[dict[str, Any]],
    cpu: int,
    memory: int,
    replicas: int,
    max_surge: int,
    max_skew: int | None,
    surge_nodes: list[str] | None = None,
) -> dict[str, Any]:
    """Replay a surge-first roll (``maxUnavailable`` 0) old Pod by old Pod.

    The surge Pods go where the scheduler would put them (or onto
    ``surge_nodes`` when the live placement is known); then, while new Pods
    are still owed, each deletion must let the next new Pod land somewhere:
    the steered order tries the old Pods on nodes without a new Pod first, the
    tightest post-release fit among them, and leaves the Pods co-located with
    new Pods for last. ``deletion_order`` carries the deletion costs to write
    on the old Pods (lowest first); ``nodes_with_a_free_slot`` the nodes that
    could take the surge Pod (exactly one is the case that needs steering);
    ``without_steering`` whether the controller's own order stalls and which
    order that is.
    """

    free = {
        node["name"]: [node["free_cpu_millicores"], node["free_memory_bytes"]]
        for node in node_states
    }
    new_counts = {name: 0 for name in free}
    placements: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "nodes_with_a_free_slot": sorted(
            node
            for node, (free_cpu, free_memory) in free.items()
            if free_cpu >= cpu and free_memory >= memory
        ),
        "surge_nodes": [],
        "deletion_order": [],
        "placements": placements,
        "progress_guaranteed": False,
        "stall": None,
        "free_after": {},
        "without_steering": {"progress_guaranteed": True, "stalling_order": []},
    }
    if surge_nodes is None:
        surge_nodes = []
        for index in range(min(max(max_surge, 0), replicas)):
            node = _node_for_new_pod(free, new_counts, cpu, memory, max_skew)
            if node is None:
                result["stall"] = (
                    f"surge Pod {index + 1} has no node: "
                    + _no_node_reason(free, new_counts, cpu, memory, max_skew)
                )
                return result
            surge_nodes.append(node)
            _occupy(free, new_counts, node, cpu, memory)
            placements.append({"new_pod": index + 1, "node": node, "after": None})
    else:
        for index, node in enumerate(surge_nodes, start=1):
            _occupy(free, new_counts, node, cpu, memory)
            placements.append({"new_pod": index, "node": node, "after": None})
    result["surge_nodes"] = list(surge_nodes)
    to_place = max(0, replicas - len(surge_nodes))
    stalling_order, stall = _every_kill_order_progresses(
        free,
        new_counts,
        list(old_pods),
        to_place=to_place,
        placed=len(surge_nodes),
        replicas=replicas,
        cpu=cpu,
        memory=memory,
        max_skew=max_skew,
    )
    result["without_steering"] = {
        "progress_guaranteed": stall is None,
        "stalling_order": stalling_order,
        **({"stall": stall} if stall is not None else {}),
    }
    remaining = list(old_pods)
    order: list[dict[str, Any]] = []
    while remaining:
        if to_place == 0:
            # The last deletions owe no new Pod; the Pods sharing a node with
            # a new Pod still go last, so a kill the model did not foresee
            # never lands next to one.
            remaining.sort(
                key=lambda pod: (new_counts[pod["node"]], pod["node"], pod["name"])
            )
            for pod in remaining:
                free = _released(free, pod)
                order.append({"pod": pod["name"], "node": pod["node"]})
            break
        choice = None
        first_failure = None
        for pod in sorted(
            remaining,
            key=lambda pod: (
                new_counts[pod["node"]],
                free[pod["node"]][0] + pod["cpu_millicores"],
                pod["name"],
            ),
        ):
            trial_free = _released(free, pod)
            node = _node_for_new_pod(trial_free, new_counts, cpu, memory, max_skew)
            if node is not None:
                choice = (pod, trial_free, node)
                break
            if first_failure is None:
                first_failure = (pod, trial_free)
        if choice is None:
            assert first_failure is not None
            pod, trial_free = first_failure
            result["stall"] = (
                f"new Pod {len(placements) + 1} of {replicas} ({cpu / MILLI:g} cores "
                f"/ {memory / GIBI:g}Gi) has no node whichever old Pod is deleted "
                f"first (deleting {pod['name']} on {pod['node']}): "
                + _no_node_reason(trial_free, new_counts, cpu, memory, max_skew)
            )
            result["deletion_order"] = order
            return result
        pod, free, node = choice
        remaining.remove(pod)
        _occupy(free, new_counts, node, cpu, memory)
        placements.append(
            {"new_pod": len(placements) + 1, "node": node, "after": pod["name"]}
        )
        order.append({"pod": pod["name"], "node": pod["node"], "frees_node_for": node})
        to_place -= 1
    for index, entry in enumerate(order, start=1):
        entry["cost"] = DELETION_COST_STEP * index
    result["deletion_order"] = order
    result["progress_guaranteed"] = True
    result["free_after"] = {node: list(values) for node, values in free.items()}
    return result


def plan_steps(
    *,
    nodes: dict[str, Any],
    pods: dict[str, Any],
    live_deployments: dict[str, Any],
    rendered: dict[str, dict[str, Any]],
    order: list[str],
) -> dict[str, Any]:
    """Model the roll role by role in ``order``; raise when a step cannot
    schedule. Returns the evidence (nodes before, per-step placements)."""

    live_by_name = {
        item["metadata"]["name"]: item for item in live_deployments.get("items", [])
    }
    first_template = (rendered[order[0]].get("spec") or {}).get("template") or {}
    node_states = eligible_nodes(nodes, pods, first_template.get("spec") or {})
    if not node_states:
        raise SchedulabilityRefused(
            "control-plane schedulability: no Ready, schedulable node is eligible "
            "for the control-plane Pods"
        )
    evidence: dict[str, Any] = {
        "order": list(order),
        "nodes_before": [dict(node) for node in node_states],
        "steps": [],
    }
    for index, role in enumerate(order, start=1):
        name = ROLE_DEPLOYMENTS[role]
        desired = rendered[role]
        demand = step_demand(desired, live_by_name.get(name))
        cpu = demand["pod_request"]["cpu_millicores"]
        memory = demand["pod_request"]["memory_bytes"]
        step: dict[str, Any] = {
            "step": index,
            "role": role,
            "deployment": name,
            **{key: value for key, value in demand.items() if key != "max_skew"},
            "free_cpu_millicores_before": {
                node["name"]: node["free_cpu_millicores"] for node in node_states
            },
        }
        sizes = _node_sizes(node_states)
        old_pods = deployment_pods(pods, name, {node["name"] for node in node_states})
        # A replica reduction frees its room before anything else is placed:
        # the controller scales the old ReplicaSet down first.
        if demand["removed_replicas"]:
            old_pods = _release_replicas(demand, node_states, old_pods)
        surge_first = (
            demand["template_rolls"]
            and demand["max_unavailable"] == 0
            and demand["max_surge"] > 0
        )
        added_placed: list[str] = []
        sequence_stall: str | None = None
        if surge_first:
            # Old Pod by old Pod: every deletion must leave the next new Pod a
            # node. The simulated end state is what the following steps see.
            steering = simulate_surge_roll(
                node_states=node_states,
                old_pods=old_pods,
                cpu=cpu,
                memory=memory,
                replicas=demand["desired_replicas"],
                max_surge=demand["max_surge"],
                max_skew=demand["max_skew"],
            )
            step["steering"] = {
                key: value for key, value in steering.items() if key != "free_after"
            }
            surge_placed = list(steering["surge_nodes"])
            reason = None
            surge = min(demand["max_surge"], demand["desired_replicas"])
            if steering["progress_guaranteed"]:
                for node in node_states:
                    node["free_cpu_millicores"], node["free_memory_bytes"] = steering[
                        "free_after"
                    ][node["name"]]
                added_placed = [
                    placement["node"]
                    for placement in steering["placements"][len(old_pods) :]
                ]
            elif len(surge_placed) < surge:
                reason = _shortfall(surge, len(surge_placed), cpu, memory, node_states)
            else:
                sequence_stall = steering["stall"]
        else:
            # The surge slot is transient (the old Pod goes once the new one is
            # Ready), the added replicas keep their room.
            surge_placed, reason = _place(
                demand["surge_pods_needing_a_free_slot"],
                cpu,
                memory,
                node_states,
                max_skew=demand["max_skew"],
                permanent=False,
            )
            if reason is None:
                added_placed, reason = _place(
                    demand["added_replicas"],
                    cpu,
                    memory,
                    node_states,
                    max_skew=demand["max_skew"],
                    permanent=True,
                )
        step["surge_placements"] = surge_placed
        step["added_placements"] = added_placed
        evidence["steps"].append(step)
        if sequence_stall is not None:
            step["refused"] = sequence_stall
            raise SchedulabilityRefused(
                f"control-plane schedulability: step {index} ({name} roll) cannot "
                f"complete its surge-first roll (maxUnavailable "
                f"{demand['max_unavailable']}, maxSurge {demand['max_surge']}) even "
                f"with the deletion order steered: {sequence_stall}; free one "
                f"{cpu / MILLI:g}-core / {memory / GIBI:g}Gi slot on a second node "
                f"(one that will host no new Pod) or add a CPU node before applying; "
                f"nodes before this step: {sizes}."
            )
        if reason is not None:
            step["refused"] = reason
            surge = demand["surge_pods_needing_a_free_slot"]
            what = (
                f"one surge Pod (maxUnavailable {demand['max_unavailable']}, "
                f"maxSurge {demand['max_surge']})"
                if surge and len(surge_placed) < surge
                else f"{demand['added_replicas']} added replica(s)"
            )
            raise SchedulabilityRefused(
                f"control-plane schedulability: step {index} ({name} roll) cannot "
                f"schedule {what}: {reason}; nodes before this step: {sizes}. "
                "Add CPU node capacity or lower the preset before applying."
            )
    evidence["nodes_after"] = [dict(node) for node in node_states]
    return evidence


def _release_replicas(
    demand: dict[str, Any],
    node_states: list[dict[str, Any]],
    old_pods: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Give the room of the replicas a reduction removes back to their nodes
    (the live Pods of the Deployment, most-loaded nodes first); returns the
    Pods a template roll still has to replace."""

    by_name = {node["name"]: node for node in node_states}
    removed = sorted(
        old_pods,
        key=lambda pod: (by_name[pod["node"]]["free_cpu_millicores"], pod["name"]),
    )[: demand["removed_replicas"]]
    for pod in removed:
        by_name[pod["node"]]["free_cpu_millicores"] += pod["cpu_millicores"]
        by_name[pod["node"]]["free_memory_bytes"] += pod["memory_bytes"]
    return [pod for pod in old_pods if pod not in removed]


def candidate_orders(
    rendered: dict[str, dict[str, Any]],
    live_deployments: dict[str, Any],
    *,
    fixed_order: list[str] | None = None,
) -> list[list[str]]:
    """The orders to try: the fixed one when the caller has reasons of its
    own (the spool-disable drain), else canonical and then room-freeing roles
    first."""

    if fixed_order is not None:
        return [list(fixed_order)]
    canonical = [role for role in CANONICAL_ORDER if role in rendered]
    live_by_name = {
        item["metadata"]["name"]: item for item in live_deployments.get("items", [])
    }
    frees_room = []
    for role in canonical:
        demand = step_demand(rendered[role], live_by_name.get(ROLE_DEPLOYMENTS[role]))
        if demand["removed_replicas"]:
            frees_room.append(role)
    shrinkers_first = frees_room + [
        role for role in canonical if role not in frees_room
    ]
    orders = [canonical]
    if shrinkers_first != canonical:
        orders.append(shrinkers_first)
    return orders


def plan_roll(
    *,
    nodes: dict[str, Any],
    pods: dict[str, Any],
    live_deployments: dict[str, Any],
    rendered: dict[str, dict[str, Any]],
    mode: str = "refuse",
    fixed_order: list[str] | None = None,
) -> dict[str, Any]:
    """Try the candidate orders; the first that fits is the apply order. In
    ``refuse`` mode none fitting raises; in ``warn`` mode the canonical order
    is returned with the verdict ``warning`` and every attempt's shortfall."""

    if mode not in MODES:
        raise ValueError(f"unknown schedulability mode {mode!r}")
    attempts: list[dict[str, Any]] = []
    for order in candidate_orders(rendered, live_deployments, fixed_order=fixed_order):
        try:
            evidence = plan_steps(
                nodes=nodes,
                pods=pods,
                live_deployments=live_deployments,
                rendered=rendered,
                order=order,
            )
        except SchedulabilityRefused as exc:
            attempts.append({"order": list(order), "refused": str(exc)})
            continue
        evidence["verdict"] = "fits"
        evidence["mode"] = mode
        evidence["attempts_refused"] = attempts
        return evidence
    first = attempts[0]
    if mode == "refuse":
        others = "; ".join(
            f"order {','.join(attempt['order'])} also refused"
            for attempt in attempts[1:]
        )
        raise SchedulabilityRefused(
            first["refused"] + (f" ({others})" if others else "")
        )
    return {
        "verdict": "warning",
        "mode": mode,
        "order": list(first["order"]),
        "attempts_refused": attempts,
        "warning": (
            "control-plane schedulability: the roll may not fit; proceeding because "
            f"the caller asked to warn only -- {first['refused']}"
        ),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nodes", required=True, type=Path)
    parser.add_argument("--pods", required=True, type=Path)
    parser.add_argument("--live-deployments", required=True, type=Path)
    parser.add_argument(
        "--rendered",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="final rendered Deployment JSON of a selected role (ingress|worker|spool)",
    )
    parser.add_argument("--mode", choices=MODES, default="refuse")
    parser.add_argument(
        "--fixed-order",
        help="comma-separated roles: model exactly this order instead of choosing one",
    )
    parser.add_argument("--evidence", type=Path, help="write the evidence JSON here")
    arguments = parser.parse_args(argv)
    rendered: dict[str, dict[str, Any]] = {}
    for item in arguments.rendered:
        role, _, path = item.partition("=")
        if role not in ROLE_DEPLOYMENTS or not path:
            parser.error(f"--rendered expects ROLE=PATH with a known role: {item!r}")
        rendered[role] = _read_json(Path(path))
    if not rendered:
        parser.error("at least one --rendered ROLE=PATH is required")
    fixed_order = None
    if arguments.fixed_order:
        fixed_order = [
            role for role in arguments.fixed_order.split(",") if role in rendered
        ]
        if sorted(fixed_order) != sorted(rendered):
            parser.error("--fixed-order must name every rendered role exactly once")
    try:
        evidence = plan_roll(
            nodes=_read_json(arguments.nodes),
            pods=_read_json(arguments.pods),
            live_deployments=_read_json(arguments.live_deployments),
            rendered=rendered,
            mode=arguments.mode,
            fixed_order=fixed_order,
        )
    except SchedulabilityRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if arguments.evidence is not None:
        arguments.evidence.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        )
    if evidence["verdict"] == "warning":
        print(evidence["warning"], file=sys.stderr)
    print(
        "control-plane schedulability: " + json.dumps(evidence, sort_keys=True),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
