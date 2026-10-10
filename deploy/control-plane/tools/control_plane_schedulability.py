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


def _step_demand(
    desired: dict[str, Any], live: dict[str, Any] | None
) -> dict[str, Any]:
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
        demand = _step_demand(desired, live_by_name.get(name))
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
        # A replica reduction frees its room before anything else is placed:
        # the controller scales the old ReplicaSet down first.
        if demand["removed_replicas"]:
            _release_replicas(demand, node_states, pods, name)
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
        added_placed: list[str] = []
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
    pods: dict[str, Any],
    name: str,
) -> None:
    """Give the room of the replicas a reduction removes back to their nodes
    (the live Pods of the Deployment, most-loaded nodes first)."""

    cpu = demand["pod_request"]["cpu_millicores"]
    memory = demand["pod_request"]["memory_bytes"]
    by_name = {node["name"]: node for node in node_states}
    on_nodes = sorted(
        (
            (pod.get("spec") or {}).get("nodeName")
            for pod in pods.get("items", [])
            if ((pod.get("metadata") or {}).get("labels") or {}).get("app") == name
            and (pod.get("spec") or {}).get("nodeName") in by_name
            and (pod.get("status") or {}).get("phase") not in ("Succeeded", "Failed")
        ),
        key=lambda node_name: by_name[node_name]["free_cpu_millicores"],
    )
    for node_name in on_nodes[: demand["removed_replicas"]]:
        by_name[node_name]["free_cpu_millicores"] += cpu
        by_name[node_name]["free_memory_bytes"] += memory


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
        demand = _step_demand(rendered[role], live_by_name.get(ROLE_DEPLOYMENTS[role]))
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
