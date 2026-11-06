"""The load generators' evidence of a capacity round (perf plan §5).

A saturated client cannot prove product latency. The burst and integrated
runners record, for the load Job they ran, how many load-generator Pods there
were, how many distinct nodes they ran on and the CPU each generator process
used against its Pod's CPU limit (``summary.json.load_generator``); a round in
which any generator used more than 70% of its CPU limit measured the client,
not the control plane, and is ``INVALID`` (status.json ``invalid``, exit
``INVALID_EXIT``).

Evidence only: the Pod count, placement and limits stay what the runners
always rendered. The plan's other gates -- at least four Pods, spread across
nodes, a no-op calibration at twice the target peak, a mean below 50% of the
limit -- are recorded as facts and not enforced: enforcing them changes the
load shape, and the numbers measured that way need a fresh baseline.

``client_cpu_cores`` is each generator's own reading: its process CPU seconds
per wall second of the send phase (``time.process_time`` over the threads of
the one generator process in the Pod).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

#: A generator above this share of its Pod's CPU limit invalidates the round.
CPU_LIMIT_INVALID_SHARE = 0.70
#: The plan's mean below which the generators are comfortably unsaturated.
CPU_LIMIT_MEAN_SHARE = 0.50
#: The plan asks for at least this many independent load-generator Pods.
MIN_LOAD_PODS = 4
#: The exit status (and status.json ``invalid``) of a round whose load
#: generator was saturated; ``MEASUREMENT_FAILED_EXIT`` is 4.
INVALID_EXIT = 5
INVALID_REASON = (
    "a load generator used more than 70% of its Pod's CPU limit: see "
    "summary.json load_generator"
)
LOAD_CONTAINER = "load"


def cpu_cores(quantity: object) -> float | None:
    """A Kubernetes CPU quantity (``4``, ``4000m``, ``0.5``) in cores."""

    text = str(quantity or "").strip()
    try:
        return float(text[:-1]) / 1000 if text.endswith("m") else float(text)
    except ValueError:
        return None


def plan(*, pods: int, cpu_request: str, cpu_limit: str) -> dict[str, Any]:
    """``run.json.load_generator``: what the runner renders, unchanged."""

    return {
        "pods": pods,
        "cpu_request": cpu_request,
        "cpu_limit": cpu_limit,
        "anti_affinity": False,
        "no_op_calibration": False,
    }


def load_pods(dataplane: Callable[..., str], job: str) -> list[dict[str, Any]]:
    """The Job's Pods: name, node and the generator container's CPU limit."""

    raw = dataplane(
        "get", "pod", "-l", f"batch.kubernetes.io/job-name={job}", "-o", "json"
    )
    pods = []
    for item in (json.loads(raw) if raw.strip() else {}).get("items") or []:
        spec = item.get("spec") or {}
        containers = spec.get("containers") or []
        load = next(
            (entry for entry in containers if entry.get("name") == LOAD_CONTAINER),
            containers[0] if containers else {},
        )
        limits = (load.get("resources") or {}).get("limits") or {}
        pods.append(
            {
                "name": (item.get("metadata") or {}).get("name"),
                "node": spec.get("nodeName"),
                "cpu_limit_cores": cpu_cores(limits.get("cpu")),
            }
        )
    return sorted(pods, key=lambda pod: str(pod["name"]))


def section(
    pods: list[dict[str, Any]],
    documents: list[dict],
    *,
    rendered_cpu_limit: float | None = None,
) -> dict[str, Any]:
    """``summary.json.load_generator`` and its verdict (``valid``)."""

    nodes: dict[str, int] = {}
    for pod in pods:
        if pod.get("node"):
            nodes[str(pod["node"])] = nodes.get(str(pod["node"]), 0) + 1
    limits = [pod["cpu_limit_cores"] for pod in pods if pod.get("cpu_limit_cores")]
    # One Job template: every Pod carries the same limit and the smallest
    # judges; without a Pod reading, the limit the runner rendered.
    limit = min(limits) if limits else rendered_cpu_limit
    readings = [
        (str(document.get("cluster_id") or document.get("cluster_offset")), cores)
        for document in documents
        if isinstance(cores := document.get("client_cpu_cores"), (int, float))
    ]
    record: dict[str, Any] = {
        "pods": len(pods),
        "nodes": len(nodes),
        "max_pods_per_node": max(nodes.values(), default=0),
        "cpu_limit_cores": limit,
        "readings": len(readings),
        "client_cpu_cores_max": max((cores for _, cores in readings), default=None),
        "client_cpu_cores_mean": (
            round(sum(cores for _, cores in readings) / len(readings), 6)
            if readings
            else None
        ),
        # Recorded, not enforced (perf plan §5): they change the load shape.
        "at_least_4_pods": len(pods) >= MIN_LOAD_PODS,
        "one_pod_per_node": bool(nodes) and max(nodes.values()) == 1,
        "invalid_above_limit_share": CPU_LIMIT_INVALID_SHARE,
    }
    if not limit or not readings:
        missing = "CPU limit" if not limit else "client_cpu_cores reading"
        record.update(judged=False, valid=True, reason=f"no {missing}: not judged")
        return record
    shares = [(source, cores / limit) for source, cores in readings]
    source, top = max(shares, key=lambda item: item[1])
    mean = sum(share for _, share in shares) / len(shares)
    record.update(
        judged=True,
        cpu_limit_share_max=round(top, 6),
        cpu_limit_share_mean=round(mean, 6),
        mean_below_half_limit=mean < CPU_LIMIT_MEAN_SHARE,
        valid=top <= CPU_LIMIT_INVALID_SHARE,
    )
    if record["valid"]:
        record["reason"] = (
            f"every load generator stayed within {CPU_LIMIT_INVALID_SHARE:.0%} of "
            f"its {limit:g}-core CPU limit (max {top:.0%})"
        )
    else:
        record["reason"] = (
            f"the load generator of {source} used {top * limit:.2f} of its "
            f"{limit:g}-core CPU limit ({top:.0%} > {CPU_LIMIT_INVALID_SHARE:.0%}): "
            "a saturated client cannot prove product latency"
        )
    return record


def evidence(
    dataplane: Callable[..., str],
    job: str,
    documents: list[dict],
    *,
    cpu_limit: str,
) -> dict[str, Any]:
    """Read the load Job's Pods (before the teardown deletes them) and judge
    the generators' CPU against their limit -- the one the runner rendered
    when the Pods cannot be read, which is then recorded."""

    try:
        pods, error = load_pods(dataplane, job), None
    except Exception as exc:  # noqa: BLE001 - recorded; the CPU gate still judges
        pods, error = [], " ".join(f"{type(exc).__name__}: {exc}".split())[-300:]
    record = section(pods, documents, rendered_cpu_limit=cpu_cores(cpu_limit))
    if error is not None:
        record["pods_read_error"] = error
    return record
