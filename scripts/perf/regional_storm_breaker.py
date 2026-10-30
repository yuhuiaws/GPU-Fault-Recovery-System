"""The remediation storm breaker step's building blocks (perf plan §8.5).

The suites keep every synthetic cluster below its breaker
(``regional_synthetic_fleet``); the step (``regional_storm_breaker_suite``)
makes one cluster cross it on purpose, at the product's constants, and judges
each part of the breaker's cycle on the product's own state, never on log
text:

* the plan: the cluster registers ``nodes`` GPU nodes, so the breaker trips at
  ``threshold = max(3, ceil(20% x nodes))`` distinct nodes of two or more
  workload domains within 600 s; with ``k`` nodes a workflow (one incident, no
  job: one domain each) the first ``admitted = max(1, ceil(threshold / k) - 1)``
  workflows enter, the next one trips the breaker and is held, and so is every
  one after it (``breaker_plan``);
* the breaker record and the workflows the product lists as held by it, and
  the step's own workflows and their remote commands (``storm_breaker_pod.py``
  through ``breaker_status``);
* the ``gpu_fault_remediation_budget_storm_breaker_*`` gauges of a
  control-plane Pod's ``/metrics`` (``storm_gauges``);
* the AMP alert state of ``GpuFaultRemediationStormBreakerOpen`` for the
  cluster, and whether the AMP rules carry it at all (``alert_states``,
  ``alert_rule_loaded``);
* the operator exits ``release`` and ``abort``: the very Pod program
  ``gpu-fault-admin remediation-storm`` runs (``storm_operator_action``). The
  verb itself cannot name a synthetic cluster -- it accepts only the clusters
  of the managed site (``workflow_reconcile.cluster_target``) -- so the step
  runs its program through the same exec channel, with the operator identity,
  ``--reason`` and ``--reference`` it would record, and keeps the evidence in
  the run directory instead of ``<state-dir>/remediation-storm/``.
"""

from __future__ import annotations

import functools
import json
import math
import statistics
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpu_fault.remediation_storm import storm_scope, storm_threshold
from scripts.e2e.regional.regional_live_fixture import component_python

if TYPE_CHECKING or __package__:
    from . import regional_capacity_registry as registry
    from .regional_capacity_timeseries import metrics_port
else:
    import regional_capacity_registry as registry
    from regional_capacity_timeseries import metrics_port

PERF_DIR = Path(__file__).resolve().parent
STATUS_SCRIPT = PERF_DIR / "storm_breaker_pod.py"
STORM_CLUSTER = "perf-cap-000"
ALERT_NAME = "GpuFaultRemediationStormBreakerOpen"
GAUGE_PREFIX = "gpu_fault_remediation_budget_storm_breaker_"
GAUGES = (
    "open",
    "window_nodes",
    "threshold_nodes",
    "tripped_timestamp_seconds",
    "held_workflows",
)
#: The action model's steps; the simulated executor runs one command each.
WORKFLOW_COMMANDS = 10
HELD_STATUSES = frozenset({"PENDING", "SAFETY_PENDING"})
TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"})

Control = Callable[..., str]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: object) -> datetime | None:
    return datetime.fromisoformat(value) if isinstance(value, str) and value else None


def breaker_plan(
    *, nodes_per_cluster: int, nodes_per_workflow: int, held_workflows: int
) -> dict[str, Any]:
    """Where the breaker trips on a ``nodes_per_cluster`` cluster (module docstring)."""

    if nodes_per_workflow < 1 or held_workflows < 1:
        raise ValueError("the breaker step needs >= 1 node a workflow and >= 1 held")
    threshold = storm_threshold(nodes_per_cluster)
    admitted = max(1, math.ceil(threshold / nodes_per_workflow) - 1)
    workflows = admitted + held_workflows
    model_nodes = workflows * nodes_per_workflow
    if model_nodes > nodes_per_cluster:
        raise ValueError(
            f"the breaker step needs {model_nodes} workflow nodes, more than the "
            f"{nodes_per_cluster} GPU nodes its cluster registers"
        )
    return {
        "cluster_id": STORM_CLUSTER,
        "nodes_per_cluster": nodes_per_cluster,
        "threshold_nodes": threshold,
        "nodes_per_workflow": nodes_per_workflow,
        "workflows": workflows,
        "admitted_workflows": admitted,
        "held_workflows": held_workflows,
        "trip_nodes": (admitted + 1) * nodes_per_workflow,
        "trip_domains": admitted + 1,
        "model_nodes": model_nodes,
        "commands": workflows * WORKFLOW_COMMANDS,
    }


def plan_line(plan: dict[str, Any]) -> str:
    return (
        f"storm breaker step: {plan['cluster_id']} registers "
        f"{plan['nodes_per_cluster']} GPU nodes, threshold "
        f"{plan['threshold_nodes']}; {plan['workflows']} job-less workflows of "
        f"{plan['nodes_per_workflow']} nodes: {plan['admitted_workflows']} admitted, "
        f"the next trips it at {plan['trip_nodes']} nodes across "
        f"{plan['trip_domains']} domains, {plan['held_workflows']} held until release"
    )


def api_pod(control_fn: Control) -> str:
    pod = control_fn(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    if not pod:
        raise RuntimeError("no Running gpu-fault-api-ha Pod")
    return pod


def last_json(output: str) -> dict[str, Any]:
    for line in reversed(output.strip().splitlines()):
        try:
            document = json.loads(line)
        except ValueError:
            continue
        if isinstance(document, dict):
            return document
    return {}


def operator() -> str:
    from gpu_fault.admin.operator_identity import (
        local_operator_identity,
        resolve_operator_identity,
    )

    return resolve_operator_identity(fallback=local_operator_identity())


def storm_operator_action(
    cluster_id: str,
    action: str,
    *,
    reason: str,
    reference: str,
    dry_run: bool = False,
    control_fn: Control | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """``remediation-storm release|abort`` of one cluster, through the verb's
    own Pod program (module docstring); raises on a refusal."""

    from gpu_fault.admin.remediation_storm import ACTIONS, SCRIPT

    if action not in ACTIONS:
        raise ValueError(f"unknown remediation storm action {action!r}")
    kube = registry.control if control_fn is None else control_fn
    payload = {
        "cluster_id": cluster_id,
        "action": action,
        "reason": reason,
        "reference": reference,
        "actor": actor or operator(),
        "dry_run": dry_run,
    }
    output = kube(
        "exec",
        "-i",
        api_pod(kube),
        "--",
        component_python("cpu"),
        "-c",
        SCRIPT,
        stdin=json.dumps(payload, separators=(",", ":")).encode(),
        check=False,
        timeout=300,
    )
    document = last_json(output)
    if not document or "error" in document or "status" not in document:
        detail = document.get("error") if document else output.strip()[-300:]
        raise RuntimeError(f"remediation-storm {action} of {cluster_id}: {detail}")
    return {
        **document,
        "operator": payload["actor"],
        "reason": reason,
        "reference": reference,
        "recorded_at": utc_now().isoformat(),
    }


def breaker_status(
    *, run_id: str, workflows: int, control_fn: Control | None = None
) -> dict[str, Any]:
    """One read of ``storm_breaker_pod.py`` in an api-ha Pod."""

    kube = registry.control if control_fn is None else control_fn
    output = kube(
        "exec",
        "-i",
        api_pod(kube),
        "--",
        "env",
        f"STORM_RUN_ID={run_id}",
        f"STORM_CLUSTER_ID={STORM_CLUSTER}",
        f"STORM_WORKFLOWS={workflows}",
        component_python("cpu"),
        "-",
        stdin=STATUS_SCRIPT.read_bytes(),
        timeout=120,
    )
    document = last_json(output)
    if document.get("cluster_id") != STORM_CLUSTER:
        raise RuntimeError("storm breaker status read returned no status")
    return document


def parse_storm_gauges(body: str, cluster_id: str) -> dict[str, float]:
    label = f'cluster_id="{cluster_id}"'
    values: dict[str, float] = {}
    for line in body.splitlines():
        if not line.startswith(GAUGE_PREFIX):
            continue
        name, _, value = line.rpartition(" ")
        family, _, labels = name.partition("{")
        if label not in labels:
            continue
        try:
            values[family.removeprefix(GAUGE_PREFIX)] = float(value)
        except ValueError:
            continue
    return values


def storm_gauges(
    cluster_id: str, *, control_fn: Control | None = None
) -> dict[str, float]:
    """The cluster's breaker gauges on one api-ha Pod's Pod-coherent /metrics."""

    kube = registry.control if control_fn is None else control_fn
    pod = api_pod(kube)
    body = kube(
        "exec",
        pod,
        "--",
        "python3",
        "-c",
        (
            "import urllib.request;print(urllib.request.urlopen("
            f"'http://127.0.0.1:{metrics_port(pod)}/metrics',timeout=15)"
            ".read().decode())"
        ),
        timeout=90,
    )
    return parse_storm_gauges(body, cluster_id)


@functools.cache
def amp_workspace() -> str:
    """The site's AMP workspace, resolved once a run (an ``aws`` call)."""

    return registry._amp_workspace_id()


def amp_get(path: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """A SigV4-signed GET of the site's AMP workspace (Prometheus API)."""

    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    region = registry.AWS_REGION
    url = (
        f"https://aps-workspaces.{region}.amazonaws.com/workspaces/"
        f"{amp_workspace()}{path}"
    )
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials are unavailable for the AMP read")
    signed = AWSRequest(method="GET", url=url)
    SigV4Auth(credentials, "aps", region).add_auth(signed)
    request = urllib.request.Request(url, headers=dict(signed.headers), method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        document = json.load(response)
    if not isinstance(document, dict) or document.get("status") != "success":
        raise RuntimeError(f"AMP {path} answered without success")
    return document


def alert_states(
    cluster_id: str, *, fetch: Callable[[str], dict[str, Any]] = amp_get
) -> list[str]:
    """``pending`` / ``firing`` of the breaker alert for one cluster in AMP."""

    alerts = (fetch("/api/v1/alerts").get("data") or {}).get("alerts")
    if not isinstance(alerts, list):
        raise RuntimeError("AMP alert inventory is missing")
    return sorted(
        {
            str(item.get("state"))
            for item in alerts
            if isinstance(item, dict)
            and (item.get("labels") or {}).get("alertname") == ALERT_NAME
            and (item.get("labels") or {}).get("cluster_id") == cluster_id
        }
    )


def alert_rule_loaded(*, fetch: Callable[[str], dict[str, Any]] = amp_get) -> bool:
    groups = (fetch("/api/v1/rules").get("data") or {}).get("groups") or []
    return any(
        isinstance(rule, dict) and rule.get("name") == ALERT_NAME
        for group in groups
        if isinstance(group, dict)
        for rule in group.get("rules") or []
    )


def classify(status: dict[str, Any]) -> dict[str, Any]:
    """One status read as the step judges it."""

    scope = storm_scope(str(status.get("cluster_id")))
    workflows = status.get("workflows") or {}
    commands = status.get("commands") or {}
    breaker = status.get("breaker") or {}

    def ids(predicate: Callable[[dict[str, Any]], bool]) -> list[str]:
        return sorted(
            request_id
            for request_id, view in workflows.items()
            if isinstance(view, dict) and predicate(view)
        )

    held = ids(
        lambda view: view["status"] in HELD_STATUSES
        and view["last_blocked_scope"] == scope
    )
    return {
        "breaker_state": breaker.get("state", "CLOSED"),
        "breaker_open": bool(breaker.get("open")),
        "held": held,
        "product_held": sorted(status.get("held_workflows") or []),
        "running": ids(lambda view: view["status"] == "RUNNING"),
        "succeeded": ids(lambda view: view["status"] == "SUCCEEDED"),
        "terminal": ids(lambda view: view["status"] in TERMINAL_STATUSES),
        "missing": sorted(key for key, view in workflows.items() if view is None),
        "with_commands": sorted(
            key for key, view in commands.items() if int(view.get("count", 0))
        ),
        "tripped_by": ids(lambda view: bool(view.get("tripped_breaker"))),
    }


def held_violations(snapshot: dict[str, Any], held_ids: list[str]) -> list[str]:
    """What a held workflow must not do while the breaker is open."""

    violations = []
    for request_id in held_ids:
        if request_id in snapshot["with_commands"]:
            violations.append(f"{request_id} has remote commands while held")
        if request_id in snapshot["running"] or request_id in snapshot["terminal"]:
            violations.append(f"{request_id} left PENDING while held")
    return violations


def spread(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "max": None}
    return {
        "count": len(values),
        "p50": round(statistics.median(values), 3),
        "max": round(max(values), 3),
    }


def workflow_times(
    status: dict[str, Any], request_id: str
) -> tuple[datetime | None, datetime | None]:
    """``(first remote command, terminal)`` of one workflow (``None``: not yet)."""

    view = (status.get("workflows") or {}).get(request_id) or {}
    command = (status.get("commands") or {}).get(request_id) or {}
    terminal = (
        parse_time(view.get("updated_at"))
        if view.get("status") in TERMINAL_STATUSES
        else None
    )
    return parse_time(command.get("first_created_at")), terminal


def drain_measurements(
    status: dict[str, Any],
    *,
    released_at: datetime,
    held_ids: list[str],
    admitted_ids: list[str],
) -> dict[str, Any]:
    """The step's headline numbers, from the final status read: how long after
    the release the held work resumed and completed, beside how long one
    workflow takes once it runs (held and admitted alike)."""

    def times(ids: list[str]) -> list[tuple[datetime | None, datetime | None]]:
        return [workflow_times(status, request_id) for request_id in ids]

    held = times(held_ids)
    resumed = [(first - released_at).total_seconds() for first, _ in held if first]
    completed = [(end - released_at).total_seconds() for _, end in held if end]
    execution = {
        name: spread(
            [
                (end - first).total_seconds()
                for first, end in times(ids)
                if first and end
            ]
        )
        for name, ids in (("held", held_ids), ("admitted", admitted_ids))
    }
    complete = len(completed) == len(held_ids) and bool(held_ids)
    return {
        "released_at": released_at.isoformat(),
        "release_to_first_held_command_seconds": min(resumed) if resumed else None,
        "release_to_last_held_command_seconds": (
            max(resumed) if len(resumed) == len(held_ids) and resumed else None
        ),
        "release_to_held_completion_seconds": max(completed) if complete else None,
        "held_completed": len(completed),
        "workflow_execution_seconds": execution,
    }
