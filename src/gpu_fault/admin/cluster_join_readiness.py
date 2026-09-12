"""The join's ``COLLECTORS_READY`` gate: fast kinds reported, slow kinds scheduled.

The control plane's ``/v1/collector-readiness/{cluster_id}`` says a node is
ready once *every* collector kind has one successful report inside its silence
threshold. On a freshly installed node that means waiting a whole report period
of the slowest kind: the DCGM and Fabric Manager collectors post their channel
unconditionally only with the health summary (300 s by default), so the gate of
join attempt 9 (2026-09-12) passed 227 s after the agents were installed while
the fast kinds had reported within the first minute.

This gate reads the same endpoint but applies the join's rule, control-plane
side only -- the node wheel is untouched:

* a node's Agent row must be live: ``POST /v1/fleet/readiness`` over the
  cluster's HyperPod nodes proves an ACTIVE row with an unexpired lease and the
  release's required identity, one node per HyperPod node;
* every *fast* kind (configured unconditional report period at most
  ``FAST_REPORT_PERIOD_CEILING_SECONDS``) must have reported once on every node
  -- the endpoint's own ``ready`` verdict, so its silence threshold still holds;
* every *slow* kind is verified as *scheduled* rather than reported: the Agent
  heartbeat shows its systemd unit active. Its first report is confirmed by the
  existing verify/status path (``control_api`` -> ``collector_readiness``) the
  next time it runs after the report lands.

Which kinds are slow is derived from the configured report period, never from
a list of names: ``COLLECTOR_REPORT_PERIODS`` names, per kind, the environment
variable the node installer writes (``deploy/node/install-gpu-fault-collector.sh``)
and its default, and the split is ``period > ceiling``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite
from gpu_fault.collector_registry import COLLECTOR_KINDS, CollectorKind

#: A kind whose unconditional report period is longer than this is not waited
#: for at the gate. One minute: the gate then costs about one GPU inventory
#: period after the agents come up instead of one health-summary period.
FAST_REPORT_PERIOD_CEILING_SECONDS = 60.0

#: kind -> (installer environment variable, default seconds): the period at
#: which the node collector posts the kind's channel *unconditionally*. Kinds
#: that also post on data (log lines, metric edges) usually report sooner, but
#: only the periodic post is configured behaviour, so only it counts here.
COLLECTOR_REPORT_PERIODS: dict[CollectorKind, tuple[str, float]] = {
    # ``dcgm`` emits the inventory every interval.
    CollectorKind.GPU_INVENTORY: ("GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS", 60),
    # ``dcgm`` metrics are edge-filtered; the health summary is the periodic post.
    CollectorKind.GPU_METRICS: ("GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS", 300),
    # The host collector posts every sample.
    CollectorKind.HOST_TELEMETRY: ("GPU_FAULT_HOST_INTERVAL_SECONDS", 15),
    # Log collectors post entries when there are any; the summary is periodic.
    CollectorKind.NODE_LOGS: ("GPU_FAULT_NODE_LOG_HEALTH_SUMMARY_SECONDS", 300),
    CollectorKind.NVIDIA_KERNEL: ("GPU_FAULT_KERNEL_HEALTH_SUMMARY_SECONDS", 300),
    CollectorKind.FABRIC_MANAGER_LOG: (
        "GPU_FAULT_FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS",
        300,
    ),
}

#: Unit states the Agent reports that count as "scheduled". ``unknown`` is what
#: the readiness endpoint itself accepts for an Agent that does not report unit
#: states; it never widens a kind the endpoint would refuse.
SCHEDULED_UNIT_STATES = frozenset({"active", "unknown"})

JOIN_READINESS_SCRIPT = r"""
import json
import os
import urllib.parse
import urllib.request

cluster_id = os.environ["ADMIN_CLUSTER_ID"]
expected = json.loads(os.environ["ADMIN_EXPECTED_NODES_JSON"])
headers = {"X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"]}


def get(path):
    request = urllib.request.Request("http://127.0.0.1:8080" + path, headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def post(path, body):
    request = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers={**headers, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


collectors = get("/v1/collector-readiness/" + urllib.parse.quote(cluster_id, safe=""))
node_ids = sorted(expected) or sorted(
    str(item["node_id"]) for item in collectors.get("nodes") or []
)
fleet = (
    post("/v1/fleet/readiness", {"cluster_id": cluster_id, "node_ids": node_ids})
    if node_ids
    else None
)
print(json.dumps({"collectors": collectors, "fleet": fleet}, separators=(",", ":")))
"""


def _validate_report_periods() -> None:
    missing = sorted(
        kind.value
        for kind, spec in COLLECTOR_KINDS.items()
        if spec.systemd_unit is not None and kind not in COLLECTOR_REPORT_PERIODS
    )
    if missing:
        raise RuntimeError(
            "node collector kinds without a configured report period: "
            + ", ".join(missing)
        )


_validate_report_periods()


def collector_report_periods(
    environ: Mapping[str, str] | None = None,
) -> dict[CollectorKind, float]:
    """The unconditional report period of every node collector kind, in seconds."""

    environment = os.environ if environ is None else environ
    periods = {}
    for kind, (name, default) in COLLECTOR_REPORT_PERIODS.items():
        raw = str(environment.get(name, "") or "").strip()
        try:
            period = float(raw) if raw else float(default)
        except ValueError as exc:
            raise BootstrapError(f"{name} is not a number: {raw!r}") from exc
        periods[kind] = period
    return periods


def classify_collector_kinds(
    periods: Mapping[CollectorKind, float],
    *,
    ceiling_seconds: float = FAST_REPORT_PERIOD_CEILING_SECONDS,
) -> tuple[frozenset[str], frozenset[str]]:
    """``(waited, deferred)`` kind names: the split is period against ceiling."""

    waited = frozenset(
        kind.value for kind, period in periods.items() if period <= ceiling_seconds
    )
    deferred = frozenset(kind.value for kind in periods) - waited
    return waited, deferred


@dataclass(frozen=True)
class JoinReadinessVerdict:
    ready: bool
    waited_kinds: tuple[str, ...]
    deferred_kinds: tuple[str, ...]
    #: node id -> {"waiting_for": [...], "deferred": {kind: {...}}, "ready": bool}
    nodes: dict[str, dict[str, Any]]
    missing_nodes: tuple[str, ...]
    fleet_ready: bool
    fleet_reasons: tuple[str, ...]
    periods_seconds: dict[str, float]

    def evidence(self) -> dict[str, Any]:
        """What ``state.json`` records beside the step timestamp."""

        deferred: dict[str, dict[str, Any]] = {}
        for kind in self.deferred_kinds:
            per_node = [
                node["deferred"][kind]
                for node in self.nodes.values()
                if kind in node["deferred"]
            ]
            deferred[kind] = {
                "report_period_seconds": self.periods_seconds.get(kind),
                "verified_as": "scheduled",
                "scheduled_nodes": sum(1 for item in per_node if item["scheduled"]),
                "reported_nodes": sum(1 for item in per_node if item["reported"]),
                "nodes": len(per_node),
            }
        return {
            "ready": self.ready,
            "node_count": len(self.nodes),
            "nodes": sorted(self.nodes),
            "missing_nodes": list(self.missing_nodes),
            "waited_kinds": list(self.waited_kinds),
            "deferred_kinds": deferred,
            "fast_report_period_ceiling_seconds": FAST_REPORT_PERIOD_CEILING_SECONDS,
            "fleet_ready": self.fleet_ready,
            "fleet_reasons": list(self.fleet_reasons),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

    def describe_shortfall(self) -> str:
        parts = []
        if self.missing_nodes:
            parts.append("no Agent row for: " + ", ".join(self.missing_nodes))
        for node_id, node in sorted(self.nodes.items()):
            if node["waiting_for"]:
                parts.append(f"{node_id} waiting for " + ", ".join(node["waiting_for"]))
            unscheduled = sorted(
                kind
                for kind, item in node["deferred"].items()
                if not (item["scheduled"] or item["ready"])
            )
            if unscheduled:
                parts.append(f"{node_id} unit not active for " + ", ".join(unscheduled))
        if not self.fleet_ready:
            parts.append(
                "fleet readiness: " + ("; ".join(self.fleet_reasons) or "not ready")
            )
        return "; ".join(parts) or "ready"


def _fleet_verdict(fleet: Mapping[str, Any] | None) -> tuple[bool, tuple[str, ...]]:
    if not isinstance(fleet, Mapping):
        return False, ("fleet readiness was not evaluated",)
    reasons: list[str] = []
    for item in fleet.get("nodes") or []:
        if isinstance(item, Mapping) and not item.get("ready"):
            node = str(item.get("node_id") or "?")
            reasons.extend(f"{node}: {reason}" for reason in item.get("reasons") or [])
    return bool(fleet.get("ready")), tuple(reasons)


def evaluate_join_readiness(
    report: Mapping[str, Any],
    *,
    expected_nodes: Sequence[str] | None,
    fleet: Mapping[str, Any] | None,
    periods: Mapping[CollectorKind, float] | None = None,
) -> JoinReadinessVerdict:
    """Apply the join's rule to one readiness payload.

    ``report`` is the ``/v1/collector-readiness`` document; ``fleet`` the
    ``/v1/fleet/readiness`` answer over the expected nodes. ``expected_nodes``
    is the cluster's HyperPod node list; ``None`` accepts the Agent rows the
    endpoint lists (an older attempt record without the node list).
    """

    resolved = dict(periods if periods is not None else collector_report_periods())
    waited, deferred = classify_collector_kinds(resolved)
    by_node = {
        str(item.get("node_id")): item
        for item in report.get("nodes") or []
        if isinstance(item, Mapping) and item.get("node_id")
    }
    selected = sorted(expected_nodes) if expected_nodes else sorted(by_node)
    missing = tuple(node for node in selected if node not in by_node)
    nodes: dict[str, dict[str, Any]] = {}
    for node_id in selected:
        item = by_node.get(node_id)
        if item is None:
            continue
        collectors = cast(Mapping[str, Any], item.get("collectors") or {})
        waiting = []
        deferred_detail: dict[str, dict[str, Any]] = {}
        for kind, status in collectors.items():
            detail = cast(Mapping[str, Any], status or {})
            if kind in deferred:
                unit_state = str(detail.get("unit_state") or "unknown")
                deferred_detail[kind] = {
                    "reported": detail.get("last_success_at") is not None,
                    "ready": bool(detail.get("ready")),
                    "unit_state": unit_state,
                    "scheduled": unit_state in SCHEDULED_UNIT_STATES,
                }
            elif not detail.get("ready"):
                # A kind outside the period table (a plugin) is waited for:
                # the conservative side of an unknown cadence.
                waiting.append(kind)
        nodes[node_id] = {
            "waiting_for": sorted(waiting),
            "deferred": deferred_detail,
            "ready": not waiting
            and all(
                item["scheduled"] or item["ready"] for item in deferred_detail.values()
            ),
        }
    fleet_ready, fleet_reasons = _fleet_verdict(fleet)
    ready = (
        bool(nodes)
        and not missing
        and all(node["ready"] for node in nodes.values())
        and fleet_ready
    )
    return JoinReadinessVerdict(
        ready=ready,
        waited_kinds=tuple(sorted(waited)),
        deferred_kinds=tuple(sorted(deferred)),
        nodes=nodes,
        missing_nodes=missing,
        fleet_ready=fleet_ready,
        fleet_reasons=fleet_reasons,
        periods_seconds={kind.value: period for kind, period in resolved.items()},
    )


def join_readiness_report(
    site: RenderedSite,
    cluster_id: str,
    *,
    expected_nodes: Sequence[str],
) -> dict[str, Any]:
    """One exec into the CPU ingress Pod: collector readiness plus fleet readiness."""

    kubectl = [
        "kubectl",
        "--kubeconfig",
        str(site.release_config["cpu_kubeconfig"]),
        "-n",
        str(site.release_config["namespace"]),
    ]
    pod = subprocess.run(
        [
            *kubectl,
            "get",
            "pod",
            "-l",
            "app=gpu-fault-api-ha",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        text=True,
        capture_output=True,
    )
    if pod.returncode or not pod.stdout.strip():
        raise BootstrapError("cannot find a Running CPU ingress Pod")
    result = subprocess.run(
        [
            *kubectl,
            "exec",
            pod.stdout.strip(),
            "--",
            "env",
            f"ADMIN_CLUSTER_ID={cluster_id}",
            "ADMIN_EXPECTED_NODES_JSON="
            + json.dumps(sorted(expected_nodes), separators=(",", ":")),
            "python",
            "-c",
            JOIN_READINESS_SCRIPT,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise BootstrapError(
            "collector readiness query failed: " + result.stderr.strip()
        )
    return cast(dict[str, Any], json.loads(result.stdout))


ReadinessReporter = Callable[..., Mapping[str, Any]]


def wait_join_collector_readiness(
    site: RenderedSite,
    cluster_id: str,
    *,
    expected_nodes: Sequence[str] | None,
    timeout_seconds: float = 600,
    interval_seconds: float = 10,
    report: ReadinessReporter = join_readiness_report,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll until the join's readiness rule holds; the evidence is the verdict.

    A fast kind that never reports, a HyperPod node without a live Agent row or
    a slow kind whose unit is not active by the deadline is a hard failure, as
    the whole-set gate was.
    """

    deadline = clock() + timeout_seconds
    periods = collector_report_periods()
    last: JoinReadinessVerdict | None = None
    last_error: str | None = None
    while True:
        try:
            document = report(
                site, cluster_id, expected_nodes=list(expected_nodes or ())
            )
            last = evaluate_join_readiness(
                cast(Mapping[str, Any], document.get("collectors") or {}),
                expected_nodes=expected_nodes,
                fleet=cast(Mapping[str, Any] | None, document.get("fleet")),
                periods=periods,
            )
            last_error = None
            if last.ready:
                return last.evidence()
        except (BootstrapError, json.JSONDecodeError, KeyError, TypeError) as exc:
            last_error = str(exc)
        if clock() >= deadline:
            break
        sleep(interval_seconds)
    detail = last_error or (last.describe_shortfall() if last else "no report")
    raise BootstrapError(
        f"{cluster_id} collectors did not become ready within "
        f"{timeout_seconds:g}s: {detail}"
    )
