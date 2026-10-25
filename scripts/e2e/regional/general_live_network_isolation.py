"""Reversible host-level network isolation shared by GF-GENERAL-LIVE 028/011.

Two product channels are cut on a node for a bounded window, each by tagged
``REJECT --reject-with tcp-reset`` rules the node probe
``probes/general_live_isolation_probe.py`` inserts after arming a
``systemd-run --on-active`` deadman that removes them even when the runner,
the probe Pod and the control plane are all gone:

* 028 ``api_unreachable`` -- ``pod-egress``: on the node that runs the
  completion watcher Pod, FORWARD rules reject TCP from that Pod's IP to the
  control-plane endpoint (the URL of ``gpu-fault-regional-connection``) on its
  port. Only the watcher's target access is blocked: the Pod IP is the source
  match, so no other Pod, no host unit and no other destination is touched,
  and the block never outlives ``WATCHER_BLOCK_SECONDS`` (the control plane
  reads a cluster's coverage UNKNOWN 600 s after the watcher's last heartbeat
  and blocks node-mutating plans; the deadman fires before that).
* 011 ``unreachable_node`` -- ``agent-ingress``: on the attempt's second GPU
  node, one INPUT rule rejects TCP to the Node Agent's listening port -- the
  channel the cluster executor pushes node actions to
  (``/v1/node-actions/submit``). The Agent's outbound heartbeats, every
  collector and kubelet keep working, so the control plane still believes the
  node is covered and has to record the gap itself.

The verdicts read only the product's own records: the attempt's completion
records and restart ledger (``general_live_store.attempt_records``), the hung
workflow's step executions (``general_live_nccl``), and this module's
read-only ``ISOLATION_RECORDS_PROBE`` -- the watcher's coverage heartbeat, the
attempt observations it published, the notifications of the attempt's
incidents and the Agent record of a node. The runner's own oracle is the
block's effect on those records (a heartbeat that stops advancing, a terminal
that does not arrive while the block holds) and the probe's L4 connectivity
reading; a block that did not bite is INCONCLUSIVE, never a product FAIL.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from scripts.e2e.regional.general_live_fixture import (
    PROBES,
    ProbeHandle,
    VariantContext,
    VariantNotRun,
)
from scripts.e2e.regional.general_live_registry import HARDWARE_OPERATIONS
from scripts.e2e.regional.general_live_verdicts import (
    TERMINAL_WORKFLOWS,
    executions,
    node_isolated,
    parse_time,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_pod_inventory import (
    pod_inventory,
    ready_pod_records,
)

ISOLATION_PROBE = PROBES / "general_live_isolation_probe.py"
WATCHER_APP = "gpu-fault-completion-watcher"
CONNECTION_SECRET = "gpu-fault-regional-connection"
CONTROL_PLANE_URL_KEY = "control-plane-url"
AGENT_UNIT = "gpu-fault-node-agent.service"
POD_EGRESS = "pod-egress"
AGENT_INGRESS = "agent-ingress"
KINDS = (POD_EGRESS, AGENT_INGRESS)
#: What the watcher's emergency fallback writes on the workload and its Pods
#: (``completion_workload_stopper.KubernetesWorkloadStopper.stop``).
EMERGENCY_MODE_ANNOTATION = "gpu-fault.io/passive-stop-mode"
EMERGENCY_MODE = "emergency-fallback"
TERMINATION_ANNOTATION = "gpu-fault.io/termination-initiator-incident-id"
#: The watcher block's ceiling: under the control plane's 600 s coverage
#: horizon, so a runner that dies mid-variant never leaves the cluster
#: reading UNKNOWN for long.
WATCHER_BLOCK_SECONDS = 540
#: The Agent block spans the stall proof (about 930 s) and the hung chain's
#: evidence steps (the 600 s hung threshold plus the summary/triage waves),
#: usually under 1200 s; the deadman gets an hour so a slow chain is never
#: restored mid-evidence, and the probe caps a deadman at 5400 s anyway.
AGENT_BLOCK_SECONDS = 3600
MIN_BLOCK_SECONDS = 120
#: Operations the hung evidence workflow collects from the Node Agents; the
#: ones after them tell the evidence phase is over.
HUNG_EVIDENCE_OPERATIONS = ("COLLECT_HUNG_SUMMARY", "COLLECT_HUNG_TRIAGE")
HUNG_LATER_OPERATIONS = (
    "COLLECT_DIAGNOSTIC_BUNDLE",
    "VALIDATE_FABRIC",
    "ESCALATE_SUPPORT",
)
#: A hung chain is diagnostic-only; none of these may run on a suspicion.
FORBIDDEN_OPERATIONS = HARDWARE_OPERATIONS | {
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "STOP_WORKLOADS",
    "RESTART_WORKLOAD",
}

#: Read-only: the watcher's coverage heartbeat of the cluster, the attempt
#: observations it published for a job, the notifications of the given
#: incidents and one node's Agent record. Never writes.
ISOLATION_RECORDS_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

cluster_id, job_id, node_id, incident_text = sys.argv[1:5]
incident_ids = {item for item in incident_text.split(",") if item}
store = ApplicationContext.from_environment().store

heartbeat = store.get_workload_coverage_heartbeat(cluster_id)
observations = [
    {
        "attempt_id": item.attempt_id,
        "job_id": item.job_id,
        "workload_phase": item.workload_phase.value,
        "observed_at": item.observed_at.isoformat(),
        "containers": len(item.containers),
    }
    for item in store.list_attempt_observations(cluster_id)
    if not job_id or item.job_id == job_id
]
notifications = []
if incident_ids:
    for notification in store.list_notifications(limit=500, newest_first=True):
        if notification.incident_id not in incident_ids:
            continue
        result = store.get_notification_result(notification.notification_id)
        notifications.append({
            "notification_id": notification.notification_id,
            "incident_id": notification.incident_id,
            "category": notification.category,
            "deduplication_key": notification.deduplication_key,
            "created_at": notification.created_at.isoformat(),
            "subject": notification.subject,
            "status": result.status.value if result else None,
        })
agent = None
if node_id:
    try:
        record = store.get_agent(cluster_id, node_id)
        agent = {
            "node_id": record.node_id,
            "endpoint": record.endpoint,
            "last_seen_at": record.last_seen_at.isoformat(),
            "lifecycle_state": record.lifecycle_state.value,
            "lease_expires_at": (
                record.lease_expires_at.isoformat() if record.lease_expires_at else None
            ),
        }
    except NotFoundError:
        agent = None
print(json.dumps({
    "cluster_id": cluster_id,
    "heartbeat": heartbeat.model_dump(mode="json") if heartbeat is not None else None,
    "observations": observations,
    "notifications": notifications,
    "agent": agent,
}, sort_keys=True, default=str))
"""


def isolation_records(
    regional: Any,
    cluster_id: str,
    *,
    job_id: str = "",
    node: str = "",
    incident_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if any("," in item or not item for item in incident_ids):
        raise ValueError("incident ids must be non-empty, no comma")
    return regional.cpu_python(
        ISOLATION_RECORDS_PROBE,
        cluster_id,
        job_id,
        node,
        ",".join(incident_ids),
        timeout=240,
    )


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def watcher_pod(regional: Any) -> dict[str, Any]:
    """The one Ready completion-watcher Pod: name, UID, node, Pod IP.

    The Deployment is a singleton (``replicas: 1``, Recreate); two Pods mean a
    rollout in progress and nothing to isolate deterministically. A hostNetwork
    Pod would make a source-IP block node-wide, so it is refused too.
    """

    inventory = pod_inventory(regional, "gpu", WATCHER_APP)
    ready = {item["uid"] for item in ready_pod_records(inventory)}
    pods = []
    for item in inventory.get("items") or []:
        metadata = item.get("metadata") or {}
        spec = item.get("spec") or {}
        status = item.get("status") or {}
        if status.get("phase") in {"Succeeded", "Failed"} or not metadata.get("uid"):
            continue
        pods.append(
            {
                "name": metadata.get("name"),
                "uid": metadata.get("uid"),
                "resource_version": metadata.get("resourceVersion"),
                "node": spec.get("nodeName"),
                "pod_ip": status.get("podIP"),
                "host_network": bool(spec.get("hostNetwork")),
                "phase": status.get("phase"),
                "ready": metadata.get("uid") in ready,
            }
        )
    if len(pods) != 1 or not pods[0]["ready"]:
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            f"expected exactly one Ready {WATCHER_APP} Pod, found "
            f"{[(p['name'], p['phase'], p['ready']) for p in pods]}",
        )
    pod = pods[0]
    if pod["host_network"] or not pod["pod_ip"] or not pod["node"]:
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            f"{WATCHER_APP} Pod {pod['name']} has no Pod IP of its own to scope the block to",
        )
    return pod


def control_plane_endpoint(regional: Any) -> tuple[str, int]:
    """``(host, port)`` of the control-plane URL the data plane connects to.

    Read from the one key of ``gpu-fault-regional-connection`` that is a URL;
    the cluster token in the same Secret is never read.
    """

    encoded = regional.kubectl(
        "gpu",
        "get",
        "secret",
        CONNECTION_SECRET,
        "-o",
        f"jsonpath={{.data.{CONTROL_PLANE_URL_KEY}}}",
    ).strip()
    if not encoded:
        raise RegionalFixtureError(
            f"{CONNECTION_SECRET} carries no {CONTROL_PLANE_URL_KEY}"
        )
    return endpoint_of(base64.b64decode(encoded).decode("utf-8").strip())


def endpoint_of(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    if parts.scheme not in {"https", "http"} or not parts.hostname:
        raise RegionalFixtureError("the control-plane URL is not an http(s) URL")
    return parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)


def agent_port(agent: Mapping[str, Any] | None) -> int:
    """The port of the Agent endpoint the control plane dispatches to."""

    if not agent or not agent.get("endpoint"):
        raise VariantNotRun(
            "FIXTURE_NOT_READY", "the node's Agent record names no endpoint"
        )
    parts = urlsplit(str(agent["endpoint"]))
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if not 1 <= port <= 65535:
        raise VariantNotRun("FIXTURE_NOT_READY", "the Agent endpoint port is invalid")
    return port


def block_budget(ctx: VariantContext, maximum: int) -> int:
    """How long a block's deadman may run: the ceiling, inside the window."""

    remaining = int(ctx.remaining_seconds()) - MIN_BLOCK_SECONDS
    return max(MIN_BLOCK_SECONDS, min(int(maximum), remaining))


def isolation_probe(ctx: VariantContext, node: str) -> ProbeHandle:
    """The isolation probe Pod on ``node`` (created lazily, torn down with the
    other probes)."""

    key = f"isolation@{node}"
    handle = ctx.probes.get(key)
    if handle is None:
        handle = ProbeHandle(
            ctx.regional,
            script=ISOLATION_PROBE,
            node=node,
            image=ctx.settings.host_probe_image,
            key=ctx.spec.key,
            run_id=ctx.run_id,
            case_dir=ctx.case_dir,
            deadline_seconds=ctx.spec.max_minutes * 60 + 1200,
        )
        ctx.probes[key] = handle
    return handle


# --------------------------------------------------------------------------- #
# the block
# --------------------------------------------------------------------------- #
class HostNetworkIsolation:
    """One tagged block on one node, with its deadman and its LIFO undo.

    ``block`` registers ``restore`` on the context's cleanup stack before the
    probe is asked for anything, so a block the probe half-established is
    still undone; ``restore`` runs the probe's cleanup and refuses to call the
    node clean unless no tagged rule and no armed timer remain (and, for the
    Agent block, the port accepts connections again).
    """

    def __init__(
        self,
        ctx: VariantContext,
        *,
        node: str,
        kind: str,
        deadman_seconds: int,
        pod_ip: str = "",
        host: str = "",
        port: int | None = None,
        cidrs: Sequence[str] = (),
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown isolation kind {kind!r}")
        if kind == POD_EGRESS and not (pod_ip and port and (host or cidrs)):
            raise ValueError(
                "a pod-egress block needs the Pod IP, the port and a host or CIDRs"
            )
        self.ctx = ctx
        self.node = node
        self.kind = kind
        self.deadman_seconds = int(deadman_seconds)
        self.pod_ip = pod_ip
        self.host = host
        self.port = port
        self.cidrs = tuple(cidrs)
        self.cleanup_name = f"isolation-{kind}@{node}"
        self.blocked: dict[str, Any] | None = None
        self.blocked_at: datetime | None = None
        self.restored: dict[str, Any] | None = None
        self.restored_at: datetime | None = None

    def _call(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        return isolation_probe(self.ctx, self.node).execute(
            *arguments, "--run-id", self.ctx.run_id, timeout=timeout
        )

    def survey(self) -> dict[str, Any]:
        value = self._call("survey", "--unit", AGENT_UNIT)
        self.ctx.evidence.write(f"isolation-{self.kind}-survey", value)
        return value

    def block(self) -> dict[str, Any]:
        self.ctx.cleanup.push(self.cleanup_name, self.restore)
        arguments = [
            "block",
            "--kind",
            self.kind,
            "--deadman-seconds",
            str(self.deadman_seconds),
        ]
        if self.kind == POD_EGRESS:
            arguments += ["--pod-ip", self.pod_ip, "--port", str(self.port)]
            if self.host:
                arguments += ["--host", self.host]
            for cidr in self.cidrs:
                arguments += ["--cidr", cidr]
        else:
            arguments += ["--unit", AGENT_UNIT]
            if self.port:
                arguments += ["--port", str(self.port)]
        value = self._call(*arguments)
        self.ctx.evidence.write(f"isolation-{self.kind}-block", value)
        errors = block_errors(value, self.kind)
        if errors:
            raise RegionalFixtureError(
                "the isolation block was not established: " + "; ".join(errors)
            )
        self.blocked = value
        self.blocked_at = self.ctx.clock()
        return value

    def status(self) -> dict[str, Any]:
        value = self._call("status", "--kind", self.kind)
        self.ctx.evidence.write(f"isolation-{self.kind}-status", value)
        return value

    def restore(self) -> dict[str, Any]:
        value = self._call("cleanup", "--kind", self.kind)
        self.ctx.evidence.write(f"isolation-{self.kind}-cleanup", value)
        errors = cleanup_errors(value, self.kind)
        if errors:
            raise RegionalFixtureError(
                "the isolation cleanup did not prove the node clean: "
                + "; ".join(errors)
            )
        self.restored = value
        self.restored_at = self.ctx.clock()
        return value


def block_errors(document: Mapping[str, Any], kind: str) -> list[str]:
    """Why the probe's block document does not show an armed, effective block."""

    errors = []
    if not document.get("rules"):
        errors.append("no tagged rule was inserted")
    timer = (document.get("deadman") or {}).get("timer") or {}
    if timer.get("ActiveState") != "active":
        errors.append(f"the deadman timer is not active: {timer}")
    if kind == AGENT_INGRESS and document.get("connectivity") is not False:
        errors.append(
            "the Node Agent port still accepts connections "
            f"(connectivity={document.get('connectivity')!r})"
        )
    return errors


def cleanup_errors(document: Mapping[str, Any], kind: str) -> list[str]:
    """Why the probe's cleanup document does not prove the node clean."""

    errors = []
    item = (document.get("kinds") or {}).get(kind) or {}
    if document.get("clean") is not True:
        errors.append("the probe does not report clean")
    if item.get("remaining"):
        errors.append(f"tagged rules remain: {item['remaining']}")
    if (item.get("timer") or {}).get("ActiveState") == "active":
        errors.append("the deadman timer is still armed")
    if kind == AGENT_INGRESS and item.get("connectivity") is not True:
        errors.append(
            "the Node Agent port does not accept connections after the restore "
            f"(connectivity={item.get('connectivity')!r})"
        )
    return errors


# --------------------------------------------------------------------------- #
# 028 api_unreachable graders (pure)
# --------------------------------------------------------------------------- #
def _observed_at(heartbeat: Mapping[str, Any] | None) -> datetime | None:
    return parse_time((heartbeat or {}).get("observed_at"))


def ineffective_block_reasons(
    during: Mapping[str, Any],
    attempt_entry: Mapping[str, Any],
    *,
    blocked_at: datetime,
) -> list[str]:
    """Why the block did not cut the watcher off (the effectiveness oracle).

    The watcher's coverage heartbeat must not advance past the block, and the
    attempt's terminal/decision must not reach the control plane while the
    block holds. Either is INCONCLUSIVE, never a product verdict.
    """

    reasons = []
    observed = _observed_at(during.get("heartbeat"))
    if observed is not None and observed >= blocked_at:
        reasons.append(
            "the watcher's coverage heartbeat still reached the control plane at "
            f"{observed.isoformat()} during the block"
        )
    if attempt_entry.get("event") or attempt_entry.get("decision"):
        reasons.append(
            "the attempt's terminal reached the control plane during the block"
        )
    return reasons


def emergency_stop_errors(
    workload: Mapping[str, Any], *, incident_id: str | None = None
) -> list[str]:
    """The watcher's emergency fallback suspended the workload itself.

    It annotates the workload with ``passive-stop-mode: emergency-fallback``
    and the containment incident id; both are the product's own records.
    """

    annotations = (workload.get("metadata") or {}).get("annotations") or {}
    errors = []
    if annotations.get(EMERGENCY_MODE_ANNOTATION) != EMERGENCY_MODE:
        errors.append(
            "the workload carries no emergency-fallback stop annotation "
            f"({EMERGENCY_MODE_ANNOTATION}={annotations.get(EMERGENCY_MODE_ANNOTATION)!r})"
        )
    initiator = annotations.get(TERMINATION_ANNOTATION)
    if not initiator:
        errors.append("the workload names no termination-initiator incident")
    elif incident_id is not None and initiator != incident_id:
        errors.append(
            f"the workload's termination initiator {initiator} is not the "
            f"attempt's containment incident {incident_id}"
        )
    return errors


def _succeeded(workflow: Mapping[str, Any], operation: str) -> int:
    return sum(
        1
        for item in executions(dict(workflow), operation)
        if item.get("status") == "SUCCEEDED"
    )


def single_action_errors(
    entry: Mapping[str, Any], records: Mapping[str, Any]
) -> list[str]:
    """Emergency stop then retry produced every action exactly once.

    One decision, one containment STOP_WORKLOADS that succeeded, at most one
    recovery workflow with at most one successful RESTART_WORKLOAD, and a
    ledger whose count equals its reservations and never exceeds one.
    """

    errors = []
    if not entry.get("decision"):
        errors.append("no completion decision was recorded after the restore")
    stops = sum(
        _succeeded(workflow, "STOP_WORKLOADS")
        for workflow in entry.get("containment_workflows") or []
    )
    if stops == 0:
        errors.append("the containment workflow's STOP_WORKLOADS never succeeded")
    elif stops > 1:
        errors.append(f"STOP_WORKLOADS succeeded {stops} times for one attempt")
    recovery = entry.get("recovery") or []
    if len(recovery) > 1:
        errors.append(f"{len(recovery)} recovery workflows for one attempt")
    restarts = sum(
        _succeeded(item.get("workflow") or {}, "RESTART_WORKLOAD") for item in recovery
    )
    if restarts > 1:
        errors.append(f"RESTART_WORKLOAD succeeded {restarts} times for one attempt")
    budget = records.get("restart_budget")
    if isinstance(budget, dict):
        count = int(budget.get("restart_count", 0))
        reservations = list(budget.get("reservation_ids") or [])
        if count > 1:
            errors.append(f"restart_count {count} exceeds the one budgeted restart")
        if count != len(reservations):
            errors.append("restart_count does not match the reservation ids")
    return errors


def restored_heartbeat_errors(
    after: Mapping[str, Any], *, restored_at: datetime
) -> list[str]:
    """The watcher reached the control plane again: a heartbeat after the restore."""

    observed = _observed_at(after.get("heartbeat"))
    if observed is None:
        return ["no watcher coverage heartbeat is recorded after the restore"]
    if observed < restored_at:
        return [
            f"the watcher's last coverage heartbeat ({observed.isoformat()}) predates "
            f"the restore ({restored_at.isoformat()})"
        ]
    return []


def duplicate_notification_errors(
    notifications: Sequence[Mapping[str, Any]],
) -> list[str]:
    """No notification of the attempt's incidents was recorded twice."""

    by_key: dict[str, int] = {}
    by_pair: dict[tuple[str, str], int] = {}
    for item in notifications:
        key = str(item.get("deduplication_key") or "")
        if key:
            by_key[key] = by_key.get(key, 0) + 1
        pair = (str(item.get("incident_id") or ""), str(item.get("category") or ""))
        by_pair[pair] = by_pair.get(pair, 0) + 1
    errors = [
        f"notification {key} was recorded {count} times"
        for key, count in sorted(by_key.items())
        if count > 1
    ]
    errors.extend(
        f"{count} {category} notifications for incident {incident}"
        for (incident, category), count in sorted(by_pair.items())
        if count > 1
    )
    return errors


def attempt_incident_ids(entry: Mapping[str, Any]) -> list[str]:
    """The containment and recovery incidents of one attempt's records."""

    ids = []
    containment = entry.get("containment_incident") or {}
    if containment.get("incident_id"):
        ids.append(str(containment["incident_id"]))
    for item in entry.get("recovery") or []:
        incident_id = (item.get("incident") or {}).get("incident_id")
        if incident_id and str(incident_id) not in ids:
            ids.append(str(incident_id))
    return ids


def observed_running(
    records: Mapping[str, Any], attempt_id: str
) -> dict[str, Any] | None:
    """The watcher's RUNNING observation of ``attempt_id``, if published."""

    for item in records.get("observations") or []:
        if (
            item.get("attempt_id") == attempt_id
            and item.get("workload_phase") == "RUNNING"
        ):
            return dict(item)
    return None


# --------------------------------------------------------------------------- #
# 011 unreachable_node graders (pure)
# --------------------------------------------------------------------------- #
def _evidence_executions(workflow: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for operation in HUNG_EVIDENCE_OPERATIONS
        for item in executions(dict(workflow), operation)
    ]


def _names_gap(details: Mapping[str, Any], node: str) -> bool:
    decision = details.get("hung_triage_decision") or {}
    survey = details.get("hung_survey") or {}
    coverage = survey.get("survey_coverage") or details.get("survey_coverage") or {}
    short = (
        isinstance(coverage.get("responding_node_count"), int)
        and isinstance(coverage.get("expected_node_count"), int)
        and coverage["responding_node_count"] < coverage["expected_node_count"]
    )
    return (
        node in (details.get("undetermined_nodes") or [])
        or node in (details.get("node_failures") or {})
        or node in (decision.get("undetermined_nodes") or [])
        or node in (decision.get("not_sampled_nodes") or [])
        or (short and node not in (details.get("node_results") or {}))
    )


def evidence_recorded(workflow: Mapping[str, Any]) -> bool:
    """The hung chain's evidence steps are done: a later step has an
    execution, the workflow is terminal, or every triage step has ended."""

    if not workflow:
        return False
    if str(workflow.get("status")) in TERMINAL_WORKFLOWS:
        return True
    if any(executions(dict(workflow), op) for op in HUNG_LATER_OPERATIONS):
        return True
    triage_indexes = [
        index
        for index, step in enumerate(workflow.get("official_steps") or [])
        if step.get("operation") == "COLLECT_HUNG_TRIAGE"
    ]
    if not triage_indexes:
        return False
    ended = {
        item.get("step_index")
        for item in executions(dict(workflow), "COLLECT_HUNG_TRIAGE")
        if item.get("status") in {"SUCCEEDED", "FAILED"}
    }
    return all(index in ended for index in triage_indexes)


def coverage_gap_errors(
    workflow: Mapping[str, Any], *, node: str, peer: str
) -> list[str]:
    """The product recorded ``node`` as a coverage gap and kept ``peer``'s evidence.

    At least one summary/triage execution names the unreachable node as
    undetermined, failed, not sampled or not responding; none carries a
    reading for it (a result row for a node whose Agent was unreachable would
    be a fabricated zero); the reachable peer's results are present.
    """

    evidence = _evidence_executions(workflow)
    if not evidence:
        return ["no COLLECT_HUNG_SUMMARY/COLLECT_HUNG_TRIAGE execution was recorded"]
    errors = []
    details = [dict(item.get("details") or {}) for item in evidence]
    if not any(_names_gap(item, node) for item in details):
        errors.append(
            f"no summary/triage record names {node} as undetermined, failed or not "
            "responding (the coverage gap is not recorded)"
        )
    fabricated = [
        item.get("operation")
        for item, detail in zip(evidence, details)
        if node in (detail.get("node_results") or {})
    ]
    if fabricated:
        errors.append(
            f"{node} has a result in {sorted(set(map(str, fabricated)))} although its "
            "Agent was unreachable (a fabricated reading)"
        )
    if not any(peer in (item.get("node_results") or {}) for item in details):
        errors.append(
            f"no summary/triage evidence was collected from the reachable {peer}"
        )
    return errors


def no_isolation_errors(
    workflow: Mapping[str, Any], node: Mapping[str, Any]
) -> list[str]:
    """Nothing was cordoned, stopped, reset or restarted on the suspicion."""

    acted = sorted(
        {
            str(item.get("operation"))
            for item in workflow.get("step_executions") or []
            if item.get("operation") in FORBIDDEN_OPERATIONS
        }
    )
    errors = []
    if acted:
        errors.append(f"the hung chain executed forbidden {acted}")
    if node_isolated(dict(node)):
        errors.append(f"{node.get('name')} was isolated on a hung suspicion")
    return errors


def channel_restored_errors(
    cleanup: Mapping[str, Any],
    agent_before: Mapping[str, Any] | None,
    agent_after: Mapping[str, Any] | None,
    *,
    restored_at: datetime,
) -> list[str]:
    """The Agent channel is back: the port accepts, the Agent still heartbeats."""

    errors = cleanup_errors(cleanup, AGENT_INGRESS)
    if not agent_after:
        return [*errors, "the node's Agent record is gone after the restore"]
    if agent_after.get("lifecycle_state") != "ACTIVE":
        errors.append(
            f"the Agent is {agent_after.get('lifecycle_state')} after the restore"
        )
    seen = parse_time(agent_after.get("last_seen_at"))
    if seen is None or seen < restored_at:
        errors.append(
            "the Agent's heartbeat did not advance after the restore "
            f"(last_seen_at={agent_after.get('last_seen_at')})"
        )
    before = parse_time((agent_before or {}).get("last_seen_at"))
    if before is not None and seen is not None and seen <= before:
        errors.append("the Agent's last_seen_at did not move during the variant")
    return errors


def later_step_statuses(workflow: Mapping[str, Any]) -> dict[str, str | None]:
    """The final status of each post-evidence step (recorded, not graded)."""

    result: dict[str, str | None] = {}
    for operation in HUNG_LATER_OPERATIONS:
        runs = executions(dict(workflow), operation)
        result[operation] = str(runs[-1].get("status")) if runs else None
    return result


__all__ = [
    "AGENT_BLOCK_SECONDS",
    "AGENT_INGRESS",
    "AGENT_UNIT",
    "EMERGENCY_MODE",
    "EMERGENCY_MODE_ANNOTATION",
    "FORBIDDEN_OPERATIONS",
    "HostNetworkIsolation",
    "ISOLATION_PROBE",
    "ISOLATION_RECORDS_PROBE",
    "KINDS",
    "POD_EGRESS",
    "TERMINATION_ANNOTATION",
    "WATCHER_APP",
    "WATCHER_BLOCK_SECONDS",
    "agent_port",
    "attempt_incident_ids",
    "block_budget",
    "block_errors",
    "channel_restored_errors",
    "cleanup_errors",
    "control_plane_endpoint",
    "coverage_gap_errors",
    "duplicate_notification_errors",
    "emergency_stop_errors",
    "endpoint_of",
    "evidence_recorded",
    "ineffective_block_reasons",
    "isolation_probe",
    "isolation_records",
    "later_step_statuses",
    "no_isolation_errors",
    "observed_running",
    "restored_heartbeat_errors",
    "single_action_errors",
    "watcher_pod",
]
