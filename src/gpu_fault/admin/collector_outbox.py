"""``gpu-fault-admin collector-outbox``: a collector's dead letters, from the
deploy host (ARCH-G2).

``gpu-fault-collector outbox --collector <c> {stats,list,requeue-dead --yes}``
inspects or requeues the NDJSON outbox a node collector writes, and until this
verb existed it was the *only* entry: an administrator had to log on to the
GPU node. This verb carries the same command to the node the way everything
else reaches a node agent -- as the single NODE_ACTION step of a workflow:

1. validate the arguments on the deploy host (known collector and action,
   ``requeue-dead`` needs ``--yes``, every action needs ``--reference`` for the
   audit trail);
2. run ``COLLECTOR_OUTBOX_SCRIPT`` in the CPU ingress Pod over the exec
   channel every other verb uses: it refuses unless the node's agent heartbeat
   advertises ``COLLECTOR_OUTBOX_MAINTENANCE`` (an older agent predates the
   operation), builds the operator incident and workflow with
   ``gpu_fault.orchestration.collector_outbox_maintenance``, writes both in
   one transaction and wakes the dispatcher;
3. poll the workflow to a terminal status and print the node's answer --
   ``stats``, the metadata rows of ``list``, the counts of ``requeue-dead`` --
   or the refusal the agent recorded (the outbox lock's recorded holder, an
   unreadable file, an agent that rejected the command);
4. archive the document under ``<state-dir>/collector-outbox/<node>/``.

Safety properties are the node-local CLI's without ``--force``: metadata only
ever leaves the node (no payload bytes), the lock is taken strictly and a held
lock is reported with its recorded holder. There is deliberately no remote
``--force``: from here nobody can see whether the collector is stopped, so a
stopped collector's leftover lock stays a node-local operation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, cast

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError, safe_name
from gpu_fault.admin.operator_identity import resolve_operator_identity
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.workflow_reconcile import run_control_plane_script
from gpu_fault.collectors.outbox_maintenance import (
    ACTION_LIST,
    ACTION_REQUEUE_DEAD,
    ACTION_STATS,
    OUTBOX_ACTIONS,
    OUTBOX_COLLECTORS,
    OutboxMaintenanceRequest,
)

STATE_ROOT = "collector-outbox"
OPERATION = "COLLECTOR_OUTBOX_MAINTENANCE"
DEFAULT_WAIT_SECONDS = 300
POLL_INTERVAL_SECONDS = 2.0
TERMINAL_WORKFLOW_STATUSES = frozenset({"SUCCEEDED", "FAILED", "SUPERSEDED", "BLOCKED"})
#: What the node-action transport records when the agent could not even parse
#: the command (an agent whose ``WorkflowOperation`` predates the operation
#: answers 422 to the envelope) or refused it as not allow-listed (403).
PREDATES_HTTP_STATUSES = frozenset({403, 422})
PREDATES_ERROR_CODES = frozenset({"HTTP_REJECTION", "OPERATION_NOT_ALLOWED"})
NODE_LOCAL_COMMAND = "gpu-fault-collector outbox --collector {collector} {action}"

COLLECTOR_OUTBOX_SCRIPT = """
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

payload = json.load(sys.stdin)
context = ApplicationContext.from_environment()
store = context.store
OPERATION = "COLLECTOR_OUTBOX_MAINTENANCE"


def dump(model):
    return None if model is None else json.loads(model.model_dump_json())


if payload["mode"] == "submit":
    try:
        from gpu_fault.orchestration.incident_closure import (
            build_collector_outbox_workflow,
        )
    except ImportError:
        raise SystemExit(
            "collector-outbox: this control plane predates collector outbox "
            "maintenance; deploy the current release first, or run the node-local "
            "gpu-fault-collector outbox command on the node"
        )
    cluster_id = payload["cluster_id"]
    node_id = payload["node_id"]
    try:
        agent = store.get_agent(cluster_id, node_id)
    except NotFoundError:
        raise SystemExit(
            f"collector-outbox: node {node_id} has no agent heartbeat in cluster "
            f"{cluster_id}; check the node name and gpu-fault-admin status --full"
        )
    allowed = {str(getattr(item, "value", item)) for item in agent.allowed_operations}
    if OPERATION not in allowed:
        raise SystemExit(
            f"collector-outbox: the node agent on {node_id} (version "
            f"{agent.agent_version}) predates outbox maintenance or does not "
            f"allow-list {OPERATION}; roll the node to the current release or use "
            "the node-local gpu-fault-collector outbox command on the node"
        )
    incident, workflow = build_collector_outbox_workflow(
        cluster_id,
        node_id,
        collector=payload["collector"],
        action=payload["action"],
        confirm=bool(payload.get("confirm")),
        path=payload.get("path"),
        operator=payload["operator"],
        reference=payload.get("reference"),
        now=datetime.now(timezone.utc),
        runtime_profile_version=payload.get("runtime_profile_version"),
    )
    store.save_incident_and_workflow(incident, workflow)
    context.dispatcher.wake()
    result = {
        "incident": dump(incident),
        "workflow": dump(workflow),
        "agent_version": agent.agent_version,
    }
elif payload["mode"] == "status":
    workflow = store.get_workflow(payload["workflow_request_id"])
    incident = store.get_incident(workflow.incident_id)
    result = {"incident": dump(incident), "workflow": dump(workflow)}
else:
    raise ValueError("unsupported collector-outbox mode")
print(json.dumps(result, sort_keys=True))
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CollectorOutboxRequest:
    """One validated command line; refuses before anything leaves the host."""

    site: RenderedSite
    cluster_id: str
    node_id: str
    collector: str
    action: str
    reference: str
    confirm: bool = False
    path: str | None = None
    wait_seconds: int = DEFAULT_WAIT_SECONDS

    def __post_init__(self) -> None:
        if not self.cluster_id.strip():
            raise BootstrapError("collector-outbox requires --cluster-id")
        if not self.node_id.strip():
            raise BootstrapError("collector-outbox requires --node")
        if not self.reference.strip():
            raise BootstrapError(
                "collector-outbox requires --reference: every remote outbox command "
                "is recorded against a change or ticket"
            )
        if self.action == ACTION_REQUEUE_DEAD and not self.confirm:
            raise BootstrapError(
                f"{ACTION_REQUEUE_DEAD} replays records the control plane already "
                "rejected, and the collector retries them on every restart until "
                "they dead-letter again; pass --yes to confirm"
            )
        try:
            OutboxMaintenanceRequest(
                collector=self.collector,
                action=self.action,
                confirm=self.confirm,
                path=self.path,
            )
        except ValueError as exc:
            raise BootstrapError(f"collector-outbox: {exc}") from exc
        if self.wait_seconds < 1:
            raise BootstrapError("--wait-seconds must be at least 1")
        clusters = {
            str(item.get("cluster_id"))
            for item in self.site.release_config.get("clusters") or []
        }
        if self.cluster_id not in clusters:
            raise BootstrapError(
                f"cluster {self.cluster_id} is not in the managed site; managed: "
                + (", ".join(sorted(clusters)) or "none")
            )


@dataclass
class CollectorOutboxResult:
    """What the verb prints and archives."""

    request: CollectorOutboxRequest
    operator: str
    incident_id: str
    workflow_request_id: str
    workflow_status: str
    incident_state: str | None
    agent_version: str | None = None
    node_result: dict[str, Any] | None = None
    error: str | None = None
    step_details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    @property
    def succeeded(self) -> bool:
        return self.workflow_status == "SUCCEEDED"

    def as_dict(self) -> dict[str, Any]:
        request = self.request
        return {
            "cluster_id": request.cluster_id,
            "node_id": request.node_id,
            "collector": request.collector,
            "action": request.action,
            "path": request.path,
            "reference": request.reference,
            "operator": self.operator,
            "incident_id": self.incident_id,
            "workflow_request_id": self.workflow_request_id,
            "workflow_status": self.workflow_status,
            "incident_state": self.incident_state,
            "agent_version": self.agent_version,
            "result": self.node_result,
            "error": self.error,
            "message": self.message,
        }


# --------------------------------------------------------------------------
# Control-plane round trips


def submit_maintenance(
    site: RenderedSite, request: CollectorOutboxRequest, *, operator: str
) -> dict[str, Any]:
    return run_control_plane_script(
        site,
        {
            "mode": "submit",
            "cluster_id": request.cluster_id,
            "node_id": request.node_id,
            "collector": request.collector,
            "action": request.action,
            "confirm": request.confirm,
            "path": request.path,
            "operator": operator,
            "reference": request.reference,
            "runtime_profile_version": str(
                (site.release_config.get("runtime_profile") or {}).get("version") or ""
            )
            or None,
        },
        script=COLLECTOR_OUTBOX_SCRIPT,
    )


def read_workflow(site: RenderedSite, workflow_request_id: str) -> dict[str, Any]:
    return run_control_plane_script(
        site,
        {"mode": "status", "workflow_request_id": workflow_request_id},
        script=COLLECTOR_OUTBOX_SCRIPT,
    )


def wait_for_terminal(
    site: RenderedSite,
    workflow_request_id: str,
    *,
    wait_seconds: int,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll the workflow until it is terminal or ``wait_seconds`` pass.

    Returns the last ``{"incident", "workflow"}`` document either way; the
    caller reads ``workflow.status`` to tell the two apart.
    """

    deadline = clock() + wait_seconds
    while True:
        document = read_workflow(site, workflow_request_id)
        workflow = cast(dict[str, Any], document.get("workflow") or {})
        if str(workflow.get("status")) in TERMINAL_WORKFLOW_STATUSES:
            return document
        if clock() >= deadline:
            return document
        sleep(POLL_INTERVAL_SECONDS)


# --------------------------------------------------------------------------
# Reading the node's answer out of the workflow record


def _last_execution(workflow: Mapping[str, Any]) -> dict[str, Any] | None:
    executions = [
        dict(item)
        for item in workflow.get("step_executions") or []
        if item.get("operation") == OPERATION
    ]
    return executions[-1] if executions else None


def agent_predates_operation(details: Mapping[str, Any]) -> bool:
    """Whether the step failed because the agent never knew the operation.

    The transport records the agent's HTTP answer on the step: a 422 means
    the envelope did not parse (the operation is not in that agent's enum), a
    403 ``OPERATION_NOT_ALLOWED`` that the agent knows it but its allowlist
    does not name it. Both read as "roll the node or use the node-local CLI".
    """

    status = details.get("http_status")
    code = details.get("node_action_error_code")
    return (isinstance(status, int) and status in PREDATES_HTTP_STATUSES) or (
        isinstance(code, str) and code in PREDATES_ERROR_CODES
    )


def _describe_result(request: CollectorOutboxRequest, result: Mapping[str, Any]) -> str:
    if request.action == ACTION_STATS:
        stats = cast(dict[str, Any], result.get("stats") or {})
        holder = result.get("lock_holder")
        return (
            f"{request.collector} outbox on {request.node_id}: depth "
            f"{stats.get('depth')}, replayable {stats.get('replayable')}, dead "
            f"{stats.get('dead')} (payload truncated "
            f"{stats.get('payload_truncated', 0)}), oldest failure "
            f"{stats.get('oldest_failed_at')}"
            + (f"; lock held: {holder}" if holder else "")
        )
    if request.action == ACTION_LIST:
        rows = cast(list[dict[str, Any]], result.get("records") or [])
        dead = sum(1 for row in rows if not row.get("replayable"))
        return (
            f"{request.collector} outbox on {request.node_id}: {len(rows)} "
            f"record(s) ({dead} dead); metadata only, payloads never leave the node"
        )
    return (
        f"requeued {result.get('requeued')} dead record(s) in the "
        f"{request.collector} outbox on {request.node_id}; "
        f"{result.get('skipped_payload_truncated', 0)} left dead because only a "
        f"payload digest was kept; dead {result.get('dead_before')} -> "
        f"{result.get('dead_after')}"
    )


def interpret(
    request: CollectorOutboxRequest,
    *,
    operator: str,
    submission: Mapping[str, Any],
    document: Mapping[str, Any],
) -> CollectorOutboxResult:
    """Turn the terminal (or timed-out) workflow record into the verb's answer."""

    workflow = cast(dict[str, Any], document.get("workflow") or {})
    incident = cast(dict[str, Any], document.get("incident") or {})
    result = CollectorOutboxResult(
        request=request,
        operator=operator,
        incident_id=str(incident.get("incident_id") or workflow.get("incident_id")),
        workflow_request_id=str(workflow.get("request_id")),
        workflow_status=str(workflow.get("status")),
        incident_state=(
            str(incident.get("state")) if incident.get("state") is not None else None
        ),
        agent_version=(
            str(submission.get("agent_version"))
            if submission.get("agent_version")
            else None
        ),
    )
    execution = _last_execution(workflow) or {}
    details = cast(dict[str, Any], execution.get("details") or {})
    result.step_details = details
    node_results = cast(dict[str, Any], details.get("node_results") or {})
    node_result = node_results.get(request.node_id)
    if result.workflow_status not in TERMINAL_WORKFLOW_STATUSES:
        result.error = (
            f"workflow {result.workflow_request_id} is still "
            f"{result.workflow_status} after {request.wait_seconds}s"
        )
        result.message = (
            f"{result.error}; the node has not answered yet -- rerun with a longer "
            f"--wait-seconds or read GET /v1/workflows/{result.workflow_request_id}"
        )
        return result
    if result.succeeded and isinstance(node_result, dict):
        result.node_result = node_result
        result.message = _describe_result(request, node_result)
        return result
    error = str(execution.get("error") or workflow.get("error") or "")
    result.error = error or f"workflow ended {result.workflow_status} without a result"
    node_local = NODE_LOCAL_COMMAND.format(
        collector=request.collector, action=request.action
    ) + (" --yes" if request.action == ACTION_REQUEUE_DEAD else "")
    if agent_predates_operation(details):
        result.message = (
            f"the node agent on {request.node_id} predates outbox maintenance "
            f"(HTTP {details.get('http_status')} "
            f"{details.get('node_action_error_code')}); roll the node to the "
            f"current release, or run the node-local `{node_local}` on the node"
        )
    elif details.get("lock_unavailable"):
        result.message = (
            f"the {request.collector} outbox lock on {request.node_id} is held "
            f"({details.get('lock_holder')}); a live collector is buffering or "
            "replaying -- retry shortly. Only if the collector is stopped and left "
            f"its lock behind, run `{node_local} --force` on the node; there is no "
            "remote --force"
        )
    else:
        result.message = (
            f"{request.action} failed on {request.node_id}: {result.error}; "
            f"incident {result.incident_id} is {result.incident_state}"
        )
    return result


# --------------------------------------------------------------------------
# Archive


def evidence_directory(site: RenderedSite, node_id: str) -> Path:
    return site.source.parent / STATE_ROOT / safe_name(node_id)


def record_result(
    site: RenderedSite, result: CollectorOutboxResult, *, now: datetime
) -> Path:
    directory = evidence_directory(site, result.request.node_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{result.request.action}-{now:%Y%m%dT%H%M%SZ}.json"
    write_json_atomic(
        path,
        {
            "status": "SUCCEEDED" if result.succeeded else "FAILED",
            "recorded_at": now.isoformat(),
            "step_details": result.step_details,
            **result.as_dict(),
        },
    )
    return path


# --------------------------------------------------------------------------
# The verb


def run_collector_outbox(
    request: CollectorOutboxRequest,
    *,
    now: Callable[[], datetime] = _utc_now,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> CollectorOutboxResult:
    """Submit, wait for the terminal state, interpret, archive."""

    site = request.site
    stamp = now()
    operator = resolve_operator_identity()
    submission = submit_maintenance(site, request, operator=operator)
    workflow = cast(dict[str, Any], submission.get("workflow") or {})
    workflow_request_id = str(workflow.get("request_id") or "")
    if not workflow_request_id:
        raise BootstrapError("collector-outbox: the control plane returned no workflow")
    document = wait_for_terminal(
        site,
        workflow_request_id,
        wait_seconds=request.wait_seconds,
        sleep=sleep,
        clock=clock,
    )
    if not document.get("workflow"):
        document = submission
    result = interpret(
        request, operator=operator, submission=submission, document=document
    )
    record_result(site, result, now=stamp)
    return result


def add_collector_outbox_command(
    commands: Any, add_managed_site_arguments: Callable[[Any], None]
) -> None:
    """Register ``gpu-fault-admin collector-outbox``; the CLI passes site options."""

    command = commands.add_parser(
        "collector-outbox",
        usage=(
            "gpu-fault-admin collector-outbox --state-dir STATE_DIR "
            "--cluster-id CLUSTER_ID --node NODE --collector "
            "{" + ",".join(OUTBOX_COLLECTORS) + "} --action "
            "{" + ",".join(OUTBOX_ACTIONS) + "} [--path PATH] [--yes] "
            "--reference CHANGE_ID [--wait-seconds N]"
        ),
        help=(
            "inspect or requeue a node collector's dead-letter outbox from here: "
            "the node-local `gpu-fault-collector outbox` command carried to the "
            "node as a NODE_ACTION step (metadata only; requeue-dead needs --yes; "
            "no remote --force)"
        ),
    )
    add_managed_site_arguments(command)
    command.add_argument(
        "--cluster-id", required=True, metavar="CLUSTER_ID", help="managed GPU cluster"
    )
    command.add_argument(
        "--node", required=True, metavar="NODE", help="Kubernetes node name"
    )
    command.add_argument(
        "--collector",
        required=True,
        choices=OUTBOX_COLLECTORS,
        help="the collector whose outbox to read (/var/lib/gpu-fault/outbox/<c>.ndjson)",
    )
    command.add_argument(
        "--action",
        required=True,
        choices=OUTBOX_ACTIONS,
        help=(
            "stats: depth, replayable/dead split, oldest failure, lock holder; "
            "list: one metadata row per record, never a payload; requeue-dead: "
            "mark dead records replayable again (payload-truncated ones stay dead)"
        ),
    )
    command.add_argument(
        "--path",
        default=None,
        metavar="PATH",
        help="only list or requeue records for this control-plane path (/v1/...)",
    )
    command.add_argument(
        "--yes",
        action="store_true",
        help="required by requeue-dead: confirm the dead records should be replayed",
    )
    command.add_argument(
        "--reference",
        required=True,
        metavar="CHANGE_ID",
        help="change ticket recorded on the incident, the requeued records and the evidence file",
    )
    command.add_argument(
        "--wait-seconds",
        type=int,
        default=DEFAULT_WAIT_SECONDS,
        metavar="N",
        help=f"how long to wait for the node's answer (default {DEFAULT_WAIT_SECONDS})",
    )


def run_collector_outbox_command(
    arguments: argparse.Namespace, *, site: RenderedSite
) -> int:
    request = CollectorOutboxRequest(
        site=site,
        cluster_id=str(arguments.cluster_id),
        node_id=str(arguments.node),
        collector=str(arguments.collector),
        action=str(arguments.action),
        reference=str(arguments.reference or ""),
        confirm=bool(arguments.yes),
        path=cast(str | None, arguments.path),
        wait_seconds=int(arguments.wait_seconds),
    )
    result = run_collector_outbox(request)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.succeeded else 1
