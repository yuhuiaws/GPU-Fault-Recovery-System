"""Read-only control-plane reads of the GF-GENERAL-LIVE runner.

Verdicts read only the product's own records -- accepted host/GPU batches,
health-signal states, latest metrics, incidents, markers, workflows with their
step executions, remote commands, notifications, completion decisions and the
restart-budget ledger -- through ``RegionalLiveFixture.cpu_python`` (the
installed control-plane interpreter inside an API Pod). Nothing here writes to
the store; the runner never posts a prefabricated finding or workflow.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Protocol


class CpuPython(Protocol):
    def cpu_python(
        self, script: str, *arguments: str, **options: Any
    ) -> dict[str, Any]: ...


#: What a variant may start from: no open incident or workflow on the node,
#: a current Agent, fresh collector channels.
PREFLIGHT_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.store import NotFoundError

cluster_id, node_id = sys.argv[1:]
store = ApplicationContext.from_environment().store
try:
    agent = store.get_agent(cluster_id, node_id).model_dump(mode="json")
except NotFoundError:
    agent = None
try:
    registration = store.get_regional_cluster(cluster_id).model_dump(mode="json")
except NotFoundError:
    registration = None
open_states = [
    IncidentState.DETECTED,
    IncidentState.ACTION_PENDING,
    IncidentState.SAFETY_PENDING,
    IncidentState.ESCALATED,
    IncidentState.QUARANTINED,
]
incidents = [
    {
        "incident_id": item.incident_id,
        "state": item.state.value,
        "event_type": item.event_type,
        "created_at": item.created_at.isoformat(),
        "workflow_request_id": item.workflow_request_id,
    }
    for item in store.list_incidents_by_state(
        cluster_id, open_states, node_ids={node_id}
    )
]
active = [
    {"incident_id": incident.incident_id, "request_id": workflow.request_id,
     "status": workflow.status.value}
    for incident, workflow in store.list_active_workflow_incidents(
        cluster_id, node_ids={node_id}
    )
]
statuses = [
    item.model_dump(mode="json")
    for item in store.list_collector_statuses(cluster_id, node_id)
]
print(json.dumps({
    "agent": agent,
    "registration": {
        key: (registration or {}).get(key)
        for key in (
            "cluster_id", "provider", "lifecycle_state", "enabled", "runtime_profile_version"
        )
    } if registration else None,
    "open_incidents": incidents,
    "active_workflows": active,
    "collector_statuses": statuses,
}, sort_keys=True, default=str))
"""

#: Everything the product recorded about one node since a moment. Incidents
#: are listed in every state (a diagnostic-only one is often RECOVERED before
#: the next poll), plus the incident of any workflow created since, however
#: old; the document names its own ``since``, cluster and node so a verdict
#: can tell this read's findings from such older ones
#: (``general_live_verdicts.minted_in_read``).
ACTIVITY_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import EvidenceKind

cluster_id, node_id, since_text, metrics_text, signals_text, kinds_text = sys.argv[1:7]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
metrics = {item for item in metrics_text.split(",") if item}
signals = [item for item in signals_text.split(",") if item]
kinds = [item for item in kinds_text.split(",") if item]
store = ApplicationContext.from_environment().store


def batch_view(item):
    payload = item.payload if isinstance(item.payload, dict) else {}
    samples = payload.get("samples") or []
    return {
        "record_id": item.record_id,
        "kind": str(item.kind),
        "observed_at": item.observed_at.isoformat(),
        "batch_id": payload.get("batch_id"),
        "received_at": payload.get("received_at"),
        "edge_filter_reasons": payload.get("edge_filter_reasons") or [],
        "collection_errors": payload.get("collection_errors") or [],
        "sample_count": len(samples),
        "samples": [
            {
                "name": sample.get("name"),
                "device": sample.get("device"),
                "value": sample.get("value"),
                "labels": sample.get("labels") or {},
            }
            for sample in samples
            if isinstance(sample, dict) and sample.get("name") in metrics
        ],
    }


batches = []
for kind in kinds:
    for item in store.list_raw_evidence(
        cluster_id, node_id=node_id, kind=EvidenceKind(kind), limit=2000
    ):
        if item.observed_at >= since:
            batches.append(batch_view(item))
batches.sort(key=lambda item: item["observed_at"])
incidents = {}
for incident in store.list_incidents_by_state(
    cluster_id, list(IncidentState), node_ids={node_id}
):
    if incident.created_at >= since:
        incidents[incident.incident_id] = incident
workflows = {}
for workflow in store.list_workflows(limit=500, newest_first=True):
    if workflow.created_at < since:
        continue
    if workflow.incident_id in incidents:
        workflows[workflow.request_id] = workflow
        continue
    try:
        incident = store.get_incident(workflow.incident_id)
    except NotFoundError:
        continue
    if incident.cluster_id == cluster_id and node_id in incident.node_ids:
        incidents[incident.incident_id] = incident
        workflows[workflow.request_id] = workflow
for incident in incidents.values():
    if incident.workflow_request_id and incident.workflow_request_id not in workflows:
        try:
            workflows[incident.workflow_request_id] = store.get_workflow(
                incident.workflow_request_id
            )
        except NotFoundError:
            pass
markers = []
for incident_id in incidents:
    markers.extend(
        item.model_dump(mode="json") for item in store.list_markers_for_incident(incident_id)
    )
commands = (
    [
        item.model_dump(mode="json", exclude={"lease_token"})
        for item in store.list_remote_commands(workflow_request_ids=list(workflows))
    ]
    if workflows
    else []
)
health = {}
for key in signals:
    state = store.get_health_signal_state(key)
    health[key] = state.model_dump(mode="json") if state is not None else None
latest = [
    {
        "name": item.name,
        "device": item.device,
        "value": item.value,
        "observed_at": item.observed_at.isoformat(),
        "labels": item.labels,
    }
    for item in store.list_telemetry_metrics_latest(cluster_id, node_id)
    if not metrics or item.name in metrics or item.name == "collector_observation_valid"
]
notifications = []
for notification in store.list_notifications(limit=500, newest_first=True):
    if notification.created_at < since:
        continue
    text = "\n".join(
        (notification.deduplication_key, notification.subject, notification.body_text)
    )
    if node_id not in text:
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
try:
    agent = store.get_agent(cluster_id, node_id).model_dump(mode="json")
except NotFoundError:
    agent = None
print(json.dumps({
    "since": since.isoformat(),
    "cluster_id": cluster_id,
    "node_id": node_id,
    "batches": batches,
    "incidents": [item.model_dump(mode="json") for item in incidents.values()],
    "workflows": [item.model_dump(mode="json") for item in workflows.values()],
    "markers": markers,
    "commands": commands,
    "health_signals": health,
    "metric_latest": latest,
    "collector_statuses": [
        item.model_dump(mode="json")
        for item in store.list_collector_statuses(cluster_id, node_id)
    ],
    "notifications": notifications,
    "agent": agent,
}, sort_keys=True, default=str))
"""

#: One training attempt's completion records (terminal event, decision,
#: containment incident and workflow, recovery incidents/workflows -- the
#: workflows the terminal decision opened, never the containment's own STOP --
#: the notifications of those incidents) and its job's restart ledger.
#: Read-only.
ATTEMPT_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

cluster_id, job_id, *attempt_ids = sys.argv[1:]
store = ApplicationContext.from_environment().store
attempts = {}
for attempt_id in attempt_ids:
    entry = {}
    try:
        entry["event"] = store.get_event_by_attempt(cluster_id, attempt_id).model_dump(mode="json")
    except NotFoundError:
        entry["event"] = None
    try:
        entry["decision"] = store.get_decision_by_attempt(cluster_id, attempt_id).model_dump(mode="json")
    except NotFoundError:
        entry["decision"] = None
    containment = store.get_incident_by_event(
        f"{cluster_id}/{attempt_id}/TrainingAttemptFailureDetected"
    )
    entry["containment_incident"] = (
        containment.model_dump(mode="json") if containment is not None else None
    )
    recovery = []
    incident_ids = set()
    if containment is not None:
        incident_ids.add(containment.incident_id)
    for incident, workflow in store.list_job_recovery_workflow_incidents(
        cluster_id, job_id, attempt_id, include_terminal=True
    ):
        # With terminal workflows included, the listing is every workflow of
        # the attempt's incidents -- the passive containment's own
        # STOP_WORKLOADS workflow among them (same job and attempt). That one
        # is the containment, reported below; the recovery history is what
        # the terminal decision opened (live 2026-10-27: counting it there
        # read one stop and one restart as two recovery workflows).
        if incident.event_type == "TRAINING_ATTEMPT_FAILURE_DETECTED" or (
            containment is not None and incident.incident_id == containment.incident_id
        ):
            continue
        incident_ids.add(incident.incident_id)
        recovery.append({
            "incident": incident.model_dump(mode="json"),
            "workflow": workflow.model_dump(mode="json"),
        })
    entry["recovery"] = recovery
    workflows = []
    if containment is not None and containment.workflow_request_id:
        try:
            workflows.append(store.get_workflow(containment.workflow_request_id).model_dump(mode="json"))
        except NotFoundError:
            pass
    entry["containment_workflows"] = workflows
    # The notifications of the attempt's own incidents (containment and
    # recovery): what a redelivered terminal must not add to (029).
    entry["notifications"] = [
        {
            "notification_id": notification.notification_id,
            "incident_id": notification.incident_id,
            "category": notification.category,
            "deduplication_key": notification.deduplication_key,
            "created_at": notification.created_at.isoformat(),
            "subject": notification.subject,
        }
        for notification in store.list_notifications(limit=500, newest_first=True)
        if notification.incident_id in incident_ids
    ]
    attempts[attempt_id] = entry
try:
    budget = store.get_restart_budget(cluster_id, job_id).model_dump(mode="json")
except NotFoundError:
    budget = None
print(json.dumps({"attempts": attempts, "restart_budget": budget}, sort_keys=True, default=str))
"""


#: Read-only: one managed job's attempt observations as the completion watcher
#: recorded them (containers with Pod UID, container name, host PID,
#: termination), the NODE_LOGS entries the node log collector delivered from a
#: fixture path since a moment, and the node's log incidents (``log-`` event
#: ids) with their workflows. Never writes.
APPLOG_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import EvidenceKind

cluster_id, node_id, since_text, job_id, needle = sys.argv[1:6]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store

attempts = [
    item.model_dump(mode="json")
    for item in store.list_attempt_observations(cluster_id)
    if item.job_id == job_id
]
entries = []
for record in store.list_raw_evidence(
    cluster_id, node_id=node_id, kind=EvidenceKind.NODE_LOGS, limit=2000
):
    if record.observed_at < since:
        continue
    payload = record.payload if isinstance(record.payload, dict) else {}
    for entry in payload.get("entries") or []:
        fields = entry.get("fields") or {}
        if needle and needle not in str(fields.get("path") or ""):
            continue
        entries.append({
            "record_id": record.record_id,
            "batch_id": payload.get("batch_id"),
            "collected_at": payload.get("collected_at"),
            "entry_id": entry.get("entry_id"),
            "source": entry.get("source"),
            "observed_at": entry.get("observed_at"),
            "message": entry.get("message"),
            "fields": fields,
        })
entries.sort(key=lambda item: (str(item["collected_at"]), str(item["observed_at"])))
incidents = {}
for incident in store.list_incidents_by_state(
    cluster_id, list(IncidentState), node_ids={node_id}
):
    if incident.created_at >= since and str(incident.event_id).startswith("log-"):
        incidents[incident.incident_id] = incident
workflows = {}
for incident in incidents.values():
    if incident.workflow_request_id:
        try:
            workflows[incident.workflow_request_id] = store.get_workflow(
                incident.workflow_request_id
            ).model_dump(mode="json")
        except NotFoundError:
            pass
print(json.dumps({
    "since": since.isoformat(),
    "cluster_id": cluster_id,
    "node_id": node_id,
    "job_id": job_id,
    "attempts": attempts,
    "entries": entries,
    "incidents": [item.model_dump(mode="json") for item in incidents.values()],
    "workflows": workflows,
}, sort_keys=True, default=str))
"""


def preflight_state(regional: CpuPython, cluster_id: str, node: str) -> dict[str, Any]:
    return regional.cpu_python(PREFLIGHT_PROBE, cluster_id, node)


def node_activity(
    regional: CpuPython,
    cluster_id: str,
    node: str,
    since: datetime,
    *,
    metrics: Iterable[str] = (),
    signal_keys: Iterable[str] = (),
    kinds: Sequence[str] = ("HOST_TELEMETRY",),
) -> dict[str, Any]:
    for value in (*metrics, *signal_keys, *kinds):
        if "," in value or not value:
            raise ValueError("activity probe filters must be non-empty, no comma")
    return regional.cpu_python(
        ACTIVITY_PROBE,
        cluster_id,
        node,
        since.isoformat(),
        ",".join(sorted(set(metrics))),
        ",".join(signal_keys),
        ",".join(kinds),
        timeout=240,
    )


def attempt_records(
    regional: CpuPython, cluster_id: str, job_id: str, attempt_ids: Sequence[str]
) -> dict[str, Any]:
    if not attempt_ids or any("," in item or not item for item in attempt_ids):
        raise ValueError("attempt ids must be non-empty")
    return regional.cpu_python(ATTEMPT_PROBE, cluster_id, job_id, *attempt_ids)


def applog_records(
    regional: CpuPython,
    cluster_id: str,
    node: str,
    since: datetime,
    *,
    job_id: str,
    path_needle: str,
) -> dict[str, Any]:
    """A managed job's attempt observations, the fixture-path log entries and
    the node's log incidents since ``since`` (``APPLOG_PROBE``)."""

    if not job_id or not path_needle:
        raise ValueError("applog records need a job id and a path needle")
    return regional.cpu_python(
        APPLOG_PROBE,
        cluster_id,
        node,
        since.isoformat(),
        job_id,
        path_needle,
        timeout=240,
    )


def signal_key(
    cluster_id: str,
    node: str,
    metric: str,
    device: str | None = None,
    rule_id: str | None = None,
) -> str:
    """``health_signal_key`` of ``gpu_fault.store.shared.health_signals``."""

    parts = [cluster_id, node, metric, device or "node"]
    if rule_id:
        parts.append(rule_id)
    return "/".join(parts)


__all__ = [
    "ACTIVITY_PROBE",
    "APPLOG_PROBE",
    "ATTEMPT_PROBE",
    "PREFLIGHT_PROBE",
    "applog_records",
    "attempt_records",
    "node_activity",
    "preflight_state",
    "signal_key",
]
