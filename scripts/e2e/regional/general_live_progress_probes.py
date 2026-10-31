"""GF-GENERAL-LIVE-027 store probes: the records read and the scan trigger.

The two scripts ``general_live_progress`` runs through ``cpu_python`` in an
API Pod, kept apart from the fixture module (the architecture gate's
1500-line file cap). Both are read-only against the product's own records;
the scan trigger exercises the product's execution-token route and nothing
else. ``HEARTBEAT_POST`` runs in the test job's own reporter sidecar instead
and sends one heartbeat through the product's ingest route, as the reporter
does (027 ``old_attempt``'s foreign attempt).
"""

from __future__ import annotations

#: Read-only: this attempt's training-progress state and observation, its
#: ``SITE_TRAINING_HEALTH`` incidents (training findings) on any node of the
#: job, their workflows and step executions, the completion decision and the
#: training notifications. Never writes.
#:
#: ``argv[2]`` is every node the job's ranks run on, comma-separated, as 011's
#: ``HUNG_PROBE`` takes it. A training finding is bound to the rank it grades
#: -- its ``node_id`` is that rank's node (``training_health.py`` ``_claim``)
#: -- and the incident the scan route raises from it names that node alone
#: (``node_ids``), so a read scoped to one node misses the other rank's
#: incidents. Live 2026-10-26 17:43Z (HyperPod, release 37db33f254a1)
#: ``straggler_group`` read the lane node only and never saw rank 1's
#: ``training_straggler`` on ``--node-b``: "no training_straggler finding
#: reached an incident" with both ranks' fresh heartbeats in the store (rank 0
#: step 240, rank 1 step 180, one ``progress_group``), and ``findings: []`` in
#: every later scan body because the signal was already active. The attempt's
#: observation (phase, containers with rank, node, Pod UID and container name)
#: and the control plane's clock at the read are kept so a binding, phase or
#: clock cause is visible in ``training-records.json`` next time.
#:
#: ``argv[6]`` (optional) names further attempt ids, comma-separated, whose
#: stored progress states are listed as ``other_progress_states``: 027
#: ``old_attempt`` proves that the heartbeat it posted for an attempt the
#: product does not run left no state behind.
TRAINING_PROBE = r"""
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.store import NotFoundError

cluster_id, node_list, job_id, attempt_id, since_text = sys.argv[1:6]
node_ids = {item for item in node_list.split(",") if item}
others = [item for item in (sys.argv[6] if len(sys.argv) > 6 else "").split(",") if item]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store

progress = []
for state in store.list_training_progress_states(cluster_id, attempt_id):
    heartbeat = state.heartbeat
    progress.append({
        "rank": heartbeat.rank,
        "node_id": heartbeat.node_id,
        "pod_uid": heartbeat.pod_uid,
        "container_name": heartbeat.container_name,
        "step": heartbeat.step,
        "loss": heartbeat.loss,
        "numerical_error": heartbeat.numerical_error,
        "checkpoint_ref": heartbeat.checkpoint_ref,
        "labels": heartbeat.labels,
        "observed_at": heartbeat.observed_at.isoformat(),
        "last_progress_at": state.last_progress_at.isoformat(),
    })

observation = None
for state in store.list_attempt_observation_states(cluster_id):
    if state.observation.attempt_id != attempt_id:
        continue
    observation = {
        "workload_phase": str(state.observation.workload_phase.value),
        "first_observed_at": state.first_observed_at.isoformat(),
        "observed_at": state.observation.observed_at.isoformat(),
        "expected_critical_ranks": state.observation.expected_critical_ranks,
        "containers": [
            {
                "rank": item.rank,
                "node_id": item.node_id,
                "pod_uid": item.pod_uid,
                "container_name": item.container_name,
                "critical": item.critical,
                "terminated": item.terminated,
            }
            for item in state.observation.containers
        ],
    }

incidents = []
workflows = {}
for incident in store.list_incidents_by_state(
    cluster_id, list(IncidentState), node_ids=node_ids
):
    if not str(incident.event_id).startswith("training-"):
        continue
    if incident.created_at < since and incident.updated_at < since:
        continue
    incidents.append({
        "incident_id": incident.incident_id,
        "event_id": incident.event_id,
        "state": incident.state.value,
        "policy_source": incident.policy_source,
        "job_id": incident.job_id,
        "attempt_id": incident.attempt_id,
        "node_ids": list(incident.node_ids),
        "reasons": list(incident.reasons or []),
        "official_action": incident.official_action,
        "workflow_request_id": incident.workflow_request_id,
    })
    if incident.workflow_request_id and incident.workflow_request_id not in workflows:
        try:
            workflow = store.get_workflow(incident.workflow_request_id)
        except NotFoundError:
            continue
        workflows[incident.workflow_request_id] = workflow.model_dump(mode="json")

try:
    decision = store.get_decision_by_attempt(cluster_id, attempt_id).model_dump(
        mode="json"
    )
except NotFoundError:
    decision = None

notifications = []
incident_ids = {item["incident_id"] for item in incidents}
for notification in store.list_notifications(limit=500, newest_first=True):
    if notification.incident_id not in incident_ids:
        continue
    result = store.get_notification_result(notification.notification_id)
    notifications.append({
        "notification_id": notification.notification_id,
        "incident_id": notification.incident_id,
        "deduplication_key": notification.deduplication_key,
        "category": notification.category,
        "status": result.status.value if result else None,
    })

other_progress = {
    other: [
        {
            "rank": state.heartbeat.rank,
            "node_id": state.heartbeat.node_id,
            "observed_at": state.heartbeat.observed_at.isoformat(),
        }
        for state in store.list_training_progress_states(cluster_id, other)
    ]
    for other in others
}

print(json.dumps({
    "progress_states": progress,
    "other_progress_states": other_progress,
    "observation": observation,
    "incidents": incidents,
    "workflows": workflows,
    "decision": decision,
    "notifications": notifications,
    "probed_at": datetime.now(timezone.utc).isoformat(),
}, sort_keys=True, default=str))
"""

#: How long ``HEARTBEAT_POST`` follows a processor receipt to its completion.
#: The product's own receipt client allows 120 s (``HttpEventSink``
#: ``processor_receipt_timeout_seconds``); a routine-tier row behind a busy
#: queue gets twice that before the variant calls it unobserved. The exec that
#: runs the script allows 120 s more: the POST's own 30-s timeout, one last
#: 30-s read past the bound and the interpreter's start-up.
RECEIPT_FOLLOW_SECONDS = 240

#: Run in a reporter sidecar (``kubectl exec -c progress-reporter-<rank>``,
#: the GPU runtime image's interpreter): POST one heartbeat for ``argv[1]``
#: (an attempt id), rank ``argv[2]``, progress fields ``argv[3]`` (JSON) --
#: built with the product's own heartbeat model and sent with the product's
#: own collector client exactly as the reporter sends its own: the sidecar's
#: control-plane URL, cluster token and CA (``SSL_CERT_FILE``), and the Pod
#: identity the reporter reports (``NODE_NAME``, ``POD_UID``,
#: ``CONTAINER_NAME``). One live attempt, no outbox and no retry
#: (``HttpEventSink.deliver_once``).
#:
#: On the regional path that answer is not the ingest's: the channel is
#: ``receipt=True`` (``channel_registry``), so the ingress admits the POST to
#: the processor queue and answers 202 ``{"accepted": true,
#: "processor_request_id", "status_url", "coalesced"}``
#: (``app/middleware/dispatch.py``) -- an admission, not a decision. The
#: processor later replays the POST through the route handler and commits the
#: handler's answer to the request row, which ``GET status_url`` returns once
#: COMPLETED (``X-GPU-Fault-Processor-Status: COMPLETED``, the handler's own
#: HTTP status and body) and answers 202 ``{"status": PENDING|LEASED}`` until
#: then (``app/routes/processor.py``). The script follows such a receipt for up
#: to ``argv[4]`` seconds as the product's own receipt client does (same
#: authentication and cluster header, 202 and retryable statuses asked again
#: with a doubling delay up to 2 s) and keeps the last read as ``receipt``.
#: Prints one JSON object: the heartbeat sent, the POST's answer (``body``) or
#: the client's error and HTTP status, and the receipt's outcome. Never prints
#: the token.
HEARTBEAT_POST = r"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gpu_fault.channel_registry import TRAINING_PROGRESS_PATH
from gpu_fault.collectors.sinks import (
    CollectorError,
    HttpEventSink,
    is_retryable_collector_status,
)
from gpu_fault.training_models import TrainingProgressHeartbeat

attempt_id, rank, fields = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
follow_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
cluster_id = os.environ["GPU_FAULT_CLUSTER_ID"]
base_url = os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
token = os.environ.get("GPU_FAULT_CONTROL_PLANE_TOKEN")
heartbeat = TrainingProgressHeartbeat(
    cluster_id=cluster_id,
    attempt_id=attempt_id,
    rank=rank,
    observed_at=datetime.now(timezone.utc),
    node_id=os.environ.get("NODE_NAME") or None,
    pod_uid=os.environ.get("POD_UID") or None,
    container_name=os.environ.get("CONTAINER_NAME") or None,
    **fields,
)
payload = heartbeat.model_dump(mode="json")
sink = HttpEventSink(base_url, bearer_token=token, timeout_seconds=30)


def read(status_url):
    headers = {"X-GPU-Fault-Cluster-ID": cluster_id, "User-Agent": "gfrs-general-live"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        request = Request(base_url + status_url, headers=headers, method="GET")
        with urlopen(request, timeout=30) as response:
            code, received, text = int(response.status), response.headers, response.read()
    except HTTPError as exc:
        code, received, text = exc.code, exc.headers, exc.read()
    except OSError as exc:
        return None, None, str(exc)[:300]
    try:
        body = json.loads(text) if text else None
    except ValueError:
        body = text.decode("utf-8", "replace")[:300]
    status = received.get("X-GPU-Fault-Processor-Status") if received else None
    return code, status, body


def follow(status_url):
    started = time.monotonic()
    delay, polls = 0.25, 0
    while True:
        polls += 1
        code, processor_status, body = read(status_url)
        if processor_status is None and code == 202 and isinstance(body, dict):
            processor_status = body.get("status")
        outcome = {
            "status_url": status_url,
            "http_status": code,
            "processor_status": processor_status,
            "body": body,
            "polls": polls,
            "waited_seconds": round(time.monotonic() - started, 3),
            "bound_seconds": follow_seconds,
        }
        if processor_status == "COMPLETED":
            return {**outcome, "terminal": True}
        if code is not None and code != 202 and not is_retryable_collector_status(code):
            return {**outcome, "terminal": False, "refused_read": True}
        if time.monotonic() - started + delay > follow_seconds:
            return {**outcome, "terminal": False}
        time.sleep(delay)
        delay = min(delay * 2, 2.0)


answer = {"heartbeat": payload}
try:
    body = sink.deliver_once(TRAINING_PROGRESS_PATH, payload)
except CollectorError as exc:
    answer.update(delivered=False, status=exc.status_code, error=str(exc)[:500])
else:
    answer.update(delivered=True, body=body)
    if isinstance(body, dict) and body.get("processor_request_id"):
        status_url = body.get("status_url")
        if not isinstance(status_url, str) or not status_url.startswith("/"):
            status_url = "/v1/processor/requests/" + str(body["processor_request_id"])
        answer["receipt"] = follow(status_url)
print(json.dumps(answer, sort_keys=True, default=str))
"""

#: Trigger one cluster scan through the product route (execution token); the
#: global monitor stays disabled. Mirrors warm_spare_fixture's in-Pod POST.
SCAN_POST = r"""
import json
import os
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

cluster_id = sys.argv[1]
request = Request(
    "http://127.0.0.1:8080/v1/training-health/" + quote(cluster_id, safe="") + "/scan",
    data=b"{}",
    method="POST",
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
try:
    with urlopen(request, timeout=60) as response:
        content = response.read()
        result = {"status": response.status, "body": json.loads(content) if content else {}}
except HTTPError as exc:
    content = exc.read()
    try:
        decoded = json.loads(content) if content else {}
    except ValueError:
        decoded = {"detail": content.decode(errors="replace")}
    result = {"status": exc.code, "body": decoded}
print(json.dumps(result, sort_keys=True))
"""
