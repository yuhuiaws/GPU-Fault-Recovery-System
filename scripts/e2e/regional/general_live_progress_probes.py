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

#: Run in a reporter sidecar (``kubectl exec -c progress-reporter-<rank>``,
#: the GPU runtime image's interpreter): POST one heartbeat for ``argv[1]``
#: (an attempt id), rank ``argv[2]``, progress fields ``argv[3]`` (JSON) --
#: built with the product's own heartbeat model and sent with the product's
#: own collector client exactly as the reporter sends its own: the sidecar's
#: control-plane URL, cluster token and CA (``SSL_CERT_FILE``), and the Pod
#: identity the reporter reports (``NODE_NAME``, ``POD_UID``,
#: ``CONTAINER_NAME``). One live attempt, no outbox and no retry
#: (``HttpEventSink.deliver_once``), so the product's answer comes back to the
#: runner instead of being buffered for a replay. Prints one JSON object: the
#: heartbeat sent, and either the product's answer (``delivered`` true,
#: ``body`` = ``TrainingHealthResult``) or the client's error and HTTP status.
#: Never prints the token.
HEARTBEAT_POST = r"""
import json
import os
import sys
from datetime import datetime, timezone

from gpu_fault.channel_registry import TRAINING_PROGRESS_PATH
from gpu_fault.collectors.sinks import CollectorError, HttpEventSink
from gpu_fault.training_models import TrainingProgressHeartbeat

attempt_id, rank, fields = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
heartbeat = TrainingProgressHeartbeat(
    cluster_id=os.environ["GPU_FAULT_CLUSTER_ID"],
    attempt_id=attempt_id,
    rank=rank,
    observed_at=datetime.now(timezone.utc),
    node_id=os.environ.get("NODE_NAME") or None,
    pod_uid=os.environ.get("POD_UID") or None,
    container_name=os.environ.get("CONTAINER_NAME") or None,
    **fields,
)
payload = heartbeat.model_dump(mode="json")
sink = HttpEventSink(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
    bearer_token=os.environ.get("GPU_FAULT_CONTROL_PLANE_TOKEN"),
    timeout_seconds=30,
)
answer = {"heartbeat": payload}
try:
    answer.update(delivered=True, body=sink.deliver_once(TRAINING_PROGRESS_PATH, payload))
except CollectorError as exc:
    answer.update(delivered=False, status=exc.status_code, error=str(exc)[:500])
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
