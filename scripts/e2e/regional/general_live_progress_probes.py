"""GF-GENERAL-LIVE-027 store probes: the records read and the scan trigger.

The two scripts ``general_live_progress`` runs through ``cpu_python`` in an
API Pod, kept apart from the fixture module (the architecture gate's
1500-line file cap). Both are read-only against the product's own records;
the scan trigger exercises the product's execution-token route and nothing
else.
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
TRAINING_PROBE = r"""
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.store import NotFoundError

cluster_id, node_list, job_id, attempt_id, since_text = sys.argv[1:6]
node_ids = {item for item in node_list.split(",") if item}
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

print(json.dumps({
    "progress_states": progress,
    "observation": observation,
    "incidents": incidents,
    "workflows": workflows,
    "decision": decision,
    "notifications": notifications,
    "probed_at": datetime.now(timezone.utc).isoformat(),
}, sort_keys=True, default=str))
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
