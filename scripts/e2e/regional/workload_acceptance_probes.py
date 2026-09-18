"""Deployed workload observation, Store and kernel-evidence probes."""

WORKLOAD_STORE_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

# The fourth argument arrived after the third; a caller that still passes
# three gets the job-scoped workflow read below.
argv = list(sys.argv[1:])
if len(argv) == 3:
    argv.append("")
cluster_id, job_id, attempt_id, workflow_request_ids_text = argv
store = ApplicationContext.from_environment().store
observations = [
    item.model_dump(mode="json")
    for item in store.list_attempt_observations(cluster_id)
    if item.job_id == job_id and item.attempt_id == attempt_id
]
try:
    decision = store.get_decision_by_attempt(cluster_id, attempt_id)
except NotFoundError:
    # A baseline attempt has no terminal decision yet.
    decision = None
try:
    budget = store.get_restart_budget(cluster_id, job_id).model_dump(mode="json")
except NotFoundError:
    budget = None
# Incidents and workflows the store already scopes to this cluster and job:
# the ones still acting on the attempt and the ones that restarted into it.
# Every backend filters both in the store, so this probe never pages the
# whole workflow table -- or, as it used to, the whole remote-command table
# every five seconds -- through the API Pod.
pairs = {}
for incident, workflow in store.list_active_workflow_incidents(
    cluster_id, job_id=job_id
):
    pairs[workflow.request_id] = (incident, workflow)
for incident, workflow in store.list_job_recovery_workflow_incidents(
    cluster_id, job_id, attempt_id
):
    pairs[workflow.request_id] = (incident, workflow)
explicit_ids = [item for item in workflow_request_ids_text.split(",") if item]
request_ids = sorted(set(pairs) | set(explicit_ids))
commands = [
    item.model_dump(mode="json")
    for item in (
        store.list_remote_commands(workflow_request_ids=request_ids)
        if request_ids
        else []
    )
    if any(
        job_id in workload_id
        for workload_id in item.step.workload_ids
    )
    or item.workflow_request_id in explicit_ids
]
print(json.dumps({
    "cluster_id": cluster_id,
    "observations": observations,
    "decision": decision.model_dump(mode="json") if decision else None,
    "restart_budget": budget,
    "commands": commands,
    "incidents": [
        {
            "incident_id": incident.incident_id,
            "cluster_id": incident.cluster_id,
            "job_id": incident.job_id,
            "attempt_id": incident.attempt_id,
            "workflow_request_id": incident.workflow_request_id,
        }
        for incident, _workflow in pairs.values()
    ],
    "workflows": [
        {
            "request_id": workflow.request_id,
            "incident_id": workflow.incident_id,
            "status": workflow.status.value,
        }
        for _incident, workflow in pairs.values()
    ],
}, sort_keys=True, default=str))
"""


OBSERVATION_POST_PROBE = r"""
import json
import os
import sys
from gpu_fault.collectors.sinks import HttpEventSink

payload = json.loads(sys.argv[1])
os.environ["SSL_CERT_FILE"] = os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
sink = HttpEventSink(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
    bearer_token=os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
    timeout_seconds=15,
    max_attempts=1,
)
result = sink.post("/v1/workload-observations", payload)
print(json.dumps(result, sort_keys=True))
"""


E2E_PREFLIGHT_PROBE = r"""
import json
import sys
from datetime import datetime, timezone
from gpu_fault.app import ApplicationContext

cluster_id = sys.argv[1]
store = ApplicationContext.from_environment().store
statuses = [
    item.model_dump(mode="json")
    for item in store.list_collector_statuses(cluster_id)
]
agents = [
    item.model_dump(mode="json")
    for item in store.list_agents(cluster_id)
]
print(json.dumps({
    "collector_statuses": statuses,
    "agents": agents,
    "observed_at": datetime.now(timezone.utc).isoformat(),
}, sort_keys=True, default=str))
"""


KMSG_EVIDENCE_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
cluster_id, reference = sys.argv[1:]
records = []
for item in ApplicationContext.from_environment().store.list_raw_evidence(cluster_id):
    if item.payload.get("evidence_ref") != reference:
        continue
    records.append({
        "record_id": item.record_id, "cluster_id": item.cluster_id, "node_id": item.node_id,
        "payload": {
            key: item.payload.get(key) for key in (
                "record_id", "cluster_id", "node_id", "source_boot_id", "evidence_ref"
            )
        },
    })
print(json.dumps({"evidence": records}, sort_keys=True))
"""
