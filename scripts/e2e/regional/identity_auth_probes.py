"""Deployed-code identity probes; transported on stdin, without adapter actions."""

AUTH008_EXECUTOR_IDENTITY_PROBE = r"""
import json
import os
import socket
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
cluster = os.environ["GPU_FAULT_CLUSTER_ID"]
print(json.dumps({
    "cluster_id": cluster,
    "executor_id": os.getenv("GPU_FAULT_CLUSTER_EXECUTOR_ID", cluster + "/" + socket.gethostname()),
    "artifact": os.environ["GPU_FAULT_EXECUTOR_ARTIFACT_SHA256"],
    "compatibility": os.environ["GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST"],
    "protocol": CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
    "probe_owner_configured": any(
        value == "gpu-fault-acceptance-probe"
        for key, value in os.environ.items()
        if key.startswith("GPU_FAULT_") and key.endswith("_OWNER")
    ),
}, sort_keys=True))
"""


AUTH008_CLAIM_PROBE = r"""
import hashlib
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
header_cluster, payload = json.loads(sys.argv[1])
request = urllib.request.Request(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/") + "/v1/regional/executors/claim",
    data=json.dumps(payload).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
        "X-GPU-Fault-Cluster-ID": header_cluster,
    },
    method="POST",
)
context = ssl.create_default_context(cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"])
try:
    with urllib.request.urlopen(request, context=context, timeout=20) as response:
        status, raw = response.status, response.read()
except urllib.error.HTTPError as exc:
    status, raw = exc.code, exc.read()
body = json.loads(raw)
safe = {"invalid_body_sha256": hashlib.sha256(raw).hexdigest()}
if body == {"detail": "regional cluster authentication failed"}:
    safe = body
elif (
    isinstance(body, dict) and set(body) == {"commands"}
    and isinstance(body["commands"], list)
    and all(isinstance(item, dict) for item in body["commands"])
):
    safe = {"commands": [
        {key: item.get(key) for key in (
            "command_id", "cluster_id", "workflow_request_id", "incident_id", "status"
        )}
        for item in body["commands"]
    ]}
print(json.dumps({"status": status, "body": safe}, sort_keys=True))
"""


AUTH008_BACKLOG_PROBE = r"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    FaultIncident, IncidentState, WorkflowOperation, WorkflowRequest,
    WorkflowStatus, WorkflowStepSpec,
)
from gpu_fault.regional import RemoteActionCommand, RemoteCommandResult, RemoteCommandStatus
from gpu_fault.store import NotFoundError

action, receipt = json.loads(sys.argv[1])
nonce = receipt["nonce"]
if not re.fullmatch(r"[0-9a-f]{32}", nonce):
    raise RuntimeError("invalid AUTH008 ownership nonce")
if os.environ["GPU_FAULT_RELEASE_ID"] != receipt["release_id"]:
    raise RuntimeError("AUTH008 release changed")
cluster_a, cluster_b = receipt["cluster_a"], receipt["cluster_b"]
if not cluster_a or not cluster_b or cluster_a == cluster_b:
    raise RuntimeError("AUTH008 needs distinct clusters")
store = ApplicationContext.from_environment().store
owner = "gpu-fault-acceptance-probe"
seed_owner = "auth008-seed-" + nonce
event_id = "auth008-event-" + nonce
incident_id = "auth008-incident-" + nonce
workflow_id = "auth008-workflow-" + nonce
command_id = "auth008-command-" + nonce
step = WorkflowStepSpec(operation=WorkflowOperation.FREEZE_EVIDENCE, execution_owner=owner)
now = datetime.now(timezone.utc)
deadline = datetime.fromisoformat(receipt["expires_at"])

def optional(reader, key):
    try:
        return reader(key)
    except NotFoundError:
        return None

def records():
    incident = optional(store.get_incident, incident_id)
    workflow = optional(store.get_workflow, workflow_id)
    command = optional(store.get_remote_command, command_id)
    if incident is not None and (
        incident.cluster_id != cluster_b or incident.event_id != event_id
        or incident.drill_id != nonce or incident.node_ids or incident.job_id
        or incident.workflow_request_id != workflow_id or incident.fencing_token != 1
        or incident.event_type != "AUTH008_ACCEPTANCE"
    ):
        raise RuntimeError("AUTH008 incident ownership changed")
    if workflow is not None and (
        workflow.incident_id != incident_id or workflow.official_action != "NO_ACTION"
        or workflow.official_steps != [step] or workflow.safety_steps
        or workflow.execution_owner_id != seed_owner or workflow.execution_epoch != 1
        or workflow.fencing_token != 1 or workflow.merge_revision != 0
    ):
        raise RuntimeError("AUTH008 workflow ownership changed")
    if command is not None and (
        command.cluster_id != cluster_b or command.workflow_request_id != workflow_id
        or command.incident_id != incident_id or command.step != step
        or command.batched_steps or command.step_index != 0 or command.fencing_token != 1
        or command.idempotency_key != workflow_id + "/0/FREEZE_EVIDENCE"
        or command.workflow.request_id != workflow_id
        or command.workflow.official_steps != [step] or command.workflow.safety_steps
        or command.incident.cluster_id != cluster_b or command.incident.node_ids
        or command.incident.job_id or command.restart_authorization is not None
    ):
        raise RuntimeError("AUTH008 command ownership changed")
    if (incident is None) != (workflow is None) or (command is not None and workflow is None):
        raise RuntimeError("AUTH008 partial ownership cannot be reconciled")
    if workflow is not None and any(
        item.command_id != command_id
        for item in store.list_remote_commands(workflow_request_ids=[workflow_id])
    ):
        raise RuntimeError("AUTH008 workflow has an unowned command")
    return incident, workflow, command

incident, workflow, command = records()
if action == "seed":
    if any(item is not None for item in (incident, workflow, command)):
        raise RuntimeError("AUTH008 owned IDs are not fresh")
    if deadline.tzinfo is None or not 60 < (deadline - now).total_seconds() <= 1800:
        raise RuntimeError("AUTH008 isolation lease is invalid")
    for cluster in (cluster_a, cluster_b):
        if not store.get_regional_cluster(cluster).is_active():
            raise RuntimeError("AUTH008 cluster is not active")
    if any(
        item.step.execution_owner == owner
        and item.status in {RemoteCommandStatus.PENDING, RemoteCommandStatus.WAITING, RemoteCommandStatus.LEASED}
        for item in store.list_remote_commands()
    ):
        raise RuntimeError("AUTH008 probe-owner backlog is not empty before creation")
    incident = FaultIncident(
        incident_id=incident_id, event_id=event_id, event_type="AUTH008_ACCEPTANCE",
        cluster_id=cluster_b, node_ids=[], policy_version="auth008/v1",
        policy_source="ACCEPTANCE", state=IncidentState.ACTION_PENDING,
        workflow_request_id=workflow_id, fencing_token=1, drill_id=nonce,
    )
    workflow = WorkflowRequest(
        request_id=workflow_id, incident_id=incident_id, status=WorkflowStatus.PENDING,
        official_action="NO_ACTION", fencing_token=1, official_steps=[step],
        execution_owner_id=seed_owner, execution_epoch=1,
        execution_lease_expires_at=deadline, not_before=deadline,
    )
    incident, workflow, created = store.create_incident_workflow_if_absent(
        event_id, lambda: (incident, workflow)
    )
    if not created:
        raise RuntimeError("AUTH008 event identity collision")
    store.ensure_remote_command(RemoteActionCommand(
        command_id=command_id, cluster_id=cluster_b,
        workflow_request_id=workflow_id, incident_id=incident_id, step_index=0,
        fencing_token=1, idempotency_key=workflow_id + "/0/FREEZE_EVIDENCE",
        step=step, workflow=workflow, incident=incident,
    ))
elif action == "cleanup":
    if command is not None and command.status is RemoteCommandStatus.LEASED:
        if command.lease_owner not in receipt["claimants"] or not command.lease_token:
            raise RuntimeError("AUTH008 command has an unowned claimant; cleanup deferred")
        store.cancel_remote_commands_for_workflow(workflow_id, reason="AUTH008 no-action probe retired")
        store.complete_remote_command(cluster_b, command_id, RemoteCommandResult(
            lease_token=command.lease_token, status=RemoteCommandStatus.FAILED,
            error="AUTH008 no-action probe retired", status_source="auth008-probe-cleanup",
            details={"no_action_executed": True},
        ))
    elif command is not None and command.status in {
        RemoteCommandStatus.PENDING, RemoteCommandStatus.WAITING
    }:
        if not store.cancel_remote_command(command_id, reason="AUTH008 no-action probe retired"):
            raise RuntimeError("AUTH008 candidate changed during cancellation")
    if workflow is not None and workflow.status is not WorkflowStatus.SUPERSEDED:
        workflow = workflow.model_copy(update={
            "status": WorkflowStatus.SUPERSEDED, "superseded_at": now,
            "updated_at": now, "execution_lease_expires_at": None,
        })
        incident = incident.model_copy(update={"state": IncidentState.RECOVERED, "updated_at": now})
        store.save_workflow_and_incident_if_leased(workflow, incident, seed_owner, 1)
elif action != "snapshot":
    raise RuntimeError("unknown AUTH008 backlog action")
incident, workflow, command = records()
retired = (
    (workflow is None or workflow.status is WorkflowStatus.SUPERSEDED)
    and (incident is None or incident.state is IncidentState.RECOVERED)
    and (command is None or (
        command.status is RemoteCommandStatus.FAILED
        and command.lease_owner is None and command.lease_token is None
        and command.lease_expires_at is None
    ))
)
pending = (
    command is not None and workflow is not None
    and command.status is RemoteCommandStatus.PENDING
    and command.lease_owner is None and command.last_lease_owner is None
    and command.lease_token is None and command.lease_expires_at is None
    and command.cancellation_requested_at is None
    and workflow.status is WorkflowStatus.PENDING
    and workflow.execution_lease_expires_at is not None
    and workflow.execution_lease_expires_at > datetime.now(timezone.utc)
)
print(json.dumps({
    "command_id": command_id, "workflow_id": workflow_id, "incident_id": incident_id,
    "cluster_id": cluster_b, "pending_unleased": pending, "retired": retired,
    "status": command.status.value if command else None,
    "lease_owner": command.lease_owner if command else None,
    "command_sha256": hashlib.sha256(command.model_dump_json().encode()).hexdigest() if command else None,
}, sort_keys=True))
"""


ROUTE_INVENTORY_PROBE = r"""
import json
from gpu_fault.app import create_app
from gpu_fault.app.authorization import (
    AUTHORIZATION_BUCKET_ATTRIBUTE,
    UNDOCUMENTED_PUBLIC_PATHS,
    ExplicitAuthorizationRegistry,
    iter_api_routes,
)

app = create_app()
registry = ExplicitAuthorizationRegistry()
registry.load(app.routes)
routes = []
for route in iter_api_routes(app.routes):
    if not (
        route.path.startswith("/v1/")
        or route.path in {"/healthz", "/livez", "/metrics"}
    ):
        continue
    routes.append({
        "path": route.path,
        "methods": sorted(route.methods or []),
        "bucket": getattr(route.endpoint, AUTHORIZATION_BUCKET_ATTRIBUTE),
    })
# The OpenAPI surface has no bucket: it is public read-only information that
# AUTH-014 records (anonymous status) rather than judges against a bucket.
for path in sorted(UNDOCUMENTED_PUBLIC_PATHS):
    routes.append({"path": path, "methods": ["GET"], "bucket": "public-undocumented"})
print(json.dumps({"routes": routes}, sort_keys=True))
"""


EXECUTION_TOKEN_DIGEST_PROBE = r"""
import hashlib
import json
import os
token = os.environ["GPU_FAULT_EXECUTION_TOKEN"]
print(json.dumps({
    "sha256": hashlib.sha256(token.encode()).hexdigest(),
    "stripped_sha256": hashlib.sha256(token.strip().encode()).hexdigest(),
}))
"""


ANONYMOUS_ROUTE_PROBE = r"""
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request

routes = json.loads(sys.argv[1])
base = os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
context = ssl.create_default_context(
    cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
)
results = []
for item in routes:
    path = re.sub(r"\{[^}]+\}", "probe", item["path"])
    method = item["method"]
    request = urllib.request.Request(
        base + path,
        data=(b"{}" if method in {"POST", "PUT", "PATCH", "DELETE"} else None),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, context=context, timeout=15) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception as exc:
        results.append({
            "path": item["path"],
            "method": method,
            "bucket": item["bucket"],
            "error": type(exc).__name__,
        })
        continue
    results.append({
        "path": item["path"],
        "method": method,
        "bucket": item["bucket"],
        "status": status,
    })
print(json.dumps({"results": results}, sort_keys=True))
"""


REMOTE_STATUS_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
# argv: command IDs whose status must be reported even once terminal, so the
# "after" reading can tell SUCCEEDED (not interrupted) from a vanished row.
tracked = set(sys.argv[1:])
commands = ApplicationContext.from_environment().store.list_remote_commands()
print(json.dumps({
    "commands": [
        {
            "command_id": item.command_id,
            "cluster_id": item.cluster_id,
            "status": item.status.value,
            "lease_owner_present": item.lease_owner is not None,
        }
        for item in commands
        if item.status.value in {"PENDING", "WAITING", "LEASED"}
        or item.command_id in tracked
    ]
}, sort_keys=True))
"""


DIRECT_CLAIM_PROBE_TEMPLATE = r"""
import json
import os
import ssl
import urllib.error
import urllib.request
payload = __PAYLOAD__
request = urllib.request.Request(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/") + "/v1/regional/executors/claim",
    data=json.dumps(payload, separators=(",", ":")).encode(),
    headers={
        "Authorization": "Bearer " + __TOKEN__,
        "Content-Type": "application/json",
        "X-GPU-Fault-Cluster-ID": __CLUSTER_ID__,
    },
    method="POST",
)
context = ssl.create_default_context(cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"])
try:
    with urllib.request.urlopen(request, context=context, timeout=10) as response:
        print(json.dumps({"status": int(response.status)}))
except urllib.error.HTTPError as exc:
    print(json.dumps({"status": exc.code}))
except Exception as exc:
    print(json.dumps({"status": type(exc).__name__}))
"""


TLS_BOUNDARY_PROBE = r"""
import json
import os
import socket
import ssl
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

url = urlsplit(os.environ["GPU_FAULT_CONTROL_PLANE_URL"])
host = url.hostname
ca_file = os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
context = ssl.create_default_context(cafile=ca_file)
certificate = {}
default_handshake_ok = False
default_handshake_error = None
try:
    with socket.create_connection((host, url.port or 443), timeout=15) as raw:
        with context.wrap_socket(raw, server_hostname=host) as tls:
            certificate = tls.getpeercert() or {}
            default_handshake_ok = True
except (ssl.SSLError, OSError) as exc:
    default_handshake_error = type(exc).__name__

empty_ca_rejected = False
with tempfile.TemporaryDirectory(prefix="auth013-") as directory:
    empty_ca = Path(directory) / "empty-ca.pem"
    empty_ca.write_text("", encoding="ascii")
    try:
        ssl.create_default_context(cafile=str(empty_ca))
    except ssl.SSLError:
        empty_ca_rejected = True

# Only OpenSSL's hostname-mismatch error proves the hostname check ran.
# A reset, timeout, or a different certificate error is inconclusive.
wrong_hostname_rejected = False
wrong_hostname_error = None
try:
    with socket.create_connection((host, url.port or 443), timeout=15) as raw:
        with context.wrap_socket(raw, server_hostname="wrong.invalid"):
            pass
except ssl.SSLCertVerificationError as exc:
    wrong_hostname_rejected = getattr(exc, "verify_code", None) == 62
    wrong_hostname_error = type(exc).__name__
except (ssl.SSLError, OSError) as exc:
    wrong_hostname_error = type(exc).__name__

not_after = certificate.get("notAfter")
expires = (
    datetime.fromtimestamp(ssl.cert_time_to_seconds(not_after), tz=timezone.utc)
    if not_after else None
)
sans = [
    value
    for kind, value in certificate.get("subjectAltName", [])
    if kind == "DNS"
]
print(json.dumps({
    "host": host,
    "ca_file": ca_file,
    "ca_exists": Path(ca_file).is_file(),
    "ssl_cert_file_set": bool(os.getenv("SSL_CERT_FILE")),
    "requests_ca_bundle_set": bool(os.getenv("REQUESTS_CA_BUNDLE")),
    "sans": sans,
    "hostname_in_san": host in sans,
    "default_handshake_ok": default_handshake_ok,
    "default_handshake_error": default_handshake_error,
    "empty_ca_rejected": empty_ca_rejected,
    "wrong_hostname_rejected": wrong_hostname_rejected,
    "wrong_hostname_error": wrong_hostname_error,
    "not_after": expires.isoformat() if expires else None,
    "remaining_days": (
        (expires - datetime.now(timezone.utc)).total_seconds() / 86400
        if expires else None
    ),
}, sort_keys=True))
"""


AGENT_SNAPSHOT_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
cluster_id, *nodes = sys.argv[1:]
store = ApplicationContext.from_environment().store
result = {}
for node in nodes:
    agent = store.get_agent(cluster_id, node)
    result[node] = {
        "generation": agent.generation,
        "lifecycle_state": agent.lifecycle_state.value,
        # AgentRecord renamed the heartbeat field to last_seen_at; the evidence
        # key stays for readers of earlier reports.
        "last_heartbeat_at": agent.last_seen_at.isoformat(),
        "node_action_key_version": agent.node_action_key_version,
    }
print(json.dumps({"agents": result}, sort_keys=True))
"""
