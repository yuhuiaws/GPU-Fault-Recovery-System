"""CPU-side source for the shared regional Store snapshot.

Only the source string is sent to the Pod; its imports belong to the installed
control-plane component, not the operator's runner checkout.
"""

from __future__ import annotations

STORE_PROBE = r"""
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import urlsplit

from gpu_fault.app import ApplicationContext
from gpu_fault.hyperpod import hyperpod_submission_idempotency_key
from gpu_fault.nvidia_logs import NvidiaKernelLogEvent, NvidiaLogNormalizer
from gpu_fault.store import NotFoundError


# Lease credentials never belong in a case's evidence directory.
def redacted_command(item):
    value = item.model_dump(mode="json")
    token = value.pop("lease_token", None)
    raw = str(token).encode() if token not in (None, "") else None
    value["lease_token_sha256"] = (
        hashlib.sha256(raw).hexdigest() if raw is not None else None
    )
    value["lease_token_length"] = len(raw) if raw is not None else None
    return value


def marker_in_message(message, marker):
    return isinstance(message, str) and re.search(
        r"(?<![A-Za-z0-9_.:-])" + re.escape(marker) + r"(?![A-Za-z0-9_.:-])",
        message,
    ) is not None


def normalized_evidence_event(record, cluster_id, node_id):
    try:
        source = NvidiaKernelLogEvent.model_validate(record.payload)
    except ValueError:
        raise ValueError("raw kernel evidence payload is invalid") from None
    if (
        record.kind != "NVIDIA_KERNEL"
        or record.cluster_id != cluster_id
        or record.node_id != node_id
        or source.cluster_id != cluster_id
        or source.node_id != node_id
        or not source.record_id
        or record.record_id != f"nvidia-kernel/{source.record_id}"
        or source.observed_at.tzinfo is None
        or record.observed_at != source.observed_at
        or not source.evidence_ref
    ):
        raise ValueError("raw kernel evidence identity is invalid")
    reference = urlsplit(source.evidence_ref)
    if reference.query or reference.fragment:
        raise ValueError("raw kernel evidence reference is invalid")
    if reference.scheme == "kmsg":
        boot_id = source.source_boot_id
        prefix = f"kmsg-{boot_id}-"
        sequence = source.record_id.removeprefix(prefix)
        if (
            not boot_id
            or not source.record_id.startswith(prefix)
            or not sequence.isascii()
            or not sequence.isdecimal()
            or reference.netloc != node_id
            or reference.path != f"/{boot_id}/{sequence}"
        ):
            raise ValueError("raw kernel evidence node/boot/sequence is invalid")
    elif reference.scheme == "api-replay":
        if not reference.netloc or reference.path != f"/{source.record_id}":
            raise ValueError("raw API replay evidence identity is invalid")
    else:
        raise ValueError("raw kernel evidence source is unknown")
    batch = NvidiaLogNormalizer().normalize_kernel(source)
    if (
        len(batch.xid_events) != 1
        or batch.sxid_events
        or any(item.unresolved_reasons for item in batch.provider_signals)
    ):
        raise ValueError("raw kernel evidence does not identify exactly one XID")
    return batch.xid_events[0]


def require_event_marker(item, event):
    if (
        item.marker_id != f"marker-{event.event_id}"
        or item.cluster_id != event.cluster_id
        or item.scope.node_ids != [event.node_id]
        or item.trusted is not True
        or item.event_source != event.event_source
        or item.source_boot_id != event.source_boot_id
        or item.observed_at != event.observed_at
        or item.raw_evidence_ref != event.evidence_ref
        or item.scope.pci_bdfs != ([event.pci_bdf] if event.pci_bdf else [])
    ):
        raise ValueError("raw kernel evidence marker identity is invalid")


def recover_kernel_event(store, cluster_id, node_id, marker, observed_after):
    # Correlation rows are a sliding working set. Re-normalize the retained
    # payload, then require persisted proof of that exact event; never infer
    # an event ID by stripping an arbitrary marker's prefix.
    candidates = []
    for record in store.list_raw_evidence(cluster_id, node_id=node_id, limit=500):
        payload = record.payload
        if not isinstance(payload, dict) or not marker_in_message(
            payload.get("message"), marker
        ):
            continue
        event = normalized_evidence_event(record, cluster_id, node_id)
        if observed_after is not None and event.observed_at < observed_after:
            continue
        candidates.append(event)
    if len(candidates) > 1:
        raise ValueError("raw kernel evidence marker is ambiguous")
    if not candidates:
        return None, None
    event = candidates[0]
    try:
        decision = store.get_xid_policy_decision(event.event_id)
    except NotFoundError:
        decision = None
    if decision is not None:
        if decision.event_id != event.event_id or decision.event_type != "XID":
            raise ValueError("raw kernel evidence decision identity is invalid")
        # The decision retains its marker even after the active marker retires.
        require_event_marker(decision.marker, event)
    else:
        matched = [
            item
            for item in store.list_recent_markers_for_nodes(
                {node_id}, observed_after or event.observed_at
            )
            if item.raw_evidence_ref == event.evidence_ref
        ]
        if not matched:
            return None, None
        if len(matched) != 1:
            raise ValueError("raw kernel evidence marker is ambiguous")
        require_event_marker(matched[0], event)
    return event, decision


# Preflights sample until the queue drains, rather than treating routine
# in-flight telemetry as a backlog. Wait loops request only one cheap sample.
def drained_queue_stats(store, attempts=20, pause=0.5):
    samples = []
    fault_backlog_depth = 0
    for index in range(attempts):
        stats = store.processor_queue_stats()
        samples.append(stats)
        fault_backlog_depth = int(store.processor_fault_backlog_depth())
        if not int(stats.get("depth") or 0):
            break
        if index + 1 < attempts:
            time.sleep(pause)
    result = dict(samples[-1])
    result["samples"] = len(samples)
    result["fault_backlog_depth"] = fault_backlog_depth
    result["max_sampled_depth"] = max(
        int(item.get("depth") or 0) for item in samples
    )
    return result


probe_argv = list(sys.argv[1:])
# Preserve the eight-argument protocol for existing callers.
if len(probe_argv) == 8:
    probe_argv.append("")
(
    cluster_id,
    node_id,
    marker,
    observed_after_text,
    job_id,
    attempt_id,
    hyperpod_cluster,
    queue_attempts_text,
    workflow_request_ids_text,
) = probe_argv
queue_attempts = int(queue_attempts_text or 20)
explicit_workflow_request_ids = [
    item for item in workflow_request_ids_text.split(",") if item
]
store = ApplicationContext.from_environment().store
observed_after = (
    datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
    if observed_after_text
    else None
)
events = (
    store.list_xid_events(
        cluster_id,
        node_id,
        observed_after=observed_after,
    )
    if node_id
    else []
)
matching_events = [
    item
    for item in events
    if (
        not marker
        or marker_in_message(item.raw_message, marker)
        or item.event_id in (marker, f"xid-{marker}")
    )
]
if marker and len(matching_events) > 1:
    raise ValueError("correlation event marker is ambiguous")
event = matching_events[-1] if matching_events else None
decision = None
recovered = False
if marker and event is None and node_id:
    event, decision = recover_kernel_event(
        store, cluster_id, node_id, marker, observed_after
    )
    recovered = event is not None
if event is not None:
    if (
        event.cluster_id != cluster_id
        or event.node_id != node_id
        or not isinstance(event.event_id, str)
        or not event.event_id
        or (observed_after is not None and event.observed_at < observed_after)
    ):
        raise ValueError("correlation event identity is invalid")
    if not recovered:
        try:
            decision = store.get_xid_policy_decision(event.event_id)
        except NotFoundError:
            pass
    if decision is not None and decision.event_id != event.event_id:
        raise ValueError("event decision identity is invalid")
incident = store.get_incident_by_event(event.event_id) if event is not None else None
if incident is not None and (
    not isinstance(incident.incident_id, str)
    or not incident.incident_id
    or incident.cluster_id != cluster_id
    or not isinstance(incident.node_ids, list)
    or any(not isinstance(node, str) or not node for node in incident.node_ids)
    or node_id not in incident.node_ids
    or (
        decision is not None
        and decision.incident_id is not None
        and decision.incident_id != incident.incident_id
    )
    or (
        incident.event_id != event.event_id
        and (decision is None or decision.incident_id != incident.incident_id)
    )
):
    raise ValueError("event incident identity is invalid")
workflow = (
    store.get_workflow(incident.workflow_request_id)
    if incident is not None and incident.workflow_request_id
    else None
)
if workflow is not None and workflow.request_id != incident.workflow_request_id:
    raise ValueError("incident workflow identity is invalid")
# Narrow commands in the Store, preserving the explicit multi-workflow read.
if explicit_workflow_request_ids:
    commands = store.list_remote_commands(
        workflow_request_ids=explicit_workflow_request_ids
    )
elif workflow is not None:
    commands = [
        item
        for item in store.list_remote_commands(
            workflow_request_ids=[workflow.request_id]
        )
        if item.workflow_request_id == workflow.request_id
    ]
else:
    commands = []
notifications = []
if incident is not None:
    for item in store.list_notifications():
        if item.incident_id != incident.incident_id:
            continue
        result = store.get_notification_result(item.notification_id)
        notifications.append({
            "notification": item.model_dump(mode="json"),
            "result": (
                result.model_dump(mode="json")
                if result is not None else None
            ),
        })
observations = [
    item.model_dump(mode="json")
    for item in store.list_attempt_observations(cluster_id)
    if (not job_id or item.job_id == job_id)
    and (not attempt_id or item.attempt_id == attempt_id)
]
restart_budget = None
if job_id:
    try:
        restart_budget = store.get_restart_budget(cluster_id, job_id).model_dump(
            mode="json"
        )
    except NotFoundError:
        pass
agent = None
profile = None
if node_id:
    try:
        agent_model = store.get_agent(cluster_id, node_id)
        agent = agent_model.model_dump(mode="json")
        profile = store.get_profile(
            agent_model.runtime_profile_version
        ).model_dump(mode="json")
    except (KeyError, NotFoundError):
        pass
submission = None
if hyperpod_cluster and commands:
    restart_command = next(
        (
            item for item in commands
            if item.step.operation.value == "RESTART_NODE"
        ),
        None,
    )
    if restart_command is not None:
        submission_key = (
            restart_command.result_details.get("submission_idempotency_key")
            or hyperpod_submission_idempotency_key(
                workflow.request_id,
                restart_command.step_index,
                restart_command.step.operation,
            )
        )
        try:
            submission = store.get_hyperpod_submission(
                hyperpod_cluster,
                submission_key,
            )
        except NotFoundError:
            pass
print(json.dumps({
    # CPU Pods do not export GPU_FAULT_RELEASE_ID, so this is None in a real
    # deployment; RegionalLiveFixture.store_snapshot fills the key from the
    # gpu-fault-regional-release-state ConfigMap. Kept so an environment that
    # does set it (local runs) is still reported as-is.
    "release_id": os.getenv("GPU_FAULT_RELEASE_ID"),
    "event": (
        {
            **event.model_dump(mode="json"),
            **({"recovered_from": "raw_evidence"} if recovered else {}),
        }
        if event is not None else None
    ),
    "decision": (
        decision.model_dump(mode="json") if decision is not None else None
    ),
    "incident": (
        incident.model_dump(mode="json") if incident is not None else None
    ),
    "workflow": (
        workflow.model_dump(mode="json") if workflow is not None else None
    ),
    "commands": [redacted_command(item) for item in commands],
    "notifications": notifications,
    "observations": observations,
    "restart_budget": restart_budget,
    "agent": agent,
    "profile": profile,
    "submission": (
        submission.model_dump(mode="json")
        if submission is not None else None
    ),
    "queue": drained_queue_stats(store, attempts=queue_attempts),
    "remote_commands": store.remote_command_stats(),
}, sort_keys=True, default=str))
"""
