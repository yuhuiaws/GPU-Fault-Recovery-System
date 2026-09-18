"""Workload identity, training, recovery and blast-radius evidence judgments."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from scripts.e2e.regional import run_destr009_workload_restart as workload_case

SUSPICIOUS_LOG_PATTERN = re.compile(
    r"\b(?:ERROR|Traceback|401|403|CERTIFICATE_VERIFY_FAILED)\b"
)


MAX_RECORDED_SUSPICIOUS_LINES = 20


LOSS_LINE_PATTERN = re.compile(r"\brank=(\d+)\b.*?\bstep=(\d+)\b.*?\bloss=(\S+)")


SUCCESS_LINE_PATTERN = re.compile(r"\bSUCCESS rank=(\d+)/(\d+)\b.*?\ball_reduce=(\S+)")
LOSS_RECORD_PREFIX = re.compile(r"rank=\d+[ \t]+step=\d+[ \t]+loss=")
SUCCESS_RECORD_PREFIX = re.compile(r"SUCCESS rank=\d+/\d+[ \t]")
TRAINING_RECORD_BOUNDARY = re.compile(
    rf"(?={LOSS_RECORD_PREFIX.pattern}|{SUCCESS_RECORD_PREFIX.pattern})"
)


def training_log_records(text: str) -> list[str]:
    # torchrun ranks share stdout: print's payload and newline can be interleaved.
    # Split only known record headers; malformed numeric payloads remain intact.
    return [
        record
        for line in text.splitlines()
        for record in TRAINING_RECORD_BOUNDARY.split(line)
        if record
    ]


def identity_baseline_errors(state: dict[str, Any]) -> list[str]:
    """Why ``state`` is not the empty pre-fault baseline for a job/attempt.

    ISO-001 step 5 and E2E-001 both require that nothing has been recorded for
    the identity yet: a restart budget, a decision or an observation left by an
    earlier run would be read as this run's isolation result.
    """

    errors = []
    required_fields = {
        "cluster_id",
        "restart_budget",
        "decision",
        "observations",
        "incidents",
        "workflows",
        "commands",
    }
    if not required_fields <= state.keys() or any(
        not isinstance(state.get(field), list)
        for field in ("observations", "incidents", "workflows", "commands")
    ):
        return ["control-plane identity baseline is incomplete"]
    if state.get("restart_budget") is not None:
        errors.append("restart budget already exists")
    if state.get("decision") is not None:
        errors.append("attempt decision already exists")
    if state.get("observations"):
        errors.append(f"{len(state['observations'])} attempt observations exist")
    if state.get("incidents") or state.get("workflows"):
        errors.append("an incident or workflow already references the job")
    if state.get("commands"):
        errors.append("remote commands already reference the job")
    return errors


def metadata_errors(
    workload: dict[str, Any],
    *,
    job_id: str,
    attempt_id: str,
    profile_version: str,
) -> list[str]:
    errors = []
    labels = workload["metadata"].get("labels", {})
    expected_labels = {
        "gpu-fault.io/managed": "true",
        "gpu-fault.io/job-id": job_id,
        "gpu-fault.io/attempt-id": attempt_id,
    }
    for key, value in expected_labels.items():
        if labels.get(key) != value:
            errors.append(f"workload label {key} differs")
    replicas = workload["spec"]["pytorchReplicaSpecs"]
    for role, expected_offset in (("Master", "0"), ("Worker", "1")):
        template = replicas[role]["template"]["metadata"]
        template_labels = template.get("labels", {})
        annotations = template.get("annotations", {})
        for key, value in {
            **expected_labels,
            "gpu-fault.io/critical": "true",
            "gpu-fault.io/role": role.lower(),
        }.items():
            if template_labels.get(key) != value:
                errors.append(f"{role} template label {key} differs")
        if annotations.get("gpu-fault.io/expected-critical-ranks") != "3":
            errors.append(f"{role} expected critical ranks differs")
        if annotations.get("gpu-fault.io/rank-offset") != expected_offset:
            errors.append(f"{role} rank offset differs")
        if annotations.get("gpu-fault.io/runtime-profile-version") != profile_version:
            errors.append(f"{role} runtime profile differs")
        if annotations.get("gpu-fault.io/restart-budget") != "1":
            errors.append(f"{role} restart budget differs")
        if annotations.get("gpu-fault.io/training-container") != "pytorch":
            errors.append(f"{role} training container differs")
    return errors


def suspicious_log_lines(
    text: str, *, limit: int = MAX_RECORDED_SUSPICIOUS_LINES
) -> list[dict[str, Any]]:
    """Executor log lines that hit a spec marker on a word boundary.

    Returns the marker and the line (truncated) so the evidence shows what
    matched rather than only that something did.
    """

    hits: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = SUSPICIOUS_LOG_PATTERN.search(line)
        if match is None:
            continue
        hits.append({"line": number, "marker": match.group(0), "text": line[:300]})
        if len(hits) >= limit:
            break
    return hits


def loss_errors(
    logs: dict[str, str], *, world_size: int = 24, steps: int = 3
) -> list[str]:
    """Why the fixture's per-rank ``loss=`` lines do not show a converging model.

    Every rank trains the same Linear(16, 1) on constant data with SGD, so the
    loss it prints must not increase from one step to the next; a rank whose
    loss climbs, or a Pod that printed none, is a training defect the SUCCESS
    line alone would hide.
    """

    errors = []
    owners: dict[int, str] = {}
    measured: dict[int, list[tuple[int, float]]] = {}
    for pod, text in sorted(logs.items()):
        series: dict[int, list[tuple[int, float]]] = {}
        for line in training_log_records(text):
            match = LOSS_LINE_PATTERN.search(line)
            if match is None:
                if LOSS_RECORD_PREFIX.match(line):
                    errors.append(f"{pod}: incomplete loss record")
                continue
            try:
                value = float(match.group(3))
            except ValueError:
                errors.append(f"{pod}: unparseable loss for rank {match.group(1)}")
                continue
            if not math.isfinite(value):
                errors.append(f"{pod}: non-finite loss for rank {match.group(1)}")
                continue
            series.setdefault(int(match.group(1)), []).append(
                (int(match.group(2)), value)
            )
        if not series:
            errors.append(f"{pod}: no loss lines")
            continue
        for rank, points in sorted(series.items()):
            if rank in owners:
                errors.append(f"rank {rank} appears in more than one Pod")
            owners[rank] = pod
            measured[rank] = points
            if sorted(step for step, _value in points) != list(range(steps)):
                errors.append(f"{pod}: rank {rank} does not have exactly {steps} steps")
                continue
            ordered = [value for _step, value in sorted(points)]
            if any(later > earlier for earlier, later in zip(ordered, ordered[1:])):
                errors.append(f"{pod}: rank {rank} loss increased: {ordered}")
            elif ordered[-1] >= ordered[0]:
                errors.append(f"{pod}: rank {rank} final loss did not decrease")
    if set(measured) != set(range(world_size)):
        errors.append(f"loss ranks do not cover exactly 0..{world_size - 1}")
    return errors


def collective_errors(logs: dict[str, str], *, world_size: int = 24) -> list[str]:
    expected = world_size * (world_size + 1) / 2
    seen: set[int] = set()
    errors = []
    for pod, text in sorted(logs.items()):
        matches = []
        for record in training_log_records(text):
            found = list(SUCCESS_LINE_PATTERN.finditer(record))
            if not found and SUCCESS_RECORD_PREFIX.match(record):
                errors.append(f"{pod}: incomplete collective SUCCESS record")
            matches.extend(found)
        if not matches:
            errors.append(f"{pod}: no collective SUCCESS records")
        for match in matches:
            rank, reported_world = int(match.group(1)), int(match.group(2))
            try:
                total = float(match.group(3))
            except ValueError:
                total = math.nan
            if reported_world != world_size or total != expected:
                errors.append(f"{pod}: rank {rank} collective differs")
            if rank in seen:
                errors.append(f"rank {rank} has duplicate collective SUCCESS records")
            seen.add(rank)
    if seen != set(range(world_size)):
        errors.append(f"collective ranks do not cover exactly 0..{world_size - 1}")
    return errors


def observation_rank_errors(
    observation: dict[str, Any],
    *,
    expected_ranks: set[int],
) -> list[str]:
    """The observation's containers must cover exactly the expected ranks."""

    ranks = [container.get("rank") for container in observation.get("containers") or []]
    if any(rank is None for rank in ranks):
        return ["an observed container has no rank"]
    actual = {int(rank) for rank in ranks if rank is not None}
    if len(actual) != len(ranks):
        return [f"duplicate container ranks: {sorted(ranks)}"]
    if actual != expected_ranks:
        return [f"container ranks {sorted(actual)} != {sorted(expected_ranks)}"]
    return []


def observation_pod_names(observation: dict[str, Any]) -> set[str]:
    return {
        str(container.get("pod_name") or "")
        for container in observation.get("containers") or []
    }


def observation_pod_uids(observation: dict[str, Any]) -> set[str]:
    return {
        str(container.get("pod_uid") or "")
        for container in observation.get("containers") or []
    }


def virtual_isolation_errors(
    virtual_states: Sequence[dict[str, Any]],
    *,
    sources: Sequence[dict[str, Any]],
    cluster_ids: Sequence[str],
    shared_node: str,
) -> list[str]:
    """Why the same virtual node name did not stay cluster-scoped.

    Both clusters posted an observation naming ``shared_node``; each store
    scope must hold exactly one, stamped with its own cluster_id and carrying
    its own Pods (names, UIDs) and GPUs -- and none of the other side's. A
    check that only compared node_id sets passed when B's observation had been
    overwritten by A's, because both said ``shared_node``.
    """

    errors = []
    if not (len(virtual_states) == len(sources) == len(cluster_ids) == 2):
        return ["virtual isolation needs exactly two clusters"]
    own_uids = [{str(item["uid"]) for item in source["pods"]} for source in sources]
    own_names = [{str(item["name"]) for item in source["pods"]} for source in sources]
    observations: list[dict[str, Any]] = []
    for index, (state, cluster_id) in enumerate(zip(virtual_states, cluster_ids)):
        items = state.get("observations") or []
        if len(items) != 1:
            errors.append(f"{cluster_id}: {len(items)} observations, expected 1")
            observations.append({})
            continue
        observation = items[0]
        observations.append(observation)
        if state.get("cluster_id") not in (None, cluster_id):
            errors.append(f"{cluster_id}: store probe answered for another cluster")
        if observation.get("cluster_id") != cluster_id:
            errors.append(
                f"{cluster_id}: observation cluster_id is {observation.get('cluster_id')!r}"
            )
        nodes = {
            container.get("node_id")
            for container in observation.get("containers") or []
        }
        if nodes != {shared_node}:
            errors.append(
                f"{cluster_id}: container node IDs are {sorted(map(str, nodes))}"
            )
        if observation_pod_uids(observation) != own_uids[index]:
            errors.append(f"{cluster_id}: observation Pod UIDs are not its own")
        if observation_pod_names(observation) != own_names[index]:
            errors.append(f"{cluster_id}: observation Pod names are not its own")
        if state.get("decision") is not None:
            errors.append(f"{cluster_id}: pre-fault decision is not 404")
        if state.get("restart_budget") is not None:
            errors.append(f"{cluster_id}: pre-fault restart budget is not 404")
    if all(observations):
        a_gpus = workload_case.observation_gpu_uuids(observations[0])
        b_gpus = workload_case.observation_gpu_uuids(observations[1])
        if a_gpus & b_gpus:
            errors.append("GPU UUIDs are shared between the two cluster scopes")
        if observation_pod_uids(observations[0]) & observation_pod_uids(
            observations[1]
        ):
            errors.append("Pod UIDs are shared between the two cluster scopes")
    return errors


def command_ids_in_logs(command_ids: Sequence[str], logs: dict[str, str]) -> list[str]:
    """Which of ``command_ids`` appear in any of the executor ``logs``."""

    found = set()
    for text in logs.values():
        for command_id in command_ids:
            if command_id and command_id in text:
                found.add(command_id)
    return sorted(found)


def control_plane_blast_errors(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    """Why the control-plane EKS is not identical across the fault window.

    Nodes and gpu-fault Jobs must match exactly. Eviction events are judged
    as "none new": the whole event set was compared before, and it grew
    whenever an unrelated event aged out of the API server between the two
    snapshots -- a FAIL with nothing to do with the workflow.
    """

    errors = []
    if any(
        not isinstance(snapshot.get("nodes"), dict)
        or not snapshot["nodes"]
        or not isinstance(snapshot.get("gpu_fault_jobs"), list)
        or not isinstance(snapshot.get("eviction_events"), list)
        for snapshot in (before, after)
    ):
        return ["control-plane blast snapshot is incomplete"]
    if before.get("nodes") != after.get("nodes"):
        errors.append("CPU node taints/cordon/gpu-fault metadata changed")
    if before.get("gpu_fault_jobs") != after.get("gpu_fault_jobs"):
        errors.append("gpu-fault labelled Jobs on the control plane changed")
    new_events = [
        item
        for item in after.get("eviction_events") or []
        if item not in (before.get("eviction_events") or [])
    ]
    if new_events:
        errors.append(f"{len(new_events)} new eviction events in the window")
    return errors


def source_observation_errors(
    observation: dict[str, Any],
    source: dict[str, Any],
    *,
    cluster_id: str,
    job_id: str,
    attempt_id: str,
) -> list[str]:
    pods = source.get("pods") or []
    errors = []
    if (
        not pods
        or observation.get("cluster_id") != cluster_id
        or observation.get("job_id") != job_id
        or observation.get("attempt_id") != attempt_id
        or observation.get("workload_phase") != "RUNNING"
        or observation_pod_uids(observation) != {item.get("uid") for item in pods}
        or observation_pod_names(observation) != {item.get("name") for item in pods}
        or {item.get("node_id") for item in observation.get("containers") or []}
        != {item.get("node") for item in pods}
        or not observation.get("workload_ids")
    ):
        errors.append("source Observation does not bind the running workload")
    try:
        observed = datetime.fromisoformat(
            str(observation["observed_at"]).replace("Z", "+00:00")
        )
        age = (datetime.now(timezone.utc) - observed).total_seconds()
        if not 0 <= age <= 120:
            errors.append("source Observation is stale or future-dated")
    except (KeyError, ValueError, TypeError):
        errors.append("source Observation freshness is unknown")
    return errors


def recovery_identity_errors(
    state: dict[str, Any],
    *,
    cluster_id: str,
    job_id: str,
    attempt_id: str,
    node: str,
    marker: str,
    observed_after: datetime,
    restart_withheld: bool = False,
) -> list[str]:
    """Bind the successful recovery to this injection, including compound commands."""

    errors = []
    event, decision = state.get("event") or {}, state.get("decision") or {}
    incident, workflow = state.get("incident") or {}, state.get("workflow") or {}
    event_id, incident_id = event.get("event_id"), incident.get("incident_id")
    workflow_id = workflow.get("request_id")
    if (
        not event_id
        or event.get("cluster_id") != cluster_id
        or event.get("node_id") != node
        or marker not in str(event.get("raw_message") or "")
        or event.get("job_id") != job_id
        or event.get("attempt_id") != attempt_id
        or decision.get("event_id") != event_id
    ):
        errors.append("event/decision does not bind the injected workload fault")
    try:
        event_time = datetime.fromisoformat(
            str(event["observed_at"]).replace("Z", "+00:00")
        )
        if event_time < observed_after or event_time > datetime.now(timezone.utc):
            errors.append("fault event is outside this injection window")
    except (KeyError, TypeError, ValueError):
        errors.append("fault event time is unknown")
    if (
        not incident_id
        or not workflow_id
        or incident.get("event_id") != event_id
        or incident.get("cluster_id") != cluster_id
        or incident.get("job_id") != job_id
        or incident.get("attempt_id") != attempt_id
        or node not in (incident.get("node_ids") or [])
        or incident.get("workflow_request_id") != workflow_id
        or workflow.get("incident_id") != incident_id
    ):
        errors.append("incident/workflow does not bind the injected workload fault")
    for key in ("official_steps", "safety_steps", "step_executions"):
        if any(
            step.get("operation") in workload_case.FORBIDDEN_OPERATIONS
            for step in workflow.get(key) or []
        ):
            errors.append("workload recovery contains a forbidden node operation")
    commands = state.get("commands") or []
    covered: dict[tuple[int, str], str] = {}
    seen = set()
    for command in commands:
        command_id = command.get("command_id")
        if (
            not command_id
            or command_id in seen
            or command.get("cluster_id") != cluster_id
            or command.get("incident_id") != incident_id
            or command.get("workflow_request_id") != workflow_id
            or command.get("status") != "SUCCEEDED"
            or not (command.get("last_lease_owner") or command.get("lease_owner"))
        ):
            errors.append(
                "remote command identity, scope or terminal state is unproven"
            )
        seen.add(command_id)
        steps = [
            {
                "step_index": command.get("step_index"),
                "step": command.get("step") or {},
            },
            *(command.get("batched_steps") or []),
        ]
        for entry in steps:
            index = entry.get("step_index")
            operation = (entry.get("step") or {}).get("operation")
            if type(index) is not int or not isinstance(operation, str):
                errors.append("remote command step identity is missing")
                continue
            step_key = (index, operation)
            if step_key in covered:
                errors.append("multiple remote commands cover the same workflow step")
            covered[step_key] = f"remote/{command_id}"
    executions = workflow.get("step_executions") or []
    required_remote = (
        ("STOP_WORKLOADS",)
        if restart_withheld
        else ("STOP_WORKLOADS", "RESTART_WORKLOAD")
    )
    for operation in required_remote:
        terminal = [
            entry
            for entry in executions
            if entry.get("operation") == operation
            and entry.get("status") == "SUCCEEDED"
        ]
        if not terminal or any(
            covered.get((entry.get("step_index"), operation))
            != entry.get("adapter_operation_id")
            for entry in terminal
        ):
            errors.append(f"{operation} completion does not bind its remote command")
    budget = state.get("restart_budget") or {}
    if (
        budget.get("cluster_id") != cluster_id
        or budget.get("job_id") != job_id
        or type(budget.get("restart_count")) is not int
        or budget["restart_count"] != (0 if restart_withheld else 1)
    ):
        errors.append("restart budget does not bind the target cluster/job")
    return errors


def notification_errors(state: dict[str, Any]) -> list[str]:
    notifications = state.get("notifications") or []
    categories: dict[str, list[dict[str, Any]]] = {}
    for item in notifications:
        notification = item.get("notification") or {}
        result = item.get("result") or {}
        categories.setdefault(str(notification.get("category")), []).append(result)
    errors = []
    incident_id = (state.get("incident") or {}).get("incident_id")
    notification_ids = [
        (item.get("notification") or {}).get("notification_id")
        for item in notifications
    ]
    if (
        not incident_id
        or None in notification_ids
        or "" in notification_ids
        or len(set(notification_ids)) != len(notification_ids)
        or any(
            (item.get("notification") or {}).get("incident_id") != incident_id
            for item in notifications
        )
    ):
        errors.append("notifications do not uniquely bind the recovery incident")
    for category in ("FAULT_DETECTED", "ACTION_COMPLETED"):
        values = categories.get(category) or []
        if len(values) != 1:
            errors.append(f"{category} notification count is not one")
            continue
        if values[0].get("status") != "SENT":
            errors.append(f"{category} notification is not SENT")
        if not values[0].get("provider_message_id"):
            errors.append(f"{category} notification has no provider message ID")
    return errors


def gpu_nodes_clean(nodes: Sequence[dict[str, Any]]) -> bool:
    """Every GPU node Ready, schedulable (spares excepted) and free of our taints."""

    return bool(nodes) and all(
        item["ready"] == "True"
        and (
            not item["unschedulable"]
            # spare-pool nodes stay cordoned by design (see e2e_preflight)
            or (item.get("labels") or {}).get("gpu-fault.io/spare") == "true"
        )
        and not any(
            str(taint.get("key", "")).startswith("gpu-fault.io/")
            for taint in item.get("taints") or []
        )
        for item in nodes
    )
