"""GF-GENERAL-LIVE-017 CRI Pod-log attribution: product-record readers (pure).

What the 017 CRI variants (``general_live_applogs``: ``live_pid``,
``exited_pid``, ``old_pod``, ``cri_partial_rotation``) read off the product's
own records -- the attempt observation the completion watcher recorded, the
delivered NODE_LOGS entries (``pod_uid``/``container_name``, the CRI event
time), the incident the product minted for an entry (by the event id
``log_recovery_context.log_event_id`` derives) and the diagnostic bundle step's
attribution parameters -- and the stage readers that turn them into errors.
Nothing here touches a cluster or a node; every function takes the documents
``general_live_store.applog_records`` returned.

A NO_ACTION application-log line is kept by the product as its NODE_LOGS record
only (``app/ingest/node_health.py`` skips a ``SITE_APPLICATION_LOG`` NO_ACTION
finding: no incident, marker or workflow), so ``evidence_only_errors`` and
``attributable_errors`` read the record and the attempt observation where
``attribution_errors`` reads an incident.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from scripts.e2e.regional.general_live_verdicts import (
    final_execution,
    operations,
    parse_time,
    step_parameters,
)

DIAGNOSTIC_WORKFLOW_OPS = frozenset(
    {"RUN_DIAGNOSTICS", "COLLECT_DIAGNOSTIC_BUNDLE", "RUN_DCGM_DIAGNOSTIC"}
)
STEPS_WITH_DIAGNOSTIC_PARAMETERS = ("COLLECT_DIAGNOSTIC_BUNDLE", "COLLECT_HUNG_TRIAGE")


def safe_component(value: str) -> str:
    """``HostHealthEvaluator._safe``: the event-id component sanitizer."""

    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value)[:160]


def log_event_id(cluster_id: str, node_id: str, entry_id: str) -> str:
    """``log_recovery_context.log_event_id`` recomputed from a delivered entry."""

    digest = hashlib.sha256(
        json.dumps(
            [cluster_id, node_id, entry_id], ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:32]
    return f"log-{safe_component(cluster_id)}-{safe_component(node_id)}-{digest}"


def entries_of(records: dict[str, Any], paths: set[str]) -> list[dict[str, Any]]:
    """The delivered NODE_LOGS entries read from any of the fixture ``paths``."""

    return [
        item
        for item in records.get("entries") or []
        if str((item.get("fields") or {}).get("path")) in paths
    ]


def incident_of(
    records: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any] | None:
    """The incident the product minted for ``entry`` (by its derived event id)."""

    wanted = log_event_id(
        str(records.get("cluster_id")),
        str(records.get("node_id")),
        str(entry.get("entry_id")),
    )
    for incident in records.get("incidents") or []:
        if incident.get("event_id") == wanted:
            return incident
    return None


def workflow_of(
    records: dict[str, Any], incident: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not incident or not incident.get("workflow_request_id"):
        return None
    value = (records.get("workflows") or {}).get(str(incident["workflow_request_id"]))
    return dict(value) if isinstance(value, dict) else None


def attempt_of(records: dict[str, Any], attempt_id: str) -> dict[str, Any] | None:
    for item in records.get("attempts") or []:
        if item.get("attempt_id") == attempt_id:
            return item
    return None


def container_of(
    attempt: dict[str, Any] | None, pod_uid: str, container: str
) -> dict[str, Any] | None:
    for item in (attempt or {}).get("containers") or []:
        if item.get("pod_uid") == pod_uid and item.get("container_name") == container:
            return item
    return None


def diagnostic_parameters(workflow: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The finding's attribution parameters, as copied onto the bundle steps."""

    if not workflow:
        return []
    return [
        parameters
        for operation in STEPS_WITH_DIAGNOSTIC_PARAMETERS
        for parameters in step_parameters(workflow, operation)
    ]


def capture_steps(workflow: dict[str, Any] | None) -> list[str]:
    """Steps that would sample a process: capture requested or PIDs targeted."""

    if not workflow:
        return []
    return [
        str(step.get("operation"))
        for step in workflow.get("official_steps") or []
        if (step.get("parameters") or {}).get("capture_process_state")
        or (step.get("parameters") or {}).get("target_pids_by_node")
    ]


def bundle_node_result(
    execution: dict[str, Any] | None, node: str
) -> dict[str, Any] | None:
    """The Node Agent's COLLECT_DIAGNOSTIC_BUNDLE receipt for ``node``.

    A node-agent step is recorded per node under ``details.node_results[<node>]``
    (``adapters/node_action/step_execution``): ``evidence_ref``, ``sha256``,
    ``size_bytes`` and the ``manifest_summary`` the agent wrote
    (``capture_count``, ``failed_capture_count``, ``diagnostic_reason``,
    ``capture_process_state``). Nothing is hoisted to the step's own
    ``details``. Live (HyperPod, 2026-10-26) a bundle of 25 captures and 0
    failures -- py-spy, ``/proc`` and strace of the targeted host PID -- read
    as "did not capture the live process state" while this oracle looked for
    ``details.manifest_summary``.
    """

    results = ((execution or {}).get("details") or {}).get("node_results")
    value = results.get(node) if isinstance(results, dict) else None
    return dict(value) if isinstance(value, dict) else None


def entry_identity_errors(
    entry: dict[str, Any] | None,
    *,
    pod_uid: str,
    container: str,
    stamp: str | None,
    message: str | None = None,
) -> list[str]:
    """The delivered entry names the genuine Pod/container and the CRI time."""

    if entry is None:
        return ["no fixture line was delivered"]
    errors = []
    fields = entry.get("fields") or {}
    if entry.get("source") != "training-log":
        errors.append(f"entry source {entry.get('source')!r} is not training-log")
    if fields.get("pod_uid") != pod_uid:
        errors.append(f"entry pod_uid {fields.get('pod_uid')!r} != {pod_uid}")
    if fields.get("container_name") != container:
        errors.append(
            f"entry container {fields.get('container_name')!r} != {container}"
        )
    if fields.get("timestamp_source") != "cri":
        errors.append(f"timestamp source {fields.get('timestamp_source')!r} is not cri")
    event_at = parse_time(entry.get("observed_at"))
    stamped = parse_time(stamp) if stamp else None
    if event_at is None or stamped is None:
        errors.append("the entry or the fixture stamp has no readable time")
    elif abs((event_at - stamped).total_seconds()) > 1.0:
        errors.append(
            f"event time {entry.get('observed_at')} is not the CRI stamp {stamp}"
        )
    if message is not None and entry.get("message") != message:
        errors.append(f"entry message {entry.get('message')!r} != {message!r}")
    return errors


def attribution_errors(
    incident: dict[str, Any] | None,
    workflow: dict[str, Any] | None,
    *,
    job_id: str,
    attempt_id: str,
    process_state: str,
    node: str,
    pid: int | None,
) -> list[str]:
    """The incident names the attempt; the bundle step carries the attribution."""

    if incident is None:
        return ["the delivered line produced no incident"]
    errors = []
    if incident.get("job_id") != job_id or incident.get("attempt_id") != attempt_id:
        errors.append(
            f"incident attributed to {incident.get('job_id')}/{incident.get('attempt_id')}"
            f", expected {job_id}/{attempt_id}"
        )
    parameters = diagnostic_parameters(workflow)
    if process_state == "LIVE":
        if not parameters:
            errors.append(
                "no diagnostic bundle step carries the attribution parameters"
            )
        for item in parameters:
            if item.get("attribution") != "OBSERVED_PROCESS":
                errors.append(
                    f"attribution {item.get('attribution')!r} != OBSERVED_PROCESS"
                )
            if item.get("process_state") != "LIVE":
                errors.append(f"process_state {item.get('process_state')!r} != LIVE")
            if item.get("log_source") != "training-log":
                errors.append(f"log_source {item.get('log_source')!r} != training-log")
            targets = (item.get("target_pids_by_node") or {}).get(node)
            if pid is not None and targets != [pid]:
                errors.append(f"target pids {targets!r} != [{pid}] on {node}")
            if item.get("capture_process_state") is not True:
                errors.append("capture_process_state is not set for a live process")
    return errors


def diagnostic_chain_errors(
    incident: dict[str, Any] | None,
    workflow: dict[str, Any] | None,
    *,
    expect_bundle: bool,
    node: str | None = None,
) -> list[str]:
    """RUN_DIAGNOSTICS; the bundle ran for a live process and never for an exited one.

    ``node`` is the node whose agent collected the bundle (required when one is
    expected): its receipt under ``details.node_results[<node>]`` says whether
    the process capture ran (``manifest_summary.capture_process_state``), and
    a node the step has no receipt for is named with the failure or skip the
    step recorded for it instead of being read as a bundle without one.
    """

    if incident is None:
        return ["no incident"]
    errors = []
    if incident.get("effective_action") != "RUN_DIAGNOSTICS":
        errors.append(
            f"effective action {incident.get('effective_action')!r} != RUN_DIAGNOSTICS"
        )
    if workflow is None:
        return [*errors, "the incident has no workflow"]
    ops = operations(workflow)
    if expect_bundle:
        if node is None:
            raise ValueError("expect_bundle needs the node whose agent ran the bundle")
        if "COLLECT_DIAGNOSTIC_BUNDLE" not in ops:
            errors.append(f"no COLLECT_DIAGNOSTIC_BUNDLE for a live process: {ops}")
        execution = final_execution(workflow, "COLLECT_DIAGNOSTIC_BUNDLE")
        if execution is None or execution.get("status") != "SUCCEEDED":
            errors.append(
                "the diagnostic bundle did not succeed: "
                f"{(execution or {}).get('status')} {(execution or {}).get('error')}"
            )
        else:
            result = bundle_node_result(execution, node)
            details = execution.get("details") or {}
            if result is None:
                cause = (details.get("node_failures") or {}).get(node) or details.get(
                    "skipped"
                )
                errors.append(
                    f"the bundle step recorded no result for {node}"
                    + (f": {cause}" if cause else "")
                )
            elif (result.get("manifest_summary") or {}).get(
                "capture_process_state"
            ) is not True:
                errors.append(
                    f"the bundle on {node} did not run the live process capture: "
                    f"manifest_summary {result.get('manifest_summary')!r}"
                )
    else:
        sampled = capture_steps(workflow)
        if sampled:
            errors.append(
                f"an exited process was still scheduled for sampling: {sampled}"
            )
        if "COLLECT_DIAGNOSTIC_BUNDLE" in ops:
            errors.append(
                "a diagnostic bundle (stack capture) was planned for an exited PID"
            )
    return errors


def stop_errors(workflow: dict[str, Any] | None) -> list[str]:
    if workflow and "STOP_WORKLOADS" in operations(workflow):
        return [f"workflow {workflow.get('request_id')} stopped the attempt"]
    return []


def no_diagnostic_errors(workflow: dict[str, Any] | None) -> list[str]:
    found = sorted(set(operations(workflow or {})) & DIAGNOSTIC_WORKFLOW_OPS)
    return [f"an evidence-only line ran a diagnostic workflow {found}"] if found else []


def new_attempt_errors(
    incident: dict[str, Any] | None,
    workflow: dict[str, Any] | None,
    *,
    new_attempt_id: str,
    new_pid: int | None,
    node: str,
) -> list[str]:
    """An old Pod's line never reaches the new attempt or its live PID."""

    if incident is None:
        return ["the delivered line produced no incident"]
    errors = []
    if incident.get("attempt_id") == new_attempt_id:
        errors.append(
            f"the old Pod's line was attributed to the new attempt {new_attempt_id}"
        )
    for item in diagnostic_parameters(workflow):
        targets = (item.get("target_pids_by_node") or {}).get(node) or []
        if new_pid is not None and new_pid in targets:
            errors.append(f"the new attempt's PID {new_pid} was targeted for sampling")
        if item.get("process_state") == "LIVE":
            errors.append("an old Pod's line was graded as a LIVE process")
    return errors


def stitch_errors(
    entries: list[dict[str, Any]],
    *,
    message: str,
    rotated_path: str,
    head_stamp: str | None,
) -> list[str]:
    """One record with the whole message, at the P fragment's time, on the rotated inode."""

    if not entries:
        return ["no record was delivered from the fixture file"]
    errors = []
    whole = [item for item in entries if item.get("message") == message]
    if len(whole) != 1:
        errors.append(
            f"{len(whole)} records carry the stitched message, expected exactly one"
        )
    fragments = [
        item.get("message") for item in entries if item.get("message") != message
    ]
    if fragments:
        errors.append(f"fragments were delivered on their own: {fragments}")
    if whole:
        entry = whole[0]
        if str((entry.get("fields") or {}).get("path")) != rotated_path:
            errors.append(
                f"the stitched record was read from {(entry.get('fields') or {}).get('path')}"
                f", not the rotated file {rotated_path}"
            )
        event_at = parse_time(entry.get("observed_at"))
        stamped = parse_time(head_stamp) if head_stamp else None
        if event_at is None or stamped is None:
            errors.append("the stitched record or the P stamp has no readable time")
        elif abs((event_at - stamped).total_seconds()) > 1.0:
            errors.append(
                f"the stitched record's time {entry.get('observed_at')} is not the "
                f"P fragment's stamp {head_stamp}"
            )
    return errors


def evidence_only_errors(
    records: dict[str, Any], entry: dict[str, Any] | None
) -> list[str]:
    """A NO_ACTION application-log line is kept as its NODE_LOGS record only.

    The product mints no incident, marker or workflow for it:
    ``app/ingest/node_health.py`` ``NodeHealthIngestionService.ingest`` skips a
    ``SITE_APPLICATION_LOG`` finding whose action is NO_ACTION ("log evidence is
    already durably captured by the ingest transaction; workload errors are not
    hardware markers or a second recovery owner"). Live 2026-10-26 (HyperPod)
    ``cri_partial_rotation`` waited 420 s for "one NO_ACTION evidence incident"
    that this design never mints and failed on it. Incidents are decided in the
    ingest transaction that stores the record, so once the record is in the
    store the absence is final -- no wait. One minted anyway is the failure,
    named with its action and any diagnostic step it ran.
    """

    if entry is None:
        return ["no stitched record was delivered"]
    incident = incident_of(records, entry)
    if incident is None:
        return []
    errors = [
        f"the product minted incident {incident.get('incident_id')} "
        f"({incident.get('effective_action')}) for a NO_ACTION application-log "
        "line; node_health.ingest keeps such a line as the NODE_LOGS record only"
    ]
    errors.extend(no_diagnostic_errors(workflow_of(records, incident)))
    return errors


def attributable_errors(
    records: dict[str, Any],
    entry: dict[str, Any] | None,
    *,
    attempt_id: str,
    pod_uid: str,
    container: str,
    node: str,
) -> list[str]:
    """The delivered record names a live container of the attempt the watcher recorded.

    With no incident to read the attribution from (``evidence_only_errors``),
    the binding the product would make is checked where it reads it:
    ``log_recovery_context._resolve_log_attempt`` binds a training-log record
    to the attempt observation whose container on this node has the record's
    ``pod_uid`` and ``container_name`` (``_matches_process``).
    """

    if entry is None:
        return ["no stitched record was delivered"]
    errors = []
    observed = container_of(attempt_of(records, attempt_id), pod_uid, container)
    if observed is None:
        errors.append(
            f"the completion watcher recorded no container {container} of Pod "
            f"{pod_uid} for attempt {attempt_id}"
        )
    else:
        if observed.get("terminated"):
            errors.append(
                "the attempt's container was recorded terminated while its line "
                "was delivered"
            )
        if observed.get("node_id") not in (None, node):
            errors.append(
                f"the attempt's container is recorded on {observed.get('node_id')}, "
                f"not {node}"
            )
    fields = entry.get("fields") or {}
    if (fields.get("pod_uid"), fields.get("container_name")) != (pod_uid, container):
        errors.append(
            f"the record names Pod {fields.get('pod_uid')}/{fields.get('container_name')}"
            f", not the attempt's {pod_uid}/{container}"
        )
    return errors
