"""Resumable bounded-concurrency survey; the node list is never sampled away."""

from __future__ import annotations

import contextvars
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.hung_summary_codec import MAX_SURVEY_RAW_BYTES, encode_summary
from gpu_fault.models import WorkflowStepStatus
from gpu_fault.node_agent.protocol import NodeActionResult, NodeActionStatus

MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_RAW_RESULT_BYTES = MAX_SURVEY_RAW_BYTES
NODE_TIMEOUT_SECONDS = 30


def scoped_context(context: WorkflowStepContext, node: str) -> WorkflowStepContext:
    parameters = dict(context.step.parameters)
    for key in (
        "workload_cgroup_paths_by_node",
        "gpu_uuids_by_node",
        "rank_by_pid_by_node",
        "target_pids_by_node",
        "target_start_ticks_by_pid_by_node",
        "target_gpu_uuids_by_pid_by_node",
    ):
        mapping = parameters.get(key)
        if isinstance(mapping, dict):
            parameters[key] = {
                node: mapping.get(node, [] if key != "rank_by_pid_by_node" else {})
            }
    parameters.pop("attempt_node_ids", None)
    parameters.pop("hung_task_summary", None)
    parameters.pop("fabric_partition_by_node", None)
    parameters.pop("not_sampled_nodes", None)
    return replace(
        context,
        step=context.step.model_copy(
            update={"node_ids": [node], "parameters": parameters}
        ),
    )


def execute_hung_survey(
    adapter: Any, context: WorkflowStepContext
) -> WorkflowStepOutcome:
    previous = next(
        (
            item
            for item in reversed(context.workflow.step_executions)
            if item.step_index == context.step_index
            and item.operation is context.step.operation
        ),
        None,
    )
    old = previous.details if previous is not None else {}
    now = datetime.now(timezone.utc)
    started = (
        datetime.fromisoformat(old["survey_started_at"])
        if old.get("survey_started_at")
        else now
    )
    results = dict(old.get("node_results") or {})
    failures = dict(old.get("node_failures") or {})
    pending = dict(old.get("pending_since") or {})
    workers = min(32, adapter.max_parallel_node_actions)
    timeout = min(
        540,
        max(
            120,
            math.ceil(len(context.step.node_ids) / workers) * NODE_TIMEOUT_SECONDS + 60,
        ),
    )
    retained = int(old.get("retained_bytes", 0))
    raw_retained = int(old.get("retained_raw_bytes", 0))

    def send(node: str) -> NodeActionResult | WorkflowStepOutcome:
        scoped = scoped_context(context, node)
        mapping = scoped.step.parameters.get("gpu_uuids_by_node") or {}
        gpus = mapping.get(node, []) if isinstance(mapping, dict) else []
        return adapter._send_action(
            scoped, node, context.step.operation, gpus, command_suffix=node
        )

    slice_deadline = time.monotonic() + 5
    while True:
        unseen = [
            node
            for node in context.step.node_ids
            if node not in results and node not in failures and node not in pending
        ]
        now = datetime.now(timezone.utc)
        if (now - started).total_seconds() >= timeout:
            failures.update({node: "SURVEY_DEADLINE" for node in [*pending, *unseen]})
            pending.clear()
            break
        window = list(pending)[:workers] + unseen[: max(0, workers - len(pending))]
        if not window:
            break
        retained, raw_retained = collect_wave(
            window,
            send,
            results,
            failures,
            pending,
            retained,
            raw_retained,
            now,
            compress=True,
        )
        if time.monotonic() >= slice_deadline:
            break
        if pending:
            time.sleep(0.25)
    remaining = set(context.step.node_ids) - results.keys() - failures.keys()
    details = {
        "node_results": results,
        "node_failures": failures,
        "survey_started_at": started.isoformat(),
        "pending_since": pending,
        "remaining_node_count": len(remaining),
        "retained_bytes": retained,
        "retained_raw_bytes": raw_retained,
        "requested_node_count": len(context.step.node_ids),
        "summary_deadline_seconds": timeout,
        "max_inflight_nodes": workers,
        "snapshot_epoch": context.step.parameters.get("snapshot_epoch"),
        "undetermined_nodes": sorted(failures),
    }
    if remaining:
        return WorkflowStepOutcome.waiting(
            operation_id=context.idempotency_key, details=details
        )
    return WorkflowStepOutcome.succeeded(
        operation_id=context.idempotency_key, details=details
    )


def collect_wave(
    window: list[str],
    send: Callable[[str], NodeActionResult | WorkflowStepOutcome],
    results: dict[str, Any],
    failures: dict[str, str],
    pending: dict[str, str],
    retained: int,
    raw_retained: int,
    now: datetime,
    *,
    compress: bool = True,
) -> tuple[int, int]:
    with ThreadPoolExecutor(
        max_workers=len(window), thread_name_prefix="hung-survey"
    ) as pool:
        futures = {
            node: pool.submit(contextvars.copy_context().run, send, node)
            for node in window
        }
        for node, future in futures.items():
            response = future.result()
            sent_at = datetime.fromisoformat(pending.get(node, now.isoformat()))
            if (
                isinstance(response, WorkflowStepOutcome)
                and response.status is WorkflowStepStatus.WAITING
            ):
                if (
                    datetime.now(timezone.utc) - sent_at
                ).total_seconds() < NODE_TIMEOUT_SECONDS:
                    pending[node] = sent_at.isoformat()
                    continue
                failures[node] = "NODE_TIMEOUT"
            elif isinstance(response, NodeActionResult) and (
                response.status is NodeActionStatus.SUCCEEDED
                or response.details.get("evidence_complete") is False
            ):
                if response.status is not NodeActionStatus.SUCCEEDED:
                    failures[node] = (response.error or "PARTIAL_COLLECTION")[:512]
                try:
                    encoded = encode_summary(response.details)
                    document = encoded if compress else response.details
                except (TypeError, ValueError):
                    failures[node] = "INVALID_SUMMARY_RESPONSE"
                    pending.pop(node, None)
                    continue
                size = len(json.dumps(document, separators=(",", ":")).encode())
                if (
                    retained + size > MAX_RESULT_BYTES
                    or raw_retained + encoded["raw_bytes"] > MAX_RAW_RESULT_BYTES
                ):
                    failures[node] = "RESULT_BUDGET_EXCEEDED"
                else:
                    results[node] = document
                    retained += size
                    raw_retained += encoded["raw_bytes"]
            else:
                failures[node] = (response.error or "SUMMARY_UNAVAILABLE")[:512]
            pending.pop(node, None)
    return retained, raw_retained
