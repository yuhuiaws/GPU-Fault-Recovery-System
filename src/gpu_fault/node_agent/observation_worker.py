"""Read-only child process. No node reset, workload stop or service mutation API."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault.node_agent.diagnostic_process import DiagnosticProcessRunner
from gpu_fault.node_agent.observation_limits import (
    OBSERVATION_REQUEST_BUDGET_BYTES,
    OBSERVATION_RESULT_BUDGET_BYTES,
    RESULT_STAND_IN_RANKS_BUDGET_BYTES,
    WORKER_OUTPUT,
)
from gpu_fault.node_agent.operations.diagnostics import DiagnosticOperationsMixin
from gpu_fault.node_agent.operations.efa import EfaOperationsMixin
from gpu_fault.node_agent.operations.flight_recorder import (
    FlightRecorderOperationsMixin,
)
from gpu_fault.node_agent.operations.hung_process import HungProcessOperationsMixin
from gpu_fault.node_agent.operations.hung_summary import collect_hung_summary
from gpu_fault.node_agent.operations.hung_triage import HungTriageOperationsMixin
from gpu_fault.node_agent.protocol import NodeActionCommand
from gpu_fault.models import WorkflowOperation as Op

_PARTIAL: dict[str, Any] = {}


class ObservationAgent(
    HungTriageOperationsMixin,
    HungProcessOperationsMixin,
    FlightRecorderOperationsMixin,
    EfaOperationsMixin,
    DiagnosticOperationsMixin,
):
    def __init__(self, settings: dict[str, Any]):
        self.proc_root = Path(settings["proc_root"])
        self.infiniband_root = Path(settings["infiniband_root"])
        self.diagnostic_output_dir = Path(settings["diagnostic_output_dir"])
        self.python_stack_tool = settings.get("python_stack_tool", "")
        self.runner = DiagnosticProcessRunner(kill_grace_seconds=0.1)
        self.sleep = time.sleep
        self.now = lambda: datetime.now(timezone.utc)
        self.diagnostic_s3_uri = settings.get("diagnostic_s3_uri")
        self.diagnostic_retention_seconds = int(
            settings.get("diagnostic_retention_seconds", 604800)
        )
        self.diagnostic_max_archives = int(settings.get("diagnostic_max_archives", 20))
        self.expand_python_cgroup_processes = False
        self.partial_evidence_sink = lambda document: publish(document, final=False)


def rank_status(row: dict[str, Any]) -> str | None:
    """One word for a rank the stand-in names.

    The row's own ``status`` or ``error`` first (a summary row's
    ``PROCESS_CHANGED_OR_UNREADABLE``), else the recorder series' status
    (``PAIRED``/``SECOND_UNAVAILABLE``/...), else the recorder's
    (``dumped``/``not_configured``/``process_changed``), else a summary row's
    ``communication_status``.
    """

    candidates: list[Any] = [row.get("status"), row.get("error")]
    for key in ("flight_recorder_series", "flight_recorder"):
        nested = row.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get("status"))
    candidates.append(row.get("communication_status"))
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return None


def ranks_summary(
    document: dict[str, Any], budget: int = RESULT_STAND_IN_RANKS_BUDGET_BYTES
) -> dict[str, Any]:
    """pid, rank and status of as many of ``document``'s ranks as fit in
    ``budget`` bytes, in the result's own order; the total and whether the list
    stopped short are counted next to it."""

    ranks = document.get("ranks")
    rows = (
        [row for row in ranks if isinstance(row, dict)]
        if isinstance(ranks, list)
        else []
    )
    listed: list[dict[str, Any]] = []
    size = 2  # the brackets
    for row in rows:
        entry = {
            "pid": row.get("pid"),
            "rank": row.get("rank"),
            "status": rank_status(row),
        }
        weight = len(json.dumps(entry, separators=(",", ":")).encode()) + bool(listed)
        if size + weight > budget:
            break
        listed.append(entry)
        size += weight
    return {
        "ranks_summary": listed,
        "ranks_total": len(rows),
        "ranks_summary_truncated": len(listed) < len(rows),
    }


def oversized_result(size: int, document: dict[str, Any]) -> dict[str, Any]:
    """The stand-in for a result line the supervisor would refuse.

    Live (GF-GENERAL-LIVE-011 ``fr_enabled`` on HyperPod, 2026-10-25), a triage
    of eight ranks with two flight recorder dumps each weighed ~1.2 MB and came
    back as the bare ``{"status": "RESULT_SIZE_LIMIT", "ranks": []}``: the
    workflow recorded ``ObservationIncomplete: RESULT_SIZE_LIMIT`` for both
    nodes, nobody could tell an oversized result from a lost one, and the
    runner read "sampled no rank process" over eight sampled ranks. The
    stand-in names the size, the budget, the heaviest fields and the stage, in
    the shape every other refusal uses (``error``, ``failed_component``,
    ``failure_stage``, ``worker_started``, ``evidence_complete``), and it names
    the ranks it replaces: ``ranks_summary`` lists pid, rank and status for as
    many ranks as fit in ``RESULT_STAND_IN_RANKS_BUDGET_BYTES``, with
    ``ranks_total`` and ``ranks_summary_truncated`` beside it. The hung triage
    trims itself down to a skeleton of every rank before it gets here, so this
    is reached only by a document whose skeleton alone is over the line.
    """

    weights = sorted(
        (
            (len(json.dumps(value, separators=(",", ":"))), str(key))
            for key, value in document.items()
        ),
        reverse=True,
    )[:3]
    heaviest = ", ".join(f"{key}={weight} bytes" for weight, key in weights)
    return {
        "status": "RESULT_SIZE_LIMIT",
        "ranks": [],
        "evidence_complete": False,
        "error": (
            "observation result exceeds budget: "
            f"{size} > {OBSERVATION_RESULT_BUDGET_BYTES} bytes"
            + (f"; largest fields: {heaviest}" if heaviest else "")
        ),
        "failed_component": "observation_worker",
        "failure_stage": WORKER_OUTPUT,
        "worker_started": True,
        "result_bytes": size,
        **ranks_summary(document),
    }


def publish(document: dict[str, Any], *, final: bool) -> None:
    if not final:
        _PARTIAL.clear()
        _PARTIAL.update(document)
    raw = json.dumps(
        {"final": final, "result": document}, separators=(",", ":")
    ).encode()
    # The newline counts: the supervisor measures the line it buffers.
    if len(raw) + 1 > OBSERVATION_RESULT_BUDGET_BYTES:
        raw = json.dumps(
            {"final": final, "result": oversized_result(len(raw) + 1, document)},
            separators=(",", ":"),
        ).encode()
    sys.stdout.buffer.write(raw + b"\n")
    sys.stdout.buffer.flush()


def execute(request: dict[str, Any]) -> dict[str, Any]:
    agent = ObservationAgent(request["settings"])
    operation = request["operation"]
    if operation == "sample":
        from gpu_fault.node_agent.passive_communication import sample_communication

        return sample_communication(agent, request["target"], request["plan"])
    command = NodeActionCommand.model_validate(request["command"])
    expected = {
        "summary": Op.COLLECT_HUNG_SUMMARY,
        "triage": Op.COLLECT_HUNG_TRIAGE,
        "bundle": Op.COLLECT_DIAGNOSTIC_BUNDLE,
    }
    if command.operation is not expected.get(operation):
        raise ValueError("observation command operation mismatch")
    if operation == "bundle" and command.parameters.get("diagnostic_reason") not in {
        "EFA_TRAFFIC_HUNG_SUSPECTED",
        "TRAINING_HUNG",
        "APPLICATION_NCCL_ERROR",
    }:
        raise ValueError("observation worker only accepts passive hung bundles")
    if operation == "summary":
        return collect_hung_summary(agent, command)
    if operation == "triage":
        return agent._collect_hung_triage(command)
    if operation == "bundle":
        return agent._collect_diagnostic_bundle(command)
    raise ValueError("unsupported observation operation")


def main() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # GPU tooling may reserve large virtual mappings. Physical memory is
    # constrained by the verified service cgroup, not an arbitrary AS limit.
    resource.setrlimit(resource.RLIMIT_CPU, (8, 9))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    os.nice(10)
    raw = sys.stdin.buffer.read(OBSERVATION_REQUEST_BUDGET_BYTES + 1)
    if len(raw) > OBSERVATION_REQUEST_BUDGET_BYTES:
        raise ValueError(
            "observation request exceeds budget: more than "
            f"{OBSERVATION_REQUEST_BUDGET_BYTES} bytes on stdin"
        )
    request = json.loads(raw)
    publish({"status": "COLLECTING", "ranks": [], "read_only": True}, final=False)
    try:
        result = execute(request)
    except Exception as exc:
        result = {
            **_PARTIAL,
            "status": "UNAVAILABLE",
            "error": f"{type(exc).__name__}: {str(exc)[:256]}",
            "evidence_complete": False,
        }
    publish(result, final=True)


if __name__ == "__main__":
    main()
