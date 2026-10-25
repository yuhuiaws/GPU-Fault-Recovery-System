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
from gpu_fault.node_agent.observation_limits import OBSERVATION_REQUEST_BUDGET_BYTES
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


def publish(document: dict[str, Any], *, final: bool) -> None:
    if not final:
        _PARTIAL.clear()
        _PARTIAL.update(document)
    raw = json.dumps(
        {"final": final, "result": document}, separators=(",", ":")
    ).encode()
    if len(raw) > 1_048_576:
        raw = json.dumps(
            {"final": final, "result": {"status": "RESULT_SIZE_LIMIT", "ranks": []}}
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
