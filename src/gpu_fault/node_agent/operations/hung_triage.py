from __future__ import annotations

import math
import subprocess
import time
from typing import Any, Callable

from gpu_fault.node_agent.diagnostic_process import run_probe
from gpu_fault.node_agent.operations.collection_budget import (
    collection_lease,
    publish_partial,
)
from gpu_fault.node_agent.operations.flight_evidence import request_flight
from gpu_fault.node_agent.operations.flight_series import (
    finish_flight_series,
    triage_timeout,
)
from gpu_fault.node_agent.operations.process_identity import (
    bind_process_ranks,
    process_start_ticks,
    scoped_processes,
)
from gpu_fault.node_agent.operations.triage_budget import bounded_triage_result
from gpu_fault.node_agent.operations.triage_samples import collect_process_series
from gpu_fault.node_agent.operations.workload_scope import resolve_workload_scope
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
)


class HungTriageOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    _await_flight_dumps: Callable[..., Any]
    _capture_python_stack_sample: Any
    _counter_delta: Callable[..., Any]
    _efa_counter_snapshot: Callable[..., Any]
    _finish_flight_recorder: Callable[..., Any]
    _process_environment: Callable[..., Any]
    _trigger_flight_recorder: Callable[..., Any]
    proc_root: Any
    python_stack_tool: Any
    runner: Callable[..., Any]
    sleep: Callable[..., Any]

    def _collect_hung_triage(self, command: NodeActionCommand) -> dict[str, Any]:
        runtime = getattr(self, "observation_runtime", None)
        if runtime is not None:
            return dict(runtime.run_command("triage", command))
        with collection_lease(self, command) as refused:
            if refused:
                return {
                    "status": refused,
                    "ranks": [],
                    "errors": [refused],
                    "read_only": True,
                }
            return self._collect_hung_triage_admitted(command)

    def _collect_hung_triage_admitted(
        self, command: NodeActionCommand
    ) -> dict[str, Any]:
        """Collect a bounded, read-only NCCL hang signal snapshot."""

        started = time.monotonic()
        # The identity checks below all read this node's resolved scope.
        command, workload_scope = resolve_workload_scope(self.proc_root, command)
        timeout_seconds = triage_timeout(command.parameters)
        errors: list[str] = []
        process_rows: list[dict[str, Any]] = []
        inventory_complete = True
        query = [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = run_probe(
                self.runner,
                query,
                check=False,
                capture_output=True,
                text=True,
                timeout=min(3.0, timeout_seconds),
            )
            if completed.returncode != 0:
                inventory_complete = False
                errors.append(
                    "nvidia-smi compute query failed: "
                    + (completed.stderr or "").strip()
                )
            else:
                inventory_complete = len(completed.stdout.splitlines()) <= 4096
                for line in completed.stdout.splitlines()[:4096]:
                    fields = [item.strip() for item in line.split(",", 2)]
                    if len(fields) != 3 or not fields[0].isdigit():
                        inventory_complete = False
                        continue
                    process_rows.append(
                        {
                            "pid": int(fields[0]),
                            "gpu_uuid": fields[1],
                            "process_name": fields[2],
                        }
                    )
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            inventory_complete = False
            errors.append(f"compute query: {type(exc).__name__}: {exc}")

        clients: dict[int, list[str]] = {}
        for row in process_rows:
            clients.setdefault(row["pid"], []).append(row["gpu_uuid"])
        target_gpu_uuids = set(command.gpu_uuids)
        if target_gpu_uuids:
            process_rows = [
                row for row in process_rows if row["gpu_uuid"] in target_gpu_uuids
            ]
        process_rows, identities = scoped_processes(
            self.proc_root, command, process_rows, errors
        )
        bind_process_ranks(command, process_rows, self._process_environment)

        max_processes = min(
            32, max(1, int(command.parameters.get("triage_max_processes", 16)))
        )
        eligible_process_count = len(process_rows)
        process_rows = [
            row
            for row in process_rows[:max_processes]
            if identities[row["pid"]]["start_time_ticks"]
            == process_start_ticks(self.proc_root, row["pid"])
        ]
        publish_partial(
            self,
            {
                "status": "COLLECTING",
                "node_id": command.node_id,
                "ranks": [
                    {
                        "pid": row["pid"],
                        "rank": row.get("rank"),
                        "process_identity": identities[row["pid"]],
                    }
                    for row in process_rows
                ],
                "evidence_complete": False,
            },
        )
        flight_requests = {
            row["pid"]: request_flight(self, command, row, "first")
            for row in process_rows
        }
        efa_before = self._efa_counter_snapshot()
        ranks = collect_process_series(
            self,
            command,
            process_rows,
            identities,
            clients if inventory_complete else None,
            started + timeout_seconds / 2,
            time.monotonic,
        )
        if not process_rows and started + timeout_seconds - time.monotonic() >= 1:
            self.sleep(1)
        publish_partial(
            self,
            {
                "status": "COLLECTING",
                "node_id": command.node_id,
                "ranks": ranks,
                "errors": errors,
                "evidence_complete": False,
            },
        )
        efa_after = self._efa_counter_snapshot()
        for row in ranks:
            row["flight_recorder"] = (
                self._finish_flight_recorder(flight_requests.get(row["pid"], {}))
                if time.monotonic() < started + timeout_seconds
                else {"status": "BUDGET_EXHAUSTED"}
            )

        flight_wait_seconds = finish_flight_series(
            self,
            command,
            process_rows,
            ranks,
            identities,
            flight_requests,
            started,
            timeout_seconds,
            time.monotonic,
        )

        # Eight ranks with two FR dumps each weighed ~1.2 MB live against the
        # worker's 1 MiB result line and came back as nothing; the ladder in
        # ``triage_budget`` keeps every rank and names what it shortened.
        return bounded_triage_result(
            {
                "triage_version": "nccl-hung-triage/v4",
                "node_id": command.node_id,
                "ranks": ranks,
                "efa": self._counter_delta(efa_before, efa_after),
                "errors": errors,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "flight_recorder_wait_seconds": flight_wait_seconds,
                "read_only": True,
                "workload_scope": workload_scope,
                "process_coverage": {
                    "eligible": eligible_process_count,
                    "collected": len(ranks),
                    "truncated": eligible_process_count > max_processes,
                    "rank_inventory_complete": False,
                },
            }
        )

    def _gpu_triage_snapshot(
        self,
        timeout_seconds: float = 3,
    ) -> dict[str, dict[str, Any]]:
        argv = [
            "nvidia-smi",
            (
                "--query-gpu=uuid,utilization.gpu,"
                "utilization.memory,clocks_throttle_reasons.active,name"
            ),
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = run_probe(
                self.runner,
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return {}
        if completed.returncode != 0:
            return {}
        result = {}
        for line in completed.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 4:
                continue
            try:
                gpu, memory = float(fields[1]), float(fields[2])
                if not all(
                    math.isfinite(value) and 0 <= value <= 100
                    for value in (gpu, memory)
                ):
                    continue
                result[fields[0]] = {
                    "utilization_gpu_percent": gpu,
                    "utilization_memory_percent": memory,
                    "clocks_throttle_reasons_active": fields[3],
                    **({"device_model": fields[4]} if len(fields) > 4 else {}),
                }
            except ValueError:
                continue
        return result
