"""Agent-lifetime passive observation with short-lived control-plane scopes."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from typing import Any

from gpu_fault.communication_monitoring import CommunicationPlan
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent.observation_supervisor import (
    SLOT_BUSY,
    ObservationIncomplete,
    ObservationSlotBusy,
    ObservationSupervisor,
    slot_wait_for,
)
from gpu_fault.node_agent.operations.hung_summary_io import read_bounded
from gpu_fault.node_agent.observation_limits import (
    OBSERVATION_REQUEST_BUDGET_BYTES,
    service_limits,
)

#: Command parameters the worker never reads. A hung bundle copies them into
#: its manifest's ``fault_context`` and nothing else, so they are the only
#: parameters a request over the budget may lose. Everything else -- the PIDs
#: and start ticks the bundle attaches to, the raw flight recorder references,
#: the cgroup scope -- steers the collection and is never touched.
CONTEXT_ONLY_PARAMETERS = ("hung_triage_decision",)
REQUEST_BUDGET_OMISSION = "OBSERVATION_REQUEST_BUDGET"


def bounded_request(
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """``request`` within the worker's budget, and what was taken out of it.

    Live, the control plane bound a bundle to a 259 KB ``hung_triage_decision``
    (sixteen processes' per-thread stacks and wait observations; a churn thread
    added a few rows per process), the request weighed 264 KB against a 256 KiB
    budget and the supervisor refused it before any worker ran -- the whole
    bundle came back as ``ranks: []`` over a context field the node never
    reads. Over budget, each context-only parameter is replaced by a stand-in
    naming the omission, the byte count and the SHA-256 of what was omitted
    (the full value stays in the workflow record: the bundle step's parameters
    and the triage step's execution details), and the manifest lists the
    omitted keys under ``observation_request_trimmed``. A request still over
    budget after that is refused with its cause.
    """

    if len(json.dumps(request)) <= OBSERVATION_REQUEST_BUDGET_BYTES:
        return request, {}
    command = request.get("command")
    if not isinstance(command, dict) or not isinstance(command.get("parameters"), dict):
        return request, {}
    trimmed: dict[str, dict[str, Any]] = {}
    parameters = dict(command["parameters"])
    for key in CONTEXT_ONLY_PARAMETERS:
        if key not in parameters:
            continue
        encoded = json.dumps(
            parameters[key], sort_keys=True, separators=(",", ":")
        ).encode()
        trimmed[key] = {
            "omitted": REQUEST_BUDGET_OMISSION,
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "recorded_in": "workflow step parameters and COLLECT_HUNG_TRIAGE details",
        }
        parameters[key] = trimmed[key]
    if not trimmed:
        return request, {}
    parameters["observation_request_trimmed"] = sorted(trimmed)
    return {**request, "command": {**command, "parameters": parameters}}, trimmed


class CommunicationRuntime:
    def __init__(self, agent: Any):
        self.agent = agent
        self.supervisor = ObservationSupervisor()
        self.lock = threading.RLock()
        self.plan: CommunicationPlan | None = None
        self.lease_deadline = 0.0
        self.last_epoch: dict[tuple[str, str], int] = {}
        self.last_samples: dict[str, Any] = {}
        try:
            self.boot_id = read_bounded(
                agent.proc_root / "sys/kernel/random/boot_id", 128
            ).strip()
        except OSError:
            self.boot_id = ""
        self.resource_limits = service_limits()
        self.settings = {
            name: str(getattr(agent, name))
            if name.endswith("_root") or name.endswith("_dir")
            else getattr(agent, name)
            for name in (
                "proc_root",
                "infiniband_root",
                "diagnostic_output_dir",
                "python_stack_tool",
                "diagnostic_s3_uri",
                "diagnostic_retention_seconds",
                "diagnostic_max_archives",
            )
        }

    def accept_plan(self, raw: Any) -> None:
        if raw is None:
            with self.lock:
                self.plan = None
            return
        plan = CommunicationPlan.model_validate(raw)
        duration = (plan.expires_at - plan.issued_at).total_seconds()
        remaining = (plan.expires_at - datetime.now(timezone.utc)).total_seconds()
        if (
            plan.node_id not in self.agent.node_ids
            or not plan.boot_id
            or plan.boot_id != self.boot_id
            or plan.generation != self.agent.agent_generation
            or not 0 < duration <= 90
            or not 0 < remaining <= duration + 5
        ):
            raise ValueError("communication observation scope identity mismatch")
        with self.lock:
            self.plan = plan
            self.lease_deadline = time.monotonic() + min(duration, remaining)
            active = {(target.job_id, target.attempt_id) for target in plan.targets}
            self.last_epoch = {
                key: value for key, value in self.last_epoch.items() if key in active
            }

    def run_command(self, operation: str, command: Any) -> dict[str, Any]:
        self.resource_limits = service_limits()
        if not self.resource_limits["verified"]:
            raise ObservationIncomplete(
                {"status": "RESOURCE_LIMITS_UNVERIFIED", "evidence_complete": False}
            )
        # A separate activity outlives a timed-out caller if its worker cannot
        # yet leave the kernel, keeping GPU maintenance fenced.
        activity = self.agent.gpu_activity.admit(
            command.operation, command.command_id + "/worker", command.incident_id
        )
        request, trimmed = bounded_request(
            {
                "operation": operation,
                "settings": self.settings,
                "command": command.model_dump(mode="json"),
            }
        )
        timeout = {"summary": 20, "triage": 28, "bundle": 120}[operation]
        # A command waits for the node's single worker slot (preempting the
        # passive sample if that is the holder) instead of being refused on
        # arrival: live, two triages that landed 1.45 s into a sampling epoch
        # were both answered WORKER_STILL_RUNNING within 1.2 s and the hung
        # workflow saw no process on either node.
        result = self.supervisor.run(
            request,
            timeout,
            release=lambda: self.agent.gpu_activity.release(activity),
            slot_wait=slot_wait_for(timeout),
        )
        if trimmed:
            result = {**result, "observation_request_trimmed": trimmed}
        if result.get("status") == SLOT_BUSY and result.get("worker_started") is False:
            raise ObservationSlotBusy(result)
        if result.get("evidence_complete") is False or result.get("status") in {
            "UNAVAILABLE",
            "RESULT_SIZE_LIMIT",
        }:
            raise ObservationIncomplete(result)
        return result

    def tick(self, stop: threading.Event | None = None) -> None:
        self.resource_limits = service_limits()
        if not self.resource_limits["verified"]:
            return
        with self.lock:
            plan = self.plan
            if plan is None or time.monotonic() >= self.lease_deadline:
                return
        epoch = int(time.time() // 30)
        for target in sorted(
            plan.targets,
            key=lambda value: self.last_epoch.get((value.job_id, value.attempt_id), -1),
        ):
            if stop is not None and stop.is_set():
                break
            with self.lock:
                if self.plan is not plan or time.monotonic() >= self.lease_deadline:
                    break
            key = (target.job_id, target.attempt_id)
            if self.last_epoch.get(key) == epoch:
                continue
            self.last_epoch[key] = epoch
            try:
                activity = self.agent.gpu_activity.admit(
                    Op.COLLECT_HUNG_SUMMARY,
                    f"passive/{target.attempt_id}/{epoch}",
                    "passive-observation",
                )
                # Preemptible: a control-plane command that finds this sample
                # holding the slot stops it and runs; the epoch is sampled
                # again 30 s later. The sample itself never waits for a slot a
                # command holds.
                result = self.supervisor.run(
                    {
                        "operation": "sample",
                        "settings": self.settings,
                        "target": target.model_dump(mode="json"),
                        "plan": plan.model_dump(mode="json"),
                    },
                    12,
                    release=lambda: self.agent.gpu_activity.release(activity),
                    preemptible=True,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                result = {"status": "UNAVAILABLE", "reason": str(exc)[:128]}
            with self.lock:
                self.last_samples[target.attempt_id] = result
                self.last_samples = dict(list(self.last_samples.items())[-8:])

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.tick(stop)
            stop.wait(1)

    def health(self) -> dict[str, Any]:
        with self.lock:
            return {
                "enabled": True,
                "lease_valid": self.plan is not None
                and time.monotonic() < self.lease_deadline,
                "expected_targets": self.plan.expected_target_count if self.plan else 0,
                "assigned_targets": len(self.plan.targets) if self.plan else 0,
                "samples": dict(self.last_samples),
                "interval_seconds": 30,
                "worker_limits": {
                    "memory_enforcement": "INHERITED_CGROUP",
                    "cpu_seconds": 8,
                    "concurrent_workers": 1,
                },
                "service_limits": self.resource_limits,
            }


def agent_lifespan(
    reporter: Any,
    observer: CommunicationRuntime | None,
    ownership_gate: Any,
    action_pool: Any,
) -> Callable[[Any], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        stop = threading.Event()
        threads = []
        for service, name in (
            (reporter, "gpu-fault-agent-heartbeat"),
            (observer, "gpu-fault-communication-monitor"),
        ):
            if service is not None:
                worker = threading.Thread(
                    target=service.run, args=(stop,), name=name, daemon=True
                )
                worker.start()
                threads.append((worker, service))
        try:
            yield
        finally:
            stop.set()
            if ownership_gate is not None:
                ownership_gate.cancel_all()
            for worker, service in threads:
                worker.join(timeout=getattr(service, "interval_seconds", 13) + 2)
            action_pool.shutdown(wait=False, cancel_futures=False)

    return lifespan
