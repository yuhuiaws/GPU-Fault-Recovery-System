"""Agent-lifetime passive observation with short-lived control-plane scopes."""

from __future__ import annotations

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
    ObservationIncomplete,
    ObservationSupervisor,
)
from gpu_fault.node_agent.operations.hung_summary_io import read_bounded
from gpu_fault.node_agent.observation_limits import service_limits


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
        result = self.supervisor.run(
            {
                "operation": operation,
                "settings": self.settings,
                "command": command.model_dump(mode="json"),
            },
            {"summary": 20, "triage": 28, "bundle": 120}[operation],
            release=lambda: self.agent.gpu_activity.release(activity),
        )
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
                result = self.supervisor.run(
                    {
                        "operation": "sample",
                        "settings": self.settings,
                        "target": target.model_dump(mode="json"),
                        "plan": plan.model_dump(mode="json"),
                    },
                    12,
                    release=lambda: self.agent.gpu_activity.release(activity),
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
