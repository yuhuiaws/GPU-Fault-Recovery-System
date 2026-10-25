"""A scripted GF-ENV-LIVE world for the watch, judge and runner tests.

A fake clock, a proxy whose ``/status`` reads advance one per collector
interval after the repoint (as the repointed collector scrapes it), and the
product's records -- the ``gpu_state`` and ``node_activity`` documents and the
Node -- that appear at scripted offsets from the arm. The document shapes are
the ones ``general_live_gpu_store.GPU_STATE_PROBE``,
``general_live_store.ACTIVITY_PROBE`` and ``project_node_snapshot`` print.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

T0 = datetime(2031, 3, 4, 5, 0, 0, tzinfo=timezone.utc)
GPU = "GPU-aaaa"
NODE = "ip-10-0-43-5"
BOOT = "boot-1"
PROXY = "http://127.0.0.1:9411/metrics"
REVIEW_OPERATIONS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "COLLECT_DIAGNOSTIC_BUNDLE",
    "TRIGGER_HEALTH_SNAPSHOT",
    "ESCALATE_SUPPORT",
)
DIAGNOSTIC_OPERATIONS = ("FREEZE_EVIDENCE", "VALIDATE_GPU")


def iso(moment: datetime) -> str:
    return moment.isoformat()


class World:
    """Fake time, proxy and store; events fire at offsets from the arm."""

    def __init__(self, *, interval: float = 15.0, start: datetime = T0) -> None:
        self.now = start
        self.interval = interval
        self.repoint_at: datetime | None = None
        self.armed_at: datetime | None = None
        self.armed = False
        self.reads_frozen_at: int | None = None
        self.aborted: str | None = None
        self.endpoint: str | None = PROXY
        self.events: list[tuple[float, Callable[[World], None]]] = []
        self.latest: dict[str, dict[str, Any]] = {}
        self.findings: dict[str, dict[str, Any]] = {}
        self.active: set[str] = set()
        self.batches: list[dict[str, Any]] = []
        self.collector_errors: list[str] = []
        self.incidents: dict[str, dict[str, Any]] = {}
        self.workflows: dict[str, dict[str, Any]] = {}
        self.notifications: dict[str, dict[str, Any]] = {}
        self.node: dict[str, Any] = {
            "name": NODE,
            "uid": "uid-1",
            "boot_id": BOOT,
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        }
        self.sleeps: list[float] = []
        self.polls = 0
        # GF-ENV-LIVE-009: the scheduled read-synchronous pause and the restart.
        self.pause_target: int | None = None
        self.pause_scheduled_at: datetime | None = None
        self.paused_at: datetime | None = None
        self.resumed_at: datetime | None = None
        self.pause_timed_out = False
        self.waiter_dead = False
        self.restart_at: datetime | None = None
        self.restart_events: list[tuple[float, Callable[[World], None]]] = []
        self.read_events: list[tuple[int, Callable[[World], None]]] = []
        # GF-ENV-LIVE-010: the control node and the control task's Pods.
        self.control_node: dict[str, Any] = {
            "name": "ip-10-0-32-162",
            "uid": "uid-2",
            "boot_id": "boot-2",
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        }
        self.control_incidents: dict[str, dict[str, Any]] = {}
        self.control_workflows: dict[str, dict[str, Any]] = {}
        self.control_pods: list[dict[str, Any]] = [
            {
                "name": "envctl-master-0",
                "uid": "pod-ctl-1",
                "node": "ip-10-0-32-162",
                "phase": "Running",
                "ready": True,
                "attempt_id": "envctl-a001",
            }
        ]

    # -- time --------------------------------------------------------------- #
    def clock(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)

    def at(self, offset: float, action: Callable[[World], None]) -> World:
        self.events.append((offset, action))
        return self

    def origin(self) -> datetime:
        return self.armed_at or self.repoint_at or self.now

    def at_read(self, reads: int, action: Callable[[World], None]) -> World:
        """Fire once the collector consumed ``reads`` proxy reads (pause-aware)."""

        self.read_events.append((reads, action))
        return self

    def at_restart(self, offset: float, action: Callable[[World], None]) -> World:
        """Fire ``offset`` seconds after the 009 collector restart."""

        self.restart_events.append((offset, action))
        return self

    def _fire(self) -> None:
        if self.repoint_at is None:
            return  # nothing reaches the product before the collector is repointed
        due = [
            item
            for item in self.events
            if self.origin() + timedelta(seconds=item[0]) <= self.now
        ]
        for item in due:
            self.events.remove(item)
            item[1](self)
        reads = self.reads()
        for item in [item for item in self.read_events if item[0] <= reads]:
            self.read_events.remove(item)
            item[1](self)
        if self.restart_at is not None:
            for item in [
                item
                for item in self.restart_events
                if self.restart_at + timedelta(seconds=item[0]) <= self.now
            ]:
                self.restart_events.remove(item)
                item[1](self)

    def moment(self, offset: float) -> datetime:
        return self.origin() + timedelta(seconds=offset)

    # -- proxy -------------------------------------------------------------- #
    def reads(self) -> int:
        """Collector ticks since the repoint, skipping those inside a pause.

        The 009 waiter stops the unit right after the target read: a tick
        inside the pause is never taken, and the first tick after the resume
        is the next grid point, so the accepted-sample gap is the pause plus
        up to one interval -- what the live collector's sleep does too.
        """

        if self.repoint_at is None:
            return 0
        count, k = 0, 1
        while True:
            tick = self.repoint_at + timedelta(seconds=k * self.interval)
            if tick > self.now:
                break
            k += 1
            if (
                self.paused_at is not None
                and tick > self.paused_at
                and (self.resumed_at is None or tick < self.resumed_at)
            ):
                continue
            count += 1
            if (
                self.pause_target is not None
                and self.paused_at is None
                and not self.pause_timed_out
                and not self.waiter_dead
                and count >= self.pause_target
            ):
                self.paused_at = tick + timedelta(milliseconds=250)
        if self.reads_frozen_at is not None:
            count = min(count, self.reads_frozen_at)
        return count

    def tick_time(self, reads: int) -> datetime:
        """When the collector took its ``reads``-th read (pause-aware)."""

        assert self.repoint_at is not None, "no reads before the repoint"
        count, k = 0, 1
        while True:
            tick = self.repoint_at + timedelta(seconds=k * self.interval)
            k += 1
            if (
                self.paused_at is not None
                and tick > self.paused_at
                and (self.resumed_at is None or tick < self.resumed_at)
            ):
                continue
            count += 1
            if count == reads:
                return tick

    def status(self) -> dict[str, Any]:
        return {
            "run_id": "r1",
            "status": {
                "reads": self.reads(),
                "next_values": {},
                "proof_scope": "live-telemetry-replay",
                "aborted": self.aborted,
                "armed": self.armed,
            },
            "effective_endpoint": self.endpoint,
        }

    def repoint(self) -> dict[str, Any]:
        self.repoint_at = self.now
        return {
            "proxy_endpoint": PROXY,
            "original_endpoint": "http://127.0.0.1:9400/metrics",
        }

    def arm(self) -> dict[str, Any]:
        self.armed_at = self.now
        self.armed = True
        return {"run_id": "r1", "armed": True, "status": self.status()["status"]}

    # -- 009: pause / resume / restart ------------------------------------- #
    def pause_collector(self, **options: Any) -> dict[str, Any]:
        self.pause_target = int(options["after_reads"])
        self.pause_scheduled_at = self.now
        return {
            "run_id": "r1",
            "target_reads": self.pause_target,
            "waiter": {"ActiveState": "inactive" if self.waiter_dead else "active"},
            "status": self.status()["status"],
        }

    def pause_status(self) -> dict[str, Any]:
        self._fire()
        record: dict[str, Any] = {}
        if self.pause_target is not None:
            record = {
                "run_id": "r1",
                "target_reads": self.pause_target,
                "scheduled_at": iso(self.pause_scheduled_at or self.now),
            }
        if self.paused_at is not None:
            record.update(
                paused_at=iso(self.paused_at),
                reads_at_pause=self.pause_target,
                missed=False,
            )
        if self.resumed_at is not None:
            record["resumed_at"] = iso(self.resumed_at)
        if self.pause_timed_out:
            record["timed_out_at"] = iso(self.now)
        waiting = self.pause_target is not None and self.paused_at is None
        return {
            "run_id": "r1",
            "pause": record,
            "stopped": self.paused_at is not None and self.resumed_at is None,
            "waiter": {
                "ActiveState": "active"
                if waiting and not self.pause_timed_out and not self.waiter_dead
                else "inactive"
            },
            "effective_endpoint": self.endpoint,
        }

    def resume_collector(self) -> dict[str, Any]:
        self.resumed_at = self.now
        return {
            "run_id": "r1",
            "stopped": False,
            "pause": {"resumed_at": iso(self.now)},
        }

    def restart_collector(self) -> dict[str, Any]:
        self.restart_at = self.now
        return {
            "run_id": "r1",
            "restarted_at": iso(self.now),
            "effective_endpoint": PROXY,
        }

    # -- 010: the control node and task ------------------------------------ #
    def control_node_snapshot(self) -> dict[str, Any]:
        self._fire()
        return {**self.control_node, "taints": list(self.control_node["taints"])}

    def control_activity(self) -> dict[str, Any]:
        self._fire()
        return {
            "since": iso(self.now),
            "incidents": list(self.control_incidents.values()),
            "workflows": list(self.control_workflows.values()),
            "notifications": [],
        }

    def control_task_snapshot(self) -> dict[str, Any]:
        self._fire()
        return {
            "job_id": "envctl",
            "attempt_id": "envctl-a001",
            "node": self.control_node["name"],
            "pods": [dict(item) for item in self.control_pods],
        }

    # -- product records ---------------------------------------------------- #
    def gpu_state(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self._fire()
        self.polls += 1
        return {
            "since": iso(self.now),
            "latest": list(self.latest.values()),
            "active_findings": [
                self.findings[key]
                for key in sorted(self.active)
                if key in self.findings
            ],
            "findings": list(self.findings.values()),
            "xid_events": [],
            "batches": list(self.batches),
            "collector_statuses": [
                {
                    "collector": "GPU_METRICS",
                    "observed_at": iso(self.now),
                    "errors": list(self.collector_errors),
                }
            ],
            "agent": {"lifecycle_state": "ACTIVE"},
            "capabilities": {},
        }

    def activity(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self._fire()
        return {
            "since": iso(self.now),
            "batches": [],
            "incidents": list(self.incidents.values()),
            "workflows": list(self.workflows.values()),
            "markers": [],
            "commands": [],
            "health_signals": {},
            "metric_latest": [],
            "collector_statuses": [],
            "notifications": list(self.notifications.values()),
            "agent": {"lifecycle_state": "ACTIVE"},
        }

    def node_snapshot(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self._fire()
        return {**self.node, "taints": list(self.node["taints"])}

    # -- builders ----------------------------------------------------------- #
    def record_batch(
        self,
        observed_at: datetime,
        values: dict[str, float],
        *,
        reasons: tuple[str, ...] = ("environmental-observation",),
        errors: tuple[str, ...] = (),
        gpu: str = GPU,
        boot: str = BOOT,
    ) -> None:
        """A delivered batch now: raw evidence with the samples, and the latest."""

        observed = iso(observed_at)
        batch_id = f"dcgm-{NODE}-{len(self.batches)}"
        self.batches.append(
            {
                "batch_id": batch_id,
                "observed_at": observed,
                "edge_filter_reasons": list(reasons),
                "collection_errors": list(errors),
                "sample_count": len(values),
                "gpus": [gpu],
                "samples": [
                    {
                        "gpu": gpu,
                        "name": name,
                        "value": value,
                        "labels": {"source_boot_id": boot},
                    }
                    for name, value in values.items()
                ],
            }
        )
        for name, value in values.items():
            self.latest[f"{gpu}:{name}"] = {
                "gpu": gpu,
                "name": name,
                "value": value,
                "observed_at": observed,
                "source": "DCGM_EXPORTER",
                "labels": {"source_boot_id": boot},
            }

    def accept(
        self,
        offset: float,
        values: dict[str, float],
        *,
        reasons: tuple[str, ...] = ("environmental-observation",),
        errors: tuple[str, ...] = (),
        gpu: str = GPU,
        boot: str = BOOT,
    ) -> World:
        """A delivered batch at ``offset`` from the arm."""

        return self.at(
            offset,
            lambda world: world.record_batch(
                world.moment(offset),
                values,
                reasons=reasons,
                errors=errors,
                gpu=gpu,
                boot=boot,
            ),
        )

    def record_finding(
        self,
        observed_at: datetime,
        finding_id: str,
        *,
        name: str,
        value: float,
        severity: str,
        count: int = 0,
        action: str | None = None,
        source: str | None = None,
        policy: str = "SITE_NVIDIA_DEVICE_LIMIT_DERIVED",
        active: bool = True,
        gpu: str = GPU,
        reason: str = "",
    ) -> None:
        self.findings[finding_id] = {
            "finding_id": finding_id,
            "canonical_name": name,
            "severity": severity,
            "value": value,
            "gpu_uuid": gpu,
            "observed_at": iso(observed_at),
            "automatic_action": action
            or ("DRAIN" if severity == "CRITICAL" else "RUN_DIAGNOSTICS"),
            "official_action": None,
            "policy_source": policy,
            "threshold_source": source,
            "confirmation_count": count,
            "workload_state": "IDLE",
            "affected_workload_ids": [],
            "reason": reason,
            "finding_kind": "METRIC",
            "affected_gpu_uuids": [],
        }
        if active:
            self.active.add(finding_id)

    def finding(
        self,
        offset: float,
        finding_id: str,
        *,
        name: str,
        value: float,
        severity: str,
        count: int = 0,
        action: str | None = None,
        source: str | None = None,
        policy: str = "SITE_NVIDIA_DEVICE_LIMIT_DERIVED",
        active: bool = True,
        gpu: str = GPU,
        reason: str = "",
    ) -> World:
        return self.at(
            offset,
            lambda world: world.record_finding(
                world.moment(offset),
                finding_id,
                name=name,
                value=value,
                severity=severity,
                count=count,
                action=action,
                source=source,
                policy=policy,
                active=active,
                gpu=gpu,
                reason=reason,
            ),
        )

    def clear(self, offset: float, *finding_ids: str) -> World:
        return self.at(
            offset, lambda world: world.active.difference_update(finding_ids)
        )

    def incident(
        self,
        offset: float,
        finding_id: str,
        *,
        state: str = "ACTION_PENDING",
        action: str = "RUN_DIAGNOSTICS",
        workflow: str | None = None,
        gpus: list[str] | None = None,
    ) -> World:
        def apply(world: World) -> None:
            incident_id = f"inc-gpu-{finding_id}"
            world.incidents[incident_id] = {
                "incident_id": incident_id,
                "event_id": f"gpu-{finding_id}",
                "event_type": "NODE_HEALTH",
                "state": state,
                "effective_action": action,
                "gpu_uuids": gpus if gpus is not None else [GPU],
                "node_ids": [NODE],
                "workflow_request_id": workflow,
                "created_at": iso(world.moment(offset)),
            }

        return self.at(offset, apply)

    def set_incident(self, offset: float, finding_id: str, **fields: Any) -> World:
        return self.at(
            offset,
            lambda world: world.incidents[f"inc-gpu-{finding_id}"].update(fields),
        )

    def workflow(
        self,
        offset: float,
        request_id: str,
        finding_id: str,
        *,
        operations: tuple[str, ...],
        status: str = "PENDING",
        executed: tuple[str, ...] = (),
        deadline_seconds: float = 1800,
    ) -> World:
        def apply(world: World) -> None:
            created = world.moment(offset)
            world.workflows[request_id] = {
                "request_id": request_id,
                "incident_id": f"inc-gpu-{finding_id}",
                "status": status,
                "official_steps": [{"operation": name} for name in operations],
                "step_executions": [
                    {
                        "operation": name,
                        "step_index": index,
                        "status": "SUCCEEDED",
                        "updated_at": iso(created),
                    }
                    for index, name in enumerate(operations)
                    if name in executed
                ],
                "completed_operations": list(executed),
                "execution_deadline": iso(
                    created + timedelta(seconds=deadline_seconds)
                ),
                "created_at": iso(created),
            }

        return self.at(offset, apply)

    def finish(
        self, offset: float, request_id: str, *, status: str = "SUCCEEDED"
    ) -> World:
        def apply(world: World) -> None:
            item = world.workflows[request_id]
            names = [step["operation"] for step in item["official_steps"]]
            outcome = "SUCCEEDED" if status == "SUCCEEDED" else "FAILED"
            item["status"] = status
            item["completed_operations"] = names if outcome == "SUCCEEDED" else []
            item["step_executions"] = [
                {
                    "operation": name,
                    "step_index": index,
                    "status": outcome,
                    "updated_at": iso(world.moment(offset)),
                    "error": None if outcome == "SUCCEEDED" else "validation refused",
                }
                for index, name in enumerate(names)
            ]

        return self.at(offset, apply)

    def notify(self, offset: float, finding_id: str, *, status: str = "SENT") -> World:
        def apply(world: World) -> None:
            world.notifications[f"n-{finding_id}"] = {
                "notification_id": f"n-{finding_id}",
                "incident_id": f"inc-gpu-{finding_id}",
                "category": "SUPPORT_ESCALATION",
                "created_at": iso(world.moment(offset)),
                "status": status,
            }

        return self.at(offset, apply)

    def isolate(self, offset: float) -> World:
        def apply(world: World) -> None:
            world.node["unschedulable"] = True
            world.node["taints"] = [
                {"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}
            ]

        return self.at(offset, apply)

    def release(self, offset: float) -> World:
        def apply(world: World) -> None:
            world.node["unschedulable"] = False
            world.node["taints"] = []

        return self.at(offset, apply)


def review_chain(
    world: World,
    *,
    name: str = "gpu_tlimit_c",
    severe: float = -3.0,
    healthy: float = 58.0,
    first_severe: float = 45.0,
    samples: int = 3,
    source: str = "NVIDIA_DEVICE_TLIMIT",
    policy: str = "SITE_NVIDIA_DEVICE_LIMIT_DERIVED",
    finish_at: float = 200.0,
) -> World:
    """The product doing GF-ENV-LIVE-004 right, at the default 15 s cadence."""

    step = world.interval
    for index in range(samples):
        world.accept(
            first_severe + index * step, {name: severe, "gpu_temperature_c": 30.0}
        )
    confirmed = first_severe + (samples - 1) * step
    if samples > 1:
        world.finding(
            first_severe,
            "f-warn",
            name=name,
            value=severe,
            severity="WARNING",
            count=1,
            source=source,
            policy=policy,
        )
        world.incident(first_severe, "f-warn", workflow="wf-diag")
        world.workflow(
            first_severe, "wf-diag", "f-warn", operations=DIAGNOSTIC_OPERATIONS
        )
        world.finish(confirmed + 5, "wf-diag", status="FAILED")
    world.finding(
        confirmed,
        "f-crit",
        name=name,
        value=severe,
        severity="CRITICAL",
        count=samples,
        source=source,
        policy=policy,
    )
    world.clear(confirmed, "f-warn")
    world.incident(confirmed, "f-crit", action="DRAIN", workflow="wf-review")
    world.workflow(
        confirmed, "wf-review", "f-crit", operations=REVIEW_OPERATIONS, status="RUNNING"
    )
    world.accept(
        confirmed + step,
        {name: healthy, "gpu_temperature_c": 30.0},
        reasons=("candidate-recovered",),
    )
    world.clear(confirmed + step, "f-crit")
    world.isolate(confirmed + 25)
    world.finish(finish_at, "wf-review")
    world.set_incident(finish_at, "f-crit", state="ESCALATED")
    world.notify(finish_at, "f-crit")
    return world


H200_LIMITS = {
    "gpu_tlimit_c": 59.0,
    "gpu_shutdown_tlimit_c": -8.0,
    "gpu_slowdown_tlimit_c": -2.0,
    "gpu_max_operating_tlimit_c": 0.0,
    "memory_max_operating_tlimit_c": 0.0,
}


def h200_preflight(**overrides: Any) -> dict[str, Any]:
    """A clean preflight of a T.Limit-only p5en H200 with its device plan."""

    from gpu_fault.gpu_metrics import GpuMetricsThresholds
    from scripts.e2e.regional import env_live_fixture as fixture
    from scripts.e2e.regional import env_live_plans as plans

    target = {
        "gpu_uuid": GPU,
        "gpu_index": "0",
        "product": "NVIDIA H200",
        "pci_bdf": "59:00",
        "limits": dict(H200_LIMITS),
    }
    plan = plans.build_device_plan(
        gpu_uuid=GPU,
        gpu_index="0",
        product="NVIDIA H200",
        raw_limits=target["limits"],
        thresholds=GpuMetricsThresholds(),
        confirmation_samples=3,
    )
    value = {
        "release_id": "rel-1",
        "cluster_id": "gpu-eks-cluster",
        "node": {"name": NODE, "boot_id": BOOT},
        "store": {"agent_state": "ACTIVE", "collector_statuses": [], "incident": None},
        "workloads": [],
        "predecessor": {"valid": True, "verdict": "NOT_REQUIRED", "case_id": None},
        "cpu_blast": {},
        "devices": [target],
        "device_plans": {GPU: fixture.device_plan_report(plan)},
        "target_gpu": target,
        "confirmation_samples": 3,
        "errors": [],
    }
    value.update(overrides)
    return value


def baseline_values(**overrides: float) -> dict[str, float]:
    """The target GPU's readings an H200 exporter delivers (T.Limit scale)."""

    values = {
        "gpu_temperature_c": 31.0,
        "power_usage_w": 112.0,
        "power_limit_w": 700.0,
        "clock_throttle_reasons": 0.0,
        "gpu_slowdown_tlimit_c": -2.0,
        "gpu_shutdown_tlimit_c": -8.0,
        "gpu_tlimit_c": 58.0,
        "memory_temperature_c": 33.0,
    }
    values.update(overrides)
    return values


COLLECTOR_ENV = {
    "GPU_FAULT_METRICS_INTERVAL_SECONDS": "15",
    "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "3",
    "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "300",
}


class WorldRegionalSettings:
    cluster_id = "gpu-eks-cluster"

    def environment(self) -> dict[str, str]:
        return {"GPU_FAULT_CLUSTER_ID": self.cluster_id}


class WorldRegional:
    """Stands in for RegionalLiveFixture over a World."""

    world: World

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "rel-1", "cluster_id": "gpu-eks-cluster"}

    def node_snapshot(self, node: str) -> dict[str, Any]:
        if node == self.world.control_node["name"]:
            return self.world.control_node_snapshot()
        return self.world.node_snapshot()


class WorldController:
    """Stands in for EnvReplayController; records the lifecycle order."""

    instances: list[WorldController] = []
    world: World
    restored_endpoint: str | None = None
    after_state = "active"

    def __init__(self, regional: Any, **options: Any) -> None:
        self.options = options
        self.calls: list[str] = []
        WorldController.instances.append(self)

    def create(self) -> None:
        self.calls.append("create")

    def preflight(self) -> dict[str, Any]:
        self.calls.append("preflight")
        return {
            "metrics_unit": {"ActiveState": "active"},
            "collector_env": dict(COLLECTOR_ENV),
        }

    def deploy_proxy(self, plan_path: Any, gpu_uuid: str, **options: Any) -> Any:
        self.calls.append("deploy_proxy")
        self.plan = Path(plan_path).read_text()
        return {"endpoint": PROXY, "status": {"armed": False}}

    def repoint(self, **options: Any) -> dict[str, Any]:
        self.calls.append("repoint")
        return self.world.repoint()

    def status(self, **options: Any) -> dict[str, Any]:
        self.calls.append("status")
        return self.world.status()

    def arm(self, **options: Any) -> dict[str, Any]:
        self.calls.append("arm")
        return self.world.arm()

    def pause_collector(self, **options: Any) -> dict[str, Any]:
        self.calls.append("pause_collector")
        return self.world.pause_collector(**options)

    def pause_status(self) -> dict[str, Any]:
        self.calls.append("pause_status")
        return self.world.pause_status()

    def resume_collector(self) -> dict[str, Any]:
        self.calls.append("resume_collector")
        return self.world.resume_collector()

    def restart_collector(self) -> dict[str, Any]:
        self.calls.append("restart_collector")
        return self.world.restart_collector()

    def cleanup(self) -> dict[str, Any]:
        self.calls.append("cleanup")
        return {
            "restore": {
                "restored_endpoint": self.restored_endpoint,
                "original_endpoint": "http://127.0.0.1:9400/metrics",
                "after": {"ActiveState": self.after_state},
                "dropin_removed": True,
            },
            "stop_proxy": {"ok": True},
            "probe_residuals": {"pod": False, "configmap": False},
        }


class WorldControlTask:
    """Stands in for env_live_scope.ControlTask over the World's control Pods."""

    instances: list[WorldControlTask] = []
    world: World

    def __init__(self, regional: Any, **options: Any) -> None:
        self.options = options
        self.calls: list[str] = []
        self.node = options.get("node")
        WorldControlTask.instances.append(self)

    def start(self, **_options: Any) -> dict[str, Any]:
        self.calls.append("start")
        return self.world.control_task_snapshot()

    def snapshot(self) -> dict[str, Any]:
        self.calls.append("snapshot")
        return self.world.control_task_snapshot()

    def cleanup(self) -> dict[str, Any]:
        self.calls.append("cleanup")
        return {"workload": "removed", "prewarm": {"pods": False}}


def install_world(monkeypatch: Any, world: World, *, preflight: Any) -> None:
    """Bind the runner's live seams to ``world`` (no cluster, fake clock)."""

    from scripts.e2e.regional import env_live_fixture as fixture
    from scripts.e2e.regional import env_live_scope as scope
    from scripts.e2e.regional import run_env_live as runner

    WorldController.instances = []
    WorldControlTask.instances = []
    monkeypatch.setattr(WorldController, "world", world, raising=False)
    monkeypatch.setattr(WorldControlTask, "world", world, raising=False)
    monkeypatch.setattr(WorldRegional, "world", world, raising=False)
    monkeypatch.setattr(runner, "RegionalLiveFixture", WorldRegional)
    monkeypatch.setattr(fixture, "EnvReplayController", WorldController)
    monkeypatch.setattr(scope, "ControlTask", WorldControlTask)
    monkeypatch.setattr(fixture, "gpu_records", world.gpu_state)

    def node_records(regional: Any, node: str, since: Any) -> dict[str, Any]:
        if node == world.control_node["name"]:
            return world.control_activity()
        return world.activity()

    def node_state(regional: Any, node: str) -> dict[str, Any]:
        if node == world.control_node["name"]:
            return world.control_node_snapshot()
        return world.node_snapshot()

    monkeypatch.setattr(fixture, "node_records", node_records)
    monkeypatch.setattr(fixture, "node_state", node_state)
    monkeypatch.setattr(fixture, "utc_now", world.clock)
    monkeypatch.setattr(runner, "SLEEP", world.sleep)
    monkeypatch.setattr(runner, "read_only_preflight", lambda s, d: preflight)
