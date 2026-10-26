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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

T0 = datetime(2031, 3, 4, 5, 0, 0, tzinfo=timezone.utc)
GPU = "GPU-aaaa"
NODE = "ip-10-0-43-5"
BOOT = "boot-1"
PROXY = "http://127.0.0.1:9411/metrics"
REAL = "http://127.0.0.1:9400/metrics"
#: The run's repoint drop-in as the probe names it, and the collector unit's
#: identity after the repoint restart (``systemctl show``).
DROPIN = (
    "/run/systemd/system/gpu-fault-metrics-collector.service.d/"
    "gpu-fault-env-replay-0123456789abcdef.conf"
)
UNIT_INVOCATION = "52ac4fabb52d4b77beb6d0e928da732e"
#: The live proxy's refusal text (tools/gpu_environment_replay.transform).
REFUSAL = "target GPU or requested fields are absent; refusing invented samples"
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


@dataclass
class _Pause:
    """One scheduled read-synchronous pause of the 009 world (gap or hold)."""

    target: int | None = None
    scheduled_at: datetime | None = None
    paused_at: datetime | None = None
    resumed_at: datetime | None = None
    timed_out: bool = False
    waiter_dead: bool = False


def _pause_record(pause: _Pause | None, label: str, now: datetime) -> dict[str, Any]:
    """The probe's pause record for one label, with its waiter's state."""

    if pause is None:
        return {"waiter": {"ActiveState": "inactive"}}
    record: dict[str, Any] = {
        "run_id": "r1",
        "label": label,
        "target_reads": pause.target,
        "scheduled_at": iso(pause.scheduled_at or now),
    }
    if pause.paused_at is not None:
        record.update(
            paused_at=iso(pause.paused_at), reads_at_pause=pause.target, missed=False
        )
    if pause.resumed_at is not None:
        record["resumed_at"] = iso(pause.resumed_at)
    if pause.timed_out:
        record["timed_out_at"] = iso(now)
    waiting = pause.paused_at is None and not pause.timed_out
    record["waiter"] = {
        "ActiveState": "active" if waiting and not pause.waiter_dead else "inactive"
    }
    return record


class World:
    """Fake time, proxy and store; events fire at offsets from the arm."""

    def __init__(self, *, interval: float = 15.0, start: datetime = T0) -> None:
        self.now = start
        self.interval = interval
        self.repoint_at: datetime | None = None
        self.armed_at: datetime | None = None
        self.armed = False
        #: The proxy's read count at the arm (the proven repoint's pass-through
        #: read): plan reads in ``at_read``/``tick_time`` count from here.
        self.reads_at_arm: int | None = None
        self.reads_frozen_at: int | None = None
        self.aborted: str | None = None
        self.endpoint: str | None = PROXY
        self.events: list[tuple[float, Callable[[World], None]]] = []
        self.latest: dict[str, dict[str, Any]] = {}
        #: Every finding recorded, by id; ``history`` is the subset the product
        #: would keep durably (``gpu_finding_history``: the first finding of a
        #: GPU/field key or a severity / automatic-action change), ``state`` the
        #: current finding per key (``gpu_finding_state``), ``active`` the ids
        #: the ``active_findings`` read lists.
        self.findings: dict[str, dict[str, Any]] = {}
        self.history: dict[str, dict[str, Any]] = {}
        self.state: dict[tuple[str, str], str] = {}
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
        #: What one remote read of the runner costs (a ``kubectl exec`` live:
        #: 6-8 s); zero keeps the reads instantaneous. A poll's reads advance
        #: the clock by it, so a polling round is the sleep plus the reads --
        #: the live cadence that missed the 15 s regrade state of 009.
        self.read_latency_seconds = 0.0
        # GF-ENV-LIVE-009: the scheduled read-synchronous pauses (the gap pause
        # under "", the recount hold under its label) and the restart.
        self.pauses: dict[str, _Pause] = {}
        self.pause_timed_out = False
        self.waiter_dead = False
        self.hold_timed_out = False
        self.hold_waiter_dead = False
        #: ``pause-collector --label hold`` refused by the node.
        self.hold_schedule_fails = False
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
        # The collector unit as the repoint left it and as a later ``snapshot``
        # sees it; the proxy's refusal counters; what the probe preflight finds.
        self.unit_invocation = UNIT_INVOCATION
        self.unit_started: str | None = None
        self.unit_active = "active"
        self.dropin_loaded = True
        self.failed_reads: int | None = 0
        self.last_error: str | None = None
        #: Live 2026-10-26 GF-ENV-LIVE-006: every armed read answered 503, the
        #: read count frozen at the arm; one refusal per collector tick.
        self.refuse_armed_reads = False
        #: A collector that stops scraping at the arm (no refusals counted).
        self.freeze_reads_at_arm = False
        self.repoint_receipt_overrides: dict[str, Any] = {}
        self.preflight_extra: dict[str, Any] = {}
        self.snapshots = 0

    # -- time --------------------------------------------------------------- #
    def clock(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)

    def latency(self) -> None:
        """One remote read of the runner costs ``read_latency_seconds``."""

        if self.read_latency_seconds:
            self.now += timedelta(seconds=self.read_latency_seconds)

    def _remote_read(self) -> None:
        """A store or Node read: its latency, then the product's events."""

        self.latency()
        self._fire()

    # -- 009 pauses (legacy single-pause view: the gap pause) ------------------ #
    @property
    def pause_target(self) -> int | None:
        return self.pauses[""].target if "" in self.pauses else None

    @property
    def paused_at(self) -> datetime | None:
        return self.pauses[""].paused_at if "" in self.pauses else None

    @property
    def resumed_at(self) -> datetime | None:
        return self.pauses[""].resumed_at if "" in self.pauses else None

    @property
    def hold(self) -> _Pause | None:
        return self.pauses.get("hold")

    def _inside_pause(self, tick: datetime) -> bool:
        """A collector tick no pause lets through (the unit is SIGSTOPped)."""

        return any(
            pause.paused_at is not None
            and tick > pause.paused_at
            and (pause.resumed_at is None or tick < pause.resumed_at)
            for pause in self.pauses.values()
        )

    def at(self, offset: float, action: Callable[[World], None]) -> World:
        self.events.append((offset, action))
        return self

    def origin(self) -> datetime:
        return self.armed_at or self.repoint_at or self.now

    def at_read(self, reads: int, action: Callable[[World], None]) -> World:
        """Fire once the collector consumed ``reads`` plan reads (pause-aware).

        Plan reads count from the arm: the proven repoint already cost one
        pass-through read, so plan read ``n`` is absolute proxy read
        ``reads_at_arm + n`` (absolute before an arm, for the baseline case).
        """

        self.read_events.append((reads, action))
        return self

    def _absolute(self, reads: int) -> int:
        return reads + (self.reads_at_arm or 0)

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
        for item in [
            item for item in self.read_events if self._absolute(item[0]) <= reads
        ]:
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
            if self._inside_pause(tick):
                continue
            count += 1
            for pause in self.pauses.values():
                if (
                    pause.target is not None
                    and pause.paused_at is None
                    and not pause.timed_out
                    and not pause.waiter_dead
                    and count >= pause.target
                ):
                    # The waiter's SIGSTOP: a quarter second after the read
                    # (its /status poll; the live probe adds a settle).
                    pause.paused_at = tick + timedelta(milliseconds=250)
        if self.reads_frozen_at is not None:
            count = min(count, self.reads_frozen_at)
        return count

    def tick_time(self, reads: int) -> datetime:
        """When the collector took its ``reads``-th plan read (pause-aware)."""

        assert self.repoint_at is not None, "no reads before the repoint"
        target = self._absolute(reads)
        count, k = 0, 1
        while True:
            tick = self.repoint_at + timedelta(seconds=k * self.interval)
            k += 1
            if self._inside_pause(tick):
                continue
            count += 1
            if count == target:
                return tick

    def refused_reads(self) -> int | None:
        """The proxy's refusals: one per collector tick since the arm when the
        armed reads are refused, else the scripted count."""

        if not self.refuse_armed_reads or self.armed_at is None:
            return self.failed_reads
        ticks = 0
        k = 1
        while self.armed_at + timedelta(seconds=k * self.interval) <= self.now:
            ticks += 1
            k += 1
        return ticks

    def status(self) -> dict[str, Any]:
        refused = self.refused_reads()
        return {
            "run_id": "r1",
            "status": {
                "reads": self.reads(),
                "next_values": {},
                "proof_scope": "live-telemetry-replay",
                "aborted": self.aborted,
                "armed": self.armed,
                "failed_reads": refused,
                "last_error": (
                    REFUSAL if self.refuse_armed_reads and refused else self.last_error
                ),
            },
            "effective_endpoint": self.endpoint,
        }

    def unit_state(self) -> dict[str, str]:
        return {
            "ActiveState": self.unit_active,
            "SubState": "running" if self.unit_active == "active" else "dead",
            "MainPID": "4242",
            "InvocationID": self.unit_invocation,
            "ExecMainStartTimestamp": self.unit_started or iso(self.now),
            "DropInPaths": DROPIN if self.dropin_loaded else "",
        }

    def repoint(self) -> dict[str, Any]:
        """The probe's proven repoint: the unit restarted, its process reads
        the proxy, and the proxy counted the first pass-through read before
        the call returned (one collector tick plus a little, as live)."""

        self.repoint_at = self.now
        self.unit_started = iso(self.now)
        # The probe returns only after the restarted collector's first
        # pass-through read: one tick on the collector's grid. The wait is the
        # probe's, not a runner poll, so it is not recorded in ``sleeps``.
        self.now += timedelta(seconds=self.interval)
        receipt: dict[str, Any] = {
            "repointed": True,
            "run_id": "r1",
            "proxy_endpoint": PROXY,
            "original_endpoint": REAL,
            "dropin": DROPIN,
            "process_environment": {"GPU_FAULT_DCGM_METRICS_URL": PROXY},
            "first_read": {
                "observed": True,
                "reads_before": 0,
                "reads_after": self.reads(),
                "waited_seconds": self.interval,
                "budget_seconds": 65,
            },
            "after": self.unit_state(),
        }
        receipt.update(self.repoint_receipt_overrides)
        return receipt

    def snapshot(self) -> dict[str, Any]:
        """The probe's ``snapshot``: the unit now, and the endpoint it reads."""

        self._fire()
        self.snapshots += 1
        return {
            "captured_at": iso(self.now),
            "metrics_unit": self.unit_state(),
            "effective_endpoint": self.endpoint,
            "baseline_endpoint": REAL,
            "open_windows": [],
            "leftover_units": [],
        }

    def restart_unit(self) -> None:
        """The collector unit restarted (a new InvocationID, a new start)."""

        self.unit_invocation = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
        self.unit_started = iso(self.now)

    def arm(self) -> dict[str, Any]:
        self.armed_at = self.now
        self.armed = True
        self.reads_at_arm = self.reads()
        if self.refuse_armed_reads or self.freeze_reads_at_arm:
            self.reads_frozen_at = self.reads_at_arm
        return {"run_id": "r1", "armed": True, "status": self.status()["status"]}

    # -- 009: pause / hold / resume / restart ------------------------------- #
    def pause_collector(self, **options: Any) -> dict[str, Any]:
        """``pause-collector``: the gap pause (no label) or the recount hold."""

        label = str(options.get("label") or "")
        if label and self.hold_schedule_fails:
            raise RuntimeError(f"pause-collector[{label}] refused by the node")
        pause = _Pause(
            target=int(options["after_reads"]),
            scheduled_at=self.now,
            timed_out=self.hold_timed_out if label else self.pause_timed_out,
            waiter_dead=self.hold_waiter_dead if label else self.waiter_dead,
        )
        self.pauses[label] = pause
        return {
            "run_id": "r1",
            "label": label,
            "target_reads": pause.target,
            "waiter": {"ActiveState": "inactive" if pause.waiter_dead else "active"},
            "status": self.status()["status"],
        }

    def pause_status(self) -> dict[str, Any]:
        self._remote_read()
        records = {
            label: _pause_record(pause, label, self.now)
            for label, pause in self.pauses.items()
        }
        gap = records.get("") or _pause_record(None, "", self.now)
        return {
            "run_id": "r1",
            "pause": {key: value for key, value in gap.items() if key != "waiter"},
            "stopped": any(
                pause.paused_at is not None and pause.resumed_at is None
                for pause in self.pauses.values()
            ),
            "waiter": gap["waiter"],
            "pauses": records,
            "effective_endpoint": self.endpoint,
        }

    def resume_collector(self, label: str = "") -> dict[str, Any]:
        pause = self.pauses.get(label)
        if pause is not None and pause.resumed_at is None:
            pause.resumed_at = self.now
        return {
            "run_id": "r1",
            "label": label,
            "stopped": False,
            "pause": {"label": label, "resumed_at": iso(self.now)},
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
        self._remote_read()
        self.polls += 1
        return {
            "since": iso(self.now),
            "latest": list(self.latest.values()),
            "active_findings": [
                self.findings[key]
                for key in sorted(self.active)
                if key in self.findings
            ],
            # The durable set only: a finding that merely changed its
            # confirmation count is in ``active_findings`` while it is the
            # current state and nowhere once a later reading replaced it.
            "findings": list(self.history.values()),
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
        self._remote_read()
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
        self._remote_read()
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
        history: bool | None = None,
    ) -> None:
        """A finding the product graded now.

        ``history`` defaults to the product's own rule (``store.update_gpu_
        findings``): the ``gpu_finding_history`` row is written only when the
        finding *activates* -- the first finding of its GPU/field key, or a
        severity or automatic-action change against the key's current state.
        A confirmation-count change alone (1/3 -> 2/3, or the post-gap 2/3 ->
        1/3 regrade) replaces the current state and leaves no durable row. An
        active finding replaces the key's previous one as the current state.
        """

        automatic_action = action or (
            "DRAIN" if severity == "CRITICAL" else "RUN_DIAGNOSTICS"
        )
        finding = {
            "finding_id": finding_id,
            "canonical_name": name,
            "severity": severity,
            "value": value,
            "gpu_uuid": gpu,
            "observed_at": iso(observed_at),
            "automatic_action": automatic_action,
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
        key = (gpu, name)
        previous = self.findings.get(self.state.get(key, ""))
        if history is None:
            history = (
                previous is None
                or previous["severity"] != severity
                or previous["automatic_action"] != automatic_action
            )
        self.findings[finding_id] = finding
        if history:
            self.history[finding_id] = finding
        if active:
            if previous is not None and previous["finding_id"] != finding_id:
                self.active.discard(previous["finding_id"])
            self.state[key] = finding_id
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
            "metrics_unit": {"ActiveState": "active", "DropInPaths": ""},
            "collector_env": dict(COLLECTOR_ENV),
            "effective_endpoint": REAL,
            "baseline_endpoint": REAL,
            "open_windows": [],
            "leftover_units": [],
            **self.world.preflight_extra,
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
        # A proxy read costs the runner the same exec, but fires no product
        # event (the proxy is not the product).
        self.world.latency()
        return self.world.status()

    def snapshot(self) -> dict[str, Any]:
        self.calls.append("snapshot")
        return self.world.snapshot()

    def arm(self, **options: Any) -> dict[str, Any]:
        self.calls.append("arm")
        return self.world.arm()

    def pause_collector(self, **options: Any) -> dict[str, Any]:
        label = str(options.get("label") or "")
        self.calls.append(f"pause_collector:{label}" if label else "pause_collector")
        return self.world.pause_collector(**options)

    def pause_status(self) -> dict[str, Any]:
        self.calls.append("pause_status")
        return self.world.pause_status()

    def resume_collector(self, **options: Any) -> dict[str, Any]:
        label = str(options.get("label") or "")
        self.calls.append(f"resume_collector:{label}" if label else "resume_collector")
        if label:
            return self.world.resume_collector(label=label)
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
