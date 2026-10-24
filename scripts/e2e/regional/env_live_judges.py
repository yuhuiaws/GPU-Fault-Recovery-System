"""Judge one GF-ENV-LIVE replay on the product's own records, poll by poll.

``env_live_watch`` polls; a judge here accumulates each poll's documents --
``general_live_gpu_store.gpu_state`` (the accepted GPU_METRICS batches with the
replayed fields' samples, the latest accepted readings, the node's GPU findings
and the GPU collector status), ``general_live_store.node_activity`` (the
node's incidents, workflows with their step executions, notifications) and the
Node document -- and answers, against the case's ``Expectation``
(``env_live_expectations``):

* INCONCLUSIVE, never PASS, when the replay was not consumed, aborted on a real
  safety reading, or no accepted record proves the replayed reading reached the
  control plane;
* FAIL when the reading was ingested and the expected reaction is missing or
  wrong: an isolating or forbidden step on a read-only signal, a wrong GPU or
  node scope, a missing workflow where the spec requires one, a confirmed
  finding without its ENVIRONMENT_REVIEW chain and receipts;
* PASS only on positive evidence: the graded finding, the workflow and its
  receipts, and for a signal the spec keeps read-only a hold after the replay
  was consumed without a forbidden step.

A decision before the deadline is final only when it is definite (a forbidden
step, every pass condition met); everything else waits for the deadline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from gpu_fault.adapters.common import QUARANTINE_TAINT
from gpu_fault.gpu_environment_policy import CONFIRMATION_MAX_GAP_SECONDS
from gpu_fault.health_recovery_contract import COOLDOWN_SECONDS

from scripts.e2e.regional import env_live_verdicts as verdicts
from scripts.e2e.regional.env_live_expectations import (
    Expectation,
    brake_mask,
    same_value,
)
from scripts.e2e.regional.env_live_watch import Judgement, ReplayProgress, Snapshot
from scripts.e2e.regional.general_live_verdicts import (
    TERMINAL_WORKFLOWS,
    final_execution,
    operations,
    parse_time,
)

#: The receipts ENVIRONMENT_REVIEW must have executed; STOP_WORKLOADS is
#: compiled only for an ACTIVE attempt (``orchestration/families/health.py``).
REVIEW_RECEIPTS = (
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "COLLECT_DIAGNOSTIC_BUNDLE",
    "TRIGGER_HEALTH_SNAPSHOT",
    "ESCALATE_SUPPORT",
)
REVIEW_OPERATIONS = ("FREEZE_EVIDENCE", *REVIEW_RECEIPTS)
SENT_NOTIFICATIONS = frozenset({"SENT", "DUPLICATE"})
THERMAL_DURATION_NAMES = frozenset(
    {"clock_throttle_reasons", "thermal_violation_total_us", "composite:THERMAL_STRESS"}
)
COMPLETENESS_FINDING = "dcgm_field_completeness"
#: Readings that do not move while a GPU idles: a pass-through baseline must
#: deliver them as the real exporter did before the repoint.
STATIC_BASELINE_FIELDS = frozenset(
    {
        "power_limit_w",
        "gpu_slowdown_temperature_c",
        "gpu_shutdown_temperature_c",
        "gpu_slowdown_tlimit_c",
        "gpu_shutdown_tlimit_c",
    }
)
ISOLATION_TAINT_PREFIX = "gpu-fault.io/"
#: The node collector's clock against the runner's: how far before the arm a
#: finding may be stamped and still be this run's.
FINDING_SKEW_SECONDS = 30
EVIDENCE_LIMIT = 40
NO_TASK = (
    "no owned test task: the preflight admits only a node without business "
    "workloads and the product recorded no ACTIVE workload for the target GPU, "
    "so no task progress or suspension is measured (STOP_WORKLOADS is compiled "
    "only for an ACTIVE attempt)"
)


def _time(value: Any) -> datetime | None:
    try:
        return parse_time(value)
    except ValueError:
        return None


def _isolation_taints(node: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("key"))
        for item in node.get("taints") or []
        if str(item.get("key") or "").startswith(ISOLATION_TAINT_PREFIX)
    }


def _why(workflow: Mapping[str, Any]) -> Any:
    return workflow.get("pending_failure_error") or workflow.get("blocked_reasons")


def max_valid_run(
    samples: Sequence[Mapping[str, Any]], triggers: Sequence[float]
) -> int:
    """The longest run of accepted trigger samples the policy would count.

    ``env_live_verdicts.valid_severe_samples`` applied prefix by prefix: a
    clear, a >60 s gap or a boot change restarts the count.
    """

    marked = [
        {
            "severe": any(same_value(item.get("value"), value) for value in triggers),
            "observed_at": item.get("observed_at"),
            "source": "DCGM_EXPORTER",
            "source_boot_id": (item.get("labels") or {}).get("source_boot_id"),
            "pci_bdf": None,
        }
        for item in samples
    ]
    best = 0
    for index in range(len(marked)):
        run, _notes = verdicts.valid_severe_samples(
            marked[: index + 1], max_gap_seconds=CONFIRMATION_MAX_GAP_SECONDS
        )
        best = max(best, run)
    return best


class _Judge:
    """Accumulates what each poll showed; the subclasses decide."""

    reads_node = True

    def __init__(
        self, expectation: Expectation, *, node_before: Mapping[str, Any] | None = None
    ) -> None:
        self.x = expectation
        self.label = expectation.case_id
        before = node_before or {}
        self.cordoned_before = bool(before.get("unschedulable"))
        self.taints_before = _isolation_taints(before)
        self.findings: dict[str, dict[str, Any]] = {}
        self.active_ids: set[str] = set()
        self.samples: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.batches: dict[str, dict[str, Any]] = {}
        self.latest: dict[tuple[str, str], dict[str, Any]] = {}
        self.collector_errors: dict[str, str] = {}
        self.incidents: dict[str, dict[str, Any]] = {}
        self.workflows: dict[str, dict[str, Any]] = {}
        self.notifications: dict[str, dict[str, Any]] = {}
        self.node_timeline: list[dict[str, Any]] = []
        self.seen: dict[str, datetime] = {}
        self.reads = {"gpu_state": 0, "activity": 0, "node": 0}
        self.started_at: datetime | None = None

    # -- intake ----------------------------------------------------------- #
    def observe(self, snapshot: Snapshot, progress: ReplayProgress) -> None:
        self.started_at = progress.started_at
        if snapshot.gpu_state is not None:
            self.reads["gpu_state"] += 1
            self._absorb_state(snapshot.gpu_state, snapshot.at)
        if snapshot.activity is not None:
            self.reads["activity"] += 1
            for kind, key, store in (
                ("incidents", "incident_id", self.incidents),
                ("workflows", "request_id", self.workflows),
                ("notifications", "notification_id", self.notifications),
            ):
                for item in snapshot.activity.get(kind) or []:
                    store[str(item.get(key))] = item
        if snapshot.node is not None:
            self.reads["node"] += 1
            cordoned = bool(snapshot.node.get("unschedulable"))
            taints = _isolation_taints(snapshot.node)
            self.node_timeline.append(
                {
                    "at": snapshot.at,
                    "cordoned": cordoned,
                    "quarantined": QUARANTINE_TAINT in taints,
                    # Isolation this run caused: a cordon or a gpu-fault taint
                    # the node did not carry before the arm.
                    "isolated": (cordoned and not self.cordoned_before)
                    or bool(taints - self.taints_before),
                    "ownership_annotations": sorted(
                        snapshot.node.get("ownership_annotations") or {}
                    ),
                }
            )

    def _absorb_state(self, state: Mapping[str, Any], at: datetime) -> None:
        active = state.get("active_findings") or []
        self.active_ids = {str(item.get("finding_id")) for item in active}
        for item in [*(state.get("findings") or []), *active]:
            key = str(item.get("finding_id"))
            self.findings.setdefault(key, {"first_seen": at.isoformat()}).update(item)
        for batch in state.get("batches") or []:
            batch_id = str(batch.get("batch_id"))
            self.batches[batch_id] = {
                key: batch.get(key)
                for key in ("observed_at", "edge_filter_reasons", "collection_errors")
            } | {"gpus": batch.get("gpus") or []}
            for sample in batch.get("samples") or []:
                if sample.get("gpu") != self.x.gpu:
                    continue
                name = str(sample.get("name"))
                self.samples[(name, str(batch.get("observed_at")), batch_id)] = {
                    "name": name,
                    "value": sample.get("value"),
                    "labels": sample.get("labels") or {},
                    "observed_at": batch.get("observed_at"),
                    "batch_id": batch_id,
                    "edge_filter_reasons": batch.get("edge_filter_reasons") or [],
                }
        for row in state.get("latest") or []:
            if row.get("gpu") == self.x.gpu:
                self.latest[(str(row.get("name")), str(row.get("observed_at")))] = row
        for status in state.get("collector_statuses") or []:
            if str(status.get("collector")) == "GPU_METRICS":
                for error in status.get("errors") or []:
                    self.collector_errors.setdefault(
                        str(error), str(status.get("observed_at"))
                    )

    def mark(self, name: str, at: datetime) -> datetime:
        return self.seen.setdefault(name, at)

    def since_arm(self, value: Any) -> bool:
        """Stamped at or after the arm: a reading the replay can have produced."""

        moment = _time(value)
        return moment is not None and (
            self.started_at is None or moment >= self.started_at
        )

    # -- facts ------------------------------------------------------------ #
    def field_samples(self, name: str | None = None) -> list[dict[str, Any]]:
        wanted = name or self.x.field
        items = [item for key, item in self.samples.items() if key[0] == wanted]
        return sorted(items, key=lambda item: str(item.get("observed_at")))

    def trigger_index(self) -> int | None:
        """The last accepted sample carrying a replayed trigger value."""

        samples = self.field_samples()
        return max(
            (
                index
                for index, item in enumerate(samples)
                if any(same_value(item.get("value"), v) for v in self.x.trigger_values)
            ),
            default=None,
        )

    def after_trigger(self) -> datetime | None:
        """The first accepted reading after the last trigger sample (the clear)."""

        last = self.trigger_index()
        if last is None:
            return None
        for item in self.field_samples()[last + 1 :]:
            moment = _time(item.get("observed_at"))
            if moment is not None:
                return moment
        return None

    def gpu_findings(self, names: Iterable[str]) -> list[dict[str, Any]]:
        """The target GPU's findings on ``names`` this run can have caused.

        A finding stamped well before the arm is a leftover the preflight let
        through, not a reaction.
        """

        wanted = set(names)
        earliest = (
            self.started_at - timedelta(seconds=FINDING_SKEW_SECONDS)
            if self.started_at is not None
            else None
        )
        return sorted(
            (
                item
                for item in self.findings.values()
                if item.get("canonical_name") in wanted
                and (
                    item.get("gpu_uuid") == self.x.gpu
                    or self.x.gpu in (item.get("affected_gpu_uuids") or [])
                )
                and (
                    earliest is None
                    or (_time(item.get("observed_at")) or earliest) >= earliest
                )
            ),
            key=lambda item: str(item.get("observed_at")),
        )

    def incidents_of(
        self, findings: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """NODE_HEALTH incidents of GPU findings (``inc-gpu-<finding id>``)."""

        events = {f"gpu-{item.get('finding_id')}" for item in findings}
        return sorted(
            (
                item
                for item in self.incidents.values()
                if item.get("event_id") in events
            ),
            key=lambda item: str(item.get("created_at")),
        )

    def workflows_of(self, incident: Mapping[str, Any]) -> list[dict[str, Any]]:
        return sorted(
            (
                item
                for item in self.workflows.values()
                if item.get("incident_id") == incident.get("incident_id")
            ),
            key=lambda item: str(item.get("created_at")),
        )

    def forbidden(self) -> list[str]:
        errors = []
        for workflow in self.workflows.values():
            found = sorted(set(operations(workflow)) & verdicts.FORBIDDEN_OPERATIONS)
            if found:
                errors.append(
                    f"{self.label}: workflow {workflow.get('request_id')} plans "
                    f"forbidden {found}"
                )
        return errors

    def newly_isolated(self) -> bool:
        return any(item["isolated"] for item in self.node_timeline)

    def held_now(self) -> bool:
        last = self.node_timeline[-1] if self.node_timeline else {}
        return bool(last.get("cordoned") and last.get("quarantined"))

    def released(self) -> bool:
        """Cordoned and quarantined at one poll, no longer both at a later one."""

        held = False
        for item in self.node_timeline:
            if item["cordoned"] and item["quarantined"]:
                held = True
            elif held:
                return True
        return False

    def task_evidence(
        self,
        findings: Sequence[Mapping[str, Any]],
        workflows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """What the product's records say about an owned test task."""

        owned = sorted(
            {
                str(w)
                for item in findings
                for w in item.get("affected_workload_ids") or []
            }
        )
        active = any(item.get("workload_state") == "ACTIVE" for item in findings)
        stops = [
            final_execution(dict(workflow), "STOP_WORKLOADS")
            for workflow in workflows
            if "STOP_WORKLOADS" in operations(dict(workflow))
        ]
        if not owned and not active and not stops:
            return {"owned_task": None, "measured": False, "reason": NO_TASK}
        executed = [item for item in stops if item is not None]
        return {
            "owned_task": owned or None,
            "workload_state_active": active,
            "measured": True,
            "stop_executions": [
                {"status": item.get("status"), "error": item.get("error")}
                for item in executed
            ],
            "stopped": any(item.get("status") == "SUCCEEDED" for item in executed),
            "progress_measured": False,
            "progress_reason": (
                "this runner reads no training-progress reporter; the product's "
                "STOP_WORKLOADS executions are what it measures"
            ),
        }

    @staticmethod
    def workflow_deadline(workflow: Mapping[str, Any]) -> datetime | None:
        moments = [
            moment
            for moment in (
                _time(workflow.get("execution_deadline")),
                _time(workflow.get("lifetime_deadline_at")),
            )
            if moment is not None
        ]
        return min(moments) if moments else None

    @classmethod
    def workflow_overdue(cls, workflow: Mapping[str, Any], now: datetime) -> bool:
        deadline = cls.workflow_deadline(workflow)
        return deadline is not None and now >= deadline

    @classmethod
    def running_until(
        cls, workflows: Iterable[Mapping[str, Any]], now: datetime
    ) -> datetime | None:
        """The latest own deadline of the workflows still running inside it
        (a far moment for one whose deadline is not stamped yet)."""

        moments = []
        for workflow in workflows:
            if str(workflow.get("status")) in TERMINAL_WORKFLOWS:
                continue
            deadline = cls.workflow_deadline(workflow)
            if deadline is None:
                moments.append(now + timedelta(days=1))
            elif now < deadline:
                moments.append(deadline)
        return max(moments) if moments else None

    def wait_until(self, progress: ReplayProgress, now: datetime) -> datetime | None:
        return None

    # -- output ----------------------------------------------------------- #
    def base_evidence(self, progress: ReplayProgress) -> dict[str, Any]:
        def pick(item: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
            return {key: item.get(key) for key in keys}

        names = {str(self.x.field), *THERMAL_DURATION_NAMES, COMPLETENESS_FINDING}
        return {
            "expectation": self.x.evidence(),
            "replay": progress.evidence(),
            "store_reads": dict(self.reads),
            "first_seen": {key: value.isoformat() for key, value in self.seen.items()},
            "findings": [
                pick(
                    item,
                    (
                        "finding_id",
                        "canonical_name",
                        "severity",
                        "value",
                        "confirmation_count",
                        "threshold_source",
                        "automatic_action",
                        "observed_at",
                        "first_seen",
                    ),
                )
                for item in self.gpu_findings(names | {"gpu_temperature_c"})
            ][-EVIDENCE_LIMIT:],
            "accepted_samples": [
                pick(item, ("name", "value", "observed_at", "edge_filter_reasons"))
                for item in sorted(
                    self.samples.values(), key=lambda item: str(item.get("observed_at"))
                )
            ][-EVIDENCE_LIMIT:],
            "incidents": [
                pick(
                    item,
                    (
                        "incident_id",
                        "event_id",
                        "state",
                        "effective_action",
                        "gpu_uuids",
                        "node_ids",
                        "created_at",
                    ),
                )
                for item in self.incidents.values()
            ][-EVIDENCE_LIMIT:],
            "workflows": [
                {
                    **pick(
                        item,
                        (
                            "request_id",
                            "incident_id",
                            "status",
                            "completed_operations",
                            "execution_deadline",
                        ),
                    ),
                    "operations": operations(item),
                }
                for item in self.workflows.values()
            ][-EVIDENCE_LIMIT:],
            "notifications": list(self.notifications.values())[-EVIDENCE_LIMIT:],
            "node": [
                {**item, "at": item["at"].isoformat()} for item in self.node_timeline
            ][-EVIDENCE_LIMIT:],
            "collector_errors": dict(
                list(self.collector_errors.items())[-EVIDENCE_LIMIT:]
            ),
        }

    def judgement(
        self,
        progress: ReplayProgress,
        stages: dict[str, list[str]],
        inconclusive: list[str],
        evidence: dict[str, Any],
    ) -> Judgement:
        failed = any(stages.values())
        outcome = "FAIL" if failed else ("INCONCLUSIVE" if inconclusive else "PASS")
        return Judgement(
            outcome=outcome,
            stages=stages,
            inconclusive=[] if failed else inconclusive,
            evidence={**self.base_evidence(progress), **evidence},
        )


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-001: pass-through baseline after the repoint
# --------------------------------------------------------------------------- #
class BaselineJudge(_Judge):
    """Fresh accepted readings through the proxy, unchanged, no new action."""

    def __init__(
        self,
        expectation: Expectation,
        *,
        repoint_at: datetime,
        before: Mapping[str, Any] | None,
        collector_mode: str,
        node_before: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(expectation, node_before=node_before)
        self.repoint_at = repoint_at
        self.collector_mode = collector_mode
        self.before = {
            str(row.get("name")): row.get("value")
            for row in (before or {}).get("latest") or []
            if row.get("gpu") == expectation.gpu
        }

    def fresh_batch(self) -> dict[str, Any] | None:
        """The repointed collector's first delivery (``initial-baseline``)."""

        for batch_id, batch in sorted(
            self.batches.items(), key=lambda item: str(item[1].get("observed_at"))
        ):
            observed = _time(batch.get("observed_at"))
            if (
                observed is not None
                and observed >= self.repoint_at
                and "initial-baseline" in (batch.get("edge_filter_reasons") or [])
                and self.x.gpu in batch["gpus"]
            ):
                return {**batch, "batch_id": batch_id}
        return None

    def fresh_readings(self) -> dict[str, dict[str, Any]]:
        """The target GPU's readings accepted from the repointed collector.

        Every batch from its ``initial-baseline`` on (the restart's first
        delivery) and every latest reading stamped after the repoint: a first
        batch that lacks the cached temperature limits is completed by a later
        one rather than judged alone.
        """

        batch = self.fresh_batch()
        result: dict[str, dict[str, Any]] = {}
        if batch is not None:
            first = str(batch.get("observed_at"))
            for item in sorted(
                self.samples.values(), key=lambda item: str(item.get("observed_at"))
            ):
                if str(item.get("observed_at")) >= first:
                    result.setdefault(str(item["name"]), item)
        for (name, _observed), row in sorted(self.latest.items()):
            observed = _time(row.get("observed_at"))
            if observed is not None and observed > self.repoint_at:
                result.setdefault(name, row)
        return result

    def decide(
        self, progress: ReplayProgress, now: datetime, *, final: bool
    ) -> Judgement | None:
        label = self.label
        stages: dict[str, list[str]] = {
            "baseline": [],
            "passthrough_values": [
                f"{label}: {item}" for item in progress.passthrough_violations
            ],
        }
        gpu_incidents = [
            item
            for item in self.incidents.values()
            if str(item.get("event_id") or "").startswith("gpu-")
        ]
        actions = [
            f"incident {item.get('incident_id')} ({item.get('state')}) "
            f"{[operations(flow) for flow in self.workflows_of(item)]}"
            for item in gpu_incidents
        ]
        if actions:
            stages["baseline"].append(
                f"{label}: the pass-through baseline created actions {actions}"
            )
        readings = self.fresh_readings()
        if readings:
            self.mark("fresh_readings", now)
        stages["passthrough_values"].extend(
            f"{label}: the pass-through changed {name}: {self.before[name]} before "
            f"the repoint, {item.get('value')} through it"
            for name, item in sorted(readings.items())
            if name in STATIC_BASELINE_FIELDS
            and name in self.before
            and not same_value(self.before[name], item.get("value"))
        )
        boots = sorted(
            {
                str((item.get("labels") or {}).get("source_boot_id"))
                for item in readings.values()
                if (item.get("labels") or {}).get("source_boot_id")
            }
        )
        boot_ok = bool(boots) and (self.x.boot_id is None or boots == [self.x.boot_id])
        if boots and not boot_ok:
            stages["baseline"].append(
                f"{label}: accepted readings carry source_boot_id {boots}, the node "
                f"booted {self.x.boot_id}"
            )
        served = progress.served or 0
        evidence = {
            "fresh_batch": self.fresh_batch(),
            "fresh_fields": sorted(readings),
            "unchanged_static_fields": sorted(
                set(readings) & STATIC_BASELINE_FIELDS & set(self.before)
            ),
            "passthrough_reads": served,
            "task": {"owned_task": None, "measured": False, "reason": NO_TASK},
        }
        if any(stages.values()):
            return self.judgement(progress, stages, [], evidence)
        coverage = verdicts.baseline_errors(
            per_gpu_readings={self.x.gpu: set(readings)},
            unsupported={},
            valid_sample_count=served,
            source_boot_id=boots[0] if boot_ok else None,
            collector_mode=self.collector_mode or None,
            new_environment_actions=[],
            task_progressed=None,
        )
        # A missing field may still arrive with a later batch: it fails only
        # at the deadline, once the pass-through itself was shown.
        if progress.consumed and readings and not coverage:
            return self.judgement(progress, stages, [], evidence)
        if not final:
            return None
        pending = progress.inconclusive()
        if progress.consumed and readings:
            stages["baseline"] = coverage
        elif not readings:
            pending.append(
                f"{label}: no GPU_METRICS batch of the repointed collector reached "
                "the control plane for the target GPU"
            )
        return self.judgement(progress, stages, pending, evidence)


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-004 / 011 / 007 (sustained, mixed): confirmation to human review
# --------------------------------------------------------------------------- #
@dataclass
class ReviewFacts:
    run: int = 0
    finding: dict[str, Any] | None = None
    incident: dict[str, Any] | None = None
    workflow: dict[str, Any] | None = None
    receipts_ok: bool = False
    escalated: bool = False
    notification_statuses: list[str] = field(default_factory=list)
    recovered: datetime | None = None
    hold_end: datetime | None = None
    held: bool = False
    task: dict[str, Any] = field(default_factory=dict)
    #: The final execution status of each receipt the review must carry.
    executed: dict[str, Any] = field(default_factory=dict)

    @property
    def notified(self) -> bool:
        return any(
            status in SENT_NOTIFICATIONS for status in self.notification_statuses
        )


class ReviewJudge(_Judge):
    """N accepted severe samples, ENVIRONMENT_REVIEW, receipts, a held node.

    The same pass conditions as ``env_live_verdicts.severe_review_errors``,
    taken in order on the polled records, so that what the window could not
    show stays INCONCLUSIVE instead of failing or passing.
    """

    def confirmed(self) -> dict[str, Any] | None:
        for item in self.gpu_findings({str(self.x.field)}):
            if item.get("severity") == "CRITICAL" and any(
                same_value(item.get("value"), value) for value in self.x.trigger_values
            ):
                return item
        return None

    def review(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        finding = self.confirmed()
        incidents = self.incidents_of([finding]) if finding is not None else []
        if not incidents:
            return None, None
        incident = incidents[0]
        workflows = self.workflows_of(incident)
        linked = self.workflows.get(str(incident.get("workflow_request_id")))
        return incident, linked or (workflows[-1] if workflows else None)

    def hold_end(self, progress: ReplayProgress) -> datetime | None:
        """``COOLDOWN_SECONDS`` + one interval after the replay ended and the
        cooled reading was accepted (whichever came later)."""

        if progress.consumed_at is None:
            return None
        recovered = self.after_trigger()
        start = max(
            moment for moment in (progress.consumed_at, recovered) if moment is not None
        )
        return start + timedelta(seconds=self.x.hold_seconds)

    def wait_until(self, progress: ReplayProgress, now: datetime) -> datetime | None:
        _incident, workflow = self.review()
        moments = [
            self.running_until([workflow] if workflow else [], now),
            self.hold_end(progress) if self.confirmed() is not None else None,
        ]
        present = [moment for moment in moments if moment is not None]
        return max(present) if present else None

    def facts(self, progress: ReplayProgress) -> ReviewFacts:
        facts = ReviewFacts(
            run=max_valid_run(self.field_samples(), self.x.trigger_values)
        )
        facts.finding = self.confirmed()
        facts.incident, facts.workflow = self.review()
        facts.recovered = self.after_trigger()
        facts.hold_end = self.hold_end(progress)
        facts.held = facts.hold_end is not None and any(
            item["at"] >= facts.hold_end and item["cordoned"] and item["quarantined"]
            for item in self.node_timeline
        )
        if facts.incident is not None:
            facts.escalated = facts.incident.get("state") == "ESCALATED"
            facts.notification_statuses = sorted(
                str(item.get("status"))
                for item in self.notifications.values()
                if item.get("incident_id") == facts.incident.get("incident_id")
            )
        facts.task = self.task_evidence(
            [facts.finding] if facts.finding is not None else [],
            [facts.workflow] if facts.workflow is not None else [],
        )
        if facts.workflow is not None:
            steps = operations(facts.workflow)
            wanted = [
                *REVIEW_RECEIPTS,
                *(["STOP_WORKLOADS"] if "STOP_WORKLOADS" in steps else []),
            ]
            facts.executed = {
                name: (final_execution(facts.workflow, name) or {}).get("status")
                for name in wanted
            }
            facts.receipts_ok = str(
                facts.workflow.get("status")
            ) == "SUCCEEDED" and all(
                value == "SUCCEEDED" for value in facts.executed.values()
            )
        return facts

    def errors(self, facts: ReviewFacts, now: datetime) -> dict[str, list[str]]:
        """Definite failures: a decision the rest of the window cannot change."""

        label = self.label
        stages: dict[str, list[str]] = {
            "reaction": self.forbidden(),
            "grading": [],
            "scope": [],
            "workflow": [],
            "isolation": [],
            "task": [],
        }
        if facts.finding is not None:
            self.mark("confirmed_finding", now)
            source = facts.finding.get("threshold_source")
            if self.x.threshold_source and source and source != self.x.threshold_source:
                stages["grading"].append(
                    f"{label}: confirmed on {source}, the device grades on "
                    f"{self.x.threshold_source}"
                )
            policy = facts.finding.get("policy_source")
            if self.x.power_delivery and policy not in {
                "SITE_NVIDIA_POWER_DELIVERY",
                None,
            }:
                stages["grading"].append(
                    f"{label}: the brake finding is {policy}, not a power-delivery check"
                )
        reviewed = (
            {str(facts.workflow.get("request_id"))}
            if facts.workflow is not None
            else set()
        )
        review_incident = (facts.incident or {}).get("incident_id")
        for other in self.incidents_of(self.gpu_findings({str(self.x.field)})):
            if other.get("incident_id") == review_incident:
                continue
            for item in self.workflows_of(other):
                if str(item.get("request_id")) in reviewed:
                    continue  # the review was merged into the earlier workflow
                isolating = sorted(
                    set(operations(item)) & verdicts.ISOLATING_OPERATIONS
                )
                if isolating:
                    stages["reaction"].append(
                        f"{label}: a pre-confirmation severe sample already planned "
                        f"{isolating} (workflow {item.get('request_id')})"
                    )
        for item in self.workflows.values():
            restored = "RESTORE_SCHEDULING" in operations(item) and final_execution(
                item, "RESTORE_SCHEDULING"
            )
            if restored or (
                str(item.get("request_id")) in reviewed
                and "RESTORE_SCHEDULING" in operations(item)
            ):
                stages["isolation"].append(
                    f"{label}: workflow {item.get('request_id')} restores scheduling "
                    "inside the review hold"
                )
        if self.released():
            stages["isolation"].append(
                f"{label}: the node was released after the review isolated it"
            )
        if facts.incident is not None:
            self.mark("review_incident", now)
            for key, wanted in (("gpu_uuids", self.x.gpu), ("node_ids", self.x.node)):
                if list(facts.incident.get(key) or []) != [wanted]:
                    stages["scope"].append(
                        f"{label}: the review incident scopes {key} "
                        f"{facts.incident.get(key)}, not [{wanted}]"
                    )
        if facts.workflow is not None:
            self._workflow_errors(facts, stages, now)
        return stages

    def _workflow_errors(
        self, facts: ReviewFacts, stages: dict[str, list[str]], now: datetime
    ) -> None:
        label, workflow = self.label, facts.workflow
        assert workflow is not None, "only called with the review workflow"
        self.mark("review_workflow", now)
        steps = operations(workflow)
        missing = [name for name in REVIEW_OPERATIONS if name not in steps]
        if missing:
            stages["workflow"].append(
                f"{label}: ENVIRONMENT_REVIEW workflow {workflow.get('request_id')} "
                f"lacks {missing} (steps {steps})"
            )
        if facts.task.get("workload_state_active") and "STOP_WORKLOADS" not in steps:
            stages["task"].append(
                f"{label}: the owned ACTIVE task {facts.task.get('owned_task')} is "
                "not stopped by the review"
            )
        status = str(workflow.get("status"))
        if status in TERMINAL_WORKFLOWS and status != "SUCCEEDED":
            stages["workflow"].append(
                f"{label}: ENVIRONMENT_REVIEW workflow ended {status}: {_why(workflow)}"
            )
        if facts.executed.get("STOP_WORKLOADS") == "FAILED":
            stages["task"].append(f"{label}: the owned task's STOP_WORKLOADS failed")
        if status == "SUCCEEDED":
            self.mark("review_succeeded", now)
            failed = sorted(
                name for name, value in facts.executed.items() if value != "SUCCEEDED"
            )
            if failed:
                stages["workflow"].append(
                    f"{label}: the review SUCCEEDED without receipts for {failed}"
                )

    def evidence(self, facts: ReviewFacts) -> dict[str, Any]:
        def iso(value: datetime | None) -> str | None:
            return None if value is None else value.isoformat()

        return {
            "valid_severe_run": facts.run,
            "confirmed_finding": facts.finding,
            "review_incident": (facts.incident or {}).get("incident_id"),
            "review_workflow": (facts.workflow or {}).get("request_id"),
            "hold": {
                "end": iso(facts.hold_end),
                "recovered_reading_at": iso(facts.recovered),
                "held": facts.held,
                "seconds": self.x.hold_seconds,
            },
            "notification_statuses": facts.notification_statuses,
            "task": facts.task,
        }

    def complete(self, progress: ReplayProgress, facts: ReviewFacts) -> bool:
        return bool(
            progress.consumed
            and facts.finding is not None
            and facts.receipts_ok
            and facts.escalated
            and facts.notified
            and self.held_now()
            and facts.held
            and (
                not facts.task.get("workload_state_active") or facts.task.get("stopped")
            )
        )

    def decide(
        self, progress: ReplayProgress, now: datetime, *, final: bool
    ) -> Judgement | None:
        facts = self.facts(progress)
        stages = self.errors(facts, now)
        evidence = self.evidence(facts)
        if any(stages.values()) or self.complete(progress, facts):
            return self.judgement(progress, stages, [], evidence)
        if not final:
            return None
        pending = progress.inconclusive() or self.unfinished(facts, stages, now)
        return self.judgement(progress, stages, pending, evidence)

    def unfinished(
        self, facts: ReviewFacts, stages: dict[str, list[str]], now: datetime
    ) -> list[str]:
        """At the deadline: what is missing, as a failure or as a gap."""

        label = self.label
        if facts.finding is None:
            if facts.run < self.x.required_samples:
                return [
                    f"{label}: the control plane accepted {facts.run} valid severe "
                    f"{self.x.field} samples of the {self.x.required_samples} the "
                    "confirmation needs (a delivery gap, a >60s spacing or a boot "
                    "change restarts the count), so the case cannot be judged"
                ]
            stages["grading"].append(
                f"{label}: {facts.run} valid severe {self.x.field} samples were "
                "accepted but no CRITICAL finding confirmed them"
            )
            return []
        if facts.incident is None or facts.workflow is None:
            stages["workflow"].append(
                f"{label}: no ENVIRONMENT_REVIEW workflow was observed after the "
                "confirmed finding"
            )
            return []
        workflow = facts.workflow
        status = str(workflow.get("status"))
        if status not in TERMINAL_WORKFLOWS:
            if self.workflow_overdue(workflow, now):
                stages["workflow"].append(
                    f"{label}: workflow {workflow.get('request_id')} is still {status} "
                    "past its own execution deadline"
                )
                return []
            return [
                f"{label}: the replay window closed with workflow "
                f"{workflow.get('request_id')} still {status} inside its own "
                f"execution deadline {workflow.get('execution_deadline')} "
                f"(completed {workflow.get('completed_operations')})"
            ]
        pending: list[str] = []
        if not facts.escalated:
            stages["workflow"].append(
                f"{label}: incident state is {facts.incident.get('state')}, not "
                "ESCALATED"
            )
        if not facts.notified:
            # QUEUED, or no delivery result recorded yet: still on its way.
            if facts.notification_statuses and set(facts.notification_statuses) <= {
                "QUEUED",
                "None",
            }:
                pending.append(
                    f"{label}: the review notification has no delivery result yet "
                    f"({facts.notification_statuses})"
                )
            else:
                stages["workflow"].append(
                    f"{label}: no delivered notification receipt for the review "
                    f"incident (statuses {facts.notification_statuses})"
                )
        last = self.node_timeline[-1] if self.node_timeline else {}
        if not last.get("cordoned"):
            stages["isolation"].append(f"{label}: node was not cordoned")
        if not last.get("quarantined"):
            stages["isolation"].append(
                f"{label}: quarantine taint {QUARANTINE_TAINT} is absent"
            )
        if facts.task.get("workload_state_active") and not facts.task.get("stopped"):
            stages["task"].append(f"{label}: the owned ACTIVE task was not stopped")
        if not facts.held and not any(stages.values()):
            pending.append(
                f"{label}: the cooled reading never reached the control plane, so "
                "the hold after cooling is not shown"
                if facts.recovered is None
                else f"{label}: the replay window closed before the "
                f"{self.x.hold_seconds:.0f}s hold past the old {COOLDOWN_SECONDS}s "
                f"cooldown ended ({facts.hold_end})"
            )
        return pending


# --------------------------------------------------------------------------- #
# Read-only signals: 003 warning, 005/007 transient, 006 duration, 007 non-brake,
# 008 unusable field
# --------------------------------------------------------------------------- #
class ReadOnlyJudge(_Judge):
    """Positive grading evidence, then a hold without a forbidden step."""

    def related_findings(self) -> list[dict[str, Any]]:
        if self.x.kind == "duration":
            return self.gpu_findings(THERMAL_DURATION_NAMES)
        return self.gpu_findings({str(self.x.field)})

    def diagnostics(self) -> list[dict[str, Any]]:
        return [
            workflow
            for incident in self.incidents_of(self.related_findings())
            for workflow in self.workflows_of(incident)
        ]

    def hold_end(self, progress: ReplayProgress) -> datetime | None:
        graded = self.seen.get("graded")
        if graded is None or progress.consumed_at is None:
            return None
        return max(graded, progress.consumed_at) + timedelta(
            seconds=self.x.hold_seconds
        )

    def clear_deadline(self, progress: ReplayProgress) -> datetime | None:
        """By when the edge filter must have delivered the clearing reading."""

        if progress.consumed_at is None:
            return None
        return progress.consumed_at + timedelta(seconds=2 * self.x.hold_seconds)

    def wait_until(self, progress: ReplayProgress, now: datetime) -> datetime | None:
        moments = [self.hold_end(progress)]
        if self.x.require_diagnostic:
            moments.append(self.running_until(self.diagnostics(), now))
        if self.x.require_clear and self.cleared_at() is None:
            moments.append(self.clear_deadline(progress))
        present = [moment for moment in moments if moment is not None]
        return max(present) if present else None

    def trigger_samples(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.field_samples()
            if self.since_arm(item.get("observed_at"))
            and any(same_value(item.get("value"), v) for v in self.x.trigger_values)
        ]

    def grading(self) -> tuple[list[str], list[dict[str, Any]]]:
        """Grading errors, and the findings that prove the reading was graded."""

        label, kind = self.label, self.x.kind
        errors: list[str] = []
        proof: list[dict[str, Any]] = []
        for item in self.related_findings():
            severity, value = item.get("severity"), item.get("value")
            replayed = any(same_value(value, v) for v in self.x.trigger_values)
            if severity == "CRITICAL" or item.get("automatic_action") == "DRAIN":
                errors.append(
                    f"{label}: {item.get('canonical_name')}={value} was graded "
                    f"{severity}/{item.get('automatic_action')} on a read-only signal"
                )
            elif kind == "no_finding":
                errors.append(
                    f"{label}: ordinary bit {value} was graded {severity} "
                    f"({item.get('threshold_source') or item.get('reason')})"
                )
            elif kind == "unknown":
                if brake_mask(value) or item.get("policy_source") == (
                    "SITE_NVIDIA_POWER_DELIVERY"
                ):
                    errors.append(
                        f"{label}: an unusable mask was read as a Power Brake"
                    )
            elif kind in {"warning", "transient"} and not replayed:
                continue
            elif kind == "transient" and int(item.get("confirmation_count") or 0) > (
                self.x.max_confirmation
            ):
                errors.append(
                    f"{label}: the count reached {item.get('confirmation_count')} "
                    f"after {self.x.max_confirmation} transient severe sample(s)"
                )
            elif (
                kind == "warning"
                and self.x.threshold_source
                and item.get("threshold_source")
                and item.get("threshold_source") != self.x.threshold_source
            ):
                errors.append(
                    f"{label}: the warning was graded on {item.get('threshold_source')}"
                    f", the device grades on {self.x.threshold_source}"
                )
            else:
                proof.append(item)
        if self.x.fallback_field:
            errors.extend(
                f"{label}: T.Limit grades the GPU but {self.x.fallback_field} also "
                f"raised {item.get('severity')} ({item.get('threshold_source')})"
                for item in self.gpu_findings({self.x.fallback_field})
            )
        return errors, proof

    def unknown_evidence(self) -> list[str]:
        """Records that report the unusable field as UNKNOWN / collection error."""

        texts = [
            *(
                error
                for error, observed in self.collector_errors.items()
                if self.since_arm(observed)
            ),
            *(
                str(error)
                for batch in self.batches.values()
                if self.since_arm(batch.get("observed_at"))
                for error in batch.get("collection_errors") or []
            ),
            *(
                str(item.get("reason"))
                for item in self.gpu_findings({COMPLETENESS_FINDING})
            ),
            *(
                str(item.get("reason"))
                for item in self.findings.values()
                if item.get("canonical_name") == COMPLETENESS_FINDING
                and item.get("gpu_uuid") is None
                and self.since_arm(item.get("observed_at"))
            ),
        ]
        return sorted({text for text in texts if str(self.x.field) in text})

    def accepted_without_field(self, progress: ReplayProgress) -> list[str]:
        """Accepted batches with the target GPU but without its replayed field.

        Only a replayed read lacks the field (the real exporter reports it, or
        the tool refuses the plan), so such a batch is a replayed read the
        control plane accepted. A health-summary batch is not kept as raw
        evidence; its trace is the latest table: the GPU's temperature refreshed
        after the arm while the field's latest stayed older.
        """

        present: dict[str, set[str]] = {}
        for name, _observed, batch_id in self.samples:
            present.setdefault(batch_id, set()).add(name)
        result = sorted(
            batch_id
            for batch_id, names in present.items()
            if "gpu_temperature_c" in names and str(self.x.field) not in names
        )
        newest: dict[str, datetime] = {}
        for name, observed in self.latest:
            moment = _time(observed)
            if moment is not None and (name not in newest or moment > newest[name]):
                newest[name] = moment
        temperature = newest.get("gpu_temperature_c")
        reading = newest.get(str(self.x.field))
        started = progress.started_at
        if (
            temperature is not None
            and started is not None
            and temperature >= started
            and (reading is None or reading < temperature)
        ):
            result.append(f"latest gpu_temperature_c at {temperature.isoformat()}")
        return result

    def graded(self, now: datetime) -> list[str]:
        """Positive evidence the reading was ingested and graded (marks when)."""

        kind = self.x.kind
        found: list[str] = []
        if kind in {"warning", "transient", "duration"}:
            found = [str(item.get("finding_id")) for item in self.grading()[1]]
        elif kind == "no_finding":
            rows = [
                row
                for row in self.latest.values()
                if row.get("name") == self.x.field
                and self.since_arm(row.get("observed_at"))
                and any(same_value(row.get("value"), v) for v in self.x.trigger_values)
            ]
            found = [
                f"accepted {self.x.field}={item.get('value')} at "
                f"{item.get('observed_at')}"
                for item in [*self.trigger_samples(), *rows]
            ]
        elif kind == "unknown":
            found = self.unknown_evidence()
        if found:
            self.mark("graded", now)
        return found

    def cleared_at(self) -> datetime | None:
        """The accepted clearing reading, once no finding on the field is active."""

        active = {
            str(self.findings[key].get("canonical_name"))
            for key in self.active_ids
            if key in self.findings and self.findings[key].get("gpu_uuid") == self.x.gpu
        }
        return None if self.x.field in active else self.after_trigger()

    def errors(self) -> dict[str, list[str]]:
        """Definite failures: a decision the rest of the window cannot change."""

        label = self.label
        stages: dict[str, list[str]] = {
            "reaction": [],
            "isolation": [],
            "grading": self.grading()[0],
            "diagnostic": [],
            "clear": [],
        }
        for workflow in self.workflows.values():
            incident = self.incidents.get(str(workflow.get("incident_id"))) or {}
            stages["reaction"].extend(
                verdicts.read_only_diagnostic_errors(
                    workflow=workflow,
                    incident_state=incident.get("state"),
                    node_isolated=False,
                    task_progressed=None,
                    label=f"{label} workflow {workflow.get('request_id')}",
                )
            )
        with_workflow = {
            str(item.get("incident_id")) for item in self.workflows.values()
        }
        stages["reaction"].extend(
            f"{label}: incident {key} is {item.get('state')} on a read-only signal"
            for key, item in self.incidents.items()
            if key not in with_workflow
            and item.get("state") in {"QUARANTINED", "ESCALATED"}
        )
        if self.newly_isolated():
            stages["isolation"].append(
                f"{label}: node was isolated for a read-only signal"
            )
        if self.x.require_diagnostic:
            for workflow in self.diagnostics():
                status = str(workflow.get("status"))
                if status in TERMINAL_WORKFLOWS and status != "SUCCEEDED":
                    stages["diagnostic"].append(
                        f"{label}: diagnostic workflow {workflow.get('request_id')} "
                        f"ended {status}: {_why(workflow)}"
                    )
                validation = final_execution(workflow, "VALIDATE_GPU") or {}
                if validation.get("status") == "FAILED":
                    stages["diagnostic"].append(
                        f"{label}: VALIDATE_GPU failed: "
                        f"{str(validation.get('error') or '')[:200]}"
                    )
        return stages

    def decide(
        self, progress: ReplayProgress, now: datetime, *, final: bool
    ) -> Judgement | None:
        stages = self.errors()
        found = self.graded(now)
        diagnostics = self.diagnostics()
        cleared = self.cleared_at() if self.x.require_clear else None
        hold_end = self.hold_end(progress)
        evidence = {
            "graded_evidence": found[:EVIDENCE_LIMIT],
            "graded_at": (self.seen.get("graded") or now).isoformat()
            if found
            else None,
            "hold_end": None if hold_end is None else hold_end.isoformat(),
            "diagnostic_workflows": [item.get("request_id") for item in diagnostics],
            "cleared_at": None if cleared is None else cleared.isoformat(),
            "task": self.task_evidence(
                self.related_findings(), list(self.workflows.values())
            ),
        }
        diagnosed = bool(diagnostics) and all(
            str(item.get("status")) == "SUCCEEDED" for item in diagnostics
        )
        complete = (
            progress.consumed
            and hold_end is not None
            and now >= hold_end
            and (diagnosed or not self.x.require_diagnostic)
            and (cleared is not None or not self.x.require_clear)
        )
        if any(stages.values()) or complete:
            return self.judgement(progress, stages, [], evidence)
        if not final:
            return None
        pending = progress.inconclusive() or self.unfinished(
            progress, stages, now, found=found, diagnosed=diagnosed, cleared=cleared
        )
        return self.judgement(progress, stages, pending, evidence)

    def unfinished(
        self,
        progress: ReplayProgress,
        stages: dict[str, list[str]],
        now: datetime,
        *,
        found: list[str],
        diagnosed: bool,
        cleared: datetime | None,
    ) -> list[str]:
        """At the deadline: what is missing, as a failure or as a gap."""

        label = self.label
        if not found:
            accepted = self.trigger_samples()
            missing = (
                self.accepted_without_field(progress)
                if self.x.kind == "unknown"
                else []
            )
            if self.x.kind in {"warning", "transient", "duration"} and accepted:
                stages["grading"].append(
                    f"{label}: {len(accepted)} accepted {self.x.field} sample(s) "
                    f"{sorted({item.get('value') for item in accepted})} were never "
                    "graded into a finding"
                )
                return []
            if missing:
                stages["grading"].append(
                    f"{label}: {missing[:3]} accepted the target GPU without its "
                    f"{self.x.field} and nothing reported it UNKNOWN (the "
                    "field-missing GPU was judged by the batch's other GPUs)"
                )
                return []
            return [
                f"{label}: no accepted record proves the replayed {self.x.field} "
                "reached the control plane and was graded"
                + (f" ({self.x.notes[0]})" if self.x.notes else "")
            ]
        pending: list[str] = []
        diagnostics = self.diagnostics()
        if self.x.require_diagnostic and not diagnosed:
            late = [item for item in diagnostics if self.workflow_overdue(item, now)]
            if not diagnostics:
                stages["diagnostic"].append(
                    f"{label}: the graded reading opened no read-only diagnostic "
                    "workflow (the spec requires the validation to complete)"
                )
            elif late:
                stages["diagnostic"].append(
                    f"{label}: diagnostic workflow {late[0].get('request_id')} is still "
                    "running past its own execution deadline"
                )
            else:
                pending.append(
                    f"{label}: the replay window closed with the diagnostic workflow "
                    "still running inside its own deadline"
                )
        settle = self.clear_deadline(progress)
        if self.x.require_clear and cleared is None:
            if settle is not None and now >= settle:
                stages["clear"].append(
                    f"{label}: the clearing reading did not pass the edge filter (no "
                    f"accepted {self.x.field} after the transient, or the finding "
                    "stayed active)"
                )
            else:
                pending.append(
                    f"{label}: the clear was not accepted before the deadline"
                )
        hold_end = self.hold_end(progress)
        if hold_end is not None and now < hold_end and not any(stages.values()):
            pending.append(
                f"{label}: the replay window closed before the "
                f"{self.x.hold_seconds:.0f}s hold ended ({hold_end.isoformat()})"
            )
        return pending


def judge_for(
    expectation: Expectation,
    *,
    repoint_at: datetime,
    before: Mapping[str, Any] | None = None,
    collector_mode: str = "dcgm",
    node_before: Mapping[str, Any] | None = None,
) -> BaselineJudge | ReviewJudge | ReadOnlyJudge:
    if expectation.kind == "baseline":
        return BaselineJudge(
            expectation,
            repoint_at=repoint_at,
            before=before,
            collector_mode=collector_mode,
            node_before=node_before,
        )
    if expectation.kind == "review":
        return ReviewJudge(expectation, node_before=node_before)
    return ReadOnlyJudge(expectation, node_before=node_before)


__all__ = [
    "BaselineJudge",
    "NO_TASK",
    "QUARANTINE_TAINT",
    "REVIEW_OPERATIONS",
    "REVIEW_RECEIPTS",
    "ReadOnlyJudge",
    "ReviewFacts",
    "ReviewJudge",
    "judge_for",
    "max_valid_run",
]
