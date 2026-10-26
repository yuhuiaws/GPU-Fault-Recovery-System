"""GF-ENV-LIVE-009: the collection interruption, judged on the product's records.

The spec (``docs/components/gpu-environment-live-acceptance.md``, 009): after
two severe samples the collection is paused through the reviewed single-node
maintenance so that the accepted samples lie more than 60 s apart, then
resumed; the confirmation must count from 1 again. At a normal temperature the
collector is restarted once; freshness and a single producer are verified. The
DCGM -> nvidia-smi source switch runs only through the normal installation and
a cross-boot re-confirmation needs an approved node reboot: this runner drives
neither and records both as gaps.

Mechanism (``probes/env_replay_proxy.py``): ``pause-collector`` schedules a
waiter unit on the node that SIGSTOPs ``gpu-fault-metrics-collector.service``
once the proxy has served the pause read, with a ``systemd-run`` deadman
that resumes the unit; ``resume-collector`` and ``restart-collector`` are the
maintenance steps this judge takes, through :class:`InterruptionSteps`, once
the records call for them. Everything else here only reads: the pause state
(the poll's ``extra`` read), the accepted GPU_METRICS batches, the findings and
their ``confirmation_count``, the node's workflows and the Node.

The recount hold. The product keeps a confirmation-count change only in the
current finding state (``store.update_gpu_findings``: the ``gpu_finding_state``
row is overwritten by every accepted reading of the field, the
``gpu_finding_history`` row is written only when the severity or the automatic
action changes, and no incident reason, marker or notification follows a
count change). The post-gap regrade "2/3 -> 1/3" (``gpu_environment_policy.
confirm_environment_decision``: a gap over 60 s makes the previous confirmation
incomparable) is therefore a state that lives exactly one collector interval,
until the clearing read replaces it -- 15 s at the default cadence, while one
poll of this runner costs 46-56 s on a live site (live 2026-10-26, plain EKS:
accepted 13:32:57Z, cleared 13:33:12Z, polled 13:32:20Z and 13:33:16Z,
INCONCLUSIVE "no finding graded it yet"). So, right before the gap pause is
resumed, the judge schedules a second read-synchronous pause (the *recount
hold*, label :data:`RECOUNT_HOLD_LABEL`) one read past the gap pause's target:
the collector stops again as soon as it has posted the first post-gap read, the
regraded state stays readable, and the judge resumes the hold the moment it has
read a finding on that sample (or after ``recount_hold_seconds``, the bound for
failed reads). The clearing read then follows as before. The proof is the
product's own current state, read while it is current -- not a weaker rule.

Verdict rule:

* INCONCLUSIVE, never PASS, when the pause did not land on its read (the
  waiter timed out, died or was late), the plan was not consumed, the gap the
  records show is not wider than the policy's 60 s, no finding on the post-gap
  sample was read (the hold not taken, its waiter dead, late or timed out,
  or the state read fewer times than the hold allows), or the restarted
  collector's first batch was not observed before the deadline;
* FAIL when the gap was shown and the product still confirmed on it (a
  CRITICAL finding, an isolating step, a node isolation), when a reading was
  accepted inside the gap or the pre-gap finding was cleared by the mere
  absence of readings, when the post-gap sample was held readable for the
  hold's polls and no finding graded it, when more than one producer
  delivered, or when the cleared reading never passed the edge filter;
* PASS only on positive evidence: the first post-gap severe sample graded
  ``confirmation 1/N`` (WARNING), the clear accepted, the restarted collector's
  ``initial-baseline`` batch with the node's own boot id, one producer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from gpu_fault.gpu_environment_policy import CONFIRMATION_MAX_GAP_SECONDS

from scripts.e2e.regional import env_live_verdicts as verdicts
from scripts.e2e.regional.env_live_expectations import (
    RECOUNT_HOLD_POLLS,
    Expectation,
    same_value,
)
from scripts.e2e.regional.env_live_judges import BaseJudge
from scripts.e2e.regional.env_live_watch import (
    PROCESSING_MARGIN_SECONDS,
    STARTUP_SPREAD_SECONDS,
    Judgement,
    ReplayProgress,
    Snapshot,
)
from scripts.e2e.regional.general_live_verdicts import operations, parse_time

#: The restarted collector's first delivery: startup spread + one interval +
#: the ingestion path, from the restart receipt.
RESTART_DELIVERY_SECONDS = STARTUP_SPREAD_SECONDS + PROCESSING_MARGIN_SECONDS
#: One collector keeps at most this many batches per interval over a window;
#: a second producer of the same identity doubles the rate.
PRODUCER_RATE_SLACK = 2
#: The probe's label of the recount hold (``pause-collector --label``): its
#: waiter, deadman and record are the gap pause's with this suffix.
RECOUNT_HOLD_LABEL = "hold"


def _time(value: Any) -> datetime | None:
    try:
        return parse_time(value)
    except ValueError:
        return None


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _receipt_time(receipt: Mapping[str, Any] | None, key: str) -> datetime | None:
    """The node's own stamp from a pause/resume receipt (``pause.<key>``)."""

    return _time(((receipt or {}).get("pause") or {}).get(key))


@dataclass
class InterruptionSteps:
    """The maintenance steps the judge takes, bound to the controller.

    ``hold`` schedules the recount hold (right before ``resume``), and
    ``resume_hold`` resumes it; both are optional so that a controller without
    the labelled pause still runs the case (then the regrade is read only if a
    poll happens to land inside its one-interval life, as before).
    """

    resume: Callable[[], dict[str, Any]]
    restart: Callable[[], dict[str, Any]]
    hold: Callable[[], dict[str, Any]] | None = None
    resume_hold: Callable[[], dict[str, Any]] | None = None


@dataclass
class HoldFacts:
    """The recount hold: what the node's record and the judge's steps say."""

    scheduled: bool = False
    schedule_receipt: dict[str, Any] | None = None
    schedule_error: str | None = None
    target_reads: int | None = None
    paused_at: datetime | None = None
    reads_at_pause: int | None = None
    missed: bool = False
    timed_out: bool = False
    waiter_state: str | None = None
    resumed_at: datetime | None = None
    resume_receipt: dict[str, Any] | None = None
    #: Store reads taken while the hold was in place (a poll that started
    #: after the hold landed, before its resume), and those among them that
    #: already showed the post-gap sample's batch: what the hold bought.
    reads_while_held: int = 0
    reads_with_post_sample: int = 0
    #: When a finding on the post-gap sample was first read (any count).
    regrade_seen_at: datetime | None = None

    @property
    def taken(self) -> bool:
        return self.paused_at is not None

    def evidence(self) -> dict[str, Any]:
        return {
            "label": RECOUNT_HOLD_LABEL,
            "scheduled": self.scheduled,
            "schedule_error": self.schedule_error,
            "target_reads": self.target_reads,
            "paused_at": _iso(self.paused_at),
            "reads_at_pause": self.reads_at_pause,
            "missed": self.missed,
            "timed_out": self.timed_out,
            "waiter_state": self.waiter_state,
            "resumed_at": _iso(self.resumed_at),
            "reads_while_held": self.reads_while_held,
            "reads_with_post_sample": self.reads_with_post_sample,
            "regrade_seen_at": _iso(self.regrade_seen_at),
            "schedule_receipt": self.schedule_receipt,
            "resume_receipt": self.resume_receipt,
        }

    def note(self) -> str:
        """Why the regraded state was (not) readable, for the INCONCLUSIVE text."""

        if self.schedule_error:
            return f"the recount hold could not be scheduled ({self.schedule_error})"
        if not self.scheduled:
            return "no recount hold was scheduled"
        if self.timed_out:
            return "the recount hold waiter did not reach the post-gap read in time"
        if self.missed:
            return (
                f"the recount hold landed after read {self.reads_at_pause}, past its "
                f"target {self.target_reads}"
            )
        if not self.taken:
            return (
                "the recount hold never landed (waiter "
                f"{self.waiter_state or 'unknown'})"
            )
        return (
            f"the collector was held after the post-gap read and the state was read "
            f"{self.reads_while_held} time(s) meanwhile"
        )


@dataclass
class PauseFacts:
    """What the node's pause record and the runner's own steps say."""

    scheduled: bool = False
    target_reads: int | None = None
    paused_at: datetime | None = None
    reads_at_pause: int | None = None
    missed: bool = False
    timed_out: bool = False
    waiter_state: str | None = None
    stopped: bool | None = None
    resumed_at: datetime | None = None
    resume_receipt: dict[str, Any] | None = None
    restart_at: datetime | None = None
    restart_receipt: dict[str, Any] | None = None
    first_seen_paused: datetime | None = None
    #: Pre-gap finding ids seen active while the unit was paused, and whether
    #: one of them vanished before any post-gap reading was accepted.
    active_while_paused: set[str] = field(default_factory=set)
    cleared_in_gap: list[str] = field(default_factory=list)
    hold: HoldFacts = field(default_factory=HoldFacts)

    def evidence(self) -> dict[str, Any]:
        return {
            "scheduled": self.scheduled,
            "target_reads": self.target_reads,
            "paused_at": _iso(self.paused_at),
            "reads_at_pause": self.reads_at_pause,
            "missed": self.missed,
            "timed_out": self.timed_out,
            "waiter_state": self.waiter_state,
            "stopped": self.stopped,
            "resumed_at": _iso(self.resumed_at),
            "resume_receipt": self.resume_receipt,
            "restart_at": _iso(self.restart_at),
            "restart_receipt": self.restart_receipt,
            "cleared_in_gap": list(self.cleared_in_gap),
            "hold": self.hold.evidence(),
        }


class InterruptionJudge(BaseJudge):
    """009: a >60 s gap recounts from 1; restart stays fresh with one producer."""

    reads_extra = True

    def __init__(
        self,
        expectation: Expectation,
        *,
        steps: InterruptionSteps,
        interval_seconds: float,
        node_before: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(expectation, node_before=node_before)
        self.steps = steps
        self.interval = float(interval_seconds)
        self.pause = PauseFacts()
        self.step_errors: list[str] = []

    # -- intake ----------------------------------------------------------- #
    def observe(self, snapshot: Snapshot, progress: ReplayProgress) -> None:
        super().observe(snapshot, progress)
        if snapshot.extra is not None:
            self._absorb_pause(snapshot.extra, snapshot.at)
            self._absorb_hold(snapshot.extra)
        self._track_gap_health(snapshot.at)
        self._track_hold(snapshot)
        self._take_steps(progress, snapshot.at)

    def _absorb_pause(self, extra: Mapping[str, Any], at: datetime) -> None:
        record = dict(extra.get("pause") or {})
        pause = self.pause
        pause.scheduled = pause.scheduled or bool(record)
        if record.get("target_reads") is not None:
            pause.target_reads = int(record["target_reads"])
        paused = _time(record.get("paused_at"))
        if paused is not None and pause.paused_at is None:
            pause.paused_at = paused
            pause.first_seen_paused = at
        if record.get("reads_at_pause") is not None:
            pause.reads_at_pause = int(record["reads_at_pause"])
        pause.missed = pause.missed or bool(record.get("missed"))
        pause.timed_out = pause.timed_out or bool(record.get("timed_out_at"))
        pause.waiter_state = str((extra.get("waiter") or {}).get("ActiveState") or "")
        stopped = extra.get("stopped")
        pause.stopped = bool(stopped) if stopped is not None else pause.stopped
        resumed = _time(record.get("resumed_at"))
        if resumed is not None and pause.resumed_at is None:
            pause.resumed_at = resumed

    def _absorb_hold(self, extra: Mapping[str, Any]) -> None:
        """The recount hold's record (``pauses[<label>]`` of ``pause-status``)."""

        record = dict((extra.get("pauses") or {}).get(RECOUNT_HOLD_LABEL) or {})
        if not record:
            return
        hold = self.pause.hold
        hold.scheduled = True
        if record.get("target_reads") is not None:
            hold.target_reads = int(record["target_reads"])
        paused = _time(record.get("paused_at"))
        if paused is not None and hold.paused_at is None:
            hold.paused_at = paused
        if record.get("reads_at_pause") is not None:
            hold.reads_at_pause = int(record["reads_at_pause"])
        hold.missed = hold.missed or bool(record.get("missed"))
        hold.timed_out = hold.timed_out or bool(record.get("timed_out_at"))
        hold.waiter_state = str((record.get("waiter") or {}).get("ActiveState") or "")
        resumed = _time(record.get("resumed_at"))
        if resumed is not None and hold.resumed_at is None:
            # The node's stamp: the runner's own resume or the deadman's.
            hold.resumed_at = resumed

    def _track_gap_health(self, at: datetime) -> None:
        """The pre-gap finding must survive the gap until a reading says otherwise."""

        pause = self.pause
        if pause.paused_at is None or self.first_post_sample() is not None:
            return
        pre = {str(item.get("finding_id")) for item in self.pre_gap_findings()}
        if pause.resumed_at is None:
            pause.active_while_paused |= pre & self.active_ids
        for finding_id in sorted(pause.active_while_paused - self.active_ids):
            if finding_id not in pause.cleared_in_gap:
                pause.cleared_in_gap.append(finding_id)

    def _track_hold(self, snapshot: Snapshot) -> None:
        """What the hold bought: store reads taken while the state was held."""

        hold = self.pause.hold
        if hold.regrade_seen_at is None and self.post_gap_findings():
            hold.regrade_seen_at = snapshot.at
        if (
            snapshot.gpu_state is None
            or hold.paused_at is None
            or hold.resumed_at is not None
            # The poll's ``at`` precedes its reads: a poll that started after
            # the hold landed read the held state.
            or snapshot.at <= hold.paused_at
        ):
            return
        hold.reads_while_held += 1
        if self.first_post_sample() is not None:
            hold.reads_with_post_sample += 1

    def _take_steps(self, progress: ReplayProgress, now: datetime) -> None:
        pause = self.pause
        hold = pause.hold
        due = (
            pause.paused_at is not None
            and pause.resumed_at is None
            and now >= pause.paused_at + timedelta(seconds=self.x.pause_seconds)
        )
        if due:
            # The hold is scheduled while the collector is still stopped, so no
            # read can slip in between the schedule and the resume; a schedule
            # that fails is recorded and the resume goes ahead regardless.
            if (
                self.steps.hold is not None
                and not hold.scheduled
                and hold.schedule_error is None
            ):
                try:
                    hold.schedule_receipt = self.steps.hold()
                    hold.scheduled = True
                    target = (hold.schedule_receipt or {}).get("target_reads")
                    if target is not None:
                        hold.target_reads = int(target)
                except Exception as exc:  # noqa: BLE001 - recorded; the resume goes on
                    hold.schedule_error = (
                        f"pause-collector[{RECOUNT_HOLD_LABEL}]: "
                        f"{type(exc).__name__}: {exc}"
                    )
            try:
                pause.resume_receipt = self.steps.resume()
            except Exception as exc:  # noqa: BLE001 - recorded; the case fails
                self.step_errors.append(
                    f"resume-collector: {type(exc).__name__}: {exc}"
                )
            # The node's stamp of the SIGCONT when the receipt carries it: the
            # poll's ``now`` precedes the step by the poll's reads (live: 36 s).
            pause.resumed_at = _receipt_time(pause.resume_receipt, "resumed_at") or now
            self.mark("resumed", now)
        hold_due = (
            hold.paused_at is not None
            and hold.resumed_at is None
            and self.steps.resume_hold is not None
            and (
                bool(self.post_gap_findings())
                or now
                >= hold.paused_at + timedelta(seconds=self.x.recount_hold_seconds)
            )
        )
        if hold_due:
            try:
                hold.resume_receipt = self.steps.resume_hold()
            except Exception as exc:  # noqa: BLE001 - recorded; the case fails
                self.step_errors.append(
                    f"resume-collector[{RECOUNT_HOLD_LABEL}]: {type(exc).__name__}: {exc}"
                )
            hold.resumed_at = _receipt_time(hold.resume_receipt, "resumed_at") or now
            self.mark("hold_resumed", now)
        if (
            pause.resumed_at is not None
            and pause.restart_at is None
            and progress.consumed
            and self.cleared_at() is not None
            and not self.step_errors
        ):
            try:
                pause.restart_receipt = self.steps.restart()
            except Exception as exc:  # noqa: BLE001 - recorded; the case fails
                self.step_errors.append(
                    f"restart-collector: {type(exc).__name__}: {exc}"
                )
            pause.restart_at = now
            self.mark("restarted", now)

    # -- facts ------------------------------------------------------------ #
    def trigger_samples(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.field_samples()
            if self.since_arm(item.get("observed_at"))
            and any(same_value(item.get("value"), v) for v in self.x.trigger_values)
        ]

    def pre_gap_samples(self) -> list[dict[str, Any]]:
        paused = self.pause.paused_at
        if paused is None:
            return self.trigger_samples()
        return [
            item
            for item in self.trigger_samples()
            if (_time(item.get("observed_at")) or paused) <= paused
        ]

    def first_post_sample(self) -> dict[str, Any] | None:
        paused = self.pause.paused_at
        if paused is None:
            return None
        for item in self.trigger_samples():
            moment = _time(item.get("observed_at"))
            if moment is not None and moment > paused:
                return item
        return None

    def gap_seconds(self) -> float | None:
        pre = self.pre_gap_samples()
        post = self.first_post_sample()
        if not pre or post is None:
            return None
        last = _time(pre[-1].get("observed_at"))
        first = _time(post.get("observed_at"))
        if last is None or first is None:
            return None
        return (first - last).total_seconds()

    def inside_gap(self) -> list[dict[str, Any]]:
        """Readings of the field accepted strictly between the gap's ends."""

        pre = self.pre_gap_samples()
        post = self.first_post_sample()
        if not pre or post is None:
            return []
        last = _time(pre[-1].get("observed_at"))
        first = _time(post.get("observed_at"))
        if last is None or first is None:
            return []
        return [
            item
            for item in self.field_samples()
            if last < (_time(item.get("observed_at")) or last) < first
        ]

    def pre_gap_findings(self) -> list[dict[str, Any]]:
        paused = self.pause.paused_at
        return [
            item
            for item in self.gpu_findings({str(self.x.field)})
            if paused is None or (_time(item.get("observed_at")) or paused) <= paused
        ]

    def post_gap_findings(self) -> list[dict[str, Any]]:
        post = self.first_post_sample()
        if post is None:
            return []
        first = _time(post.get("observed_at"))
        return [
            item
            for item in self.gpu_findings({str(self.x.field)})
            if first is not None
            and (_time(item.get("observed_at")) or first) >= first
            and any(same_value(item.get("value"), v) for v in self.x.trigger_values)
        ]

    def cleared_at(self) -> datetime | None:
        """The accepted clearing reading, once no finding on the field is active."""

        active = {
            str(self.findings[key].get("canonical_name"))
            for key in self.active_ids
            if key in self.findings and self.findings[key].get("gpu_uuid") == self.x.gpu
        }
        return None if self.x.field in active else self.after_trigger()

    def sources(self) -> set[str]:
        return {
            str(row.get("source"))
            for row in self.latest.values()
            if row.get("source") and self.since_arm(row.get("observed_at"))
        }

    def fresh_after_restart(self) -> dict[str, Any] | None:
        restart = self.pause.restart_at
        if restart is None:
            return None
        for batch_id, batch in sorted(
            self.batches.items(), key=lambda item: str(item[1].get("observed_at"))
        ):
            observed = _time(batch.get("observed_at"))
            if (
                observed is not None
                and observed >= restart - timedelta(seconds=PROCESSING_MARGIN_SECONDS)
                and "initial-baseline" in (batch.get("edge_filter_reasons") or [])
                and self.x.gpu in batch["gpus"]
            ):
                return {**batch, "batch_id": batch_id}
        return None

    def batches_after_restart(self) -> list[datetime]:
        restart = self.pause.restart_at
        if restart is None:
            return []
        moments = [_time(batch.get("observed_at")) for batch in self.batches.values()]
        return sorted(moment for moment in moments if moment and moment >= restart)

    def restart_boot_ids(self) -> set[str]:
        batch = self.fresh_after_restart()
        if batch is None:
            return set()
        first = str(batch.get("observed_at"))
        return {
            str((item.get("labels") or {}).get("source_boot_id"))
            for item in self.samples.values()
            if str(item.get("observed_at")) >= first
            and (item.get("labels") or {}).get("source_boot_id")
        }

    def ungraded_while_held(self) -> bool:
        """The post-gap sample's batch was readable for the hold's polls and no
        finding on it appeared: the state the hold froze does not grade it."""

        hold = self.pause.hold
        return (
            hold.taken
            and self.first_post_sample() is not None
            and not self.post_gap_findings()
            and hold.reads_with_post_sample >= RECOUNT_HOLD_POLLS
        )

    def wait_until(self, progress: ReplayProgress, now: datetime) -> datetime | None:
        pause = self.pause
        hold = pause.hold
        moments: list[datetime] = []
        if pause.paused_at is not None and pause.resumed_at is None:
            moments.append(
                pause.paused_at
                + timedelta(seconds=self.x.pause_seconds + self.interval)
            )
        if hold.paused_at is not None and hold.resumed_at is None:
            moments.append(
                hold.paused_at
                + timedelta(seconds=self.x.recount_hold_seconds + self.interval)
            )
        if pause.restart_at is not None and self.fresh_after_restart() is None:
            moments.append(
                pause.restart_at
                + timedelta(seconds=RESTART_DELIVERY_SECONDS + 2 * self.interval)
            )
        if progress.consumed_at is not None and self.cleared_at() is None:
            moments.append(
                progress.consumed_at + timedelta(seconds=2 * self.x.hold_seconds)
            )
        return max(moments) if moments else None

    # -- decision --------------------------------------------------------- #
    def errors(self, now: datetime) -> dict[str, list[str]]:
        """Definite failures: a decision the rest of the window cannot change."""

        label = self.label
        stages: dict[str, list[str]] = {
            "reaction": self.forbidden(),
            "recount": [],
            "gap": [],
            "producer": [],
            "restart": [],
            "clear": [],
            "steps": list(self.step_errors),
        }
        if self.pause.missed or self.pause.timed_out:
            # A late or absent pause is the harness's failure: whatever the
            # product did with three consecutive severe samples is not graded.
            return stages
        for workflow in self.workflows.values():
            isolating = sorted(
                set(operations(workflow)) & verdicts.ISOLATING_OPERATIONS
            )
            if isolating:
                stages["reaction"].append(
                    f"{label}: workflow {workflow.get('request_id')} plans "
                    f"{isolating} on an interrupted, unconfirmed signal"
                )
            if "RESTORE_SCHEDULING" in operations(workflow):
                stages["reaction"].append(
                    f"{label}: workflow {workflow.get('request_id')} restores "
                    "scheduling: an unauthorized recovery"
                )
        if self.newly_isolated():
            stages["reaction"].append(
                f"{label}: the node was isolated although the gap must restart "
                "the confirmation"
            )
        for item in self.gpu_findings({str(self.x.field)}):
            if item.get("severity") == "CRITICAL" and any(
                same_value(item.get("value"), v) for v in self.x.trigger_values
            ):
                stages["recount"].append(
                    f"{label}: finding {item.get('finding_id')} confirmed CRITICAL "
                    f"(confirmation_count {item.get('confirmation_count')}) across "
                    "the collection gap"
                )
        post = self.post_gap_findings()
        gap = self.gap_seconds()
        if post and gap is not None and gap > CONFIRMATION_MAX_GAP_SECONDS:
            first = post[0]
            count = int(first.get("confirmation_count") or 0)
            if count != 1:
                stages["recount"].append(
                    f"{label}: the first severe sample after the {gap:.0f}s gap was "
                    f"counted confirmation {count}, not 1"
                )
            self.mark("recounted", now)
        if self.ungraded_while_held():
            sample = self.first_post_sample() or {}
            hold = self.pause.hold
            stages["recount"].append(
                f"{label}: the severe {self.x.field} sample accepted at "
                f"{sample.get('observed_at')} after the {gap or 0:.0f}s gap was not "
                "graded into a finding while the collector was held after it "
                f"({hold.reads_with_post_sample} store reads showed the batch, the "
                "field's finding state never named the sample)"
            )
        inside = self.inside_gap()
        if inside:
            stages["gap"].append(
                f"{label}: {len(inside)} {self.x.field} reading(s) were accepted "
                "inside the collection gap "
                f"({sorted(str(item.get('observed_at')) for item in inside)[:3]})"
            )
        if self.pause.cleared_in_gap:
            stages["gap"].append(
                f"{label}: finding(s) {self.pause.cleared_in_gap} were cleared while "
                "no reading was accepted: the gap was treated as healthy"
            )
        sources = self.sources()
        if len(sources) > 1:
            stages["producer"].append(
                f"{label}: accepted readings carry {sorted(sources)} sources; one "
                "producer is required"
            )
        moments = self.batches_after_restart()
        if len(moments) > 1:
            elapsed = (moments[-1] - moments[0]).total_seconds()
            allowed = int(elapsed // self.interval) + 1 + PRODUCER_RATE_SLACK
            if len(moments) > allowed:
                stages["producer"].append(
                    f"{label}: {len(moments)} batches in {elapsed:.0f}s after the "
                    f"restart exceed one producer's {allowed} at {self.interval:.0f}s"
                )
        boots = self.restart_boot_ids()
        if boots and self.x.boot_id and boots != {self.x.boot_id}:
            stages["restart"].append(
                f"{label}: the restarted collector's readings carry source_boot_id "
                f"{sorted(boots)}, the node booted {self.x.boot_id}"
            )
        return stages

    def evidence(self, progress: ReplayProgress) -> dict[str, Any]:
        pre = self.pre_gap_samples()
        post = self.first_post_sample()
        cleared = self.cleared_at()
        return {
            "pause": self.pause.evidence(),
            "recount_hold_seconds": self.x.recount_hold_seconds,
            "pre_gap_severe_samples": len(pre),
            "pre_gap_findings": [
                {
                    "finding_id": item.get("finding_id"),
                    "severity": item.get("severity"),
                    "confirmation_count": item.get("confirmation_count"),
                    "observed_at": item.get("observed_at"),
                }
                for item in self.pre_gap_findings()
            ][-8:],
            "first_post_gap_sample_at": post.get("observed_at") if post else None,
            "gap_seconds": self.gap_seconds(),
            "post_gap_findings": [
                {
                    "finding_id": item.get("finding_id"),
                    "severity": item.get("severity"),
                    "confirmation_count": item.get("confirmation_count"),
                    "observed_at": item.get("observed_at"),
                }
                for item in self.post_gap_findings()
            ][-8:],
            "cleared_at": _iso(cleared),
            "sources": sorted(self.sources()),
            "fresh_batch_after_restart": self.fresh_after_restart(),
            "batches_after_restart": len(self.batches_after_restart()),
            "task": {"owned_task": None, "measured": False, "reason": NO_TASK_009},
        }

    def complete(self, progress: ReplayProgress) -> bool:
        gap = self.gap_seconds()
        post = self.post_gap_findings()
        return bool(
            progress.consumed
            and self.pause.resumed_at is not None
            and gap is not None
            and gap > CONFIRMATION_MAX_GAP_SECONDS
            and post
            and int(post[0].get("confirmation_count") or 0) == 1
            and self.cleared_at() is not None
            and self.pause.restart_at is not None
            and self.fresh_after_restart() is not None
            and len(self.sources()) == 1
        )

    def decide(
        self, progress: ReplayProgress, now: datetime, *, final: bool
    ) -> Judgement | None:
        stages = self.errors(now)
        evidence = self.evidence(progress)
        pause = self.pause
        harness = pause.missed or pause.timed_out
        if any(stages.values()) or (self.complete(progress) and not harness):
            return self.judgement(progress, stages, [], evidence)
        # The waiter died before its read while the plan still runs: the
        # runner's cleanup restores the real endpoint before the third severe
        # read can be served (decided now, not at the deadline).
        waiter_dead = (
            pause.scheduled
            and pause.paused_at is None
            and pause.waiter_state not in {None, "", "active", "activating"}
        )
        if harness or waiter_dead:
            return self.judgement(progress, stages, self.unfinished(progress), evidence)
        if not final:
            return None
        return self.judgement(
            progress,
            stages,
            progress.inconclusive() or self.unfinished(progress),
            evidence,
        )

    def unfinished(self, progress: ReplayProgress) -> list[str]:
        """At the deadline (or on a harness failure): why there is no verdict."""

        label, pause = self.label, self.pause
        if pause.timed_out:
            return [f"{label}: the pause waiter did not reach its read in time"]
        if pause.missed:
            return [
                f"{label}: the pause landed after read {pause.reads_at_pause}, past "
                f"its target {pause.target_reads}; the product's reaction to three "
                "consecutive severe samples is not this case's evidence"
            ]
        if pause.paused_at is None:
            return [
                f"{label}: the collector was never paused (waiter "
                f"{pause.waiter_state or 'unknown'})"
            ]
        if pause.resumed_at is None:
            return [f"{label}: the collector was not resumed before the deadline"]
        gap = self.gap_seconds()
        if gap is None:
            return [
                f"{label}: no accepted severe {self.x.field} sample after the resume "
                "shows the gap"
            ]
        if gap <= CONFIRMATION_MAX_GAP_SECONDS:
            return [
                f"{label}: the accepted samples are only {gap:.0f}s apart, not past "
                f"the {CONFIRMATION_MAX_GAP_SECONDS}s confirmation gap"
            ]
        if not self.post_gap_findings():
            return [
                f"{label}: the post-gap severe sample was accepted but no finding "
                f"graded it in any store read; {pause.hold.note()} (the product "
                "keeps a confirmation-count change only in the current finding "
                "state, which the clearing read replaces)"
            ]
        if pause.hold.taken and pause.hold.resumed_at is None:
            return [f"{label}: the recount hold was not resumed before the deadline"]
        if self.cleared_at() is None:
            return [
                f"{label}: the clearing reading was not accepted before the deadline"
            ]
        if pause.restart_at is None:
            return [f"{label}: the collector restart was not taken before the deadline"]
        if self.fresh_after_restart() is None:
            return [
                f"{label}: the restarted collector's initial-baseline batch did not "
                "reach the control plane before the deadline"
            ]
        return [f"{label}: the case did not complete before the deadline"]


NO_TASK_009 = (
    "no owned test task: the preflight admits only a node without business "
    "workloads; an interrupted, unconfirmed signal must not stop anything, and "
    "nothing was there to stop"
)

__all__ = [
    "HoldFacts",
    "InterruptionJudge",
    "InterruptionSteps",
    "PauseFacts",
    "RECOUNT_HOLD_LABEL",
    "RESTART_DELIVERY_SECONDS",
]
