from __future__ import annotations

from gpu_fault.training_progress_state import last_valid_step

import math
import os
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any

from pydantic import Field

from gpu_fault.host_health import (
    NodeHealthCategory,
    NodeHealthFinding,
)
from gpu_fault.models import (
    RecoveryAction,
    Severity,
    StrictModel,
    WorkloadState,
)
from gpu_fault.training_phase import observe_phase, paused, phase
from gpu_fault.orchestration.families.identity import derived_record_id
from gpu_fault.store.shared.health_signals import (
    training_signal_key as _training_signal_key,
)
from gpu_fault.training_models import (
    TrainingProgressHeartbeat as TrainingProgressHeartbeat,
)
from gpu_fault.training_models import (
    TrainingProgressState as TrainingProgressState,
)
from gpu_fault.watcher import WorkloadPhase


class TrainingHealthResult(StrictModel):
    heartbeat_id: str | None = None
    accepted: bool = True
    findings: list[NodeHealthFinding] = Field(default_factory=list)


class TrainingHealthPolicy(StrictModel):
    heartbeat_timeout_seconds: int = Field(default=120, ge=15)
    startup_grace_seconds: int = Field(default=300, ge=15)
    max_step_lag: int = Field(default=20, ge=1)
    min_throughput_ratio: float = Field(default=0.5, gt=0, lt=1)

    @classmethod
    def from_environment(cls) -> TrainingHealthPolicy:
        return cls(
            heartbeat_timeout_seconds=int(
                os.getenv(
                    "GPU_FAULT_TRAINING_HEARTBEAT_TIMEOUT_SECONDS",
                    "120",
                )
            ),
            startup_grace_seconds=int(
                os.getenv(
                    "GPU_FAULT_TRAINING_STARTUP_GRACE_SECONDS",
                    "300",
                )
            ),
            max_step_lag=int(os.getenv("GPU_FAULT_TRAINING_MAX_STEP_LAG", "20")),
            min_throughput_ratio=float(
                os.getenv(
                    "GPU_FAULT_TRAINING_MIN_THROUGHPUT_RATIO",
                    "0.5",
                )
            ),
        )


TRAINING_EVIDENCE_SCHEME = "training-progress://"


def training_health_signal_key(finding: NodeHealthFinding) -> str | None:
    """The health-signal key ``TrainingHealthService`` claimed for a finding.

    Rebuilt from the fields the finding carries -- ``evidence_ref`` names the
    attempt and rank, ``metric_name`` the kind -- so the deliverer can latch
    it after the incident commit (P0-38B). None for a finding this service did
    not produce.
    """

    evidence_ref = finding.evidence_ref or ""
    metric_name = str(finding.metric_name)
    if not evidence_ref.startswith(TRAINING_EVIDENCE_SCHEME):
        return None
    if not metric_name.startswith("training_"):
        return None
    attempt_id, separator, rank = evidence_ref[
        len(TRAINING_EVIDENCE_SCHEME) :
    ].rpartition("/rank-")
    if not separator or not attempt_id or not rank.isdigit():
        return None
    return _training_signal_key(
        finding.cluster_id,
        attempt_id,
        int(rank),
        metric_name[len("training_") :],
    )


class TrainingHealthService:
    def __init__(self, store: Any, policy: TrainingHealthPolicy | None = None) -> None:
        self.store = store
        self.policy = policy or TrainingHealthPolicy()

    def mark_notified(self, findings: list[NodeHealthFinding]) -> None:
        """Latch the signals behind delivered findings (P0-38B).

        The claim in ``_claim`` only decides to emit; the caller that persisted
        the findings' incidents calls this afterwards. A finding whose write
        failed is never marked and is emitted again on the next evaluation.
        """

        for finding in findings:
            signal_key = training_health_signal_key(finding)
            if signal_key is not None:
                self.store.mark_health_signal_notified(
                    signal_key, notified_at=finding.observed_at
                )

    def ingest(self, heartbeat: TrainingProgressHeartbeat) -> TrainingHealthResult:
        state, container = self._allocation(
            heartbeat.cluster_id,
            heartbeat.attempt_id,
            heartbeat.rank,
        )
        if state is None or container is None or container.terminated:
            return TrainingHealthResult(
                heartbeat_id=heartbeat.heartbeat_id, accepted=False
            )
        if (
            not -30
            <= (heartbeat.observed_at - state.observation.observed_at).total_seconds()
            <= 120
        ):
            return TrainingHealthResult(
                heartbeat_id=heartbeat.heartbeat_id, accepted=False
            )
        if container is not None:
            if (
                heartbeat.node_id is not None
                and container.node_id is not None
                and heartbeat.node_id != container.node_id
            ):
                raise ValueError("training heartbeat node does not match allocation")
            if (
                (
                    heartbeat.pod_uid is not None
                    and heartbeat.pod_uid != container.pod_uid
                )
                or (
                    heartbeat.container_name is not None
                    and heartbeat.container_name != container.container_name
                )
                or (
                    heartbeat.gpu_uuids
                    and not set(heartbeat.gpu_uuids) <= set(container.gpu_uuids)
                )
            ):
                raise ValueError(
                    "training heartbeat identity does not match allocation"
                )
            heartbeat = heartbeat.model_copy(
                update={
                    "node_id": heartbeat.node_id or container.node_id,
                    "gpu_uuids": (heartbeat.gpu_uuids or container.gpu_uuids),
                    "pod_uid": heartbeat.pod_uid or container.pod_uid,
                    "container_name": (
                        heartbeat.container_name or container.container_name
                    ),
                }
            )
        previous = self.store.observe_training_progress(heartbeat)
        if previous is False:
            return TrainingHealthResult(
                heartbeat_id=heartbeat.heartbeat_id,
                accepted=False,
            )

        findings = []
        old_step = last_valid_step(previous)
        progressed = heartbeat.step is not None and (
            old_step is None or heartbeat.step > old_step
        )
        observe_phase(self.store, heartbeat, progressed=progressed)
        if progressed:
            self._claim(heartbeat, "hang", False, heartbeat.observed_at, state)
        self._claim(heartbeat, "reporter-silent", False, heartbeat.observed_at, state)
        if heartbeat.step is not None:
            self._claim(
                heartbeat, "progress-unavailable", False, heartbeat.observed_at, state
            )
        if heartbeat.numerical_error or (
            heartbeat.loss is not None and not math.isfinite(heartbeat.loss)
        ):
            finding = self._claim(
                heartbeat,
                "nonfinite-loss",
                True,
                heartbeat.observed_at,
                state,
                reason="training loss is NaN or infinite",
                severity=Severity.CRITICAL,
            )
            if finding is not None:
                findings.append(finding)
        else:
            self._claim(
                heartbeat,
                "nonfinite-loss",
                False,
                heartbeat.observed_at,
                state,
            )
        if (
            previous is not None
            and previous is not False
            and heartbeat.step is not None
            and previous.step is not None
            and heartbeat.step < previous.step
            and not (
                phase(heartbeat) in {"CHECKPOINT_LOAD", "RESTORING"}
                and heartbeat.checkpoint_ref
                and paused(self.store, heartbeat, heartbeat.observed_at)
            )
        ):
            finding = self._claim(
                heartbeat,
                "step-regression",
                True,
                heartbeat.observed_at,
                state,
                reason="training step regressed",
            )
            if finding is not None:
                findings.append(finding)
        else:
            self._claim(
                heartbeat,
                "step-regression",
                False,
                heartbeat.observed_at,
                state,
            )
        return TrainingHealthResult(
            heartbeat_id=heartbeat.heartbeat_id,
            findings=findings,
        )

    def scan(
        self,
        cluster_id: str,
        *,
        now: datetime | None = None,
    ) -> TrainingHealthResult:
        checked_at = now or datetime.now(timezone.utc)
        findings: list[NodeHealthFinding] = []
        progress_states = {
            (item.heartbeat.attempt_id, item.heartbeat.rank): item
            for item in self.store.list_training_progress_states(cluster_id)
        }
        for state in self.store.list_attempt_observation_states(cluster_id):
            observation = state.observation
            if observation.workload_phase is not WorkloadPhase.RUNNING:
                continue
            active = [
                item
                for item in observation.containers
                if item.critical and not item.terminated
            ]
            fresh = []
            for container in active:
                progress_state = progress_states.get(
                    (observation.attempt_id, container.rank)
                )
                if progress_state is not None and (
                    progress_state.heartbeat.node_id != container.node_id
                    or progress_state.heartbeat.pod_uid != container.pod_uid
                    or progress_state.heartbeat.container_name
                    != container.container_name
                ):
                    progress_state = None
                heartbeat = (
                    progress_state.heartbeat if progress_state is not None else None
                )
                if heartbeat is not None and heartbeat.observed_at > checked_at:
                    continue
                anchor = (
                    heartbeat.observed_at
                    if heartbeat is not None
                    else state.first_observed_at
                )
                timeout = (
                    self.policy.heartbeat_timeout_seconds
                    if heartbeat is not None
                    else self.policy.startup_grace_seconds
                )
                stale = checked_at - anchor > timedelta(seconds=timeout)
                missing_progress = bool(
                    heartbeat is not None
                    and heartbeat.step is None
                    and progress_state is not None
                    and (checked_at - progress_state.last_progress_at).total_seconds()
                    > (
                        self.policy.heartbeat_timeout_seconds
                        if last_valid_step(heartbeat) is not None
                        else self.policy.startup_grace_seconds
                    )
                    and not paused(self.store, heartbeat, checked_at)
                )
                stalled = (
                    progress_state is not None
                    and heartbeat is not None
                    and heartbeat.step is not None
                    and checked_at - progress_state.last_progress_at
                    > timedelta(seconds=(self.policy.heartbeat_timeout_seconds))
                )
                if (
                    heartbeat is not None
                    and not stale
                    and paused(self.store, heartbeat, checked_at)
                ):
                    stalled = False
                probe = heartbeat or TrainingProgressHeartbeat(
                    cluster_id=cluster_id,
                    attempt_id=observation.attempt_id,
                    rank=container.rank,
                    node_id=container.node_id,
                    pod_uid=container.pod_uid,
                    container_name=container.container_name,
                    gpu_uuids=container.gpu_uuids,
                    observed_at=checked_at,
                )
                finding = self._claim(
                    probe,
                    "reporter-silent"
                    if stale
                    else "progress-unavailable"
                    if heartbeat is not None and heartbeat.step is None
                    else "hang",
                    stale or stalled or missing_progress,
                    checked_at,
                    state,
                    reason=(
                        "training step stopped advancing"
                        if stalled and not stale
                        else "training progress is unavailable; liveness is not step progress"
                        if heartbeat is not None
                        and heartbeat.step is None
                        and not stale
                        else "training rank heartbeat timed out"
                    ),
                    severity=Severity.WARNING
                    if missing_progress and not stale
                    else Severity.CRITICAL,
                )
                if finding is not None:
                    findings.append(finding)
                if not stale and heartbeat is not None:
                    fresh.append(heartbeat)
            findings.extend(self._stragglers(state, fresh, checked_at))
        return TrainingHealthResult(findings=findings)

    def scan_all(self, *, now: datetime | None = None) -> TrainingHealthResult:
        cluster_ids = {
            item.observation.cluster_id
            for item in self.store.list_attempt_observation_states()
        }
        findings = [
            finding
            for cluster_id in cluster_ids
            for finding in self.scan(cluster_id, now=now).findings
        ]
        return TrainingHealthResult(findings=findings)

    def _stragglers(
        self,
        state: Any,
        heartbeats: list[TrainingProgressHeartbeat],
        observed_at: datetime,
    ) -> list[NodeHealthFinding]:
        groups: dict[str, list[TrainingProgressHeartbeat]] = {}
        for heartbeat in heartbeats:
            group = heartbeat.labels.get("progress_group")
            if group and phase(heartbeat) == "TRAINING":
                groups.setdefault(group, []).append(heartbeat)
        return [
            finding
            for grouped in groups.values()
            for finding in self._group_stragglers(state, grouped, observed_at)
        ]

    def _group_stragglers(
        self,
        state: Any,
        heartbeats: list[TrainingProgressHeartbeat],
        observed_at: datetime,
    ) -> list[NodeHealthFinding]:
        if len(heartbeats) < 2:
            return []
        steps = [item.step for item in heartbeats if item.step is not None]
        throughputs = [
            item.samples_per_second
            for item in heartbeats
            if item.samples_per_second is not None
        ]
        median_step = median(steps) if len(steps) >= 2 else None
        median_throughput = median(throughputs) if len(throughputs) >= 2 else None
        findings = []
        for heartbeat in heartbeats:
            step_lag = (
                median_step - heartbeat.step
                if median_step is not None and heartbeat.step is not None
                else 0
            )
            throughput_low = (
                median_throughput is not None
                and median_throughput > 0
                and heartbeat.samples_per_second is not None
                and heartbeat.samples_per_second
                < median_throughput * self.policy.min_throughput_ratio
            )
            active = step_lag >= self.policy.max_step_lag or throughput_low
            finding = self._claim(
                heartbeat,
                "straggler",
                active,
                observed_at,
                state,
                reason="training rank is lagging peer progress",
            )
            if finding is not None:
                findings.append(finding)
        return findings

    def _allocation(
        self, cluster_id: str, attempt_id: str, rank: int
    ) -> tuple[Any, Any]:
        for state in self.store.list_attempt_observation_states(cluster_id):
            if state.observation.attempt_id != attempt_id:
                continue
            if state.observation.workload_phase not in {
                WorkloadPhase.RUNNING,
                WorkloadPhase.PENDING,
            }:
                continue
            container = next(
                (item for item in state.observation.containers if item.rank == rank),
                None,
            )
            return state, container
        return None, None

    def _claim(
        self,
        heartbeat: TrainingProgressHeartbeat,
        kind: str,
        active: bool,
        observed_at: datetime,
        state: Any,
        *,
        reason: str | None = None,
        severity: Severity = Severity.WARNING,
    ) -> NodeHealthFinding | None:
        signal_key = _training_signal_key(
            heartbeat.cluster_id, heartbeat.attempt_id, heartbeat.rank, kind
        )
        # Decides to emit, does not latch ``notified``: ``mark_notified`` does,
        # once the caller has persisted the finding (P0-38B).
        if not self.store.claim_health_signal_transition(
            signal_key, active, observed_at
        ):
            return None
        observation = state.observation if state is not None else None
        # Keep notification latch keys stable, but bind new event IDs to the tenant.
        event_id = derived_record_id(
            f"training-{kind}",
            "training-health",
            heartbeat.cluster_id,
            heartbeat.attempt_id,
            str(heartbeat.rank),
            str(int(observed_at.timestamp())),
        )
        return NodeHealthFinding(
            finding_id=f"finding-{event_id}",
            event_id=event_id,
            cluster_id=heartbeat.cluster_id,
            node_id=heartbeat.node_id or "UNKNOWN",
            observed_at=observed_at,
            category=NodeHealthCategory.TRAINING,
            severity=severity,
            reason=reason or f"training {kind} detected",
            recommended_action=(
                RecoveryAction.STOP_WORKLOAD
                if kind == "nonfinite-loss"
                else RecoveryAction.RUN_DIAGNOSTICS
            ),
            metric_name=f"training_{kind}",
            value=1,
            device=f"rank-{heartbeat.rank}",
            evidence_ref=(
                f"training-progress://{heartbeat.attempt_id}/rank-{heartbeat.rank}"
            ),
            runtime_profile_version=(
                observation.runtime_profile_version if observation is not None else None
            ),
            affected_workload_ids=(
                observation.workload_ids if observation is not None else []
            ),
            job_id=observation.job_id if observation is not None else None,
            attempt_id=heartbeat.attempt_id,
            gpu_uuids=heartbeat.gpu_uuids,
            workload_state=WorkloadState.ACTIVE,
            policy_source="SITE_TRAINING_HEALTH",
            diagnostic_parameters={
                "diagnostic_reason": (
                    "TRAINING_PROGRESS_STALL"
                    if kind == "hang"
                    else "TRAINING_REPORTER_SILENT"
                    if kind == "reporter-silent"
                    else "TRAINING_APPLICATION_ERROR"
                ),
                "capture_process_state": kind == "straggler",
                "checkpoint_ref": heartbeat.checkpoint_ref,
                "checkpoint_disposition": "REQUIRES_APPLICATION_VALIDATION",
                "training_diagnostic": {
                    "attempt_id": heartbeat.attempt_id,
                    "rank": heartbeat.rank,
                    "pod_uid": heartbeat.pod_uid,
                    "step": heartbeat.step,
                    "kind": kind,
                    "phase": phase(heartbeat),
                },
            },
        )
