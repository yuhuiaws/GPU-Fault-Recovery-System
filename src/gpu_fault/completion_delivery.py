"""Per-attempt delivery: observation -> terminal / containment POSTs.

One attempt's reconcile (``_reconcile_attempt``), the failure-containment
delivery and the emergency stop that arms when the control plane has not
accepted the event. Split out of ``completion_controller`` as a pure move
(F6b); the observe half of ``_reconcile_attempt`` stays fused to the delivery
half because splitting the function is not a move.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from typing import Any, Callable

from gpu_fault.collectors import EventSink
from gpu_fault.completion_attempt_state import (
    AttemptSpec,
    cache_terminal_attempt_observation,
    log_reconcile_failure,
    publish_attempt_observation,
)
from gpu_fault.completion_observation import (
    RECOVERED_VERDICT_REASON,
    is_unknown_profile_rejection,
    reconcile_attempt_observation,
)
from gpu_fault.completion_outbox import completion_delivery_deferred
from gpu_fault.completion_pod_parsing import (
    RUNTIME_PROFILE_ANNOTATION,
    CompletionControllerError,
)
from gpu_fault.watcher import (
    AttemptObservation,
    CompletionWatcher,
    FailureDetectedEvent,
    failure_containment_ids,
)

# Borrowed on purpose: every line below used to be logged by
# ``gpu_fault.completion_controller`` and the log format prints the logger
# name, so the split must not rename what operators grep for.
LOGGER = logging.getLogger("gpu_fault.completion_controller")


class CompletionDeliveryMixin:
    """Delivery half of ``KubernetesCompletionController``."""

    # Attributes supplied by the composed concrete implementation.
    _attempt_specs: dict[str, AttemptSpec]
    _emergency_incident_ids: dict[str, str]
    _failure_delivery_started: dict[str, datetime]
    _failure_events: dict[str, FailureDetectedEvent]
    _failure_sent: set[str]
    _last_observations: dict[str, AttemptObservation]
    _synthesized_verdicts: set[str]
    _terminal_sent: set[str]
    _workloads_stopped: set[str]
    emergency_fallback_seconds: float | None
    now: Callable[[], datetime]
    publish_observations: bool
    reconcile_failures_total: int
    sink: EventSink
    watcher: CompletionWatcher
    workload_stopper: Any | None

    def _reconcile_attempt(
        self,
        attempt_id: str,
        attempt_pods: list[dict[str, Any]],
        observed_at: datetime,
        results: list[dict[str, Any]],
    ) -> None:
        try:
            observation = reconcile_attempt_observation(
                self, attempt_id, attempt_pods, observed_at
            )
            if observation is None:
                return
            result = self.watcher.observe(observation)
            if result.terminal_event is not None:
                # Unknown live ranks cannot survive a terminal observation.
                observation = cache_terminal_attempt_observation(
                    self, attempt_id, result.terminal_event, observation
                )
            self._last_observations[attempt_id] = observation
        except (CompletionControllerError, ValueError) as exc:
            self.reconcile_failures_total += 1
            log_reconcile_failure(self, attempt_id, exc)
            return
        if self.publish_observations:
            publish_attempt_observation(self, observation, attempt_id)
        synthesized = attempt_id in self._synthesized_verdicts
        if result.failure_detected is not None and not synthesized:
            failure_event = result.failure_detected
            if self.workload_stopper is not None and hasattr(
                self.workload_stopper, "capture_logs"
            ):
                incident_id, _ = failure_containment_ids(failure_event.event_key)
                try:
                    snapshots = self.workload_stopper.capture_logs(
                        attempt_id,
                        incident_id,
                        pods=attempt_pods,
                    )
                except Exception:
                    LOGGER.exception(
                        "cannot capture failure logs for attempt %s",
                        attempt_id,
                    )
                else:
                    if snapshots:
                        failure_event = failure_event.model_copy(
                            update={"workload_log_snapshots": snapshots}
                        )
            self._failure_events[attempt_id] = failure_event
        if attempt_id in self._failure_events:
            failure_event = self._failure_events[attempt_id]
            passive_incident_id, _ = failure_containment_ids(failure_event.event_key)
            spec = self._attempt_specs.get(attempt_id)
            initiator = (
                spec.termination_initiator_incident_id if spec is not None else None
            )
            if initiator is None or initiator == passive_incident_id:
                self._deliver_failure_containment(attempt_id)
        if result.failure_detected is not None and synthesized:
            # No Pod of this attempt was ever seen by this process, so the
            # event carries no rank, node or exit code -- and must not be
            # published: ``handle_failure_detected`` would open an incident and
            # STOP_WORKLOADS the workload IDs, which are stable across
            # attempts, suspending whatever runs under them now (C1).
            LOGGER.warning(
                "restored attempt %s ended %s: %s; reporting the terminal "
                "without containment",
                attempt_id,
                result.terminal_event.terminal_status.value
                if result.terminal_event is not None
                else "unresolved",
                RECOVERED_VERDICT_REASON,
            )
        elif result.failure_detected is not None:
            LOGGER.warning(
                "training failure detected: attempt=%s rank=%s node=%s exit_code=%s",
                attempt_id,
                result.failure_detected.first_failed_rank,
                result.failure_detected.node_id,
                result.failure_detected.exit_code,
            )
        if (
            result.terminal_event is not None
            and result.terminal_event.event_key not in self._terminal_sent
        ):
            try:
                response = self.sink.post(
                    "/v1/attempts/terminal",
                    result.terminal_event.model_dump(mode="json"),
                )
            except Exception as exc:
                # A 404 here means the control plane has no such
                # runtime profile. Retrying is still correct (an
                # operator can register it and the next poll
                # succeeds), but the generic "will retry" message
                # buries the one thing that has to be fixed, so the
                # loop just reprints an opaque traceback every poll
                # while the attempt never reaches a decision.
                if is_unknown_profile_rejection(exc):
                    LOGGER.error(
                        "control plane does not know runtime "
                        "profile %r declared by attempt %s "
                        "(annotation %s); the terminal event "
                        "cannot be accepted until that profile is "
                        "registered via "
                        "POST /v1/runtime-profiles. Retrying, but "
                        "this will not clear on its own: %s",
                        self._runtime_profile_version(attempt_id),
                        attempt_id,
                        RUNTIME_PROFILE_ANNOTATION,
                        exc,
                    )
                    return
                LOGGER.exception(
                    "cannot submit training terminal for "
                    "attempt %s; will retry without blocking "
                    "other attempts",
                    attempt_id,
                )
                return
            if completion_delivery_deferred(response):
                # An earlier pass buffered this terminal and never delivered
                # it; the outbox replay owns the retry (F8), so posting it live
                # again this pass would only double the load. It is *not*
                # delivered, so it must not be recorded as sent: a record the
                # control plane rejects is quarantined and never replayed by
                # the loop again, and the live path is what has to pick it up
                # from there. The outbox suppresses the duplicate live POST
                # for as long as the record really is replay's.
                LOGGER.warning(
                    "training terminal for attempt %s is buffered in the "
                    "outbox; its delivery is owned by the replay",
                    attempt_id,
                )
                return
            self._terminal_sent.add(result.terminal_event.event_key)
            LOGGER.info(
                "training terminal submitted: attempt=%s status=%s event_key=%s",
                attempt_id,
                result.terminal_event.terminal_status.value,
                result.terminal_event.event_key,
            )
            results.append(response)

    def _runtime_profile_version(self, attempt_id: str) -> str | None:
        spec = self._attempt_specs.get(attempt_id)
        return spec.runtime_profile_version if spec is not None else None

    def _deliver_failure_containment(self, attempt_id: str) -> None:
        if attempt_id in self._failure_sent:
            return
        event = self._failure_events[attempt_id]
        try:
            response = self.sink.post(
                "/v1/attempts/failure-detected",
                event.model_dump(mode="json"),
            )
        except Exception as exc:
            started = self._failure_delivery_started.setdefault(attempt_id, self.now())
            if is_unknown_profile_rejection(exc):
                # Same misconfiguration as the terminal path, but here
                # it also drives the emergency fallback below, so the
                # operator has to be able to tell "profile not
                # registered" apart from "control plane unreachable".
                LOGGER.error(
                    "control plane does not know runtime profile %r "
                    "declared by attempt %s (annotation %s); failure "
                    "containment cannot start until that profile is "
                    "registered via POST /v1/runtime-profiles. The "
                    "emergency workload stop below is the only "
                    "remaining protection: %s",
                    self._runtime_profile_version(attempt_id),
                    attempt_id,
                    RUNTIME_PROFILE_ANNOTATION,
                    exc,
                )
            else:
                LOGGER.exception(
                    "cannot submit failure containment for attempt %s; "
                    "control-plane workflow will be retried",
                    attempt_id,
                )
            self._emergency_stop_if_overdue(attempt_id, event, started)
            return
        if completion_delivery_deferred(response):
            # The outbox still holds an undelivered copy of this event, so it
            # owns the retry (F8) and we must not post it a second time this
            # pass. It is *not* delivered though: the control plane has not
            # accepted anything, so the containment clock keeps running and
            # the emergency stop stays armed exactly as if the POST failed.
            started = self._failure_delivery_started.setdefault(attempt_id, self.now())
            LOGGER.warning(
                "failure containment for attempt %s is buffered in the outbox; "
                "its delivery is owned by the replay and the emergency "
                "fallback stays armed",
                attempt_id,
            )
            self._emergency_stop_if_overdue(attempt_id, event, started)
            return
        self._failure_sent.add(attempt_id)
        self._failure_delivery_started.pop(attempt_id, None)
        LOGGER.warning(
            "failure containment submitted to control plane: attempt=%s event=%s",
            attempt_id,
            event.event_key,
        )

    def _emergency_stop_if_overdue(
        self,
        attempt_id: str,
        event: Any,
        started: datetime,
    ) -> None:
        """Suspend the workload ourselves once containment is overdue.

        Reached from every path on which the control plane has not accepted the
        failure event: a failed POST, and a POST that was left to the outbox
        replay because an earlier copy is still buffered. Both mean the same
        thing operationally -- no workflow exists -- so both keep this last
        line of defence armed.
        """

        if (
            self.emergency_fallback_seconds is None
            or self.workload_stopper is None
            or attempt_id in self._workloads_stopped
            or (self.now() - started).total_seconds() < self.emergency_fallback_seconds
        ):
            return
        spec = self._attempt_specs.get(attempt_id)
        if spec is None:
            LOGGER.error(
                "emergency workload stop skipped: attempt %s has no spec",
                attempt_id,
            )
            return
        if not spec.workload_ids:
            return
        try:
            incident_id, _ = failure_containment_ids(event.event_key)
            existing_snapshots = list(event.workload_log_snapshots)
            if existing_snapshots and hasattr(self.workload_stopper, "capture_logs"):
                self.workload_stopper.stop(
                    spec.workload_ids,
                    attempt_id,
                    incident_id,
                    capture_logs=False,
                )
                snapshots = existing_snapshots
            else:
                snapshots = self.workload_stopper.stop(
                    spec.workload_ids,
                    attempt_id,
                    incident_id,
                )
        except Exception:
            LOGGER.exception(
                "emergency workload stop failed for attempt %s; will retry",
                attempt_id,
            )
            return
        self._failure_events[attempt_id] = event.model_copy(
            update={"workload_log_snapshots": snapshots or []}
        )
        self._attempt_specs[attempt_id] = replace(
            spec,
            termination_initiator_incident_id=incident_id,
        )
        self._emergency_incident_ids[attempt_id] = incident_id
        self._workloads_stopped.add(attempt_id)
        LOGGER.error(
            "control plane unavailable; emergency fallback "
            "suspended attempt=%s workloads=%s",
            attempt_id,
            ",".join(spec.workload_ids),
        )
