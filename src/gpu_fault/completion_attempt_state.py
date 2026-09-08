from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, cast

from pydantic import ValidationError

from gpu_fault.attempt_observation_state import terminal_attempt_observation
from gpu_fault.completion_attempt_store import (
    attempt_observation_payload,
    attempt_state_record,
)
from gpu_fault.completion_observation import (
    TERMINAL_ORIGIN_MISSING_TOMBSTONE,
    TERMINAL_ORIGIN_OBSERVED,
)
from gpu_fault.completion_outbox import CompletionOutboxFull
from gpu_fault.models import Environment, TerminalEvent
from gpu_fault.telemetry import (
    ATTEMPT_COVERAGE_PATH,
    WorkloadCoverageHeartbeat,
)
from gpu_fault.watcher import AttemptObservation, WorkloadPhase

LOGGER = logging.getLogger(__name__)
ACTIVE_PHASES = frozenset({WorkloadPhase.PENDING, WorkloadPhase.RUNNING})
#: The only two Kubernetes Pod phases that mean "nothing of this Pod is
#: running". Everything else -- ``Pending``, ``Running``, the ``Unknown`` of a
#: kubelet that stopped answering, a status the API server has not written yet,
#: a phase a future release adds -- may still hold a GPU, so
#: :func:`active_pass_counts` counts it as running rather than guessing.
FINISHED_POD_PHASES = frozenset({"SUCCEEDED", "FAILED"})
#: How many (attempt, error) pairs :func:`log_reconcile_failure` remembers.
#: Bounded because a cluster that churns attempts would otherwise grow the map
#: for the lifetime of the process; the oldest pair is dropped, so its next
#: failure prints one more traceback and nothing is lost but a repeat.
REPEATED_FAILURE_CAP = 1024
#: Failures whose message states a measurement rather than only a cause.
#: ``CompletionOutboxFull`` names the bytes it measured (``... exceeds its byte
#: bound (13245 > 7000 bytes)``), and that number moves whenever any *other*
#: attempt in the same document changes size -- a container restart is enough.
#: Keying the log-once map on the text therefore reported a "new" failure on
#: every pass, which is the storm this whole path exists to stop, so these are
#: keyed on the exception type instead: one over-bound attempt is one problem
#: however the measurement drifts. The DEBUG repeat still carries the current
#: numbers, so the growth is visible to anyone who turns DEBUG on.
MEASURED_FAILURES: tuple[type[BaseException], ...] = (CompletionOutboxFull,)


@dataclass(frozen=True)
class AttemptSpec:
    cluster_id: str
    environment: Environment
    job_id: str
    attempt_id: str
    expected_critical_ranks: int
    runtime_profile_version: str
    cleanup_timeout_seconds: int
    workload_ids: tuple[str, ...] = ()
    checkpoint_manifest_ref: str | None = None
    termination_initiator_incident_id: str | None = None
    restart_budget: int = 1


def attempt_spec_from_observation(observation: AttemptObservation) -> AttemptSpec:
    return AttemptSpec(
        cluster_id=observation.cluster_id,
        environment=observation.environment,
        job_id=observation.job_id,
        attempt_id=observation.attempt_id,
        expected_critical_ranks=observation.expected_critical_ranks,
        runtime_profile_version=observation.runtime_profile_version,
        cleanup_timeout_seconds=observation.cleanup_timeout_seconds,
        workload_ids=tuple(observation.workload_ids),
        checkpoint_manifest_ref=observation.checkpoint_manifest_ref,
        termination_initiator_incident_id=(
            observation.termination_initiator_incident_id
        ),
        restart_budget=observation.restart_budget,
    )


def restore_persisted_attempt_observations(controller: Any) -> None:
    """Rebuild in-memory attempt state from the sink's persisted records.

    A record that cannot be parsed, or that belongs to another cluster,
    environment or is no longer active, is skipped and counted in
    ``controller.restore_skipped_total`` rather than raised: the persisted
    state is shared by every Watcher generation, and one foreign record used
    to keep the whole process from starting (F-G4 / P1-24B).
    """
    controller.restore_skipped_total = 0
    load = getattr(controller.sink, "load_attempt_observations", None)
    if load is None:
        return
    for payload in load():
        try:
            observation = AttemptObservation.model_validate(
                attempt_observation_payload(payload)
                if isinstance(payload, dict)
                else payload
            )
        except ValidationError as exc:
            controller.restore_skipped_total += 1
            LOGGER.warning(
                "skipping unparsable persisted attempt observation %r: %s",
                (payload or {}).get("attempt_id")
                if isinstance(payload, dict)
                else None,
                exc,
            )
            continue
        if (
            observation.cluster_id != controller.cluster_id
            or observation.environment is not controller.environment
            or observation.workload_phase not in ACTIVE_PHASES
        ):
            controller.restore_skipped_total += 1
            LOGGER.warning(
                "skipping persisted attempt observation that does not belong to "
                "this watcher: attempt=%s cluster=%s environment=%s phase=%s",
                observation.attempt_id,
                observation.cluster_id,
                observation.environment.value,
                observation.workload_phase.value,
            )
            continue
        try:
            controller.watcher.observe(observation)
        except ValueError as exc:
            controller.restore_skipped_total += 1
            LOGGER.warning(
                "skipping persisted attempt observation rejected by the watcher "
                "core: attempt=%s: %s",
                observation.attempt_id,
                exc,
            )
            continue
        controller._attempt_specs[observation.attempt_id] = (
            attempt_spec_from_observation(observation)
        )
        controller._last_observations[observation.attempt_id] = observation
        # This process has never seen a Pod of this attempt, so the absence of
        # Pods proves nothing about how it ended: the tombstone path has to ask
        # the workload object first (F7). Cleared as soon as one of its Pods is
        # listed.
        controller._restored_attempts.add(observation.attempt_id)


def log_attempt_failure(
    controller: Any,
    attempt_id: str,
    exc: BaseException,
    *,
    action: str,
) -> None:
    """Report a per-attempt failure once at ERROR, then at DEBUG (F9 / M1).

    Two failures repeat for as long as their cause exists, once per
    ``reconcile_interval_seconds``: a Pod whose status will never become valid
    (a ``startTime`` that is not a timestamp, a field a future kubelet writes
    differently) and an attempt whose state does not fit the bounded
    attempt-state ConfigMap. Each used to print a full traceback per pass. A day
    of that buries the failures an operator can act on and costs real money in
    log ingest, while dropping the repeats entirely would hide a permanent
    breakage. So the first sighting of each failure is an ERROR with its
    traceback and every repeat is one DEBUG line; the counters
    (``reconcile_failures_total``, ``gpu_fault_completion_outbox_...``) still
    count them all, which is what the alerts read.

    Keyed by ``action`` and the error text, not by the exception type: two
    malformed fields on the same attempt are two different problems, and the
    same text is the same problem however many passes it survives. The
    exceptions in ``MEASURED_FAILURES`` are keyed by type, because their text
    embeds a measurement that moves without the problem changing.
    """

    seen = getattr(controller, "_reconcile_failures_logged", None)
    if seen is None:
        # Created here rather than in the controller's constructor: this is the
        # logger's own bookkeeping, and no other caller may read it.
        seen = OrderedDict()
        controller._reconcile_failures_logged = seen
    description = f"{type(exc).__name__}: {exc}"
    identity = type(exc).__name__ if isinstance(exc, MEASURED_FAILURES) else description
    digest = hashlib.sha256(f"{action}/{identity}".encode()).hexdigest()[:12]
    key = f"{attempt_id}/{digest}"
    if key in seen:
        seen.move_to_end(key)
        LOGGER.debug(
            "cannot %s attempt %s, unchanged since it was first reported: %s",
            action,
            attempt_id,
            description,
        )
        return
    seen[key] = None
    while len(seen) > REPEATED_FAILURE_CAP:
        seen.popitem(last=False)
    LOGGER.error("cannot %s attempt %s", action, attempt_id, exc_info=exc)


def log_reconcile_failure(controller: Any, attempt_id: str, exc: BaseException) -> None:
    """Report a failure to reconcile one attempt; see ``log_attempt_failure``."""

    log_attempt_failure(controller, attempt_id, exc, action="reconcile")


def cache_terminal_attempt_observation(
    controller: Any,
    attempt_id: str,
    event: TerminalEvent,
    observation: AttemptObservation,
) -> AttemptObservation:
    # The origin is decided before the tracker is cleared: only a terminal the
    # tracker itself produced from missing Pods is a tombstone, and only a
    # tombstone may be evicted later by Pods that come back (F2). A terminal
    # that was read off Pod status -- including the TIMED_OUT of an attempt
    # whose failure we did observe -- stays authoritative.
    origin = (
        TERMINAL_ORIGIN_MISSING_TOMBSTONE
        if (
            controller._missing_attempts.is_tombstoned(attempt_id)
            and observation.workload_phase is WorkloadPhase.STOPPED
        )
        else TERMINAL_ORIGIN_OBSERVED
    )
    controller._missing_attempts.clear(attempt_id)
    return cast(
        AttemptObservation,
        controller._terminal_observations.setdefault(
            attempt_id,
            terminal_attempt_observation(event, observation),
            origin,
        ),
    )


def active_pass_counts(controller: Any, pods: list[dict[str, Any]]) -> tuple[int, int]:
    """What a completed full pass still sees running: Pods, then attempts.

    Both counts mean "in use by a workload", which is the only question the
    coverage heartbeat answers, so neither is a raw total. A Succeeded Pod
    holds no GPU and lingers until Kubernetes collects it -- counting it would
    keep a cluster whose last job ended days ago from ever reading IDLE, the
    live DESTR-016 case this exists for. An attempt kept only for its terminal
    retention window is finished for the same reason.

    An attempt this process still believes is PENDING or RUNNING counts even
    when none of its Pods were listed: a restored attempt whose Pods this
    process has never seen, or one inside its missing-attempt grace, is exactly
    the state where the control plane has no fresh observation and IDLE would be
    the fail-open answer.

    Which is also why the Pod count is "not finished" rather than "Pending or
    Running": the Pod this count exists for is the managed one that could not be
    grouped at all -- no attempt-id label, so it produces no observation -- and
    a Pod in phase ``Unknown`` (the kubelet stopped answering, which is what a
    GPU fault does) or with no status written yet may still be training. Only
    ``Succeeded`` and ``Failed`` are safe to skip.
    """

    active_pods = sum(
        1
        for item in pods
        if (
            (controller.serializer(item).get("status") or {}).get("phase") or ""
        ).upper()
        not in FINISHED_POD_PHASES
    )
    active_attempts = sum(
        1
        for observation in controller._last_observations.values()
        if observation.workload_phase in ACTIVE_PHASES
    )
    return active_pods, active_attempts


def publish_coverage_heartbeat(
    controller: Any,
    *,
    watched_pods: int,
    watched_attempts: int,
    resource_version: str | None,
    reconcile_failures: int,
) -> None:
    """Tell the control plane this pass watched the whole cluster and it is idle
    (F4).

    Only a completed full pass calls this, which is what makes the
    statement true: a pass that raised never reaches it, and a debounced
    pass over one attempt looked at one attempt. A watcher that does not
    publish observations stays silent as well -- claiming coverage while
    the control plane cannot see the attempts that *are* running would turn
    a busy node into an IDLE one, which is the fail-open direction.

    Four more silences, each of them that same direction:

    * A namespace-scoped watcher never vouches. It lists one namespace, so it
      cannot say anything about a node running a job in another one, and the
      control plane's answer is per cluster.
    * A pass that saw a running Pod or a live attempt says nothing. The
      heartbeat's only job is to distinguish "nothing to see" from "nobody
      watching", and on a busy cluster the observations already answer that.
    * A pass in which handling one attempt raised says nothing. Those failures
      are swallowed per attempt on purpose, so a pass can finish having failed
      to publish an observation -- and then "no observation" would be a lost
      write, not an idle cluster.
    * At most one heartbeat per ``coverage_heartbeat_interval_seconds``. The
      row is superseded by design and the control plane's window is minutes, so
      a heartbeat per pass is pure load on a strictly ordered, reserved-capacity
      ingress path shared with workflow dispatch.

    Delivery failures are counted and dropped: the heartbeat is weak
    evidence with a natural retry (the next pass), and a control plane that
    is refusing writes must not also fail the reconcile. The rate limit counts
    attempts rather than successes, so a control plane that is refusing this
    write is not asked again on every pass.
    """

    if not controller.publish_observations:
        return
    if controller.namespace:
        if not controller._coverage_scope_logged:
            controller._coverage_scope_logged = True
            LOGGER.info(
                "watching namespace %s only, so this watcher publishes no coverage "
                "heartbeat: an idle-cluster statement needs every namespace",
                controller.namespace,
            )
        return
    if watched_pods or watched_attempts or reconcile_failures:
        return
    observed_at = controller.now()
    last = controller._last_coverage_post_at
    if (
        last is not None
        and (observed_at - last).total_seconds()
        < controller.coverage_heartbeat_interval_seconds
    ):
        return
    controller._last_coverage_post_at = observed_at
    heartbeat = WorkloadCoverageHeartbeat(
        cluster_id=controller.cluster_id,
        observed_at=observed_at,
        watched_pods=watched_pods,
        watched_attempts=watched_attempts,
        resource_version=resource_version,
        watcher_instance=controller.watcher_instance,
    )
    try:
        controller.sink.post(ATTEMPT_COVERAGE_PATH, heartbeat.model_dump(mode="json"))
    except Exception:
        controller.coverage_heartbeat_failures_total += 1
        LOGGER.warning(
            "cannot publish coverage heartbeat for cluster %s",
            controller.cluster_id,
            exc_info=True,
        )
        return
    controller.coverage_heartbeats_total += 1
    controller.last_coverage_heartbeat_at = heartbeat.observed_at


def publish_attempt_observation(
    controller: Any,
    observation: AttemptObservation,
    attempt_id: str,
) -> None:
    payload = observation.model_dump(mode="json")
    # What is persisted is the compact record -- the spec plus each Pod's
    # attribution identity, without the log tail and the other fields the next
    # pass re-reads from a live Pod -- and what is posted is the whole payload.
    # Shrinking the persisted copy must not shrink the control plane's (F6).
    record = attempt_state_record(payload)
    if observation.workload_phase in ACTIVE_PHASES:
        persist = getattr(controller.sink, "save_attempt_observation", None)
        if persist is not None:
            try:
                persist(record)
            except Exception as exc:
                # Log-once rather than evicting the stalest record to make room:
                # what is refused here is the *newest* state, which the next
                # pass re-derives from the live Pod, whereas evicting an older
                # attempt would drop exactly the record a restart cannot
                # rebuild -- the one whose Pods are already gone (M1).
                log_attempt_failure(
                    controller,
                    attempt_id,
                    exc,
                    action="persist the active state of",
                )
    try:
        controller.sink.post("/v1/workload-observations", payload)
    except Exception:
        LOGGER.exception(
            "cannot publish workload observation for %s",
            attempt_id,
        )
        return
    if observation.workload_phase in ACTIVE_PHASES:
        return
    remove = getattr(controller.sink, "remove_attempt_observation", None)
    if remove is not None:
        try:
            remove(record)
        except Exception as exc:
            log_attempt_failure(
                controller,
                attempt_id,
                exc,
                action="drop the persisted state of",
            )
