from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, cast

from gpu_fault.watcher import AttemptObservation, ContainerObservation, WorkloadPhase

LOGGER = logging.getLogger(__name__)
TERMINATION_INCIDENT_ANNOTATION = "gpu-fault.io/termination-initiator-incident-id"
_CUSTOM_WORKLOADS: dict[str, tuple[str, str, str]] = {
    "pytorchjob": ("kubeflow.org", "v1", "pytorchjobs"),
    "jobset": ("jobset.x-k8s.io", "v1alpha2", "jobsets"),
}
# Terminal observations are cached per attempt and shadow every later pass, so
# where one came from decides whether Pods that show up afterwards may evict
# it: a terminal read off Pod status is evidence, a terminal derived from the
# *absence* of Pods is a guess (F2).
TERMINAL_ORIGIN_OBSERVED = "observed"
TERMINAL_ORIGIN_MISSING_TOMBSTONE = "missing-tombstone"
# (connect, read) budget for the workload-object reads on the reconcile path.
# The kubernetes client leaves the read timeout unset by default, and a parked
# API-server endpoint would then hold the whole reconcile forever (F3).
WORKLOAD_READ_TIMEOUT = (5.0, 10.0)
_FAILURE_CONDITION_TYPES = frozenset({"failed"})
# What a FAILED verdict recovered from a workload object can say about the
# attempt: the Pods are gone, so there is no failed rank, no node and no exit
# code to attribute, and containment must not be raised on a guess (C1).
RECOVERED_VERDICT_REASON = "recovered from workload object; no per-rank evidence"


def _parse_workload_id(value: str) -> tuple[str, str, str]:
    parts = value.split("/")
    if len(parts) == 2:
        return parts[0], "job", parts[1]
    if len(parts) == 3:
        return parts[0], parts[1].lower(), parts[2]
    raise ValueError(f"invalid workload ID: {value}")


def read_workload_objects(
    controller: Any, workload_ids: list[str]
) -> list[dict[str, Any]]:
    """Every readable workload object of an attempt, read exactly once.

    Both questions the missing-Pod path asks -- "did we stop this ourselves?"
    and "what does the object say happened?" -- have to be answered from the
    *same* snapshot, and in that order (C1): deriving an outcome from a status
    that our own stop produced turns a stop into an unattributed failure.
    """

    objects: list[dict[str, Any]] = []
    for workload_id in workload_ids:
        data = read_workload_object(controller, workload_id)
        if data is not None:
            objects.append(data)
    return objects


def workload_objects_initiator(objects: list[dict[str, Any]]) -> str | None:
    """The initiator incident recorded on the workload object itself.

    ``STOP_WORKLOADS`` writes ``gpu-fault.io/termination-initiator-incident-id``
    on the Job/PyTorchJob/JobSet when it suspends it, so the annotation
    survives the Pods.
    """

    for data in objects:
        metadata = data.get("metadata") or {}
        annotations = metadata.get("annotations") or {}
        initiator = annotations.get(TERMINATION_INCIDENT_ANNOTATION)
        if initiator:
            return str(initiator)
    return None


def read_workload_object(controller: Any, workload_id: str) -> dict[str, Any] | None:
    """The Job/PyTorchJob/JobSet behind ``namespace/kind/name``, or ``None``.

    Read-only through the stopper's own API clients and bounded by
    ``_request_timeout``: this runs on the reconcile path, where an unbounded
    read of a wedged API-server endpoint would park the whole watcher. Any
    failure -- absent object, RBAC, timeout, unparsable ID -- is "unknown",
    never an exception on the observation path, and every caller has to fail
    closed on ``None``.
    """

    stopper = controller.workload_stopper
    try:
        namespace, kind, name = _parse_workload_id(workload_id)
        raw: Any
        if kind == "job":
            batch = getattr(stopper, "batch", None)
            if batch is None:
                return None
            raw = batch.read_namespaced_job(
                name, namespace, _request_timeout=WORKLOAD_READ_TIMEOUT
            )
        else:
            custom = getattr(stopper, "custom", None)
            if custom is None or kind not in _CUSTOM_WORKLOADS:
                return None
            group, version, plural = _CUSTOM_WORKLOADS[kind]
            raw = custom.get_namespaced_custom_object(
                group,
                version,
                namespace,
                plural,
                name,
                _request_timeout=WORKLOAD_READ_TIMEOUT,
            )
    except Exception as exc:  # noqa: BLE001 - observation must not fail
        LOGGER.debug("could not read workload %s: %s", workload_id, exc)
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return dict(controller.serializer(raw))
    except Exception as exc:  # noqa: BLE001 - observation must not fail
        LOGGER.debug("could not deserialize workload %s: %s", workload_id, exc)
        return None


def pod_container_has_started(pod: dict[str, Any], container_name: str) -> bool:
    """True when the Pod's training container is running or has terminated.

    A Pod that exists is not a Pod that started: an unschedulable or
    image-pulling Pod has no ``containerStatuses`` at all, and publishing it as
    RUNNING misreported both the phase and ``started_at`` (F10).

    A container that *has* run and is waiting to be restarted counts as
    started: a CrashLoopBackOff window shows ``state.waiting`` with the exit in
    ``lastState``, and reading that as PENDING would hide the attempt from hang
    detection for as long as the backoff lasts.
    """

    status = pod.get("status") or {}
    statuses = status.get("containerStatuses") or status.get("container_statuses") or []
    selected: dict[str, Any] = next(
        (item for item in statuses if item.get("name") == container_name),
        {},
    )
    if _as_count(selected.get("restartCount") or selected.get("restart_count")) > 0:
        return True
    states = [
        selected.get("state") or {},
        selected.get("lastState") or selected.get("last_state") or {},
    ]
    # Presence, not truthiness: a freshly started container reports
    # ``state: {running: {}}`` -- an empty, falsy dict.
    return any(
        state.get("running") is not None or state.get("terminated") is not None
        for state in states
    )


def _as_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


class WorkloadVerdict(Enum):
    """What a workload object says about an attempt whose Pods are all gone.

    ``ACTIVE`` is not an outcome: it means the object is still working (a
    ``backoffLimit`` retry, an elastic replica set that lost members) and the
    attempt has to stay in the missing state instead of being terminalized.
    ``UNKNOWN`` is the fail-closed default and leaves the stop in place.
    """

    UNKNOWN = "unknown"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class _WorkloadStatusFacts:
    succeeded: int = 0
    failed: int = 0
    active: int = 0
    failure_condition: bool = False


def _workload_status_facts(status: dict[str, Any]) -> _WorkloadStatusFacts:
    """Replica counts and the terminal failure condition, across our kinds.

    ``Job`` reports the counts at the top level, ``PyTorchJob`` per replica
    type, ``JobSet`` per replicated job, and all three carry a ``Failed``
    condition. An unknown shape contributes nothing, which reads as "no
    verdict" and leaves the caller with its fail-closed default.
    """

    succeeded = _as_count(status.get("succeeded"))
    failed = _as_count(status.get("failed"))
    active = _as_count(status.get("active"))
    groups: list[dict[str, Any]] = []
    replica_statuses = status.get("replicaStatuses") or status.get("replica_statuses")
    if isinstance(replica_statuses, dict):
        groups.extend(
            item for item in replica_statuses.values() if isinstance(item, dict)
        )
    replicated = status.get("replicatedJobsStatus") or status.get(
        "replicated_jobs_status"
    )
    if isinstance(replicated, list):
        groups.extend(item for item in replicated if isinstance(item, dict))
    for group in groups:
        succeeded += _as_count(group.get("succeeded"))
        failed += _as_count(group.get("failed"))
        active += _as_count(group.get("active"))
    failure_condition = False
    for condition in status.get("conditions") or []:
        if not isinstance(condition, dict):
            continue
        if str(condition.get("status", "")).lower() != "true":
            continue
        if str(condition.get("type", "")).lower() in _FAILURE_CONDITION_TYPES:
            failure_condition = True
    return _WorkloadStatusFacts(
        succeeded=succeeded,
        failed=failed,
        active=active,
        failure_condition=failure_condition,
    )


def _status_verdict(
    status: dict[str, Any], expected_critical_ranks: int
) -> WorkloadVerdict:
    facts = _workload_status_facts(status)
    # A true ``Failed`` condition is the object's own terminal verdict and
    # outranks every count: ``{"succeeded": 3, "failed": 5}`` used to be read
    # as a success because the success count was checked first (I1).
    if facts.failure_condition:
        return WorkloadVerdict.FAILED
    if facts.active > 0:
        # Still working: neither a success (ranks may yet fail) nor a failure
        # (the retry may yet succeed) may be declared here (I2).
        return WorkloadVerdict.ACTIVE
    if facts.succeeded >= max(1, expected_critical_ranks):
        return WorkloadVerdict.SUCCEEDED
    if facts.failed > 0:
        return WorkloadVerdict.FAILED
    return WorkloadVerdict.UNKNOWN


def workload_objects_verdict(
    objects: list[dict[str, Any]], expected_critical_ranks: int
) -> WorkloadVerdict:
    """What the workload objects say happened, defaulting to UNKNOWN (F7).

    The Pods of an attempt that finished while the watcher was down are
    garbage-collected (``ttlSecondsAfterFinished``, ``cleanPodPolicy``), so
    their absence used to be recorded as a user stop and the job's workflows
    were withdrawn. The workload object outlives its Pods and still carries the
    outcome.

    A success needs every critical rank accounted for; a partial count, an
    unreadable object or a shape we do not recognise stays UNKNOWN, because
    this path must never invent a success. ``ACTIVE`` from any object wins over
    a sibling's outcome: the attempt is not over while one of its workloads
    still is.
    """

    verdicts = [
        _status_verdict(status, expected_critical_ranks)
        for status in (data.get("status") for data in objects)
        if isinstance(status, dict)
    ]
    for candidate in (
        WorkloadVerdict.ACTIVE,
        WorkloadVerdict.FAILED,
        WorkloadVerdict.SUCCEEDED,
    ):
        if candidate in verdicts:
            return candidate
    return WorkloadVerdict.UNKNOWN


MANAGED_LABEL = "gpu-fault.io/managed"
JOB_LABEL = "gpu-fault.io/job-id"
ATTEMPT_LABEL = "gpu-fault.io/attempt-id"
RANK_ANNOTATION = "gpu-fault.io/rank"
EXPECTED_RANKS_ANNOTATION = "gpu-fault.io/expected-critical-ranks"
RUNTIME_PROFILE_ANNOTATION = "gpu-fault.io/runtime-profile-version"
RESTART_BUDGET_ANNOTATION = "gpu-fault.io/restart-budget"
JOBSET_NAME_LABEL = "jobset.sigs.k8s.io/jobset-name"
PYTORCH_JOB_LABELS = (
    "training.kubeflow.org/job-name",
    "pytorch-job-name",
)


class MissingAttemptTracker:
    """Since when every Pod of an attempt has been absent, and for how long.

    The grace is a *whole-attempt* budget of minutes, deliberately not
    ``cleanup_timeout_seconds``: that one is the container-cleanup budget the
    watcher core uses for the TIMED_OUT deadline of an attempt that already
    reported a failure, and reusing it here declared a user stop 30 s after a
    ``spec.suspend`` toggle or a Kueue readmission (F2).
    """

    def __init__(self, grace_seconds: float) -> None:
        if grace_seconds < 1:
            raise ValueError("attempt missing grace must be at least one second")
        self.grace_seconds = grace_seconds
        self.since: dict[str, datetime] = {}
        self.tombstoned: set[str] = set()

    def clear(self, attempt_id: str) -> None:
        self.since.pop(attempt_id, None)
        self.tombstoned.discard(attempt_id)

    def is_tombstoned(self, attempt_id: str) -> bool:
        """True when this attempt's last observation was a missing tombstone."""

        return attempt_id in self.tombstoned

    def defer_tombstone(self, attempt_id: str) -> None:
        """Withdraw a tombstone the workload object contradicts (I2).

        ``since`` is deliberately kept: the grace has already expired, so the
        first pass on which the workload object goes quiet terminalizes the
        attempt immediately instead of waiting out another whole grace.
        """

        self.tombstoned.discard(attempt_id)

    def observe_missing(
        self,
        attempt_id: str,
        observation: AttemptObservation,
        observed_at: datetime,
    ) -> AttemptObservation:
        missing_since = self.since.setdefault(attempt_id, observed_at)
        if (observed_at - missing_since).total_seconds() < self.grace_seconds:
            # Republished as-is *except* for the instant (final review I1):
            # the control plane ignores an observation older than its
            # ``max_age_seconds`` (120 s), and for a restored attempt the
            # persisted ``observed_at`` is hours old, so the RUNNING re-post
            # inside a 300 s grace was inert after two minutes and the
            # faulted GPU resolved IDLE -- the flip C1 exists to prevent.
            return observation.model_copy(update={"observed_at": observed_at})
        self.tombstoned.add(attempt_id)
        return cast(
            AttemptObservation,
            observation.model_copy(
                update={
                    "workload_phase": WorkloadPhase.STOPPED,
                    "observed_at": observed_at,
                    "containers": [],
                }
            ),
        )


def restored_attempt_observation(
    controller: Any,
    attempt_id: str,
    previous: AttemptObservation,
    tombstone: AttemptObservation,
    phase: WorkloadPhase,
    observed_at: datetime,
) -> AttemptObservation:
    """Replace a restored attempt's tombstone with the workload's own verdict.

    Only attempts rebuilt from persisted state reach here: this process never
    saw their Pods, so the absence of Pods says nothing about how they ended
    (F7).

    The ranks are marked terminated because the watcher core only derives
    SUCCEEDED/FAILED once every critical rank is terminated, but they all keep
    exit code 0: the Pods are gone, so no real code exists, and a synthesized
    non-zero code would name a first failed rank and a node that nothing
    observed (C1). A FAILED verdict rides on the phase alone, and the attempt
    is marked as carrying a synthesized verdict so that the caller posts the
    terminal without ever raising containment.
    """

    critical = [item for item in previous.containers if item.critical]
    if len(critical) < previous.expected_critical_ranks:
        LOGGER.warning(
            "restored attempt %s has %d of %d critical ranks persisted, so the "
            "workload verdict %s cannot be attributed; recording the stop",
            attempt_id,
            len(critical),
            previous.expected_critical_ranks,
            phase.value,
        )
        return tombstone
    if phase is WorkloadPhase.SUCCEEDED and any(
        item.terminated and item.exit_code not in (0, None) for item in critical
    ):
        LOGGER.warning(
            "restored attempt %s has a rank that exited non-zero, so the "
            "workload success verdict is not trusted; recording the stop",
            attempt_id,
        )
        return tombstone
    containers = [
        item
        if item.terminated
        else item.model_copy(
            update={
                "terminated": True,
                "exit_code": 0,
                "finished_at": observed_at,
            }
        )
        for item in critical
    ]
    controller._synthesized_verdicts.add(attempt_id)
    LOGGER.warning(
        "restored attempt %s has no Pods left and its workload object reports "
        "%s; recording that instead of a stop",
        attempt_id,
        phase.value,
    )
    return tombstone.model_copy(
        update={"workload_phase": phase, "containers": containers}
    )


def observe_containers(
    controller: Any, pods: list[dict[str, Any]]
) -> list[ContainerObservation]:
    """One ``ContainerObservation`` per Pod, stamping progress after each.

    A RUNNING Pod without a GPU UUID annotation costs one ``nvidia-smi`` exec
    (10 s on a hung kubelet), serially, and the in-memory backoff that spaces
    the retries is lost on restart: 32 such Pods were 320 s of loop with
    nothing refreshing the liveness clock, past the 310 s budget on a watcher
    that was working (final review M1). Each Pod read is one finished step.
    """

    containers: list[ContainerObservation] = []
    for pod in pods:
        containers.append(controller._container(pod))
        controller.note_progress()
    return containers


def reconcile_attempt_observation(
    controller: Any,
    attempt_id: str,
    attempt_pods: list[dict[str, Any]],
    observed_at: datetime,
) -> AttemptObservation | None:
    terminal = cast(
        AttemptObservation | None,
        controller._terminal_observations.get(attempt_id),
    )
    if terminal is not None:
        if not (
            attempt_pods
            and controller._terminal_observations.is_missing_tombstone(attempt_id)
        ):
            # A terminal read off Pod status is evidence: Pods listed after it
            # (a delayed relist, a replacement Pod) must not regress it.
            controller._missing_attempts.clear(attempt_id)
            return terminal
        # The tombstone only ever meant "no Pod of this attempt is left", and
        # live Pods under the same attempt-id disprove it. Forget everything
        # derived from it -- including the sent terminal keys -- and observe the
        # Pods as if the attempt were new, the way a metadata takeover does.
        controller.resume_tombstoned_attempt(attempt_id)
    if attempt_pods:
        controller._missing_attempts.clear(attempt_id)
        controller._restored_attempts.discard(attempt_id)
        controller._synthesized_verdicts.discard(attempt_id)
        return cast(
            AttemptObservation,
            controller._observation(attempt_id, attempt_pods, observed_at),
        )
    previous = cast(
        AttemptObservation | None,
        controller._last_observations.get(attempt_id),
    )
    if previous is None:
        return None
    if attempt_id in controller._failure_events:
        return cast(
            AttemptObservation,
            previous.model_copy(update={"observed_at": observed_at}),
        )
    result = cast(
        AttemptObservation,
        controller._missing_attempts.observe_missing(
            attempt_id,
            previous,
            observed_at,
        ),
    )
    if result.workload_phase is not WorkloadPhase.STOPPED:
        return result
    return resolve_missing_attempt_tombstone(
        controller, attempt_id, previous, result, observed_at
    )


def resolve_missing_attempt_tombstone(
    controller: Any,
    attempt_id: str,
    previous: AttemptObservation,
    tombstone: AttemptObservation,
    observed_at: datetime,
) -> AttemptObservation:
    """Let the workload objects qualify a tombstone the Pods cannot explain.

    Read once, and ask the questions in the order that keeps the path
    fail-closed (C1): who stopped this first, an outcome only afterwards.
    """

    if tombstone.termination_initiator_incident_id is not None:
        return tombstone
    objects = read_workload_objects(controller, previous.workload_ids)
    # STOP_WORKLOADS annotates the attempt Pods with the initiator incident
    # and then suspends the workload; a Pod that exits within one poll of the
    # patch is never observed annotated, so the tombstone would read as a user
    # stop and the control plane would withdraw the workflow that is waiting
    # to RESTART_WORKLOAD (DESTR-015, live). The stop step writes the same
    # annotation on the workload object, which outlives the Pods.
    initiator = workload_objects_initiator(objects)
    if initiator:
        # A stop we initiated also leaves ``status.failed`` behind -- it
        # deleted the Pods -- so the outcome must not be derived from it.
        return tombstone.model_copy(
            update={"termination_initiator_incident_id": initiator}
        )
    if attempt_id not in controller._restored_attempts:
        return tombstone
    verdict = workload_objects_verdict(objects, previous.expected_critical_ranks)
    if verdict is WorkloadVerdict.ACTIVE:
        controller._missing_attempts.defer_tombstone(attempt_id)
        LOGGER.info(
            "restored attempt %s has no Pods left but its workload object is "
            "still active; holding the stop",
            attempt_id,
        )
        return previous.model_copy(update={"observed_at": observed_at})
    if verdict is WorkloadVerdict.UNKNOWN:
        return tombstone
    phase = (
        WorkloadPhase.SUCCEEDED
        if verdict is WorkloadVerdict.SUCCEEDED
        else WorkloadPhase.FAILED
    )
    return restored_attempt_observation(
        controller, attempt_id, previous, tombstone, phase, observed_at
    )


class ObservationOnlyTracker:
    def __init__(
        self,
        enabled: bool,
        runtime_profile: str | None,
        retention_cycles: int,
    ) -> None:
        if retention_cycles < 1:
            raise ValueError("observation retention cycles must be positive")
        self.retention_cycles = retention_cycles
        self.enabled = enabled
        self.runtime_profile = runtime_profile
        if enabled and not runtime_profile:
            raise ValueError("observation-only mode requires a runtime profile")
        self.attempts: set[str] = set()
        self.missed: dict[str, int] = {}

    def update(
        self,
        controller,
        observed: set[str],
        active: set[str],
    ) -> None:
        self.attempts.update(observed)
        for attempt_id in list(self.attempts):
            if attempt_id in active:
                self.missed.pop(attempt_id, None)
                continue
            count = self.missed.get(attempt_id, 0) + 1
            if count < self.retention_cycles:
                self.missed[attempt_id] = count
                continue
            self.missed.pop(attempt_id, None)
            self.attempts.discard(attempt_id)
            _clear_attempt(controller, attempt_id)

    def contains(self, attempt_id: str) -> bool:
        return attempt_id in self.attempts

    def group(self, controller, pods, serializer):
        grouped, observed = group_completion_pods(
            pods,
            serializer=serializer,
            observe_unmanaged=self.enabled,
            runtime_profile=self.runtime_profile,
        )
        self.update(controller, observed, set(grouped))
        return grouped


def group_completion_pods(
    pods,
    *,
    serializer,
    observe_unmanaged: bool,
    runtime_profile: str | None,
):
    grouped = {}
    observation_only = set()
    for item in pods:
        pod = serializer(item)
        metadata = pod.get("metadata") or {}
        labels = metadata.get("labels") or {}
        managed = labels.get(MANAGED_LABEL, "").lower() == "true"
        attempt_id = labels.get(ATTEMPT_LABEL)
        if not managed:
            if not observe_unmanaged:
                continue
            attempt_id = observation_only_attempt_id(pod)
            if attempt_id is None:
                continue
            pod = observation_only_pod(
                pod,
                attempt_id,
                runtime_profile,
            )
            observation_only.add(attempt_id)
        if not attempt_id:
            LOGGER.warning(
                "managed Pod %s has no attempt ID",
                metadata.get("name"),
            )
            continue
        grouped.setdefault(attempt_id, []).append(pod)
    for attempt_id in observation_only:
        attempt_pods = grouped.get(attempt_id, [])
        for rank, pod in enumerate(
            sorted(
                attempt_pods,
                key=lambda value: ((value.get("metadata") or {}).get("name") or ""),
            )
        ):
            annotations = pod["metadata"].setdefault("annotations", {})
            annotations.setdefault(
                EXPECTED_RANKS_ANNOTATION,
                str(len(attempt_pods)),
            )
            annotations.setdefault(RANK_ANNOTATION, str(rank))
    return grouped, observation_only


def observation_only_attempt_id(pod) -> str | None:
    metadata = pod.get("metadata") or {}
    labels = metadata.get("labels") or {}
    jobset_name = labels.get(JOBSET_NAME_LABEL)
    job = jobset_name or next(
        (labels.get(name) for name in PYTORCH_JOB_LABELS if labels.get(name)),
        None,
    )
    if not job:
        return None
    namespace = metadata.get("namespace") or "default"
    owners = metadata.get("ownerReferences") or []
    owner = next(
        (item for item in owners if item.get("controller")),
        owners[0] if owners else {},
    )
    identity = (
        f"jobset-{jobset_name}"
        if jobset_name
        else owner.get("uid") or owner.get("name") or job
    )
    return f"observed/{namespace}/{job}/{identity}"


def observation_only_pod(
    pod,
    attempt_id: str,
    runtime_profile: str | None,
):
    metadata = pod.setdefault("metadata", {})
    labels = metadata.setdefault("labels", {})
    annotations = metadata.setdefault("annotations", {})
    namespace = metadata.get("namespace") or "default"
    job = labels.get(JOBSET_NAME_LABEL) or next(
        (labels.get(name) for name in PYTORCH_JOB_LABELS if labels.get(name)),
        "unknown",
    )
    labels[MANAGED_LABEL] = "true"
    labels[ATTEMPT_LABEL] = attempt_id
    labels[JOB_LABEL] = f"{namespace}/{job}"
    annotations[RUNTIME_PROFILE_ANNOTATION] = str(runtime_profile)
    annotations[RESTART_BUDGET_ANNOTATION] = "0"
    return pod


def completion_list_arguments(
    namespace: str | None,
    observe_unmanaged: bool,
) -> dict:
    arguments = {}
    if not observe_unmanaged:
        arguments["label_selector"] = f"{MANAGED_LABEL}=true"
    if namespace:
        arguments["namespace"] = namespace
    return arguments


def list_completion_pods(
    core_api,
    namespace: str | None,
    observe_unmanaged: bool,
):
    selector = None if observe_unmanaged else f"{MANAGED_LABEL}=true"
    if namespace:
        response = core_api.list_namespaced_pod(namespace, label_selector=selector)
    else:
        response = core_api.list_pod_for_all_namespaces(label_selector=selector)
    if isinstance(response, dict):
        metadata = response.get("metadata") or {}
        return response.get("items", []), (
            metadata.get("resourceVersion") or metadata.get("resource_version")
        )
    metadata = getattr(response, "metadata", None)
    return response.items, getattr(metadata, "resource_version", None)


def is_unknown_profile_rejection(exc: BaseException) -> bool:
    text = str(exc)
    return (
        getattr(exc, "status_code", None) == 404 and "resource not found" in text
    ) or (
        getattr(exc, "status_code", None) is None
        and "(404)" in text
        and "resource not found" in text
    )


class TerminalObservationCache:
    """Per-attempt terminal observations plus the origin that produced them.

    The origin is what makes the eviction rule expressible (F2): a terminal
    derived from Pod status is evidence and is never dropped, while a
    ``missing-tombstone`` -- derived from the *absence* of Pods -- is dropped as
    soon as Pods of that attempt-id are listed again. Terminal observations are
    never persisted (only PENDING/RUNNING ones are), so no origin can be lost
    across a restart: a restored attempt is active by construction and starts
    with no cache entry at all.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[AttemptObservation, str]] = {}

    def __contains__(self, attempt_id: object) -> bool:
        return attempt_id in self._entries

    def get(self, attempt_id: str) -> AttemptObservation | None:
        entry = self._entries.get(attempt_id)
        return None if entry is None else entry[0]

    def origin(self, attempt_id: str) -> str | None:
        entry = self._entries.get(attempt_id)
        return None if entry is None else entry[1]

    def is_missing_tombstone(self, attempt_id: str) -> bool:
        return self.origin(attempt_id) == TERMINAL_ORIGIN_MISSING_TOMBSTONE

    def setdefault(
        self,
        attempt_id: str,
        observation: AttemptObservation,
        origin: str,
    ) -> AttemptObservation:
        return self._entries.setdefault(attempt_id, (observation, origin))[0]

    def pop(
        self, attempt_id: str, default: AttemptObservation | None = None
    ) -> AttemptObservation | None:
        entry = self._entries.pop(attempt_id, None)
        return default if entry is None else entry[0]


def _clear_attempt(controller, attempt_id: str) -> None:
    controller._attempt_specs.pop(attempt_id, None)
    controller._last_observations.pop(attempt_id, None)
    controller._terminal_observations.pop(attempt_id, None)
    controller._missing_attempts.clear(attempt_id)
    controller._restored_attempts.discard(attempt_id)
    controller._synthesized_verdicts.discard(attempt_id)
    controller._failure_events.pop(attempt_id, None)
    controller._failure_sent.discard(attempt_id)
    controller._failure_delivery_started.pop(attempt_id, None)
    controller._workloads_stopped.discard(attempt_id)
    controller._emergency_incident_ids.pop(attempt_id, None)
    controller.watcher.reset_attempt(attempt_id)
    fragment = f"/{attempt_id}/"
    controller._terminal_sent = {
        key for key in controller._terminal_sent if fragment not in key
    }
