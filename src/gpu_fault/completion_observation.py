from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, cast

from gpu_fault.watcher import AttemptObservation, WorkloadPhase

LOGGER = logging.getLogger(__name__)

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
    def __init__(self) -> None:
        self.since: dict[str, datetime] = {}

    def clear(self, attempt_id: str) -> None:
        self.since.pop(attempt_id, None)

    def observe_missing(
        self,
        attempt_id: str,
        observation: AttemptObservation,
        observed_at: datetime,
    ) -> AttemptObservation:
        missing_since = self.since.setdefault(attempt_id, observed_at)
        if (
            observed_at - missing_since
        ).total_seconds() < observation.cleanup_timeout_seconds:
            return observation
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
        controller._missing_attempts.clear(attempt_id)
        return terminal
    if attempt_pods:
        controller._missing_attempts.clear(attempt_id)
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
    return cast(
        AttemptObservation,
        controller._missing_attempts.observe_missing(
            attempt_id,
            previous,
            observed_at,
        ),
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


def _clear_attempt(controller, attempt_id: str) -> None:
    controller._attempt_specs.pop(attempt_id, None)
    controller._last_observations.pop(attempt_id, None)
    controller._terminal_observations.pop(attempt_id, None)
    controller._missing_attempts.clear(attempt_id)
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
