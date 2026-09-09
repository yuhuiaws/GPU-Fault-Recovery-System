"""Pod metadata -> attempt spec / container observation for the watcher.

The label and annotation vocabulary the Completion Watcher reads off managed
Pods, the controller error type, and the parsing methods that turn a list of
Pods into an ``AttemptSpec`` and an ``AttemptObservation``. Split out of
``completion_controller`` as a pure move (F6b); the mixin reads the
controller state it declares below and nothing else.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Callable

from gpu_fault.completion_attempt_state import AttemptSpec
from gpu_fault.completion_observation import (
    TERMINATION_INCIDENT_ANNOTATION,
    observe_containers,
    pod_container_has_started,
)
from gpu_fault.models import Environment
from gpu_fault.watcher import (
    AttemptObservation,
    ContainerObservation,
    WorkloadPhase,
)

# Borrowed on purpose: every line below used to be logged by
# ``gpu_fault.completion_controller`` and the log format prints the logger
# name, so the split must not rename what operators grep for.
LOGGER = logging.getLogger("gpu_fault.completion_controller")
MANAGED_LABEL = "gpu-fault.io/managed"
JOB_LABEL = "gpu-fault.io/job-id"
ATTEMPT_LABEL = "gpu-fault.io/attempt-id"
ROLE_LABEL = "gpu-fault.io/role"
CRITICAL_LABEL = "gpu-fault.io/critical"
RANK_ANNOTATION = "gpu-fault.io/rank"
RANK_OFFSET_ANNOTATION = "gpu-fault.io/rank-offset"
RANK_JOB_STRIDE_ANNOTATION = "gpu-fault.io/rank-job-stride"
INDEXED_JOB_RANK = "batch.kubernetes.io/job-completion-index"
KUBEFLOW_REPLICA_INDEX = "training.kubeflow.org/replica-index"
JOBSET_JOB_INDEX = "jobset.sigs.k8s.io/job-index"
EXPECTED_RANKS_ANNOTATION = "gpu-fault.io/expected-critical-ranks"
TRAINING_CONTAINER_ANNOTATION = "gpu-fault.io/training-container"
RUNTIME_PROFILE_ANNOTATION = "gpu-fault.io/runtime-profile-version"
GPU_UUIDS_ANNOTATION = "gpu-fault.io/gpu-uuids"
HOST_PID_ANNOTATION = "gpu-fault.io/host-pid"
CGROUP_PATH_ANNOTATION = "gpu-fault.io/cgroup-path"
CHECKPOINT_ANNOTATION = "gpu-fault.io/checkpoint-manifest"
WORKLOAD_IDS_ANNOTATION = "gpu-fault.io/workload-ids"
JOBSET_NAME_LABEL = "jobset.sigs.k8s.io/jobset-name"
PYTORCH_JOB_LABELS = (
    "training.kubeflow.org/job-name",
    "pytorch-job-name",
)
WORKLOAD_LOG_SNAPSHOT_ANNOTATION = "gpu-fault.io/workload-log-snapshot"
RESTART_BUDGET_ANNOTATION = "gpu-fault.io/restart-budget"


class CompletionControllerError(ValueError):
    pass


class CompletionPodParsingMixin:
    """Pod -> spec/container parsing half of ``KubernetesCompletionController``."""

    # Attributes supplied by the composed concrete implementation.
    _attempt_specs: dict[str, AttemptSpec]
    _emergency_incident_ids: dict[str, str]
    _forget_attempt_progress: Callable[[str], None]
    _gpu_uuid_cache: dict[str, list[str]]
    _gpu_uuid_failures: dict[str, tuple[int, datetime]]
    _gpu_uuids: Callable[[str | None], list[str]]
    _pod_key: Callable[[dict[str, Any]], str]
    cleanup_timeout_seconds: int
    cluster_id: str
    environment: Environment
    gpu_uuid_resolver: Callable[[dict[str, Any], str], list[str]] | None
    metadata_takeovers_total: int
    now: Callable[[], datetime]

    def _observation(
        self,
        attempt_id: str,
        pods: list[dict[str, Any]],
        observed_at: datetime,
    ) -> AttemptObservation:
        if pods:
            spec = self._spec_from_pods(attempt_id, pods)
            previous = self._attempt_specs.get(attempt_id)
            if previous is not None and previous != spec:
                comparable_previous = {
                    **previous.__dict__,
                    "termination_initiator_incident_id": None,
                }
                comparable_current = {
                    **spec.__dict__,
                    "termination_initiator_incident_id": None,
                }
                if not (
                    comparable_previous == comparable_current
                    and previous.termination_initiator_incident_id is None
                    and spec.termination_initiator_incident_id is not None
                ):
                    self._take_over_attempt_spec(attempt_id, previous, spec)
            self._attempt_specs[attempt_id] = spec
        else:
            spec = self._attempt_specs[attempt_id]

        critical_pods = [pod for pod in pods if self._is_critical(pod)]
        containers = observe_containers(self, critical_pods)
        started = any(
            pod_container_has_started(pod, item.container_name)
            for pod, item in zip(critical_pods, containers, strict=True)
        )
        terminated = [item for item in containers if item.terminated]
        failed = [
            item
            for item in terminated
            if item.exit_code is not None and item.exit_code != 0
        ]
        if (
            spec.termination_initiator_incident_id
            and len(terminated) >= spec.expected_critical_ranks
        ):
            phase = WorkloadPhase.STOPPED
        elif failed:
            phase = WorkloadPhase.FAILED
        elif len(terminated) >= spec.expected_critical_ranks and all(
            item.exit_code == 0 for item in terminated
        ):
            phase = WorkloadPhase.SUCCEEDED
        elif containers and started:
            phase = WorkloadPhase.RUNNING
        else:
            # A Pod that exists is not a Pod that started: unschedulable and
            # image-pulling Pods carry no container state at all, and PENDING
            # used to be unreachable once a Pod existed (F10).
            phase = WorkloadPhase.PENDING
        pod_start_times = [
            started_at
            for pod in pods
            if (
                started_at := self._time(
                    (pod.get("status") or {}).get("startTime")
                    or (pod.get("status") or {}).get("start_time")
                    or (pod.get("metadata") or {}).get("creationTimestamp")
                    or (pod.get("metadata") or {}).get("creation_timestamp")
                )
            )
            is not None
        ]
        return AttemptObservation(
            cluster_id=spec.cluster_id,
            environment=spec.environment,
            job_id=spec.job_id,
            attempt_id=spec.attempt_id,
            workload_phase=phase,
            observed_at=observed_at,
            started_at=(min(pod_start_times) if pod_start_times else None),
            expected_critical_ranks=(spec.expected_critical_ranks),
            containers=containers,
            workload_ids=list(spec.workload_ids),
            cleanup_timeout_seconds=(spec.cleanup_timeout_seconds),
            checkpoint_manifest_ref=(spec.checkpoint_manifest_ref),
            termination_initiator_incident_id=(spec.termination_initiator_incident_id),
            runtime_profile_version=(spec.runtime_profile_version),
            restart_budget=spec.restart_budget,
        )

    def _take_over_attempt_spec(
        self,
        attempt_id: str,
        previous: AttemptSpec,
        current: AttemptSpec,
    ) -> None:
        self.metadata_takeovers_total += 1
        LOGGER.warning(
            "attempt metadata generation changed; accepting the "
            "new internally-consistent Pod spec: attempt=%s "
            "old_job=%s new_job=%s old_expected_ranks=%s "
            "new_expected_ranks=%s old_profile=%s new_profile=%s",
            attempt_id,
            previous.job_id,
            current.job_id,
            previous.expected_critical_ranks,
            current.expected_critical_ranks,
            previous.runtime_profile_version,
            current.runtime_profile_version,
        )
        self._forget_attempt_progress(attempt_id)

    def _spec_from_pods(
        self, attempt_id: str, pods: list[dict[str, Any]]
    ) -> AttemptSpec:
        values = []
        for pod in pods:
            metadata = pod.get("metadata") or {}
            labels = metadata.get("labels") or {}
            annotations = metadata.get("annotations") or {}
            try:
                expected = int(annotations[EXPECTED_RANKS_ANNOTATION])
            except (KeyError, TypeError, ValueError) as exc:
                raise CompletionControllerError(
                    f"attempt {attempt_id} requires integer {EXPECTED_RANKS_ANNOTATION}"
                ) from exc
            runtime_profile = annotations.get(RUNTIME_PROFILE_ANNOTATION)
            raw_restart_budget = annotations.get(
                RESTART_BUDGET_ANNOTATION,
                os.getenv("GPU_FAULT_DEFAULT_RESTART_BUDGET", "1"),
            )
            try:
                restart_budget = int(raw_restart_budget)
            except (TypeError, ValueError) as exc:
                raise CompletionControllerError(
                    f"attempt {attempt_id} requires non-negative "
                    f"{RESTART_BUDGET_ANNOTATION}"
                ) from exc
            job_id = labels.get(JOB_LABEL)
            if not job_id or not runtime_profile or expected < 1 or restart_budget < 0:
                raise CompletionControllerError(
                    "managed Pod requires job ID, runtime profile "
                    "and valid rank/restart budget"
                )
            values.append(
                (
                    job_id,
                    expected,
                    runtime_profile,
                    tuple(dict.fromkeys(self._workload_ids(pod))),
                    annotations.get(CHECKPOINT_ANNOTATION),
                    annotations.get(TERMINATION_INCIDENT_ANNOTATION),
                    restart_budget,
                )
            )
        if len(set(values)) != 1:
            raise CompletionControllerError(
                "attempt Pods disagree on immutable metadata"
            )
        (
            job_id,
            expected,
            profile,
            workload_ids,
            checkpoint,
            initiator,
            restart_budget,
        ) = values[0]
        return AttemptSpec(
            cluster_id=self.cluster_id,
            environment=self.environment,
            job_id=job_id,
            attempt_id=attempt_id,
            expected_critical_ranks=expected,
            runtime_profile_version=profile,
            cleanup_timeout_seconds=self.cleanup_timeout_seconds,
            workload_ids=workload_ids,
            checkpoint_manifest_ref=checkpoint,
            termination_initiator_incident_id=(
                initiator or self._emergency_incident_ids.get(attempt_id)
            ),
            restart_budget=restart_budget,
        )

    def _container(self, pod: dict[str, Any]) -> ContainerObservation:
        metadata = pod.get("metadata") or {}
        labels = metadata.get("labels") or {}
        annotations = metadata.get("annotations") or {}
        spec = pod.get("spec") or {}
        status = pod.get("status") or {}
        name = annotations.get(TRAINING_CONTAINER_ANNOTATION)
        if not name:
            containers = spec.get("containers") or []
            if len(containers) != 1:
                raise CompletionControllerError(
                    f"Pod {metadata.get('name')} must identify its training container"
                )
            name = containers[0].get("name")
        container_spec = next(
            (
                item
                for item in (spec.get("containers") or [])
                if item.get("name") == name
            ),
            {},
        )
        resources = container_spec.get("resources") or {}
        requests = resources.get("requests") or {}
        limits = resources.get("limits") or {}
        try:
            gpu_count = max(
                int(requests.get("nvidia.com/gpu", 0)),
                int(limits.get("nvidia.com/gpu", 0)),
            )
        except (TypeError, ValueError) as exc:
            raise CompletionControllerError(
                f"Pod {metadata.get('name')} has invalid nvidia.com/gpu quantity"
            ) from exc
        statuses = (
            status.get("containerStatuses") or status.get("container_statuses") or []
        )
        selected = next(
            (item for item in statuses if item.get("name") == name),
            {},
        )
        terminated = (selected.get("state") or {}).get("terminated") or {}
        is_terminated = bool(terminated)
        try:
            explicit_rank = annotations.get(RANK_ANNOTATION)
            if explicit_rank is not None:
                rank = int(explicit_rank)
            else:
                local_rank_value = (
                    annotations.get(INDEXED_JOB_RANK)
                    or labels.get(INDEXED_JOB_RANK)
                    or labels.get(KUBEFLOW_REPLICA_INDEX)
                )
                offset_value = annotations.get(RANK_OFFSET_ANNOTATION)
                if local_rank_value is None and offset_value is None:
                    raise KeyError(RANK_OFFSET_ANNOTATION)
                local_rank = int(local_rank_value or 0)
                offset = int(offset_value or 0)
                job_index = int(labels.get(JOBSET_JOB_INDEX, 0))
                stride = int(annotations.get(RANK_JOB_STRIDE_ANNOTATION, 0))
                rank = offset + job_index * stride + local_rank
        except (KeyError, TypeError, ValueError) as exc:
            raise CompletionControllerError(
                f"Pod {metadata.get('name')} requires integer rank "
                f"or {INDEXED_JOB_RANK}"
            ) from exc
        gpu_uuids = self._gpu_uuids(annotations.get(GPU_UUIDS_ANNOTATION))
        workload_log_snapshot = None
        raw_snapshot = annotations.get(WORKLOAD_LOG_SNAPSHOT_ANNOTATION)
        if raw_snapshot:
            try:
                parsed_snapshot = json.loads(raw_snapshot)
            except json.JSONDecodeError as exc:
                raise CompletionControllerError(
                    f"Pod {metadata.get('name')} has invalid "
                    f"{WORKLOAD_LOG_SNAPSHOT_ANNOTATION}"
                ) from exc
            if not isinstance(parsed_snapshot, dict):
                raise CompletionControllerError(
                    f"Pod {metadata.get('name')} requires object "
                    f"{WORKLOAD_LOG_SNAPSHOT_ANNOTATION}"
                )
            workload_log_snapshot = parsed_snapshot
        if (
            not gpu_uuids
            and not is_terminated
            and (status.get("phase") or "").upper() == "RUNNING"
            and (selected.get("state") or {}).get("running") is not None
        ):
            gpu_uuids = self._discover_gpu_uuids(pod, name)
        return ContainerObservation(
            pod_uid=metadata.get("uid") or metadata.get("name"),
            pod_name=metadata.get("name"),
            container_name=name,
            container_id=(selected.get("containerID") or selected.get("container_id")),
            host_pid=(
                int(annotations[HOST_PID_ANNOTATION])
                if annotations.get(HOST_PID_ANNOTATION)
                else None
            ),
            cgroup_path=annotations.get(CGROUP_PATH_ANNOTATION),
            role=labels.get(ROLE_LABEL, "worker"),
            rank=rank,
            critical=True,
            node_id=spec.get("nodeName") or spec.get("node_name"),
            instance_id=labels.get("node.kubernetes.io/instance-id"),
            gpu_uuids=gpu_uuids,
            gpu_count=gpu_count,
            workload_log_snapshot=workload_log_snapshot,
            terminated=is_terminated,
            exit_code=(
                terminated.get("exitCode")
                if "exitCode" in terminated
                else terminated.get("exit_code")
            ),
            signal=terminated.get("signal"),
            finished_at=self._time(
                terminated.get("finishedAt") or terminated.get("finished_at")
            ),
            restart_count=selected.get(
                "restartCount",
                selected.get("restart_count", 0),
            ),
        )

    @staticmethod
    def _is_critical(pod: dict[str, Any]) -> bool:
        labels = (pod.get("metadata") or {}).get("labels") or {}
        return labels.get(CRITICAL_LABEL, "true").lower() == "true"

    def _discover_gpu_uuids(
        self, pod: dict[str, Any], container_name: str
    ) -> list[str]:
        if self.gpu_uuid_resolver is None:
            return []
        key = self._pod_key(pod)
        cached = self._gpu_uuid_cache.get(key)
        if cached is not None:
            return list(cached)
        failure = self._gpu_uuid_failures.get(key)
        if failure is not None and self.now() < failure[1]:
            return []
        metadata = pod.get("metadata") or {}
        try:
            values = [
                value.strip()
                for value in self.gpu_uuid_resolver(pod, container_name)
                if value.strip()
            ]
            if not values:
                raise CompletionControllerError(
                    "nvidia-smi returned no visible GPU UUIDs"
                )
            invalid = [
                value for value in values if not value.startswith(("GPU-", "MIG-"))
            ]
            if invalid:
                raise CompletionControllerError("nvidia-smi returned invalid GPU UUIDs")
            if len(values) != len(set(values)):
                raise CompletionControllerError(
                    "nvidia-smi returned duplicate GPU UUIDs"
                )
        except Exception as exc:
            attempts = self._gpu_uuid_failures.get(key, (0, self.now()))[0] + 1
            delay = min(60, 2 ** min(attempts, 6))
            self._gpu_uuid_failures[key] = (
                attempts,
                self.now() + timedelta(seconds=delay),
            )
            LOGGER.warning(
                "cannot discover GPU UUIDs for Pod %s/%s; retry in %ss: %s",
                metadata.get("namespace", "default"),
                metadata.get("name"),
                delay,
                exc,
            )
            return []
        self._gpu_uuid_cache[key] = values
        self._gpu_uuid_failures.pop(key, None)
        return list(values)

    @staticmethod
    def _string_list(value: str | None, description: str) -> list[str]:
        if not value:
            return []
        if value.lstrip().startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list) or not all(
                isinstance(item, str) for item in parsed
            ):
                raise CompletionControllerError(
                    f"{description} must be a JSON string list"
                )
            return parsed
        return [item.strip() for item in value.split(",") if item.strip()]

    def _workload_ids(self, pod: dict[str, Any]) -> list[str]:
        metadata = pod.get("metadata") or {}
        annotations = metadata.get("annotations") or {}
        explicit = self._string_list(
            annotations.get(WORKLOAD_IDS_ANNOTATION),
            "workload IDs",
        )
        if explicit:
            return explicit
        labels = metadata.get("labels") or {}
        namespace = metadata.get("namespace") or "default"
        jobset_name = labels.get(JOBSET_NAME_LABEL)
        if jobset_name:
            return [f"{namespace}/jobset/{jobset_name}"]
        for label in PYTORCH_JOB_LABELS:
            pytorch_name = labels.get(label)
            if pytorch_name:
                return [f"{namespace}/pytorchjob/{pytorch_name}"]
        owners = (
            metadata.get("ownerReferences") or metadata.get("owner_references") or []
        )
        controller_owners = [
            owner for owner in owners if owner.get("controller", False)
        ]
        selected = controller_owners or owners
        for owner in selected:
            kind = str(owner.get("kind", "")).lower()
            name = owner.get("name")
            if kind in {"job", "pytorchjob", "jobset"} and name:
                return [f"{namespace}/{kind}/{name}"]
        return []

    @staticmethod
    def _time(value: str | datetime | None) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
