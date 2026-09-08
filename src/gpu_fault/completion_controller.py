from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import RLock, Timer, current_thread
from typing import Any, Callable
from urllib.parse import urlparse

from gpu_fault.collectors import EventSink
from gpu_fault.collectors.sinks import RETRY_AFTER_CAP_SECONDS
from gpu_fault.env import env_bool
from gpu_fault.completion_attempt_state import (
    AttemptSpec,
    cache_terminal_attempt_observation,
    publish_attempt_observation,
    restore_persisted_attempt_observations,
)
from gpu_fault.completion_metrics_server import start_completion_metrics_server
from gpu_fault.completion_observation import (
    TERMINATION_INCIDENT_ANNOTATION,
    MissingAttemptTracker,
    ObservationOnlyTracker,
    _clear_attempt,
    completion_list_arguments,
    is_unknown_profile_rejection,
    list_completion_pods,
    reconcile_attempt_observation,
)
from gpu_fault.completion_outbox import (
    completion_delivery_deferred,
    completion_sink_from_environment,
    replay_completion_outbox,
)
from gpu_fault.env_validation import validate_gpu_fault_environment
from gpu_fault.logging_setup import configure_logging
from gpu_fault.models import Environment
from gpu_fault.watcher import (
    AttemptObservation,
    CompletionWatcher,
    ContainerObservation,
    WorkloadPhase,
    failure_containment_ids,
)

LOGGER = logging.getLogger(__name__)
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
# One budget answers both "how long may the loop spend inside a single
# blocking step before it counts as stuck?" (the /healthz liveness window, F3)
# and "how long does the end of a watch cycle wait for a debounce timer that is
# still reconciling?" (F5). They have to be the same number: a join that
# outlives the liveness window would guarantee a restart every time it fired,
# and a window shorter than one legal delivery would kill a working watcher
# mid-pass.
#
# The longest legal blocking step is one delivery, and one delivery is: every
# HTTP attempt at its socket timeout, a capped ``Retry-After`` sleep between
# them, and then the processor receipt poll that follows the accepted POST.
# Deliveries are what stamp progress (see ``_ProgressStampingSink``), so this
# really is the largest gap a working loop can produce; the margin covers the
# non-blocking bookkeeping around it.
PROGRESS_BUDGET_MARGIN_SECONDS = 60.0
# A liveness window may not grow without limit: a node whose telemetry stops
# reads UNKNOWN after 600 s and every node-mutating plan is then BLOCKED, so a
# watcher that is really wedged has to be replaced well inside that horizon.
# Only the delivery term is capped -- ``PROGRESS_BUDGET_RELISTS`` below is a
# floor, and lowering it under one relist would fail a healthy watch.
MAX_DELIVERY_BUDGET_SECONDS = 480.0
# A stuck watch has also missed this many relists. The floor scales with the
# watch timeout so an operator who raises
# GPU_FAULT_WATCHER_WATCH_TIMEOUT_SECONDS does not create a restart loop.
PROGRESS_BUDGET_RELISTS = 3
# Used only when the sink hides its timings (a test double, a future sink).
DEFAULT_SINK_RECEIPT_TIMEOUT_SECONDS = 120.0
DEFAULT_SINK_HTTP_TIMEOUT_SECONDS = 10.0
DEFAULT_SINK_MAX_ATTEMPTS = 4.0
# The sink may be an outbox wrapping the HTTP sink that owns the timeouts.
MAX_SINK_CHAIN_DEPTH = 5


class CompletionControllerError(ValueError):
    pass


class _ProgressStampingSink:
    """Sink proxy that stamps loop progress after every delivery returns.

    The liveness budget is expressed in *one delivery*, and one delivery is the
    only thing this loop does that can legitimately block for minutes (HTTP
    attempts, a capped ``Retry-After`` sleep, then the processor receipt poll).
    Stamping at the reconcile or attempt boundary was not enough: the outbox
    replay delivers one record per buffered event before the first attempt even
    starts, and a single failing attempt delivers an observation, a
    failure-detected event and a terminal. Putting the stamp here makes "one
    unstamped gap = at most one delivery" true by construction, so no future
    call site can reintroduce the gap.

    Everything else is delegated untouched, including the depth gauges and the
    ``replay``/``stats`` helpers, so the proxy is invisible to its callers.
    """

    #: Delivery entry points: their return -- success or failure -- is a step.
    STAMPED_METHODS = frozenset({"post", "deliver"})

    def __init__(self, sink: Any, note_progress: Callable[[], None]) -> None:
        self._sink = sink
        self._note_progress = note_progress

    @property
    def wrapped_sink(self) -> Any:
        """The sink underneath, for tests and for identity checks."""

        return self._sink

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._sink, name)
        if name in self.STAMPED_METHODS and callable(value):
            return self._stamped(value)
        return value

    def _stamped(self, call: Callable[..., Any]) -> Callable[..., Any]:
        def stamped(*args: Any, **kwargs: Any) -> Any:
            try:
                return call(*args, **kwargs)
            finally:
                # ``finally``: a delivery that raised still proves the loop is
                # moving, and the retry path is the loop working (I2).
                self._note_progress()

        return stamped


def _stamping_sink(sink: Any, note_progress: Callable[[], None]) -> Any:
    """Wrap ``sink`` -- and the sink it wraps -- for progress stamping.

    ``KubernetesCompletionOutbox`` replays buffered records through the sink it
    holds, so that inner one is wrapped in place; otherwise a backlog replay is
    a single unstamped block at the top of every pass. A sink that will not
    accept the swap keeps the outer wrap alone, which is still correct for the
    live path.
    """

    inner = getattr(sink, "sink", None)
    if (
        inner is not None
        and not isinstance(inner, _ProgressStampingSink)
        and callable(getattr(inner, "post", None))
    ):
        try:
            sink.sink = _ProgressStampingSink(inner, note_progress)
        except (AttributeError, TypeError):
            LOGGER.warning(
                "cannot stamp progress on the buffered-record sink of %s; a "
                "long replay will not refresh the liveness clock",
                type(sink).__name__,
            )
    if isinstance(sink, _ProgressStampingSink):
        return sink
    return _ProgressStampingSink(sink, note_progress)


class KubernetesWorkloadStopper:
    """Emergency fallback when control-plane containment is unavailable."""

    def __init__(
        self,
        batch_api,
        custom_api,
        core_api=None,
        *,
        workload_log_tail_lines: int = 2000,
        workload_log_max_bytes: int = 262144,
        workload_log_s3_uri: str | None = None,
        workload_log_s3_max_bytes: int = 104857600,
        workload_log_annotation_tail_bytes: int = 8192,
        workload_log_uploader=None,
        workload_log_timeout_seconds: float = 30.0,
    ) -> None:
        if workload_log_timeout_seconds <= 0:
            raise CompletionControllerError(
                "workload log capture timeout must be positive"
            )
        self.batch = batch_api
        self.custom = custom_api
        self.core = core_api
        self.workload_log_timeout_seconds = workload_log_timeout_seconds
        self.workload_log_tail_lines = workload_log_tail_lines
        self.workload_log_max_bytes = workload_log_max_bytes
        uri = workload_log_s3_uri
        self.workload_log_s3_uri = uri.rstrip("/") if uri else None
        self.workload_log_s3_max_bytes = workload_log_s3_max_bytes
        self.workload_log_annotation_tail_bytes = workload_log_annotation_tail_bytes
        self.workload_log_uploader = workload_log_uploader

    def stop(
        self,
        workload_ids: tuple[str, ...],
        attempt_id: str,
        incident_id: str | None = None,
        *,
        capture_logs: bool = True,
    ) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        if incident_id is not None and self.core is not None:
            items = self._attempt_pods(attempt_id)
            if capture_logs:
                snapshots = self.capture_logs(
                    attempt_id,
                    incident_id,
                    pods=items,
                    annotate=False,
                )
            snapshots_by_uid = {
                str(snapshot.get("pod_uid")): snapshot
                for snapshot in snapshots
                if snapshot.get("pod_uid")
            }
            for pod in items:
                name, namespace, pod_uid = self._pod_identity(pod)
                if not name:
                    raise CompletionControllerError(
                        "managed Pod selected for emergency stop has no metadata.name"
                    )
                pod_annotations = {
                    TERMINATION_INCIDENT_ANNOTATION: incident_id,
                    "gpu-fault.io/passive-stop-attempt": attempt_id,
                    "gpu-fault.io/passive-stop-mode": ("emergency-fallback"),
                }
                snapshot = snapshots_by_uid.get(pod_uid)
                if snapshot is not None and snapshot.get("record_id"):
                    pod_annotations[WORKLOAD_LOG_SNAPSHOT_ANNOTATION] = (
                        self._snapshot_annotation(snapshot)
                    )
                self.core.patch_namespaced_pod(
                    name,
                    namespace,
                    {"metadata": {"annotations": pod_annotations}},
                )
        for workload_id in dict.fromkeys(workload_ids):
            namespace, kind, name = self._parse(workload_id)
            body: dict[str, Any] = {
                "metadata": {
                    "annotations": {
                        "gpu-fault.io/passive-stop-attempt": attempt_id,
                        "gpu-fault.io/passive-stop-mode": ("emergency-fallback"),
                    }
                },
                "spec": (
                    {"runPolicy": {"suspend": True}}
                    if kind == "pytorchjob"
                    else {"suspend": True}
                ),
            }
            if kind == "job":
                self.batch.patch_namespaced_job(name, namespace, body)
                continue
            group, version, plural = {
                "pytorchjob": (
                    "kubeflow.org",
                    "v1",
                    "pytorchjobs",
                ),
                "jobset": (
                    "jobset.x-k8s.io",
                    "v1alpha2",
                    "jobsets",
                ),
            }[kind]
            self.custom.patch_namespaced_custom_object(
                group, version, namespace, plural, name, body
            )
        return snapshots

    def _attempt_pods(self, attempt_id: str) -> list[Any]:
        listing = self.core.list_pod_for_all_namespaces(
            label_selector=(f"{MANAGED_LABEL}=true,{ATTEMPT_LABEL}={attempt_id}")
        )
        return (
            listing.get("items", [])
            if isinstance(listing, dict)
            else list(getattr(listing, "items", []))
        )

    @staticmethod
    def _pod_identity(pod: Any) -> tuple[str, str, str]:
        metadata = (
            pod.get("metadata", {})
            if isinstance(pod, dict)
            else getattr(pod, "metadata", None)
        )
        name = (
            metadata.get("name")
            if isinstance(metadata, dict)
            else getattr(metadata, "name", None)
        )
        namespace = (
            metadata.get("namespace", "default")
            if isinstance(metadata, dict)
            else getattr(metadata, "namespace", "default")
        )
        uid = (
            metadata.get("uid")
            if isinstance(metadata, dict)
            else getattr(metadata, "uid", None)
        )
        return str(name or ""), str(namespace or "default"), str(uid or name or "")

    def capture_logs(
        self,
        attempt_id: str,
        incident_id: str,
        *,
        pods: list[Any] | None = None,
        annotate: bool = True,
    ) -> list[dict[str, Any]]:
        if self.core is None:
            return []
        items = list(pods) if pods is not None else self._attempt_pods(attempt_id)
        if not items:
            return []

        def capture(pod: Any) -> tuple[Any, dict[str, Any]]:
            name, namespace, _pod_uid = self._pod_identity(pod)
            try:
                snapshot = self._capture_workload_log(
                    pod,
                    namespace=namespace,
                    attempt_id=attempt_id,
                    incident_id=incident_id,
                )
            except Exception as exc:
                LOGGER.exception(
                    "cannot capture failure workload log for %s/%s",
                    namespace,
                    name,
                )
                snapshot = {
                    "pod_name": name,
                    "namespace": namespace,
                    "capture_error": str(exc),
                }
            return pod, snapshot

        # Bounded: each read carries ``_request_timeout`` and the whole batch
        # gets one budget per wave of workers. A capture that has not finished
        # by then is recorded as an error and left to finish on its own thread;
        # the reconcile loop that called us must not wait for a hung kubelet.
        completed: list[tuple[Any, dict[str, Any]]] = []
        max_workers = min(8, len(items))
        budget = self.workload_log_timeout_seconds * math.ceil(len(items) / max_workers)
        executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="gpu-fault-workload-log",
        )
        futures = {executor.submit(capture, pod): pod for pod in items}
        try:
            for future in as_completed(futures, timeout=budget):
                completed.append(future.result())
        except TimeoutError:
            for future, pod in futures.items():
                if future.done():
                    continue
                name, namespace, _pod_uid = self._pod_identity(pod)
                LOGGER.error(
                    "workload log capture for %s/%s timed out after %ss",
                    namespace,
                    name,
                    budget,
                )
                completed.append(
                    (
                        pod,
                        {
                            "pod_name": name,
                            "namespace": namespace,
                            "capture_error": (
                                f"workload log capture timed out after {budget}s"
                            ),
                        },
                    )
                )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        snapshots = []
        for pod, snapshot in completed:
            snapshots.append(snapshot)
            if not annotate or not snapshot.get("record_id"):
                continue
            name, namespace, _pod_uid = self._pod_identity(pod)
            self.core.patch_namespaced_pod(
                name,
                namespace,
                {
                    "metadata": {
                        "annotations": {
                            WORKLOAD_LOG_SNAPSHOT_ANNOTATION: (
                                self._snapshot_annotation(snapshot)
                            )
                        }
                    }
                },
            )
        return sorted(
            snapshots,
            key=lambda item: (
                str(item.get("node_id") or ""),
                str(item.get("pod_uid") or item.get("pod_name") or ""),
            ),
        )

    def _snapshot_annotation(self, snapshot: dict[str, Any]) -> str:
        annotation_snapshot = dict(snapshot)
        tail = str(annotation_snapshot.get("tail") or "")
        annotation_snapshot["tail"] = tail.encode("utf-8")[
            -self.workload_log_annotation_tail_bytes :
        ].decode("utf-8", errors="replace")
        annotation_snapshot["annotation_tail_truncated"] = len(
            annotation_snapshot["tail"]
        ) < len(tail)
        return json.dumps(
            annotation_snapshot,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _capture_workload_log(
        self,
        pod: Any,
        *,
        namespace: str,
        attempt_id: str,
        incident_id: str,
    ) -> dict[str, Any]:
        metadata = (
            pod.get("metadata", {}) if isinstance(pod, dict) else pod.metadata.to_dict()
        )
        spec = pod.get("spec", {}) if isinstance(pod, dict) else pod.spec.to_dict()
        annotations = metadata.get("annotations") or {}
        pod_name = str(metadata.get("name") or "")
        pod_uid = str(metadata.get("uid") or pod_name)
        container_name = annotations.get(TRAINING_CONTAINER_ANNOTATION)
        containers = spec.get("containers") or []
        if not container_name and len(containers) == 1:
            container_name = containers[0].get("name")
        if not pod_name or not container_name:
            raise CompletionControllerError(
                "emergency log capture requires Pod and training container identity"
            )
        captured_at = datetime.now(timezone.utc)
        if self.workload_log_s3_uri:
            value = self.core.read_namespaced_pod_log(
                pod_name,
                namespace,
                container=container_name,
                timestamps=True,
                limit_bytes=self.workload_log_s3_max_bytes,
                _request_timeout=self.workload_log_timeout_seconds,
            )
            raw = str(value).encode("utf-8", errors="replace")
            archive_truncated = len(raw) >= self.workload_log_s3_max_bytes
        else:
            value = self.core.read_namespaced_pod_log(
                pod_name,
                namespace,
                container=container_name,
                timestamps=True,
                tail_lines=self.workload_log_tail_lines,
                _request_timeout=self.workload_log_timeout_seconds,
            )
            raw = str(value).encode("utf-8", errors="replace")
            archive_truncated = len(raw.splitlines()) >= self.workload_log_tail_lines
        tail = raw[-self.workload_log_max_bytes :]
        digest = hashlib.sha256(raw).hexdigest()
        record_hash = hashlib.sha256(
            (f"{incident_id}\x1f{pod_uid}\x1f{container_name}").encode()
        ).hexdigest()[:24]
        snapshot = {
            "record_id": f"workload-log/{record_hash}",
            "node_id": str(spec.get("nodeName") or "UNKNOWN"),
            "attempt_id": attempt_id,
            "namespace": namespace,
            "pod_name": pod_name,
            "pod_uid": pod_uid,
            "container_name": container_name,
            "captured_at": captured_at.isoformat(),
            "sha256": digest,
            "tail": tail.decode("utf-8", errors="replace"),
            "tail_bytes": len(tail),
            "tail_lines": self.workload_log_tail_lines,
            "truncated": (len(raw) > len(tail) or archive_truncated),
            "archive_truncated": archive_truncated,
            "s3_uri": None,
        }
        if self.workload_log_s3_uri:
            try:
                snapshot["s3_uri"] = self._upload_workload_log(
                    data=raw,
                    attempt_id=attempt_id,
                    incident_id=incident_id,
                    pod_uid=pod_uid,
                    container_name=str(container_name),
                    captured_at=captured_at,
                )
            except Exception as exc:
                LOGGER.exception(
                    "cannot archive workload log to S3 for %s/%s",
                    namespace,
                    pod_name,
                )
                snapshot["archive_error"] = f"{type(exc).__name__}: {exc}"
        return snapshot

    def _upload_workload_log(
        self,
        *,
        data: bytes,
        attempt_id: str,
        incident_id: str,
        pod_uid: str,
        container_name: str,
        captured_at: datetime,
    ) -> str:
        parsed = urlparse(self.workload_log_s3_uri or "")
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("GPU_FAULT_WORKLOAD_LOG_S3_URI must be an s3:// URI")
        safe_container = re.sub(r"[^A-Za-z0-9_.-]+", "-", container_name)
        suffix = (
            f"{attempt_id}/{incident_id}/{pod_uid}/"
            f"{safe_container}-{captured_at.strftime('%Y%m%dT%H%M%SZ')}"
            ".log.gz"
        )
        prefix = parsed.path.strip("/")
        key = f"{prefix}/{suffix}" if prefix else suffix
        destination = f"s3://{parsed.netloc}/{key}"
        compressed = gzip.compress(data)
        if self.workload_log_uploader is not None:
            return str(self.workload_log_uploader(destination, compressed))
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for workload log S3 archival"
            ) from exc
        boto3.client("s3").put_object(
            Bucket=parsed.netloc,
            Key=key,
            Body=compressed,
            ContentType="text/plain",
            ContentEncoding="gzip",
            Metadata={"sha256": hashlib.sha256(data).hexdigest()},
        )
        return destination

    @staticmethod
    def _parse(value: str) -> tuple[str, str, str]:
        parts = value.split("/")
        if len(parts) == 2:
            namespace, name = parts
            kind = "job"
        elif len(parts) == 3:
            namespace, kind, name = parts
            kind = kind.lower()
        else:
            raise CompletionControllerError(
                "workload ID must be namespace/name or "
                "namespace/{job|pytorchjob|jobset}/name"
            )
        if kind not in {"job", "pytorchjob", "jobset"} or not namespace or not name:
            raise CompletionControllerError(f"invalid workload ID: {value}")
        return namespace, kind, name


class KubernetesCompletionController:
    """Polls managed training Pods and emits durable attempt terminals."""

    _is_unknown_profile_rejection = staticmethod(is_unknown_profile_rejection)

    def __init__(
        self,
        core_api,
        sink: EventSink,
        *,
        cluster_id: str,
        environment: Environment = Environment.HYPERPOD_EKS,
        namespace: str | None = None,
        poll_interval_seconds: float = 5,
        watch_timeout_seconds: int = 30,
        cleanup_timeout_seconds: int = 120,
        now: Callable[[], datetime] | None = None,
        serializer: Callable[[Any], dict[str, Any]] | None = None,
        watch_factory: Callable[[], Any] | None = None,
        workload_stopper: Any | None = None,
        emergency_fallback_seconds: float | None = None,
        reconcile_debounce_seconds: float = 0.5,
        publish_observations: bool = False,
        observe_unmanaged_workloads: bool = False,
        observation_runtime_profile_version: str | None = None,
        observation_only_retention_cycles: int = 3,
        gpu_uuid_resolver: (Callable[[dict[str, Any], str], list[str]] | None) = None,
        terminal_retention_seconds: int = 3600,
        watcher_max_attempts: int = 10000,
    ) -> None:
        if not cluster_id:
            raise CompletionControllerError("cluster_id is required")
        if poll_interval_seconds <= 0:
            raise CompletionControllerError("poll interval must be positive")
        if watch_timeout_seconds < 1:
            raise CompletionControllerError("watch timeout must be positive")
        if emergency_fallback_seconds is not None and emergency_fallback_seconds < 1:
            raise CompletionControllerError(
                "emergency fallback must be at least one second"
            )
        if reconcile_debounce_seconds < 0:
            raise CompletionControllerError("reconcile debounce must be non-negative")
        self.core_api = core_api
        # Every delivery through this sink refreshes the liveness clock, so the
        # largest legitimate unstamped gap is one delivery -- which is exactly
        # what ``progress_stall_budget_seconds`` is derived from.
        self.sink = _stamping_sink(sink, self.note_progress)
        self.cluster_id = cluster_id
        self.environment = environment
        self.namespace = namespace
        self.poll_interval_seconds = poll_interval_seconds
        self.watch_timeout_seconds = watch_timeout_seconds
        self.cleanup_timeout_seconds = cleanup_timeout_seconds
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.serializer = serializer or (
            lambda value: value if isinstance(value, dict) else value.to_dict()
        )
        self.watch_factory = watch_factory
        self.workload_stopper = workload_stopper
        self.emergency_fallback_seconds = emergency_fallback_seconds
        self.reconcile_debounce_seconds = reconcile_debounce_seconds
        self.publish_observations = publish_observations
        self.observation_only = ObservationOnlyTracker(
            observe_unmanaged_workloads,
            observation_runtime_profile_version,
            observation_only_retention_cycles,
        )
        self.gpu_uuid_resolver = gpu_uuid_resolver
        try:
            self.watcher = CompletionWatcher(
                terminal_retention_seconds=terminal_retention_seconds,
                max_attempts=watcher_max_attempts,
            )
        except ValueError as exc:
            raise CompletionControllerError(str(exc)) from exc
        self._attempt_specs: dict[str, AttemptSpec] = {}
        self._last_observations: dict[str, AttemptObservation] = {}
        self._terminal_observations: dict[str, AttemptObservation] = {}
        self._missing_attempts = MissingAttemptTracker()
        self._terminal_sent: set[str] = set()
        self._workloads_stopped: set[str] = set()
        self._failure_events = {}
        self._failure_sent: set[str] = set()
        self._failure_delivery_started: dict[str, datetime] = {}
        self._emergency_incident_ids: dict[str, str] = {}
        self._gpu_uuid_cache: dict[str, list[str]] = {}
        self._gpu_uuid_failures: dict[str, tuple[int, datetime]] = {}
        self.metadata_takeovers_total = 0
        self.reconcile_runs_total = 0
        self.reconciled_attempts_total = 0
        # One attempt's reconcile raising is logged and counted here; it no
        # longer aborts the pass for every other attempt (F-G4 / P1-47A).
        self.reconcile_failures_total = 0
        # Attempts whose controller-side state was dropped because the watcher
        # core pruned them and no Pod of theirs is left (F-G4 / P1-50C).
        self.evicted_attempts_total = 0
        # Persisted attempt records skipped at start-up (set by the restore).
        self.restore_skipped_total = 0
        # Liveness (F3/C1): ``last_progress_at`` is what /healthz judges, and
        # every step the loop finishes moves it -- a relist, one attempt's
        # reconcile, one watch event, even a failed cycle that went back to the
        # retry sleep. A hung watch stream or a parked LIST finishes nothing, so
        # it stops moving, while a slow-but-working pass and an idle cluster
        # both keep it fresh. ``last_cycle_completed_at`` is deliberately *not*
        # the probe's signal: one full pass can legitimately outlast several
        # watch timeouts. It stays as the metric humans alert on.
        self.started_at: datetime = self.now()
        self.last_progress_at: datetime = self.started_at
        self.last_cycle_completed_at: datetime | None = None
        # One reconcile at a time per process (F5). The lock used to be a
        # per-cycle local, so a debounce timer still inside ``_reconcile`` ran
        # concurrently with the next cycle's full pass. It is an ``RLock`` so
        # the inline zero-debounce flush on the watch thread can re-enter.
        self.reconcile_lock = RLock()
        try:
            restore_persisted_attempt_observations(self)
        except ValueError as exc:
            raise CompletionControllerError(
                "cannot restore persisted Completion Watcher attempt state"
            ) from exc

    @property
    def outbox_append_failures_total(self) -> int:
        """Write-ahead failures counted by the outbox sink (F1/F12).

        The sink owns the counter, but ``/metrics`` scrapes the controller, so
        it is republished here. A sink without an outbox (a plain HTTP sink, a
        test double) reports zero.
        """

        return int(getattr(self.sink, "append_failures_total", 0))

    @property
    def outbox_depth(self) -> int:
        """Buffered completion records as of the last replay pass (F12).

        Every record in the ConfigMap, not only the critical events: the
        latest undelivered workload observation of an attempt is one too.
        """

        return int(getattr(self.sink, "last_depth", 0))

    @property
    def outbox_quarantined_depth(self) -> int:
        """Buffered records no replay will retry again (F12).

        ``replay`` skips a quarantined record and no production caller passes
        ``include_quarantined``, so anything above zero here is waiting on the
        live path or on an operator, never on the loop.
        """

        return int(getattr(self.sink, "last_quarantined_depth", 0))

    def _sink_timing(self, attribute: str, default: float) -> float:
        """A delivery timeout read off the sink, or off the sink it wraps.

        ``KubernetesCompletionOutbox`` fronts the HTTP sink that owns the
        timeouts, so the value has to be looked up down the chain. Anything
        unreadable falls back to the shipped default: this feeds a liveness
        window, so it must never raise and never return zero.
        """

        sink: Any = self.sink
        for _ in range(MAX_SINK_CHAIN_DEPTH):
            if sink is None:
                break
            value = getattr(sink, attribute, None)
            try:
                if value is not None and float(value) > 0:
                    return float(value)
            except (TypeError, ValueError):
                pass
            sink = getattr(sink, "sink", None)
        return default

    @property
    def progress_stall_budget_seconds(self) -> float:
        """How long the loop may finish nothing before it counts as stuck.

        Derived, not configured: one whole delivery (every HTTP attempt plus
        the processor receipt poll) with a margin, and never less than
        ``PROGRESS_BUDGET_RELISTS`` watch timeouts. ``/healthz`` and the timer
        join in ``run_watch_cycle`` both read it, so a join that fires can
        never by itself push the Pod past the liveness window.
        """

        receipt = self._sink_timing(
            "processor_receipt_timeout_seconds",
            DEFAULT_SINK_RECEIPT_TIMEOUT_SECONDS,
        )
        http_timeout = self._sink_timing(
            "timeout_seconds", DEFAULT_SINK_HTTP_TIMEOUT_SECONDS
        )
        attempts = self._sink_timing("max_attempts", DEFAULT_SINK_MAX_ATTEMPTS)
        delivery = (
            receipt
            + http_timeout * attempts
            + RETRY_AFTER_CAP_SECONDS * max(attempts - 1.0, 0.0)
            + PROGRESS_BUDGET_MARGIN_SECONDS
        )
        relists = PROGRESS_BUDGET_RELISTS * float(self.watch_timeout_seconds)
        return max(min(delivery, MAX_DELIVERY_BUDGET_SECONDS), relists)

    def note_progress(self) -> None:
        """Record that the loop just finished a step (the liveness signal).

        Called around every blocking step rather than once per pass, so a slow
        pass reads as alive and only a step that never returns reads as stuck.
        """

        self.last_progress_at = self.now()

    def run_once(self) -> list[dict[str, Any]]:
        pods, _ = list_completion_pods(
            self.core_api,
            self.namespace,
            self.observation_only.enabled,
        )
        self.note_progress()
        return self._reconcile(pods)

    def _reconcile(
        self,
        pods: list[dict[str, Any]],
        *,
        attempt_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """One reconcile pass, serialized against every other pass (F5).

        The lock lives here rather than at the call sites so every entry point
        -- the cycle's full pass, a debounce flush on a timer thread, the
        polling fallback's ``run_once`` -- is covered structurally and a new
        caller cannot forget it. It is an ``RLock``, so a nested pass on the
        same thread still works.
        """

        with self.reconcile_lock:
            return self._reconcile_pass(pods, attempt_filter=attempt_filter)

    def _reconcile_pass(
        self,
        pods: list[dict[str, Any]],
        *,
        attempt_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        replay_completion_outbox(self.sink, LOGGER)
        grouped = self.observation_only.group(self, pods, self.serializer)

        if attempt_filter is None:
            active_pod_keys = {
                self._pod_key(pod)
                for attempt_pods in grouped.values()
                for pod in attempt_pods
            }
            self._gpu_uuid_cache = {
                key: value
                for key, value in self._gpu_uuid_cache.items()
                if key in active_pod_keys
            }
            self._gpu_uuid_failures = {
                key: value
                for key, value in self._gpu_uuid_failures.items()
                if key in active_pod_keys
            }
        results: list[dict[str, Any]] = []
        attempt_ids = (
            set(grouped).union(self._attempt_specs)
            if attempt_filter is None
            else set(attempt_filter)
        )
        self.reconcile_runs_total += 1
        self.reconciled_attempts_total += len(attempt_ids)
        observed_at = self.now()
        for attempt_id in sorted(attempt_ids):
            attempt_pods = grouped.get(attempt_id, [])
            # The whole body is isolated, not just the observation step: a
            # failure anywhere in one attempt's handling used to abort the
            # pass for every attempt after it in sort order (P1-47A).
            # Liveness (C1): one attempt is the smallest unit the pass can
            # finish, and handling it can block for a whole delivery, so the
            # stamp goes on both sides of it. Stamping once per pass instead
            # made a storm of terminals look like a hung loop.
            self.note_progress()
            try:
                self._reconcile_attempt(attempt_id, attempt_pods, observed_at, results)
            except Exception:
                self.reconcile_failures_total += 1
                LOGGER.exception("cannot reconcile attempt %s", attempt_id)
            self.note_progress()
        if attempt_filter is None:
            self._evict_pruned_attempts(grouped)
            # Not the liveness signal (see ``last_progress_at``): this is the
            # value humans alert on, and only a pass that relisted every Pod
            # moves it.
            self.last_cycle_completed_at = self.now()
        return results

    def _evict_pruned_attempts(self, grouped: dict[str, list[dict[str, Any]]]) -> None:
        """Drop controller state for attempts the watcher core has pruned.

        Only attempts with no live Pod are evicted: one whose Pods are still
        listed is rebuilt from them on the next pass anyway, and dropping its
        sent-keys would re-post its terminal. The watcher accumulates pruned
        ids across filtered passes; a full pass drains them.
        """
        for attempt_id in sorted(self.watcher.take_pruned_attempt_ids()):
            if attempt_id in grouped:
                continue
            _clear_attempt(self, attempt_id)
            self.evicted_attempts_total += 1
            LOGGER.info(
                "evicted terminal attempt state after retention: attempt=%s",
                attempt_id,
            )

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
        except (CompletionControllerError, ValueError):
            self.reconcile_failures_total += 1
            LOGGER.exception("cannot reconcile attempt %s", attempt_id)
            return
        if self.publish_observations:
            publish_attempt_observation(self, observation, attempt_id)
        if self.observation_only.contains(attempt_id):
            results.append(
                {
                    "attempt_id": attempt_id,
                    "observation_only": True,
                }
            )
            return
        if result.failure_detected is not None:
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
        if result.failure_detected is not None:
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
                # delivered, so it must not be recorded as sent: replay
                # quarantines a record after ``max_replay_attempts`` and then
                # never touches it again, and the live path is what has to pick
                # it up from there. The outbox suppresses the duplicate live
                # POST for as long as the record really is replay's.
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

    def run(self, *, metrics_port: int | None = None) -> None:
        """Serve ``/metrics`` + ``/healthz`` and loop until the process ends.

        ``metrics_port`` overrides the environment setting; ``0`` disables the
        server, which the Deployment must never do -- the liveness probe reads
        that endpoint, so no server means kubelet keeps restarting the Pod.
        """

        metrics_server = start_completion_metrics_server(self, port=metrics_port)
        try:
            self._run_forever()
        finally:
            if metrics_server is not None:
                metrics_server.stop()

    def _run_forever(self) -> None:
        if self.watch_factory is None:
            LOGGER.warning("watch client is unavailable; using polling fallback")
            self._run_polling()
            return

        while True:
            # Reaching the top of the loop is progress, and so is coming back
            # from a failure (I2): an API server that refuses every LIST keeps
            # the loop turning, and CrashLooping the only Completion Watcher
            # during a control-plane outage would only add a cold start to it.
            self.note_progress()
            try:
                self.run_watch_cycle()
            except Exception:
                LOGGER.exception("Kubernetes Pod watch cycle failed; retrying")
                self.note_progress()
                time.sleep(self.poll_interval_seconds)

    def _run_polling(self) -> None:
        while True:
            self.note_progress()
            try:
                self.run_once()
            except Exception:
                LOGGER.exception("Kubernetes completion reconciliation failed")
            self.note_progress()
            time.sleep(self.poll_interval_seconds)

    def run_watch_cycle(self) -> None:
        """One list + watch + reconcile cycle: the loop's unit of work.

        ``run()`` never returns and ``run_once()`` only covers the polling
        fallback, so this is the public seam for anything that wants a single
        watch cycle -- a supervisor, an operator one-shot, or a test that has
        to observe what the cycle's ``finally`` does.
        """

        pods, resource_version = list_completion_pods(
            self.core_api,
            self.namespace,
            self.observation_only.enabled,
        )
        self.note_progress()
        serialized = [self.serializer(item) for item in pods]
        cache = {self._pod_key(pod): pod for pod in serialized}
        # ``_reconcile`` takes ``reconcile_lock`` itself (F5), which is what
        # keeps a timer that outlived its cycle from running beside this pass.
        self._reconcile(list(cache.values()))
        cache_lock = RLock()
        pending_lock = RLock()
        pending_attempts: set[str] = set()
        pending_timer: list[Timer | None] = [None]
        # Timers stay reachable after they have fired so the ``finally`` can
        # join a flush that is still running; ``pending_timer`` is cleared by
        # the flush itself so ``schedule`` can arm the next one, which means a
        # slow flush and a freshly armed timer can both be live (the reconcile
        # lock serializes them). Finished timers are dropped on every arm, so
        # the list holds at most the live ones.
        started_timers: list[Timer] = []

        def attempt_id(pod: dict[str, Any] | None) -> str | None:
            if not pod:
                return None
            metadata = pod.get("metadata") or {}
            labels = metadata.get("labels") or {}
            value = labels.get(ATTEMPT_LABEL)
            return value if isinstance(value, str) and value else None

        def flush_pending() -> None:
            with pending_lock:
                attempts = set(pending_attempts)
                pending_attempts.clear()
                pending_timer[0] = None
            if not attempts:
                return
            with cache_lock:
                affected_pods = [
                    pod for pod in cache.values() if attempt_id(pod) in attempts
                ]
            self._reconcile(
                affected_pods,
                attempt_filter=attempts,
            )

        def schedule(attempts: set[str]) -> None:
            if not attempts:
                return
            with pending_lock:
                pending_attempts.update(attempts)
                if pending_timer[0] is not None:
                    return
                if self.reconcile_debounce_seconds == 0:
                    pass
                else:
                    timer = Timer(
                        self.reconcile_debounce_seconds,
                        flush_pending,
                    )
                    timer.daemon = True
                    pending_timer[0] = timer
                    started_timers[:] = [
                        item for item in started_timers if item.is_alive()
                    ]
                    started_timers.append(timer)
                    timer.start()
                    return
            flush_pending()

        watcher = self.watch_factory()
        LOGGER.info(
            "starting Kubernetes Pod watch: resource_version=%s cached_pods=%s",
            resource_version or "current",
            len(cache),
        )
        try:
            try:
                for event in watcher.stream(
                    self._list_method(),
                    **completion_list_arguments(
                        self.namespace,
                        self.observation_only.enabled,
                    ),
                    resource_version=resource_version or None,
                    timeout_seconds=self.watch_timeout_seconds,
                    allow_watch_bookmarks=True,
                    # F3: without this the client leaves ``timeout=None`` and a
                    # stream the API server drops without an RST (dead
                    # endpoint, NAT idle eviction) parks this thread for ever:
                    # no relist, no reconcile, and every node UNKNOWN after
                    # 600 s. The read budget sits above the server-side
                    # ``timeout_seconds`` so a healthy stream always ends by
                    # the server closing it, never by the client.
                    _request_timeout=(5, self.watch_timeout_seconds + 15),
                ):
                    event_type = str(event.get("type", "")).upper()
                    raw_object = event.get("object")
                    pod = self.serializer(raw_object) if raw_object is not None else {}
                    if event_type == "ERROR":
                        code = pod.get("code")
                        if code == 410:
                            LOGGER.info("Pod watch resourceVersion expired; relisting")
                            return
                        raise CompletionControllerError(
                            f"Kubernetes Pod watch error: {pod}"
                        )
                    if event_type == "BOOKMARK":
                        continue
                    if event_type not in {
                        "ADDED",
                        "MODIFIED",
                        "DELETED",
                    }:
                        LOGGER.warning(
                            "ignoring unknown Kubernetes watch event %s",
                            event_type,
                        )
                        continue
                    # An event delivered and applied is progress: a busy
                    # stream keeps the probe green between relists.
                    self.note_progress()
                    key = self._pod_key(pod)
                    with cache_lock:
                        previous = cache.get(key)
                        if event_type == "DELETED":
                            cache.pop(key, None)
                            self._gpu_uuid_cache.pop(key, None)
                            self._gpu_uuid_failures.pop(key, None)
                        else:
                            cache[key] = pod
                    schedule(
                        {
                            value
                            for value in (
                                attempt_id(previous),
                                attempt_id(pod),
                            )
                            if value is not None
                        }
                    )
            except Exception as exc:
                if getattr(exc, "status", None) == 410:
                    LOGGER.info("Pod watch resourceVersion expired; relisting")
                    return
                raise
        finally:
            with pending_lock:
                timer = pending_timer[0]
                if timer is not None:
                    timer.cancel()
                    pending_timer[0] = None
                in_flight = [item for item in started_timers if item.is_alive()]
                started_timers.clear()
            # Join OUTSIDE every lock (F5). A timer that already fired holds no
            # lock this thread wants, and this thread holds none it wants, so
            # the join cannot deadlock; taking ``reconcile_lock`` here would.
            join_timeout = self.progress_stall_budget_seconds
            for item in in_flight:
                if item is current_thread():
                    continue
                item.join(timeout=join_timeout)
                if item.is_alive():
                    LOGGER.warning(
                        "debounced reconcile still running after %ss; the next "
                        "full pass will wait on the reconcile lock",
                        join_timeout,
                    )
            # F11: the flush is the only step here that can raise (one bad Pod
            # is enough), and skipping ``stop()`` leaked the API-server stream
            # and its thread on every such cycle.
            try:
                flush_pending()
            finally:
                watcher.stop()
        LOGGER.info("Kubernetes Pod watch timed out; resyncing")

    def _list_method(self):
        if self.namespace:
            return self.core_api.list_namespaced_pod
        return self.core_api.list_pod_for_all_namespaces

    @staticmethod
    def _pod_key(pod: dict[str, Any]) -> str:
        metadata = pod.get("metadata") or {}
        uid = metadata.get("uid")
        if uid:
            return uid
        namespace = metadata.get("namespace", "default")
        name = metadata.get("name")
        if not name:
            raise CompletionControllerError("watched Pod requires UID or name")
        return f"{namespace}/{name}"

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

        containers = [self._container(pod) for pod in pods if self._is_critical(pod)]
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
        elif containers:
            phase = WorkloadPhase.RUNNING
        else:
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
        self.watcher.reset_attempt(attempt_id)
        self._missing_attempts.clear(attempt_id)
        self._last_observations.pop(attempt_id, None)
        self._terminal_observations.pop(attempt_id, None)
        self._failure_events.pop(attempt_id, None)
        self._failure_sent.discard(attempt_id)
        self._failure_delivery_started.pop(attempt_id, None)
        self._workloads_stopped.discard(attempt_id)
        self._emergency_incident_ids.pop(attempt_id, None)
        terminal_key_fragment = f"/{attempt_id}/"
        self._terminal_sent = {
            key for key in self._terminal_sent if terminal_key_fragment not in key
        }

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

    @staticmethod
    def _gpu_uuids(value: str | None) -> list[str]:
        return KubernetesCompletionController._string_list(value, "GPU UUID annotation")

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


def controller_from_environment() -> KubernetesCompletionController:
    try:
        from kubernetes import client, config, watch
        from kubernetes.config.config_exception import ConfigException
        from kubernetes.stream import stream
    except ImportError as exc:
        raise RuntimeError("install gpu-fault-control-plane[collectors]") from exc
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()
    api_client = client.ApiClient()
    core_api = client.CoreV1Api()

    def discover_gpu_uuids(pod: dict[str, Any], container_name: str) -> list[str]:
        metadata = pod.get("metadata") or {}
        output = stream(
            core_api.connect_get_namespaced_pod_exec,
            metadata["name"],
            metadata.get("namespace") or "default",
            container=container_name,
            command=[
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            stderr=False,
            stdin=False,
            stdout=True,
            tty=False,
            _request_timeout=10,
        )
        return [line.strip() for line in output.splitlines() if line.strip()]

    resolver = (
        discover_gpu_uuids
        if env_bool("GPU_FAULT_DISCOVER_POD_GPU_UUIDS", True)
        else None
    )
    fallback_value = os.getenv("GPU_FAULT_PASSIVE_STOP_FALLBACK_SECONDS", "30").strip()
    fallback_seconds = float(fallback_value) if fallback_value else None
    return KubernetesCompletionController(
        core_api,
        completion_sink_from_environment(core_api),
        cluster_id=os.environ["GPU_FAULT_CLUSTER_ID"],
        environment=Environment(
            os.getenv(
                "GPU_FAULT_WATCHER_ENVIRONMENT",
                Environment.HYPERPOD_EKS.value,
            )
        ),
        namespace=os.getenv("GPU_FAULT_WATCH_NAMESPACE") or None,
        poll_interval_seconds=float(os.getenv("GPU_FAULT_WATCHER_POLL_SECONDS", "5")),
        watch_timeout_seconds=int(
            os.getenv("GPU_FAULT_WATCHER_WATCH_TIMEOUT_SECONDS", "30")
        ),
        cleanup_timeout_seconds=int(
            os.getenv(
                "GPU_FAULT_WATCHER_CLEANUP_TIMEOUT_SECONDS",
                "120",
            )
        ),
        serializer=api_client.sanitize_for_serialization,
        watch_factory=watch.Watch,
        workload_stopper=KubernetesWorkloadStopper(
            client.BatchV1Api(),
            client.CustomObjectsApi(),
            core_api,
            workload_log_tail_lines=int(
                os.getenv("GPU_FAULT_WORKLOAD_LOG_TAIL_LINES", "2000")
            ),
            workload_log_max_bytes=int(
                os.getenv("GPU_FAULT_WORKLOAD_LOG_MAX_BYTES", "262144")
            ),
            workload_log_s3_uri=(os.getenv("GPU_FAULT_WORKLOAD_LOG_S3_URI") or None),
            workload_log_s3_max_bytes=int(
                os.getenv(
                    "GPU_FAULT_WORKLOAD_LOG_S3_MAX_BYTES",
                    "104857600",
                )
            ),
            workload_log_annotation_tail_bytes=int(
                os.getenv(
                    "GPU_FAULT_WORKLOAD_LOG_ANNOTATION_TAIL_BYTES",
                    "8192",
                )
            ),
            workload_log_timeout_seconds=float(
                os.getenv("GPU_FAULT_WORKLOAD_LOG_TIMEOUT_SECONDS", "30")
            ),
        ),
        terminal_retention_seconds=int(
            os.getenv("GPU_FAULT_WATCHER_TERMINAL_RETENTION_SECONDS", "3600")
        ),
        emergency_fallback_seconds=fallback_seconds,
        reconcile_debounce_seconds=float(
            os.getenv(
                "GPU_FAULT_WATCHER_RECONCILE_DEBOUNCE_SECONDS",
                "0.5",
            )
        ),
        publish_observations=env_bool("GPU_FAULT_PUBLISH_WORKLOAD_OBSERVATIONS", True),
        observe_unmanaged_workloads=env_bool(
            "GPU_FAULT_COMPLETION_OBSERVE_UNMANAGED", False
        ),
        observation_runtime_profile_version=(
            os.getenv("GPU_FAULT_COMPLETION_OBSERVATION_RUNTIME_PROFILE") or None
        ),
        observation_only_retention_cycles=int(
            os.getenv(
                "GPU_FAULT_COMPLETION_OBSERVATION_RETENTION_CYCLES",
                "3",
            )
        ),
        gpu_uuid_resolver=resolver,
    )


def main() -> None:
    configure_logging()
    validate_gpu_fault_environment(process_name="gpu-fault-completion-watcher")
    controller_from_environment().run()


if __name__ == "__main__":
    main()
