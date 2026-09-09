from __future__ import annotations

import logging
import os
import socket
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable

from gpu_fault.collectors import EventSink
from gpu_fault.completion_attempt_state import (
    AttemptSpec,
    active_pass_counts,
    evict_pruned_attempts,
    log_reconcile_failure,
    publish_coverage_heartbeat,
    restore_persisted_attempt_observations,
)
from gpu_fault.completion_delivery import CompletionDeliveryMixin
from gpu_fault.completion_liveness import (
    DEFAULT_SINK_HTTP_TIMEOUT_SECONDS,
    DEFAULT_SINK_MAX_ATTEMPTS,
    DEFAULT_SINK_RECEIPT_TIMEOUT_SECONDS,
    MAX_DELIVERY_BUDGET_SECONDS,
    MAX_SINK_CHAIN_DEPTH,
    PROGRESS_BUDGET_MARGIN_SECONDS,
    PROGRESS_BUDGET_RELISTS,
    CompletionLivenessMixin,
    _ProgressStampingSink,
    _stamping_sink,
)
from gpu_fault.completion_observation import (
    MissingAttemptTracker,
    TerminalObservationCache,
    group_completion_pods,
    is_unknown_profile_rejection,
    list_completion_pods,
)
from gpu_fault.completion_outbox import (
    completion_sink_from_environment,
    replay_completion_outbox,
    replay_quarantined_once,
    replay_quarantined_requested,
)
from gpu_fault.completion_pod_parsing import (
    ATTEMPT_LABEL,
    CGROUP_PATH_ANNOTATION,
    CHECKPOINT_ANNOTATION,
    CRITICAL_LABEL,
    EXPECTED_RANKS_ANNOTATION,
    GPU_UUIDS_ANNOTATION,
    HOST_PID_ANNOTATION,
    INDEXED_JOB_RANK,
    JOB_LABEL,
    JOBSET_JOB_INDEX,
    JOBSET_NAME_LABEL,
    KUBEFLOW_REPLICA_INDEX,
    MANAGED_LABEL,
    PYTORCH_JOB_LABELS,
    RANK_ANNOTATION,
    RANK_JOB_STRIDE_ANNOTATION,
    RANK_OFFSET_ANNOTATION,
    RESTART_BUDGET_ANNOTATION,
    ROLE_LABEL,
    RUNTIME_PROFILE_ANNOTATION,
    TRAINING_CONTAINER_ANNOTATION,
    WORKLOAD_IDS_ANNOTATION,
    WORKLOAD_LOG_SNAPSHOT_ANNOTATION,
    CompletionControllerError,
    CompletionPodParsingMixin,
)
from gpu_fault.completion_reconcile_loop import CompletionReconcileLoopMixin
from gpu_fault.completion_workload_stopper import KubernetesWorkloadStopper
from gpu_fault.env import env_bool
from gpu_fault.env_validation import validate_gpu_fault_environment
from gpu_fault.logging_setup import configure_logging
from gpu_fault.models import Environment
from gpu_fault.watcher import AttemptObservation, CompletionWatcher

# Every name this module used to define stays importable from here: the
# console script ``gpu-fault-completion-watcher`` is ``completion_controller:main``,
# and the split modules (F6b) are an implementation detail of this one.
__all__ = [
    "ATTEMPT_LABEL",
    "CGROUP_PATH_ANNOTATION",
    "CHECKPOINT_ANNOTATION",
    "CRITICAL_LABEL",
    "DEFAULT_SINK_HTTP_TIMEOUT_SECONDS",
    "DEFAULT_SINK_MAX_ATTEMPTS",
    "DEFAULT_SINK_RECEIPT_TIMEOUT_SECONDS",
    "EXPECTED_RANKS_ANNOTATION",
    "GPU_UUIDS_ANNOTATION",
    "HOST_PID_ANNOTATION",
    "INDEXED_JOB_RANK",
    "JOB_LABEL",
    "JOBSET_JOB_INDEX",
    "JOBSET_NAME_LABEL",
    "KUBEFLOW_REPLICA_INDEX",
    "LOGGER",
    "MANAGED_LABEL",
    "MAX_DELIVERY_BUDGET_SECONDS",
    "MAX_SINK_CHAIN_DEPTH",
    "PROGRESS_BUDGET_MARGIN_SECONDS",
    "PROGRESS_BUDGET_RELISTS",
    "PYTORCH_JOB_LABELS",
    "RANK_ANNOTATION",
    "RANK_JOB_STRIDE_ANNOTATION",
    "RANK_OFFSET_ANNOTATION",
    "RESTART_BUDGET_ANNOTATION",
    "ROLE_LABEL",
    "RUNTIME_PROFILE_ANNOTATION",
    "TRAINING_CONTAINER_ANNOTATION",
    "WORKLOAD_IDS_ANNOTATION",
    "WORKLOAD_LOG_SNAPSHOT_ANNOTATION",
    "CompletionControllerError",
    "KubernetesCompletionController",
    "KubernetesWorkloadStopper",
    "_ProgressStampingSink",
    "_stamping_sink",
    "controller_from_environment",
    "main",
]

LOGGER = logging.getLogger(__name__)


class KubernetesCompletionController(
    CompletionPodParsingMixin,
    CompletionDeliveryMixin,
    CompletionReconcileLoopMixin,
    CompletionLivenessMixin,
):
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
        attempt_missing_grace_seconds: int = 300,
        now: Callable[[], datetime] | None = None,
        serializer: Callable[[Any], dict[str, Any]] | None = None,
        watch_factory: Callable[[], Any] | None = None,
        workload_stopper: Any | None = None,
        emergency_fallback_seconds: float | None = None,
        reconcile_debounce_seconds: float = 0.5,
        publish_observations: bool = False,
        gpu_uuid_resolver: (Callable[[dict[str, Any], str], list[str]] | None) = None,
        terminal_retention_seconds: int = 3600,
        watcher_max_attempts: int = 10000,
        watcher_instance: str | None = None,
        coverage_heartbeat_interval_seconds: float = 120,
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
        if coverage_heartbeat_interval_seconds < 0:
            raise CompletionControllerError(
                "coverage heartbeat interval must be non-negative"
            )
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
        # How long every Pod of an attempt may be absent before the attempt is
        # tombstoned. Minutes, and separate from ``cleanup_timeout_seconds``:
        # see ``MissingAttemptTracker`` (F2).
        self.attempt_missing_grace_seconds = attempt_missing_grace_seconds
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.serializer = serializer or (
            lambda value: value if isinstance(value, dict) else value.to_dict()
        )
        self.watch_factory = watch_factory
        self.workload_stopper = workload_stopper
        if workload_stopper is not None and hasattr(workload_stopper, "capture_logs"):
            workload_stopper.note_progress = self.note_progress
        self.emergency_fallback_seconds = emergency_fallback_seconds
        self.reconcile_debounce_seconds = reconcile_debounce_seconds
        self.publish_observations = publish_observations
        # Names the process in its coverage heartbeats so an operator can
        # tell a heartbeat from a watcher a rollout has replaced from a
        # current one. Never read by the resolver.
        self.watcher_instance = watcher_instance or socket.gethostname()
        # How often an idle cluster's coverage may be restated. A fifth of the
        # control plane's default coverage window (600 s), so four heartbeats
        # can be lost before the cluster reads UNKNOWN, while the strictly
        # ordered ingress lane this shares with workflow dispatch carries one
        # write every two minutes instead of one per pass.
        self.coverage_heartbeat_interval_seconds = coverage_heartbeat_interval_seconds
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
        self._terminal_observations = TerminalObservationCache()
        # Attempts rebuilt from persisted state whose Pods this process has
        # never seen (F7); the restore fills it.
        self._restored_attempts: set[str] = set()
        # Attempts whose SUCCEEDED/FAILED came from the workload object rather
        # than from a Pod: the terminal is published, but the failure never
        # reaches containment, because there is no rank, node or exit code to
        # attribute and the workload IDs outlive the attempt (C1).
        self._synthesized_verdicts: set[str] = set()
        try:
            self._missing_attempts = MissingAttemptTracker(
                attempt_missing_grace_seconds
            )
        except ValueError as exc:
            raise CompletionControllerError(str(exc)) from exc
        self._terminal_sent: set[str] = set()
        self._workloads_stopped: set[str] = set()
        self._failure_events = {}
        self._failure_sent: set[str] = set()
        self._failure_delivery_started: dict[str, datetime] = {}
        self._emergency_incident_ids: dict[str, str] = {}
        self._gpu_uuid_cache: dict[str, list[str]] = {}
        self._gpu_uuid_failures: dict[str, tuple[int, datetime]] = {}
        self.metadata_takeovers_total = 0
        # Attempts whose missing-Pod tombstone was withdrawn because Pods of
        # that attempt-id came back (F2).
        self.resumed_attempts_total = 0
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
        # Coverage heartbeats (F4): an idle cluster publishes no observation,
        # so a completed full pass says so explicitly and the control plane can
        # answer IDLE instead of UNKNOWN. Weak evidence by design: it rides the
        # ordinary sink, never the critical outbox, and a lost one is counted
        # and forgotten because the next pass restates it.
        self.coverage_heartbeats_total = 0
        self.coverage_heartbeat_failures_total = 0
        self.last_coverage_heartbeat_at: datetime | None = None
        # When a heartbeat was last *attempted*, which is what the rate limit
        # counts: a refusing control plane must not be asked once per pass.
        self._last_coverage_post_at: datetime | None = None
        # A namespace-scoped watcher cannot vouch for a cluster, and says so
        # once rather than on every pass.
        self._coverage_scope_logged = False
        # The list revision the most recent relist started from; operator
        # context carried in the heartbeat, not an input to any decision.
        self._last_resource_version: str | None = None
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
    def active_state_unavailable(self) -> int:
        """0/1 from the sink: is the attempt-state ConfigMap refusing (I1)?

        Republished here because ``/metrics`` scrapes the controller; the gauge's
        HELP text carries the consequence. A sink without an outbox reports 0.
        """

        return int(getattr(self.sink, "active_state_unavailable", 0))

    @property
    def outbox_depth(self) -> int:
        """Buffered completion records as of the last replay pass (F12).

        Every record in the ConfigMap, not only the critical events: the
        latest undelivered workload observation of an attempt is one too.
        """

        return int(getattr(self.sink, "last_depth", 0))

    @property
    def outbox_quarantined_depth(self) -> int:
        """Buffered records the loop's replay will not retry again (F12).

        Anything above zero is waiting on the live path or on the operator's
        ``--replay-quarantined`` one-shot, never on the loop.
        """

        return int(getattr(self.sink, "last_quarantined_depth", 0))

    @property
    def outbox_expired_total(self) -> int:
        """Retry records removed after the ERROR that names them, 24 h after
        they were buffered (final review C1, R5) -- never a rejected one."""

        return int(getattr(self.sink, "expired_total", 0))

    @property
    def outbox_quarantine_evictions_total(self) -> int:
        """Quarantined records evicted to make room for a critical event (R5)."""

        return int(getattr(self.sink, "quarantine_evictions_total", 0))

    def run_once(self) -> list[dict[str, Any]]:
        pods, resource_version = list_completion_pods(self.core_api, self.namespace)
        self._last_resource_version = resource_version
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
        grouped = group_completion_pods(pods, serializer=self.serializer)

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
        # Per-attempt failures below are swallowed so one attempt cannot abort
        # the pass; the coverage heartbeat has to know they happened, because a
        # pass that failed to publish an observation must not then claim the
        # cluster is idle.
        failures_before = self.reconcile_failures_total
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
            except Exception as exc:
                self.reconcile_failures_total += 1
                log_reconcile_failure(self, attempt_id, exc)
            self.note_progress()
        if attempt_filter is None:
            evict_pruned_attempts(self, grouped)
            # Not the liveness signal (see ``last_progress_at``): this is the
            # value humans alert on, and only a pass that relisted every Pod
            # moves it.
            self.last_cycle_completed_at = self.now()
            active_pods, active_attempts = active_pass_counts(self, pods)
            publish_coverage_heartbeat(
                self,
                watched_pods=active_pods,
                watched_attempts=active_attempts,
                resource_version=self._last_resource_version,
                reconcile_failures=self.reconcile_failures_total - failures_before,
            )
        return results

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

    def resume_tombstoned_attempt(self, attempt_id: str) -> None:
        """Withdraw a missing-Pod tombstone because the Pods came back (F2).

        The tombstone claimed the attempt no longer existed; a live Pod under
        the same attempt-id is proof it does. Everything derived from the
        tombstone has to go -- the cached terminal, the sent terminal keys, the
        core's attempt state -- or the returning ranks stay invisible and their
        failure is never reported. Deliberately the same reset a metadata
        takeover performs, minus the spec swap: the Pods are re-observed from
        scratch on this very pass.
        """

        self.resumed_attempts_total += 1
        LOGGER.warning(
            "attempt %s was tombstoned as missing but its Pods are listed "
            "again; withdrawing the tombstone and observing the Pods afresh",
            attempt_id,
        )
        self._forget_attempt_progress(attempt_id)

    def _forget_attempt_progress(self, attempt_id: str) -> None:
        """Drop everything this process learned about one attempt-id.

        ``_attempt_specs`` is deliberately left alone: both callers are about to
        rebuild it from the Pods they are holding.
        """

        self.watcher.reset_attempt(attempt_id)
        self._missing_attempts.clear(attempt_id)
        self._restored_attempts.discard(attempt_id)
        self._synthesized_verdicts.discard(attempt_id)
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

    @staticmethod
    def _gpu_uuids(value: str | None) -> list[str]:
        return KubernetesCompletionController._string_list(value, "GPU UUID annotation")


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
        attempt_missing_grace_seconds=int(
            os.getenv(
                "GPU_FAULT_COMPLETION_ATTEMPT_MISSING_GRACE_SECONDS",
                "300",
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
        gpu_uuid_resolver=resolver,
    )


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    validate_gpu_fault_environment(process_name="gpu-fault-completion-watcher")
    controller = controller_from_environment()
    if replay_quarantined_requested(argv):
        # Operator one-shot (C1): one outbox pass, then exit -- never the loop.
        raise SystemExit(replay_quarantined_once(controller.sink, LOGGER))
    controller.run()


if __name__ == "__main__":
    main()
