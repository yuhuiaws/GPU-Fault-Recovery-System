"""The regional cluster action executor, one layer per module.

``cluster_executor.py`` grew to 2,600 lines; the package keeps its whole
import surface and splits the code along the seams it already had:

* ``regional_client`` -- the HTTP client to the control plane and the
  store-shaped ``Regional*`` proxies built on it.
* ``lease`` -- ``CommandLeaseWatch`` and ``CommandLifecycle``: one command
  from claim to posted verdict (renewal, execution cap, result report).
* ``dispatch`` -- ``CommandDispatch``: validation, fleet preflight, adapter
  selection, and the classification of whatever the adapter raised.
* ``batching`` -- ``execute_batched_command``: a compound command's covered
  steps run in order through the dispatch pieces, with progress posts.
* ``executor`` -- ``ClusterActionExecutor``: the claim loop (long-poll, the
  network-degraded back-off), the counters, the breadcrumbs and the stop
  signal.
* ``bootstrap`` -- ``executor_from_environment``, ``readiness_probe`` and
  ``main``, the console-script targets.
* ``metrics`` -- the ``gpu_fault_cluster_executor_`` Prometheus family the
  executor's counters are mirrored into, and the ``/healthz`` predicate.

``SpareReservationSweep`` lives in ``gpu_fault.spare_reservation_sweep`` and is
re-exported here, where it always was.

Every layer logs as ``gpu_fault.cluster_executor``, the name the log format
carries and operators filter on. Nothing here is lazy: the names below are the
names the module always published, and a missing one must fail at import.
"""

from __future__ import annotations

from gpu_fault.cluster_executor.batching import (
    STOPPED_BETWEEN_STEPS_STATUS_SOURCE,
    execute_batched_command,
)
from gpu_fault.cluster_executor.bootstrap import (
    executor_from_environment,
    executor_health,
    main,
    readiness_probe,
    start_executor_metrics,
)
from gpu_fault.cluster_executor.dispatch import CommandDispatch
from gpu_fault.cluster_executor.executor import (
    DEFAULT_LIVENESS_INTERVAL_SECONDS,
    DEFAULT_LIVENESS_STATE_PATH,
    LIVENESS_STALE_AFTER_SECONDS,
    ClusterActionExecutor,
    ClusterExecutorClaimError,
    transport_degraded_idle_delay,
)
from gpu_fault.cluster_executor.lease import (
    ABANDONED_WORKER_HOLD_REASON,
    DEFAULT_MAX_EXECUTION_SECONDS,
    EXECUTION_TIMEOUT_STATUS_SOURCE,
    EXECUTION_TIMEOUT_UNKNOWN_STATUS_SOURCE,
    RETRYABLE_MARKER_KEYS,
    TRANSPORT_RETRYABLE_SOURCE,
    CommandLeaseWatch,
    CommandLifecycle,
    CommandOutcome,
    adapter_facing_details,
)
from gpu_fault.cluster_executor.metrics import (
    CLUSTER_EXECUTOR_METRICS_PREFIX,
    ClusterExecutorMetrics,
    cluster_executor_metrics,
    loop_breadcrumb_is_fresh,
)
from gpu_fault.cluster_executor.regional_client import (
    ClusterExecutorError,
    RegionalExecutorClient,
    RegionalFleetRegistry,
    RegionalHyperPodSubmissionStore,
    RegionalIncidentOwnershipProvider,
)
from gpu_fault.spare_reservation_sweep import (
    SPARE_RESERVATION_SWEEP_INTERVAL_SECONDS,
    SPARE_RESERVATION_TTL_SECONDS,
    SpareReservationSweep,
)

__all__ = [
    "ABANDONED_WORKER_HOLD_REASON",
    "CLUSTER_EXECUTOR_METRICS_PREFIX",
    "DEFAULT_LIVENESS_INTERVAL_SECONDS",
    "DEFAULT_LIVENESS_STATE_PATH",
    "DEFAULT_MAX_EXECUTION_SECONDS",
    "EXECUTION_TIMEOUT_STATUS_SOURCE",
    "EXECUTION_TIMEOUT_UNKNOWN_STATUS_SOURCE",
    "LIVENESS_STALE_AFTER_SECONDS",
    "RETRYABLE_MARKER_KEYS",
    "SPARE_RESERVATION_SWEEP_INTERVAL_SECONDS",
    "SPARE_RESERVATION_TTL_SECONDS",
    "STOPPED_BETWEEN_STEPS_STATUS_SOURCE",
    "TRANSPORT_RETRYABLE_SOURCE",
    "ClusterActionExecutor",
    "ClusterExecutorClaimError",
    "ClusterExecutorError",
    "ClusterExecutorMetrics",
    "CommandDispatch",
    "CommandLeaseWatch",
    "CommandLifecycle",
    "CommandOutcome",
    "RegionalExecutorClient",
    "RegionalFleetRegistry",
    "RegionalHyperPodSubmissionStore",
    "RegionalIncidentOwnershipProvider",
    "SpareReservationSweep",
    "adapter_facing_details",
    "cluster_executor_metrics",
    "execute_batched_command",
    "executor_from_environment",
    "executor_health",
    "loop_breadcrumb_is_fresh",
    "main",
    "readiness_probe",
    "start_executor_metrics",
    "transport_degraded_idle_delay",
]
