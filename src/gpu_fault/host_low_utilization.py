"""Which managed attempt a host batch is graded for, and which GPUs it holds.

``LOW_CPU_UTILIZATION`` and ``LOW_GPU_UTILIZATION``, the two
``active_workload_only`` rules of ``NodeHealthPolicy``, time a window only while
the control plane resolves exactly one managed attempt with a live container on
the batch's node (:func:`resolve_managed_attempt`), and only for the GPUs that
attempt's live containers on that node hold. Both rules read per-GPU samples:
``host_gpu_utilization_percent`` (nvidia-smi ``utilization.gpu``) and
``training_rank_cpu_percent`` (CPU time of the processes holding a compute
context on that GPU, in percent of one core; ``collectors/host/gpu_rank.py``).

Grading every GPU of the node read a partial-node attempt's unallocated GPUs --
0 % by definition -- as "GPU idle while training is active" after 300 s (a
one-GPU job leaves seven). Which GPUs an attempt holds is the union of the GPU
UUIDs its live containers on the node carry: the completion watcher takes them
from the ``gpu-fault.io/gpu-uuids`` annotation or from ``nvidia-smi
--query-gpu=uuid`` inside the container, on HyperPod and plain EKS alike, and
the host collector names the same UUIDs. A container that declares more
``nvidia.com/gpu`` than it lists UUIDs leaves the allocation unresolved. It is
then resolved only when the declared GPUs cover every GPU the node reports: the
product requires Kubernetes allocatable to equal the host inventory
(``gpu_kubernetes_allocatable_mismatch``), so such an attempt holds them all.
Otherwise only the listed UUIDs are graded -- no false positive, at the price of
a GPU that is not timed; the gap is logged (:class:`UnresolvedAllocationWarnings`).
A GPU the attempt does not hold is an idle sample: it resets the unscoped window
and is never timed for the attempt.

The CPU rule is the CPU of the attempt's own processes, in percent of one core,
not the host's ``cpu_usage_percent`` (eight ranks spinning a core each read 4.2 %
of a 192-vCPU host and 8.3 % of a 96-vCPU one). Its premise is CUDA's default
scheduling, cited below: a rank waiting on its GPU spins a core, a blocked rank
reads ~0, and a job that opts into blocking sync can read ~0 while healthy (the
rule's known limitation; its action is VALIDATE_HOST only).
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple

from gpu_fault.models import WorkloadState
from gpu_fault.store.shared.health_signals import attempt_signal_key

if TYPE_CHECKING:
    from gpu_fault.host_health import (
        HostMetricSample,
        HostTelemetryBatch,
        NodeHealthPolicy,
    )
    from gpu_fault.watcher import AttemptObservation

#: The policy's own logger: these lines were written by ``host_health`` before
#: they moved here, and operators grep for them under that name.
LOGGER = logging.getLogger("gpu_fault.host_health")
#: ``(signal_key, active, observed_at, minimum_active_seconds)`` of one claim.
Transition = tuple[str, bool, datetime, float]
GPU_UTILIZATION_METRIC = "host_gpu_utilization_percent"
#: NVIDIA, CUDA Runtime API, cudaSetDeviceFlags (cudaDeviceScheduleAuto is the
#: default; cudaDeviceScheduleBlockingSync the opt-in this rule cannot tell from
#: a hang). Pinned against both editions of the detailed design by a test.
CUDA_SCHEDULING_REFERENCE = (
    "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html"
)
CUDA_SCHEDULING_DOCS_VERSION = "CUDA Runtime API Reference Manual 13.4"
CUDA_SCHEDULING_SECTION = ("6.3. Device Management", "cudaSetDeviceFlags")
CUDA_SCHEDULE_AUTO_QUOTE = (
    "If C > P, then CUDA will yield to other OS threads when waiting for the "
    "device, otherwise CUDA will not yield while waiting for results and "
    "actively spin on the processor."
)
CUDA_SCHEDULE_BLOCKING_SYNC_QUOTE = (
    "Instruct CUDA to block the CPU thread on a synchronization primitive when "
    "waiting for the device to finish work."
)


def gpu_key(device: str) -> str:
    """One spelling of a GPU UUID for comparison (annotations are free text)."""

    return device.strip().lower()


class GpuOwnership(NamedTuple):
    """The GPUs of one node a managed attempt holds, and how that is known."""

    #: :func:`gpu_key` spellings of the held GPUs.
    gpu_uuids: frozenset[str]
    #: ``gpu_uuid`` (listed by the containers), ``gpu_count`` (the declared
    #: count covers the node) or ``none`` (nothing held as far as is known).
    source: str
    #: ``nvidia.com/gpu`` the attempt's live containers on the node declare.
    declared: int
    #: A container declares more GPUs than it lists UUIDs, and the count does
    #: not cover the node: those GPUs cannot be told apart from unheld ones.
    unresolved: bool


def node_gpu_ownership(
    attempt: AttemptObservation, batch: HostTelemetryBatch
) -> GpuOwnership:
    """Which of ``batch``'s GPUs ``attempt`` holds on the batch's node."""

    live = [
        container
        for container in attempt.containers
        if container.node_id == batch.node_id and not container.terminated
    ]
    listed = frozenset(
        gpu_key(uuid) for container in live for uuid in container.gpu_uuids
    )
    declared = sum(
        max(container.gpu_count, len(container.gpu_uuids)) for container in live
    )
    unresolved = any(
        container.gpu_count > len(container.gpu_uuids) for container in live
    )
    node_gpus = frozenset(
        gpu_key(sample.device)
        for sample in batch.samples
        if sample.name == GPU_UTILIZATION_METRIC and sample.device
    )
    if unresolved and node_gpus and declared >= len(node_gpus):
        return GpuOwnership(node_gpus | listed, "gpu_count", declared, False)
    return GpuOwnership(listed, "gpu_uuid" if listed else "none", declared, unresolved)


def resolve_managed_attempt(
    store: Any, batch: HostTelemetryBatch
) -> AttemptObservation | None:
    """The one managed attempt live on ``batch``'s node, or ``None``.

    ``workload_state``/``affected_workload_ids`` are the collector's static
    environment, filled from the topology when it says UNKNOWN; the attempt
    itself must be PENDING/RUNNING in an observation from 30 s after to 120 s
    before the batch, name one of those workloads and have a live container on
    the node. Two such attempts are ambiguous and resolve to none.
    """

    if (
        batch.workload_state is not WorkloadState.ACTIVE
        or not batch.affected_workload_ids
    ):
        return None
    workload_ids = set(batch.affected_workload_ids)
    candidates = []
    for observation in store.list_attempt_observations(batch.cluster_id):
        age = (batch.observed_at - observation.observed_at).total_seconds()
        if (
            age < -30
            or age > 120
            or observation.workload_phase.value not in {"PENDING", "RUNNING"}
            or not workload_ids.intersection(observation.workload_ids)
            or not any(
                container.node_id == batch.node_id and not container.terminated
                for container in observation.containers
            )
        ):
            continue
        candidates.append(observation)
    identities = {(item.job_id, item.attempt_id) for item in candidates}
    if len(identities) > 1:
        LOGGER.warning(
            "ambiguous managed attempt ownership: cluster=%s node=%s identities=%s",
            batch.cluster_id,
            batch.node_id,
            sorted(identities),
        )
        return None
    return candidates[0] if len(identities) == 1 else None


class UnresolvedAllocationWarnings:
    """Warn once per attempt and node that its GPUs cannot be graded.

    A starting container lists no UUIDs until the watcher has read them, so
    this waits until the attempt has been RUNNING for one sustain window: past
    that, a breach on a GPU the attempt may hold goes untimed, and the operator
    has to restore the UUIDs (the watcher's ``GPU_FAULT_DISCOVER_POD_GPU_UUIDS``
    discovery needs ``nvidia-smi`` in the container and ``pods/exec``, or the
    ``gpu-fault.io/gpu-uuids`` annotation). Bounded: the oldest key is
    forgotten first, so a forgotten attempt can warn again. The ingest runs
    batches on several Store I/O threads, hence the lock.
    """

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._lock = threading.Lock()
        self._warned: OrderedDict[tuple[str, str, str, str], None] = OrderedDict()

    def note(
        self,
        batch: HostTelemetryBatch,
        attempt: AttemptObservation,
        ownership: GpuOwnership,
        window_seconds: float,
    ) -> bool:
        """Log the warning when it is due; whether it was logged."""

        started_at = attempt.started_at
        if (
            attempt.workload_phase.value != "RUNNING"
            or started_at is None
            or (batch.observed_at - started_at).total_seconds() < window_seconds
        ):
            return False
        key = (batch.cluster_id, batch.node_id, attempt.job_id, attempt.attempt_id)
        with self._lock:
            if key in self._warned:
                return False
            self._warned[key] = None
            while len(self._warned) > self.capacity:
                self._warned.popitem(last=False)
        LOGGER.warning(
            "low CPU/GPU utilisation cannot grade every GPU of attempt %s/%s on "
            "cluster=%s node=%s: its live containers there declare %d GPUs but "
            "list %d GPU UUIDs, so only the listed GPUs are graded; restore the "
            "UUIDs (the watcher's GPU_FAULT_DISCOVER_POD_GPU_UUIDS discovery or "
            "the gpu-fault.io/gpu-uuids annotation)",
            attempt.job_id,
            attempt.attempt_id,
            batch.cluster_id,
            batch.node_id,
            ownership.declared,
            len(ownership.gpu_uuids),
        )
        return True


class BatchManagedAttempt:
    """Resolve, once, the managed attempt owning a batch; scope its windows.

    A batch carries eight GPUs times several rules and the resolver walks
    every attempt observation in the cluster; the answer cannot change inside
    one ``evaluate_metrics`` call, so it is memoised here and discarded with
    the call. Nothing lands on the policy but the bounded warning memory.
    """

    __slots__ = (
        "_attempt",
        "_batch",
        "_explained",
        "_gates",
        "_ownership",
        "_policy",
        "_resolved",
        "_ungraded_explained",
    )

    def __init__(self, policy: NodeHealthPolicy, batch: HostTelemetryBatch) -> None:
        self._policy = policy
        self._batch = batch
        self._resolved = False
        self._attempt: Any = None
        self._ownership: GpuOwnership | None = None
        self._explained = False
        self._ungraded_explained = False
        self._gates: dict[str, Transition] = {}

    def scope(self, transition: Transition) -> Transition:
        """Time a graded sample's window per resolved managed attempt.

        A replacement attempt must not inherit the node's low time or latch
        (round 10: E2E-002 restarted a minute after taking over from
        COLLECT-017). The claim runs under the attempt's key; the unscoped
        transition rides along as its gate, the only key that sees -- and is
        reset by -- a sample the attempt is not graded on. Both are store rows:
        every replica shares them and they survive restarts. Only called for a
        sample :meth:`grades` accepted, so the attempt has resolved.
        """

        attempt = self.resolve()
        signal_key, active, observed_at, duration = transition
        scoped = attempt_signal_key(signal_key, attempt.job_id, attempt.attempt_id)
        self._gates[scoped] = transition
        return scoped, active, observed_at, duration

    def claim(
        self, transitions: list[Transition], fingerprints: list[str]
    ) -> list[bool]:
        """Claim transitions and gates in one store call; a scoped window emits
        only if its gate, never latched, has been active as long."""

        gated = [i for i, item in enumerate(transitions) if item[0] in self._gates]
        emitted = self._policy.store.claim_health_signal_transitions(
            [*transitions, *(self._gates[transitions[i][0]] for i in gated)],
            received_at=self._batch.received_at,
            semantic_fingerprints=[*fingerprints, *(fingerprints[i] for i in gated)],
        )
        verdicts: list[bool] = emitted[: len(transitions)]
        for index, gate in zip(gated, emitted[len(transitions) :], strict=True):
            verdicts[index] = verdicts[index] and gate
        return verdicts

    def resolve(self) -> Any:
        """The one managed attempt on the batch's node, or ``None``; memoized."""

        if not self._resolved:
            self._attempt = self._policy._active_attempt(self._batch)
            self._resolved = True
        return self._attempt

    def ownership(self) -> GpuOwnership:
        """The GPUs the resolved attempt holds on this node; memoized."""

        if self._ownership is None:
            self._ownership = node_gpu_ownership(self.resolve(), self._batch)
        return self._ownership

    def grades(self, rule_id: str, sample: HostMetricSample, active: bool) -> bool:
        """Whether an active-workload rule times ``sample`` for the attempt.

        Only a GPU the resolved attempt holds on this node is graded; any other
        sample is idle. ``active`` (the reading breaches) only decides whether
        the reason is worth a debug line.
        """

        attempt = self.resolve()
        if attempt is None:
            if active:
                self.explain_inactive(rule_id, sample)
            return False
        ownership = self.ownership()
        if sample.device is not None and gpu_key(sample.device) in ownership.gpu_uuids:
            return True
        if active:
            self._explain_ungraded(rule_id, sample, attempt, ownership)
        return False

    def explain_inactive(self, rule_id: str, sample: HostMetricSample) -> None:
        """Debug-log, once per batch, why an active-workload rule stays idle."""

        if self._explained:
            return
        self._explained = True
        batch = self._batch
        if (
            batch.workload_state is not WorkloadState.ACTIVE
            or not batch.affected_workload_ids
        ):
            reason = "collector reports no ACTIVE workload"
        else:
            reason = (
                "collector declares an ACTIVE workload but no single managed "
                "attempt with a live container resolves on this node"
            )
        LOGGER.debug(
            "%s on cluster=%s node=%s (first breach %s%s) treated as inactive: %s; "
            "workload_state=%s affected_workload_ids=%s",
            rule_id,
            batch.cluster_id,
            batch.node_id,
            sample.name,
            f"/{sample.device}" if sample.device else "",
            reason,
            batch.workload_state.value,
            batch.affected_workload_ids,
        )

    def _explain_ungraded(
        self,
        rule_id: str,
        sample: HostMetricSample,
        attempt: AttemptObservation,
        ownership: GpuOwnership,
    ) -> None:
        """Debug-log, once per batch, a breach on a GPU the attempt does not
        hold; warn once per attempt when its allocation stays unresolved."""

        if ownership.unresolved:
            self._policy.unresolved_gpu_allocations.note(
                self._batch,
                attempt,
                ownership,
                self._policy.low_utilization_duration_seconds,
            )
        if self._ungraded_explained:
            return
        self._ungraded_explained = True
        LOGGER.debug(
            "%s on cluster=%s node=%s (first breach %s/%s) not graded: attempt "
            "%s/%s holds %d GPUs there by %s (declares %d, unresolved=%s)",
            rule_id,
            self._batch.cluster_id,
            self._batch.node_id,
            sample.name,
            sample.device or "node",
            attempt.job_id,
            attempt.attempt_id,
            len(ownership.gpu_uuids),
            ownership.source,
            ownership.declared,
            ownership.unresolved,
        )
