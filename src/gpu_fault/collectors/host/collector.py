from __future__ import annotations

import argparse
import contextlib
import logging
import os
import shlex
import socket
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from gpu_fault.channel_registry import HOST_TELEMETRY_PATH
from gpu_fault.env import env_bool
from gpu_fault.models import WorkloadState
from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
    HostTelemetryHistoryPoint,
    NodeHealthPolicy,
)


from gpu_fault.collectors.gpu.discovery import expected_accelerator_counts
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import (
    CollectorError,
    EventSink,
    deliver_event,
)
from gpu_fault.collectors.host.gpu_rank import HostGpuRankMixin
from gpu_fault.collectors.host.inventory import HostInventoryMixin
from gpu_fault.collectors.host.network import HostNetworkMixin
from gpu_fault.collectors.host.system_metrics import (
    BoundedStatvfs,
    HostSystemMetricsMixin,
)

LOGGER = logging.getLogger(__name__)


def sd_notify(state: str) -> bool:
    """Send one ``sd_notify(3)`` datagram, or do nothing off systemd.

    ``Type=notify`` with ``WatchdogSec=`` is the only thing that makes a wedged
    collector visible to the supervisor: every contributor reads the host, and
    both ``statvfs`` on a hard NFS/Lustre mount and ``nvidia-smi`` inside a hung
    driver block in uninterruptible sleep, where the unit stays
    "active (running)" forever and ``Restart=always`` never fires. Returns
    whether a notification was sent, so a collector run by hand (no
    ``NOTIFY_SOCKET``) is a no-op rather than an error.
    """

    address = os.getenv("NOTIFY_SOCKET")
    if not address:
        return False
    if address.startswith("@"):
        # An abstract namespace socket: systemd writes the leading NUL as "@".
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify_socket:
            notify_socket.connect(address)
            notify_socket.sendall(state.encode("utf-8"))
    except OSError as exc:
        LOGGER.warning("could not notify systemd (%s): %s", state, exc)
        return False
    return True


def _watchdog_heartbeat() -> Callable[[], None]:
    """A ``WATCHDOG=1`` sender that costs nothing off systemd.

    ``NOTIFY_SOCKET`` is read once: without it every ping would otherwise pay a
    ``getenv`` and a log line per contributor per tick for a collector nobody
    supervises.
    """

    if not os.getenv("NOTIFY_SOCKET"):
        return lambda: None

    def ping() -> None:
        sd_notify("WATCHDOG=1")

    return ping


def _reap_abandoned_process(argv: list[str], process: subprocess.Popen[str]) -> None:
    """Wait out a killed child that has not left the kernel yet.

    Runs on a daemon thread and touches nothing but its own child: the circuit
    breaker and every other piece of collector state stay owned by the
    collection thread.
    """

    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()
    with contextlib.suppress(Exception):
        process.wait()
    LOGGER.info(
        "abandoned %s (pid %s) finally exited with %s",
        argv[0] if argv else "?",
        process.pid,
        process.returncode,
    )


class BoundedProcessRunner:
    """``subprocess.run`` that a child in uninterruptible sleep cannot outlive.

    CPython implements ``run(timeout=...)`` as ``kill()`` followed by a
    *blocking* ``wait()``. The failure this collector exists to report --
    nvidia-smi wedged inside the driver (XID 79, a GPU off the bus, a
    fabric-manager hang) -- is exactly the case where SIGKILL is not acted on
    until the syscall returns, so ``TimeoutExpired`` never reached the caller:
    the nvidia-smi circuit breaker never engaged, ``collect_once`` never
    returned, and the node sent no host telemetry at all. Here the second wait
    is bounded too and a child that outlives it is handed to a daemon reaper,
    so a call returns within ``timeout + kill_grace_seconds``. Every
    ``self.runner`` call site shares this bound, so smartctl on a dying NVMe
    and ethtool on a wedged EFA netdev cannot hold a tick either.
    """

    def __init__(
        self,
        *,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        kill_grace_seconds: float = 1.0,
    ) -> None:
        self.popen = popen
        self.kill_grace_seconds = kill_grace_seconds

    def __call__(
        self,
        argv: list[str],
        *,
        capture_output: bool = False,
        text: bool = False,
        timeout: float | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        pipe = subprocess.PIPE if capture_output else None
        process = self.popen(argv, stdout=pipe, stderr=pipe, text=text)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._abandon(argv, process)
            raise
        completed = subprocess.CompletedProcess(
            argv, process.returncode or 0, stdout, stderr
        )
        if check:
            completed.check_returncode()
        return completed

    def _abandon(self, argv: list[str], process: subprocess.Popen[str]) -> None:
        with contextlib.suppress(OSError):
            process.kill()
        try:
            process.wait(timeout=self.kill_grace_seconds)
        except subprocess.TimeoutExpired:
            pass
        else:
            # ``communicate`` timed out, so its pipes were never drained or
            # closed. A tick runs every 15 s and the collector's RLIMIT_NOFILE
            # is the unit's default, so leaking two descriptors per timed-out
            # call ends in EMFILE -- where every probe fails, not just this one.
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    with contextlib.suppress(OSError):
                        stream.close()
            return
        LOGGER.warning(
            "%s did not die within %gs of SIGKILL (uninterruptible sleep in a "
            "driver); abandoning it to a background reaper",
            argv[0] if argv else "?",
            self.kill_grace_seconds,
        )
        threading.Thread(
            target=_reap_abandoned_process,
            args=(argv, process),
            name="gpu-fault-process-reaper",
            daemon=True,
        ).start()


class HostTelemetryCollector(
    HostGpuRankMixin,
    HostInventoryMixin,
    HostNetworkMixin,
    HostSystemMetricsMixin,
):
    """Collects host, storage, network, RDMA, SMART and BMC signals."""

    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        interval_seconds: float = 15,
        filesystems: list[str] | None = None,
        required_interfaces: list[str] | None = None,
        infiniband_root: str = "/sys/class/infiniband",
        net_class_root: str = "/sys/class/net",
        proc_root: str = "/proc",
        pci_devices_root: str | None = None,
        node_instance_type: str | None = None,
        expected_gpu_count: int | None = None,
        expected_efa_device_count: int | None = None,
        inventory_mismatch_consecutive_samples: int = 2,
        nvswitch_topology_command: list[str] | None = None,
        now: Callable[[], datetime] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        statvfs: Callable[[str], os.statvfs_result] | None = None,
        statvfs_timeout_seconds: float = 5,
        statvfs_budget_seconds: float = 15,
        edge_filter_enabled: bool | None = None,
        health_summary_seconds: int | None = None,
        history_max_points: int | None = None,
        startup_spread_seconds: int | None = None,
        force_snapshot_path: str | None = None,
        nvidia_smi_timeout_seconds: float = 15,
        nvidia_smi_breaker_rounds: int = 3,
        nvidia_smi_breaker_cooldown_rounds: int = 4,
    ) -> None:
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.interval_seconds = interval_seconds
        self.filesystems = filesystems or ["/", "/var", "/tmp"]
        self.required_interfaces = set(required_interfaces or [])
        self.infiniband_root = Path(infiniband_root)
        self.net_class_root = Path(net_class_root)
        self.proc_root = Path(proc_root)
        self.pci_devices_root = (
            Path(pci_devices_root)
            if pci_devices_root is not None
            else (
                Path("/sys/bus/pci/devices")
                if infiniband_root == "/sys/class/infiniband"
                else None
            )
        )
        self.node_instance_type = node_instance_type
        self.expected_gpu_count = expected_gpu_count
        self.expected_efa_device_count = expected_efa_device_count
        self._default_expected_counts_from_instance_type()
        if (
            nvidia_smi_timeout_seconds <= 0
            or nvidia_smi_breaker_rounds < 1
            or nvidia_smi_breaker_cooldown_rounds < 1
        ):
            raise ValueError(
                "nvidia-smi timeout must be positive and breaker rounds at least 1"
            )
        self.nvidia_smi_timeout_seconds = nvidia_smi_timeout_seconds
        self.nvidia_smi_breaker_rounds = nvidia_smi_breaker_rounds
        self.nvidia_smi_breaker_cooldown_rounds = nvidia_smi_breaker_cooldown_rounds
        self._nvidia_smi_round: dict[
            tuple[str, ...], subprocess.CompletedProcess[str] | CollectorError
        ] = {}
        self._nvidia_smi_round_at: datetime | None = None
        self._nvidia_smi_round_timed_out = False
        self._nvidia_smi_round_skipped = False
        self._nvidia_smi_consecutive_timeouts = 0
        self._nvidia_smi_breaker_skips_remaining = 0
        if inventory_mismatch_consecutive_samples < 1:
            raise ValueError("inventory mismatch consecutive samples must be positive")
        self.inventory_mismatch_consecutive_samples = (
            inventory_mismatch_consecutive_samples
        )
        self._inventory_mismatch_counts = {
            "gpu": 0,
            "efa": 0,
        }
        self.nvswitch_topology_command = (
            nvswitch_topology_command
            if nvswitch_topology_command is not None
            else shlex.split(os.getenv("GPU_FAULT_NVSWITCH_TOPOLOGY_COMMAND", ""))
        )
        self.now = now or (lambda: datetime.now(timezone.utc))
        # The default runner bounds its own kill path; ``subprocess.run`` does
        # not, and a D-state nvidia-smi wedged the whole collector in it.
        self.runner = runner if runner is not None else BoundedProcessRunner()
        if statvfs_timeout_seconds <= 0:
            raise ValueError("statvfs timeout must be positive")
        if statvfs_budget_seconds <= 0:
            raise ValueError("statvfs budget must be positive")
        self.statvfs_timeout_seconds = statvfs_timeout_seconds
        # One unreachable file server is every bind mount served from it, so
        # the tick caps its *total* mount waiting, not just each call's.
        self.statvfs_budget_seconds = statvfs_budget_seconds
        self._statvfs_probe = BoundedStatvfs(
            statvfs if statvfs is not None else os.statvfs,
            timeout_seconds=statvfs_timeout_seconds,
        )
        self._statvfs_tick: datetime | None = None
        self._statvfs_spent_seconds: float = 0.0
        self._unresponsive_mounts: set[str] = set()
        #: Samples a contributor produced before it failed (see
        #: :meth:`collect_once`).
        self._partial_samples: list[HostMetricSample] = []
        self._previous: dict[str, tuple[float, datetime]] = {}
        self.edge_filter_enabled = (
            edge_filter_enabled
            if edge_filter_enabled is not None
            else env_bool("GPU_FAULT_HOST_EDGE_FILTER_ENABLED", True)
        )
        self.health_summary_seconds = (
            health_summary_seconds
            if health_summary_seconds is not None
            else int(os.getenv("GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS", "300"))
        )
        self.startup_spread_seconds = (
            startup_spread_seconds
            if startup_spread_seconds is not None
            else int(
                os.getenv(
                    "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS",
                    "15",
                )
            )
        )
        self.force_snapshot_path = Path(
            force_snapshot_path
            or os.getenv(
                "GPU_FAULT_HOST_HEALTH_SNAPSHOT_REQUEST_PATH",
                ("/var/lib/gpu-fault/health-snapshot/host.request"),
            )
        )
        history_points = (
            history_max_points
            if history_max_points is not None
            else int(os.getenv("GPU_FAULT_HOST_HISTORY_MAX_POINTS", "20"))
        )
        if (
            self.health_summary_seconds <= 0
            or history_points <= 0
            or self.startup_spread_seconds <= 0
        ):
            raise ValueError("host edge filter intervals and counts must be positive")
        self.low_utilization_threshold_percent = float(
            os.getenv("GPU_FAULT_LOW_UTILIZATION_THRESHOLD_PERCENT", "5")
        )
        self.memory_available_warning_percent = float(
            os.getenv("GPU_FAULT_MEMORY_AVAILABLE_WARNING_PERCENT", "5")
        )
        self.page_cache_warning_percent = float(
            os.getenv("GPU_FAULT_PAGE_CACHE_WARNING_PERCENT", "90")
        )
        self.local_filesystem_warning_percent = float(
            os.getenv("GPU_FAULT_LOCAL_FILESYSTEM_WARNING_PERCENT", "90")
        )
        self.efa_minimum_active_bps = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "1048576")
        )
        self.efa_drop_ratio = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_DROP_RATIO", "0.25")
        )
        self.efa_spike_ratio = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO", "4")
        )
        if not all(
            0 <= value <= 100
            for value in (
                self.low_utilization_threshold_percent,
                self.memory_available_warning_percent,
                self.page_cache_warning_percent,
                self.local_filesystem_warning_percent,
            )
        ):
            raise ValueError("host edge percentage thresholds must be from 0 to 100")
        if not 0 < self.efa_drop_ratio < 1:
            raise ValueError("GPU_FAULT_EFA_TRAFFIC_DROP_RATIO must be in (0, 1)")
        if self.efa_spike_ratio <= 1:
            raise ValueError("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO must be greater than 1")
        self.rank_liveness_enabled = env_bool("GPU_FAULT_RANK_LIVENESS_ENABLED", True)
        self.rank_progress_min_write_bps = float(
            os.getenv("GPU_FAULT_RANK_PROGRESS_MIN_WRITE_BPS", "1048576")
        )
        self.rank_progress_min_cpu_cores = float(
            os.getenv("GPU_FAULT_RANK_PROGRESS_MIN_CPU_CORES", "0.5")
        )
        self.rank_progress_gpu_idle_percent = float(
            os.getenv("GPU_FAULT_RANK_PROGRESS_GPU_IDLE_PERCENT", "5")
        )
        if (
            self.rank_progress_min_write_bps < 0
            or self.rank_progress_min_cpu_cores <= 0
            or not 0 <= self.rank_progress_gpu_idle_percent <= 100
        ):
            raise ValueError(
                "rank progress thresholds must be non-negative and the "
                "GPU idle percentage must be from 0 to 100"
            )
        self._clock_ticks = float(os.sysconf("SC_CLK_TCK") or 100)
        self._history: deque[HostTelemetryHistoryPoint] = deque(maxlen=history_points)
        self._last_delivered_at: datetime | None = None
        self._next_health_summary_at: datetime | None = None
        self._active_edge_reasons: set[str] = set()
        self._last_efa_rate: float | None = None
        self._last_gpu_utilization_percent: float | None = None
        self._rank_progress_at: datetime | None = None
        self._rank_liveness_warned = False

    #: Contributor methods, in collection order. This is a class constant rather
    #: than a literal inside :meth:`collect_once` so a caller that has to account
    #: for every contributor — a test silencing them, for instance — reads the
    #: same list the collector runs. Two host tests silenced a hand-copied subset
    #: and left ``_rank_liveness`` live, so it shelled out to ``nvidia-smi``.
    CONTRIBUTORS: tuple[str, ...] = (
        "_cpu",
        "_memory",
        "_filesystems",
        "_shared_filesystems",
        "_lustre",
        "_diskstats",
        "_network",
        "_tcp",
        "_gpu_utilization",
        "_rank_liveness",
        "_gpu_inventory",
        "_efa_inventory",
        "_rdma",
        "_efa_network",
        "_nvswitch_topology",
        "_smart",
        "_bmc",
    )

    def _default_expected_counts_from_instance_type(self) -> None:
        """Fill the expected GPU/EFA counts from the instance type.

        ``--expected-gpu-count`` was optional and the installer left it
        unset, so the ``gpu_inventory_mismatch`` finding never fired on a
        fleet whose instance type already names the count. An explicit count
        still wins; an unknown type is logged once, not guessed.
        """

        if self.expected_gpu_count is not None and (
            self.expected_efa_device_count is not None
        ):
            return
        counts = expected_accelerator_counts(self.node_instance_type)
        if counts is None:
            if self.node_instance_type:
                LOGGER.warning(
                    "instance type %s is not in the accelerator table and no "
                    "explicit expected GPU/EFA count was configured; the "
                    "inventory invariant is not checked",
                    self.node_instance_type,
                )
            return
        if self.expected_gpu_count is None:
            self.expected_gpu_count = counts["gpu"]
        if self.expected_efa_device_count is None:
            self.expected_efa_device_count = counts["efa"]
        LOGGER.info(
            "expected accelerator counts default from instance type %s: gpu=%d efa=%d",
            self.node_instance_type,
            self.expected_gpu_count,
            self.expected_efa_device_count,
        )

    def _begin_nvidia_smi_round(self) -> None:
        """Reset the per-round nvidia-smi memo and advance the circuit breaker."""

        self._nvidia_smi_round = {}
        if self._nvidia_smi_round_skipped:
            self._nvidia_smi_breaker_skips_remaining -= 1
        if self._nvidia_smi_round_timed_out:
            self._nvidia_smi_consecutive_timeouts += 1
            if self._nvidia_smi_consecutive_timeouts >= self.nvidia_smi_breaker_rounds:
                self._nvidia_smi_breaker_skips_remaining = (
                    self.nvidia_smi_breaker_cooldown_rounds
                )
                LOGGER.warning(
                    "nvidia-smi timed out %d rounds in a row; skipping GPU "
                    "queries for the next %d rounds",
                    self._nvidia_smi_consecutive_timeouts,
                    self.nvidia_smi_breaker_cooldown_rounds,
                )
        self._nvidia_smi_round_timed_out = False
        self._nvidia_smi_round_skipped = self._nvidia_smi_breaker_skips_remaining > 0

    def _nvidia_smi(
        self, argv: list[str], observed_at: datetime
    ) -> subprocess.CompletedProcess[str]:
        """Run one nvidia-smi query per round, bounded and circuit-broken.

        Three contributors used to shell out separately with 30 s timeouts,
        so a hung driver stretched a 15 s round to 90 s and the batch never
        said why. The result is memoised per round so the utilization and
        inventory readers share one query, a timeout is bounded per call,
        and after ``nvidia_smi_breaker_rounds`` consecutive timed-out rounds
        the queries are skipped for a cooldown with the reason recorded as a
        collection error rather than as a slow, quiet node.
        """

        if observed_at != self._nvidia_smi_round_at:
            # A contributor called outside ``collect_once`` with a new sample
            # time starts a fresh memo; the round bookkeeping is unchanged.
            self._nvidia_smi_round = {}
            self._nvidia_smi_round_at = observed_at
        key = tuple(argv)
        cached = self._nvidia_smi_round.get(key)
        if isinstance(cached, CollectorError):
            # The same query already hung this round; a second caller must
            # not pay the timeout again.
            raise cached
        if cached is not None:
            return cached
        if self._nvidia_smi_round_skipped:
            raise CollectorError(
                "nvidia-smi circuit breaker open "
                f"({self._nvidia_smi_consecutive_timeouts} consecutive timeouts); "
                "GPU queries skipped this round"
            )
        try:
            completed: subprocess.CompletedProcess[str] = self.runner(
                argv,
                capture_output=True,
                text=True,
                timeout=self.nvidia_smi_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self._nvidia_smi_round_timed_out = True
            error = CollectorError(
                f"nvidia-smi timed out after {self.nvidia_smi_timeout_seconds:g}s"
            )
            # The ``TimeoutExpired`` is deliberately *not* attached as the
            # cause: this error is memoised for the whole round, and the
            # exception holds the child's captured output, hence its pipes.
            self._nvidia_smi_round[key] = error
            raise error
        self._nvidia_smi_consecutive_timeouts = 0
        self._nvidia_smi_round[key] = completed
        return completed

    def collect_once(
        self, heartbeat: Callable[[], None] | None = None
    ) -> HostTelemetryBatch:
        """Run every contributor once and post the batch.

        ``heartbeat`` is called after each contributor. The tick's worst case
        is the sum of every per-call bound in it -- two nvidia-smi calls, a
        topology dump, ethtool per EFA netdev, a smartctl scan plus one call
        per drive, ipmitool, the mount budget and the sink's retry ladder --
        which on a sick host is minutes. Reporting progress only at the end of
        the tick meant the systemd watchdog killed a tick that was slow but
        advancing, and it was killed *before* ``sink.post``: the node then
        posted nothing at all and restarted forever.
        """

        observed_at = self.now()
        samples: list[HostMetricSample] = []
        errors: list[str] = []
        self._begin_nvidia_smi_round()
        self._nvidia_smi_round_at = observed_at
        for name in self.CONTRIBUTORS:
            collector = getattr(self, name)
            try:
                samples.extend(collector(observed_at))
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
            if heartbeat is not None:
                heartbeat()
            # A contributor that fails *after* it has something the consumer
            # needs parks it here rather than returning it, so the failure
            # still names the contributor and still opens an edge. The GPU
            # inventory does this on a failed driver query: the raise used to
            # discard the ``gpu_inventory_mismatch`` sample that is the whole
            # point of the query (F-H6).
            if self._partial_samples:
                samples.extend(self._partial_samples)
                self._partial_samples.clear()
        batch = HostTelemetryBatch(
            batch_id=(
                f"host-{self.node_id}-{int(observed_at.timestamp() * 1_000_000)}"
            ),
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            observed_at=observed_at,
            samples=samples,
            collection_errors=errors,
            runtime_profile_version=(self.context.runtime_profile_version),
            workload_state=self.context.workload_state,
            affected_workload_ids=(self.context.affected_workload_ids),
            evidence_ref=f"host://{self.node_id}",
        )
        self._history.append(
            HostTelemetryHistoryPoint(
                observed_at=observed_at,
                samples=samples,
                collection_errors=errors,
            )
        )
        reasons = self._edge_reasons(batch)
        delivery_reasons = reasons - self._active_edge_reasons
        if not self.edge_filter_enabled:
            delivery_reasons.add("filter-disabled")
        elif self._last_delivered_at is None:
            delivery_reasons.add("baseline")
        if self._active_edge_reasons - reasons:
            delivery_reasons.add("recovered")
        if self._last_delivered_at is not None and (
            (
                self._next_health_summary_at is not None
                and observed_at >= self._next_health_summary_at
            )
            or (
                self._next_health_summary_at is None
                and (observed_at - self._last_delivered_at).total_seconds()
                >= self.health_summary_seconds
            )
        ):
            delivery_reasons.add("health-summary")
        should_deliver = not self.edge_filter_enabled or bool(delivery_reasons)
        if should_deliver:
            batch = batch.model_copy(
                update={
                    "edge_filter_reasons": sorted(delivery_reasons),
                }
            )
            result = deliver_event(
                self.sink,
                HOST_TELEMETRY_PATH,
                batch.model_dump(mode="json"),
            )
            # A batch the durable outbox took is delivered as far as this
            # collector is concerned (ARCH-G3): skipping the bookkeeping made
            # every following tick re-satisfy the summary condition, block in
            # the retry ladder and buffer another 100-200 KB batch with a fresh
            # batch_id until the outbox evicted. Only a batch that went nowhere
            # keeps the edge open.
            result.raise_for_failure()
            if result.buffered:
                LOGGER.warning(
                    "host telemetry batch %s persisted to the collector outbox; "
                    "advancing the edge filter: %s",
                    batch.batch_id,
                    result.error,
                )
            self._last_delivered_at = observed_at
            if (
                self._next_health_summary_at is None
                or observed_at >= self._next_health_summary_at
            ):
                self._next_health_summary_at = next_stable_phase(
                    observed_at,
                    cluster_id=self.context.cluster_id,
                    node_id=self.node_id,
                    channel="host-telemetry",
                    interval_seconds=self.health_summary_seconds,
                )
        self._active_edge_reasons = reasons
        return batch

    def collect_rdma_samples(
        self,
        observed_at: datetime | None = None,
    ) -> list[HostMetricSample]:
        """Collect only RDMA/EFA counters without delivering a full batch."""
        return self._rdma(observed_at or self.now())

    def _edge_reasons(self, batch: HostTelemetryBatch) -> set[str]:
        reasons = {
            f"collection-error:{item.split(':', 1)[0]}"
            for item in batch.collection_errors
        }
        sustained = {
            "memory_available_percent": (
                "lte",
                self.memory_available_warning_percent,
            ),
            "page_cache_percent": (
                "gte",
                self.page_cache_warning_percent,
            ),
            "local_filesystem_used_percent": (
                "gte",
                self.local_filesystem_warning_percent,
            ),
        }
        if batch.workload_state is WorkloadState.ACTIVE:
            sustained.update(
                {
                    "cpu_usage_percent": (
                        "lte",
                        self.low_utilization_threshold_percent,
                    ),
                    "host_gpu_utilization_percent": (
                        "lte",
                        self.low_utilization_threshold_percent,
                    ),
                }
            )
        for sample in batch.samples:
            immediate = NodeHealthPolicy.METRIC_RULES.get(sample.name)
            if immediate is not None and sample.value >= immediate[0]:
                reasons.add(f"threshold:{sample.name}")
            candidate = sustained.get(sample.name)
            if candidate is not None:
                comparison, threshold = candidate
                if (comparison == "lte" and sample.value <= threshold) or (
                    comparison == "gte" and sample.value >= threshold
                ):
                    reasons.add(f"sustained:{sample.name}")
            if sample.name == "efa_traffic_bytes_per_second":
                previous = self._last_efa_rate
                if (
                    batch.workload_state is WorkloadState.ACTIVE
                    and previous is not None
                    and previous >= self.efa_minimum_active_bps
                ):
                    ratio = sample.value / previous
                    if ratio <= self.efa_drop_ratio:
                        reasons.add("efa-traffic-drop")
                    if ratio >= self.efa_spike_ratio:
                        reasons.add("efa-traffic-spike")
                self._last_efa_rate = sample.value
        return reasons

    def run(self) -> None:
        # ``Type=notify``: announce readiness before the startup spread so the
        # unit does not sit in "activating" for the length of the spread, then
        # ping the watchdog as the tick advances -- between contributors, not
        # only when the tick ends, because the sum of the tick's per-call
        # bounds is minutes on a sick host. A tick that stops *advancing* -- a
        # sysfs read or a driver ioctl that never answers -- stops the pings
        # and systemd restarts the unit, which is the last line of defence
        # behind the per-call bounds below.
        sd_notify("READY=1")
        heartbeat = _watchdog_heartbeat()
        force_snapshot = self.force_snapshot_path.exists()
        if not force_snapshot:
            startup_time = self.now()
            delay = (
                next_stable_phase(
                    startup_time,
                    cluster_id=self.context.cluster_id,
                    node_id=self.node_id,
                    channel="host-collector-startup",
                    interval_seconds=self.startup_spread_seconds,
                )
                - startup_time
            ).total_seconds()
            if delay > 0:
                time.sleep(delay)
        while True:
            succeeded = False
            try:
                self.collect_once(heartbeat)
                succeeded = True
            except Exception:
                LOGGER.exception("host telemetry collection failed")
            if force_snapshot and succeeded:
                self.force_snapshot_path.unlink(missing_ok=True)
                force_snapshot = False
            # A tick whose delivery failed still completed: the process is
            # alive and reading the host, and a control plane that rejects a
            # batch must not restart every collector in the fleet.
            sd_notify("WATCHDOG=1")
            time.sleep(self.interval_seconds)

    @staticmethod
    def _sample(
        name: str,
        value: float,
        unit: str | None = None,
        device: str | None = None,
    ) -> HostMetricSample:
        return HostMetricSample(name=name, value=value, unit=unit, device=device)

    def _delta(
        self, key: str, value: float, observed_at: datetime
    ) -> tuple[float, float] | None:
        previous = self._previous.get(key)
        self._previous[key] = (value, observed_at)
        if previous is None or observed_at <= previous[1]:
            return None
        return max(0.0, value - previous[0]), (
            observed_at - previous[1]
        ).total_seconds()


def build_from_environment(
    sink: EventSink, context: CollectorContext, arguments: argparse.Namespace
) -> HostTelemetryCollector:
    """The ``gpu-fault-collector host`` factory named by the registry."""

    if not arguments.node_id:
        raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
    expected_gpu_count = os.getenv("GPU_FAULT_EXPECTED_GPU_COUNT")
    expected_efa_device_count = os.getenv("GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT")
    return HostTelemetryCollector(
        sink,
        context,
        node_id=arguments.node_id,
        interval_seconds=arguments.interval_seconds,
        filesystems=[
            item
            for item in os.getenv("GPU_FAULT_FILESYSTEMS", "/,/var,/tmp").split(",")
            if item
        ],
        required_interfaces=[
            item
            for item in os.getenv("GPU_FAULT_REQUIRED_INTERFACES", "").split(",")
            if item
        ],
        pci_devices_root=(os.getenv("GPU_FAULT_PCI_DEVICES_ROOT") or None),
        node_instance_type=(os.getenv("GPU_FAULT_NODE_INSTANCE_TYPE") or None),
        expected_gpu_count=(int(expected_gpu_count) if expected_gpu_count else None),
        expected_efa_device_count=(
            int(expected_efa_device_count) if expected_efa_device_count else None
        ),
        inventory_mismatch_consecutive_samples=int(
            os.getenv(
                "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES",
                "2",
            )
        ),
    )
