from __future__ import annotations

from typing import Any, Callable

import json
import logging
from datetime import datetime


from gpu_fault.host_health import (
    HostMetricSample,
)


from gpu_fault.collectors.sinks import CollectorError

LOGGER = logging.getLogger(__name__)


class HostGpuRankMixin:
    # Attributes supplied by the composed concrete implementation.
    nvswitch_topology_command: Any
    rank_liveness_enabled: Any

    _clock_ticks: Any
    _delta: Callable[..., Any]
    _previous: Any
    _sample: Callable[..., Any]
    proc_root: Any
    rank_progress_gpu_idle_percent: Any
    rank_progress_min_cpu_cores: Any
    rank_progress_min_write_bps: Any
    runner: Callable[..., Any]

    def _gpu_utilization(self, _: datetime) -> list[HostMetricSample]:
        self._last_gpu_utilization_percent = None
        argv = [
            "nvidia-smi",
            "--query-gpu=uuid,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        completed = self.runner(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise CollectorError(
                "GPU utilization query failed: " + completed.stderr.strip()
            )
        samples = []
        for line in completed.stdout.splitlines():
            fields = [item.strip() for item in line.split(",", 1)]
            if len(fields) != 2:
                continue
            try:
                value = float(fields[1])
            except ValueError:
                continue
            samples.append(
                self._sample(
                    "host_gpu_utilization_percent",
                    max(0.0, min(100.0, value)),
                    "percent",
                    fields[0],
                )
            )
        self._last_gpu_utilization_percent = (
            max(sample.value for sample in samples) if samples else None
        )
        return samples

    def _warn_rank_liveness(self, detail: str) -> None:
        # The probe is best effort: a node without nvidia-smi simply has
        # no ranks to watch, and warning on every cycle would drown the
        # journal that the log collector reads.
        if self._rank_liveness_warned:
            return
        self._rank_liveness_warned = True
        LOGGER.warning(
            "rank liveness probe cannot list compute apps: %s",
            detail,
        )

    def _gpu_compute_pids(self) -> list[int]:
        try:
            completed = self.runner(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except OSError as exc:
            self._warn_rank_liveness(str(exc))
            return []
        if completed.returncode != 0:
            self._warn_rank_liveness(completed.stderr.strip())
            return []
        pids = []
        for line in completed.stdout.splitlines():
            token = line.strip()
            if token.isdigit():
                pids.append(int(token))
        return sorted(set(pids))

    def _rank_counters(self, pid: int) -> tuple[str, dict[str, float]] | None:
        """Cheap per-rank progress counters straight from procfs.

        ``write_bytes``/``wchar`` say whether the rank is still
        producing output, and ``cpu_ticks`` says whether it is still
        running code. Reading three small procfs files per rank costs
        far less than the py-spy attach that a hang escalation triggers,
        so the probe can run on every collection cycle.
        """

        root = self.proc_root / str(pid)
        try:
            stat_fields = (
                (root / "stat")
                .read_text(encoding="utf-8", errors="replace")
                .rpartition(")")[2]
                .split()
            )
            # Fields are numbered from ``state`` at 3 in proc(5); the
            # split above drops the first two, so utime/stime/starttime
            # land at 11/12/19.
            cpu_ticks = float(stat_fields[11]) + float(stat_fields[12])
            starttime = stat_fields[19]
            counters = {"cpu_ticks": cpu_ticks}
            for line in (
                (root / "io").read_text(encoding="utf-8", errors="replace").splitlines()
            ):
                name, _, raw = line.partition(":")
                if name in {
                    "write_bytes",
                    "wchar",
                    "syscw",
                }:
                    # ``write_bytes`` only counts block-device traffic, so
                    # a checkpoint written to FSx or streamed to S3 shows
                    # up in ``wchar`` alone. Both are read and the larger
                    # rate decides.
                    counters[name] = float(raw.strip())
            for line in (
                (root / "status")
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            ):
                if line.startswith("voluntary_ctxt_switches:"):
                    counters["voluntary_ctxt_switches"] = float(
                        line.split(":", 1)[1].strip()
                    )
                    break
        except (OSError, ValueError, IndexError):
            # The rank exited between listing and sampling, or the
            # collector lacks the privilege to read /proc/<pid>/io.
            return None
        return starttime, counters

    def _prune_rank_counters(self, live_keys: set[str]) -> None:
        """Drop delta baselines of ranks that no longer exist.

        Ranks come and go with every attempt, and a node that finishes a
        job keeps running this collector for weeks. Without pruning on
        the "no GPU process at all" path too, every attempt would leave
        its pids behind in the baseline for the life of the process.
        """

        for key in [
            item
            for item in self._previous
            if item.startswith("rank/") and item not in live_keys
        ]:
            del self._previous[key]

    def _rank_liveness(self, observed_at: datetime) -> list[HostMetricSample]:
        """Per-rank liveness evidence that needs no application hooks.

        Zero EFA traffic cannot by itself separate a wedged collective
        from a checkpoint write or a graph compile. Those two look
        opposite from procfs: a checkpoint moves ``write_bytes``, a
        compile burns host CPU with idle GPUs, and a rank stuck in a
        collective does neither while its kernel keeps the SMs busy.
        The node publishes how long ago it last saw a rank advance so
        the control plane can restart the hang clock from measured
        progress instead of escalating on a quiet fabric.
        """

        if not self.rank_liveness_enabled:
            return []
        pids = self._gpu_compute_pids()
        if not pids:
            self._rank_progress_at = None
            self._prune_rank_counters(set())
            return []
        gpu_utilization = self._last_gpu_utilization_percent
        gpu_idle = (
            gpu_utilization is not None
            and gpu_utilization <= self.rank_progress_gpu_idle_percent
        )
        totals = {
            "write_bytes": 0.0,
            "wchar": 0.0,
            "syscw": 0.0,
            "cpu_ticks": 0.0,
            "voluntary_ctxt_switches": 0.0,
        }
        sampled = 0
        advancing = 0
        live_keys = set()
        for pid in pids:
            reading = self._rank_counters(pid)
            if reading is None:
                continue
            sampled += 1
            starttime, counters = reading
            deltas = {}
            interval = 0.0
            for name, value in counters.items():
                key = f"rank/{pid}/{starttime}/{name}"
                live_keys.add(key)
                change = self._delta(key, value, observed_at)
                if change is None:
                    continue
                deltas[name] = change[0]
                interval = max(interval, change[1])
                totals[name] = totals.get(name, 0.0) + change[0]
            if interval <= 0:
                continue
            write_bps = (
                max(
                    deltas.get("write_bytes", 0.0),
                    deltas.get("wchar", 0.0),
                )
                / interval
            )
            cpu_cores = deltas.get("cpu_ticks", 0.0) / self._clock_ticks / interval
            # A rank stuck in a collective keeps its spin kernel resident,
            # so the GPU reads busy while nothing leaves the process. Host
            # CPU alone is not progress: the NCCL wait spins too. It only
            # counts as progress when the GPUs are idle, which is what a
            # compile or a data-loading stall looks like and what a wedged
            # collective cannot look like.
            if write_bps >= self.rank_progress_min_write_bps or (
                gpu_idle and cpu_cores >= self.rank_progress_min_cpu_cores
            ):
                advancing += 1
        self._prune_rank_counters(live_keys)
        if advancing:
            self._rank_progress_at = observed_at
        result = [
            self._sample(
                "training_rank_process_count",
                float(sampled),
                "count",
            ),
            self._sample(
                "training_rank_advancing_count",
                float(advancing),
                "count",
            ),
            self._sample(
                "training_rank_write_bytes_delta",
                totals["write_bytes"],
                "bytes",
            ),
            self._sample(
                "training_rank_write_chars_delta",
                totals["wchar"],
                "bytes",
            ),
            self._sample(
                "training_rank_write_syscalls_delta",
                totals["syscw"],
                "events",
            ),
            self._sample(
                "training_rank_cpu_ticks_delta",
                totals["cpu_ticks"],
                "ticks",
            ),
            self._sample(
                "training_rank_voluntary_ctxt_switches_delta",
                totals["voluntary_ctxt_switches"],
                "events",
            ),
        ]
        if self._rank_progress_at is not None:
            result.append(
                self._sample(
                    "training_rank_seconds_since_progress",
                    max(
                        0.0,
                        (observed_at - self._rank_progress_at).total_seconds(),
                    ),
                    "seconds",
                )
            )
        return result

    def _nvswitch_topology(self, _: datetime) -> list[HostMetricSample]:
        if not self.nvswitch_topology_command:
            return []
        completed = self.runner(
            self.nvswitch_topology_command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if completed.returncode != 0:
            raise CollectorError(
                "NVSwitch topology query failed: " + completed.stderr.strip()
            )
        payload = json.loads(completed.stdout)
        ports = payload.get("ports", []) if isinstance(payload, dict) else payload
        if not isinstance(ports, list):
            raise CollectorError("NVSwitch topology query must return a ports array")
        samples = []
        for item in ports:
            if not isinstance(item, dict):
                continue
            switch_id = str(item.get("switch_id", "")).strip()
            port = str(item.get("port", "")).strip()
            link_scope = str(item.get("link_scope", "")).strip().upper()
            peer_type = str(item.get("peer_type", "")).strip().upper()
            if not switch_id or not port or link_scope not in {"ACCESS", "TRUNK"}:
                continue
            labels = {
                "trusted": "true",
                "source": "nvswitch-topology-query",
                "switch_id": switch_id,
                "port": port,
                "link_scope": link_scope,
            }
            if peer_type:
                labels["peer_type"] = peer_type
            for key in ("gpu_uuid", "fabric_partition"):
                value = str(item.get(key, "")).strip()
                if value:
                    labels[key] = value
            samples.append(
                HostMetricSample(
                    name="nvswitch_port_topology",
                    value=1,
                    device=f"{switch_id}/{port}",
                    labels=labels,
                )
            )
        return samples
