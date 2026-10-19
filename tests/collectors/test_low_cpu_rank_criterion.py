"""LOW_CPU_UTILIZATION reads the attempt's own processes, not the host.

The rule was ``cpu_usage_percent`` (``/proc/stat``, every CPU of the host) at or
below 5 % for 300 s. A healthy GPU-bound job whose eight ranks each spin about
one core while they wait on their GPU uses 8 cores: 4.2 % of the 192 vCPUs of a
plain EKS ``p5en.48xlarge`` -- "near zero" -- and 8.3 % of the 96 a HyperPod
``ml.p5en.48xlarge`` exposes, so the same job read idle on one provider and busy
on the other. The rule now reads ``training_rank_cpu_percent``: per GPU, the
CPU time the processes holding a compute context on it used, in percent of one
core (``collectors/host/gpu_rank.py``), graded for the GPUs the attempt holds.

Each case runs the real host collector over a simulated host -- ``/proc/stat``,
``/proc/<pid>/stat`` of eight ranks and nvidia-smi's GPU and compute-app
queries -- and feeds every batch to the real policy with the managed attempt
observed on the node, on both host sizes.
"""

from __future__ import annotations

import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

from gpu_fault import host_low_utilization as low_utilization
from gpu_fault.collectors import HostTelemetryCollector
from gpu_fault.host_health import NodeHealthPolicy
from gpu_fault.models import WorkloadState
from tests._builders import attempt_observation, build_store, container_observation
from tests.collectors._support import NOW, RecordingSink, context, write_fake_rank

INTERVAL = 15
WINDOW = 300
NODE = "worker-1"
WORKLOAD = "gpu-fault-system/pytorchjob/gpu-fault-ranks"
GPUS = tuple(f"GPU-{index:08x}-1111-2222-3333-{index:012x}" for index in range(8))
PIDS = tuple(4100 + index for index in range(8))
#: kubelet, containerd, the collectors and the DaemonSets: busy on any host.
BACKGROUND_CORES = 0.4


class _Host:
    """A node with ``vcpus`` CPUs whose eight processes each burn ``cores``.

    ``cuda`` decides whether those processes hold a compute context, one GPU
    each (training ranks), or none (NOTIFY-005's CPU burners); ``gpu_percent``
    is every GPU's ``utilization.gpu``.
    """

    def __init__(self, tmp_path, monkeypatch, *, vcpus, cores, cuda, gpu_percent):
        self.root = tmp_path
        self.vcpus = vcpus
        self.cores = cores
        self.cuda = cuda
        self.gpu_percent = gpu_percent
        self.elapsed = 0.0
        monkeypatch.setattr(
            HostTelemetryCollector,
            "CONTRIBUTORS",
            ("_cpu", "_gpu_utilization", "_rank_liveness"),
        )
        original = type(tmp_path).read_text

        def read_text(path, *args, **kwargs):
            if str(path) == "/proc/stat":
                return self._proc_stat()
            if str(path) == "/proc/loadavg":
                return "8.00 8.00 8.00 9/900 4200\n"
            return original(path, *args, **kwargs)

        monkeypatch.setattr(type(tmp_path), "read_text", read_text)
        self.collector = HostTelemetryCollector(
            RecordingSink(),
            context(),
            node_id=NODE,
            proc_root=str(tmp_path),
            runner=self._nvidia_smi,
            now=lambda: NOW + timedelta(seconds=self.elapsed),
            edge_filter_enabled=False,
        )

    def _proc_stat(self) -> str:
        ticks = self.collector._clock_ticks * self.elapsed
        busy = round(ticks * (8 * self.cores + BACKGROUND_CORES))
        return f"cpu {busy} 0 0 {round(ticks * self.vcpus) - busy} 0 0 0 0 0 0\n"

    def _nvidia_smi(self, argv, **_kwargs):
        if tuple(argv) == HostTelemetryCollector.COMPUTE_APPS_ARGV:
            lines = [f"{pid}, {gpu}" for pid, gpu in zip(PIDS, GPUS, strict=True)]
            stdout = "\n".join(lines) + "\n" if self.cuda else ""
        else:
            stdout = "".join(f"{gpu}, {self.gpu_percent}\n" for gpu in GPUS)
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    def tick(self, offset):
        self.elapsed = float(offset)
        for pid in PIDS:
            write_fake_rank(
                self.root,
                pid,
                cpu_ticks=round(self.collector._clock_ticks * offset * self.cores),
                write_bytes=0,
                wchar=0,
            )
        return self.collector.collect_once()


def _run(host, *, managed=True):
    """Every batch through the policy, as the control plane enriches it."""

    store = build_store()
    policy = NodeHealthPolicy(store)
    fired, host_cpu = {}, []
    for offset in range(0, WINDOW + 2 * INTERVAL + 1, INTERVAL):
        batch = host.tick(offset)
        host_cpu.extend(
            item.value for item in batch.samples if item.name == "cpu_usage_percent"
        )
        if managed:
            store.save_attempt_observation(
                attempt_observation(
                    "job-ranks",
                    "attempt-ranks",
                    batch.observed_at,
                    cluster_id=batch.cluster_id,
                    workload_ids=[WORKLOAD],
                    containers=[
                        container_observation(
                            "pod-0", "master-0", 0, NODE, gpu_uuids=list(GPUS)
                        )
                    ],
                )
            )
        fired[offset] = policy.evaluate_metrics(
            batch.model_copy(
                update={
                    "workload_state": (
                        WorkloadState.ACTIVE if managed else WorkloadState.IDLE
                    ),
                    "affected_workload_ids": [WORKLOAD] if managed else [],
                    "received_at": batch.observed_at,
                }
            )
        )
    return fired, host_cpu


def _low_cpu(fired):
    return {
        item.device: offset
        for offset, findings in sorted(fired.items(), reverse=True)
        for item in findings
        if item.diagnostic_parameters.get("signal") == "LOW_CPU_UTILIZATION"
    }


@pytest.mark.parametrize("vcpus", [96, 192])
def test_a_healthy_gpu_bound_job_never_reads_as_idle_cpu_on_either_host(
    tmp_path, monkeypatch, vcpus
):
    host = _Host(
        tmp_path, monkeypatch, vcpus=vcpus, cores=1.0, cuda=True, gpu_percent=97
    )

    fired, host_cpu = _run(host)

    assert _low_cpu(fired) == {}, (
        f"{vcpus} vCPUs: eight ranks spinning a core each read as idle: {fired}"
    )
    # What the host-wide reading made of the same job: the breach was decided
    # by the host's size (8.4 cores of 192 is under 5 %, of 96 it is not).
    expected = 100 * (8 + BACKGROUND_CORES) / vcpus
    assert host_cpu and all(
        value == pytest.approx(expected, abs=0.01) for value in host_cpu
    ), host_cpu
    assert (expected <= 5) is (vcpus == 192)
    rank_cpu = [
        item.value
        for item in host.tick(WINDOW + 3 * INTERVAL).samples
        if item.name == "training_rank_cpu_percent"
    ]
    assert rank_cpu == pytest.approx([100.0] * 8), rank_cpu


@pytest.mark.parametrize("vcpus", [96, 192])
def test_ranks_sitting_off_cpu_trip_the_rule_on_either_host(
    tmp_path, monkeypatch, vcpus
):
    """A blocked attempt: every rank asleep (a dead data loader, a deadlock)."""

    host = _Host(
        tmp_path, monkeypatch, vcpus=vcpus, cores=0.0, cuda=True, gpu_percent=0
    )

    fired, _ = _run(host)

    # The first tick has no CPU baseline, so each window opens at the second.
    assert _low_cpu(fired) == dict.fromkeys(GPUS, INTERVAL + WINDOW), _low_cpu(fired)
    finding = next(
        item
        for item in fired[INTERVAL + WINDOW]
        if item.metric_name == "training_rank_cpu_percent"
    )
    assert finding.category.value == "CPU"
    assert finding.recommended_action.value == "RUN_DIAGNOSTICS"
    assert finding.value == 0.0
    assert finding.diagnostic_parameters["processes"] == "1"
    assert (finding.job_id, finding.attempt_id) == ("job-ranks", "attempt-ranks")


@pytest.mark.parametrize("vcpus", [96, 192])
def test_notify005s_cpu_burners_never_trip_the_cpu_rule(tmp_path, monkeypatch, vcpus):
    """Eight burning processes that never touch CUDA hold every GPU idle: the
    GPU rule fires on each GPU, the CPU rule has no process to grade."""

    host = _Host(
        tmp_path, monkeypatch, vcpus=vcpus, cores=1.0, cuda=False, gpu_percent=0
    )

    fired, _ = _run(host)

    assert _low_cpu(fired) == {}
    gpu = {
        item.device
        for item in fired[WINDOW]
        if item.metric_name == "host_gpu_utilization_percent"
    }
    assert gpu == set(GPUS)


def test_an_idle_node_reads_no_rank_cpu_and_raises_nothing(tmp_path, monkeypatch):
    host = _Host(tmp_path, monkeypatch, vcpus=192, cores=0.0, cuda=False, gpu_percent=0)

    fired, host_cpu = _run(host, managed=False)

    assert not any(fired.values()), fired
    assert host_cpu and max(host_cpu) < 1


def test_the_cpu_rules_cuda_premise_is_quoted_verbatim_in_both_editions():
    """The premise -- a rank waiting on its GPU spins a core, a blocking-sync
    job need not -- is NVIDIA's documented behaviour, cited where it is used."""

    root = Path(__file__).resolve().parents[2]
    for edition in ("docs/详细设计.md", "docs/en/detailed-design.md"):
        text = " ".join((root / edition).read_text(encoding="utf-8").split())
        for value in (
            low_utilization.CUDA_SCHEDULING_REFERENCE,
            low_utilization.CUDA_SCHEDULING_DOCS_VERSION,
            *low_utilization.CUDA_SCHEDULING_SECTION,
            low_utilization.CUDA_SCHEDULE_AUTO_QUOTE,
            low_utilization.CUDA_SCHEDULE_BLOCKING_SYNC_QUOTE,
        ):
            assert value in text, f"{edition} does not cite {value!r}"
