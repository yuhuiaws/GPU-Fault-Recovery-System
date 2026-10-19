"""The node half of the low CPU/GPU utilisation rules.

``training_rank_cpu_percent`` is the CPU the processes holding a compute context
on one GPU used since the last tick, in percent of one core: nvidia-smi's
compute-app query names the GPUs of every process, ``/proc/<pid>/stat`` its CPU
time. And each GPU's own crossing of the threshold is a delivery edge, whatever
the collector's static ``GPU_FAULT_WORKLOAD_STATE`` says (no deploy path sets
it): the control plane grades only the GPUs a managed attempt holds, so one of
them recovering while the unallocated GPUs stay at 0 % has to reach it.
"""

from __future__ import annotations

import pytest

from gpu_fault.channel_registry import channel_for_path
from gpu_fault.collectors import HostTelemetryCollector
from tests.collectors import _cov95_runtime_collect as common
from tests.collectors import _cov95_runtime_collect_host as support
from tests.collectors._support import write_fake_rank

isolated_runtime = common.isolated_runtime
host_case = support.host_case

COMPUTE_APPS = (
    "nvidia-smi",
    "--query-compute-apps=pid,gpu_uuid",
    "--format=csv,noheader,nounits",
)
GPU_QUERY = HostTelemetryCollector.GPU_QUERY_ARGV
#: No health summary falls due inside these few ticks (its lead is 5 % of it).
EDGES_ONLY = {"edge_filter_enabled": True, "health_summary_seconds": 3600}


def _rank_cpu(batch):
    return {
        sample.device: (sample.value, sample.labels["processes"])
        for sample in batch.samples
        if sample.name == "training_rank_cpu_percent"
    }


def _ranks(case, cores, seconds):
    """``/proc/<pid>/stat`` of ranks that burned ``cores[pid]`` cores so far."""

    for pid, value in cores.items():
        write_fake_rank(
            case.roots.proc,
            pid,
            cpu_ticks=round(100 * value * seconds),
            write_bytes=0,
            wchar=0,
        )


def test_rank_cpu_is_counted_on_every_gpu_a_process_drives(host_case):
    host_case.outputs[COMPUTE_APPS] = (
        # Two processes on GPU-a, one process on two GPUs, one line from a
        # driver that printed no GPU column, one GPU field nvidia-smi left N/A.
        "11, GPU-a\n22, GPU-b\n22, GPU-c\n33, GPU-a\n44\n55, [N/A]\n66, MIG-m\n",
        "",
        0,
    )
    cores = {11: 1.0, 22: 2.0, 33: 0.5, 44: 1.0, 55: 1.0, 66: 0.25}
    collector = host_case.build("_rank_liveness")
    _ranks(host_case, cores, 0)
    first = collector.collect_once()
    host_case.clock.sleep(15)
    _ranks(host_case, cores, 15)
    second = collector.collect_once()

    assert _rank_cpu(first) == {}, "a 0 was reported before any CPU baseline"
    assert _rank_cpu(second) == {
        "GPU-a": (pytest.approx(150.0), "2"),
        "GPU-b": (pytest.approx(200.0), "1"),
        "GPU-c": (pytest.approx(200.0), "1"),
        "MIG-m": (pytest.approx(25.0), "1"),
    }
    values = support.values(second)
    assert values[("training_rank_process_count", None)] == 6
    assert values[("training_rank_cpu_ticks_delta", None)] == pytest.approx(
        100 * 15 * sum(cores.values())
    )


def test_a_gpu_whose_process_has_no_baseline_yet_gets_no_sample(host_case):
    host_case.outputs[COMPUTE_APPS] = ("11, GPU-a\n", "", 0)
    collector = host_case.build("_rank_liveness")
    _ranks(host_case, {11: 1.0}, 0)
    collector.collect_once()
    host_case.clock.sleep(15)
    host_case.outputs[COMPUTE_APPS] = ("11, GPU-a\n22, GPU-b\n", "", 0)
    _ranks(host_case, {11: 1.0, 22: 0.0}, 15)

    batch = collector.collect_once()

    assert _rank_cpu(batch) == {"GPU-a": (pytest.approx(100.0), "1")}


def _utilization(case, readings):
    case.outputs[GPU_QUERY] = (
        "".join(f"{uuid}, {value}\n" for uuid, value in readings.items()),
        "",
        0,
    )


def _deliveries(case, collector, timeline):
    """Edge reasons of the batch each tick delivered (``None``: not delivered)."""

    delivered = []
    for readings in timeline:
        _utilization(case, readings)
        before = len(case.sink.requests)
        batch = collector.collect_once()
        delivered.append(
            sorted(batch.edge_filter_reasons)
            if len(case.sink.requests) > before
            else None
        )
        case.clock.sleep(15)
    return delivered


@pytest.mark.parametrize("declared", ["UNKNOWN", "ACTIVE"])
def test_each_gpu_crossing_is_an_edge_while_the_unallocated_gpus_stay_idle(
    host_case, declared
):
    """GPU-a is the attempt's and goes idle, then busy again; GPU-b is
    unallocated and reads 0 % throughout. The deploy leaves UNKNOWN."""

    collector = host_case.build("_gpu_utilization", **EDGES_ONLY)
    collector.context = common.collector_context(workload_state=declared)

    delivered = _deliveries(
        host_case,
        collector,
        [
            {"GPU-a": 60, "GPU-b": 0},
            {"GPU-a": 0, "GPU-b": 0},
            {"GPU-a": 0, "GPU-b": 0},
            {"GPU-a": 60, "GPU-b": 0},
        ],
    )

    assert delivered == [
        ["baseline"],
        ["sustained:host_gpu_utilization_percent:GPU-a"],
        None,
        ["recovered"],
    ], delivered


def test_an_idle_nodes_collector_start_still_posts_a_routine_baseline(host_case):
    """Idle GPUs read low from the first tick: a state, not an edge. A
    baseline carrying their reasons would be evidence-class traffic on every
    collector start (a fleet's Agent wave), not a routine spoolable batch."""

    collector = host_case.build("_gpu_utilization", **EDGES_ONLY)
    collector.context = common.collector_context(workload_state="UNKNOWN")
    _utilization(host_case, {"GPU-a": 0, "GPU-b": 0})

    collector.collect_once()

    [(path, payload)] = host_case.sink.requests
    assert payload["edge_filter_reasons"] == ["baseline"], payload
    assert channel_for_path(path).is_routine_payload(payload), payload


def test_an_explicitly_idle_node_ships_no_low_utilization_edge(host_case):
    collector = host_case.build("_gpu_utilization", **EDGES_ONLY)
    collector.context = common.collector_context(workload_state="IDLE")

    delivered = _deliveries(
        host_case, collector, [{"GPU-a": 60}, {"GPU-a": 0}, {"GPU-a": 60}]
    )

    assert delivered == [["baseline"], None, None], delivered


def test_idle_rank_cpu_is_an_edge_and_the_host_wide_reading_no_longer_is(host_case):
    host_case.outputs[COMPUTE_APPS] = ("11, GPU-a\n", "", 0)
    collector = host_case.build("_cpu", "_rank_liveness", **EDGES_ONLY)
    collector.context = common.collector_context(workload_state="ACTIVE")
    support.write_file(host_case.roots.proc / "loadavg", "0.10 0.10 0.10 1/100 1\n")
    reasons = []
    for second, busy_seconds in ((0, 0), (15, 15), (30, 15), (45, 30)):
        # The host reads 0 % busy throughout; the rank burns one core for
        # 15 s, sleeps for 15 s, and burns again.
        support.write_file(
            host_case.roots.proc / "stat",
            f"cpu 0 0 0 {200 * (second + 1)} 0 0 0 0 0 0\n",
        )
        write_fake_rank(
            host_case.roots.proc,
            11,
            cpu_ticks=100 * busy_seconds,
            write_bytes=0,
            wchar=0,
        )
        before = len(host_case.sink.requests)
        batch = collector.collect_once()
        reasons.append(
            sorted(batch.edge_filter_reasons)
            if len(host_case.sink.requests) > before
            else None
        )
        host_case.clock.sleep(15)

    assert reasons == [
        ["baseline"],
        None,
        ["sustained:training_rank_cpu_percent:GPU-a"],
        ["recovered"],
    ], reasons
    assert all(
        "cpu_usage_percent" not in reason for reason in sum(filter(None, reasons), [])
    ), f"the host-wide CPU reading is still an edge: {reasons}"
