"""Low CPU/GPU utilisation grades only the GPUs the managed attempt holds.

``LOW_GPU_UTILIZATION`` graded every GPU of a node while one managed attempt
was live there. A job holding one of eight GPUs (DESTR-012 group D, any
partial-node job in production) left seven unallocated GPUs that read 0 % by
definition, and after 300 s each of them was reported as "GPU idle while a
training workload is active". Only the GPUs the attempt's live containers on
the node hold are graded now -- by the GPU UUIDs the completion watcher reads
for each container, or by a declared ``nvidia.com/gpu`` count that covers every
GPU of the node -- and every other GPU is an idle sample that resets its window.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from gpu_fault.host_low_utilization import GpuOwnership, UnresolvedAllocationWarnings
from tests._builders import (
    attempt_observation,
    build_store,
    container_observation,
    host_telemetry_batch,
)
from tests.host_health._support import NOW

WINDOW = 300
NODE = "ip-10-0-0-1.us-west-2.compute.internal"
WORKLOAD = "gpu-fault-system/pytorchjob/gpu-fault-partial-node"
#: nvidia-smi ``--query-gpu=uuid`` spelling of a p5en node's eight GPUs.
NODE_GPUS = tuple(f"GPU-{index:08x}-11d3-5c96-fdb5-76df542ca80e" for index in range(8))
GPU = "host_gpu_utilization_percent"
CPU = "training_rank_cpu_percent"


def _container(rank, *, gpu_uuids=(), gpu_count=None, node=NODE):
    return container_observation(
        f"pod-{rank}",
        f"worker-{rank}",
        rank,
        node,
        gpu_uuids=list(gpu_uuids),
        gpu_count=len(gpu_uuids) if gpu_count is None else gpu_count,
    )


def _observe(store, offset, containers):
    store.save_attempt_observation(
        attempt_observation(
            "job-partial",
            "attempt-partial",
            NOW + timedelta(seconds=offset),
            started_at=NOW,
            expected_critical_ranks=len(containers),
            workload_ids=[WORKLOAD],
            containers=list(containers),
        )
    )


def _batch(offset, *, rank_cpu=(), node=NODE, gpus=NODE_GPUS):
    """Every GPU of the node at 0 %; ``rank_cpu`` GPUs report idle processes."""

    observed_at = NOW + timedelta(seconds=offset)
    return host_telemetry_batch(
        f"partial-{offset}",
        observed_at,
        [
            *(
                HostMetricSample(name=GPU, value=0.0, unit="percent", device=uuid)
                for uuid in gpus
            ),
            *(
                HostMetricSample(name=CPU, value=0.0, unit="percent", device=uuid)
                for uuid in rank_cpu
            ),
        ],
        node_id=node,
        workload_state="ACTIVE",
        affected_workload_ids=[WORKLOAD],
        received_at=observed_at,
    )


def _run(
    policy, store, containers, *, start=0, until=WINDOW + 60, step=30, rank_cpu=()
):
    """``{offset: findings}`` of a low node, the attempt refreshed per batch."""

    fired = {}
    for offset in range(start, until + 1, step):
        _observe(
            store, offset, containers(offset) if callable(containers) else containers
        )
        fired[offset] = policy.evaluate_metrics(_batch(offset, rank_cpu=rank_cpu))
    return fired


def _graded(fired):
    """``{(metric, device): first offset}`` over a run."""

    first = {}
    for offset, findings in sorted(fired.items()):
        for item in findings:
            first.setdefault((item.metric_name, item.device), offset)
    return first


def test_a_one_gpu_attempt_is_graded_on_the_gpu_it_holds_only():
    """DESTR-012 group D's shape: one GPU of eight, the node otherwise idle."""

    store = build_store()
    policy = NodeHealthPolicy(store)
    held = NODE_GPUS[3]

    fired = _run(policy, store, [_container(0, gpu_uuids=[held])], rank_cpu=[held])

    assert _graded(fired) == {(GPU, held): WINDOW, (CPU, held): WINDOW}, (
        "the attempt's unallocated GPUs were graded as idle training GPUs: "
        f"{sorted(_graded(fired))}"
    )
    finding = next(item for item in fired[WINDOW] if item.metric_name == GPU)
    assert (finding.job_id, finding.attempt_id) == ("job-partial", "attempt-partial")


def test_two_containers_holding_half_the_node_are_graded_on_their_four_gpus():
    store = build_store()
    policy = NodeHealthPolicy(store)
    containers = [
        _container(0, gpu_uuids=NODE_GPUS[0:2]),
        _container(1, gpu_uuids=NODE_GPUS[2:4]),
    ]

    fired = _run(policy, store, containers)

    assert _graded(fired) == {(GPU, uuid): WINDOW for uuid in NODE_GPUS[:4]}


def test_a_gpu_the_attempt_does_not_hold_is_an_idle_sample_that_resets_its_window():
    """Rank 1 moves onto GPU 5 at 240 s: GPU 5 is timed from then, not from 0."""

    store = build_store()
    policy = NodeHealthPolicy(store)

    def containers(offset):
        moved = [_container(1, gpu_uuids=[NODE_GPUS[5]])] if offset >= 240 else []
        return [_container(0, gpu_uuids=[NODE_GPUS[0]]), *moved]

    fired = _run(policy, store, containers, until=240 + WINDOW + 60)

    assert _graded(fired) == {
        (GPU, NODE_GPUS[0]): WINDOW,
        (GPU, NODE_GPUS[5]): 240 + WINDOW,
    }, f"GPU 5 carried low time from before the attempt held it: {_graded(fired)}"


def test_an_unresolved_allocation_grades_nothing_and_says_so_once(caplog):
    """A container declares one GPU but its UUID was never read (discovery
    failed): which of the eight it holds is unknown, so none is graded, and
    once the attempt has run a full window the operator is told."""

    store = build_store()
    policy = NodeHealthPolicy(store)
    containers = [_container(0, gpu_count=1)]

    def warnings():
        return [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]

    with caplog.at_level(logging.DEBUG, logger="gpu_fault.host_health"):
        early = _run(policy, store, containers, until=WINDOW - 30)
        assert warnings() == [], "warned while the watcher could still read the UUID"
        late = _run(policy, store, containers, start=WINDOW, until=WINDOW + 60)

    assert not any(early.values()) and not any(late.values()), (
        "an unresolved allocation was graded as if the attempt held every GPU"
    )
    assert len(warnings()) == 1, warnings()
    (warning,) = warnings()
    assert "declare 1 GPUs but list 0 GPU UUIDs" in warning, warning
    assert "attempt-partial" in warning and NODE in warning, warning
    ungraded = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG and "not graded" in record.getMessage()
    ]
    assert len(ungraded) == len(early) + len(late), (
        "expected one debug line per batch naming the ungraded GPUs, got "
        f"{len(ungraded)} for {len(early) + len(late)} batches"
    )
    assert "by none (declares 1, unresolved=True)" in ungraded[0], ungraded[0]


@pytest.mark.parametrize(
    "counts", [(8,), (4, 4)], ids=["one-full-node-container", "two-halves"]
)
def test_a_declared_count_that_covers_the_node_grades_every_gpu(counts, caplog):
    """No UUID was read, but the declared GPUs are all of the node's: the
    product requires Kubernetes allocatable to equal the host inventory, so
    the attempt holds every GPU (a full-node job keeps its coverage)."""

    store = build_store()
    policy = NodeHealthPolicy(store)
    containers = [
        _container(rank, gpu_count=count) for rank, count in enumerate(counts)
    ]

    with caplog.at_level(logging.WARNING, logger="gpu_fault.host_health"):
        fired = _run(policy, store, containers)

    assert _graded(fired) == {(GPU, uuid): WINDOW for uuid in NODE_GPUS}
    assert all(
        item.diagnostic_parameters["signal"] == "LOW_GPU_UTILIZATION"
        for item in fired[WINDOW]
    ), fired[WINDOW]
    assert not caplog.records, [record.getMessage() for record in caplog.records]


def test_a_container_that_declares_no_gpu_holds_none_and_is_not_a_gap(caplog):
    store = build_store()
    policy = NodeHealthPolicy(store)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.host_health"):
        fired = _run(policy, store, [_container(0, gpu_count=0)])

    assert not any(fired.values()), _graded(fired)
    assert not caplog.records, [record.getMessage() for record in caplog.records]


def test_gpu_uuids_compare_as_one_spelling():
    """``gpu-fault.io/gpu-uuids`` is free text; the collector's form wins."""

    store = build_store()
    policy = NodeHealthPolicy(store)
    held = NODE_GPUS[6]

    fired = _run(policy, store, [_container(0, gpu_uuids=[f"  {held.lower()} "])])

    assert _graded(fired) == {(GPU, held): WINDOW}


def test_a_terminated_container_holds_nothing_on_the_node():
    """Rank 0 still runs on GPU 0; rank 1 exited and released GPU 1."""

    store = build_store()
    policy = NodeHealthPolicy(store)
    containers = [
        _container(0, gpu_uuids=[NODE_GPUS[0]]),
        container_observation(
            "pod-1",
            "worker-1",
            1,
            NODE,
            gpu_uuids=[NODE_GPUS[1]],
            gpu_count=1,
            terminated=True,
            exit_code=0,
        ),
    ]

    fired = _run(policy, store, containers)

    assert _graded(fired) == {(GPU, NODE_GPUS[0]): WINDOW}


def test_a_healthy_reading_needs_no_explanation_held_or_not(caplog):
    """Only a breach is worth a debug line: busy GPUs, held or unheld, and a
    busy GPU on a node without any managed attempt log nothing."""

    store = build_store()
    policy = NodeHealthPolicy(store)

    def busy(offset):
        observed_at = NOW + timedelta(seconds=offset)
        return host_telemetry_batch(
            f"busy-{offset}",
            observed_at,
            [
                HostMetricSample(name=GPU, value=60.0, unit="percent", device=uuid)
                for uuid in NODE_GPUS[:2]
            ],
            node_id=NODE,
            workload_state="ACTIVE",
            affected_workload_ids=[WORKLOAD],
            received_at=observed_at,
        )

    with caplog.at_level(logging.DEBUG, logger="gpu_fault.host_health"):
        assert policy.evaluate_metrics(busy(0)) == []
        _observe(store, 30, [_container(0, gpu_uuids=[NODE_GPUS[0]])])
        assert policy.evaluate_metrics(busy(30)) == []

    assert caplog.records == [], [record.getMessage() for record in caplog.records]


def test_the_unresolved_warning_memory_is_bounded_and_forgets_the_oldest(caplog):
    warnings = UnresolvedAllocationWarnings(capacity=2)
    ownership = GpuOwnership(frozenset(), "none", 1, True)
    batch = _batch(WINDOW)

    def attempt(name):
        return attempt_observation(
            "job-partial",
            name,
            batch.observed_at,
            started_at=NOW,
            workload_ids=[WORKLOAD],
            containers=[_container(0, gpu_count=1)],
        )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.host_health"):
        logged = [
            warnings.note(batch, attempt(name), ownership, WINDOW)
            for name in ("a1", "a2", "a1", "a3", "a1")
        ]

    # a3 pushed a1 out (capacity 2), so a1 is new again and warns once more.
    assert logged == [True, True, False, True, True], logged
    assert len(caplog.records) == 4, [record.getMessage() for record in caplog.records]
