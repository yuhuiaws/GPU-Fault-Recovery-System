"""Replay one XID through both of its reports, then let the dispatcher run it.

The kernel collector posts the NVRM line (``NVRM: Xid (PCI:0000:b9:00): ...``)
and DCGM field 230 changes on the exporter's next refresh, labelled
``pci_bus_id="00000000:B9:00.0"``: the same occurrence twice. A replay posts
both through the public ingestion routes (decision -> incident -> workflow) and
drives the production dispatcher and executor over a recording adapter, so
every gate between a workflow and a node -- the remediation budget's per-node
exclusion, the storm breaker, node-exclusive predecessors, the rule-A busy-node
hold, the executor's claim -- decides what really runs. The adapter answers
every step and holds each node-mutating one WAITING until released, so the
second report can land while the first remediation is in flight.

Node states: ``idle`` (the watcher's coverage heartbeat, nothing running),
``uncovered`` (no coverage: UNKNOWN, destructive plans compile BLOCKED),
``active`` (a managed attempt bound to the faulted GPU) and ``active-unbound``
(the node runs a job the faulted GPU cannot be bound to -- its pods use other
GPUs, or the watcher has not mapped their GPUs yet). Timings: ``in-flight``
(the first workflow is past its first node-mutating step when the second
report arrives -- the steady state, the gauge trails the line by 15-30 s),
``after-completion`` (the first remediation finished, the attempt restarted),
``before-dispatch`` (both reports land before any claim), ``late-second`` (the
second report arrives after the first one's correlation window closed).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.execution import (
    ProductionExecutorConfig,
    WorkflowDispatcherConfig,
    WorkflowStepOutcome,
)
from gpu_fault.gpu_fabric_manager import FABRIC_MANAGER_STATUS
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample, GpuMetricSource
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import WorkloadCoverageHeartbeat
from gpu_fault.watcher import WorkloadPhase
from tests._builders import asgi_client, attempt_observation, container_observation

CLUSTER = "cluster-a"
NODE = "node-a"
GPU = "GPU-h200-0"
OTHER_GPU = "GPU-h200-1"
EXPORTER_PCI = "00000000:B9:00.0"
KERNEL_PCI = "0000:b9:00"
WORKLOAD = "training/pytorchjob/train"
KERNEL = "KERNEL_LOG"
GAUGE = "DCGM_EXPORTER"

#: Every step that changes the node, the job or an operator's queue. The
#: node-mutating ones wait for the replay to release them.
ACTING = frozenset(
    {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.REPLACE_NODE,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESTART_WORKLOAD,
        WorkflowOperation.RECOVER_DRAIN_P2P,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESTART_FABRIC_MANAGER,
        WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
        WorkflowOperation.RESTART_VM,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.ESCALATE_SUPPORT,
        WorkflowOperation.CHECK_MECHANICALS,
    }
)
HELD = frozenset(
    {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESTART_WORKLOAD,
        WorkflowOperation.RECOVER_DRAIN_P2P,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.QUARANTINE,
    }
)


@dataclass(frozen=True)
class XidCase:
    xid: int
    text: str
    product: str = "NVIDIA H200"


#: One XID per class that compiles a workflow, with the kernel text its decode
#: reads. XID 45 has no NVSwitch inventory here, so it is the catalog's solo
#: RESTART_FM, not the graded-node observe-only decision.
CASES = {
    "48-reset-gpu": XidCase(
        48, "pid=1234, name=python, An uncorrectable double bit error"
    ),
    "79-restart-bm": XidCase(79, "GPU has fallen off the bus."),
    "13-restart-app": XidCase(
        13, "pid=1234, name=python, Graphics Exception: ESR 0x404600"
    ),
    "154-drain-p2p": XidCase(
        154, "GPU recovery action changed from 0x0 (None) to 0x3 (Drain P2P)"
    ),
    "154-reset": XidCase(
        154, "GPU recovery action changed from 0x0 (None) to 0x1 (GPU reset required)"
    ),
    "154-reboot": XidCase(
        154, "GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)"
    ),
    "74-reset": XidCase(
        74, "pid=1234, name=python, Link 3, 0x100 0x0 0x0 0x0 0x0 0x0 0x0"
    ),
    "45-restart-fm": XidCase(45, "pid=1234, name=python, Preemptive Channel Removal"),
    "54-check-mechanicals": XidCase(
        54, "Auxiliary power is not connected to the GPU board"
    ),
    "151-restart-vm": XidCase(151, "Key rotation error"),
    "78-update-swfw": XidCase(78, "vGPU start failed"),
    "159-not-applicable": XidCase(159, "CHECK_UVM"),
    "145-b200-nvlink5": XidCase(
        145,
        "RLW_RXPIPE Fatal XC0 i0 Link 03 (0x00000086 0x00000001)",
        product="NVIDIA B200",
    ),
}


class RecordingAdapter:
    """Answers every step; a node-mutating one waits until released."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, WorkflowOperation]] = []
        self.hold = True

    def supports(self, step) -> bool:
        return True

    def execute(self, context):
        operation = context.step.operation
        self.calls.append((context.workflow.request_id, context.step_index, operation))
        if operation in HELD and self.hold:
            return WorkflowStepOutcome.waiting(
                operation_id=f"held/{context.idempotency_key}"
            )
        return WorkflowStepOutcome.succeeded(operation_id=context.idempotency_key)

    def executions(self) -> Counter[WorkflowOperation]:
        """How many distinct steps (workflow, index) ran each acting operation."""

        return Counter(
            operation
            for _request, _index, operation in dict.fromkeys(self.calls)
            if operation in ACTING
        )


@dataclass
class Replay:
    context: ApplicationContext
    adapter: RecordingAdapter
    base: datetime
    post_errors: list[str] = field(default_factory=list)

    @property
    def store(self):
        return self.context.store

    def post(self, path: str, body: dict[str, Any]) -> None:
        async def call() -> None:
            async with asgi_client(self.context) as client:
                response = await client.post(path, json=body)
                if response.status_code != 200:
                    self.post_errors.append(f"{path}: {response.text}")

        asyncio.run(call())

    def close_windows(self, advance: timedelta = timedelta(minutes=5)) -> None:
        """Run the correlation loop as if ``advance`` had passed."""

        coordinator = self.context.xid_correlation
        original = coordinator.now
        coordinator.now = lambda: datetime.now(timezone.utc) + advance
        try:
            coordinator.run_once()
        finally:
            coordinator.now = original

    def dispatch(self, passes: int) -> None:
        for _ in range(passes):
            # The aggregation window has passed: nothing else moves the clock.
            for workflow in self.store.list_workflows(set(WorkflowStatus), limit=200):
                if workflow.not_before is not None and workflow.status in {
                    WorkflowStatus.PENDING,
                    WorkflowStatus.SAFETY_PENDING,
                }:
                    self.store.save_workflow(
                        workflow.model_copy(update={"not_before": None}),
                        expected=workflow,
                    )
            self.context.dispatcher.run_once()

    def finish(self) -> None:
        self.adapter.hold = False
        self.dispatch(8)

    def reports(self) -> dict[str, tuple[Any, Any, Any]]:
        """``event_source -> (event, decision, incident)``."""

        return {
            event.event_source: (
                event,
                self.store.get_xid_policy_decision(event.event_id),
                self.store.get_incident_by_event(event.event_id),
            )
            for event in self.store.list_xid_events(CLUSTER, NODE)
        }

    def incidents(self):
        return self.store.list_incidents_by_state(
            CLUSTER, set(IncidentState), node_ids={NODE}
        )

    def workflows(self):
        return self.store.list_workflows(set(WorkflowStatus), limit=200)


def new_replay(
    state: str, case: XidCase, *, graded: bool = False, store=None
) -> Replay:
    store = store if store is not None else InMemoryStore()
    adapter = RecordingAdapter()
    context = ApplicationContext(
        store=store,
        production_executor_config=ProductionExecutorConfig(
            enabled=True,
            executor_id="b237-executor",
            allowed_operations=frozenset(WorkflowOperation),
        ),
        production_adapters=[adapter],
        dispatcher_config=WorkflowDispatcherConfig(
            enabled=True, poll_interval_seconds=1, batch_size=50
        ),
    )
    store.save_profile(default_simulated_profile())
    replay = Replay(
        context, adapter, datetime.now(timezone.utc) - timedelta(seconds=90)
    )
    observe_node(replay, state)
    replay.post(GPU_METRICS_PATH, gauge_batch("baseline", replay.base, 0, case))
    if graded:
        replay.post(GPU_METRICS_PATH, fabric_batch(replay.base, case))
    return replay


def fabric_batch(at: datetime, case: XidCase) -> dict[str, Any]:
    """Eight GPUs report an NVLink fabric state: a graded NVSwitch node, where a
    solo XID 45 is observe-only (``orchestration/xid45_policy``)."""

    return GpuMetricBatch(
        batch_id="fabric",
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=at,
        source=GpuMetricSource.DCGM_EXPORTER,
        product=case.product,
        driver_branch=580,
        cuda_version="13.0",
        runtime_profile_version="simulated-v1",
        samples=[
            GpuMetricSample(
                metric_name="DCGM_FI_DEV_FABRIC_MANAGER_STATUS",
                canonical_name=FABRIC_MANAGER_STATUS,
                value=3.0,
                gpu_index=str(index),
                gpu_uuid=GPU if index == 0 else f"GPU-h200-{index}",
                labels={"modelName": case.product},
            )
            for index in range(8)
        ],
    ).model_dump(mode="json")


def observe_node(replay: Replay, state: str) -> None:
    at = replay.base
    if state == "idle":
        replay.context.topology.observe_coverage(
            WorkloadCoverageHeartbeat(
                cluster_id=CLUSTER,
                observed_at=at,
                watcher_instance="watcher-a",
                resource_version="1",
            )
        )
    elif state in {"active", "active-unbound"}:
        replay.store.save_attempt_observation(
            attempt(
                "train-a001",
                at,
                started_at=at - timedelta(minutes=30),
                gpu_uuids=[GPU] if state == "active" else [OTHER_GPU],
            )
        )
    else:
        assert state == "uncovered", state


def attempt(
    attempt_id: str,
    at: datetime,
    *,
    started_at: datetime,
    gpu_uuids: list[str],
    phase: WorkloadPhase = WorkloadPhase.RUNNING,
):
    terminated = phase is not WorkloadPhase.RUNNING
    return attempt_observation(
        "train",
        attempt_id,
        at,
        workload_phase=phase,
        started_at=started_at,
        containers=[
            container_observation(
                f"pod-{attempt_id}",
                "trainer-0",
                0,
                NODE,
                gpu_uuids=gpu_uuids,
                terminated=terminated,
                **({"exit_code": 1, "finished_at": at} if terminated else {}),
            )
        ],
        workload_ids=[WORKLOAD],
        restart_budget=3,
    )


def gauge_batch(
    batch_id: str, at: datetime, xid: float, case: XidCase
) -> dict[str, Any]:
    labels = {"gpu": "0", "UUID": GPU, "pci_bus_id": EXPORTER_PCI}
    return GpuMetricBatch(
        batch_id=batch_id,
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=at,
        source=GpuMetricSource.DCGM_EXPORTER,
        product=case.product,
        driver_branch=580,
        cuda_version="13.0",
        runtime_profile_version="simulated-v1",
        samples=[
            GpuMetricSample(
                metric_name=metric,
                canonical_name=canonical,
                value=value,
                gpu_index="0",
                gpu_uuid=GPU,
                pci_bdf=EXPORTER_PCI,
                labels=dict(labels),
            )
            for metric, canonical, value in (
                ("DCGM_FI_DEV_XID_ERRORS", "xid_last_error", xid),
                ("DCGM_FI_DEV_GPU_TEMP", "gpu_temperature_c", 40),
            )
        ],
    ).model_dump(mode="json")


def kernel_line(
    case: XidCase, at: datetime, *, record_id: str | None = None
) -> dict[str, Any]:
    return {
        "cluster_id": CLUSTER,
        "node_id": NODE,
        "record_id": record_id or f"kmsg-xid{case.xid}",
        "observed_at": at.isoformat(),
        "source_boot_id": "boot-a",
        "message": f"NVRM: Xid (PCI:{KERNEL_PCI}): {case.xid}, {case.text}",
        "product": case.product,
        "driver_branch": 580,
        "cuda_version": "13.0",
        "runtime_profile_version": "simulated-v1",
    }


def replay_both(
    case: XidCase,
    *,
    order: str,
    state: str,
    timing: str,
    graded: bool = False,
    store=None,
) -> Replay:
    """The kernel line and the field-230 change of one XID, 20 s apart, on
    ``store`` (an in-memory one by default)."""

    replay = new_replay(state, case, graded=graded, store=store)
    first_at, second_at = (
        replay.base + timedelta(seconds=10),
        replay.base + timedelta(seconds=30),
    )
    kernel_at, gauge_at = (
        (first_at, second_at) if order == "kernel-first" else (second_at, first_at)
    )
    kernel = (NVIDIA_KERNEL_PATH, kernel_line(case, kernel_at))
    gauge = (GPU_METRICS_PATH, gauge_batch("gauge", gauge_at, case.xid, case))
    first, second = (kernel, gauge) if order == "kernel-first" else (gauge, kernel)
    replay.post(*first)
    # Only windows due by the time the second report arrives close.
    replay.close_windows(
        timedelta(minutes=5) if timing == "late-second" else timedelta(seconds=20)
    )
    if timing == "in-flight":
        replay.dispatch(3)
    elif timing == "after-completion":
        replay.finish()
        if state.startswith("active"):
            # The first remediation restarted the job: a new attempt runs.
            restarted = replay.base + timedelta(seconds=60)
            gpus = [GPU] if state == "active" else [OTHER_GPU]
            replay.store.save_attempt_observation(
                attempt(
                    "train-a001",
                    restarted,
                    started_at=replay.base - timedelta(minutes=31),
                    gpu_uuids=gpus,
                    phase=WorkloadPhase.FAILED,
                )
            )
            replay.store.save_attempt_observation(
                attempt("train-a002", restarted, started_at=restarted, gpu_uuids=gpus)
            )
        replay.adapter.hold = True
    else:
        assert timing in {"before-dispatch", "late-second"}, timing
    replay.post(*second)
    replay.close_windows()
    replay.dispatch(3)
    replay.finish()
    return replay
