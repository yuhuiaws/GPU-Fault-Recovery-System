"""A multi-node VALIDATE_GPU asks each node only for the GPUs it owns.

Live on HyperPod (GF-REGIONAL-COLLECT-016): a merge widened a reset drill's
VALIDATE_GPU from the reset node to a second node of the job, carrying the
union of GPU UUIDs. The per-GPU presence rule asked the second node to report
a GPU it never had, the step stayed WAITING for the whole 600 s per-step cap
and failed into a reboot ladder for both nodes. The single-node rule is
unchanged: every GPU the step names must report; a multi-node step binds each
GPU to the node that owns it (``gpu_uuids_by_node`` when carried, else the GPUs
the node has ever reported).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpu_fault.adapters import GpuValidationAdapter
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepStatus,
)
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import build_store, copy_model, workflow_step_execution
from tests.execution._support import workflow_state

RESET_NODE = "node-a"
SIBLING = "node-b"


class _Metrics:
    """Per-node GPU telemetry: ``samples[node] = {gpu: observed_at}``."""

    def __init__(self, reset_at: datetime, samples: dict[str, dict[str, datetime]]):
        self.samples = samples
        reset_at_plus = reset_at + timedelta(seconds=5)

        class _Store:
            def list_collector_statuses(self, cluster_id, node_id):
                return [
                    CollectorStatus(
                        cluster_id=cluster_id,
                        node_id=node_id,
                        collector=CollectorKind.GPU_METRICS,
                        observed_at=reset_at_plus,
                        ingested_at=reset_at_plus,
                        last_success_at=reset_at_plus,
                    )
                ]

            def list_telemetry_metrics_latest(self, cluster_id, node_id):
                return []

        self.store = _Store()

    def latest(self, cluster_id, node_id):
        return [
            SimpleNamespace(
                observed_at=observed_at,
                sample=SimpleNamespace(
                    canonical_name="gpu_temperature_c", gpu_uuid=gpu
                ),
            )
            for gpu, observed_at in self.samples.get(node_id, {}).items()
        ]

    def findings(self, cluster_id, node_id):
        return []


def _context(store, reset_at, *, node_ids, gpu_uuids, parameters=None):
    incident, workflow = workflow_state(
        store, [WorkflowOperation.RESET_GPU, WorkflowOperation.VALIDATE_GPU]
    )
    workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(0, WorkflowOperation.RESET_GPU, updated_at=reset_at)
        ],
    )
    step = copy_model(
        workflow.official_steps[1],
        execution_owner="gpu-fault-validation-adapter",
        node_ids=list(node_ids),
        gpu_uuids=list(gpu_uuids),
        parameters=dict(parameters or {}),
    )
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=1,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="validation/multi-node",
    )


def _adapter(metrics: _Metrics) -> GpuValidationAdapter:
    return GpuValidationAdapter(metrics, store=metrics.store)


def test_each_node_of_a_widened_validation_answers_for_its_own_gpus() -> None:
    reset_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    fresh = reset_at + timedelta(seconds=10)
    metrics = _Metrics(
        reset_at, {RESET_NODE: {"GPU-a": fresh}, SIBLING: {"GPU-b": fresh}}
    )
    context = _context(
        build_store(),
        reset_at,
        node_ids=[RESET_NODE, SIBLING],
        gpu_uuids=["GPU-a", "GPU-b"],
    )

    outcome = _adapter(metrics).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.details
    assert outcome.details["validated_nodes"] == [RESET_NODE, SIBLING]


def test_a_node_that_owns_none_of_the_named_gpus_validates_node_wide() -> None:
    """The live shape: the sibling owns GPU-c and reports it; the step names
    only the reset node's GPU. The sibling is not asked for GPU-a."""

    reset_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    fresh = reset_at + timedelta(seconds=10)
    metrics = _Metrics(
        reset_at, {RESET_NODE: {"GPU-a": fresh}, SIBLING: {"GPU-c": fresh}}
    )
    context = _context(
        build_store(), reset_at, node_ids=[RESET_NODE, SIBLING], gpu_uuids=["GPU-a"]
    )

    outcome = _adapter(metrics).execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.details


def test_a_gpu_the_node_once_reported_is_still_required_after_the_reset() -> None:
    """A GPU that fell off the bus keeps its last (old) sample; the node that
    owns it must still report it fresh, as before."""

    reset_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    fresh = reset_at + timedelta(seconds=10)
    metrics = _Metrics(
        reset_at,
        {
            RESET_NODE: {"GPU-a": fresh, "GPU-x": reset_at - timedelta(hours=1)},
            SIBLING: {"GPU-b": fresh},
        },
    )
    context = _context(
        build_store(),
        reset_at,
        node_ids=[RESET_NODE, SIBLING],
        gpu_uuids=["GPU-a", "GPU-x", "GPU-b"],
    )

    outcome = _adapter(metrics).execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["pending_nodes"] == [RESET_NODE]
    assert outcome.details["node_pending"][RESET_NODE][
        "missing_recent_metrics_by_gpu"
    ] == {"GPU-x": ["gpu_temperature_c"]}


def test_the_per_node_mapping_wins_over_inferred_ownership() -> None:
    reset_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    fresh = reset_at + timedelta(seconds=10)
    metrics = _Metrics(
        reset_at, {RESET_NODE: {"GPU-a": fresh}, SIBLING: {"GPU-b": fresh}}
    )
    context = _context(
        build_store(),
        reset_at,
        node_ids=[RESET_NODE, SIBLING],
        gpu_uuids=["GPU-a", "GPU-b"],
        parameters={"gpu_uuids_by_node": {RESET_NODE: ["GPU-a", "GPU-b"]}},
    )

    outcome = _adapter(metrics).execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["pending_nodes"] == [RESET_NODE], (
        "the mapping says the reset node owns both GPUs; it owes GPU-b's sample"
    )


def test_a_single_node_step_still_requires_every_gpu_it_names() -> None:
    reset_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    fresh = reset_at + timedelta(seconds=10)
    metrics = _Metrics(reset_at, {RESET_NODE: {"GPU-a": fresh}})
    context = _context(
        build_store(), reset_at, node_ids=[RESET_NODE], gpu_uuids=["GPU-a", "GPU-b"]
    )

    outcome = _adapter(metrics).execute(context)

    assert outcome.status is WorkflowStepStatus.WAITING, (
        "unchanged: a single node answers for every GPU the step names"
    )
    assert outcome.details["node_pending"][RESET_NODE][
        "missing_recent_metrics_by_gpu"
    ] == {"GPU-b": ["gpu_temperature_c"]}
