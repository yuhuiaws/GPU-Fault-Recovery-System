"""The read-only evidence gates of ``health_evidence_validation``, case by case.

A health workflow's last step carries the recovery contract it was compiled
with (``health_recovery_contract``). The OBSERVABILITY and TRAINING contracts
judge collector state and training progress; THERMAL_COOLDOWN asks for fresh
temperatures of every affected GPU spanning the cooldown. These tests drive
each gate with a real ``InMemoryStore`` (collector statuses, training
progress, attempt observations, health-signal states and GPU latest rows) and
real workflow records, and read the outcome a validation step would report.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.execution import WorkflowStepContext
from gpu_fault.gpu_metrics import (
    GpuMetricLatest,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
)
from gpu_fault.health_evidence_validation import (
    cooldown_key,
    reset_unproven_cooldown,
    special_health_validation,
    thermal_cooldown,
)
from gpu_fault.health_recovery_contract import (
    COOLDOWN_MAX_GAP_SECONDS,
    COOLDOWN_SECONDS,
    HEALTH_CONTRACT,
)
from gpu_fault.host_diagnostics import HOST_DIAGNOSTIC_FAILURE
from gpu_fault.models import (
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from gpu_fault.training_models import TrainingProgressHeartbeat
from tests._builders import (
    attempt_observation,
    container_observation,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

CLUSTER = "cluster-a"
JOB = "job-a"
ATTEMPT = "attempt-a"
NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
DETECTED = NOW - timedelta(minutes=10)


def step_context(
    contract: Any,
    *,
    node_ids: tuple[str, ...] = ("node-a",),
    gpu_uuids: tuple[str, ...] = (),
    step_executions: tuple[Any, ...] = (),
    created_at: datetime = DETECTED,
) -> WorkflowStepContext:
    """The context the validation adapter hands a gate for a workflow's step 0."""

    step = workflow_step(
        WorkflowOperation.VALIDATE_GPU,
        "gpu-validation",
        node_ids=list(node_ids),
        gpu_uuids=list(gpu_uuids),
        parameters={HEALTH_CONTRACT: contract},
    )
    workflow = workflow_request(
        "workflow-a",
        "incident-a",
        status=WorkflowStatus.RUNNING,
        official_steps=[step],
        step_executions=list(step_executions),
        created_at=created_at,
        updated_at=created_at,
    )
    incident = fault_incident(
        "incident-a", "event-a", "NODE_HEALTH", node_ids=list(node_ids), job_id=JOB
    )
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="workflow-a/0/VALIDATE_GPU",
    )


def pending_reason(outcome: Any) -> str:
    assert outcome is not None, "the gate must answer, not fall through"
    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details[HOST_DIAGNOSTIC_FAILURE] is True, outcome.details
    reason: str = outcome.details["health_evidence_pending"]
    return reason


# ------------------------------------------------------------ contract parsing


@pytest.mark.parametrize(
    "contract",
    [
        "OBSERVABILITY",
        {"mode": "OBSERVABILITY", "detected_at": "2026-11-01T11:50:00"},
        {"mode": "TRAINING", "attempt_id": ATTEMPT, "rank": 0},
        {"mode": "TRAINING", "detected_at": 17, "attempt_id": ATTEMPT, "rank": 0},
        {
            "mode": "TRAINING",
            "detected_at": DETECTED.isoformat(),
            "attempt_id": ATTEMPT,
            "rank": "first",
        },
        {
            "mode": "OBSERVABILITY",
            "detected_at": DETECTED.isoformat(),
            "collectors": ["NOT_A_COLLECTOR"],
        },
    ],
    ids=[
        "not-a-mapping",
        "naive-event-time",
        "no-event-time",
        "event-time-not-text",
        "rank-not-a-number",
        "unknown-collector",
    ],
)
def test_a_malformed_contract_waits_as_invalid_instead_of_raising(contract) -> None:
    outcome = special_health_validation(step_context(contract), InMemoryStore(), NOW)

    assert pending_reason(outcome) == "invalid recovery contract"


def test_an_observability_contract_without_collectors_has_nothing_to_prove() -> None:
    contract = {
        "mode": "OBSERVABILITY",
        "detected_at": DETECTED.isoformat(),
        "collectors": [],
    }

    outcome = special_health_validation(step_context(contract), InMemoryStore(), NOW)

    assert pending_reason(outcome) == "observation scope is empty"


def _status(kind: CollectorKind, node: str, *, sample_count: int) -> CollectorStatus:
    seen = NOW - timedelta(seconds=30)
    return CollectorStatus(
        cluster_id=CLUSTER,
        node_id=node,
        collector=kind,
        observed_at=seen,
        ingested_at=seen,
        last_success_at=seen,
        sample_count=sample_count,
    )


def test_a_clean_gpu_scrape_with_no_samples_is_an_empty_observation() -> None:
    """A fresh, error-free GPU_METRICS batch that carried no sample says nothing
    about the GPUs; the host collector's clean batch is evidence on its own."""

    store = InMemoryStore()
    store.save_collector_status(
        _status(CollectorKind.GPU_METRICS, "node-a", sample_count=0)
    )
    store.save_collector_status(
        _status(CollectorKind.HOST_TELEMETRY, "node-a", sample_count=0)
    )
    contract = {
        "mode": "OBSERVABILITY",
        "detected_at": DETECTED.isoformat(),
        "collectors": ["GPU_METRICS", "HOST_TELEMETRY"],
    }

    outcome = special_health_validation(step_context(contract), store, NOW)

    assert pending_reason(outcome) == "collector state is UNKNOWN"
    assert outcome.details["missing_observations"] == {
        "node-a/GPU_METRICS": "EMPTY_GPU_OBSERVATION"
    }, "only the GPU channel needs samples"

    store.save_collector_status(
        _status(CollectorKind.GPU_METRICS, "node-a", sample_count=8).model_copy(
            update={"observed_at": NOW - timedelta(seconds=20)}
        )
    )
    healed = special_health_validation(step_context(contract), store, NOW)
    assert healed.status is WorkflowStepStatus.SUCCEEDED, healed
    assert healed.details["observation_state"] == "HEALTHY"


# ------------------------------------------------------------ training gates


def _progress(
    store: InMemoryStore,
    rank: int,
    observed_at: datetime,
    *,
    node: str = "node-a",
    step: int | None = 10,
    **values: Any,
) -> None:
    store.observe_training_progress(
        TrainingProgressHeartbeat(
            cluster_id=CLUSTER,
            attempt_id=ATTEMPT,
            rank=rank,
            observed_at=observed_at,
            node_id=node,
            pod_uid=f"pod-{rank}",
            container_name="trainer",
            step=step,
            **values,
        )
    )


def _training(**values: Any) -> dict[str, Any]:
    return {
        "mode": "TRAINING",
        "detected_at": DETECTED.isoformat(),
        "attempt_id": ATTEMPT,
        "rank": 0,
        **values,
    }


def test_training_without_progress_of_the_original_rank_waits() -> None:
    store = InMemoryStore()
    _progress(store, 1, NOW - timedelta(seconds=5))

    outcome = special_health_validation(step_context(_training()), store, NOW)

    assert pending_reason(outcome) == "missing progress for the original attempt/rank"


@pytest.mark.parametrize(
    ("values", "contract_step"),
    [
        ({"numerical_error": True}, None),
        ({"loss": float("inf")}, None),
        ({"step": None}, None),
        ({}, 10),
    ],
    ids=["numerical-error", "non-finite-loss", "no-step", "step-not-past-contract"],
)
def test_training_progress_that_is_not_valid_application_progress_waits(
    values, contract_step
) -> None:
    store = InMemoryStore()
    _progress(store, 0, NOW - timedelta(seconds=5), **values)

    outcome = special_health_validation(
        step_context(_training(step=contract_step)), store, NOW
    )

    assert pending_reason(outcome) == (
        "original rank has not made valid application progress"
    )


def test_fresh_progress_of_the_original_rank_validates_a_stall() -> None:
    store = InMemoryStore()
    _progress(store, 0, NOW - timedelta(seconds=5), step=11)

    outcome = special_health_validation(step_context(_training(step=10)), store, NOW)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert outcome.adapter_operation_id == "workflow-a/0/VALIDATE_GPU"
    assert outcome.details == {
        "validated_nodes": ["node-a"],
        "validation": "VALIDATE_GPU",
        "validated_attempt": ATTEMPT,
        "validated_rank": 0,
        "validated_step": 11,
    }


def _hang_context(*, expected_rank_count: Any = None) -> WorkflowStepContext:
    survey = workflow_step_execution(
        0,
        WorkflowOperation.COLLECT_HUNG_SUMMARY,
        details={"hung_survey": {"expected_rank_count": expected_rank_count}},
    )
    return step_context(
        _training(kind="hang"), node_ids=("node-a", "node-b"), step_executions=(survey,)
    )


def _allocation(
    store: InMemoryStore,
    observed_at: datetime,
    *,
    ranks: tuple[int, ...] = (0, 1),
    expected: int = 2,
    terminated: tuple[int, ...] = (),
) -> None:
    store.save_attempt_observation(
        attempt_observation(
            JOB,
            ATTEMPT,
            observed_at,
            expected_critical_ranks=expected,
            containers=[
                container_observation(
                    f"pod-{rank}",
                    f"trainer-{rank}",
                    rank,
                    "node-a" if rank == 0 else "node-b",
                    **(
                        {"terminated": True, "exit_code": 137}
                        if rank in terminated
                        else {}
                    ),
                )
                for rank in ranks
            ],
        )
    )


def _hang_outcome(store: InMemoryStore, **context: Any) -> Any:
    _progress(store, 0, NOW - timedelta(seconds=5), step=11)
    return special_health_validation(_hang_context(**context), store, NOW)


def test_a_hang_without_the_attempts_allocation_cannot_be_judged() -> None:
    outcome = _hang_outcome(InMemoryStore())

    assert pending_reason(outcome) == "attempt progress coverage is incomplete"
    assert outcome.details["missing_progress"] == ["allocation_unknown"]


def test_a_hang_on_a_stale_allocation_cannot_be_judged() -> None:
    store = InMemoryStore()
    _allocation(store, NOW - timedelta(seconds=121))

    outcome = _hang_outcome(store)

    assert outcome.details["missing_progress"] == ["allocation_stale"]


@pytest.mark.parametrize(
    "allocation",
    [{"terminated": (0, 1)}, {"ranks": (0,), "expected": 2}],
    ids=["every-critical-container-terminated", "fewer-ranks-than-expected"],
)
def test_a_hang_on_an_incomplete_allocation_cannot_be_judged(allocation) -> None:
    store = InMemoryStore()
    _allocation(store, NOW - timedelta(seconds=10), **allocation)

    outcome = _hang_outcome(store)

    assert outcome.details["missing_progress"] == ["allocation_incomplete"]


def test_the_surveyed_rank_count_names_every_rank_without_fresh_progress() -> None:
    """The hung survey counted three ranks; the allocation shows two. Rank 1
    has a heartbeat but no progress since detection, rank 2 none at all."""

    store = InMemoryStore()
    _allocation(store, NOW - timedelta(seconds=10))
    _progress(store, 1, DETECTED - timedelta(seconds=30), node="node-b")

    outcome = _hang_outcome(store, expected_rank_count=3)

    assert pending_reason(outcome) == "attempt progress coverage is incomplete"
    assert outcome.details["missing_progress"] == ["1", "2"]


def test_without_a_surveyed_count_every_allocated_rank_must_report() -> None:
    store = InMemoryStore()
    _allocation(store, NOW - timedelta(seconds=10))

    outcome = _hang_outcome(store, expected_rank_count="2")

    assert outcome.details["missing_progress"] == ["1"], (
        "a count that is not an integer is ignored; rank 1 never reported"
    )


def test_every_surveyed_rank_progressing_validates_the_hang() -> None:
    store = InMemoryStore()
    _allocation(store, NOW - timedelta(seconds=10))
    _progress(store, 1, NOW - timedelta(seconds=4), node="node-b", step=11)

    outcome = _hang_outcome(store, expected_rank_count=2)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert outcome.details["validated_rank"] == 0
    assert outcome.details["validated_step"] == 11


# ------------------------------------------------------------ thermal cooldown

GPU_A = "GPU-a"
GPU_B = "GPU-b"


def _thermal(**values: Any) -> dict[str, Any]:
    return {"mode": "THERMAL_COOLDOWN", "detected_at": DETECTED.isoformat(), **values}


def _temperature(
    store: InMemoryStore,
    node: str,
    gpu: str,
    observed_at: datetime,
    *,
    name: str = "gpu_temperature_c",
    value: float = 45.0,
) -> None:
    sample = GpuMetricSample(
        metric_name=name, canonical_name=name, value=value, gpu_uuid=gpu
    )
    store.observe_gpu_metrics(
        [
            (
                (CLUSTER, node, gpu, name),
                GpuMetricLatest(
                    cluster_id=CLUSTER,
                    node_id=node,
                    observed_at=observed_at,
                    source=GpuMetricSource.DCGM_EXPORTER,
                    sample=sample,
                ),
            )
        ]
    )


def _owned(mapping: dict[str, list[str]]):
    calls: list[str] = []

    def resolve(context: Any, node: str, latest: list[Any]) -> list[str]:
        calls.append(node)
        return mapping[node]

    return resolve, calls


def _cooldown_state(store: InMemoryStore, context: Any, node: str) -> Any:
    return store.get_health_signal_state(cooldown_key(context, node))


def test_a_cooldown_without_gpu_identity_waits() -> None:
    metrics = GpuMetricsService(store=InMemoryStore())
    resolve, calls = _owned({})

    outcome = thermal_cooldown(step_context(_thermal()), metrics, NOW, resolve)

    assert pending_reason(outcome) == "thermal scope has no GPU identity"
    assert calls == [], "no ownership is resolved without a GPU to own"


def test_a_cooldown_whose_gpus_no_node_owns_waits() -> None:
    metrics = GpuMetricsService(store=InMemoryStore())
    resolve, calls = _owned({"node-a": [GPU_A]})

    outcome = thermal_cooldown(
        step_context(_thermal(), gpu_uuids=(GPU_A, GPU_B)), metrics, NOW, resolve
    )

    assert pending_reason(outcome) == "thermal GPU ownership is incomplete"
    assert calls == ["node-a"]


def test_a_cooldown_restarts_while_a_gpu_lacks_a_fresh_temperature() -> None:
    """GPU-b's only temperature is older than the workflow: the node's
    cooldown is reset, and node-b, which owns none of the GPUs, is not asked."""

    store = InMemoryStore()
    metrics = GpuMetricsService(store=store)
    context = step_context(
        _thermal(), node_ids=("node-a", "node-b"), gpu_uuids=(GPU_A, GPU_B)
    )
    key = cooldown_key(context, "node-a")
    store.claim_health_signal_transition(key, True, NOW - timedelta(seconds=60))
    _temperature(store, "node-a", GPU_A, NOW - timedelta(seconds=10))
    _temperature(store, "node-a", GPU_B, DETECTED - timedelta(seconds=1))
    resolve, _calls = _owned({"node-a": [GPU_A, GPU_B], "node-b": []})

    outcome = thermal_cooldown(context, metrics, NOW, resolve)

    assert pending_reason(outcome) == "thermal cooldown not complete"
    assert outcome.details["node_pending"] == {
        "node-a": "missing fresh temperature for every affected GPU"
    }
    assert _cooldown_state(store, context, "node-a").active is False, (
        "an unproven node restarts its cooldown"
    )
    assert _cooldown_state(store, context, "node-b") is None


def test_a_cooldown_waits_for_a_sample_newer_than_the_last_one_counted() -> None:
    store = InMemoryStore()
    metrics = GpuMetricsService(store=store)
    context = step_context(_thermal(), gpu_uuids=(GPU_A,))
    seen = NOW - timedelta(seconds=20)
    store.claim_health_signal_transition(cooldown_key(context, "node-a"), True, seen)
    _temperature(store, "node-a", GPU_A, seen, name="gpu_tlimit_c", value=30)
    resolve, _calls = _owned({"node-a": [GPU_A]})

    outcome = thermal_cooldown(context, metrics, NOW, resolve)

    assert outcome.details["node_pending"] == {
        "node-a": "waiting for a new healthy sample"
    }
    assert _cooldown_state(store, context, "node-a").active is True, (
        "the counted sample is not withdrawn"
    )


def test_a_cooldown_completes_once_healthy_samples_span_the_cooldown() -> None:
    store = InMemoryStore()
    metrics = GpuMetricsService(store=store)
    context = step_context(_thermal(), gpu_uuids=(GPU_A,))
    resolve, _calls = _owned({"node-a": [GPU_A]})
    first = NOW - timedelta(seconds=COOLDOWN_SECONDS + 5)
    _temperature(store, "node-a", GPU_A, first)

    started = thermal_cooldown(context, metrics, first + timedelta(seconds=1), resolve)

    assert started.details["node_pending"] == {
        "node-a": f"healthy observations must span {COOLDOWN_SECONDS}s"
    }
    assert _cooldown_state(store, context, "node-a").active_since == first

    _temperature(store, "node-a", GPU_A, NOW - timedelta(seconds=5), value=44)

    assert thermal_cooldown(context, metrics, NOW, resolve) is None, (
        "a cooldown spanning its duration lets the validation go on"
    )


def test_a_gap_between_healthy_samples_restarts_the_cooldown() -> None:
    store = InMemoryStore()
    metrics = GpuMetricsService(store=store)
    context = step_context(_thermal(), gpu_uuids=(GPU_A,))
    key = cooldown_key(context, "node-a")
    resolve, _calls = _owned({"node-a": [GPU_A]})
    latest = NOW - timedelta(seconds=5)
    store.claim_health_signal_transition(
        key, True, latest - timedelta(seconds=COOLDOWN_MAX_GAP_SECONDS + 1)
    )
    _temperature(store, "node-a", GPU_A, latest, name="memory_temperature_c")

    outcome = thermal_cooldown(context, metrics, NOW, resolve)

    assert outcome.details["node_pending"] == {
        "node-a": f"healthy observations must span {COOLDOWN_SECONDS}s"
    }
    state = _cooldown_state(store, context, "node-a")
    assert state.observed_at == latest, state
    assert not state.active or state.active_since == latest, (
        "the old active run is not continued across the gap"
    )


def test_unproven_nodes_restart_their_cooldown_and_only_theirs() -> None:
    store = InMemoryStore()
    context = step_context(
        _thermal(), node_ids=("node-a", "node-b", "node-c"), gpu_uuids=(GPU_A,)
    )
    started = NOW - timedelta(seconds=90)
    for node in ("node-a", "node-b", "node-c"):
        store.claim_health_signal_transition(cooldown_key(context, node), True, started)

    reset_unproven_cooldown(
        store,
        context,
        NOW,
        {"node-a": {"gpu_temperature_c": ["GPU-a"]}},
        {"node-b": ["temperature above limit"]},
    )

    assert {
        node: _cooldown_state(store, context, node).active
        for node in ("node-a", "node-b", "node-c")
    } == {"node-a": False, "node-b": False, "node-c": True}

    other = step_context({"mode": "OBSERVABILITY"}, node_ids=("node-c",))
    reset_unproven_cooldown(store, other, NOW, {"node-c": {}}, {})
    assert _cooldown_state(store, context, "node-c").active is True, (
        "a contract that is not a thermal cooldown resets nothing"
    )
