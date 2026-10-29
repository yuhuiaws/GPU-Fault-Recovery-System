"""The dispatch batch follows the dispatcher's capacity (batch 88's tick model).

The dispatcher works the oldest ``batch_size`` open workflows a tick. Unset,
``GPU_FAULT_WORKFLOW_BATCH_SIZE`` is the rows the dispatcher's workers work in
half the dispatch lease -- about 1 s + 0.15 s per workflow over the workers,
the 8.5 s the live site took for 400 on 8 -- between 100 and 400. No
remediation budget enters it: no fixed number caps how many remediations run
at once (each cluster's storm breaker bounds a correlated fault), so the
retired region budget no longer moves it. Startup warns when a tick cannot
finish inside the dispatch lease.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.config import (
    ProductionExecutorConfig,
    WorkflowDispatcherConfig,
    dispatch_batch_size,
    validate_timing_from_environment,
    validate_timing_relationships,
)
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.models import WorkflowStatus
from tests._builders import build_store
from tests.execution._budget_hold_support import (
    REBOOT_PLAN,
    ScriptedAdapter,
    all_workflows,
    budget_dispatcher,
    large_fleet,
    seed,
)

RETIRED_REGION = "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION"
BATCH = "GPU_FAULT_WORKFLOW_BATCH_SIZE"
WORKERS = "GPU_FAULT_WORKFLOW_DISPATCHER_WORKERS"
LEASE = "GPU_FAULT_WORKFLOW_DISPATCH_LEASE_SECONDS"
NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("workers", "lease", "batch"),
    [
        (8, 15, 346),
        (4, 15, 173),
        (2, 15, 100),
        (1, 15, 100),
        (16, 15, 400),
        (8, 30, 400),
        (8, 0, 346),
    ],
)
def test_the_batch_is_what_the_workers_work_in_half_the_lease(
    workers: int, lease: int, batch: int
) -> None:
    assert dispatch_batch_size(workers, lease) == batch
    mapping = {WORKERS: str(workers), LEASE: str(lease)}
    assert WorkflowDispatcherConfig.from_mapping(mapping, True).batch_size == batch


def test_the_shipped_defaults_derive_a_batch_of_346_inside_the_lease() -> None:
    assert WorkflowDispatcherConfig.from_mapping({}, True).batch_size == 346
    assert validate_timing_from_environment({}) == []


def test_the_retired_region_budget_moves_nothing() -> None:
    for region in ("1", "350", "900"):
        config = WorkflowDispatcherConfig.from_mapping({RETIRED_REGION: region}, True)
        assert config.batch_size == 346, region
        assert validate_timing_from_environment({RETIRED_REGION: region}) == []
    assert validate_timing_from_environment({BATCH: "60", RETIRED_REGION: "80"}) == []


def test_an_explicit_batch_size_is_kept() -> None:
    assert WorkflowDispatcherConfig.from_mapping({BATCH: "60"}, True).batch_size == 60
    assert validate_timing_from_environment({BATCH: "60"}) == []


def test_a_tick_that_cannot_finish_inside_the_dispatch_lease_is_a_warning() -> None:
    executor = ProductionExecutorConfig.from_mapping({})
    dispatcher = WorkflowDispatcherConfig.from_mapping({BATCH: "400"}, True)

    def check(config: WorkflowDispatcherConfig) -> list[str]:
        return validate_timing_relationships(
            executor, config, verify_max_attempts=60, branch_max_rungs=2
        )

    assert check(dispatcher) == [], "400 rows on 8 workers: about 8.5 s, inside 15 s"
    (warning,) = check(replace(dispatcher, max_workers=4))
    assert warning.startswith(
        "warning: a dispatch tick of 400 workflows on 4 workers takes about 16.0s"
    ), warning
    assert (
        check(replace(dispatcher, max_workers=4, cycle_deadline_seconds=10.0)) == []
    ), "a cycle deadline under the lease bounds the tick"
    derived = WorkflowDispatcherConfig.from_mapping({WORKERS: "4"}, True)
    assert (derived.batch_size, check(derived)) == (173, []), (
        "a derived batch stays inside the lease"
    )
    (floor,) = check(WorkflowDispatcherConfig.from_mapping({WORKERS: "1"}, True))
    assert floor.startswith("warning: a dispatch tick of 100 workflows on 1 workers"), (
        "the floor keeps the selection and says the tick no longer fits"
    )


@pytest.mark.usefixtures(large_fleet.__name__)
def test_a_storm_of_150_workflows_starts_in_one_tick() -> None:
    """150 workflows over 50 clusters: the derived batch of 346 starts them all
    in one tick; the region-derived cut of 100 started 100. A fleet of 1000
    nodes per cluster keeps each cluster's breaker (200 nodes) out of reach."""

    store = build_store()
    for index in range(150):
        seed(
            store,
            f"{index:03d}",
            cluster_id=f"cluster-{index % 50:03d}",
            created_at=NOW + timedelta(microseconds=index),
        )
    batch = WorkflowDispatcherConfig.from_mapping({}, True).batch_size
    dispatcher = budget_dispatcher(
        store, ScriptedAdapter(REBOOT_PLAN), RemediationBudgetPolicy(), batch_size=batch
    )

    dispatcher.run_once()

    running = [w for w in all_workflows(store) if w.status is WorkflowStatus.RUNNING]
    assert len(running) == 150
