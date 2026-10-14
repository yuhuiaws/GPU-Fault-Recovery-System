"""The dispatch batch follows the region remediation budget (batch 88).

The dispatcher works the oldest ``batch_size`` open workflows a tick, so with a
region budget above the batch the selection -- not the budget -- bounds how many
remediations run at once. Unset, ``GPU_FAULT_WORKFLOW_BATCH_SIZE`` is derived as
max(100, region budget + 50), capped at 400 so a tick (batch 88's model: about
1 s + 0.15 s per workflow over 8 workers, 8.5 s at 400) stays under the 15-s
dispatch lease; startup refuses a region budget the batch cannot hold and warns
when a tick cannot finish inside the dispatch lease.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.config import (
    ProductionExecutorConfig,
    TimingConfigurationError,
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
    seed,
)

REGION = "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION"
BATCH = "GPU_FAULT_WORKFLOW_BATCH_SIZE"
NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("region", "batch"),
    [(1, 100), (20, 100), (50, 100), (51, 101), (150, 200), (350, 400), (900, 400)],
)
def test_the_batch_is_the_region_budget_plus_headroom_between_100_and_400(
    region: int, batch: int
) -> None:
    assert dispatch_batch_size(region) == batch
    config = WorkflowDispatcherConfig.from_mapping({REGION: str(region)}, True)
    assert config.batch_size == batch


def test_the_shipped_defaults_keep_a_batch_of_100() -> None:
    assert WorkflowDispatcherConfig.from_mapping({}, True).batch_size == 100
    assert validate_timing_from_environment({}) == []


def test_an_explicit_batch_size_is_kept_and_checked_against_the_budget() -> None:
    assert WorkflowDispatcherConfig.from_mapping({BATCH: "60"}, True).batch_size == 60
    assert validate_timing_from_environment({BATCH: "60"}) == []

    with pytest.raises(TimingConfigurationError) as refused:
        validate_timing_from_environment({BATCH: "60", REGION: "80"})
    assert (
        "the region remediation budget (80) exceeds the dispatch batch size (60)"
        in str(refused.value)
    )


def test_startup_refuses_a_region_budget_the_capped_batch_cannot_hold() -> None:
    assert validate_timing_from_environment({REGION: "400"}) == []
    with pytest.raises(TimingConfigurationError) as refused:
        validate_timing_from_environment({REGION: "401"})
    assert (
        "the region remediation budget (401) exceeds the dispatch batch size (400)"
        in str(refused.value)
    )


def test_a_tick_that_cannot_finish_inside_the_dispatch_lease_is_a_warning() -> None:
    executor = ProductionExecutorConfig.from_mapping({REGION: "350"})
    dispatcher = WorkflowDispatcherConfig.from_mapping({REGION: "350"}, True)

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


def test_a_region_budget_above_100_is_not_capped_by_the_selection() -> None:
    """150 workflows over 50 clusters, region budget 150: the derived batch of
    200 starts them all in one tick; the old 100-row cut started 100."""

    store = build_store()
    for index in range(150):
        seed(
            store,
            f"{index:03d}",
            cluster_id=f"cluster-{index % 50:03d}",
            created_at=NOW + timedelta(microseconds=index),
        )
    policy = RemediationBudgetPolicy(region_limit=150, resource_class_limit=5)
    batch = WorkflowDispatcherConfig.from_mapping({REGION: "150"}, True).batch_size
    dispatcher = budget_dispatcher(
        store, ScriptedAdapter(REBOOT_PLAN), policy, batch_size=batch
    )

    dispatcher.run_once()

    running = [w for w in all_workflows(store) if w.status is WorkflowStatus.RUNNING]
    assert len(running) == 150
