"""The inventory gate fails a node on the row its erroring host collector delivered.

Live 2026-10-27 (GF-GENERAL-LIVE-013 ``post_repair_failure``, R13g, plain EKS
ip-10-0-43-5 10:18-10:33Z and HyperPod 07:08-07:51Z): REMEDIATE_EFA_DRIVER
rebound the function (SUCCEEDED 10:19:46.7Z), the harness trap re-unbound it
4 s later (10:19:50.3Z), RESTART_EFA_DEVICE_PLUGIN had passed on the plugin's
unchanged allocatable (10:19:48.2Z), TRIGGER_HEALTH_SNAPSHOT restarted both
collectors (10:20:01.7Z) and VALIDATE_FABRIC waited out the 600 s cap
(10:30:06.2Z) with ``node_pending = {node: {missing_or_stale_collectors:
[HOST_TELEMETRY]}}``. The restarted host collector's batches carried
``efa_inventory_active_count=15`` from its first tick on, but every one of them
also carried ``_rdma``'s "incomplete EFA traffic coverage: 15/16 devices", so
HOST_TELEMETRY stayed COLLECTION_ERROR and the gate read no row; the
REBOOT_NODE the plan asked for was withheld as missing evidence and the node
went to an operator cordoned.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters import GpuValidationAdapter
from gpu_fault.adapters.inventory_verdict import inventory_contributor_error
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.validation_evidence import (
    validation_without_observed_failure,
    withheld_escalation,
)
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import build_store, copy_model, workflow_step_execution
from tests.execution._support import workflow_state

NODE = "node-a"
RDMA_COVERAGE_ERROR = (
    "_rdma: HostObservationError: incomplete EFA traffic coverage: 15/16 devices"
)
GPU_QUERY_ERROR = (
    "_gpu_inventory: CollectorError: GPU inventory query failed: Unable to "
    "determine the device handle for GPU 0000:07:00.0: Unknown Error"
)
EFA_CHAIN = [
    WorkflowOperation.REMEDIATE_EFA_DRIVER,
    WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
    WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
    WorkflowOperation.VALIDATE_FABRIC,
]
EFA_REQUIREMENT = {
    NODE: {
        "metrics": {"efa_inventory_active_count": 16, "gpu_inventory_active_count": 8},
        "node_instance_type": "p5en.48xlarge",
    }
}


class _Clock:
    """The live chain's instants, relative to ``now``."""

    def __init__(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.last_clean_batch = self.now - timedelta(seconds=60)
        self.rebound_at = self.now - timedelta(seconds=20)
        self.plugin_at = self.now - timedelta(seconds=18)
        self.snapshot_at = self.now - timedelta(seconds=5)
        # The restarted collector's first tick: after the last recovery step.
        self.batch_at = self.now - timedelta(seconds=3)


def _row(name: str, value: float, at: datetime, device: str | None = None) -> Any:
    return SimpleNamespace(name=name, value=value, observed_at=at, device=device)


class _Store:
    def __init__(self, statuses: list[CollectorStatus], rows: list[Any]) -> None:
        self.statuses = statuses
        self.rows = rows

    def list_collector_statuses(self, cluster_id, node_id):
        return list(self.statuses)

    def list_telemetry_metrics_latest(self, cluster_id, node_id):
        return list(self.rows)


class _Metrics:
    def __init__(self, store: _Store) -> None:
        self.store = store

    def latest(self, cluster_id, node_id):
        return []

    def findings(self, cluster_id, node_id):
        return []


def _status(
    collector: CollectorKind,
    *,
    observed_at: datetime,
    last_success_at: datetime | None,
    last_error_at: datetime | None = None,
    errors: list[str] | None = None,
) -> CollectorStatus:
    return CollectorStatus(
        cluster_id="cluster-a",
        node_id=NODE,
        collector=collector,
        observed_at=observed_at,
        ingested_at=observed_at,
        last_success_at=last_success_at,
        last_error_at=last_error_at,
        errors=list(errors or []),
    )


def _erroring_host(clock: _Clock, error: str = RDMA_COVERAGE_ERROR) -> CollectorStatus:
    """The live status shape: the newest batch erroring since the restart, the
    last clean one from before the unbind (sticky ``last_success_at``)."""

    return _status(
        CollectorKind.HOST_TELEMETRY,
        observed_at=clock.batch_at,
        last_success_at=clock.last_clean_batch,
        last_error_at=clock.batch_at,
        errors=[error],
    )


def _clean(collector: CollectorKind, at: datetime) -> CollectorStatus:
    return _status(collector, observed_at=at, last_success_at=at)


def _efa_rows(clock: _Clock, active: int, at: datetime | None = None) -> list[Any]:
    at = at or clock.batch_at
    return [
        _row("efa_inventory_expected_count", 16, at),
        _row("efa_inventory_active_count", active, at),
        _row("gpu_inventory_active_count", 8, at),
        _row("network_link_up", 1, at, "enp71s0"),
    ]


def _validate(
    operations: list[WorkflowOperation],
    executions: list[tuple[WorkflowOperation, datetime]],
    requirement: dict[str, Any],
    store: _Store,
):
    """Run the chain's validation (its last step) over ``store``."""

    state_store = build_store()
    incident, workflow = workflow_state(state_store, operations)
    index = len(operations) - 1
    validation_step = copy_model(
        workflow.official_steps[index],
        execution_owner="gpu-fault-validation-adapter",
        parameters={
            "failure_escalation_action": "REBOOT_NODE",
            "inventory_requirements_by_node": requirement,
        },
    )
    workflow = copy_model(
        workflow,
        official_steps=[*workflow.official_steps[:index], validation_step],
        step_executions=[
            workflow_step_execution(position, operation, updated_at=at)
            for position, (operation, at) in enumerate(executions)
        ],
    )
    outcome = GpuValidationAdapter(_Metrics(store), store=store).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=validation_step,
            step_index=index,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="validation/inventory-despite-collection-error",
        )
    )
    return workflow, index, outcome


def _efa_chain(clock: _Clock) -> list[tuple[WorkflowOperation, datetime]]:
    return [
        (WorkflowOperation.REMEDIATE_EFA_DRIVER, clock.rebound_at),
        (WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN, clock.plugin_at),
        (WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT, clock.snapshot_at),
    ]


def test_post_repair_failure_is_a_verdict_read_from_the_erroring_batch() -> None:
    clock = _Clock()
    store = _Store(
        [_clean(CollectorKind.GPU_METRICS, clock.now), _erroring_host(clock)],
        _efa_rows(clock, active=15),
    )

    workflow, index, outcome = _validate(
        EFA_CHAIN, _efa_chain(clock), EFA_REQUIREMENT, store
    )

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    details = outcome.details
    assert details["failed_nodes"] == [NODE]
    assert details["node_failures"] == {
        NODE: ["efa_inventory_active_count=15,expected=16"]
    }
    evidence = details["inventory_evidence"][NODE]["efa_inventory_active_count"]
    assert evidence["value"] == 15 and evidence["expected"] == 16
    assert evidence["observed_at"] == clock.batch_at.isoformat()
    assert details["host_collector_errors"] == {NODE: [RDMA_COVERAGE_ERROR]}
    assert "node_pending" not in details
    # The ladder reads it as a verdict: the plan's REBOOT_NODE is owed, not
    # withheld as missing evidence.
    assert validation_without_observed_failure(details) is False
    execution = WorkflowStepExecution(
        step_index=index,
        operation=WorkflowOperation.VALIDATE_FABRIC,
        status=WorkflowStepStatus.FAILED,
        details=details,
        error=outcome.error,
    )
    assert withheld_escalation(workflow, execution) is None


def test_an_erroring_batch_that_reads_the_expected_count_still_waits() -> None:
    # A batch with a collection error never passes a validation: the rows it
    # lacks would read as clean. Only a failure is read from it.
    clock = _Clock()
    store = _Store(
        [_clean(CollectorKind.GPU_METRICS, clock.now), _erroring_host(clock)],
        _efa_rows(clock, active=16),
    )

    _, _, outcome = _validate(EFA_CHAIN, _efa_chain(clock), EFA_REQUIREMENT, store)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_pending"] == {
        NODE: {"missing_or_stale_collectors": ["HOST_TELEMETRY"]}
    }
    assert "host_collector_errors" not in outcome.details


def test_a_count_from_before_the_last_recovery_step_is_no_verdict() -> None:
    # The failing row predates the rebind: it proves nothing about the repair
    # (inventory_observation_window), erroring collector or not.
    clock = _Clock()
    store = _Store(
        [_clean(CollectorKind.GPU_METRICS, clock.now), _erroring_host(clock)],
        _efa_rows(clock, active=15, at=clock.rebound_at - timedelta(seconds=1)),
    )

    _, _, outcome = _validate(EFA_CHAIN, _efa_chain(clock), EFA_REQUIREMENT, store)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_pending"] == {
        NODE: {"missing_or_stale_collectors": ["HOST_TELEMETRY"]}
    }


def test_an_unreadable_inventory_source_stays_unknown() -> None:
    # The inventory contributor itself failed (an unreadable sysfs tree raises
    # out of ``_efa_inventory`` with no samples): the batch carries no fresh
    # count, and the stale one from before the action is not read either.
    clock = _Clock()
    store = _Store(
        [
            _clean(CollectorKind.GPU_METRICS, clock.now),
            _erroring_host(
                clock,
                error=(
                    "_efa_inventory: CollectorError: EFA inventory observation "
                    "unavailable: [Errno 13] Permission denied"
                ),
            ),
        ],
        _efa_rows(clock, active=15, at=clock.last_clean_batch),
    )

    _, _, outcome = _validate(EFA_CHAIN, _efa_chain(clock), EFA_REQUIREMENT, store)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_pending"] == {
        NODE: {"missing_or_stale_collectors": ["HOST_TELEMETRY"]}
    }


def test_a_partial_gpu_query_is_not_a_verdict_but_a_coexisting_error_is() -> None:
    # Post-reboot VALIDATE_GPU. A ``_gpu_inventory`` error ships the GPUs that
    # still answered with the error (DRIVER_QUERY_FAILED, F-H6): the count was
    # not read in full, so no verdict until a clean batch. The same 7/8 beside
    # another contributor's error is the node's own count.
    clock = _Clock()
    chain = [WorkflowOperation.RESTART_NODE, WorkflowOperation.VALIDATE_GPU]
    executions = [(WorkflowOperation.RESTART_NODE, clock.rebound_at)]
    requirement = {NODE: {"metrics": {"gpu_inventory_active_count": 8}}}
    rows = [
        _row("gpu_inventory_expected_count", 8, clock.batch_at),
        _row("gpu_inventory_active_count", 7, clock.batch_at),
    ]

    _, _, partial = _validate(
        chain,
        executions,
        requirement,
        _Store(
            [
                _clean(CollectorKind.GPU_METRICS, clock.now),
                _erroring_host(clock, error=GPU_QUERY_ERROR),
            ],
            rows,
        ),
    )
    assert partial.status is WorkflowStepStatus.WAITING, partial
    assert partial.details["node_pending"] == {
        NODE: {"missing_or_stale_collectors": ["HOST_TELEMETRY"]}
    }

    _, _, coexisting = _validate(
        chain,
        executions,
        requirement,
        _Store(
            [
                _clean(CollectorKind.GPU_METRICS, clock.now),
                _erroring_host(
                    clock, error="_smart: HostObservationError: smartctl scan failed"
                ),
            ],
            rows,
        ),
    )
    assert coexisting.status is WorkflowStepStatus.FAILED, coexisting
    assert coexisting.details["node_failures"] == {
        NODE: ["gpu_inventory_active_count=7,expected=8"]
    }


def test_a_stale_erroring_status_is_no_verdict() -> None:
    # The newest batch, erroring or not, must be within the post-action age:
    # a collector that stopped reporting proves nothing about the node now.
    clock = _Clock()
    stale_at = clock.now - timedelta(minutes=7)
    rebound_at = clock.now - timedelta(minutes=8)
    store = _Store(
        [
            _clean(CollectorKind.GPU_METRICS, clock.now),
            _status(
                CollectorKind.HOST_TELEMETRY,
                observed_at=stale_at,
                last_success_at=clock.now - timedelta(minutes=9),
                last_error_at=stale_at,
                errors=[RDMA_COVERAGE_ERROR],
            ),
        ],
        _efa_rows(clock, active=15, at=stale_at),
    )

    _, _, outcome = _validate(
        EFA_CHAIN,
        [
            (WorkflowOperation.REMEDIATE_EFA_DRIVER, rebound_at),
            (WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN, rebound_at),
            (WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT, rebound_at),
        ],
        EFA_REQUIREMENT,
        store,
    )

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_pending"] == {
        NODE: {"missing_or_stale_collectors": ["HOST_TELEMETRY"]}
    }


def test_a_dark_gpu_collector_does_not_hide_a_fresh_host_inventory_failure() -> None:
    # The host collector is clean and fresh; DCGM is the stale one. The EFA
    # count is the host collector's reading, so the node fails on it instead of
    # waiting for the exporter until the cap withholds the escalation.
    clock = _Clock()
    store = _Store(
        [
            _clean(CollectorKind.GPU_METRICS, clock.now - timedelta(minutes=10)),
            _clean(CollectorKind.HOST_TELEMETRY, clock.batch_at),
        ],
        _efa_rows(clock, active=15),
    )

    _, _, outcome = _validate(EFA_CHAIN, _efa_chain(clock), EFA_REQUIREMENT, store)

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details["node_failures"] == {
        NODE: ["efa_inventory_active_count=15,expected=16"]
    }
    assert "host_collector_errors" not in outcome.details


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RDMA_COVERAGE_ERROR, False),
        ("_smart: HostObservationError: smartctl scan failed", False),
        ("rejected-event:host-telemetry/abc", False),
        (GPU_QUERY_ERROR, True),
        ("_efa_inventory: CollectorError: EFA inventory observation unavailable", True),
    ],
)
def test_inventory_contributor_errors_are_told_by_their_prefix(
    error: str, expected: bool
) -> None:
    assert inventory_contributor_error(error) is expected
