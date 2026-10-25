"""VALIDATE_HOST / VALIDATE_FABRIC judge the interfaces the node still reports.

Live 2026-10-25 (plain EKS ip-10-0-43-5, GF-GENERAL-LIVE-008 interface_missing):
the required test veth pair was deleted, the collector window that had
declared it required was closed, and the validated restore ended FAILED on
the pair's last reading -- ``network_link_down=1`` / ``network_link_up=0``,
written by the collector while the window was open, still inside the
validation window -- although the collector no longer reported either end.
"""

from __future__ import annotations

from typing import Any

from tests._builders import build_store, copy_model

from ._support import (
    CollectorKind,
    CollectorStatus,
    GpuValidationAdapter,
    SimpleNamespace,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepContext,
    WorkflowStepStatus,
    datetime,
    pytest,
    timedelta,
    timezone,
    workflow_state,
)

TEST_LINK = "gfrsbf9076b"


def _rows(at: datetime, *, link_at: datetime, link_down: int) -> list[Any]:
    """The node's latest-value rows: the newest batch at ``at``, the test
    interface's last link reading at ``link_at``."""

    current = {
        "load1_per_cpu": 0.1,
        "memory_used_percent": 10,
        "filesystem_used_percent": 20,
    }
    rows = [
        SimpleNamespace(name=name, value=value, observed_at=at, device=None)
        for name, value in current.items()
    ]
    rows.append(
        SimpleNamespace(
            name="network_link_up", value=1, observed_at=at, device="enp71s0"
        )
    )
    rows += [
        SimpleNamespace(
            name="network_link_down",
            value=link_down,
            observed_at=link_at,
            device=TEST_LINK,
        ),
        SimpleNamespace(
            name="network_link_up",
            value=1 - link_down,
            observed_at=link_at,
            device=TEST_LINK,
        ),
    ]
    return rows


class _Store:
    def __init__(self, batch_at: datetime, rows: list[Any]) -> None:
        self.batch_at = batch_at
        self.rows = rows

    def list_collector_statuses(self, cluster_id, node_id):
        return [
            CollectorStatus(
                cluster_id=cluster_id,
                node_id=node_id,
                collector=collector,
                observed_at=self.batch_at,
                ingested_at=self.batch_at,
                last_success_at=self.batch_at,
            )
            for collector in (CollectorKind.GPU_METRICS, CollectorKind.HOST_TELEMETRY)
        ]

    def list_telemetry_metrics_latest(self, cluster_id, node_id):
        return list(self.rows)


class _Metrics:
    def __init__(self, store: _Store) -> None:
        self.store = store

    def latest(self, cluster_id, node_id):
        return [
            SimpleNamespace(
                observed_at=self.store.batch_at,
                sample=SimpleNamespace(canonical_name=name),
            )
            for name in (
                "nvlink_crc_aggregate_error_total",
                "nvlink_recovery_aggregate_error_total",
                "nvlink_replay_aggregate_error_total",
            )
        ]

    def findings(self, cluster_id, node_id):
        return []


def _validate(operation: WorkflowOperation, rows_store: _Store):
    store = build_store()
    incident, workflow = workflow_state(store, [operation])
    step = copy_model(
        workflow.official_steps[0], execution_owner="gpu-fault-validation-adapter"
    )
    metrics = _Metrics(rows_store)
    return GpuValidationAdapter(metrics, store=rows_store).execute(
        WorkflowStepContext(
            workflow=copy_model(workflow, official_steps=[step]),
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key=f"validation/interface-scope/{operation.value}",
        )
    )


OPERATIONS = [WorkflowOperation.VALIDATE_HOST, WorkflowOperation.VALIDATE_FABRIC]


@pytest.mark.parametrize("operation", OPERATIONS, ids=lambda item: item.value)
def test_an_interface_the_newest_batch_no_longer_reports_does_not_fail_it(
    operation: WorkflowOperation,
) -> None:
    batch_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    store = _Store(
        batch_at, _rows(batch_at, link_at=batch_at - timedelta(seconds=90), link_down=1)
    )

    outcome = _validate(operation, store)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, (
        f"{operation.value} judged {TEST_LINK}'s last reading, which the node's "
        f"newest complete host batch no longer carries: {outcome}"
    )


@pytest.mark.parametrize("operation", OPERATIONS, ids=lambda item: item.value)
def test_an_interface_the_newest_batch_reads_down_still_fails_it(
    operation: WorkflowOperation,
) -> None:
    batch_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    store = _Store(batch_at, _rows(batch_at, link_at=batch_at, link_down=1))

    outcome = _validate(operation, store)

    expected = (
        "network_link_down"
        if operation is WorkflowOperation.VALIDATE_HOST
        else "network_link_up"
    )
    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert expected in outcome.details["node_failures"]["node-a"], outcome.details


def test_an_out_of_scope_reading_never_stands_in_for_a_missing_batch() -> None:
    # The scope is the newest *complete* batch: while the host collector's
    # newest batch carries a collection error the validation keeps waiting,
    # whatever the older link rows say.
    batch_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    store = _Store(
        batch_at, _rows(batch_at, link_at=batch_at - timedelta(seconds=90), link_down=1)
    )
    errored = [
        copy_model(
            status,
            errors=["_network: HostObservationError: cannot list /sys/class/net"],
            last_error_at=batch_at,
        )
        for status in store.list_collector_statuses("cluster-a", "node-a")
    ]
    store.list_collector_statuses = lambda cluster_id, node_id: errored

    outcome = _validate(WorkflowOperation.VALIDATE_HOST, store)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_pending"]["node-a"] == {
        "missing_or_stale_collectors": ["HOST_TELEMETRY"]
    }, outcome.details
