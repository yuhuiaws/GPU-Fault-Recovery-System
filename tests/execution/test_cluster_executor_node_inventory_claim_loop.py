"""The Kubernetes node inventory is ``ClaimLoop`` housekeeping, like the key sync.

Batch 98 replaced the executor's batch loop with ``ClaimLoop``, which runs the
due housekeeping (the spare sweep, the node-key sync) on every turn, whether or
not the turn claimed. Batch 102 added the node inventory report -- the
departed-node cleanup's Kubernetes evidence -- to the loop it replaced. Five
long restarts holding every slot claim nothing for as long as they run; the
report must still reach the control plane on its own schedule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.models import WorkflowOperation
from tests.execution.test_cluster_executor_claim_slots import (
    FastRenewalTick,
    GatedAdapter,
    LeasingControlPlane,
    eventually,
    running,
    slot_executor,
)
from tests.execution.test_cluster_executor_lease_and_report import remote_command
from tests.execution.test_cluster_executor_saturated_slots import ScheduledJob, Ticker


class InventoryReporter:
    """``NodeInventoryReporter.report``: due-gated, never raises into the loop."""

    def __init__(self, job: ScheduledJob) -> None:
        self.job = job

    def report(self) -> None:
        if self.job.due():
            self.job.run()


def test_the_node_inventory_reports_on_schedule_while_every_slot_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("gpu_fault.cluster_executor.lease.Event", FastRenewalTick)
    restarts = [
        remote_command(f"restart-{index}", operation=WorkflowOperation.RESTART_NODE)
        for index in range(5)
    ]
    client = LeasingControlPlane(restarts)
    adapter = GatedAdapter(*(command.command_id for command in restarts))
    ticker = Ticker()
    inventory = ScheduledJob(ticker, 60)
    executor = slot_executor(
        client,
        adapter,
        tmp_path,
        batch_size=5,
        max_concurrent_commands=5,
        lease_seconds=120,
        node_inventory=InventoryReporter(inventory),
    )

    with running(executor, adapter):
        eventually(
            lambda: len(adapter.started_ids()) == 5, "the five restarts never ran"
        )
        for minute in range(1, 4):
            ticker.seconds += 60
            eventually(
                lambda: len(inventory.runs) == 1 + minute,
                f"the node inventory missed minute {minute}: {inventory.runs}",
            )
        claims_while_busy = [claim.command_ids for claim in client.claims]

    assert inventory.runs == [0.0, 60.0, 120.0, 180.0], inventory.runs
    assert claims_while_busy == [tuple(command.command_id for command in restarts)], (
        "nothing may be claimed while every slot is busy"
    )
