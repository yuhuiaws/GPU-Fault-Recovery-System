"""E2E software regressions; physical GPU and cluster acceptance is separate."""

from __future__ import annotations

import pytest

from gpu_fault.cluster_scope import FABRIC_MANAGER_UNIT
from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.node_agent import NodeActionStatus
from tests.execution._xid_recovery_e2e_support import RecoveryRig


@pytest.fixture
def rig(tmp_path):
    value = RecoveryRig(tmp_path)
    try:
        yield value
    finally:
        value.close()


@pytest.mark.parametrize(
    ("xid", "driver", "cuda"),
    [(95, 550, "12.4"), (95, None, None), (95, 575, "12.9"), (48, 550, "12.4")],
    ids=["old-version", "missing-version", "linkage-boundary", "xid48-old-version"],
)
def test_linkage_does_not_block_original_xid_recovery(rig, xid, driver, cuda):
    decision = rig.ingest(rig.payload(xid=xid, driver_branch=driver, cuda_version=cuda))

    assert decision["disposition"] == "EXECUTABLE", decision
    assert decision["action"] == "RESET_GPU", decision
    result = rig.execute(decision)

    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert rig.hardware.trace.count("gpu-reset") == 1, rig.hardware.trace
    assert rig.hardware.trace.count(Op.RESTART_WORKLOAD.value) == 1
    assert (
        rig.store.get_incident(decision["incident_id"]).state is IncidentState.RECOVERED
    )


@pytest.mark.parametrize(
    "action",
    ["None", "Reset", "Drain and Reset", "Drain P2P"],
    ids=["none", "reset", "drain-and-reset", "still-drain-p2p"],
)
def test_drain_p2p_orders_release_query_recovery_and_readmission(rig, action):
    rig.hardware.recovery_action = action
    rig.hardware.stop_ready = False
    payload = rig.payload()
    decision = rig.ingest(payload)

    waiting = rig.execute(decision)
    assert waiting.status is WorkflowStatus.RUNNING, waiting
    assert rig.envelopes == [], "no node action may precede confirmed workload stop"

    rig.hardware.stop_ready = True
    rig.hardware.validation_ready = False
    waiting = rig.execute(decision)
    assert waiting.status is WorkflowStatus.RUNNING, waiting
    trace = rig.hardware.trace
    assert trace.index(Op.MARK_UNSCHEDULABLE.value) < trace.index("persistence-off")
    assert trace.index("persistence-off") < trace.index("recovery-query")
    assert trace.count("gpu-reset") == (0 if action == "None" else 1), trace
    if action != "None":
        assert trace.index("recovery-query") < trace.index("gpu-reset")
    assert trace.index("recovery-query") < trace.index("persistence-restored")
    assert Op.RESTORE_SCHEDULING.value not in trace, trace
    assert Op.RESTART_WORKLOAD.value not in trace, trace
    assert rig.hardware.persistence["GPU-b"] == "Enabled"
    assert not any(
        command[:2] in (["systemctl", "stop"], ["systemctl", "restart"])
        and command[2] == FABRIC_MANAGER_UNIT
        for command in rig.hardware.commands
    ), "per-GPU recovery must keep Fabric Manager running"

    replay = rig.ingest(payload)
    assert replay["workflow_request_id"] == decision["workflow_request_id"]
    envelope = next(
        e for e in rig.envelopes if e.command.operation is Op.RECOVER_DRAIN_P2P
    )
    before = list(rig.hardware.commands)
    assert rig.agent.execute(envelope).status is NodeActionStatus.SUCCEEDED
    assert rig.hardware.commands == before, (
        "ledger replay must not query or reset again"
    )

    rig.hardware.validation_ready = True
    result = rig.execute(decision)
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert trace.index(Op.VALIDATE_FABRIC.value) < trace.index(
        Op.RESTORE_SCHEDULING.value
    )
    assert trace.count(Op.RESTART_WORKLOAD.value) == 1
    assert rig.hardware.persistence["GPU-a"] == "Enabled"
    assert (
        rig.store.get_incident(decision["incident_id"]).state is IncidentState.RECOVERED
    )


def test_drain_p2p_reboot_is_escalated_without_gpu_reset(rig):
    rig.hardware.recovery_action = "Reboot"
    decision = rig.ingest(rig.payload())
    result = rig.execute(decision)
    if result.status is WorkflowStatus.RUNNING:
        result = rig.execute(decision)

    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert "gpu-reset" not in rig.hardware.trace, rig.hardware.trace
    assert rig.hardware.trace.count(Op.RESTART_NODE.value) == 1, rig.hardware.trace
    assert rig.hardware.trace.count(Op.RESTART_WORKLOAD.value) == 1
    recovery = next(r for r in rig.agent_results if r.operation is Op.RECOVER_DRAIN_P2P)
    assert recovery.details["gpu_recovery_reboot_required"] is True


@pytest.mark.parametrize(
    "failure",
    [
        "unknown-action",
        "missing-gpu",
        "persistence-stuck",
        "query-timeout",
        "reset-timeout",
    ],
)
def test_drain_p2p_unproven_recovery_never_restarts_training(rig, failure):
    rig.hardware.failure = failure
    rig.hardware.recovery_action = "N/A" if failure == "unknown-action" else "Reset"
    decision = rig.ingest(rig.payload())
    result = rig.execute(decision)

    assert result.status is not WorkflowStatus.SUCCEEDED, result
    trace = rig.hardware.trace
    assert Op.RESTART_NODE.value not in trace, trace
    assert Op.RESTORE_SCHEDULING.value not in trace, trace
    assert Op.RESTART_WORKLOAD.value not in trace, trace
    assert trace.count("gpu-reset") == (1 if failure == "reset-timeout" else 0), trace
    assert Op.RECOVER_DRAIN_P2P.value in trace, (
        "the failure must reach the recovery step"
    )
    if failure == "reset-timeout":
        assert "persistence-restored" in trace, (
            "existing terminal-failure compensation restores services, not scheduling"
        )
        recovery = next(
            r for r in rig.agent_results if r.operation is Op.RECOVER_DRAIN_P2P
        )
        assert recovery.details["outcome_unknown"] is True
        assert not recovery.retryable, recovery
        assert result.status is WorkflowStatus.BLOCKED, result
        before = list(rig.hardware.commands)
        assert rig.execute(decision).status is WorkflowStatus.BLOCKED
        assert rig.hardware.commands == before, (
            "an unknown reset must not be resubmitted"
        )


def test_drain_p2p_idle_node_has_no_workload_stop_or_restart(rig):
    rig.hardware.training_running = False
    decision = rig.ingest(rig.payload(active=False))
    result = rig.execute(decision)

    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert Op.STOP_WORKLOADS.value not in rig.hardware.trace
    assert Op.RESTART_WORKLOAD.value not in rig.hardware.trace
    assert "recovery-query" in rig.hardware.trace


def test_drain_p2p_unsupported_agent_operation_fails_closed(rig):
    rig.agent.allowed_operations.remove(Op.RECOVER_DRAIN_P2P)
    decision = rig.ingest(rig.payload())
    result = rig.execute(decision)

    assert result.status is not WorkflowStatus.SUCCEEDED, result
    assert "gpu-reset" not in rig.hardware.trace
    assert Op.RESTART_NODE.value not in rig.hardware.trace
    assert Op.RESTART_WORKLOAD.value not in rig.hardware.trace
