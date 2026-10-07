"""A reboot the provider confirms automatically lifts the eks-managed quiesce taint.

Live 2026-10-07 (gpu-eks-cluster DESTR-025): RESET_GPU failed on a busy device,
the branch escalated in place to RESTART_NODE (EC2 ``RebootInstances``), the
node came back with a new Agent incarnation -- and still carried the
``gpu-fault.io/quiesce`` NoExecute taint the retired RESTORE_GPU_SERVICES would
have lifted. The exporter DaemonSet could not schedule, VALIDATE_GPU waited out
its cap on ``quiesce_taint_present``, and the ladder climbed to a REPLACE_NODE
no spare could serve. The provider-Running-plus-new-incarnation confirmation
is the path the live run took; it strips the taint now, as the explicit
operation-id confirmation does (``tests/execution/test_kubernetes_guards.py``).
"""

from __future__ import annotations

from gpu_fault.adapters.common import QUIESCE_TAINT, quarantine_taint_value
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from tests.hyperpod._cov95_runtime_confirmation import ConfirmationHarness


def _taint_keys(harness: ConfirmationHarness) -> list[str]:
    return [item["key"] for item in harness.scheduler.core.node["spec"]["taints"]]


def test_the_automatic_confirmation_strips_the_quiesce_taint() -> None:
    harness = ConfirmationHarness(WorkflowOperation.RESTART_NODE)
    harness.scheduler.core.node["spec"]["taints"].append(
        {
            "key": QUIESCE_TAINT,
            "value": quarantine_taint_value(harness.incident.incident_id),
            "effect": "NoExecute",
        }
    )

    result = harness.execute()

    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert result.details["confirmation_source"].endswith("new-agent-incarnation"), (
        result.details
    )
    assert QUIESCE_TAINT not in _taint_keys(harness), _taint_keys(harness)
    assert "gpu-fault.io/quarantined" in _taint_keys(harness), (
        "the quarantine taint belongs to RESTORE_SCHEDULING"
    )
    assert result.details["quiesce_taint_removed_nodes"] == ["node-a"], result.details
    assert harness.provider.submissions == [], harness.provider.submissions


def test_a_poll_that_does_not_confirm_yet_leaves_the_taint() -> None:
    harness = ConfirmationHarness(WorkflowOperation.RESTART_NODE, stabilization=60)
    harness.scheduler.core.node["spec"]["taints"].append(
        {
            "key": QUIESCE_TAINT,
            "value": quarantine_taint_value(harness.incident.incident_id),
            "effect": "NoExecute",
        }
    )

    result = harness.execute()

    assert result.status is WorkflowStepStatus.WAITING, result
    assert QUIESCE_TAINT in _taint_keys(harness), _taint_keys(harness)
    assert "quiesce_taint_removed_nodes" not in result.details, result.details

    harness.advance(60)
    result = harness.execute()

    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert QUIESCE_TAINT not in _taint_keys(harness), _taint_keys(harness)
    assert result.details["quiesce_taint_removed_nodes"] == ["node-a"], result.details
