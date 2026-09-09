"""A FAILED remote command tells the step *why* it failed, by status source.

Control-plane review 2026-09-08, D-8. ``RegionalRemoteWorkflowAdapter`` mapped
a FAILED command to a failed step with the command's ``result_details`` only;
a command the workflow itself cancelled (``status_source`` ``workflow-preempted``
at a step boundary, ``workflow-timeout`` at the deadline) looked exactly like a
node refusing the action, and the branch escalator took a rung for it. The
outcome now carries ``remote_status_source`` so the executor can tell them apart.
"""

from __future__ import annotations

from gpu_fault.models import WorkflowStepStatus
from gpu_fault.regional import RegionalRemoteWorkflowAdapter
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests._builders import build_store
from tests.regional._regional_support import TOKEN_A, registration, workflow_state


def test_a_cancelled_command_reports_its_status_source_on_the_failed_step() -> None:
    store = build_store()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    adapter = RegionalRemoteWorkflowAdapter(
        store, owners={"gpu-fault-kubernetes-adapter"}
    )
    context = workflow_state()
    first = adapter.execute(context)
    command_id = first.details["remote_command_id"]
    assert store.cancel_remote_command(command_id, reason="cancelled by a successor"), (
        "a PENDING command must be cancellable"
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.error == "cancelled by a successor"
    assert outcome.details["remote_status_source"] == "workflow-preempted"
    assert store.get_remote_command(command_id).status is RemoteCommandStatus.FAILED
