"""The workflow step keeps the holders a final-ownership refusal names.

``_fold_result`` copies the agent's details into the step record only for an
unknown-outcome failure; an ``OwnershipRefused`` carries
``manual_confirmation_required`` so it takes that branch, and the
``persistent_device_clients`` attribution added for the live 2026-10-01
observation has to survive the hop unchanged -- the step details and the
``node_failures`` text are what the operator and the support reason read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.models import WorkflowStepStatus
from tests.execution.test_node_action_transport_retry import (
    ENDPOINT,
    SECRET,
    step_context,
)
from tests.node_agent.test_final_ownership_holder_attribution import (
    HOLDER,
    REASON,
    refused_reset_result,
)


def test_the_step_details_keep_the_holders_the_agent_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _agent, result, _hardware = refused_reset_result(tmp_path, monkeypatch)
    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT}, SECRET, sender=lambda *_args: result
    )

    outcome = adapter.execute(step_context(adapter))

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details["reason"] == REASON, outcome.details
    assert outcome.details["persistent_device_clients"] == [HOLDER], outcome.details
    assert outcome.details["persistent_device_client_count"] == 1, outcome.details
    assert outcome.details["manual_confirmation_required"] is True, outcome.details
    assert outcome.details["node_action_not_started"] is True, outcome.details
    # The per-node failure text the support reason quotes names the holder.
    assert outcome.details["node_failures"] == {
        "node-a": [f"node action outcome unknown: {result.error}"]
    }, outcome.details
    assert "GPU-a:4242:dcgm-exporter" in outcome.details["node_failures"]["node-a"][0]
    assert (outcome.error or "").startswith(
        f"node agent node-a: ResetProgressError: {REASON}: GPU-a:4242:dcgm-exporter"
    ), outcome.error
