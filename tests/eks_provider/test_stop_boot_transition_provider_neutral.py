"""The STOP boot-transition proof accepts either provider's reboot confirmation.

``confirmed_reboot`` in ``tests/regional/test_stop_ownership_reboot_transition``
builds a real confirmed HyperPod reboot chain; here its confirmation source is
rewritten to the plain-EKS one (every other field is shared) and the pre-submit
ownership guard must still accept the boot change, while any other source
remains ``STOP_OWNERSHIP_DRIFT``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.adapters.kubernetes.stop_ownership import (
    node_submission_ownership_guard,
    stop_ownership_scope,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_provider import (
    EKS_REBOOT_CONFIRMATION_SOURCE,
    HYPERPOD_REBOOT_CONFIRMATION_SOURCE,
)
from tests.regional.test_stop_ownership_reboot_transition import confirmed_reboot


def _guard_outcome(source: str):
    state, validator, context = confirmed_reboot()
    execution = context.workflow.step_executions[1]
    assert (
        execution.details["confirmation_source"] == HYPERPOD_REBOOT_CONFIRMATION_SOURCE
    )
    execution.details["confirmation_source"] = source
    context = replace(
        context,
        step=context.step.model_copy(
            update={"operation": WorkflowOperation.REPLACE_NODE}
        ),
    )
    with stop_ownership_scope(validator):
        return node_submission_ownership_guard(context)


@pytest.mark.parametrize(
    "source", [HYPERPOD_REBOOT_CONFIRMATION_SOURCE, EKS_REBOOT_CONFIRMATION_SOURCE]
)
def test_either_provider_confirmation_proves_the_boot_transition(source: str) -> None:
    assert _guard_outcome(source) is None


@pytest.mark.parametrize("source", ["explicit-operation-id", "ec2-running", ""])
def test_any_other_confirmation_source_remains_drift(source: str) -> None:
    outcome = _guard_outcome(source)

    assert outcome is not None
    assert outcome.details is not None
    assert outcome.details["safety_rejection"] is True
    assert outcome.details["manual_confirmation_required"] is True
