"""The regional harness's terminal workflow-status sets must match the product.

COLLECT-014 injects a fail-closed SXID, restores it (superseding the
fail-closed workflow), then injects the positive SXID. ``wait_marker`` widens
``observed_after`` backward by ``KMSG_CLOCK_SKEW_SECONDS`` (30s), so the
positive snapshot sweeps in the fail-closed workflow the restore left
``SUPERSEDED`` about 25s before injection. When the shared terminal set omitted
``SUPERSEDED`` that swept-in workflow never counted as settled, so the wait loop
spun to its 1800s timeout and failed the case even though the positive
``RESET_ALL_GPUS_AND_NVSWITCHES`` had run every step and SUCCEEDED. The
control plane treats every status outside PENDING/SAFETY_PENDING/RUNNING as
terminal (``models.py``); the harness must agree.
"""

from __future__ import annotations

from gpu_fault.models import EXECUTABLE_WORKFLOW_STATUSES, WorkflowStatus
from scripts.e2e.regional.collector_acceptance_fixture import (
    TERMINAL_WORKFLOW_STATUSES as COLLECTOR_TERMINAL,
)
from scripts.e2e.regional.regional_live_fixture import (
    TERMINAL_WORKFLOW_STATUSES as REGIONAL_TERMINAL,
)

CANONICAL_TERMINAL = {
    status.value
    for status in WorkflowStatus
    if status not in EXECUTABLE_WORKFLOW_STATUSES
}


def test_canonical_terminal_set_contains_superseded() -> None:
    assert WorkflowStatus.SUPERSEDED.value in CANONICAL_TERMINAL


def test_collector_fixture_terminal_set_matches_product() -> None:
    assert set(COLLECTOR_TERMINAL) == CANONICAL_TERMINAL
    assert "SUPERSEDED" in COLLECTOR_TERMINAL


def test_regional_fixture_terminal_set_matches_product() -> None:
    assert set(REGIONAL_TERMINAL) == CANONICAL_TERMINAL
    assert "SUPERSEDED" in REGIONAL_TERMINAL
