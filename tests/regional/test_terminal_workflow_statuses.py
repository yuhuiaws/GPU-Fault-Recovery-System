"""The regional harness's terminal sets match the product, including SUPERSEDED."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import EXECUTABLE_WORKFLOW_STATUSES, WorkflowStatus
from scripts.e2e.regional import regional_live_fixture as live
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
    assert WorkflowStatus.SUPERSEDED.value in CANONICAL_TERMINAL, (
        "superseded workflows must not keep a marker wait loop open"
    )


def test_collector_fixture_terminal_set_matches_product() -> None:
    assert set(COLLECTOR_TERMINAL) == CANONICAL_TERMINAL, (
        "collector settlement must use the product terminal statuses"
    )
    assert "SUPERSEDED" in COLLECTOR_TERMINAL, (
        "a restored predecessor must not block a later marker read"
    )


def test_regional_fixture_terminal_set_matches_product() -> None:
    assert set(REGIONAL_TERMINAL) == CANONICAL_TERMINAL, (
        "regional settlement must use the product terminal statuses"
    )
    assert "SUPERSEDED" in REGIONAL_TERMINAL, (
        "a superseded workflow is terminal, not successful"
    )


@pytest.mark.parametrize("status", ["SUPERSEDED", "UNKNOWN"])
def test_wait_loop_distinguishes_superseded_from_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    kubeconfig = tmp_path / "isolated.kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    regional = live.RegionalLiveFixture(
        live.RegionalLiveSettings(
            cpu_kubeconfig=kubeconfig,
            gpu_kubeconfig=kubeconfig,
            gpu_context="test-context",
            namespace="test-namespace",
            cluster_id="test-cluster",
            region="us-west-2",
        )
    )
    clock = {"now": 0.0}
    calls: list[dict[str, Any]] = []

    def snapshot(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"workflow": {"request_id": "workflow-a", "status": status}}

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    monkeypatch.setattr(regional, "store_snapshot", snapshot)
    monkeypatch.setattr(live.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(live.time, "sleep", sleep)
    arguments = {
        "node": "node-a",
        "marker": "marker-a",
        "observed_after": datetime(2026, 9, 14, tzinfo=timezone.utc),
        "case_dir": tmp_path,
        "timeout_seconds": 5,
    }
    if status == "SUPERSEDED":
        result = regional.wait_for_workflow(**arguments)
        assert result["workflow"]["status"] == "SUPERSEDED", (
            "polling completion must not relabel supersession as success"
        )
        assert "verdict" not in result, "a terminal read must not synthesize a PASS"
    else:
        with pytest.raises(live.RegionalFixtureError, match="did not reach"):
            regional.wait_for_workflow(**arguments)
    assert len(calls) == 1, "this check must use one isolated Store snapshot"
