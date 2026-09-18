"""Each recovery branch forwards its own Store proof before replacement reads."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_e2e002_multicluster_fault as case
from tests.regional._identity_lifecycle_support import recovery_state


@pytest.mark.parametrize(
    ("budget_denied", "index"), [(False, 0), (False, 1), (True, 0), (True, 1)]
)
@pytest.mark.parametrize("custody_rejected", [False, True])
def test_branch_authorizes_only_its_expected_restart(
    tmp_path: Path, budget_denied: bool, index: int, custody_rejected: bool
) -> None:
    cluster = ("a", "b")[index]
    state = recovery_state(cluster, f"node-{cluster}", "marker")
    calls: list[str] = []
    source = {"pods": [{"uid": "source-pod", "node": f"node-{cluster}"}]}
    target = {"pods": [{"uid": "target-pod"}]}

    def authorize(proof: dict[str, Any]) -> None:
        assert proof is state
        calls.append("authorize")
        if custody_rejected:
            raise case.RegionalFixtureError("restart custody rejected")

    def wait(uids: set[str], **kwargs: Any) -> dict[str, Any]:
        assert calls == ["authorize"]
        assert uids == {"source-pod"}
        calls.append("wait")
        return target

    def pods() -> list[dict[str, Any]]:
        calls.append("pods")
        return source["pods"]

    fixture = SimpleNamespace(
        settings=SimpleNamespace(cluster_id=cluster),
        wait_for_workflow=lambda **kwargs: state,
    )
    workload = SimpleNamespace(
        authorize_restart=authorize, wait_restarted=wait, pods=pods
    )
    branches = case.RecoveryBranches(
        settings=SimpleNamespace(job_id="job-test", attempt_id="attempt-test"),
        targets=(SimpleNamespace(cluster_id="a"), SimpleNamespace(cluster_id="b")),
        fixtures=[fixture, fixture],
        workloads=[workload, workload],
        prewarms=[],
        submission_started=[True, True],
        injection_contexts=[{}, {}],
        case_dir=tmp_path,
        expect_a_budget_denial=budget_denied,
        sources=[source, source],
    )
    refused_branch = budget_denied and index == 0
    if custody_rejected and not refused_branch:
        with pytest.raises(case.RegionalFixtureError, match="custody rejected"):
            branches.settle(index, datetime.now(timezone.utc), {"record_id": "marker"})
        assert calls == ["authorize"]
    else:
        observed, restarted = branches.settle(
            index, datetime.now(timezone.utc), {"record_id": "marker"}
        )
        assert observed is state
        assert restarted == (source if refused_branch else target)
        assert calls == (["pods"] if refused_branch else ["authorize", "wait"])
