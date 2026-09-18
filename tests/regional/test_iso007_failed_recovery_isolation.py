from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import run_e2e002_multicluster_fault as shared
from scripts.e2e.regional import run_iso007_failed_recovery_isolation as runner
from tests.regional.test_identity_causal_review import lifecycle_harness


def test_zero_budget_is_rejected_without_spending_a_restart() -> None:
    state, reserved = InMemoryStore().reserve_job_restart(
        "a", "job-test", 0, "reservation-test"
    )
    assert reserved is False
    assert state.budget == 0 and state.restart_count == 0


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "wrong-reason",
        "unexpected-restart",
        "running",
        "b-failed",
        "cleanup",
        "ack",
    ],
)
def test_iso007_requires_precise_a_failure_and_independent_b_success(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path, defect=defect)
    settings = shared.Settings(
        multi=site.multi(*targets),
        site_file=site.site_file,
        job_id="job-test",
        attempt_id="attempt-test",
        predecessor_path=tmp_path / "predecessor.json",
    )
    monkeypatch.setattr(runner, "read_only_preflight", shared.read_only_preflight)
    code = runner.execute_case(
        settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    result = json.loads(
        (tmp_path / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json").read_text()
    )
    assert code == (0 if defect == "none" else 1), result.get("errors", [])
    assert result["case_id"] == runner.CASE_ID
    assert site.regional(targets[0]).workload.restart_budget == 0
    assert site.regional(targets[1]).workload.restart_budget == 1
    assert "settle-a" in events and "settle-b" in events
    assert "delete-a" in events and "delete-b" in events
    if defect == "none":
        assert [state["workflow"]["status"] for state in result["states"]] == [
            "FAILED",
            "SUCCEEDED",
        ]
        assert result["verdict"] == "PASS"
    else:
        assert result["verdict"] == "FAIL"


@pytest.mark.parametrize("pair", [["a", "b"], ["a", "c"], [], None])
def test_iso007_preflight_requires_the_same_successful_predecessor_pair(
    pair: list[str] | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predecessor = tmp_path / "previous.json"
    predecessor.write_text(json.dumps({"cluster_ids": pair}))
    settings = SimpleNamespace(
        predecessor_path=predecessor,
        multi=SimpleNamespace(
            cluster_a=SimpleNamespace(cluster_id="a"),
            cluster_b=SimpleNamespace(cluster_id="b"),
        ),
    )
    node = {
        "name": "node",
        "uid": "node-uid",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "labels": {},
    }
    calls = []

    def preflight(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "errors": [],
            "nodes_a": [copy.deepcopy(node)],
            "nodes_b": [copy.deepcopy(node)],
        }

    monkeypatch.setattr(shared, "read_only_preflight", preflight)
    result = runner.read_only_preflight(settings, tmp_path)
    assert bool(result["errors"]) is (pair != ["a", "b"])
    assert calls[0]["predecessor_case_id"] == "GF-REGIONAL-E2E-002"


def test_iso007_main_uses_the_shared_guard_and_stays_plan_only_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arguments = runner.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--cluster-a",
            "a",
            "--gpu-a-context",
            "a",
            "--gpu-a-kubeconfig",
            str(tmp_path / "a"),
            "--cluster-b",
            "b",
            "--gpu-b-context",
            "b",
            "--gpu-b-kubeconfig",
            str(tmp_path / "b"),
        ]
    )
    assert arguments.execute is False
    assert runner.CASE.confirmation == "ISO007_A_BUDGET_DENIED_B_RECOVERS"
    seen = []
    monkeypatch.setattr(
        runner, "run_standard_case", lambda case: seen.append(case) or 0
    )
    assert runner.main() == 0
    assert seen == [runner.CASE]
