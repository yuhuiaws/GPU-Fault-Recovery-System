from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from scripts.e2e.regional import run_e2e002_multicluster_fault as runner
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_identity_causal_review import lifecycle_harness


def configured(site: Any, targets: list[Any], tmp_path: Path) -> runner.Settings:
    return runner.Settings(
        multi=site.multi(*targets),
        site_file=site.site_file,
        job_id="job-test",
        attempt_id="attempt-test",
        predecessor_path=tmp_path / "previous.json",
    )


@pytest.mark.parametrize("budget_denial", [False, True])
def test_concurrent_recovery_keeps_cluster_order_when_b_starts_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, budget_denial: bool
) -> None:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path)
    settings = configured(site, targets, tmp_path)
    original = runner.inject_cluster_faults

    def b_first(*args: Any, **kwargs: Any) -> Any:
        receipts, starts = original(*args, **kwargs)
        assert set(starts) == {0, 1}
        return receipts, {index: starts[index] for index in (1, 0)}

    monkeypatch.setattr(runner, "inject_cluster_faults", b_first)
    case_id = "GF-REGIONAL-ISO-007" if budget_denial else runner.CASE_ID
    assert (
        runner.execute_case(
            settings,
            tmp_path,
            1,
            datetime.now(timezone.utc) + timedelta(hours=1),
            case_id=case_id,
            expect_a_budget_denial=budget_denial,
        )
        == 0
    )
    result = json.loads((tmp_path / "cases" / case_id / f"{case_id}.json").read_text())
    assert [state["event"]["cluster_id"] for state in result["states"]] == ["a", "b"]
    assert [state["workflow"]["status"] for state in result["states"]] == [
        "FAILED" if budget_denial else "SUCCEEDED",
        "SUCCEEDED",
    ]
    assert events.index("delete-a") > max(
        events.index("settle-a"), events.index("settle-b")
    )
    assert "delete-b" in events


@pytest.mark.parametrize(
    "defect", ["observation", "budget", "prewarm-residual", "prewarm-error"]
)
def test_concurrent_branches_cleanup_both_scopes_after_preparation_or_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path)
    settings = configured(site, targets, tmp_path)
    if defect in {"observation", "budget"}:
        original = runner.workload_case.wait_observation
        both_submitted = Barrier(2)

        def observation(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = original(*args, **kwargs)
            both_submitted.wait(timeout=5)
            if defect == "observation":
                result["containers"][0]["pod_uid"] = "foreign"
            else:
                result["restart_budget"] = True
            return result

        monkeypatch.setattr(runner.workload_case, "wait_observation", observation)
    else:
        original_prewarm = runner.ImagePrewarmFixture

        class Prewarm(original_prewarm):
            def cleanup(self) -> dict[str, bool]:
                super().cleanup()
                if defect == "prewarm-error":
                    raise RuntimeError("synthetic prewarm cleanup failure")
                return {"pod": True}

        monkeypatch.setattr(runner, "ImagePrewarmFixture", Prewarm)
    assert (
        runner.execute_case(
            settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
        )
        == 1
    )
    result = json.loads(
        (tmp_path / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json").read_text()
    )
    assert result["verdict"] == "FAIL"
    assert "delete-a" in events and "delete-b" in events
    if defect in {"observation", "budget"}:
        assert not any(event.startswith("inject-") for event in events), (
            "an unproved source Observation must stop both fault injections"
        )
    else:
        assert any("prewarm" in error for error in result["errors"]), result["errors"]


@pytest.mark.parametrize(
    "defect",
    [
        "shared-workflow",
        "budget-count",
        "budget-cluster",
        "notification-cluster",
        "notification-id",
        "notification-incident",
        "notification-receipt",
        "no-notifications",
        "no-commands",
        "no-owner",
        "wrong-command",
    ],
)
def test_concurrent_recovery_never_passes_unbound_terminal_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path)
    settings = configured(site, targets, tmp_path)
    region = site.regional(targets[1])
    original = region.wait_for_workflow

    def state(**kwargs: Any) -> dict[str, Any]:
        value = original(**kwargs)
        if defect == "shared-workflow":
            value["workflow"]["request_id"] = "workflow-a"
        elif defect == "budget-count":
            value["restart_budget"]["restart_count"] = 2
        elif defect == "budget-cluster":
            value["restart_budget"]["cluster_id"] = "a"
        elif defect == "notification-cluster":
            value["notifications"][0]["notification"]["cluster_name"] = "foreign"
        elif defect == "notification-id":
            value["notifications"][0]["notification"]["notification_id"] = ""
        elif defect == "notification-incident":
            value["notifications"][0]["notification"]["incident_id"] = "foreign"
        elif defect == "notification-receipt":
            value["notifications"][0]["result"]["provider_message_id"] = ""
        elif defect == "no-notifications":
            value["notifications"] = []
        elif defect == "no-commands":
            value["commands"] = []
        elif defect == "no-owner":
            value["commands"][0]["last_lease_owner"] = None
        else:
            value["commands"][0]["incident_id"] = "foreign"
        return value

    monkeypatch.setattr(region, "wait_for_workflow", state)
    assert (
        runner.execute_case(
            settings, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
        )
        == 1
    )
    result = json.loads(
        (tmp_path / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json").read_text()
    )
    assert result["errors"]
    assert "settle-a" in events and "settle-b" in events
    assert events.index("delete-a") > max(
        events.index("settle-a"), events.index("settle-b")
    )
    assert "delete-b" in events
