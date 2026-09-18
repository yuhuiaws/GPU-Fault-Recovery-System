from __future__ import annotations

import copy
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import iso006_primary_recovery as primary
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_identity_causal_review import lifecycle_harness


def primary_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[primary.PrimaryRecovery, Any, list[str]]:
    site, targets, events = lifecycle_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(
        primary, "ImagePrewarmFixture", primary.workload.ImagePrewarmFixture
    )
    region = site.regional(targets[0])
    fixture = primary.PrimaryRecovery(
        region,
        site_file=site.site_file,
        case_dir=tmp_path,
        job_id="job-test",
        attempt_id="attempt-test",
        attempt=1,
        maintenance_window_end=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    return fixture, region, events


@pytest.mark.parametrize("defect", ["expired", "busy", "node"])
def test_primary_preparation_stops_before_submission_on_unsafe_baselines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    fixture, region, events = primary_fixture(monkeypatch, tmp_path)
    if defect == "expired":
        fixture.maintenance_window_end = datetime.now(timezone.utc) - timedelta(
            seconds=1
        )
    elif defect == "busy":
        monkeypatch.setattr(region, "gpu_workloads", lambda: [{"uid": "foreign"}])
    else:
        nodes = region.gpu_nodes()
        nodes[0]["ready"] = "False"
        monkeypatch.setattr(region, "gpu_nodes", lambda: nodes)
    with pytest.raises(
        primary.RegionalFixtureError, match="window|GPU workload|not clean"
    ):
        fixture.prepare()
    assert events == []
    assert fixture.submitted is False


@pytest.mark.parametrize("defect", ["budget", "uid", "stale"])
def test_primary_observation_failure_retains_scoped_cleanup_intent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    fixture, _region, events = primary_fixture(monkeypatch, tmp_path)
    original = primary.recovery.wait_observation

    def observation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = original(*args, **kwargs)
        if defect == "budget":
            value["restart_budget"] = 0
        elif defect == "uid":
            value["containers"][0]["pod_uid"] = "foreign"
        else:
            value["observed_at"] = "2000-01-01T00:00:00Z"
        return value

    monkeypatch.setattr(primary.recovery, "wait_observation", observation)
    with pytest.raises(primary.RegionalFixtureError, match="fresh and workload-bound"):
        fixture.prepare()
    assert fixture.submitted is True
    assert "inject-a" not in events
    assert fixture.cleanup({}) == []
    assert events.index("delete-a") > events.index("submit-a")


@pytest.mark.parametrize(
    "defect", ["reference", "old-attempt", "missing-attempt", "mixed-attempt"]
)
def test_primary_recovery_requires_causal_replay_and_a_unique_new_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    fixture, region, events = primary_fixture(monkeypatch, tmp_path)
    fixture.prepare()
    if defect == "reference":
        original = region.wait_for_workflow

        def state(**kwargs: Any) -> dict[str, Any]:
            value = original(**kwargs)
            value["event"]["evidence_ref"] = "api-replay://foreign"
            return value

        monkeypatch.setattr(region, "wait_for_workflow", state)
    else:
        target = copy.deepcopy(fixture.source)
        for index, pod in enumerate(target["pods"]):
            pod["uid"] += "-new"
            pod["attempt_id"] = (
                fixture.attempt_id
                if defect == "old-attempt"
                else None
                if defect == "missing-attempt"
                else f"new-{index}"
            )
        monkeypatch.setattr(
            fixture.workload, "wait_restarted", lambda *args, **kwargs: target
        )
    with pytest.raises(
        primary.RegionalFixtureError,
        match="submitted software replay|new unique attempt",
    ):
        fixture.recover(time.monotonic() + 120, cut_is_active=lambda: True)
    assert events.index("settle-a") > events.index("inject-a")
    assert fixture.cleanup({}) == []
    assert events.index("delete-a") > events.index("settle-a")


def test_primary_restart_custody_refusal_prevents_replacement_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture, region, events = primary_fixture(monkeypatch, tmp_path)
    fixture.prepare()

    def refuse(state: dict[str, Any]) -> None:
        assert state is region.state
        events.append("custody-refused")
        raise primary.RegionalFixtureError("restart custody rejected")

    monkeypatch.setattr(fixture.workload, "authorize_restart", refuse)
    monkeypatch.setattr(
        fixture.workload,
        "wait_restarted",
        lambda *args, **kwargs: events.append("wait"),
    )
    with pytest.raises(primary.RegionalFixtureError, match="custody rejected"):
        fixture.recover(time.monotonic() + 120, cut_is_active=lambda: True)
    assert "custody-refused" in events and "wait" not in events
    assert fixture.cleanup({}) == []


@pytest.mark.parametrize("defect", ["residual", "prewarm-error", "node", "unsubmitted"])
def test_primary_cleanup_checks_prewarm_and_nodes_even_after_workload_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    fixture, region, events = primary_fixture(monkeypatch, tmp_path)
    if defect != "unsubmitted":
        fixture.prepare()
    if defect == "residual":
        monkeypatch.setattr(fixture.prewarm, "cleanup", lambda: {"pod": True})
    elif defect == "prewarm-error":

        def failed() -> Any:
            raise RuntimeError("synthetic prewarm failure")

        monkeypatch.setattr(fixture.prewarm, "cleanup", failed)
    elif defect == "node":
        nodes = region.gpu_nodes()
        nodes[0]["uid"] = "replacement"
        monkeypatch.setattr(region, "gpu_nodes", lambda: nodes)
    result: dict[str, Any] = {}
    errors = fixture.cleanup(result)
    assert bool(errors) is (defect != "unsubmitted")
    assert ("delete-a" in events) is (defect != "unsubmitted")
    if defect == "node":
        assert errors == ["A node state changed across recovery/cleanup"]
