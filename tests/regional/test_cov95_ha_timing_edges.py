from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
from tests.regional._cov95_ha_reset_harness import ResetHarness


def test_timing_window_scales_and_restores_a_smaller_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha004)
    harness.api.document["spec"]["replicas"] = 1
    harness.api.document["status"].update(
        replicas=1, readyReplicas=1, availableReplicas=1, updatedReplicas=1
    )
    baseline = copy.deepcopy(harness.api.document["spec"])
    fixture = ha004.ExecutorTimingFixture(harness.api, harness.directory)
    assert fixture.apply()["replicas"] == 2
    assert fixture.restore()["replicas"] == 1
    assert harness.api.document["spec"] == baseline
    assert harness.watchdog.returncode == 0


@pytest.mark.parametrize("defect", ["uid", "replicas", "environment", "watchdog"])
def test_timing_restore_refuses_identity_or_unverified_restoration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha004)
    fixture = ha004.ExecutorTimingFixture(harness.api, harness.directory)
    fixture.apply()
    if defect == "uid":
        harness.api.document["metadata"]["uid"] = "replacement"
    elif defect == "replicas":
        harness.api.document["spec"]["replicas"] = 3
    elif defect == "environment":
        harness.api.document["spec"]["template"]["spec"]["containers"][0]["env"][0][
            "value"
        ] = "999"
    else:
        monkeypatch.setattr(
            ha004, "stop_process_group", lambda _: {"stop_error": "unit"}
        )
    with pytest.raises(
        RuntimeError, match="replacement|restore|restoration|watchdog|window"
    ):
        fixture.restore()
    if defect in {"replicas", "environment"}:
        assert harness.watchdog.returncode is None


def test_timing_restore_without_an_open_window_is_a_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha004)
    fixture = ha004.ExecutorTimingFixture(harness.api, harness.directory)
    result = fixture.restore()
    assert result["watchdog"] == {"armed": False}
    assert result["restore_errors"] == []


def test_timing_scale_never_targets_a_replaced_deployment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = ResetHarness(monkeypatch, tmp_path, ha004)
    fixture = ha004.ExecutorTimingFixture(harness.api, harness.directory)
    harness.api.document["metadata"]["uid"] = "replacement"
    with pytest.raises(RuntimeError, match="UID changed"):
        fixture.scale(2)
