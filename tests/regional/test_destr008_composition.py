"""Composition retains a single journal and binds every implementation helper."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional import destr008_watchdog_admission as admission
from scripts.e2e.regional import destr008_watchdog_journal as journal
from scripts.e2e.regional import destr008_watchdog_retirement as retirement
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._destr008_cancellation_controller import build_api


@pytest.mark.parametrize(
    "helper",
    [journal, admission, retirement],
    ids=["journal", "admission", "retirement"],
)
def test_each_composed_helper_is_part_of_execution_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, helper: ModuleType
) -> None:
    api, plan, runtime = build_api(tmp_path, monkeypatch)
    source = tmp_path / "helper-source"
    source.write_text("original controlled helper bytes\n")
    monkeypatch.setattr(helper, "__file__", str(source))
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    source.write_text("changed controlled helper bytes\n")
    mutations = [call for call in api.calls if call[0] in {"create", "patch", "delete"}]
    with pytest.raises(RegionalFixtureError, match="source changed"):
        watchdog.validate_running()
    fresh = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError, match="source changed"):
        fresh.arm()
    assert [
        call for call in api.calls if call[0] in {"create", "patch", "delete"}
    ] == mutations, "helper changes cannot grant new execution authority"
    proof = fresh.resume_cleanup(seconds=10)
    assert proof.case_failed and proof.producer_revoked, (
        "compatible source changes may only perform explicit cleanup"
    )
    fresh.cleanup()
    assert fresh.record.closed, (
        "composed retirement must update the same durable journal"
    )


def test_legacy_source_key_shape_remains_cleanup_only_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api, plan, runtime = build_api(tmp_path, monkeypatch)
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    saved = api.journal()
    assert set(saved["sources"]) == {
        "controller",
        "resources",
        "control",
        "locking",
        "protocol",
        "probe",
    }, "implementation extraction must not invalidate the recorded source key shape"
    saved["sources"]["controller"] = "a" * 64
    write_json_atomic(watchdog.path, saved)
    fresh = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError, match="source changed"):
        fresh.arm()
    proof = fresh.resume_cleanup(seconds=10)
    assert proof.state == "QUIESCENT" and proof.case_failed, (
        "older controller source identity must not be promoted into a fresh arm"
    )


def test_retirement_requires_recorded_cessation_after_each_stop_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api, plan, runtime = build_api(tmp_path, monkeypatch)
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    stopped = []

    def unacknowledged_stop(
        _component: retirement.WatchdogRetirement, index: int
    ) -> None:
        stopped.append(index)

    monkeypatch.setattr(retirement.WatchdogRetirement, "stop_job", unacknowledged_stop)
    with pytest.raises(
        RegionalFixtureError, match="earlier watchdog observer is still present"
    ):
        watchdog.cleanup()
    assert stopped == [0], "the retirement operation must attempt its owned observer"
    assert not watchdog.record.jobs[0].stopped and not watchdog.record.closed, (
        "a returned sub-step is not proof of a persisted stop acknowledgement"
    )
    assert ("job", watchdog.name) in api.objects, (
        "supporting resources must remain with the live observer"
    )
