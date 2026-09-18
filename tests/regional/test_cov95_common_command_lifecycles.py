from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.e2e.regional import run_cmd017_barrier_hold as barrier
from scripts.e2e.regional import run_cmd018_open_sibling_hold as sibling
from tests.regional._cov95_common_commands import CommandModel


@pytest.mark.parametrize(
    "failure",
    [
        "",
        "preflight",
        "register",
        "ready",
        "seed",
        "seed-ack",
        "marker",
        "phase",
        "cleanup",
    ],
)
def test_barrier_case_keeps_holds_nonphysical_and_tracks_partial_cleanup(
    failure, tmp_path, monkeypatch
) -> None:
    model = CommandModel(barrier, monkeypatch)
    model.failure = failure
    result_code = barrier.run_case(
        tmp_path, 2, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    result = json.loads(
        (tmp_path / "cases" / barrier.CASE_ID / f"{barrier.CASE_ID}.json").read_text()
    )
    assert result_code == (1 if failure else 0), (
        "every failed barrier stage must fail the case"
    )
    assert result["verdict"] == ("FAIL" if failure else "PASS"), (
        "a held command is valid only when all boundary and cleanup checks pass"
    )
    assert model.events[-1] == "cleanup", "partial preparation always enters cleanup"
    if not failure:
        assert model.events.index("stop-claimant") < model.events.index(
            "final-command"
        ), "read the final nonterminal state after stopping the only claimant"
        assert result["adapter_executed"] is None, (
            "no adapter action should be observed"
        )
        assert result["held_command"]["status"] == "WAITING", "a hold is not completion"
    if failure == "register":
        assert model.cleanup_state["registry_started"] is True, (
            "lost registry acknowledgement must retain a cleanup obligation"
        )
    if failure == "seed-ack":
        assert model.cleanup_state["seed"]["workflow_id"] == "workflow-fixture", (
            "seed identity must be recorded before a write can commit"
        )


@pytest.mark.parametrize(
    "failure",
    [
        "",
        "preflight",
        "register",
        "dispatch",
        "release",
        "metrics-one",
        "metrics-all",
        "purge",
        "purge-residual",
        "cleanup",
    ],
)
def test_open_sibling_case_stops_claimant_before_release_and_purges_all_ids(
    failure, tmp_path, monkeypatch
) -> None:
    model = CommandModel(sibling, monkeypatch)
    model.failure = failure
    failed = failure not in {"", "metrics-one"}
    result_code = sibling.run_case(
        tmp_path, 3, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    result = json.loads(
        (tmp_path / "cases" / sibling.CASE_ID / f"{sibling.CASE_ID}.json").read_text()
    )
    assert result_code == int(failed), "sibling case exit must include cleanup failures"
    assert result["verdict"] == ("FAIL" if failed else "PASS"), (
        "never promote partial evidence"
    )
    assert "cleanup" in model.events, "all partial runs must retain their cleanup path"
    if "release" in model.events:
        assert model.events.index("stop-claimant") < model.events.index("release"), (
            "the second command must not be minted until the only claimant is stopped"
        )
        assert model.cleanup_state["command_ids"] == ["command-a", "command-b"], (
            "both the original and released command need cleanup accounting"
        )
    if failure in {"purge", "purge-residual"}:
        assert "cleanup_error" in result, "unproved seed purge needs a causal error"


@pytest.mark.parametrize("module", [barrier, sibling])
def test_expired_command_window_never_registers_or_seeds(
    module, tmp_path, monkeypatch
) -> None:
    model = CommandModel(module, monkeypatch)
    assert (
        module.run_case(tmp_path, 1, datetime.now(timezone.utc) - timedelta(seconds=1))
        == 1
    ), "an expired maintenance window must be refused"
    assert model.events == ["cleanup"], (
        "expiry must stop before environment or registry actions"
    )
    assert model.cleanup_state["seed"] == {}, (
        "no seed may be attributed to an expired run"
    )
