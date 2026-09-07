"""Contract tests for GF-REGIONAL-CMD-018 (open-sibling command hold)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import cmd018_verdicts as verdicts
from scripts.e2e.regional import run_cmd018_open_sibling_hold as cmd018
from scripts.e2e.regional import seeded_command_fixture as seeded

ROOT = Path(__file__).resolve().parents[2]
FIRST = "remote-aaaaaaaaaaaaaaaaaaaaaaaa"
HELD = "remote-bbbbbbbbbbbbbbbbbbbbbbbb"


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


def _dispatch(**overrides: Any) -> dict[str, Any]:
    dispatch = {
        "first": {
            "status": "WAITING",
            "remote_command_id": FIRST,
            "remote_status": "PENDING",
        },
        "held": {
            "status": "WAITING",
            "details": {
                "reason": verdicts.HOLD_REASON,
                "remote_command_id": FIRST,
                "remote_cluster_id": "perf-cap-000",
                "remote_status": "PENDING",
                "held_command_id": HELD,
                "mutation_submitted_by_control_plane": False,
            },
        },
        "open_sibling_holds_total": 1,
        "open_command_ids": [FIRST],
        "registered_agents": [],
    }
    dispatch.update(overrides)
    return dispatch


def test_the_dispatch_contract_passes_on_a_held_rewrite() -> None:
    assert verdicts.dispatch_errors(_dispatch()) == []


def test_the_dispatch_contract_rejects_a_second_command_or_a_same_id() -> None:
    assert "expected [" in _text(
        verdicts.dispatch_errors(_dispatch(open_command_ids=[FIRST, HELD]))
    )
    same = _dispatch()
    same["held"]["details"]["held_command_id"] = FIRST
    assert "did not produce a different command id" in _text(
        verdicts.dispatch_errors(same)
    )
    unheld = _dispatch()
    unheld["held"]["details"]["reason"] = None
    assert "hold reason is None" in _text(verdicts.dispatch_errors(unheld))
    assert "expected 1" in _text(
        verdicts.dispatch_errors(_dispatch(open_sibling_holds_total=0))
    )
    assert "hard stop does not hold" in _text(
        verdicts.dispatch_errors(_dispatch(registered_agents=["node-x"]))
    )


def test_the_executed_once_and_release_contracts() -> None:
    ledger = {"physical_count": 1, "keys": ["wf/0/TRIGGER_HEALTH_SNAPSHOT"]}
    completed = {"command_id": FIRST, "status": "SUCCEEDED"}
    assert verdicts.executed_once_errors(ledger, completed, first_id=FIRST) == []
    assert "holds 2 attempts" in _text(
        verdicts.executed_once_errors(
            {**ledger, "physical_count": 2}, completed, first_id=FIRST
        )
    )
    assert "expected SUCCEEDED" in _text(
        verdicts.executed_once_errors(
            ledger, {**completed, "status": "FAILED"}, first_id=FIRST
        )
    )
    release = {
        "outcome": {"status": "WAITING", "details": {"remote_command_id": HELD}},
        "cancelled": True,
        "cancelled_command": {"command_id": HELD, "status": "FAILED"},
    }
    assert verdicts.release_errors(release, first_id=FIRST) == []
    held = {
        "outcome": {
            "status": "WAITING",
            "details": {"reason": verdicts.HOLD_REASON, "remote_command_id": FIRST},
        }
    }
    assert "still held after the sibling" in _text(
        verdicts.release_errors(held, first_id=FIRST)
    )
    assert "was not cancelled" in _text(
        verdicts.release_errors({**release, "cancelled": False}, first_id=FIRST)
    )
    assert (
        verdicts.metric_family_errors(
            [f"# TYPE {verdicts.HOLDS_METRIC} counter\n{verdicts.HOLDS_METRIC} 0"]
        )
        == []
    )
    assert "not exported" in _text(verdicts.metric_family_errors(["nothing"]))


def test_the_probe_definition_and_plan_are_plan_only_and_name_the_hard_stop() -> None:
    probe = cmd018.probe_definition()
    assert probe.script.is_file() and probe.owner == verdicts.OWNER, probe
    manifest = seeded.pod_manifest(
        probe,
        "img",
        {"executor_artifact_sha256": "a", "executor_compatibility_digest": "b"},
    )
    assert (
        manifest["spec"]["containers"][0]["command"][-1]
        == "/scripts/cmd018_ledger_executor.py"
    )
    parser = cmd018.parser()
    assert parser.parse_args(["--run-dir", "/tmp/run"]).execute is False, (
        "plan by default"
    )
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in parser.format_help(), flag
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])
    details = cmd018.plan_details()
    assert details["risk"] == "live-non-destructive", details
    assert details["expected_hold_reason"] == verdicts.HOLD_REASON
    assert "local ledger" in details["hard_stop"], details["hard_stop"]
    assert cmd018.CONFIRMATION == "CMD018_EXECUTE"
    assert verdicts.PREDECESSOR_CASE_ID == "GF-REGIONAL-CMD-017"
    for path in (
        ROOT / "scripts/e2e/regional/run_cmd018_open_sibling_hold.py",
        ROOT / "scripts/e2e/regional/probes/cmd018_ledger_executor.py",
    ):
        assert path.stat().st_mode & 0o777 == 0o775, path
        assert (
            path.read_text(encoding="utf-8").splitlines()[0] == "#!/usr/bin/env python3"
        )
