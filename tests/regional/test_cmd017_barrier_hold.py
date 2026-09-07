"""Contract tests for GF-REGIONAL-CMD-017.

The verdicts are judged against synthetic evidence -- the control plane's view
of the held command, the probe executor's counters and breadcrumb, its log and
the stand-in adapter's marker file -- once on the intended run and once per way
the run can be wrong. Nothing here touches a cluster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import cmd017_verdicts as verdicts
from scripts.e2e.regional import run_cmd017_barrier_hold as cmd017
from scripts.e2e.regional import seeded_command_fixture as seeded

ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_ID = "cmd017-barrier-executor"


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


def _seed(**overrides: Any) -> dict[str, Any]:
    seed = {
        "operation": verdicts.OPERATION,
        "node_ids": list(verdicts.NODE_IDS),
        "registered_agents": [],
    }
    seed.update(overrides)
    return seed


def test_the_seed_must_be_a_two_node_barrier_for_a_cluster_nobody_serves() -> None:
    assert verdicts.seed_errors(_seed()) == [], "the intended seed passes"
    assert "a barrier needs two" in _text(
        verdicts.seed_errors(_seed(node_ids=["only-one"]))
    ), "a single node is not a barrier"
    assert "hard stop does not hold" in _text(
        verdicts.seed_errors(_seed(registered_agents=["node-x"]))
    ), "an agent registered under the synthetic cluster voids the hard stop"
    assert "seeded operation" in _text(
        verdicts.seed_errors(_seed(operation="RESET_GPU"))
    )
    assert verdicts.OPERATION == "RESET_ALL_GPUS_NVSWITCHES", verdicts.OPERATION


def test_the_probe_must_own_the_step_and_declare_no_coordinator() -> None:
    ready = {"owner": verdicts.OWNER, "adapter_has_barrier_coordinator": False}
    assert verdicts.ready_errors(ready) == [], "the stand-in's contract passes"
    assert "claims a barrier coordinator" in _text(
        verdicts.ready_errors({**ready, "adapter_has_barrier_coordinator": True})
    )


def _held(**overrides: Any) -> dict[str, Any]:
    command = {
        "status": "WAITING",
        "status_source": verdicts.STATUS_SOURCE,
        "error": None,
        "result_details": {
            "multi_node_barrier_unavailable": True,
            "operation": verdicts.OPERATION,
            "node_ids": list(verdicts.NODE_IDS),
            "executor_id": EXECUTOR_ID,
            "reason": (
                "RESET_ALL_GPUS_NVSWITCHES across 2 nodes needs a multi-node barrier "
                "coordinator, and this regional executor has none"
            ),
        },
    }
    command.update(overrides)
    return command


def _held_errors(command: dict[str, Any]) -> list[str]:
    return verdicts.held_command_errors(
        command, node_ids=list(verdicts.NODE_IDS), executor_id=EXECUTOR_ID
    )


def test_the_held_command_contract_passes_on_the_fail_closed_record() -> None:
    assert _held_errors(_held()) == [], "the C6 hold record passes"
    assert _held_errors(_held(status="PENDING")) == [], "re-claimable between rounds"


def test_the_held_command_contract_rejects_terminal_or_unexplained_records() -> None:
    assert "must not reach a terminal status" in _text(
        _held_errors(_held(status="FAILED"))
    )
    assert "status_source" in _text(
        _held_errors(_held(status_source="executor-rejected"))
    )
    bare = _held()
    bare["result_details"] = {}
    errors = _held_errors(bare)
    assert "multi_node_barrier_unavailable" in _text(errors), errors
    assert "missing coordinator" in _text(errors), errors
    other = _held()
    other["result_details"]["node_ids"] = ["someone-else"]
    assert "result_details.node_ids" in _text(_held_errors(other))
    errored = _held(error="reset failed")
    assert "carries an error" in _text(_held_errors(errored))


def _state(**overrides: Any) -> dict[str, Any]:
    state = {
        "barrier_unavailable_holds_total": 3,
        "claimed_total": 3,
        "unexpected_failures": 0,
        "reported_failures": 0,
        "adapter_executed": False,
    }
    state.update(overrides)
    return state


def test_the_counter_contract_needs_the_minimum_holds_and_no_adapter_execution() -> (
    None
):
    assert verdicts.executor_state_errors(_state()) == [], "three held rounds pass"
    assert "below the 2 claim rounds" in _text(
        verdicts.executor_state_errors(_state(barrier_unavailable_holds_total=1))
    )
    assert "stand-in adapter was executed" in _text(
        verdicts.executor_state_errors(_state(adapter_executed=True))
    )
    assert "rejected a hold result" in _text(
        verdicts.executor_state_errors(_state(reported_failures=1))
    )
    assert "claimed_total 1 is below" in _text(
        verdicts.executor_state_errors(_state(claimed_total=1))
    )


def test_the_breadcrumb_log_marker_and_final_command_contracts() -> None:
    state = _state()
    crumb = {"counters": {"barrier_unavailable_holds_total": 3, "claimed_total": 3}}
    assert verdicts.breadcrumb_errors(crumb, state) == [], "matching breadcrumb passes"
    assert "never recorded a barrier hold" in _text(
        verdicts.breadcrumb_errors(
            {"counters": {"barrier_unavailable_holds_total": 0}}, state
        )
    )
    assert "no counters" in _text(verdicts.breadcrumb_errors({}, state))
    assert (
        verdicts.log_errors(f"... remote command held: {verdicts.HOLD_LOG}: ...") == []
    )
    assert "no 'multi-node barrier unavailable'" in _text(
        verdicts.log_errors("nothing")
    )
    assert verdicts.adapter_marker_errors(None) == [], "no marker means never reached"
    assert "recorded an execution" in _text(
        verdicts.adapter_marker_errors(
            {"operation": verdicts.OPERATION, "node_ids": ["a"]}
        )
    )
    assert verdicts.final_command_errors({"status": "WAITING"}) == [], "still open"
    assert "not fail-closed" in _text(
        verdicts.final_command_errors({"status": "SUCCEEDED"})
    )


def test_the_probe_definition_and_plan_are_plan_only_and_name_the_hard_stop() -> None:
    probe = cmd017.probe_definition()
    assert probe.script.is_file(), probe.script
    assert probe.owner == verdicts.OWNER and probe.pod == verdicts.POD, probe
    manifest = seeded.pod_manifest(
        probe,
        "img",
        {"executor_artifact_sha256": "a", "executor_compatibility_digest": "b"},
    )
    assert manifest["spec"]["containers"][0]["command"][-1] == (
        "/scripts/cmd017_barrier_executor.py"
    ), manifest["spec"]["containers"][0]["command"]
    parser = cmd017.parser()
    assert parser.parse_args(["--run-dir", "/tmp/run"]).execute is False, (
        "plan by default"
    )
    help_text = parser.format_help()
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in help_text, flag
    details = cmd017.plan_details()
    assert details["risk"] == "live-non-destructive", details
    assert details["expected_status_source"] == verdicts.STATUS_SOURCE, details
    assert "no reset is reachable" in details["hard_stop"], details["hard_stop"]
    assert "stand-in adapter is executed" in "\n".join(details["stop_conditions"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])
    for path in (
        ROOT / "scripts/e2e/regional/run_cmd017_barrier_hold.py",
        ROOT / "scripts/e2e/regional/probes/cmd017_barrier_executor.py",
    ):
        assert path.stat().st_mode & 0o777 == 0o775, path
        assert (
            path.read_text(encoding="utf-8").splitlines()[0] == "#!/usr/bin/env python3"
        )
