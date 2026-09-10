"""Contract tests for GF-REGIONAL-PREEMPT-037 (dispatcher loop liveness)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import preempt037_verdicts as verdicts
from scripts.e2e.regional import run_preempt037_dispatcher_liveness as preempt037

ROOT = Path(__file__).resolve().parents[2]
RULES = (ROOT / "deploy/observability/amp-rules.yaml").read_text(encoding="utf-8")
RUNBOOK = (ROOT / "docs/管理员日常运维.md").read_text(encoding="utf-8")


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


def test_the_shipped_rule_is_read_with_its_threshold_for_and_runbook() -> None:
    parameters = verdicts.stall_rule_parameters(RULES)
    assert (
        parameters["threshold_seconds"] == 300 and parameters["for_seconds"] == 300
    ), parameters
    assert parameters["metric"] == verdicts.DISPATCH_METRIC
    assert verdicts.rule_errors(parameters, RUNBOOK) == []
    assert "no runbook anchor" in _text(verdicts.rule_errors(parameters, "nothing"))
    with pytest.raises(ValueError, match="not defined"):
        verdicts.stall_rule_parameters("groups: []")


def _metrics(dispatch: float | None, periodic: float) -> str:
    lines = [f"gpu_fault_periodic_last_cycle_timestamp_seconds {periodic}"]
    if dispatch is not None:
        lines.append(
            f"gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds {dispatch}"
        )
    return "\n".join(lines)


def test_the_expression_is_evaluated_like_the_rule() -> None:
    now = 10_000.0
    assert (
        verdicts.stalled([_metrics(now - 301, now)], now=now, threshold_seconds=300)
        is True
    )
    assert (
        verdicts.stalled([_metrics(now - 10, now)], now=now, threshold_seconds=300)
        is False
    )
    assert (
        verdicts.stalled([_metrics(None, now)], now=now, threshold_seconds=300) is True
    ), "a replica that never ticked reads as stalled"
    assert (
        verdicts.stalled(
            [_metrics(now - 900, now), _metrics(now - 10, now)],
            now=now,
            threshold_seconds=300,
        )
        is False
    ), "max across replicas: one live replica keeps the fleet un-stalled"
    assert verdicts.periodic_alive(
        [_metrics(0, now - 10)], now=now, threshold_seconds=300
    ), "a fresh periodic stamp is alive"
    assert not verdicts.periodic_alive(
        [_metrics(0, now - 400)], now=now, threshold_seconds=300
    ), "an aged periodic stamp is not alive"


def _timeline(
    *flags: bool, periodic: bool = True, step: int = 30
) -> list[dict[str, Any]]:
    return [
        {
            "observed_epoch": 1000 + index * step,
            "stalled": flag,
            "periodic_alive": periodic,
        }
        for index, flag in enumerate(flags)
    ]


def test_the_stall_timeline_contract_passes_when_the_expression_holds_for_the_rule_for() -> (
    None
):
    timeline = _timeline(False, *([True] * 12))
    assert verdicts.stall_timeline_errors(timeline, for_seconds=300) == []


def test_each_stall_timeline_deviation_fails() -> None:
    assert "never became true" in _text(
        verdicts.stall_timeline_errors(_timeline(False, False), for_seconds=300)
    )
    assert "flickered" in _text(
        verdicts.stall_timeline_errors(
            _timeline(True, True, False, True), for_seconds=60
        )
    )
    assert "below the rule's for" in _text(
        verdicts.stall_timeline_errors(_timeline(True, True, True), for_seconds=300)
    )
    assert "whole process stalled" in _text(
        verdicts.stall_timeline_errors(
            _timeline(*([True] * 12), periodic=False), for_seconds=300
        )
    )
    assert "no stall samples" in _text(
        verdicts.stall_timeline_errors([], for_seconds=300)
    )


def test_quiescence_window_restore_and_recovery_contracts() -> None:
    assert (
        verdicts.quiescence_errors([{"request_id": "w", "status": "SUCCEEDED"}]) == []
    )
    assert "still in flight" in _text(
        verdicts.quiescence_errors([{"request_id": "w", "status": "RUNNING"}])
    )
    record = {
        "variable": verdicts.VARIABLE,
        "value": "false",
        "replicas": [{"pod": "p", "values": {verdicts.VARIABLE: "false"}}],
        "baseline": {"present": False, "value": None},
        "restored_state": {"present": False, "value": None},
        "replicas_after_close": [{"pod": "p", "values": {verdicts.VARIABLE: None}}],
    }
    assert verdicts.window_errors(record) == []
    assert "does not read the window value" in _text(
        verdicts.window_errors({**record, "replicas": [{"pod": "p", "values": {}}]})
    )
    assert verdicts.restore_errors(record) == []
    assert "not restored to the recorded baseline" in _text(
        verdicts.restore_errors(
            {**record, "restored_state": {"present": True, "value": "false"}}
        )
    )
    assert (
        verdicts.recovery_errors(
            [{"stalled": True}, {"stalled": False}], threshold_seconds=300
        )
        == []
    )
    assert "did not become fresh" in _text(
        verdicts.recovery_errors([{"stalled": True}], threshold_seconds=300)
    )


def test_the_runner_is_plan_by_default_and_names_the_service_action(
    tmp_path: Path,
) -> None:
    parser = preempt037.parser()
    arguments = parser.parse_args(["--run-dir", str(tmp_path)])
    assert arguments.execute is False and arguments.confirm == ""
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in parser.format_help(), flag
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", str(tmp_path), "--plan", "--execute"])
    assert preempt037.CASE_ID == "GF-REGIONAL-PREEMPT-037"
    assert preempt037.CONFIRMATION == "PREEMPT037_EXECUTE"
    assert verdicts.PREDECESSOR_CASE_ID == "GF-REGIONAL-PREEMPT-036"
    assert verdicts.VARIABLE == "GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER"
    assert "PENDING, RUNNING or WAITING" in "\n".join(preempt037.stop_conditions())


def test_the_restore_waits_for_the_value_the_processes_read_before_the_window() -> None:
    """The Deployment carried no literal entry, yet every process read ``true``
    from an envFrom ConfigMap; a restore that waited for ``None`` never
    completed (attempt 1, 2026-09-09)."""
    replicas = [
        {"pod": "w-1", "values": {verdicts.VARIABLE: "true"}},
        {"pod": "w-2", "values": {verdicts.VARIABLE: "true"}},
    ]
    assert verdicts.effective_variable_value(replicas) == "true"
    assert verdicts.effective_variable_value([{"pod": "w-1", "values": {}}]) is None
    with pytest.raises(ValueError, match="disagree"):
        verdicts.effective_variable_value(
            [*replicas, {"pod": "w-3", "values": {verdicts.VARIABLE: "false"}}]
        )


def test_the_restore_verdict_compares_replicas_with_the_pre_window_value() -> None:
    record = {
        "baseline": {"present": False, "value": None},
        "restored_state": {"present": False, "value": None},
        "effective_before": "true",
        "replicas_after_close": [
            {"pod": "w-1", "values": {verdicts.VARIABLE: "true"}},
            {"pod": "w-2", "values": {verdicts.VARIABLE: "true"}},
        ],
    }
    assert verdicts.restore_errors(record) == []
    record["replicas_after_close"].append({"pod": "w-3", "values": {}})
    errors = verdicts.restore_errors(record)
    assert len(errors) == 1 and "w-3" in errors[0] and "'true'" in errors[0], errors
    legacy = {**record, "replicas_after_close": [{"pod": "w-1", "values": {}}]}
    del legacy["effective_before"]
    assert verdicts.restore_errors(legacy) == [], (
        "without the record the literal baseline still applies"
    )
