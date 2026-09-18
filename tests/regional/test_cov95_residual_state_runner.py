"""State-table audit boundaries using the real runner and fake CPU inventory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta

import pytest

from scripts.e2e.regional import run_state_table_acceptance as runner
from tests.regional import _cov95_residual_support as support
from tests.regional.test_state_table_acceptance import Regional, proof

residual_isolation = support.residual_isolation


@pytest.mark.parametrize("kind,mode", [("unknown", "dual"), ("workflow", "unknown")])
def test_invalid_audit_target_never_queries_a_cpu_role(kind, mode):
    regional = Regional()
    with pytest.raises(runner.RegionalFixtureError, match="unsupported"):
        runner.audit(
            regional,
            kind,
            mode,
            verify=True,
            deadline=support.NOW + timedelta(minutes=5),
        )
    assert regional.calls == []
    assert "unsupported state-table acceptance target" in runner.state_errors(
        proof(), kind, mode, verify=True
    )


@pytest.mark.parametrize("seconds", [None, 20])
def test_audit_requires_aware_sufficient_deadline_before_sql(monkeypatch, seconds):
    regional = Regional()

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return support.NOW

    monkeypatch.setattr(runner, "datetime", Clock)
    deadline = (
        support.NOW.replace(tzinfo=None)
        if seconds is None
        else support.NOW + timedelta(seconds=seconds)
    )
    with pytest.raises(runner.RegionalFixtureError, match="timezone|insufficient"):
        runner.audit(regional, "remote_command", "dual", verify=True, deadline=deadline)
    assert regional.pod_proofs == 0


def test_deadline_overrun_after_last_probe_invalidates_the_whole_audit(monkeypatch):
    regional = Regional()
    clock = support.Clock()
    original = regional.kubectl

    class ClockDate(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now()

    def command(*args, **kwargs):
        result = original(*args, **kwargs)
        if regional.pod_proofs == 3:
            clock.seconds = 61
        return result

    monkeypatch.setattr(regional, "kubectl", command)
    monkeypatch.setattr(runner, "datetime", ClockDate)
    with pytest.raises(runner.RegionalFixtureError, match="exceeded"):
        runner.audit(
            regional,
            "remote_command",
            "dual",
            verify=True,
            deadline=support.NOW + timedelta(seconds=60),
        )
    assert regional.pod_proofs == 3


def test_preflight_without_a_predecessor_still_checks_full_cpu_population(
    monkeypatch, tmp_path
):
    regional = Regional()
    monkeypatch.setattr(runner, "predecessor_path", lambda *args: (None, None))
    arguments = argparse.Namespace(
        case="GF-REGIONAL-BOOT-030",
        expected_mode="dual",
        run_dir=tmp_path,
        predecessor_evidence="",
    )
    result = runner.preflight(regional, arguments)
    assert result["predecessor"] == {"valid": True, "verdict": "NOT_REQUIRED"}
    assert set(result["population"]) == set(runner.CPU_ROLES)
    assert result["errors"] == []


def test_execute_refuses_changed_identity_before_starting_any_probe(tmp_path):
    regional = Regional()
    case = "GF-REGIONAL-BOOT-030"
    directory = tmp_path / "cases" / case
    directory.mkdir(parents=True)
    (directory / "plan.json").write_text(
        json.dumps({"details": {"identity": {"release_id": "other"}}})
    )
    arguments = argparse.Namespace(run_dir=tmp_path, case=case, expected_mode="dual")
    with pytest.raises(runner.RegionalFixtureError, match="identity changed"):
        runner.execute(
            regional,
            arguments,
            support.NOW + timedelta(minutes=5),
            {"errors": [], "identity": regional.evidence_identity()},
        )
    assert regional.calls == []


@pytest.mark.parametrize("report", [{}, {"state": []}, {"verdict": "FAIL"}])
def test_incomplete_probe_output_cannot_be_interpreted_as_a_clean_state_table(report):
    errors = runner.state_errors(report, "workflow", "dual", verify=True)
    assert "migration state is missing" in errors
    assert "schema/read-only writer proof is incomplete" in errors
