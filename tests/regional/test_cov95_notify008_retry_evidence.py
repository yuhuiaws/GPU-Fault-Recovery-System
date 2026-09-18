from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from scripts.e2e.regional import notify008_runner as runner
from tests.regional._cov95_notify008_lifecycle import setup_run, uid


def old_pass(case_dir):
    path = case_dir / f"{runner.CASE_ID}.json"
    previous = {
        "case_id": runner.CASE_ID,
        "verdict": "PASS",
        "status": "COMPLETED",
        "execution_scope": "formal",
        "formal_sequence_satisfied": True,
        "release_id": "release-local",
        "cluster_id": "cluster-local",
    }
    path.write_text(json.dumps(previous))
    return path, previous


@pytest.mark.parametrize(
    "failure", ["plan", "bundle", "window", "target", "predecessor"]
)
def test_authorized_retry_invalidates_prior_pass_before_any_fallible_preflight(
    tmp_path, monkeypatch, failure
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    path, previous = old_pass(case_dir)
    plan_path = case_dir / "plan.json"
    if failure == "plan":
        plan_path.write_text("not-json")
    elif failure == "bundle":
        plan = json.loads(plan_path.read_text())
        plan["details"]["bundle_sha256"] = "0" * 64
        plan_path.write_text(json.dumps(plan))
    elif failure == "window":
        deadline = datetime.now(UTC) + timedelta(seconds=10)
    elif failure == "target":
        api.source["node"]["metadata"]["uid"] = uid(99)
    else:
        (tmp_path / "cases/GF-REGIONAL-NOTIFY-007/GF-REGIONAL-NOTIFY-007.json").unlink()

    assert runner.execute_case(settings, tmp_path, 2, deadline) == 1, (
        f"authorized {failure} failure must not retain the previous success"
    )
    result = json.loads(path.read_text())
    assert result["verdict"] == "FAIL" and result["status"] == "FAILED", (
        "the current canonical case report must be explicitly non-PASS"
    )
    assert not any(call[0] in {"create", "patch", "delete"} for call in api.calls), (
        "an early refusal must not issue remote mutations"
    )
    (history,) = case_dir.glob(f"{runner.CASE_ID}.before-2-*.json")
    assert json.loads(history.read_text()) == previous, (
        "the previous evidence must be retained as history, not remain canonical"
    )


def test_abort_before_resource_creation_preserves_nonpass_evidence(
    tmp_path, monkeypatch
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    path, _ = old_pass(case_dir)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "source_target", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.execute_case(settings, tmp_path, 2, deadline)
    assert json.loads(path.read_text())["verdict"] == "FAIL", (
        "an abort must never restore or leave the old PASS"
    )
    assert api.calls == [], "pre-resource abort must not invoke remote cleanup"


@pytest.mark.parametrize("abort", [False, True])
def test_cleanup_exception_preserves_failure_evidence_without_retrying_remote_commands(
    tmp_path, monkeypatch, abort
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    path, _ = old_pass(case_dir)
    if abort:
        api.fail_probe = "run"
        api.probe_exception = KeyboardInterrupt()
    snapshots = []

    def failed_cleanup(self):
        snapshots.append(list(api.calls))
        raise RuntimeError("local cleanup acknowledgement failure")

    monkeypatch.setattr(runner.Sandbox, "cleanup", failed_cleanup)
    if abort:
        with pytest.raises(KeyboardInterrupt):
            runner.execute_case(settings, tmp_path, 2, deadline)
    else:
        assert runner.execute_case(settings, tmp_path, 2, deadline) == 1, (
            "cleanup failure must override an otherwise successful probe"
        )
    result = json.loads(path.read_text())
    assert result["verdict"] == "FAIL" and result["status"] == "FAILED", (
        "a thrown cleanup error must still replace the canonical PASS"
    )
    assert result["cleanup"]["process_termination_proven"] is False, (
        "throwing cleanup cannot prove process termination"
    )
    assert len(snapshots) == 1 and api.calls == snapshots[0], (
        "recording cleanup failure must not make additional remote attempts"
    )
