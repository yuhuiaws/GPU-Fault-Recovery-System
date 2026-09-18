from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_managed import ManagedHarness, case


def test_managed_residual_audit_requires_a_successful_absence_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)

    def unavailable(value: Any, args: Any, kwargs: Any) -> Any:
        if "--ignore-not-found" in args:
            if kwargs.get("check", True):
                raise RegionalFixtureError("fake absence read unavailable")
            return ""
        return value

    h.transforms["regional.kubectl"] = unavailable
    code, report = h.execute(tmp_path)
    assert code == 1 and report["workload_residuals"]["D"] is True, report
    assert "absence read unavailable" in report["group_d_residual_error"], report


@pytest.mark.parametrize("rerun", [False, True])
def test_managed_case_orders_groups_and_keeps_unprovisioned_group_not_run(
    rerun: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch, rerun=rerun)
    h.plan(tmp_path)
    code, report = h.execute(tmp_path)
    assert code == 0 and report["errors"] == [], report
    assert report["groups_not_run"] == ["C"], report
    assert report["groups"]["C"]["verdict"] == "NOT_RUN", report
    assert report["groups"]["A"]["group_a_source"] == (
        case.GROUP_A_WORKLOAD_SOURCE if rerun else case.GROUP_A_EVIDENCE_SOURCE
    ), report
    assert h.jobs["job-d"].deleted and h.jobs["job-d"].annotation is None, h.jobs
    assert ("job-a" in h.jobs) is rerun, h.jobs
    names = [name for name, _, _ in h.calls]
    assert names.count("workload.job-d.delete") == 1, names
    assert names.count("workload.job-a.delete") == int(rerun), names
    assert h.jobs["job-d"].restart_state == h.state_d
    assert names.index("workload.job-d.authorize_restart") < names.index(
        "workload.job-d.wait_restarted"
    )
    if rerun:
        assert h.jobs["job-a"].restart_state == h.state_a


@pytest.mark.parametrize(
    "phase",
    [
        "prewarm.create",
        "workload.job-a.submit",
        "workload.job-d.submit",
        "workload.job-d.annotate_auto_resume",
        "workload.job-d.authorize_restart",
        "workload.job-d.wait_restarted",
        "regional.provider_events",
        "prewarm.cleanup",
        "regional.verify_runtime_identity",
    ],
)
def test_managed_phase_failure_stops_advancement_and_checks_residuals(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch, rerun=True)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    if phase == "prewarm.cleanup":
        assert (
            report["prewarm_cleanup_error"]
            == "RuntimeError: fake failure at prewarm.cleanup"
        ), report
        assert report["prewarm_residuals"] == {"cleanup_error": True}, report
    else:
        assert "error" in report or report.get("errors"), report
    names = [name for name, _, _ in h.calls]
    assert "prewarm.cleanup" in names, names
    if phase == "workload.job-a.submit":
        assert "workload.job-d.submit" not in names, names
    if phase == "workload.job-d.authorize_restart":
        assert "workload.job-d.wait_restarted" not in names


@pytest.mark.parametrize(
    "defect",
    [
        "bad-block",
        "pod-changed",
        "annotation-remains",
        "blocked-receipt",
        "retry-receipt",
    ],
)
def test_managed_negative_guard_must_be_proven_before_retry(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "bad-block":
        h.blocked["workflow"]["step_executions"][0]["error"] = "unrelated"
    elif defect == "pod-changed":
        calls = 0

        def snapshot(value: dict[str, Any], *_args: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 2:
                value["pods"][0]["uid"] = "changed-before-refusal"
            return value

        h.transforms["workload.job-d.snapshot"] = snapshot
    elif defect == "annotation-remains":
        h.ignore_annotation_clear = True
    else:

        def receipt(value: dict[str, Any], args: Any, kwargs: Any) -> dict[str, Any]:
            target = "-block-" if defect == "blocked-receipt" else "-retry-"
            return (
                {"receipt": {"status": 500}}
                if target in args[0]["record_id"]
                else value
            )

        h.transforms["regional.post_xid_event"] = receipt
    code, report = h.execute(tmp_path)
    assert code == 1 and "group D failed" in report["error"], report
    posts = [
        args[0]["record_id"]
        for name, args, _ in h.calls
        if name == "regional.post_xid_event"
    ]
    if defect != "retry-receipt":
        assert not any("-retry-" in marker for marker in posts), posts
    assert h.jobs["job-d"].deleted, h.jobs


@pytest.mark.parametrize(
    "phase",
    [
        "before",
        "workload.job-a.wait_running",
        "workload.job-d.wait_running",
        "workload.job-d.annotate_auto_resume",
    ],
)
def test_managed_group_rechecks_window_before_mutations(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch, rerun=True)
    h.plan(tmp_path)
    if phase == "before":
        h.clock.sleep(61)
    else:
        h.advance_at[phase] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and report["error"], report
    assert "prewarm.cleanup" in [name for name, _, _ in h.calls], h.calls


@pytest.mark.parametrize(
    "defect", ["profile", "namespace", "missing-pods", "invalid-json"]
)
def test_group_b_audit_refuses_drift_and_auto_resume_in_managed_namespaces(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch)
    if defect == "profile":
        h.profile_disagreement = True
    elif defect == "namespace":
        h.namespace_objects = [
            {
                "kind": "Job",
                "metadata": {
                    "name": "violating",
                    "annotations": {case.AUTO_RESUME_ANNOTATION: "true"},
                },
            }
        ]
    elif defect == "missing-pods":
        h.empty_pods = True
    else:
        h.invalid_profile_response = True
    if defect in {"missing-pods", "invalid-json"}:
        with pytest.raises(RegionalFixtureError):
            case.group_b_audit(h.regional, profile_version="profile-a")
    else:
        result = case.group_b_audit(h.regional, profile_version="profile-a")
        assert result["errors"], result


def test_managed_abort_keeps_cleanup_local_and_does_not_start_next_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch, rerun=True)
    h.plan(tmp_path)
    h.failures["workload.job-a.submit"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _, _ in h.calls]
    assert "workload.job-a.delete" in names and "prewarm.cleanup" in names, names
    assert "workload.job-d.submit" not in names, names


def test_managed_group_b_failure_and_plan_drift_refuse_before_prewarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.preflight["group_b"]["errors"] = ["fake unsafe namespace"]
    code, report = h.execute(tmp_path)
    assert code == 1 and "group B failed" in report["error"], report
    assert "prewarm.create" not in [name for name, _, _ in h.calls], h.calls
    h.preflight["release_id"] = "changed"
    with pytest.raises(RegionalFixtureError, match="plan drifted"):
        h.execute(tmp_path)
