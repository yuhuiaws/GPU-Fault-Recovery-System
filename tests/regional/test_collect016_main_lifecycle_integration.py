"""COLLECT-016 caller ordering and cleanup with all live boundaries replaced."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from gpu_fault.models import Environment
from gpu_fault.watcher import AttemptObservation, WorkloadPhase
from scripts.e2e.regional import run_collect016_training_recovery as runner


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    calls: list[str] = []
    job_id = "c016-actual-returned-job"
    attempts = {"source-attempt", "replacement-attempt"}
    regional_settings = SimpleNamespace(cluster_id="cluster-a")
    settings = runner.Settings(
        regional=regional_settings,
        site_file=tmp_path / "site.yaml",
        host_probe_image="unit-image",
        predecessor_path=tmp_path / "pred.json",
    )
    report = {
        "observations": [
            AttemptObservation(
                cluster_id="cluster-a",
                environment=Environment.HYPERPOD_EKS,
                job_id=job_id,
                attempt_id=attempt,
                workload_phase=WorkloadPhase.STOPPED,
                observed_at=datetime.now(timezone.utc),
                expected_critical_ranks=3,
                runtime_profile_version="profile-a",
                containers=[],
            ).model_dump(mode="json")
            for attempt in sorted(attempts)
        ]
    }

    def store_snapshot(**kwargs: Any) -> dict[str, Any]:
        if not kwargs:
            calls.append("cleanup-profile")
            return {"profile": {"profile_version": "profile-a"}}
        assert kwargs == {"job_id": job_id, "queue_attempts": 1}
        calls.append("drain")
        return report

    regional = SimpleNamespace(
        settings=regional_settings,
        evidence_identity=lambda: {
            "cluster_id": "cluster-a",
            "release_id": "unit-release",
        },
        store_snapshot=Mock(side_effect=store_snapshot),
        cpu_blast_snapshot=lambda: {},
    )
    workload = SimpleNamespace(
        settings=SimpleNamespace(job_id=job_id, attempt_id="source-attempt"),
        delete=Mock(side_effect=lambda: calls.append("delete-workload")),
    )
    probe = SimpleNamespace(
        cleanup=Mock(side_effect=lambda: calls.append("cleanup-probe"))
    )
    prewarm = SimpleNamespace(
        create=Mock(side_effect=lambda _nodes: calls.append("prewarm")),
        cleanup=Mock(side_effect=lambda: calls.append("cleanup-prewarm") or {}),
    )
    ab = {
        "errors": [],
        "a": {
            "job_id": job_id,
            "source_attempt_id": "source-attempt",
            "target_attempt_id": "replacement-attempt",
        },
        "b": {},
    }

    def restart_sections(
        *_args: Any, workloads: list[Any], fixtures: list[Any], **_kwargs: Any
    ) -> dict[str, Any]:
        calls.append("a-b")
        workloads.append(workload)
        fixtures.append(probe)
        return ab

    reset = Mock(
        side_effect=lambda *a, **k: calls.append("reset") or {"errors": [], "d": {}}
    )
    cleanup = SimpleNamespace(
        seed_markers=["unit-seed"],
        incident_states=[],
        finish=Mock(
            side_effect=lambda **_: calls.append("finish-incidents") or {"errors": []}
        ),
    )
    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(runner, "ImagePrewarmFixture", lambda *a, **k: prewarm)
    monkeypatch.setattr(runner.base, "CaseCleanup", lambda: cleanup)
    monkeypatch.setattr(
        runner,
        "read_only_preflight",
        lambda *a: {
            "errors": [],
            "candidate_nodes": [{"name": "node-a"}],
            "cpu_blast": {},
        },
    )
    monkeypatch.setattr(runner, "run_restart_budget_sections", restart_sections)
    monkeypatch.setattr(runner, "run_reset_section", reset)
    return SimpleNamespace(
        settings=settings,
        regional=regional,
        report=report,
        ab=ab,
        workload=workload,
        prewarm=prewarm,
        probe=probe,
        reset=reset,
        cleanup=cleanup,
        calls=calls,
        root=tmp_path,
    )


def execute(case: SimpleNamespace) -> tuple[int, dict[str, Any]]:
    code = runner.execute_case(
        case.settings, case.root, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    result = json.loads(
        (case.root / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json").read_text()
    )
    return code, result


def assert_cleanup(case: SimpleNamespace) -> None:
    assert case.calls[-4:] == [
        "finish-incidents",
        "delete-workload",
        "cleanup-probe",
        "cleanup-prewarm",
    ]
    case.cleanup.finish.assert_called_once_with(
        profile_version="profile-a", reason="COLLECT-016 validated cleanup"
    )
    case.probe.cleanup.assert_called_once()
    case.prewarm.cleanup.assert_called_once()


def test_reset_follows_deleted_and_drained_returned_job_then_all_cleanup(
    case: SimpleNamespace,
) -> None:
    code, result = execute(case)

    assert code == 0 and result["verdict"] == "PASS", result
    assert result["a_drain"]["job_id"] == case.ab["a"]["job_id"]
    assert case.calls[:5] == ["prewarm", "a-b", "delete-workload", "drain", "reset"]
    case.reset.assert_called_once()
    assert_cleanup(case)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "foreign-job"),
        ("job_id", ""),
        ("job_id", None),
        ("job_id", 123),
        ("source_attempt_id", "foreign-attempt"),
        ("source_attempt_id", None),
        ("target_attempt_id", ""),
        ("target_attempt_id", None),
        ("target_attempt_id", "source-attempt"),
    ],
)
def test_unbound_a_result_refuses_drain_and_reset_but_cleans_owned_resources(
    case: SimpleNamespace, field: str, value: Any
) -> None:
    case.ab["a"][field] = value

    code, result = execute(case)

    assert code == 1 and "identity is unbound" in result["error"], result
    assert "drain" not in case.calls
    case.reset.assert_not_called()
    assert_cleanup(case)


def test_missing_a_job_refuses_reset_and_keeps_cleanup(case: SimpleNamespace) -> None:
    case.ab["a"].pop("job_id")

    code, result = execute(case)

    assert code == 1 and result["verdict"] == "FAIL"
    case.reset.assert_not_called()
    assert_cleanup(case)


@pytest.mark.parametrize("change", ["missing-report", "missing-attempt", "wrong-job"])
def test_invalid_store_drain_report_never_allows_reset(
    case: SimpleNamespace, change: str
) -> None:
    if change == "missing-report":
        case.report.pop("observations")
    elif change == "missing-attempt":
        case.report["observations"].pop()
    else:
        case.report["observations"][0]["job_id"] = "foreign-job"

    code, result = execute(case)

    assert code == 1 and result["verdict"] == "FAIL", result
    assert "drain" in case.calls
    proof = json.loads(
        (case.root / "cases" / runner.CASE_ID / "a-drain.json").read_text()
    )
    assert proof["drained"] is False
    case.reset.assert_not_called()
    assert_cleanup(case)


@pytest.mark.parametrize(
    "report",
    [
        None,
        {},
        {"drained": False},
        {"drained": "true"},
        {"drained": True, "job_id": "foreign"},
        {
            "drained": True,
            "job_id": "c016-actual-returned-job",
            "cluster_id": "foreign",
        },
        {
            "drained": True,
            "job_id": "c016-actual-returned-job",
            "cluster_id": "cluster-a",
            "expected_attempt_ids": ["source-attempt"],
        },
    ],
)
def test_invalid_returned_drain_proof_refuses_reset(
    case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, report: Any
) -> None:
    monkeypatch.setattr(runner, "wait_a_workload_drained", Mock(return_value=report))

    code, result = execute(case)

    assert code == 1 and "drain proof is missing or unbound" in result["error"], result
    case.reset.assert_not_called()
    assert_cleanup(case)


def test_expired_drain_still_finishes_incident_workload_probe_and_prewarm_cleanup(
    case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "wait_a_workload_drained",
        Mock(
            side_effect=runner.RegionalFixtureError("A-phase workload drain timed out")
        ),
    )

    code, result = execute(case)

    assert code == 1 and result["verdict"] == "FAIL", result
    case.reset.assert_not_called()
    assert_cleanup(case)


def test_failed_workload_deletion_prevents_the_drain_and_reset(
    case: SimpleNamespace,
) -> None:
    case.workload.delete.side_effect = [
        runner.RegionalFixtureError("delete failed"),
        None,
    ]

    code, result = execute(case)

    assert code == 1 and "delete failed" in result["error"], result
    assert "drain" not in case.calls
    case.reset.assert_not_called()
    assert case.workload.delete.call_count == 2
    case.probe.cleanup.assert_called_once()
    case.prewarm.cleanup.assert_called_once()


@pytest.mark.parametrize("replacement_missing", [False, True])
def test_a_observes_the_replacement_before_b_and_returns_both_attempt_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement_missing: bool
) -> None:
    job_id = "c016-a-unit"
    settings = runner.Settings(
        regional=SimpleNamespace(cluster_id="cluster-a"),
        site_file=tmp_path / "site.yaml",
        host_probe_image="unit-image",
        predecessor_path=tmp_path / "pred.json",
    )
    source_pods = [{"uid": f"old-{rank}", "node": f"node-{rank}"} for rank in range(3)]
    target_pods = [{"uid": f"new-{rank}", "node": f"node-{rank}"} for rank in range(3)]
    workload = SimpleNamespace(
        submit=Mock(), wait_running=Mock(return_value={"pods": source_pods})
    )
    collector = SimpleNamespace(
        create=Mock(),
        snapshot=lambda: {"gpu_inventory": [{"pci_bdf": "0000:59:00.0"}]},
        execute=Mock(),
    )
    state_a = {
        "incident": {"workload_identity_source": "SOLE_ACTIVE_ATTEMPT_ON_NODE"},
        "workflow": {
            "official_steps": [
                {"operation": "RESTART_WORKLOAD", "parameters": {"job_id": job_id}}
            ]
        },
    }
    state_b = {
        "workflow": {
            "status": "FAILED",
            "completed_operations": ["FREEZE_EVIDENCE", "STOP_WORKLOADS"],
            "step_executions": [
                {
                    "operation": "RESTART_WORKLOAD",
                    "status": "FAILED",
                    "details": {"reason": "RESTART_BUDGET_EXHAUSTED"},
                }
            ],
        },
        "commands": [{"step": {"operation": "STOP_WORKLOADS"}}],
        "restart_budget": {"restart_count": 1},
    }
    regional = SimpleNamespace(wait_for_workflow=Mock(side_effect=[state_a, state_b]))
    replacement = Mock(
        return_value={
            "pods": target_pods,
            "observation": {"attempt_id": "verified-replacement"},
        },
        side_effect=(
            runner.RegionalFixtureError("replacement observation is missing")
            if replacement_missing
            else None
        ),
    )
    monkeypatch.setattr(runner, "managed_fixture", lambda *a, **k: workload)
    monkeypatch.setattr(
        runner.base, "render_named_training_manifest", lambda path, **k: path
    )
    monkeypatch.setattr(runner, "CollectorAcceptanceFixture", lambda *a, **k: collector)
    monkeypatch.setattr(runner.workload_case, "wait_observation", Mock())
    monkeypatch.setattr(runner.workload_case, "workflow_errors", Mock(return_value=[]))
    monkeypatch.setattr(runner, "capture_post_restart_workload", replacement)
    workloads: list[Any] = []
    fixtures: list[Any] = []
    cleanup = SimpleNamespace(register_seed=Mock())
    arguments = {"workloads": workloads, "fixtures": fixtures, "cleanup": cleanup}

    if replacement_missing:
        with pytest.raises(
            runner.RegionalFixtureError, match="replacement observation"
        ):
            runner.run_restart_budget_sections(
                settings, regional, tmp_path, "unit", **arguments
            )
        assert collector.execute.call_count == 1
        assert regional.wait_for_workflow.call_count == 1
    else:
        result = runner.run_restart_budget_sections(
            settings, regional, tmp_path, "unit", **arguments
        )
        assert result["errors"] == [], result
        assert result["a"]["job_id"] == job_id
        assert result["a"]["source_attempt_id"] == f"{job_id}-a001"
        assert result["a"]["target_attempt_id"] == "verified-replacement"
        assert (
            regional.wait_for_workflow.call_args.kwargs["attempt_id"]
            == "verified-replacement"
        )
        assert collector.execute.call_count == 2
    replacement.assert_called_once()
    assert replacement.call_args.kwargs["source_uids"] == {
        pod["uid"] for pod in source_pods
    }
    assert workloads == [workload] and fixtures == [collector]
