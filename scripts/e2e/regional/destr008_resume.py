"""Cleanup-only recovery of a previously started DESTR-008 execution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.e2e.regional.destr008_controller_lock import controller_ownership
from scripts.e2e.regional.destr008_journal import ExecutionJournal, ScenarioState
from scripts.e2e.regional.destr008_safety import ShortageSafety
from scripts.e2e.regional.managed_workload_fixture import (
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
)
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture

if TYPE_CHECKING:
    from scripts.e2e.regional.run_destr008_warm_spare_shortage import Settings


def resume_scenario(
    settings: Settings,
    *,
    regional: RegionalLiveFixture,
    warm: WarmSpareLiveFixture,
    journal: ExecutionJournal,
    scenario: str,
    state: ScenarioState,
    case_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    from scripts.e2e.regional import run_destr008_warm_spare_shortage as case

    result: dict[str, Any] = {
        "scenario": scenario,
        "cleanup_only": True,
        "verdict": "FAIL",
        "errors": [],
    }
    directory = case_dir / "scenarios" / scenario
    record = journal.record
    job_id, attempt_id = case.scenario_identity(run_dir, record.attempt, scenario)
    event_id = f"destr008-{scenario}-{job_id}-event"
    fixture = case.ScenarioFixture(
        settings,
        warm,
        scenario=scenario,
        run_id=job_id,
        state_directory=directory / "host-probes",
    )
    with controller_ownership(directory / "scenario-owner.json"):
        try:
            bounded = scenario in {*case.SERVICE_UNIT, "active-gpu-pod"}
            if not bounded and state.fixture_started:
                raise RegionalFixtureError(
                    "interrupted metadata shortage requires original mutation reconciliation"
                )
            safety: ShortageSafety | None = None
            proof: dict[str, Any] = {}
            if state.safety_started:
                safety = ShortageSafety(
                    regional,
                    run_id=job_id,
                    attempt_id=attempt_id,
                    event_id=event_id,
                    fault_node=settings.fault_node,
                    spare_node=settings.spare_node,
                    spare_uid=record.spare_uid,
                    release_id=record.release_id,
                    directory=directory / "safety",
                )
                proof = safety.resume_cleanup()
                result["independent_safety_cleanup"] = proof
                if proof["quiescent"] is not True or proof["retired"] is not True:
                    raise RegionalFixtureError(
                        "original cancellation cleanup is unproven"
                    )
            incident_id = ""
            if state.post_started:
                receipt = proof.get("receipt")
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("source_complete") is not True
                    or receipt.get("producer_revoked") is not True
                ):
                    raise RegionalFixtureError(
                        "original producer completion is unproven"
                    )
                root = receipt.get("root")
                if isinstance(root, dict):
                    incident_id = str(root["incident_id"])
                    warm.wait_incident_idle(incident_id)
            workload_path = directory / "workload-owner.json"
            if state.workload_started and not workload_path.is_file():
                raise RegionalFixtureError(
                    "original workload ownership journal is missing"
                )
            if workload_path.exists() or workload_path.is_symlink():
                workload = ManagedWorkloadFixture(
                    regional,
                    ManagedWorkloadSettings(
                        manifest=directory / "pinned-workload.yaml",
                        site_file=settings.site_file,
                        job_id=job_id,
                        attempt_id=attempt_id,
                        restart_budget=1,
                        expected_pods=1,
                        expected_gpu_count=8,
                    ),
                    state_path=workload_path,
                )
                workload.delete()
            if state.fixture_started:
                if safety is None or safety.plan is None:
                    raise RegionalFixtureError(
                        "original bounded fixture plan is missing"
                    )
                fixture.bind_safety(
                    safety.plan,
                    datetime.fromtimestamp(record.maintenance_expires_at, timezone.utc),
                )
                result["scenario_cleanup"] = fixture.resume_cleanup()
                if result["scenario_cleanup"]["errors"]:
                    raise RegionalFixtureError("original bounded fixture remains")
            result["fault_cleanup"] = case.restore_fault_node(
                warm,
                settings=settings,
                incident_id=incident_id,
                profile_version=record.profile_version,
            )
            if result["fault_cleanup"]["errors"]:
                raise RegionalFixtureError(
                    "original fault-node restoration is incomplete"
                )
            case.audit_scenario_nodes(warm, settings, result)
            if result.get("postflight_errors") or "postflight_error" in result:
                raise RegionalFixtureError("original node baseline is not restored")
            result["cleanup_complete"] = True
        except Exception as exc:
            result["errors"].append(f"{type(exc).__name__}: {exc}")
            result["cleanup_complete"] = False
        finally:
            try:
                fixture.close()
            except Exception as exc:
                result["errors"].append(
                    f"fixture ownership close: {type(exc).__name__}"
                )
                result["cleanup_complete"] = False
        if result["cleanup_complete"]:
            try:
                journal.cleaned_scenario(scenario)
            except Exception as exc:
                result["errors"].append(f"cleanup checkpoint: {type(exc).__name__}")
                result["cleanup_complete"] = False
    return result


def resume_execution(
    settings: Settings,
    *,
    regional: RegionalLiveFixture,
    warm: WarmSpareLiveFixture,
    journal: ExecutionJournal,
    case_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    from scripts.e2e.regional import run_destr008_warm_spare_shortage as case

    result: dict[str, Any] = {
        "case_id": case.CASE_ID,
        "attempt": journal.record.attempt,
        "verdict": "FAIL",
        "cleanup_only": True,
        "errors": [],
        "scenarios": [],
    }
    try:
        record = journal.record
        scope = regional.evidence_identity()
        if (
            scope.get("release_id") != record.release_id
            or scope.get("cluster_id") != settings.regional.cluster_id
            or warm.node_snapshot(settings.fault_node)["uid"] != record.fault_uid
            or warm.node_snapshot(settings.spare_node)["uid"] != record.spare_uid
        ):
            raise RegionalFixtureError(
                "original cleanup release or Node identity drifted"
            )
        for scenario, state in record.scenarios.items():
            if state.state == "STARTED":
                report = resume_scenario(
                    settings,
                    regional=regional,
                    warm=warm,
                    journal=journal,
                    scenario=scenario,
                    state=state,
                    case_dir=case_dir,
                    run_dir=run_dir,
                )
                result["scenarios"].append(report)
                if report["cleanup_complete"] is not True:
                    result["errors"].append(
                        "original scenario cleanup remains unresolved"
                    )
        prewarm_path = case_dir / "prewarm-owner.json"
        if record.prewarm_started and not prewarm_path.is_file():
            raise RegionalFixtureError("original prewarm ownership journal is missing")
        if prewarm_path.exists() or prewarm_path.is_symlink():
            prewarm = ImagePrewarmFixture(
                regional,
                case_id=case.CASE_ID,
                run_id=case.scenario_identity(run_dir, record.attempt, "prewarm")[0],
                state_path=prewarm_path,
            )
            residuals = prewarm.cleanup()
            result["prewarm_residuals"] = residuals
            if any(residuals.values()):
                raise RegionalFixtureError("original prewarm resources remain")
        journal.cleaned_prewarm()
        if not result["errors"]:
            journal.complete()
        result["cleanup_complete"] = not result["errors"]
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["cleanup_complete"] = False
    return result
