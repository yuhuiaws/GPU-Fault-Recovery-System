"""Cleanup-only routing over real journals and fake external resource owners."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional import destr008_journal as journal_module
from scripts.e2e.regional import destr008_resume as resume
from scripts.e2e.regional import run_destr008_warm_spare_shortage as case
from scripts.e2e.regional.destr008_journal import ExecutionJournal, Stage
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveSettings,
)
from tests.regional.test_destr008_journal import (
    STAGES,
    execution_record,
    make_journal,
    reload_journal,
)


class Harness:
    """Record cleanup I/O without replacing either journal or resume routing."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.run_dir = tmp_path / "approved-run"
        self.case_dir = self.run_dir / "cases" / case.CASE_ID
        self.journal = make_journal(
            self.case_dir, scenarios={name: {} for name in case.SCENARIOS}
        )
        for filename in ("cpu-config", "gpu-config"):
            (tmp_path / filename).write_text("apiVersion: v1\n")
        (tmp_path / "site.yaml").write_text("scope: hermetic-test\n")
        self.settings = case.Settings(
            regional=RegionalLiveSettings(
                cpu_kubeconfig=tmp_path / "cpu-config",
                gpu_kubeconfig=tmp_path / "gpu-config",
                gpu_context="gpu-context",
                namespace="gpu-fault-system",
                cluster_id="cluster-a",
                region="us-west-2",
            ),
            site_file=tmp_path / "site.yaml",
            manifest=tmp_path / "workload.yaml",
            hyperpod_cluster="hyperpod-a",
            fault_node="fault-a",
            spare_node="spare-a",
            host_probe_image="registry.example/probe@sha256:" + "a" * 64,
            scenarios=case.SCENARIOS,
            predecessor_path=tmp_path / "predecessor.json",
        )
        self.calls: list[str] = []
        self.arguments: dict[str, list[dict[str, Any]]] = {}
        self.failures: dict[str, Exception] = {}
        self.scope: Any = {"release_id": "release-original", "cluster_id": "cluster-a"}
        self.nodes: dict[str, Any] = {
            "fault-a": {"uid": "fault-original-uid"},
            "spare-a": {"uid": "spare-original-uid"},
        }
        self.scenario = "active-gpu-pod"
        self.plan: wire.Plan | None = None
        self.proof: Any = {}
        self.fixture_result: Any = {"errors": []}
        self.restore_result: Any = {"errors": []}
        self.audit_result: dict[str, Any] = {}
        self.prewarm_residuals: Any = {"pods": [], "jobs": []}
        self.regional: Any = SimpleNamespace(
            settings=self.settings.regional,
            evidence_identity=self.evidence_identity,
            api_request=self.forbidden,
            run=self.forbidden,
        )
        self.warm: Any = SimpleNamespace(
            node_snapshot=self.node_snapshot,
            wait_incident_idle=self.wait_incident_idle,
            create_restore_workflow=self.forbidden,
            run=self.forbidden,
        )
        monkeypatch.setattr(case, "ScenarioFixture", self.fixture)
        monkeypatch.setattr(resume, "ShortageSafety", self.safety)
        monkeypatch.setattr(resume, "ManagedWorkloadFixture", self.workload)
        monkeypatch.setattr(resume, "ImagePrewarmFixture", self.prewarm)
        monkeypatch.setattr(case, "restore_fault_node", self.restore_fault)
        monkeypatch.setattr(case, "audit_scenario_nodes", self.audit)

    @property
    def directory(self) -> Path:
        return self.case_dir / "scenarios" / self.scenario

    def note(self, name: str, **values: Any) -> None:
        self.calls.append(name)
        self.arguments.setdefault(name, []).append(values)
        if name in self.failures:
            raise self.failures[name]

    def forbidden(self, *args: Any, **kwargs: Any) -> None:
        pytest.fail("cleanup-only recovery attempted new execution or external I/O")

    def evidence_identity(self) -> Any:
        self.note("identity")
        return copy.deepcopy(self.scope)

    def node_snapshot(self, node: str) -> Any:
        self.note("snapshot", node=node)
        return copy.deepcopy(self.nodes[node])

    def wait_incident_idle(self, incident_id: str) -> None:
        self.note("idle", incident_id=incident_id)

    def fixture(self, settings: case.Settings, warm: Any, **kwargs: Any) -> Any:
        assert settings is self.settings and warm is self.warm
        self.note("fixture-init", **kwargs)

        def bind(plan: wire.Plan, expires_at: datetime) -> None:
            self.note("fixture-bind", plan=plan, expires_at=expires_at)

        def cleanup() -> Any:
            self.note("fixture-cleanup")
            return copy.deepcopy(self.fixture_result)

        return SimpleNamespace(
            bind_safety=bind,
            resume_cleanup=cleanup,
            close=lambda: self.note("fixture-close"),
            apply=self.forbidden,
            stop_service=self.forbidden,
        )

    def safety(self, regional: Any, **kwargs: Any) -> Any:
        assert regional is self.regional
        self.note("safety-init", **kwargs)

        def cleanup() -> Any:
            self.note("safety-cleanup")
            return copy.deepcopy(self.proof)

        return SimpleNamespace(
            plan=self.plan,
            resume_cleanup=cleanup,
            arm=self.forbidden,
            before_post=self.forbidden,
            acknowledge=self.forbidden,
        )

    def workload(self, regional: Any, settings: Any, **kwargs: Any) -> Any:
        assert regional is self.regional
        self.note("workload-init", settings=settings, **kwargs)
        return SimpleNamespace(
            delete=lambda: self.note("workload-delete"), submit=self.forbidden
        )

    def prewarm(self, regional: Any, **kwargs: Any) -> Any:
        assert regional is self.regional
        self.note("prewarm-init", **kwargs)

        def cleanup() -> Any:
            self.note("prewarm-cleanup")
            return copy.deepcopy(self.prewarm_residuals)

        return SimpleNamespace(cleanup=cleanup, prepare=self.forbidden)

    def restore_fault(self, warm: Any, **kwargs: Any) -> Any:
        assert warm is self.warm
        self.note("fault-restore", **kwargs)
        return copy.deepcopy(self.restore_result)

    def audit(self, warm: Any, settings: case.Settings, result: dict[str, Any]) -> None:
        assert warm is self.warm and settings is self.settings
        self.note("audit")
        result.update(copy.deepcopy(self.audit_result))

    def seed(
        self,
        *,
        scenario: str = "active-gpu-pod",
        stages: tuple[Stage, ...] = STAGES,
        start_scenario: bool = True,
    ) -> None:
        self.scenario = scenario
        self.journal.start_prewarm()
        write_json_atomic(
            self.case_dir / "prewarm-owner.json", {"fake_owner": "original"}
        )
        if start_scenario:
            self.journal.start_scenario(scenario)
            for stage in stages:
                self.journal.stage(scenario, stage)
            if "workload_started" in stages:
                write_json_atomic(
                    self.directory / "workload-owner.json", {"fake_owner": "original"}
                )
                (self.directory / "pinned-workload.yaml").write_text("kind: Job\n")
        job, attempt = case.scenario_identity(self.run_dir, 7, scenario)
        self.plan = wire.Plan(
            schema_version=1,
            run_id=job,
            cluster_id="cluster-a",
            job_id=job,
            attempt_id=attempt,
            event_id=f"destr008-{scenario}-{job}-event",
            release_id="release-original",
            fault_node="fault-a",
            spare_node="spare-a",
            runtime_profile_version="profile-original",
            workload_ids=[f"training/{job}"],
            probe_sha256="b" * 64,
            created_at=1699999400,
            deadline_at=1699999880,
            fence=wire.Fence(
                policy="policy-original",
                policy_uid="policy-original-uid",
                binding="binding-original",
                binding_uid="binding-original-uid",
                marker="c" * 64,
                node="spare-a",
                node_uid="spare-original-uid",
            ),
        )
        self.set_receipt(submitted="post_started" in stages and start_scenario)

    def set_receipt(self, *, submitted: bool) -> None:
        assert self.plan is not None
        bound = self.plan
        now = bound.created_at
        control = wire.initial_control(bound)
        root = None
        if submitted:
            control = wire.claim_submission(
                bound, control, claim_id="claim-original", now=now + 1
            )
            acknowledgement = wire.Acknowledgement(
                claim_id="claim-original",
                event_id=bound.event_id,
                completed_at=now + 2,
                incident_id="incident-original",
                workflow_request_id="workflow-original",
            )
            control = wire.acknowledge_submission(
                bound, control, acknowledgement=acknowledgement, now=now + 2
            )
            root = wire.Root(
                incident_id="incident-original", workflow_request_id="workflow-original"
            )
        control = wire.revoke(control, now=now + 3, reason="PARENT_CLOSE")
        receipt = wire.receipt(
            bound,
            control,
            None,
            uid="control-original-uid",
            now=now + 4 + wire.QUIET_SECONDS,
            state="QUIESCENT",
            source_complete=True,
            root=root,
            workflow_ids=[] if root is None else [root.workflow_request_id],
            commands_active=0,
            workflows_active=0,
            pending_creation=False,
            inventory_sha256="d" * 64,
            quiet_since=now + 4,
            monitoring=False,
        )
        self.proof = {
            "quiescent": True,
            "retired": True,
            "errors": [],
            "receipt": receipt.model_dump(mode="json"),
        }

    def run(self) -> dict[str, Any]:
        self.journal = reload_journal(self.journal)
        assert self.journal.resuming is True
        return resume.resume_execution(
            self.settings,
            regional=self.regional,
            warm=self.warm,
            journal=self.journal,
            case_dir=self.case_dir,
            run_dir=self.run_dir,
        )

    def assert_unfinished(self, result: dict[str, Any]) -> dict[str, Any]:
        assert result["verdict"] == "FAIL"
        assert result["cleanup_only"] is True
        assert result["cleanup_complete"] is False
        assert result["errors"]
        assert reload_journal(self.journal).record.completed is False
        report = result["scenarios"][0]
        assert isinstance(report, dict), (
            "scenario cleanup must return a structured report"
        )
        assert report["verdict"] == "FAIL"
        assert report["cleanup_complete"] is False
        assert report["errors"]
        assert (
            reload_journal(self.journal).record.scenarios[self.scenario].state
            == "STARTED"
        )
        assert self.calls.count("fixture-close") == 1
        return report


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "scenario", ["active-gpu-pod", "agent-unavailable", "kubernetes-not-ready"]
)
def test_resume_closes_original_resources_in_order_but_never_passes_the_case(
    harness: Harness, scenario: str
) -> None:
    harness.seed(scenario=scenario)
    evidence = harness.case_dir / "original-failed-case.json"
    write_json_atomic(evidence, {"verdict": "FAIL", "attempt": 7})
    original = evidence.read_bytes()
    result = harness.run()
    assert result["verdict"] == "FAIL"
    assert result["attempt"] == 7
    assert result["cleanup_only"] is True
    assert result["cleanup_complete"] is True
    assert result["errors"] == []
    report = result["scenarios"][0]
    assert report["verdict"] == "FAIL" and report["cleanup_complete"] is True
    assert report["errors"] == []
    assert harness.calls == [
        "identity",
        "snapshot",
        "snapshot",
        "fixture-init",
        "safety-init",
        "safety-cleanup",
        "idle",
        "workload-init",
        "workload-delete",
        "fixture-bind",
        "fixture-cleanup",
        "fault-restore",
        "audit",
        "fixture-close",
        "prewarm-init",
        "prewarm-cleanup",
    ]
    assert evidence.read_bytes() == original
    final = reload_journal(harness.journal).record
    assert final.completed and final.prewarm_cleaned
    assert final.scenarios[scenario].state == "CLEANED"
    assert final.scenarios["no-spare"].state == "PENDING"
    assert final.scenarios[scenario].post_started is True


def test_resume_uses_original_attempt_targets_and_expiry_even_after_a_later_attempt(
    harness: Harness,
) -> None:
    harness.seed()
    original = harness.journal.record
    later = execution_record(
        harness.journal.binding(),
        attempt=900,
        maintenance_expires_at=2000000000,
        release_id="later-release",
        scenarios={name: {} for name in case.SCENARIOS},
    )
    harness.journal = ExecutionJournal(
        harness.journal.path, harness.journal.binding, initial=later
    )
    result = harness.run()
    assert result["attempt"] == original.attempt == 7
    assert result["cleanup_complete"] is True
    job, attempt = case.scenario_identity(harness.run_dir, 7, harness.scenario)
    safety = harness.arguments["safety-init"][0]
    assert safety == {
        "run_id": job,
        "attempt_id": attempt,
        "event_id": f"destr008-{harness.scenario}-{job}-event",
        "fault_node": "fault-a",
        "spare_node": "spare-a",
        "spare_uid": original.spare_uid,
        "release_id": original.release_id,
        "directory": harness.directory / "safety",
    }
    workload = harness.arguments["workload-init"][0]
    assert workload["state_path"] == harness.directory / "workload-owner.json"
    assert workload["settings"].manifest == harness.directory / "pinned-workload.yaml"
    assert workload["settings"].site_file == harness.settings.site_file
    assert workload["settings"].job_id == job
    assert workload["settings"].attempt_id == attempt
    assert workload["settings"].restart_budget == 1
    assert harness.arguments["fixture-bind"] == [
        {
            "plan": harness.plan,
            "expires_at": datetime.fromtimestamp(
                original.maintenance_expires_at, timezone.utc
            ),
        }
    ]
    assert (
        harness.arguments["fault-restore"][0]["profile_version"] == "profile-original"
    )
    assert harness.arguments["fault-restore"][0]["incident_id"] == "incident-original"
    assert (
        harness.arguments["prewarm-init"][0]["run_id"]
        == case.scenario_identity(harness.run_dir, 7, "prewarm")[0]
    )
    assert reload_journal(harness.journal).record.maintenance_expires_at == 1700000000


@pytest.mark.parametrize("stages", [(), STAGES[:1], STAGES[:2], STAGES[:3], STAGES])
def test_each_persisted_stage_routes_only_its_owned_cleanup(
    harness: Harness, stages: tuple[Stage, ...]
) -> None:
    harness.seed(stages=stages)
    result = harness.run()
    assert result["cleanup_complete"] is True
    assert result["verdict"] == "FAIL"
    for stage, operation in (
        ("workload_started", "workload-delete"),
        ("safety_started", "safety-cleanup"),
        ("fixture_started", "fixture-cleanup"),
        ("post_started", "idle"),
    ):
        assert (operation in harness.calls) is (stage in stages)
    assert harness.calls.count("fixture-close") == 1


def test_never_started_producer_receipt_is_valid_after_post_intent(
    harness: Harness,
) -> None:
    harness.seed()
    harness.set_receipt(submitted=False)
    result = harness.run()
    assert result["cleanup_complete"] is True
    assert result["verdict"] == "FAIL"
    assert "idle" not in harness.calls
    assert harness.arguments["fault-restore"][0]["incident_id"] == ""
    assert (
        result["scenarios"][0]["independent_safety_cleanup"]["receipt"]["root"] is None
    )


@pytest.mark.parametrize("prewarm_started", [False, True])
def test_pending_scenarios_are_not_started_by_cleanup(
    harness: Harness, prewarm_started: bool
) -> None:
    if prewarm_started:
        harness.seed(start_scenario=False)
    result = harness.run()
    assert result["cleanup_complete"] is True
    assert result["verdict"] == "FAIL"
    assert result["scenarios"] == []
    assert "fixture-init" not in harness.calls
    assert ("prewarm-cleanup" in harness.calls) is prewarm_started
    assert all(
        value.state == "PENDING"
        for value in reload_journal(harness.journal).record.scenarios.values()
    ), "cleanup must leave unstarted scenarios pending"


def test_completed_cleanup_remains_a_failed_case_and_cannot_recreate_a_scenario(
    harness: Harness,
) -> None:
    harness.seed()
    first = harness.run()
    assert first["cleanup_complete"] is True
    harness.calls.clear()
    again = harness.run()
    assert again["cleanup_complete"] is True
    assert again["verdict"] == "FAIL"
    assert again["scenarios"] == []
    assert harness.calls == [
        "identity",
        "snapshot",
        "snapshot",
        "prewarm-init",
        "prewarm-cleanup",
    ]


@pytest.mark.parametrize(
    "change",
    ["release", "cluster", "fault-uid", "spare-uid", "missing-fault-uid", "bad-scope"],
)
def test_replaced_or_unproven_identity_blocks_all_cleanup_mutation(
    harness: Harness, change: str
) -> None:
    harness.seed()
    if change in {"release", "cluster"}:
        harness.scope["release_id" if change == "release" else "cluster_id"] = "other"
    elif change == "bad-scope":
        harness.scope = []
    elif change == "missing-fault-uid":
        harness.nodes["fault-a"] = {}
    else:
        harness.nodes["fault-a" if change == "fault-uid" else "spare-a"]["uid"] = "new"
    original = harness.journal.path.read_bytes()
    result = harness.run()
    assert result["cleanup_complete"] is False
    assert result["verdict"] == "FAIL"
    assert result["errors"]
    assert result["scenarios"] == []
    assert set(harness.calls) <= {"identity", "snapshot"}
    assert harness.journal.path.read_bytes() == original


@pytest.mark.parametrize(
    "scenario", ["no-spare", "topology-mismatch", "reserved-by-other"]
)
def test_interrupted_metadata_fixture_remains_explicitly_unresolved(
    harness: Harness, scenario: str
) -> None:
    harness.seed(
        scenario=scenario,
        stages=("fixture_started", "workload_started", "post_started"),
    )
    result = harness.run()
    report = harness.assert_unfinished(result)
    assert "original mutation reconciliation" in report["errors"][0]
    assert "workload-delete" not in harness.calls
    assert "fault-restore" not in harness.calls
    assert harness.journal.record.prewarm_cleaned is True


@pytest.mark.parametrize(
    "proof",
    [
        None,
        {},
        {"quiescent": False, "retired": True},
        {"quiescent": 1, "retired": True},
        {"quiescent": True, "retired": False},
        {"quiescent": True, "retired": 1},
    ],
)
def test_missing_untyped_or_unretired_safety_proof_never_allows_resource_cleanup(
    harness: Harness, proof: Any
) -> None:
    harness.seed()
    harness.proof = proof
    report = harness.assert_unfinished(harness.run())
    assert report["errors"]
    assert "workload-delete" not in harness.calls
    assert "fixture-cleanup" not in harness.calls
    assert "fault-restore" not in harness.calls


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        [],
        {},
        {"source_complete": False, "producer_revoked": True},
        {"source_complete": 1, "producer_revoked": True},
        {"source_complete": True, "producer_revoked": False},
        {"source_complete": True, "producer_revoked": 1},
    ],
)
def test_lost_ack_or_unrevoked_producer_is_not_finished_source_authority(
    harness: Harness, receipt: Any
) -> None:
    harness.seed()
    harness.proof["receipt"] = receipt
    report = harness.assert_unfinished(harness.run())
    assert "producer completion is unproven" in report["errors"][0]
    assert "idle" not in harness.calls
    assert "workload-delete" not in harness.calls
    assert "fixture-cleanup" not in harness.calls


def test_missing_workload_owner_is_not_reconstructed_after_controller_loss(
    harness: Harness,
) -> None:
    harness.seed()
    path = harness.directory / "workload-owner.json"
    path.unlink()
    report = harness.assert_unfinished(harness.run())
    assert "workload ownership journal is missing" in report["errors"][0]
    assert not path.exists(), "missing workload ownership must not be recreated"
    assert "workload-init" not in harness.calls
    assert "fixture-cleanup" not in harness.calls


@pytest.mark.parametrize("source", ["manifest", "site"])
def test_real_workload_settings_refuse_missing_original_local_source(
    harness: Harness, source: str
) -> None:
    harness.seed()
    path = (
        harness.directory / "pinned-workload.yaml"
        if source == "manifest"
        else harness.settings.site_file
    )
    path.unlink()
    report = harness.assert_unfinished(harness.run())
    assert "does not exist" in report["errors"][0]
    assert "workload-init" not in harness.calls
    assert "workload-delete" not in harness.calls
    assert "fixture-cleanup" not in harness.calls
    assert not path.exists(), "cleanup must not recreate a missing original source"


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        ("safety-init", FileNotFoundError("saved source is missing")),
        (
            "safety-cleanup",
            RegionalFixtureError("original watchdog resource disappeared"),
        ),
        ("idle", RegionalFixtureError("a source command still owns its lease")),
        ("workload-init", FileNotFoundError("original manifest is missing")),
        ("workload-delete", RegionalFixtureError("workload UID was replaced")),
        ("fixture-bind", RegionalFixtureError("original fixture plan differs")),
        (
            "fixture-cleanup",
            RegionalFixtureError("original holder resource is missing"),
        ),
        ("fault-restore", RegionalFixtureError("cleanup owner is not in this family")),
        ("audit", RegionalFixtureError("postflight source unavailable")),
    ],
)
def test_source_loss_replaced_owners_and_incomplete_closure_keep_live_intents(
    harness: Harness, operation: str, error: Exception
) -> None:
    harness.seed()
    harness.failures[operation] = error
    report = harness.assert_unfinished(harness.run())
    assert str(error) in report["errors"][0]
    assert operation in harness.calls
    if operation in {"safety-init", "safety-cleanup", "idle", "workload-init"}:
        assert "workload-delete" not in harness.calls
    if operation not in {"fault-restore", "audit"}:
        assert "fault-restore" not in harness.calls


def test_lost_saved_plan_cannot_be_replaced_by_a_new_fixture_plan(
    harness: Harness,
) -> None:
    harness.seed()
    harness.plan = None
    report = harness.assert_unfinished(harness.run())
    assert "bounded fixture plan is missing" in report["errors"][0]
    assert "fixture-bind" not in harness.calls
    assert "fixture-cleanup" not in harness.calls


@pytest.mark.parametrize(
    "defect",
    ["fixture-residual", "fault-residual", "postflight-errors", "postflight-error"],
)
def test_returned_cleanup_failures_are_not_tombstoned(
    harness: Harness, defect: str
) -> None:
    harness.seed()
    if defect == "fixture-residual":
        harness.fixture_result = {"errors": ["holder still exists"]}
    elif defect == "fault-residual":
        harness.restore_result = {"errors": ["quarantine owner remains"]}
    elif defect == "postflight-errors":
        harness.audit_result = {"postflight_errors": ["spare is not available"]}
    else:
        harness.audit_result = {"postflight_error": "node read failed"}
    report = harness.assert_unfinished(harness.run())
    assert report["errors"]


def test_fixture_close_failure_preserves_scenario_for_a_second_cleanup_attempt(
    harness: Harness,
) -> None:
    harness.seed()
    harness.failures["fixture-close"] = RegionalFixtureError(
        "resource owner still open"
    )
    first = harness.run()
    report = harness.assert_unfinished(first)
    assert any("fixture ownership close" in item for item in report["errors"]), (
        "fixture close failures must remain in the cleanup report"
    )
    harness.failures.pop("fixture-close")
    second = harness.run()
    assert second["cleanup_complete"] is True
    assert second["verdict"] == "FAIL"
    assert len(second["scenarios"]) == 1
    assert harness.calls.count("fixture-close") == 2
    assert harness.calls.count("fixture-cleanup") == 2


def test_cleanup_error_and_close_error_are_both_retained(harness: Harness) -> None:
    harness.seed()
    harness.failures["workload-delete"] = RegionalFixtureError("foreign owner")
    harness.failures["fixture-close"] = OSError("closure failed")
    report = harness.assert_unfinished(harness.run())
    assert len(report["errors"]) == 2
    assert "foreign owner" in report["errors"][0]
    assert "fixture ownership close: OSError" in report["errors"][1]


def test_missing_prewarm_journal_is_unresolved_even_after_scenario_cleanup(
    harness: Harness,
) -> None:
    harness.seed()
    path = harness.case_dir / "prewarm-owner.json"
    path.unlink()
    result = harness.run()
    assert result["verdict"] == "FAIL" and result["cleanup_complete"] is False
    assert "prewarm ownership journal is missing" in result["errors"][0]
    assert result["scenarios"][0]["cleanup_complete"] is True
    assert harness.journal.record.completed is False
    assert harness.journal.record.prewarm_cleaned is False
    assert "prewarm-init" not in harness.calls
    assert not path.exists(), "missing prewarm ownership must not be recreated"


@pytest.mark.parametrize(
    "residuals", [{"pods": ["owned-pod"]}, {"jobs": ["owned-job"]}, None]
)
def test_prewarm_failure_retries_only_unfinished_cleanup_without_a_new_scenario(
    harness: Harness, residuals: Any
) -> None:
    harness.seed()
    harness.prewarm_residuals = residuals
    first = harness.run()
    assert first["verdict"] == "FAIL" and first["cleanup_complete"] is False
    assert first["errors"]
    assert harness.journal.record.scenarios[harness.scenario].state == "CLEANED"
    assert not harness.journal.record.prewarm_cleaned, (
        "prewarm residuals cannot be marked clean"
    )
    assert not harness.journal.record.completed, (
        "unresolved prewarm cleanup must keep execution open"
    )
    harness.prewarm_residuals = {"pods": [], "jobs": []}
    harness.calls.clear()
    again = harness.run()
    assert again["cleanup_complete"] is True and again["verdict"] == "FAIL"
    assert again["scenarios"] == []
    assert "fixture-init" not in harness.calls
    assert "fault-restore" not in harness.calls
    assert harness.calls.count("prewarm-cleanup") == 1


@pytest.mark.parametrize("operation", ["prewarm-init", "prewarm-cleanup"])
def test_prewarm_source_or_owner_loss_cannot_complete_the_execution(
    harness: Harness, operation: str
) -> None:
    harness.seed(start_scenario=False)
    harness.failures[operation] = RegionalFixtureError(
        "original prewarm UID unavailable"
    )
    result = harness.run()
    assert result["verdict"] == "FAIL" and result["cleanup_complete"] is False
    assert "original prewarm UID unavailable" in result["errors"][0]
    assert not harness.journal.record.prewarm_cleaned, (
        "lost prewarm authority cannot prove cleanup"
    )
    assert not harness.journal.record.completed, (
        "lost prewarm authority must keep execution open"
    )


def test_failed_final_journal_write_is_not_successful_resource_retirement(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.seed()
    original_write = write_json_atomic

    def cannot_record(path: Path, document: dict[str, Any]) -> None:
        assert path == harness.journal.path
        if document["scenarios"][harness.scenario]["state"] == "CLEANED":
            assert harness.calls[-1] == "fixture-close"
            harness.note("cleanup-checkpoint")
            raise OSError("journal storage unavailable")
        original_write(path, document)

    monkeypatch.setattr(journal_module, "write_json_atomic", cannot_record)
    report = harness.assert_unfinished(harness.run())
    assert report["errors"] == ["cleanup checkpoint: OSError"]
    assert harness.calls.count("cleanup-checkpoint") == 1
    assert "prewarm-cleanup" in harness.calls
    monkeypatch.setattr(journal_module, "write_json_atomic", original_write)
    again = harness.run()
    assert again["cleanup_complete"] is True
    assert again["verdict"] == "FAIL"
    assert len(again["scenarios"]) == 1
    assert harness.calls.count("fixture-close") == 2
    assert reload_journal(harness.journal).record.completed is True
