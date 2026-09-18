"""Config compensation requires release evidence, including the commit boundary."""

from __future__ import annotations

import copy
import json
import subprocess
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cli, release_child, release_state
from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigError,
    admin_config_desired_path,
    admin_config_history_path,
    canonical_sha256,
    load_desired_admin_config,
    load_pending_admin_config_apply,
    persist_desired_admin_config,
)
from gpu_fault.admin.config_file import (
    initialize_desired_admin_config,
    write_admin_config_file,
)
from gpu_fault.admin.operation_lock import SITE_OPERATION_LOCK_FD_ENV
from gpu_fault.admin.site import SiteConfigError, load_site
from gpu_fault_release import regional_release_transaction
from gpu_fault_release.regional_release_diff import (
    ReleaseChangeKind,
    ReleaseDiff,
    build_execution_plan,
)
from scripts import release_failure_recovery
from tests.admin.test_admin_site import site_file


def release_document(config: AdminConfig, **updates: Any) -> dict[str, Any]:
    return {
        "release_id": "release-a",
        "phase": "complete",
        "transaction_committed": True,
        "release_lifecycle": "COMMITTED",
        "admin_config": config.as_dict(),
        "admin_config_sha256": config.sha256(),
        "admin_config_role_sha256": config.role_sha256(),
        **updates,
    }


class ConfigScenario:
    def __init__(self, path: Path) -> None:
        self.site = load_site(path)
        self.root = path.parent
        self.before = initialize_desired_admin_config(self.root)
        self.desired = self.before.patched(
            {
                "aurora": {"minAcu": 16, "maxAcu": 64},
                "workflow": {"dispatcherWorkers": 9},
            }
        )
        write_admin_config_file(
            self.root / "admin-config.yaml", self.desired, overwrite=True
        )
        self.live = release_document(self.before)
        self.capacity = self.before.aurora
        self.events: list[str] = []
        self.driver_action: Callable[[], None] = lambda: None
        self.driver_code = 7
        self.driver_error: Exception | None = None
        self.read_error: Exception | None = None
        self.request_error: Exception | None = None
        self.await_error: Exception | None = None
        self.rollback_error: Exception | None = None
        self.rollback_calls = 0
        self.driver_calls = 0
        self.reads: list[dict[str, Any]] = []
        self.arguments = cli.parser().parse_args(
            ["config", "--state-dir", str(self.root), "--reference", "CHG-REVIEW"]
        )

    def read(self, arguments: list[str], **_options: Any):
        self.events.append("read")
        if self.read_error is not None:
            raise self.read_error
        self.reads.append(copy.deepcopy(self.live))
        return subprocess.CompletedProcess(
            arguments,
            0,
            json.dumps({"data": {"state.json": json.dumps(self.live)}}),
            "",
        )

    def driver(self, arguments: list[str], **options: Any):
        self.events.append("release")
        self.driver_calls += 1
        descriptor = int(options["env"][SITE_OPERATION_LOCK_FD_ENV])
        assert descriptor in options["pass_fds"], (
            "config release did not inherit its exclusive site lock"
        )
        self.driver_action()
        if self.driver_error is not None:
            raise self.driver_error
        return subprocess.CompletedProcess(arguments, self.driver_code, "", "")

    def request(self, **options: Any) -> dict[str, Any]:
        self.events.append("request")
        self.capacity = options["desired"]
        if self.request_error is not None:
            raise self.request_error
        return {"modified": True, "scale_up": True}

    def wait(self, **_options: Any) -> dict[str, Any]:
        self.events.append("await")
        if self.await_error is not None:
            raise self.await_error
        return {"min_acu": self.capacity.min_acu}

    def rollback(self, **options: Any) -> dict[str, Any]:
        self.events.append("rollback")
        self.rollback_calls += 1
        assert options["expected"] == self.desired.aurora
        assert options["desired"] == self.before.aurora
        if self.rollback_error is not None:
            raise self.rollback_error
        self.capacity = options["desired"]
        return {"modified": True}

    def candidate(self, **updates: Any) -> dict[str, Any]:
        previous = {"release_id": "release-a", "admin_config": self.before.as_dict()}
        diff = ReleaseDiff(
            ReleaseChangeKind.CONTROL_PLANE_ONLY,
            frozenset(
                "admin_config_" + role
                for role, digest in self.desired.role_sha256().items()
                if digest != self.before.role_sha256()[role]
            ),
        )
        return release_document(
            self.desired,
            **{
                "phase": "cpu-staged",
                "transaction_committed": False,
                "release_lifecycle": "PREPARING",
                "previous": previous,
                "previous_snapshot_sha256": canonical_sha256(previous),
                "completed_phases": ["uploaded"],
                "completed_cluster_ids": [],
                "release_diff": diff.as_dict(),
                "execution_plan": build_execution_plan(diff).as_dict(),
                "component_progress": {
                    "schema_version": 1,
                    "global": {"cpu-stage": {"status": "STARTED"}},
                    "clusters": {},
                },
                **updates,
            },
        )

    def completed_rollback(self, **updates: Any) -> dict[str, Any]:
        return self.candidate(
            **{
                "phase": "rolled-back",
                "release_lifecycle": "ROLLED_BACK",
                "rollback_cleanup_completed": True,
                "rollback_result": {"status": "PASSED"},
                **updates,
            }
        )

    def synced_rollback(self, **updates: Any) -> dict[str, Any]:
        return release_document(
            replace(self.before, aurora=self.desired.aurora),
            **{
                "release_diff": {"kind": "NOOP", "changed": []},
                "previous": None,
                **updates,
            },
        )

    def write_driver_result(self, **updates: Any) -> dict[str, Any]:
        timestamp = datetime.now(UTC).isoformat()
        record = {
            "schema_version": 1,
            "release_id": "release-a",
            "site_file": str(self.site.source),
            "phase": "FAILED",
            "failed_at": timestamp,
            "rollback": {
                "status": "PASSED",
                "management_state_synced": True,
                "started_at": timestamp,
                "completed_at": timestamp,
            },
            **updates,
        }
        path = self.root / "release-deploy/release-a/state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        return record

    def run(self) -> int:
        return cli.run(self.arguments)

    def failure(self) -> dict[str, Any]:
        pending = load_pending_admin_config_apply(self.root)
        assert pending is not None, "failed apply lost its pending target"
        history = admin_config_history_path(self.root, str(pending["history"]))
        return json.loads((history / "result.json").read_text())

    def assert_retained(self, reason: str) -> None:
        assert self.rollback_calls == 0, "unproven release caused an Aurora rollback"
        assert self.capacity == self.desired.aurora
        assert load_desired_admin_config(self.root) == self.desired
        result = self.failure()
        assert result["details"]["release_recovery"]["restore_before"] is False
        assert result["details"]["release_recovery"]["reason"] == reason
        assert "pending configuration target retained" in result["error"]


@pytest.fixture
def scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ConfigScenario:
    value = ConfigScenario(site_file(tmp_path))
    monkeypatch.setattr(release_state, "run_command", value.read)
    monkeypatch.setattr(release_child, "run_driver", value.driver)
    monkeypatch.setattr(cli, "verify_prebuilt_release", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "request_aurora_capacity", value.request)
    monkeypatch.setattr(cli, "await_aurora_capacity", value.wait)
    monkeypatch.setattr(cli, "reconcile_aurora_capacity", value.rollback)
    return value


@pytest.mark.parametrize("failure_kind", ["exit", "exception", "aurora-request"])
def test_unchanged_committed_before_state_authorizes_compensation(
    scenario: ConfigScenario, failure_kind: str
) -> None:
    if failure_kind == "exception":
        scenario.driver_error = RuntimeError("example release failure")
    elif failure_kind == "aurora-request":
        scenario.request_error = RuntimeError("example request failed after its write")
    if failure_kind == "exit":
        assert scenario.run() == 7
    else:
        with pytest.raises(RuntimeError, match="example"):
            scenario.run()
    assert scenario.rollback_calls == 1
    assert scenario.capacity == scenario.before.aurora
    assert load_desired_admin_config(scenario.root) == scenario.before
    assert len(scenario.reads) == 2
    assert scenario.failure()["details"]["release_recovery"]["reason"] == (
        "UNCHANGED_COMMITTED_BASELINE"
    )
    if failure_kind == "aurora-request":
        assert scenario.driver_calls == 0


def test_real_commit_cleanup_failure_preserves_committed_target(
    scenario: ConfigScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_cleanup(*_args: Any) -> None:
        raise RuntimeError("post-commit cleanup failed")

    monkeypatch.setattr(
        regional_release_transaction,
        "delete_stale_release_secret_backups",
        fail_cleanup,
    )

    def child() -> None:
        scenario.live = scenario.candidate(phase="complete")
        release = SimpleNamespace(
            state=scenario.live,
            _save_state=lambda phase, **values: scenario.live.update(
                phase=phase, **values
            ),
        )
        with pytest.raises(RuntimeError, match="post-commit cleanup"):
            regional_release_transaction.commit_release(release)
        result = release_failure_recovery.recover_release_failure(
            SimpleNamespace(),
            site_file=scenario.site.source,
            root=scenario.site.repository_root,
            environment={},
            failure_error="cleanup failed",
            failed_at=datetime.now(UTC).isoformat(),
            deployment_succeeded=True,
            commit_started=True,
            automatic_rollback=True,
            run_release_mode=lambda *_args, **_kwargs: pytest.fail(
                "committed rollback"
            ),
            read_live_state=lambda _path: copy.deepcopy(scenario.live),
            update_phase=lambda *_args, **_kwargs: None,
        )
        assert result is not None, "commit failure produced no recovery result"
        assert result["status"] == "SKIPPED_COMMITTED"
        scenario.write_driver_result(rollback=result)

    scenario.driver_action = child
    with pytest.raises(AdminConfigError, match="TARGET_COMMITTED"):
        scenario.run()
    assert scenario.live["transaction_committed"] is True
    assert scenario.live["commit_cleanup_completed"] is False
    scenario.assert_retained("TARGET_COMMITTED")


@pytest.mark.parametrize(
    "state",
    [
        {"phase": "complete"},
        {"phase": "cpu-staged"},
        {"phase": "failed"},
        {"phase": "rollback-failed", "rollback_result": {"status": "FAILED"}},
        {"phase": "rollback-cpu-restoring"},
        {"phase": "unknown"},
        {"phase": []},
        {"release_id": "different-release"},
    ],
)
def test_partial_unknown_and_foreign_releases_cannot_authorize_compensation(
    scenario: ConfigScenario, state: dict[str, Any]
) -> None:
    scenario.driver_action = lambda: setattr(
        scenario, "live", scenario.candidate(**state)
    )
    with pytest.raises(AdminConfigError, match="pending configuration target retained"):
        scenario.run()
    scenario.assert_retained(
        "RELEASE_IDENTITY_CHANGED"
        if "release_id" in state
        else "RELEASE_OUTCOME_UNPROVEN"
    )


def test_policy_disabled_rollback_is_not_restoration(scenario: ConfigScenario) -> None:
    def child() -> None:
        scenario.live = scenario.candidate(phase="complete")
        result = release_failure_recovery.recover_release_failure(
            SimpleNamespace(),
            site_file=scenario.site.source,
            root=scenario.site.repository_root,
            environment={},
            failure_error="verification failed",
            failed_at=datetime.now(UTC).isoformat(),
            deployment_succeeded=True,
            commit_started=False,
            automatic_rollback=False,
            run_release_mode=lambda *_args, **_kwargs: pytest.fail("policy rollback"),
            read_live_state=lambda _path: copy.deepcopy(scenario.live),
            update_phase=lambda *_args, **_kwargs: None,
        )
        assert result is not None, "policy refusal produced no result"
        assert result["status"] == "SKIPPED_POLICY"
        scenario.write_driver_result(rollback=result)

    scenario.driver_action = child
    with pytest.raises(AdminConfigError, match="RELEASE_OUTCOME_UNPROVEN"):
        scenario.run()
    scenario.assert_retained("RELEASE_OUTCOME_UNPROVEN")


@pytest.mark.parametrize("ambiguous_commit", [False, True])
def test_real_failed_or_ambiguous_driver_recovery_cannot_authorize_compensation(
    scenario: ConfigScenario, ambiguous_commit: bool
) -> None:
    def child() -> None:
        scenario.live = scenario.candidate(phase="complete")

        def read(_path: Path) -> dict[str, Any]:
            if ambiguous_commit:
                raise SiteConfigError("commit result cannot be read")
            return copy.deepcopy(scenario.live)

        def rollback(*_args: Any, **_options: Any) -> None:
            assert not ambiguous_commit, "ambiguous commit initiated a rollback"
            scenario.live["phase"] = "rollback-failed"
            raise RuntimeError("example release rollback failed")

        result = release_failure_recovery.recover_release_failure(
            SimpleNamespace(),
            site_file=scenario.site.source,
            root=scenario.site.repository_root,
            environment={},
            failure_error="example child failure",
            failed_at=datetime.now(UTC).isoformat(),
            deployment_succeeded=True,
            commit_started=ambiguous_commit,
            automatic_rollback=True,
            run_release_mode=rollback,
            read_live_state=read,
            update_phase=lambda *_args, **_kwargs: None,
        )
        assert result is not None, "failed recovery produced no result"
        assert result["status"] == (
            "SKIPPED_AMBIGUOUS_COMMIT" if ambiguous_commit else "FAILED"
        )
        scenario.write_driver_result(rollback=result)

    scenario.driver_action = child
    with pytest.raises(AdminConfigError, match="RELEASE_OUTCOME_UNPROVEN"):
        scenario.run()
    scenario.assert_retained("RELEASE_OUTCOME_UNPROVEN")


def test_exception_after_partial_release_retains_target_and_original_cause(
    scenario: ConfigScenario,
) -> None:
    scenario.driver_action = lambda: setattr(scenario, "live", scenario.candidate())
    scenario.driver_error = RuntimeError("example interrupted driver")
    with pytest.raises(AdminConfigError, match="example interrupted driver") as caught:
        scenario.run()
    assert caught.value.__cause__ is scenario.driver_error
    scenario.assert_retained("RELEASE_OUTCOME_UNPROVEN")


@pytest.mark.parametrize(
    "field", ["admin_config", "admin_config_sha256", "admin_config_role_sha256"]
)
def test_unchanged_legacy_state_without_config_proof_cannot_compensate(
    scenario: ConfigScenario, field: str
) -> None:
    scenario.live.pop(field)
    with pytest.raises(AdminConfigError, match="RELEASE_OUTCOME_UNPROVEN"):
        scenario.run()
    scenario.assert_retained("RELEASE_OUTCOME_UNPROVEN")


@pytest.mark.parametrize(
    "state",
    [
        {"release_lifecycle": None},
        {"release_lifecycle": "UNKNOWN"},
        {"release_lifecycle": "FAILED"},
        {"rollback_result": None},
        {"rollback_result": {"status": "FAILED"}},
        {"rollback_failure": "unconfirmed restoration"},
    ],
)
def test_contradictory_committed_baseline_cannot_authorize_compensation(
    scenario: ConfigScenario, state: dict[str, Any]
) -> None:
    scenario.live.update(state)
    with pytest.raises(AdminConfigError, match="RELEASE_OUTCOME_UNPROVEN"):
        scenario.run()
    scenario.assert_retained("RELEASE_OUTCOME_UNPROVEN")


def test_cpu_only_partial_release_preserves_desired_without_aurora_calls(
    scenario: ConfigScenario,
) -> None:
    scenario.desired = scenario.before.patched({"workflow": {"dispatcherWorkers": 9}})
    write_admin_config_file(
        scenario.root / "admin-config.yaml", scenario.desired, overwrite=True
    )
    scenario.driver_action = lambda: setattr(scenario, "live", scenario.candidate())
    with pytest.raises(AdminConfigError, match="RELEASE_OUTCOME_UNPROVEN"):
        scenario.run()
    assert scenario.events == ["read", "release", "read"]
    scenario.assert_retained("RELEASE_OUTCOME_UNPROVEN")


@pytest.mark.parametrize(
    "error", [SiteConfigError("unreadable state"), ValueError("bad JSON")]
)
def test_unreadable_postfailure_state_preserves_pending_target(
    scenario: ConfigScenario, error: Exception
) -> None:
    scenario.driver_action = lambda: setattr(scenario, "read_error", error)
    with pytest.raises(AdminConfigError, match="LIVE_STATE_UNREADABLE"):
        scenario.run()
    scenario.assert_retained("LIVE_STATE_UNREADABLE")


@pytest.mark.parametrize(
    "phase", ["preflight", "uploaded", "candidate-preflight-ready", "failed"]
)
def test_explicit_unstarted_config_progress_authorizes_compensation(
    scenario: ConfigScenario, phase: str
) -> None:
    scenario.driver_action = lambda: setattr(
        scenario,
        "live",
        scenario.candidate(
            phase=phase,
            component_progress={"schema_version": 1, "global": {}, "clusters": {}},
        ),
    )
    assert scenario.run() == 7
    assert scenario.rollback_calls == 1
    assert load_desired_admin_config(scenario.root) == scenario.before
    assert scenario.failure()["details"]["release_recovery"]["reason"] == (
        "CONFIG_ROLLOUT_NOT_STARTED"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"completed_phases": ["uploaded", "cpu-staged"]},
        {"completed_phases": None},
        {"completed_phases": [{}]},
        {"completed_cluster_ids": ["unexpected"]},
        {"component_progress": None},
        {"component_progress": {"schema_version": True, "global": {}, "clusters": {}}},
        {"component_progress": {"schema_version": 2, "global": {}, "clusters": {}}},
        {
            "component_progress": {
                "schema_version": 1,
                "global": {"cpu-stage": {}},
                "clusters": {},
            }
        },
        {"execution_plan": {"nodes": ["unknown"]}},
        {"release_diff": {"kind": "FULL", "changed": ["admin_config_worker"]}},
        {"release_diff": None},
        {"release_diff": {"kind": "CONTROL_PLANE_ONLY", "changed": []}},
        {"release_diff": {"kind": "CONTROL_PLANE_ONLY", "changed": [None]}},
        {
            "release_diff": {
                "kind": "CONTROL_PLANE_ONLY",
                "changed": ["admin_config_ingress"],
            }
        },
        {"release_diff": {"kind": "CONTROL_PLANE_ONLY", "changed": ["cpu_manifests"]}},
        {"previous_snapshot_sha256": "0" * 64},
        {"previous": None},
        {"rollback_result": None},
        {"rollback_failure": "uncertain"},
        {"transaction_committed": None},
    ],
)
def test_incomplete_or_started_progress_is_not_negative_mutation_evidence(
    scenario: ConfigScenario, mutation: dict[str, Any]
) -> None:
    def child() -> None:
        scenario.live = scenario.candidate(
            phase="uploaded",
            component_progress={"schema_version": 1, "global": {}, "clusters": {}},
        )
        scenario.live.update(mutation)

    scenario.driver_action = child
    with pytest.raises(AdminConfigError, match="pending configuration target retained"):
        scenario.run()
    scenario.assert_retained(
        "LIVE_STATE_UNREADABLE"
        if "previous_snapshot_sha256" in mutation
        else "RELEASE_OUTCOME_UNPROVEN"
    )


def test_verified_completed_rollback_authorizes_compensation(
    scenario: ConfigScenario,
) -> None:
    scenario.driver_action = lambda: setattr(
        scenario, "live", scenario.completed_rollback()
    )
    assert scenario.run() == 7
    assert scenario.rollback_calls == 1
    assert load_desired_admin_config(scenario.root) == scenario.before
    assert (
        scenario.failure()["details"]["release_recovery"]["reason"]
        == "VERIFIED_ROLLBACK"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"rollback_cleanup_completed": False},
        {"rollback_cleanup_completed": "true"},
        {"rollback_result": {"status": "FAILED"}},
        {"rollback_result": {}},
        {"release_lifecycle": "PREPARING"},
        {"transaction_committed": True},
        {"previous_snapshot_sha256": "0" * 64},
        {"previous": {"release_id": "different-release"}},
        {"admin_config_sha256": "0" * 64},
        {"admin_config_role_sha256": {}},
        {"admin_config": {}},
    ],
)
def test_incomplete_or_mismatched_rollback_retains_target(
    scenario: ConfigScenario, mutation: dict[str, Any]
) -> None:
    def child() -> None:
        scenario.live = scenario.completed_rollback(**mutation)
        if isinstance(mutation.get("previous"), dict):
            scenario.live["previous_snapshot_sha256"] = canonical_sha256(
                scenario.live["previous"]
            )

    scenario.driver_action = child
    with pytest.raises(AdminConfigError, match="pending configuration target retained"):
        scenario.run()
    scenario.assert_retained(
        "LIVE_STATE_UNREADABLE"
        if "previous_snapshot_sha256" in mutation
        else "RELEASE_OUTCOME_UNPROVEN"
    )


def test_previous_snapshot_must_match_original_config(scenario: ConfigScenario) -> None:
    def child() -> None:
        scenario.live = scenario.completed_rollback()
        scenario.live["previous"]["admin_config"] = scenario.desired.as_dict()
        scenario.live["previous_snapshot_sha256"] = canonical_sha256(
            scenario.live["previous"]
        )

    scenario.driver_action = child
    with pytest.raises(AdminConfigError, match="RELEASE_OUTCOME_UNPROVEN"):
        scenario.run()
    scenario.assert_retained("RELEASE_OUTCOME_UNPROVEN")


def test_fresh_management_sync_receipt_proves_the_rollback(
    scenario: ConfigScenario,
) -> None:
    def child() -> None:
        scenario.live = scenario.synced_rollback()
        scenario.write_driver_result()

    scenario.driver_action = child
    assert scenario.run() == 7
    assert scenario.rollback_calls == 1
    assert load_desired_admin_config(scenario.root) == scenario.before
    assert scenario.failure()["details"]["release_recovery"]["reason"] == (
        "VERIFIED_SYNCED_ROLLBACK"
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "invalid-json",
        "not-object",
        "wrong-schema",
        "boolean-schema",
        "wrong-release",
        "wrong-site",
        "not-failed",
        "failed-rollback",
        "no-rollback",
        "not-synced",
        "stale",
        "future",
        "unordered",
        "missing-time",
        "naive-time",
        "invalid-time",
        "foreign-config",
        "wrong-role-digest",
        "not-noop",
        "retained-previous",
        "rollback-failure",
        "not-committed",
    ],
)
def test_unbound_or_stale_driver_receipts_cannot_authorize_compensation(
    scenario: ConfigScenario, change: str
) -> None:
    def child() -> None:
        scenario.live = scenario.synced_rollback()
        record = scenario.write_driver_result()
        if change == "wrong-schema":
            record["schema_version"] = 2
        elif change == "boolean-schema":
            record["schema_version"] = True
        elif change == "wrong-release":
            record["release_id"] = "different-release"
        elif change == "wrong-site":
            record["site_file"] = "/different/site.yaml"
        elif change == "not-failed":
            record["phase"] = "COMPLETED"
        elif change == "failed-rollback":
            record["rollback"]["status"] = "FAILED"
        elif change == "no-rollback":
            record["rollback"] = None
        elif change == "not-synced":
            record["rollback"]["management_state_synced"] = False
        elif change == "stale":
            record["failed_at"] = "2020-01-01T00:00:00+00:00"
        elif change == "future":
            record["rollback"]["completed_at"] = (
                datetime.now(UTC) + timedelta(days=1)
            ).isoformat()
        elif change == "unordered":
            record["rollback"]["completed_at"] = "2020-01-01T00:00:00+00:00"
        elif change == "missing-time":
            record["rollback"].pop("started_at")
        elif change == "naive-time":
            record["rollback"]["started_at"] = "2026-09-14T00:00:00"
        elif change == "invalid-time":
            record["rollback"]["started_at"] = "invalid"
        elif change == "foreign-config":
            scenario.live["admin_config"] = {}
        elif change == "wrong-role-digest":
            scenario.live["admin_config_role_sha256"] = {}
        elif change == "not-noop":
            scenario.live["release_diff"] = {"kind": "FULL", "changed": []}
        elif change == "retained-previous":
            scenario.live["previous"] = {}
        elif change == "rollback-failure":
            scenario.live["rollback_failure"] = "unproven"
        elif change == "not-committed":
            scenario.live["transaction_committed"] = False
        path = scenario.root / "release-deploy/release-a/state.json"
        path.write_text(json.dumps(record))
        if change == "missing":
            path.unlink()
        elif change == "invalid-json":
            path.write_text("invalid")
        elif change == "not-object":
            path.write_text("[]")

    scenario.driver_action = child
    with pytest.raises(AdminConfigError, match="RELEASE_OUTCOME_UNPROVEN"):
        scenario.run()
    scenario.assert_retained("RELEASE_OUTCOME_UNPROVEN")


def test_failed_management_sync_does_not_leave_a_hybrid_desired_config(
    scenario: ConfigScenario,
) -> None:
    def child() -> None:
        scenario.live = scenario.completed_rollback(rollback_cleanup_completed=False)
        persist_desired_admin_config(
            scenario.root,
            config=replace(scenario.before, aurora=scenario.desired.aurora),
            source="partial-rollback-alignment",
        )

    scenario.driver_action = child
    with pytest.raises(AdminConfigError, match="pending configuration target retained"):
        scenario.run()
    scenario.assert_retained("RELEASE_OUTCOME_UNPROVEN")
    persisted = json.loads(admin_config_desired_path(scenario.root).read_text())
    assert persisted["reference"] == "CHG-REVIEW"
    assert persisted["approver_identity"], (
        "retained target lost its administrator identity"
    )


def test_aurora_rollback_failure_does_not_claim_restored_desired(
    scenario: ConfigScenario,
) -> None:
    scenario.rollback_error = RuntimeError("example rollback request failed")
    with pytest.raises(AdminConfigError, match="Aurora rollback failed"):
        scenario.run()
    assert scenario.rollback_calls == 1
    assert scenario.capacity == scenario.desired.aurora
    assert load_desired_admin_config(scenario.root) == scenario.desired
    assert "Aurora rollback failed" in scenario.failure()["error"]


def test_post_success_aurora_wait_failure_retries_without_reverting_target(
    scenario: ConfigScenario,
) -> None:
    scenario.driver_code = 0
    scenario.driver_action = lambda: setattr(
        scenario, "live", release_document(scenario.desired)
    )
    scenario.await_error = AdminConfigError("capacity did not converge")
    with pytest.raises(AdminConfigError, match="finish waiting for Aurora"):
        scenario.run()
    scenario.assert_retained("RELEASE_SUCCEEDED")
    first = scenario.failure()
    assert "gpu-fault-admin deploy" not in first["error"], (
        "a successful release's ACU wait incorrectly requested another deploy"
    )
    scenario.await_error = None
    assert scenario.run() == 0
    assert load_desired_admin_config(scenario.root) == scenario.desired
    assert load_pending_admin_config_apply(scenario.root) is None
    assert scenario.rollback_calls == 0
    assert scenario.driver_calls == 2
    assert (
        admin_config_history_path(scenario.root, first["history"]) / "result.json"
    ).is_file(), "retry removed the original failed-wait evidence"


def test_retry_request_failure_on_committed_target_cannot_restore_old_capacity(
    scenario: ConfigScenario,
) -> None:
    scenario.driver_code = 0
    scenario.driver_action = lambda: setattr(
        scenario, "live", release_document(scenario.desired)
    )
    scenario.await_error = AdminConfigError("capacity did not converge")
    with pytest.raises(AdminConfigError, match="finish waiting for Aurora"):
        scenario.run()
    scenario.await_error = None
    scenario.request_error = RuntimeError("retry capacity observation failed")
    with pytest.raises(AdminConfigError, match="TARGET_COMMITTED"):
        scenario.run()
    assert scenario.driver_calls == 1
    scenario.assert_retained("TARGET_COMMITTED")
