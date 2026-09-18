"""Offline repair transactions must prove identity, rollback data and ordering."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_admin_commands as COMMANDS
from gpu_fault_release import regional_aurora_credentials as CREDENTIALS
from gpu_fault_release import regional_release_aurora_refresh as REFRESH
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_release_config import ReleaseError, canonical_sha256
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    build_execution_plan,
    diff_from_changed,
)
from gpu_fault_release.regional_release_prerequisite_repair import REPAIR_KEY
from tests.regional._prerequisite_repair_support import repair_release
from tests.regional._release_orchestrator_support import config_file
from tests.regional.test_release_aurora_refresh_preflight import RecordingRunner
from tests.regional.test_release_aurora_refresh_transaction import (
    NAMESPACE,
    ObjectRunner,
    objects,
    release,
)


@pytest.fixture
def bootstrap_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    instance = ROLLOUT.RegionalRelease(
        ROLLOUT.ReleaseConfig.load(config_file(tmp_path)), ROLLOUT.Runner(dry_run=True)
    )
    events: list[tuple[str, str]] = []

    def save(phase: str, **updates: Any) -> None:
        instance.state.update(phase=phase, **updates)
        events.append(("state", phase))

    for name in (
        "_ensure_contexts",
        "_apply_rds_ca_bundle",
        "_require_cpu_secrets",
        "_initialize_registry",
        "_prepare_nlb",
        "_upload_release",
        "_prepare_bootstrap_workflows",
        "_ensure_schema",
        "_apply_cpu",
        "_wait_nlb",
        "_apply_control_plane_observability",
    ):
        monkeypatch.setattr(instance, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(instance, "_save_state", save)
    monkeypatch.setattr(
        instance,
        "_cancel_active_installer_jobs",
        lambda target: events.append(("cancel-jobs", target.cluster_id)),
    )
    monkeypatch.setattr(
        instance,
        "_scale_if_present",
        lambda _kubectl, name, _replicas, **_kwargs: events.append(("scale", name)),
    )
    monkeypatch.setattr(ROLLOUT, "ensure_runtime_profile", lambda _release: None)

    def fail_gpu(*_args: Any) -> None:
        events.append(("gpu", "failed"))
        raise ReleaseError("GPU bootstrap failed")

    monkeypatch.setattr(ROLLOUT, "bootstrap_gpu_clusters", fail_gpu)
    return SimpleNamespace(release=instance, events=events)


def test_consecutive_bootstrap_failures_both_cancel_jobs_and_scale_workloads(
    bootstrap_model: SimpleNamespace,
) -> None:
    instance = bootstrap_model.release
    for _attempt in range(2):
        with pytest.raises(ReleaseError, match="GPU bootstrap failed"):
            instance.bootstrap()
        assert instance.state["phase"] == "bootstrap-cleaned", (
            "a failed attempt did not finish cleanup"
        )
    events = bootstrap_model.events
    assert events.count(("cancel-jobs", "gpu-a")) == 2, (
        "the second deployment reused the first attempt's cancelled-Jobs checkpoint"
    )
    names = {name for event, name in events if event == "scale"}
    assert names, "cleanup never scaled a workload"
    assert all(events.count(("scale", name)) == 2 for name in names), (
        "the second failure skipped a workload that the new attempt could restart"
    )


def test_cleanup_resume_keeps_progress_then_new_attempt_resets_it(
    bootstrap_model: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = bootstrap_model.release
    events = bootstrap_model.events
    fail_once = True

    def scale(_kubectl: Any, name: str, _replicas: int, **_kwargs: Any) -> None:
        nonlocal fail_once
        events.append(("scale", name))
        if fail_once and name == "gpu-fault-cluster-executor":
            fail_once = False
            raise ReleaseError("cleanup interrupted")

    monkeypatch.setattr(instance, "_scale_if_present", scale)
    with pytest.raises(ReleaseError, match="cleanup interrupted"):
        instance.bootstrap()
    assert instance.state["bootstrap_cleanup_completed_steps"] == [
        "installer-jobs-cancelled"
    ], "cleanup did not persist the completed step before its interruption"
    with pytest.raises(ReleaseError, match="GPU bootstrap failed"):
        instance.bootstrap()
    assert events.count(("cancel-jobs", "gpu-a")) == 2, (
        "cleanup resume must skip its completed step, but the new attempt must repeat it"
    )
    assert instance.state["phase"] == "bootstrap-cleaned", (
        "the retried attempt was not fully cleaned"
    )


def test_lost_supervision_does_not_enter_bootstrap_compensation(
    bootstrap_model: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    class QuiescenceUnproved(BaseException):
        pass

    def lost() -> None:
        raise QuiescenceUnproved()

    monkeypatch.setattr(bootstrap_model.release, "_prepare_bootstrap_workflows", lost)
    with pytest.raises(QuiescenceUnproved):
        bootstrap_model.release.bootstrap()
    assert not any(
        event in {"scale", "cancel-jobs"} for event, _name in bootstrap_model.events
    ), "a fatal supervision exception must not trigger an unproved cleanup"


class PreflightRunner(ObjectRunner):
    def __init__(self, *, installed: bool, missing_refresh: bool) -> None:
        super().__init__([] if missing_refresh else objects())
        self.installed = installed
        self.resource_error = ""
        self.amp_error = ""
        self.foreign_anchor = False

    def run(self, args: list[str], **kwargs: Any) -> str:
        if args[:2] == ["aws", "amp"]:
            if args[2] == "describe-workspace":
                return json.dumps(
                    {
                        "workspace": {
                            "workspaceId": "ws-test",
                            "status": {"statusCode": "ACTIVE"},
                        }
                    }
                )
            raise ReleaseError("AMP definition is missing")
        if args[:2] == ["aws", "sns"]:
            return json.dumps({"Subscriptions": []})
        if "get" in args:
            resource = args[args.index("get") + 1]
            if resource in {"cronjob", "cronjob.batch"} and self.resource_error:
                raise ReleaseError(self.resource_error)
            if resource == "configmap":
                if not self.installed:
                    return ""
                return json.dumps(
                    {
                        "kind": "ConfigMap",
                        "metadata": {
                            "name": COMMANDS.STATE_CONFIG_MAP,
                            "namespace": NAMESPACE,
                            "uid": "state-uid",
                        },
                    }
                )
            if resource in {"deployment", "namespace"}:
                return json.dumps(
                    {
                        "kind": "Namespace"
                        if resource == "namespace"
                        else "Deployment",
                        "metadata": {
                            "name": NAMESPACE
                            if resource == "namespace"
                            else "gpu-fault-api-ha",
                            "namespace": "foreign"
                            if self.foreign_anchor
                            else NAMESPACE,
                            "uid": "anchor-uid",
                        },
                    }
                )
        return str(super().run(args, **kwargs))

    def probe_output(self, args: list[str], **kwargs: Any) -> tuple[int, str, str]:
        if args[:2] == ["aws", "amp"]:
            return 254, "", self.amp_error or "ResourceNotFoundException"
        return super().probe_output(args, **kwargs)


@pytest.fixture
def preflight_model(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in (
        "_check_tools",
        "_check_local_inputs",
        "_check_aws_identity",
        "_check_contexts",
        "_check_cpu_capacity",
        "check_cpu_secrets",
        "_check_load_balancer_controller",
        "_check_nlb_inputs",
        "_check_aurora",
        "check_control_record_archive_bucket",
        "check_email_notifications",
    ):
        monkeypatch.setattr(
            CHECKS,
            name,
            lambda *_args, **_kwargs: CHECKS.CheckValue("prerequisite proved"),
        )
    monkeypatch.setattr(CHECKS, "workflow_safety_snapshot", lambda _release: {})

    def build(
        *,
        installed: bool = True,
        missing_refresh: bool = True,
        changed: frozenset[str] = frozenset(
            {"aurora_refresh_drift", "observability_drift"}
        ),
    ) -> SimpleNamespace:
        runner = PreflightRunner(installed=installed, missing_refresh=missing_refresh)
        instance = release(
            runner,
            _load_state=lambda: instance.state,
            _get_json=lambda args: json.loads(runner.run(list(args))),
        )
        instance.config.site_name = "site-test"
        instance.config.clusters = ("gpu-a",)
        instance.config.health = SimpleNamespace(
            amp_workspace_id="ws-test",
            amp_rule_namespace="gpu-fault-control-plane-capacity",
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:alerts",
            require_confirmed_sns_subscription=False,
        )
        instance.config.notifications = SimpleNamespace(admin_email=None)
        instance.release_id = "candidate"
        instance.state = {"release_id": "old", "phase": "complete"}
        monkeypatch.setattr(
            COMMANDS, "classify_release", lambda *_args: diff_from_changed(changed)
        )
        return instance

    return build


@pytest.mark.parametrize("installed", [False, True])
def test_readonly_foundation_preflight_defers_only_planned_verified_absence(
    preflight_model: Any, installed: bool
) -> None:
    instance = preflight_model(installed=installed)
    report = CHECKS.build_preflight_report(
        instance,
        repair_plan=build_execution_plan(
            diff_from_changed({"aurora_refresh_drift", "observability_drift"})
        ),
        bootstrap=not installed,
    )
    checks = {item["name"]: item for item in report["checks"]}
    assert report["healthy"] is True, report["checks"]
    assert checks["monitoring"]["status"] == "WARN", (
        "missing AMP definitions were presented as healthy instead of pending installation"
    )
    assert checks["aurora_credential_refresh"]["status"] == "WARN", (
        "missing refresher was not explicitly deferred to its planned repair"
    )
    assert not ({"apply", "delete", "refresh"} & set(instance.runner.events)), (
        "deployment preflight mutated resources before the release snapshot"
    )
    strict = CHECKS.build_preflight_report(instance)
    assert strict["healthy"] is False, (
        "an ordinary preflight inherited the deploy-only missing-resource exception"
    )


def test_absence_is_not_deferred_for_an_unrelated_application_plan(
    preflight_model: Any,
) -> None:
    instance = preflight_model(changed=frozenset({"cpu_worker_manifests"}))
    report = COMMANDS.build_deploy_preflight_report(instance)
    failed = {item["name"] for item in report["checks"] if item["status"] == "FAIL"}
    assert {"aurora_credential_refresh", "monitoring"} <= failed, (
        "a plan that does not repair missing resources was allowed to proceed"
    )


@pytest.mark.parametrize(
    "error", ["Forbidden", "timeout", "Unauthorized", "invalid JSON"]
)
def test_unreadable_refresher_cannot_be_called_a_repairable_absence(
    preflight_model: Any, error: str
) -> None:
    instance = preflight_model()
    instance.runner.resource_error = error
    report = COMMANDS.build_deploy_preflight_report(instance)
    check = next(
        item for item in report["checks"] if item["name"] == "aurora_credential_refresh"
    )
    assert check["status"] == "FAIL", check
    assert error in check["summary"], "the failed read lost its diagnostic"
    assert not ({"apply", "delete", "refresh"} & set(instance.runner.events)), (
        "an unknown refresher state reached a mutation"
    )


@pytest.mark.parametrize(
    "error", ["AccessDeniedException", "RequestTimeout", "ThrottlingException"]
)
def test_unreadable_amp_definitions_remain_fail_closed(
    preflight_model: Any, error: str
) -> None:
    instance = preflight_model()
    instance.runner.amp_error = error
    report = COMMANDS.build_deploy_preflight_report(instance)
    check = next(item for item in report["checks"] if item["name"] == "monitoring")
    assert check["status"] == "FAIL", check
    assert error in check["summary"], "AMP read failures must retain their cause"


def test_missing_resources_require_the_cpu_namespace_identity(
    preflight_model: Any,
) -> None:
    instance = preflight_model()
    instance.runner.foreign_anchor = True
    report = COMMANDS.build_deploy_preflight_report(instance)
    failed = {item["name"] for item in report["checks"] if item["status"] == "FAIL"}
    assert {"aurora_credential_refresh", "monitoring"} <= failed, (
        "a foreign namespace anchor authorized resource creation"
    )


def test_first_install_keeps_the_gpu_cluster_guard(preflight_model: Any) -> None:
    instance = preflight_model(installed=False)
    instance.config.clusters = ()
    with pytest.raises(ReleaseError, match="at least one GPU cluster"):
        COMMANDS.build_deploy_preflight_report(instance)
    assert instance.runner.events == [], "invalid bootstrap entered resource checks"


@pytest.mark.parametrize("error", ["Forbidden", "timeout", "NotFound"])
def test_unreadable_release_state_does_not_authorize_first_install_repair(
    preflight_model: Any, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    instance = preflight_model()
    monkeypatch.setattr(
        instance.runner, "probe_output", lambda *_args, **_kwargs: (1, "", error)
    )
    with pytest.raises(ReleaseError, match="cannot read configmap"):
        COMMANDS.build_deploy_preflight_report(instance)
    assert instance.runner.events == [], "an unreadable state was treated as bootstrap"


@pytest.mark.parametrize("state", [{}, [], {"phase": "unknown", "release_id": "old"}])
def test_unknown_release_state_never_authorizes_a_repair_plan(
    preflight_model: Any, monkeypatch: pytest.MonkeyPatch, state: object
) -> None:
    instance = preflight_model()
    monkeypatch.setattr(instance, "_load_state", lambda: state)
    with pytest.raises(ReleaseError, match="incomplete or unknown"):
        COMMANDS.build_deploy_preflight_report(instance)
    assert not ({"apply", "delete", "refresh"} & set(instance.runner.events)), (
        "an untrusted state reached a resource mutation"
    )


class ProofRunner(RecordingRunner):
    def __init__(self, status: object) -> None:
        super().__init__()
        self.status = status

    def run(self, args: list[str], **kwargs: Any) -> str:
        if "get" in args:
            self.commands.append(list(args))
            assert r"jsonpath={.data.last-refresh-status\.json}" in args, (
                "AWSCURRENT proof must not read the database password"
            )
            return base64.b64encode(json.dumps(self.status).encode()).decode()
        return str(super().run(args, **kwargs))


@pytest.mark.parametrize("status", ["failed", "old", "missing", "future"])
def test_candidate_job_completion_without_fresh_awscurrent_proof_is_refused(
    status: str,
) -> None:
    finished = datetime.now(UTC)
    if status == "old":
        finished -= timedelta(hours=1)
    elif status == "future":
        finished += timedelta(hours=1)
    payload = (
        None
        if status == "missing"
        else {
            "status": "failed" if status == "failed" else "ok",
            "finished_at": finished.isoformat(),
        }
    )
    runner = ProofRunner(payload)
    instance = SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *args: ["kubectl", *args],
    )
    with pytest.raises(ReleaseError, match="fresh successful AWSCURRENT proof"):
        CREDENTIALS.refresh_aurora_credentials(instance, required=True)
    assert not any("delete" in command for command in runner.commands), (
        "a Job whose credential proof failed was deleted before diagnosis"
    )


def test_a_required_refresh_never_skips_an_absent_cronjob() -> None:
    runner = RecordingRunner(cronjob_exists=False)
    instance = SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *args: ["kubectl", *args],
    )
    with pytest.raises(ReleaseError, match="AWSCURRENT has not been proved"):
        CREDENTIALS.refresh_aurora_credentials(instance, required=True)
    assert runner.commands == [], "a missing required program was executed"


def test_successful_required_refresh_proves_status_before_deleting_its_job() -> None:
    runner = ProofRunner({"status": "ok", "finished_at": datetime.now(UTC).isoformat()})
    instance = SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *args: ["kubectl", *args],
    )

    result = CREDENTIALS.refresh_aurora_credentials(instance, required=True)

    assert result["status"] == "refreshed", (
        "the fresh AWSCURRENT proof was not accepted"
    )
    assert "get" in runner.commands[-2] and "delete" in runner.commands[-1], (
        "the verification Job was deleted before its status was proved"
    )


@pytest.mark.parametrize("failure_minutes_ago", [10, 0])
def test_strict_verification_orders_failed_jobs_against_repair_proof(
    failure_minutes_ago: int,
) -> None:
    documents = {item["kind"]: item for item in objects()}
    status = {
        "status": "ok",
        "finished_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
    }
    jobs = [
        {
            "metadata": {"name": "failed-refresh"},
            "status": {
                "conditions": [
                    {
                        "type": "Failed",
                        "status": "True",
                        "lastTransitionTime": (
                            datetime.now(UTC) - timedelta(minutes=failure_minutes_ago)
                        ).isoformat(),
                    }
                ]
            },
        }
    ]

    def get_json(args: list[str]) -> dict[str, Any]:
        if "role" in args:
            return documents["Role"]
        if "secret" in args:
            return {
                "data": {
                    CREDENTIALS.REFRESH_STATUS_KEY: base64.b64encode(
                        json.dumps(status).encode()
                    ).decode()
                }
            }
        if "job" in args:
            return {"items": jobs}
        return documents["CronJob"]

    instance = release(ObjectRunner(list(documents.values())), _get_json=get_json)
    if failure_minutes_ago == 0:
        with pytest.raises(ReleaseError, match="Job failed"):
            CHECKS.check_aurora_refresh(instance, require_success=True)
    else:
        result = CHECKS.check_aurora_refresh(instance, require_success=True)
        assert result.details["last_successful_time"] is None, (
            "the one-shot verification invented a scheduled CronJob success"
        )
        assert result.details["last_refresh_status"]["status"] == "ok", (
            "an older failed Job masked the newer successful repair"
        )


def test_snapshot_refuses_literal_database_credentials_without_echoing_them() -> None:
    documents = objects()
    literal = "private-database-value-must-not-be-captured"
    documents[-1]["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0][
        "env"
    ].append({"name": "GPU_FAULT_STORE_URL", "value": literal})
    runner = ObjectRunner(documents)
    with pytest.raises(ReleaseError, match="sensitive literal") as failure:
        REFRESH.capture_aurora_refresh_snapshot(release(runner))
    assert literal not in str(failure.value), "snapshot refusal disclosed a credential"
    assert "apply" not in runner.events, "unsafe snapshot reached a mutation"


def test_pending_rollback_preflight_uses_its_trusted_repair_scope(
    preflight_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = preflight_model()
    instance.config.clusters = ()
    previous = {
        "aurora_refresh": {"namespace": NAMESPACE, "objects": objects(), "absent": []}
    }
    instance.state.update(
        phase="rollback-failed",
        previous=previous,
        previous_snapshot_sha256=canonical_sha256(previous),
        execution_plan={"nodes": ["aurora-refresh", "verify"]},
        component_progress={
            "schema_version": 1,
            "global": {"aurora-refresh": {"status": "STARTED"}},
            "clusters": {},
        },
    )
    monkeypatch.setattr(
        CHECKS, "_check_monitoring", lambda _release: CHECKS.CheckValue("proved")
    )
    report = COMMANDS.build_deploy_preflight_report(instance)
    assert report["healthy"] is True, report["checks"]
    assert (
        next(
            item["status"]
            for item in report["checks"]
            if item["name"] == "aurora_credential_refresh"
        )
        == "WARN"
    ), "rollback was blocked by the missing candidate it must restore"
    instance.state["previous_snapshot_sha256"] = "wrong"
    with pytest.raises(ReleaseError, match="trusted previous snapshot"):
        COMMANDS.build_deploy_preflight_report(instance)


@pytest.fixture
def upgrade_model(monkeypatch: pytest.MonkeyPatch) -> Any:
    def build(changed: set[str], *, auto_rollback: bool = False) -> SimpleNamespace:
        instance = repair_release(monkeypatch)
        instance.config.auto_rollback = auto_rollback
        runner = instance.runner
        calls = runner.events
        monkeypatch.setattr(
            ORCHESTRATION,
            "preflight_upgrade_mutations",
            lambda *_args: calls.append("candidate-preflight"),
        )
        return SimpleNamespace(
            release=instance,
            calls=calls,
            runner=runner,
            diff=diff_from_changed(changed),
        )

    return build


@pytest.mark.parametrize(
    ("changed", "consumers"),
    [
        ({"aurora_refresh_drift"}, []),
        (
            {"aurora_refresh_drift", "schema_manifests", "cpu_worker_manifests"},
            ["schema", "cpu-finalize"],
        ),
    ],
)
def test_broken_refresher_repairs_after_snapshot_before_consumers(
    upgrade_model: Any, changed: set[str], consumers: list[str]
) -> None:
    model = upgrade_model(changed)
    ORCHESTRATION.upgrade_release(model.release, diff=model.diff)
    calls = model.calls
    assert "old-proof" not in calls, (
        "upgrade tried to execute the program it must repair"
    )
    assert (
        calls.index("approval") < calls.index("persist") < calls.index("credential-job")
    ), "candidate repair ran before snapshot and approval"
    assert calls.index("credential-job") < calls.index("store-gate"), (
        "rotated credentials still blocked Store gates before the repair"
    )
    assert all(
        max(calls.index("credential-job"), calls.index("candidate-preflight"))
        < calls.index(name)
        for name in consumers
    ), "a database consumer started before AWSCURRENT was proved"
    assert (
        model.release.state["previous"]["aurora_refresh"]
        == (model.runner.snapshots[0][REPAIR_KEY]["previous_refresher"])
    ), "the full snapshot lost the original pre-repair refresher"
    if not consumers:
        assert not ({"schema", "cpu-stage", "cpu-finalize"} & set(calls)), (
            "repair-only unnecessarily changed database or application consumers"
        )


def test_refresher_repair_requires_a_complete_snapshot_even_when_fail_forward(
    upgrade_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = upgrade_model({"aurora_refresh_drift"})
    model.runner.live["role.rbac.authorization.k8s.io"]["metadata"]["name"] = "foreign"
    with pytest.raises(ReleaseError, match="snapshot"):
        ORCHESTRATION.upgrade_release(model.release, diff=model.diff)
    assert "apply" not in model.runner.events, (
        "an incomplete snapshot allowed candidate mutation"
    )
    assert "credential-job" not in model.calls, "the untrusted repair ran its program"


@pytest.mark.parametrize(
    "failure", ["approval", "candidate-preflight", "candidate-proof"]
)
def test_repair_failures_never_reach_schema_or_consumer_rollout(
    upgrade_model: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    model = upgrade_model(
        {"aurora_refresh_drift", "schema_manifests", "cpu_worker_manifests"}
    )

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise ReleaseError(failure)

    if failure == "approval":
        monkeypatch.setattr(model.release, "enforce_manifest_plan_pin", fail)
    elif failure == "candidate-preflight":
        monkeypatch.setattr(ORCHESTRATION, "preflight_upgrade_mutations", fail)
    else:
        model.runner.fail_job = "credential"
    with pytest.raises(
        ReleaseError,
        match="credential Job failed" if failure == "candidate-proof" else failure,
    ):
        ORCHESTRATION.upgrade_release(model.release, diff=model.diff)
    assert not ({"schema", "cpu-stage", "cpu-finalize"} & set(model.calls)), (
        "a failed repair still restarted a consumer or ran DDL"
    )
    if failure == "approval":
        assert "apply" not in model.runner.events, (
            "candidate was applied before its gates"
        )


def test_resume_reproves_awscurrent_without_reinstalling_a_completed_refresher(
    upgrade_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = upgrade_model({"aurora_refresh_drift", "schema_manifests"})

    def fail_schema() -> None:
        raise ReleaseError("schema unavailable")

    monkeypatch.setattr(model.release, "_ensure_schema", fail_schema)
    with pytest.raises(ReleaseError, match="schema unavailable"):
        ORCHESTRATION.upgrade_release(model.release, diff=model.diff)
    assert "aurora-refresh-ready" in model.release.state["completed_phases"], (
        "the completed repair was not durable before schema work"
    )
    monkeypatch.setattr(
        model.release, "_ensure_schema", lambda: model.calls.append("schema")
    )
    ORCHESTRATION.upgrade_release(model.release, resume=True, diff=model.diff)
    assert model.calls.count("credential-job") == 2, (
        "resume reused an old credential proof across an interruption"
    )
    assert model.runner.events.count("apply") == 2, (
        "resume reinstalled an already proved candidate program"
    )
    assert model.calls.count("resume-checkpoint") == 1, (
        "resume omitted checkpoint validation"
    )


def test_resume_rejects_snapshot_drift_before_running_a_refresher(
    upgrade_model: Any,
) -> None:
    model = upgrade_model({"aurora_refresh_drift"})
    previous = {"aurora_refresh": {"corrupt": True}}
    model.release.state = {
        "phase": "failed",
        "release_id": "candidate",
        "release_diff": model.diff.as_dict(),
        "execution_plan": build_execution_plan(model.diff).as_dict(),
        "previous": previous,
        "previous_snapshot_sha256": canonical_sha256({}),
    }
    with pytest.raises(ReleaseError, match="snapshot"):
        ORCHESTRATION.upgrade_release(model.release, resume=True, diff=model.diff)
    assert not ({"credential-job", "old-proof"} & set(model.calls)), (
        "resume ran a refresher before validating its original snapshot"
    )


def test_refresher_only_plan_contains_no_schema_or_consumer_component() -> None:
    plan = build_execution_plan(diff_from_changed({"aurora_refresh_drift"}))
    assert plan.nodes == (ReleaseComponent.AURORA_REFRESH, ReleaseComponent.VERIFY), (
        "repair classification unnecessarily selected a consumer restart"
    )
