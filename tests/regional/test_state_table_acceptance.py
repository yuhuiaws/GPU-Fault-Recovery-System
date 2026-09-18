from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.schema_migrations import LATEST_POSTGRES_SCHEMA_VERSION
from scripts.e2e.regional import run_state_table_acceptance as runner
from scripts.e2e.regional.probes import state_table_snapshot as probe


def proof(
    kind: str = "remote_command", mode: str = "dual", *, verify: bool = True
) -> dict[str, Any]:
    report = {
        "schema_version": LATEST_POSTGRES_SCHEMA_VERSION,
        "schema_valid": True,
        "writer": True,
        "read_only": True,
        "database_sha256": "a" * 64,
        "state": {
            "kind": kind,
            "mode": mode,
            "revision": 3,
            "legacy_rows": 2,
            "dedicated_rows": 2,
            "backfill_complete": True,
            "legacy_purged": False,
            "verification_applicable": mode == "dual",
            "verification_performed": verify and mode == "dual",
        },
    }
    if verify and mode == "dual":
        report["state"].update(
            {
                "verified": True,
                "missing_rows": 0,
                "mismatched_rows": 0,
                "extra_rows": 0,
                "invalid_records": 0,
                "noncanonical_records": 0,
            }
        )
    return report


class Regional:
    def __init__(self, kind: str = "remote_command") -> None:
        self.report = proof(kind)
        self.roles = {}
        self.calls = []
        self.pod_proofs = 0
        self.change_uid = False
        self.mode_drift = False
        self.fail_exec = False
        self.settings = SimpleNamespace(environment=lambda: {})
        for app in runner.CPU_ROLES:
            self.roles[app] = {
                "metadata": {"name": app, "uid": f"{app}-uid", "generation": 1},
                "spec": {"replicas": 1},
                "status": {
                    "observedGeneration": 1,
                    "replicas": 1,
                    "readyReplicas": 1,
                    "availableReplicas": 1,
                    "updatedReplicas": 1,
                },
            }

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.calls.append((plane, args, kwargs))
        assert plane == "cpu", "state-table audit must never contact GPU Kubernetes"
        if args[:2] == ("get", "deployment"):
            return json.dumps(self.roles[args[2]])
        assert args[0] == "exec", (
            "state-table audit must use only get or read-only exec"
        )
        if self.fail_exec:
            raise runner.RegionalFixtureError("probe failed")
        self.pod_proofs += 1
        value = deepcopy(self.report)
        if args[-1] == "metadata":
            value["state"]["verification_performed"] = False
            for key in (
                "verified",
                "missing_rows",
                "mismatched_rows",
                "extra_rows",
                "invalid_records",
                "noncanonical_records",
            ):
                value["state"].pop(key, None)
        if self.pod_proofs > 1 and self.mode_drift:
            value["state"]["revision"] += 1
        if self.change_uid:
            self.roles[runner.CPU_ROLES[0]]["metadata"]["uid"] = "replacement"
        return json.dumps(value)

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        return (
            [{"name": app, "uid": f"{app}-pod-uid"}]
            if self.roles[app]["spec"]["replicas"]
            else []
        )

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "release-test", "cluster_id": "cluster-test"}


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
def test_full_audit_reads_every_cpu_replica_with_one_full_verification(
    kind: str,
) -> None:
    regional = Regional(kind)
    result = runner.audit(
        regional,
        kind,
        "dual",
        verify=True,
        deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    assert result["verdict"] == "PASS", result
    assert len(result["snapshots"]) == 3, result
    assert result["mutation_performed"] is False, result
    commands = [args for _, args, _ in regional.calls if args[0] == "exec"]
    assert sum(args[-1] == "verify" for args in commands) == 1, commands
    assert all(
        "/opt/gpu-fault/control-plane/bin/python" in args for args in commands
    ), commands
    assert all(
        kwargs["timeout"] == 40
        for _, args, kwargs in regional.calls
        if args[0] == "exec"
    ), regional.calls


def test_probe_response_cannot_replace_observed_pod_or_release_identity() -> None:
    regional = Regional()
    regional.report.update(
        {
            "app": "untrusted-role",
            "pod": "untrusted-pod",
            "pod_uid": "untrusted-uid",
            "release_id": "untrusted-release",
            "cluster_id": "untrusted-cluster",
            "mutation_performed": True,
            "private_response": "must-not-be-persisted",
        }
    )
    regional.report["state"]["private_response"] = "must-not-be-persisted"
    result = runner.audit(
        regional,
        "remote_command",
        "dual",
        verify=True,
        deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    assert (
        result["release_id"] == "release-test"
        and result["cluster_id"] == "cluster-test"
    )
    assert result["mutation_performed"] is False
    assert {
        (item["app"], item["pod"], item["pod_uid"]) for item in result["snapshots"]
    } == {(app, app, f"{app}-pod-uid") for app in runner.CPU_ROLES}
    assert "must-not-be-persisted" not in json.dumps(result)


def test_all_replicas_are_probed_even_when_the_first_role_has_multiple_pods(
    monkeypatch,
) -> None:
    regional = Regional()
    for deployment in regional.roles.values():
        deployment["spec"]["replicas"] = 2
        for key in (
            "replicas",
            "readyReplicas",
            "availableReplicas",
            "updatedReplicas",
        ):
            deployment["status"][key] = 2
    monkeypatch.setattr(
        regional,
        "ready_pods",
        lambda _plane, app: [
            {"name": f"{app}-{index}", "uid": f"{app}-{index}-uid"} for index in (1, 0)
        ],
    )
    result = runner.audit(
        regional,
        "remote_command",
        "dual",
        verify=True,
        deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    commands = [args for _, args, _ in regional.calls if args[0] == "exec"]
    assert len(result["snapshots"]) == len(commands) == 6
    assert sum(args[-1] == "verify" for args in commands) == 1
    assert {(item["pod"], item["pod_uid"]) for item in result["snapshots"]} == {
        (f"{app}-{index}", f"{app}-{index}-uid")
        for app in runner.CPU_ROLES
        for index in (0, 1)
    }


@pytest.mark.parametrize(
    "failure",
    [
        "missing-pod",
        "duplicate-pod-name",
        "duplicate-pod-uid",
        "missing-pod-uid",
        "duplicate-deployment-uid",
        "terminating-deployment",
        "boolean-replicas",
        "stale-generation",
        "missing-role-replica",
        "residual-disabled-spool",
    ],
)
def test_incomplete_or_ambiguous_cpu_population_never_starts_a_probe(
    monkeypatch, failure
) -> None:
    regional = Regional()
    first, second, spool = (regional.roles[app] for app in runner.CPU_ROLES)
    if failure == "missing-pod":
        monkeypatch.setattr(regional, "ready_pods", lambda *_args: [])
    elif failure in {"duplicate-pod-name", "duplicate-pod-uid", "missing-pod-uid"}:
        monkeypatch.setattr(
            regional,
            "ready_pods",
            lambda _plane, app: [
                {
                    "name": "shared" if failure == "duplicate-pod-name" else app,
                    "uid": ""
                    if failure == "missing-pod-uid"
                    else ("shared" if failure == "duplicate-pod-uid" else f"{app}-uid"),
                }
            ],
        )
    elif failure == "duplicate-deployment-uid":
        second["metadata"]["uid"] = first["metadata"]["uid"]
    elif failure == "terminating-deployment":
        first["metadata"]["deletionTimestamp"] = "2026-09-12T00:00:00Z"
    elif failure == "boolean-replicas":
        first["spec"]["replicas"] = True
    elif failure == "stale-generation":
        first["status"]["observedGeneration"] = 0
    elif failure == "missing-role-replica":
        second["spec"]["replicas"] = 2
        for key in (
            "replicas",
            "readyReplicas",
            "availableReplicas",
            "updatedReplicas",
        ):
            second["status"][key] = 2
    else:
        spool["spec"]["replicas"] = 0
    with pytest.raises(runner.RegionalFixtureError):
        runner.audit(
            regional,
            "remote_command",
            "dual",
            verify=True,
            deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
    assert regional.pod_proofs == 0, "all CPU roles must be complete before any probe"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "legacy"),
        ("kind", "workflow"),
        ("backfill_complete", False),
        ("verified", False),
        ("verification_performed", "true"),
        ("missing_rows", 1),
        ("mismatched_rows", 1),
        ("extra_rows", 1),
        ("invalid_records", 1),
        ("noncanonical_records", 1),
        ("legacy_rows", True),
        ("dedicated_rows", -1),
        ("revision", None),
    ],
)
def test_incomplete_state_cannot_pass(field: str, value: Any) -> None:
    report = proof()
    report["state"][field] = value
    assert runner.state_errors(report, "remote_command", "dual", verify=True), report


@pytest.mark.parametrize(
    "field",
    ["schema_version", "read_only", "writer", "database_sha256", "schema_valid"],
)
def test_missing_proof_field_cannot_pass(field: str) -> None:
    report = proof()
    del report[field]
    assert runner.state_errors(report, "remote_command", "dual", verify=True), (
        f"missing proof field {field!r} was accepted: {report!r}"
    )


def test_dedicated_mode_allows_retained_legacy_until_explicit_purge() -> None:
    report = proof(mode="dedicated")
    assert not runner.state_errors(
        report, "remote_command", "dedicated", verify=True
    ), f"dedicated mode must allow retained legacy rows before purge: {report!r}"
    report["state"]["legacy_purged"] = True
    assert runner.state_errors(report, "remote_command", "dedicated", verify=True), (
        f"purged legacy state must reject nonzero legacy rows: {report!r}"
    )
    report["state"]["legacy_rows"] = 0
    assert not runner.state_errors(
        report, "remote_command", "dedicated", verify=True
    ), f"dedicated mode must accept completed legacy purge: {report!r}"


@pytest.mark.parametrize(
    "failure", ["change_uid", "mode_drift", "fail_exec", "unready", "deadline"]
)
def test_changed_or_incomplete_inventory_fails_closed(failure: str) -> None:
    regional = Regional()
    if failure in {"change_uid", "mode_drift", "fail_exec"}:
        setattr(regional, failure, True)
    elif failure == "unready":
        regional.roles[runner.CPU_ROLES[0]]["status"]["readyReplicas"] = 0
    deadline = datetime.now(timezone.utc) + timedelta(
        seconds=20 if failure == "deadline" else 300
    )
    with pytest.raises(runner.RegionalFixtureError):
        runner.audit(regional, "remote_command", "dual", verify=True, deadline=deadline)
    if failure in {"deadline", "unready"}:
        assert regional.pod_proofs == 0


def test_only_disabled_optional_spool_can_be_omitted() -> None:
    regional = Regional()
    spool = regional.roles[runner.CPU_ROLES[-1]]
    spool["spec"]["replicas"] = 0
    spool["status"] = {"observedGeneration": 1}
    assert not runner.cpu_population(regional)[runner.CPU_ROLES[-1]]["pods"]
    regional.roles[runner.CPU_ROLES[0]]["spec"]["replicas"] = 0
    with pytest.raises(runner.RegionalFixtureError):
        runner.cpu_population(regional)


def test_probe_refuses_unreadable_current_credential_file_before_connect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://obsolete@127.0.0.1/probe")
    monkeypatch.setattr(
        "psycopg.connect", lambda *_args, **_kwargs: pytest.fail("stale DSN used")
    )
    with pytest.raises(RuntimeError, match="credentials"):
        probe.snapshot("remote_command", verify=True)


@pytest.mark.parametrize(
    "schemas", [["pg_catalog", "public"], ["pg_catalog", "alternate"]]
)
def test_probe_uses_readonly_session_and_existing_status_api(
    monkeypatch: pytest.MonkeyPatch, schemas
) -> None:
    calls = []
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://test@127.0.0.1/probe")

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def execute(self, sql):
            calls.append(sql)
            if sql.startswith("SELECT current_database"):
                return SimpleNamespace(
                    fetchone=lambda: ("probe", "127.0.0.1", 5432, schemas, False, "on")
                )
            if "schema_version" in sql:
                return SimpleNamespace(
                    fetchall=lambda: [(LATEST_POSTGRES_SCHEMA_VERSION,)]
                )
            return SimpleNamespace(
                fetchall=lambda: [
                    (m.version, m.name, m.checksum)
                    for m in probe.POSTGRES_SCHEMA_MIGRATIONS
                ]
            )

    def connect(_url, **kwargs):
        assert "default_transaction_read_only=on" in kwargs["options"]
        assert kwargs["connect_timeout"] == 10
        return Connection()

    monkeypatch.setattr("psycopg.connect", connect)
    monkeypatch.setattr(probe, "validate_state_table_schema", lambda _connection: None)
    monkeypatch.setattr(
        probe,
        "state_table_status",
        lambda _connection, kind, verify: proof(kind)["state"],
    )
    report = probe.snapshot("workflow", verify=True)
    assert report["state"]["kind"] == "workflow"
    assert report["read_only"] is True
    assert (
        report["database_sha256"]
        == hashlib.sha256(
            json.dumps(("probe", "127.0.0.1", 5432, schemas)).encode()
        ).hexdigest()
    ), "database identity must include the effective schema search path"
    assert all(sql.startswith("SELECT ") for sql in calls), calls


@pytest.mark.parametrize(
    "row",
    [
        None,
        (),
        ("probe", "127.0.0.1", 5432, ["public"], True, "on"),
        ("probe", "127.0.0.1", 5432, ["public"], "false", "on"),
        ("probe", "127.0.0.1", 5432, ["public"], False, "off"),
        ("probe", None, 5432, ["public"], False, "on"),
        ("probe", "127.0.0.1", True, ["public"], False, "on"),
        ("probe", "127.0.0.1", 5432, [], False, "on"),
        ("probe", "127.0.0.1", 5432, "public", False, "on"),
    ],
)
def test_probe_refuses_unverified_writer_identity_before_schema_reads(
    monkeypatch, row
) -> None:
    from contextlib import nullcontext

    calls = []

    def execute(query):
        calls.append(query)
        return SimpleNamespace(fetchone=lambda: row)

    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://test@127.0.0.1/probe")
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setattr(
        "psycopg.connect",
        lambda *_args, **_kwargs: nullcontext(SimpleNamespace(execute=execute)),
    )
    with pytest.raises(RuntimeError, match="read-only connection to the writer"):
        probe.snapshot("workflow", verify=True)
    assert len(calls) == 1, (
        "unverified writer identity must fail before schema or record reads"
    )


def test_probe_failure_does_not_print_database_error_details(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("private-connection-value")

    monkeypatch.setattr(probe, "snapshot", fail)
    assert probe.main(["workflow", "verify"]) == 1
    output = capsys.readouterr().out
    assert "private-connection-value" not in output
    assert json.loads(output)["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "arguments", [[], ["workflow"], ["workflow", "unknown"], ["unknown", "verify"]]
)
def test_probe_rejects_invalid_arguments_without_connecting(
    monkeypatch, capsys, arguments
) -> None:
    monkeypatch.setattr(
        "psycopg.connect", lambda *_args, **_kwargs: pytest.fail("unexpected SQL")
    )
    assert probe.main(arguments) == 1
    assert json.loads(capsys.readouterr().out) == {
        "verdict": "FAIL",
        "error_type": "ValueError",
    }


def test_probe_main_metadata_returns_only_the_report(monkeypatch, capsys) -> None:
    calls = []

    def snapshot(kind, *, verify):
        calls.append((kind, verify))
        return proof(kind, verify=verify)

    monkeypatch.setattr(probe, "snapshot", snapshot)
    assert probe.main(["workflow", "metadata"]) == 0
    assert calls == [("workflow", False)]
    assert json.loads(capsys.readouterr().out) == proof("workflow", verify=False)


def main_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case_id: str = "GF-REGIONAL-BOOT-030",
    mode: str = "dual",
) -> SimpleNamespace:
    from scripts.e2e.regional import live_driver_guard as guard

    regional = Regional(runner.CASES[case_id])
    regional.report = proof(runner.CASES[case_id], mode)
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner, "settings_from_arguments", lambda _: regional.settings)
    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda _: regional)
    monkeypatch.setattr(guard, "source_digest", lambda: "state-table-unit-source")
    monkeypatch.setattr(guard, "applied_site_profile", lambda: None)
    monkeypatch.delenv(guard.SITE_PROFILE_ENV, raising=False)
    monkeypatch.setenv("GPU_FAULT_ACCEPTANCE_EXECUTION_SCOPE", "formal")
    monkeypatch.delenv("GPU_FAULT_ACCEPTANCE_SELECTION_REFERENCE", raising=False)
    previous, predecessor = runner.predecessor_path(tmp_path, case_id, "")
    assert previous is not None and predecessor is not None
    predecessor.parent.mkdir(parents=True, exist_ok=True)
    predecessor.write_text(
        json.dumps(
            {
                "case_id": previous,
                "verdict": "PASS",
                "status": "COMPLETED",
                **regional.evidence_identity(),
            }
        )
    )
    argv = [
        runner.__file__,
        "--run-dir",
        str(tmp_path),
        "--case",
        case_id,
        "--expected-mode",
        mode,
    ]

    def invoke(*extra: str, execute: bool = False) -> int:
        selected = list(argv)
        if execute:
            selected += [
                "--execute",
                "--confirm",
                runner.CONFIRMATION,
                "--maintenance-window-end",
                "2099-01-01T00:00:00Z",
            ]
        monkeypatch.setattr(sys, "argv", [*selected, *extra])
        return runner.main()

    return SimpleNamespace(
        regional=regional,
        invoke=invoke,
        case_id=case_id,
        predecessor=predecessor,
        plan=tmp_path / "cases" / case_id / "plan.json",
        result=runner.case_evidence_path(tmp_path, case_id),
        root=tmp_path,
    )


@pytest.mark.parametrize("case_id", tuple(runner.CASES))
@pytest.mark.parametrize("mode", runner.MODES)
def test_main_uses_real_schema3_plan_and_readonly_execute(
    monkeypatch, tmp_path, case_id, mode
) -> None:
    case = main_environment(monkeypatch, tmp_path, case_id, mode)
    assert case.invoke() == 0
    plan = json.loads(case.plan.read_text())
    assert plan["schema_version"] == 3 and plan["preflight_passed"] is True
    assert plan["arguments_sha256"] and plan["details_sha256"]
    assert plan["details"]["kind"] == runner.CASES[case_id]
    assert plan["details"]["expected_mode"] == mode
    assert case.regional.pod_proofs == 0
    assert case.invoke(execute=True) == 0
    result = json.loads(case.result.read_text())
    assert result["status"] == "COMPLETED" and result["verdict"] == "PASS"
    assert result["case_id"] == case_id and result["attempt"] == 1
    assert (
        result["release_id"] == "release-test"
        and result["cluster_id"] == "cluster-test"
    )
    assert result["predecessor"]["valid"] is True
    assert (
        result["mutation_performed"] is False and result["cleanup"]["required"] is False
    )
    assert result["completed_at"]
    assert case.result.stat().st_mode & 0o777 == 0o600
    commands = [args for _, args, _ in case.regional.calls if args[0] == "exec"]
    assert len(commands) == 3
    assert sum(args[-1] == "verify" for args in commands) == int(mode == "dual")


@pytest.mark.parametrize(
    "failure", ["settings", "identity", "inventory", "predecessor"]
)
def test_failed_replan_invalidates_the_previous_approval(
    monkeypatch, tmp_path, failure, capsys
) -> None:
    case = main_environment(monkeypatch, tmp_path)
    assert case.invoke() == 0

    def fail(*args, **kwargs):
        raise RuntimeError("private-connection-diagnostic")

    if failure == "settings":
        monkeypatch.setattr(runner, "settings_from_arguments", fail)
    elif failure == "identity":
        monkeypatch.setattr(case.regional, "evidence_identity", fail)
    elif failure == "inventory":
        monkeypatch.setattr(case.regional, "kubectl", fail)
    else:
        case.predecessor.unlink()
    assert case.invoke() == 1
    failed = json.loads(case.plan.read_text())
    assert failed["preflight_passed"] is False
    assert failed["status"] == "FAILED"
    assert case.invoke(execute=True) == 1
    assert case.regional.pod_proofs == 0
    assert json.loads(case.result.read_text())["verdict"] == "FAIL"
    assert "private-connection-diagnostic" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "failure",
    [
        "missing-plan",
        "old-schema",
        "changed-source",
        "changed-mode",
        "wrong-confirmation",
        "expired-window",
        "settings",
        "identity",
        "inventory",
        "predecessor",
        "probe",
    ],
)
def test_execute_failures_never_leave_an_old_pass(
    monkeypatch, tmp_path, failure, capsys
) -> None:
    from scripts.e2e.regional import live_driver_guard as guard

    case = main_environment(monkeypatch, tmp_path)
    assert case.invoke() == 0
    case.result.write_text(
        json.dumps({"case_id": case.case_id, "verdict": "PASS", "status": "COMPLETED"})
    )
    extra = []

    def fail(*args, **kwargs):
        assert json.loads(case.result.read_text())["verdict"] != "PASS"
        raise RuntimeError("private-connection-diagnostic")

    if failure == "missing-plan":
        case.plan.unlink()
    elif failure == "old-schema":
        plan = json.loads(case.plan.read_text())
        plan["schema_version"] = 2
        case.plan.write_text(json.dumps(plan))
    elif failure == "changed-source":
        monkeypatch.setattr(guard, "source_digest", lambda: "different-source")
    elif failure == "changed-mode":
        extra = ["--expected-mode", "legacy"]
    elif failure == "wrong-confirmation":
        extra = ["--confirm", "wrong"]
    elif failure == "expired-window":
        extra = ["--maintenance-window-end", "2000-01-01T00:00:00Z"]
    elif failure == "settings":
        monkeypatch.setattr(runner, "settings_from_arguments", fail)
    elif failure == "identity":
        monkeypatch.setattr(case.regional, "evidence_identity", fail)
    elif failure == "inventory":
        monkeypatch.setattr(case.regional, "kubectl", fail)
    elif failure == "predecessor":
        previous = json.loads(case.predecessor.read_text())
        previous["status"] = "FAILED"
        case.predecessor.write_text(json.dumps(previous))
    else:
        case.regional.fail_exec = True
    assert case.invoke(*extra, execute=True) == 1
    result = json.loads(case.result.read_text())
    assert result["verdict"] == "FAIL" and result["status"] == "FAILED"
    assert result["error_type"] and result["completed_at"]
    assert "private-connection-diagnostic" not in capsys.readouterr().out
    assert "private-connection-diagnostic" not in case.result.read_text()


@pytest.mark.parametrize("drift", ["release", "population", "kind", "expected_mode"])
def test_audit_checks_the_exact_approved_target_before_probe_exec(
    monkeypatch, tmp_path, drift
) -> None:
    from scripts.e2e.regional.live_driver_guard import details_sha256

    case = main_environment(monkeypatch, tmp_path)
    assert case.invoke() == 0
    if drift == "release":
        calls = iter(
            [
                {"release_id": "release-test", "cluster_id": "cluster-test"},
                {"release_id": "replacement-release", "cluster_id": "cluster-test"},
            ]
        )
        monkeypatch.setattr(case.regional, "evidence_identity", lambda: next(calls))
    elif drift == "population":
        case.regional.roles[runner.CPU_ROLES[0]]["metadata"]["uid"] = "replacement"
    else:
        plan = json.loads(case.plan.read_text())
        plan["details"][drift] = "workflow" if drift == "kind" else "dedicated"
        plan["details_sha256"] = details_sha256(plan["details"])
        case.plan.write_text(json.dumps(plan))
    assert case.invoke(execute=True) == 1
    assert case.regional.pod_proofs == 0
    assert json.loads(case.result.read_text())["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "response", ["", "not-json", "[]", "null", '{"verdict":"FAIL"}']
)
def test_main_rejects_empty_malformed_or_failed_probe_responses(
    monkeypatch, tmp_path, response
) -> None:
    case = main_environment(monkeypatch, tmp_path)
    assert case.invoke() == 0
    kubectl = case.regional.kubectl

    def read(plane, *arguments, **kwargs):
        if arguments[0] == "exec":
            return response
        return kubectl(plane, *arguments, **kwargs)

    monkeypatch.setattr(case.regional, "kubectl", read)
    assert case.invoke(execute=True) == 1
    result = json.loads(case.result.read_text())
    assert result["verdict"] == "FAIL" and result["status"] == "FAILED"
    assert "snapshots" not in result, (
        "partial or malformed proof must not become acceptance evidence"
    )


@pytest.mark.parametrize("failure", ["abort-handler", "fixture", "environment"])
@pytest.mark.parametrize("execute", [False, True])
def test_early_setup_failures_invalidate_existing_evidence(
    monkeypatch, tmp_path, capsys, failure, execute
) -> None:
    case = main_environment(monkeypatch, tmp_path)
    assert case.invoke() == 0
    path = case.result if execute else case.plan
    if execute:
        path.write_text(json.dumps({"verdict": "PASS", "status": "COMPLETED"}))

    def fail(*_args, **_kwargs):
        saved = json.loads(path.read_text())
        assert (
            saved.get("verdict") != "PASS" and saved.get("preflight_passed") is not True
        )
        raise RuntimeError("private-startup-diagnostic")

    if failure == "abort-handler":
        monkeypatch.setattr(runner, "install_abort_signals", fail)
    elif failure == "fixture":
        monkeypatch.setattr(runner, "RegionalLiveFixture", fail)
    else:
        case.regional.settings.environment = fail
    assert case.invoke(execute=execute) == 1
    result = json.loads(path.read_text())
    assert result["verdict"] == "FAIL" and result["status"] == "FAILED"
    assert "private-startup-diagnostic" not in path.read_text()
    assert "private-startup-diagnostic" not in capsys.readouterr().out


def test_supervision_loss_persists_failure_and_blocks_a_retry(
    monkeypatch, tmp_path
) -> None:
    from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
    from scripts.e2e.regional import regional_commands
    from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture

    case = main_environment(monkeypatch, tmp_path)
    assert case.invoke() == 0
    calls = []

    def lost(*args, **kwargs):
        calls.append(True)
        raise ProcessSupervisionLost("unit command lost")

    monkeypatch.setattr(regional_commands, "run_command", lost)
    monkeypatch.setattr(
        case.regional,
        "kubectl",
        lambda *args, **kwargs: RegionalLiveFixture.run(["unit-read"]).stdout,
    )
    with pytest.raises(ProcessSupervisionLost):
        case.invoke(execute=True)
    assert json.loads(case.result.read_text())["verdict"] == "FAIL"
    assert (tmp_path / "command-supervision-lost.json").is_file(), (
        f"supervision loss must persist its retry-blocking marker in {tmp_path}"
    )
    assert case.invoke(execute=True) == 1
    assert calls == [True]
