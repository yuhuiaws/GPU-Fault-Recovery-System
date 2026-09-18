from __future__ import annotations

import json
import runpy
import sys

import pytest

from scripts.e2e.regional.probes import notify008_probe as probe
from scripts.e2e.regional.probes.notify008_protocol import ProbeError
from tests.regional._cov95_notify008_probe import (
    FACTS,
    argv,
    fake_connection,
    probe_environment,
)
from tests.regional._cov95_notify008_support import report_record


def test_probe_inspection_and_arm_are_bound_and_explicit(tmp_path, monkeypatch, capsys):
    target, arguments, identity = probe_environment(tmp_path, monkeypatch)
    assert probe.checked_context(arguments) == (target, identity), (
        "the actual bundle, Pod and runtime identity must all agree"
    )
    assert probe.main(argv("inspect", arguments)) == 0, "inert inspection must succeed"
    assert json.loads(capsys.readouterr().out)["armed"] is False, (
        "inspection must never arm database or runtime work"
    )
    assert probe.main(argv("arm", arguments)) == 0, "explicit ARM must be acknowledged"
    assert json.loads(capsys.readouterr().out)["armed"] is True, (
        "ARM must persist its marker"
    )
    probe.arm(target)
    probe.require_armed(target)
    assert probe.main(argv("stop", arguments)) == 0, (
        "owned stop request must be acknowledged"
    )
    assert json.loads(capsys.readouterr().out)["stop_requested"] is True, (
        "cleanup must publish the stop marker"
    )
    with pytest.raises(ProbeError, match="rearmed"):
        probe.arm(target)
    with pytest.raises(ProbeError, match="not armed"):
        probe.require_armed(target)


@pytest.mark.parametrize(
    "field", ["expected_pod_uid", "expected_namespace_uid", "bundle_sha256"]
)
def test_probe_refuses_identity_drift_before_arm(tmp_path, monkeypatch, field):
    _, arguments, _ = probe_environment(tmp_path, monkeypatch)
    setattr(arguments, field, "different")
    with pytest.raises(ProbeError, match="identity differs"):
        probe.checked_context(arguments)
    assert not probe.marker("arm").exists(), "identity drift cannot cross ARM"


@pytest.mark.parametrize(
    "name",
    [
        "AWS_ROLE_ARN",
        "GPU_FAULT_EXECUTION_TOKEN",
        "PGHOST",
        "GPU_FAULT_STORE_URL_FILE",
        "KUBERNETES_SERVICE_ACCOUNT_TOKEN",
    ],
)
def test_probe_refuses_inherited_authority_without_using_it(
    tmp_path, monkeypatch, name, capsys
):
    _, arguments, _ = probe_environment(tmp_path, monkeypatch)
    monkeypatch.setenv(name, "local-unapproved-reference")
    assert probe.main(argv("arm", arguments)) == 1, "ambient authority must be refused"
    output = json.loads(capsys.readouterr().out)
    assert output["verdict"] == "FAIL" and output["error_type"] == "ProbeError", (
        "the failure must be explicit without exposing environment values"
    )
    assert "local-unapproved-reference" not in json.dumps(output), (
        "credential-like input values must not enter the report"
    )
    assert not probe.marker("arm").exists(), (
        "refused authority must not arm the sandbox"
    )


def test_probe_refuses_changed_runtime_or_bundle(tmp_path, monkeypatch):
    _, arguments, identity = probe_environment(tmp_path, monkeypatch)
    identity["module_digest"] = "f" * 64
    with pytest.raises(ProbeError, match="runtime differs"):
        probe.checked_context(arguments)
    (probe.CASE_ROOT / "postgres-start").write_text("different payload")
    with pytest.raises(ProbeError, match="bundle changed"):
        probe.load_config()


@pytest.mark.parametrize("payload", ["[]", "{}", "x" * 16385])
def test_config_shape_and_size_are_checked_before_any_operation(
    tmp_path, monkeypatch, payload
):
    probe_environment(tmp_path, monkeypatch)
    (probe.CASE_ROOT / "config.json").write_text(payload)
    with pytest.raises(ProbeError):
        probe.load_config()


def test_control_markers_refuse_symlinks_and_foreign_arm_contents(
    tmp_path, monkeypatch
):
    target, _, _ = probe_environment(tmp_path, monkeypatch)
    outside = tmp_path / "untouched"
    outside.write_text("untouched")
    probe.marker("arm").symlink_to(outside)
    with pytest.raises(ProbeError, match="symlink"):
        probe.write_private(probe.marker("arm"), target.run_id)
    with pytest.raises(ProbeError, match="differs"):
        probe.arm(target)
    assert outside.read_text() == "untouched", (
        "marker refusal must not overwrite another file"
    )
    probe.marker("arm").unlink()
    probe.marker("arm").write_text("foreign")
    with pytest.raises(ProbeError, match="differs"):
        probe.arm(target)


def test_prepare_waits_for_read_only_readiness_then_runs_schema_once(
    tmp_path, monkeypatch
):
    import psycopg

    target, _, _ = probe_environment(tmp_path, monkeypatch)
    probe.arm(target)
    attempts = []

    def connecting():
        attempts.append("connect")
        if len(attempts) == 1:
            raise psycopg.OperationalError("local startup not ready")
        return fake_connection()

    prepared = []
    monkeypatch.setattr(probe, "connect", connecting)
    monkeypatch.setattr(probe, "database_identity", lambda connection: FACTS)
    monkeypatch.setattr(
        probe, "prepare_schema", lambda run_id: prepared.append(run_id) or FACTS
    )
    assert probe.prepare(target) == FACTS, (
        "schema preparation must report observed backend facts"
    )
    assert len(attempts) == 2 and prepared == [target.run_id], (
        "read-only connection retries must not replay schema mutation"
    )
    with pytest.raises(FileExistsError):
        probe.prepare(target)
    assert prepared == [target.run_id], (
        "lost preparation ACK cannot authorize a second DDL run"
    )


def test_run_requires_preparation_and_never_replays_an_existing_work_directory(
    tmp_path, monkeypatch, capsys
):
    target, arguments, identity = probe_environment(tmp_path, monkeypatch)
    probe.arm(target)
    with pytest.raises(ProbeError, match="preparation"):
        probe.run(target, identity)
    (probe.WORK / "prepared.json").write_text(json.dumps(FACTS))
    monkeypatch.setattr(probe, "connect", fake_connection)
    monkeypatch.setattr(probe, "database_identity", lambda connection: FACTS)
    observed = []

    def exercised(factory, root, run_id, **kwargs):
        observed.append((root, run_id, kwargs["seconds"]))
        report = report_record()
        return {key: report[key] for key in ("provider", "variants", "children_reaped")}

    monkeypatch.setattr(probe, "exercise", exercised)
    assert probe.main(argv("run", arguments)) == 0, (
        "complete process evidence must be reported"
    )
    result = json.loads(capsys.readouterr().out)
    assert result["verdict"] == "PASS" and result["provider"] == "SIMULATED", (
        "the probe report must retain its simulator scope"
    )
    assert observed == [(probe.WORK / "runtime", target.run_id, target.seconds - 60)], (
        "the runtime must use only its owned work directory and bounded budget"
    )
    with pytest.raises(FileExistsError):
        probe.run(target, identity)
    assert len(observed) == 1, (
        "a duplicate command cannot silently rerun provider acceptance"
    )


def test_idle_startup_has_no_database_or_runtime_work(tmp_path, monkeypatch):
    probe_environment(tmp_path, monkeypatch)
    sleeps = []

    def stopped(seconds):
        sleeps.append(seconds)
        probe.marker("stop").write_text("stop")

    monkeypatch.setattr(probe.time, "sleep", stopped)
    monkeypatch.setattr(
        probe, "connect", lambda: pytest.fail("idle startup reached SQL")
    )
    monkeypatch.setattr(
        probe,
        "exercise",
        lambda *args, **kwargs: pytest.fail("idle startup started a runtime"),
    )
    assert probe.main(["idle"]) == 0, "idle container must stop on the owned marker"
    assert sleeps == [0.2], "idle start must remain in its bounded no-work barrier"


@pytest.mark.parametrize("failure", ["environment", "pod-uid"])
def test_startup_refuses_missing_fixed_environment_and_invalid_pod_uid(
    tmp_path, monkeypatch, failure
):
    probe_environment(tmp_path, monkeypatch)
    if failure == "environment":
        monkeypatch.delenv("AWS_EC2_METADATA_DISABLED")
    else:
        monkeypatch.setenv("NOTIFY008_POD_UID", "not-a-pod-uid")
    with pytest.raises(ProbeError, match="environment differs|UID is unavailable"):
        probe.idle()


def test_prepare_command_emits_backend_facts_after_explicit_arm(
    tmp_path, monkeypatch, capsys
):
    target, arguments, _ = probe_environment(tmp_path, monkeypatch)
    probe.arm(target)
    monkeypatch.setattr(probe, "connect", fake_connection)
    monkeypatch.setattr(probe, "database_identity", lambda connection: FACTS)
    prepared = []
    monkeypatch.setattr(
        probe, "prepare_schema", lambda run_id: prepared.append(run_id) or FACTS
    )
    assert probe.main(argv("prepare", arguments)) == 0, (
        "the preparation entry must execute the separately authorized schema step"
    )
    assert json.loads(capsys.readouterr().out) == FACTS and prepared == [
        target.run_id
    ], "preparation output must match the observed isolated-backend facts"


def test_incomplete_process_report_is_persisted_as_failure_not_promoted(
    tmp_path, monkeypatch, capsys
):
    target, arguments, _ = probe_environment(tmp_path, monkeypatch)
    probe.arm(target)
    (probe.WORK / "prepared.json").write_text(json.dumps(FACTS))
    monkeypatch.setattr(probe, "connect", fake_connection)
    monkeypatch.setattr(probe, "database_identity", lambda connection: FACTS)
    monkeypatch.setattr(
        probe,
        "exercise",
        lambda *args, **kwargs: {
            "provider": "SIMULATED",
            "variants": [],
            "children_reaped": True,
        },
    )
    assert probe.main(argv("run", arguments)) == 1, (
        "an incomplete matrix must fail even if its children exited cleanly"
    )
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "FAIL" and report["errors"], (
        "missing process evidence must be explicit"
    )
    assert json.loads((probe.WORK / "result.json").read_text()) == report, (
        "failure evidence must be durable before command return"
    )


def test_portable_script_entry_refuses_environment_before_any_probe_io(
    tmp_path, monkeypatch, capsys
):
    probe_environment(tmp_path, monkeypatch)
    monkeypatch.delenv("AWS_EC2_METADATA_DISABLED")
    monkeypatch.setattr(sys, "argv", [probe.__file__, "idle"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(probe.__file__, run_name="__main__")
    assert raised.value.code == 1, (
        "script startup must enforce the same environment guard"
    )
    assert json.loads(capsys.readouterr().out)["error_type"] == "ProbeError", (
        "standalone refusal must not print environment values"
    )
