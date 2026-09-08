"""The first minute of ``scripts/staging_deploy.py deploy()``.

What the command knows before the source scan, the gates and the build: the
cluster ARNs and administrator email (from the command or ``site.yaml``), the
SES/SNS confirmation check, and the two consent refusals the outer hop can
already make from the quick status report and a reused release manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin.notification_precheck import (
    EmailConfirmation,
    email_confirmation_refusal,
)
from gpu_fault_release.regional_admin_commands import (
    SUPERSEDE_FAILED_TRANSACTION_ENV,
    SUPERSEDE_FAILED_TRANSACTION_FLAG,
)
from gpu_fault_release.regional_schema_change import ACCEPT_SCHEMA_CHANGE_ENV
from scripts import staging_deploy

CPU_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/control"
GPU_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"
EMAIL = "operations@example.com"


def _write_site(state: Path, *, admin_email: str | None = EMAIL) -> None:
    state.mkdir(parents=True, exist_ok=True)
    document = {
        "kind": "RegionalSite",
        "spec": {
            "autoRollback": True,
            "cpu": {"eksArn": CPU_ARN, "hyperpodClusterName": "control"},
            "clusters": [{"eksClusterArn": GPU_ARN, "hyperpodClusterName": "hp-a"}],
            "notifications": {"adminEmail": admin_email} if admin_email else {},
        },
    }
    (state / "site.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")


def _arguments(state: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "repo_root": state.parent / "repo",
        "state_dir": state,
        "cpu_cluster_arn": None,
        "gpu_cluster_arn": [],
        "admin_email": None,
        "base": "origin/main",
        "wait_for_email_confirmation": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _email(*, ses: bool, sns: str) -> EmailConfirmation:
    return EmailConfirmation(
        sender=EMAIL,
        admin_email=EMAIL,
        ses_verified=ses,
        ses_identity_created=not ses,
        sns_topic_arn="arn:aws:sns:us-east-1:123456789012:gpu-fault-site-alerts",
        sns_status=sns,
        sns_subscription_arn=None,
    )


# --- inputs ------------------------------------------------------------------


def test_parser_makes_the_cluster_inputs_optional() -> None:
    parsed = staging_deploy.parser().parse_args(["--state-dir", "/tmp/x"])

    assert parsed.cpu_cluster_arn is None
    assert parsed.gpu_cluster_arn == []
    assert parsed.admin_email is None
    assert parsed.wait_for_email_confirmation == 0


def test_deploy_inputs_come_from_the_managed_site(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write_site(state)
    arguments = _arguments(state)

    resolved = staging_deploy.resolve_deploy_inputs(arguments, state_dir=state)

    assert resolved == (CPU_ARN, (GPU_ARN,), EMAIL)
    assert arguments.cpu_cluster_arn == CPU_ARN, (
        "the apply hands the inner CLI a full command"
    )
    assert arguments.gpu_cluster_arn == [GPU_ARN]
    assert arguments.admin_email == EMAIL


def test_command_inputs_win_over_the_site(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write_site(state)
    new_gpu = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b"

    resolved = staging_deploy.resolve_deploy_inputs(
        _arguments(state, gpu_cluster_arn=[GPU_ARN, new_gpu], admin_email="ops@x.io"),
        state_dir=state,
    )

    assert resolved == (CPU_ARN, (GPU_ARN, new_gpu), "ops@x.io")


def test_first_deploy_requires_all_three_inputs(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()

    with pytest.raises(staging_deploy.StagingDeployError, match="--cpu-cluster-arn"):
        staging_deploy.resolve_deploy_inputs(_arguments(state), state_dir=state)
    with pytest.raises(staging_deploy.StagingDeployError, match="--admin-email"):
        staging_deploy.resolve_deploy_inputs(
            _arguments(state, cpu_cluster_arn=CPU_ARN, gpu_cluster_arn=[GPU_ARN]),
            state_dir=state,
        )


def test_rerun_command_is_one_parameter_on_a_managed_site(tmp_path: Path) -> None:
    upgrade = staging_deploy.deploy_rerun_command(
        state_dir=tmp_path,
        first_deploy=False,
        cpu_cluster_arn=CPU_ARN,
        gpu_cluster_arns=(GPU_ARN,),
        admin_email=EMAIL,
    )
    first = staging_deploy.deploy_rerun_command(
        state_dir=tmp_path,
        first_deploy=True,
        cpu_cluster_arn=CPU_ARN,
        gpu_cluster_arns=(GPU_ARN,),
        admin_email=EMAIL,
    )

    assert upgrade == f"gpu-fault-admin deploy --state-dir {tmp_path}"
    assert first.startswith("gpu-fault-admin deploy --cpu-cluster-arn "), (
        "a first deploy is rerun with all four inputs"
    )
    assert f"--gpu-cluster-arn {GPU_ARN}" in first and f"--admin-email {EMAIL}" in first


# --- the orchestration up to the first-minute stops ---------------------------


def _stub_until_first_minute(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: Path,
    events: list[str],
    report: dict[str, object] | None,
    manifest: dict[str, object] | None,
) -> None:
    """Everything before the deploy host and the gates, recorded as events; the
    deploy host, the gates and the apply fail the test if reached."""

    snapshot = state.parent / "snapshot"
    snapshot.mkdir(exist_ok=True)
    if manifest is not None:
        (snapshot / "dist").mkdir(exist_ok=True)
        (snapshot / "dist/current-release.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    source = staging_deploy.SourceCheckout(
        repository_root=snapshot,
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    signing = staging_deploy.SigningMaterial(
        private_key=state / "k",
        public_key=state / "p",
        password_file=state / "w",
        password="",
    )
    previous = {
        "schema_version": 1,
        "status": "PASSED",
        "identities": {
            "application": {"sha256": "9" * 64},
            "deploy_host": {"sha256": "b" * 64, "bundle": {"sha256": "c" * 64}},
        },
        "source": {"fingerprint": "0" * 64, "git_commit": "b" * 40},
        "live": {},
    }

    class Lock:
        def __enter__(self) -> int:
            events.append("lock-enter")
            return 17

        def __exit__(self, *_arguments: object) -> None:
            events.append("lock-exit")

    monkeypatch.setattr(
        staging_deploy,
        "validate_source_checkout",
        lambda _root, **_k: events.append("scan"),
    )
    monkeypatch.setattr(
        staging_deploy, "prepare_source_checkout", lambda *_a, **_k: source
    )
    monkeypatch.setattr(
        staging_deploy, "ensure_signing_material", lambda *_a, **_k: signing
    )
    monkeypatch.setattr(
        staging_deploy,
        "source_deploy_identity",
        lambda *_a, **_k: {
            "application": {"sha256": "a" * 64},
            "deploy_host": {"sha256": "b" * 64, "bundle": {"sha256": "c" * 64}},
        },
    )
    monkeypatch.setattr(
        staging_deploy, "load_successful_source_deploy", lambda *_a, **_k: previous
    )
    monkeypatch.setattr(staging_deploy, "site_operation_lock", lambda *_a, **_k: Lock())
    monkeypatch.setattr(
        staging_deploy,
        "precheck_email_confirmations",
        lambda **_k: events.append("email-check") or _email(ses=True, sns="CONFIRMED"),
    )
    monkeypatch.setattr(
        staging_deploy,
        "read_live_status",
        lambda **_k: events.append("status") or report,
    )
    monkeypatch.setattr(
        staging_deploy,
        "collect_live_deploy_evidence",
        lambda **_k: (_ for _ in ()).throw(
            staging_deploy.LiveEvidenceError("not noop")
        ),
    )
    for name in (
        "restore_trusted_ci_candidate",
        "deploy_host_artifacts",
        "ensure_deploy_host_bundle",
        "ensure_deploy_host_venv",
        "run_source_impact_gate",
        "run_admin_deploy",
        "apply_source_deploy",
    ):
        monkeypatch.setattr(
            staging_deploy,
            name,
            lambda *_a, _name=name, **_k: pytest.fail(f"{_name} ran past the refusal"),
        )
    (state / "deployer-venv/bin").mkdir(parents=True, exist_ok=True)
    (state / "deployer-venv/bin/gpu-fault-admin").write_text("", encoding="utf-8")


def _failed_report(*, phase: str = "failed") -> dict[str, object]:
    return {
        "mode": "status",
        "healthy": False,
        "health_scope": "quick",
        "live_release": {"release_id": "release-old", "phase": phase},
        "configured_release": {
            "release_id": "release-old",
            "database_schema_version": 12,
        },
        "next_deploy": {
            "kind": "FULL",
            "changed": ["control_plane_wheel"],
            "resume": True,
        },
    }


def _noop_report(*, schema_version: int = 12) -> dict[str, object]:
    return {
        "mode": "status",
        "healthy": True,
        "health_scope": "quick",
        "live_release": {"release_id": "release-old", "phase": "complete"},
        "configured_release": {
            "release_id": "release-old",
            "database_schema_version": schema_version,
        },
        "next_deploy": {"kind": "NOOP", "changed": []},
    }


def test_email_precheck_stops_before_the_source_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both addresses in one message, exit status 2, nothing scanned or built."""

    state = tmp_path / "state"
    _write_site(state)
    events: list[str] = []
    _stub_until_first_minute(
        monkeypatch, state=state, events=events, report=None, manifest=None
    )
    pending = _email(ses=False, sns="PENDING")

    def precheck(**kwargs: object) -> EmailConfirmation:
        events.append("email-check")
        raise staging_deploy.StagingDeployError(
            email_confirmation_refusal(
                pending, rerun_command=str(kwargs["rerun_command"])
            )
        )

    monkeypatch.setattr(staging_deploy, "precheck_email_confirmations", precheck)

    status = staging_deploy.main(
        ["--state-dir", str(state), "--repo-root", str(tmp_path / "repo"), "--quiet"]
    )

    assert status == 2
    assert events == ["email-check"], "the refusal comes before the source scan"
    err = capsys.readouterr().err
    assert "SES sender identity operations@example.com" in err
    assert "SNS alert subscription operations@example.com" in err
    assert f"rerun: gpu-fault-admin deploy --state-dir {state}" in err
    assert "--wait-for-email-confirmation" in err


def test_email_precheck_forwards_the_wait_and_the_rerun_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    _write_site(state)
    events: list[str] = []
    seen: dict[str, object] = {}
    _stub_until_first_minute(
        monkeypatch, state=state, events=events, report=None, manifest=None
    )

    def precheck(**kwargs: object) -> EmailConfirmation:
        seen.update(kwargs)
        raise staging_deploy.StagingDeployError("stop here")

    monkeypatch.setattr(staging_deploy, "precheck_email_confirmations", precheck)

    with pytest.raises(staging_deploy.StagingDeployError, match="stop here"):
        staging_deploy.deploy(_arguments(state, wait_for_email_confirmation=15))

    assert seen["wait_minutes"] == 15
    assert seen["admin_email"] == EMAIL
    assert seen["cpu_cluster_arn"] == CPU_ARN
    assert seen["rerun_command"] == f"gpu-fault-admin deploy --state-dir {state}"


def test_early_supersede_refusal_happens_before_gates_and_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed transaction and a different, already-built candidate: refused
    right after the quick status with the engine's wording; the flag lets it on."""

    state = tmp_path / "state"
    _write_site(state)
    events: list[str] = []
    _stub_until_first_minute(
        monkeypatch,
        state=state,
        events=events,
        report=_failed_report(),
        manifest={"release_id": "release-new", "database_schema_version": 12},
    )
    monkeypatch.delenv(SUPERSEDE_FAILED_TRANSACTION_ENV, raising=False)

    with pytest.raises(staging_deploy.StagingDeployError) as failure:
        staging_deploy.deploy(_arguments(state))

    message = str(failure.value)
    assert "release release-old stopped in phase failed" in message
    assert "release-new" in message
    assert SUPERSEDE_FAILED_TRANSACTION_FLAG in message
    assert events[-2:] == ["status", "lock-exit"], (
        "refused straight after the pre-deploy reading, under the lock"
    )

    monkeypatch.setenv(SUPERSEDE_FAILED_TRANSACTION_ENV, "1")
    events.clear()
    monkeypatch.setattr(
        staging_deploy,
        "restore_trusted_ci_candidate",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("past the refusal")),
    )
    with pytest.raises(RuntimeError, match="past the refusal"):
        staging_deploy.deploy(_arguments(state))


def test_early_schema_refusal_happens_before_gates_and_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    _write_site(state)
    events: list[str] = []
    _stub_until_first_minute(
        monkeypatch,
        state=state,
        events=events,
        report=_noop_report(schema_version=12),
        manifest={"release_id": "release-new", "database_schema_version": 13},
    )
    monkeypatch.delenv(ACCEPT_SCHEMA_CHANGE_ENV, raising=False)

    with pytest.raises(staging_deploy.StagingDeployError) as failure:
        staging_deploy.deploy(_arguments(state))

    assert "cannot be rolled back" in str(failure.value)
    assert "--accept-schema-change" in str(failure.value)
    assert events[-2:] == ["status", "lock-exit"]

    monkeypatch.setenv(ACCEPT_SCHEMA_CHANGE_ENV, "snapshot")
    monkeypatch.setattr(
        staging_deploy,
        "restore_trusted_ci_candidate",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("past the refusal")),
    )
    with pytest.raises(RuntimeError, match="past the refusal"):
        staging_deploy.deploy(_arguments(state))


def test_no_refusal_without_a_built_candidate_or_a_failed_live_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No manifest in the snapshot: the inner hop decides after the build. A
    healthy NOOP site with the same schema version: nothing to refuse."""

    state = tmp_path / "state"
    _write_site(state)
    events: list[str] = []
    _stub_until_first_minute(
        monkeypatch, state=state, events=events, report=_failed_report(), manifest=None
    )
    monkeypatch.setattr(
        staging_deploy,
        "restore_trusted_ci_candidate",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("past the refusal")),
    )

    with pytest.raises(RuntimeError, match="past the refusal"):
        staging_deploy.deploy(_arguments(state))

    source = staging_deploy.SourceCheckout(
        repository_root=tmp_path / "snapshot",
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    (tmp_path / "snapshot/dist").mkdir(parents=True, exist_ok=True)
    (tmp_path / "snapshot/dist/current-release.json").write_text(
        json.dumps({"release_id": "release-old", "database_schema_version": 12}),
        encoding="utf-8",
    )
    assert (
        staging_deploy.early_consent_refusal(
            _noop_report(), source=source, auto_rollback=True, environment={}
        )
        is None
    )
