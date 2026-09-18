"""The BOOT-011..018 entrypoint records every failure, including interrupts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import run_boot_acceptance as boot
from scripts.e2e.regional.boot_acceptance_common import BootAcceptanceError


def test_failure_outcome_marks_interrupts_and_keeps_frames() -> None:
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt as exc:
        interrupted = boot.failure_outcome(exc)
    try:
        raise BootAcceptanceError("probe returned nothing")
    except BootAcceptanceError as exc:
        failed = boot.failure_outcome(exc)

    assert interrupted["verdict"] == "FAIL"
    assert interrupted["interrupted"] is True
    assert "interrupted" in interrupted["limitations"][0]
    assert failed["interrupted"] is False
    assert failed["error"] == "BootAcceptanceError: probe returned nothing"
    assert any("BootAcceptanceError" in line for line in failed["traceback"]), failed[
        "traceback"
    ]


def test_main_handles_base_exceptions_and_reraises_after_writing(
    tmp_path, monkeypatch
) -> None:
    arguments = boot.parser().parse_args(
        [
            "--case",
            "GF-REGIONAL-BOOT-013",
            "--site",
            str(tmp_path / "synthetic-site"),
            "--run-dir",
            str(tmp_path),
            "--execute",
            "--confirm",
            "BOOT013_EXECUTE",
        ]
    )
    monkeypatch.setattr(
        boot, "parser", lambda: SimpleNamespace(parse_args=lambda: arguments)
    )
    monkeypatch.setattr(boot, "install_site_profile", lambda: None)
    monkeypatch.setattr(boot.os, "umask", lambda _mask: 0o077)
    monkeypatch.setattr(
        boot, "evidence_identity_for", lambda _a: {"release_id": "r", "cluster_id": "c"}
    )
    monkeypatch.setattr(boot, "predecessor_path", lambda *_a: (None, None))
    monkeypatch.setattr(boot, "authorize_execution", lambda *_a, **_k: None)
    monkeypatch.setattr(boot, "SiteFixture", lambda *_a: object())

    def interrupted(_fixture):
        raise KeyboardInterrupt

    monkeypatch.setattr(boot, "run_boot013", interrupted)

    with pytest.raises(KeyboardInterrupt):
        boot.main()
    path = tmp_path / "cases/GF-REGIONAL-BOOT-013/GF-REGIONAL-BOOT-013.json"
    result = json.loads(path.read_text())
    assert result["verdict"] == "FAIL", (
        "interruption must be recorded before re-raising"
    )
    assert result["interrupted"] is True, (
        "an interrupt is distinct from an assertion failure"
    )


def test_boot021_requires_its_own_case_and_refuses_early_recording() -> None:
    parser = boot.parser()
    arguments = parser.parse_args(
        [
            "--case",
            "GF-REGIONAL-BOOT-021",
            "--site",
            "/tmp/site.yaml",
            "--run-dir",
            "/tmp/run",
        ]
    )
    boot.validate_arguments(arguments)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--case",
                "GF-REGIONAL-BOOT-012",
                "--site",
                "/tmp/site.yaml",
                "--run-dir",
                "/tmp/run",
                "--also-record-boot021",
            ]
        )


def test_formal_boot018_must_retain_the_regional_fixture(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_ACCEPTANCE_EXECUTION_SCOPE", "formal")
    arguments = boot.parser().parse_args(
        [
            "--case",
            "GF-REGIONAL-BOOT-018",
            "--bootstrap-state-dir",
            "/tmp/state",
            "--run-dir",
            "/tmp/run",
        ]
    )
    with pytest.raises(BootAcceptanceError, match="retain-bootstrap-site"):
        boot.validate_arguments(arguments)
    arguments.retain_bootstrap_site = True
    boot.validate_arguments(arguments)


def test_boot021_entrypoint_records_only_its_current_canonical_case(
    tmp_path, monkeypatch
) -> None:
    arguments = boot.parser().parse_args(
        [
            "--case",
            "GF-REGIONAL-BOOT-021",
            "--site",
            str(tmp_path / "synthetic-site"),
            "--run-dir",
            str(tmp_path),
            "--execute",
            "--confirm",
            "BOOT021_EXECUTE",
        ]
    )
    monkeypatch.setattr(
        boot, "parser", lambda: SimpleNamespace(parse_args=lambda: arguments)
    )
    monkeypatch.setattr(boot, "install_site_profile", lambda: None)
    monkeypatch.setattr(
        boot,
        "evidence_identity_for",
        lambda _a: {"release_id": "full", "cluster_id": "a"},
    )
    monkeypatch.setattr(
        boot,
        "predecessor_path",
        lambda *_a: ("GF-REGIONAL-BOOT-020", tmp_path / "previous"),
    )
    monkeypatch.setattr(boot, "predecessor_evidence", lambda *_a, **_k: {"valid": True})
    monkeypatch.setattr(boot, "authorize_execution", lambda *_a, **_k: None)
    monkeypatch.setattr(boot, "SiteFixture", lambda *_a: object())
    monkeypatch.setattr(
        boot, "run_boot021", lambda _f: {"verdict": "PASS", "checks": {"matrix": True}}
    )

    assert boot.main() == 0, (
        "standalone BOOT-021 should complete after its guarded predecessor"
    )
    path = tmp_path / "cases/GF-REGIONAL-BOOT-021/GF-REGIONAL-BOOT-021.json"
    document = json.loads(path.read_text())
    assert document["case_id"] == "GF-REGIONAL-BOOT-021", (
        "only the executed case may be recorded"
    )
    assert document["release_id"] == "full" and document["cluster_id"] == "a", (
        "the current post-BOOT020 release/cluster must bind its evidence"
    )
    assert not (tmp_path / "cases/GF-REGIONAL-BOOT-012").exists(), (
        "no other case is synthesized"
    )


def test_evidence_identity_fails_closed(tmp_path, monkeypatch) -> None:
    class Broken:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("no kubeconfig")

    monkeypatch.setattr(boot, "SiteFixture", Broken)
    arguments = SimpleNamespace(
        case="GF-REGIONAL-BOOT-012",
        site=tmp_path / "site.yaml",
        bootstrap_state_dir=None,
        cluster_id="",
    )
    with pytest.raises(BootAcceptanceError, match="cannot bind"):
        boot.evidence_identity_for(arguments)

    lifecycle = SimpleNamespace(
        case="GF-REGIONAL-BOOT-017",
        site=None,
        bootstrap_state_dir=tmp_path,
        cluster_id="",
    )
    with pytest.raises(BootAcceptanceError, match="cannot bind"):
        boot.evidence_identity_for(lifecycle)
    lifecycle.case = "GF-REGIONAL-BOOT-016"
    assert boot.evidence_identity_for(lifecycle) is None, (
        "first bootstrap has no live site"
    )


@pytest.mark.parametrize("valid", [True, False, "false", None])
def test_plan_passes_namespace_and_exact_preflight_boolean(
    valid, tmp_path, monkeypatch
) -> None:
    arguments = boot.parser().parse_args(
        [
            "--case",
            "GF-REGIONAL-BOOT-013",
            "--site",
            str(tmp_path / "synthetic-site"),
            "--run-dir",
            str(tmp_path),
        ]
    )
    monkeypatch.setattr(
        boot, "parser", lambda: SimpleNamespace(parse_args=lambda: arguments)
    )
    monkeypatch.setattr(boot, "install_site_profile", lambda: None)
    monkeypatch.setattr(boot.os, "umask", lambda _mask: 0o077)
    monkeypatch.setattr(
        boot, "evidence_identity_for", lambda _a: {"release_id": "r", "cluster_id": "c"}
    )
    monkeypatch.setattr(
        boot,
        "predecessor_path",
        lambda *_a: ("GF-REGIONAL-BOOT-012", tmp_path / "predecessor"),
    )
    monkeypatch.setattr(
        boot, "predecessor_evidence", lambda *_a, **_k: {"valid": valid}
    )
    captured = {}

    def plan(**kwargs):
        captured.update(kwargs)
        return {"schema_version": 3}

    monkeypatch.setattr(boot, "build_plan", plan)

    assert boot.main() == (0 if valid is True else 1), (
        "unknown/failed proof must not pass"
    )
    assert captured["arguments"] is arguments, (
        "the parsed target arguments must be bound"
    )
    assert captured["preflight_passed"] is (valid is True), (
        "preflight must be an exact bool"
    )
