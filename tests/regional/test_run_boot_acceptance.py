"""The BOOT-011..018 entrypoint records every failure, including interrupts."""

from __future__ import annotations

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


def test_main_handles_base_exceptions_and_reraises_after_writing() -> None:
    source = boot.__file__
    text = open(source, encoding="utf-8").read()
    assert "except BaseException as exc:" in text
    assert "except Exception as exc:" not in text
    assert "raise interrupted" in text


def test_boot021_recording_is_a_boot012_option() -> None:
    parser = boot.parser()
    arguments = parser.parse_args(
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
    boot.validate_arguments(arguments)

    other = parser.parse_args(
        [
            "--case",
            "GF-REGIONAL-BOOT-013",
            "--site",
            "/tmp/site.yaml",
            "--run-dir",
            "/tmp/run",
            "--also-record-boot021",
        ]
    )
    with pytest.raises(BootAcceptanceError, match="BOOT-012 only"):
        boot.validate_arguments(other)


def test_evidence_identity_is_best_effort(tmp_path, monkeypatch) -> None:
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
    assert boot.evidence_identity_for(arguments) is None

    lifecycle = SimpleNamespace(
        case="GF-REGIONAL-BOOT-017",
        site=None,
        bootstrap_state_dir=tmp_path,
        cluster_id="",
    )
    assert boot.evidence_identity_for(lifecycle) is None, "no site.yaml yet"
