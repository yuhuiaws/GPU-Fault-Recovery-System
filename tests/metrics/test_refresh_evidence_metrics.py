"""Configured refresh evidence is unknown on read, shape or clock failures."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.app.aurora_refresh_metrics import (
    AURORA_REFRESH_STATUS_FILENAME,
    aurora_credential_refresh_metric_lines,
    aurora_refresh_status_path,
)
from gpu_fault.app.process_metrics import parse_lines

UNKNOWN = "gpu_fault_aurora_credential_refresh_status_unreadable"
OK = "gpu_fault_aurora_credential_refresh_last_run_ok"
AGE = "gpu_fault_aurora_credential_refresh_last_success_age_seconds"


def test_default_status_path_is_metadata_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    assert str(aurora_refresh_status_path()) == (
        "/etc/gpu-fault/aurora/last-refresh-status.json"
    )


def values() -> dict[str, str]:
    lines = aurora_credential_refresh_metric_lines(SimpleNamespace())
    return {
        sample.name: sample.value
        for group in parse_lines(lines).samples.values()
        for sample in group
    }


@pytest.mark.parametrize("configured", ["", " ", None])
def test_unconfigured_roles_do_not_read_status_or_raise_refresh_alarms(
    monkeypatch: pytest.MonkeyPatch, configured: str | None
) -> None:
    if configured is None:
        monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    else:
        monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", configured)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an unconfigured role must not read a status file")

    monkeypatch.setattr(Path, "read_text", forbidden)
    assert values() == {}


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError, OSError])
def test_configured_read_failures_are_unknown_without_exposing_the_error(
    monkeypatch: pytest.MonkeyPatch, failure: type[OSError]
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", "/unused-test/status-dsn")

    def broken(*_args, **_kwargs):
        raise failure("private status content must not be rendered")

    monkeypatch.setattr(Path, "read_text", broken)
    assert values() == {UNKNOWN: "1"}


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b"{",
        b"[]",
        b"null",
        b'{"status":"ok"}',
        b'{"status":"ok","finished_at":"invalid"}',
        b'{"status":"ok","finished_at":42}',
        b'{"status":"ok","finished_at":"2080-01-01T00:00:00Z"}',
        b'{"status":"ok","finished_at":"2080-01-01T00:00:00"}',
        b'{"status":"unrecognized","finished_at":"2026-01-01T00:00:00Z"}',
        b'{"finished_at":"2026-01-01T00:00:00Z"}',
    ],
)
def test_bad_status_evidence_cannot_become_a_zero_age_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "not-a-dsn"))
    (tmp_path / AURORA_REFRESH_STATUS_FILENAME).write_bytes(payload)
    assert values() == {UNKNOWN: "1"}


def test_unknown_evidence_recovers_to_a_real_success_and_then_a_failed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "not-a-dsn"))
    path = tmp_path / AURORA_REFRESH_STATUS_FILENAME
    assert values() == {UNKNOWN: "1"}
    finished = datetime.now(timezone.utc) - timedelta(seconds=30)
    path.write_text(json.dumps({"status": "ok", "finished_at": finished.isoformat()}))
    recovered = values()
    assert recovered[UNKNOWN] == "0"
    assert recovered[OK] == "1"
    assert 30 <= float(recovered[AGE]) < 60
    path.write_text(
        json.dumps({"status": "failed", "finished_at": finished.isoformat()})
    )
    failed = values()
    assert failed[UNKNOWN] == "0"
    assert failed[OK] == "0"
    assert AGE not in failed
