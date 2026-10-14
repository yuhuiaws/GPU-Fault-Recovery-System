"""Named, timed phases in the ``gpu-fault-admin config`` log.

Live 2026-10-14 the first ``config`` of every perf suite spent 13.4 minutes
before its release-diff with four command lines in its log; the time was the
Aurora settle loop (RDS raising the floor from 8 to 124 ACU), whose reads print
nothing. Each step now opens with a begin line and closes with its seconds.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import operator_identity, release_state
from gpu_fault.admin.command_log import ADMIN_PHASE_BEGIN, ADMIN_PHASE_END, admin_phase
from gpu_fault.admin.config_file import initialize_desired_admin_config
from tests.admin.test_admin_site import mock_live_release, site_file
from tests.admin.test_admin_site_config import AURORA_16_64, Stubs

PHASE_LINE = re.compile(
    r"^(?P<kind>admin-phase-begin|admin-phase-end) "
    r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ "
    r"command=(?P<command>\S+) phase=(?P<phase>\S+)"
    r"(?: status=(?P<status>ok|failed)(?: exit_code=(?P<exit_code>\d+))?"
    r" elapsed=(?P<elapsed>\d+\.\d)s)?$"
)
BEFORE_RELEASE = ["release-identity", "attestation", "live-release", "apply-record"]


@pytest.fixture(autouse=True)
def fixed_approver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        operator_identity,
        "caller_identity_arn",
        lambda **_kwargs: "arn:aws:sts::123456789012:assumed-role/Admin/alice",
    )


def phase_lines(stderr: str) -> list[re.Match[str]]:
    """Every phase line, each checked against the documented shape."""

    lines = [
        line
        for line in stderr.splitlines()
        if line.startswith((ADMIN_PHASE_BEGIN, ADMIN_PHASE_END))
    ]
    matches = [PHASE_LINE.match(line) for line in lines]
    assert all(matches), f"malformed phase line in {lines}"
    return [match for match in matches if match is not None]


def closed_phases(stderr: str) -> list[tuple[str, str]]:
    """``(phase, status)`` per phase; every begin is closed before the next opens."""

    result = []
    lines = phase_lines(stderr)
    assert len(lines) % 2 == 0, f"a phase was left open: {lines}"
    for begin, end in zip(lines[::2], lines[1::2], strict=True):
        assert (begin["kind"], end["kind"]) == (ADMIN_PHASE_BEGIN, ADMIN_PHASE_END)
        assert begin["phase"] == end["phase"] and end["command"] == "config"
        assert float(end["elapsed"]) >= 0
        result.append((end["phase"], end["status"]))
    return result


def config_file(path: Path, spec: dict[str, object]) -> Path:
    document = {"apiVersion": "gpu-fault.aws/v1alpha1", "kind": "AdminConfig"}
    path.write_text(yaml.safe_dump({**document, "spec": spec}), encoding="utf-8")
    path.chmod(0o600)
    return path


def apply(tmp_path: Path, spec: dict[str, object]) -> int:
    config = config_file(tmp_path / "admin-input.yaml", spec)
    return admin_cli.run(
        admin_cli.parser().parse_args(
            ["config", "--state-dir", str(tmp_path), "--file", str(config)]
        )
    )


def test_a_phase_prints_begin_and_end_with_its_seconds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with admin_phase("config", "attestation") as phase:
        assert phase.exit_code is None

    lines = phase_lines(capsys.readouterr().err)
    assert [(line["kind"], line["phase"]) for line in lines] == [
        (ADMIN_PHASE_BEGIN, "attestation"),
        (ADMIN_PHASE_END, "attestation"),
    ]
    assert lines[1]["status"] == "ok" and lines[1]["exit_code"] is None


def test_a_raising_phase_is_closed_as_failed_and_the_error_propagates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(RuntimeError, match="RDS throttled"):
        with admin_phase("config", "aurora-request"):
            raise RuntimeError("RDS throttled")

    end = phase_lines(capsys.readouterr().err)[-1]
    assert (end["phase"], end["status"]) == ("aurora-request", "failed")


def test_a_child_exit_status_closes_the_phase_as_failed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with admin_phase("release-deploy", "prepare") as phase:
        phase.exit_code = 7

    end = phase_lines(capsys.readouterr().err)[-1]
    assert (end["command"], end["status"], end["exit_code"]) == (
        "release-deploy",
        "failed",
        "7",
    )


def test_an_aurora_scale_up_apply_times_every_step_around_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_file(tmp_path)
    mock_live_release(monkeypatch)
    initialize_desired_admin_config(tmp_path)
    stubs = Stubs(monkeypatch)

    assert apply(tmp_path, AURORA_16_64) == 0

    assert closed_phases(capsys.readouterr().err) == [
        (name, "ok")
        for name in [
            *BEFORE_RELEASE,
            "aurora-request",
            "release-deploy",
            "aurora-await",
            "apply-complete",
        ]
    ]
    assert stubs.events == ["request", "release", "await"]
    # The request already settled this change; the floor needs one observation.
    assert stubs.calls["await"][0]["window_settled"] is True


def test_an_apply_that_leaves_aurora_alone_verifies_it_and_awaits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_file(tmp_path)
    mock_live_release(monkeypatch)
    initialize_desired_admin_config(tmp_path)
    stubs = Stubs(monkeypatch)

    assert apply(tmp_path, {"workflow": {"dispatcherWorkers": 9}}) == 0

    assert closed_phases(capsys.readouterr().err) == [
        (name, "ok")
        for name in [
            *BEFORE_RELEASE,
            "aurora-verify",
            "release-deploy",
            "apply-complete",
        ]
    ]
    assert stubs.events == ["verify", "release"]


def test_a_failed_release_closes_its_phase_with_the_child_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_file(tmp_path)
    before = initialize_desired_admin_config(tmp_path)
    monkeypatch.setattr(
        admin_cli, "verify_prebuilt_release", lambda *_args, **_kwargs: None
    )
    live = {
        "release_id": "release-a",
        "phase": "complete",
        "transaction_committed": True,
        "release_lifecycle": "COMMITTED",
        "admin_config": before.as_dict(),
        "admin_config_sha256": before.sha256(),
        "admin_config_role_sha256": before.role_sha256(),
    }
    monkeypatch.setattr(
        release_state,
        "run_command",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, json.dumps({"data": {"state.json": json.dumps(live)}}), ""
        ),
    )
    Stubs(monkeypatch, release_returncode=7)

    assert apply(tmp_path, AURORA_16_64) == 7

    phases = phase_lines(capsys.readouterr().err)
    release_end = [
        line
        for line in phases
        if line["phase"] == "release-deploy" and line["kind"] == ADMIN_PHASE_END
    ]
    assert [(line["status"], line["exit_code"]) for line in release_end] == [
        ("failed", "7")
    ]
    assert not {"aurora-await", "apply-complete"} & {line["phase"] for line in phases}
