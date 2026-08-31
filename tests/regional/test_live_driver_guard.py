from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.e2e.regional.live_driver_guard import (
    add_live_arguments,
    authorize_execution,
    build_plan,
)

CASE_ID = "GF-REGIONAL-TEST-001"
CONFIRMATION = "TEST_LIVE_CONFIRMATION"
ENVIRONMENT = {
    "CPU_KUBECONFIG": "/secure/cpu.kubeconfig",
    "GPU_KUBECONFIG": "/secure/gpu.kubeconfig",
    "GPU_EKS_CONTEXT": "gpu-context",
}


def _arguments(
    run_dir: Path, *, confirm: str = CONFIRMATION, deadline: datetime | None = None
) -> argparse.Namespace:
    return argparse.Namespace(
        execute=True,
        confirm=confirm,
        maintenance_window_end=(
            deadline or datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(),
        run_dir=run_dir,
        attempt=1,
    )


def test_live_driver_defaults_to_non_execute_mode(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_live_arguments(parser, confirmation=CONFIRMATION)

    arguments = parser.parse_args(["--run-dir", str(tmp_path)])

    assert arguments.execute is False
    assert arguments.plan is False


def test_live_driver_authorizes_matching_plan(tmp_path: Path) -> None:
    build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={"mutation": "test-only"},
    )

    deadline = authorize_execution(
        _arguments(tmp_path),
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
    )

    assert deadline > datetime.now(timezone.utc)


def test_live_driver_rejects_confirmation_and_environment_drift(tmp_path: Path) -> None:
    build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={},
    )

    with pytest.raises(RuntimeError, match="confirmation"):
        authorize_execution(
            _arguments(tmp_path, confirm="wrong"),
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            environment=ENVIRONMENT,
        )
    with pytest.raises(RuntimeError, match="environment"):
        authorize_execution(
            _arguments(tmp_path),
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            environment={**ENVIRONMENT, "GPU_EKS_CONTEXT": "drifted"},
        )


def test_live_driver_rejects_expired_window(tmp_path: Path) -> None:
    build_plan(
        run_dir=tmp_path,
        case_id=CASE_ID,
        attempt=1,
        confirmation=CONFIRMATION,
        environment=ENVIRONMENT,
        details={},
    )

    with pytest.raises(RuntimeError, match="window has ended"):
        authorize_execution(
            _arguments(
                tmp_path, deadline=datetime.now(timezone.utc) - timedelta(seconds=1)
            ),
            case_id=CASE_ID,
            confirmation=CONFIRMATION,
            environment=ENVIRONMENT,
        )
