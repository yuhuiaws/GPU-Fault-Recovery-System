"""The executor readiness matrix must keep failing under ``python -O``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.e2e.regional.audit_executor_readiness import (
    ReadinessMatrixError,
    validate_readiness_matrix,
)

ROOT = Path(__file__).resolve().parents[2]


def _matrix() -> dict:
    artifact = "a" * 64
    wrong = "0" * 64
    base = {
        "ready": True,
        "reasons": [],
        "unsupported_execution_owners": [],
        "execution_owners": ["gpu-fault-kubernetes-adapter"],
        "executor_artifact_sha256": artifact,
        "last_successful_claim_age_seconds": 0,
    }
    return {
        "valid": (200, base),
        "wrong_token": (403, {"detail": "regional cluster authentication failed"}),
        "wrong_pin": (
            503,
            {
                **base,
                "ready": False,
                "executor_artifact_sha256": wrong,
                "reasons": [
                    f"regional executor artifact mismatch: expected {artifact}, got {wrong}"
                ],
            },
        ),
        "no_owner": (
            503,
            {
                **base,
                "ready": False,
                "execution_owners": [],
                "reasons": [
                    "executor advertised no execution owners, so it can claim nothing"
                ],
            },
        ),
        "stale": (
            503,
            {
                **base,
                "ready": False,
                "last_successful_claim_age_seconds": 86400,
                "reasons": ["last successful claim was 86400s ago (limit 900s)"],
            },
        ),
        "wrong_artifact": wrong,
        "stale_age_seconds": 86400,
    }


def test_matrix_errors_are_explicit_assertion_errors() -> None:
    matrix = _matrix()
    status, payload = matrix["stale"]
    matrix["stale"] = (status, {**payload, "reasons": ["executor is unavailable"]})

    with pytest.raises(ReadinessMatrixError) as raised:
        validate_readiness_matrix(**matrix)
    assert isinstance(raised.value, AssertionError), (
        "ReadinessMatrixError stays an AssertionError so the audit's catch-all keeps treating it as a failed check"
    )
    assert "stale claim reason" in str(raised.value)


def test_matrix_still_fails_under_python_optimisation() -> None:
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from scripts.e2e.regional.audit_executor_readiness import validate_readiness_matrix;"
        "from tests.regional.test_audit_executor_readiness import _matrix;"
        "m = _matrix(); m['wrong_token'] = (200, {});"
        "validate_readiness_matrix(**m)"
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script, str(ROOT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "ReadinessMatrixError" in completed.stderr
    assert "wrong token must be 403" in completed.stderr
