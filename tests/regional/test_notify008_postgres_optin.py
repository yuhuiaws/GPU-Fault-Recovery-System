from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpu_fault.admin.postgres_grant import ALLOCATION_ENV, POSTGRES_URL_ENV
from tests.regional import _cov95_notify008_postgres as native

ROOT = Path(__file__).resolve().parents[2]
NODEID = (
    "tests/regional/test_cov95_notify008_postgres.py::"
    "test_postgres_runtime_process_loss_and_independent_acceptance"
    "[before-provider-gpu-reset-inline]"
)


@pytest.mark.parametrize("explicit", [False, True])
def test_notify008_native_execution_requires_an_explicit_url_and_private_grant(
    tmp_path: Path, explicit: bool
) -> None:
    environment = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "AWS_CONFIG_FILE": os.devnull,
        "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
        "AWS_EC2_METADATA_DISABLED": "true",
        "KUBECONFIG": os.devnull,
        ALLOCATION_ENV: str(tmp_path / "absent-grant"),
        POSTGRES_URL_ENV: "postgresql://127.0.0.1:1/explicit-test" if explicit else "",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "-p",
            "xdist.plugin",
            "-n",
            "0",
            "--tb=short",
            NODEID,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    output = result.stdout + result.stderr
    if explicit:
        assert result.returncode == 1, output
        assert "allocation files are missing or invalid" in output, (
            "explicit native selection must reach the unchanged private-grant guard"
        )
        assert "1 skipped" not in output, (
            "a configured native failure cannot be skipped"
        )
    else:
        assert result.returncode == 0 and "1 skipped" in output, output
        assert "allocation files are missing or invalid" not in output, (
            "ordinary pytest must not attempt to consume an inherited native grant"
        )


def test_notify008_parallel_native_execution_still_refuses_before_grant_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("parallel native execution reached grant filesystem access")

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw-controlled")
    monkeypatch.setattr(native, "Path", forbidden)
    with pytest.raises(pytest.fail.Exception, match="require explicit serial -n0"):
        native.validated_grant()
