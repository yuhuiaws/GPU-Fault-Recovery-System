from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpu_fault.admin.postgres_grant import ALLOCATION_ENV, POSTGRES_URL_ENV

ROOT = Path(__file__).resolve().parents[1]


def test_kubeconfig_requires_explicit_test_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert "KUBECONFIG" not in os.environ, "tests must not inherit a deploy kubeconfig"
    configured = str(tmp_path / "explicit.kubeconfig")
    monkeypatch.setenv("KUBECONFIG", configured)
    assert os.environ["KUBECONFIG"] == configured, (
        "an explicit test configuration must remain available"
    )


def test_deploy_repository_requires_explicit_test_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name = "GPU_FAULT_REPOSITORY_ROOT"
    assert name not in os.environ, (
        "tests must not inherit the deploy repository locator"
    )
    monkeypatch.setenv(name, str(tmp_path))
    assert os.environ[name] == str(tmp_path), (
        "an installed-command fixture may explicitly select its repository"
    )


@pytest.mark.parametrize(
    "postgres_url",
    [None, "", " ", "postgresql://127.0.0.1:1/explicit-test"],
    ids=["unset", "empty", "blank", "explicit"],
)
def test_deploy_input_fixture_preserves_only_explicit_native_allocations(
    tmp_path: Path, postgres_url: str | None
) -> None:
    (tmp_path / "conftest.py").write_text(
        "from tests.conftest import deploy_inputs_are_never_ambient\n", encoding="utf-8"
    )
    selected = bool(postgres_url and postgres_url.strip())
    (tmp_path / "test_inputs.py").write_text(
        "import os\n"
        "from gpu_fault.admin.postgres_grant import ALLOCATION_ENV\n"
        "def test_inputs(monkeypatch):\n"
        "    assert 'GPU_FAULT_REPOSITORY_ROOT' not in os.environ\n"
        f"    assert (ALLOCATION_ENV in os.environ) is {selected!r}\n"
        "    monkeypatch.setenv(ALLOCATION_ENV, 'explicit-unit-fixture')\n"
        "    assert os.environ[ALLOCATION_ENV] == 'explicit-unit-fixture'\n",
        encoding="utf-8",
    )
    environment = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src"))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "AWS_CONFIG_FILE": os.devnull,
        "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
        "AWS_EC2_METADATA_DISABLED": "true",
        "KUBECONFIG": os.devnull,
        "GPU_FAULT_REPOSITORY_ROOT": str(tmp_path / "deployment"),
        ALLOCATION_ENV: str(tmp_path / "inherited-allocation"),
    }
    if postgres_url is not None:
        environment[POSTGRES_URL_ENV] = postgres_url
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
            "test_inputs.py",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout, (
        "the real fixture must run for both ordinary and explicitly selected PG tests"
    )
