from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.e2e.regional import regional_commands
from tools import pytest_result_identity

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = "a" * 64
BOOTSTRAP = """
import sys
import pytest
from tools import pytest_case_reporter
pytest_case_reporter.source_identity = lambda root: "a" * 64
pytest_case_reporter.repository_root = lambda root: root.resolve()
raise SystemExit(pytest.main(sys.argv[1:]))
"""


@pytest.fixture
def child_transport(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(
        pytest_result_identity, "source_identity", lambda root: IDENTITY
    )

    def execute(command, *, environment, cwd, timeout_seconds, input_text):
        assert input_text is None, "focused pytest must not receive command stdin"
        calls.append((list(command), dict(environment or {})))
        return subprocess.run(
            [sys.executable, "-c", BOOTSTRAP, *command[3:]],
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )

    monkeypatch.setattr(regional_commands, "run_command", execute)
    return calls


def tiny_command(tmp_path: Path, body: str) -> list[str]:
    test = tmp_path / "test_focused.py"
    test.write_text(body, encoding="ascii")
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--noconftest",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "--rootdir",
        str(tmp_path),
        str(test),
    ]


def child_environment() -> dict[str, str]:
    return {
        "HOME": "/tmp",
        "PATH": os.environ["PATH"],
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "AWS_CONFIG_FILE": os.devnull,
        "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
        "AWS_EC2_METADATA_DISABLED": "true",
        "KUBECONFIG": os.devnull,
    }


@pytest.mark.parametrize(
    "body",
    [
        "import pytest\ndef test_skipped():\n    pytest.skip('unavailable fixture')\n",
        "import pytest\n"
        "@pytest.mark.xfail(reason='known failure')\n"
        "def test_expected_failure():\n"
        "    assert False, 'expected failure is not passing evidence'\n",
        "import pytest\n"
        "def test_control():\n"
        "    assert 2 + 2 == 4, 'arithmetic control'\n"
        "def test_skipped():\n"
        "    pytest.skip('unavailable fixture')\n",
    ],
    ids=["all-skipped", "xfail", "mixed"],
)
def test_live_focused_pytest_cannot_pass_from_zero_exit_with_unpassed_tests(
    tmp_path: Path, child_transport, body: str
) -> None:
    result = regional_commands.run_fixture_command(
        tiny_command(tmp_path, body),
        cwd=tmp_path,
        env=child_environment(),
        timeout=30,
        check=False,
    )
    assert result.returncode != 0, (
        "exit zero alone must not authorize the live preflight",
        result.stdout,
    )
    assert child_transport, "the test must execute a real pytest child"


def test_live_focused_pytest_rejects_a_filtered_generic_matrix(
    tmp_path: Path, child_transport
) -> None:
    command = tiny_command(
        tmp_path,
        "import pytest\n"
        "@pytest.mark.parametrize('value', [0, 1], ids=['zero', 'one'])\n"
        "def test_matrix(value):\n"
        "    assert value in {0, 1}, 'matrix member'\n",
    )
    result = regional_commands.run_fixture_command(
        [*command, "-k", "zero"],
        cwd=tmp_path,
        env=child_environment(),
        timeout=30,
        check=False,
    )
    assert result.returncode != 0, (
        "one filtered parameter cannot prove the complete focused matrix",
        result.stdout,
    )


@pytest.mark.parametrize("selector", ["test_matrix", "test_matrix[zero]", "TestGroup"])
def test_live_focused_pytest_accepts_complete_explicit_selectors(
    tmp_path: Path, child_transport, selector: str
) -> None:
    command = tiny_command(
        tmp_path,
        "import pytest\n"
        "@pytest.mark.parametrize('value', [0, 1], ids=['zero', 'one'])\n"
        "def test_matrix(value):\n"
        "    assert value in {0, 1}\n"
        "class TestGroup:\n"
        "    def test_member(self):\n"
        "        assert True\n"
        "def test_unrequested():\n"
        "    pytest.fail('unrequested sibling must not run')\n",
    )
    command[-1] += "::" + selector
    result = regional_commands.run_fixture_command(
        command, cwd=tmp_path, env=child_environment(), timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(child_transport) == 1


@pytest.mark.parametrize("filter_args", [["-k", "zero"], ["--deselect", "ONE"]])
def test_live_focused_pytest_rejects_an_incomplete_explicit_matrix(
    tmp_path: Path, child_transport, filter_args: list[str]
) -> None:
    command = tiny_command(
        tmp_path,
        "import pytest\n"
        "@pytest.mark.parametrize('value', [0, 1], ids=['zero', 'one'])\n"
        "def test_matrix(value):\n"
        "    assert value in {0, 1}\n"
        "def test_sibling():\n"
        "    assert True\n",
    )
    target = command[-1] + "::test_matrix"
    command[-1] = target
    filters = [target + "[one]" if arg == "ONE" else arg for arg in filter_args]
    result = regional_commands.run_fixture_command(
        [*command, *filters],
        cwd=tmp_path,
        env=child_environment(),
        timeout=30,
        check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "filtered or partitioned" in result.stderr


def test_live_focused_pytest_accepts_complete_passes_without_live_credentials(
    tmp_path: Path, child_transport
) -> None:
    environment = child_environment() | {
        "GPU_FAULT_EXECUTION_TOKEN": "unit-placeholder",
        "AWS_ACCESS_KEY_ID": "unit-placeholder",
        "PYTEST_ADDOPTS": "-k absent",
        "PYTEST_GPU_FAULT_PARTITION_COUNT": "2",
        "PYTEST_GPU_FAULT_PARTITION_INDEX": "1",
    }
    result = regional_commands.run_fixture_command(
        tiny_command(
            tmp_path,
            "def test_control():\n    assert 2 + 2 == 4, 'arithmetic control'\n",
        ),
        cwd=tmp_path,
        env=environment,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _, sent = child_transport[0]
    assert "GPU_FAULT_EXECUTION_TOKEN" not in sent, (
        "focused unit tests must not receive the live execution token"
    )
    assert "AWS_ACCESS_KEY_ID" not in sent, "live AWS credentials must not be inherited"
    assert not sent.get("PYTEST_ADDOPTS"), "inherited filtering must be cleared"
    assert not sent.get("PYTEST_GPU_FAULT_PARTITION_COUNT"), (
        "a deployment-host pytest partition must not truncate the focused matrix"
    )


def test_explicit_isolated_database_reaches_real_pytest_child_without_other_authority(
    tmp_path: Path, child_transport
) -> None:
    url = "postgresql://127.0.0.1:55432/gpu_fault_cap005_0123456789ab"
    environment = child_environment() | {
        "GPU_FAULT_STORE_URL": "postgresql://production.invalid/business",
        "GPU_FAULT_TEST_POSTGRES_URL": "postgresql://foreign.invalid/business",
        "PGHOST": "production.invalid",
        "AWS_PROFILE": "production",
        "PYTEST_ADDOPTS": "-k absent",
        "PYTEST_XDIST_WORKER": "gw9",
        "PYTEST_GPU_FAULT_CASE_REPORT": "/private/foreign-report.json",
    }
    result = regional_commands.run_fixture_command(
        tiny_command(
            tmp_path,
            "import os\n"
            "def test_environment():\n"
            f"    assert os.environ['GPU_FAULT_TEST_POSTGRES_URL'] == {url!r}, 'explicit URL'\n"
            "    assert os.environ['GPU_FAULT_STORE_URL'] == '', 'production disabled'\n"
            "    assert 'PGHOST' not in os.environ, 'no connection override'\n"
            "    assert 'AWS_PROFILE' not in os.environ, 'no cloud authority'\n"
            "    assert 'PYTEST_ADDOPTS' not in os.environ, 'no inherited filter'\n"
            "    assert 'PYTEST_XDIST_WORKER' not in os.environ, 'no inherited worker'\n",
        ),
        cwd=tmp_path,
        env=environment,
        isolated_postgres_url=url,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(child_transport) == 1, (
        "the explicit environment must be verified by a real receipt-emitting child"
    )
    command, sent = child_transport[0]
    assert url not in repr(command) and sent["GPU_FAULT_TEST_POSTGRES_URL"] == url, (
        "the opt-in must remain environment-only through the actual child boundary"
    )
