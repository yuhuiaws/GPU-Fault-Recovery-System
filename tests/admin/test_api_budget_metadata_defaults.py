"""A deployment scope gives its children more than one IMDS attempt.

The AWS CLI and boto3 ask the instance metadata service once with a 1 s timeout
before they give up with "Unable to locate credentials"; live on 2026-10-15 one
such miss on the deploy host's instance role failed a verify and rolled a
healthy change back. The scope now sets ``AWS_METADATA_SERVICE_NUM_ATTEMPTS`` and
``AWS_METADATA_SERVICE_TIMEOUT`` for every child -- unless the operator chose a
value, in the environment or in the AWS config file -- and takes them back when
it exits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from gpu_fault.admin import api_budget as budget
from gpu_fault.admin.execution import run_command

NAMES = tuple(budget.METADATA_SERVICE_DEFAULTS)


@pytest.fixture
def clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(budget.ROOT_ENV, raising=False)
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "aws-config"
    config.write_text("[default]\nregion = us-west-2\n", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    return config


def test_the_scope_sets_both_defaults_and_takes_them_back(clean: Path) -> None:
    with budget.deployment_api_budget():
        inside = {name: os.environ.get(name) for name in NAMES}
    assert inside == {
        "AWS_METADATA_SERVICE_NUM_ATTEMPTS": "3",
        "AWS_METADATA_SERVICE_TIMEOUT": "2",
    }
    assert all(name not in os.environ for name in NAMES), (
        "the scope leaked its defaults past its exit"
    )


def test_an_operator_value_in_the_environment_wins(
    clean: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_METADATA_SERVICE_TIMEOUT", "7")
    with budget.deployment_api_budget():
        assert os.environ["AWS_METADATA_SERVICE_TIMEOUT"] == "7"
        assert os.environ["AWS_METADATA_SERVICE_NUM_ATTEMPTS"] == "3"
    assert os.environ["AWS_METADATA_SERVICE_TIMEOUT"] == "7", "an operator value moved"


@pytest.mark.parametrize("section", ["default", "profile deploy"])
def test_an_operator_value_in_the_aws_config_file_wins(
    clean: Path, section: str
) -> None:
    """The environment outranks the file in the CLI: a default would override it."""

    clean.write_text(
        f"[{section}]\nmetadata_service_num_attempts = 5\n", encoding="utf-8"
    )
    assert budget.metadata_service_defaults(dict(os.environ)) == {
        "AWS_METADATA_SERVICE_TIMEOUT": "2"
    }


def test_a_config_file_that_cannot_be_read_adds_nothing(clean: Path) -> None:
    clean.write_text("this is not an ini file\n", encoding="utf-8")
    assert budget.metadata_service_defaults(dict(os.environ)) == {}, (
        "an unreadable choice must not be overridden by a guess"
    )


def test_a_missing_config_file_is_no_operator_choice(
    clean: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent"))
    assert set(budget.metadata_service_defaults(dict(os.environ))) == set(NAMES)


@pytest.mark.allows_cluster_binaries("aws")
def test_an_aws_child_of_the_scope_sees_the_defaults(
    clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "bin" / "aws"
    fake.parent.mkdir()
    fake.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "print(os.environ.get('AWS_METADATA_SERVICE_NUM_ATTEMPTS'),"
        " os.environ.get('AWS_METADATA_SERVICE_TIMEOUT'))\n",
        encoding="utf-8",
    )
    fake.chmod(0o700)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")

    with budget.deployment_api_budget():
        result = run_command(["aws", "--version"])

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["3", "2"], "the child ran without the defaults"
