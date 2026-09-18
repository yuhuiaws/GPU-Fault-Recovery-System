from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from contextlib import nullcontext
from pathlib import Path

import pytest

from gpu_fault.admin import api_budget, execution, rotate_token
from gpu_fault.admin.operation_lock import SITE_OPERATION_LOCK_FD_ENV
from gpu_fault.admin.python_environment import python_environment
from gpu_fault.admin.site import RenderedSite, effective_environment
from gpu_fault_release import REPOSITORY_ROOT_ENV
from scripts import staging_deploy
from tests.admin._cov95_removal_support import RemovalTransport


@pytest.mark.parametrize("inherited", [None, "", "/usr/bin", "/old/bin:/usr/bin"])
def test_python_environment_selects_cli_without_changing_other_inputs(
    tmp_path: Path, inherited: str | None
) -> None:
    original = {"PYTHONPATH": "/checked/source", "KEEP": "unchanged"}
    if inherited is not None:
        original["PATH"] = inherited
    before = dict(original)
    selected = tmp_path / "selected/bin/python"

    result = python_environment(original, executable=selected)

    assert original == before, "environment preparation must not mutate its caller"
    assert result["PATH"].split(os.pathsep)[0] == str(selected.parent)
    assert result["PATH"].split(os.pathsep)[1:] == (
        inherited if inherited is not None else os.defpath
    ).split(os.pathsep)
    assert result["PYTHONPATH"] == original["PYTHONPATH"]
    assert result["KEEP"] == "unchanged"


def test_python_environment_preserves_symlink_entry_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected/bin/python"
    selected.parent.mkdir(parents=True)
    selected.symlink_to(sys.executable)
    monkeypatch.chdir(tmp_path)
    result = python_environment(
        {"PATH": f"{selected.parent}:/usr/bin:{selected.parent}"},
        executable="selected/bin/python",
    )

    assert result["PATH"] == f"{selected.parent}:/usr/bin"
    assert python_environment(result, executable=selected) == result
    assert selected.is_symlink(), "selecting an environment must not rewrite its entry"


@pytest.mark.parametrize("lock_fd", [None, 12])
def test_preflight_handoff_selects_new_python_and_preserves_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_fd: int | None
) -> None:
    selected = tmp_path / "new-venv"
    calls = []
    monkeypatch.setenv("PATH", "/previous/bin:/usr/bin")
    monkeypatch.setenv("PYTHONPATH", "/previous/source")
    monkeypatch.setenv("PYTHONHOME", "/previous/home")
    monkeypatch.setenv(REPOSITORY_ROOT_ENV, "/previous/repository")
    monkeypatch.setattr(
        staging_deploy, "_run", lambda args, **kwargs: calls.append((args, kwargs))
    )

    staging_deploy.run_admin_preflight(
        repository_root=tmp_path,
        state_dir=tmp_path / "state",
        venv=selected,
        lock_fd=lock_fd,
    )

    arguments, options = calls[0]
    assert arguments[:2] == [str(selected / "bin/gpu-fault-admin"), "preflight"]
    assert options["env"]["PATH"] == f"{selected}/bin:/previous/bin:/usr/bin"
    assert options["env"][REPOSITORY_ROOT_ENV] == str(tmp_path)
    assert "PYTHONPATH" not in options["env"]
    assert "PYTHONHOME" not in options["env"]
    assert options["pass_fds"] == ((lock_fd,) if lock_fd is not None else ())
    if lock_fd is not None:
        assert options["env"][SITE_OPERATION_LOCK_FD_ENV] == str(lock_fd)


def test_deploy_handoff_selects_new_python_and_preserves_tool_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        staging_deploy,
        "tool_cache_environment",
        lambda _state: {
            "PATH": "/previous/bin:/usr/bin",
            "MYPY_CACHE_DIR": "/cache",
            "PYTHONPATH": "/previous/source",
        },
    )
    monkeypatch.setattr(
        staging_deploy, "_run", lambda args, **kwargs: calls.append((args, kwargs))
    )
    selected = tmp_path / "new-venv"

    staging_deploy.run_admin_deploy(
        repository_root=tmp_path,
        state_dir=tmp_path / "state",
        venv=selected,
        cpu_cluster_arn="cpu",
        gpu_cluster_arns=["gpu"],
        admin_email="operator@example.invalid",
        staging_only_release=True,
        impact_base="main",
        lock_fd=12,
    )

    arguments, options = calls[0]
    assert arguments[:2] == [str(selected / "bin/gpu-fault-admin"), "deploy"]
    assert options["env"] == {
        "PATH": f"{selected}/bin:/previous/bin:/usr/bin",
        "MYPY_CACHE_DIR": "/cache",
        SITE_OPERATION_LOCK_FD_ENV: "12",
        REPOSITORY_ROOT_ENV: str(tmp_path),
    }
    assert options["pass_fds"] == (12,)


@pytest.mark.parametrize(
    "helper",
    [
        "verify-control-plane-role-split.sh",
        "provision-node-action-keys.sh",
        "cleanup-cluster.sh",
        "rollout-regional-release.sh",
    ],
)
def test_checked_site_commands_keep_python_api_shim_and_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, helper: str
) -> None:
    calls = []
    monkeypatch.setenv("PATH", "/old/bin:/usr/bin")
    site = RenderedSite(
        source=tmp_path / "site.yaml",
        repository_root=tmp_path,
        release_config={},
        environment={"SITE_OPTION": "kept"},
        source_sha256="a" * 64,
    )

    def command(arguments, **options):
        calls.append((arguments, options))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(execution, "run_owned_command", command)
    with api_budget.deployment_api_budget(), execution.deadline_scope("helper", 30):
        environment = effective_environment(site)
        environment[SITE_OPERATION_LOCK_FD_ENV] = "12"
        execution.run_command(["bash", helper], environment=environment, pass_fds=(12,))
        shim = str(api_budget.budget_root() / "bin")

    arguments, options = calls[0]
    prepared = options["environment"]
    assert arguments == ["bash", helper]
    assert prepared["PATH"].split(os.pathsep)[:2] == [
        shim,
        str(Path(sys.executable).absolute().parent),
    ]
    assert prepared["SITE_OPTION"] == "kept"
    assert prepared["PYTHONPATH"] == str(tmp_path / "src")
    assert prepared[SITE_OPERATION_LOCK_FD_ENV] == "12"
    assert prepared[execution.DEADLINE_LABEL_ENV] == "helper"
    assert options["pass_fds"] == (12,)


@pytest.mark.parametrize("inherited_pythonhome", [False, True])
@pytest.mark.parametrize("source_shadow", [False, True])
def test_inert_shell_helper_imports_the_selected_venv_package(
    tmp_path: Path, inherited_pythonhome: bool, source_shadow: bool
) -> None:
    selected = tmp_path / "selected"
    previous = tmp_path / "previous"
    for path, value in ((selected, "selected"), (previous, "previous")):
        venv.EnvBuilder(with_pip=False, symlinks=True).create(path)
        packages = (
            path
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        (packages / "checked_dependency.py").write_text(
            f"VALUE = {value!r}\n", encoding="utf-8"
        )
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "checked_dependency.py").write_text("VALUE = 'shadow'\n")
    environment = staging_deploy.admin_cli_environment(
        {
            **os.environ,
            "PATH": f"{previous}/bin:{os.defpath}",
            **({"PYTHONHOME": sys.base_prefix} if inherited_pythonhome else {}),
            **({"PYTHONPATH": str(shadow)} if source_shadow else {}),
        },
        repository_root=tmp_path,
        venv=selected,
    )
    assert "PYTHONPATH" not in environment, "old source must not shadow the checked CLI"
    assert "PYTHONHOME" not in environment, (
        "an inherited base path must not override the checked venv"
    )

    result = execution.run_command(
        [
            "sh",
            "-c",
            "exec python3 -c 'import json, sys, checked_dependency; "
            "print(json.dumps([sys.prefix, checked_dependency.VALUE]))'",
        ],
        environment=environment,
        timeout_seconds=30,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(selected), "selected"]


@pytest.mark.parametrize("body_error", [False, True])
def test_rotation_environment_keeps_exclusions_and_restores_caller(
    tmp_path, monkeypatch, body_error
) -> None:
    monkeypatch.setenv("PYTHONHOME", "/inherited/home")
    site = RenderedSite(
        source=tmp_path / "site.yaml",
        repository_root=tmp_path,
        release_config={},
        environment={"SITE_OPTION": "kept"},
        source_sha256="a" * 64,
    )
    previous = dict(os.environ)
    refusal = (
        pytest.raises(RuntimeError, match="controlled") if body_error else nullcontext()
    )
    with refusal, rotate_token.site_process_environment(site):
        assert "PYTHONHOME" not in os.environ
        assert os.environ["SITE_OPTION"] == "kept"
        assert os.environ["PYTHONPATH"] == str(tmp_path / "src")
        if body_error:
            raise RuntimeError("controlled")
    assert dict(os.environ) == previous, "the caller environment was not restored"


def test_full_remove_keeps_the_checked_interpreter_environment(tmp_path, monkeypatch):
    class EnvironmentTransport(RemovalTransport):
        def driver(self, arguments, **options):
            environments.append(options["env"])
            return super().driver(arguments, **options)

    environments = []
    monkeypatch.setenv("PYTHONHOME", "/inherited/home")
    transport = EnvironmentTransport(tmp_path, monkeypatch)
    result = transport.remove()

    assert result["phase"] == "COMPLETED"
    assert len(environments) == 5, "every checked removal driver must be exercised"
    for environment in environments:
        assert "PYTHONHOME" not in environment
        assert environment["PATH"].split(os.pathsep)[0] == str(
            Path(sys.executable).absolute().parent
        )
        assert environment["PYTHONPATH"] == str(transport.site.repository_root / "src")
