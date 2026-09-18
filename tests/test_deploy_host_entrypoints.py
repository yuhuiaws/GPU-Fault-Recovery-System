"""Exercise the real deploy-host wheel without checkout import fallback."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import venv
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from gpu_fault_release import STATE_DIR_BINDING
from scripts import component_wheels
from tests._installed_test_dependencies import dependency_site_directories
from tests.admin.test_admin_site import site_file

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ("gpu-fault-admin", "gpu-training-submit", "gpu-fault-workload-annotate")
OFFLINE_GUARD = """\
import sys

def reject_external_actions(event, arguments):
    if event in {
        "socket.connect", "socket.bind", "sqlite3.connect",
        "subprocess.Popen", "os.system",
    }:
        raise RuntimeError("offline CLI attempted forbidden action: " + event)

sys.addaudithook(reject_external_actions)
"""


def offline_environment(root: Path) -> dict[str, str]:
    return {
        "PATH": os.pathsep.join((str(Path(sys.executable).parent), "/usr/bin", "/bin")),
        "HOME": str(root),
        "TMPDIR": str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_NO_INDEX": "1",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_CONFIG_FILE": os.devnull,
        "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
        "KUBECONFIG": os.devnull,
    }


@dataclass(frozen=True)
class InstalledDeployHost:
    python: Path
    site_packages: Path
    site: Path
    cwd: Path
    environment: dict[str, str]

    def run(self, command: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.python.parent / command), *arguments],
            cwd=self.cwd,
            env=self.environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )


@pytest.fixture(scope="module")
def installed_deploy_host(
    tmp_path_factory: pytest.TempPathFactory,
) -> InstalledDeployHost:
    root = tmp_path_factory.mktemp("installed-deploy-host")
    environment = offline_environment(root)
    wheels = root / "wheels"
    wheels.mkdir()
    with patch.dict(os.environ, environment, clear=True):
        wheel, _digest, _modules = component_wheels.build_component(
            python=sys.executable,
            name="deploy_host",
            build_root=root / "build",
            output=wheels,
        )

    host = root / "host"
    venv.EnvBuilder(with_pip=False).create(host)
    python = host / "bin/python"
    installed = (
        host
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    # Match the dependency overlay without executing another environment's .pth files.
    (installed / "fixture-dependencies.pth").write_text(
        "\n".join(dependency_site_directories()) + "\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "pip",
            "--python",
            str(python),
            "install",
            "--no-index",
            "--no-deps",
            "--no-cache-dir",
            "--no-compile",
            "--force-reinstall",
            str(wheel),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    (installed / "_gpu_fault_offline_guard.py").write_text(
        OFFLINE_GUARD, encoding="utf-8"
    )
    (installed / "fixture-offline-guard.pth").write_text(
        "import _gpu_fault_offline_guard\n", encoding="utf-8"
    )

    bound_site = site_file(root / "site-fixture")
    repository = root / ("release snapshot with spaces " * 4).strip()
    repository.symlink_to(ROOT, target_is_directory=True)
    document = yaml.safe_load(bound_site.read_text(encoding="utf-8"))
    document["spec"]["repositoryRoot"] = os.path.relpath(repository, bound_site.parent)
    bound_site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    (host / STATE_DIR_BINDING).write_text(
        json.dumps({"schema_version": 1, "state_dir": str(bound_site.parent)}),
        encoding="utf-8",
    )
    environment["PATH"] = str(python.parent)
    cwd = root / "commands"
    cwd.mkdir()
    return InstalledDeployHost(python, installed, bound_site, cwd, environment)


@pytest.mark.parametrize("command", COMMANDS)
def test_installed_help_uses_only_the_bound_site(
    installed_deploy_host: InstalledDeployHost, command: str
) -> None:
    result = installed_deploy_host.run(command, "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert command in result.stdout


def test_installed_commands_import_only_the_component_packages(
    installed_deploy_host: InstalledDeployHost,
) -> None:
    result = installed_deploy_host.run(
        "python",
        "-I",
        "-B",
        "-c",
        textwrap.dedent(
            """
            import importlib
            import importlib.metadata
            import json
            import os
            import sys
            from pathlib import Path

            installed = Path(sys.argv[1])
            assert "PYTHONPATH" not in os.environ
            assert "GPU_FAULT_REPOSITORY_ROOT" not in os.environ
            for name in (
                "gpu_fault.admin.cli",
                "gpu_fault.admin.python_environment",
                "gpu_fault.training_submit_cli",
                "gpu_fault.workload_annotate_cli",
            ):
                importlib.import_module(name)
            for name, module in sys.modules.copy().items():
                if name.startswith(("gpu_fault.", "gpu_fault_release")):
                    origin = getattr(module, "__file__", None)
                    if origin:
                        assert Path(origin).is_relative_to(installed), name
            distribution = importlib.metadata.distribution("gpu-fault-deploy-host")
            print(json.dumps({
                point.name: point.value
                for point in distribution.entry_points
                if point.group == "console_scripts"
            }))
            """
        ),
        str(installed_deploy_host.site_packages),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "gpu-fault-admin": "gpu_fault.admin.cli:main",
        "gpu-training-submit": "gpu_fault.training_submit_cli:main",
        "gpu-fault-workload-annotate": "gpu_fault.workload_annotate_cli:main",
    }


@pytest.fixture
def workload_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "workload.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": "offline-training"},
                "spec": {
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "trainer",
                                    "image": "example.invalid/training:fixture",
                                }
                            ],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.mark.parametrize(
    ("command", "options"),
    [("gpu-training-submit", ("--dry-run",)), ("gpu-fault-workload-annotate", ())],
)
def test_installed_workload_rendering_is_offline_and_uses_the_site_profile(
    installed_deploy_host: InstalledDeployHost,
    workload_manifest: Path,
    command: str,
    options: tuple[str, ...],
) -> None:
    original = workload_manifest.read_bytes()
    result = installed_deploy_host.run(
        command,
        str(workload_manifest),
        "--site",
        str(installed_deploy_host.site),
        "--job-id",
        "offline-job",
        "--attempt-number",
        "3",
        "--namespace",
        "training",
        *options,
    )

    assert result.returncode == 0, result.stderr
    rendered = yaml.safe_load(result.stdout)
    assert rendered["metadata"]["namespace"] == "training"
    metadata = rendered["spec"]["template"]["metadata"]
    assert metadata["labels"]["gpu-fault.io/managed"] == "true"
    assert metadata["labels"]["gpu-fault.io/attempt-id"] == "offline-job-a003"
    assert metadata["annotations"]["gpu-fault.io/runtime-profile-version"] == (
        "hyperpod-v1"
    )
    assert metadata["annotations"]["gpu-fault.io/expected-critical-ranks"] == "1"
    assert workload_manifest.read_bytes() == original


@pytest.mark.parametrize("command", COMMANDS[1:])
def test_installed_workload_clis_reject_profile_drift(
    installed_deploy_host: InstalledDeployHost, workload_manifest: Path, command: str
) -> None:
    result = installed_deploy_host.run(
        command,
        str(workload_manifest),
        "--site",
        str(installed_deploy_host.site),
        "--runtime-profile-version",
        "different-profile",
    )

    assert result.returncode == 1, result.stderr
    assert "does not match the site Profile" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "event", ["socket.connect", "sqlite3.connect", "subprocess.Popen"]
)
def test_installed_fixture_rejects_external_action_audit_events(
    installed_deploy_host: InstalledDeployHost, event: str
) -> None:
    result = installed_deploy_host.run(
        "python", "-I", "-B", "-c", f"import sys; sys.audit({event!r})"
    )

    assert result.returncode == 1
    assert f"offline CLI attempted forbidden action: {event}" in result.stderr
