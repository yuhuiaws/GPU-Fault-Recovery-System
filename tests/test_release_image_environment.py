from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import venv

import pytest

from scripts.verify_release_images import (
    LEGACY_COMPONENT_DOCKERFILE_SHA256,
    SINGLE_ENVIRONMENT_CHECK,
    verify_images,
)
from tests.test_release_image_container_cleanup import FakeDocker


@pytest.fixture
def environment(tmp_path):
    runtime = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False).create(runtime)
    packages = runtime / f"lib/python{sys.version_info.major}.{sys.version_info.minor}"
    packages /= "site-packages"
    for module in ("gpu_fault", "pydantic", "psycopg"):
        (packages / module).mkdir()
        (packages / module / "__init__.py").write_text("def main(): return None\n")
    distribution = packages / "gpu_fault_control_plane-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: gpu-fault-control-plane\nVersion: 1.0\n"
    )
    (distribution / "top_level.txt").write_text("gpu_fault\n")
    (distribution / "entry_points.txt").write_text(
        "[console_scripts]\ngpu-fault-probe = gpu_fault:main\n"
    )
    command = runtime / "bin/gpu-fault-probe"
    command.write_text(f"#!{runtime}/bin/python\nfrom gpu_fault import main\nmain()\n")
    command.chmod(0o755)
    alias = tmp_path / "control-plane"
    alias.symlink_to(runtime.name, target_is_directory=True)
    env = {
        "PATH": str(runtime / "bin") + os.pathsep + os.defpath,
        "PYTHONPATH": str(tmp_path / "untrusted"),
        "PYTHONHOME": str(tmp_path / "untrusted"),
        "PYTHONUSERBASE": str(tmp_path / "untrusted"),
    }

    def run(python="python"):
        return subprocess.run(
            [
                str(python),
                "-I",
                "-B",
                "-c",
                SINGLE_ENVIRONMENT_CHECK,
                str(runtime),
                "gpu-fault-control-plane",
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )

    return runtime, packages, alias, env, run


@pytest.mark.parametrize("compatibility_path", [False, True])
def test_single_environment_starts_in_isolated_mode_from_both_paths(
    environment, compatibility_path
):
    _runtime, _packages, alias, _env, run = environment
    result = run(alias / "bin/python" if compatibility_path else "python")
    assert result.returncode == 0, result.stderr


def test_manifest_path_alias_selects_the_same_python_and_console_scripts(environment):
    _runtime, _packages, alias, env, run = environment
    env["PATH"] = str(alias / "bin") + os.pathsep + os.defpath
    result = run()
    assert result.returncode == 0, result.stderr


def test_base_python_cannot_pass_the_single_environment_check(environment):
    result = environment[-1](sys.executable)
    assert result.returncode != 0
    assert "incorrect application interpreter" in result.stderr


def test_separate_component_environment_is_not_a_compatibility_alias(
    environment, tmp_path
):
    second = tmp_path / "other-runtime"
    venv.EnvBuilder(with_pip=False).create(second)
    result = environment[-1](second / "bin/python")
    assert result.returncode != 0
    assert "incorrect application interpreter" in result.stderr


def test_shared_system_packages_are_rejected_even_with_isolated_startup(environment):
    runtime, _packages, _alias, _env, run = environment
    configuration = runtime / "pyvenv.cfg"
    configuration.write_text(
        configuration.read_text().replace(
            "include-system-site-packages = false",
            "include-system-site-packages = true",
        )
    )
    result = run()
    assert result.returncode != 0
    assert "shared packages enabled" in result.stderr


def test_dependency_bridge_to_another_package_environment_is_rejected(
    environment, tmp_path
):
    _runtime, packages, _alias, _env, run = environment
    shared = tmp_path / "base/lib/site-packages"
    shared.mkdir(parents=True)
    (packages / "dependency-bridge.pth").write_text(str(shared) + "\n")
    result = run()
    assert result.returncode != 0
    assert "external package search path" in result.stderr


def test_distribution_outside_the_runtime_is_rejected(environment, tmp_path):
    _runtime, packages, _alias, _env, run = environment
    shared = tmp_path / "shared"
    distribution = shared / "external-1.0.dist-info"
    distribution.mkdir(parents=True)
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: external\nVersion: 1.0\n"
    )
    (packages / "dependency-bridge.pth").write_text(str(shared) + "\n")
    result = run()
    assert result.returncode != 0
    assert "external distribution" in result.stderr


@pytest.mark.parametrize("module", ["gpu_fault", "pydantic", "psycopg"])
def test_symlinked_application_or_dependency_cannot_escape_runtime(
    environment, tmp_path, module
):
    _runtime, packages, _alias, _env, run = environment
    external = tmp_path / "external.py"
    external.write_text("def main(): return None\n")
    source = packages / module / "__init__.py"
    source.unlink()
    source.symlink_to(external)
    result = run()
    assert result.returncode != 0
    assert "external module" in result.stderr


def test_combined_cpu_executor_distributions_cannot_pass(environment):
    _runtime, packages, _alias, _env, run = environment
    distribution = packages / "gpu_fault_cluster_executor-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: gpu-fault-cluster-executor\nVersion: 1.0\n"
    )
    (distribution / "top_level.txt").write_text("gpu_fault\n")
    result = run()
    assert result.returncode != 0
    assert "component distribution differs" in result.stderr


def test_cli_must_resolve_to_the_application_environment(environment, tmp_path):
    _runtime, _packages, _alias, env, run = environment
    external = tmp_path / "other-bin"
    external.mkdir()
    command = external / "gpu-fault-probe"
    command.write_text("#!/bin/sh\nexit 0\n")
    command.chmod(0o755)
    env["PATH"] = str(external) + os.pathsep + env["PATH"]
    result = run()
    assert result.returncode != 0
    assert "incorrect CLI path" in result.stderr


def test_cli_shebang_cannot_select_base_python(environment):
    runtime, _packages, _alias, _env, run = environment
    (runtime / "bin/gpu-fault-probe").write_text(f"#!{sys.executable}\n")
    result = run()
    assert result.returncode != 0
    assert "incorrect CLI interpreter" in result.stderr


def test_console_entrypoint_imports_are_really_executed(environment):
    _runtime, packages, _alias, _env, run = environment
    (packages / "gpu_fault_control_plane-1.0.dist-info/entry_points.txt").write_text(
        "[console_scripts]\ngpu-fault-probe = absent_entry_dependency:main\n"
    )
    result = run()
    assert result.returncode != 0
    assert "absent_entry_dependency" in result.stderr


@pytest.fixture
def image_check():
    inventory = json.dumps({"files": {"dependency.whl": {"sha256": "a" * 64}}})
    inventory_digest = hashlib.sha256(inventory.encode()).hexdigest()
    descriptor = {
        "schema_version": 3,
        "images": {
            name: {"tag": name, "dockerfile_sha256": "d" * 64}
            for name in ("control_plane", "executor")
        },
        "components": {
            name: {"module_digest": digest * 64}
            for name, digest in (("control_plane", "c"), ("executor", "e"))
        },
    }
    descriptor["images"]["node_dependencies"] = {
        "tag": "node_dependencies",
        "wheelhouse_sha256": inventory_digest,
    }

    def execute(reference: str, arguments: list[str]) -> tuple[int, str]:
        if "module_digest; print(module_digest())" in arguments[-1]:
            return 0, descriptor["components"][reference]["module_digest"]
        if arguments == ["/bin/cat", "/opt/gpu-fault/wheelhouse/inventory.json"]:
            return 0, inventory
        if "sha256sum" in arguments[-1]:
            return (
                0,
                f"{'a' * 64}  dependency.whl\n{inventory_digest}  inventory.json\n",
            )
        return 0, ""

    return descriptor, FakeDocker(execute)


@pytest.mark.parametrize("legacy", [False, True])
def test_only_known_legacy_image_identity_uses_the_old_environment_contract(
    image_check, legacy
):
    descriptor, docker = image_check
    if legacy:
        for name in ("control_plane", "executor"):
            descriptor["images"][name]["dockerfile_sha256"] = (
                LEGACY_COMPONENT_DOCKERFILE_SHA256
            )
    verify_images(descriptor, runner=docker)
    for name in ("control_plane", "executor"):
        command = next(
            creation["command"]
            for creation in docker.creations
            if creation["reference"] == name
        )
        assert command[:2] == ["/bin/sh", "-ec"]
        arguments = shlex.split(command[2])
        assert arguments.count(SINGLE_ENVIRONMENT_CHECK) == (0 if legacy else 2)
        if not legacy:
            assert arguments.count("/opt/gpu-fault/runtime") == 3
            assert "check" in arguments and "pip" in arguments
            assert "--help" in arguments
    assert docker.containers == {}


@pytest.mark.parametrize("identity", [None, "", "legacy", 17, "z" * 64])
def test_missing_or_malformed_dockerfile_identity_is_not_legacy(image_check, identity):
    descriptor, docker = image_check
    descriptor["images"]["control_plane"]["dockerfile_sha256"] = identity
    with pytest.raises(ValueError, match="Dockerfile identity"):
        verify_images(descriptor, runner=docker)
    assert docker.creations == []
