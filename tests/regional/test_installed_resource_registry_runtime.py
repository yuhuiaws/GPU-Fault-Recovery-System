from __future__ import annotations

import json
import os
import sys
import tarfile
import textwrap
import venv
import zipfile
from pathlib import Path

import pytest

from gpu_fault.admin.execution import run_command
from scripts.component_wheels import component_modules
from tests._installed_test_dependencies import dependency_site_directories
from tests.deploy import test_reconciler_template_override as reconciler_support

override_environment = reconciler_support.override_environment

ROOT = Path(__file__).resolve().parents[2]
HELPERS = {"gpu_fault.admin.execution", "gpu_fault_release.regional_resource_probe"}


@pytest.fixture(scope="module")
def packaged_registry(tmp_path_factory):
    root = tmp_path_factory.mktemp("installed-registry-packaging")
    output = root / "wheels"
    output.mkdir()
    result = run_command(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import json
                import sys
                from pathlib import Path
                from scripts.component_wheels import build_component

                root = Path(sys.argv[1])
                wheels = {}
                for name in ("deploy_host", "node_runtime"):
                    wheel, _digest, _modules = build_component(
                        python=sys.executable, name=name,
                        build_root=root / "build", output=root / "wheels",
                    )
                    wheels[name] = str(wheel)
                print(json.dumps(wheels))
                """
            ),
            str(root),
        ],
        cwd=ROOT,
        timeout_seconds=120,
    )
    assert result.returncode == 0, result.stderr
    wheels = json.loads(result.stdout)
    bundle_output = root / "bundles"
    result = run_command(
        [
            "bash",
            str(ROOT / "deploy/node/build-node-installer-bundle.sh"),
            str(bundle_output),
        ],
        environment={**os.environ, "GPU_FAULT_NODE_WHEEL": wheels["node_runtime"]},
        cwd=ROOT,
        timeout_seconds=30,
    )
    assert result.returncode == 0, result.stderr
    bundles = list(bundle_output.glob("gpu-fault-node-installer-*.tar.gz"))
    assert len(bundles) == 1
    extracted = root / "node-bundle"
    with tarfile.open(bundles[0]) as archive:
        archive.extractall(extracted, filter="data")
    tool = next(
        extracted.glob(
            "*/deploy/control-plane/tools/sync_installed_resource_registry.py"
        )
    )
    host = root / "deploy-host"
    venv.EnvBuilder(with_pip=False).create(host)
    installed = (
        host
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    with zipfile.ZipFile(wheels["deploy_host"]) as archive:
        archive.extractall(installed)
    # Add dependency directories without executing the source checkout's .pth
    # files. The unpacked wheel supplies both complete local packages first.
    (installed / "fixture-dependencies.pth").write_text(
        "\n".join(dependency_site_directories()) + "\n"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": "",
        "GPU_FAULT_REPOSITORY_ROOT": str(ROOT),
    }
    return host / "bin/python3", installed, tool, environment


@pytest.mark.parametrize(
    "component", ["deploy_host", "control_plane", "executor", "node_runtime"]
)
def test_registry_helpers_are_packaged_only_for_the_deploy_host(component: str) -> None:
    included = HELPERS & component_modules(component)
    assert included == (HELPERS if component == "deploy_host" else set())


def test_the_bundled_sync_tool_imports_from_the_isolated_deploy_host_wheel(
    packaged_registry,
) -> None:
    python, installed, tool, environment = packaged_registry
    result = run_command(
        [
            str(python),
            "-I",
            "-c",
            textwrap.dedent(
                """
                import importlib
                import json
                import runpy
                import sys
                from pathlib import Path

                installed = Path(sys.argv[1])
                tool = sys.argv[2]
                sys.argv = [tool, "--help"]
                try:
                    runpy.run_path(tool, run_name="__main__")
                except SystemExit as exc:
                    assert exc.code == 0
                names = [
                    "gpu_fault.admin.execution",
                    "gpu_fault_release.regional_resource_probe",
                ]
                for name in names:
                    module = importlib.import_module(name)
                    assert Path(module.__file__).is_relative_to(installed)
                for name, module in sys.modules.copy().items():
                    if name.startswith(("gpu_fault.", "gpu_fault_release.")):
                        origin = getattr(module, "__file__", None)
                        if origin:
                            assert Path(origin).is_relative_to(installed), name
                print(json.dumps({"isolated_helpers": names}))
                """
            ),
            str(installed),
            str(tool),
        ],
        environment=environment,
        cwd=tool.parent,
        timeout_seconds=30,
    )
    assert result.returncode == 0, result.stderr
    assert (
        set(json.loads(result.stdout.splitlines()[-1])["isolated_helpers"]) == HELPERS
    )


@pytest.mark.parametrize("sync_registry", ["false", "true"])
def test_reconciler_shell_sync_branch_runs_with_the_deploy_host_wheel(
    packaged_registry, override_environment, tmp_path: Path, sync_registry: str
) -> None:
    python, _installed, _tool, _environment = packaged_registry
    binary = tmp_path / "bin"
    (binary / "python3").unlink()
    original = binary / "fixture-kubectl"
    (binary / "kubectl").rename(original)
    wrapper = binary / "kubectl"
    wrapper.write_text(
        f"#!{python}\n"
        + textwrap.dedent(
            """
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            root = Path(os.environ["TEST_TEMPLATE_ROOT"])
            if "config" in args and "view" in args:
                print(json.dumps({"apiVersion": "v1", "kind": "Config", "users": []}))
            elif "get" in args and "gpu-fault-installed-resources" in args:
                pass
            elif "get" in args and args[args.index("get") + 1] in {
                "deployment", "daemonset", "poddisruptionbudget", "serviceaccount",
                "clusterrole", "clusterrolebinding", "role", "rolebinding",
            }:
                print(json.dumps({"apiVersion": "v1", "kind": "List", "items": []}))
            elif "apply" in args and args[args.index("-f") + 1] == "-":
                value = json.load(sys.stdin)
                assert value["metadata"]["name"] == "gpu-fault-installed-resources"
                (root / "registry-applied.json").write_text(json.dumps(value))
            else:
                original = root / "bin/fixture-kubectl"
                os.execv(original, [str(original), *args])
            """
        )
    )
    wrapper.chmod(0o755)
    result = run_command(
        ["bash", str(ROOT / "deploy/node/deploy-node-installer-reconciler.sh")],
        environment={
            **override_environment,
            "PATH": os.pathsep.join(
                (str(binary), str(python.parent), os.environ["PATH"])
            ),
            "PYTHONPATH": "",
            "GPU_FAULT_REPOSITORY_ROOT": str(ROOT),
            "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": sync_registry,
        },
        cwd=tmp_path,
        timeout_seconds=30,
    )
    assert result.returncode == 0, result.stderr
    applied = tmp_path / "registry-applied.json"
    assert applied.exists() is (sync_registry == "true")
    if sync_registry == "true":
        document = json.loads(json.loads(applied.read_text())["data"]["inventory.json"])
        assert document["plane"] == "gpu"
        assert document["namespace"] == "gpu-fault-system"
        assert json.loads(result.stdout.splitlines()[-1])["written"] is True
