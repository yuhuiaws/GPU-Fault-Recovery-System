"""Component wheels must carry every module a component reaches by name.

The node-runtime wheel built on 2026-09-07 lost the DCGM, nvidia-smi, host,
kernel, Fabric Manager and node-log collectors: ``gpu_fault.collector_registry``
names their factories as ``"module:attr"`` strings, which the import walk
cannot see, and the collector services on the first upgraded node died with
ModuleNotFoundError. The closure has to follow those references.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import component_wheels  # noqa: E402

REGISTRY_FACTORY_MODULES = (
    "gpu_fault.collectors.gpu.dcgm",
    "gpu_fault.collectors.gpu.nvidia_smi",
    "gpu_fault.collectors.host.collector",
    "gpu_fault.collectors.logs.kernel",
    "gpu_fault.collectors.logs.fabric_manager",
    "gpu_fault.collectors.logs.node",
    "gpu_fault.collectors.training_progress",
)


@pytest.mark.parametrize("component", ["node_runtime", "executor"])
def test_collector_wheels_include_every_registry_factory_module(component: str) -> None:
    modules = component_wheels.component_modules(component)
    missing = [name for name in REGISTRY_FACTORY_MODULES if name not in modules]
    assert missing == [], f"{component} wheel would lack {missing}"


def test_the_deploy_host_wheel_carries_the_release_engine_the_admin_cli_imports() -> (
    None
):
    """The admin CLI imports ``gpu_fault_release`` at module level (status
    header, deploy consent, schema-change acceptance, rollback). A walk that
    followed only ``gpu_fault`` shipped a wheel whose ``gpu-fault-admin`` died
    with ModuleNotFoundError on the first deploy-host install from main after
    aca7a37 (2026-09-09)."""
    modules = component_wheels.component_modules("deploy_host")
    assert "gpu_fault_release.regional_admin_commands" in modules, sorted(
        name for name in modules if name.startswith("gpu_fault_release")
    )
    assert "gpu_fault.admin.deploy_consent" in modules, "the consent module is shipped"


def test_deploy_host_carries_the_cleanup_tools_rbac_dependency() -> None:
    module = "gpu_fault.admin.cluster_removal_rbac"
    assert module in component_wheels.component_modules("deploy_host"), (
        "the copied cleanup tools must import their RBAC helper from the installed wheel"
    )
    for name in component_wheels.COMPONENTS:
        assert module not in component_wheels.component_modules(name), (
            f"{name} must not receive deployment-only cleanup authority"
        )


@pytest.mark.parametrize("component", ["deploy_host", "control_plane"])
def test_workload_clis_are_delivered_without_removing_cpu_compatibility(
    component: str,
) -> None:
    expected = {
        "gpu-training-submit": "gpu_fault.training_submit_cli:main",
        "gpu-fault-workload-annotate": "gpu_fault.workload_annotate_cli:main",
    }
    definition = component_wheels.component_definition(component)
    for command, target in expected.items():
        assert definition.scripts.get(command) == target, (component, command)
    modules = component_wheels.component_modules(component)
    assert {target.partition(":")[0] for target in expected.values()} <= modules
    assert "gpu_fault.admin.python_environment" in modules, (
        "site helpers must retain the parent's selected-interpreter binding"
    )


@pytest.mark.parametrize(
    "component", sorted(component_wheels.COMPONENTS) + ["deploy_host"]
)
def test_every_component_closure_is_import_complete(component: str) -> None:
    """Whatever a shipped module imports from a local package is shipped too."""
    modules = component_wheels.component_modules(component)
    missing = {
        f"{module} -> {imported}"
        for module in modules
        for imported in component_wheels.local_imports(module)
        if imported not in modules
    }
    assert missing == set(), sorted(missing)


def test_factory_reference_strings_name_their_module() -> None:
    import ast

    tree = ast.parse(
        'FACTORY = "gpu_fault.collectors.gpu.dcgm:build_from_environment"\n'
        'OTHER = "gpu_fault.collectors.gpu.dcgm is mentioned in prose"\n'
        'LOADED = import_module("gpu_fault.collectors.logs.kernel")\n'
    )
    found = component_wheels.referenced_modules(tree)
    assert found == {
        "gpu_fault.collectors.gpu.dcgm",
        "gpu_fault.collectors.logs.kernel",
    }, found


def test_repack_wheel_stored_keeps_content_and_drops_per_file_deflate(tmp_path) -> None:
    """Whole-file xz over deflated members gains nothing; the ConfigMap ceiling
    is 1 MiB, and the deflated control-plane wheel already exceeds it."""
    import zipfile

    wheel = tmp_path / "demo-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        entry = zipfile.ZipInfo("demo/__init__.py", date_time=(1980, 1, 1, 0, 0, 0))
        entry.external_attr = 0o644 << 16
        entry.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(entry, "x = 1\n" * 2000)
        zf.writestr("demo-0.1.dist-info/RECORD", "demo/__init__.py,sha256=abc,12000\n")
    before = wheel.stat().st_size

    component_wheels.repack_wheel_stored(wheel)

    with zipfile.ZipFile(wheel) as zf:
        infos = zf.infolist()
        assert [item.filename for item in infos] == [
            "demo/__init__.py",
            "demo-0.1.dist-info/RECORD",
        ], infos
        assert {item.compress_type for item in infos} == {zipfile.ZIP_STORED}, infos
        assert zf.read("demo/__init__.py") == b"x = 1\n" * 2000, "content changed"
        assert infos[0].external_attr == 0o644 << 16, infos[0].external_attr
        assert infos[0].date_time == (1980, 1, 1, 0, 0, 0), infos[0].date_time
    assert wheel.stat().st_size > before, "stored wheel must be the larger one"


def test_the_installed_module_digest_matches_the_staged_wheel_digest(tmp_path) -> None:
    """``gpu_fault.module_digest`` (what the installer reads back from the venv)
    and ``package_digest`` (what the bundle manifest records) must be one
    computation: with the release engine staged next to ``gpu_fault`` they
    disagreed on ordering and deploy #13 (2026-09-09) refused the install as
    "project identity does not match bundle"."""
    import gpu_fault

    src = tmp_path / "src"
    (src / "gpu_fault/admin").mkdir(parents=True)
    (src / "gpu_fault/__init__.py").write_text("x = 1\n", encoding="utf-8")
    (src / "gpu_fault/admin/cli.py").write_text("y = 2\n", encoding="utf-8")
    (src / "gpu_fault/data").mkdir()
    (src / "gpu_fault/data/env-inventory.json").write_text("{}", encoding="utf-8")
    without_engine = gpu_fault.module_digest(src / "gpu_fault")
    assert without_engine == component_wheels.package_digest(src / "gpu_fault"), (
        "gpu_fault-only trees hash identically"
    )
    (src / "gpu_fault_release").mkdir()
    (src / "gpu_fault_release/__init__.py").write_text("z = 3\n", encoding="utf-8")
    (src / "gpu_fault_release/__pycache__").mkdir()
    (src / "gpu_fault_release/__pycache__/skip.pyc").write_bytes(b"\x00")
    with_engine = gpu_fault.module_digest(src / "gpu_fault")
    assert with_engine == component_wheels.package_digest(src / "gpu_fault"), (
        "the sibling engine is folded in identically on both sides"
    )
    assert with_engine != without_engine, "the engine is part of the identity"


def test_node_runtime_does_not_ship_the_control_plane_orchestration_families() -> None:
    """A module inside the node-runtime wheel changes the wheel digest, and a
    changed digest re-rolls every GPU node. The orchestration families are
    composed by the control-plane coordinator alone; no node entrypoint reaches
    them. Yet ``gpu_fault.orchestration.families.__init__`` imported all eleven
    eagerly, so the collector path ``training_health -> families.identity``
    shipped the whole package to every node and a control-plane-only change to
    ``node_lifecycle`` rolled the data plane (every deploy through #21,
    2026-09-19)."""
    node_runtime = component_wheels.component_modules("node_runtime")
    control_plane = component_wheels.component_modules("control_plane")
    for module in (
        "gpu_fault.orchestration.families.node_lifecycle",
        "gpu_fault.orchestration.families.health",
        "gpu_fault.orchestration.families.grouped_health",
        "gpu_fault.orchestration.families.drain",
        "gpu_fault.orchestration.families.reset",
    ):
        assert module not in node_runtime, f"{module} is control-plane code"
        assert module in control_plane, f"the coordinator still composes {module}"
