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
