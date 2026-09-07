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
