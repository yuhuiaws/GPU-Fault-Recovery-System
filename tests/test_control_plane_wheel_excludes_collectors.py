"""The control-plane wheel must not carry the data-plane collectors.

The wheel is an import-walk closure, and until 2026-09-19 it held 26 data-plane
modules -- every ``gpu_fault.collectors.*`` implementation, the registry and
``gpu_fault.collectors_cli`` -- because ``gpu_fault.telemetry`` and
``gpu_fault.collector_requirements`` read a few kind rows from
``gpu_fault.collector_registry``, whose ``COLLECTOR_REGISTRY`` names every
collector factory as a ``"module:attr"`` string the walk follows. A
collector-only edit therefore changed the control-plane wheel digest and
re-rolled the control plane. The kind rows live in the leaf module
``gpu_fault.collector_kinds`` now; the CLI table stays on the data plane.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import component_wheels  # noqa: E402

from tests.test_component_wheel_closures import REGISTRY_FACTORY_MODULES  # noqa: E402

# The registry's factory strings plus the modules only the collector CLI reaches.
# ``gpu_fault.collectors.sinks`` and the outbox modules are shared code and stay.
DATA_PLANE_ONLY_MODULES = (
    "gpu_fault.collectors_cli",
    "gpu_fault.collector_registry",
    "gpu_fault.collectors.gpu.dcgm",
    "gpu_fault.collectors.gpu.nvidia_smi",
    "gpu_fault.collectors.host.collector",
    "gpu_fault.collectors.logs.kernel",
    "gpu_fault.collectors.logs.fabric_manager",
    "gpu_fault.collectors.logs.node",
    "gpu_fault.collectors.training_progress",
    "gpu_fault.collectors.cloud.kubernetes",
)


def test_control_plane_wheel_excludes_the_collector_implementations() -> None:
    modules = component_wheels.component_modules("control_plane")
    shipped = [name for name in DATA_PLANE_ONLY_MODULES if name in modules]
    assert shipped == [], f"control_plane wheel would ship data-plane code {shipped}"


def test_control_plane_wheel_keeps_the_kind_table_and_telemetry() -> None:
    modules = component_wheels.component_modules("control_plane")
    assert "gpu_fault.collector_kinds" in modules, (
        "the kind table is control-plane code"
    )
    assert "gpu_fault.telemetry" in modules, "telemetry stays in the control plane"


@pytest.mark.parametrize("component", ["node_runtime", "executor"])
def test_data_plane_wheels_still_carry_every_registry_factory_module(
    component: str,
) -> None:
    modules = component_wheels.component_modules(component)
    missing = [name for name in REGISTRY_FACTORY_MODULES if name not in modules]
    assert missing == [], f"{component} wheel would lack {missing}"
