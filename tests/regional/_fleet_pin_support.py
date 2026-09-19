"""Shared fixtures for the fleet pin hot-reload tests.

One start-up environment carrying an agent pin set and an executor pin set, the
ConfigMap ``.data`` that environment was resolved from, and a reader the poll
loop is pointed at instead of the Kubernetes API.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from gpu_fault.fleet_compatibility import CURRENT_AGENT_PROTOCOL_VERSION
from gpu_fault.fleet_pins import CONFIG_MAP_KEY_ENVIRONMENT
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from tests.fleet._support import ARTIFACT, CONFIG

CONFIG_MAP = "gpu-fault-release-metadata"
NAMESPACE = "gpu-fault-system"
NEW_ARTIFACT = "b" * 64
EXECUTOR_ARTIFACT = "1" * 64
NEW_EXECUTOR_ARTIFACT = "2" * 64


def startup_environment(**overrides: str) -> dict[str, str]:
    """The environment a control-plane Pod starts with: pins plus one non-pin."""

    values = {
        "GPU_FAULT_ENABLE_AGENT_REGISTRY": "true",
        "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": ARTIFACT,
        "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": CONFIG,
        "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": str(
            CURRENT_AGENT_PROTOCOL_VERSION
        ),
        "GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION": "1",
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": str(
            CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
        ),
        "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS": "",
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": EXECUTOR_ARTIFACT,
        # Not a pin: the runtime must carry it through unchanged.
        "GPU_FAULT_SERVICE_ROLE": "worker",
    }
    values.update(overrides)
    return values


def pins_from(environment: Mapping[str, str]) -> dict[str, str]:
    """The ConfigMap ``.data`` that ``environment`` was resolved from."""

    return {
        key: environment[name]
        for key, name in CONFIG_MAP_KEY_ENVIRONMENT.items()
        if name in environment
    }


def config_map_data(**overrides: str) -> dict[str, str]:
    """``.data`` of the ConfigMap holding the start-up pins, some rewritten.

    Keys are the ConfigMap's own hyphenated names, spelled the way the release
    script writes them: ``config_map_data(**{"compatible-agent-artifact-sha256s":
    digest})``.
    """

    data = pins_from(startup_environment())
    data.update(overrides)
    return data


class FakeConfigMapReader:
    """Stands in for the Kubernetes API: serves ``data`` or raises ``error``."""

    def __init__(self, data: Mapping[str, str], resource_version: str = "1") -> None:
        self.data = dict(data)
        self.resource_version = resource_version
        self.error: Exception | None = None
        self.calls = 0

    def __call__(self) -> tuple[dict[str, str], str | None]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return dict(self.data), self.resource_version


def never_read() -> tuple[dict[str, str], str | None]:
    raise AssertionError("the pin runtime must not poll in this test")


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()
