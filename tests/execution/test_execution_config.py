from __future__ import annotations

import pytest

from gpu_fault.execution import ProductionExecutorConfig
from gpu_fault.execution.config import WorkflowDispatcherConfig


@pytest.mark.parametrize("token", ["1", "yes", "on", "TRUE"])
def test_preemption_switch_accepts_every_enabled_token(token: str) -> None:
    """A default-on switch spelled ``=1`` used to switch preemption off."""

    config = ProductionExecutorConfig.from_mapping(
        {"GPU_FAULT_ENABLE_WORKFLOW_PREEMPTION": token}
    )

    assert config.workflow_preemption_enabled is True


@pytest.mark.parametrize("token", ["0", "no", "off"])
def test_preemption_switch_accepts_every_disabled_token(token: str) -> None:
    config = ProductionExecutorConfig.from_mapping(
        {"GPU_FAULT_ENABLE_WORKFLOW_PREEMPTION": token}
    )

    assert config.workflow_preemption_enabled is False


@pytest.mark.parametrize("token", ["1", "yes", "on"])
def test_dispatcher_switch_accepts_every_enabled_token(token: str) -> None:
    config = WorkflowDispatcherConfig.from_mapping(
        {"GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER": token}, executor_enabled=True
    )

    assert config.enabled is True


def test_dispatcher_switch_typo_is_loud() -> None:
    with pytest.raises(ValueError, match="GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER"):
        WorkflowDispatcherConfig.from_mapping(
            {"GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER": "ture"}, executor_enabled=True
        )
