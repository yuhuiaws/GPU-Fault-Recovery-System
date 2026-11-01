"""The product's RESET_GPU is a Function Level Reset, never a bus reset (b233).

nvidia-smi(1) (v610-era page), "-r, --gpu-reset": "Reset triggered without
extra arguments, will be a default Function Level Reset (FLR). To issue a Bus
Reset, use -r bus. For certain platforms only Function Level Reset is
possible." The node action passes no reset type, so it is an FLR -- which is
why NVML v615's GPU recovery action 6, BUS_RESET ("Reset the GPU’s PCIe
bus."), must never map to RESET_GPU (``gpu_recovery_action_policy`` fails it
closed; ``tests/metrics/test_dcgm_gap_field_rules.py``).
"""

from __future__ import annotations

from gpu_fault.gpu_recovery_action_policy import NVIDIA_SMI_RESET_TYPE_QUOTE

from ._support import (
    FakeRunner,
    NodeActionStatus,
    WorkflowOperation,
    command,
    envelope,
    executor,
)


def test_reset_gpu_issues_the_default_function_level_reset(tmp_path) -> None:
    runner = FakeRunner()
    agent = executor(tmp_path, runner)

    result = agent.execute(envelope(command(WorkflowOperation.RESET_GPU)))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    resets = [item for item in runner.commands if "--gpu-reset" in item]
    assert resets == [["nvidia-smi", "--gpu-reset", "-i", "GPU-a"]]
    assert not any({"bus", "-r"} & set(item) for item in resets), (
        "no reset type: nvidia-smi's default Function Level Reset"
    )
    assert "default Function Level Reset (FLR)" in NVIDIA_SMI_RESET_TYPE_QUOTE
