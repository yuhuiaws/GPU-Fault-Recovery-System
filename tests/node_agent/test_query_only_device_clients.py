"""A read-only ``nvidia-smi`` query is not a GPU client the reset must yield to.

Live (gpu-eks-cluster, XID 46 on ip-10-0-41-164, release 5377615bd785): the
quiesce had stopped ``nvidia-persistenced``, so every ``nvidia-smi`` call
initialised the driver and held ``/dev/nvidia0`` for seconds; the acceptance
sampler's query was seen in both 0.5 s samples of the final ownership recheck
and the reset was refused ``OWNERSHIP_FINAL_CLIENTS_CHANGED:
GPU-3df3...:212320:nvidia-smi`` on a GPU nobody computed on. ``nvidia-smi``
speaks NVML through ``/dev/nvidiactl`` and ``/dev/nvidiaN`` and never opens
``/dev/nvidia-uvm``, which every CUDA context needs; such a holder is now
classified query-only, recorded, and ignored by both the sampled preflight and
the final recheck. Anything else -- another name, a UVM descriptor, a process
whose descriptors cannot be read -- still refuses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.late_ownership import (
    OwnershipRefused,
    classify_final_device_clients,
    execute_with_final_ownership,
)
from gpu_fault.node_agent.protocol import NodeActionStatus, SignedNodeAction
from gpu_fault.node_agent.query_clients import query_only_device_client
from tests.node_agent._support import (
    FakeRunner,
    command,
    envelope,
    node_action_executor,
)
from tests.regional.test_late_ownership_agent_gate import command as command_fixture
from tests.regional.test_late_ownership_agent_gate_edges import FakeNodeAgent

LOGGER_NAME = "gpu_fault.node_agent.late_ownership"


def _process(proc: Path, pid: str, name: str, *targets: str) -> None:
    process = proc / pid
    (process / "fd").mkdir(parents=True)
    (process / "comm").write_text(name + "\n", encoding="ascii")
    for index, target in enumerate(targets):
        (process / "fd" / str(index + 3)).symlink_to(target)


def _holder(pid: str, name: str) -> dict[str, str]:
    return {
        "gpu_uuid": "GPU-a",
        "pid": pid,
        "process_name": name,
        "device": "/dev/nvidia0",
    }


def test_the_classification_needs_positive_evidence(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _process(proc, "212320", "nvidia-smi", "/dev/nvidiactl", "/dev/nvidia0")
    _process(
        proc, "4243", "nvidia-smi", "/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"
    )
    _process(proc, "4244", "python", "/dev/nvidiactl", "/dev/nvidia0")

    assert query_only_device_client(proc, _holder("212320", "nvidia-smi")) is True
    assert query_only_device_client(proc, _holder("4243", "nvidia-smi")) is False, (
        "a UVM descriptor is a CUDA context, whatever the process is called"
    )
    assert query_only_device_client(proc, _holder("4244", "python")) is False
    assert query_only_device_client(proc, _holder("9999", "nvidia-smi")) is False, (
        "a process whose descriptors cannot be read is not vouched for"
    )
    assert query_only_device_client(None, _holder("212320", "nvidia-smi")) is False


def _signed() -> SignedNodeAction:
    return SignedNodeAction(command=command_fixture(), signature="unused")


def test_the_final_recheck_ignores_a_persistent_query_and_records_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    proc = tmp_path / "proc"
    _process(proc, "212320", "nvidia-smi", "/dev/nvidiactl", "/dev/nvidia0")
    agent = FakeNodeAgent()
    agent.proc_root = proc  # type: ignore[attr-defined]
    agent.device_client_finder = lambda targets: [_holder("212320", "nvidia-smi")]  # type: ignore[method-assign]
    agent.sleep = lambda seconds: None  # type: ignore[attr-defined]

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        details = execute_with_final_ownership(agent, _signed(), agent.effect)

    assert details["effect"] is True, "the handler ran: the query did not refuse it"
    (receipt,) = details["physical_ownership_checks"]
    assert receipt["query_only_device_clients"] == [
        {
            "gpu_uuid": "GPU-a",
            "pid": "212320",
            "device": "/dev/nvidia0",
            "process_name": "nvidia-smi",
        }
    ]
    assert receipt["query_only_device_client_count"] == 1
    assert "ignored query-only device clients" in caplog.text
    assert classify_final_device_clients(agent, {"GPU-a"}) == (
        [],
        [_holder("212320", "nvidia-smi")],
    )


def test_a_persistent_holder_that_is_not_a_query_still_refuses(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _process(proc, "4244", "python", "/dev/nvidiactl", "/dev/nvidia0")
    agent = FakeNodeAgent()
    agent.proc_root = proc  # type: ignore[attr-defined]
    agent.device_client_finder = lambda targets: [_holder("4244", "python")]  # type: ignore[method-assign]
    agent.sleep = lambda seconds: None  # type: ignore[attr-defined]

    with pytest.raises(OwnershipRefused) as refused:
        execute_with_final_ownership(agent, _signed(), agent.effect)

    assert "effect" not in agent.calls
    assert refused.value.action_details["persistent_device_clients"][0]["pid"] == "4244"


def test_the_sampled_preflight_passes_over_a_query_and_names_it(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _process(proc, "212320", "nvidia-smi", "/dev/nvidiactl", "/dev/nvidia0")
    agent = node_action_executor(
        tmp_path,
        "query-only.db",
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        reset_enabled=True,
        runner=FakeRunner(),
        proc_root=str(proc),
        device_client_finder=lambda _: [_holder("212320", "nvidia-smi")],
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        device_client_samples=3,
        sleep=lambda _: None,
    )

    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert result.details["verified_no_gpu_clients"] is True
    (ignored,) = result.details["query_only_device_clients"]
    assert ignored["pid"] == "212320" and ignored["classification"] == "query-only"
    assert result.details["transient_device_clients"] == [ignored], (
        "the query travels with the transient holders, classified"
    )


def test_a_persistent_compute_holder_still_fails_the_preflight(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _process(
        proc, "4244", "python", "/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"
    )
    agent = node_action_executor(
        tmp_path,
        "compute-holder.db",
        allowed_operations={WorkflowOperation.VERIFY_NO_GPU_CLIENTS},
        reset_enabled=True,
        runner=FakeRunner(),
        proc_root=str(proc),
        device_client_finder=lambda _: [_holder("4244", "python")],
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        sleep=lambda _: None,
    )

    result = agent.execute(envelope(command(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)))

    assert result.status is NodeActionStatus.FAILED
    assert "GPU device clients are still active" in (result.error or "")
