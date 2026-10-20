"""A busy fabric reset on a node whose GPUs went cold still gets its retry.

Live (gpu-eks-cluster, ip-10-0-45-43, p5en.48xlarge, 8x H200, AL2023 NVIDIA
AMI, GF-REGIONAL-DESTR-010): the quiesce had persistence mode off and the
display stack was all that kept the eight GPUs initialised. The fabric reset
was refused "In use by another client", the Agent unloaded ``nvidia_drm`` --
and the first ``nvidia-smi --query-compute-apps`` after it, the final
ownership boundary's pre-permit recheck before ``rmmod nvidia_modeset``, had to
initialise all eight GPUs. Its flat 15 s ran out, ``TimeoutExpired`` became
``OWNERSHIP_FINAL_SAFETY_CHANGED`` and a recoverable busy reset ended as a
FAILED remediation on an isolated node.

A read-only ``nvidia-smi`` query is now bounded for every GPU of the node (the
driver's own list). After the display-stack release the busy loop also leaves
the client re-read to the boundary, which makes it right before the retry's
spawn anyway: on eight cold GPUs each read costs about 18 s, and the duplicate
would have pushed the retry's checkpoint past the command's two minutes
(``OWNERSHIP_COMMAND_EXPIRED`` instead of the timeout). A client that appears
is still refused, and a driver that never answers still fails closed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.common import nvidia_gpu_count
from gpu_fault.node_agent.operations.clients import (
    NVIDIA_SMI_COLD_INIT_SECONDS_PER_GPU,
    NVIDIA_SMI_QUERY_TIMEOUT_SECONDS,
    nvidia_smi_query_timeout,
)
from gpu_fault.node_agent.protocol import NodeActionStatus
from tests.node_agent._cold_node import (
    COMMAND_TTL,
    GPU_BUS_IDS,
    GPU_UUIDS,
    INIT_SECONDS,
    QUERY_SECONDS,
    ColdNode,
    cold_agent,
    fabric_reset,
    gpu_reset,
)

READ_ONLY_QUERIES = (
    "nvidia-smi --query-compute-apps",
    "nvidia-smi -q -x",
    "nvidia-smi --query-gpu=uuid ",
)


def _spawned_resets(node: ColdNode) -> list[float]:
    return [
        item["at"] for item in node.calls if item["command"] == "nvidia-smi --gpu-reset"
    ]


def test_the_busy_fabric_reset_retries_on_eight_cold_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = ColdNode(tmp_path)
    agent = cold_agent(tmp_path, node, monkeypatch)

    result = agent.execute(fabric_reset())

    assert result.status is NodeActionStatus.SUCCEEDED, (
        result.error,
        result.details.get("refusal_cause"),
        node.killed(),
    )
    assert [item for item in node.killed() if item.startswith("nvidia-smi")] == [], (
        "no read-only query was killed by its bound"
    )
    assert result.details["reset_attempts"] == 2
    assert result.details["reset_busy_refusals"] == 1
    assert node.resets_spawned == 1
    first, retry = _spawned_resets(node)
    assert retry < COMMAND_TTL.total_seconds(), (
        f"the retry was spawned {retry} s after the command was issued; its "
        "checkpoint must still lie inside the command's lifetime"
    )
    assert [item["outcome"] for item in result.details["physical_calls"]] == [
        "busy",
        "released",
        "released",
        "reset",
    ]
    assert len(result.details["physical_ownership_checks"]) == 4


def test_each_read_only_query_is_bounded_for_every_gpu_of_the_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = ColdNode(tmp_path)
    agent = cold_agent(tmp_path, node, monkeypatch)

    agent.execute(fabric_reset())

    cold = QUERY_SECONDS + INIT_SECONDS * len(GPU_BUS_IDS)
    for prefix in READ_ONLY_QUERIES:
        timeouts = node.timeouts(prefix)
        assert timeouts, prefix
        assert min(timeouts) >= 2 * cold, (prefix, timeouts, cold)
    assert nvidia_smi_query_timeout(len(GPU_BUS_IDS)) == (
        NVIDIA_SMI_QUERY_TIMEOUT_SECONDS
        + NVIDIA_SMI_COLD_INIT_SECONDS_PER_GPU * len(GPU_BUS_IDS)
    )


def test_the_bound_follows_the_driver_s_gpu_list(tmp_path: Path) -> None:
    node = ColdNode(tmp_path)
    assert nvidia_gpu_count(node.proc) == len(GPU_BUS_IDS)
    for bus_id in GPU_BUS_IDS[1:]:
        (node.proc / "driver" / "nvidia" / "gpus" / bus_id).rmdir()
    assert nvidia_gpu_count(node.proc) == 1
    assert nvidia_smi_query_timeout(1) == (
        NVIDIA_SMI_QUERY_TIMEOUT_SECONDS + NVIDIA_SMI_COLD_INIT_SECONDS_PER_GPU
    )
    assert nvidia_gpu_count(tmp_path / "no-driver") is None
    assert nvidia_smi_query_timeout(None) == nvidia_smi_query_timeout(8), (
        "a driver list that cannot be read bounds for the largest node shape"
    )


def test_a_driver_that_never_answers_still_refuses_the_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = ColdNode(tmp_path)
    agent = cold_agent(tmp_path, node, monkeypatch)
    node.after_unload_of(None, node.hang)

    result = agent.execute(fabric_reset())

    assert result.status is NodeActionStatus.FAILED
    assert result.details["reason"] == "OWNERSHIP_FINAL_SAFETY_CHANGED"
    assert result.details["refusal_cause"] == "TimeoutExpired"
    assert node.resets_spawned == 0, "fail closed: the retry is never spawned"
    assert result.details["reset_verdict"] == "not-applied"


def test_a_compute_client_that_appears_after_the_release_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = ColdNode(tmp_path)
    agent = cold_agent(tmp_path, node, monkeypatch)
    node.after_unload_of("nvidia_modeset", lambda: node.start_compute_app(GPU_UUIDS[3]))

    result = agent.execute(fabric_reset())

    assert result.status is NodeActionStatus.FAILED
    assert result.details["safety_rejection"] is True
    assert result.details["refusal_cause"] == "GpuClientsActive"
    assert node.resets_spawned == 0
    assert [item["outcome"] for item in result.details["physical_calls"]] == [
        "busy",
        "released",
        "released",
        "refused: OWNERSHIP_FINAL_SAFETY_CHANGED",
    ]


def test_a_device_holder_that_appears_after_the_release_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = ColdNode(tmp_path)
    agent = cold_agent(tmp_path, node, monkeypatch)
    node.after_unload_of("nvidia_modeset", lambda: node.open_device(3))

    result = agent.execute(fabric_reset())

    assert result.status is NodeActionStatus.FAILED
    assert result.details["reason"] == "OWNERSHIP_FINAL_CLIENTS_CHANGED"
    assert result.details["persistent_device_clients"][0]["pid"] == "4242"
    assert node.resets_spawned == 0


def test_without_the_boundary_the_loop_still_verifies_before_the_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ownership boundary guards the spawn: the busy loop's own sampled
    verification is the only check before the retry, and it stays in full."""

    node = ColdNode(tmp_path)
    agent = cold_agent(tmp_path, node, monkeypatch, guarded=False)
    node.after_unload_of("nvidia_modeset", lambda: node.start_compute_app(GPU_UUIDS[3]))

    result = agent.execute(fabric_reset(guarded=False))

    assert result.status is NodeActionStatus.FAILED
    assert "GPU compute clients are still active" in (result.error or "")
    assert node.resets_spawned == 0


def test_a_refusal_inside_the_release_lists_the_unload_it_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live shape: refused at the checkpoint of the second unload.
    ``physical_calls[i]`` answers ``physical_ownership_checks[i]`` -- the busy
    reset, the ``rmmod nvidia_drm`` that ran, and the refused ``rmmod
    nvidia_modeset`` -- instead of the busy reset alone."""

    node = ColdNode(tmp_path)
    agent = cold_agent(tmp_path, node, monkeypatch)
    node.after_unload_of(None, node.hang)

    result = agent.execute(fabric_reset())

    assert result.status is NodeActionStatus.FAILED
    checks = result.details["physical_ownership_checks"]
    calls = result.details["physical_calls"]
    assert len(checks) == 2, checks
    assert [item["command"] for item in calls] == [
        "nvidia-smi --gpu-reset",
        "rmmod nvidia_drm",
        "rmmod nvidia_modeset",
    ]
    assert [item["outcome"] for item in calls] == [
        "busy",
        "released",
        "refused: OWNERSHIP_FINAL_SAFETY_CHANGED",
    ]


@pytest.mark.parametrize(
    "operation",
    [WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES, WorkflowOperation.RESET_GPU],
    ids=["fabric", "gpu"],
)
def test_the_boundary_reads_the_clients_the_busy_loop_no_longer_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: WorkflowOperation
) -> None:
    """The busy loop skips its own client read after the release only because
    the boundary makes the same read before the retry's spawn -- for both
    resets. A compute app that starts after the release is refused there."""

    node = ColdNode(tmp_path)
    agent = cold_agent(tmp_path, node, monkeypatch, operation=operation)
    node.after_unload_of("nvidia_modeset", lambda: node.start_compute_app(GPU_UUIDS[7]))
    signed = (
        fabric_reset()
        if operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
        else gpu_reset(GPU_UUIDS[7], guarded=True)
    )

    result = agent.execute(signed)

    assert result.status is NodeActionStatus.FAILED
    assert result.details["safety_rejection"] is True
    assert result.details["refusal_cause"] == "GpuClientsActive"
    assert result.details["reset_verdict"] == "not-applied"
    assert node.resets_spawned == 0, "the retry is never spawned"
