"""The DESTR-015 witness knows the quiesce's persistence calls, and only those.

Live (GF-REGIONAL-DESTR-015 attempt 2, plain EKS, release ce56ac611c1d): the
product's parallel reset SUCCEEDED on both nodes, and both witnesses refused
"unknown NVIDIA invocation cannot prove no action". The traced Node Agent's
QUIESCE_GPU_SERVICES had read every GPU's persistence mode, released the
target's (``nvidia-smi -i 00000000:59:00.0 -pm 0``, recorded in each node's
QUIESCE receipt) and read the modes again; RESTORE_GPU_SERVICES read them once
more -- the target-scoped release the quiesce has made since it stopped
killing ``nvidia-persistenced``, which no DESTR-015 run had traced before. The
traces below are those nodes' spawn sequences, built from the receipts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gpu_fault.node_agent.quiesce_persistence import (
    PERSISTENCE_QUERY,
    TargetPersistence,
)
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.probes import destr015_physical_probe as probe
from tests.regional.test_acceptance_physical_interval_alignment import trace_process

NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
QUERY = ("nvidia-smi", *probe.PERSISTENCE_QUERY_ARGS)
CALIBRATION = ("nvidia-smi", *probe.QUERY_ARGS)
#: Each live node's reset GPU and its bus id, from its QUIESCE and RESET receipts.
LIVE_NODES = {
    "ip-10-0-37-86": ("GPU-8b50ba26-2fc6-86fe-fffc-46c010b9f042", "00000000:59:00.0"),
    "ip-10-0-42-179": ("GPU-f61dc01a-11d3-5c96-fdb5-76df542ca80e", "00000000:59:00.0"),
}
UNKNOWN = "unknown NVIDIA invocation cannot prove no action"


def window(
    gpu_uuid: str, release: tuple[str, ...] | None
) -> list[tuple[tuple[str, ...], int]]:
    """QUIESCE (read, release, read), VERIFY, RESET busy then applied, RESTORE."""

    reset = ("nvidia-smi", "--gpu-reset", "-i", gpu_uuid)
    steps: list[tuple[tuple[str, ...], int]] = [(QUERY, 0)]
    if release is not None:
        steps.append((release, 0))
    steps += [
        (QUERY, 0),
        (CALIBRATION, 0),
        (("nvidia-smi", "-q", "-x"), 0),
        (reset, 255),
        (CALIBRATION, 0),
        (reset, 0),
        (QUERY, 0),
    ]
    return steps


def trace(steps: list[tuple[tuple[str, ...], int]]) -> bytes:
    return b"".join(
        trace_process(300 + index, 2 * index + 1, 2 * index + 2, argv, code)
        for index, (argv, code) in enumerate(steps)
    )


@pytest.mark.parametrize("node", sorted(LIVE_NODES))
def test_each_live_node_window_was_refused_and_now_proves_its_resets(node: str) -> None:
    gpu_uuid, bus = LIVE_NODES[node]
    raw = trace(window(gpu_uuid, ("nvidia-smi", "-i", bus, "-pm", "0")))

    with pytest.raises(BoundaryDenied, match=UNKNOWN):
        probe.reset_events(raw, executable=NVIDIA_SMI, gpu_uuid=gpu_uuid)

    counts: dict[str, int] = {}
    resets, calibrations = probe.reset_events(
        raw,
        executable=NVIDIA_SMI,
        gpu_uuid=gpu_uuid,
        pci_bus_id=bus,
        persistence=counts,
    )
    assert [item["returncode"] for item in resets] == [255, 0]
    assert calibrations == 2
    assert counts == {"queries": 3, "releases": 1}


@pytest.mark.parametrize(
    "release",
    [
        pytest.param(
            ("nvidia-smi", "-i", "00000000:5A:00.0", "-pm", "0"),
            id="another-gpus-release",
        ),
        pytest.param(("nvidia-smi", "-pm", "0"), id="whole-node-release"),
        pytest.param(
            ("nvidia-smi", "-i", "00000000:59:00.0", "-pm", "1"),
            id="persistence-enable",
        ),
        pytest.param(
            ("nvidia-smi", "-i", "00000000:59:00.0", "-pm", "0", "-r"),
            id="extra-arguments",
        ),
        pytest.param(
            ("nvidia-smi", "--query-gpu=pci.bus_id,persistence_mode", "--format=csv"),
            id="another-query-spelling",
        ),
    ],
)
def test_anything_else_still_cannot_prove_no_action(release: tuple[str, ...]) -> None:
    gpu_uuid, bus = LIVE_NODES["ip-10-0-37-86"]
    raw = trace(window(gpu_uuid, release))

    with pytest.raises(BoundaryDenied, match=UNKNOWN):
        probe.reset_events(
            raw, executable=NVIDIA_SMI, gpu_uuid=gpu_uuid, pci_bus_id=bus
        )


def test_the_release_must_belong_to_the_quiesce_before_the_reset() -> None:
    gpu_uuid, bus = LIVE_NODES["ip-10-0-42-179"]
    steps = window(gpu_uuid, None)
    steps.append((("nvidia-smi", "-i", bus, "-pm", "0"), 0))

    with pytest.raises(BoundaryDenied, match="outside the quiesce before reset"):
        probe.reset_events(
            trace(steps), executable=NVIDIA_SMI, gpu_uuid=gpu_uuid, pci_bus_id=bus
        )


def test_the_bus_id_compares_without_regard_to_hex_case() -> None:
    gpu_uuid, _ = LIVE_NODES["ip-10-0-37-86"]
    raw = trace(window(gpu_uuid, ("nvidia-smi", "-i", "00000000:5a:00.0", "-pm", "0")))

    resets, _ = probe.reset_events(
        raw, executable=NVIDIA_SMI, gpu_uuid=gpu_uuid, pci_bus_id="00000000:5A:00.0"
    )
    assert [item["returncode"] for item in resets] == [255, 0]


def test_the_witness_spellings_are_the_agent_quiesce_calls(tmp_path: Path) -> None:
    """Pinned to the product: a quiesce that changes how it reads or releases
    persistence fails here before it fails a live witness."""

    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    persistence = TargetPersistence(run=run, holders=list, proc_root=tmp_path)
    persistence.stop_services(
        active_services=["gpu-fault-gpu-persistence"],
        pci_bus_ids=["00000000:59:00.0"],
        whole_node=False,
    )

    assert tuple(PERSISTENCE_QUERY[1:]) == probe.PERSISTENCE_QUERY_ARGS
    assert [argv for argv in calls if argv[0] == "nvidia-smi"] == [
        list(PERSISTENCE_QUERY),
        ["nvidia-smi", "-i", "00000000:59:00.0", *probe.PERSISTENCE_RELEASE_FLAGS],
    ]
