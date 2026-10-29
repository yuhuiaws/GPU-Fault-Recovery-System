"""The DESTR-015 witness knows the Agent's passive GPU samples, and only those.

Live (GF-REGIONAL-DESTR-015, round 13, release 007595f77e11 on product
8995fc06): on HyperPod and on plain EKS the parallel RESET_GPU SUCCEEDED on
both nodes, and every witness refused "unknown NVIDIA invocation cannot prove
no action" with no reset classified (``actions == []``, ``calibration_execs ==
0``; the persistence split that runs before the classifier still counted its 3
queries and 1 release). That release's Node Agent runs a passive communication
monitor: in every 30-s epoch in which a control-plane lease names a running
managed job on the node -- this case's own PyTorchJob, before the injection and
again after RESTART_WORKLOAD, both inside the witness's arm..collect trace --
its read-only observation worker reads ``gpu_sample``'s two ``nvidia-smi``
queries, spellings no DESTR-015 trace had held before. The witness keeps no raw
argv, so the traces below are those windows' spawn sequences rebuilt from the
product's code path and each node's receipts, not copies of the live traces.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.gpu_activity import GPU_ACTIVITY_OPERATIONS
from gpu_fault.node_agent.operations.hung_summary import gpu_sample
from scripts.e2e.regional import late_ownership_trace as trace
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.probes import destr015_physical_probe as probe
from tests.regional.test_acceptance_physical_interval_alignment import (
    BASE,
    SECOND,
    quoted,
    trace_process,
)

NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
UNKNOWN = "unknown NVIDIA invocation cannot prove no action"
OVERLAP = "passive GPU sample overlapped a physical reset"
UTILIZATION = ("nvidia-smi", *probe.OBSERVATION_QUERY_ARGS[0])
CLIENTS = ("nvidia-smi", *probe.OBSERVATION_QUERY_ARGS[1])
QUERY = ("nvidia-smi", *probe.PERSISTENCE_QUERY_ARGS)
CALIBRATION = ("nvidia-smi", *probe.QUERY_ARGS)
#: Each live node's reset GPU and the bus id its witness read when armed.
LIVE_NODES = {
    "hyperpod-node-a": ("GPU-1a588e85-965e-fc68-40e4-225aaa3315b6", "00000000:59:00.0"),
    "hyperpod-node-b": ("GPU-5bd1b880-836d-eec4-44cf-48502ab91a20", "00000000:59:00.0"),
    "ip-10-0-43-5": ("GPU-9d21f50b-8c80-b4e0-8f24-eff9d453315f", "00000000:59:00.0"),
    "ip-10-0-32-162": ("GPU-80e1ad6c-ca4f-c21b-07ac-80d802782c62", "00000000:59:00.0"),
}
#: The reset's process interval in every window below, in seconds.
RESET = (20, 40)

Step = tuple[tuple[str, ...], int, int, int]


def window(gpu_uuid: str, bus: str) -> list[Step]:
    """A sample before the injection, QUIESCE (read, release, read), VERIFY,
    the reset, RESTORE's read-back, and two samples after RESTART_WORKLOAD."""

    return [
        (UTILIZATION, 1, 2, 0),
        (CLIENTS, 3, 4, 0),
        (QUERY, 10, 11, 0),
        (("nvidia-smi", "-i", bus, "-pm", "0"), 12, 13, 0),
        (QUERY, 14, 15, 0),
        (CALIBRATION, 16, 17, 0),
        (("nvidia-smi", "--gpu-reset", "-i", gpu_uuid), *RESET, 0),
        (QUERY, 50, 51, 0),
        (UTILIZATION, 60, 61, 0),
        (CLIENTS, 62, 63, 0),
        (UTILIZATION, 90, 91, 0),
        (CLIENTS, 92, 93, 0),
    ]


def render(steps: list[Step]) -> bytes:
    return b"".join(
        trace_process(400 + index, start, end, argv, code)
        for index, (argv, start, end, code) in enumerate(steps)
    )


def replaced(steps: list[Step], index: int, argv: tuple[str, ...]) -> list[Step]:
    _, start, end, code = steps[index]
    return [*steps[:index], (argv, start, end, code), *steps[index + 1 :]]


@pytest.mark.parametrize("node", sorted(LIVE_NODES))
def test_each_live_node_window_was_refused_and_now_proves_its_reset(node: str) -> None:
    gpu_uuid, bus = LIVE_NODES[node]
    # What refused it: the classifier reads neither sample spelling.
    for argv in (UTILIZATION, CLIENTS):
        events = trace.parse_exec_trace(
            render([(CALIBRATION, 1, 2, 0), (argv, 3, 4, 0)])
        )
        with pytest.raises(trace.UnknownInvocation, match=UNKNOWN) as refused:
            trace.physical_actions(
                events, nvidia_smi=NVIDIA_SMI, calibration_argv=CALIBRATION
            )
        assert refused.value.argv_sha256 == trace.argv_digest(argv)

    persistence: dict[str, int] = {}
    observation: dict[str, int] = {}
    resets, calibrations = probe.reset_events(
        render(window(gpu_uuid, bus)),
        executable=NVIDIA_SMI,
        gpu_uuid=gpu_uuid,
        pci_bus_id=bus,
        persistence=persistence,
        observation=observation,
    )

    assert [(item["gpu_uuid"], item["returncode"]) for item in resets] == [
        (gpu_uuid, 0)
    ]
    assert (resets[0]["started_ns"], resets[0]["ended_ns"]) == (
        BASE + RESET[0] * SECOND,
        BASE + RESET[1] * SECOND,
    )
    assert calibrations == 1
    assert persistence == {"queries": 3, "releases": 1}
    assert observation == {"queries": 6}


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(
            ("nvidia-smi", "--query-gpu=uuid,utilization.gpu,utilization.memory"),
            id="no-format",
        ),
        pytest.param(
            (
                "nvidia-smi",
                "--query-gpu=uuid,utilization.gpu,utilization.memory",
                "--format=csv",
            ),
            id="another-format",
        ),
        pytest.param(
            (
                "nvidia-smi",
                "--query-gpu=uuid,utilization.gpu,utilization.memory,clocks.sm",
                "--format=csv,noheader,nounits",
            ),
            id="another-field",
        ),
        pytest.param(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name",
                "--format=csv,noheader,nounits",
            ),
            id="the-hung-triage-spelling",
        ),
        pytest.param(
            (
                "nvidia-smi",
                "--format=csv,noheader,nounits",
                "--query-compute-apps=pid,gpu_uuid",
            ),
            id="reordered",
        ),
        pytest.param((*CLIENTS, "-i", "0"), id="target-selector"),
        pytest.param((*UTILIZATION, "-f", "/tmp/sample.csv"), id="file-output"),
        pytest.param((*UTILIZATION, "-lms", "100"), id="looping-sampler"),
        pytest.param((*CLIENTS, "-pm", "0"), id="state-change-riding-on-a-sample"),
    ],
)
def test_anything_else_still_cannot_prove_no_action(argv: tuple[str, ...]) -> None:
    gpu_uuid, bus = LIVE_NODES["hyperpod-node-a"]
    steps = replaced(window(gpu_uuid, bus), 9, argv)

    with pytest.raises(trace.UnknownInvocation, match=UNKNOWN) as refused:
        probe.reset_events(
            render(steps), executable=NVIDIA_SMI, gpu_uuid=gpu_uuid, pci_bus_id=bus
        )
    # The refusal names what it refused by digest, never by argv.
    assert refused.value.argv_sha256 == trace.argv_digest(argv)
    assert gpu_uuid not in str(refused.value) and argv[1] not in str(refused.value)


def test_a_sample_of_another_executable_is_not_taken_out() -> None:
    gpu_uuid, bus = LIVE_NODES["ip-10-0-43-5"]
    raw = render(window(gpu_uuid, bus)) + trace_process(
        499, 70, 71, UTILIZATION, executable="/usr/local/bin/nvidia-smi"
    )

    with pytest.raises(BoundaryDenied, match="unapproved executable"):
        probe.reset_events(
            raw, executable=NVIDIA_SMI, gpu_uuid=gpu_uuid, pci_bus_id=bus
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(25, 26, id="inside"),
        pytest.param(18, 22, id="across-the-start"),
        pytest.param(38, 45, id="across-the-end"),
        pytest.param(19, 41, id="around"),
    ],
)
def test_a_sample_that_overlaps_a_reset_is_refused(start: int, end: int) -> None:
    """The Agent's GPU-activity gate refuses a sample while a quiesce window is
    open, and VERIFY and the reset's re-checks fail closed while one is in
    flight: a sample beside the reset is a fence the Agent did not hold."""

    gpu_uuid, bus = LIVE_NODES["ip-10-0-32-162"]
    steps = [*window(gpu_uuid, bus), (CLIENTS, start, end, 0)]

    with pytest.raises(BoundaryDenied, match=OVERLAP):
        probe.reset_events(
            render(steps), executable=NVIDIA_SMI, gpu_uuid=gpu_uuid, pci_bus_id=bus
        )


def test_the_passive_sample_is_a_gated_gpu_activity() -> None:
    """The premise of the overlap refusal: the monitor admits each sample as a
    ``COLLECT_HUNG_SUMMARY`` GPU activity, which an open window refuses."""

    assert WorkflowOperation.COLLECT_HUNG_SUMMARY in GPU_ACTIVITY_OPERATIONS


def test_a_sample_that_only_touches_a_reset_is_not_an_overlap() -> None:
    gpu_uuid, bus = LIVE_NODES["ip-10-0-32-162"]
    steps = [
        *window(gpu_uuid, bus),
        (UTILIZATION, 18, RESET[0], 0),
        (CLIENTS, RESET[1], 42, 0),
    ]

    observation: dict[str, int] = {}
    resets, _ = probe.reset_events(
        render(steps),
        executable=NVIDIA_SMI,
        gpu_uuid=gpu_uuid,
        pci_bus_id=bus,
        observation=observation,
    )
    assert len(resets) == 1 and observation == {"queries": 8}


def test_a_sample_killed_on_its_timeout_is_still_only_a_read() -> None:
    """``gpu_sample`` gives each query 2 s; on a cold GPU the observation
    worker kills it, and the trace closes it with ``+++ killed by``."""

    gpu_uuid, bus = LIVE_NODES["hyperpod-node-b"]
    stamp = BASE // SECOND
    killed = (
        f"499 {stamp + 70}.000000000 execve({quoted(str(NVIDIA_SMI))}, "
        f"[{', '.join(quoted(item) for item in UTILIZATION)}], NULL) = 0\n"
        f"499 {stamp + 72}.000000000 +++ killed by SIGKILL +++\n"
    ).encode()

    observation: dict[str, int] = {}
    resets, _ = probe.reset_events(
        render(window(gpu_uuid, bus)) + killed,
        executable=NVIDIA_SMI,
        gpu_uuid=gpu_uuid,
        pci_bus_id=bus,
        observation=observation,
    )
    assert [item["returncode"] for item in resets] == [0]
    assert observation == {"queries": 7}


def test_the_witness_spellings_are_the_agent_passive_sample() -> None:
    """Pinned to the product: a sample that changes how it reads the GPUs fails
    here before it fails a live witness."""

    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    gpu_sample(SimpleNamespace(runner=run))

    assert calls == [list(UTILIZATION), list(CLIENTS)]
