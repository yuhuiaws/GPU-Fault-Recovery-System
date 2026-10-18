"""Every Agent reports the version of its node's running DCGM engine.

NVIDIA requires the exporter's DCGM client to be at least the host engine's
version; ``status --full`` can only compare the two if the engine's version is
known. The heartbeat's unit map is the carrier batch 122 used for the engine's
state: one more entry, ``nvidia-dcgm/version=<v>``, which an older control
plane verifies (the signature covers it) and drops, and which an older Agent
never sends -- no protocol or version change. ``dcgmi -v`` costs a connection
to the engine, so it is asked only when the engine has just become active and
on the enablement refresh tick, never on every heartbeat.
"""

from __future__ import annotations

from collections.abc import Callable
from subprocess import CompletedProcess, TimeoutExpired

import pytest

from gpu_fault import collector_requirements
from gpu_fault.dcgm_versions import hostengine_version
from gpu_fault.fleet import AgentRecord, SignedAgentHeartbeat
from gpu_fault.node_agent.heartbeat import CollectorServiceStates
from tests.fleet._support import heartbeat, registry, signed

ENGINE = "nvidia-dcgm"
VERSION_KEY = "nvidia-dcgm/version=4.7.0"
RUNNING = {"active": "active", "enabled": "enabled"}
DCGMI_V = (
    "\nLocal build info:\nVersion : 4.7.0\nBuild ID : 3\n\n"
    "Hostengine build info:\nVersion : 4.7.0\nBuild ID : 3\n"
)


class Node:
    """``systemctl`` and ``dcgmi`` of one node, with the engine's state settable."""

    def __init__(self) -> None:
        self.engine = "active"
        self.dcgmi: Callable[[], str] = lambda: DCGMI_V
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **_kwargs: object) -> CompletedProcess[str]:
        self.calls.append(list(argv))
        if argv[0] == "dcgmi":
            return CompletedProcess(argv, 0, self.dcgmi(), "")
        if argv[1] == "is-active":
            state = self.engine if argv[2] == ENGINE else "active"
            return CompletedProcess(argv, 0, state + "\n", "")
        return CompletedProcess(argv, 0, "enabled\n", "")

    def dcgmi_calls(self) -> int:
        return sum(1 for argv in self.calls if argv[0] == "dcgmi")


def _states(node: Node, refresh_every: int = 30) -> CollectorServiceStates:
    return CollectorServiceStates(
        units=lambda: [ENGINE, "gpu-fault-kernel-collector"],
        runner=node,
        refresh_every=refresh_every,
    )


def test_the_agent_reports_the_running_engines_version_beside_its_state() -> None:
    node = Node()

    services = _states(node)()

    assert set(services) == {ENGINE, "gpu-fault-kernel-collector", VERSION_KEY}
    assert services[VERSION_KEY] == services[ENGINE], (
        "the entry carries the unit's own state, so a reader of the map sees a "
        "running engine"
    )
    assert node.calls.count(["dcgmi", "-v"]) == 1, node.calls
    assert hostengine_version(services) == "4.7.0"


def test_dcgmi_is_asked_on_activation_and_refresh_ticks_only() -> None:
    node = Node()
    states = _states(node, refresh_every=4)

    for _ in range(4):
        states()
    assert node.dcgmi_calls() == 1, "one call for the first active tick, then cached"

    states()  # the next refresh tick: every 4th after the first
    assert node.dcgmi_calls() == 2, "re-read with the enablement refresh"

    node.engine = "inactive"
    quiet = states()
    assert VERSION_KEY not in quiet, "a stopped engine reports no version"
    assert node.dcgmi_calls() == 2, "and is never asked"

    node.engine = "active"
    node.dcgmi = lambda: DCGMI_V.rsplit("4.7.0", 1)[0] + "4.6.1\nBuild ID : 3\n"
    restarted = states()
    assert node.dcgmi_calls() == 3, "a restarted engine (a package change) is re-read"
    assert hostengine_version(restarted) == "4.6.1", restarted


@pytest.mark.parametrize(
    "answer",
    [
        lambda: "",  # DCGM 4.6: an unreachable engine prints nothing
        lambda: "\nLocal build info:\nVersion : 4.7.0\n\nHostengine build info "
        "not available: could not connect to hostengine at localhost\n",
    ],
)
def test_no_version_is_reported_when_dcgmi_cannot_read_it(
    answer: Callable[[], str],
) -> None:
    node = Node()
    node.dcgmi = answer

    services = _states(node)()

    assert VERSION_KEY not in services and hostengine_version(services) is None, (
        services
    )
    assert services[ENGINE].active.value == "active", "the state is still reported"


def test_a_dcgmi_that_hangs_costs_one_bounded_call_and_no_entry() -> None:
    node = Node()

    def hang() -> str:
        raise TimeoutExpired(["dcgmi", "-v"], 5)

    node.dcgmi = hang
    states = _states(node)

    first = states()
    second = states()

    assert VERSION_KEY not in first and VERSION_KEY not in second
    assert node.dcgmi_calls() == 1, "not retried before the next trigger"


def test_the_record_keeps_one_version_entry() -> None:
    record = registry().register(
        signed(
            heartbeat(
                "node-a", collector_services={ENGINE: RUNNING, VERSION_KEY: RUNNING}
            )
        )
    )

    assert set(record.collector_services) == {ENGINE, VERSION_KEY}
    assert hostengine_version(record.collector_services) == "4.7.0"

    ambiguous = registry().register(
        signed(
            heartbeat(
                "node-a",
                collector_services={
                    ENGINE: RUNNING,
                    VERSION_KEY: RUNNING,
                    "nvidia-dcgm/version=4.6.1": RUNNING,
                },
            )
        )
    )
    assert set(ambiguous.collector_services) == {ENGINE}, (
        "a map naming two versions reports none"
    )


def _older_control_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    """The record filter of a control plane before the version entry."""

    monkeypatch.setattr(collector_requirements, "RECORDED_UNIT_SHAPES", ())


def test_an_older_control_plane_verifies_the_version_and_drops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _older_control_plane(monkeypatch)
    envelope = signed(
        heartbeat("node-a", collector_services={ENGINE: RUNNING, VERSION_KEY: RUNNING})
    )
    received = SignedAgentHeartbeat.model_validate_json(envelope.model_dump_json())

    record = registry().register(received)

    assert VERSION_KEY in received.heartbeat.collector_services, (
        "the heartbeat model takes any unit, so the signature still verifies"
    )
    assert set(record.collector_services) == {ENGINE}, "the record drops it"


def test_a_record_with_the_version_reads_back_under_an_older_control_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written = registry().register(
        signed(
            heartbeat(
                "node-a", collector_services={ENGINE: RUNNING, VERSION_KEY: RUNNING}
            )
        )
    )
    document = written.model_dump_json()
    _older_control_plane(monkeypatch)

    read = AgentRecord.model_validate_json(document)

    assert set(read.collector_services) == {ENGINE}, (
        "a rollback reads the record and keeps the engine's state"
    )
