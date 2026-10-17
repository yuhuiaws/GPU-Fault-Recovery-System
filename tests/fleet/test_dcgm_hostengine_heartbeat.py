"""Every Agent reports its node's host DCGM engine in each heartbeat.

The heartbeat's unit map (``collector_services``) has always taken any unit --
the signature covers what the Agent sent -- and an Agent record keeps the
units its control plane knows (``RECORDED_UNITS``). So the engine's unit
``nvidia-dcgm`` rides beside the collectors without a protocol change: an older
control plane verifies the heartbeat and drops the entry, a newer one keeps
it for ``status``, and an older Agent simply has no entry. A record written
with it reads back under the older filter too (a rollback).
"""

from __future__ import annotations

from subprocess import CompletedProcess

import pytest

from gpu_fault import collector_requirements
from gpu_fault.collector_kinds import COLLECTOR_SYSTEMD_UNITS
from gpu_fault.fleet import AgentRecord, SignedAgentHeartbeat, sign_agent_heartbeat
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.heartbeat import (
    AgentHeartbeatReporter,
    CollectorServiceStates,
)
from tests.fleet._support import (
    ARTIFACT,
    CONFIG,
    NOW,
    SECRET,
    heartbeat,
    registry,
    signed,
)

ENGINE = "nvidia-dcgm"
#: Reported the same way since batch 127: VALIDATE_GPU requires it running on a
#: Hopper NVSwitch node whose systemd enables it.
FABRIC_MANAGER = "nvidia-fabricmanager"
FAILED = {"active": "failed", "enabled": "enabled"}
COLLECTOR = "gpu-fault-kernel-collector"
RUNNING = {"active": "active", "enabled": "enabled"}


def _older_control_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    """The record filter of a control plane that knows only its collectors."""

    monkeypatch.setattr(
        collector_requirements,
        "RECORDED_UNITS",
        frozenset(COLLECTOR_SYSTEMD_UNITS.values()),
    )


def test_the_agent_reports_the_engine_beside_its_collectors() -> None:
    asked: list[list[str]] = []

    def systemctl(argv, **_kwargs):
        asked.append(list(argv))
        failed = argv[2] == ENGINE and argv[1] == "is-active"
        return CompletedProcess(argv, 0, "failed\n" if failed else "active\n", "")

    sent: list[SignedAgentHeartbeat] = []
    reporter = AgentHeartbeatReporter(
        control_plane_url="http://control-plane",
        secret=SECRET,
        cluster_id="cluster-a",
        node_id="node-a",
        endpoint="http://node-a:9099",
        agent_version="0.9.0",
        artifact_sha256=ARTIFACT,
        policy_version="catalog-a",
        runtime_profile_version="profile-a",
        config_digest=CONFIG,
        allowed_operations=[WorkflowOperation.RESET_GPU],
        collector_status_provider=CollectorServiceStates(runner=systemctl),
        boot_id="boot-a",
        sender=lambda _url, envelope: sent.append(envelope) or {},
        now=lambda: NOW,
    )

    reporter.report_once()

    (envelope,) = sent
    services = envelope.heartbeat.collector_services
    assert set(services) == {*COLLECTOR_SYSTEMD_UNITS.values(), ENGINE, FABRIC_MANAGER}
    assert services[ENGINE].active.value == "failed"
    assert services[FABRIC_MANAGER].active.value == "active", (
        "Fabric Manager's unit rides in the same map, asked the same way"
    )
    assert ["systemctl", "is-active", ENGINE] in asked
    assert ["systemctl", "is-active", FABRIC_MANAGER] in asked
    assert envelope.signature == sign_agent_heartbeat(envelope.heartbeat, SECRET), (
        "the signature covers the engine's entry"
    )


def test_the_record_keeps_the_engine_and_still_drops_unknown_units() -> None:
    value = heartbeat(
        "node-a",
        collector_services={
            ENGINE: FAILED,
            FABRIC_MANAGER: FAILED,
            COLLECTOR: RUNNING,
            "gpu-fault-future-collector": RUNNING,
        },
    )

    record = registry().register(signed(value))

    assert set(record.collector_services) == {ENGINE, FABRIC_MANAGER, COLLECTOR}
    assert record.collector_services[ENGINE].active.value == "failed"
    assert record.collector_services[FABRIC_MANAGER].active.value == "failed", (
        "VALIDATE_GPU reads Fabric Manager's state off the record"
    )
    assert collector_requirements.RECORDED_UNITS == frozenset(
        {*COLLECTOR_SYSTEMD_UNITS.values(), ENGINE, FABRIC_MANAGER}
    )


def test_an_older_control_plane_verifies_the_report_and_drops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _older_control_plane(monkeypatch)
    envelope = signed(
        heartbeat("node-a", collector_services={ENGINE: FAILED, COLLECTOR: RUNNING})
    )
    # What the control plane parses from the wire.
    received = SignedAgentHeartbeat.model_validate_json(envelope.model_dump_json())

    record = registry().register(received)

    assert ENGINE in received.heartbeat.collector_services, (
        "the heartbeat model takes any unit, so the signature still verifies"
    )
    assert set(record.collector_services) == {COLLECTOR}, "the record drops it"


def test_an_older_agent_simply_has_no_entry() -> None:
    record = registry().register(
        signed(heartbeat("node-a", collector_services={COLLECTOR: RUNNING}))
    )

    assert ENGINE not in record.collector_services


def test_a_record_with_the_engine_reads_back_under_an_older_control_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet = registry()
    written = fleet.register(
        signed(heartbeat("node-a", collector_services={ENGINE: FAILED}))
    )
    document = written.model_dump_json()
    _older_control_plane(monkeypatch)

    read = AgentRecord.model_validate_json(document)
    again = fleet.register(
        signed(
            heartbeat(
                "node-a",
                observed_at=NOW,
                collector_services={ENGINE: FAILED, COLLECTOR: RUNNING},
            )
        )
    )

    assert ENGINE not in read.collector_services, "a rollback reads the record"
    assert set(again.collector_services) == {COLLECTOR}, (
        "and keeps registering the same Agent over its stored record"
    )
