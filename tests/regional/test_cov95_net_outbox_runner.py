"""Execute NET-008 phases with real verdicts and a stateful fake host."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_net008_outbox_dead_letter as runner
from tests.regional._cov95_collect_net import (  # noqa: F401
    StopLoop,
    no_external_effects,
)
from tests.regional._cov95_net_outbox import OutboxHost


@pytest.fixture
def host(monkeypatch: Any) -> OutboxHost:
    value = OutboxHost()
    monkeypatch.setattr(runner, "time", value.clock)
    return value


def execute(host: OutboxHost, tmp_path: Path, **changes: Any) -> Any:
    settings = SimpleNamespace(
        node=host.node, endpoint_host="control.invalid", **changes
    )
    return runner.execute(
        settings, host, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )


@pytest.mark.parametrize(
    "fallback,pending", [(False, False), (True, False), (False, True)]
)
def test_net008_runs_three_distinct_phases_then_purges_only_its_seed(
    host: OutboxHost, tmp_path: Path, fallback: bool, pending: bool
) -> None:
    host.fallback, host.pending = fallback, pending
    result = execute(host, tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert host.waits == ["transient", "replay", "blackout-replay"]
    commands = [call[0] for call in host.calls]
    assert commands.index("stop-unit") < commands.index("seed-outbox-record")
    assert commands.index("seed-outbox-record") < commands.index("close-window")
    assert (
        commands.index("close-window")
        < commands.index("block")
        < commands.index("unblock")
    )
    assert commands[-3:] == ["stop-unit", "purge-outbox-record", "start-unit"]
    assert (
        not host.window and not host.blocked and not host.stopped and not host.seeded
    ), "successful phases must release every owned temporary state"
    assert result["stages"]["stream_identity"] == []
    assert result["dead_letter"]["requeued"]["stats"]["replayable"] == 1


@pytest.mark.parametrize(
    "problem", ["window", "transient", "dead-letter", "evidence", "listing"]
)
def test_failed_phase_prevents_later_injections(
    host: OutboxHost, tmp_path: Path, problem: str
) -> None:
    host.problem = problem
    result = execute(host, tmp_path)
    assert result["verdict"] == "FAIL"
    assert not any(call[0] == "block" for call in host.calls), (
        "phase failure must stop blackout injection"
    )
    if problem in {"window", "transient"}:
        assert not any(call[0] == "seed-outbox-record" for call in host.calls), (
            "phase A failure must stop phase B"
        )
    if problem == "listing":
        assert not any("requeue-dead" in call for call in host.calls), (
            "bad listing must not authorize requeue"
        )
    assert not host.window and not host.stopped, (
        "failure must close the window and restart its stopped unit"
    )


@pytest.mark.parametrize(
    "command",
    [
        "open-window",
        "stop-unit",
        "seed-outbox-record",
        "close-window",
        "start-unit",
        "outbox:list",
        "outbox:requeue-dead",
        "block",
        "unblock",
        "purge-outbox-record",
    ],
)
def test_lost_ack_and_command_failures_still_attempt_all_owned_cleanup(
    host: OutboxHost, tmp_path: Path, command: str
) -> None:
    host.failures[command] = [RuntimeError("lost ACK")]
    result = execute(host, tmp_path)
    assert result["verdict"] == "FAIL"
    assert result["errors"], "failed command needs observable failure evidence"
    commands = [call[0] for call in host.calls]
    if command == "stop-unit":
        assert commands[-1] == "start-unit", "stop intent precedes its ACK"
    if command in {
        "seed-outbox-record",
        "close-window",
        "start-unit",
        "outbox:list",
        "outbox:requeue-dead",
        "block",
        "unblock",
    }:
        assert "purge-outbox-record" in commands, (
            "seed intent must survive an ACK failure"
        )
        assert commands[-1] == "start-unit", "purge failure cannot suppress restart"
    if command == "block":
        assert "unblock" in commands, "block intent must survive an ACK failure"


@pytest.mark.parametrize(
    "problem", ["stop-unit", "start-unit", "no-ip", "block", "unblock"]
)
def test_unconfirmed_host_transition_fails_and_keeps_independent_cleanup(
    host: OutboxHost, tmp_path: Path, problem: str
) -> None:
    host.problem = problem
    result = execute(host, tmp_path)
    assert result["verdict"] == "FAIL"
    assert result["errors"], "unconfirmed host state must not produce a PASS"
    if problem in {"block", "unblock"}:
        assert any(call[0] == "unblock" for call in host.calls), (
            "restore blocked egress even after failed verification"
        )


def test_cleanup_attempts_unblock_purge_and_restart_independently(
    host: OutboxHost, tmp_path: Path
) -> None:
    host.failures["block"] = [RuntimeError("block ACK lost")]
    host.failures["unblock"] = [RuntimeError("unblock unavailable")]
    host.failures["purge-outbox-record"] = [RuntimeError("purge unavailable")]
    host.failures["start-unit"] = [None, RuntimeError("cleanup restart failed")]
    result = execute(host, tmp_path)
    assert result["verdict"] == "FAIL"
    assert any("unblock:" in error for error in result["stages"]["cleanup"]), (
        "unblock failure must be recorded"
    )
    assert any("purge:" in error for error in result["stages"]["cleanup"]), (
        "purge must still be attempted after unblock fails"
    )
    assert any("start-unit:" in error for error in result["stages"]["cleanup"]), (
        "restart must still be attempted after purge fails"
    )


def test_abort_after_seed_intent_still_attempts_purge_and_restart(
    host: OutboxHost, tmp_path: Path
) -> None:
    host.failures["seed-outbox-record"] = [StopLoop()]
    with pytest.raises(StopLoop):
        execute(host, tmp_path)
    commands = [call[0] for call in host.calls]
    assert "purge-outbox-record" in commands
    assert commands[-1] == "start-unit"


@pytest.mark.parametrize("problem", ["depth", "inventory", "endpoint"])
def test_prepare_refuses_unknown_baseline_before_window(
    host: OutboxHost, tmp_path: Path, problem: str
) -> None:
    if problem == "depth":
        host.depth = None
    if problem == "inventory":
        host.inventory = []
    settings = SimpleNamespace(
        node=host.node, endpoint_host="" if problem == "endpoint" else "control.invalid"
    )
    with pytest.raises(runner.RegionalFixtureError):
        runner.execute(
            settings, host, tmp_path, 1, datetime.now(timezone.utc) + timedelta(hours=1)
        )
    assert not any(call[0] == "open-window" for call in host.calls), (
        "invalid baseline must not mutate"
    )


def test_expired_case_stops_before_first_phase(
    host: OutboxHost, tmp_path: Path
) -> None:
    result = runner.execute(
        SimpleNamespace(node=host.node, endpoint_host="control.invalid"),
        host,
        tmp_path,
        1,
        datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    assert result["verdict"] == "FAIL"
    assert [call[0] for call in host.calls] == ["resolve"]


def test_main_delegates_to_shared_window_lifecycle(monkeypatch: Any) -> None:
    calls = []
    monkeypatch.setattr(
        runner, "run_window_case", lambda **kwargs: calls.append(kwargs) or 7
    )
    assert runner.main() == 7
    assert calls[0]["execute"] is runner.execute
    details = calls[0]["plan_details"](
        SimpleNamespace(node="node-a"), {"predecessor": {"valid": True}}
    )
    assert details["predecessor"]["valid"] is True
