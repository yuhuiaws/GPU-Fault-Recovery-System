"""NET-003 verdicts: the control plane committed, the client lost the response."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_net003_result_retry as net003

T0 = datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc)
NOTIFICATION_ID = "notification-net003-x"


def _ready(**overrides: Any) -> dict[str, Any]:
    ready = {
        "drop_rollback_seconds": net003.DROP_ROLLBACK_SECONDS,
        "http_timeout_seconds": float(net003.HTTP_TIMEOUT_SECONDS),
        "lease_seconds": net003.LEASE_SECONDS,
        "result_connection_reset": True,
        "response_loss_mode": net003.RESPONSE_LOSS_MODE,
        "response_quiet_seconds": net003.RESPONSE_QUIET_SECONDS,
        "replay_delay_seconds": net003.REPLAY_DELAY_SECONDS,
        "terminal_result_replays": 1,
    }
    ready.update(overrides)
    return ready


def _evidence(**overrides: Any) -> dict[str, Any]:
    committed_at = T0 + timedelta(seconds=7)
    replay_sent_at = committed_at + timedelta(seconds=net003.REPLAY_DELAY_SECONDS)
    evidence: dict[str, Any] = {
        "ready": _ready(),
        "leased": {
            "status": "LEASED",
            "lease_expires_at": (
                T0 + timedelta(seconds=net003.LEASE_SECONDS)
            ).isoformat(),
        },
        "committed": {"status": "SUCCEEDED", "lease_expires_at": None},
        "final": {
            "status": "SUCCEEDED",
            "updated_at": committed_at.isoformat(),
            "result_details": {"cached": False, "notification_id": NOTIFICATION_ID},
        },
        "result_interrupted": {
            "first_post_succeeded": False,
            "exception": "RemoteDisconnected",
        },
        "result_replays": {
            "count": 1,
            "replay_sent_at_epoch": replay_sent_at.timestamp(),
            "responses": [
                {"status": "SUCCEEDED", "updated_at": committed_at.isoformat()}
            ],
        },
        "executor_state": {
            "claimed_total": 1,
            "reported_failures": 0,
            "unexpected_failures": 0,
            "lease_renewal_failures": 0,
        },
        "ledger": {"physical_count": 1, "keys": ["workflow/0/FREEZE_EVIDENCE"]},
        "logs": f"WARNING {net003.LOST_RESPONSE_LOG}: RemoteDisconnected: closed",
        "rollback_triggered": False,
        "drop_observed": {
            "connection_reset": True,
            "request_forwarded": True,
            "upstream_response_bytes": 2048,
        },
        "notification_id": NOTIFICATION_ID,
        "notification_baseline": {
            "objects": {
                "notification": {"count": 1},
                "notification_delivery": {"count": 1},
            },
            "dedup_link_count": 1,
        },
        "notification_final": {
            "objects": {
                "notification": {"count": 1},
                "notification_delivery": {"count": 1},
                "notification_result": {"count": 1, "status": "SKIPPED"},
            },
            "dedup_link_count": 1,
        },
    }
    evidence.update(overrides)
    return evidence


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


def test_the_intended_run_passes() -> None:
    assert net003.net003_errors(**_evidence()) == []


def test_a_reset_before_forwarding_is_the_old_degenerate_case_and_fails() -> None:
    errors = net003.net003_errors(
        **_evidence(
            drop_observed={
                "connection_reset": True,
                "request_forwarded": False,
                "upstream_response_bytes": 0,
            }
        )
    )
    assert "before forwarding" in _text(errors)
    assert "before the control plane answered" in _text(errors)


def test_the_control_plane_must_have_committed_before_the_reset() -> None:
    errors = net003.net003_errors(
        **_evidence(committed={"status": "LEASED", "lease_expires_at": "2026"})
    )
    assert "did not commit the first result" in _text(errors)
    assert "still carries a lease" in _text(errors)


def test_a_reclaim_means_the_first_result_was_not_held() -> None:
    errors = net003.net003_errors(
        **_evidence(
            executor_state={
                "claimed_total": 2,
                "reported_failures": 0,
                "unexpected_failures": 0,
                "lease_renewal_failures": 0,
            },
            final={
                "status": "SUCCEEDED",
                "updated_at": (T0 + timedelta(seconds=7)).isoformat(),
                "result_details": {"cached": True, "notification_id": NOTIFICATION_ID},
            },
        )
    )
    assert "claimed again" in _text(errors)
    assert "not the first physical execution" in _text(errors)


def test_the_commit_must_predate_the_replay_by_half_the_delay() -> None:
    committed_at = T0 + timedelta(seconds=7)
    errors = net003.net003_errors(
        **_evidence(
            result_replays={
                "count": 1,
                "replay_sent_at_epoch": committed_at.timestamp() + 0.4,
                "responses": [
                    {"status": "SUCCEEDED", "updated_at": committed_at.isoformat()}
                ],
            }
        )
    )
    assert "not attributable to the first post" in _text(errors)


def test_a_replay_that_rewrote_the_command_fails() -> None:
    committed_at = T0 + timedelta(seconds=7)
    rewritten = committed_at + timedelta(seconds=net003.REPLAY_DELAY_SECONDS)
    errors = net003.net003_errors(
        **_evidence(
            result_replays={
                "count": 1,
                "replay_sent_at_epoch": rewritten.timestamp(),
                "responses": [
                    {"status": "SUCCEEDED", "updated_at": rewritten.isoformat()}
                ],
            }
        )
    )
    assert "rewrote the committed command" in _text(errors)


def test_exactly_one_replay_is_required() -> None:
    evidence = _evidence()
    evidence["result_replays"]["count"] = 2
    evidence["result_replays"]["responses"] = (
        evidence["result_replays"]["responses"] * 2
    )
    assert "replayed exactly once" in _text(net003.net003_errors(**evidence))
    assert "terminal result replay count" in _text(
        net003.net003_errors(**_evidence(ready=_ready(terminal_result_replays=2)))
    )


def test_a_first_post_that_got_its_answer_fails() -> None:
    errors = net003.net003_errors(
        **_evidence(result_interrupted={"first_post_succeeded": True})
    )
    assert "did not see a transport error" in _text(errors)


def test_a_409_or_a_missing_lost_response_log_fails() -> None:
    assert "409 path" in _text(
        net003.net003_errors(**_evidence(logs="rejected request (409) ..."))
    )
    assert "lost-response" in _text(net003.net003_errors(**_evidence(logs="")))


def test_timing_keeps_the_exchange_inside_the_first_renewal_interval() -> None:
    assert net003.timing_errors() == []
    exchange = (
        net003.ACTION_SECONDS
        + net003.RESPONSE_QUIET_SECONDS
        + net003.REPLAY_DELAY_SECONDS
    )
    assert exchange + 10 < net003.renewal_interval_seconds(net003.LEASE_SECONDS)


def test_a_twenty_second_lease_cannot_fit_the_exchange() -> None:
    errors = net003.timing_errors(lease_seconds=20)
    assert "renewal interval" in _text(errors)


def test_a_replay_delay_too_short_to_attribute_the_commit_is_refused() -> None:
    assert "too short" in _text(net003.timing_errors(replay_delay_seconds=1))


def test_plan_details_carry_the_predecessor_and_the_loss_mode() -> None:
    predecessor = {"valid": True, "case_id": "GF-REGIONAL-NET-002", "verdict": "PASS"}
    details = net003.plan_details(predecessor)
    assert details["predecessor"] is predecessor
    assert details["response_loss_mode"] == "forward-then-reset"
    assert details["terminal_result_replays"] == 1


def test_the_private_seed_leases_the_workflow_to_the_probe(monkeypatch) -> None:
    """NET-003 seeds its own workflow (a notification rides along too); it must
    carry the same probe-owned execution lease as the shared seed, or the
    deployed dispatcher claims and fails it on the save-triggered wakeup within
    the second, then the orphan sweep cancels the command."""
    from scripts.e2e.regional import seeded_command_fixture as seeded

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        net003.fixture, "cpu_python", lambda script, *args: calls.append(args) or {}
    )
    net003.seed_command("run")
    assert calls[-1][-1] == str(seeded.SEED_LEASE_SECONDS)
    assert 'execution_owner_id=f"{owner}-seed"' in net003._SEED_COMMAND
    assert "execution_lease_expires_at=" in net003._SEED_COMMAND


def _gated_adapter(tmp_path, monkeypatch):
    """A LedgerAdapter with its /state files redirected into a temp dir and
    its sleeps replaced by a spy that records what was on disk when the
    simulated action ran."""
    from scripts.e2e.regional.probes import net003_executor as probe

    for attr in ("BLOCK", "ACTION_STARTED", "ACTION_GATE_OBSERVED", "LEDGER"):
        monkeypatch.setattr(probe, attr, tmp_path / attr.lower())
    at_action: list[dict[str, bool]] = []

    def spy_sleep(seconds):
        if seconds == 5:
            at_action.append(
                {
                    "gate_observed": probe.ACTION_GATE_OBSERVED.exists(),
                    "ledger_written": probe.LEDGER.exists(),
                }
            )

    monkeypatch.setattr(probe.time, "sleep", spy_sleep)
    return probe, probe.LedgerAdapter(NOTIFICATION_ID), at_action


def test_the_probe_holds_its_action_until_the_network_gate_is_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With /state/block never armed the adapter must give up without running
    the action, so the runner's leased snapshot can never race a commit."""
    probe, adapter, at_action = _gated_adapter(tmp_path, monkeypatch)
    clock = iter([0.0, 31.0, 31.0])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(clock))

    with pytest.raises(RuntimeError, match="network block was not armed"):
        adapter.execute(SimpleNamespace(idempotency_key="key-1"))

    assert at_action == [], "the action ran although the gate was never armed"
    assert not probe.ACTION_GATE_OBSERVED.exists(), "gate-observed written early"
    assert not probe.LEDGER.exists(), "the ledger committed an ungated action"


def test_the_probe_records_the_gate_then_acts_then_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once /state/block is armed the adapter records action-gate-observed,
    runs the action, and only then commits the ledger; a replay of the same
    key is served from the ledger without a second action."""
    probe, adapter, at_action = _gated_adapter(tmp_path, monkeypatch)
    probe.BLOCK.touch()

    first = adapter.execute(SimpleNamespace(idempotency_key="key-1"))

    observed = json.loads(probe.ACTION_GATE_OBSERVED.read_text(encoding="utf-8"))
    assert observed["idempotency_key"] == "key-1"
    assert at_action == [{"gate_observed": True, "ledger_written": False}]
    assert first.details["cached"] is False
    assert first.details["physical_count"] == 1

    replay = adapter.execute(SimpleNamespace(idempotency_key="key-1"))

    assert len(at_action) == 1, "a cached key must not run the action again"
    assert replay.details["cached"] is True
    assert replay.details["physical_count"] == 1


def test_the_runner_arms_the_gate_after_snapshotting_the_leased_command() -> None:
    """The runner snapshots the leased command first, then arms /state/block
    and waits for the probe to observe it. The block is never removed: unlike
    NET-002 this case does not force the lease to expire."""
    source = Path(net003.__file__).read_text(encoding="utf-8")
    snapshot = source.index('write_json(case_dir / "leased-command.json"')
    lease_check = source.index("command was not actively leased before injection")
    arm = source.index('fixture.touch(probe, "/state/block")')
    observed = source.index('"/state/action-gate-observed.json"')
    submit = source.index('"/state/result-submit-started.json"')
    assert snapshot < lease_check < arm < observed < submit
    assert 'fixture.remove(probe, "/state/block")' not in source


def _fake_replay(command_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        command_id=command_id,
        status=SimpleNamespace(value="SUCCEEDED"),
        status_source="control-plane",
        updated_at=T0 + timedelta(seconds=7),
    )


def _interrupting_client(tmp_path, monkeypatch):
    """A probe client with its /state writes redirected into a temp dir and its
    replay backoff neutralised, so complete()'s classification can be exercised
    without a live control plane."""
    from scripts.e2e.regional.probes import net003_executor as probe

    for attr in (
        "RESULT_SUBMIT_STARTED",
        "RESULT_INTERRUPTED",
        "RESULT_REPLAYS",
        "DROP_NEXT",
    ):
        monkeypatch.setattr(probe, attr, tmp_path / attr.lower())
    monkeypatch.setattr(probe.time, "sleep", lambda *_a, **_k: None)
    client = probe.InterruptingRegionalExecutorClient.__new__(
        probe.InterruptingRegionalExecutorClient
    )
    client._drop_injected = False
    return probe, client


def test_a_reset_with_no_status_code_is_a_lost_response_and_is_replayed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reset this case injects reaches the client as a ClusterExecutorError
    with status_code None -- no verdict was received -- so the probe records an
    interrupted first post and replays the same terminal result exactly once."""
    probe, client = _interrupting_client(tmp_path, monkeypatch)
    calls: list[int] = []

    def fake_complete(self, command, result):
        calls.append(1)
        if len(calls) == 1:
            raise probe.ClusterExecutorError(
                "regional control plane request failed: connection reset",
                status_code=None,
            )
        return _fake_replay(command.command_id)

    monkeypatch.setattr(probe.RegionalExecutorClient, "complete", fake_complete)
    command = SimpleNamespace(command_id="cmd-net003")

    returned = client.complete(command, {"status": "SUCCEEDED"})

    assert len(calls) == 2  # first post lost its response, then one replay
    interrupted = json.loads(probe.RESULT_INTERRUPTED.read_text(encoding="utf-8"))
    assert interrupted["first_post_succeeded"] is False
    assert interrupted["exception"] == "ClusterExecutorError"
    replays = json.loads(probe.RESULT_REPLAYS.read_text(encoding="utf-8"))
    assert replays["count"] == 1
    assert replays["responses"][0]["status"] == "SUCCEEDED"
    assert returned.status.value == "SUCCEEDED"


def test_an_http_rejection_surfaces_and_is_never_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ClusterExecutorError carrying a real HTTP status is an answer the
    client received; it must surface as the rejection it is and never be
    mistaken for the lost-response window, so no interrupted record is written
    and no replay is sent."""
    probe, client = _interrupting_client(tmp_path, monkeypatch)
    calls: list[int] = []

    def fake_complete(self, command, result):
        calls.append(1)
        raise probe.ClusterExecutorError(
            "regional control plane rejected the command", status_code=409
        )

    monkeypatch.setattr(probe.RegionalExecutorClient, "complete", fake_complete)
    command = SimpleNamespace(command_id="cmd-net003")

    with pytest.raises(probe.ClusterExecutorError) as caught:
        client.complete(command, {"status": "SUCCEEDED"})

    assert caught.value.status_code == 409
    assert len(calls) == 1  # rejected, not replayed
    assert not probe.RESULT_INTERRUPTED.exists(), "a 409 is not a lost response"
    assert not probe.RESULT_REPLAYS.exists(), "a rejected result must not replay"
