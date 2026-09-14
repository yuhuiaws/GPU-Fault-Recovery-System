"""NET-003 verdicts: the control plane committed, the client lost the response."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

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


def test_the_probe_holds_its_action_behind_the_network_gate() -> None:
    """The probe must not run its action until the runner has armed
    /state/block, so the runner's leased snapshot cannot race the action that
    commits the result; it records action-gate-observed once it proceeds."""
    from pathlib import Path

    source = (
        Path(net003.__file__).resolve().parent / "probes" / "net003_executor.py"
    ).read_text(encoding="utf-8")
    assert 'BLOCK = STATE / "block"' in source
    assert 'ACTION_GATE_OBSERVED = STATE / "action-gate-observed.json"' in source
    gate = source.index("while not BLOCK.exists()")
    observed = source.index("ACTION_GATE_OBSERVED.write_text")
    action = source.index("time.sleep(5)")
    assert gate < observed < action


def test_the_runner_arms_the_gate_after_snapshotting_the_leased_command() -> None:
    """The runner snapshots the leased command first, then arms /state/block
    and waits for the probe to observe it. The block is never removed: unlike
    NET-002 this case does not force the lease to expire."""
    from pathlib import Path

    source = Path(net003.__file__).read_text(encoding="utf-8")
    snapshot = source.index('write_json(case_dir / "leased-command.json"')
    lease_check = source.index("command was not actively leased before injection")
    arm = source.index('fixture.touch(probe, "/state/block")')
    observed = source.index('"/state/action-gate-observed.json"')
    submit = source.index('"/state/result-submit-started.json"')
    assert snapshot < lease_check < arm < observed < submit
    assert 'fixture.remove(probe, "/state/block")' not in source
