"""NET-002 verdicts and the shared network-case fixture."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import net_command_fixture as fixture
from scripts.e2e.regional import run_net002_command_recovery as net002

T0 = datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc)
COMMAND_ID = "remote-net002-test"
UNBLOCKED_AT = T0 + timedelta(seconds=net002.BLOCK_SECONDS + 10)


def _ready(**overrides: Any) -> dict[str, Any]:
    ready = {
        "block_rollback_seconds": net002.BLOCK_ROLLBACK_SECONDS,
        "http_timeout_seconds": float(net002.HTTP_TIMEOUT_SECONDS),
        "lease_seconds": net002.LEASE_SECONDS,
        "result_submission_gate": True,
        "first_result_submission_receipt": True,
        "result_gate_closes_caller_transport_pool": True,
        "action_requires_network_block": True,
    }
    ready.update(overrides)
    return ready


def _receipt(**overrides: Any) -> dict[str, Any]:
    receipt = {
        "command_id": COMMAND_ID,
        "submission_index": 1,
        "status_code": 409,
        "stale_lease_reason": net002.STALE_LEASE_DETAIL,
        "submitted_at_epoch": UNBLOCKED_AT.timestamp() + 0.1,
        "observed_at_epoch": UNBLOCKED_AT.timestamp() + 0.2,
        "gate_released_at_epoch": UNBLOCKED_AT.timestamp(),
        "caller_transport_pool_closed": True,
    }
    receipt.update(overrides)
    return receipt


def _errors(**overrides: Any) -> list[str]:
    leased = {
        "command_id": COMMAND_ID,
        "status": "LEASED",
        "lease_expires_at": (T0 + timedelta(seconds=net002.LEASE_SECONDS)).isoformat(),
    }
    arguments: dict[str, Any] = {
        "ready": _ready(),
        "leased": leased,
        "expired": dict(leased),
        "final": {
            "command_id": COMMAND_ID,
            "status": "SUCCEEDED",
            "result_details": {"cached": True},
        },
        "executor_state": {
            "claimed_total": 2,
            "reported_failures": 1,
            "unexpected_failures": 0,
            "lease_renewal_failures": 1,
        },
        "ledger": {"physical_count": 1, "keys": ["workflow/0/FREEZE_EVIDENCE"]},
        "logs": (
            "regional control plane rejected request (409): "
            "remote command lease is missing, stale, or changed"
        ),
        "blocked_seconds": net002.BLOCK_SECONDS + 3.0,
        "unblocked_at": UNBLOCKED_AT,
        "unblock_requested_at": UNBLOCKED_AT - timedelta(seconds=0.1),
        "expected_command_id": COMMAND_ID,
        "result_submit_waiting": {
            "command_id": COMMAND_ID,
            "observed_at_epoch": T0.timestamp() + 5,
        },
        "result_submit_released": {
            "command_id": COMMAND_ID,
            "observed_at_epoch": UNBLOCKED_AT.timestamp(),
        },
        "first_result_submission": _receipt(),
        "first_result_submission_after_reclaim": _receipt(),
    }
    arguments.update(overrides)
    return net002.net002_errors(**arguments)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def test_the_block_outlasts_the_lease_by_a_margin_not_by_a_minute() -> None:
    assert net002.LEASE_SECONDS < net002.BLOCK_SECONDS <= net002.LEASE_SECONDS + 15
    assert net002.BLOCK_SECONDS < net002.BLOCK_ROLLBACK_SECONDS
    assert net002.timing_errors() == []


def test_a_block_inside_the_lease_is_refused() -> None:
    errors = net002.timing_errors(block_seconds=net002.LEASE_SECONDS)
    assert any("does not outlast the lease" in item for item in errors), errors


def test_a_timeout_shorter_than_the_block_is_refused() -> None:
    errors = net002.timing_errors(http_timeout_seconds=15)
    assert any("HTTP timeout" in item for item in errors), errors


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
def test_the_intended_run_passes() -> None:
    assert _errors() == []


@pytest.mark.parametrize("detail", net002.STALE_LEASE_DETAILS)
def test_current_and_legacy_stale_lease_responses_are_recognized(detail: str) -> None:
    assert (
        _errors(
            logs=f"regional control plane rejected request (409): {detail}",
            first_result_submission=_receipt(stale_lease_reason=detail),
            first_result_submission_after_reclaim=_receipt(stale_lease_reason=detail),
        )
        == []
    )


@pytest.mark.parametrize(
    "logs",
    [
        "regional control plane rejected request (409): unrelated conflict",
        "regional control plane rejected request (403): remote command lease is stale",
        "remote command lease is stale",
    ],
)
def test_unrelated_conflicts_or_missing_http_status_do_not_prove_expiry(
    logs: str,
) -> None:
    assert "executor log has no stale-lease HTTP 409" in _errors(logs=logs)


@pytest.mark.parametrize(
    ("status_code", "reason"),
    [
        (None, None),
        (409, "unrelated conflict"),
        (403, net002.STALE_LEASE_DETAIL),
        (200, None),
        ("409", net002.STALE_LEASE_DETAIL),
        (409.0, net002.STALE_LEASE_DETAIL),
        (True, net002.STALE_LEASE_DETAIL),
        (409, net002.STALE_LEASE_DETAIL + ": untrusted suffix"),
    ],
)
def test_renewal_only_409_cannot_replace_first_result_proof(
    status_code: Any, reason: Any
) -> None:
    receipt = _receipt(status_code=status_code, stale_lease_reason=reason)
    errors = _errors(
        logs="renewal: regional control plane rejected request (409): "
        + net002.STALE_LEASE_DETAIL,
        first_result_submission=receipt,
        first_result_submission_after_reclaim=dict(receipt),
    )
    assert "first result submission has no stale-lease HTTP 409 receipt" in errors


def test_missing_first_receipt_fails_despite_stale_lease_logs() -> None:
    assert "first result submission receipt is missing" in _errors(
        first_result_submission={}
    )


@pytest.mark.parametrize(
    "field",
    [
        "first_result_submission",
        "result_submit_waiting",
        "result_submit_released",
        "leased",
        "expired",
        "final",
    ],
)
def test_unrelated_command_receipts_gates_and_snapshots_are_rejected(
    field: str,
) -> None:
    errors = _errors(**{field: {"command_id": "remote-unrelated"}})
    assert any("intended command" in item for item in errors), errors


@pytest.mark.parametrize("index", [None, 0, 2, True, "1"])
def test_later_or_unidentified_submission_cannot_be_used_as_first(index: Any) -> None:
    assert "result submission receipt is not the first attempt" in _errors(
        first_result_submission=_receipt(submission_index=index)
    )


@pytest.mark.parametrize(
    "replacement",
    [
        {},
        _receipt(status_code=200, stale_lease_reason=None),
        _receipt(command_id="other"),
    ],
    ids=["missing-after-reclaim", "cached-success", "other-command"],
)
def test_first_receipt_cannot_disappear_or_be_overwritten(
    replacement: dict[str, Any],
) -> None:
    assert "first result submission receipt changed after reclaim" in _errors(
        first_result_submission_after_reclaim=replacement
    )


@pytest.mark.parametrize(
    "field", ["gate_released_at_epoch", "submitted_at_epoch", "observed_at_epoch"]
)
@pytest.mark.parametrize("value", [None, "later", True, float("nan"), float("inf")])
def test_receipt_requires_finite_numeric_timestamps(field: str, value: Any) -> None:
    assert (
        "first result submission receipt has missing or invalid timestamps"
        in _errors(first_result_submission=_receipt(**{field: value}))
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"gate_released_at_epoch": UNBLOCKED_AT.timestamp() - 1},
        {"submitted_at_epoch": UNBLOCKED_AT.timestamp() - 1},
        {"observed_at_epoch": UNBLOCKED_AT.timestamp() - 1},
    ],
    ids=["wrong-release", "submitted-before-unblock", "response-before-submit"],
)
def test_receipt_must_follow_the_actual_result_gate_release(
    overrides: dict[str, Any],
) -> None:
    assert "first result submission was not recorded after unblock" in _errors(
        first_result_submission=_receipt(**overrides)
    )


def test_result_gate_release_must_follow_the_unblock_request() -> None:
    assert "first result submission was not recorded after unblock" in _errors(
        unblock_requested_at=UNBLOCKED_AT + timedelta(seconds=1)
    )


def test_receipt_can_arrive_before_the_unblock_command_acknowledgement() -> None:
    assert _errors(unblocked_at=UNBLOCKED_AT + timedelta(seconds=1)) == []


def test_the_http_timeout_is_a_recorded_fact_not_a_pass_condition() -> None:
    assert _errors(ready=_ready(http_timeout_seconds=200.0)) == []
    assert _errors(ready=_ready(http_timeout_seconds=180.0)) == []


def test_the_limitation_is_declared_in_plan_and_result() -> None:
    assert net002.LIMITATIONS == [
        "HTTP timeout 180s (production 15s) is used to reach the 409 path",
        "After the result gate releases, only the probe result caller's thread-local "
        "HTTP pool is closed to avoid an expired keepalive; TLS verification is unchanged",
    ]
    predecessor = {"valid": True, "case_id": "GF-REGIONAL-NET-001", "verdict": "PASS"}
    details = net002.plan_details(predecessor)
    assert details["limitations"] == net002.LIMITATIONS
    assert details["predecessor"] is predecessor
    assert details["timing"]["block_seconds"] == net002.BLOCK_SECONDS
    assert details["mutations"][1] == (
        "converged CPU registry revision publication without Deployment rollout"
    )
    assert details["pod_active_deadline_seconds"] == 900


def test_a_block_shorter_than_intended_fails_with_the_intended_length() -> None:
    errors = _errors(blocked_seconds=net002.BLOCK_SECONDS - 5.0)
    assert f"before {net002.BLOCK_SECONDS} seconds" in "\n".join(errors)


def test_a_lease_that_outlives_the_block_fails() -> None:
    errors = _errors(unblocked_at=T0 + timedelta(seconds=30))
    assert "had not expired" in "\n".join(errors)


def test_a_wrong_probe_lease_length_fails() -> None:
    errors = _errors(ready=_ready(lease_seconds=20))
    assert "lease length" in "\n".join(errors)


# --------------------------------------------------------------------------- #
# Shared fixture
# --------------------------------------------------------------------------- #
def test_run_identity_does_not_depend_on_the_run_dir_name(monkeypatch) -> None:
    first = fixture.run_identity(Path("/tmp/acceptance"), 1, "net002")
    second = fixture.run_identity(Path("/tmp/other-acceptance"), 1, "net002")
    assert first != second, "two directories with the same name suffix collided"
    assert first.startswith("net002-") and first.endswith("-a1")
    second_attempt = fixture.run_identity(Path("/tmp/acceptance"), 2, "net002")
    assert second_attempt.endswith("-a2"), second_attempt
    with pytest.raises(ValueError):
        fixture.run_identity(Path("/tmp/x"), 1, "net 002")


def test_cpu_pod_is_looked_up_once_per_process(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def control(*args: str, **_kwargs: Any) -> str:
        calls.append(args)
        if "exec" in args:
            return 'noise\n{"ok": true}\n'
        return "api-pod-1"

    monkeypatch.setattr(fixture, "control", control)
    fixture.reset_cpu_pod_cache()

    assert fixture.cpu_python("print(1)") == {"ok": True}
    assert fixture.cpu_python("print(1)") == {"ok": True}

    lookups = [item for item in calls if item[0] == "get"]
    assert len(lookups) == 1, calls
    fixture.reset_cpu_pod_cache()


def test_cpu_python_does_not_replay_a_failed_mutation(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []
    pods = iter(["api-pod-1", "api-pod-2"])

    def control(*args: str, **_kwargs: Any) -> str:
        calls.append(args)
        if args[0] == "get":
            return next(pods)
        if "api-pod-1" in args:
            raise RuntimeError("command failed (1): pod not found")
        return '{"ok": true}'

    monkeypatch.setattr(fixture, "control", control)
    fixture.reset_cpu_pod_cache()

    with pytest.raises(RuntimeError, match="pod not found"):
        fixture.cpu_python("possibly_committed_mutation()")
    assert len(calls) == 2
    assert calls[0][0] == "get" and calls[1][0] == "exec"
    assert fixture.cpu_python("independent_read()") == {"ok": True}
    assert [item for item in calls if item[0] == "get"] == [calls[0], calls[2]]
    assert fixture.cpu_pod() == "api-pod-2"
    fixture.reset_cpu_pod_cache()


def test_predecessor_gate_refuses_a_missing_predecessor(tmp_path: Path) -> None:
    gate = fixture.predecessor_gate(tmp_path, net002.CASE_ID, None)
    assert gate["valid"] is False
    assert gate["case_id"] == "GF-REGIONAL-NET-001"
    with pytest.raises(fixture.NetCommandError, match="predecessor"):
        fixture.require_predecessor(gate)


def test_predecessor_gate_accepts_a_formal_pass(tmp_path: Path) -> None:
    path = tmp_path / "cases" / "GF-REGIONAL-NET-001" / "GF-REGIONAL-NET-001.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"case_id": "GF-REGIONAL-NET-001", "verdict": "PASS"}),
        encoding="utf-8",
    )

    gate = fixture.predecessor_gate(tmp_path, net002.CASE_ID, None)

    assert gate["valid"] is True
    fixture.require_predecessor(gate)


def test_predecessor_gate_honours_an_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "elsewhere.json"
    gate = fixture.predecessor_gate(tmp_path, net002.CASE_ID, explicit)
    assert gate["path"] == str(explicit.resolve())


def test_wait_command_polls_until_the_status_matches(monkeypatch) -> None:
    states = iter([{"status": "LEASED"}, {"status": "LEASED"}, {"status": "SUCCEEDED"}])
    monkeypatch.setattr(fixture, "command_snapshot", lambda _id: next(states))
    ticks = iter(range(100))

    final = fixture.wait_command(
        "remote-x", "SUCCEEDED", 50, clock=lambda: next(ticks), sleep=lambda _s: None
    )

    assert final == {"status": "SUCCEEDED"}


def test_wait_command_raises_with_the_last_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(fixture, "command_snapshot", lambda _id: {"status": "LEASED"})
    ticks = iter([0, 1, 2, 100, 101])

    with pytest.raises(fixture.NetCommandError, match="did not reach SUCCEEDED"):
        fixture.wait_command(
            "remote-x", "SUCCEEDED", 3, clock=lambda: next(ticks), sleep=lambda _s: None
        )


def test_run_main_installs_the_site_profile_before_building_the_parser(
    monkeypatch, tmp_path: Path
) -> None:
    """The shared NET-002/003 ``main`` must read the site profile first: the
    parser's defaults come from the environment the profile fills, so a parser
    built earlier would silently ignore the operator's site values."""

    order: list[str] = []
    monkeypatch.setattr(
        fixture, "install_site_profile", lambda: order.append("profile")
    )

    class Parser:
        def parse_args(self) -> Any:
            order.append("parse")
            return type(
                "Args",
                (),
                {
                    "run_dir": tmp_path,
                    "predecessor_evidence": None,
                    "execute": False,
                    "attempt": 1,
                    "cluster_id": None,
                },
            )()

    monkeypatch.setattr(
        fixture,
        "predecessor_gate",
        lambda *_args, **_kwargs: order.append("gate") or {"valid": True},
    )

    def build_plan(*, arguments: Any, preflight_passed: bool, **kwargs: Any) -> dict:
        assert arguments.run_dir == tmp_path
        assert preflight_passed is True
        assert kwargs["details"]["predecessor"] == {"valid": True}
        return {"case_id": "GF-REGIONAL-NET-002"}

    monkeypatch.setattr(fixture, "build_plan", build_plan)
    exit_code = fixture.run_main(
        case_id="GF-REGIONAL-NET-002",
        confirmation="X",
        parser=lambda: Parser(),
        plan_details=lambda predecessor: {"predecessor": predecessor},
        run_case=lambda *_a, **_k: 0,
    )
    assert exit_code == 0, "a valid predecessor must let the plan path exit 0"
    assert order == ["profile", "parse", "gate"], (
        "the site profile must be installed before the parser is built and "
        f"before the predecessor gate runs; saw {order}"
    )
