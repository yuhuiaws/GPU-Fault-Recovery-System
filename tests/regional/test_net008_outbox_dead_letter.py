"""Contract tests for GF-REGIONAL-NET-008 (outbox dead-letter semantics)."""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import net008_verdicts as verdicts
from scripts.e2e.regional import run_net008_outbox_dead_letter as net008


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


def _record(**overrides: Any) -> dict[str, Any]:
    record = {
        "path": "/v1/collector-events/nvidia-kernel",
        "replayable": True,
        "error": "HTTP 403: regional cluster authentication failed",
        "failed_at": "t",
        "marker_present": True,
    }
    record.update(overrides)
    return record


def test_the_transient_outbox_contract_passes_on_replayable_403_records() -> None:
    records = [_record(), _record()]
    stats = {"depth": 2, "replayable": 2, "dead": 0}
    assert verdicts.transient_outbox_errors(records, stats, expected=2) == []


def test_the_transient_outbox_contract_rejects_a_dead_letter_or_a_missing_record() -> (
    None
):
    assert "not replayable" in _text(
        verdicts.transient_outbox_errors(
            [_record(replayable=False), _record()],
            {"replayable": 1, "dead": 1},
            expected=2,
        )
    )
    assert "1 marked outbox events in 1 record(s), expected 2" in _text(
        verdicts.transient_outbox_errors(
            [_record()], {"replayable": 1, "dead": 0}, expected=2
        )
    )
    assert "does not name the 403" in _text(
        verdicts.transient_outbox_errors(
            [_record(error="HTTP 500"), _record()],
            {"replayable": 2, "dead": 0},
            expected=2,
        )
    )


def test_two_transient_events_batched_into_one_record_count_as_two() -> None:
    """A record is one POST body; the collector batches a cycle's events into
    it, so two marked lines ten seconds apart were one record with two markers
    (attempt 1, 2026-09-10). The verdict counts marked events."""
    records = [_record(marker_count=2)]
    stats = {"depth": 1, "replayable": 1, "dead": 0}
    assert verdicts.transient_outbox_errors(records, stats, expected=2) == []
    errors = verdicts.transient_outbox_errors(
        [_record(marker_count=1)], stats, expected=2
    )
    assert any("1 marked outbox events in 1 record(s)" in item for item in errors), (
        errors
    )
    assert verdicts.COLLECTOR_SETTLE_SECONDS >= 10


def test_the_service_contract_counts_only_automatic_restarts() -> None:
    before = {"ActiveState": "active", "NRestarts": "0"}
    assert (
        verdicts.service_errors(before, {"ActiveState": "active", "NRestarts": "0"})
        == []
    )
    assert "restarted on its own" in _text(
        verdicts.service_errors(before, {"ActiveState": "active", "NRestarts": "2"})
    )


def test_the_delivery_contract_wants_exactly_one_record_per_marker() -> None:
    evidence = [
        {"payload": {"message": "marker=m-t1"}},
        {"payload": {"message": "marker=m-t2"}},
    ]
    assert (
        verdicts.delivered_errors(
            evidence, [], {"replayable": 0}, markers=["m-t1", "m-t2"]
        )
        == []
    )
    assert "has 0 evidence records" in _text(
        verdicts.delivered_errors(evidence, [], {"replayable": 0}, markers=["m-t3"])
    )
    assert "has 2 evidence records" in _text(
        verdicts.delivered_errors(
            evidence + [evidence[0]], [], {"replayable": 0}, markers=["m-t1"]
        )
    )
    assert "still in the outbox" in _text(
        verdicts.delivered_errors(
            evidence, [_record()], {"replayable": 0}, markers=["m-t1"]
        )
    )


def test_the_dead_letter_listing_and_requeue_contracts() -> None:
    dead = _record(
        path=verdicts.RETIRED_CHANNEL_PATH,
        replayable=False,
        error="HTTP 404: resource not found",
    )
    assert verdicts.dead_letter_errors([dead], {"dead": 1}, marker="m") == []
    assert "was not dead-lettered" in _text(
        verdicts.dead_letter_errors(
            [{**dead, "replayable": True}], {"dead": 0}, marker="m"
        )
    )
    assert "does not name the 404" in _text(
        verdicts.dead_letter_errors(
            [{**dead, "error": "HTTP 403"}], {"dead": 1}, marker="m"
        )
    )
    lines = [f"0\t{verdicts.RETIRED_CHANNEL_PATH}\tdead\tt\tHTTP 404"]
    assert verdicts.listing_errors(lines) == []
    assert "leaks the record payload" in _text(
        verdicts.listing_errors(lines + ["acceptance-dead-letter-m"])
    )
    assert "does not show the dead" in _text(
        verdicts.listing_errors(["0\t/p\treplayable"])
    )
    requeued = {
        "refused_without_yes": True,
        "output": "requeued 1 dead record(s) in /var/lib/gpu-fault/outbox/kernel.ndjson",
        "stats": {"replayable": 1, "dead": 0},
    }
    assert verdicts.requeue_errors(requeued) == []
    assert "ran without --yes" in _text(
        verdicts.requeue_errors({**requeued, "refused_without_yes": False})
    )
    assert "did not report one record" in _text(
        verdicts.requeue_errors({**requeued, "output": "requeued 0 dead record(s)"})
    )


def test_the_stream_identity_contract_catches_a_reopen() -> None:
    before = {"pid": 10, "invocation_id": "i1", "kmsg_streams": [{"fd": 5, "pos": 100}]}
    assert (
        verdicts.stream_identity_errors(
            before,
            {"pid": 10, "invocation_id": "i1", "kmsg_streams": [{"fd": 5, "pos": 130}]},
        )
        == []
    )
    assert "fd set changed" in _text(
        verdicts.stream_identity_errors(
            before,
            {"pid": 10, "invocation_id": "i1", "kmsg_streams": [{"fd": 7, "pos": 0}]},
        )
    )
    assert "went backwards" in _text(
        verdicts.stream_identity_errors(
            before,
            {"pid": 10, "invocation_id": "i1", "kmsg_streams": [{"fd": 5, "pos": 3}]},
        )
    )
    assert "PID changed" in _text(
        verdicts.stream_identity_errors(
            before,
            {"pid": 11, "invocation_id": "i1", "kmsg_streams": [{"fd": 5, "pos": 130}]},
        )
    )
    assert "held no /dev/kmsg stream" in _text(
        verdicts.stream_identity_errors(
            {"pid": 10, "invocation_id": "i1", "kmsg_streams": []},
            {"pid": 10, "invocation_id": "i1", "kmsg_streams": []},
        )
    )


def test_the_blackout_and_purge_contracts() -> None:
    blocked = {"connectivity": {"10.0.0.1": False}, "timer": {"ActiveState": "active"}}
    unblocked = {"rules": [], "connectivity": {"10.0.0.1": True}}
    assert verdicts.blackout_errors(blocked, unblocked) == []
    assert "stayed reachable" in _text(
        verdicts.blackout_errors(
            {**blocked, "connectivity": {"10.0.0.1": True}}, unblocked
        )
    )
    assert "rules remain" in _text(
        verdicts.blackout_errors(blocked, {**unblocked, "rules": ["-A OUTPUT ..."]})
    )
    assert "rollback timer was not active" in _text(
        verdicts.blackout_errors({**blocked, "timer": {}}, unblocked)
    )
    assert verdicts.purge_errors({"removed": 1}) == []
    assert "expected 1" in _text(verdicts.purge_errors({"removed": 0}))


def test_the_case_constants_and_plan_name_every_phase() -> None:
    assert net008.CASE_ID == "GF-REGIONAL-NET-008"
    assert net008.CONFIRMATION == "NET008_EXECUTE"
    assert verdicts.PREDECESSOR_CASE_ID == "GF-REGIONAL-NET-007"
    assert verdicts.BLOCK_SECONDS < verdicts.BLOCK_TTL_SECONDS, (
        "rollback outlasts the block"
    )
    settings = type(
        "S",
        (),
        {
            "node": "node-a",
            "endpoint_host": "h",
            "regional": type("R", (), {"cluster_id": "c"})(),
        },
    )()
    details = net008.plan_details(settings, {"predecessor": {"valid": True}})
    assert details["risk"] == "live-kernel-log-injection", details
    for phase in ("A:", "B:", "C:"):
        assert phase in details["mutation"], details["mutation"]
    assert (
        details["rollback"]["firewall_rollback_seconds"] == verdicts.BLOCK_TTL_SECONDS
    )
    assert (
        "purges seed" in "".join(details["rollback"])
        or details["rollback"]["runner_finally_closes_window_unblocks_and_purges_seed"]
    ), details["rollback"]
