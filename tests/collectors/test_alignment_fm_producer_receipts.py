"""Producer receipts prove real sink handoffs without an idle fleet log flood."""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.channel_registry import FABRIC_MANAGER_PATH
from gpu_fault.collectors.logs import fabric_manager as fabric
from gpu_fault.collectors.logs import fabric_manager_receipts as receipts
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.sinks import CollectorError

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)
LOGGER_NAME = fabric.__name__


class Clock:
    value = 0.0

    def monotonic(self) -> float:
        return self.value

    def utc(self) -> datetime:
        return NOW + timedelta(seconds=self.value)

    def advance(self, seconds: float = 5) -> None:
        self.value += seconds


class Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.error: Exception | None = None
        self.before_post = None

    def post(self, path: str, payload: dict) -> dict:
        if path == FABRIC_MANAGER_PATH:
            if self.before_post:
                self.before_post(payload)
            self.events.append(payload)
            if self.error is not None:
                raise self.error
        return {"accepted": True}


def line(index: int, extra: str = "") -> bytes:
    return (
        f"SXid (PCI:0000:ab:00.0): 99999, Non-fatal, private-{index} {extra}\n"
    ).encode()


def records(caplog) -> list[dict]:
    result = []
    for record in caplog.records:
        if not record.getMessage().startswith(receipts.FM_RECEIPT_PREFIX):
            continue
        assert record.levelno == logging.INFO, "producer receipts must use INFO"
        message = record.getMessage()
        assert len(message.encode()) <= receipts.FM_RECEIPT_MAX_BYTES
        payload = json.loads(message.removeprefix(receipts.FM_RECEIPT_PREFIX))
        assert set(payload) == receipts.FM_RECEIPT_FIELDS, (
            "producer receipts must not acquire free-form fields"
        )
        for name in receipts.FM_RECEIPT_COUNTERS:
            assert type(payload[name]) is int and payload[name] >= 0
        assert type(payload["counter_exhausted"]) is bool
        result.append(payload)
    return result


def reader(
    tmp_path: Path,
    sink: Sink,
    clock: Clock,
    *,
    cluster_id: str = "private-cluster",
    node_id: str = "private-node",
    boot_id: str = "private-boot",
) -> fabric.FabricManagerLogCollector:
    return fabric.FabricManagerLogCollector(
        sink,
        CollectorContext(cluster_id=cluster_id),
        node_id=node_id,
        boot_id=boot_id,
        journal_enabled=False,
        log_paths=[str(tmp_path / "private.log")],
        state_path=str(tmp_path / "cursor.json"),
        now=clock.utc,
        monotonic=clock.monotonic,
    )


def prepare(tmp_path: Path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    log = tmp_path / "private.log"
    log.write_bytes(b"")
    sink, clock = Sink(), Clock()
    collector = reader(tmp_path, sink, clock)
    collector.collect_once()
    caplog.clear()
    clock.advance()
    return log, sink, clock, collector


def test_attempt_precedes_sink_and_completion_and_round_follow_real_outcome(
    tmp_path, caplog, monkeypatch
) -> None:
    monkeypatch.setenv("INVOCATION_ID", "1" * 32)
    log, sink, _clock, collector = prepare(tmp_path, caplog)
    log.write_bytes(line(1))

    def before_post(payload):
        observed = records(caplog)
        assert [item["stage"] for item in observed] == ["ATTEMPT"], (
            "the sink must not be entered before its attempt receipt"
        )
        attempt = observed[0]
        assert attempt["record_id_sha256"] == receipts.identity_sha256(
            payload["record_id"]
        )
        assert attempt["attempts_total"] == 1 and attempt["delivered_total"] == 0

    sink.before_post = before_post
    stats = collector.collect_once()
    observed = records(caplog)
    assert [item["stage"] for item in observed] == ["ATTEMPT", "COMPLETION", "ROUND"]
    assert [item["outcome"] for item in observed] == [
        "STARTED",
        "DELIVERED",
        "COMPLETE",
    ]
    assert [item["receipt_seq"] for item in observed] == [2, 3, 4]
    assert {item["attempt_seq"] for item in observed} == {1}
    assert {item["round_seq"] for item in observed} == {2}
    assert len({item["producer_invocation_id"] for item in observed}) == 1
    assert {item["systemd_invocation_id"] for item in observed} == {"1" * 32}
    assert observed[-1]["rounds_completed_total"] == 2
    assert observed[-1]["omitted_receipts_total"] == 0
    assert stats.delivered == 1
    checkpoint = json.loads((tmp_path / "cursor.json").read_text())["files"][str(log)]
    assert checkpoint["offset"] == log.stat().st_size


@pytest.mark.parametrize(
    "outcome,error",
    [
        ("BUFFERED", CollectorError("unavailable", buffered=True, replayable=True)),
        ("FAILED", CollectorError("rejected")),
        ("FAILED", RuntimeError("unexpected sink failure")),
    ],
    ids=["buffered", "rejected", "unexpected"],
)
def test_completion_never_confuses_buffered_or_failed_delivery_with_acknowledgement(
    tmp_path, caplog, outcome, error
) -> None:
    log, sink, _clock, collector = prepare(tmp_path, caplog)
    log.write_bytes(line(1))
    sink.error = error
    if outcome == "FAILED":
        with pytest.raises(type(error)):
            collector.collect_once()
    else:
        collector.collect_once()
    observed = records(caplog)
    assert observed[1]["stage"] == "COMPLETION"
    assert observed[1]["outcome"] == outcome
    assert observed[-1]["outcome"] == ("FAILED" if outcome == "FAILED" else "COMPLETE")
    assert observed[-1]["delivered_total"] == 0
    assert observed[-1]["failed_total"] == int(outcome == "FAILED")
    assert observed[-1]["buffered_total"] == int(outcome == "BUFFERED")
    checkpoint = json.loads((tmp_path / "cursor.json").read_text())["files"][str(log)]
    assert checkpoint["offset"] == (0 if outcome == "FAILED" else log.stat().st_size)


def test_restart_changes_producer_invocation_and_proves_zero_source_replay(
    tmp_path, caplog, monkeypatch
) -> None:
    monkeypatch.setenv("INVOCATION_ID", "1" * 32)
    log, sink, clock, collector = prepare(tmp_path, caplog)
    log.write_bytes(line(1))
    collector.collect_once()
    before = records(caplog)[-1]
    caplog.clear()
    monkeypatch.setenv("INVOCATION_ID", "2" * 32)
    restarted = reader(tmp_path, sink, clock)
    for _ in range(2):
        clock.advance()
        restarted.collect_once()
    after = records(caplog)
    assert len(sink.events) == 1, "a new process invocation must not replay its cursor"
    assert [item["stage"] for item in after] == ["ROUND", "ROUND"]
    assert [item["round_seq"] for item in after] == [1, 2]
    assert [item["receipt_seq"] for item in after] == [1, 2]
    assert all(item["attempts_total"] == 0 for item in after), (
        "restart must not attempt delivery from the acknowledged cursor"
    )
    assert all(item["omitted_receipts_total"] == 0 for item in after), (
        "restarted idle rounds must not conceal omitted receipts"
    )
    assert after[0]["producer_invocation_id"] != before["producer_invocation_id"]
    assert after[0]["systemd_invocation_id"] == "2" * 32
    assert after[0]["source_config_sha256"] == before["source_config_sha256"]


def test_normal_idle_polls_are_aggregated_at_the_existing_summary_cadence(
    tmp_path, caplog
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    (tmp_path / "private.log").write_bytes(b"")
    sink, clock = Sink(), Clock()
    collector = reader(tmp_path, sink, clock)
    for _ in range(721):
        collector.collect_once()
        clock.advance()
    observed = records(caplog)
    assert len(observed) == 13, (
        "one hour of five-second idle polls must not create 721 INFO receipts"
    )
    assert all(item["stage"] == "ROUND" for item in observed), (
        "idle poll summaries must only emit ROUND receipts"
    )
    assert [item["receipt_seq"] for item in observed] == list(range(1, 14))
    assert [item["round_seq"] for item in observed] == [1, 2, *range(62, 663, 60)]
    assert observed[-1]["rounds_completed_total"] == 662
    assert observed[-1]["attempts_total"] == 0
    assert observed[-1]["omitted_receipts_total"] == 0
    assert sink.events == []


def test_logging_budget_limits_volume_without_losing_faults_or_hiding_omissions(
    tmp_path, caplog, monkeypatch
) -> None:
    monkeypatch.setattr(receipts, "FM_RECEIPT_PAIR_BUDGET", 2)
    log, sink, clock, collector = prepare(tmp_path, caplog)
    log.write_bytes(b"".join(line(index) for index in range(8)))
    collector.collect_once()
    observed = records(caplog)
    assert len(sink.events) == 8, "logging pressure must not suppress fault delivery"
    assert [item["stage"] for item in observed] == [
        "ATTEMPT",
        "COMPLETION",
        "ATTEMPT",
        "COMPLETION",
        "ROUND",
    ]
    assert observed[-1]["attempts_total"] == observed[-1]["delivered_total"] == 8
    assert observed[-1]["omitted_receipts_total"] == 12
    assert observed[-1]["receipt_seq"] == 18, (
        "omitted receipts must leave sequence gaps"
    )
    invocation = observed[-1]["producer_invocation_id"]
    caplog.clear()
    clock.advance(60)
    with log.open("ab") as stream:
        stream.write(line(9))
    collector.collect_once()
    refreshed = records(caplog)
    assert refreshed[0]["producer_invocation_id"] == invocation
    assert refreshed[0]["attempt_seq"] == 9, "budget renewal must not reset identity"
    assert refreshed[-1]["omitted_receipts_total"] == 12
    assert refreshed[-1]["delivered_total"] == 9


def test_unreadable_source_is_failed_progress_even_when_other_sources_can_continue(
    tmp_path, caplog, monkeypatch
) -> None:
    log, _sink, clock, collector = prepare(tmp_path, caplog)
    real_open = Path.open

    def denied(path, mode="r", *args, **kwargs):
        if path == log and mode == "rb":
            raise PermissionError("private source read denied")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    collector.collect_once()
    observed = records(caplog)
    assert observed[-1]["stage"] == "ROUND"
    assert observed[-1]["outcome"] == "FAILED"
    assert observed[-1]["rounds_failed_total"] == 1
    monkeypatch.setattr(Path, "open", real_open)
    clock.advance(300)
    collector.collect_once()
    assert records(caplog)[-1]["outcome"] == "COMPLETE"


def test_sink_ack_with_failed_checkpoint_cannot_be_reported_as_complete_round(
    tmp_path, caplog, monkeypatch
) -> None:
    log, _sink, _clock, collector = prepare(tmp_path, caplog)
    log.write_bytes(line(1))

    def denied_replace(*_args):
        raise OSError("private checkpoint replacement refused")

    monkeypatch.setattr(fabric.os, "replace", denied_replace)
    with pytest.raises(OSError, match="checkpoint replacement refused"):
        collector.collect_once()
    observed = records(caplog)
    assert observed[1]["outcome"] == "DELIVERED"
    assert observed[-1]["outcome"] == "FAILED"
    assert observed[-1]["rounds_failed_total"] == 1


def test_receipts_and_failure_diagnostics_never_copy_sensitive_values(
    tmp_path, caplog, monkeypatch
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    secret = secrets.token_urlsafe(32)
    private_url = f"https://private.invalid/?token={secret}"
    monkeypatch.setenv("INVOCATION_ID", private_url)
    log = tmp_path / "private.log"
    log.write_bytes(b"")
    sink, clock = Sink(), Clock()
    collector = reader(
        tmp_path,
        sink,
        clock,
        cluster_id="cluster-" + secret,
        node_id="node-" + secret,
        boot_id="boot-" + secret,
    )
    collector.collect_once()
    log.write_bytes(line(1, private_url))
    sink.error = CollectorError(private_url, buffered=True, replayable=True)
    clock.advance()
    collector.collect_once()
    observed = records(caplog)
    assert observed[-1]["buffered_total"] == 1
    assert observed[-1]["systemd_invocation_id"] is None
    assert observed[-1]["cluster_id_sha256"] == receipts.identity_sha256(
        "cluster-" + secret
    )
    assert secret not in caplog.text, "receipt or warning leaked a private value"
    assert "https://" not in caplog.text, "receipt or warning copied a source URL"
    assert str(log) not in caplog.text, "receipt or warning copied a source path"


def test_counter_exhaustion_is_explicit_and_never_creates_unbounded_numbers(
    tmp_path, caplog, monkeypatch
) -> None:
    monkeypatch.setattr(receipts, "FM_RECEIPT_COUNTER_MAX", 3)
    log, _sink, _clock, collector = prepare(tmp_path, caplog)
    log.write_bytes(b"".join(line(index) for index in range(5)))
    collector.collect_once()
    observed = records(caplog)
    assert observed[-1]["counter_exhausted"] is True
    assert all(
        value[name] <= 3 for value in observed for name in receipts.FM_RECEIPT_COUNTERS
    ), "exhausted producer counters must stay within the configured maximum"


def test_round_log_budget_keeps_a_periodic_checkpoint_for_visible_omissions(
    tmp_path, caplog, monkeypatch
) -> None:
    monkeypatch.setattr(receipts, "FM_RECEIPT_ROUND_BUDGET", 1)
    log, sink, clock, collector = prepare(tmp_path, caplog)
    real_open = Path.open

    def denied(path, mode="r", *args, **kwargs):
        if path == log and mode == "rb":
            raise PermissionError("private source read denied")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    for _ in range(5):
        collector.collect_once()
    assert len(records(caplog)) == 1, "only the reserved second startup round may log"
    clock.advance(300)
    collector.collect_once()
    latest = records(caplog)[-1]
    assert latest["stage"] == "ROUND" and latest["outcome"] == "FAILED"
    assert latest["omitted_receipts_total"] == 4
    assert latest["rounds_failed_total"] == 6
    assert sink.events == []


def test_disabled_info_level_is_an_explicit_proof_gap_when_logging_resumes(
    tmp_path, caplog
) -> None:
    log, _sink, clock, collector = prepare(tmp_path, caplog)
    log.write_bytes(line(1))
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    collector.collect_once()
    assert records(caplog) == []
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    clock.advance(300)
    collector.collect_once()
    latest = records(caplog)[-1]
    assert latest["receipt_seq"] == 5
    assert latest["omitted_receipts_total"] == 3
    assert latest["delivered_total"] == 1


def test_checkpoint_schema_size_failure_does_not_stop_the_collector(
    tmp_path, caplog, monkeypatch
) -> None:
    log, sink, clock, collector = prepare(tmp_path, caplog)
    log.write_bytes(line(1))
    with monkeypatch.context() as limited:
        limited.setattr(receipts, "FM_RECEIPT_MAX_BYTES", 1)
        collector.collect_once()
    assert len(sink.events) == 1
    assert records(caplog) == []
    clock.advance(300)
    collector.collect_once()
    latest = records(caplog)[-1]
    assert latest["counter_exhausted"] is True
    assert latest["omitted_receipts_total"] == 3


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_summary_cadence_is_rejected_before_receipt_emission(
    caplog, interval
) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        receipts.FabricManagerReceiptLog(
            logging.getLogger(LOGGER_NAME),
            cluster_id="private-cluster",
            node_id="private-node",
            boot_id="private-boot",
            source_configuration="private",
            summary_seconds=interval,
            monotonic=lambda: 0,
        )
    assert records(caplog) == []
