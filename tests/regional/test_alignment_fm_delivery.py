from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import collector_delivery_evidence as evidence
from scripts.e2e.regional import run_collector_acceptance as runner
from scripts.e2e.regional.probes import collector_node_probe as probe
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._alignment_fm_receipts import (
    NEW_UNIT,
    OLD_UNIT,
    RECORD_ID,
    SCOPE,
    FmProducerFixture,
    baseline,
    receipt,
    replay_window,
    window,
)
from tests.regional._cov95_collect_net import Clock


def window_errors(projection: dict[str, Any], *, restarted: bool = True) -> list[str]:
    return evidence.delivery_window_errors(
        baseline()["records"][0],
        projection,
        record_id=RECORD_ID,
        **SCOPE,
        restarted_invocation=NEW_UNIT if restarted else None,
    )


def test_producer_receipts_prove_restart_without_confusing_it_with_store_dedup() -> (
    None
):
    assert evidence.original_anchor(baseline(), **SCOPE) == baseline()["records"][0]
    assert window_errors(window(), restarted=False) == []
    assert window_errors(window(restarted=True)) == []
    errors = window_errors(replay_window())
    assert any("replayed the original record" in error for error in errors), (
        "source replay must be diagnosed even when the Store deduplicates its row"
    )
    assert evidence.completed_round_end(window(restarted=True), NEW_UNIT) == 6000
    assert evidence.completed_round_end(window(), NEW_UNIT) is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("producer_invocation_id", ""),
        ("systemd_invocation_id", None),
        ("pid", True),
        ("pid", 1),
        ("cluster_id_sha256", ""),
        ("node_id_sha256", None),
        ("receipt_seq", True),
        ("receipt_seq", 0),
        ("round_seq", 0),
        ("attempts_total", -1),
        ("stage", "OTHER"),
        ("outcome", "STARTED"),
        ("record_id_sha256", "a" * 64),
        ("source", "file"),
        ("source_config_sha256", ""),
        ("counter_exhausted", True),
        ("counter_exhausted", None),
        ("outcome", []),
    ],
)
def test_fm_receipt_schema_rejects_unknown_identity_counts_or_round_shape(
    key: str, value: Any
) -> None:
    row = receipt(1, 1, "ROUND", "COMPLETE")
    row[key] = value
    with pytest.raises(probe.ProbeError):
        probe.checked_fm_receipt(row)


def test_fm_receipt_schema_never_returns_an_extra_field_from_a_log() -> None:
    row = receipt(1, 1, "ROUND", "COMPLETE")
    row["unexpected_sensitive_field"] = "do-not-export-this"
    with pytest.raises(probe.ProbeError, match="fields"):
        probe.checked_fm_receipt(row)
    for source, digest in (("api", "a" * 64), ("file", None)):
        row = receipt(2, 2, "ATTEMPT", "STARTED", attempt=1)
        row.update(source=source, record_id_sha256=digest)
        with pytest.raises(probe.ProbeError):
            probe.checked_fm_receipt(row)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("root", "complete", False),
        ("root", "captured_monotonic_us", 0),
        ("root", "captured_monotonic_us", True),
        ("root", "records", []),
        ("service", "MainPID", "1"),
        ("service", "MainPID", "missing"),
        ("service", "InvocationID", None),
        ("service", "ActiveState", "inactive"),
        ("record", "cursor", ""),
        ("record", "monotonic_us", 100_001),
        ("record", "monotonic_us", -1),
        ("record", "receipt", {}),
        ("body", "cluster_id_sha256", "f" * 64),
        ("body", "node_id_sha256", "f" * 64),
        ("body", "boot_id_sha256", "f" * 64),
    ],
)
def test_fm_projection_requires_bound_complete_scope_and_journal_clock(
    section: str, key: str, value: Any
) -> None:
    projection = window()
    target = {
        "root": projection,
        "service": projection["service"],
        "record": projection["records"][0],
        "body": projection["records"][0]["receipt"],
    }[section]
    target[key] = value
    assert window_errors(projection, restarted=False), (
        "delivery proof must reject incomplete or unbound receipt projections"
    )


@pytest.mark.parametrize(
    "problem",
    [
        "anchor",
        "wrong-anchor-field",
        "service",
        "first-sequence",
        "first-round",
        "pid-switch",
        "old-pid-switch",
        "scope-switch",
        "sequence-gap",
        "omitted",
        "regression",
        "alias",
        "failed",
        "round-failed",
        "no-attempt",
        "pair",
        "attempt-counter",
        "completion-counter",
        "unmatched",
        "duplicate-round",
        "round-total",
        "hidden-delivery",
        "missing-second",
        "return-old",
        "buffered",
        "source-config",
    ],
)
def test_no_replay_gate_rejects_incomplete_or_contradictory_receipt_windows(
    problem: str,
) -> None:
    projection = window(restarted=True)
    rows = projection["records"]
    first, second, completion, end, new_first, new_second = [
        row["receipt"] for row in rows
    ]
    if problem == "anchor":
        rows.pop(0)
    elif problem == "wrong-anchor-field":
        projection["anchor_cursor"] = "s=another"
    elif problem == "service":
        projection["service"]["InvocationID"] = OLD_UNIT
    elif problem == "first-sequence":
        new_first["receipt_seq"] = 2
    elif problem == "first-round":
        new_first["round_seq"] = 2
    elif problem == "pid-switch":
        new_second["pid"] = 999
    elif problem == "old-pid-switch":
        end["pid"] = 999
    elif problem == "scope-switch":
        new_second["systemd_invocation_id"] = "5" * 32
    elif problem == "sequence-gap":
        completion["receipt_seq"] += 1
    elif problem == "omitted":
        end["omitted_receipts_total"] = 1
    elif problem == "regression":
        end["attempts_total"] = 0
    elif problem == "alias":
        second["attempt_seq"] = 2
    elif problem == "failed":
        completion["outcome"] = "FAILED"
        completion["delivered_total"] = 0
        completion["failed_total"] = 1
    elif problem == "round-failed":
        new_second["outcome"] = "FAILED"
        new_second["rounds_failed_total"] = 1
    elif problem == "no-attempt":
        rows.pop(1)
    elif problem == "pair":
        completion["record_id_sha256"] = "f" * 64
    elif problem == "attempt-counter":
        second["delivered_total"] = 1
    elif problem == "completion-counter":
        completion["delivered_total"] = 0
    elif problem == "unmatched":
        rows[:] = rows[:2]
        projection["service"] = window()["service"]
    elif problem == "duplicate-round":
        new_second["round_seq"] = new_second["rounds_completed_total"] = 1
    elif problem == "round-total":
        new_second["rounds_completed_total"] = 3
    elif problem == "hidden-delivery":
        new_second.update(attempt_seq=1, attempts_total=1, delivered_total=1)
    elif problem == "missing-second":
        rows.pop()
    elif problem == "return-old":
        rows[-1]["receipt"] = deepcopy(end)
        rows[-1]["receipt"]["receipt_seq"] += 1
    elif problem == "buffered":
        completion.update(outcome="BUFFERED", delivered_total=0, buffered_total=1)
        end.update(delivered_total=0, buffered_total=1)
    elif problem == "source-config":
        new_second["source_config_sha256"] = "f" * 64
    assert window_errors(projection), f"accepted {problem}"


def test_original_anchor_and_restarted_window_cannot_use_an_unstarted_process() -> None:
    value = baseline()
    value["records"][0]["receipt"]["outcome"] = "FAILED"
    with pytest.raises(RegionalFixtureError, match="anchor"):
        evidence.original_anchor(value, **SCOPE)
    assert evidence.delivery_window_errors(
        baseline()["records"][0],
        window(),
        record_id=RECORD_ID,
        restarted_invocation=OLD_UNIT,
        **SCOPE,
    ), "the original service invocation must not prove a producer restart"
    value = window(restarted=True)
    value["records"].append(deepcopy(value["records"][-1]))
    assert window_errors(value), (
        "a duplicate receipt must invalidate the restart window"
    )


def journal_lines(projection: dict[str, Any]) -> str:
    return "\n".join(
        json.dumps(
            {
                "MESSAGE": "INFO:collector:"
                + probe.FM_RECEIPT_PREFIX
                + json.dumps(row["receipt"]),
                "__CURSOR": row["cursor"],
                "_SYSTEMD_INVOCATION_ID": row["receipt"]["systemd_invocation_id"],
                "_PID": str(row["receipt"]["pid"]),
                "__MONOTONIC_TIMESTAMP": str(row["monotonic_us"]),
            }
        )
        for row in projection["records"]
    )


def test_journal_probe_reads_inclusive_bounded_cursor_and_projects_only_receipts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    projection = window()
    commands = []
    monkeypatch.setattr(
        probe, "service_snapshot", lambda: {probe.FM_UNIT: projection["service"]}
    )
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs))
        or SimpleNamespace(
            returncode=0, stdout=journal_lines(projection), stderr="do-not-export"
        ),
    )
    arguments = probe.parser().parse_args(
        ["fm-delivery-evidence", "--cursor", projection["anchor_cursor"]]
    )
    arguments.handler(arguments)
    text = capsys.readouterr().out
    result = json.loads(text)
    assert result["records"] == projection["records"]
    assert result["complete"] is True
    assert f"--lines=+{probe.FM_JOURNAL_LIMIT + 1}" in commands[0][0]
    assert f"--cursor={projection['anchor_cursor']}" in commands[0][0]
    assert commands[0][1] == {"check": False, "timeout": 60}
    assert "MESSAGE" not in text and "do-not-export" not in text


def test_journal_probe_does_not_rely_on_grep_ordering_and_skips_plain_log_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """systemd 252 returns ``--grep`` matches newest-first once ``--lines`` is
    given, and with ``--cursor`` it even walks backwards (live 2026-09-17,
    COLLECT-005 "journal order/clock is unproven"). The probe must read the
    unit's journal without ``--grep``, skip the collector's plain log lines
    itself, and only emit a window in journal order.
    """
    projection = window()
    commands = []
    plain = json.dumps(
        {
            "MESSAGE": "2026-09-17 16:07:28,916 INFO gpu_fault.collectors.logs.fabric "
            "manager collector caught up",
            "__CURSOR": "s=fixture;i=plain",
            "_SYSTEMD_INVOCATION_ID": projection["records"][0]["receipt"][
                "systemd_invocation_id"
            ],
            "_PID": str(projection["records"][0]["receipt"]["pid"]),
            "__MONOTONIC_TIMESTAMP": str(projection["records"][0]["monotonic_us"] + 1),
        }
    )

    def journalctl(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        lines = journal_lines(projection).splitlines()
        lines.insert(1, plain)
        if any(item.startswith("--grep=") for item in command):
            lines.reverse()  # the systemd 252 behaviour the live node showed
        return SimpleNamespace(returncode=0, stdout="\n".join(lines), stderr="")

    monkeypatch.setattr(
        probe, "service_snapshot", lambda: {probe.FM_UNIT: projection["service"]}
    )
    monkeypatch.setattr(probe, "run", journalctl)
    arguments = probe.parser().parse_args(
        ["fm-delivery-evidence", "--cursor", projection["anchor_cursor"]]
    )
    arguments.handler(arguments)
    result = json.loads(capsys.readouterr().out)
    assert not any(item.startswith("--grep=") for item in commands[0]), (
        'test_journal_probe_does_not_rely_on_grep_ordering_and_skips_plain_log_lines: expected no any(item.startswith("--grep=") for item in comma...'
    )
    assert result["records"] == projection["records"]
    clocks = [row["monotonic_us"] for row in result["records"]]
    assert clocks == sorted(clocks)
    # An out-of-order window (whatever produced it) is refused, never re-sorted.
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="\n".join(reversed(journal_lines(projection).splitlines())),
            stderr="",
        ),
    )
    with pytest.raises(probe.ProbeError, match="journal order"):
        probe.fm_delivery_evidence(SimpleNamespace(cursor=""))


@pytest.mark.parametrize(
    "problem",
    [
        "cursor",
        "inactive",
        "exit",
        "size",
        "json",
        "payload",
        "pid",
        "unit",
        "clock",
        "lost-anchor",
        "truncated",
        "changed",
    ],
)
def test_journal_probe_refuses_unproven_coverage_without_emitting_log_bodies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], problem: str
) -> None:
    projection = window()
    service = projection["service"]
    stdout = journal_lines(projection)
    if problem == "inactive":
        service["ActiveState"] = "inactive"
    elif problem == "size":
        stdout = "x" * (4 * 1024 * 1024 + 1)
    elif problem == "json":
        stdout = "not JSON"
    elif problem == "payload":
        stdout = json.dumps({"MESSAGE": "unrelated-private-message"})
    elif problem in {"pid", "unit", "clock"}:
        rows = [json.loads(line) for line in stdout.splitlines()]
        key = {
            "pid": "_PID",
            "unit": "_SYSTEMD_INVOCATION_ID",
            "clock": "__MONOTONIC_TIMESTAMP",
        }[problem]
        rows[0][key] = "unknown"
        stdout = "\n".join(json.dumps(row) for row in rows)
    elif problem == "lost-anchor":
        stdout = "\n".join(stdout.splitlines()[1:])
    elif problem == "truncated":
        monkeypatch.setattr(probe, "FM_JOURNAL_LIMIT", 1)
    if problem == "changed":
        states = iter([service, {**service, "InvocationID": "f" * 32}])
        monkeypatch.setattr(
            probe, "service_snapshot", lambda: {probe.FM_UNIT: next(states)}
        )
    else:
        monkeypatch.setattr(probe, "service_snapshot", lambda: {probe.FM_UNIT: service})
    monkeypatch.setattr(
        probe,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1 if problem == "exit" else 0, stdout=stdout
        ),
    )
    with pytest.raises(probe.ProbeError):
        probe.fm_delivery_evidence(
            SimpleNamespace(
                cursor="bad\ncursor" if problem == "cursor" else "s=fixture;i=1"
            )
        )
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("problem", [None, "initial-count", "replay", "workflow"])
def test_collect005_proves_source_no_replay_even_when_persisted_row_count_stays_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, problem: str | None
) -> None:
    monkeypatch.setattr(runner, "time", Clock())
    fixture = FmProducerFixture()
    fixture.problem = problem
    if problem in {"initial-count", "replay"}:
        with pytest.raises(RegionalFixtureError):
            runner.run_collect005(fixture, tmp_path, 1)
        assert ("restart-service" in fixture.calls) == (problem == "replay")
        if problem == "replay":
            assert len(fixture.store_snapshot("marker")["evidence"]) == 1
    else:
        result = runner.run_collect005(fixture, tmp_path, 1)
        assert result["verdict"] == ("PASS" if problem is None else "FAIL")
        assert len(result["producer_observation_samples"]) == 3
        assert result["persisted_record_counts"] == [1, 1, 1]
        assert fixture.calls.index("fm-delivery-evidence") < fixture.calls.index(
            "append-sxid"
        )


def test_wait_for_delivery_handles_inflight_attempt_but_not_a_sequence_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    incomplete = window()
    incomplete["records"] = incomplete["records"][:2]
    reads = iter([incomplete, window()])
    fixture = SimpleNamespace(execute=lambda *a: next(reads))
    proof = runner.wait_fm_delivery(
        fixture, baseline()["records"][0], record_id=RECORD_ID, **SCOPE
    )
    assert proof == window()
    assert clock.sleeps == [3]
    stuck = SimpleNamespace(execute=lambda *a: incomplete)
    with pytest.raises(RegionalFixtureError, match="delivery proof"):
        runner.wait_fm_delivery(
            stuck,
            baseline()["records"][0],
            record_id=RECORD_ID,
            timeout_seconds=1,
            **SCOPE,
        )


def test_consumer_accepts_actual_bounded_runtime_receipt_logger(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from gpu_fault.collectors.logs.fabric_manager_receipts import (
        FM_RECEIPT_FIELDS,
        FM_RECEIPT_PREFIX,
        FabricManagerReceiptLog,
    )

    assert probe.FM_RECEIPT_KEYS == FM_RECEIPT_FIELDS
    logger = logging.getLogger("collector-fm-acceptance-test")
    caplog.set_level(logging.INFO, logger=logger.name)

    def producer() -> FabricManagerReceiptLog:
        return FabricManagerReceiptLog(
            logger,
            cluster_id=SCOPE["cluster_id"],
            node_id=SCOPE["node"],
            boot_id=SCOPE["boot_id"],
            source_configuration="fixture-source",
            summary_seconds=300,
            monotonic=lambda: 0,
        )

    monkeypatch.setenv("INVOCATION_ID", OLD_UNIT)
    first = producer()
    first.complete_round(first.begin_round(), complete=True)
    active = first.begin_round()
    attempt = first.begin_attempt(active, record_id=RECORD_ID, source="file")
    first.complete_attempt(attempt, "DELIVERED")
    first.complete_round(active, complete=True)
    monkeypatch.setenv("INVOCATION_ID", NEW_UNIT)
    second = producer()
    for _ in range(2):
        second.complete_round(second.begin_round(), complete=True)
    bodies = [
        json.loads(row.getMessage().removeprefix(FM_RECEIPT_PREFIX))
        for row in caplog.records
        if row.name == logger.name
    ]
    assert len(bodies) == 6
    projection = window(restarted=True)
    projection["service"]["MainPID"] = str(os.getpid())
    for row, body in zip(projection["records"], bodies, strict=True):
        row["receipt"] = body
    anchor = projection["records"][0]
    assert (
        evidence.delivery_window_errors(
            anchor,
            projection,
            record_id=RECORD_ID,
            restarted_invocation=NEW_UNIT,
            **SCOPE,
        )
        == []
    )


def test_real_fm_collector_restart_receipts_match_the_live_consumer_without_store_upsert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from gpu_fault.channel_registry import FABRIC_MANAGER_PATH
    from gpu_fault.collectors.logs.fabric_manager import FabricManagerLogCollector
    from gpu_fault.collectors.logs.fabric_manager_receipts import FM_RECEIPT_PREFIX
    from gpu_fault.collectors.models import CollectorContext

    logger = "gpu_fault.collectors.logs.fabric_manager"
    caplog.set_level(logging.INFO, logger=logger)
    log = tmp_path / "private-fm.log"
    log.write_text("preexisting history\n")
    checkpoint = tmp_path / "private-cursor.json"
    sends = []
    sink = SimpleNamespace(post=lambda path, value: sends.append((path, value)) or {})
    now = datetime.now(timezone.utc)

    def collector() -> FabricManagerLogCollector:
        return FabricManagerLogCollector(
            sink,
            CollectorContext(cluster_id=SCOPE["cluster_id"]),
            node_id=SCOPE["node"],
            boot_id=SCOPE["boot_id"],
            journal_enabled=False,
            log_paths=[str(log)],
            state_path=str(checkpoint),
            now=lambda: now,
            monotonic=lambda: 0,
        )

    monkeypatch.setenv("INVOCATION_ID", OLD_UNIT)
    first = collector()
    first.collect_once()
    with log.open("a") as stream:
        stream.write(
            f"[{now.isoformat()}] nvidia-nvswitch0: SXid (PCI:0000:ab:00.0): "
            "99999, Non-fatal, Link 3 marker=private-test\n"
        )
    assert first.collect_once().delivered == 1
    record_id = next(
        value["record_id"] for path, value in sends if path == FABRIC_MANAGER_PATH
    )
    monkeypatch.setenv("INVOCATION_ID", NEW_UNIT)
    second = collector()
    assert second.collect_once().delivered == 0
    assert second.collect_once().delivered == 0
    assert sum(path == FABRIC_MANAGER_PATH for path, _value in sends) == 1
    bodies = [
        json.loads(row.getMessage().removeprefix(FM_RECEIPT_PREFIX))
        for row in caplog.records
        if row.name == logger and row.getMessage().startswith(FM_RECEIPT_PREFIX)
    ]
    projection = window(restarted=True)
    projection["service"]["MainPID"] = str(os.getpid())
    assert len(bodies) == len(projection["records"]) == 6
    for row, body in zip(projection["records"], bodies, strict=True):
        row["receipt"] = body
    assert (
        evidence.delivery_window_errors(
            projection["records"][0],
            projection,
            record_id=record_id,
            restarted_invocation=NEW_UNIT,
            **SCOPE,
        )
        == []
    )
