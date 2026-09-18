"""First-result receipt requirements through the complete, local NET002 runner."""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts.e2e.regional import run_net002_command_recovery as net002
from tests.regional import _cov95_net_commands as support
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401

command_runner = support.command_runner
pytestmark = pytest.mark.parametrize(
    "command_runner", [net002], indirect=True, ids=["net002"]
)
RECEIPT_PATH = "/state/first-result-submission.json"


def report(harness: Any) -> dict[str, Any]:
    return json.loads(
        (harness.root / "cases" / net002.CASE_ID / f"{net002.CASE_ID}.json").read_text()
    )


def test_runner_preserves_first_receipt_before_and_after_reclaim(
    command_runner: Any,
) -> None:
    harness = command_runner
    assert support.run_command(harness) == 0
    document = report(harness)
    expected = harness.states[RECEIPT_PATH]
    assert document["first_result_submission"] == expected
    assert document["first_result_submission_after_reclaim"] == expected
    reads = [
        index
        for index, (name, args) in enumerate(harness.calls)
        if name == "read-state" and args == (RECEIPT_PATH,)
    ]
    reclaim = next(
        index for index, (name, _) in enumerate(harness.calls) if name == "wait-command"
    )
    assert len(reads) == 2
    assert reads[0] < reclaim < reads[1]
    assert document["limitations"] == net002.LIMITATIONS


@pytest.mark.parametrize(
    "replacement",
    [
        {},
        {"status_code": None, "stale_lease_reason": None},
        {"status_code": 409, "stale_lease_reason": "unrelated conflict"},
        {"command_id": "remote-unrelated"},
        {"submission_index": 2},
        {"submitted_at_epoch": 1.0},
    ],
    ids=[
        "missing",
        "renewal-only-409",
        "conflict",
        "foreign-command",
        "later",
        "early",
    ],
)
def test_runner_rejects_bad_receipts_despite_all_legacy_log_proofs(
    command_runner: Any, replacement: dict[str, Any]
) -> None:
    harness = command_runner
    if replacement:
        harness.states[RECEIPT_PATH].update(replacement)
    else:
        harness.states[RECEIPT_PATH] = {}
    assert support.run_command(harness) == 1
    document = report(harness)
    assert document["verdict"] == "FAIL"
    assert any(
        "first result" in error or "first attempt" in error
        for error in document["errors"]
    ), (
        'test_runner_rejects_bad_receipts_despite_all_legacy_log_proofs: expected any( "first result" in error or "first attempt" in error for err...'
    )
    assert any(name == "cleanup" for name, _ in harness.calls), (
        'test_runner_rejects_bad_receipts_despite_all_legacy_log_proofs: expected any(name == "cleanup" for name, _ in harness.calls)'
    )


@pytest.mark.parametrize(
    "replacement", [{}, {"status_code": 200}], ids=["missing", "cached"]
)
def test_runner_rejects_receipt_replacement_after_first_read(
    command_runner: Any, monkeypatch: pytest.MonkeyPatch, replacement: dict[str, Any]
) -> None:
    harness = command_runner
    read_state = harness.transport.read_state
    reads = 0

    def read(probe: Any, path: str) -> dict[str, Any]:
        nonlocal reads
        value = read_state(probe, path)
        if path == RECEIPT_PATH:
            reads += 1
            if reads == 2:
                return replacement
        return value

    monkeypatch.setattr(harness.transport, "read_state", read)
    assert support.run_command(harness) == 1
    document = report(harness)
    assert "first result submission receipt changed after reclaim" in document["errors"]
    assert document["first_result_submission"]["status_code"] == 409
    assert document["first_result_submission_after_reclaim"] == replacement
