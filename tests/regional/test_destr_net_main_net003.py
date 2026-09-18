"""NET-003 action-gate and real product report-retry integration."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.cluster_executor import ClusterActionExecutor
from scripts.e2e.regional import run_net003_result_retry as runner
from scripts.e2e.regional.probes import net003_executor as probe
from tests.execution.test_cluster_executor_lease_and_report import (
    FakeExecutorClient,
    RecordingAdapter,
    remote_command,
)
from tests.regional.test_net003_result_retry import T0, _evidence


@pytest.fixture
def gated_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[probe.LedgerAdapter, list[dict[str, bool]]]:
    for name in ("BLOCK", "ACTION_STARTED", "ACTION_GATE_OBSERVED", "LEDGER"):
        monkeypatch.setattr(probe, name, tmp_path / name.lower())
    at_action: list[dict[str, bool]] = []

    def sleep(seconds: float) -> None:
        if seconds == 5:
            at_action.append(
                {
                    "gate_observed": probe.ACTION_GATE_OBSERVED.exists(),
                    "ledger_written": probe.LEDGER.exists(),
                }
            )

    monkeypatch.setattr(probe.time, "sleep", sleep)
    return probe.LedgerAdapter("notification-net003"), at_action


def test_the_probe_holds_its_action_until_the_network_gate_is_armed(
    gated_adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, at_action = gated_adapter
    clock = iter([0.0, 31.0])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="network block was not armed"):
        adapter.execute(SimpleNamespace(idempotency_key="key-1"))
    assert at_action == []
    assert not probe.ACTION_GATE_OBSERVED.exists(), (
        "an unarmed gate cannot emit an observed-gate receipt"
    )
    assert not probe.LEDGER.exists(), "an unarmed gate cannot commit an action"
    assert probe.ACTION_STARTED.exists(), "the runner must be able to observe the gate"


def test_the_probe_records_the_gate_then_acts_then_commits(gated_adapter: Any) -> None:
    adapter, at_action = gated_adapter
    probe.BLOCK.touch()
    first = adapter.execute(SimpleNamespace(idempotency_key="key-1"))
    observed = json.loads(probe.ACTION_GATE_OBSERVED.read_text())
    assert observed["idempotency_key"] == "key-1"
    assert at_action == [{"gate_observed": True, "ledger_written": False}]
    assert first.details["cached"] is False
    assert first.details["physical_count"] == 1
    replay = adapter.execute(SimpleNamespace(idempotency_key="key-1"))
    assert len(at_action) == 1
    assert replay.details["cached"] is True
    assert replay.details["physical_count"] == 1


@pytest.fixture
def interrupting_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> probe.InterruptingRegionalExecutorClient:
    for name in (
        "RESULT_SUBMIT_STARTED",
        "RESULT_INTERRUPTED",
        "RESULT_REPLAYS",
        "DROP_NEXT",
    ):
        monkeypatch.setattr(probe, name, tmp_path / name.lower())
    monkeypatch.setattr(
        probe.RegionalExecutorClient, "__init__", lambda self, *a, **k: None
    )
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    return probe.InterruptingRegionalExecutorClient()


def terminal(command_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        command_id=command_id,
        status=SimpleNamespace(value="SUCCEEDED"),
        status_source="executor-result",
        updated_at=T0 + timedelta(seconds=7),
    )


def test_a_reset_without_status_is_reported_then_replayed_only_by_the_caller(
    interrupting_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def complete(self: Any, command: Any, result: Any) -> Any:
        calls.append(command.command_id)
        if len(calls) == 1:
            raise probe.ClusterExecutorError("connection reset", status_code=None)
        return terminal(command.command_id)

    monkeypatch.setattr(probe.RegionalExecutorClient, "complete", complete)
    command = SimpleNamespace(command_id="command-net003")
    with pytest.raises(probe.ClusterExecutorError):
        interrupting_client.complete(command, {"status": "SUCCEEDED"})
    assert calls == [command.command_id], "the substitute client must not retry"
    assert not probe.RESULT_REPLAYS.exists(), (
        "the first lost response must not trigger a substitute-client replay"
    )
    interrupted = json.loads(probe.RESULT_INTERRUPTED.read_text())
    assert interrupted["first_post_succeeded"] is False
    assert interrupted["exception"] == "ClusterExecutorError"
    assert interrupted["status_code"] is None
    assert (
        interrupting_client.complete(command, {"status": "SUCCEEDED"}).status.value
        == "SUCCEEDED"
    )
    replay = json.loads(probe.RESULT_REPLAYS.read_text())
    assert calls == [command.command_id, command.command_id]
    assert replay["count"] == 1 and replay["retry_owner"] == "product-executor"
    assert replay["responses"][0]["command_id"] == command.command_id


def test_wrapped_transport_failure_is_retried_by_the_production_executor(
    interrupting_client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = interrupting_client
    client.cluster_id = "cluster-a"
    command = remote_command("command-net003")
    lease_client = FakeExecutorClient([command])
    calls: list[str] = []

    def complete(self: Any, current: Any, result: Any) -> Any:
        calls.append(current.command_id)
        if len(calls) == 1:
            raise probe.ClusterExecutorError("connection reset", status_code=None)
        return terminal(current.command_id)

    monkeypatch.setattr(probe.RegionalExecutorClient, "complete", complete)
    monkeypatch.setattr(client, "claim", lease_client.claim)
    monkeypatch.setattr(client, "renew", lease_client.renew)
    adapter = RecordingAdapter()
    admission_at_entry: list[list[tuple[str, str, int]]] = []
    execute = adapter.execute

    def record_entry(context: Any) -> Any:
        admission_at_entry.append(list(lease_client.renewals))
        return execute(context)

    monkeypatch.setattr(adapter, "execute", record_entry)
    delays: list[float] = []
    executor = ClusterActionExecutor(
        client,
        [adapter],
        executor_id="net003-integration",
        allowed_namespaces={"training"},
        lease_seconds=runner.LEASE_SECONDS,
        claim_state_path=str(tmp_path / "claim.json"),
        liveness_state_path=str(tmp_path / "liveness.json"),
        sleep=delays.append,
    )
    assert executor.run_once() == 1
    assert calls == [command.command_id, command.command_id]
    assert len(adapter.contexts) == 1
    assert admission_at_entry == [
        [(command.command_id, "net003-integration", runner.LEASE_SECONDS)]
    ], "the server lease must be renewed before the one adapter execution"
    assert len(delays) == 1
    assert executor.metrics_snapshot()["transport_retries_total"] == 1
    assert executor.metrics_snapshot()["reported_failures"] == 0
    assert executor.metrics_snapshot()["lease_lost_total"] == 0, (
        "the retry must run under a valid lease"
    )
    assert executor.metrics_snapshot()["results_withheld_total"] == 0, (
        "both result posts must be authorized"
    )
    assert json.loads(probe.RESULT_REPLAYS.read_text())["count"] == 1


@pytest.mark.parametrize("status_code", [403, 409, 503])
def test_received_http_rejections_are_never_marked_as_lost_responses(
    interrupting_client: Any, monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    calls: list[str] = []

    def complete(self: Any, command: Any, result: Any) -> Any:
        calls.append(command.command_id)
        raise probe.ClusterExecutorError("received rejection", status_code=status_code)

    monkeypatch.setattr(probe.RegionalExecutorClient, "complete", complete)
    with pytest.raises(probe.ClusterExecutorError) as caught:
        interrupting_client.complete(SimpleNamespace(command_id="command-net003"), None)
    assert caught.value.status_code == status_code
    assert calls == ["command-net003"]
    assert not probe.RESULT_INTERRUPTED.exists(), (
        "a received HTTP rejection is not a lost-response injection"
    )
    assert not probe.RESULT_REPLAYS.exists(), (
        "a rejected HTTP response cannot manufacture a replay receipt"
    )


@pytest.mark.parametrize(
    "record",
    [
        "leased",
        "committed",
        "final",
        "result_interrupted",
        "result_replays",
        "response",
    ],
)
def test_all_result_exchange_records_must_identify_one_command(record: str) -> None:
    evidence = _evidence()
    target = (
        evidence[record]
        if record != "response"
        else evidence["result_replays"]["responses"][0]
    )
    target["command_id"] = "foreign"
    errors = runner.net003_errors(**evidence)
    assert "result exchange records do not identify the same command" in errors


@pytest.mark.parametrize(
    "field", ["workflow_request_id", "incident_id", "cluster_id", "idempotency_key"]
)
def test_a_success_cannot_be_borrowed_from_another_execution(field: str) -> None:
    evidence = _evidence()
    evidence["final"][field] = "foreign"
    assert f"result exchange changed or omitted {field}" in runner.net003_errors(
        **evidence
    )


@pytest.mark.parametrize(
    "counter",
    [
        "claimed_total",
        "reported_failures",
        "unexpected_failures",
        "lease_renewal_failures",
        "transport_retries_total",
    ],
)
@pytest.mark.parametrize("value", [None, True, "0"])
def test_missing_or_coerced_counters_cannot_prove_the_retry(
    counter: str, value: Any
) -> None:
    evidence = _evidence()
    evidence["executor_state"][counter] = value
    assert runner.net003_errors(**evidence), (
        "missing or coerced counters cannot establish product-owned retry"
    )


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -1])
def test_replay_timestamps_must_be_finite_epochs(value: Any) -> None:
    evidence = _evidence()
    evidence["result_replays"]["replay_sent_at_epoch"] = value
    assert "replay time or command update time is missing" in runner.net003_errors(
        **evidence
    )


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "field", ["quiet_seconds", "replay_delay_seconds", "http_timeout_seconds"]
)
def test_timing_margins_do_not_accept_nonfinite_or_invalid_values(
    field: str, value: Any
) -> None:
    assert runner.timing_errors(**{field: value}), (
        "nonfinite or nonpositive timings cannot authorize the result exchange"
    )


def test_replay_must_leave_the_actual_first_result_unchanged() -> None:
    evidence = _evidence()
    evidence["final"]["result_details"]["extra"] = "rewritten"
    assert "terminal replay changed the first committed result" in runner.net003_errors(
        **evidence
    )


def test_the_interrupted_receipt_cannot_hide_a_received_http_status() -> None:
    evidence = _evidence()
    evidence["result_interrupted"]["status_code"] = 409
    assert (
        "the client did not see a transport error on the first post"
        in runner.net003_errors(**evidence)
    )


def test_action_gate_requires_a_fresh_receipt_for_the_seed_key() -> None:
    seed = {"idempotency_key": "key-net003"}
    receipt = {"idempotency_key": "key-net003", "observed_at_epoch": T0.timestamp()}
    assert runner.action_gate_errors(receipt, seed, armed_at=T0, observed_at=T0) == []
    receipt["observed_at_epoch"] -= 60
    assert runner.action_gate_errors(receipt, seed, armed_at=T0, observed_at=T0), (
        "a receipt predating gate arming must be rejected"
    )
    receipt.update(observed_at_epoch=T0.timestamp(), idempotency_key="foreign")
    assert runner.action_gate_errors(receipt, seed, armed_at=T0, observed_at=T0), (
        "a foreign idempotency key cannot prove this seed's gate"
    )
