"""Command probes use the real executor with fake transport and local ledgers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gpu_fault.cluster_executor import ClusterActionExecutor
from gpu_fault.models import WorkflowOperation
from scripts.e2e.regional import cmd017_verdicts, cmd018_verdicts
from scripts.e2e.regional.probes import cmd017_barrier_executor as barrier
from scripts.e2e.regional.probes import cmd018_ledger_executor as ledger
from tests.execution.test_cluster_executor_lease_and_report import (
    FakeExecutorClient,
    remote_command,
)
from tests.regional import _cov95_residual_support as support

residual_isolation = support.residual_isolation


@pytest.fixture(params=[barrier, ledger], ids=["barrier", "ledger"])
def probe(request, monkeypatch, tmp_path, residual_isolation):
    module = request.param
    for name in (
        "STATE",
        "READY",
        "LEDGER",
        "EXECUTOR_STATE",
        "CLAIM_STATE",
        "ADAPTER_EXECUTED",
    ):
        if hasattr(module, name):
            path = tmp_path if name == "STATE" else tmp_path / (name.lower() + ".json")
            monkeypatch.setattr(module, name, path)
    tokens = tmp_path / "registration.json"
    tokens.write_text(json.dumps([{"cluster_id": "cluster-a", "token": "test-token"}]))
    support.redirect_path(monkeypatch, module, {"/tokens/clusters.json": tokens})
    for name, value in {
        "CONTROL_PLANE_URL": "https://control.invalid/",
        "EXECUTOR_ARTIFACT_SHA256": "a" * 64,
        "EXECUTOR_COMPATIBILITY_DIGEST": "b" * 64,
    }.items():
        monkeypatch.setenv(name, value)
    return module


def test_default_probe_entrypoint_uses_real_executor_without_any_node_transport(
    probe, monkeypatch
):
    operation = (
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
        if probe is barrier
        else WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT
    )
    nodes = ["synthetic-a", "synthetic-b"] if probe is barrier else ["synthetic-a"]
    command = remote_command(
        "command-a", operation=operation, node_ids=nodes, execution_owner=probe.OWNER
    )
    client = FakeExecutorClient([command], [command])
    constructed = []
    client_calls = []
    recorder_threads = []

    class TwoRounds(ClusterActionExecutor):
        def run(self):
            self.run_once()
            self.run_once()

    def executor(*args, **kwargs):
        instance = TwoRounds(*args, **kwargs)
        constructed.append(instance)
        return instance

    def transport(*args, **kwargs):
        client_calls.append((args, kwargs))
        return client

    monkeypatch.setattr(probe, "RegionalExecutorClient", transport)
    monkeypatch.setattr(probe, "ClusterActionExecutor", executor)
    monkeypatch.setattr(
        probe,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: recorder_threads.append(kwargs)),
    )
    probe.main()
    ready = json.loads(probe.READY.read_text())
    assert ready["owner"] == probe.OWNER
    assert ready["cluster_id"] == "cluster-a"
    assert client_calls[0][0][0] == "https://control.invalid"
    assert client_calls[0][1]["ca_file"] == "/tls/ca.crt"
    assert client_calls[0][1]["timeout_seconds"] == probe.HTTP_TIMEOUT_SECONDS
    assert recorder_threads[0]["target"] is probe.record_executor_state
    assert len(client.completed) == 2

    clock = support.Clock()

    def stop(seconds):
        clock.sleep(seconds)
        raise support.StopLoop

    monkeypatch.setattr(probe, "time", SimpleNamespace(time=clock.time, sleep=stop))
    with pytest.raises(support.StopLoop):
        probe.record_executor_state(constructed[0])
    snapshot = json.loads(probe.EXECUTOR_STATE.read_text())
    assert snapshot["claimed_total"] == 2
    if probe is barrier:
        assert cmd017_verdicts.executor_state_errors(snapshot) == []
        assert not probe.ADAPTER_EXECUTED.exists(), (
            "barrier hold must not reach the stand-in adapter"
        )
        assert [result.status.value for _, result in client.completed] == [
            "WAITING",
            "WAITING",
        ]
    else:
        saved = json.loads(probe.LEDGER.read_text())
        assert saved["physical_count"] == 1
        assert saved["keys"] == ["idem-command-a"]
        assert snapshot["ledger"] == saved
        assert [result.status.value for _, result in client.completed] == [
            "SUCCEEDED",
            "SUCCEEDED",
        ]
        assert (
            cmd018_verdicts.executed_once_errors(
                saved,
                {"command_id": "command-a", "status": "SUCCEEDED"},
                first_id="command-a",
            )
            == []
        )


def test_adapter_owner_and_direct_execution_leave_explicit_local_evidence(probe):
    adapter = (
        probe.BarrierStandInAdapter() if probe is barrier else probe.LedgerAdapter()
    )
    assert adapter.supports(SimpleNamespace(execution_owner=probe.OWNER)), (
        "probe adapter must advertise only its seeded owner"
    )
    assert not adapter.supports(SimpleNamespace(execution_owner="unrelated")), (
        "foreign commands must not be selected by the probe adapter"
    )
    context = SimpleNamespace(
        idempotency_key="key-a",
        command_id="command-a",
        step=SimpleNamespace(
            operation=WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
            node_ids=["synthetic-a"],
        ),
    )
    result = adapter.execute(context)
    if probe is barrier:
        assert result.status.value == "FAILED"
        marker = json.loads(probe.ADAPTER_EXECUTED.read_text())
        assert marker["idempotency_key"] == "key-a"
        assert marker["node_ids"] == ["synthetic-a"]
        assert cmd017_verdicts.adapter_marker_errors(marker), (
            "an adapter invocation must invalidate barrier acceptance"
        )
    else:
        assert result.status.value == "SUCCEEDED"
        repeated = probe.LedgerAdapter().execute(context)
        assert repeated.details["cached"] is True
        assert repeated.details["physical_count"] == 1


@pytest.mark.parametrize(
    "contents", ["[]", "{}", '{"physical_count":0,"keys":null}', "{"]
)
def test_corrupt_ledger_cannot_record_a_successful_execution(
    monkeypatch, tmp_path, contents
):
    path = tmp_path / "ledger.json"
    path.write_text(contents)
    monkeypatch.setattr(ledger, "LEDGER", path)
    context = SimpleNamespace(idempotency_key="key-a", command_id="command-a")
    with pytest.raises((KeyError, TypeError, ValueError)):
        ledger.LedgerAdapter().execute(context)
    assert path.read_text() == contents
