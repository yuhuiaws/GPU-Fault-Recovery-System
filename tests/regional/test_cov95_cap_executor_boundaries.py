from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from gpu_fault.cluster_executor import ClusterExecutorError
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import WorkflowExecutionRequest
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from scripts.e2e.regional import capacity_acceptance_executor as proof
from scripts.e2e.regional.probes import cap004_commands as commands
from tests.regional import test_cap004_executor_proof as cap_support
from tests.regional.test_cap004_executor_proof import RUN_ID, TOKEN, URL

local_api = cap_support.local_api


@pytest.mark.parametrize("failure", ["missing", "duplicate-lease", "handback"])
def test_bulk_api_rejects_partial_claims_duplicate_leases_and_unreleased_results(
    failure, local_api
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")
    first = []
    claims = []

    def transform(path, _payload, status, body):
        document = json.loads(body)
        if path.endswith("/claim"):
            claims.append(path)
            if len(claims) == 1:
                first.extend(document["commands"])
                if failure == "missing":
                    document["commands"].pop()
            elif failure == "duplicate-lease":
                document["commands"] = [first[0]]
        elif path.endswith("/result") and failure == "handback":
            document["status"] = "SUCCEEDED"
        return status, json.dumps(document).encode()

    local_api.transform = transform
    with pytest.raises(
        proof.Cap004Error, match="exactly 25|duplicated an active lease|handback"
    ):
        proof.measure_bulk_api(URL, TOKEN, RUN_ID)


@pytest.mark.parametrize("failure", ["lease", "duration"])
def test_executor_claim_requires_a_live_positive_duration_lease(
    failure, local_api
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")

    def transform(path, _payload, status, body):
        document = json.loads(body)
        if path.endswith("/claim"):
            row = document["commands"][0]
            if failure == "lease":
                row["lease_token"] = None
            else:
                row["updated_at"] = row["lease_expires_at"]
        return status, json.dumps(document).encode()

    local_api.transform = transform
    client = proof.Cap004Client(URL, TOKEN, RUN_ID)
    options = {
        "execution_owners": [commands.OWNER],
        "max_commands": 5,
        "lease_seconds": 10,
    }
    with pytest.raises(ClusterExecutorError, match="controlled claim"):
        client.claim("fixture-executor", **options)
    with pytest.raises(proof.Cap004Error, match="valid lease|duration is invalid"):
        client.claim("fixture-executor", **options)
    assert client.first_claimed.is_set() is False, (
        "invalid leases must never admit adapter execution"
    )


def test_result_on_the_initial_fault_claim_invalidates_capacity_proof(
    local_api,
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")
    client = proof.Cap004Client(URL, TOKEN, RUN_ID)
    options = {
        "execution_owners": [commands.OWNER],
        "max_commands": 5,
        "lease_seconds": 60,
    }
    with pytest.raises(ClusterExecutorError, match="controlled claim"):
        client.claim("fixture-executor", **options)
    claimed = {}
    for _batch in range(commands.COMMAND_COUNT // proof.EXECUTOR_BATCH_SIZE):
        claimed.update(
            (item.command_id, item)
            for item in client.claim("fixture-executor", **options)
        )
        if client.failure_command_id in claimed:
            break
    command = claimed[client.failure_command_id]
    client.complete(
        command,
        RemoteCommandResult(
            lease_token=str(command.lease_token), status=RemoteCommandStatus.SUCCEEDED
        ),
    )
    assert "result sent despite the injected lost lease" in client.problems, (
        "capacity proof must not pass without withholding the designated fault-claim result"
    )


def ledger(tmp_path):
    rows = commands.commands_for_run(RUN_ID)
    client = SimpleNamespace(
        expected={item.command_id: item for item in rows},
        failure_command_id=rows[0].command_id,
        original_lease_seconds={item.command_id: 10 for item in rows},
        renewed_once=lambda _command_id: False,
    )
    adapter = proof.NonphysicalLedgerAdapter(
        client, tmp_path / "ledger.sqlite", threading.Event()
    )
    return rows, adapter


def test_ledger_requires_an_active_lease_guard_before_any_mutation(tmp_path) -> None:
    rows, adapter = ledger(tmp_path)
    command = rows[0]
    context = WorkflowStepContext(
        workflow=command.workflow,
        incident=command.incident,
        step=command.step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=1),
        idempotency_key=command.idempotency_key,
    )
    try:
        with pytest.raises(proof.Cap004Error, match="owned command and live guard"):
            adapter.execute(context)
        assert adapter.snapshot() == [], (
            "unproved lease ownership must not write the ledger"
        )
    finally:
        adapter.close()


@pytest.mark.parametrize("failure", ["hold", "deadline"])
def test_healthy_action_refuses_lost_lease_or_expired_evidence_deadline(
    failure, tmp_path, monkeypatch
) -> None:
    rows, adapter = ledger(tmp_path)
    monkeypatch.setattr(
        proof,
        "lease_hold_reason",
        lambda: "fixture hold" if failure == "hold" else None,
    )
    monkeypatch.setattr(proof, "time", SimpleNamespace(monotonic=lambda: 2.0))
    adapter.action_timeout_seconds = 1
    try:
        with pytest.raises(
            proof.Cap004Error, match="healthy command lost|evidence deadline"
        ):
            adapter.wait_for_lease_evidence(rows[1].command_id, 0)
        assert adapter.snapshot() == [], (
            "failed lease evidence cannot create an action result"
        )
    finally:
        adapter.close()
