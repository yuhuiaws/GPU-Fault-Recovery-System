from __future__ import annotations

from scripts.e2e.regional.capacity_queued_lease import run_queued_lease_proof
from scripts.e2e.regional.probes.cap004_commands import command_snapshot
from tests.regional.test_cap004_executor_proof import RUN_ID, TOKEN, URL
from tests.regional.test_cap004_executor_proof import local_api as shared_api_fixture

local_api = shared_api_fixture


def test_nondefault_batch_two_concurrency_one_does_not_execute_a_stolen_queued_lease(
    local_api, tmp_path
) -> None:
    command_snapshot(local_api.store, RUN_ID, "seed")
    proof = run_queued_lease_proof(URL, TOKEN, RUN_ID, tmp_path, timeout_seconds=5)
    assert proof["passed"] is True
    assert proof["adapter_calls"] == 1 and proof["queued_adapter_calls"] == 0
    assert proof["stale_renewal_status"] == 409
    assert proof["competitor_token_changed"] is True
    assert proof["executor_counters"]["results_withheld_total"] == 1
    snapshot = command_snapshot(local_api.store, RUN_ID, "inspect")
    assert snapshot["status_counts"] == {"WAITING": 25}
    assert not any(row["lease_present"] for row in snapshot["commands"]), (
        "the queue proof must hand back every owned lease"
    )
