from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from scripts.e2e.regional import destr015_verdicts as parallel
from scripts.e2e.regional import destr018_verdicts as lifetime
from scripts.e2e.regional import destr021_verdicts as metadata
from scripts.e2e.regional import destr022_verdicts as spare
from scripts.e2e.regional import destr023_verdicts as coverage
from scripts.e2e.regional import destr024_verdicts as watcher
from scripts.e2e.regional import run_destr001_gpu_reset as reset
from scripts.e2e.regional import run_destr010_fabric_manager_restart as fabric
from tests.regional.test_destr015_parallel_branch_join import (
    NODES,
    happy_budget,
    happy_incident,
    happy_workflow,
)
from tests.regional.test_destr023_idle_cluster_reset import coverage as covered_cluster


@pytest.mark.parametrize("state", ["FAILED", "IN_PROGRESS", None])
def test_reset_requires_a_successful_ledger_result(state: str | None) -> None:
    before = {
        "ledger": [],
        "gpu_inventory": [{"uuid": "GPU-a"}],
        "gpu_fault_timers": [],
        "services": {},
    }
    after = {
        **before,
        "ledger": [
            {
                "command_id": "reset/node-a/agent-1",
                "operation": "RESET_GPU",
                "attempt": 1,
                "state": state,
            }
        ],
        "sampler": {"sample_count": 3, "min_gpu_count": 0, "last": {"gpu_count": 1}},
        "kernel_reset_journal": {"target_reset_count": 0},
    }

    errors = reset.host_errors(
        before, after, expected_gpu_count=1, target_bdf="0000:01:00.0"
    )

    assert any("SUCCEEDED" in error for error in errors), errors


def fabric_snapshots() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    before = {
        "fabric_manager": {
            "MainPID": "10",
            "InvocationID": "old",
            "ActiveState": "active",
        },
        "ledger": [],
        "gpu_fault_timers": [],
        "journal": {"started_count": 0},
    }
    first = {
        "fabric_manager": {
            "MainPID": "11",
            "InvocationID": "new",
            "ActiveState": "active",
        },
        "ledger": [
            {
                "command_id": "fm/node-a/agent-1",
                "operation": "RESTART_FABRIC_MANAGER",
                "attempt": 1,
                "state": "SUCCEEDED",
            }
        ],
        "gpu_fault_timers": [],
        "journal": {"started_count": 1},
    }
    return before, first, deepcopy(first)


def test_fabric_replay_cannot_hide_an_attempt_under_an_existing_command_id() -> None:
    before, first, replay = fabric_snapshots()
    replay["ledger"].append({**replay["ledger"][0], "attempt": 2})

    errors = fabric.host_errors(before, first, replay, "fm/node-a/agent-1")

    assert any("ledger" in error for error in errors), errors


@pytest.mark.parametrize("field", ["InvocationID", "started_count", "state"])
def test_fabric_replay_compares_service_and_ledger_evidence(field: str) -> None:
    before, first, replay = fabric_snapshots()
    if field == "InvocationID":
        replay["fabric_manager"][field] = "unexpected-restart"
    elif field == "started_count":
        replay["journal"][field] = 2
    else:
        replay["ledger"][0][field] = "FAILED"

    assert fabric.host_errors(before, first, replay, "fm/node-a/agent-1"), field


@pytest.mark.parametrize("field", ["started_at", "updated_at"])
@pytest.mark.parametrize("value", [None, "", "invalid", "2026-09-06T10:00:00"])
def test_parallel_proof_requires_every_execution_timestamp(
    field: str, value: Any
) -> None:
    workflow = happy_workflow()
    workflow["step_executions"][2][field] = value

    errors = parallel.workflow_errors(
        workflow,
        happy_incident(),
        nodes=NODES,
        expected_gpu_count=16,
        restart_budget=happy_budget(),
    )

    assert any("timestamp" in error for error in errors), errors


def test_parallel_join_requires_its_own_start_timestamp() -> None:
    workflow = happy_workflow()
    workflow["step_executions"][-1].pop("started_at")

    errors = parallel.workflow_errors(
        workflow,
        happy_incident(),
        nodes=NODES,
        expected_gpu_count=16,
        restart_budget=happy_budget(),
    )

    assert any("timestamp" in error for error in errors), errors


def test_compensation_must_start_after_cancellation_not_merely_finish_after() -> None:
    cancelled = datetime(2026, 9, 12, tzinfo=timezone.utc)
    row = {
        "command_id": "restore/node-a/agent-1",
        "operation": lifetime.COMPENSATION_STEP,
        "state": "SUCCEEDED",
        "started_at": (cancelled - timedelta(seconds=1)).isoformat(),
        "completed_at": (cancelled + timedelta(seconds=1)).isoformat(),
    }

    errors = lifetime.compensation_row_errors(
        [row], t_cancel=cancelled, baseline_command_ids=set()
    )

    assert any("before" in error for error in errors), errors


def test_spare_reclaim_needs_at_least_one_fresh_counter_witness() -> None:
    cancelled = datetime(2026, 9, 12, tzinfo=timezone.utc)
    stale = {
        "pod": "executor-a",
        "claim_state": {
            "counters": {spare.RECLAIM_COUNTER: 3},
            "last_successful_claim_at": (cancelled - timedelta(seconds=10)).isoformat(),
        },
    }

    errors = spare.counter_errors([stale], [deepcopy(stale)], reclaimed_at=cancelled)

    assert any("breadcrumb" in error for error in errors), errors


@pytest.mark.parametrize(
    "overrides",
    [
        {"running": True},
        {"elapsed_seconds": None},
        {"elapsed_seconds": float("nan")},
        {"elapsed_seconds": float("inf")},
        {"error": "patch transport failed"},
    ],
)
def test_writer_requires_a_stopped_observed_healthy_loop(
    overrides: dict[str, Any],
) -> None:
    report = {
        "patches": 60,
        "elapsed_seconds": 90.0,
        "running": False,
        "stopped": True,
        "cleared": True,
        "exceeded_max_seconds": False,
        **overrides,
    }

    assert metadata.writer_errors(report), overrides


@pytest.mark.parametrize("missing", ["watched_pods", "watched_attempts", "observed_at"])
def test_idle_coverage_requires_explicit_heartbeat_fields(missing: str) -> None:
    sample = covered_cluster()
    sample["heartbeat"].pop(missing)

    assert coverage.fresh_coverage_errors(sample), missing


@pytest.mark.parametrize("age", [-1.0, float("nan"), float("inf")])
def test_idle_coverage_rejects_unusable_heartbeat_age(age: float) -> None:
    sample = covered_cluster()
    sample["heartbeat_age_seconds"] = age

    assert coverage.fresh_coverage_errors(sample), age


def test_stale_coverage_cannot_treat_unknown_heartbeat_age_as_expired() -> None:
    sample = covered_cluster(heartbeat_age=700.0, state="UNKNOWN")
    sample["heartbeat_age_seconds"] = None

    assert coverage.stale_coverage_errors(sample), sample


def test_watcher_down_rejects_a_new_attempt_under_an_old_physical_command() -> None:
    row = {
        "command_id": "reset/node-a/agent-1",
        "operation": "RESET_GPU",
        "attempt": 1,
        "state": "FAILED",
    }
    before = {"ledger": [row], "gpu_inventory": [], "services": {}}
    after = {**before, "ledger": [row, {**row, "attempt": 2}]}

    assert watcher.host_untouched_errors(before, after), after


@pytest.mark.parametrize(
    "field", ["observation_count", "probed_at", "latest_observation_at"]
)
def test_coverage_cannot_infer_missing_inventory_fields(field: str) -> None:
    sample = covered_cluster()
    sample.pop(field)
    assert coverage.fresh_coverage_errors(sample), field


def test_coverage_age_must_match_the_observed_timestamp() -> None:
    sample = covered_cluster(heartbeat_age=700.0)
    sample["heartbeat_age_seconds"] = 1.0
    assert coverage.fresh_coverage_errors(sample), (
        "the age cannot contradict its timestamp"
    )


@pytest.mark.parametrize(
    "field",
    [
        "uid",
        "generation",
        "observed_generation",
        "status_replicas",
        "updated_replicas",
        "available_replicas",
    ],
)
def test_ready_count_alone_does_not_prove_watcher_recovery(field: str) -> None:
    from tests.regional.test_destr023_idle_cluster_reset import deployment

    summary = coverage.deployment_summary(deployment())
    summary.pop(field)
    assert coverage.watcher_errors(summary), field


def test_a_heartbeat_from_before_restore_is_not_recovery() -> None:
    before = covered_cluster(heartbeat_age=700.0, state="UNKNOWN")
    after = covered_cluster(heartbeat_age=5.0)
    restored = datetime.fromisoformat(after["probed_at"])
    assert watcher.heartbeat_recovered_errors(before, after, not_before=restored), (
        "a pre-restore heartbeat cannot prove recovery"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"uid": "replacement"},
        {"observed_at": "2026-09-12T00:00:00+00:00"},
        {"claim_state_error": "unreadable"},
    ],
)
def test_reclaim_counters_require_the_same_fresh_executor(
    change: dict[str, Any],
) -> None:
    now = datetime(2026, 9, 12, tzinfo=timezone.utc)

    def record(offset: int, counter: int) -> dict[str, Any]:
        claimed = now + timedelta(seconds=offset)
        return {
            "pod": "executor-a",
            "uid": "original",
            "observed_at": (claimed + timedelta(seconds=1)).isoformat(),
            "claim_state": {
                "last_successful_claim_at": claimed.isoformat(),
                "counters": {spare.RECLAIM_COUNTER: counter},
            },
        }

    before, after = record(-10, 3), record(10, 4)
    assert spare.counter_errors([before], [after], reclaimed_at=now) == []
    assert spare.counter_errors([before], [{**after, **change}], reclaimed_at=now), (
        change
    )


@pytest.mark.parametrize("value", [True, -1, 1.5, "1", float("inf")])
def test_reclaim_counter_is_an_observed_nonnegative_integer(value: Any) -> None:
    assert spare.counter_of({"counters": {spare.RECLAIM_COUNTER: value}}) is None


def test_sibling_gpu_disappearance_cannot_prove_the_target_reset() -> None:
    before = {
        "gpu_inventory": [
            {"uuid": "GPU-a", "pci_bdf": "0000:01:00"},
            {"uuid": "GPU-b", "pci_bdf": "0000:02:00"},
        ],
        "ledger": [],
        "services": {},
        "gpu_fault_timers": [],
    }
    after = {
        **before,
        "ledger": [
            {
                "command_id": "reset",
                "operation": "RESET_GPU",
                "state": "SUCCEEDED",
                "attempt": 1,
                "gpu_uuids": ["GPU-a"],
            }
        ],
        "sampler": {
            "sample_count": 3,
            "min_gpu_count": 1,
            "last": {"gpu_count": 2, "gpu_uuids": ["GPU-a", "GPU-b"]},
            "observed_gpu_uuid_sets": [["GPU-b"]],
        },
    }
    assert (
        reset.host_errors(before, after, expected_gpu_count=2, target_bdf="0000:01:00")
        == []
    )
    after["sampler"]["observed_gpu_uuid_sets"] = [["GPU-a"]]
    errors = reset.host_errors(
        before, after, expected_gpu_count=2, target_bdf="0000:01:00"
    )
    assert any("only the target" in error for error in errors), errors


def test_node_ledger_correlation_uses_the_actual_node_action_id() -> None:
    row = {
        "command_id": "verify/node-a/agent-4",
        "operation": "VERIFY_NO_GPU_CLIENTS",
        "workflow_request_id": "workflow-a",
        "incident_id": "incident-a",
        "fencing_token": 3,
        "agent_generation": 4,
    }
    command = {
        "command_id": "remote-a",
        "idempotency_key": "verify",
        "workflow_request_id": "workflow-a",
        "incident_id": "incident-a",
        "fencing_token": 3,
        "step": {"operation": "VERIFY_NO_GPU_CLIENTS", "node_ids": ["node-a"]},
    }
    assert lifetime.ledger_command_matches(row, command), "exact node command identity"
    assert not lifetime.ledger_command_matches({**row, "fencing_token": 2}, command), (
        "a different workflow fence cannot match"
    )
    assert not lifetime.ledger_command_matches(
        {**row, "agent_generation": 5}, command
    ), "a different Agent generation cannot match"
