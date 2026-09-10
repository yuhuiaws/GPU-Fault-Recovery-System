"""Contract tests for GF-REGIONAL-DESTR-023.

The case proves that a *truly* idle cluster -- no managed attempt, no
coverage canary, every attempt observation older than the freshness window --
still resolves IDLE through the completion watcher's coverage heartbeat, so an
XID 46 on an idle node compiles and runs the eight-step RESET_GPU contract
instead of falling closed to ``node workload state is UNKNOWN``.

Every verdict is judged against synthetic coverage probes, Pod listings and
store snapshots: once on the intended run and once per way the run can be
wrong. Nothing here touches a cluster.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from scripts.e2e.regional import destr023_verdicts as verdicts
from scripts.e2e.regional import run_destr023_idle_cluster_reset as destr023
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata

NODE = "node-a"
NOW = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
FRESHNESS = 600.0


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def coverage(
    *,
    heartbeat_age: float | None = 20.0,
    observation_age: float | None = None,
    state: str = "IDLE",
    supported: bool = True,
    watched_pods: int = 0,
) -> dict[str, Any]:
    heartbeat = None
    if heartbeat_age is not None:
        heartbeat = {
            "cluster_id": "cluster-a",
            "observed_at": (NOW - timedelta(seconds=heartbeat_age)).isoformat(),
            "watched_pods": watched_pods,
            "watched_attempts": 0,
            "resource_version": "4711",
            "watcher_instance": "completion-watcher-0",
        }
    return {
        "heartbeat_supported": supported,
        "probed_at": NOW.isoformat(),
        "freshness_seconds": FRESHNESS,
        "max_age_seconds": 120,
        "heartbeat": heartbeat,
        "heartbeat_age_seconds": heartbeat_age,
        "observation_count": 0 if observation_age is None else 1,
        "latest_observation_age_seconds": observation_age,
        "workload_state": state,
    }


def _step(operation: str) -> dict[str, Any]:
    return {"operation": operation, "node_ids": [NODE]}


def _command(operation: str, status: str = "SUCCEEDED") -> dict[str, Any]:
    return {"status": status, "step": {"operation": operation}}


def reset_state(*, status: str = "SUCCEEDED") -> dict[str, Any]:
    """A store snapshot that satisfies DESTR-001's ``workflow_errors``."""

    steps = list(verdicts.RESET_OPERATIONS)
    executions = [
        {
            "operation": operation,
            "status": "SUCCEEDED",
            "adapter_operation_id": f"remote/{operation.lower()}",
            "details": {},
        }
        for operation in steps
    ]
    return {
        "event": {"xid": 46, "evidence_ref": "kmsg://node-a/1"},
        "decision": {"official_action": "RESET_GPU"},
        "incident": {"incident_id": "inc-destr023", "state": "RECOVERED"},
        "workflow": {
            "request_id": "wf-destr023",
            "status": status,
            "official_action": "RESET_GPU",
            "official_steps": [_step(item) for item in steps],
            "completed_operations": steps if status == "SUCCEEDED" else [],
            "step_executions": executions if status == "SUCCEEDED" else [],
            "blocked_reasons": [],
        },
        "observed_waiting_step_executions": [],
        "commands": [
            _command(item)
            for item in (
                "MARK_UNSCHEDULABLE",
                "QUIESCE_GPU_SERVICES",
                "VERIFY_NO_GPU_CLIENTS",
                "RESET_GPU",
                "RESTORE_GPU_SERVICES",
                "RESTORE_SCHEDULING",
            )
        ],
    }


def deployment(*, replicas: int = 1, ready: int = 1) -> dict[str, Any]:
    return {
        "metadata": {"uid": "dep-1", "generation": 3},
        "spec": {"replicas": replicas, "strategy": {"type": "Recreate"}},
        "status": {"readyReplicas": ready},
    }


# --------------------------------------------------------------------------- #
# Case contract
# --------------------------------------------------------------------------- #
def test_the_case_identity_and_predecessor_are_pinned() -> None:
    metadata = RegionalCaseMetadata(
        case_id=destr023.CASE_ID,
        title="",
        category="regional-destructive-acceptance",
        level="staging",
        risk="destructive",
        automation="manual",
        procedure="docs/x.md#gf-regional-destr-023",
        predecessor=destr023.PREDECESSOR_CASE_ID,
    )
    assert destr023.CASE_ID == "GF-REGIONAL-DESTR-023"
    assert destr023.CONFIRMATION == metadata.confirmation == "DESTR023_EXECUTE"
    assert destr023.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-001", (
        "the reset contract itself is proven by DESTR-001; this case only adds "
        "the heartbeat-only coverage premise"
    )
    assert verdicts.CASE_ID == destr023.CASE_ID


def test_the_parser_is_plan_only_by_default() -> None:
    arguments = destr023.parser().parse_args(["--run-dir", "/tmp/destr023-test"])
    assert arguments.execute is False
    assert arguments.plan is False
    assert hasattr(arguments, "predecessor_evidence"), (
        "the shared predecessor-evidence flag must be wired"
    )
    assert hasattr(arguments, "host_probe_image"), (
        "the shared host-probe-image flag must be wired"
    )


def test_the_unknown_reason_matches_the_workflow_builder() -> None:
    from gpu_fault.orchestration.workflow_builder import WORKLOAD_STATE_UNKNOWN_REASON

    assert verdicts.WORKLOAD_STATE_UNKNOWN_REASON == WORKLOAD_STATE_UNKNOWN_REASON


# --------------------------------------------------------------------------- #
# Coverage probe verdicts
# --------------------------------------------------------------------------- #
def test_a_fresh_heartbeat_with_expired_observations_is_heartbeat_only_idle() -> None:
    assert verdicts.fresh_coverage_errors(coverage()) == []
    assert verdicts.fresh_coverage_errors(coverage(observation_age=650.0)) == []


def test_fresh_coverage_refuses_a_control_plane_without_the_heartbeat() -> None:
    errors = verdicts.fresh_coverage_errors(coverage(supported=False))
    assert "heartbeat" in _text(errors) and "control plane" in _text(errors)


def test_fresh_coverage_refuses_a_missing_or_stale_heartbeat() -> None:
    assert "no coverage heartbeat" in _text(
        verdicts.fresh_coverage_errors(coverage(heartbeat_age=None, state="UNKNOWN"))
    )
    errors = verdicts.fresh_coverage_errors(
        coverage(heartbeat_age=700.0, state="UNKNOWN")
    )
    assert "stale" in _text(errors)


def test_fresh_coverage_refuses_a_heartbeat_that_saw_running_work() -> None:
    # The resolver reads only ``watched_pods == watched_attempts == 0`` as an
    # idle statement; a fresh heartbeat that counted work is not coverage.
    errors = verdicts.fresh_coverage_errors(coverage(watched_pods=2))
    assert "saw running work" in _text(errors)
    assert verdicts.heartbeat_saw_work(coverage(watched_pods=2)) is True
    assert verdicts.heartbeat_saw_work(coverage()) is False


def test_fresh_coverage_refuses_a_state_other_than_idle() -> None:
    errors = verdicts.fresh_coverage_errors(coverage(state="UNKNOWN"))
    assert "IDLE" in _text(errors)
    errors = verdicts.fresh_coverage_errors(coverage(state="ACTIVE"))
    assert "IDLE" in _text(errors)


def test_fresh_coverage_refuses_idle_that_a_leftover_job_could_explain() -> None:
    # An attempt observation younger than the freshness window would make the
    # node IDLE with or without the heartbeat; the verdict must not accept it.
    errors = verdicts.fresh_coverage_errors(coverage(observation_age=30.0))
    assert "attempt observation" in _text(errors)
    assert (
        verdicts.fresh_coverage_errors(
            coverage(observation_age=30.0), require_stale_observations=False
        )
        == []
    )


def test_stale_coverage_is_unknown_with_no_fresh_heartbeat_or_observation() -> None:
    assert (
        verdicts.stale_coverage_errors(coverage(heartbeat_age=700.0, state="UNKNOWN"))
        == []
    )
    assert (
        verdicts.stale_coverage_errors(coverage(heartbeat_age=None, state="UNKNOWN"))
        == []
    )


def test_stale_coverage_refuses_a_fresh_heartbeat_or_observation_or_idle() -> None:
    assert "still fresh" in _text(
        verdicts.stale_coverage_errors(coverage(state="UNKNOWN"))
    )
    assert "attempt observation" in _text(
        verdicts.stale_coverage_errors(
            coverage(heartbeat_age=700.0, observation_age=10.0, state="UNKNOWN")
        )
    )
    assert "UNKNOWN" in _text(
        verdicts.stale_coverage_errors(coverage(heartbeat_age=700.0, state="IDLE"))
    )
    assert "control plane" in _text(
        verdicts.stale_coverage_errors(coverage(supported=False, state="UNKNOWN"))
    )


def test_expiry_wait_covers_the_youngest_observation_plus_margin() -> None:
    assert verdicts.expiry_wait_seconds(coverage(observation_age=None)) == 0
    assert verdicts.expiry_wait_seconds(coverage(observation_age=650.0)) == 0
    assert (
        verdicts.expiry_wait_seconds(coverage(observation_age=500.0), margin=30) == 130
    )
    # The heartbeat is only part of the wait when the caller asks for it
    # (DESTR-024 wants it expired; DESTR-023 wants it fresh).
    assert verdicts.expiry_wait_seconds(coverage(heartbeat_age=100.0)) == 0
    assert (
        verdicts.expiry_wait_seconds(
            coverage(heartbeat_age=100.0), include_heartbeat=True, margin=30
        )
        == 530
    )


def test_expiry_wait_refuses_an_unbounded_freshness() -> None:
    value = coverage(observation_age=10.0)
    value["freshness_seconds"] = verdicts.MAX_EXPIRY_WAIT_SECONDS + 1000
    with pytest.raises(ValueError):
        verdicts.expiry_wait_seconds(value)


# --------------------------------------------------------------------------- #
# Idle-cluster and watcher preconditions
# --------------------------------------------------------------------------- #
def test_an_idle_cluster_has_no_managed_pod_and_no_canary() -> None:
    assert verdicts.idle_cluster_errors([], []) == []


def test_managed_pods_or_a_canary_job_break_the_idle_premise() -> None:
    pods = verdicts.managed_pod_summary(
        [
            {
                "metadata": {
                    "name": "train-0",
                    "namespace": "team-a",
                    "labels": {verdicts.MANAGED_LABEL: "true"},
                },
                "spec": {"nodeName": "node-b"},
                "status": {"phase": "Running"},
            }
        ]
    )
    assert pods == [
        {"namespace": "team-a", "name": "train-0", "node": "node-b", "phase": "Running"}
    ]
    assert "managed" in _text(verdicts.idle_cluster_errors(pods, []))
    assert "canary" in _text(
        verdicts.idle_cluster_errors([], [verdicts.COVERAGE_CANARY_JOB])
    )


def test_the_watcher_must_be_a_single_ready_replica() -> None:
    summary = verdicts.deployment_summary(deployment())
    assert summary["replicas"] == 1 and summary["ready_replicas"] == 1
    assert verdicts.watcher_errors(summary) == []
    assert "replicas" in _text(
        verdicts.watcher_errors(
            verdicts.deployment_summary(deployment(replicas=0, ready=0))
        )
    )
    assert "ready" in _text(
        verdicts.watcher_errors(verdicts.deployment_summary(deployment(ready=0)))
    )


# --------------------------------------------------------------------------- #
# Reset verdict
# --------------------------------------------------------------------------- #
def test_the_reset_verdict_accepts_the_destr001_contract() -> None:
    assert verdicts.blocked_by_unknown_errors(reset_state()) == []
    assert verdicts.reset_errors(reset_state()) == []


def test_the_reset_verdict_names_the_unknown_block_first() -> None:
    state = reset_state(status="BLOCKED")
    state["workflow"]["blocked_reasons"] = [verdicts.WORKLOAD_STATE_UNKNOWN_REASON]
    errors = verdicts.reset_errors(state)
    assert errors[0].startswith("workflow was BLOCKED by UNKNOWN workload state"), (
        "the UNKNOWN workload block must lead the errors"
    )
    assert "reset workflow is BLOCKED" in errors
    # DESTR-001's own contract still speaks: nothing completed.
    assert any("not SUCCEEDED" in item for item in errors), (
        "DESTR-001's not-SUCCEEDED contract must still be reported"
    )


def test_a_block_for_another_reason_is_still_a_block() -> None:
    state = reset_state(status="BLOCKED")
    state["workflow"]["blocked_reasons"] = ["repeat fault cooldown"]
    errors = verdicts.blocked_by_unknown_errors(state)
    assert errors == ["reset workflow is BLOCKED"]


# --------------------------------------------------------------------------- #
# Plan identity
# --------------------------------------------------------------------------- #
def test_plan_identity_pins_release_node_agent_profile_and_watcher() -> None:
    preflight = {
        "release_id": "rel-1",
        "node": {"uid": "uid-1"},
        "store": {"agent": {"generation": 4}, "profile": {"profile_version": "v9"}},
        "watcher": {
            "uid": "dep-1",
            "generation": 3,
            "replicas": 1,
            "ready_replicas": 1,
        },
    }
    identity = destr023.plan_identity(preflight)
    assert identity == {
        "release_id": "rel-1",
        "node_uid": "uid-1",
        "agent_generation": 4,
        "runtime_profile_version": "v9",
        "watcher_deployment_uid": "dep-1",
    }


def test_the_reset_contract_reads_the_operations_a_batched_command_carried() -> None:
    """DESTR-023's live reset ran QUIESCE..RESTORE_GPU_SERVICES as one command.

    With protocol-3 batching a node command carries the head step in ``step``
    and the rest in ``batched_steps``; a verdict that read only the head saw
    three of six remote operations and failed a SUCCEEDED workflow.
    """

    from scripts.e2e.regional import run_destr001_gpu_reset as destr001
    from scripts.e2e.regional.remote_command_shapes import command_operations

    def command(head: str, *batched: str) -> dict[str, Any]:
        return {
            "status": "SUCCEEDED",
            "step": {"operation": head},
            "batched_steps": [{"step": {"operation": item}} for item in batched],
        }

    assert command_operations(
        command("QUIESCE_GPU_SERVICES", "VERIFY_NO_GPU_CLIENTS", "RESET_GPU")
    ) == ["QUIESCE_GPU_SERVICES", "VERIFY_NO_GPU_CLIENTS", "RESET_GPU"], (
        "the head step comes first, then the batch in order"
    )
    assert command_operations({"status": "SUCCEEDED"}) == [], "no step, no operations"

    state = reset_state()
    state["commands"] = [
        command("MARK_UNSCHEDULABLE"),
        command(
            "QUIESCE_GPU_SERVICES",
            "VERIFY_NO_GPU_CLIENTS",
            "RESET_GPU",
            "RESTORE_GPU_SERVICES",
        ),
        command("RESTORE_SCHEDULING"),
    ]
    assert destr001.workflow_errors(state) == [], (
        "a batched reset command satisfies the remote-operation contract"
    )
