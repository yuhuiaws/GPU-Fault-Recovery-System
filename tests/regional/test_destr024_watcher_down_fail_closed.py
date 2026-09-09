"""Contract tests for GF-REGIONAL-DESTR-024.

The case proves the other half of the coverage heartbeat: with the completion
watcher scaled to zero and every heartbeat and observation older than the
freshness window, the topology service resolves UNKNOWN and an XID 46 on an
idle node compiles BLOCKED with ``node workload state is UNKNOWN`` -- the
node is contained (cordon + quarantine taint) and never reset. The watcher is
then restored, the heartbeat has to come back, and the isolation is lifted
only through the validation-first restore workflow.

Every verdict is judged against synthetic snapshots; nothing here touches a
cluster.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.e2e.regional import destr023_verdicts as coverage_verdicts
from scripts.e2e.regional import destr024_verdicts as verdicts
from scripts.e2e.regional import run_destr024_watcher_down_fail_closed as destr024
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata

NODE = "node-a"
NOW = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


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
        "freshness_seconds": 600.0,
        "max_age_seconds": 120,
        "heartbeat": heartbeat,
        "heartbeat_age_seconds": heartbeat_age,
        "observation_count": 0 if observation_age is None else 1,
        "latest_observation_age_seconds": observation_age,
        "workload_state": state,
    }


def _execution(operation: str, status: str = "SUCCEEDED") -> dict[str, Any]:
    return {"operation": operation, "status": status, "phase": "safety", "details": {}}


def _command(operation: str, status: str = "SUCCEEDED") -> dict[str, Any]:
    return {"status": status, "step": {"operation": operation}}


def blocked_state() -> dict[str, Any]:
    """The shape DESTR-016 attempt 4 produced live on 2026-09-08."""

    containment = list(verdicts.CONTAINMENT_OPERATIONS)
    return {
        "event": {"xid": 46, "evidence_ref": "kmsg://node-a/1"},
        "decision": {"official_action": "RESET_GPU"},
        "incident": {"incident_id": "inc-destr024", "state": "QUARANTINED"},
        "workflow": {
            "request_id": "wf-destr024",
            "status": "BLOCKED",
            "blocked_kind": "SAFETY_SETTLED",
            "blocked_reasons": [verdicts.WORKLOAD_STATE_UNKNOWN_REASON],
            "official_action": "RESET_GPU",
            "official_steps": [],
            "safety_steps": [{"operation": item} for item in containment],
            "completed_operations": containment,
            "step_executions": [_execution(item) for item in containment],
        },
        "commands": [_command("MARK_UNSCHEDULABLE"), _command("QUARANTINE")],
    }


def host(
    *, ledger_extra: list[dict[str, Any]] | None = None, quiesce: bool = False
) -> dict[str, Any]:
    return {
        "gpu_inventory": [{"pci_bdf": "0000:10:1c.0"}, {"pci_bdf": "0000:10:1d.0"}],
        "compute_clients": [],
        "quiesce_states": ["quiesce-abc.json"] if quiesce else [],
        "ledger": [
            {"command_id": "old-1", "operation": "VALIDATE_GPU", "attempt": 1},
            *(ledger_extra or []),
        ],
        "services": {"nvidia-fabricmanager.service": {"ActiveState": "active"}},
        "gpu_fault_timers": [],
    }


def node(
    *, unschedulable: bool = True, tainted: bool = True, owned: bool = True
) -> dict[str, Any]:
    return {
        "name": NODE,
        "uid": "uid-1",
        "ready": "True",
        "unschedulable": unschedulable,
        "taints": [{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}]
        if tainted
        else [],
        "ownership_annotations": {"gpu-fault.io/owner": "inc"} if owned else {},
    }


# --------------------------------------------------------------------------- #
# Case contract
# --------------------------------------------------------------------------- #
def test_the_case_identity_and_predecessor_are_pinned() -> None:
    metadata = RegionalCaseMetadata(
        case_id=destr024.CASE_ID,
        title="",
        category="regional-destructive-acceptance",
        level="staging",
        risk="live-isolation",
        automation="manual",
        procedure="docs/x.md#gf-regional-destr-024",
        predecessor=destr024.PREDECESSOR_CASE_ID,
    )
    assert destr024.CASE_ID == "GF-REGIONAL-DESTR-024"
    assert destr024.CONFIRMATION == metadata.confirmation == "DESTR024_EXECUTE"
    assert destr024.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-023", (
        "the fail-closed half is only meaningful once the heartbeat-only IDLE "
        "half has passed on the same release"
    )
    assert verdicts.CASE_ID == destr024.CASE_ID
    assert verdicts.WATCHER_DEPLOYMENT == coverage_verdicts.WATCHER_DEPLOYMENT


def test_the_parser_is_plan_only_by_default() -> None:
    arguments = destr024.parser().parse_args(["--run-dir", "/tmp/destr024-test"])
    assert arguments.execute is False
    assert hasattr(arguments, "predecessor_evidence"), (
        "the shared predecessor-evidence flag must be wired"
    )
    assert hasattr(arguments, "host_probe_image"), (
        "the shared host-probe-image flag must be wired"
    )


# --------------------------------------------------------------------------- #
# The BLOCKED workflow
# --------------------------------------------------------------------------- #
def test_the_live_blocked_shape_passes() -> None:
    state = blocked_state()
    assert verdicts.blocked_workflow_errors(state) == []
    assert verdicts.containment_errors(state) == []


def test_a_workflow_that_ran_is_the_failure_the_case_exists_to_catch() -> None:
    state = blocked_state()
    state["workflow"]["status"] = "SUCCEEDED"
    state["workflow"]["blocked_reasons"] = []
    state["workflow"]["completed_operations"] = [
        *verdicts.CONTAINMENT_OPERATIONS,
        "QUIESCE_GPU_SERVICES",
        "RESET_GPU",
    ]
    state["commands"].append(_command("RESET_GPU"))
    errors = verdicts.blocked_workflow_errors(state)
    assert "not BLOCKED" in _text(errors)
    assert verdicts.WORKLOAD_STATE_UNKNOWN_REASON in _text(errors)
    assert "RESET_GPU" in _text(errors)


def test_a_block_for_another_reason_is_not_this_case() -> None:
    state = blocked_state()
    state["workflow"]["blocked_reasons"] = ["repeat fault cooldown"]
    assert verdicts.WORKLOAD_STATE_UNKNOWN_REASON in _text(
        verdicts.blocked_workflow_errors(state)
    )


def test_the_block_must_settle_through_the_safety_steps() -> None:
    state = blocked_state()
    state["workflow"]["blocked_kind"] = "NEEDS_OPERATOR"
    assert "SAFETY_SETTLED" in _text(verdicts.blocked_workflow_errors(state))


def test_a_physical_remote_command_or_execution_fails_even_when_blocked() -> None:
    state = blocked_state()
    state["commands"].append(_command("QUIESCE_GPU_SERVICES", status="WAITING"))
    assert "QUIESCE_GPU_SERVICES" in _text(verdicts.blocked_workflow_errors(state))
    state = blocked_state()
    state["workflow"]["step_executions"].append(_execution("RESET_GPU", "RUNNING"))
    assert "RESET_GPU" in _text(verdicts.blocked_workflow_errors(state))


def test_the_event_must_be_the_injected_kmsg_xid_46_decided_as_reset() -> None:
    state = blocked_state()
    state["event"]["xid"] = 79
    assert "XID 46" in _text(verdicts.blocked_workflow_errors(state))
    state = blocked_state()
    state["event"]["evidence_ref"] = "api://x"
    assert "kmsg" in _text(verdicts.blocked_workflow_errors(state))
    state = blocked_state()
    state["decision"]["official_action"] = "RESTART_NODE"
    assert "RESET_GPU" in _text(verdicts.blocked_workflow_errors(state))


def test_containment_must_have_cordoned_the_node() -> None:
    state = blocked_state()
    state["workflow"]["completed_operations"] = ["FREEZE_EVIDENCE"]
    assert "MARK_UNSCHEDULABLE" in _text(verdicts.containment_errors(state))


def test_the_node_must_be_isolated_after_the_block() -> None:
    assert verdicts.node_isolated_errors(node()) == []
    assert verdicts.node_isolated_errors(node(unschedulable=False, owned=False)) == []
    assert "not isolated" in _text(
        verdicts.node_isolated_errors(
            node(unschedulable=False, tainted=False, owned=False)
        )
    )


# --------------------------------------------------------------------------- #
# The host was never touched
# --------------------------------------------------------------------------- #
def test_an_untouched_host_passes() -> None:
    assert verdicts.host_untouched_errors(host(), host()) == []


def test_a_reset_or_quiesce_row_or_state_on_the_host_fails() -> None:
    after = host(
        ledger_extra=[{"command_id": "new-1", "operation": "RESET_GPU", "attempt": 1}]
    )
    assert "RESET_GPU" in _text(verdicts.host_untouched_errors(host(), after))
    after = host(
        ledger_extra=[
            {"command_id": "new-2", "operation": "QUIESCE_GPU_SERVICES", "attempt": 1}
        ]
    )
    assert "QUIESCE_GPU_SERVICES" in _text(
        verdicts.host_untouched_errors(host(), after)
    )
    assert "quiesce state" in _text(
        verdicts.host_untouched_errors(host(), host(quiesce=True))
    )


def test_a_changed_inventory_or_stopped_service_fails() -> None:
    after = host()
    after["gpu_inventory"] = after["gpu_inventory"][:1]
    assert "inventory" in _text(verdicts.host_untouched_errors(host(), after))
    after = host()
    after["services"]["nvidia-fabricmanager.service"] = {"ActiveState": "inactive"}
    assert "nvidia-fabricmanager.service" in _text(
        verdicts.host_untouched_errors(host(), after)
    )


# --------------------------------------------------------------------------- #
# Watcher scale, watchdog, heartbeat recovery
# --------------------------------------------------------------------------- #
def test_the_watcher_must_be_gone_before_the_wait_starts() -> None:
    absent = {"replicas": 0, "ready_replicas": 0}
    assert verdicts.watcher_absent_errors(absent, []) == []
    assert "replicas" in _text(
        verdicts.watcher_absent_errors({"replicas": 1, "ready_replicas": 0}, [])
    )
    assert "Pod" in _text(verdicts.watcher_absent_errors(absent, [{"name": "w-1"}]))


def test_the_watchdog_script_sleeps_then_restores_the_baseline_replicas() -> None:
    script = verdicts.watchdog_script(
        ["kubectl", "--kubeconfig", "/k", "--context", "c", "-n", "ns"],
        replicas=1,
        delay_seconds=900,
    )
    assert script.startswith("sleep 900; "), (
        "the watchdog must delay before restoring replicas"
    )
    assert "scale deployment/gpu-fault-completion-watcher --replicas=1" in script


def test_the_watchdog_delay_outlives_every_phase_it_guards() -> None:
    delay = verdicts.watchdog_delay_seconds(expiry_wait_seconds=630)
    assert delay == (
        630
        + verdicts.STALE_SETTLE_BUDGET_SECONDS
        + verdicts.WORKFLOW_BUDGET_SECONDS
        + verdicts.WATCHDOG_MARGIN_SECONDS
    )


def test_heartbeat_recovery_needs_a_newer_fresh_heartbeat_and_idle() -> None:
    stale = coverage(heartbeat_age=700.0, state="UNKNOWN")
    recovered = coverage(heartbeat_age=5.0)
    assert verdicts.heartbeat_recovered_errors(stale, recovered) == []
    # A leftover observation stream is not the watcher coming back.
    assert (
        verdicts.heartbeat_recovered_errors(
            stale, coverage(heartbeat_age=700.0, observation_age=10.0)
        )
        != []
    )
    same = coverage(heartbeat_age=700.0, state="UNKNOWN")
    assert "did not advance" in _text(verdicts.heartbeat_recovered_errors(stale, same))
    still_unknown = coverage(heartbeat_age=5.0, state="UNKNOWN")
    assert "IDLE" in _text(verdicts.heartbeat_recovered_errors(stale, still_unknown))


def test_recovery_from_no_heartbeat_at_all_only_needs_the_fresh_one() -> None:
    assert (
        verdicts.heartbeat_recovered_errors(
            coverage(heartbeat_age=None, state="UNKNOWN"), coverage(heartbeat_age=5.0)
        )
        == []
    )


# --------------------------------------------------------------------------- #
# Validated restore
# --------------------------------------------------------------------------- #
def test_the_restore_must_succeed_and_release_the_node() -> None:
    clean = node(unschedulable=False, tainted=False, owned=False)
    assert verdicts.restore_errors({"status": "SUCCEEDED"}, clean) == []
    assert "restore workflow" in _text(
        verdicts.restore_errors({"status": "FAILED"}, clean)
    )
    assert "unschedulable" in _text(
        verdicts.restore_errors(
            {"status": "SUCCEEDED"}, node(tainted=False, owned=False)
        )
    )
    assert "taint" in _text(
        verdicts.restore_errors(
            {"status": "SUCCEEDED"}, node(unschedulable=False, owned=False)
        )
    )
    assert "ownership" in _text(
        verdicts.restore_errors(
            {"status": "SUCCEEDED"}, node(unschedulable=False, tainted=False)
        )
    )
    not_ready = dict(clean, ready="False")
    assert "Ready" in _text(verdicts.restore_errors({"status": "SUCCEEDED"}, not_ready))
