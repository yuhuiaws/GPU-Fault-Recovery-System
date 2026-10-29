"""DESTR-019 on a graded Hopper NVSwitch node: the audited command is the reset.

The case restarts the Node Agent and then audits the next real command's trail.
Its injection is DESTR-010's solo XID 45, and on a node whose NVLink fabric the
control plane grades (batch 128) that is no longer one Fabric Manager restart
but the full GPU + NVSwitch reset chain. The audit then reads the chain's
RESET_ALL_GPUS_NVSWITCHES -- one ledger row at attempt 1, three journald lines
with the nine identifier fields -- and the health counters must move exactly as
the Agent's ledger did since the restart. A node the product does not grade
keeps the one-command contract.

The window may meet one other command, the control plane's observation refresh
(``observation_refresh``): its one ledger row and every ``rejected`` the
restarted Agent logged for it are accepted in exactly the product's shape, and
nothing else is. On HyperPod the quiesce stops kubelet, so the Node's Ready
condition may move once, into the restore that started kubelet again.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.app import notify_silent_collectors
from gpu_fault.fleet import AgentRecord
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.config import protected_quiesce_services
from gpu_fault.regional import RegionalClusterRegistration
from scripts.e2e.regional import destr019_verdicts as verdicts
from scripts.e2e.regional import fabric_restart_shape as shapes
from scripts.e2e.regional import observation_refresh as refresh
from scripts.e2e.regional import run_destr019_agent_restart_ledger as destr019
from scripts.e2e.regional import xid45_opt_in
from scripts.e2e.regional.probes import destr019_node_probe as probe
from tests.execution._health_policy_e2e_support import HealthRig
from tests.node_agent._support import (
    NodeActionStatus,
    ServiceRunner,
    command,
    envelope,
    no_device_clients,
    node_action_executor,
    quiesce_manager,
)
from tests.regional._cov95_destr_agent import (
    GENERATION,
    INCIDENT,
    NODE,
    WORKFLOW,
    AgentHarness,
    audit,
    health,
    journal,
)
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW

GPUS = [f"GPU-{index}" for index in range(8)]
GRADED = {
    "available": True,
    "graded_gpus": GPUS,
    "states": dict.fromkeys(GPUS, 3.0),
    "product": "NVIDIA H200",
}
RESET_KEY = f"{WORKFLOW}/4/RESET_ALL_GPUS_NVSWITCHES"
RESET_COMMAND_ID = f"{RESET_KEY}/{NODE}/agent-{GENERATION}"
OWNERS = {
    "FREEZE_EVIDENCE": "gpu-fault-control-plane",
    "MARK_UNSCHEDULABLE": "gpu-fault-kubernetes-adapter",
    "RESTORE_SCHEDULING": "gpu-fault-kubernetes-adapter",
    "VALIDATE_GPU": "gpu-fault-validation-adapter",
    "VALIDATE_FABRIC": "gpu-fault-validation-adapter",
}


def _chain_state() -> dict[str, Any]:
    steps = []
    for operation in shapes.FABRIC_RESET_STEPS:
        step: dict[str, Any] = {
            "operation": operation,
            "execution_owner": OWNERS.get(operation, "gpu-fault-node-agent"),
            "gpu_uuids": GPUS,
            "parameters": {},
        }
        if operation == "RESET_ALL_GPUS_NVSWITCHES":
            step["parameters"] = {"fabric_partition": "c/n/local-nvswitch", "xid": 45}
        steps.append(step)
    commands = [
        {
            "status": "SUCCEEDED",
            "idempotency_key": f"{WORKFLOW}/{index}/{operation}",
            "step": {"operation": operation},
            "batched_steps": [],
        }
        for index, operation in (
            (1, "MARK_UNSCHEDULABLE"),
            (2, "QUIESCE_GPU_SERVICES"),
            (3, "VERIFY_NO_GPU_CLIENTS"),
            (4, "RESET_ALL_GPUS_NVSWITCHES"),
            (5, "RESTORE_GPU_SERVICES"),
            (8, "RESTORE_SCHEDULING"),
        )
    ]
    return {
        "event": {"xid": 45},
        "decision": {"official_action": "RESTART_FM", "disposition": "EXECUTABLE"},
        "workflow": {
            "request_id": WORKFLOW,
            "status": "SUCCEEDED",
            "fencing_token": 2,
            "official_steps": steps,
            "completed_operations": list(shapes.FABRIC_RESET_STEPS),
        },
        "incident": {"incident_id": INCIDENT, "node_ids": [NODE]},
        "commands": commands,
    }


def _row(operation: str, state: str = "SUCCEEDED", attempt: int = 1) -> dict[str, Any]:
    return {
        "command_id": f"{operation}-id",
        "operation": operation,
        "state": state,
        "attempt": attempt,
        "incident_id": INCIDENT,
        "workflow_request_id": WORKFLOW,
    }


CHAIN_ROWS = [
    _row("QUIESCE_GPU_SERVICES"),
    _row("VERIFY_NO_GPU_CLIENTS", "FAILED"),
    _row("VERIFY_NO_GPU_CLIENTS"),
    _row("RESET_ALL_GPUS_NVSWITCHES"),
    _row("RESTORE_GPU_SERVICES"),
]


def test_a_graded_node_needs_the_chain_s_node_actions() -> None:
    errors = verdicts.preflight_errors(
        node={"ready": "True"},
        agent={
            "lifecycle_state": "ACTIVE",
            "generation": GENERATION,
            "allowed_operations": ["RESTART_FABRIC_MANAGER"],
        },
        profile={},
        workloads=[],
        queue={"depth": 0},
        remote_commands={},
        recent_event=None,
        host={
            "agent": {"ActiveState": "active"},
            "ledger": {"present": True, "user_version": verdicts.LEDGER_SCHEMA_VERSION},
        },
        tests_passed=True,
        shape=shapes.FABRIC_RESET,
    )

    assert sorted(errors) == sorted(
        f"target Node Agent does not allow {item}"
        for item in shapes.FABRIC_RESET_NODE_STEPS
    )


def test_the_chain_passes_and_the_audited_command_is_its_reset() -> None:
    state = _chain_state()

    assert (
        verdicts.workflow_errors(
            state, node=NODE, shape=shapes.FABRIC_RESET, expected_gpus=GPUS
        )
        == []
    )
    command = verdicts.audited_command(state, shapes.FABRIC_RESET)
    assert (
        verdicts.expected_command_id(command, node=NODE, generation=GENERATION)
        == RESET_COMMAND_ID
    )
    assert (
        verdicts.AUDITED_OPERATION[shapes.FABRIC_RESET] == "RESET_ALL_GPUS_NVSWITCHES"
    )


def test_the_two_step_restart_or_a_widened_incident_fails_on_a_graded_node() -> None:
    state = _chain_state()
    state["workflow"]["official_steps"] = state["workflow"]["official_steps"][:1]
    widened = _chain_state()
    widened["incident"]["node_ids"] = [NODE, "node-b"]

    assert verdicts.workflow_errors(
        state, node=NODE, shape=shapes.FABRIC_RESET, expected_gpus=GPUS
    ), "the two-step restart is no pass on a graded node"
    errors = verdicts.workflow_errors(
        widened, node=NODE, shape=shapes.FABRIC_RESET, expected_gpus=GPUS
    )
    assert any("not just node-a" in item for item in errors), errors


def test_the_counters_follow_the_ledger_on_a_graded_node() -> None:
    assert verdicts.counter_expectation(shapes.FABRIC_RESET, {"rows": CHAIN_ROWS}) == {
        "accepted": 5,
        "completed": 4,
        "failed": 1,
        "rejected": 0,
    }
    assert verdicts.counter_expectation(shapes.FABRIC_MANAGER_RESTART) == {
        "accepted": 1,
        "completed": 1,
        "failed": 0,
        "rejected": 0,
    }
    assert verdicts.chain_ledger_errors({"rows": CHAIN_ROWS}) == []


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([*CHAIN_ROWS, _row("RESTART_FABRIC_MANAGER")], "outside the chain"),
        ([*CHAIN_ROWS, _row("RESET_ALL_GPUS_NVSWITCHES", attempt=2)], "2 RESET_ALL"),
        (CHAIN_ROWS[:1] + CHAIN_ROWS[3:], "no client verification"),
    ],
    ids=["foreign-operation", "second-reset-attempt", "no-verification"],
)
def test_a_chain_that_is_not_one_window_fails(rows, message) -> None:
    errors = verdicts.chain_ledger_errors({"rows": rows})

    assert any(message in item for item in errors), errors


def _graded_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentHarness:
    h = AgentHarness(tmp_path, monkeypatch)
    h.grading = deepcopy(GRADED)
    h.agent["allowed_operations"] = list(shapes.FABRIC_RESET_NODE_STEPS)
    h.state = _chain_state()
    lines = journal()
    for line in lines["lines"]:
        line["fields"].update(
            command_id=RESET_COMMAND_ID,
            operation="RESET_ALL_GPUS_NVSWITCHES",
            gpu_count="8",
        )
    h.journal = lines
    reset_audit = audit()
    reset_audit["rows"][0].update(
        command_id=RESET_COMMAND_ID,
        operation="RESET_ALL_GPUS_NVSWITCHES",
        gpu_uuid_count=8,
    )
    h.audit = reset_audit
    h.chain_audit = {
        "rows": [
            {**row, "started_at": (NOW + timedelta(seconds=index)).isoformat()}
            for index, row in enumerate(CHAIN_ROWS)
        ]
    }
    h.health_after = health({"accepted": 5, "completed": 4, "failed": 1, "rejected": 0})
    return h


def test_the_runner_audits_the_reset_of_a_graded_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _graded_harness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    details = destr019.plan_details(h.settings, preflight)
    assert details["risk"] == "destructive", details["risk"]
    assert details["workflow_shape"] == shapes.FABRIC_RESET
    assert "full GPU + NVSwitch reset chain" in details["mutation"]

    code, report = h.execute(tmp_path)

    assert code == 0 and report["errors"] == [], report
    audits = [detail for name, detail in h.calls if name == "agent.ledger-audit"]
    assert len(audits) == 2, audits
    assert audits[0]["args"] == ("--command-id", RESET_COMMAND_ID), audits
    assert audits[1]["args"][0] == "--since", "the chain audit reads every attempt"


def test_counters_that_disagree_with_the_ledger_fail_the_graded_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _graded_harness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.health_after = health({"accepted": 1, "completed": 1, "failed": 0, "rejected": 0})

    code, report = h.execute(tmp_path)

    assert code == 1, report
    assert "counters" in " ".join(report["errors"]), report["errors"]


# --- batch 131: the chain under the XID 45 opt-in window -------------------------


def _opt_in_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentHarness:
    """A graded node whose control plane makes a solo XID 45 observe-only."""

    h = _graded_harness(tmp_path, monkeypatch)
    h.grading["observe_only_default"] = True
    h.xid45_switch = {"opted_in": False, "replicas": []}
    after = ready_runtime()
    after["deployments"]["cpu"]["gpu-fault-control-worker"].update(
        generation=3, observed_generation=3
    )
    h.identity_after = after
    return h


def test_the_audit_runs_the_chain_inside_the_opt_in_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _opt_in_harness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    details = destr019.plan_details(h.settings, preflight)
    assert details["workflow_shape"] == shapes.FABRIC_RESET
    assert details["xid45_opt_in_window"] is True
    assert xid45_opt_in.SWITCH in details["mutation"], details["mutation"]
    assert details["rollback"]["cleanup_closes_the_xid45_opt_in_window"] is True

    code, report = h.execute(tmp_path)

    assert code == 0 and report["errors"] == [], report
    names = [name for name, _detail in h.calls]
    assert names.index("xid45.open") < names.index("agent.restart-agent"), (
        "the window rolls the control-worker before the quiet check and restart"
    )
    assert names.index("injector.write-xid45") > names.index("xid45.open")
    assert h.windows == ["open", "close"], "closed again in the cleanup"
    assert "runtime_identity" in report["cleanup"], report["cleanup"]


def test_a_window_that_was_not_closed_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _opt_in_harness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    monkeypatch.setattr(
        destr019.xid45_opt_in,
        "close_window",
        lambda _regional, _case_dir: {"state": "OPEN"},
    )
    h.identity_after = ready_runtime()

    code, report = h.execute(tmp_path)

    assert code == 1, report
    errors = " ".join(report["cleanup"]["errors"])
    assert "opt-in window was not closed again" in errors, errors
    assert "runtime_identity" in errors, "and the worker's rolls do not add up"


def test_a_site_that_opted_in_runs_the_chain_without_a_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _opt_in_harness(tmp_path, monkeypatch)
    h.xid45_switch = {"opted_in": True, "replicas": []}
    h.identity_after = None
    preflight = h.plan(tmp_path)

    assert destr019.plan_details(h.settings, preflight)["xid45_opt_in_window"] is False
    code, report = h.execute(tmp_path)

    assert code == 0, report
    assert h.windows == [], "no window, the strict runtime identity check"


def test_a_half_rolled_switch_refuses_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _opt_in_harness(tmp_path, monkeypatch)
    h.xid45_switch = {"opted_in": None, "replicas": []}

    preflight = h.plan(tmp_path)

    assert any(xid45_opt_in.SWITCH in item for item in preflight["errors"]), preflight


# --- R13i: the control plane's observation refresh beside the chain -----------
#
# Live (HyperPod, release 007595f77e11): the chain's QUIESCE stopped the
# collectors, the control plane's silence scan asked for an observation refresh
# of the node, the Agent refused its TRIGGER_HEALTH_SNAPSHOT 41 times while the
# window was open and ran it once during the restore's settle; the counters read
# accepted=5 completed=5 failed=0 rejected=41. The quiesce also stopped kubelet,
# so the Node's Ready condition flipped and was back from the restore's kubelet
# start.

# The live run on anonymised identities (the public release names no site):
# its timestamps and counts, and the ids the product derives for its refresh.
LIVE_CLUSTER = "hp-cluster-a"
LIVE_NODE = "hyperpod-node-a"
LIVE_GENERATION = 41
LIVE_WORKFLOW = "workflow-00000000-0000-4000-8000-000000000045"
LIVE_INCIDENT = "inc-kernel-log-kmsg-boot-a-3749-xid-45-cluster-a"
LIVE_CASE_START = datetime(2026, 10, 29, 0, 26, 7, tzinfo=timezone.utc)
LIVE_AUDIT_AT = datetime(2026, 10, 29, 0, 37, 8, tzinfo=timezone.utc)
[LIVE_REQUEST] = refresh.requests(
    cluster_id=LIVE_CLUSTER,
    node=LIVE_NODE,
    generation=LIVE_GENERATION,
    start=LIVE_CASE_START,
    end=LIVE_AUDIT_AT,
).values()
LIVE_REFRESH_WORKFLOW = LIVE_REQUEST.workflow_request_id
LIVE_REFRESH_INCIDENT = LIVE_REQUEST.incident_id
LIVE_REFRESH = LIVE_REQUEST.command_id
LIVE_INVOCATION = "invocation-after-restart"


def _at(clock: str) -> datetime:
    return datetime.fromisoformat(f"2026-10-29T{clock}+00:00")


def _live_row(
    operation: str, step: str, started: str, completed: str, **extra: Any
) -> dict[str, Any]:
    return {
        "agent_generation": LIVE_GENERATION,
        "attempt": 1,
        "command_id": f"{LIVE_WORKFLOW}/{step}/{LIVE_NODE}/agent-41",
        "completed_at": _at(completed).isoformat(),
        "exit_code": None,
        "fencing_token": 1,
        "gpu_uuid_count": 8,
        "gpu_uuids_present": True,
        "incident_id": LIVE_INCIDENT,
        "operation": operation,
        "started_at": _at(started).isoformat(),
        "state": "SUCCEEDED",
        "workflow_request_id": LIVE_WORKFLOW,
        **extra,
    }


def _live_refresh_row(**overrides: Any) -> dict[str, Any]:
    return {
        "agent_generation": LIVE_GENERATION,
        "attempt": 1,
        "command_id": LIVE_REFRESH,
        "completed_at": _at("00:35:32.503808").isoformat(),
        "exit_code": None,
        "fencing_token": 1,
        "gpu_uuid_count": 0,
        "gpu_uuids_present": True,
        "incident_id": LIVE_REFRESH_INCIDENT,
        "operation": "TRIGGER_HEALTH_SNAPSHOT",
        "started_at": _at("00:35:27.289895").isoformat(),
        "state": "SUCCEEDED",
        "workflow_request_id": LIVE_REFRESH_WORKFLOW,
        **overrides,
    }


def _live_chain_rows() -> list[dict[str, Any]]:
    """``ledger-audit-chain.json`` of the live run, the refresh row aside."""

    return [
        _live_row(
            "QUIESCE_GPU_SERVICES",
            "2/QUIESCE_GPU_SERVICES",
            "00:31:59.598621",
            "00:32:28.470590",
        ),
        _live_row(
            "VERIFY_NO_GPU_CLIENTS",
            "3/VERIFY_NO_GPU_CLIENTS",
            "00:32:28.549032",
            "00:32:47.327325",
            command_id=(
                f"{LIVE_WORKFLOW}/3/VERIFY_NO_GPU_CLIENTS/{LIVE_NODE}/attempt-1/agent-41"
            ),
        ),
        _live_row(
            "RESET_ALL_GPUS_NVSWITCHES",
            "4/RESET_ALL_GPUS_NVSWITCHES",
            "00:32:49.603730",
            "00:35:12.001240",
        ),
        _live_row(
            "RESTORE_GPU_SERVICES",
            "5/RESTORE_GPU_SERVICES",
            "00:35:14.804833",
            "00:35:56.452974",
        ),
    ]


def _rejected_line(
    logged: datetime,
    *,
    invocation: str = LIVE_INVOCATION,
    reason: str | None = None,
    **fields: str,
) -> dict[str, Any]:
    """One journald line the way the probe parses the Agent's refusal."""

    values = {
        "command_id": LIVE_REFRESH,
        "incident_id": LIVE_REFRESH_INCIDENT,
        "workflow_request_id": LIVE_REFRESH_WORKFLOW,
        "operation": "TRIGGER_HEALTH_SNAPSHOT",
        "node_id": LIVE_NODE,
        "fencing_token": "1",
        "agent_generation": str(LIVE_GENERATION),
        "gpu_count": "0",
        "attempt": "0",
        "reason": refresh.window_refusal(LIVE_INCIDENT) if reason is None else reason,
        **fields,
    }
    return {
        "phase": "rejected",
        "fields": {name: value for name, value in values.items() if value != "-"},
        "realtime_timestamp": str(int(logged.timestamp() * 1_000_000)),
        "invocation_id": invocation,
    }


def _live_refusals(count: int = 41) -> list[dict[str, Any]]:
    """The window's refusals, one every five seconds from just after QUIESCE."""

    first = _at("00:32:02.600000")
    return [
        _rejected_line(first + timedelta(seconds=5 * index)) for index in range(count)
    ]


def _judge(
    rows: list[dict[str, Any]], lines: list[dict[str, Any]], counters: dict[str, int]
) -> list[str]:
    """The runner's audit of the chain, its refresh and the counters."""

    chain_audit = {"rows": rows}
    attempts = verdicts.refresh_attempts(
        chain_audit,
        cluster_id=LIVE_CLUSTER,
        node=LIVE_NODE,
        generation=LIVE_GENERATION,
        start=LIVE_CASE_START,
        end=LIVE_AUDIT_AT,
    )
    window = verdicts.chain_window(chain_audit, workflow_request_id=LIVE_WORKFLOW)
    errors = verdicts.chain_ledger_errors(
        chain_audit, workflow_request_id=LIVE_WORKFLOW, foreign=list(attempts)
    )
    errors += verdicts.refresh_errors(
        attempts, generation=LIVE_GENERATION, window=window
    )
    refused, refusal_errors = verdicts.refusal_count(
        lines,
        attempts,
        invocation_id=LIVE_INVOCATION,
        chain_incident=LIVE_INCIDENT,
        window=window,
        node=LIVE_NODE,
        generation=LIVE_GENERATION,
    )
    errors += refusal_errors
    errors += verdicts.health_errors(
        health(counters),
        label="after command",
        expect_counters=verdicts.counter_expectation(
            shapes.FABRIC_RESET, chain_audit, refused=refused
        ),
    )
    return errors


def test_the_refresh_is_one_request_per_cluster_node_generation_and_hour() -> None:
    requests = refresh.requests(
        cluster_id=LIVE_CLUSTER,
        node=LIVE_NODE,
        generation=LIVE_GENERATION,
        start=LIVE_CASE_START,
        end=LIVE_AUDIT_AT,
    )

    assert list(requests) == [LIVE_REFRESH], "the case ran inside one hour"
    assert re.fullmatch(
        r"inc-observation-refresh-[0-9a-f]{24}", LIVE_REFRESH_INCIDENT
    ), LIVE_REFRESH_INCIDENT
    assert re.fullmatch(r"workflow-[0-9a-f]{24}", LIVE_REFRESH_WORKFLOW), (
        LIVE_REFRESH_WORKFLOW
    )
    assert LIVE_REFRESH == (
        f"{LIVE_REFRESH_WORKFLOW}/1/TRIGGER_HEALTH_SNAPSHOT/{LIVE_NODE}/agent-41"
    )
    for other in (
        {"cluster_id": "eks-cluster-a", "node": LIVE_NODE, "generation": 41},
        {"cluster_id": LIVE_CLUSTER, "node": "hyperpod-node-b", "generation": 41},
        {"cluster_id": LIVE_CLUSTER, "node": LIVE_NODE, "generation": 42},
    ):
        found = refresh.requests(**other, start=LIVE_CASE_START, end=LIVE_AUDIT_AT)
        assert LIVE_REFRESH not in found, other
    next_hour = refresh.requests(
        cluster_id=LIVE_CLUSTER,
        node=LIVE_NODE,
        generation=LIVE_GENERATION,
        start=LIVE_CASE_START + timedelta(hours=1),
        end=LIVE_AUDIT_AT + timedelta(hours=1),
    )
    assert LIVE_REFRESH not in next_hour, "another hour is another request"
    spanning = refresh.requests(
        cluster_id=LIVE_CLUSTER,
        node=LIVE_NODE,
        generation=LIVE_GENERATION,
        start=LIVE_CASE_START,
        end=LIVE_AUDIT_AT + timedelta(hours=1),
    )
    assert list(spanning) == [LIVE_REFRESH, *next_hour], "a case across the hour"


def test_the_refresh_ids_are_the_ones_the_silence_scan_mints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = HealthRig(tmp_path, monkeypatch)
    try:
        rig.store.save_regional_cluster(
            RegionalClusterRegistration(
                cluster_id="cluster-a",
                region="us-west-2",
                hyperpod_cluster_name="hp-a",
                eks_cluster_arn="arn:aws:eks:us-west-2:1:cluster/a",
                token_sha256="a" * 64,
                agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
            )
        )
        rig.store.save_agent(
            AgentRecord(
                cluster_id="cluster-a",
                node_id="node-a",
                endpoint="https://127.0.0.1",
                agent_version="test",
                artifact_sha256="a" * 64,
                policy_version="test",
                runtime_profile_version="simulated-v1",
                config_digest="b" * 64,
                allowed_operations=[WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT],
                first_seen_at=rig.clock,
                last_seen_at=rig.clock,
                lease_expires_at=rig.clock + timedelta(hours=1),
            )
        )
        monkeypatch.setattr(
            rig.context.advisory_notifications, "send", lambda _notification: None
        )
        notify_silent_collectors(
            rig.context,
            observed_at=rig.clock,
            silent_after_seconds=420,
            alert_interval_seconds=3600,
        )
        [workflow] = rig.store.list_workflows()
    finally:
        rig.store.close()
    [request] = refresh.requests(
        cluster_id="cluster-a",
        node="node-a",
        generation=7,
        start=rig.clock,
        end=rig.clock,
    ).values()

    assert workflow.request_id == request.workflow_request_id, workflow.request_id
    assert workflow.incident_id == request.incident_id, workflow.incident_id
    operations = [step.operation.value for step in workflow.official_steps]
    assert operations[refresh.STEP_INDEX] == refresh.OPERATION, operations
    assert request.command_id == verdicts.expected_command_id(
        {"idempotency_key": f"{workflow.request_id}/1/TRIGGER_HEALTH_SNAPSHOT"},
        node="node-a",
        generation=7,
    ), "the node action the transport names for that step"


def test_the_agent_s_window_refusal_is_one_rejected_and_the_line_the_audit_reads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runner = ServiceRunner(active={"nvidia-fabricmanager", "nvidia-dcgm", "kubelet"})
    operations = {
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
    }
    agent = node_action_executor(
        tmp_path,
        "refusal.db",
        allowed_operations=operations,
        reset_enabled=True,
        service_quiesce_enabled=True,
        quiesce_manager=quiesce_manager(tmp_path, runner),
        runner=runner,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        sleep=lambda _: None,
    )
    agent.set_agent_generation(7)
    quiesced = agent.execute(
        envelope(
            command(
                WorkflowOperation.QUIESCE_GPU_SERVICES, command_id="wf/quiesce/node-a"
            )
        )
    )
    assert quiesced.status is NodeActionStatus.SUCCEEDED, quiesced.error
    opened = datetime.now(timezone.utc)
    [request] = refresh.requests(
        cluster_id="cluster-a", node="node-a", generation=7, start=opened, end=opened
    ).values()
    snapshot = command(
        WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
        command_id=request.command_id,
        gpu_uuids=[],
    ).model_copy(
        update={
            "incident_id": request.incident_id,
            "workflow_request_id": request.workflow_request_id,
            "agent_generation": 7,
        }
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="GPU maintenance window is open"):
            agent.validate_submission(envelope(snapshot))
    [message] = [
        record.getMessage()
        for record in caplog.records
        if "node action rejected" in record.getMessage()
    ]
    parsed = probe.parse_log_line(f"2026-10-29 00:32:02,600 WARNING agent {message}")
    assert parsed is not None, message
    logged = opened + timedelta(seconds=1)
    line = {
        **parsed,
        "realtime_timestamp": str(int(logged.timestamp() * 1_000_000)),
        "invocation_id": "invocation-2",
    }
    row = {
        "command_id": request.command_id,
        "incident_id": request.incident_id,
        "workflow_request_id": request.workflow_request_id,
        "fencing_token": 1,
        "started_at": (opened + timedelta(minutes=3)).isoformat(),
    }

    assert agent.counters_snapshot()["rejected"] == 1, agent.counters_snapshot()
    assert parsed["fields"]["reason"] == refresh.window_refusal("incident-a"), parsed
    assert refresh.refusal_errors(
        [line],
        {request.command_id: row},
        invocation_id="invocation-2",
        window_incident="incident-a",
        window_opened=opened,
        node="node-a",
        generation=7,
    ) == (1, []), "the Agent's own refusal line is the one the audit accepts"


def test_a_hyperpod_quiesce_stops_kubelet_and_plain_eks_keeps_it() -> None:
    assert protected_quiesce_services({"GPU_FAULT_CLUSTER_PROVIDER": "hyperpod"}) == ()
    assert "kubelet" in protected_quiesce_services(
        {"GPU_FAULT_CLUSTER_PROVIDER": "eks-managed"}
    ), "plain EKS never stops kubelet: its Ready condition does not move"


def test_the_live_chain_with_the_refresh_and_its_refusals_passes() -> None:
    rows = [*_live_chain_rows(), _live_refresh_row()]
    counters = {"accepted": 5, "completed": 5, "failed": 0, "rejected": 41}

    assert _judge(rows, _live_refusals(), counters) == []


def test_a_chain_without_a_refresh_still_counts_no_rejection() -> None:
    counters = {"accepted": 4, "completed": 4, "failed": 0, "rejected": 0}

    assert _judge(_live_chain_rows(), [], counters) == [], "the round-12 shape"
    errors = _judge(_live_chain_rows(), [], {**counters, "rejected": 41})
    assert any("counters" in item for item in errors), (
        "a rejection the journal does not show is still a failure"
    )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            _live_refresh_row(
                command_id=(
                    "workflow-0123456789abcdef01234567/1/TRIGGER_HEALTH_SNAPSHOT/"
                    f"{LIVE_NODE}/agent-41"
                )
            ),
            "outside the chain",
        ),
        (_live_refresh_row(state="FAILED"), "state='FAILED'"),
        (_live_refresh_row(attempt=2), "attempt=2"),
        (_live_refresh_row(gpu_uuid_count=8), "gpu_uuid_count=8"),
        (_live_refresh_row(incident_id=LIVE_INCIDENT), "incident_id="),
        (_live_refresh_row(agent_generation=40), "agent generation 40"),
        (
            _live_refresh_row(started_at=_at("00:33:00").isoformat()),
            "inside the chain's quiesce window",
        ),
        (
            _live_row(
                "RUN_DCGM_DIAGNOSTIC",
                "1/RUN_DCGM_DIAGNOSTIC",
                "00:35:40",
                "00:35:50",
                workflow_request_id="workflow-other",
            ),
            "outside the chain",
        ),
        (
            _live_row(
                "VERIFY_NO_GPU_CLIENTS",
                "3/VERIFY_NO_GPU_CLIENTS/again",
                "00:35:40",
                "00:35:50",
                workflow_request_id="workflow-other",
            ),
            "another workflow",
        ),
    ],
    ids=[
        "snapshot-of-another-request",
        "failed-refresh",
        "second-refresh-attempt",
        "refresh-with-gpus",
        "refresh-of-the-chain-incident",
        "refresh-of-another-generation",
        "refresh-inside-the-window",
        "foreign-diagnostic",
        "chain-step-of-another-workflow",
    ],
)
def test_foreign_work_that_is_not_the_refresh_still_fails(
    row: dict[str, Any], message: str
) -> None:
    errors = _judge(
        [*_live_chain_rows(), row],
        [],
        {"accepted": 5, "completed": 5, "failed": 0, "rejected": 0},
    )

    assert any(message in item for item in errors), errors


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            _rejected_line(
                _at("00:33:00"),
                command_id=f"{LIVE_WORKFLOW}/4/RESET_ALL_GPUS_NVSWITCHES/{LIVE_NODE}/agent-41",
                operation="RESET_ALL_GPUS_NVSWITCHES",
                reason="node action agent generation is not known yet",
            ),
            "not an observation refresh that ran",
        ),
        (
            _rejected_line(
                _at("00:33:00"),
                command_id=f"{LIVE_REFRESH_WORKFLOW}/1/RUN_DCGM_DIAGNOSTIC/{LIVE_NODE}/agent-41",
                operation="RUN_DCGM_DIAGNOSTIC",
            ),
            "not an observation refresh that ran",
        ),
        (
            _rejected_line(_at("00:33:00"), reason="node action command has expired"),
            "not the chain's window refusal",
        ),
        (
            _rejected_line(
                _at("00:33:00"), reason=refresh.window_refusal("inc-another-window")
            ),
            "not the chain's window refusal",
        ),
        (_rejected_line(_at("00:33:00"), gpu_count="8"), "['gpu_count']"),
        (_rejected_line(_at("00:33:00"), attempt="1"), "['attempt']"),
        (_rejected_line(_at("00:31:50")), "not between the window's opening"),
        (_rejected_line(_at("00:35:30")), "not between the window's opening"),
        (
            _rejected_line(
                _at("00:33:00"),
                command_id="workflow-x/1/TRIGGER_HEALTH_SNAPSHOT/other/agent-41",
                incident_id="-",
                workflow_request_id="-",
                node_id="-",
                fencing_token="-",
                agent_generation="-",
                gpu_count="-",
                attempt="-",
            ),
            "not an observation refresh that ran",
        ),
    ],
    ids=[
        "refused-chain-command",
        "refused-diagnostic",
        "another-refusal-reason",
        "another-window",
        "refusal-naming-gpus",
        "refusal-at-an-attempt",
        "refused-before-the-window",
        "refused-after-the-refresh-ran",
        "pool-line-of-another-command",
    ],
)
def test_a_rejection_that_is_not_the_window_s_refusal_of_the_refresh_fails(
    line: dict[str, Any], message: str
) -> None:
    counters = {"accepted": 5, "completed": 5, "failed": 0, "rejected": 42}
    rows = [*_live_chain_rows(), _live_refresh_row()]
    if not all(name in line["fields"] for name in refresh.IDENTIFIER_FIELDS):
        counters["rejected"] = 41  # the action pool's line is not one more count

    errors = _judge(rows, [*_live_refusals(), line], counters)

    assert any(message in item for item in errors), errors
    assert not any("counters" in item for item in errors), (
        "the count still matches; the line itself is the failure"
    )


def test_the_refusals_are_counted_exactly_and_only_the_restarted_agent_s() -> None:
    rows = [*_live_chain_rows(), _live_refresh_row()]
    counters = {"accepted": 5, "completed": 5, "failed": 0, "rejected": 41}
    before_restart = _rejected_line(_at("00:30:45"), invocation="invocation-before")
    pool = _rejected_line(
        _at("00:33:00"),
        incident_id="-",
        workflow_request_id="-",
        node_id="-",
        fencing_token="-",
        agent_generation="-",
        gpu_count="-",
        attempt="-",
    )

    assert _judge(rows, [before_restart, pool, *_live_refusals()], counters) == [], (
        "the previous Agent's lines and the pool's own line beside a counted "
        "refusal are not counted"
    )
    for lines, rejected in ((_live_refusals(40), 41), (_live_refusals(), 40)):
        errors = _judge(rows, lines, {**counters, "rejected": rejected})
        assert any("counters" in item for item in errors), (lines, rejected)


def _kubelet_state(restored: list[str]) -> dict[str, Any]:
    return {
        "commands": [
            {
                "status": "SUCCEEDED",
                "step": {"operation": "RESTORE_GPU_SERVICES"},
                "result_details": {"restored_services": {LIVE_NODE: restored}},
            }
        ]
    }


LIVE_SERVICES = [
    "nvidia-fabricmanager",
    "nvidia-dcgm",
    "gpu-fault-gpu-persistence",
    "gpu-fault-metrics-collector",
    "gpu-fault-host-collector",
    "kubelet",
]
LIVE_NODE_BASELINE = {
    "boot_id": "boot-a",
    "gpu_allocatable": "8",
    "installer_boot_id": "boot-a",
    "name": LIVE_NODE,
    "ownership_annotations": {},
    "ready": "True",
    "ready_since": "2026-10-29T00:04:23Z",
    "spare_pool_annotations": {},
    "taints": [],
    "tolerated_taints": [],
    "uid": "node-uid-a",
    "unschedulable": False,
}


def _node_errors(final: dict[str, Any], restored: list[str]) -> list[str]:
    restarted = verdicts.kubelet_restart(
        _kubelet_state(restored),
        {"rows": _live_chain_rows()},
        node=LIVE_NODE,
        workflow_request_id=LIVE_WORKFLOW,
    )
    return verdicts.node_errors(LIVE_NODE_BASELINE, final, kubelet_restarted=restarted)


def test_the_ready_flip_of_the_hyperpod_quiesce_is_the_chain_s_own() -> None:
    final = {**LIVE_NODE_BASELINE, "ready_since": "2026-10-29T00:35:24Z"}

    assert _node_errors(final, LIVE_SERVICES) == [], "the live node passes"
    assert _node_errors(LIVE_NODE_BASELINE, LIVE_SERVICES) == [], "no flip at all"
    assert _node_errors(LIVE_NODE_BASELINE, LIVE_SERVICES[:-1]) == [], (
        "plain EKS: kubelet kept, Ready untouched"
    )


@pytest.mark.parametrize(
    ("final", "restored", "message"),
    [
        (
            {**LIVE_NODE_BASELINE, "ready_since": "2026-10-29T00:35:24Z"},
            LIVE_SERVICES[:-1],
            "differs from baseline",
        ),
        (
            {**LIVE_NODE_BASELINE, "ready_since": "2026-10-29T00:33:03Z"},
            LIVE_SERVICES,
            "not since the restore started kubelet again",
        ),
        (
            {**LIVE_NODE_BASELINE, "ready_since": "2026-10-29T00:36:30Z"},
            LIVE_SERVICES,
            "not since the restore started kubelet again",
        ),
        (
            {
                **LIVE_NODE_BASELINE,
                "ready_since": "2026-10-29T00:35:24Z",
                "unschedulable": True,
            },
            LIVE_SERVICES,
            "differs from baseline",
        ),
        (
            {
                **LIVE_NODE_BASELINE,
                "ready_since": "2026-10-29T00:35:24Z",
                "boot_id": "another-boot",
            },
            LIVE_SERVICES,
            "differs from baseline",
        ),
        (
            {
                **LIVE_NODE_BASELINE,
                "ready_since": "2026-10-29T00:35:24Z",
                "ready": "False",
            },
            LIVE_SERVICES,
            "differs from baseline",
        ),
    ],
    ids=[
        "flip-without-a-kubelet-stop",
        "ready-again-before-the-restore",
        "ready-again-after-the-restore",
        "flip-and-a-cordon",
        "flip-and-a-reboot",
        "not-ready-at-the-end",
    ],
)
def test_any_other_node_change_still_fails(
    final: dict[str, Any], restored: list[str], message: str
) -> None:
    errors = _node_errors(final, restored)

    assert any(message in item for item in errors), errors


def _refresh_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, rejected: int
) -> AgentHarness:
    """The graded run with the refresh beside the chain: its row after the
    restore began, three refusals inside the window, kubelet restored by the
    restore and the Node Ready again inside it."""

    h = _graded_harness(tmp_path, monkeypatch)
    [request] = refresh.requests(
        cluster_id=h.settings.regional.cluster_id,
        node=NODE,
        generation=GENERATION,
        start=NOW,
        end=NOW,
    ).values()
    rows = h.chain_audit["rows"]
    restore = next(row for row in rows if row["operation"] == "RESTORE_GPU_SERVICES")
    restore["completed_at"] = (NOW + timedelta(seconds=60)).isoformat()
    quiesce = next(row for row in rows if row["operation"] == "QUIESCE_GPU_SERVICES")
    opened = datetime.fromisoformat(quiesce["started_at"])
    rows.append(
        {
            "command_id": request.command_id,
            "operation": "TRIGGER_HEALTH_SNAPSHOT",
            "state": "SUCCEEDED",
            "attempt": 1,
            "incident_id": request.incident_id,
            "workflow_request_id": request.workflow_request_id,
            "fencing_token": 1,
            "gpu_uuid_count": 0,
            "exit_code": None,
            "agent_generation": GENERATION,
            "started_at": (NOW + timedelta(seconds=30)).isoformat(),
            "completed_at": (NOW + timedelta(seconds=35)).isoformat(),
        }
    )
    refusals = [
        {
            "phase": "rejected",
            "fields": {
                "command_id": request.command_id,
                "incident_id": request.incident_id,
                "workflow_request_id": request.workflow_request_id,
                "operation": "TRIGGER_HEALTH_SNAPSHOT",
                "node_id": NODE,
                "fencing_token": "1",
                "agent_generation": str(GENERATION),
                "gpu_count": "0",
                "attempt": "0",
                "reason": refresh.window_refusal(INCIDENT),
            },
            "realtime_timestamp": str(
                int((opened + timedelta(seconds=1 + index)).timestamp() * 1_000_000)
            ),
            "invocation_id": h.restart_report["after"]["InvocationID"],
        }
        for index in range(3)
    ]
    h.journal_since = {"lines": [*h.journal["lines"], *refusals]}
    h.health_after = health(
        {"accepted": 6, "completed": 5, "failed": 1, "rejected": rejected}
    )
    for item in h.state["commands"]:
        if item["step"]["operation"] == "RESTORE_GPU_SERVICES":
            item["result_details"] = {"restored_services": {NODE: ["kubelet"]}}
    h.node = {**h.node, "ready_since": "2026-09-12T11:00:00Z"}
    h.node_after = {**h.node, "ready_since": (NOW + timedelta(seconds=40)).isoformat()}
    h.plan(tmp_path)
    return h


def test_the_runner_accepts_the_refresh_its_refusals_and_the_ready_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _refresh_harness(tmp_path, monkeypatch, rejected=3)

    code, report = h.execute(tmp_path)

    assert code == 0 and report["errors"] == [], report
    assert report["cleanup"]["errors"] == [], report["cleanup"]
    names = [name for name, _detail in h.calls]
    journals = [detail for name, detail in h.calls if name == "agent.journal"]
    assert [("--command-id" in detail["args"]) for detail in journals] == [
        True,
        False,
    ], journals
    health_read = names.index("agent.agent-health", names.index("injector.write-xid45"))
    last_journal = len(names) - 1 - names[::-1].index("agent.journal")
    assert health_read < last_journal, (
        "the refusal journal is read after the health that counted them"
    )


def test_the_runner_fails_a_rejection_the_journal_does_not_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = _refresh_harness(tmp_path, monkeypatch, rejected=4)

    code, report = h.execute(tmp_path)

    assert code == 1, report
    assert "counters" in " ".join(report["errors"]), report["errors"]
    assert report["cleanup"]["errors"] == [], "the Ready flip is still the chain's"
