"""Contract tests for GF-REGIONAL-DESTR-019.

Every verdict is judged against synthetic evidence documents -- the probe's
/healthz, journald and ledger-audit answers, store snapshots of the Fabric
Manager workflow -- once on the intended run and once per way the run can be
wrong. Nothing here touches a cluster; the runner's live phases only call these
functions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr019_verdicts as verdicts
from scripts.e2e.regional import run_destr019_agent_restart_ledger as destr019
from scripts.e2e.regional.regional_case_contract import RegionalCaseMetadata

ROOT = Path(__file__).resolve().parents[2]
NODE = "node-a"
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
RESTARTED_AT = T0 + timedelta(minutes=1)
GENERATION = 4
INCIDENT = "inc-45"
WORKFLOW = "workflow-45"
COMMAND = {"idempotency_key": f"{WORKFLOW}/1/RESTART_FABRIC_MANAGER"}
COMMAND_ID = verdicts.expected_command_id(COMMAND, node=NODE, generation=GENERATION)


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


# --------------------------------------------------------------------------- #
# Case contract
# --------------------------------------------------------------------------- #
def test_the_confirmation_names_this_case_and_the_predecessor_is_destr010() -> None:
    metadata = RegionalCaseMetadata(
        case_id=destr019.CASE_ID,
        title="",
        category="regional-destructive-acceptance",
        level="staging",
        risk="live-service-action",
        automation="manual",
        procedure="docs/x.md#gf-regional-destr-019",
        predecessor=destr019.PREDECESSOR_CASE_ID,
    )
    prefix = metadata.confirmation.removesuffix("EXECUTE")
    assert prefix == "DESTR019_", prefix
    assert destr019.CONFIRMATION.startswith(prefix), destr019.CONFIRMATION
    assert destr019.PREDECESSOR_CASE_ID == "GF-REGIONAL-DESTR-010", (
        "the Fabric Manager restart path must already be proven"
    )
    assert verdicts.CASE_ID == "GF-REGIONAL-DESTR-019", verdicts.CASE_ID


def test_the_expected_command_id_is_the_pinned_single_node_form() -> None:
    assert COMMAND_ID == f"{WORKFLOW}/1/RESTART_FABRIC_MANAGER/{NODE}/agent-4", (
        COMMAND_ID
    )


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def _node() -> dict[str, Any]:
    return {
        "uid": "uid-1",
        "boot_id": "boot-1",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }


def _agent(**overrides: Any) -> dict[str, Any]:
    agent = {
        "lifecycle_state": "ACTIVE",
        "generation": GENERATION,
        "agent_incarnation_id": "inc-a",
        "boot_id": "boot-1",
        "allowed_operations": ["RESTART_FABRIC_MANAGER", "VERIFY_NO_GPU_CLIENTS"],
        "last_seen_at": (RESTARTED_AT + timedelta(seconds=20)).isoformat(),
    }
    agent.update(overrides)
    return agent


def _host(**overrides: Any) -> dict[str, Any]:
    host = {
        "boot_id": "boot-1",
        "agent": {"ActiveState": "active", "MainPID": "100", "InvocationID": "i1"},
        "ledger": {
            "present": True,
            "user_version": verdicts.LEDGER_SCHEMA_VERSION,
            "interrupted_count": 0,
        },
        "gpu_fault_timers": ["gpu-fault-certificate-check.timer"],
        "restore_timer": {"ActiveState": "inactive"},
    }
    host.update(overrides)
    return host


def _preflight(**overrides: Any) -> list[str]:
    arguments: dict[str, Any] = {
        "node": _node(),
        "agent": _agent(),
        "profile": {"warnings": []},
        "workloads": [],
        "queue": {"depth": 0},
        "remote_commands": {"open_by_cluster": {}},
        "recent_event": None,
        "host": _host(),
        "tests_passed": True,
    }
    arguments.update(overrides)
    return verdicts.preflight_errors(**arguments)


def test_the_preflight_passes_on_an_idle_node_with_a_migrated_ledger() -> None:
    assert _preflight() == [], "the intended preflight must be clean"


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"node": {**_node(), "unschedulable": True}}, "already unschedulable"),
        ({"workloads": [{"namespace": "x", "name": "y"}]}, "non-system running Pods"),
        ({"agent": _agent(lifecycle_state="DRAINING")}, "not ACTIVE"),
        ({"agent": _agent(allowed_operations=[])}, "RESTART_FABRIC_MANAGER"),
        ({"queue": {"depth": 3}}, "processor queue"),
        ({"remote_commands": {"open_by_cluster": {"c": 1}}}, "remote command queue"),
        ({"recent_event": {"xid": 45}}, "recent XID"),
        (
            {"host": _host(ledger={"present": True, "user_version": 0})},
            "predates ARCH-C4",
        ),
        ({"host": _host(restore_timer={"ActiveState": "active"})}, "fail-safe"),
        ({"tests_passed": False}, "focused regression"),
    ],
)
def test_the_preflight_refuses_each_unsafe_precondition(
    overrides: dict[str, Any], fragment: str
) -> None:
    assert fragment in _text(_preflight(**overrides)), (fragment, overrides)


# --------------------------------------------------------------------------- #
# /healthz (ARCH-C5)
# --------------------------------------------------------------------------- #
def _health(
    *,
    status: int = 200,
    counters: dict[str, int] | None = None,
    writable: bool = True,
    heartbeat_success: bool = True,
    age: float = 12.0,
) -> dict[str, Any]:
    return {
        "http_status": status,
        "payload": {
            "status": "ok" if status == 200 else "degraded",
            "ledger": {"writable": writable},
            "heartbeat": {
                "configured": True,
                "consecutive_failures": 0,
                "last_success_at": T0.isoformat() if heartbeat_success else None,
                "last_success_age_seconds": age if heartbeat_success else None,
            },
            "counters": counters
            if counters is not None
            else dict.fromkeys(verdicts.COUNTER_NAMES, 0),
        },
    }


def test_the_health_payload_contract_passes_on_a_fresh_agent() -> None:
    zero = dict.fromkeys(verdicts.COUNTER_NAMES, 0)
    assert (
        verdicts.health_errors(
            _health(), label="x", expect_counters=zero, max_heartbeat_age_seconds=120
        )
        == []
    ), "a healthy, freshly heartbeating agent must pass"


def test_a_degraded_or_unwritable_ledger_fails_the_health_contract() -> None:
    errors = verdicts.health_errors(_health(status=503, writable=False), label="x")
    assert "returned 503" in _text(errors), errors
    assert "ledger writable" in _text(errors), errors


def test_counters_and_heartbeat_age_are_judged_exactly() -> None:
    moved = _health(
        counters={"accepted": 1, "completed": 1, "failed": 0, "rejected": 0}
    )
    assert (
        verdicts.health_errors(
            moved,
            label="after command",
            expect_counters={"accepted": 1, "completed": 1, "failed": 0, "rejected": 0},
        )
        == []
    ), "one accepted/completed pair is the expected delta"
    stale = _health(age=900.0)
    assert "heartbeat age" in _text(
        verdicts.health_errors(stale, label="x", max_heartbeat_age_seconds=120)
    ), "a stale heartbeat must fail"
    never = _health(heartbeat_success=False)
    assert "never succeeded" in _text(verdicts.health_errors(never, label="x"))
    shape = _health()
    shape["payload"]["counters"] = {"accepted": 0}
    assert "counters are not" in _text(verdicts.health_errors(shape, label="x"))
    delta = verdicts.counter_delta(_health(), moved)
    assert delta == {"accepted": 1, "completed": 1, "failed": 0, "rejected": 0}, delta


# --------------------------------------------------------------------------- #
# Restart and agent record
# --------------------------------------------------------------------------- #
def _restart(**overrides: Any) -> dict[str, Any]:
    value = {
        "before": {"ActiveState": "active", "MainPID": "100", "InvocationID": "i1"},
        "after": {"ActiveState": "active", "MainPID": "200", "InvocationID": "i2"},
        "boot_id": "boot-1",
        "restore_unit": "gpu-fault-destr019-restore-abc.timer",
        "armed_at": (RESTARTED_AT - timedelta(seconds=1)).isoformat(),
        "restarted_at": RESTARTED_AT.isoformat(),
    }
    value.update(overrides)
    return value


def test_the_restart_contract_passes_when_only_the_agent_process_changed() -> None:
    assert verdicts.restart_errors(_restart(), baseline_boot_id="boot-1") == [], (
        "a pid/invocation change on the same boot behind an armed fail-safe passes"
    )


def test_the_restart_contract_rejects_a_reboot_an_unchanged_pid_or_a_late_fail_safe() -> (
    None
):
    reboot = verdicts.restart_errors(
        _restart(boot_id="boot-2"), baseline_boot_id="boot-1"
    )
    assert "boot id changed" in _text(reboot), reboot
    same = verdicts.restart_errors(
        _restart(
            after={"ActiveState": "active", "MainPID": "100", "InvocationID": "i1"}
        ),
        baseline_boot_id="boot-1",
    )
    assert "MainPID did not change" in _text(same), same
    late = verdicts.restart_errors(
        _restart(armed_at=(RESTARTED_AT + timedelta(seconds=5)).isoformat()),
        baseline_boot_id="boot-1",
    )
    assert "armed before the restart" in _text(late), late
    down = verdicts.restart_errors(
        _restart(after={"ActiveState": "failed", "MainPID": "0"}),
        baseline_boot_id="boot-1",
    )
    assert "not active after the restart" in _text(down), down


def test_the_agent_record_keeps_generation_and_incarnation_but_heartbeats_again() -> (
    None
):
    assert (
        verdicts.agent_record_errors(_agent(), _agent(), restarted_at=RESTARTED_AT)
        == []
    ), "same generation, same incarnation, fresh heartbeat passes"
    bumped = verdicts.agent_record_errors(
        _agent(), _agent(generation=GENERATION + 1), restarted_at=RESTARTED_AT
    )
    assert "generation moved" in _text(bumped), bumped
    silent = verdicts.agent_record_errors(
        _agent(),
        _agent(last_seen_at=(RESTARTED_AT - timedelta(minutes=5)).isoformat()),
        restarted_at=RESTARTED_AT,
    )
    assert "no heartbeat" in _text(silent), silent
    retired = verdicts.agent_record_errors(
        _agent(), _agent(agent_incarnation_id="inc-b"), restarted_at=RESTARTED_AT
    )
    assert "incarnation id changed" in _text(retired), retired


# --------------------------------------------------------------------------- #
# Workflow (DESTR-010 contract)
# --------------------------------------------------------------------------- #
def _state(**overrides: Any) -> dict[str, Any]:
    state = {
        "event": {"xid": 45},
        "decision": {"official_action": "RESTART_FM", "disposition": "EXECUTABLE"},
        "workflow": {
            "request_id": WORKFLOW,
            "status": "SUCCEEDED",
            "fencing_token": 2,
            "official_steps": [
                {"operation": op, "execution_owner": owner}
                for op, owner in zip(verdicts.EXPECTED_STEPS, verdicts.EXPECTED_OWNERS)
            ],
            "step_executions": [
                {"operation": op, "status": "SUCCEEDED"}
                for op in verdicts.EXPECTED_STEPS
            ],
        },
        "incident": {"incident_id": INCIDENT, "node_ids": [NODE]},
        "commands": [{**COMMAND, "status": "SUCCEEDED"}],
    }
    state.update(overrides)
    return state


def test_the_workflow_contract_passes_on_the_two_step_fabric_manager_restart() -> None:
    assert verdicts.workflow_errors(_state(), node=NODE) == [], "DESTR-010 shape passes"


def test_the_workflow_contract_rejects_isolation_or_a_widened_incident() -> None:
    isolated = _state()
    isolated["workflow"]["step_executions"].append(
        {"operation": "MARK_UNSCHEDULABLE", "status": "SUCCEEDED"}
    )
    assert "forbidden operations" in _text(
        verdicts.workflow_errors(isolated, node=NODE)
    )
    widened = _state(incident={"incident_id": INCIDENT, "node_ids": [NODE, "node-b"]})
    assert "not just node-a" in _text(verdicts.workflow_errors(widened, node=NODE))
    failed = _state()
    failed["workflow"]["status"] = "FAILED"
    assert "not SUCCEEDED" in _text(verdicts.workflow_errors(failed, node=NODE))


# --------------------------------------------------------------------------- #
# journald (ARCH-C3)
# --------------------------------------------------------------------------- #
def _fields(**extra: str) -> dict[str, str]:
    return {
        "command_id": COMMAND_ID,
        "incident_id": INCIDENT,
        "workflow_request_id": WORKFLOW,
        "operation": "RESTART_FABRIC_MANAGER",
        "node_id": NODE,
        "fencing_token": "2",
        "agent_generation": str(GENERATION),
        "gpu_count": "0",
        "attempt": "1",
        **extra,
    }


def _lines() -> list[dict[str, Any]]:
    return [
        {"phase": "accepted", "fields": _fields()},
        {"phase": "started", "fields": _fields()},
        {
            "phase": "completed",
            "fields": _fields(status="SUCCEEDED", duration_ms="812"),
        },
    ]


def _journal_errors(lines: list[dict[str, Any]]) -> list[str]:
    return verdicts.journal_errors(
        lines,
        command_id=COMMAND_ID,
        incident_id=INCIDENT,
        workflow_request_id=WORKFLOW,
        node=NODE,
        fencing_token=2,
        generation=GENERATION,
    )


def test_the_journal_contract_passes_on_accepted_started_completed() -> None:
    assert _journal_errors(_lines()) == [], "three identifier-only lines pass"


def test_the_journal_contract_rejects_a_missing_phase_a_wrong_field_or_a_leak() -> None:
    short = _journal_errors(_lines()[:2])
    assert "journal phases" in _text(short), short
    wrong = _lines()
    wrong[1]["fields"]["incident_id"] = "inc-other"
    assert "incident_id='inc-other'" in _text(_journal_errors(wrong))
    leaked = _lines()
    leaked[0]["fields"]["gpu_uuids"] = "GPU-1"
    assert "forbidden fields ['gpu_uuids']" in _text(_journal_errors(leaked))
    missing = _lines()
    del missing[2]["fields"]["agent_generation"]
    assert "missing fields ['agent_generation']" in _text(_journal_errors(missing))
    failed = _lines()
    failed[2]["fields"]["status"] = "FAILED"
    assert "status='FAILED'" in _text(_journal_errors(failed))


# --------------------------------------------------------------------------- #
# Ledger audit (ARCH-C4)
# --------------------------------------------------------------------------- #
def _audit(**row_overrides: Any) -> dict[str, Any]:
    row = {
        "command_id": COMMAND_ID,
        "attempt": 1,
        "state": "SUCCEEDED",
        "operation": "RESTART_FABRIC_MANAGER",
        "started_at": T0.isoformat(),
        "completed_at": (T0 + timedelta(seconds=2)).isoformat(),
        "incident_id": INCIDENT,
        "workflow_request_id": WORKFLOW,
        "fencing_token": 2,
        "gpu_uuid_count": 0,
        "gpu_uuids_present": True,
        "parameters_digest": "a" * 64,
        "signature_digest": "b" * 64,
        "exit_code": None,
    }
    row.update(row_overrides)
    return {
        "present": True,
        "user_version": verdicts.LEDGER_SCHEMA_VERSION,
        "columns": ["command_id", "payload", *verdicts.AUDIT_COLUMNS],
        "primary_key": list(verdicts.LEDGER_PRIMARY_KEY),
        "row_count": 10,
        "interrupted_count": 0,
        "rows": [row],
    }


def _ledger_errors(
    audit: dict[str, Any], *, baseline_interrupted: int = 0
) -> list[str]:
    return verdicts.ledger_row_errors(
        audit,
        command_id=COMMAND_ID,
        incident_id=INCIDENT,
        workflow_request_id=WORKFLOW,
        fencing_token=2,
        baseline_interrupted=baseline_interrupted,
    )


def test_the_ledger_contract_passes_on_one_audited_attempt_row() -> None:
    assert _ledger_errors(_audit()) == [], "one complete attempt-1 row passes"


def test_the_ledger_contract_rejects_missing_columns_bad_digests_and_interruptions() -> (
    None
):
    old = _audit()
    old["user_version"] = 0
    old["primary_key"] = ["command_id"]
    old["columns"] = ["command_id", "payload", "attempt", "state"]
    errors = _ledger_errors(old)
    assert "user_version is 0" in _text(errors), errors
    assert "lacks audit columns" in _text(errors), errors
    assert "primary key is" in _text(errors), errors
    digest = _ledger_errors(_audit(parameters_digest="not-a-digest"))
    assert "parameters_digest is not a sha256" in _text(digest), digest
    interrupted = _audit()
    interrupted["interrupted_count"] = 1
    assert "gained INTERRUPTED rows" in _text(_ledger_errors(interrupted))
    exit_code = _ledger_errors(_audit(exit_code=3))
    assert "carries exit_code" in _text(exit_code), exit_code
    two_rows = _audit()
    two_rows["rows"].append({**two_rows["rows"][0], "attempt": 2})
    assert "expected one ledger row" in _text(_ledger_errors(two_rows))
    wrong_token = _ledger_errors(_audit(fencing_token=9))
    assert "fencing_token is 9" in _text(wrong_token), wrong_token


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
def test_cleanup_requires_the_fail_safe_disarmed_and_no_new_timers() -> None:
    clean = {
        "restore_timer": {"ActiveState": "inactive"},
        "agent": {"ActiveState": "active"},
        "gpu_fault_timers": ["gpu-fault-certificate-check.timer"],
    }
    baseline = ["gpu-fault-certificate-check.timer"]
    assert verdicts.cleanup_errors(clean, baseline_timers=baseline) == [], (
        "clean passes"
    )
    armed = {**clean, "restore_timer": {"ActiveState": "active"}}
    assert "still armed" in _text(
        verdicts.cleanup_errors(armed, baseline_timers=baseline)
    )
    extra = {
        **clean,
        "gpu_fault_timers": [*baseline, "gpu-fault-destr019-restore-x.timer"],
    }
    assert "timers remain" in _text(
        verdicts.cleanup_errors(extra, baseline_timers=baseline)
    )
    assert verdicts.node_errors(_node(), {**_node(), "unschedulable": True}) == [
        "target Kubernetes Node state differs from baseline"
    ], "a node that drifted must fail"


# --------------------------------------------------------------------------- #
# Plan and command line
# --------------------------------------------------------------------------- #
def _settings(tmp_path: Path) -> destr019.Settings:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings

    return destr019.Settings(
        regional=RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        ),
        node=NODE,
        host_probe_image="registry.example/node-installer@sha256:" + "0" * 64,
        predecessor_path=tmp_path / "GF-REGIONAL-DESTR-010.json",
        restore_seconds=180,
    )


def _preflight_document() -> dict[str, Any]:
    return {
        "release_id": "rel-1",
        "node": _node(),
        "store": {"agent": _agent(), "profile": {"profile_version": "hyperpod-v2"}},
        "host": _host(),
        "predecessor": {"valid": True, "case_id": destr019.PREDECESSOR_CASE_ID},
    }


def test_the_plan_names_the_risk_the_hard_stops_and_the_fail_safe(
    tmp_path: Path,
) -> None:
    details = destr019.plan_details(_settings(tmp_path), _preflight_document())
    assert details["risk"] == "live-service-action", details["risk"]
    assert details["target_node"] == NODE, details
    assert "no command in flight" in details["mutation"], details["mutation"]
    assert "No reset, no reboot" in details["mutation"], details["mutation"]
    assert details["preflight_identity"]["agent_generation"] == GENERATION, details
    assert details["preflight_identity"]["ledger_user_version"] == 2, details
    assert (
        details["rollback"]["fail_safe_start_timer_is_armed_before_the_restart"] is True
    )
    assert details["rollback"]["migration_drill_uses_a_scratch_ledger_only"] is True
    conditions = "\n".join(details["stop_conditions"])
    assert "fail-safe restore timer" in conditions, conditions
    assert "new generation" in conditions, conditions
    assert "predates ARCH-C4" in conditions, conditions


def test_a_plan_that_drifted_from_its_preflight_is_refused(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / destr019.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"preflight_identity": {"release_id": "rel-0"}}}),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="plan drifted"):
        destr019.verify_plan_identity(case_dir, _preflight_document())


def test_the_runner_is_plan_by_default_and_needs_an_exact_confirmation() -> None:
    parser = destr019.parser()
    plan = parser.parse_args(["--run-dir", "/tmp/run"])
    assert plan.execute is False, "the runner must default to plan mode"
    assert plan.restore_seconds == verdicts.RESTORE_SECONDS, plan.restore_seconds
    execute = parser.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--execute",
            "--confirm",
            destr019.CONFIRMATION,
            "--maintenance-window-end",
            "2026-09-06T12:00:00+00:00",
            "--node",
            NODE,
        ]
    )
    assert execute.execute is True and execute.confirm == destr019.CONFIRMATION, execute
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "/tmp/run", "--plan", "--execute"])
    assert isinstance(parser, argparse.ArgumentParser), "parser type"


def test_configure_refuses_an_unbounded_fail_safe(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    arguments = destr019.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--cpu-kubeconfig",
            str(cpu),
            "--gpu-kubeconfig",
            str(gpu),
            "--gpu-context",
            "ctx",
            "--cluster-id",
            "cluster-a",
            "--region",
            "us-west-2",
            "--node",
            NODE,
            "--host-probe-image",
            "img",
            "--restore-seconds",
            "30",
        ]
    )
    with pytest.raises(Exception, match="restore seconds"):
        destr019.configure(arguments)


def test_the_help_text_offers_the_four_documented_live_flags() -> None:
    help_text = destr019.parser().format_help()
    for flag in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert flag in help_text, flag


def test_the_runner_and_probe_are_executable_with_a_shebang() -> None:
    for path in (
        ROOT / "scripts/e2e/regional/run_destr019_agent_restart_ledger.py",
        ROOT / "scripts/e2e/regional/probes/destr019_node_probe.py",
    ):
        mode = path.stat().st_mode & 0o777
        assert mode == 0o775, f"{path.name} is {oct(mode)}, not 0o775"
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == "#!/usr/bin/env python3", first
