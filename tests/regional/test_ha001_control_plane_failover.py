"""HA-001: derived limits, settle-based observation and control-plane-only closure evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.e2e.regional import run_ha001_control_plane_failover as ha001


def _deployment(pre_stop_sleep: int | None = 20, graceful: int | None = 60) -> dict:
    container: dict = {
        "args": [
            "exec uvicorn gpu_fault.app:create_app --timeout-graceful-shutdown "
            f"{graceful} --no-access-log"
            if graceful is not None
            else "exec uvicorn gpu_fault.app:create_app"
        ]
    }
    if pre_stop_sleep is not None:
        container["lifecycle"] = {
            "preStop": {
                "exec": {"command": ["/bin/sh", "-c", f"sleep {pre_stop_sleep}"]}
            }
        }
    return {
        "pre_stop": container.get("lifecycle", {}).get("preStop"),
        "pre_stop_sleep_seconds": ha001.pre_stop_sleep_seconds(
            container.get("lifecycle", {}).get("preStop")
        ),
        "graceful_shutdown_seconds": ha001.graceful_shutdown_seconds(container),
    }


def test_failure_window_limit_is_derived_from_the_deployment_not_a_constant() -> None:
    limit = ha001.failure_window_limit(
        _deployment(20, 60), nlb_deregistration_seconds=30
    )

    assert limit == {
        "pre_stop_sleep_seconds": 20,
        "graceful_shutdown_seconds": 60,
        "nlb_deregistration_seconds": 30,
        "limit_seconds": 110,
    }
    assert ha001.failure_window_limit(_deployment(5, 10))["limit_seconds"] == (
        5 + 10 + ha001.NLB_DEREGISTRATION_SECONDS
    )
    assert not hasattr(ha001, "MAX_FAILURE_WINDOW_SECONDS"), (
        "the failure window must be derived from the Deployment, not a constant"
    )
    with pytest.raises(ha001.CaseError, match="cannot derive"):
        ha001.failure_window_limit(_deployment(None, 60))
    with pytest.raises(ha001.CaseError, match="cannot derive"):
        ha001.failure_window_limit(_deployment(20, None))


def test_replica_counts_come_from_spec_replicas() -> None:
    topology = {
        "deployments": {
            ha001.INGRESS_APP: {"replicas": 4},
            ha001.WORKER_APP: {"replicas": 8},
            ha001.SPOOL_APP: {"replicas": 0},
        }
    }
    replicas = ha001.declared_replicas(topology)

    assert replicas == {ha001.INGRESS_APP: 4, ha001.WORKER_APP: 8, ha001.SPOOL_APP: 0}
    assert ha001.minimum_ready_from_replicas(replicas) == {
        "ingress": 3,
        "worker": 7,
        "spool": None,
    }
    with pytest.raises(ha001.CaseError, match="no Deployment"):
        ha001.declared_replicas({"deployments": {ha001.INGRESS_APP: {"replicas": 3}}})


def test_validate_roles_uses_declared_replicas_and_requires_null_leadership() -> None:
    replicas = {ha001.INGRESS_APP: 1, ha001.WORKER_APP: 2, ha001.SPOOL_APP: 0}
    ingress = {
        "name": "api-a",
        "health": {
            "service_role": "ingress",
            "processor_role": "inactive",
            "leadership": None,
        },
        "processor_active_consumer": 0.0,
    }
    worker = {
        "name": "worker-a",
        "health": {
            "service_role": "worker",
            "processor_role": "active-consumer",
            "leadership": None,
        },
        "processor_active_consumer": 1.0,
    }
    clean = {
        ha001.INGRESS_APP: [ingress],
        ha001.WORKER_APP: [worker, {**worker, "name": "worker-b"}],
    }
    assert ha001.validate_roles(clean, replicas) == []

    leader = {
        **worker,
        "name": "worker-c",
        "health": {**worker["health"], "leadership": {"owner": "worker-c"}},
    }
    errors = ha001.validate_roles(
        {ha001.INGRESS_APP: [ingress], ha001.WORKER_APP: [worker, leader]}, replicas
    )
    assert errors == ["worker-c reports leadership; expected null"], errors

    missing_leadership_key = {
        **worker,
        "health": {k: v for k, v in worker["health"].items() if k != "leadership"},
    }
    assert any(
        "leadership" in item
        for item in ha001.validate_roles(
            {
                ha001.INGRESS_APP: [ingress],
                ha001.WORKER_APP: [worker, missing_leadership_key],
            },
            replicas,
        )
    ), "a healthz without the leadership field is not proof of null leadership"

    short = ha001.validate_roles(
        {ha001.INGRESS_APP: [ingress], ha001.WORKER_APP: [worker]}, replicas
    )
    assert "worker role snapshot does not contain 2 Pods" in short


def test_minimum_probe_cycles_scales_with_observed_time() -> None:
    assert ha001.minimum_probe_cycles(0) == 0
    assert ha001.minimum_probe_cycles(930) == int(930 / 2.0 * 0.8)
    assert ha001.minimum_probe_cycles(300) < ha001.minimum_probe_cycles(930)


def test_observe_phase_ends_after_settle_and_not_at_the_cap(monkeypatch) -> None:
    clock = {"now": 0.0}
    probe_state = {"current_failure_window_seconds": 5.0}
    ready = {"ingress_ready": 2}

    def control_sample(include_queue: bool) -> dict:
        return {
            "observed_at": "t",
            "ingress_ready": ready["ingress_ready"],
            "worker_ready": 6,
            "spool_ready": 0,
            "endpoint_ready": 3,
            "ingress_pods": ["api-b", "api-c"]
            + (["api-new"] if ready["ingress_ready"] == 3 else []),
            "worker_pods": [],
            "spool_pods": [],
            **({"queue": {"depth": 0}} if include_queue else {}),
        }

    def read_probe(path: str = "/state/stats.json") -> dict:
        return {
            "current_failure_window_seconds": probe_state[
                "current_failure_window_seconds"
            ],
            "max_failure_window_seconds": 5.0,
            "error_counts": {},
        }

    def sleep(seconds: float) -> None:
        clock["now"] += max(seconds, 2.0)
        if clock["now"] >= 20:
            ready["ingress_ready"] = 3
            probe_state["current_failure_window_seconds"] = 0.0

    monkeypatch.setattr(ha001, "control_sample", control_sample)
    monkeypatch.setattr(ha001, "read_probe", read_probe)
    monkeypatch.setattr(ha001, "log", lambda _message: None)
    target = {"app": ha001.INGRESS_APP, "name": "api-a", "phase": "ingress-1"}

    phase = ha001.observe_phase(
        "ingress-1",
        300,
        0,
        minimum_ready={"ingress": 2, "worker": 5, "spool": None},
        max_failure_window_seconds=110,
        settled=ha001.replacement_settled(target, desired=3),
        settle_seconds=60,
        sleep=sleep,
        clock=lambda: clock["now"],
    )

    assert phase["settle_reached"] is True
    assert 80 <= phase["duration_seconds"] < 300, phase["duration_seconds"]
    assert phase["cap_seconds"] == 300


def test_observe_phase_reports_cap_reached_when_never_settled(monkeypatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr(
        ha001,
        "control_sample",
        lambda include_queue: {
            "observed_at": "t",
            "ingress_ready": 2,
            "worker_ready": 6,
            "spool_ready": 0,
            "endpoint_ready": 2,
            "ingress_pods": ["api-a", "api-b"],
            "worker_pods": [],
            "spool_pods": [],
            **({"queue": {"depth": 0}} if include_queue else {}),
        },
    )
    monkeypatch.setattr(
        ha001,
        "read_probe",
        lambda path="/state/stats.json": {
            "current_failure_window_seconds": 0.0,
            "max_failure_window_seconds": 0.0,
            "error_counts": {},
        },
    )
    monkeypatch.setattr(ha001, "log", lambda _message: None)

    def sleep(seconds: float) -> None:
        clock["now"] += 10

    phase = ha001.observe_phase(
        "worker-1",
        100,
        0,
        minimum_ready={"ingress": 2, "worker": 5, "spool": None},
        max_failure_window_seconds=110,
        settled=lambda sample, probe: False,
        sleep=sleep,
        clock=lambda: clock["now"],
    )

    assert phase["settle_reached"] is False
    assert phase["duration_seconds"] >= 100


def _closure(statuses: list[str], owners: list[str | None], cached: list[bool]) -> dict:
    return {
        "workflow_id": "workflow-x",
        "workflow_status_observed": "BLOCKED",
        "commands": [
            {
                "command_id": f"remote-x-{index}",
                "step_index": index,
                "operation": op,
                "status": status,
                "lease_owner": None,
                "last_lease_owner": owner,
                "result_details": {"cached": flag, "physical_count": index + 1},
            }
            for index, (op, status, owner, flag) in enumerate(
                zip(
                    ["FREEZE_EVIDENCE", "STOP_WORKLOADS", "RESTART_WORKLOAD"],
                    statuses,
                    owners,
                    cached,
                    strict=True,
                )
            )
        ],
    }


SEED = {
    "workflow_id": "workflow-x",
    "command_ids": ["remote-x-0", "remote-x-1", "remote-x-2"],
    "operations": ["FREEZE_EVIDENCE", "STOP_WORKLOADS", "RESTART_WORKLOAD"],
}
LEDGER = {
    "physical_count": 3,
    "operations": ["FREEZE_EVIDENCE", "STOP_WORKLOADS", "RESTART_WORKLOAD"],
}


def test_closure_is_judged_from_control_plane_records_not_a_runner_write() -> None:
    executor = "ha001-probe/pod-a"
    closure = _closure(["SUCCEEDED"] * 3, [executor] * 3, [False] * 3)
    summary = ha001.closure_summary(SEED, closure, LEDGER)

    assert summary["succeeded_by_step_index"] == {"0": 1, "1": 1, "2": 1}
    assert summary["physical_executions"] == 3
    assert summary["workflow_status_observed"] == "BLOCKED", (
        "the runner records the workflow as the control plane left it"
    )
    assert ha001.closure_errors(SEED, summary, executor_id=executor) == []
    assert not hasattr(ha001, "finalize_closure"), (
        "the runner must not write the workflow status it then asserts on"
    )


def test_closure_errors_catch_replayed_or_foreign_completions() -> None:
    executor = "ha001-probe/pod-a"
    replayed = _closure(["SUCCEEDED"] * 3, [executor] * 3, [False, True, False])
    errors = ha001.closure_errors(
        SEED, ha001.closure_summary(SEED, replayed, LEDGER), executor_id=executor
    )
    assert any("executed physically once" in item for item in errors), errors

    foreign = _closure(
        ["SUCCEEDED"] * 3, [executor, "other/pod", executor], [False] * 3
    )
    errors = ha001.closure_errors(
        SEED, ha001.closure_summary(SEED, foreign, LEDGER), executor_id=executor
    )
    assert any("completed by other/pod" in item for item in errors), errors

    incomplete = _closure(
        ["SUCCEEDED", "LEASED", "SUCCEEDED"], [executor] * 3, [False] * 3
    )
    errors = ha001.closure_errors(
        SEED, ha001.closure_summary(SEED, incomplete, LEDGER), executor_id=executor
    )
    assert "synthetic command remote-x-1 is not SUCCEEDED" in errors
    assert any(
        "exactly one SUCCEEDED command per step_index" in item for item in errors
    ), f"a duplicate step_index must be named in the closure errors: {errors}"

    short_ledger = ha001.closure_summary(
        SEED,
        _closure(["SUCCEEDED"] * 3, [executor] * 3, [False] * 3),
        {**LEDGER, "physical_count": 2},
    )
    short_errors = ha001.closure_errors(SEED, short_ledger, executor_id=executor)
    assert any("ledger physical count" in item for item in short_errors), (
        f"a ledger short of the command count must be an error: {short_errors}"
    )


def test_cleanup_steps_run_every_step_and_collect_errors(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def gpu(*args: str, **_kwargs: object) -> str:
        calls.append("gpu " + " ".join(args[:2]))
        if args[0] == "exec":
            raise ha001.CaseError("kubectl exec timed out")
        return ""

    def cpu(*args: str, **_kwargs: object) -> str:
        calls.append("cpu " + " ".join(args[:3]))
        return ""

    monkeypatch.setattr(ha001, "gpu", gpu)
    monkeypatch.setattr(ha001, "cpu", cpu)
    monkeypatch.setattr(
        ha001,
        "cleanup_closure",
        lambda seed: calls.append("cleanup_closure") or {"deleted": {}},
    )
    monkeypatch.setattr(ha001, "database_residuals", lambda: {"objects": 0, "links": 0})
    monkeypatch.setattr(ha001, "probe_resources", lambda: {"count": 0, "resources": {}})
    monkeypatch.setattr(
        ha001,
        "ready_pods",
        lambda app: [{"name": "p"}] * (3 if app == ha001.INGRESS_APP else 6),
    )
    result: dict = {}

    errors = ha001.cleanup_steps(
        probe_created=True,
        seed={"command_ids": ["a"]},
        case_dir=tmp_path,
        result=result,
        expected_replicas={ha001.INGRESS_APP: 3, ha001.WORKER_APP: 6},
    )

    assert errors == ["stop probe: CaseError: kubectl exec timed out"], errors
    assert "cleanup_closure" in calls, (
        "a failing kubectl step must not skip the store cleanup"
    )
    assert "closure_cleanup" in result and "postflight" in result
    assert (tmp_path / "postflight.json").is_file(), (
        "the postflight record must be written even after a failed cleanup step"
    )


def test_execute_case_refuses_a_plan_without_derived_fields(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / ha001.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text('{"attempt": 1, "targets": []}')
    with pytest.raises(ha001.CaseError, match="re-plan"):
        ha001.execute_case(tmp_path, 1, ha001.CONFIRMATION)
