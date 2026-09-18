from __future__ import annotations

import argparse
import asyncio
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app.routes.admin import AdminRouterDependencies, healthz
from scripts.e2e.regional import live_driver_guard as guard
from scripts.e2e.regional import run_ha001_control_plane_failover as ha001
from scripts.e2e.regional import run_ha002_pdb_topology as ha002
from tests.regional.test_ha_notify_plan_contracts import (
    execution_args,
    install_cpu_observations,
)
from tests.regional.test_ha_notify_plan_contracts import (
    isolated_guard as isolated_guard,
)


def role_health(role: str) -> dict[str, Any]:
    processor = SimpleNamespace(
        leadership=None,
        active_consumers=role == "worker",
        is_healthy=lambda: True,
        unhealthy_reason=None,
    )
    context = SimpleNamespace(
        executor_mode="regional",
        dispatcher=SimpleNamespace(config=SimpleNamespace(enabled=True)),
        fleet_registry=None,
    )
    payload = asyncio.run(
        healthz(
            AdminRouterDependencies(
                context=context,
                processor=processor,
                processor_mode="active-active",
                service_role=role,
                environment={"GPU_FAULT_SERVICE_ROLE": role},
                regional_registry_runtime=None,
            )
        )
    )
    assert isinstance(payload, dict), "the real healthy endpoint must return an object"
    return payload


def role_snapshot() -> dict[str, Any]:
    return {
        app: [
            {
                "name": f"{role}-{index}",
                "health": role_health(role),
                "processor_active_consumer": 0.0 if role == "ingress" else 4.0,
            }
            for index in range(count)
        ]
        for app, role, count in (
            (ha001.INGRESS_APP, "ingress", 1),
            (ha001.WORKER_APP, "worker", 2),
        )
    }


REPLICAS = {ha001.INGRESS_APP: 1, ha001.WORKER_APP: 2}


@pytest.mark.parametrize("consumers", [1, 1.0, 4, 4.0, 8.0])
def test_actual_healthz_and_summed_consumer_gauge_satisfy_the_role_contract(
    consumers: int | float,
) -> None:
    snapshot = role_snapshot()
    snapshot[ha001.WORKER_APP][0]["processor_active_consumer"] = consumers
    assert "leadership" not in snapshot[ha001.WORKER_APP][0]["health"]
    assert ha001.validate_roles(snapshot, REPLICAS) == []


def test_script_style_runner_import_keeps_the_current_health_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(ha001.__file__).parent))
    imported = runpy.run_path(ha001.__file__, run_name="ha001-import-test")
    snapshot = role_snapshot()
    assert imported["validate_roles"](snapshot, REPLICAS) == []
    snapshot[ha001.WORKER_APP][0]["health"].pop("processor_epoch")
    assert imported["validate_roles"](snapshot, REPLICAS), (
        "script-style imports must still reject a missing processor epoch"
    )


@pytest.mark.parametrize(
    "consumers",
    [None, True, False, "4", [], {}, float("nan"), float("inf"), -1, 0, 0.5, 1.5],
)
def test_malformed_or_idle_consumer_metrics_cannot_prove_healthy_workers(
    consumers: Any,
) -> None:
    snapshot = role_snapshot()
    snapshot[ha001.WORKER_APP][0]["processor_active_consumer"] = consumers
    errors = ha001.validate_roles(snapshot, REPLICAS)
    assert any("active-consumer metric" in error for error in errors), errors


@pytest.mark.parametrize("epoch", [None, False, 0, [], {}, "7", " "])
def test_an_empty_string_is_the_only_leaderless_epoch(epoch: Any) -> None:
    snapshot = role_snapshot()
    snapshot[ha001.WORKER_APP][0]["health"]["processor_epoch"] = epoch
    assert ha001.validate_roles(snapshot, REPLICAS), (
        "only an explicitly empty string proves a leaderless processor"
    )


@pytest.mark.parametrize("health", [None, [], "ok"])
def test_malformed_health_objects_cannot_prove_a_role(health: Any) -> None:
    snapshot = role_snapshot()
    snapshot[ha001.WORKER_APP][0]["health"] = health
    assert any(
        "healthz" in error for error in ha001.validate_roles(snapshot, REPLICAS)
    ), "a malformed health response must not prove a healthy role"


@pytest.mark.parametrize("duplicate", [False, True])
def test_replica_count_cannot_be_replaced_by_an_aggregate_consumer_count(
    duplicate: bool,
) -> None:
    snapshot = role_snapshot()
    worker = snapshot[ha001.WORKER_APP][0]
    worker["processor_active_consumer"] = 64.0
    snapshot[ha001.WORKER_APP] = [worker, worker] if duplicate else [worker]
    assert ha001.validate_roles(snapshot, REPLICAS), (
        "consumer counts cannot replace a complete unique Pod population"
    )


def plan_arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=tmp_path,
        attempt=1,
        plan=True,
        execute=False,
        confirm="",
        maintenance_window_end="",
        gpu_context="unit-context",
    )


def install_current_health(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    install_cpu_observations(monkeypatch)
    snapshot = ha001.role_snapshot()
    for app, role in ((ha001.INGRESS_APP, "ingress"), (ha001.WORKER_APP, "worker")):
        for pod in snapshot[app]:
            pod["health"] = role_health(role)
            pod["processor_active_consumer"] = 0.0 if role == "ingress" else 4.0
    return snapshot


@pytest.mark.parametrize("module", [ha001, ha002])
@pytest.mark.usefixtures("isolated_guard")
def test_integrated_plans_bind_the_full_details_with_the_stronger_shared_guard(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_current_health(monkeypatch)
    monkeypatch.setattr(sys, "argv", [module.__file__])
    arguments = plan_arguments(tmp_path)
    plan = module.build_plan(tmp_path, 1, arguments=arguments)
    details = plan["details"]
    assert plan["schema_version"] == guard.PLAN_SCHEMA_VERSION == 3
    assert plan["preflight_passed"] is True
    assert plan["details_sha256"] == guard.details_sha256(details)
    assert plan["arguments_sha256"] and plan["source_digest"] and "connections" in plan
    assert details["mutation"] and details["maintenance_window_required_at_execute"]
    assert details["risk"] == (
        "live-control-plane-pod-deletion"
        if module is ha001
        else "live-control-plane-pdb-eviction"
    )
    assert all(
        isinstance(limit, dict) and limit["limit_seconds"] > 0
        for limit in details["failure_window_limits"].values()
    ), "keep the full locally derived limit evidence inside the signed details"
    target = details["targets"][0] if module is ha001 else details["target_node"]
    assert target["uid"]
    execute = execution_args(arguments, module.CONFIRMATION)
    assert guard.authorize_execution(
        execute,
        case_id=module.CASE_ID,
        confirmation=module.CONFIRMATION,
        environment={},
    ) > datetime.now(timezone.utc)

    target["uid"] = "different-object"
    path = tmp_path / "cases" / module.CASE_ID / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match="details_sha256"):
        guard.authorize_execution(
            execute,
            case_id=module.CASE_ID,
            confirmation=module.CONFIRMATION,
            environment={},
        )
    plan["details_sha256"] = guard.details_sha256(details)
    plan["schema_version"] = 2
    path.write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match="schema_version"):
        guard.authorize_execution(
            execute,
            case_id=module.CASE_ID,
            confirmation=module.CONFIRMATION,
            environment={},
        )


@pytest.mark.parametrize("module", [ha001, ha002])
@pytest.mark.parametrize("defect", ["missing-epoch", "under-replicated", "nonfinite"])
@pytest.mark.usefixtures("isolated_guard")
def test_plans_with_incomplete_health_cannot_authorize_execution(
    module: ModuleType, defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = install_current_health(monkeypatch)
    if defect == "missing-epoch":
        snapshot[ha001.WORKER_APP][0]["health"].pop("processor_epoch")
    elif defect == "under-replicated":
        snapshot[ha001.WORKER_APP].pop()
    else:
        snapshot[ha001.WORKER_APP][0]["processor_active_consumer"] = float("nan")
    monkeypatch.setattr(sys, "argv", [module.__file__])
    arguments = plan_arguments(tmp_path)
    plan = module.build_plan(tmp_path, 1, arguments=arguments)
    assert plan["preflight_passed"] is False
    assert plan["details"]["errors"]
    with pytest.raises(RuntimeError, match="preflight"):
        guard.authorize_execution(
            execution_args(arguments, module.CONFIRMATION),
            case_id=module.CASE_ID,
            confirmation=module.CONFIRMATION,
            environment={},
        )
