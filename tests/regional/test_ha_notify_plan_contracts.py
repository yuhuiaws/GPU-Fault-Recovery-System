from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.e2e.regional import ha_evidence
from scripts.e2e.regional import live_driver_guard as guard
from scripts.e2e.regional import regional_case_contract as contract
from scripts.e2e.regional import run_ha001_control_plane_failover as ha001
from scripts.e2e.regional import run_ha002_pdb_topology as ha002
from scripts.e2e.regional import run_ha005_rollout_continuity as ha005
from scripts.e2e.regional import run_ha006_executor_takeover as ha006
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from scripts.e2e.regional import run_notification_acceptance as notify
from scripts.e2e.regional import run_notify007_delivery_states as notify007


@pytest.fixture
def isolated_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(guard, "source_digest", lambda: "unit-source")
    monkeypatch.setattr(guard, "applied_site_profile", lambda: None)
    monkeypatch.setattr(notify, "install_abort_signals", lambda: None)
    monkeypatch.setattr(notify007, "install_abort_signals", lambda: None)
    for key in guard.COMMON_ENVIRONMENT_KEYS:
        monkeypatch.setenv(key, f"unit-{key.lower()}")
    monkeypatch.setattr(ha_evidence, "settings_from_arguments", lambda args: args)
    identity = {"release_id": "unit-release", "cluster_id": "unit-cluster"}
    monkeypatch.setattr(
        ha_evidence,
        "RegionalLiveFixture",
        lambda _: SimpleNamespace(evidence_identity=lambda: identity),
    )
    # Exercise the parent's accepted HA-005 restoration without editing its catalog.
    order = yaml.safe_load(contract.ORDER_PATH.read_text())
    for phase in order["phases"]:
        if phase["sequence"] == 12:
            entries = phase["entries"]
            if not any(entry.get("case") == ha005.CASE_ID for entry in entries):
                position = next(
                    i
                    for i, entry in enumerate(entries)
                    if entry.get("case") == ha002.CASE_ID
                )
                entries.insert(position + 1, {"case": ha005.CASE_ID})
    order["do_not_run"] = [
        entry for entry in order["do_not_run"] if entry["case"] != ha005.CASE_ID
    ]
    order_path = tmp_path / "order.yaml"
    order_path.write_text(yaml.safe_dump(order))
    monkeypatch.setattr(contract, "ORDER_PATH", order_path)
    contract.expanded_order.cache_clear()
    for module in (ha001, ha002, ha005, ha006, ha009):
        previous = contract.formal_predecessor(module.CASE_ID)
        if previous:
            path = contract.case_evidence_path(tmp_path, previous)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"case_id": previous, "verdict": "PASS", **identity})
            )
    yield identity
    contract.expanded_order.cache_clear()


def execution_args(
    plan_args: argparse.Namespace, confirmation: str
) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            **vars(plan_args),
            "execute": True,
            "plan": False,
            "confirm": confirmation,
            "maintenance_window_end": "2099-01-01T00:00:00Z",
        }
    )


def install_cpu_observations(
    monkeypatch: pytest.MonkeyPatch, *, failed: bool = False
) -> None:
    replicas = {ha001.INGRESS_APP: 3, ha001.WORKER_APP: 6, ha001.SPOOL_APP: 0}
    pods = {
        app: [
            {"name": f"{app}-{i}", "uid": f"uid-{app}-{i}", "node": f"node-{i % 3}"}
            for i in range(count)
        ]
        for app, count in replicas.items()
    }
    roles = {}
    for app, role in ((ha001.INGRESS_APP, "ingress"), (ha001.WORKER_APP, "worker")):
        roles[app] = [
            {
                **pod,
                "health": {
                    "service_role": role,
                    "processor_mode": "active-active",
                    "processor_epoch": "",
                    "processor_role": "inactive"
                    if role == "ingress"
                    else "active-consumer",
                },
                "processor_active_consumer": 0.0 if role == "ingress" else 4.0,
            }
            for pod in pods[app]
        ]
    if failed:
        roles[ha001.WORKER_APP][0]["health"]["processor_mode"] = "direct"
    monkeypatch.setattr(ha001, "ready_pods", lambda app: pods[app])
    monkeypatch.setattr(ha001, "role_snapshot", lambda: roles)
    monkeypatch.setattr(ha001, "queue_stats", lambda: {"depth": 0})
    monkeypatch.setattr(ha001, "environment_values", lambda: {})
    monkeypatch.setattr(
        ha001,
        "deployment_and_pdb_snapshot",
        lambda: {
            "deployments": {
                app: {
                    "replicas": count,
                    "pre_stop_sleep_seconds": 10,
                    "graceful_shutdown_seconds": 30,
                }
                for app, count in replicas.items()
            },
            "pdbs": {},
        },
    )
    node_document = {
        "items": [
            {
                "metadata": {"name": f"node-{i}", "uid": f"node-uid-{i}"},
                "spec": {},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            }
            for i in range(3)
        ]
    }
    pod_document = {
        "items": [
            {
                "metadata": {
                    "name": pod["name"],
                    "uid": pod["uid"],
                    "labels": {"app": app},
                },
                "spec": {"nodeName": pod["node"], "containers": [{"name": "main"}]},
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [{"name": "main", "ready": True}],
                },
            }
            for app, records in pods.items()
            for pod in records
        ]
    }
    monkeypatch.setattr(
        ha001,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess([], 0, json.dumps(node_document)),
    )
    monkeypatch.setattr(ha001, "cpu", lambda *a, **kw: json.dumps(pod_document))


@pytest.mark.parametrize("module", [ha001, ha002])
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize(
    "profile", [None, {"path": "/private/test-site.yaml", "sha256": "a" * 64}]
)
def test_local_ha_builder_uses_the_shared_schema_and_real_preflight_result(
    module,
    failed: bool,
    profile,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    isolated_guard,
) -> None:
    install_cpu_observations(monkeypatch, failed=failed)
    monkeypatch.setattr(guard, "applied_site_profile", lambda: profile)
    monkeypatch.setattr(sys, "argv", [module.__file__])
    args = argparse.Namespace(
        run_dir=tmp_path,
        attempt=1,
        plan=True,
        execute=False,
        confirm="",
        maintenance_window_end="",
        gpu_context="unit-context",
    )
    plan = module.build_plan(tmp_path, 1, arguments=args)
    assert plan["schema_version"] == 3
    assert plan["source_digest"] == "unit-source"
    assert plan["site_profile"] == profile, (
        "custom HA plans must preserve the actual site profile binding"
    )
    assert plan["arguments_sha256"]
    assert plan["preflight_passed"] is (not failed)
    execute = execution_args(args, module.CONFIRMATION)
    if failed:
        with pytest.raises(RuntimeError, match="preflight"):
            guard.authorize_execution(
                execute,
                case_id=module.CASE_ID,
                confirmation=module.CONFIRMATION,
                environment={},
            )
    else:
        assert guard.authorize_execution(
            execute,
            case_id=module.CASE_ID,
            confirmation=module.CONFIRMATION,
            environment={},
        ) > datetime.now(timezone.utc)
        monkeypatch.setattr(
            guard, "applied_site_profile", lambda: {"path": "/different-site.yaml"}
        )
        with pytest.raises(RuntimeError, match="site_profile"):
            guard.authorize_execution(
                execute,
                case_id=module.CASE_ID,
                confirmation=module.CONFIRMATION,
                environment={},
            )
        monkeypatch.setattr(guard, "applied_site_profile", lambda: profile)
        execute.gpu_context = "changed-context"
        with pytest.raises(RuntimeError, match="arguments_sha256"):
            guard.authorize_execution(
                execute,
                case_id=module.CASE_ID,
                confirmation=module.CONFIRMATION,
                environment={},
            )


@pytest.mark.parametrize("module", [ha005, ha006, ha009])
@pytest.mark.parametrize("failed", [False, True])
def test_direct_ha_main_binds_a_passing_preflight_before_execution(
    module,
    failed: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    isolated_guard,
) -> None:
    monkeypatch.setattr(module, "install_site_profile", lambda: None)
    base = module.BASE if module is ha009 else module
    monkeypatch.setattr(
        base, "database_residuals", lambda: {"total": 1 if failed else 0}
    )
    monkeypatch.setattr(base, "registry_residuals", lambda: {"count": 0})
    monkeypatch.setattr(base, "kubernetes_residuals", lambda: {"count": 0})
    if module is ha009:
        for name in ("RDS_CLUSTER_ID", "AWS_REGION", "CRONJOB", "SECRET_NAME"):
            monkeypatch.setattr(module, name, getattr(module, name))
        monkeypatch.setattr(
            module,
            "aurora_guard",
            lambda: SimpleNamespace(read=lambda: {"unit": "proof"}),
        )
    calls = []
    monkeypatch.setattr(module, "run_case", lambda *a, **kw: calls.append((a, kw)) or 0)
    argv = [module.__file__, "--run-dir", str(tmp_path)]
    if module is ha005:
        argv.append("--all-deployments")
    if module is ha009:
        argv += ["--rds-cluster-id", "unit-aurora"]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == int(failed)
    plan = json.loads((tmp_path / "cases" / module.CASE_ID / "plan.json").read_text())
    assert plan["schema_version"] == 3
    assert plan["preflight_passed"] is (not failed)
    if module is ha005:
        assert plan["details"]["chain"]["predecessor"]["case_id"] == ha002.CASE_ID
        assert contract.formal_predecessor(ha006.CASE_ID) == ha005.CASE_ID
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *argv,
            "--execute",
            "--confirm",
            module.CONFIRMATION,
            "--maintenance-window-end",
            "2099-01-01T00:00:00Z",
        ],
    )
    if failed:
        with pytest.raises(RuntimeError, match="preflight"):
            module.main()
        assert calls == []
    else:
        assert module.main() == 0
        assert len(calls) == 1
        plan["schema_version"] = 2
        (tmp_path / "cases" / module.CASE_ID / "plan.json").write_text(json.dumps(plan))
        with pytest.raises(RuntimeError, match="schema_version"):
            module.main()
        assert len(calls) == 1


@pytest.mark.parametrize("module", [ha001, ha002, ha005, ha006, ha009])
def test_chain_refuses_failed_or_foreign_predecessors(
    module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_guard
) -> None:
    args = argparse.Namespace(run_dir=tmp_path)
    planned = ha_evidence.chain_preflight(args, module.CASE_ID)
    assert planned["errors"] == []
    path = Path(planned["predecessor"]["path"])
    for defect in (
        {"verdict": "FAIL"},
        {"status": "FAILED"},
        {"formal_sequence_satisfied": "false"},
        {"release_id": "other-release"},
        {"cluster_id": "other-cluster"},
    ):
        path.write_text(
            json.dumps(
                {
                    "case_id": planned["predecessor"]["case_id"],
                    "verdict": "PASS",
                    **isolated_guard,
                    **defect,
                }
            )
        )
        current = ha_evidence.chain_preflight(args, module.CASE_ID)
        assert current["errors"]
        with pytest.raises(Exception, match="predecessor"):
            ha_evidence.require_chain(planned, current)


def test_notification_main_records_identity_and_binds_failed_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_guard
) -> None:
    identity = {"release_id": "unit-release", "cluster_id": "unit-cluster"}
    target = SimpleNamespace(cluster_id="unit-cluster", context="unit-context")
    regional = SimpleNamespace(evidence_identity=lambda: identity)
    ready = {"pods": ["unit-pod"]}
    site_path = tmp_path / "site"
    cpu_kubeconfig = tmp_path / "cpu.kubeconfig"
    gpu_kubeconfig = tmp_path / "gpu.kubeconfig"
    for path in (site_path, cpu_kubeconfig, gpu_kubeconfig):
        path.write_text("unit-only-input\n", encoding="utf-8")
    site = SimpleNamespace(
        cpu_kubeconfig=cpu_kubeconfig,
        gpu_kubeconfig=gpu_kubeconfig,
        target=lambda _: target,
        regional=lambda _: regional,
        ready_pods=lambda *a: ready["pods"],
        pod_json=lambda *a, **kw: {"result": "DENIED", "code": "AccessDenied"},
    )
    monkeypatch.setattr(notify, "install_site_profile", lambda: None)
    monkeypatch.setattr(notify, "IdentitySite", lambda _: site)
    monkeypatch.setattr(
        notify, "predecessor_path", lambda *a: ("prev", tmp_path / "prev.json")
    )
    seen = []
    monkeypatch.setattr(
        notify,
        "predecessor_evidence",
        lambda *a, **kw: (seen.append(kw) or {"valid": True}),
    )
    argv = [
        notify.__file__,
        "--run-dir",
        str(tmp_path),
        "--site",
        str(site_path),
        "--case",
        "GF-REGIONAL-NOTIFY-004",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert notify.main() == 0
    planned = json.loads(
        (tmp_path / "cases/GF-REGIONAL-NOTIFY-004/plan.json").read_text()
    )
    assert planned["environment"]["GPU_FAULT_CONTROL_KUBECONFIG"] == str(cpu_kubeconfig)
    assert planned["environment"]["KUBECONFIG"] == str(gpu_kubeconfig)
    assert planned["connections"], "the plan must bind the fixture kubeconfig files"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *argv,
            "--execute",
            "--confirm",
            "NOTIFY004_EXECUTE",
            "--maintenance-window-end",
            "2099-01-01T00:00:00Z",
        ],
    )
    assert notify.main() == 0
    result = json.loads(
        (
            tmp_path / "cases/GF-REGIONAL-NOTIFY-004/GF-REGIONAL-NOTIFY-004.json"
        ).read_text()
    )
    assert all(item == identity for item in seen), {
        "expected_identity": identity,
        "observed": seen,
    }
    assert {key: result[key] for key in identity} == identity
    ready["pods"] = []
    monkeypatch.setattr(sys, "argv", argv)
    assert notify.main() == 1
    assert not json.loads(
        (tmp_path / "cases/GF-REGIONAL-NOTIFY-004/plan.json").read_text()
    )["preflight_passed"]


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"gpu-fault-api-ha/api/uid": ""},
        {"gpu-fault-control-worker/worker/uid": "", "unknown/pod/uid": ""},
        {"gpu-fault-control-worker/worker/uid": ""},
    ],
    ids=["empty", "missing-worker", "unknown-role", "missing-worker-families"],
)
def test_notify007_empty_metric_plan_cannot_execute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_guard, metrics: dict
) -> None:
    monkeypatch.setattr(notify007, "install_site_profile", lambda: None)
    monkeypatch.setattr(
        notify007,
        "settings_from_arguments",
        lambda _: SimpleNamespace(environment=lambda: {}),
    )
    monkeypatch.setattr(
        notify007,
        "RegionalLiveFixture",
        lambda _: SimpleNamespace(
            evidence_identity=lambda: {"release_id": "r", "cluster_id": "c"}
        ),
    )
    monkeypatch.setattr(
        notify007, "predecessor_path", lambda *a: ("prev", tmp_path / "prev")
    )
    monkeypatch.setattr(
        notify007, "predecessor_evidence", lambda *a, **kw: {"valid": True}
    )
    monkeypatch.setattr(notify007, "control_plane_metrics", lambda _: metrics)
    calls = []
    monkeypatch.setattr(notify007, "execute", lambda *a, **kw: calls.append(True))
    argv = [notify007.__file__, "--run-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    assert notify007.main() == 1, "incomplete metric evidence must fail planning"
    plan = json.loads(
        (tmp_path / "cases" / notify007.CASE_ID / "plan.json").read_text()
    )
    assert plan["preflight_passed"] is False, (
        "role and metric validation failures must be persisted in the plan"
    )
    assert plan["details"]["preflight_errors"], (
        "a failed plan must retain the metric validation reason"
    )
    assert plan["mutation_performed"] is False, (
        "failed planning must never run a notification drill"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *argv,
            "--execute",
            "--confirm",
            notify007.CONFIRMATION,
            "--maintenance-window-end",
            "2099-01-01T00:00:00Z",
        ],
    )
    with pytest.raises(RuntimeError, match="preflight"):
        notify007.main()
    assert calls == [], "failed preflight must stop execution before the drill"
