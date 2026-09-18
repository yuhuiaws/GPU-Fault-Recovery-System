from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_ha001_control_plane_failover as ha
from tests.regional._cov95_ha001_harness import CHAIN, DEADLINE, Clock, HA001Harness


@pytest.mark.parametrize("fallback", [False, True])
def test_configure_and_transport_keep_cpu_gpu_scope_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fallback: bool
) -> None:
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    cpu.write_text("apiVersion: v1\n")
    gpu.write_text("apiVersion: v1\n")
    for name in (
        "CPU_KUBECONFIG",
        "GPU_KUBECONFIG",
        "GPU_CONTEXT",
        "AWS_REGION",
        "NAMESPACE",
    ):
        monkeypatch.setattr(ha, name, getattr(ha, name))
    arguments = argparse.Namespace(
        cpu_kubeconfig=str(cpu),
        gpu_kubeconfig=str(gpu),
        gpu_context="gpu-unit",
        region="unit-region",
        namespace="unit-ns",
    )
    if fallback:
        for field, key in (
            ("cpu_kubeconfig", "CPU_KUBECONFIG"),
            ("gpu_kubeconfig", "GPU_KUBECONFIG"),
            ("gpu_context", "GPU_EKS_CONTEXT"),
            ("region", "AWS_REGION"),
        ):
            monkeypatch.setenv(key, getattr(arguments, field))
            setattr(arguments, field, "")
    ha.configure(arguments)
    calls = []

    def transport(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "unit-result", "")

    monkeypatch.setattr(ha, "run_fixture_command", transport)
    assert ha.cpu("get", "pods", timeout=17) == "unit-result"
    assert ha.gpu("get", "pods", check=False, timeout=23) == "unit-result"
    assert calls[0][0] == [
        "kubectl",
        "--kubeconfig",
        str(cpu),
        "-n",
        "unit-ns",
        "get",
        "pods",
    ]
    assert calls[1][0] == [
        "kubectl",
        "--kubeconfig",
        str(gpu),
        "--context",
        "gpu-unit",
        "-n",
        "unit-ns",
        "get",
        "pods",
    ]
    assert calls[0][1] == {"input_text": None, "check": True, "timeout": 17}
    assert calls[1][1] == {"input_text": None, "check": False, "timeout": 23}
    assert ha.environment_values()["GPU_EKS_CONTEXT"] == "gpu-unit"


@pytest.mark.parametrize("missing", ["inputs", "file"])
def test_configure_rejects_incomplete_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    for name in ("CPU_KUBECONFIG", "GPU_KUBECONFIG"):
        monkeypatch.setattr(ha, name, getattr(ha, name))
    args = argparse.Namespace(
        cpu_kubeconfig="",
        gpu_kubeconfig="",
        gpu_context="",
        region="",
        namespace="unit",
    )
    if missing == "file":
        args.cpu_kubeconfig = args.gpu_kubeconfig = str(tmp_path / "absent")
        args.gpu_context, args.region = "unit", "unit-region"
    with pytest.raises(
        ha.CaseError, match="does not exist" if missing == "file" else "required"
    ):
        ha.configure(args)


def test_store_reads_retry_but_cleanup_mutations_get_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    clock = Clock()
    monkeypatch.setattr(ha, "time", clock)
    monkeypatch.setattr(ha, "first_ready_cpu_pod", lambda _: "unit-ingress")

    def cpu(*args: str, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return "not-json" if len(calls) == 1 else '{"observed":true}'

    monkeypatch.setattr(ha, "cpu", cpu)
    assert ha.cpu_python("print('{}')") == {"observed": True}
    assert len(calls) == 2
    assert clock.now == 1
    with pytest.raises(ValueError, match="at least 1"):
        ha.cpu_python("", attempts=0)

    def probe(script: str, *args: str, **kwargs: Any) -> dict[str, int]:
        assert kwargs["attempts"] == 1
        assert list(args) == [
            seed["incident_id"],
            seed["event_id"],
            seed["workflow_id"],
            *seed["command_ids"],
        ]
        return {"remaining_objects": 0, "remaining_links": 0}

    seed = ha.closure_seed("ha001-unit")
    monkeypatch.setattr(ha, "cpu_python", probe)
    assert ha.cleanup_closure(seed) == {"remaining_objects": 0, "remaining_links": 0}


@pytest.mark.parametrize("remaining", [(1, 0), (0, 1)])
def test_cleanup_residuals_fail_instead_of_claiming_success(
    monkeypatch: pytest.MonkeyPatch, remaining: tuple[int, int]
) -> None:
    monkeypatch.setattr(
        ha,
        "cpu_python",
        lambda *a, **kw: {
            "remaining_objects": remaining[0],
            "remaining_links": remaining[1],
        },
    )
    with pytest.raises(ha.CaseError, match="cleanup left residuals"):
        ha.cleanup_closure(ha.closure_seed("ha001-unit"))


@pytest.mark.parametrize(
    "app,field,value,error",
    [
        (ha.INGRESS_APP, "service_role", "worker", "is not ingress"),
        (ha.INGRESS_APP, "processor_role", "active-consumer", "is not inactive"),
        (ha.INGRESS_APP, "processor_active_consumer", 1, "not zero"),
        (ha.WORKER_APP, "service_role", "ingress", "is not worker"),
        (ha.WORKER_APP, "processor_role", "inactive", "not active-consumer"),
        (ha.WORKER_APP, "processor_active_consumer", 0, "expected >= 1"),
        (ha.WORKER_APP, "processor_epoch", "7", "holds processor leadership epoch"),
        (ha.WORKER_APP, "processor_epoch", None, "omits processor_epoch"),
        (ha.INGRESS_APP, "processor_epoch", None, "omits processor_epoch"),
        (ha.WORKER_APP, "processor_active_consumer", float("nan"), "expected >= 1"),
    ],
)
def test_role_contract_rejects_misassigned_replicas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    app: str,
    field: str,
    value: Any,
    error: str,
) -> None:
    harness = HA001Harness(monkeypatch, tmp_path)
    roles = ha.role_snapshot()
    row = roles[app][0]
    (row if field == "processor_active_consumer" else row["health"])[field] = value
    errors = ha.validate_roles(roles, harness.replicas)
    assert len(errors) == 1
    assert error in errors[0]
    roles[ha.INGRESS_APP] = []
    assert "ingress role snapshot does not contain 3 Pods" in ha.validate_roles(
        roles, harness.replicas
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("min_ingress_ready", 1, "lost ingress quorum"),
        ("min_worker_ready", 1, "lost worker quorum"),
        ("min_endpoint_ready", 1, "lost NLB endpoints"),
        ("final_queue_depth", 6, "queue did not recover"),
        ("max_probe_failure_window_seconds", 111, "failure window exceeded"),
        ("settle_reached", False, "did not settle"),
    ],
)
def test_verdict_rechecks_observed_phase_failures_and_still_cleans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: Any, error: str
) -> None:
    harness = HA001Harness(monkeypatch, tmp_path)
    observe = ha.observe_phase

    def failed_observation(label: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = observe(label, *args, **kwargs)
        if label == "worker-1":
            (result if field == "settle_reached" else result["summary"])[field] = value
        return result

    monkeypatch.setattr(ha, "observe_phase", failed_observation)
    code, report = harness.execute()
    assert code == 1
    assert any(error in entry for entry in report["errors"]), report["errors"]
    assert harness.cleaned is True
    assert report["cleanup_errors"] == []


@pytest.mark.parametrize("succeeds", [False, True])
def test_probe_file_wait_is_bounded(
    monkeypatch: pytest.MonkeyPatch, succeeds: bool
) -> None:
    clock = Clock()
    responses = iter(["", "present" if succeeds else "", ""])
    monkeypatch.setattr(ha, "time", clock)
    monkeypatch.setattr(ha, "gpu", lambda *a, **kw: next(responses))
    if succeeds:
        assert ha.wait_probe_file("/state/ready.json", 2) is None
        assert clock.now == 1
    else:
        with pytest.raises(ha.CaseError, match="did not create"):
            ha.wait_probe_file("/state/ready.json", 2)
        assert clock.now == 2


@pytest.mark.parametrize("succeeds", [False, True])
def test_closure_wait_observes_nonterminal_records_before_timeout_or_success(
    monkeypatch: pytest.MonkeyPatch, succeeds: bool
) -> None:
    clock = Clock()
    responses = iter(["LEASED", "SUCCEEDED" if succeeds else "WAITING"])
    monkeypatch.setattr(ha, "time", clock)
    monkeypatch.setattr(
        ha, "closure_status", lambda _: {"commands": [{"status": next(responses)}]}
    )
    if succeeds:
        assert ha.wait_closure({}, 4)["commands"][0]["status"] == "SUCCEEDED"
    else:
        with pytest.raises(ha.CaseError, match="did not complete"):
            ha.wait_closure({}, 4)
    assert clock.now == (2 if succeeds else 4)


def test_closure_verdict_requires_owner_and_ordered_ledger() -> None:
    seed = {"command_ids": ["unit"], "operations": ["FREEZE_EVIDENCE"]}
    closure = {
        "commands": [
            {
                "command_id": "unit",
                "status": "SUCCEEDED",
                "step_index": 0,
                "result_details": {"cached": False},
            }
        ]
    }
    summary = ha.closure_summary(seed, closure, {"physical_count": 1, "operations": []})
    assert ha.closure_errors(seed, summary, executor_id="unit") == [
        "synthetic command unit has no lease owner",
        "synthetic closure operations are incomplete or reordered",
    ]


def test_execute_main_uses_full_fake_lifecycle_without_host_or_cloud_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA001Harness(monkeypatch, tmp_path)
    chain = {**copy.deepcopy(CHAIN), "errors": []}
    harness.plan["chain"] = chain
    plan_path = tmp_path / "cases" / ha.CASE_ID / "plan.json"
    plan_path.write_text(json.dumps({"attempt": 1, "details": harness.plan}))
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    cpu.write_text("apiVersion: v1\n")
    gpu.write_text("apiVersion: v1\n")
    for name in (
        "CPU_KUBECONFIG",
        "GPU_KUBECONFIG",
        "GPU_CONTEXT",
        "AWS_REGION",
        "NAMESPACE",
    ):
        monkeypatch.setattr(ha, name, getattr(ha, name))
    monkeypatch.setattr(ha, "install_site_profile", lambda: None)
    monkeypatch.setattr(ha, "install_abort_signals", lambda: None)
    monkeypatch.setattr(
        ha, "os", SimpleNamespace(**{**vars(ha.os), "umask": lambda _: None})
    )
    monkeypatch.setattr(ha, "guard_authorize_execution", lambda *a, **kw: DEADLINE)
    monkeypatch.setattr(ha, "chain_preflight", lambda *a: chain)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unit-ha001",
            "--execute",
            "--run-dir",
            str(tmp_path),
            "--confirm",
            ha.CONFIRMATION,
            "--cpu-kubeconfig",
            str(cpu),
            "--gpu-kubeconfig",
            str(gpu),
            "--gpu-context",
            "unit-gpu",
            "--region",
            "unit-region",
        ],
    )
    assert ha.main() == 0
    assert len(harness.deletions) == 3
    assert harness.resources == {}


def test_standalone_import_keeps_the_same_derived_limit_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(ha.__file__).parent))
    spec = importlib.util.spec_from_file_location("cov95_ha001_standalone", ha.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.failure_window_limit(
            {"pre_stop_sleep_seconds": 5, "graceful_shutdown_seconds": 10}
        )["limit_seconds"]
        == 45
    )


@pytest.mark.parametrize(
    "defect", ["confirmation", "missing-plan", "attempt", "window"]
)
def test_execute_rejects_invalid_envelopes_before_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    harness = HA001Harness(monkeypatch, tmp_path)
    confirmation, attempt, deadline = ha.CONFIRMATION, 1, DEADLINE
    if defect == "confirmation":
        confirmation = "wrong"
    elif defect == "attempt":
        attempt = 2
    elif defect == "window":
        deadline = None
    else:
        (tmp_path / "cases" / ha.CASE_ID / "plan.json").unlink()
    with pytest.raises(ha.CaseError):
        ha.execute_case(
            tmp_path, attempt, confirmation, maintenance_window_end=deadline
        )
    assert harness.events == []


def test_expired_window_blocks_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA001Harness(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="maintenance window"):
        ha.execute_case(
            tmp_path,
            1,
            ha.CONFIRMATION,
            maintenance_window_end=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
    assert harness.events == []


@pytest.mark.parametrize("defect", ["ingress", "worker", "spool", "queue", "probe"])
def test_observation_stop_conditions_are_enforced_during_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    harness = HA001Harness(monkeypatch, tmp_path)
    minimum = {"ingress": 2, "worker": 2, "spool": None}
    if defect in {"ingress", "worker"}:
        app = ha.INGRESS_APP if defect == "ingress" else ha.WORKER_APP
        for name in [f"{app}-0", f"{app}-1"]:
            del harness.pods[name]
    elif defect == "spool":
        minimum["spool"] = 1
    elif defect == "queue":
        monkeypatch.setattr(ha, "queue_stats", lambda: {"depth": 26})
    else:
        monkeypatch.setattr(
            ha, "read_probe", lambda: {"current_failure_window_seconds": 111}
        )
    with pytest.raises(
        ha.CaseError, match="availability|queue exceeded|failure window"
    ):
        ha.observe_phase(
            "unit", 30, 0, minimum_ready=minimum, max_failure_window_seconds=110
        )
    assert harness.clock.now == 0


@pytest.mark.parametrize("defect", ["baseline", "workers", "ingress"])
def test_plan_refuses_insufficient_ready_replicas_before_approval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    harness = HA001Harness(monkeypatch, tmp_path)
    monkeypatch.setattr(ha, "chain_preflight", lambda *a: {**CHAIN, "errors": []})
    if defect == "workers":
        harness.replicas[ha.WORKER_APP] = 1
        del harness.pods[f"{ha.WORKER_APP}-1"]
        del harness.pods[f"{ha.WORKER_APP}-2"]
    else:
        del harness.pods[f"{ha.INGRESS_APP}-2"]
        if defect == "ingress":
            harness.replicas[ha.INGRESS_APP] = 2
    with pytest.raises(ha.CaseError, match="baseline|fewer|at least three"):
        ha.build_plan(tmp_path, 1, arguments=argparse.Namespace())
    assert harness.events == []


@pytest.mark.parametrize("defect", ["unchanged", "short", "unidentified"])
def test_deletion_requires_a_distinct_replacement_at_declared_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    harness = HA001Harness(monkeypatch, tmp_path)
    target = harness.plan["targets"][0]
    names = set(harness.pods)
    if defect == "unchanged":
        transport = ha.cpu
        monkeypatch.setattr(
            ha, "cpu", lambda *a, **kw: "" if a[0] == "delete" else transport(*a, **kw)
        )
    elif defect == "short":
        harness.after_delete = lambda: harness.pods.pop(
            f"{ha.INGRESS_APP}-replacement-0"
        )
    else:
        names.add(f"{ha.INGRESS_APP}-replacement-0")
    with pytest.raises(ha.CaseError, match="still exists|desired Ready|no replacement"):
        ha.delete_and_observe(
            target,
            names,
            0,
            desired=3,
            minimum_ready={"ingress": 2, "worker": 2},
            max_failure_window_seconds=110,
        )


def test_missing_executor_identity_and_missing_ingress_are_not_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "env": [
                                {"name": "wanted", "value": None},
                                {"name": "other", "value": "unit"},
                            ]
                        }
                    ]
                }
            }
        }
    }
    with pytest.raises(ha.CaseError, match="no literal wanted"):
        ha.env_value(deployment, "wanted")
    monkeypatch.setattr(ha, "ready_pods", lambda _: [])
    with pytest.raises(ha.CaseError, match="no Ready CPU Pod"):
        ha.first_ready_cpu_pod(ha.INGRESS_APP)


@pytest.mark.parametrize("defect", ["foreign-probe", "roles", "seed-ack", "final-http"])
def test_late_or_foreign_evidence_cannot_override_a_failed_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    harness = HA001Harness(monkeypatch, tmp_path)
    if defect == "foreign-probe":
        harness.resources["Pod", ha.PROBE_POD] = {"metadata": {"uid": "foreign"}}
    elif defect == "roles":
        monkeypatch.setattr(ha, "role_snapshot", lambda: {})
    elif defect == "seed-ack":
        cpu = ha.cpu

        def changed_ack(*args: str, **kwargs: Any) -> str:
            output = cpu(*args, **kwargs)
            if args[0] == "exec" and ha.OWNER in args:
                return json.dumps({**json.loads(output), "workflow_id": "foreign"})
            return output

        monkeypatch.setattr(ha, "cpu", changed_ack)
    else:
        role_snapshot = ha.role_snapshot
        reads = []

        def roles() -> dict:
            reads.append(True)
            if len(reads) == 2:
                harness.probe_errors["http-401"] = 1
            return role_snapshot()

        monkeypatch.setattr(ha, "role_snapshot", roles)
    code, report = harness.execute()
    assert code == 1
    if defect == "foreign-probe":
        assert "probe resources already exist" in report["error"]
        assert "stop-probe" not in harness.events
    elif defect == "roles":
        assert "role preflight failed" in report["error"]
        assert harness.deletions == []
    elif defect == "seed-ack":
        assert "acknowledgement identity changed" in report["error"]
        assert harness.cleaned is True
    else:
        assert "probe observed a forbidden HTTP response" in report["errors"]
        assert harness.cleaned is True


@pytest.mark.parametrize("receipt", [False, True])
def test_cleanup_requires_receipts_and_the_declared_final_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, receipt: bool
) -> None:
    harness = HA001Harness(monkeypatch, tmp_path)
    expected = {**harness.replicas, ha.WORKER_APP: 4}
    report: dict[str, Any] = {}
    errors = ha.cleanup_steps(
        probe_created=not receipt,
        seed={},
        case_dir=tmp_path,
        result=report,
        expected_replicas=expected,
    )
    assert any("declared replicas" in error for error in errors), errors
    if not receipt:
        assert "probe resources have no ownership receipt; preserving them" in errors
    assert "stop-probe" not in harness.events


@pytest.mark.parametrize("passed", [False, True])
def test_plan_main_reports_real_preflight_result_without_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passed: bool
) -> None:
    monkeypatch.setattr(ha, "install_site_profile", lambda: None)
    monkeypatch.setattr(ha, "install_abort_signals", lambda: None)
    monkeypatch.setattr(ha, "configure", lambda args: None)
    monkeypatch.setattr(ha, "os", SimpleNamespace(umask=lambda _: None))
    monkeypatch.setattr(ha, "build_plan", lambda *a, **kw: {"preflight_passed": passed})
    monkeypatch.setattr(sys, "argv", ["unit", "--run-dir", str(tmp_path)])
    assert ha.main() == (0 if passed else 1)
