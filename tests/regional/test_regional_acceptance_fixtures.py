from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import (
    audit_regional_command_protocol_live as live_protocol_audit,
)
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from scripts.e2e.regional.audit_auth_boundary import validate_matrix
from scripts.e2e.regional.audit_executor_readiness import validate_readiness_matrix
from scripts.e2e.regional.audit_regional_command_protocol_live import (
    AUDITED_CASE_IDS,
    LiveProtocolAudit,
)
from scripts.e2e.regional.run_boot019_admin_lifecycle import run_admin_lifecycle
from scripts.e2e.regional.run_boot020_release_rolling import (
    configure_gpu_kubeconfig,
    run_release_rolling,
)
from scripts.e2e.regional.run_ha007_control_worker_shutdown import (
    _child_environment,
    run_probe,
)
from scripts.e2e.regional.run_ha008_processor_exit_acceptance import (
    run_acceptance as run_ha008_acceptance,
)

ROOT = Path(__file__).resolve().parents[2]
PROMOTED_MANUAL_DRIVERS = {
    "GF-REGIONAL-NET-002": "run_net002_command_recovery.py",
    "GF-REGIONAL-NET-003": "run_net003_result_retry.py",
    "GF-REGIONAL-HA-001": "run_ha001_control_plane_failover.py",
    "GF-REGIONAL-HA-002": "run_ha002_pdb_topology.py",
    "GF-REGIONAL-HA-003": "run_ha003_aurora_failover_reset.py",
    "GF-REGIONAL-HA-004": "run_ha004_waiting_reclaim_reset.py",
    "GF-REGIONAL-HA-005": "run_ha005_rollout_continuity.py",
    "GF-REGIONAL-HA-006": "run_ha006_executor_takeover.py",
    "GF-REGIONAL-DESTR-001": "run_destr001_gpu_reset.py",
    "GF-REGIONAL-DESTR-002": "run_destr002_hyperpod_reboot.py",
    "GF-REGIONAL-DESTR-003": "run_destr003_warm_spare_failover.py",
    "GF-REGIONAL-DESTR-008": "run_destr008_warm_spare_shortage.py",
    "GF-REGIONAL-DESTR-009": "run_destr009_workload_restart.py",
    "GF-REGIONAL-DESTR-012": "run_destr012_managed_recovery_guard.py",
    "GF-REGIONAL-DESTR-014": "run_destr014_branch_exhaustion.py",
    "GF-REGIONAL-DESTR-015": "run_destr015_parallel_branch_join.py",
    "GF-REGIONAL-DESTR-016": "run_destr016_preempting_reboot.py",
    "GF-REGIONAL-DESTR-017": "run_destr017_out_of_band_reboot_fence.py",
    "GF-REGIONAL-DESTR-018": "run_destr018_lifetime_deadline.py",
    "GF-REGIONAL-ISO-006": "run_iso006_cluster_offline.py",
    "GF-REGIONAL-E2E-002": "run_e2e002_multicluster_fault.py",
    "GF-REGIONAL-COLLECT-001": "run_collector_acceptance.py",
    "GF-REGIONAL-COLLECT-002": "run_collector_acceptance.py",
    "GF-REGIONAL-COLLECT-003": "run_collector_acceptance.py",
    "GF-REGIONAL-COLLECT-004": "run_collector_destructive.py",
    "GF-REGIONAL-COLLECT-005": "run_collector_acceptance.py",
    "GF-REGIONAL-COLLECT-008": "run_collector_destructive.py",
    "GF-REGIONAL-COLLECT-009": "run_collector_acceptance.py",
    "GF-REGIONAL-COLLECT-010": "run_collector_acceptance.py",
    "GF-REGIONAL-COLLECT-011": "run_collector_acceptance.py",
    "GF-REGIONAL-COLLECT-012": "run_collector_acceptance.py",
    "GF-REGIONAL-COLLECT-013": "run_collector_destructive.py",
    "GF-REGIONAL-COLLECT-014": "run_collector_destructive.py",
    "GF-REGIONAL-COLLECT-016": "run_collect016_training_recovery.py",
    "GF-REGIONAL-COLLECT-017": "run_collect017_efa_plugin.py",
    "GF-REGIONAL-COLLECT-015": "run_collector_destructive.py",
}
PLAN_ONLY_SMOKE_CASES = {
    "GF-REGIONAL-NET-002": "NET002_LIVE_REGISTRY_INTERRUPTION",
    "GF-REGIONAL-NET-003": "NET003_RESULT_CONNECTION_RESET",
    "GF-REGIONAL-HA-005": "HA005_CONTROL_PLANE_ROLLOUT",
    "GF-REGIONAL-HA-006": "HA006_FORCE_DELETE_TEST_EXECUTOR",
}


def test_regional_fixture_layout_has_no_legacy_scattered_directories() -> None:
    assert not (ROOT / "scripts/boot-guard-probe").exists(), (
        "legacy boot guard directory must stay removed"
    )
    assert not (ROOT / "scripts/e2e/manifests").exists(), (
        "legacy E2E manifest directory must stay removed"
    )
    assert not (ROOT / "tests/manifests").exists(), (
        "live manifests must not return to the pytest tree"
    )
    regional = ROOT / "scripts/e2e/regional"
    assert (regional / "boot_guard/derive.sh").is_file(), (
        "regional boot guard fixture is missing"
    )
    assert (regional / "manifests/fault-injection/kmsg-xid54.yaml").is_file(), (
        "regional fault-injection manifests are missing"
    )
    assert (
        regional / "manifests/training/xid11-three-node-pytorchjob.yaml"
    ).is_file(), "regional training manifests are missing"
    assert (regional / "run_boot019_admin_lifecycle.py").is_file(), (
        "BOOT-019 lifecycle runner is missing"
    )
    assert (regional / "run_boot020_release_rolling.py").is_file(), (
        "BOOT-020 release runner is missing"
    )
    top_level = {
        path.name
        for path in (ROOT / "scripts/e2e").iterdir()
        if path.is_file() and path.name != "__pycache__"
    }
    assert top_level == {
        "README.md",
        "__init__.py",
        "isolated_api.py",
        "render_manifest.py",
    }


def test_public_regional_contracts_do_not_reference_task_local_runners() -> None:
    public_contracts = (
        ROOT / "testcases/fault-scenarios.yaml",
        ROOT / "docs/区域模式端到端验收测试用例.md",
    )

    for path in public_contracts:
        assert ".codex/" not in path.read_text(encoding="utf-8"), path
    fixture_readme = (ROOT / "scripts/e2e/regional/README.md").read_text(
        encoding="utf-8"
    )
    assert "ignored `.codex/`" in fixture_readme
    assert "never referenced by the catalog" in fixture_readme


def test_local_executor_guard_fixture_covers_iso002_and_cmd011() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/e2e/regional/audit_executor_local_guards.py"),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert set(payload) == {"GF-REGIONAL-ISO-002", "GF-REGIONAL-CMD-011"}
    assert all(item["status"] == "FAILED" for item in payload.values()), payload
    assert all(
        item["status_source"] == "executor-rejected" for item in payload.values()
    ), payload


def test_command_protocol_fixtures_cover_every_cmd_case() -> None:
    local = {"GF-REGIONAL-CMD-011"}
    expected = {f"GF-REGIONAL-CMD-{number:03d}" for number in range(1, 17)}

    assert set(AUDITED_CASE_IDS) | local == expected
    assert set(AUDITED_CASE_IDS).isdisjoint(local), (
        "CMD-011 must remain owned by the Executor-local fixture"
    )


def test_cmd006_fixture_waits_for_the_documented_lease_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CMD-006 has to outlast the lease it took, not just retry quickly.

    The second Executor claims a 10 second lease; completing with its token
    before that lease expires proves nothing about expiry handling, so the
    fixture sleeps past it and expects 409 from the now-expired token.
    """

    slept: list[float] = []
    leases: list[int] = []
    tokens = iter(("token-1", "token-2", "token-3"))
    completions = iter(
        (
            (200, {}),
            (409, {}),
            (409, {}),
            (200, {"status": "SUCCEEDED", "last_lease_owner": "cmd006-a3"}),
        )
    )
    recorded: dict[str, dict] = {}
    monkeypatch.setattr(
        live_protocol_audit.time, "sleep", lambda seconds: slept.append(seconds)
    )

    def claim(*, executor_id: str, lease_seconds: int):
        leases.append(lease_seconds)
        return 200, {"commands": [{"executor_id": executor_id, "token": next(tokens)}]}

    audit = SimpleNamespace(
        seed=lambda _name: SimpleNamespace(command_id="command-1"),
        claim=claim,
        complete=lambda _command_id, payload: next(completions),
        record=lambda case_id, **fields: recorded.update({case_id: fields}),
        _lease_token=lambda command: command["token"],
    )

    LiveProtocolAudit.run_006(audit)

    assert slept and slept[0] > leases[1], (
        "the fixture completed with the second token before its lease expired"
    )
    assert recorded == {
        "GF-REGIONAL-CMD-006": {
            "stale_token_one": 409,
            "expired_token_two": 409,
            "final_owner": "cmd006-a3",
        }
    }


def test_auth_boundary_fixture_validates_precise_results() -> None:
    statuses = {
        "AUTH-001": 401,
        "AUTH-002-no-auth": 401,
        "AUTH-002-basic": 401,
        "AUTH-002-empty-bearer": 403,
        "AUTH-003": 403,
        "AUTH-004-zero": 403,
        "AUTH-004-near": 403,
        "AUTH-005": 403,
        "AUTH-006": 403,
        "AUTH-008-A-normal": 200,
        "AUTH-008-A-fake-executor": 200,
        "AUTH-008-B-header-A-token": 403,
        "AUTH-008-A-header-B-token": 403,
        "AUTH-009 /v1/gpu-events/xid": 403,
        "AUTH-011-health": 200,
        "AUTH-011-metrics": 403,
        "AUTH-011-clusters-anon": 403,
        "AUTH-011-clusters-cluster-token": 403,
    }
    results = {
        name: {"status": status, "body": {}} for name, status in statuses.items()
    }
    for name in ("AUTH-004-zero", "AUTH-004-near"):
        results[name]["body"]["detail"] = "regional cluster authentication failed"
    for name in ("AUTH-008-A-normal", "AUTH-008-A-fake-executor"):
        results[name]["body"]["commands"] = [{"cluster_id": "cluster-a"}]

    validate_matrix(results, cluster_a="cluster-a")

    results["AUTH-004-near"]["body"]["detail"] = "token mismatch"
    with pytest.raises(AssertionError):
        validate_matrix(results, cluster_a="cluster-a")


def test_collector_outbox_fixture_covers_current_delivery_contract() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/e2e/regional/audit_collector_outbox.py")],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "'replay_order': [1, 2, 1]" in result.stdout
    assert "'background_drained_without_new_live_event': True" in result.stdout
    assert "'background_replay_order': [99, 30, 31, 32, 33, 34, 35, 36]" in (
        result.stdout
    )
    assert "'bounded_sequences': [12, 13, 14]" in result.stdout
    assert "'unwritable_buffered': False" in result.stdout


def test_net004_dependency_audit_is_environment_driven() -> None:
    path = ROOT / "scripts/e2e/regional/audit_net004_dependency_boundary.py"
    source = path.read_text(encoding="utf-8")

    assert "/secure/gpu-fault-bootstrap" not in source
    assert "gpu-fault-gpu-1-" not in source
    assert "514385905925" not in source

    help_result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--cpu-kubeconfig" in help_result.stdout
    assert "--gpu-kubeconfig" in help_result.stdout
    assert "--gpu-context" in help_result.stdout
    assert "--region" in help_result.stdout

    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "")}
    missing = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode == 1
    assert "CPU_KUBECONFIG" in missing.stdout


@pytest.mark.parametrize(("case_id", "script_name"), PROMOTED_MANUAL_DRIVERS.items())
def test_promoted_manual_live_driver_has_safety_entrypoint(
    case_id: str, script_name: str
) -> None:
    path = ROOT / "scripts/e2e/regional" / script_name
    source = path.read_text(encoding="utf-8")

    assert "/secure/gpu-fault-bootstrap" not in source
    assert "gpu-fault-gpu-1-" not in source
    assert "514385905925" not in source
    assert "2026, 8, 31" not in source

    help_result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    for option in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
        assert option in help_result.stdout, (case_id, option)


@pytest.mark.parametrize(("case_id", "confirmation"), PLAN_ONLY_SMOKE_CASES.items())
def test_synthetic_registry_live_driver_plan_is_non_mutating(
    tmp_path: Path, case_id: str, confirmation: str
) -> None:
    script_name = PROMOTED_MANUAL_DRIVERS[case_id]
    env = {
        **os.environ,
        "GPU_FAULT_CONTROL_KUBECONFIG": "/tmp/cpu.kubeconfig",
        "KUBECONFIG": "/tmp/gpu.kubeconfig",
        "GPU_FAULT_DATAPLANE_CONTEXT": "test-gpu-context",
        "GPU_FAULT_PERF_AWS_REGION": "us-west-2",
        "GPU_FAULT_PERF_CONTROL_NAMESPACE": "gpu-fault-system",
        "GPU_FAULT_PERF_DATAPLANE_NAMESPACE": "gpu-fault-system",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/e2e/regional" / script_name),
            "--run-dir",
            str(tmp_path),
            "--attempt",
            "7",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(
        (tmp_path / "cases" / case_id / "plan.json").read_text(encoding="utf-8")
    )

    assert json.loads(completed.stdout)["case_id"] == case_id
    assert plan["attempt"] == 7
    assert plan["confirmation"] == confirmation
    assert plan["mutation_performed"] is False


def _readiness_matrix() -> dict:
    artifact = "a" * 64
    wrong_artifact = "0" * 64
    base = {
        "ready": True,
        "reasons": [],
        "unsupported_execution_owners": [],
        "execution_owners": ["gpu-fault-kubernetes-adapter"],
        "executor_artifact_sha256": artifact,
        "last_successful_claim_age_seconds": 0,
    }
    return {
        "valid": (200, base),
        "wrong_token": (403, {"detail": "regional cluster authentication failed"}),
        "wrong_pin": (
            503,
            {
                **base,
                "ready": False,
                "executor_artifact_sha256": wrong_artifact,
                "reasons": [
                    "regional executor artifact mismatch: "
                    f"expected {artifact}, got {wrong_artifact}"
                ],
            },
        ),
        "no_owner": (
            503,
            {
                **base,
                "ready": False,
                "execution_owners": [],
                "reasons": [
                    "executor advertised no execution owners, so it can claim nothing"
                ],
            },
        ),
        "stale": (
            503,
            {
                **base,
                "ready": False,
                "last_successful_claim_age_seconds": 86400,
                "reasons": ["last successful claim was 86400s ago (limit 900s)"],
            },
        ),
        "wrong_artifact": wrong_artifact,
        "stale_age_seconds": 86400,
    }


def test_executor_readiness_fixture_validates_precise_reasons() -> None:
    evidence = validate_readiness_matrix(**_readiness_matrix())

    assert evidence["wrong_token"]["detail"] == (
        "regional cluster authentication failed"
    )
    assert evidence["wrong_pin"]["reasons"] == [
        f"regional executor artifact mismatch: expected {'a' * 64}, got {'0' * 64}"
    ]
    assert evidence["no_owner"]["reasons"] == [
        "executor advertised no execution owners, so it can claim nothing"
    ]
    assert evidence["stale_claim"]["reasons"] == [
        "last successful claim was 86400s ago (limit 900s)"
    ]


def test_executor_readiness_fixture_rejects_a_generic_503() -> None:
    matrix = _readiness_matrix()
    status, payload = matrix["wrong_pin"]
    matrix["wrong_pin"] = (status, {**payload, "reasons": ["executor is unavailable"]})

    with pytest.raises(AssertionError):
        validate_readiness_matrix(**matrix)


class FakeAdminLifecycleBackend:
    def __init__(self) -> None:
        self.cluster_ids = {"cluster-a"}
        self.calls = []

    def snapshot(self) -> dict:
        values = sorted(self.cluster_ids)
        return {
            "site_cluster_ids": values,
            "registry_secret_cluster_ids": values,
            "release_state_cluster_ids": values,
            "installation_registry_cluster_ids": values,
            "cpu_control_plane_ready": True,
        }

    def join(self, fault=None) -> dict:
        self.calls.append(("join", fault))
        if fault == "before-site-commit":
            return {"phase": "ROLLED_BACK", "cluster_id": "cluster-b"}
        self.cluster_ids.add("cluster-b")
        if fault == "after-site-commit":
            return {"phase": "FAILED_AFTER_COMMIT", "cluster_id": "cluster-b"}
        return {"phase": "COMPLETED", "cluster_id": "cluster-b"}

    def capture_joined_token(self, cluster_id: str) -> dict:
        self.calls.append(("capture", cluster_id))
        return {
            "cluster_id": cluster_id,
            "token_path": "/secure/revoked-token",
            "token_sha256": "a" * 64,
        }

    def remove(self, cluster_id: str) -> dict:
        self.calls.append(("remove", cluster_id))
        self.cluster_ids.remove(cluster_id)
        return {
            "phase": "COMPLETED",
            "cluster_id": cluster_id,
            "remaining_cluster_ids": sorted(self.cluster_ids),
        }

    def probe_revoked_token(self, capture: dict) -> dict:
        self.calls.append(("probe", capture["cluster_id"]))
        return {"status": 403, "detail": "regional cluster authentication failed"}

    def uninstall(self) -> dict:
        self.calls.append(("uninstall", None))
        return {
            "cpu_cluster": "keep",
            "gpu_clusters": "preserved",
            "delete_policy_residuals": 0,
        }

    def cleanup_sensitive_files(self) -> None:
        self.calls.append(("cleanup", None))


def test_boot019_runner_executes_and_records_the_full_lifecycle(tmp_path: Path) -> None:
    evidence = tmp_path / "GF-REGIONAL-BOOT-019.json"
    backend = FakeAdminLifecycleBackend()
    recorder = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-019", inputs={"site": "test"}
    )

    result = run_admin_lifecycle(backend, recorder)

    assert result["status"] == "COMPLETED"
    assert set(result["stages"]) == {
        "baseline",
        "join_failure_before_site_commit",
        "join_failure_after_site_commit",
        "join_resumed",
        "joined_snapshot",
        "joined_token_captured",
        "joined_cluster_removed",
        "removed_token_rejected",
        "post_remove_snapshot",
        "last_cluster_removed",
        "empty_registry_snapshot",
        "uninstall_keep_cpu",
    }
    assert backend.calls == [
        ("join", "before-site-commit"),
        ("join", "after-site-commit"),
        ("join", None),
        ("capture", "cluster-b"),
        ("remove", "cluster-b"),
        ("probe", "cluster-b"),
        ("remove", "cluster-a"),
        ("uninstall", None),
        ("cleanup", None),
    ]
    assert evidence.stat().st_mode & 0o777 == 0o600


class FakeReleaseRollingBackend:
    def __init__(self) -> None:
        self.completed = set()
        self.live = {
            "cpu_wheel": "cpu-v1",
            "runtime_profile_version": "profile-v1",
            "clusters": {
                "cluster-a": {
                    "wheel": "executor-v1",
                    "reconciler_wheel": "node-v1",
                    "bundle": "bundle-v1",
                }
            },
        }
        self.cpu_generations = {"ingress": 1, "worker": 1}
        self.gpu_generations = {"cluster-a": {"executor": 1, "reconciler": 1}}
        self.calls = []

    def classify(self, scenario: str) -> dict:
        kind = (
            "NOOP"
            if scenario in self.completed
            else {
                "noop": "NOOP",
                "control_plane": "CONTROL_PLANE_ONLY",
                "executor": "DATA_PLANE_COMPATIBLE",
                "agent": "DATA_PLANE_COMPATIBLE",
                "full": "FULL",
            }[scenario]
        )
        return {"kind": kind, "changed": [] if kind == "NOOP" else [scenario]}

    def snapshot(self, scenario: str) -> dict:
        return {
            "phase": "complete",
            "release_id": scenario,
            "live": copy.deepcopy(self.live),
            "cpu_generations": dict(self.cpu_generations),
            "gpu_generations": copy.deepcopy(self.gpu_generations),
            "next_deploy": self.classify(scenario),
        }

    def deploy(
        self,
        scenario: str,
        *,
        diff: dict,
        fault_phase=None,
        resume=False,
        auto_rollback=None,
    ) -> dict:
        self.calls.append((scenario, fault_phase, resume, auto_rollback, diff["kind"]))
        plans = {
            "control_plane": {
                "clusters": {},
                "restores_data_plane": False,
                "needs_controller": False,
            },
            "executor": {
                "clusters": {
                    "cluster-a": ["collector", "executor", "reconciler", "watcher"]
                },
                "restores_data_plane": True,
                "needs_controller": False,
            },
            "agent": {
                "clusters": {"cluster-a": ["reconciler", "agent"]},
                "restores_data_plane": True,
                "needs_controller": True,
            },
            "full": {
                "clusters": {"cluster-a": ["executor", "reconciler", "agent"]},
                "restores_data_plane": True,
                "needs_controller": True,
            },
        }
        if fault_phase:
            if auto_rollback:
                return {
                    "phase": "rolled-back",
                    "injected_failure": fault_phase,
                    "rollback_plan": plans[scenario],
                    "rollback_timing": {"t_safe_seconds": 30.0, "t_full_seconds": 45.0},
                    "operation_duration_seconds": 45.0,
                }
            return {
                "phase": "failed",
                "injected_failure": fault_phase,
                "operation_duration_seconds": 1.0,
            }
        if scenario == "control_plane":
            self.live["cpu_wheel"] = "cpu-v2"
            self.cpu_generations = {"ingress": 2, "worker": 2}
        elif scenario == "executor":
            self.live["clusters"]["cluster-a"]["wheel"] = "executor-v2"
            self.gpu_generations["cluster-a"]["executor"] = 2
        elif scenario == "agent":
            self.live["clusters"]["cluster-a"]["reconciler_wheel"] = "node-v2"
            self.live["clusters"]["cluster-a"]["bundle"] = "bundle-v2"
            self.gpu_generations["cluster-a"]["reconciler"] = 2
        elif scenario == "full":
            self.live = {
                "cpu_wheel": "cpu-v3",
                "runtime_profile_version": "profile-v2",
                "clusters": {
                    "cluster-a": {
                        "wheel": "executor-v3",
                        "reconciler_wheel": "node-v2",
                        "bundle": "bundle-v2",
                    }
                },
            }
            self.cpu_generations = {"ingress": 3, "worker": 3}
            self.gpu_generations = {"cluster-a": {"executor": 3, "reconciler": 2}}
        self.completed.add(scenario)
        return {
            "phase": "complete",
            "injected_failure": None,
            "operation_duration_seconds": 10.0 if resume else 20.0,
        }


def test_boot020_runner_covers_diff_resume_and_rollback(tmp_path: Path) -> None:
    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    backend = FakeReleaseRollingBackend()
    recorder = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )

    result = run_release_rolling(backend, recorder)

    assert result["status"] == "COMPLETED"
    assert backend.calls == [
        ("noop", None, False, None, "NOOP"),
        ("control_plane", "cpu-finalized", False, True, "CONTROL_PLANE_ONLY"),
        ("control_plane", None, False, None, "CONTROL_PLANE_ONLY"),
        ("executor", "data-converged", False, True, "DATA_PLANE_COMPATIBLE"),
        ("executor", "cpu-staged", False, False, "DATA_PLANE_COMPATIBLE"),
        ("executor", None, True, False, "DATA_PLANE_COMPATIBLE"),
        ("agent", "data-converged", False, True, "DATA_PLANE_COMPATIBLE"),
        ("agent", None, False, None, "DATA_PLANE_COMPATIBLE"),
        ("full", "data-converged", False, True, "FULL"),
        ("full", None, False, None, "FULL"),
    ]
    assert (
        result["stages"]["full_rollback_snapshot"]["live"]
        == (result["stages"]["full_before"]["live"])
    )
    assert evidence.stat().st_mode & 0o777 == 0o600


def test_boot020_configures_explicit_gpu_kubeconfig(
    tmp_path: Path, monkeypatch
) -> None:
    kubeconfig = tmp_path / "gpu.kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    monkeypatch.delenv("KUBECONFIG", raising=False)

    result = configure_gpu_kubeconfig(kubeconfig)

    assert result == kubeconfig.resolve()
    assert os.environ["KUBECONFIG"] == str(kubeconfig.resolve())


def test_ha007_runner_waits_for_in_flight_requests(tmp_path: Path) -> None:
    report = run_probe(tmp_path, [0.01, 0.02])

    assert report["status"] == "PASS"
    assert [item["completed"] for item in report["runs"]] == [1, 1]
    assert all(not item["shutdown_failures"] for item in report["runs"]), report


def test_ha007_child_uses_the_repository_source_tree(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing/path")

    child_path = _child_environment()["PYTHONPATH"].split(os.pathsep)

    assert child_path[0] == str(ROOT / "src")
    assert child_path[1:] == ["/existing/path"]


def test_ha008_acceptance_runs_both_fatal_exit_branches(tmp_path: Path) -> None:
    report = run_ha008_acceptance(tmp_path)

    assert report["verdict"] == "PASS"
    assert [item["exit_code"] for item in report["branches"]] == [70, 70]
    assert [item["status_after_exit"] for item in report["branches"]] == [
        "PENDING",
        "LEASED",
    ]
    assert all(item["stale_result_rejected"] for item in report["branches"]), (
        "a fatal-exit branch accepted a stale result"
    )
    assert all(item["final_status"] == "COMPLETED" for item in report["branches"]), (
        "a fatal-exit branch did not converge to COMPLETED"
    )
    assert not list(tmp_path.glob("*.db")), "HA-008 left a temporary database"
    assert not list(tmp_path.glob("*claim*")), "HA-008 left a claim state file"
