from __future__ import annotations

import contextlib
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
from scripts.e2e.regional import seeded_command_fixture
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from scripts.e2e.regional.audit_auth_boundary import validate_matrix
from scripts.e2e.regional.audit_executor_readiness import validate_readiness_matrix
from scripts.e2e.regional.audit_regional_command_protocol_live import (
    AUDITED_CASE_IDS,
    LiveProtocolAudit,
)
from scripts.e2e.regional.regional_case_contract import formal_predecessor
from scripts.e2e.regional.run_boot019_admin_lifecycle import run_admin_lifecycle
from scripts.e2e.regional.run_boot020_release_rolling import (
    EXECUTOR_DEPLOYMENT,
    STAGES,
    LiveReleaseRollingBackend,
    configure_gpu_kubeconfig,
    deployment_generations,
    resume_release_rolling,
    resume_target,
    run_release_rolling,
)
from scripts.e2e.regional.run_ha007_control_worker_shutdown import (
    DEFAULT_DURATIONS,
    WORKER_THREAD_NAME,
    _child_environment,
    control_worker_budgets,
    run_probe,
)
from scripts.e2e.regional.run_ha008_processor_exit_acceptance import (
    run_acceptance as run_ha008_acceptance,
)

ROOT = Path(__file__).resolve().parents[2]

# The drivers import gpu_fault; run them against THIS checkout's src, not
# whatever editable install the interpreter would otherwise resolve.
_CHECKOUT_ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

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
    "GF-REGIONAL-DESTR-019": "run_destr019_agent_restart_ledger.py",
    "GF-REGIONAL-DESTR-020": "run_destr020_identity_mismatch_isolation.py",
    "GF-REGIONAL-DESTR-021": "run_destr021_adversarial_node_metadata.py",
    "GF-REGIONAL-DESTR-022": "run_destr022_spare_reservation_reclaim.py",
    "GF-REGIONAL-HA-010": "run_ha010_aurora_blackout_liveness.py",
    "GF-REGIONAL-BOOT-023": "run_boot023_release_history.py",
    "GF-REGIONAL-CMD-017": "run_cmd017_barrier_hold.py",
    "GF-REGIONAL-CMD-018": "run_cmd018_open_sibling_hold.py",
    "GF-REGIONAL-NET-006": "run_net006_lease_loss_withheld_result.py",
    "GF-REGIONAL-NET-008": "run_net008_outbox_dead_letter.py",
    "GF-REGIONAL-NOTIFY-007": "run_notify007_delivery_states.py",
    "GF-REGIONAL-PREEMPT-037": "run_preempt037_dispatcher_liveness.py",
    "GF-REGIONAL-PREEMPT-038": "run_preempt038_evidence_pins.py",
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
    "GF-REGIONAL-COLLECT-018": "run_collect018_rejected_event.py",
    "GF-REGIONAL-COLLECT-019": "run_collect019_nvidia_smi_hang.py",
    "GF-REGIONAL-COLLECT-020": "run_collect020_gpu_identity.py",
    "GF-REGIONAL-COLLECT-015": "run_collector_destructive.py",
}
PLAN_ONLY_SMOKE_CASES = {
    "GF-REGIONAL-NET-002": "NET002_LIVE_REGISTRY_INTERRUPTION",
    "GF-REGIONAL-NET-003": "NET003_RESULT_CONNECTION_RESET",
    "GF-REGIONAL-HA-005": "HA005_CONTROL_PLANE_ROLLOUT",
    "GF-REGIONAL-HA-006": "HA006_FORCE_DELETE_TEST_EXECUTOR",
    "GF-REGIONAL-NET-006": "NET006_LIVE_LEASE_LOSS_WITHHELD_RESULT",
    "GF-REGIONAL-CMD-017": "CMD017_LIVE_BARRIER_HOLD",
    "GF-REGIONAL-CMD-018": "CMD018_EXECUTE",
}
# Plan-only drivers whose ``--plan`` path already gates on the formal
# predecessor's PASS evidence (exit 1 when it is missing).
PREDECESSOR_GATED_PLAN_CASES = {"GF-REGIONAL-NET-002", "GF-REGIONAL-NET-003"}


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
    # Orphans the 2026-09-07 review removed: nothing under scripts/, tests/,
    # src/, deploy/, docs/ or testcases/ named them, and one of them
    # (spare_health_p156_probe.py) cordoned a real node with no guard.
    removed = (
        "q118_requeue_containment.py",
        "spare_health_p156_probe.py",
        "audit_collector_runtime_probe.py",
        "audit_kernel_collector_p110.py",
        "audit_workload_log_evidence.py",
        "manifests/audit-collector-host-probe.yaml",
        "manifests/fault-injection/cross-fault-outside-window-sxid.yaml",
        "manifests/fault-injection/cross-fault-outside-window-xid79.yaml",
        "manifests/fault-injection/cross-fault-within-window.yaml",
        "manifests/fault-injection/dual-node-sxid10003-kmsg.yaml",
        "manifests/fault-injection/dual-node-training-reset-quiesce.yaml",
        "manifests/fault-injection/dual-node-xid95-kmsg.yaml",
        "manifests/fault-injection/dual-node-xid95-quiesced-kmsg.yaml",
        "manifests/fault-injection/kmsg-xid74-live.yaml",
        "manifests/fault-injection/kmsg-xid78.yaml",
        "manifests/fault-injection/kmsg-xid79-reboot-resume.yaml",
        "manifests/training/three-node-nccl-hung-triage-pytorchjob.yaml",
        "manifests/training/two-node-hung-e2e-pytorchjob.yaml",
    )
    present = [name for name in removed if (regional / name).exists()]
    assert present == [], "unreferenced regional fixtures must stay removed"
    assert sorted(
        path.name for path in (regional / "manifests/fault-injection").glob("*.yaml")
    ) == ["kmsg-xid45-xid14.yaml", "kmsg-xid54.yaml", "kmsg-xid63-xid48.yaml"]
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
    # CMD-011 uses the Executor-local guard fixture; CMD-017 seeds a synthetic
    # barrier command for its own probe executor (run_cmd017_barrier_hold.py);
    # CMD-018 drives the deployed adapter in-Pod (run_cmd018_open_sibling_hold.py).
    local = {"GF-REGIONAL-CMD-011", "GF-REGIONAL-CMD-017", "GF-REGIONAL-CMD-018"}
    expected = {f"GF-REGIONAL-CMD-{number:03d}" for number in range(1, 19)}

    assert set(AUDITED_CASE_IDS) | local == expected
    assert set(AUDITED_CASE_IDS).isdisjoint(local), (
        "CMD-011, CMD-017 and CMD-018 must remain owned by their own fixtures"
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
        env=_CHECKOUT_ENV,
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
        env=_CHECKOUT_ENV,
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
    # The gated drivers' plan path refuses without the formal predecessor's
    # PASS evidence, so seed it under the run directory; the plan-only gate
    # binds no release_id/cluster_id, so case_id and verdict are enough.
    predecessor = None
    if case_id in PREDECESSOR_GATED_PLAN_CASES:
        predecessor = formal_predecessor(case_id)
        assert predecessor is not None, case_id
        evidence = tmp_path / "cases" / predecessor / f"{predecessor}.json"
        evidence.parent.mkdir(parents=True)
        evidence.write_text(
            json.dumps({"case_id": predecessor, "verdict": "PASS"}), encoding="utf-8"
        )
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
    if predecessor is not None:
        gate = plan["details"]["predecessor"]
        assert gate["case_id"] == predecessor
        assert gate["verdict"] == "PASS"
        assert gate["valid"] is True
        assert gate["execution_allowed"] is True
        # The seeded evidence is the only file under cases/<predecessor>/.
        assert sorted(path.name for path in evidence.parent.iterdir()) == [
            evidence.name
        ]


@pytest.mark.parametrize("case_id", sorted(PREDECESSOR_GATED_PLAN_CASES))
def test_predecessor_gated_plan_refuses_without_pass_evidence(
    tmp_path: Path, case_id: str
) -> None:
    """The plan is still written but exits 1 when the predecessor has no PASS."""

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
            "1",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    plan = json.loads(
        (tmp_path / "cases" / case_id / "plan.json").read_text(encoding="utf-8")
    )

    assert completed.returncode == 1, completed.stderr
    assert plan["mutation_performed"] is False
    assert plan["details"]["predecessor"]["verdict"] == "MISSING"
    assert plan["details"]["predecessor"]["valid"] is False
    assert plan["details"]["predecessor"]["case_id"] == formal_predecessor(case_id)


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
        # Like the live backend: the token stays in memory, the record carries
        # only its digest.
        return {
            "cluster_id": cluster_id,
            "token_storage": "memory",
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
        # The runner asserts what varies with the live run: at least the CPU
        # cluster and the GPU cluster record preserved, and no resource still
        # waiting to be deleted in the final registry snapshot.
        return {
            "cpu_cluster": "keep",
            "gpu_clusters": "preserved",
            "delete_policy_residuals": 0,
            "registry_entries_preserved": 2,
            "final_registry_statuses": {
                "cluster/cluster-a/eks": "PRESERVED",
                "cpu/eks": "PRESERVED",
            },
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
    assert result["stages"]["joined_token_captured"]["token_storage"] == "memory"
    assert "token_path" not in result["stages"]["joined_token_captured"]
    assert result["stages"]["uninstall_keep_cpu"]["registry_entries_preserved"] == 2
    assert evidence.stat().st_mode & 0o777 == 0o600


def test_boot019_runner_rejects_an_uninstall_that_left_delete_pending(
    tmp_path: Path,
) -> None:
    class Backend(FakeAdminLifecycleBackend):
        def uninstall(self) -> dict:
            result = super().uninstall()
            result["final_registry_statuses"]["aurora/cluster"] = "DELETE_PENDING"
            return result

    backend = Backend()
    recorder = EvidenceRecorder(
        tmp_path / "GF-REGIONAL-BOOT-019.json",
        case_id="GF-REGIONAL-BOOT-019",
        inputs={"site": "test"},
    )

    with pytest.raises(RuntimeError, match="DELETE_PENDING"):
        run_admin_lifecycle(backend, recorder)

    # The captured token is cleared on the failure path as well.
    assert backend.calls[-1] == ("cleanup", None)


PREVIOUS_EXECUTOR_PIN = "a" * 64
CANDIDATE_EXECUTOR_PIN = "b" * 64


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
        self.cpu_generations = {"gpu-fault-api-ha": 1, "gpu-fault-worker": 1}
        # Keyed by the real Deployment names: the executor stage asserts the
        # generation changes are exactly {GPU_EXECUTOR_DEPLOYMENT} per cluster.
        self.gpu_generations = {
            "cluster-a": {EXECUTOR_DEPLOYMENT: 1, "gpu-fault-completion-watcher": 1}
        }
        self.calls = []
        # Every live snapshot costs 20-30 s of kubectl reads, so the runner is
        # held to the number it takes, not only to the deploys it issues.
        self.snapshots: list[str] = []

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
        self.snapshots.append(scenario)
        return {
            "phase": "complete",
            "release_id": scenario,
            "live": copy.deepcopy(self.live),
            "cpu_generations": dict(self.cpu_generations),
            "gpu_generations": copy.deepcopy(self.gpu_generations),
            "next_deploy": self.classify(scenario),
        }

    @staticmethod
    def _pins(phase: str) -> dict:
        """The two-phase executor pin window the runner asserts at each checkpoint."""

        return {
            "rolled-back": {"required": PREVIOUS_EXECUTOR_PIN, "compatible": []},
            "staged": {
                "required": PREVIOUS_EXECUTOR_PIN,
                "compatible": [CANDIDATE_EXECUTOR_PIN],
            },
            "finalized": {"required": CANDIDATE_EXECUTOR_PIN, "compatible": []},
        }[phase] | {
            "candidate": CANDIDATE_EXECUTOR_PIN,
            "previous_required": PREVIOUS_EXECUTOR_PIN,
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
                "global_components": ["cpu_stage", "cpu_finalize"],
                "restores_data_plane": False,
                "needs_controller": False,
            },
            "executor": {
                "clusters": {
                    "cluster-a": ["collector", "executor", "reconciler", "watcher"]
                },
                "global_components": [],
                "restores_data_plane": True,
                "needs_controller": False,
            },
            "agent": {
                "clusters": {"cluster-a": ["reconciler", "agent"]},
                "global_components": [],
                "restores_data_plane": True,
                "needs_controller": True,
            },
            "full": {
                "clusters": {"cluster-a": ["executor", "reconciler", "agent"]},
                "global_components": ["cpu_stage"],
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
                    "pins": self._pins("rolled-back"),
                    "operation_duration_seconds": 45.0,
                }
            return {
                "phase": "failed",
                "injected_failure": fault_phase,
                "pins": self._pins("staged"),
                "operation_duration_seconds": 1.0,
            }
        if scenario == "control_plane":
            self.live["cpu_wheel"] = "cpu-v2"
            self.cpu_generations = {"gpu-fault-api-ha": 2, "gpu-fault-worker": 2}
        elif scenario == "executor":
            # The interrupted attempt already staged the CPU side; the resume
            # rolls only the executor Deployment.
            self.live["clusters"]["cluster-a"]["wheel"] = "executor-v2"
            self.gpu_generations["cluster-a"][EXECUTOR_DEPLOYMENT] = 2
        elif scenario == "agent":
            self.live["clusters"]["cluster-a"]["reconciler_wheel"] = "node-v2"
            self.live["clusters"]["cluster-a"]["bundle"] = "bundle-v2"
            self.gpu_generations["cluster-a"]["gpu-fault-completion-watcher"] = 2
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
            self.cpu_generations = {"gpu-fault-api-ha": 3, "gpu-fault-worker": 3}
            self.gpu_generations = {
                "cluster-a": {EXECUTOR_DEPLOYMENT: 3, "gpu-fault-completion-watcher": 3}
            }
        self.completed.add(scenario)
        return {
            "phase": "complete",
            "injected_failure": None,
            "pins": self._pins("finalized"),
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
    stages = result["stages"]
    # Executor stage: the two-phase pin window at each checkpoint, only the
    # executor Deployment rolled, and the CPU generation held across the resume.
    assert stages["executor_injected_failure_and_rollback"]["pins"]["required"] == (
        PREVIOUS_EXECUTOR_PIN
    )
    assert stages["executor_interrupted_failure"]["pins"]["compatible"] == [
        CANDIDATE_EXECUTOR_PIN
    ]
    assert stages["executor_resumed"]["pins"] == {
        "required": CANDIDATE_EXECUTOR_PIN,
        "compatible": [],
        "candidate": CANDIDATE_EXECUTOR_PIN,
        "previous_required": PREVIOUS_EXECUTOR_PIN,
    }
    assert (
        stages["executor_interrupted_snapshot"]["cpu_generations"]
        == stages["executor_after"]["cpu_generations"]
    )
    before = stages["executor_before"]["gpu_generations"]["cluster-a"]
    after = stages["executor_after"]["gpu_generations"]["cluster-a"]
    assert {name for name in after if after[name] != before[name]} == {
        EXECUTOR_DEPLOYMENT
    }
    assert stages["control_plane_injected_failure_and_rollback"]["rollback_plan"][
        "global_components"
    ] == ["cpu_stage", "cpu_finalize"]
    assert evidence.stat().st_mode & 0o777 == 0o600
    # A stage's ``*_after`` and the next stage's ``*_before`` observe the same
    # live state -- only read-only classifications run between them -- so the
    # runner takes the ``*_after`` once and reuses it, marked, as the next
    # ``*_before``. ``noop_before`` has no predecessor and stays fresh.
    # The executor stage takes one extra mid-stage snapshot
    # (``executor_interrupted_snapshot``) so the CPU generations can be held
    # across the resume; like the ``*_rollback_snapshot`` reads it is never a
    # stage boundary and is not reused.
    assert backend.snapshots == [
        "noop",
        "noop",
        "control_plane",
        "control_plane",
        "executor",
        "executor",
        "executor",
        "agent",
        "agent",
        "full",
        "full",
    ], backend.snapshots
    stages = result["stages"]
    assert "reused_from" not in stages["noop_before"]
    for previous, stage in zip(STAGES, STAGES[1:]):
        before = stages[f"{stage}_before"]
        assert before["reused_from"] == f"{previous}_after", (stage, before)
        assert before["live"] == stages[f"{previous}_after"]["live"], stage
        assert (
            before["cpu_generations"] == stages[f"{previous}_after"]["cpu_generations"]
        )
        assert (
            before["gpu_generations"] == stages[f"{previous}_after"]["gpu_generations"]
        )
        # ``next_deploy`` is the one config-dependent field; a reused snapshot
        # reports the new stage's own classification, which is what a fresh
        # ``next_deploy`` computes for a committed ``complete`` release.
        assert before["next_deploy"] == stages[f"{stage}_classification"], stage
        assert "reused_from" not in stages[f"{stage}_after"]
        assert "reused_from" not in stages[f"{stage}_rollback_snapshot"]


def test_boot020_replayed_after_snapshot_is_never_reused(tmp_path: Path) -> None:
    """Only a ``*_after`` taken fresh in this process may seed the next
    ``*_before``: one replayed from evidence describes an older observation."""

    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    recorder = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )
    run_release_rolling(FakeReleaseRollingBackend(), recorder)
    document = json.loads(evidence.read_text(encoding="utf-8"))
    for name in [key for key in document["stages"] if key.startswith("full")]:
        document["stages"].pop(name)
    document["status"] = "FAILED"
    evidence.write_text(json.dumps(document), encoding="utf-8")

    second = FakeReleaseRollingBackend()
    second.completed.update({"control_plane", "executor", "agent"})
    resumed = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )

    result = run_release_rolling(second, resumed)

    assert result["status"] == "COMPLETED"
    # ``agent_after`` replayed, so ``full_before`` was observed afresh.
    assert second.snapshots == ["full", "full", "full"], second.snapshots
    assert "reused_from" not in result["stages"]["full_before"]


CPU_ARGS = ["kubectl", "--kubeconfig", "/cpu.kubeconfig"]


def _context_args(context: str) -> list[str]:
    return ["kubectl", "--context", context]


class FakeRelease:
    """A release whose ``_get_json`` answers ``get deployment`` lists per
    kube context and records every argv it was asked for.

    ``load_state`` and ``capture_previous`` are what the engine's own
    ``_load_state``/``_capture_previous`` answer; the fake records that they
    were called inside its read snapshot.
    """

    def __init__(
        self, listings: dict[str, list[dict]], *, load_state=None, capture_previous=None
    ) -> None:
        self.config = SimpleNamespace(
            namespace="gpu-fault",
            cpu_kubeconfig="/cpu.kubeconfig",
            clusters=[
                SimpleNamespace(cluster_id="cluster-a", context="ctx-a"),
                SimpleNamespace(cluster_id="cluster-b", context="ctx-b"),
            ],
        )
        self.listings = listings
        self.load_state = load_state
        self.capture_previous = capture_previous
        self.reads: list[list[str]] = []
        self.calls: list[str] = []
        self.snapshot_depth = 0
        self.reads_inside_snapshot = 0
        self.state: dict | None = None

    def _cpu(self, *args: str) -> list[str]:
        return [*CPU_ARGS, *args]

    def _gpu(self, target, *args: str) -> list[str]:
        return [*_context_args(target.context), *args]

    def _get_json(self, args: list[str]) -> dict:
        self.reads.append(list(args))
        if self.snapshot_depth:
            self.reads_inside_snapshot += 1
        assert args[-2:] == ["get", "deployment"], args
        context = args[2]
        return {"items": self.listings[context]}

    @contextlib.contextmanager
    def _read_snapshot(self):
        self.snapshot_depth += 1
        try:
            yield
        finally:
            self.snapshot_depth -= 1

    def _load_state(self) -> dict:
        self.calls.append("load_state")
        assert self.snapshot_depth == 1
        self.state = self.load_state()
        return self.state

    def _capture_previous(self) -> dict:
        self.calls.append("capture_previous")
        assert self.snapshot_depth == 1
        return self.capture_previous()


def _deployment_item(name: str, generation: int) -> dict:
    return {"metadata": {"name": name, "generation": generation}, "spec": {}}


def test_boot020_generations_come_from_one_list_per_context() -> None:
    release = FakeRelease(
        {
            "/cpu.kubeconfig": [
                _deployment_item("gpu-fault-api-ha", 7),
                _deployment_item("gpu-fault-control-worker", 3),
                _deployment_item("unrelated", 99),
            ],
            "ctx-a": [_deployment_item("gpu-fault-cluster-action-executor", 5)],
        }
    )

    cpu = deployment_generations(
        release, CPU_ARGS, ("gpu-fault-api-ha", "gpu-fault-control-worker")
    )
    gpu = deployment_generations(
        release,
        _context_args(release.config.clusters[0].context),
        ("gpu-fault-cluster-action-executor", "gpu-fault-missing"),
    )

    assert cpu == {"gpu-fault-api-ha": 7, "gpu-fault-control-worker": 3}
    # A Deployment absent from the list reads as generation 0, the same value
    # the old per-name read fell back to when kubectl returned nothing.
    assert gpu == {"gpu-fault-cluster-action-executor": 5, "gpu-fault-missing": 0}
    # Exactly one list read per kube context, with the argv the engine's own
    # `prime_deployment_snapshot` uses so a warm read cache answers it.
    assert release.reads == [
        [
            "kubectl",
            "--kubeconfig",
            "/cpu.kubeconfig",
            "-n",
            "gpu-fault",
            "get",
            "deployment",
        ],
        ["kubectl", "--context", "ctx-a", "-n", "gpu-fault", "get", "deployment"],
    ]


def test_boot020_live_snapshot_reads_once_per_context_inside_one_snapshot(
    monkeypatch,
) -> None:
    """One ``snapshot()`` loads the state once, captures the previous release
    once, derives ``next_deploy`` from the engine's own ``next_deploy`` and
    lists Deployments once per context, all inside one read snapshot."""

    from gpu_fault_release import regional_deployment_inventory as inventory

    cpu_items = [
        _deployment_item(name, index + 1)
        for index, name in enumerate(inventory.CPU_RUNTIME_DEPLOYMENTS)
    ]
    gpu_items = [
        _deployment_item(name, index + 10)
        for index, name in enumerate(inventory.DEPLOYMENTS)
    ]
    state = {"phase": "complete", "release_id": "rel-1", "previous": {}}
    release = FakeRelease(
        {"/cpu.kubeconfig": cpu_items, "ctx-a": gpu_items, "ctx-b": gpu_items[:-1]},
        load_state=lambda: state,
        capture_previous=lambda: {
            "release_id": "rel-1",
            "clusters": {"cluster-a": {}, "cluster-b": {}},
        },
    )

    backend = LiveReleaseRollingBackend({"noop": Path("/noop.json")})
    monkeypatch.setattr(backend, "_release", lambda scenario, **_: release)

    def _next_deploy(given_release, given_state) -> dict:
        release.calls.append("next_deploy")
        assert given_release is release and given_state is state
        assert release.snapshot_depth == 1
        return {"kind": "NOOP", "changed": []}

    monkeypatch.setattr(backend, "commands", SimpleNamespace(next_deploy=_next_deploy))

    snapshot = backend.snapshot("noop")

    assert release.calls == ["load_state", "capture_previous", "next_deploy"]
    assert snapshot == {
        "phase": "complete",
        "release_id": "rel-1",
        "live": {"release_id": "rel-1", "clusters": {"cluster-a": {}, "cluster-b": {}}},
        "cpu_generations": {
            name: index + 1
            for index, name in enumerate(inventory.CPU_RUNTIME_DEPLOYMENTS)
        },
        "gpu_generations": {
            "cluster-a": {
                name: index + 10 for index, name in enumerate(inventory.DEPLOYMENTS)
            },
            "cluster-b": {
                **{
                    name: index + 10 for index, name in enumerate(inventory.DEPLOYMENTS)
                },
                inventory.DEPLOYMENTS[-1]: 0,
            },
        },
        "next_deploy": {"kind": "NOOP", "changed": []},
    }
    # One list per context (CPU, cluster-a, cluster-b), nothing per Deployment,
    # and every read inside the snapshot the whole call shares.
    assert len(release.reads) == 3, release.reads
    assert release.reads_inside_snapshot == 3
    assert release.snapshot_depth == 0


def test_boot020_live_snapshot_keeps_next_deploy_none_when_it_fails(
    monkeypatch,
) -> None:
    """``build_release_summary`` swallowed a ``next_deploy`` failure into
    ``next_deploy_error`` and the snapshot recorded ``None``; the direct call
    keeps that recorded value."""

    release = FakeRelease(
        {"/cpu.kubeconfig": [], "ctx-a": [], "ctx-b": []},
        load_state=lambda: {"phase": "complete", "release_id": "rel-1"},
        capture_previous=lambda: {"clusters": {}},
    )
    backend = LiveReleaseRollingBackend({"noop": Path("/noop.json")})
    monkeypatch.setattr(backend, "_release", lambda scenario, **_: release)

    def _next_deploy(*_):
        raise RuntimeError("classification unavailable")

    monkeypatch.setattr(backend, "commands", SimpleNamespace(next_deploy=_next_deploy))

    assert backend.snapshot("noop")["next_deploy"] is None


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
    """Two in-budget requests complete; one over-budget request is the
    coordinator's documented failure. The budgets are injected so the
    over-budget path costs well under a second instead of the live 130s."""

    budgets = {"lifespan_budget_seconds": 0.5, "kubernetes_grace_seconds": 10.0}

    report = run_probe(tmp_path, [0.01, 0.02, 3.0], budgets=budgets)

    assert report["verdict"] == "PASS", report["errors"]
    assert report["errors"] == []
    assert report["lifespan_budget_seconds"] == 0.5
    assert report["kubernetes_grace_seconds"] == 10.0
    assert [item["expected_outcome"] for item in report["runs"]] == [
        "completed",
        "completed",
        "deadline-exceeded",
    ]
    assert [item["completed"] for item in report["runs"]] == [1, 1, 0]
    assert [item["returncode"] for item in report["runs"]] == [0, 0, 1]
    assert [item["shutdown_failures"] for item in report["runs"]] == [
        [],
        [],
        [WORKER_THREAD_NAME],
    ]
    assert report["runs"][2]["shutdown_seconds"] >= 0.5
    assert all(item["errors"] == [] for item in report["runs"]), report
    written = json.loads((tmp_path / "GF-REGIONAL-HA-007.json").read_text())
    assert written["verdict"] == "PASS"


def test_ha007_budgets_come_from_the_generated_control_worker_manifest() -> None:
    budgets = control_worker_budgets()

    assert budgets["sources"] == {
        "lifespan_budget_seconds": (
            "deploy/control-plane/regional/generated/"
            "gpu-fault-control-worker-config-core.yaml"
        ),
        "kubernetes_grace_seconds": (
            "deploy/control-plane/regional/generated/gpu-fault-control-worker.yaml"
        ),
    }
    assert budgets["lifespan_budget_seconds"] < budgets["kubernetes_grace_seconds"]
    # Three defaults sit inside the lifespan budget so a completing request is
    # exercised; the last is over budget so the give-up path is exercised too.
    inside = [d for d in DEFAULT_DURATIONS if d <= budgets["lifespan_budget_seconds"]]
    beyond = [d for d in DEFAULT_DURATIONS if d > budgets["lifespan_budget_seconds"]]
    assert len(inside) == 3, DEFAULT_DURATIONS
    assert len(beyond) == 1, DEFAULT_DURATIONS
    # An over-budget run still has to exit inside the Pod grace period.
    assert budgets["lifespan_budget_seconds"] + 5 < budgets["kubernetes_grace_seconds"]


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


def test_boot020_runner_resumes_at_a_later_stage(tmp_path: Path) -> None:
    """A rerun over the same evidence must not touch the site for stages that
    already ran; --start-stage skips them and records where it restarted."""

    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    first = FakeReleaseRollingBackend()
    recorder = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )
    run_release_rolling(first, recorder)
    document = json.loads(evidence.read_text(encoding="utf-8"))
    for name in (
        "agent_apply_after_rollback",
        "agent_after",
        "agent_next_classification",
    ):
        document["stages"].pop(name)
    for name in [key for key in document["stages"] if key.startswith("full")]:
        document["stages"].pop(name)
    document["status"] = "FAILED"
    evidence.write_text(json.dumps(document), encoding="utf-8")

    second = FakeReleaseRollingBackend()
    second.completed.update({"control_plane", "executor"})
    second.live["cpu_wheel"] = "cpu-v2"
    second.live["clusters"]["cluster-a"]["wheel"] = "executor-v2"
    resumed = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )

    result = run_release_rolling(second, resumed, start_stage="agent")

    assert result["status"] == "COMPLETED", result["status"]
    assert result["stages"]["resumed_at_agent"]["skipped_stages"] == [
        "noop",
        "control_plane",
        "executor",
    ], result["stages"]["resumed_at_agent"]
    # Only the missing agent apply and the whole full stage touched the site.
    assert second.calls == [
        ("agent", None, False, None, "DATA_PLANE_COMPATIBLE"),
        ("full", "data-converged", False, True, "FULL"),
        ("full", None, False, None, "FULL"),
    ], second.calls
    # ``agent_after`` was taken fresh here, so it seeds ``full_before``.
    assert second.snapshots == ["agent", "full", "full"], second.snapshots
    assert result["stages"]["full_before"]["reused_from"] == "agent_after"


def test_boot020_runner_refuses_to_resume_without_earlier_evidence(
    tmp_path: Path,
) -> None:
    recorder = EvidenceRecorder(
        tmp_path / "GF-REGIONAL-BOOT-020.json",
        case_id="GF-REGIONAL-BOOT-020",
        inputs={"configs": "test"},
    )
    with pytest.raises(RuntimeError, match="earlier stages have no recorded result"):
        run_release_rolling(FakeReleaseRollingBackend(), recorder, start_stage="agent")


def test_probe_pod_commands_run_the_script_the_configmap_publishes() -> None:
    """NET-002 once mounted ``net002_executor.py`` but exec'd a stale probe name.

    Every live runner that ships a probe through a ConfigMap must exec the same
    file name it publishes, or the pod dies with ENOENT before it is Ready.
    """
    import importlib

    for module_name in (
        "scripts.e2e.regional.run_net002_command_recovery",
        "scripts.e2e.regional.run_net003_result_retry",
        "scripts.e2e.regional.run_ha005_rollout_continuity",
        "scripts.e2e.regional.run_ha006_executor_takeover",
    ):
        module = importlib.import_module(module_name)
        source = Path(module.__file__).read_text(encoding="utf-8")
        published = f"/scripts/{module.SCRIPT.name}"
        stale = [
            line.strip()
            for line in source.splitlines()
            if '"/scripts/' in line
            and "{SCRIPT.name}" not in line
            and published not in line
        ]
        assert not stale, f"{module_name} execs a probe it does not publish: {stale}"
        assert module.SCRIPT.is_file(), (
            f"{module_name} probe is missing: {module.SCRIPT}"
        )
        if "/scripts/" not in source:
            # NET-002/003 build their Pod through seeded_command_fixture, which
            # execs the name of the very ``script`` the runner hands it -- the
            # same file the ConfigMap publishes -- so the check moves there.
            fixture_source = Path(seeded_command_fixture.__file__).read_text(
                encoding="utf-8"
            )
            assert "script=SCRIPT" in source, (
                f"{module_name} must hand SCRIPT to the seeded probe"
            )
            assert "/scripts/{probe.script.name}" in fixture_source, (
                "seeded_command_fixture must exec the script name it publishes"
            )
            continue
        assert "/scripts/{SCRIPT.name}" in source, (
            f"{module_name} must exec SCRIPT.name"
        )


def test_runners_with_their_own_plan_builder_record_the_site_profile() -> None:
    """HA-001 once built its own plan without ``site_profile``.

    ``authorize_execution`` compares the plan against ``applied_site_profile()``
    at --execute time, so a runner that writes its own plan.json must record
    the profile or every --execute under a profile fails with plan drift.
    """
    regional = Path(__file__).resolve().parents[2] / "scripts" / "e2e" / "regional"
    offenders = []
    for path in sorted(regional.glob("run_*.py")):
        source = path.read_text(encoding="utf-8")
        if "def build_plan(" not in source or "authorize_execution(" not in source:
            continue
        if '"site_profile": applied_site_profile()' not in source:
            offenders.append(path.name)
    assert not offenders, f"plan builders without site_profile: {offenders}"


def test_boot020_resume_target_is_the_first_stage_without_a_passed_marker() -> None:
    # A stage only carries its ``_passed`` marker once every assertion in it
    # ran, so the marker -- not a mid-stage record -- is what proves a stage is
    # done and can be replayed.
    partial = {
        "stages": {
            "noop_passed": {},
            "control_plane_passed": {},
            "executor_classification": {},
            "executor_before": {},
        }
    }
    assert resume_target(partial) == (2, "executor")
    done = {"stages": {f"{stage}_passed": {} for stage in STAGES}}
    assert resume_target(done) == (len(STAGES), None)
    assert resume_target({"stages": {}}) == (0, "noop")


def test_boot020_auto_resume_reconverges_and_replays_passed_stages(
    tmp_path: Path,
) -> None:
    """--resume reads the evidence, converges the live state back to the failed
    stage's precondition, and restarts only that stage -- the semantics are
    unchanged, only the cost of a driver defect is."""

    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    first = FakeReleaseRollingBackend()
    recorder = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )
    run_release_rolling(first, recorder)

    # A driver assertion fails partway through the agent stage: it never reached
    # its end-of-stage marker and everything from it onward is stale.
    document = json.loads(evidence.read_text(encoding="utf-8"))
    for name in [
        key
        for key in document["stages"]
        if key == "agent" or key.startswith("agent_") or key.startswith("full")
    ]:
        document["stages"].pop(name)
    document["status"] = "FAILED"
    evidence.write_text(json.dumps(document), encoding="utf-8")

    # A fresh backend at the executor precondition -- what the agent stage's
    # auto-rollback left live -- reopens the same evidence.
    second = FakeReleaseRollingBackend()
    second.completed.update({"control_plane", "executor"})
    second.live["cpu_wheel"] = "cpu-v2"
    second.live["clusters"]["cluster-a"]["wheel"] = "executor-v2"
    second.gpu_generations["cluster-a"]["executor"] = 2
    second.cpu_generations = {"ingress": 2, "worker": 2}
    resumed = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )

    result = resume_release_rolling(second, resumed)

    assert result["status"] == "COMPLETED", result.get("status")
    assert result["stages"]["resumed_at_agent"]["skipped_stages"] == [
        "noop",
        "control_plane",
        "executor",
    ]
    # The convergence deployed the executor precondition once, and because the
    # live state already matched it, as a NOOP.
    assert result["convergence"][-1]["converged_to"] == "executor"
    assert [call for call in second.calls if call[0] == "executor"] == [
        ("executor", None, False, None, "NOOP")
    ]
    # The passed stages were replayed, never re-run against the site.
    assert not any(call[0] in {"noop", "control_plane"} for call in second.calls), (
        "a passed stage was re-run against the site instead of replayed"
    )
    # The agent stage ran again from its injected-failure classification, then
    # the full stage ran.
    assert second.calls == [
        ("executor", None, False, None, "NOOP"),
        ("agent", "data-converged", False, True, "DATA_PLANE_COMPATIBLE"),
        ("agent", None, False, None, "DATA_PLANE_COMPATIBLE"),
        ("full", "data-converged", False, True, "FULL"),
        ("full", None, False, None, "FULL"),
    ], second.calls
    # The resumed stage's ``before`` is observed afresh -- the convergence
    # deploy sits between it and any earlier snapshot -- and only the in-process
    # ``agent_after`` is reused for ``full_before``.
    assert second.snapshots == ["agent", "agent", "agent", "full", "full"], (
        second.snapshots
    )
    assert "reused_from" not in result["stages"]["agent_before"]
    assert result["stages"]["full_before"]["reused_from"] == "agent_after"


def test_boot020_auto_resume_over_completed_evidence_touches_nothing(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "GF-REGIONAL-BOOT-020.json"
    recorder = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )
    run_release_rolling(FakeReleaseRollingBackend(), recorder)

    second = FakeReleaseRollingBackend()
    resumed = EvidenceRecorder(
        evidence, case_id="GF-REGIONAL-BOOT-020", inputs={"configs": "test"}
    )

    result = resume_release_rolling(second, resumed)

    assert result["status"] == "COMPLETED"
    assert second.calls == [], "a completed run must not touch the site to resume"
