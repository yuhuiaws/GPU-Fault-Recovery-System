from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from scripts.e2e.regional.audit_executor_readiness import validate_readiness_matrix
from scripts.e2e.regional.audit_regional_command_protocol_live import AUDITED_CASE_IDS
from scripts.e2e.regional.run_boot019_admin_lifecycle import run_admin_lifecycle
from scripts.e2e.regional.run_boot020_release_rolling import run_release_rolling

ROOT = Path(__file__).resolve().parents[2]


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
    assert "'bounded_sequences': [12, 13, 14]" in result.stdout
    assert "'unwritable_buffered': False" in result.stdout


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
                "data_plane": "DATA_PLANE_COMPATIBLE",
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
        if scenario == "data_plane" and fault_phase:
            return {"phase": "failed", "injected_failure": fault_phase}
        if scenario == "full" and fault_phase:
            return {"phase": "rolled-back", "injected_failure": fault_phase}
        if scenario == "control_plane":
            self.live["cpu_wheel"] = "cpu-v2"
            self.cpu_generations = {"ingress": 2, "worker": 2}
        elif scenario == "data_plane":
            self.live["clusters"]["cluster-a"]["wheel"] = "executor-v2"
            self.gpu_generations["cluster-a"]["executor"] = 2
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
        return {"phase": "complete", "injected_failure": None}


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
        ("control_plane", None, False, None, "CONTROL_PLANE_ONLY"),
        ("data_plane", "cpu-staged", False, False, "DATA_PLANE_COMPATIBLE"),
        ("data_plane", None, True, False, "DATA_PLANE_COMPATIBLE"),
        ("full", "data-converged", False, True, "FULL"),
        ("full", None, False, None, "FULL"),
    ]
    assert (
        result["stages"]["full_rollback_snapshot"]["live"]
        == (result["stages"]["full_before"]["live"])
    )
    assert evidence.stat().st_mode & 0o777 == 0o600
