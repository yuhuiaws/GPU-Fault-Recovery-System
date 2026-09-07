"""Behavioural form of the review item-7 preflight/plan contracts.

Every DESTR runner's ``read_only_preflight`` binds the preflight to the release
and cluster it was read against (``evidence_identity``), hands that identity to
the predecessor check, and passes the ``reuse_focused_tests`` flag through to
``focused_tests``; every ``plan_details`` records the focused-test result with
its source digest so ``--execute`` can reuse it. These tests drive the real
functions with a fake fixture and spies instead of grepping their source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional import run_destr001_gpu_reset as destr001
from scripts.e2e.regional import run_destr002_hyperpod_reboot as destr002
from scripts.e2e.regional import run_destr009_workload_restart as destr009
from scripts.e2e.regional import run_destr010_fabric_manager_restart as destr010
from scripts.e2e.regional import run_destr012_managed_recovery_guard as destr012
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings

IDENTITY = {"release_id": "release-sentinel", "cluster_id": "cluster-sentinel"}
PROBE_IMAGE = "registry.example/probe@sha256:" + "a" * 64
PREDECESSOR_MODULES = [destr001, destr002, destr009, destr012]
ALL_MODULES = [*PREDECESSOR_MODULES, destr010]


def _node(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "uid": f"uid-{name}",
        "boot_id": f"boot-{name}",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }


class FakeRegional:
    """The read-only surface the five preflights touch, with fixed answers."""

    def __init__(self, _settings: Any) -> None:
        pass

    def evidence_identity(self) -> dict[str, str]:
        return dict(IDENTITY)

    def store_snapshot(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "release_id": "release-store",
            "agent": {"lifecycle_state": "ACTIVE", "generation": 3},
            "profile": {"profile_version": "profile-a", "capabilities": []},
            "queue": {"depth": 0},
            "remote_commands": {"open_by_cluster": {}},
        }

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return _node(node)

    def business_workloads(self, _node: str) -> list[dict[str, str]]:
        return []

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {}

    def executor_python(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"positive": {"targets": []}, "negative": {}}

    def gpu_nodes(self) -> list[dict[str, Any]]:
        return [_node("gpu-1"), _node("gpu-2"), _node("gpu-3")]

    def gpu_workloads(self) -> list[dict[str, Any]]:
        return []

    def runtime_identity(self) -> dict[str, Any]:
        return {}


def _regional_settings(tmp_path: Path) -> RegionalLiveSettings:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return RegionalLiveSettings(
        cpu_kubeconfig=cpu,
        gpu_kubeconfig=gpu,
        gpu_context="gpu-context",
        namespace="gpu-fault-system",
        cluster_id="cluster-a",
        region="us-west-2",
    )


def _fixture_file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("apiVersion: v1\n", encoding="utf-8")
    return path


def _settings(module: Any, tmp_path: Path) -> Any:
    regional = _regional_settings(tmp_path)
    predecessor = tmp_path / "predecessor.json"
    if module is destr001:
        return destr001.Settings(
            regional=regional,
            node="node-a",
            host_probe_image=PROBE_IMAGE,
            predecessor_path=predecessor,
        )
    if module is destr002:
        return destr002.Settings(
            regional=regional,
            node="node-a",
            host_probe_image=PROBE_IMAGE,
            hyperpod_cluster="hp-a",
            executor_role_arn="arn:aws:iam::000000000000:role/executor-a",
            predecessor_path=predecessor,
        )
    if module is destr009:
        return destr009.Settings(
            regional=regional,
            site_file=_fixture_file(tmp_path, "site.yaml"),
            manifest=_fixture_file(tmp_path, "training.yaml"),
            job_id="job-a",
            attempt_id="job-a-a001",
            predecessor_path=predecessor,
        )
    if module is destr010:
        return destr010.Settings(
            regional=regional, node="node-a", host_probe_image=PROBE_IMAGE
        )
    return destr012.Settings(
        regional=regional,
        site_file=_fixture_file(tmp_path, "site.yaml"),
        a_manifest=_fixture_file(tmp_path, "a.yaml"),
        d_manifest=_fixture_file(tmp_path, "d.yaml"),
        a_job_id="job-a",
        a_attempt_id="job-a-a001",
        d_job_id="job-d",
        d_attempt_id="job-d-a001",
        predecessor_path=predecessor,
    )


def _isolate(module: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Swap the live collaborators for fakes; return the spy log."""

    seen: dict[str, Any] = {}
    monkeypatch.setattr(module, "RegionalLiveFixture", FakeRegional)

    def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
        seen["focused_reuse"] = reuse
        return {"passed": True, "focused_tests_reused": reuse}

    monkeypatch.setattr(module, "focused_tests", focused_tests)
    if module is destr010:
        monkeypatch.setattr(
            module,
            "fabric_probe",
            lambda *_a, **_k: {
                "active_workflow_incidents": [],
                "recent_xid_events": [],
            },
        )
    else:

        def predecessor_evidence(path: Path, case_id: str, **identity: Any) -> Any:
            seen["predecessor_identity"] = dict(identity)
            seen["predecessor_case_id"] = case_id
            return {"valid": True, "verdict": "PASS", "path": str(path)}

        monkeypatch.setattr(module, "predecessor_evidence", predecessor_evidence)
    if module is destr012:
        monkeypatch.setattr(
            module, "group_b_audit", lambda *_a, **_k: {"errors": [], "profile": {}}
        )
    return seen


@pytest.mark.parametrize("module", ALL_MODULES, ids=lambda m: m.__name__[-30:])
@pytest.mark.parametrize("reuse", [True, False])
def test_preflight_binds_the_evidence_identity_and_forwards_focused_reuse(
    module: Any, reuse: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _isolate(module, monkeypatch)
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    result = module.read_only_preflight(
        _settings(module, tmp_path), case_dir, reuse_focused_tests=reuse
    )

    # Identity comes from the fixture's evidence_identity(), not the store.
    assert result["evidence_identity"] == IDENTITY, module.__name__
    assert result["release_id"] == "release-store", module.__name__
    assert seen["focused_reuse"] is reuse, module.__name__
    assert result["focused_tests"]["focused_tests_reused"] is reuse, module.__name__


@pytest.mark.parametrize("module", PREDECESSOR_MODULES, ids=lambda m: m.__name__[-30:])
def test_preflight_checks_the_predecessor_against_the_same_identity(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _isolate(module, monkeypatch)
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    result = module.read_only_preflight(_settings(module, tmp_path), case_dir)

    # ``**identity`` reaches predecessor_evidence, so a predecessor PASS earned
    # against another release or cluster is refused.
    assert seen["predecessor_identity"] == IDENTITY, module.__name__
    assert seen["predecessor_case_id"] == module.PREDECESSOR_CASE_ID, module.__name__
    assert result["predecessor"]["valid"] is True, module.__name__


def test_destr010_has_no_predecessor_to_bind() -> None:
    assert not hasattr(destr010, "PREDECESSOR_CASE_ID"), (
        "DESTR-010 opens the destructive chain; it must not name a predecessor"
    )


@pytest.mark.parametrize("module", ALL_MODULES, ids=lambda m: m.__name__[-30:])
def test_plan_details_record_the_focused_tests_with_their_source_digest(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(module, monkeypatch)
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "digest-sentinel")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    settings = _settings(module, tmp_path)
    preflight = module.read_only_preflight(settings, case_dir, reuse_focused_tests=True)

    details = module.plan_details(settings, preflight)

    assert details["focused_tests"] == preflight["focused_tests"], module.__name__
    assert details["focused_tests_source_digest"] == "digest-sentinel", module.__name__
    assert details["preflight"]["evidence_identity"] == IDENTITY, module.__name__
