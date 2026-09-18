from __future__ import annotations

import copy
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import boot_acceptance_lifecycle as lifecycle
from tests.regional._cov95_boot_site import BootSite, report


def arguments(root: Path, *, retain: bool = False) -> Namespace:
    return Namespace(
        bootstrap_state_dir=root / "state",
        cpu_cluster_arn="arn:aws:eks:us-east-1:000000000000:cluster/cpu",
        gpu_cluster_arn=["arn:aws:eks:us-east-1:000000000000:cluster/gpu"],
        admin_email="unit@example.invalid",
        retain_bootstrap_site=retain,
    )


@pytest.mark.parametrize(
    "fault",
    ["", "deploy", "status", "gate", "generation", "missing-site", "registry", "phase"],
)
def test_boot016_requires_the_full_bootstrap_proof_and_cleans_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    model.failure = fault
    if fault == "registry":
        model.registry = ["another-cluster"]
    elif fault == "phase":
        model.phase = "failed"
    args = arguments(tmp_path)
    result = lifecycle.run_boot016(args, tmp_path / "case")
    assert result["verdict"] == ("FAIL" if fault else "PASS")
    removed = any(call[0][0] == "uninstall" for call in model.admin_calls)
    assert removed is bool(fault and fault != "missing-site")
    if not fault:
        assert all(result["checks"].values()), "complete bootstrap proof did not pass"
        assert result["cluster_id"] == "cluster-a"
        assert result["release_id"] == "unit-release"
        assert result["cleanup"]["retained"] is True
        assert model.deploys == 2
    if removed:
        assert result["checks"]["failure_cleanup"] is True
        uninstall = model.admin_calls[-1][0]
        assert uninstall[-2:] == ("--confirm", lifecycle.UNINSTALL_CONFIRMATION)
    if fault == "generation":
        assert result["checks"]["rerun_no_rollout_generation_change"] is False
    if fault == "missing-site":
        assert result["cleanup"]["site_present"] is False


def test_boot016_does_not_reuse_a_nonempty_state_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    args = arguments(tmp_path)
    args.bootstrap_state_dir.mkdir()
    (args.bootstrap_state_dir / "existing").write_text("owned before test")
    with pytest.raises(lifecycle.BootAcceptanceError, match="new or empty"):
        lifecycle.run_boot016(args, tmp_path / "case")
    assert model.admin_calls == []


@pytest.mark.parametrize(
    "fault", ["", "gate", "failed-status", "missing-path", "read-failure", "wrong-case"]
)
def test_boot017_reuses_only_a_readable_matching_successful_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    model.failure = fault
    args = arguments(tmp_path)
    path = tmp_path / "boot016.json"
    document = {
        "case_id": "GF-REGIONAL-BOOT-016",
        "verdict": "PASS",
        "state_dir": str(args.bootstrap_state_dir),
        "checks": {"status_passed": fault != "failed-status"},
    }
    if fault == "wrong-case":
        document["case_id"] = "GF-REGIONAL-BOOT-015"
    path.write_text(json.dumps(document))
    if fault == "read-failure":
        path.write_text("[]")
    predecessor = {
        "verdict": "PASS",
        "path": None if fault == "missing-path" else str(path),
    }
    result = lifecycle.run_boot017(args, predecessor, tmp_path / "case")
    expected = (
        "PARTIAL"
        if fault in {"missing-path", "read-failure", "wrong-case"}
        else "FAIL"
        if fault
        else "PASS"
    )
    assert result["verdict"] == expected
    assert model.admin_calls == []
    assert len(model.commands) == 2
    if fault == "read-failure":
        assert "cannot read" in result["boot016_reuse"]["reason"]


@pytest.mark.parametrize(
    "fault", ["", "artifact-drift", "status", "missing-agents", "empty-clusters"]
)
def test_boot018_compares_rebuilds_to_live_identity_before_recording_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    model = BootSite(tmp_path, monkeypatch)
    model.failure = fault
    args = arguments(tmp_path)
    args.bootstrap_state_dir.mkdir()
    (args.bootstrap_state_dir / "site.yaml").write_text("unit")
    if fault == "missing-agents":
        model.agents = []
    elif fault == "empty-clusters":
        model.config["clusters"] = []
    builds = []

    def build(mask: str, *, base: Path, case_dir: Path) -> dict[str, Any]:
        builds.append(mask)
        checkout = base / f"repo-{mask}"
        target = checkout / "src/gpu_fault/app/builtin_metric_contributors.py"
        target.parent.mkdir(parents=True)
        target.write_text("unit = 1\n")
        manifest = copy.deepcopy(model.manifest)
        if fault == "artifact-drift" and mask == "002":
            manifest["bundle_sha256"] = "d" * 64
        return {"checkout": checkout, "manifest": manifest}

    def tamper(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        model.commands.append((args, kwargs))
        return subprocess.CompletedProcess(args, 1, "unit tamper was rejected", "")

    monkeypatch.setattr(lifecycle, "build_release_under_umask", build)
    monkeypatch.setattr(lifecycle, "run", tamper)
    result = lifecycle.run_boot018(args, tmp_path / "case")
    assert sorted(builds) == ["002", "077"]
    assert result["verdict"] == ("FAIL" if fault else "PASS")
    assert result["checks"]["bootstrap_site_cleanup"] is True
    assert model.admin_calls[-1][0][0] == "uninstall"
    status = next(
        options for command, options in model.admin_calls if command[0] == "status"
    )
    assert status["environment_overrides"][
        lifecycle.QUICK_VALIDATION_EVIDENCE_ENV
    ].endswith("no-quick-evidence-reuse"), (
        "BOOT-018 must force fresh runtime identity evidence instead of quick-cache reuse"
    )
    if not fault:
        assert result["runtime_identity"]["replica_count"] == 2
        assert result["runtime_identity"]["agent_count"] == 1
    elif fault == "artifact-drift":
        assert result["checks"]["wheel_and_bundle_hashes_identical"] is False


@pytest.mark.parametrize("stage", ["build", "artifact"])
def test_umask_build_errors_stop_before_returning_a_manifest(
    tmp_path: Path, stage: str
) -> None:
    calls = []

    def copy_tree(checkout: Path) -> None:
        checkout.mkdir()

    def execute(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        failed = len(calls) == 1 if stage == "build" else len(calls) == 2
        return subprocess.CompletedProcess(args, int(failed), "", "synthetic failure")

    with pytest.raises(
        lifecycle.BootAcceptanceError, match="build failed|artifact tests failed"
    ):
        lifecycle.build_release_under_umask(
            "077",
            base=tmp_path,
            case_dir=tmp_path / "case",
            runner=execute,
            copy=copy_tree,
        )
    assert len(calls) == (1 if stage == "build" else 2)
    assert calls[0][1]["umask"] == 0o77


def test_checkout_copy_leaves_state_build_outputs_and_git_metadata_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    for name in (".git", ".venv", ".codex", "dist", "artifacts", "__pycache__"):
        (root / name).mkdir()
        (root / name / "ignored").write_text("private fixture")
    (root / "kept.txt").write_text("source fixture")
    monkeypatch.setattr(lifecycle, "ROOT", root)
    destination = tmp_path / "copied"
    lifecycle.copy_checkout(destination)
    assert [path.name for path in destination.iterdir()] == ["kept.txt"]
    assert (destination / "kept.txt").read_text() == "source fixture"


@pytest.mark.parametrize(
    "field", ["manifest", "metadata", "agent", "replicas", "check"]
)
def test_boot018_identity_rejects_missing_or_mismatched_components(field: str) -> None:
    from tests.regional.test_boot_acceptance_lifecycle import AGENTS, MANIFEST, METADATA

    manifest, metadata, agents = (
        copy.deepcopy(MANIFEST),
        dict(METADATA),
        copy.deepcopy(AGENTS),
    )
    status = report()
    if field == "manifest":
        manifest["components"]["executor"]["module_digest"] = ""
    elif field == "metadata":
        metadata["required-regional-executor-artifact-sha256"] = "f" * 64
        metadata["required-agent-config-digest"] = ""
        metadata["required-node-action-key-version"] = "not-a-version"
    elif field == "agent":
        agents[0]["compatibility_digest"] = "f" * 64
    elif field == "replicas":
        status["checks"][0]["details"] = {
            "control_plane": {"deployments": {}},
            "executor": {"clusters": {}},
        }
    else:
        status["checks"] = []
    result = lifecycle.runtime_identity_matches_release(
        status, manifest=manifest, metadata=metadata, agents=agents
    )
    assert result["passed"] is False
    assert result["reasons"], "an unproved identity was rejected without diagnostics"
