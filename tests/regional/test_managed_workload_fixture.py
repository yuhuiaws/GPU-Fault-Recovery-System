"""ManagedWorkloadFixture: the test workload the destructive cases submit."""

from __future__ import annotations

from pathlib import Path

from scripts.e2e.regional.managed_workload_fixture import (
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)

REGIONAL = Path(__file__).resolve().parents[2] / "scripts" / "e2e" / "regional"


def _regional(tmp_path: Path) -> RegionalLiveFixture:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        )
    )


def test_managed_workload_delete_also_removes_restarted_copies(
    tmp_path: Path, monkeypatch
) -> None:
    """RESTART_WORKLOAD resubmits the job as ``<name>-r-<hash>`` with the same
    ``gpu-fault.io/job-id`` label; a name-only delete left that copy holding a
    GPU for two hours and failed every later live preflight (DESTR-015)."""
    site = tmp_path / "site.yaml"
    site.write_text("schemaVersion: 1\n", encoding="utf-8")
    regional = _regional(tmp_path)
    calls = []
    monkeypatch.setattr(
        regional, "kubectl", lambda *args, **kwargs: calls.append(args) or ""
    )
    fixture = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=(REGIONAL / "manifests/training/xid11-single-node-job.yaml"),
            site_file=site,
            job_id="destr012-d-abc",
            attempt_id="destr012-d-abc-a001",
            restart_budget=1,
            expected_pods=1,
            expected_gpu_count=1,
        ),
    )

    fixture.delete()

    assert [call[:4] for call in calls] == [
        ("gpu", "delete", "job", "gpu-fault-xid11-auto-resume-guard"),
        ("gpu", "delete", "job", "-l"),
    ], calls
    assert "gpu-fault.io/job-id=destr012-d-abc" in calls[1], calls[1]
