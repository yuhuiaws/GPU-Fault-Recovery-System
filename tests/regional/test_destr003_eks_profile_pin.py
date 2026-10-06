"""DESTR-003 on a mixed site stamps and names the cluster's own Runtime Profile.

The workload's observation carried the site's profile version on the eks
cluster, the synthetic finding repeated it, and the plan compiled REPLACE_NODE
for the HyperPod adapter. The runner now stamps the workload with the
cluster's effective profile (the preflight's store reading) and names it in
the synthetic finding; the control plane re-pins either way.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gpu_fault import training_submit_cli
from scripts.e2e.regional import run_destr003_warm_spare_failover as destr003
from scripts.e2e.regional.managed_workload_fixture import (
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)


def _settings() -> Any:
    return SimpleNamespace(
        regional=SimpleNamespace(cluster_id="gpu-eks-cluster"),
        fault_node="node-a",
        job_id="destr003-job",
        attempt_id="destr003-job-a001",
    )


def test_the_synthetic_finding_names_the_clusters_effective_profile() -> None:
    observation = {
        "runtime_profile_version": "hyperpod-v1",
        "workload_ids": ["training/pytorchjob/destr003"],
        "containers": [{"gpu_uuids": ["GPU-a"]}],
    }
    now = datetime.now(timezone.utc)
    repinned = destr003.replacement_payload(
        _settings(),
        event_id="e-1",
        observation=observation,
        observed_at=now,
        profile_version="regional-eks-abc",
    )
    assert repinned["runtime_profile_version"] == "regional-eks-abc"
    assert repinned["cluster_id"] == "gpu-eks-cluster"
    legacy = destr003.replacement_payload(
        _settings(), event_id="e-1", observation=observation, observed_at=now
    )
    assert legacy["runtime_profile_version"] == "hyperpod-v1", (
        "without a cluster reading the observation's version is kept"
    )


def test_the_managed_workload_is_stamped_with_the_pinned_profile(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = tmp_path / "workload.yaml"
    manifest.write_text(
        "apiVersion: kubeflow.org/v1\nkind: PyTorchJob\nmetadata:\n  name: job\n"
        "  namespace: training\nspec: {}\n",
        encoding="utf-8",
    )
    site = tmp_path / "site.yaml"
    site.write_text("site\n", encoding="utf-8")
    regional = SimpleNamespace(
        settings=SimpleNamespace(
            gpu_kubeconfig=tmp_path / "kube", gpu_context="gpu", namespace="training"
        )
    )
    seen: list[list[str]] = []

    def run(args: Any, *, runner: Any) -> int:
        seen.append([args.runtime_profile_version, str(args.site), args.job_id])
        return 0

    # ``submit`` imports the CLI module locally; patching the module's ``run``
    # intercepts exactly the call it makes.
    monkeypatch.setattr(training_submit_cli, "run", run)
    stamped = ManagedWorkloadFixture(
        regional,  # type: ignore[arg-type]
        ManagedWorkloadSettings(
            manifest=manifest,
            site_file=site,
            job_id="job-a",
            attempt_id="job-a-a001",
            restart_budget=1,
            expected_pods=1,
            expected_gpu_count=8,
            runtime_profile_version="regional-eks-abc",
        ),
    )
    stamped.submit()
    unstamped = ManagedWorkloadFixture(
        regional,  # type: ignore[arg-type]
        ManagedWorkloadSettings(
            manifest=manifest,
            site_file=site,
            job_id="job-b",
            attempt_id="job-b-a001",
            restart_budget=1,
            expected_pods=1,
            expected_gpu_count=8,
        ),
    )
    unstamped.submit()
    assert seen == [
        ["regional-eks-abc", str(site), "job-a"],
        [None, str(site), "job-b"],
    ], seen
