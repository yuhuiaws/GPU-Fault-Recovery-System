from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gpu_fault.admin import bootstrap_checkpoint
from gpu_fault.admin.bootstrap_checkpoint import bind_bootstrap_inputs
from gpu_fault.admin.bootstrap_common import BootstrapRequest, BootstrapState
from tests.admin._bootstrap_support import _cluster


def test_ca_binding_change_invalidates_readiness_without_unrelated_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu = _cluster()
    gpu = replace(
        cpu,
        role="gpu",
        eks_name="gpu-a",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        hyperpod_name="gpu-a",
        hyperpod_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a",
    )
    root = tmp_path / "repository"
    monkeypatch.setattr(
        bootstrap_checkpoint, "RDS_CA_BUNDLE_PATH", "/etc/example/first.pem"
    )
    request = BootstrapRequest(
        cpu_cluster_arn=cpu.input_arn,
        gpu_cluster_arns=(gpu.eks_arn,),
        repository_root=root,
        state_dir=tmp_path / "state",
    )
    state = BootstrapState(request.state_dir / "bootstrap.json", site_id="site-a")

    bind_bootstrap_inputs(state, request=request, cpu=cpu, gpu_clusters=(gpu,))
    before = dict(state.value["task_input_sha256"])
    for task in ("aurora_ready", "pki"):
        state.record(task, {})
        state.complete(task)
    monkeypatch.setattr(
        bootstrap_checkpoint, "RDS_CA_BUNDLE_PATH", "/etc/example/second.pem"
    )
    bind_bootstrap_inputs(state, request=request, cpu=cpu, gpu_clusters=(gpu,))

    after = state.value["task_input_sha256"]
    assert before["aurora_ready"] != after["aurora_ready"], (
        "a changed CA binding must invalidate the old DSN readiness checkpoint"
    )
    assert "aurora_ready" not in state.value["completed_tasks"]
    assert before["pki"] == after["pki"]
    assert "pki" in state.value["completed_tasks"]
