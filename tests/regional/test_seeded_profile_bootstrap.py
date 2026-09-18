from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALUES = {
    "GPU_FAULT_PERF_AWS_REGION": "us-east-2",
    "GPU_FAULT_PERF_CONTROL_NAMESPACE": "profile-control",
    "GPU_FAULT_PERF_DATAPLANE_NAMESPACE": "profile-data",
    "GPU_FAULT_PERF_IDENTITY_NAMESPACE": "profile-data",
    "GPU_FAULT_DATAPLANE_CONTEXT": "profile-context",
    "GPU_FAULT_CONTROL_KUBECONFIG": "/unit/profile-cpu",
    "KUBECONFIG": "/unit/profile-gpu",
}


@pytest.mark.parametrize(
    "entrypoint",
    [
        "scripts.e2e.regional.run_net002_command_recovery",
        "scripts.e2e.regional.run_net003_result_retry",
        "scripts.e2e.regional.run_net006_lease_loss_withheld_result",
    ],
)
@pytest.mark.parametrize("inherited_region", [None, "us-west-1"])
def test_profile_is_installed_before_routing_helpers_capture_targets(
    tmp_path: Path, entrypoint: str, inherited_region: str | None
) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"environment": VALUES}))
    profile.chmod(0o600)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in VALUES and key != "GPU_FAULT_ACCEPTANCE_SITE_PROFILE"
    }
    if inherited_region is not None:
        environment["GPU_FAULT_PERF_AWS_REGION"] = inherited_region
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "src")
    code = """
import importlib,json,os,sys
sys.argv = [sys.argv[1], "--site-profile", sys.argv[2]]
importlib.import_module(sys.argv[0])
from scripts.e2e.regional import seeded_command_fixture as fixture
print(json.dumps({
    "region": fixture.AWS_REGION,
    "control_namespace": fixture.CONTROL_NAMESPACE,
    "data_namespace": fixture.NAMESPACE,
    "data_context": fixture.DATAPLANE_CONTEXT,
    "cpu_config": os.environ["GPU_FAULT_CONTROL_KUBECONFIG"],
    "gpu_config": os.environ["KUBECONFIG"],
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, entrypoint, str(profile)],
        env=environment,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "region": inherited_region or "us-east-2",
        "control_namespace": "profile-control",
        "data_namespace": "profile-data",
        "data_context": "profile-context",
        "cpu_config": "/unit/profile-cpu",
        "gpu_config": "/unit/profile-gpu",
    }
