"""The deploy-host identity covers everything the release engine runs from the snapshot.

Live 2026-09-13: a fix to an in-Pod probe (deploy/control-plane/regional/probes)
changed neither the application wheels nor the deploy-host bundle, so `deploy`
classified it QUALITY_ONLY, applied nothing, and the site kept running the old
probe. The engine's shell entry points, probes and control-plane tools are
orchestration inputs, so such a change refreshes the deploy host and snapshot.
"""

from __future__ import annotations

from pathlib import Path

from scripts import deploy_source_identity as MODULE

ROOT = Path(__file__).resolve().parents[1]


def test_engine_probes_scripts_and_tools_are_deploy_host_orchestration_inputs() -> None:
    files = MODULE.deploy_host_identity(ROOT)["orchestration"]["files"]

    for expected in (
        "deploy/control-plane/regional/probes/gpu_endpoint_gate.py",
        "deploy/control-plane/regional/prepare-clean-redeploy.sh",
        "deploy/control-plane/regional/prepare-clean-redeploy-delete.sh",
        "deploy/control-plane/regional/rollout-regional-release.sh",
        "deploy/control-plane/tools/ensure-postgres-schema.sh",
        "deploy/control-plane/tools/cleanup_state.py",
        "src/gpu_fault_release/rollout.py",
    ):
        assert expected in files, f"{expected} must move the deploy-host identity"
