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
        "deploy/control-plane/tools/cleanup_kubernetes.py",
        "deploy/control-plane/regional/rollout-regional-release.sh",
        "deploy/control-plane/tools/ensure-postgres-schema.sh",
        "deploy/control-plane/tools/cleanup_state.py",
        "src/gpu_fault_release/rollout.py",
    ):
        assert expected in files, f"{expected} must move the deploy-host identity"


def test_every_release_engine_module_is_a_deploy_host_orchestration_input() -> None:
    """``scripts/release_deploy.py`` and the modules split from it run from the
    site snapshot. ``release_deploy_profile.py`` (the Runtime Profile plan and
    approval of a release) was in no identity input, so a change to it alone
    left both identities unchanged and the deploy applied nothing."""

    files = MODULE.deploy_host_identity(ROOT)["orchestration"]["files"]
    engine = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts").glob("release_deploy*.py")
    )

    assert "scripts/release_deploy_profile.py" in engine
    missing = [path for path in engine if path not in files]
    assert missing == [], (
        f"release engine modules outside the deploy-host identity: {missing}"
    )
