"""The legacy deploy tells the operator, twice, that what it creates is not in
the installation resource registry.

``deploy/hyperpod/deploy.sh`` writes no ``bootstrap-state.json`` and no
registry record. ``gpu-fault-admin uninstall --cpu-cluster-arn ...`` can still
rebuild ownership of the Aurora cluster, subnet group, security group and IRSA
roles -- but only by reading Secret ``<namespace>/gpu-fault-aurora`` from the
live namespace, and the cluster parameter group is registered by neither path.
The banner is rendered under bash here so the names it prints are the ones the
script actually uses.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy/hyperpod/deploy.sh"


def _banner(cluster_id: str, namespace: str) -> str:
    script = DEPLOY.read_text(encoding="utf-8")
    start = script.index("registry_warning() {")
    end = script.index("\n}\n", start) + 3
    completed = subprocess.run(
        ["bash", "-c", script[start:end] + "registry_warning"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "AURORA_CLUSTER_ID": cluster_id, "NAMESPACE": namespace},
    )
    assert completed.stdout == "", "the warning belongs on stderr, not in output"
    return completed.stderr


def test_the_warning_names_every_unregistered_resource_and_the_way_out() -> None:
    banner = _banner("gpu-fault-canary", "gpu-fault-canary-ns")

    assert "no installation resource registry" in banner
    assert "Aurora cluster gpu-fault-canary" in banner
    assert "security group gpu-fault-canary" in banner
    assert "gpu-fault-admin uninstall --cpu-cluster-arn" in banner
    assert "gpu-fault-canary-ns/gpu-fault-aurora" in banner
    assert "do not delete that Secret first" in banner
    assert "parameter group gpu-fault-canary-pg" in banner
    assert "only while it is the group" in banner


def test_the_warning_is_printed_at_aurora_time_and_in_the_deploy_summary() -> None:
    script = DEPLOY.read_text(encoding="utf-8")
    ensure = script.index("ensure_aurora() {")
    ensure_end = script.index("\n}\n", ensure)
    summary = script.index("Deployment workflow completed successfully")

    assert "registry_warning" in script[ensure:ensure_end]
    assert '[[ "${MODE}" != "deploy" ]] || registry_warning' in script[summary:]
