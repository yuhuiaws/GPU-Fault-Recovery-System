from __future__ import annotations

from typing import Any

import pytest

from gpu_fault import node_installer_reconciler as installer
from tests.node_agent import test_node_installer_reconciler as existing


@pytest.fixture(autouse=True)
def isolated_installer_io(tmp_path, monkeypatch):
    path = str(tmp_path / "reconciler-heartbeat")
    monkeypatch.setattr(installer, "RECONCILER_HEARTBEAT_PATH", path)
    monkeypatch.setattr(existing, "RECONCILER_HEARTBEAT_PATH", path)
    return path


def make_reconciler(core=None, batch=None, **options: Any):
    arguments = {
        "namespace": "gpu-fault-system",
        "cluster_name": "hp-cluster-a",
        "version": "0.10.0",
        "config_digest": "config-sha",
        "artifact_sha256": existing.ARTIFACT,
        "bundle_sha256": existing.BUNDLE,
        "template_sha256": existing.TEMPLATE,
        "job_template": existing.template(),
        "dcgm_metrics_url_template": "http://{node_ip}:9400/metrics",
        "now": lambda: existing.NOW,
        "agent_alive": lambda address, port: False,
    }
    arguments.update(options)
    return installer.NodeInstallerReconciler(
        existing.CoreApi([]) if core is None else core,
        existing.BatchApi() if batch is None else batch,
        **arguments,
    )


def node_document(**annotations: str) -> dict[str, Any]:
    return {
        "metadata": {
            "name": "hyperpod-i-123",
            "uid": "node-uid",
            "labels": {"node.kubernetes.io/instance-type": "ml.p5en.48xlarge"},
            "annotations": annotations,
        },
        "status": {
            "nodeInfo": {"bootID": "boot-current"},
            "conditions": [
                {"type": "DiskPressure", "status": "False"},
                {"type": "Ready", "status": "True"},
            ],
            "addresses": [
                {"type": "ExternalIP", "address": "192.0.2.1"},
                {"type": "InternalIP", "address": "10.0.1.25"},
            ],
        },
    }
