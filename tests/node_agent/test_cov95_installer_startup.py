from __future__ import annotations

import hashlib
import runpy
import sys
from types import SimpleNamespace

import pytest
import yaml
from kubernetes import client, config

from gpu_fault import logging_setup
from gpu_fault import node_installer_reconciler as installer
from tests.node_agent._cov95_installer_support import (
    isolated_installer_io as isolated_installer_io,
)
from tests.node_agent.test_node_installer_reconciler import (
    ARTIFACT,
    BUNDLE,
    TEMPLATE,
    ApiError,
    BatchApi,
    CoreApi,
    template,
)


class StopLoop(BaseException):
    pass


def install_startup_fakes(monkeypatch, tmp_path, *, from_file):
    text = yaml.safe_dump(template()).replace("\n", "\r\n")
    environment = {
        "GPU_FAULT_HYPERPOD_CLUSTER": "hp-cluster-a",
        "GPU_FAULT_INSTALLER_VERSION": "0.10.0",
        "GPU_FAULT_INSTALLER_CONFIG_DIGEST": "config-sha",
        "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": ARTIFACT,
        "GPU_FAULT_INSTALLER_BUNDLE_SHA256": BUNDLE,
        "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": TEMPLATE,
        installer.TEMPLATE_CONTENT_SHA256_ENV: hashlib.sha256(
            text.encode()
        ).hexdigest(),
        "GPU_FAULT_NODE_INSTALLER_METRICS_PORT": "0",
        "GPU_FAULT_INSTALLER_ALLOWED_NODES": " node-a, node-b, " if from_file else "*",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    if from_file:
        path = tmp_path / "job.yaml"
        path.write_bytes(text.encode())
        monkeypatch.setenv("GPU_FAULT_INSTALLER_TEMPLATE_PATH", str(path))

    class StartupCore(CoreApi):
        def __init__(self):
            super().__init__([])
            self.template_reads = []
            self.passes = 0

        def read_namespaced_config_map(self, name, namespace, _request_timeout=None):
            self.template_reads.append((name, namespace, _request_timeout))
            return SimpleNamespace(data={"job.yaml": text})

        def list_node(self, **kwargs):
            self.passes += 1
            if self.passes == 1:
                raise ApiError(503)
            return super().list_node(**kwargs)

    core, batch = StartupCore(), BatchApi()
    configured = []
    monkeypatch.setattr(
        config, "load_incluster_config", lambda: configured.append(True)
    )
    monkeypatch.setattr(client, "CoreV1Api", lambda: core)
    monkeypatch.setattr(client, "BatchV1Api", lambda: batch)
    monkeypatch.setattr(installer, "configure_logging", lambda: None)
    monkeypatch.setattr(logging_setup, "configure_logging", lambda: None)
    metrics_calls = []
    monkeypatch.setattr(
        installer,
        "start_metrics_server",
        lambda family, **kwargs: metrics_calls.append((family, kwargs)),
    )
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise StopLoop

    monkeypatch.setattr(installer.time, "sleep", sleep)
    return core, batch, configured, metrics_calls, sleeps


@pytest.mark.parametrize("from_file", [True, False])
def test_startup_validates_pinned_template_and_recovers_from_list_failure(
    tmp_path, monkeypatch, caplog, from_file
) -> None:
    core, batch, configured, metrics, sleeps = install_startup_fakes(
        monkeypatch, tmp_path, from_file=from_file
    )
    with pytest.raises(StopLoop):
        installer.main()
    assert configured == [True]
    assert core.passes == 2
    assert sleeps == [5, 5]
    assert len(core.template_reads) == (0 if from_file else 1)
    if core.template_reads:
        assert core.template_reads[0] == (
            "gpu-fault-node-installer-template",
            "gpu-fault-system",
            installer.REQUEST_TIMEOUT,
        )
    assert len(metrics) == 1
    assert metrics[0][1]["port"] == 0
    assert metrics[0][1]["health"] is installer.heartbeat_is_fresh
    assert "node installer reconcile failed" in caplog.text
    assert batch.created == []


def test_startup_requires_cluster_identity_before_loop_or_metrics(
    tmp_path, monkeypatch
) -> None:
    core, batch, _, metrics, sleeps = install_startup_fakes(
        monkeypatch, tmp_path, from_file=False
    )
    monkeypatch.delenv("GPU_FAULT_HYPERPOD_CLUSTER")
    with pytest.raises(RuntimeError, match="GPU_FAULT_HYPERPOD_CLUSTER is required"):
        installer.main()
    assert core.passes == 0
    assert metrics == []
    assert sleeps == []
    assert batch.created == []


def test_module_entrypoint_refuses_missing_identity_with_fake_kubernetes(
    tmp_path, monkeypatch
) -> None:
    core, batch, _, _, sleeps = install_startup_fakes(
        monkeypatch, tmp_path, from_file=False
    )
    monkeypatch.delenv("GPU_FAULT_HYPERPOD_CLUSTER")
    monkeypatch.delitem(sys.modules, installer.__name__)
    with pytest.raises(RuntimeError, match="GPU_FAULT_HYPERPOD_CLUSTER is required"):
        runpy.run_module(installer.__name__, run_name="__main__")
    assert core.passes == 0
    assert sleeps == []
    assert batch.created == []
