from __future__ import annotations

import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import capacity_acceptance_base as base

APPS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
    "gpu-fault-adot",
)


def deployment(name):
    return {
        "metadata": {"name": name, "generation": 1},
        "spec": {
            "replicas": 1,
            "template": {
                "spec": {
                    "serviceAccountName": "cpu-service-account",
                    "containers": [
                        {
                            "name": "api",
                            "image": "runtime@sha256:" + "a" * 64,
                            "args": ["uvicorn", "--workers", "2"],
                            "volumeMounts": [
                                {
                                    "name": "rds-ca-bundle",
                                    "mountPath": "/rds",
                                    "readOnly": True,
                                }
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "rds-ca-bundle",
                            "configMap": {"name": "public-rds-ca"},
                        }
                    ],
                }
            },
        },
        "status": {"readyReplicas": 1},
    }


def pod(name, app):
    return {
        "metadata": {"name": name, "uid": name + "-uid", "labels": {"app": app}},
        "spec": {"containers": [{"name": "api"}], "nodeName": "cpu-node"},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [{"name": "api", "ready": True, "restartCount": 0}],
        },
    }


class FakeApi:
    def __init__(self):
        self.calls = []
        self.applied = []
        self.deployments = [deployment(name) for name in APPS]
        self.pods = [pod(name + "-pod", name) for name in APPS]
        self.services = []
        self.processor_mode = "active-active"
        self.release_metadata = {
            "required-regional-executor-artifact-sha256": "e" * 64,
            "required-regional-executor-compatibility-digest": "f" * 64,
            "compatible-regional-executor-artifact-sha256s": "",
        }
        self.pool_size = "8"
        self.exec_output = "100\n"
        self.get_override = None
        self.fail = None
        self.probe_pods = None
        self.deleted = set()

    def run(self, *args, **options):
        self.calls.append((args, options))
        if self.fail is not None and self.fail(args):
            raise base.CapError("simulated API failure")
        if args[:3] == (
            "get",
            "configmap",
            "gpu-fault-control-worker-config-processor",
        ):
            value = {"data": {"GPU_FAULT_PROCESSOR_MODE": self.processor_mode}}
        elif args[:3] == ("get", "configmap", "gpu-fault-release-metadata"):
            value = {"data": self.release_metadata}
        elif args[:3] == ("get", "deployment", "gpu-fault-control-worker"):
            value = self.deployments[1]
        elif args[:2] == ("get", "deployments"):
            value = {"items": self.deployments}
        elif args[:2] == ("get", "services"):
            value = {"items": self.services}
        elif args[:2] == ("get", "configmap") and args[2].endswith("-config-postgres"):
            value = {
                "items": [
                    {
                        "metadata": {"name": name + "-config-postgres"},
                        "data": {"GPU_FAULT_POSTGRES_POOL_MAX_SIZE": self.pool_size},
                    }
                    for name in APPS[:3]
                ]
            }
        elif args[:2] == ("get", "pods"):
            selector = args[args.index("-l") + 1] if "-l" in args else ""
            if selector.startswith("gpu-fault.io/capacity-probe="):
                owned = [
                    item
                    for item in self.applied
                    if item["kind"] == "Deployment"
                    and item["metadata"]["name"] not in self.deleted
                ]
                generated = [
                    pod(item["metadata"]["name"] + "-pod", "gpu-fault-control-worker")
                    for item in owned
                ]
                for item in generated:
                    key, selected = selector.split("=", 1)
                    item["metadata"]["labels"][key] = selected
                value = {
                    "items": self.probe_pods
                    if self.probe_pods is not None
                    else generated
                }
            elif selector:
                key, selected = selector.split("=", 1)
                value = {
                    "items": [
                        item
                        for item in self.pods
                        if item["metadata"]["labels"].get(key) == selected
                    ]
                }
            else:
                value = {"items": self.pods}
        elif args[:1] == ("get",):
            return subprocess.CompletedProcess(args, 0, self.get_override or "", "")
        elif args[:1] == ("apply",):
            self.applied.append(json.loads(options["input_text"]))
            return subprocess.CompletedProcess(args, 0, "", "")
        elif args[:1] == ("exec",):
            return subprocess.CompletedProcess(args, 0, self.exec_output, "")
        elif args[:1] == ("delete",):
            self.deleted.add(args[2])
            return subprocess.CompletedProcess(args, 0, "", "")
        else:
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(
            args, 0, json.dumps(copy.deepcopy(value)), ""
        )


class Harness(base.CapHarnessBase):
    def kubectl(self, *args, **options):
        return self.transport.run(*args, **options)


@pytest.fixture(name="capacity")
def capacity_fixture(tmp_path, monkeypatch):
    api = FakeApi()
    config = {
        "cpu_kubeconfig": str(tmp_path / "cpu-kubeconfig"),
        "namespace": "capacity-test",
        "aws_region": "us-east-1",
        "health": {
            "amp_workspace_id": "unit-workspace",
            "aurora_cluster_id": "unit-database",
        },
    }
    monkeypatch.setattr(
        base, "load_site", lambda path: SimpleNamespace(release_config=config)
    )
    monkeypatch.setattr(
        base,
        "run_fixture_command",
        lambda *args, **kwargs: pytest.fail("external command"),
    )
    monkeypatch.setattr(
        base.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("external process"),
    )
    monkeypatch.setattr(
        base.boto3, "client", lambda *args, **kwargs: pytest.fail("external AWS client")
    )
    monkeypatch.setattr(
        base.boto3,
        "Session",
        lambda *args, **kwargs: pytest.fail("ambient AWS credentials"),
    )
    monkeypatch.setattr(
        base.httpx, "get", lambda *args, **kwargs: pytest.fail("external HTTP")
    )
    monkeypatch.setattr(
        base.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("external URL"),
    )
    harness = Harness.__new__(Harness)
    harness.transport = api
    harness.__init__(
        site_path=tmp_path / "site.yaml",
        run_dir=tmp_path / "run",
        case_id="GF-REGIONAL-CAP-001",
        predecessor={"valid": True},
    )
    return harness, api
