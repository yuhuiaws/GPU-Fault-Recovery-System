from __future__ import annotations

import base64
import builtins
import copy
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import Request

import pytest

from scripts.e2e.regional.auth015_deployed import AGENT_PROBE
from scripts.e2e.regional.identity_acceptance_common import ClusterTarget
from tests.regional._cov95_auth015_release import ReleaseFiles
from tests.regional._cov95_auth015_support import AgentPair, Response

API_TOKEN = "synthetic-cpu-only-" + "t" * 48


class LiveSite:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.files = ReleaseFiles(root, monkeypatch)
        self.pair = AgentPair(root, monkeypatch)
        self.namespace = "gpu-fault-system"
        self.target = ClusterTarget(
            "cluster-a",
            "context-a",
            "test-1",
            "hyperpod-a",
            "arn:aws:eks:test-1:111122223333:cluster/test-a",
            "arn:aws:iam::111122223333:role/test-executor",
            "https://control.invalid",
            root / "public-ca.pem",
        )
        self.events: list[tuple[str, ...]] = []
        self.api_requests = []
        self.capture_count = 0
        self.restrict_cpu_imports = False
        self.after_capture = lambda: None
        image = self.files.delivery["images"]["runtime"]["reference"]
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "api-a",
                "uid": "pod-uid-a",
                "namespace": self.namespace,
                "labels": {"app": "gpu-fault-api-ha"},
            },
            "spec": {
                "containers": [{"name": "api", "image": image}],
                "nodeName": "cpu-a",
            },
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [
                    {
                        "name": "api",
                        "ready": True,
                        "state": {"running": {"startedAt": "2026-01-01T00:00:00Z"}},
                        "imageID": "docker-pullable://" + image,
                        "containerID": "containerd://cpu-container-a",
                    }
                ],
            },
        }
        state = {
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
            "release_delivery_sha256": self.files.delivery["sha256"],
            "bundle_sha256": "c" * 64,
            "runtime_image": image,
            "node_template_sha256": "d" * 64,
            "agent_config_digest": "e" * 64,
            "runtime_profile_version": "profile-a",
        }
        version = {
            "module_digest": "f" * 64,
            "deployment_mode": "regional",
            "required_agent_artifact_sha256": "a" * 64,
            "required_agent_compatibility_digest": "b" * 64,
            "required_agent_protocol_version": 3,
            "required_agent_config_digest": "e" * 64,
            "required_runtime_profile_version": "profile-a",
            "required_node_action_key_version": 2,
        }
        self.raw = {
            "namespace": self.namespace,
            "pod": pod,
            "release_metadata": {
                "name": "gpu-fault-regional-release-state",
                "namespace": self.namespace,
                "uid": "release-state-uid",
            },
            "release_state": state,
            "registration": {
                "cluster_id": "cluster-a",
                "enabled": True,
                "lifecycle_state": "ACTIVE",
                "agent_endpoint_allowed_cidrs": ["10.0.1.0/24"],
            },
            "agent_snapshot": {
                "release_id": "release-a",
                "version": version,
                "agents": {
                    name: record.model_dump(mode="json")
                    for name, record in self.pair.records.items()
                },
            },
            "nodes": {
                "apiVersion": "v1",
                "kind": "NodeList",
                "items": [
                    {
                        "apiVersion": "v1",
                        "kind": "Node",
                        "metadata": {
                            "name": node,
                            "uid": "uid-" + node,
                            "labels": {
                                "sagemaker.amazonaws.com/cluster-name": "hyperpod-a"
                            },
                        },
                        "spec": {"providerID": "aws:///test-1a/instance-" + node},
                        "status": {
                            "nodeInfo": {"bootID": "boot-" + node},
                            "conditions": [{"type": "Ready", "status": "True"}],
                            "addresses": [
                                {
                                    "type": "InternalIP",
                                    "address": urlsplit(record.endpoint).hostname,
                                }
                            ],
                        },
                    }
                    for node, record in self.pair.records.items()
                ],
            },
        }
        self.key_document = {
            "apiVersion": "v1",
            "kind": "Secret",
            "type": "Opaque",
            "metadata": {
                "name": "gpu-fault-node-action-keys",
                "namespace": self.namespace,
                "uid": "cpu-key-uid",
                "resourceVersion": "1",
            },
            "data": {
                node: base64.b64encode(value.encode()).decode()
                for node, value in self.pair.keys.items()
            },
        }
        self.current_key_document = copy.deepcopy(self.key_document)

    def close(self):
        self.pair.close()

    def registry(self):
        self.events.append(("registry-read",))
        return [copy.deepcopy(self.raw["registration"])]

    def api_query(self, request, *, timeout):
        assert request.get_method() == "GET", (
            "the CPU probe may only read existing API state"
        )
        assert request.get_header("X-gpu-fault-execution-token") == API_TOKEN
        assert timeout == 5
        self.api_requests.append(request)
        parts = urlsplit(request.full_url)
        assert parts.netloc == "127.0.0.1:8080", "the CPU credential stays on loopback"
        if parts.path == "/v1/version":
            value = self.raw["agent_snapshot"]["version"]
        else:
            _, version, fleet, agents, cluster, node = parts.path.split("/")
            assert (version, fleet, agents, cluster) == (
                "v1",
                "fleet",
                "agents",
                "cluster-a",
            )
            value = self.raw["agent_snapshot"]["agents"][unquote(node)]
        return Response(200, json.dumps(value).encode())

    def cpu(self, *arguments, **kwargs):
        self.events.append(("cpu", *arguments))
        if arguments[:2] == ("get", "pod"):
            self.capture_count += 1
            if self.capture_count == 2:
                self.after_capture()
            return json.dumps(
                {"apiVersion": "v1", "kind": "PodList", "items": [self.raw["pod"]]}
            )
        if arguments[:2] == ("get", "configmap"):
            return json.dumps(
                {
                    "metadata": self.raw["release_metadata"],
                    "data": {"state.json": json.dumps(self.raw["release_state"])},
                }
            )
        if arguments[:2] == ("get", "secret"):
            return json.dumps(self.current_key_document)
        assert arguments[:6] == ("exec", "-i", "api-a", "-c", "api", "--"), (
            "execution must name the already bound API Pod and container"
        )
        assert arguments[6:8] == ("/opt/gpu-fault/control-plane/bin/python", "-")
        assert kwargs["timeout"] == 30
        output = io.StringIO()
        with self.monkeypatch.context() as patch, redirect_stdout(output):
            patch.setattr(sys, "argv", [str(AGENT_PROBE), *arguments[8:]])
            patch.setattr(subprocess, "run", self.http_worker)
            patch.setenv("GPU_FAULT_EXECUTION_TOKEN", API_TOKEN)
            patch.setenv(
                "GPU_FAULT_RELEASE_ID", self.raw["agent_snapshot"]["release_id"]
            )
            if self.restrict_cpu_imports:
                original_import = builtins.__import__

                def shipped_only(name, *args, **kwargs):
                    if name == "scripts" or name.startswith(
                        ("scripts.", "gpu_fault.admin")
                    ):
                        raise ImportError("unshipped module")
                    return original_import(name, *args, **kwargs)

                patch.setattr(builtins, "__import__", shipped_only)
            with pytest.raises(SystemExit) as completed:
                exec(
                    compile(kwargs["input_text"], str(AGENT_PROBE), "exec"),
                    {"__name__": "__main__"},
                )
        if completed.value.code:
            raise RuntimeError("read-only CPU probe refused")
        return output.getvalue()

    def http_worker(self, command, **kwargs):
        assert command[:5] == [sys.executable, "-I", "-S", "-B", "-c"]
        assert API_TOKEN not in repr(command), (
            "the CPU token must be private stdio, never argv"
        )
        document = json.loads(kwargs["input"])
        request = Request(
            document["url"],
            method=document["method"],
            headers=document["headers"],
            data=base64.b64decode(document["body"])
            if document["body"] is not None
            else None,
        )
        with self.api_query(request, timeout=document["seconds"]) as response:
            result = {
                "status": response.status,
                "body": base64.b64encode(response.payload).decode(),
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(result).encode(), b"")

    def gpu(self, target, *arguments, **kwargs):
        assert target == self.target
        assert arguments == ("get", "nodes", "node-a", "node-b", "-o", "json")
        self.events.append(("gpu", *arguments))
        return json.dumps(self.raw["nodes"])
