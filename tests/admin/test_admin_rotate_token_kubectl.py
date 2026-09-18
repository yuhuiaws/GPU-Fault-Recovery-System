"""Native kubectl framing/query checks against an isolated loopback log fixture.

The HTTP fixture uses no credentials and is not a Kubelet or CRI runtime.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.execution import run_command
from gpu_fault_release.regional_deployment_inventory import CPU_INGRESS_DEPLOYMENT
from tests.admin.test_admin_rotate_token_acceptance import (
    AUTH,
    NOW,
    QUIET_START,
    LogHost,
)

pytestmark = pytest.mark.allows_cluster_binaries("kubectl")


@pytest.fixture
def local_log_api(tmp_path: Path):
    if shutil.which("kubectl") is None:
        pytest.skip("native log framing check requires kubectl")
    host = LogHost()
    host.site.release_config["cpu_kubeconfig"] = "/dev/null"
    namespace = host.site.release_config["namespace"]
    pod = {"apiVersion": "v1", "kind": "Pod", **host.pods[0]}
    pod["metadata"]["namespace"] = namespace
    anchor = host.prefix["api-a"]
    old = (QUIET_START - timedelta(seconds=1)).isoformat()
    host.prefix["api-a"] += f"{old} {'x' * 5000}\n".encode()
    host.after_prefix["api-a"] = (
        host.prefix["api-a"] + f"{NOW.isoformat()} appended after snapshot\n".encode()
    )
    pods_path = f"/api/v1/namespaces/{namespace}/pods"
    apps_path = f"/apis/apps/v1/namespaces/{namespace}"
    resources = {
        f"{apps_path}/deployments/{CPU_INGRESS_DEPLOYMENT}": "deployment",
        f"{apps_path}/replicasets": "replicasets",
        pods_path: "pods",
    }
    apps_group = {
        "name": "apps",
        "versions": [{"groupVersion": "apps/v1", "version": "v1"}],
        "preferredVersion": {"groupVersion": "apps/v1", "version": "v1"},
    }
    responses = {
        "/api": {"apiVersion": "v1", "kind": "APIVersions", "versions": ["v1"]},
        "/apis": {"apiVersion": "v1", "kind": "APIGroupList", "groups": [apps_group]},
        "/apis/apps": {"apiVersion": "v1", "kind": "APIGroup", **apps_group},
        "/apis/apps/v1": {
            "apiVersion": "v1",
            "kind": "APIResourceList",
            "groupVersion": "apps/v1",
            "resources": [
                {
                    "name": name,
                    "singularName": singular,
                    "kind": kind,
                    "namespaced": True,
                    "verbs": ["get", "list"],
                }
                for name, singular, kind in (
                    ("deployments", "deployment", "Deployment"),
                    ("replicasets", "replicaset", "ReplicaSet"),
                )
            ],
        },
        "/api/v1": {
            "apiVersion": "v1",
            "kind": "APIResourceList",
            "groupVersion": "v1",
            "resources": [
                {
                    "name": "pods",
                    "singularName": "pod",
                    "kind": "Pod",
                    "namespaced": True,
                    "verbs": ["get", "list"],
                }
            ],
        },
        f"{pods_path}/api-a": pod,
    }
    queries = []
    outputs = []
    authorized = []
    prefix_reads = []
    failures: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request = urlsplit(self.path)
            query = parse_qs(request.query)
            authorized.append(self.headers.get("Authorization") is not None)
            if request.path == f"{pods_path}/api-a/log":
                queries.append(query)
                if "limitBytes" in query:
                    source = host.after_prefix if prefix_reads else host.prefix
                    limit = int(query["limitBytes"][0])
                    payload = source["api-a"][:limit]
                    prefix_reads.append(limit)
                else:
                    payload = host.window["api-a"]
                status, content_type = 200, "text/plain"
            else:
                resource = resources.get(request.path)
                if resource in failures:
                    status = failures[resource]
                    value = {
                        "apiVersion": "v1",
                        "kind": "Status",
                        "status": "Failure",
                        "reason": "Forbidden",
                        "message": "synthetic-unstructured-auth-value-24680",
                        "code": status,
                    }
                else:
                    value = (
                        host.read_resource(resource)
                        if resource
                        else responses.get(request.path)
                    )
                    status = 200 if value else 404
                content_type = "application/json"
                payload = json.dumps(
                    value
                    or {
                        "apiVersion": "v1",
                        "kind": "Status",
                        "status": "Failure",
                        "reason": "NotFound",
                        "code": 404,
                    }
                ).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            pass

    environment = {
        "HOME": str(tmp_path),
        "PATH": os.environ["PATH"],
        "KUBECONFIG": "/dev/null",
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def native_run(arguments, *, timeout_seconds):
        assert arguments[:3] == ["kubectl", "--kubeconfig", "/dev/null"]
        result = run_command(
            [
                arguments[0],
                f"--server=http://127.0.0.1:{server.server_port}",
                f"--cache-dir={tmp_path / 'discovery'}",
                "--request-timeout=5s",
                *arguments[1:],
            ],
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
        if "logs" in arguments:
            outputs.append(result.stdout)
        return result

    host.run = native_run
    try:
        yield host, queries, outputs, anchor, failures
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "loopback log server thread did not stop"
        assert authorized and not any(authorized)


@pytest.mark.parametrize("has_authentication", [False, True])
def test_native_kubectl_preserves_prefix_caps_and_fixed_window_queries(
    local_log_api, has_authentication: bool
) -> None:
    host, queries, outputs, anchor, _failures = local_log_api
    host.window["api-a"] = (
        f"{(NOW - timedelta(seconds=1)).isoformat()} {AUTH}\n".encode()
        if has_authentication
        else b""
    )
    matches = host.collect()

    assert len(matches) == int(has_authentication)
    assert queries == [
        {
            "container": ["api"],
            "limitBytes": ["4096"],
            "timeout": ["5s"],
            "timestamps": ["true"],
        },
        {
            "container": ["api"],
            "sinceTime": ["2026-09-11T10:19:00Z"],
            "timeout": ["5s"],
            "timestamps": ["true"],
        },
        {
            "container": ["api"],
            "limitBytes": [str(len(anchor))],
            "timeout": ["5s"],
            "timestamps": ["true"],
        },
    ]
    prefix = "[pod/api-a/api] "
    assert outputs[0].startswith(prefix + anchor.decode()), (
        "kubectl output must retain the source prefix and anchor"
    )
    assert len(outputs[0].encode()) == 4096 + 2 * len(prefix)
    assert not outputs[0].endswith("\n"), (
        "byte-capped output must keep its incomplete trailing fragment"
    )
    assert outputs[1] == (
        prefix + host.window["api-a"].decode() if has_authentication else ""
    )
    assert outputs[2] == prefix + anchor.decode()
    assert (
        host.resource_queries == ["deployment", "replicasets", "pods", "deployment"] * 2
    )


def test_native_kubectl_cannot_accept_one_of_two_desired_replicas(
    local_log_api,
) -> None:
    host, queries, outputs, _anchor, _failures = local_log_api
    host.deployment["spec"]["replicas"] = 2
    for field in ("replicas", "readyReplicas", "updatedReplicas", "availableReplicas"):
        host.deployment["status"][field] = 2

    with pytest.raises(BootstrapError, match="replica"):
        host.collect()
    assert queries == [], "native kubectl must not read an incomplete log source set"
    assert outputs == [], "an incomplete Deployment cannot produce quiet evidence"
    assert host.resource_queries == ["deployment", "replicasets", "pods"]


@pytest.mark.parametrize("resource", ["deployment", "replicasets", "pods"])
def test_native_source_read_failure_refuses_without_echoing_stderr(
    local_log_api, resource: str
) -> None:
    host, queries, _outputs, _anchor, failures = local_log_api
    failures[resource] = 403

    with pytest.raises(BootstrapError, match="cannot verify") as failure:
        host.collect()
    assert "synthetic-unstructured-auth-value-24680" not in str(failure.value)
    assert queries == [], "a failed source read must prevent log acceptance"


def test_native_deployment_generation_drift_refuses_after_receipt_checks(
    local_log_api,
) -> None:
    host, queries, outputs, anchor, _failures = local_log_api
    host.after_deployment["metadata"]["generation"] += 1
    host.after_deployment["status"]["observedGeneration"] += 1

    with pytest.raises(BootstrapError, match="sources changed"):
        host.collect()
    assert len(queries) == 3
    assert outputs[-1] == "[pod/api-a/api] " + anchor.decode()
    assert (
        host.resource_queries == ["deployment", "replicasets", "pods", "deployment"] * 2
    )
