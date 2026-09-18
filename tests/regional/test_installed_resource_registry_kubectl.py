from __future__ import annotations

import json
import shutil
import ssl
import sys
import threading
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from gpu_fault.admin.execution import run_command
from tests.regional.test_installed_resource_registry import MODULE

# Every client uses the explicit loopback fixture kubeconfig, never local contexts.
pytestmark = pytest.mark.allows_cluster_binaries("kubectl")

NAMESPACE = "gpu-fault-system"
API_RESOURCES = (
    ("v1", "Namespace", "namespaces", False),
    ("v1", "ConfigMap", "configmaps", True),
    ("apps/v1", "Deployment", "deployments", True),
    ("batch/v1", "CronJob", "cronjobs", True),
    ("rbac.authorization.k8s.io/v1", "ClusterRole", "clusterroles", False),
    ("monitoring.coreos.com/v1", "PrometheusRule", "prometheusrules", True),
)


@pytest.fixture
def local_api(tmp_path: Path):
    if shutil.which("kubectl") is None:
        pytest.skip("isolated client integration requires a kubectl binary")
    groups = {
        version.rsplit("/", 1)[0]: version
        for version, _kind, _plural, _namespaced in API_RESOURCES
        if "/" in version
    }
    responses = {
        "/api": {"apiVersion": "v1", "kind": "APIVersions", "versions": ["v1"]},
        "/apis": {
            "apiVersion": "v1",
            "kind": "APIGroupList",
            "groups": [
                {
                    "name": group,
                    "versions": [{"groupVersion": version, "version": "v1"}],
                    "preferredVersion": {"groupVersion": version, "version": "v1"},
                }
                for group, version in groups.items()
            ],
        },
    }
    for version, kind, plural, namespaced in API_RESOURCES:
        prefix = f"/apis/{version}" if "/" in version else f"/api/{version}"
        responses.setdefault(
            prefix,
            {
                "apiVersion": "v1",
                "kind": "APIResourceList",
                "groupVersion": version,
                "resources": [],
            },
        )["resources"].append(
            {
                "name": plural,
                "singularName": kind.lower(),
                "kind": kind,
                "namespaced": namespaced,
                "verbs": ["get", "list"],
            }
        )
        # Typed API lists may omit TypeMeta on each member. kubectl, not this
        # fixture, must supply the kind/API version when it flattens the list.
        members = [
            {
                "metadata": {
                    "name": "gpu-fault-fixture",
                    "uid": f"uid-{kind}-{namespace}",
                    **({"namespace": namespace} if namespaced else {}),
                }
            }
            for namespace in ([NAMESPACE, "gf-regional-old"] if namespaced else [""])
        ]
        responses[f"{prefix}/{plural}"] = {
            "apiVersion": version,
            "kind": kind + "List",
            "metadata": {"resourceVersion": "1"},
            "items": members,
        }
        if namespaced:
            responses[f"{prefix}/namespaces/{NAMESPACE}/{plural}"] = {
                **responses[f"{prefix}/{plural}"],
                "items": members[:1],
            }
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            requests.append(path)
            value = responses.get(path)
            self.send_response(200 if value is not None else 404)
            payload = json.dumps(
                value
                if value is not None
                else {
                    "apiVersion": "v1",
                    "kind": "Status",
                    "status": "Failure",
                    "reason": "NotFound",
                    "code": 404,
                }
            ).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            pass

    certificate, private_key = tmp_path / "server.crt", tmp_path / "server.key"
    completed = run_command(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(private_key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        timeout_seconds=15,
    )
    assert completed.returncode == 0, "cannot create the isolated TLS fixture"
    private_key.chmod(0o600)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(certificate, private_key)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"https://127.0.0.1:{server.server_port}"
    kubeconfig = tmp_path / "fixture.kubeconfig"
    config = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": "fixture",
                "cluster": {
                    "server": endpoint,
                    "certificate-authority": str(certificate),
                },
            }
        ],
        "contexts": [
            {"name": "fixture", "context": {"cluster": "fixture", "user": "fixture"}}
        ],
        "users": [{"name": "fixture", "user": {"token": "test-only"}}],
        "current-context": "fixture",
    }
    kubeconfig.write_text(json.dumps(config))
    kubeconfig.chmod(0o600)
    try:
        yield kubeconfig, config, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "registry API fixture thread did not stop"


@pytest.mark.parametrize("version,kind,plural,namespaced", API_RESOURCES)
def test_actual_kubectl_supplies_types_for_typed_list_members(
    local_api, version: str, kind: str, plural: str, namespaced: bool
) -> None:
    kubeconfig, _config, _requests = local_api
    with closing(MODULE.Kubectl(kubeconfig=str(kubeconfig), context=None)) as kubectl:
        arguments = ["get", kind.lower(), "-o", "json"]
        if namespaced:
            arguments += ["-A"]
        result = kubectl.run(arguments)
    items = MODULE.resource_list(result.stdout)
    assert {item["kind"] for item in items} == {kind}
    assert {item["apiVersion"] for item in items} == {version}
    assert {item["metadata"].get("namespace") for item in items} == (
        {NAMESPACE, "gf-regional-old"} if namespaced else {None}
    )
    assert any(path.endswith("/" + plural) for path in _requests), (
        "kubectl must query the requested resource endpoint"
    )


def test_sync_accepts_actual_kubectl_typed_list_output(
    local_api, tmp_path: Path
) -> None:
    kubeconfig, _config, requests = local_api
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cpu": {
                    "resources": [
                        {
                            "kind": "deployment",
                            "name": "gpu-fault-fixture",
                            "scope": "namespaced",
                            "phase": "ingress",
                            "order": 10,
                            "clean": "delete",
                        }
                    ]
                },
            }
        )
    )
    with closing(MODULE.Kubectl(kubeconfig=str(kubeconfig), context=None)) as kubectl:
        document = MODULE.synchronize(
            kubectl,
            plane="cpu",
            namespace=NAMESPACE,
            inventory_path=inventory,
            release_id="fixture",
            apply=False,
        )
    assert len(document["resources"]) == 1
    assert document["resources"][0]["namespace"] == NAMESPACE
    assert f"/apis/apps/v1/namespaces/{NAMESPACE}/deployments" in requests


@pytest.mark.parametrize("provide_cluster_info", [False, True])
def test_generic_exec_protocol_is_left_to_native_kubectl(
    local_api, tmp_path: Path, provide_cluster_info: bool
) -> None:
    kubeconfig, config, _requests = local_api
    observed = tmp_path / "exec-info.json"
    plugin = tmp_path / "fixture_plugin.py"
    plugin.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "info = json.loads(os.environ['KUBERNETES_EXEC_INFO'])\n"
        "Path(sys.argv[1]).write_text(json.dumps(info))\n"
        "print(json.dumps({'apiVersion': info['apiVersion'], "
        "'kind': 'ExecCredential', 'status': {'token': 'test-only'}}))\n"
    )
    config["users"][0]["user"] = {
        "exec": {
            "apiVersion": "client.authentication.k8s.io/v1",
            "command": sys.executable,
            "args": [str(plugin), str(observed)],
            "interactiveMode": "Never",
            "provideClusterInfo": provide_cluster_info,
        }
    }
    kubeconfig.write_text(json.dumps(config))
    with closing(
        MODULE.Kubectl(
            kubeconfig=str(kubeconfig), context=None, reuse_exec_credential=True
        )
    ) as kubectl:
        result = kubectl.run(["get", "deployment", "-n", NAMESPACE, "-o", "json"])
        assert MODULE.resource_list(result.stdout)[0]["kind"] == "Deployment"
        assert kubectl.session_directory is None
    info = json.loads(observed.read_text())
    assert info["spec"]["interactive"] is False
    if provide_cluster_info:
        assert (
            info["spec"]["cluster"]["server"]
            == config["clusters"][0]["cluster"]["server"]
        )
    else:
        assert "cluster" not in info["spec"]
