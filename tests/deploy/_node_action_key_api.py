#!/usr/bin/env python3
"""Local, stateful kubectl double for node-key provisioning behavior."""

from __future__ import annotations

import base64
import copy
import json
import os
import ssl
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SECRET = "gpu-fault-node-action-keys"
NAMESPACE = "gpu-fault-system"
HYPERPOD = "fixture-hyperpod"
LAST_APPLIED = "kubectl.kubernetes.io/last-applied-configuration"


def encoded(label: str) -> str:
    return base64.b64encode(("fixture-only:" + (label + "-") * 16).encode()).decode()


def secret(
    scope: str, data: dict[str, str], *, last_applied: bool = False
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": SECRET,
            "namespace": NAMESPACE,
            "uid": f"fixture-{scope}-secret",
            "resourceVersion": "1",
        },
        "type": "Opaque",
        "data": dict(data),
    }
    if last_applied:
        document["metadata"]["annotations"] = {
            LAST_APPLIED: json.dumps(copy.deepcopy(document))
        }
    return document


def node_list(*names: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "NodeList",
        "metadata": {},
        "items": [
            {
                "apiVersion": "v1",
                "kind": "Node",
                "metadata": {
                    "name": name,
                    "uid": f"fixture-uid-{name}",
                    "labels": {"sagemaker.amazonaws.com/cluster-name": HYPERPOD},
                },
            }
            for name in names
        ],
    }


class Api:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.state.setdefault("calls", [])
        self.state.setdefault("writes", [])
        self.state.setdefault("write_attempts", [])
        self.state.setdefault("counts", {})
        self.state.setdefault("events", [])

    def run(
        self, arguments: list[str], *, input_text: str | None = None, **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        scope = (
            "cpu"
            if "--kubeconfig" in arguments
            or (
                "--context" in arguments
                and "cpu" in arguments[arguments.index("--context") + 1]
            )
            else "gpu"
        )
        verb = next(
            (
                item
                for item in ("get", "create", "apply", "replace", "patch")
                if item in arguments
            ),
            "",
        )
        self.state["calls"].append(list(arguments))
        action = f"{scope}:{verb}"
        count = self.state["counts"].get(action, 0) + 1
        self.state["counts"][action] = count
        resource_action = action + (
            "-nodes" if verb == "get" and "nodes" in arguments else "-secret"
        )
        resource_count = self.state["counts"].get(resource_action, 0) + 1
        self.state["counts"][resource_action] = resource_count
        event: dict[str, Any] = next(
            (
                item
                for item in self.state["events"]
                if (
                    item["on"] == action
                    and item.get("occurrence", 1) == count
                    or item["on"] == resource_action
                    and item.get("occurrence", 1) == resource_count
                )
            ),
            {},
        )
        if verb in {"create", "replace"} and "--dry-run=client" not in arguments:
            submitted = json.loads(input_text or "{}")
            self.state["write_attempts"].append(
                {
                    "scope": scope,
                    "verb": verb,
                    "uid": submitted.get("metadata", {}).get("uid"),
                    "version": submitted.get("metadata", {}).get("resourceVersion"),
                }
            )
        target = event.get("target", scope)
        if "document" in event:
            self.state["secrets"][target] = copy.deepcopy(event["document"])
        if "merge_data" in event or "merge_metadata" in event:
            changed = self.state["secrets"][target]
            changed["data"].update(event.get("merge_data", {}))
            changed["metadata"].update(event.get("merge_metadata", {}))
            changed["metadata"]["resourceVersion"] = str(
                int(changed["metadata"]["resourceVersion"]) + 1
            )
        if "nodes" in event:
            self.state["nodes"] = copy.deepcopy(event["nodes"])
        if event.get("returncode") and not event.get("lost_ack"):
            return subprocess.CompletedProcess(
                arguments,
                event["returncode"],
                event.get("stdout", ""),
                event.get("stderr", ""),
            )
        if verb == "get" and "nodes" in arguments:
            output = (
                "\n".join(
                    item["metadata"]["name"] for item in self.state["nodes"]["items"]
                )
                if any(item.startswith("jsonpath=") for item in arguments)
                else json.dumps(self.state["nodes"])
            )
            return subprocess.CompletedProcess(arguments, 0, output, "")
        if verb == "get":
            document = self.state["secrets"].get(scope)
            if "output" in event:
                return subprocess.CompletedProcess(arguments, 0, event["output"], "")
            if document is None:
                code = 0 if "--ignore-not-found" in arguments else 1
                return subprocess.CompletedProcess(
                    arguments, code, "", "fixture missing" if code else ""
                )
            output = json.dumps(document) if "-o" in arguments else ""
            return subprocess.CompletedProcess(arguments, 0, output, "")
        if verb == "create" and "--dry-run=client" in arguments:
            directory = Path(
                next(
                    item.removeprefix("--from-file=")
                    for item in arguments
                    if item.startswith("--from-file=")
                )
            )
            document = secret(
                scope,
                {
                    path.name: base64.b64encode(path.read_bytes()).decode()
                    for path in directory.iterdir()
                    if path.is_file()
                },
            )
            document["metadata"].pop("uid")
            document["metadata"].pop("resourceVersion")
            return subprocess.CompletedProcess(arguments, 0, json.dumps(document), "")
        if verb not in {"create", "apply", "replace"} or input_text is None:
            return subprocess.CompletedProcess(
                arguments, 2, "", "unexpected fake kubectl request"
            )
        submitted = json.loads(input_text)
        old = self.state["secrets"].get(scope)
        if verb == "create" and old is not None:
            return subprocess.CompletedProcess(
                arguments, 1, "", "fixture AlreadyExists"
            )
        if verb == "replace" and (
            old is None
            or submitted["metadata"].get("uid") != old["metadata"]["uid"]
            or submitted["metadata"].get("resourceVersion")
            != old["metadata"]["resourceVersion"]
        ):
            return subprocess.CompletedProcess(arguments, 1, "", "fixture CAS rejected")
        updated = copy.deepcopy(submitted)
        if verb == "apply" and old is not None:
            previous = json.loads(
                old["metadata"].get("annotations", {}).get(LAST_APPLIED, "{}")
            )
            data = dict(old["data"])
            for key in set(previous.get("data", {})) - set(submitted["data"]):
                data.pop(key, None)
            data.update(submitted["data"])
            updated["data"] = data
        updated["metadata"].update(
            uid=old["metadata"]["uid"] if old else f"fixture-{scope}-secret",
            resourceVersion=str(int(old["metadata"]["resourceVersion"]) + 1)
            if old
            else "1",
        )
        if verb == "apply":
            updated["metadata"].setdefault("annotations", {})[LAST_APPLIED] = (
                json.dumps(submitted)
            )
        if "admission_data" in event:
            updated["data"] = copy.deepcopy(event["admission_data"])
        self.state["secrets"][scope] = updated
        self.state["writes"].append(
            {"scope": scope, "verb": verb, "keys": sorted(updated["data"])}
        )
        return subprocess.CompletedProcess(
            arguments,
            event.get("returncode", 0),
            "" if event.get("lost_ack") else event.get("output", json.dumps(updated)),
            event.get("stderr", ""),
        )


@contextmanager
def kubectl_node_api(
    tmp_path: Path, nodes: dict[str, Any]
) -> Iterator[tuple[Path, list[str]]]:
    responses = {
        "/api": {"apiVersion": "v1", "kind": "APIVersions", "versions": ["v1"]},
        "/apis": {"apiVersion": "v1", "kind": "APIGroupList", "groups": []},
        "/api/v1": {
            "apiVersion": "v1",
            "kind": "APIResourceList",
            "groupVersion": "v1",
            "resources": [
                {
                    "name": "nodes",
                    "singularName": "node",
                    "kind": "Node",
                    "namespaced": False,
                    "verbs": ["get", "list"],
                }
            ],
        },
        "/api/v1/nodes": nodes,
    }
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            requests.append(self.path)
            document = responses.get(path)
            self.send_response(200 if document is not None else 404)
            payload = json.dumps(
                document
                if document is not None
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

        def log_message(self, _format: str, *args: Any) -> None:
            pass

    certificate, private_key = tmp_path / "server.crt", tmp_path / "server.key"
    completed = subprocess.run(
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
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 0, "cannot create the isolated TLS fixture"
    private_key.chmod(0o600)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(certificate, private_key)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    kubeconfig = tmp_path / "fixture.kubeconfig"
    kubeconfig.write_text(
        json.dumps(
            {
                "apiVersion": "v1",
                "kind": "Config",
                "clusters": [
                    {
                        "name": "fixture",
                        "cluster": {
                            "server": f"https://127.0.0.1:{server.server_port}",
                            "certificate-authority": str(certificate),
                        },
                    }
                ],
                "contexts": [
                    {
                        "name": "fixture",
                        "context": {"cluster": "fixture", "user": "fixture"},
                    }
                ],
                "users": [{"name": "fixture", "user": {"token": "test-only"}}],
                "current-context": "fixture",
            }
        )
    )
    kubeconfig.chmod(0o600)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield kubeconfig, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "node-key API fixture thread did not stop"


def main() -> int:
    path = Path(os.environ["TEST_NODE_KEY_API"])
    api = Api(json.loads(path.read_text()))
    arguments = ["kubectl", *sys.argv[1:]]
    input_text = sys.stdin.read() if "-f" in arguments and "-" in arguments else None
    result = api.run(arguments, input_text=input_text)
    path.write_text(json.dumps(api.state))
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
