from __future__ import annotations

import hashlib
import io
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import yaml

from scripts.e2e.regional import auth016_lifecycle as lifecycle
from tests.regional._cov95_auth015_http import server_certificate
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional._security_token_rotation_support import RotationWorld

ROOT = Path(__file__).resolve().parents[2]
CA_VARIABLES = ("GPU_FAULT_CONTROL_PLANE_CA_FILE", "SSL_CERT_FILE")
MANIFESTS = (
    "cluster-action-executor.yaml",
    "completion-watcher.yaml",
    "kubernetes-node-resource-collector.yaml",
)


@pytest.fixture
def certificates(tmp_path):
    contexts = {}
    for name in ("trusted", "foreign"):
        directory = tmp_path / name
        directory.mkdir(mode=0o700)
        server_certificate(directory)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(directory / "server.pem", directory / "server.key")
        contexts[name] = SimpleNamespace(
            ca_file=directory / "server.pem", server=context
        )
    return contexts


@pytest.fixture(params=MANIFESTS)
def consumer(request, monkeypatch, certificates):
    documents = yaml.safe_load_all(
        (ROOT / "deploy/dataplane" / request.param).read_text()
    )
    deployment = next(
        item
        for item in documents
        if isinstance(item, dict) and item.get("kind") == "Deployment"
    )
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    references = {
        "cluster-id": "cluster-a",
        "cluster-token": "local-auth016-fixture-" + "x" * 40,
        "control-plane-url": "https://127.0.0.1",
    }
    environment = {}
    for item in container["env"]:
        if "value" in item:
            environment[item["name"]] = item["value"]
        else:
            key = item.get("valueFrom", {}).get("secretKeyRef", {}).get("key")
            if key in references:
                environment[item["name"]] = references[key]
    variables = [name for name in CA_VARIABLES if name in environment]
    assert variables == [
        CA_VARIABLES[0] if container["name"] == "executor" else CA_VARIABLES[1]
    ]
    variable = variables[0]
    assert environment[variable] == "/etc/gpu-fault/tls/ca.crt"
    environment[variable] = str(certificates["trusted"].ca_file)
    monkeypatch.setattr(os, "environ", environment)
    mode = "executor" if container["name"] == "executor" else "standard"
    monkeypatch.setattr(sys, "argv", ["auth016-probe", mode])
    return SimpleNamespace(
        environment=environment,
        ca_variable=variable,
        mode=mode,
        token_sha256=hashlib.sha256(references["cluster-token"].encode()).hexdigest(),
    )


def handshake(client_context, server_context, hostname):
    client_in, client_out, server_in, server_out = (ssl.MemoryBIO() for _ in range(4))
    client = client_context.wrap_bio(
        client_in, client_out, server_side=False, server_hostname=hostname
    )
    server = server_context.wrap_bio(server_in, server_out, server_side=True)
    completed = set()
    for _ in range(16):
        for name, endpoint in (("client", client), ("server", server)):
            if name not in completed:
                try:
                    endpoint.do_handshake()
                    completed.add(name)
                except ssl.SSLWantReadError:
                    pass
        for outgoing, incoming in ((client_out, server_in), (server_out, client_in)):
            if outgoing.pending:
                incoming.write(outgoing.read())
        if len(completed) == 2:
            assert client.getpeercert(), "the TLS peer must be authenticated"
            return
    pytest.fail("the in-memory TLS handshake did not converge")


@pytest.fixture
def transport(monkeypatch, consumer, certificates):
    state = {
        "calls": [],
        "server": certificates["trusted"].server,
        "body": [{"cluster_id": "cluster-a"}],
        "status": 200,
        "error": None,
    }

    def urlopen(request, *, context, timeout):
        state["calls"].append(request.full_url)
        assert request.get_method() == "GET" and request.data is None
        assert request.full_url == (
            consumer.environment["GPU_FAULT_CONTROL_PLANE_URL"] + "/v1/fleet/agents"
        )
        headers = {key.lower(): value for key, value in request.header_items()}
        authorization = headers["authorization"]
        assert authorization.startswith("Bearer "), (
            "the consumer probe must send bearer authentication"
        )
        assert (
            hashlib.sha256(authorization.removeprefix("Bearer ").encode()).hexdigest()
            == consumer.token_sha256
        )
        assert (
            headers["x-gpu-fault-cluster-id"]
            == (consumer.environment["GPU_FAULT_CLUSTER_ID"])
        )
        assert timeout == 20
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        assert context.cert_store_stats()["x509"] == 1
        handshake(context, state["server"], urlsplit(request.full_url).hostname)
        if state["error"] is not None:
            raise state["error"]
        response = io.BytesIO(json.dumps(state["body"]).encode())
        if state["status"] != 200:
            raise urllib.error.HTTPError(
                request.full_url, state["status"], "denied", {}, response
            )
        response.status = 200
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return state


def run_probe():
    output = io.StringIO()
    with redirect_stdout(output):
        exec(
            compile(lifecycle.POD_TOKEN_PROBE, "<auth016-pod-token-probe>", "exec"), {}
        )
    return json.loads(output.getvalue())


def test_each_manifest_trust_environment_authenticates_the_exact_consumer(
    consumer, transport
):
    result = run_probe()
    assert result == {
        "cluster_id": "cluster-a",
        "token_sha256": consumer.token_sha256,
        "status": 200,
        "local_scope": True,
    }
    assert len(transport["calls"]) == 1
    assert consumer.environment["GPU_FAULT_CONTROL_PLANE_TOKEN"] not in json.dumps(
        result
    )


@pytest.mark.parametrize("system_trust", ["same-file", "foreign-file"])
def test_each_consumers_defined_ca_precedence_is_preserved(
    consumer, transport, certificates, system_trust
):
    other = next(name for name in CA_VARIABLES if name != consumer.ca_variable)
    consumer.environment[other] = str(
        certificates["trusted" if system_trust == "same-file" else "foreign"].ca_file
    )
    assert run_probe()["status"] == 200
    assert len(transport["calls"]) == 1


@pytest.mark.parametrize("consumer", [MANIFESTS[0]], indirect=True)
def test_executor_empty_dedicated_ca_uses_its_configured_standard_ca(
    consumer, transport, certificates
):
    consumer.environment[CA_VARIABLES[0]] = ""
    consumer.environment[CA_VARIABLES[1]] = str(certificates["trusted"].ca_file)
    assert run_probe()["status"] == 200


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "empty",
        "whitespace",
        "relative",
        "control-character",
        "invalid-pem",
        "missing-file",
        "directory",
    ],
)
def test_unproven_ca_configuration_stops_before_transport(
    consumer, transport, certificates, tmp_path, defect
):
    environment, variable = consumer.environment, consumer.ca_variable
    if defect == "missing":
        environment.pop(variable)
        environment["SSL_CERT_DIR"] = str(certificates["foreign"].ca_file.parent)
        if consumer.mode == "standard":
            environment[CA_VARIABLES[0]] = str(certificates["trusted"].ca_file)
    elif defect in {"empty", "whitespace", "relative", "control-character"}:
        environment[variable] = {
            "empty": "",
            "whitespace": " ",
            "relative": "server.pem",
            "control-character": "/invalid\nca.pem",
        }[defect]
    else:
        invalid = tmp_path / "invalid-ca.pem"
        if defect == "invalid-pem":
            invalid.write_text("not a certificate")
        environment[variable] = str(tmp_path if defect == "directory" else invalid)
        other = next(name for name in CA_VARIABLES if name != variable)
        environment[other] = str(certificates["trusted"].ca_file)
    with pytest.raises((RuntimeError, OSError)):
        run_probe()
    assert transport["calls"] == []


@pytest.mark.parametrize("defect", ["foreign-ca", "wrong-hostname"])
def test_tls_identity_failure_cannot_be_retried_with_other_trust(
    consumer, transport, certificates, defect
):
    if defect == "foreign-ca":
        transport["server"] = certificates["foreign"].server
    else:
        consumer.environment["GPU_FAULT_CONTROL_PLANE_URL"] = "https://peer.invalid"
    with pytest.raises(ssl.SSLCertVerificationError):
        run_probe()
    assert len(transport["calls"]) == 1


def test_a_foreign_selected_ca_cannot_fall_back_to_an_unrelated_valid_ca(
    consumer, transport, certificates
):
    consumer.environment[consumer.ca_variable] = str(certificates["foreign"].ca_file)
    other = next(name for name in CA_VARIABLES if name != consumer.ca_variable)
    consumer.environment[other] = str(certificates["trusted"].ca_file)
    with pytest.raises(ssl.SSLCertVerificationError):
        run_probe()
    assert len(transport["calls"]) == 1


@pytest.mark.parametrize("mode", ["missing", "unknown", "extra-argument"])
def test_an_unbound_consumer_mode_stops_before_transport(
    consumer, transport, monkeypatch, mode
):
    arguments = ["auth016-probe"]
    if mode != "missing":
        arguments.append("unknown" if mode == "unknown" else consumer.mode)
    if mode == "extra-argument":
        arguments.append("standard")
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(RuntimeError, match="trust mode is unbound"):
        run_probe()
    assert transport["calls"] == []


@pytest.mark.parametrize(
    "error", [TimeoutError, ConnectionResetError, ssl.SSLError, urllib.error.URLError]
)
def test_unknown_transport_outcome_never_becomes_success(transport, error):
    transport["error"] = error("local transport failure")
    with pytest.raises(error):
        run_probe()
    assert len(transport["calls"]) == 1


@pytest.mark.parametrize("status", [401, 403, 503])
def test_http_denial_keeps_status_and_digest_without_claiming_local_scope(
    consumer, transport, status
):
    transport["status"] = status
    assert run_probe() == {
        "cluster_id": "cluster-a",
        "token_sha256": consumer.token_sha256,
        "status": status,
        "local_scope": False,
    }


@pytest.mark.parametrize("defect", ["peer", "mixed", "empty", "token", "cluster"])
def test_consumer_verdict_still_requires_local_identity_and_expected_token(
    consumer, transport, monkeypatch, tmp_path, defect
):
    world = RotationWorld(tmp_path, monkeypatch)
    monkeypatch.setattr(world.site, "pod_json", lambda *_a, **_k: run_probe())
    expected = consumer.token_sha256
    if defect in {"peer", "mixed", "empty"}:
        transport["body"] = {
            "peer": [{"cluster_id": "cluster-b"}],
            "mixed": [{"cluster_id": "cluster-a"}, {"cluster_id": "cluster-b"}],
            "empty": [],
        }[defect]
    elif defect == "token":
        expected = "f" * 64
    else:
        consumer.environment["GPU_FAULT_CLUSTER_ID"] = "cluster-b"
    with pytest.raises(lifecycle.IdentityAcceptanceError, match="accepted credential"):
        lifecycle.pod_consumers(world.site, world.target, new_digest=expected)


def test_consumer_probe_mode_is_bound_to_the_actual_manifest_deployment(
    monkeypatch, tmp_path
):
    world = RotationWorld(tmp_path, monkeypatch)
    expected = {}
    for manifest in MANIFESTS:
        documents = yaml.safe_load_all(
            (ROOT / "deploy/dataplane" / manifest).read_text()
        )
        deployment = next(
            item
            for item in documents
            if isinstance(item, dict) and item.get("kind") == "Deployment"
        )
        container = deployment["spec"]["template"]["spec"]["containers"][0]["name"]
        expected[deployment["metadata"]["name"]] = (
            "executor" if container == "executor" else "standard"
        )
    calls = []
    original = world.site.pod_json

    def pod_json(plane, target, pod, script, mode, *, timeout):
        calls.append(mode)
        assert script == lifecycle.POD_TOKEN_PROBE
        assert plane == "gpu" and target is world.target and timeout == 60
        return original(plane, target, pod, script, mode, timeout=timeout)

    monkeypatch.setattr(world.site, "pod_json", pod_json)
    lifecycle.pod_consumers(world.site, world.target)
    assert calls == [expected[name] for name in lifecycle.DEPLOYMENTS]
