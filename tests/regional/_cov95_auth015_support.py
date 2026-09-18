from __future__ import annotations

import ipaddress
import ssl
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

from gpu_fault.fleet import AgentRecord
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import app as agent_app
from gpu_fault.node_agent.executor import NodeActionExecutor
from gpu_fault.node_agent.ledger import NodeActionLedger
from scripts.e2e.regional import auth015_protocol as protocol

KEY_A = "synthetic-node-a-" + "a" * 48
KEY_B = "synthetic-node-b-" + "b" * 48


@lru_cache
def certificate() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "auth015-test")])
    now = datetime.now(timezone.utc)
    value = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.IPAddress(ipaddress.ip_address("10.0.1.1")),
                    x509.IPAddress(ipaddress.ip_address("10.0.1.2")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return value.public_bytes(serialization.Encoding.PEM).decode("ascii")


def agent(node: str = "node-a", **updates: Any) -> AgentRecord:
    now = datetime.now(timezone.utc)
    values = {
        "cluster_id": "cluster-a",
        "node_id": node,
        "endpoint": f"https://10.0.1.{1 if node == 'node-a' else 2}:9099",
        "tls_certificate_pem": certificate(),
        "agent_protocol_version": 3,
        "node_action_key_version": 2,
        "agent_version": "0.10.0",
        "artifact_sha256": "a" * 64,
        "compatibility_digest": "b" * 64,
        "installer_bundle_sha256": "c" * 64,
        "installer_template_sha256": "d" * 64,
        "policy_version": "policy-a",
        "runtime_profile_version": "profile-a",
        "config_digest": "e" * 64,
        "allowed_operations": [WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE],
        "boot_id": "boot-" + node,
        "node_instance_id": "instance-" + node,
        "agent_incarnation_id": "incarnation-" + node,
        "first_seen_at": now - timedelta(hours=1),
        "last_seen_at": now - timedelta(seconds=1),
        "lease_expires_at": now + timedelta(seconds=120),
        "generation": 7,
    }
    return AgentRecord.model_validate({**values, **updates})


class Response:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self.payload = payload
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self.payload if size < 0 else self.payload[:size]

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class AgentPair:
    def __init__(self, path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.records = {"node-a": agent(), "node-b": agent("node-b")}
        self.keys = {"node-a": KEY_A, "node-b": KEY_B}
        self.targets = {
            node: protocol.SignatureTarget(record, self.keys[node])
            for node, record in self.records.items()
        }
        self.reads: list[tuple[str, str]] = []
        self.requests: list[Request] = []
        self.executors: dict[str, NodeActionExecutor] = {}
        self.clients: dict[str, TestClient] = {}
        self.stack = ExitStack()
        monkeypatch.setattr(
            agent_app, "heartbeat_reporter_from_environment", lambda _: None
        )
        for node, record in self.records.items():
            ledger = NodeActionLedger(path / f"{node}.db")
            self.stack.callback(ledger.close)
            original_get = ledger.get

            def get(command_id, *, selected=node, read=original_get):
                self.reads.append((selected, command_id))
                return read(command_id)

            monkeypatch.setattr(ledger, "get", get)
            monkeypatch.setattr(
                ledger,
                "attempt_history",
                lambda _: pytest.fail("an inadmissible command reached the ledger"),
            )
            executor = NodeActionExecutor(
                secret=self.keys[node],
                node_action_key_version=2,
                node_ids={node},
                allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
                reset_enabled=False,
                ledger=ledger,
                agent_generation=record.generation,
                runner=lambda *a, **k: pytest.fail(
                    "a protocol challenge dispatched a handler"
                ),
            )
            self.executors[node] = executor
            self.clients[node] = self.stack.enter_context(
                TestClient(agent_app.create_node_agent_app(executor))
            )
        monkeypatch.setattr(protocol, "bounded_request", self.bounded_request)

    def close(self) -> None:
        self.stack.close()

    def bounded_request(
        self, url, *, method, headers, data, certificate, timeout, max_response_bytes
    ):
        assert 0 < timeout <= protocol.REQUEST_TIMEOUT_SECONDS
        assert max_response_bytes == protocol.MAX_RESPONSE_BYTES
        ssl_context = ssl.create_default_context(cadata=certificate)
        assert ssl_context.verify_mode == ssl.CERT_REQUIRED
        assert ssl_context.check_hostname is True
        request = Request(url, method=method, headers=headers, data=data)
        self.requests.append(request)
        parts = urlsplit(request.full_url)
        node = next(
            name
            for name, record in self.records.items()
            if urlsplit(record.endpoint).netloc == parts.netloc
        )
        result = self.clients[node].request(
            request.get_method(),
            parts.path + ("?" + parts.query if parts.query else ""),
            content=request.data,
            headers=dict(request.header_items()),
        )
        return result.status_code, result.content
