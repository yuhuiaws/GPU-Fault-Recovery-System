from __future__ import annotations

import ipaddress
import ssl
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from gpu_fault.fleet import AgentRecord
from gpu_fault.models import WorkflowOperation
from scripts.e2e.regional.auth015_protocol import SignatureTarget


def tls_context(root: Path):
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "peer-loopback")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
        .public_bytes(serialization.Encoding.PEM)
    )
    certificate_file = root / "peer-loopback.crt"
    key_file = root / "peer-loopback.key"
    certificate_file.write_bytes(certificate)
    with key_file.open("xb") as stream:
        key_file.chmod(0o600)
        stream.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate_file, key_file)
    return context, certificate.decode("ascii"), (certificate_file, key_file)


def signature_targets(endpoint: str, certificate: str):
    now = datetime.now(timezone.utc)
    return tuple(
        SignatureTarget(
            AgentRecord(
                cluster_id="peer-loopback-cluster",
                node_id=f"node-{node}",
                endpoint=endpoint,
                tls_certificate_pem=certificate,
                node_action_key_version=2,
                agent_version="unit-version",
                artifact_sha256="a" * 64,
                policy_version="unit-policy",
                runtime_profile_version="unit-profile",
                config_digest="b" * 64,
                allowed_operations=[WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE],
                first_seen_at=now,
                last_seen_at=now,
            ),
            f"synthetic-peer-{node}-" + node * 48,
        )
        for node in ("a", "b")
    )


@contextmanager
def response_server(root: Path, allowed_ports: set[int], body: bytes, *, interval=0.0):
    context, certificate, paths = tls_context(root)
    stop = Event()
    requests = []
    server_errors = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            requests.append((self.command, self.path.partition("?")[0]))
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                if interval:
                    for byte in body:
                        if stop.is_set():
                            break
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        if stop.wait(interval):
                            break
                else:
                    self.wfile.write(body)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                pass

        def log_message(self, *args):
            return

    class Server(HTTPServer):
        def handle_error(self, request, client_address):
            server_errors.append("TLS loopback handler failed")

    server = Server(("127.0.0.1", 0), Handler)
    allowed_ports.add(server.server_port)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.02))
    thread.start()
    try:
        yield SimpleNamespace(
            targets=signature_targets(
                f"https://127.0.0.1:{server.server_port}", certificate
            ),
            requests=requests,
        )
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        allowed_ports.discard(server.server_port)
        for path in paths:
            path.unlink(missing_ok=True)
        assert not thread.is_alive(), "the owned TLS listener did not terminate"
        assert server_errors == [], (
            "the owned TLS listener failed independently of the proof"
        )
