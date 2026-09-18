from __future__ import annotations

import ctypes
import ipaddress
import ssl
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


def server_certificate(path: Path) -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "auth015-loopback")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    pem = cert.public_bytes(serialization.Encoding.PEM)
    (path / "server.pem").write_bytes(pem)
    private = path / "server.key"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    private.chmod(0o600)
    return pem.decode("ascii")


@contextmanager
def owned_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0, (
        "the test must read its own child-subreaper state"
    )
    assert libc.prctl(36, 1, 0, 0, 0) == 0, (
        "the test must adopt and reap only its newly orphaned HTTP worker"
    )
    try:
        yield
    finally:
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0, (
            "the original child-subreaper state must be restored"
        )


@contextmanager
def loopback_server(path: Path, *, tls: bool = False):
    stop = Event()
    seen = Event()
    requests = []
    byte_limit = 1024

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            return

        def do_POST(self):
            self.do_GET()

        def do_GET(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append((self.command, self.path, dict(self.headers), body))
            seen.set()
            try:
                if self.path.startswith("/headers"):
                    self.connection.sendall(b"HTTP/1.1 200 OK\r\nX-Trickle: ")
                    while not stop.wait(0.03):
                        self.connection.sendall(b"x")
                    return
                if self.path.startswith("/redirect"):
                    self.send_response(302)
                    self.send_header("Location", "/unexpected")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if self.path.startswith(("/oversize", "/body")):
                    self.send_response(200)
                    self.send_header("Content-Length", "1000000000")
                    self.end_headers()
                    if self.path.startswith("/oversize"):
                        self.wfile.write(b"x" * (byte_limit + 1))
                        self.wfile.flush()
                        stop.wait(4)
                    else:
                        while not stop.wait(0.03):
                            self.wfile.write(b"x")
                            self.wfile.flush()
                    return
                content = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    certificate = None
    if tls:
        certificate = server_certificate(path)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(path / "server.pem", path / "server.key")
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        yield SimpleNamespace(
            url=f"{'https' if tls else 'http'}://127.0.0.1:{server.server_port}",
            certificate=certificate,
            requests=requests,
            byte_limit=byte_limit,
            seen=seen,
        )
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive(), (
            "the owned loopback server must stop at test cleanup"
        )
