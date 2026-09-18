from __future__ import annotations

from io import BytesIO


class MemorySocket:
    def __init__(self, incoming: bytes):
        self.incoming = incoming
        self.outgoing = bytearray()

    def makefile(self, mode, buffering=None):
        return BytesIO(self.incoming)

    def sendall(self, data):
        self.outgoing.extend(data)


class FakeHTTPServer:
    def __init__(self, address, controller):
        self.server_address = address
        self.controller = controller
        self.calls = []

    def serve_forever(self):
        self.calls.append("serve")

    def shutdown(self):
        self.calls.append("shutdown")

    def server_close(self):
        self.calls.append("close")


class FakeThread:
    def __init__(self, *, target, name, daemon):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.running = False
        self.joins = []

    def start(self):
        self.running = True
        self.target()

    def is_alive(self):
        return self.running

    def join(self, timeout=None):
        self.joins.append(timeout)
        self.running = False
