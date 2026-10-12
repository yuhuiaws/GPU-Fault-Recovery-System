"""A minimal ASGI application for the ``gpu_fault.app.serve`` tests.

Answers ``GET /pid`` with the serving process id and ``GET /listener`` with
the registered :class:`gpu_fault.app.listener.ServiceListener` snapshot, so a
test can tell which worker accepted a connection and what it counted. Any
other path is a 404. ``broken`` is a factory that raises, for the
startup-failure path.
"""

from __future__ import annotations

import json
import os
from typing import Any

from gpu_fault.app.listener import service_listener


async def _respond(send: Any, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def app(scope: Any, receive: Any, send: Any) -> None:
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    if scope["type"] != "http":
        return
    path = scope.get("path", "")
    if path == "/pid":
        await _respond(send, 200, {"pid": os.getpid()})
    elif path == "/listener":
        listener = service_listener()
        await _respond(
            send,
            200,
            {"pid": os.getpid(), "listener": listener and listener.snapshot()},
        )
    else:
        await _respond(send, 404, {"detail": "Not Found"})


def broken() -> Any:
    raise RuntimeError("this factory never produces an application")
