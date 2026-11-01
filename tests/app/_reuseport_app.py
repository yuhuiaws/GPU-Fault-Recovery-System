"""A minimal ASGI application for the ``gpu_fault.app.serve`` tests.

Answers ``GET /pid`` with the serving process id and ``GET /listener`` with
the registered :class:`gpu_fault.app.listener.ServiceListener` snapshot, so a
test can tell which worker accepted a connection and what it counted. Any
other path is a 404. ``broken`` is a factory that raises, for the
startup-failure path; ``waiting`` is one that waits the way a worker waits out
an Aurora failover inside ``create_app``; ``gave_up`` runs a real start-up
store wait against a store that never answers.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
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


def waiting() -> Any:
    """Return ``app`` once ``REUSEPORT_APP_RELEASE_FILE`` exists (60 s at most).

    Each worker first drops a file named after its pid into
    ``REUSEPORT_APP_ENTERED_DIR``, so the test knows it is inside the factory.
    """

    entered = Path(os.environ["REUSEPORT_APP_ENTERED_DIR"])
    (entered / str(os.getpid())).write_text("waiting", encoding="utf-8")
    release = Path(os.environ["REUSEPORT_APP_RELEASE_FILE"])
    deadline = time.monotonic() + 60
    while not release.exists():
        if time.monotonic() > deadline:
            raise RuntimeError("the test never released the factory")
        time.sleep(0.05)
    return app


def gave_up() -> Any:
    """A start-up store wait of one second against a pool that never answers."""

    from psycopg_pool import PoolTimeout

    from gpu_fault.app.startup_store_wait import STORE_STEP, StartupStoreWait

    def check_schema() -> None:
        raise PoolTimeout("couldn't get a connection after 10.00 sec")

    StartupStoreWait(1.0).run(STORE_STEP, check_schema)
    return app
