"""Isolated app wrapper with loopback-only Store I/O hold controls."""

from __future__ import annotations

import asyncio
from threading import Event
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from gpu_fault.app import create_app as create_base_app


def create_app() -> FastAPI:
    app = create_base_app()
    holds: dict[str, tuple[list[Event], list[asyncio.Task[Any]]]] = {}

    @app.middleware("http")  # type: ignore[untyped-decorator]
    async def capacity_controls(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        if path not in {"/__cap__/hold", "/__cap__/release"}:
            return await call_next(request)
        client_host = request.client.host if request.client else None
        if client_host not in {"127.0.0.1", "::1", "localhost"}:
            return JSONResponse(status_code=403, content={"detail": "loopback only"})
        payload = await request.json()
        tag = str(payload.get("tag", "")).strip()
        if not tag:
            return JSONResponse(status_code=422, content={"detail": "tag required"})
        if path.endswith("/hold"):
            durations = [float(item) for item in payload.get("durations", [])]
            if not durations or len(durations) > 4:
                return JSONResponse(
                    status_code=422,
                    content={"detail": "durations must contain 1..4 values"},
                )
            if any(item <= 0 or item > 900 for item in durations):
                return JSONResponse(
                    status_code=422,
                    content={"detail": "duration out of range"},
                )
            if tag in holds:
                return JSONResponse(
                    status_code=409, content={"detail": "tag already active"}
                )
            events = [Event() for _ in durations]
            baseline = app.state.store_io.in_flight
            tasks = [
                asyncio.create_task(app.state.store_io.run(event.wait, duration))
                for event, duration in zip(events, durations, strict=True)
            ]
            holds[tag] = (events, tasks)
            deadline = asyncio.get_running_loop().time() + 10
            target = baseline + len(events)
            while app.state.store_io.in_flight < target:
                if asyncio.get_running_loop().time() >= deadline:
                    for event in events:
                        event.set()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    holds.pop(tag, None)
                    return JSONResponse(
                        status_code=503,
                        content={"detail": "holds did not acquire Store I/O slots"},
                    )
                await asyncio.sleep(0.01)
            return JSONResponse(
                content={
                    "tag": tag,
                    "slots": len(events),
                    "in_flight": app.state.store_io.in_flight,
                    "max_in_flight": app.state.store_io.max_in_flight,
                }
            )
        active = holds.pop(tag, None)
        if active is None:
            return JSONResponse(content={"tag": tag, "released": 0})
        events, tasks = active
        for event in events:
            event.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        return JSONResponse(
            content={
                "tag": tag,
                "released": len(events),
                "in_flight": app.state.store_io.in_flight,
            }
        )

    return app
