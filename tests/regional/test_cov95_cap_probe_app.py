from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from scripts.e2e.regional.probes import cap_probe_app as probe


class HoldIO:
    max_in_flight = 4

    def __init__(self, *, reject: bool = False) -> None:
        self.in_flight = 0
        self.attempts = 0
        self.reject = reject

    async def run(self, action: Any, duration: float) -> None:
        self.attempts += 1
        if self.reject:
            raise RuntimeError("fake Store I/O slots unavailable")
        self.in_flight += 1
        try:
            await asyncio.to_thread(action, duration)
        finally:
            self.in_flight -= 1


def app_with_io(monkeypatch: pytest.MonkeyPatch, io: HoldIO) -> FastAPI:
    base = FastAPI()
    base.state.store_io = io
    base.get("/health")(lambda: {"healthy": True})
    monkeypatch.setattr(probe, "create_base_app", lambda: base)
    return probe.create_app()


@pytest.mark.parametrize("client", [None, ("192.0.2.1", 1234), ("2001:db8::1", 1234)])
def test_remote_or_missing_peer_cannot_open_holds_even_with_forwarded_header(
    monkeypatch: pytest.MonkeyPatch, client: Any
) -> None:
    io = HoldIO()
    app = app_with_io(monkeypatch, io)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=client),
            base_url="http://example",
        ) as request:
            denied = await request.post(
                "/__cap__/hold",
                json={"tag": "remote", "durations": [5]},
                headers={"X-Forwarded-For": "127.0.0.1"},
            )
            assert denied.status_code == 403
            assert denied.json() == {"detail": "loopback only"}
            health = await request.get("/health")
            assert health.json() == {"healthy": True}

    asyncio.run(run())
    assert io.attempts == 0


@pytest.mark.parametrize("client", [("127.0.0.1", 1), ("::1", 1), ("localhost", 1)])
def test_loopback_hold_release_is_tag_scoped_and_releases_every_slot(
    monkeypatch: pytest.MonkeyPatch, client: tuple[str, int]
) -> None:
    io = HoldIO()
    app = app_with_io(monkeypatch, io)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=client),
            base_url="http://example",
        ) as request:
            try:
                hold = await request.post(
                    "/__cap__/hold", json={"tag": " example ", "durations": [5, 5]}
                )
                assert hold.status_code == 200
                assert hold.json() == {
                    "tag": "example",
                    "slots": 2,
                    "in_flight": 2,
                    "max_in_flight": 4,
                }
                duplicate = await request.post(
                    "/__cap__/hold", json={"tag": "example", "durations": [5]}
                )
                assert duplicate.status_code == 409
                unrelated = await request.post(
                    "/__cap__/release", json={"tag": "other"}
                )
                assert unrelated.json() == {"tag": "other", "released": 0}
                assert io.in_flight == 2
            finally:
                released = await request.post(
                    "/__cap__/release", json={"tag": "example"}
                )
            assert released.json() == {"tag": "example", "released": 2, "in_flight": 0}
            again = await request.post("/__cap__/release", json={"tag": "example"})
            assert again.json() == {"tag": "example", "released": 0}

    asyncio.run(run())
    assert io.in_flight == 0
    assert io.attempts == 2


@pytest.mark.parametrize(
    "payload,detail",
    [
        ({}, "tag required"),
        ({"tag": " "}, "tag required"),
        ({"tag": "empty"}, "durations must contain 1..4 values"),
        ({"tag": "five", "durations": [1] * 5}, "durations must contain 1..4 values"),
        ({"tag": "zero", "durations": [0]}, "duration out of range"),
        ({"tag": "negative", "durations": [1, -1]}, "duration out of range"),
        ({"tag": "long", "durations": [901]}, "duration out of range"),
    ],
)
def test_invalid_hold_request_never_allocates_store_slots(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any], detail: str
) -> None:
    io = HoldIO()
    app = app_with_io(monkeypatch, io)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://example",
        ) as request:
            response = await request.post("/__cap__/hold", json=payload)
            assert response.status_code == 422
            assert response.json() == {"detail": detail}

    asyncio.run(run())
    assert io.attempts == 0


def test_acquisition_timeout_clears_the_tag_and_drains_failed_slot_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = HoldIO(reject=True)
    app = app_with_io(monkeypatch, io)
    clock = [0.0]

    async def advance(_seconds: float) -> None:
        clock[0] += 11
        await asyncio.sleep(0)

    monkeypatch.setattr(
        probe,
        "asyncio",
        SimpleNamespace(
            create_task=asyncio.create_task,
            gather=asyncio.gather,
            get_running_loop=lambda: SimpleNamespace(time=lambda: clock[0]),
            sleep=advance,
        ),
    )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://example",
        ) as request:
            for _attempt in range(2):
                response = await request.post(
                    "/__cap__/hold", json={"tag": "retry", "durations": [5, 5]}
                )
                assert response.status_code == 503
                assert response.json() == {
                    "detail": "holds did not acquire Store I/O slots"
                }
            released = await request.post("/__cap__/release", json={"tag": "retry"})
            assert released.json() == {"tag": "retry", "released": 0}

    asyncio.run(run())
    assert io.attempts == 4
    assert io.in_flight == 0
