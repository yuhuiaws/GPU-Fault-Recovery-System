from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone, tzinfo
from typing import cast

import pytest
from fastapi import FastAPI

from gpu_fault.app import ApplicationContext
from gpu_fault.app.routes import regional_registry as registry_routes
from gpu_fault.regional import RegionalRegistryMember
from gpu_fault.regional_registry_runtime import RegionalRegistryRuntime
from tests._builders import asgi_client
from tests.regional._regional_support import TOKEN_A
from tests.regional.test_regional_registry_api import context_and_app, operator_headers


@pytest.fixture
def registry_app(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ApplicationContext, FastAPI, list[datetime]]:
    context, app = context_and_app()
    clock = [datetime.now(timezone.utc)]
    runtime = cast(RegionalRegistryRuntime, app.state.regional_registry_runtime)
    runtime.now = lambda: clock[0]

    class RegistryClock:
        @staticmethod
        def now(_timezone: tzinfo) -> datetime:
            return clock[0]

    monkeypatch.setattr(registry_routes, "datetime", RegistryClock)
    return context, app, clock


def test_status_retires_an_expired_required_member_but_not_an_absent_row(
    registry_app: tuple[ApplicationContext, FastAPI, list[datetime]],
) -> None:
    context, app, clock = registry_app
    runtime = cast(RegionalRegistryRuntime, app.state.regional_registry_runtime)
    departing = RegionalRegistryRuntime.bootstrap(
        context.store,
        member_id="departing/process",
        service_role="control-worker",
        release_id=runtime.release_id,
        stale_seconds=runtime.stale_seconds,
        now=lambda: clock[0],
    )
    required = sorted([runtime.member_id, departing.member_id])

    async def scenario() -> None:
        async with asgi_client(app) as client:
            published = await client.post(
                "/v1/regional/registry/revisions",
                headers=operator_headers(),
                json={
                    "expected_generation": 1,
                    "registrations": [
                        item.model_dump(mode="json")
                        for item in context.store.list_regional_clusters()
                    ],
                    "reason": "synthetic rolling process replacement",
                },
            )
            assert published.status_code == 200, "the local publish must succeed"
            assert published.json()["required_member_ids"] == required
            assert published.json()["missing_member_ids"] == [departing.member_id]
            assert published.json()["converged"] is False

            clock[0] += timedelta(seconds=runtime.stale_seconds)
            assert runtime.refresh_once(), "the survivor ACK stays fresh"
            assert departing.is_ready(), "the last heartbeat is valid at the boundary"
            boundary = await client.get(
                "/v1/regional/registry/status", headers=operator_headers()
            )
            assert boundary.json()["converged"] is False
            assert boundary.json()["missing_member_ids"] == [departing.member_id]

            clock[0] += timedelta(microseconds=1)
            assert runtime.refresh_once(), "the survivor still serves the new head"
            assert not departing.is_ready(), "the retired row cannot authorize traffic"
            departed = await client.get(
                "/v1/regional/registry/status", headers=operator_headers()
            )
            body = departed.json()
            assert departed.status_code == 200
            assert body["converged"] is True
            assert body["required_member_ids"] == required
            assert body["acked_member_ids"] == [runtime.member_id]
            assert body["active_member_ids"] == [runtime.member_id]
            assert body["missing_member_ids"] == []

            assert (
                context.store.cleanup_stale_regional_registry_members(
                    older_than=departing.started_at, limit=10
                )
                == 1
            ), "remove only the synthetic departed row"
            missing = await client.get(
                "/v1/regional/registry/status", headers=operator_headers()
            )
            assert missing.json()["converged"] is False
            assert missing.json()["missing_member_ids"] == [departing.member_id], (
                "a missing required row cannot supply departure evidence"
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("empty_required", [False, True], ids=["captured", "empty"])
@pytest.mark.parametrize(
    "defect", ["not-ready", "old-generation", "newer-generation", "digest", "future"]
)
def test_status_includes_post_publish_members_in_both_blockers_and_acks(
    registry_app: tuple[ApplicationContext, FastAPI, list[datetime]],
    empty_required: bool,
    defect: str,
) -> None:
    context, app, clock = registry_app
    runtime = cast(RegionalRegistryRuntime, app.state.regional_registry_runtime)

    async def scenario() -> None:
        async with asgi_client(app) as client:
            published = await client.post(
                "/v1/regional/registry/revisions",
                headers=operator_headers(),
                json={
                    "expected_generation": 1,
                    "registrations": [],
                    "required_member_ids": [] if empty_required else None,
                    "reason": "synthetic post-publish arrival",
                },
            )
            assert published.status_code == 200
            target = context.store.get_regional_registry_revision(2)
            newcomer = RegionalRegistryMember(
                member_id="newcomer/process",
                service_role="control-worker",
                release_id=runtime.release_id,
                generation=(
                    1
                    if defect == "old-generation"
                    else 3
                    if defect == "newer-generation"
                    else 2
                ),
                content_sha256="b" * 64
                if defect == "digest"
                else target.content_sha256,
                ready=defect != "not-ready",
                started_at=clock[0],
                last_seen_at=clock[0]
                + timedelta(seconds=1 if defect == "future" else 0),
            )
            context.store.save_regional_registry_member(newcomer)
            status = await client.get(
                "/v1/regional/registry/status", headers=operator_headers()
            )
            body = status.json()
            assert status.status_code == 200
            assert (
                body["required_member_ids"] == published.json()["required_member_ids"]
            )
            assert body["converged"] is False
            assert body["missing_member_ids"] == [newcomer.member_id]
            assert body["acked_member_ids"] == [runtime.member_id]

            context.store.save_regional_registry_member(
                newcomer.model_copy(
                    update={
                        "generation": target.generation,
                        "content_sha256": target.content_sha256,
                        "ready": True,
                        "last_seen_at": clock[0],
                    }
                )
            )
            acked = await client.get(
                "/v1/regional/registry/status", headers=operator_headers()
            )
            assert acked.json()["converged"] is True
            assert acked.json()["missing_member_ids"] == []
            assert acked.json()["acked_member_ids"] == sorted(
                [runtime.member_id, newcomer.member_id]
            ), "ACK diagnostics must include post-publish arrivals"

    asyncio.run(scenario())


def test_status_requires_a_live_replacement_after_the_whole_required_fleet_expires(
    registry_app: tuple[ApplicationContext, FastAPI, list[datetime]],
) -> None:
    context, app, clock = registry_app
    runtime = cast(RegionalRegistryRuntime, app.state.regional_registry_runtime)

    async def scenario() -> None:
        async with asgi_client(app) as client:
            published = await client.post(
                "/v1/regional/registry/revisions",
                headers=operator_headers(),
                json={
                    "expected_generation": 1,
                    "registrations": [],
                    "reason": "synthetic complete process replacement",
                },
            )
            assert published.status_code == 200
            clock[0] += timedelta(seconds=runtime.stale_seconds + 1)
            expired = await client.get(
                "/v1/regional/registry/status", headers=operator_headers()
            )
            assert expired.json()["converged"] is False
            assert expired.json()["missing_member_ids"] == [runtime.member_id]
            assert expired.json()["active_member_ids"] == []
            assert expired.json()["acked_member_ids"] == []

            replacement = RegionalRegistryRuntime.bootstrap(
                context.store,
                member_id="replacement/process",
                service_role="ingress",
                release_id=runtime.release_id,
                stale_seconds=runtime.stale_seconds,
                now=lambda: clock[0],
            )
            status = await client.get(
                "/v1/regional/registry/status", headers=operator_headers()
            )
            assert status.json()["converged"] is True
            assert status.json()["required_member_ids"] == [runtime.member_id]
            assert status.json()["acked_member_ids"] == [replacement.member_id]
            assert status.json()["missing_member_ids"] == []

    asyncio.run(scenario())


def test_bootstrap_status_is_not_converged_when_its_known_members_all_expire(
    registry_app: tuple[ApplicationContext, FastAPI, list[datetime]],
) -> None:
    _context, app, clock = registry_app
    runtime = cast(RegionalRegistryRuntime, app.state.regional_registry_runtime)
    clock[0] += timedelta(seconds=runtime.stale_seconds + 1)

    async def scenario() -> None:
        async with asgi_client(app) as client:
            status = await client.get(
                "/v1/regional/registry/status", headers=operator_headers()
            )
            assert status.status_code == 200
            assert status.json()["required_member_ids"] == []
            assert status.json()["converged"] is False
            assert status.json()["missing_member_ids"] == [runtime.member_id]
            assert status.json()["acked_member_ids"] == []
            assert status.json()["active_member_ids"] == []

    asyncio.run(scenario())


def test_actual_cluster_traffic_stops_when_only_the_heartbeat_write_is_stale(
    registry_app: tuple[ApplicationContext, FastAPI, list[datetime]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, app, clock = registry_app
    runtime = cast(RegionalRegistryRuntime, app.state.regional_registry_runtime)
    (member,) = context.store.list_regional_registry_members()
    save_member = context.store.save_regional_registry_member

    def unavailable(_member: RegionalRegistryMember) -> None:
        raise ConnectionError("synthetic heartbeat write outage")

    monkeypatch.setattr(context.store, "save_regional_registry_member", unavailable)
    headers = {
        **operator_headers(),
        "Authorization": f"Bearer {TOKEN_A}",
        "X-GPU-Fault-Cluster-ID": "cluster-a",
    }
    path = "/v1/regional/executors/hyperpod-submissions"
    params = {"cluster_name": "hp-cluster-a", "idempotency_key": "synthetic-missing"}

    async def scenario() -> None:
        async with asgi_client(app) as client:
            clock[0] = member.last_seen_at + timedelta(seconds=runtime.stale_seconds)
            assert not runtime.refresh_once(), "the durable heartbeat cannot be renewed"
            boundary = await client.get(path, headers=headers, params=params)
            assert boundary.status_code == 200, "bounded transient I/O grace remains"

            clock[0] += timedelta(microseconds=1)
            assert not runtime.refresh_once(), "validated reads still succeed"
            assert runtime.status()["last_successful_refresh"] == clock[0]
            refused = await client.get(path, headers=headers, params=params)
            assert refused.status_code == 503, "a retired process cannot authenticate"
            assert refused.headers["retry-after"] == "2"
            health = await client.get("/healthz")
            assert health.status_code == 503
            assert health.json()["regional_registry"]["ready"] is False

            monkeypatch.setattr(
                context.store, "save_regional_registry_member", save_member
            )
            assert runtime.refresh_once(), "both read and write proofs are repaired"
            restored = await client.get(path, headers=headers, params=params)
            assert restored.status_code == 200

    asyncio.run(scenario())
