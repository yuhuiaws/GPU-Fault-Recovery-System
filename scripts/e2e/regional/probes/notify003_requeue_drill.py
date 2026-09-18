"""Exercise the installed public notification routes without production I/O."""

from __future__ import annotations

import asyncio
import http.client
import json
import secrets
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import uvicorn
from fastapi import FastAPI

from gpu_fault.app.authorization import (
    EXECUTION_TOKEN_HEADER,
    ExplicitAuthorizationRegistry,
)
from gpu_fault.app.middleware.auth import (
    RegionalAuthDependencies,
    install_regional_authorization,
)
from gpu_fault.app.routes.incidents import (
    IncidentRouterDependencies,
    get_incident_dependencies,
    router,
)
from gpu_fault.async_store import AsyncStoreExecutor
from gpu_fault.models import (
    AdvisoryNotification,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.store import InMemoryStore


class RecordingNotifier:
    def __init__(self) -> None:
        self.calls = 0

    def send(self, notification: AdvisoryNotification) -> NotificationResult:
        self.calls += 1
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id="isolated-requeue-receipt",
        )


@asynccontextmanager
async def isolated_http(
    app: FastAPI,
) -> AsyncIterator[
    Callable[[str, dict[str, str]], Awaitable[tuple[int, dict[str, Any]]]]
]:
    """Use only production dependencies and an ephemeral loopback listener."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                loop="asyncio",
                http="h11",
                ws="none",
                lifespan="off",
                access_log=False,
                log_config=None,
                log_level="critical",
            )
        )
        task = asyncio.create_task(server.serve(sockets=[listener]))

        def request(path: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.request(
                    "POST",
                    path,
                    body=json.dumps({"limit": 1}),
                    headers={"Content-Type": "application/json", **headers},
                )
                response = connection.getresponse()
                body = json.loads(response.read(1024 * 1024))
                if not isinstance(body, dict):
                    raise RuntimeError(
                        "isolated notification response is not an object"
                    )
                return response.status, body
            finally:
                connection.close()

        async def post(
            path: str, headers: dict[str, str]
        ) -> tuple[int, dict[str, Any]]:
            return await asyncio.to_thread(request, path, headers)

        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("isolated notification listener stopped")
                    await asyncio.sleep(0.01)
            yield post
        finally:
            server.should_exit = True
            try:
                await asyncio.wait_for(task, timeout=5)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)


async def route_drill() -> dict[str, Any]:
    store = InMemoryStore()
    notifier = RecordingNotifier()
    notification = AdvisoryNotification(
        notification_id="notify003-isolated",
        incident_id="notify003-isolated-incident",
        category="ADVISORY",
        cluster_name="notify003-isolated-cluster",
        subject="[DRILL] isolated backlog route proof",
        body_text="No production notification or provider call.",
        support_case_draft="No support action.",
        deduplication_key="notify003-isolated",
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    store.save_notification_if_absent(notification)
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)
    token = secrets.token_urlsafe(32)
    context = SimpleNamespace(
        store=store,
        advisory_notifications=service,
        execution_token=token,
        regional_mode=True,
    )
    store_io = AsyncStoreExecutor(
        workers=1, max_in_flight=2, admission_timeout_seconds=1
    )
    dependencies = IncidentRouterDependencies(
        context=context,
        store_io=store_io,
        notification_owner="notify003-isolated",
        notification_lease_seconds=30,
        notification_max_attempts=2,
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_incident_dependencies] = lambda: dependencies
    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)
    install_regional_authorization(
        app,
        RegionalAuthDependencies(
            context=context,
            replay_authorized=lambda _request: False,
            authorization_bucket=registry.effective,
            authenticate_cluster=lambda *_args: None,
            decode_io=store_io,
            decode_json_body=lambda body, _encoding: (body, json.loads(body)),
            payload_cluster_ids=lambda _payload: set(),
            processor_max_request_bytes=1024,
            route_exists=registry.route_exists,
        ),
    )
    path = f"/v1/advisory-notifications/{notification.notification_id}"
    headers = {EXECUTION_TOKEN_HEADER: token}
    try:
        async with isolated_http(app) as request:

            async def post(endpoint: str) -> dict[str, Any]:
                status, body = await request(endpoint, headers)
                if not 200 <= status < 300:
                    raise RuntimeError(
                        f"isolated notification request failed with HTTP {status}"
                    )
                return body

            initial = await post("/v1/advisory-notifications/dispatch")
            denied_status, _denied_body = await request(path + "/requeue", {})
            sent = await post(path + "/send")
            before = store.get_notification_delivery(notification.notification_id)
            requeued = await post(path + "/requeue")
            after = store.get_notification_delivery(notification.notification_id)
            dispatched = await post("/v1/advisory-notifications/dispatch")
            duplicate = await post(path + "/requeue")
            second = await post("/v1/advisory-notifications/dispatch")
            assert before is not None and after is not None
            result = {
                "suppressed_backlog": initial["suppressed_backlog"],
                "unauthorized_status": denied_status,
                "send_status": sent["status"],
                "status_before_requeue": before.status.value,
                "requeue_status": requeued["status"],
                "status_after_requeue": after.status.value,
                "attempts_after_requeue": after.attempts,
                "dispatch_sent": dispatched["sent"],
                "duplicate_requeue_status": duplicate["status"],
                "second_dispatch_sent": second["sent"],
                "notifier_calls": notifier.calls,
                "store_scope": "isolated-memory",
                "request_path": "/v1/advisory-notifications/{notification_id}/requeue",
                "authorization_bucket": registry.declared(path + "/requeue", "POST"),
            }
    finally:
        store_io.close()
    return {**result, "store_io_closed": True}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(route_drill()), sort_keys=True))
