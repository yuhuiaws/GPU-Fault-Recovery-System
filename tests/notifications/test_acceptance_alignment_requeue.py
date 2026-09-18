from __future__ import annotations

import asyncio
import builtins

import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app.authorization import ExplicitAuthorizationRegistry
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional.probes.notify003_requeue_drill import route_drill
from scripts.e2e.regional.run_notification_acceptance import requeue_route_errors


def test_dead_letter_requeue_exercises_http_auth_send_requeue_dispatch_and_dedup() -> (
    None
):
    evidence = asyncio.run(route_drill())
    assert requeue_route_errors(evidence) == [], evidence
    for key in evidence:
        changed = {**evidence, key: None}
        assert requeue_route_errors(changed), key


def test_requeue_is_in_the_real_application_execution_token_inventory() -> None:
    app = create_app(ApplicationContext(store=InMemoryStore()))
    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)
    assert (
        registry.declared(
            "/v1/advisory-notifications/unit-notification/requeue", "POST"
        )
        == "execution-token"
    )


def test_requeue_probe_does_not_require_development_http_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def without_development_clients(name, *args, **kwargs):
        if name.partition(".")[0] in {"httpx", "httpx2"}:
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_development_clients)
    evidence = asyncio.run(route_drill())
    assert requeue_route_errors(evidence) == [], evidence
