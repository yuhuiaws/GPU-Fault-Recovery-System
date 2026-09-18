"""NET-007 deadman admission, retry and UID-precondition behavior."""

from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
from kubernetes import client, config
from urllib3.exceptions import HTTPError

from scripts.e2e.regional.probes import net007_deadman as probe
from tests.regional._cov95_collect_net import Clock, no_external_effects  # noqa: F401


def webhook(uid: str = "uid-a", **changes: Any) -> Any:
    return SimpleNamespace(
        metadata=SimpleNamespace(
            **{
                "uid": uid,
                "resource_version": "version-a",
                "labels": {probe.RUN_LABEL: "run-a"},
                **changes,
            }
        )
    )


def api_reading(*values: Any) -> Any:
    pending = deque(values)
    deleted = []

    def read(name: str, **kwargs: Any) -> Any:
        assert kwargs["_request_timeout"] == (5, 10)
        value = pending.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    return SimpleNamespace(
        deleted=deleted,
        read_validating_webhook_configuration=read,
        delete_validating_webhook_configuration=lambda name, **kw: deleted.append(
            kw["body"]
        ),
    )


@pytest.mark.parametrize(
    "outcome",
    ["absent", "read-error", "removed", "verify-error", "pending", "replaced"],
)
def test_removal_requires_exact_preconditions_and_confirmed_absence(
    outcome: str,
) -> None:
    first = (
        client.exceptions.ApiException(status=404 if outcome == "absent" else 403)
        if outcome in {"absent", "read-error"}
        else webhook()
    )
    second = (
        client.exceptions.ApiException(status=404 if outcome == "removed" else 403)
        if outcome in {"removed", "verify-error"}
        else webhook("replacement" if outcome == "replaced" else "uid-a")
    )
    api = api_reading(first, second)
    if outcome in {"read-error", "verify-error"}:
        with pytest.raises(client.exceptions.ApiException):
            probe.remove_webhook(api, name="webhook", run_id="run-a")
    elif outcome == "replaced":
        with pytest.raises(RuntimeError, match="replaced"):
            probe.remove_webhook(api, name="webhook", run_id="run-a")
    else:
        assert probe.remove_webhook(api, name="webhook", run_id="run-a") is (
            outcome != "pending"
        )
    if outcome not in {"absent", "read-error"}:
        assert api.deleted[0].preconditions.uid == "uid-a"
        assert api.deleted[0].preconditions.resource_version == "version-a"


@pytest.mark.parametrize(
    "changes",
    [
        {"uid": ""},
        {"resource_version": ""},
        {"labels": None},
        {"labels": {probe.RUN_LABEL: "foreign"}},
    ],
)
def test_removal_refuses_unproven_or_foreign_ownership(changes: Any) -> None:
    api = api_reading(webhook(**changes))
    with pytest.raises(RuntimeError, match="ownership"):
        probe.remove_webhook(api, name="webhook", run_id="run-a")
    assert api.deleted == []


@pytest.mark.parametrize(
    "changes",
    [{"name": ""}, {"run_id": ""}, {"restore_at": float("nan")}, {"restore_at": 999}],
)
def test_watch_rejects_invalid_admission(monkeypatch: Any, changes: Any) -> None:
    monkeypatch.setattr(probe, "time", Clock())
    with pytest.raises(RuntimeError, match="invalid"):
        probe.watch(
            api_reading(),
            **{"name": "webhook", "run_id": "run-a", "restore_at": 1010, **changes},
        )


@pytest.mark.parametrize(
    "initial", [webhook(), client.exceptions.ApiException(status=403)]
)
def test_watch_cannot_adopt_existing_webhook_or_failed_read(
    monkeypatch: Any, initial: Any
) -> None:
    monkeypatch.setattr(probe, "time", Clock())
    with pytest.raises((RuntimeError, client.exceptions.ApiException)):
        probe.watch(
            api_reading(initial), name="webhook", run_id="run-a", restore_at=1010
        )


@pytest.mark.parametrize(
    "failure",
    [
        client.exceptions.ApiException(status=503),
        HTTPError("offline"),
        TimeoutError("timeout"),
        False,
    ],
)
def test_watch_retries_transient_removal_and_reports_only_confirmed_success(
    monkeypatch: Any, capsys: Any, failure: Any
) -> None:
    clock = Clock()
    monkeypatch.setattr(probe, "time", clock)
    attempts = []

    def remove(*a: Any, **k: Any) -> bool:
        attempts.append(clock.now)
        if len(attempts) == 1:
            if isinstance(failure, Exception):
                raise failure
            return False
        return True

    monkeypatch.setattr(probe, "remove_webhook", remove)
    probe.watch(
        api_reading(client.exceptions.ApiException(status=404)),
        name="webhook",
        run_id="run-a",
        restore_at=1010,
    )
    assert attempts == [1010, 1012]
    assert [
        json.loads(line)["state"] for line in capsys.readouterr().out.splitlines()
    ] == ["ARMED", "REMOVED"]


@pytest.mark.parametrize("status", [403, 503])
def test_watch_stops_on_permanent_failure_or_deadline(
    monkeypatch: Any, status: int
) -> None:
    clock = Clock()
    monkeypatch.setattr(probe, "time", clock)
    calls = []

    def remove(*a: Any, **k: Any) -> bool:
        calls.append(clock.now)
        if status == 503:
            clock.now += 121
        raise client.exceptions.ApiException(status=status)

    monkeypatch.setattr(probe, "remove_webhook", remove)
    with pytest.raises(
        RuntimeError, match="refused" if status == 403 else "could not confirm"
    ):
        probe.watch(
            api_reading(client.exceptions.ApiException(status=404)),
            name="webhook",
            run_id="run-a",
            restore_at=1010,
        )
    assert len(calls) == 1


def test_main_uses_incluster_client_and_explicit_identity(monkeypatch: Any) -> None:
    calls = []
    api = object()
    monkeypatch.setenv("NET007_WEBHOOK_NAME", "webhook-a")
    monkeypatch.setenv("NET007_RUN_ID", "run-a")
    monkeypatch.setenv("NET007_RESTORE_AT", "1010")
    monkeypatch.setattr(config, "load_incluster_config", lambda: calls.append("config"))
    monkeypatch.setattr(client, "AdmissionregistrationV1Api", lambda: api)
    monkeypatch.setattr(
        probe, "watch", lambda instance, **kw: calls.append((instance, kw))
    )
    probe.main()
    assert calls == [
        "config",
        (api, {"name": "webhook-a", "run_id": "run-a", "restore_at": 1010.0}),
    ]
