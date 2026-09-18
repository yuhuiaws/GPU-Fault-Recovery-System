from __future__ import annotations

import base64
import json
import runpy
import sys
import traceback
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from kubernetes import client, config

from gpu_fault import aurora_credential_refresh as aurora
from gpu_fault import logging_setup
from tests.notifications._cov95_runtime_aurora import (
    SECRET_REFERENCE,
    SYNTHETIC_DSN,
    core,
    encode,
    main_fakes,
    refresh,
)
from tests.regional.test_aurora_credential_refresh import FakeApps


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://local%40example:unit-old@Database.invalid/local?sslmode=verify-full#fragment",
        "postgresql://local:unit-old@[2001:db8::1]:5433/local?sslmode=verify-full",
    ],
    ids=["escaped-username", "ipv6-authority"],
)
def test_password_rotation_preserves_escaped_identity_and_host_authority(dsn) -> None:
    updated = aurora.replace_password(dsn, "unit:new/value")
    before, after = urlsplit(dsn), urlsplit(updated)
    assert after.username == before.username
    assert after.hostname == before.hostname
    assert after.port == before.port
    assert after.path == before.path
    assert after.query == before.query
    assert after.fragment == before.fragment
    assert after.password == "unit%3Anew%2Fvalue"
    assert after.netloc.rsplit("@", 1)[-1] == before.netloc.rsplit("@", 1)[-1]


def test_final_failure_traceback_does_not_reemit_the_driver_exception(monkeypatch):
    target, apps, _ = main_fakes(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_AURORA_REFRESH_MAX_ATTEMPTS", "1")

    def reject(dsn):
        raise RuntimeError("unit-driver-detail-not-for-terminal")

    monkeypatch.setattr(aurora, "verify_dsn", reject)
    with pytest.raises(RuntimeError, match="failed after retries") as failure:
        aurora.main([])
    leaked_driver_context = "unit-driver-detail-not-for-terminal" in "".join(
        traceback.format_exception(failure.value)
    )
    assert leaked_driver_context is False, (
        "driver details belong only in redacted status/logging, not the raw terminal traceback"
    )
    assert apps.patches == []
    assert base64.b64decode(target.data["postgres-url"]).decode() == SYNTHETIC_DSN


@pytest.mark.parametrize(
    "dsn,message",
    [
        ("postgresql:///local", "no host"),
        ("postgresql://database.invalid/local", "no username"),
        (
            "postgresql://local:unit-old@database.invalid:bad/local",
            "Port could not be cast",
        ),
    ],
    ids=["missing-host", "missing-username", "invalid-port"],
)
def test_missing_dsn_identity_is_rejected(dsn, message):
    with pytest.raises(ValueError, match=message):
        aurora.replace_password(dsn, "unit-placeholder")


@pytest.mark.parametrize(
    "payload", [{"password": "unit-placeholder"}, {}, {"password": ""}]
)
def test_password_source_requests_only_current_stage_from_fake_sdk(
    monkeypatch, payload
):
    clients, requests = [], []

    def get_secret_value(**kwargs):
        requests.append(kwargs)
        return {"SecretString": json.dumps(payload)}

    def sdk_client(service, **kwargs):
        clients.append((service, kwargs))
        return SimpleNamespace(get_secret_value=get_secret_value)

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=sdk_client))
    if payload.get("password"):
        assert (
            aurora.read_current_password(SECRET_REFERENCE, "us-west-2")
            == payload["password"]
        )
    else:
        with pytest.raises(RuntimeError, match="no password field"):
            aurora.read_current_password(SECRET_REFERENCE, "us-west-2")
    assert clients == [("secretsmanager", {"region_name": "us-west-2"})]
    assert requests == [{"SecretId": SECRET_REFERENCE, "VersionStage": "AWSCURRENT"}]


@pytest.mark.parametrize("missing", ["boto3", "psycopg"])
def test_optional_dependency_guards_do_not_contact_services(monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(RuntimeError, match="install gpu-fault-control-plane"):
        if missing == "boto3":
            aurora.read_current_password(SECRET_REFERENCE, "us-west-2")
        else:
            aurora.verify_dsn(SYNTHETIC_DSN)


@pytest.mark.parametrize("fail", [True, False])
def test_dsn_verifier_uses_one_bounded_connection_and_closes_cursor(monkeypatch, fail):
    events = []

    class Cursor:
        def execute(self, query):
            events.append(("execute", query))
            if fail:
                raise RuntimeError("unit probe refused")

        def fetchone(self):
            events.append("fetchone")
            return (1,)

    @contextmanager
    def cursor():
        events.append("cursor-open")
        try:
            yield Cursor()
        finally:
            events.append("cursor-close")

    @contextmanager
    def connect(dsn, **kwargs):
        assert dsn == SYNTHETIC_DSN
        events.append(("connect", kwargs))
        try:
            yield SimpleNamespace(cursor=cursor)
        finally:
            events.append("connection-close")

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect))
    if fail:
        with pytest.raises(RuntimeError, match="unit probe refused"):
            aurora.verify_dsn(SYNTHETIC_DSN, connect_timeout=7)
    else:
        aurora.verify_dsn(SYNTHETIC_DSN, connect_timeout=7)
    assert events == [
        ("connect", {"connect_timeout": 7}),
        "cursor-open",
        ("execute", "SELECT 1"),
        *([] if fail else ["fetchone"]),
        "cursor-close",
        "connection-close",
    ]


def test_kubernetes_client_factory_only_uses_injected_clients(monkeypatch):
    loaded = []
    expected = (object(), object())
    monkeypatch.setattr(config, "load_incluster_config", lambda: loaded.append(True))
    monkeypatch.setattr(client, "CoreV1Api", lambda: expected[0])
    monkeypatch.setattr(client, "AppsV1Api", lambda: expected[1])
    assert aurora.load_kubernetes_clients() == expected
    assert loaded == [True]


@pytest.mark.parametrize("restart", [True, False])
def test_arn_only_change_retains_password_and_resumes_only_explicit_rollout(restart):
    target, apps = core(), FakeApps()
    target.annotations[aurora.RESTART_ANNOTATION] = "previous-rollout"
    result = refresh(
        target,
        apps,
        fetch_password=lambda arn, region: "unit-old",
        expected_secret_arn=f"{SECRET_REFERENCE}-updated",
        restart_deployments=restart,
        verify=lambda dsn: pytest.fail(
            "unchanged credentials should not open a connection"
        ),
    )
    assert result.rotated is True
    assert result.restarted is restart
    assert result.reason == (
        "managed secret ARN updated and deployment rollout resumed"
        if restart
        else "managed secret ARN updated"
    )
    assert len(target.patches) == 1
    assert set(target.patches[0]["data"]) == {"master-secret-arn"}
    assert target.data["postgres-url"] == encode(SYNTHETIC_DSN)
    assert apps.patched_names == (["gpu-fault-api-ha"] if restart else [])


def test_main_retries_transient_fetch_with_bounded_backoff_and_no_default_rollout(
    monkeypatch,
):
    target, apps, sleeps = main_fakes(monkeypatch)
    calls = []

    def fetch(arn, region):
        calls.append((arn, region))
        if len(calls) < 3:
            raise RuntimeError("unit transient provider failure")
        return "unit-new"

    monkeypatch.setattr(aurora, "read_current_password", fetch)
    aurora.main([])
    assert len(calls) == 3
    assert sleeps == [1, 2]
    assert apps.patches == []
    status = json.loads(base64.b64decode(target.data[aurora.STATUS_KEY]))
    assert status["status"] == "ok"
    assert status["rotated"] is True
    assert status["restarted"] is False
    assert status["error"] is None


def test_failure_reporting_errors_do_not_hide_the_failed_refresh(monkeypatch, caplog):
    target, _, sleeps = main_fakes(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_AURORA_REFRESH_MAX_ATTEMPTS", "1")

    def refuse(*args, **kwargs):
        raise RuntimeError("unit transport refused")

    monkeypatch.setattr(aurora, "read_current_password", refuse)
    monkeypatch.setattr(target, "patch_namespaced_secret", refuse)
    monkeypatch.setattr(target, "create_namespaced_event", refuse)
    with pytest.raises(RuntimeError, match="failed after retries"):
        aurora.main([])
    assert sleeps == []
    assert "failed to record" in caplog.text
    assert "failed to create" in caplog.text


def test_main_requires_region_before_loading_any_clients(monkeypatch):
    for name in ("GPU_FAULT_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(aurora, "configure_logging", lambda: None)
    monkeypatch.setattr(
        aurora,
        "load_kubernetes_clients",
        lambda: pytest.fail("unexpected client creation"),
    )
    with pytest.raises(RuntimeError, match="AWS_REGION is required"):
        aurora.main([])


def test_module_entrypoint_stops_on_missing_region_without_side_effects(monkeypatch):
    for name in ("GPU_FAULT_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(logging_setup, "configure_logging", lambda: None)
    monkeypatch.setattr(sys, "argv", ["aurora-credential-refresh"])
    monkeypatch.delitem(sys.modules, aurora.__name__)
    with pytest.raises(RuntimeError, match="AWS_REGION is required"):
        runpy.run_module(aurora.__name__, run_name="__main__")
