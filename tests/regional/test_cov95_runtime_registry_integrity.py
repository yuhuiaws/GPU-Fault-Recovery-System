from __future__ import annotations

import json
import logging
import traceback
from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.regional import RegionalRegistryHead, RegionalRegistryRevision
from gpu_fault.regional_registry_runtime import RegionalRegistryRuntime
from gpu_fault.store import InMemoryStore
from tests.app_services._cov95_runtime_workers import Stop
from tests.app_services.test_periodic_registry_heartbeat import CLUSTER_A, T0
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime

SYNTHETIC_SENSITIVE_VALUE = "synthetic-unit-credential-not-real"


def decoding_store(
    monkeypatch: pytest.MonkeyPatch, mode: str, record_kind: str
) -> tuple[InMemoryStore, RegionalRegistryRuntime, Any]:
    store = InMemoryStore()
    store.save_regional_cluster(CLUSTER_A)
    runtime = RegionalRegistryRuntime.bootstrap(
        store,
        member_id="unit-member",
        service_role="worker",
        release_id="unit",
        now=lambda: T0,
    )
    reader_name = f"get_regional_registry_{record_kind}"
    original = getattr(store, reader_name)
    arguments = (1,) if record_kind == "revision" else ()
    payload = original(*arguments).model_dump(mode="json")
    model = (
        RegionalRegistryRevision if record_kind == "revision" else RegionalRegistryHead
    )
    if record_kind == "revision":
        payload["registrations"][0]["token_sha256"] = SYNTHETIC_SENSITIVE_VALUE
    else:
        payload["unexpected_registration_token"] = SYNTHETIC_SENSITIVE_VALUE

    def decode(*args: Any) -> RegionalRegistryRevision | RegionalRegistryHead:
        assert args == arguments
        if mode == "json":
            return model.model_validate(json.loads(json.dumps(payload)[:-1]))
        return model.model_validate(payload)

    monkeypatch.setattr(store, reader_name, decode)
    return store, runtime, original


@pytest.mark.parametrize("mode", ["model", "json"])
@pytest.mark.parametrize("record_kind", ["head", "revision"])
def test_store_decode_failure_fences_same_head_cache_until_validated_recovery(
    monkeypatch: pytest.MonkeyPatch, mode: str, record_kind: str
) -> None:
    store, runtime, original = decoding_store(monkeypatch, mode, record_kind)
    snapshot = runtime.snapshot()
    assert runtime.refresh_once() is False
    assert runtime.is_ready() is False, (
        "invalid stored data must not borrow the transient I/O grace"
    )
    assert runtime.snapshot() == snapshot
    assert runtime.status()["last_successful_refresh"] == T0

    def unavailable(*args: Any) -> Any:
        raise OSError("synthetic subsequent read outage")

    reader_name = f"get_regional_registry_{record_kind}"
    monkeypatch.setattr(store, reader_name, unavailable)
    assert runtime.refresh_once() is False
    assert runtime.is_ready() is False
    monkeypatch.setattr(store, reader_name, original)
    assert runtime.refresh_once() is True
    assert runtime.is_ready() is True


@pytest.mark.parametrize("mode", ["model", "json"])
@pytest.mark.parametrize("record_kind", ["head", "revision"])
def test_store_decode_failure_exposes_only_a_safe_reason_in_status_members_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mode: str,
    record_kind: str,
) -> None:
    store, runtime, _original = decoding_store(monkeypatch, mode, record_kind)
    with pytest.raises(ValueError) as raised:
        runtime.refresh_once(raise_on_failure=True)
    failure = raised.value
    logging.getLogger("unit.registry-caller").error(
        "registry refresh refused",
        exc_info=(type(failure), failure, failure.__traceback__),
    )
    safe_reason = f"regional registry {record_kind} validation failed"
    assert str(failure) == safe_reason
    assert runtime.status()["error"] == "ValueError: " + safe_reason
    (member,) = store.list_regional_registry_members()
    assert member.error == runtime.status()["error"]
    assert member.ready is False
    assert SYNTHETIC_SENSITIVE_VALUE not in "".join(traceback.format_exception(failure))
    assert SYNTHETIC_SENSITIVE_VALUE not in caplog.text


@pytest.mark.parametrize("mode", ["model", "json"])
@pytest.mark.parametrize("record_kind", ["head", "revision"])
def test_bootstrap_decode_failure_is_sanitized_and_not_retried_as_an_outage(
    monkeypatch: pytest.MonkeyPatch, mode: str, record_kind: str
) -> None:
    store, _runtime, _original = decoding_store(monkeypatch, mode, record_kind)
    waits = []
    with pytest.raises(
        ValueError, match=f"regional registry {record_kind} validation failed"
    ) as raised:
        RegionalRegistryRuntime.bootstrap(
            store,
            member_id="unit-restart",
            service_role="worker",
            release_id="unit",
            now=lambda: T0,
            retry_budget_seconds=10,
            sleep=waits.append,
        )
    assert waits == []
    assert SYNTHETIC_SENSITIVE_VALUE not in "".join(
        traceback.format_exception(raised.value)
    )


@pytest.mark.parametrize("corruption", ["generation", "digest"])
def test_same_head_integrity_failure_fences_cached_registry_until_a_valid_refresh(
    monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    store = InMemoryStore()
    store.save_regional_cluster(CLUSTER_A)
    runtime = RegionalRegistryRuntime.bootstrap(
        store,
        member_id="unit-member",
        service_role="worker",
        release_id="unit",
        now=lambda: T0,
        stale_seconds=10,
    )
    original = runtime.snapshot()
    registrations = (
        [CLUSTER_A]
        if corruption == "generation"
        else [
            CLUSTER_A.model_copy(
                update={"agent_endpoint_allowed_cidrs": ["10.1.0.0/16"]}
            )
        ]
    )
    wrong = RegionalRegistryRevision.build(
        generation=2 if corruption == "generation" else 1,
        previous_generation=1 if corruption == "generation" else None,
        registrations=registrations,
        required_member_ids=[],
        reason="synthetic inconsistent read",
        created_at=T0,
    )
    read_revision = store.get_regional_registry_revision
    monkeypatch.setattr(
        store, "get_regional_registry_revision", lambda generation: wrong
    )
    assert runtime.refresh_once() is False
    assert runtime.is_ready() is False, (
        "an integrity failure is not a transient read outage"
    )
    assert "mismatch" in runtime.status()["error"]
    assert runtime.status()["last_successful_refresh"] == T0
    assert runtime.snapshot() == original

    def unavailable(*args: Any) -> Any:
        raise OSError("synthetic subsequent read outage")

    monkeypatch.setattr(store, "get_regional_registry_revision", unavailable)
    assert runtime.refresh_once() is False
    assert runtime.is_ready() is False
    monkeypatch.setattr(store, "get_regional_registry_revision", read_revision)
    assert runtime.refresh_once() is True
    assert runtime.is_ready() is True
    assert runtime.status()["error"] is None


@pytest.mark.parametrize("record_kind", ["head", "revision"])
def test_transient_same_head_read_failure_keeps_only_the_bounded_cache_grace(
    monkeypatch: pytest.MonkeyPatch, record_kind: str
) -> None:
    store = InMemoryStore()
    store.save_regional_cluster(CLUSTER_A)
    clock = [T0]
    runtime = RegionalRegistryRuntime.bootstrap(
        store,
        member_id="unit-member",
        service_role="worker",
        release_id="unit",
        now=lambda: clock[0],
        stale_seconds=10,
    )

    def unavailable(*args: Any) -> Any:
        raise OSError("synthetic transient registry read failure")

    monkeypatch.setattr(store, f"get_regional_registry_{record_kind}", unavailable)
    assert runtime.refresh_once() is False
    assert runtime.is_ready() is True
    clock[0] += timedelta(seconds=11)
    assert runtime.is_ready() is False
    assert runtime.get("cluster-a") == CLUSTER_A


@pytest.mark.parametrize("poll,stale", [(0, 10), (1, 1)])
def test_registry_interval_validation_happens_before_a_store_access(
    poll: float, stale: float
) -> None:
    with pytest.raises(ValueError, match="interval|threshold"):
        RegionalRegistryRuntime(
            object(),
            member_id="unit",
            service_role="worker",
            release_id="unit",
            poll_seconds=poll,
            stale_seconds=stale,
        )


def test_unloaded_registry_reports_unknown_and_does_not_fabricate_a_config_digest() -> (
    None
):
    runtime = RegionalRegistryRuntime(
        InMemoryStore(), member_id="unit", service_role="worker", release_id="unit"
    )
    runtime.secret_config_sha256 = "0" * 64
    assert runtime.is_ready() is False
    assert runtime.durable_config_sha256() is None
    assert runtime.secret_drift() is False
    assert runtime.status()["generation"] == 0
    with pytest.raises(RuntimeError, match="snapshot is unavailable"):
        runtime.get("cluster-a")


def test_bootstrap_commit_with_lost_ack_reads_the_winning_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStore()
    store.save_regional_cluster(CLUSTER_A)
    publish = store.publish_regional_registry_revision
    calls = []

    def committed_then_conflict(revision: Any, **kwargs: Any) -> Any:
        calls.append(revision.generation)
        publish(revision, **kwargs)
        raise ValueError("synthetic concurrent bootstrap won")

    monkeypatch.setattr(
        store, "publish_regional_registry_revision", committed_then_conflict
    )
    runtime = RegionalRegistryRuntime.bootstrap(
        store,
        member_id="unit",
        service_role="worker",
        release_id="unit",
        now=lambda: T0,
    )
    assert calls == [1]
    assert runtime.snapshot().generation == 1
    assert runtime.get("cluster-a") == CLUSTER_A
    assert runtime.is_ready() is True


def test_secret_drift_warns_on_each_transition_without_replacing_durable_membership(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryStore()
    store.save_regional_cluster(CLUSTER_A)
    runtime = RegionalRegistryRuntime.bootstrap(
        store,
        member_id="unit",
        service_role="worker",
        release_id="unit",
        now=lambda: T0,
        secret_config_sha256="0" * 64,
    )
    assert runtime.secret_drift() is True
    assert runtime.refresh_once() is True
    assert caplog.text.count("Secret drifts from the durable head") == 1
    runtime.secret_config_sha256 = runtime.durable_config_sha256()
    assert runtime.refresh_once() is True
    assert runtime.secret_drift() is False
    runtime.secret_config_sha256 = "f" * 64
    assert runtime.refresh_once() is True
    assert caplog.text.count("Secret drifts from the durable head") == 2
    assert runtime.get("cluster-a") == CLUSTER_A


def test_registry_poll_loop_stops_after_its_owned_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStore()
    runtime = RegionalRegistryRuntime.bootstrap(
        store,
        member_id="unit",
        service_role="worker",
        release_id="unit",
        now=lambda: T0,
    )
    calls = []
    read_head = store.get_regional_registry_head

    def head() -> Any:
        calls.append("head")
        return read_head()

    monkeypatch.setattr(store, "get_regional_registry_head", head)
    stop = Stop(1)
    runtime.run(stop)
    assert calls == ["head", "head"]
    assert stop.waits == [1.0, 1.0]
    assert runtime.is_ready() is True
