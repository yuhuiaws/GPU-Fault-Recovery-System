from __future__ import annotations

import asyncio
from contextlib import closing

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.channel_registry import HOST_TELEMETRY_PATH
from gpu_fault.collectors.host.collector import HostTelemetryCollector
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.host_health import HostTelemetryBatch
from gpu_fault.models import RecoveryAction, WorkloadState
from gpu_fault.store import InMemoryStore, SqliteStore
from tests._builders import asgi_client
from tests.host_health._support import RecordingNotifier
from tests.store._health_signal_semantics import KEY, at, host_batch


@pytest.fixture(params=["memory", "sqlite"])
def context(request, tmp_path):
    if request.param == "memory":
        yield ApplicationContext(
            store=InMemoryStore(), notification_notifier=RecordingNotifier()
        )
    else:
        with closing(SqliteStore(str(tmp_path / "delivery.db"))) as store:
            yield ApplicationContext(
                store=store, notification_notifier=RecordingNotifier()
            )


def batch(mode, seconds, *, value=1):
    return host_batch(mode, seconds, value=value).model_copy(
        update={
            "runtime_profile_version": "simulated-v1",
            "workload_state": WorkloadState.IDLE,
        }
    )


def post(context, value):
    async def send():
        async with asgi_client(context) as client:
            return await client.post(
                HOST_TELEMETRY_PATH, json=value.model_dump(mode="json")
            )

    return asyncio.run(send())


def test_failed_new_meaning_delivery_retries_and_latches_only_after_commit(
    context, monkeypatch
):
    first = post(context, batch("LINK_INACTIVE", 0))
    assert first.status_code == 200, first.text
    assert context.store.get_health_signal_state(KEY).notified is True

    def fail_incident(_finding, **_kwargs):
        raise RuntimeError("semantic incident write failed")

    with monkeypatch.context() as failed:
        failed.setattr(context.orchestrator, "ingest_node_health", fail_incident)
        with pytest.raises(RuntimeError, match="semantic incident write failed"):
            post(context, batch("PCI_DEVICE_MISSING", 1))
    retried = post(context, batch("PCI_DEVICE_MISSING", 2))
    assert retried.status_code == 200, retried.text
    assert len(retried.json()["findings"]) == 1
    assert context.store.get_health_signal_state(KEY).notified is True
    unchanged = post(context, batch("PCI_DEVICE_MISSING", 3))
    assert unchanged.status_code == 200, unchanged.text
    assert unchanged.json()["findings"] == [], (
        "a successfully committed semantic delivery was not latched"
    )
    recovered = post(context, batch("PCI_DEVICE_MISSING", 4, value=0))
    assert recovered.status_code == 200, recovered.text
    assert context.store.get_health_signal_state(KEY).active is False
    rearmed = post(context, batch("PCI_DEVICE_MISSING", 5))
    assert rearmed.status_code == 200, rearmed.text
    assert len(rearmed.json()["findings"]) == 1


@pytest.mark.parametrize("return_to_a", [False, True])
def test_finish_of_an_old_delivery_does_not_latch_a_newer_semantic_episode(
    context, monkeypatch, return_to_a
):
    original = context.store.mark_health_signal_notified
    intercepted = False

    def delayed_ack(signal_key, *, notified_at, semantic_fingerprint=None):
        nonlocal intercepted
        if signal_key == KEY and not intercepted:
            intercepted = True
            assert (
                len(
                    context.node_health.evaluate_metrics(batch("PCI_DEVICE_MISSING", 1))
                )
                == 1
            )
            if return_to_a:
                assert (
                    len(context.node_health.evaluate_metrics(batch("LINK_INACTIVE", 2)))
                    == 1
                )
        original(
            signal_key,
            notified_at=notified_at,
            semantic_fingerprint=semantic_fingerprint,
        )

    monkeypatch.setattr(context.store, "mark_health_signal_notified", delayed_ack)
    response = post(context, batch("LINK_INACTIVE", 0))
    assert response.status_code == 200, response.text
    assert intercepted is True
    current = context.store.get_health_signal_state(KEY)
    assert current.notified is False, (
        "an old delivery acknowledged newer host semantics"
    )
    assert current.semantic_since == at(2 if return_to_a else 1)
    mode = "LINK_INACTIVE" if return_to_a else "PCI_DEVICE_MISSING"
    fresh = post(context, batch(mode, 3))
    assert fresh.status_code == 200, fresh.text
    assert len(fresh.json()["findings"]) == 1
    assert context.store.get_health_signal_state(KEY).notified is True


@pytest.mark.parametrize(
    ("changed_mode", "changed_action"),
    [
        ("PCI_DEVICE_MISSING", RecoveryAction.REBOOT_NODE),
        ("DRIVER_UNBOUND", RecoveryAction.REMEDIATE_EFA_DRIVER),
        ("EXCESS_DEVICE", RecoveryAction.RUN_DIAGNOSTICS),
    ],
)
def test_collector_delivers_changed_efa_semantics_without_waiting_for_a_clear(
    context, changed_mode, changed_action
):
    results = []
    clock = [at(0)]
    samples = []

    class InProcessSink:
        def post(self, path, payload):
            assert path == HOST_TELEMETRY_PATH
            response = post(context, HostTelemetryBatch.model_validate(payload))
            assert response.status_code == 200, response.text
            result = response.json()
            results.append(result)
            return result

    class SampleCollector(HostTelemetryCollector):
        # Sampling is synthetic; edge filtering, delivery, policy and ACK are real.
        CONTRIBUTORS = ("read_samples",)

        def read_samples(self, _observed_at):
            return samples

    collector = SampleCollector(
        InProcessSink(),
        CollectorContext(
            cluster_id="semantic-cluster",
            runtime_profile_version="simulated-v1",
            workload_state=WorkloadState.IDLE,
        ),
        node_id="semantic-node",
        now=lambda: clock[0],
        edge_filter_enabled=True,
        health_summary_seconds=86400,
    )
    delivered = 0
    active_since = None
    for seconds, mode, value, expected_action, should_deliver in (
        (0, "LINK_INACTIVE", 1, RecoveryAction.RUN_DIAGNOSTICS, True),
        (1, "LINK_INACTIVE", 1, RecoveryAction.RUN_DIAGNOSTICS, False),
        (2, changed_mode, 1, changed_action, True),
        (3, changed_mode, 2, changed_action, False),
        (4, "LINK_INACTIVE", 1, RecoveryAction.RUN_DIAGNOSTICS, True),
        (5, "LINK_INACTIVE", 0, None, True),
        (6, "LINK_INACTIVE", 1, RecoveryAction.RUN_DIAGNOSTICS, True),
    ):
        clock[0] = at(seconds)
        samples = host_batch(
            mode, seconds, value=value, labels={"note": f"sample {seconds}"}
        ).samples
        collected = collector.collect_once()
        assert collected.collection_errors == [], "synthetic sampling failed"
        delivered += int(should_deliver)
        assert len(results) == delivered, (
            f"collector did not preserve the delivery edge for {mode} at {seconds}"
        )
        if not should_deliver:
            assert collected.edge_filter_reasons == [], (
                "unchanged semantics must not post incidental label or value changes"
            )
            continue
        assert "health-summary" not in collected.edge_filter_reasons, (
            "a summary must not mask a missing semantic edge"
        )
        state = context.store.get_health_signal_state(KEY)
        assert state is not None, "the delivered sample must retain its stable key"
        if expected_action is None:
            assert results[-1]["findings"] == []
            assert state.active is state.notified is False
        else:
            assert [
                finding["recommended_action"] for finding in results[-1]["findings"]
            ] == [expected_action.value]
            assert state.notified is True, (
                "post-commit ACK did not latch the new meaning"
            )
            if seconds == 0:
                active_since = state.active_since
            elif seconds < 5:
                assert state.active_since == active_since, (
                    "a semantic change reset the continuous-active window"
                )
