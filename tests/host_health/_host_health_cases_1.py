from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from gpu_fault.app import ApplicationContext
from gpu_fault.app.ingest import telemetry as telemetry_ingest
from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
    NodeHealthPolicy,
    NodeLogBatch,
    NodeLogEntry,
)
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.store.shared.health_signals import finding_health_signal_key
from tests._builders import (
    asgi_client,
    build_store,
    copy_model,
    host_telemetry_batch,
    workflow_step_execution,
)
from tests.host_health._support import (
    NOW,
    RecordingNotifier,
    efa_telemetry,
    observe_running_attempt,
    telemetry,
    zero_traffic_with_liveness,
    zero_traffic_with_progress,
)

ADMIN_TOKEN = "a" * 32


def test_collector_batches_reject_blank_node_identity() -> None:
    # A collector that resolves an empty node id must be rejected at the
    # door: accepting it stores evidence and latest metrics under a
    # phantom node that no recovery path can ever act on.
    for field in ("cluster_id", "node_id"):
        with pytest.raises(ValidationError):
            HostTelemetryBatch(
                batch_id="blank-identity",
                **{**{"cluster_id": "cluster-a", "node_id": "node-a"}, field: ""},
                observed_at=NOW,
                samples=[HostMetricSample(name="load_average_1m", value=1.0)],
            )
        with pytest.raises(ValidationError):
            NodeLogBatch(
                batch_id="blank-identity",
                **{**{"cluster_id": "cluster-a", "node_id": "node-a"}, field: ""},
                collected_at=NOW,
                entries=[
                    NodeLogEntry(
                        entry_id="entry-a",
                        source="journal",
                        observed_at=NOW,
                        message="out of memory",
                    )
                ],
            )


def test_efa_traffic_spike_and_drop_are_state_transitions(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "100")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO", "2")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_DROP_RATIO", "0.5")
    store = build_store()
    policy = NodeHealthPolicy(store)
    observe_running_attempt(store, NOW)

    assert not policy.evaluate_metrics(efa_telemetry("baseline", 1_000, NOW)), (
        'expected policy.evaluate_metrics(efa_telemetry("baseline", 1_000, NOW)) to be falsy'
    )
    observe_running_attempt(store, NOW + timedelta(seconds=15))
    spike = policy.evaluate_metrics(
        efa_telemetry("spike", 3_000, NOW + timedelta(seconds=15))
    )
    observe_running_attempt(store, NOW + timedelta(seconds=30))
    drop = policy.evaluate_metrics(
        efa_telemetry("drop", 100, NOW + timedelta(seconds=30))
    )

    assert spike[0].policy_source == "SITE_EFA_TRAFFIC"
    assert "increased abruptly" in spike[0].reason
    assert "dropped abruptly" in drop[0].reason


def test_sustained_zero_efa_traffic_captures_hung_context(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "100")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_STARTUP_GRACE_SECONDS", "0")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_ZERO_WARNING_SECONDS", "15")
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_ZERO_HUNG_SECONDS", "30")
    store = build_store()
    policy = NodeHealthPolicy(store)

    findings = []
    for index, value in enumerate((1_000, 0, 0, 0)):
        observed_at = NOW + timedelta(seconds=index * 15)
        observe_running_attempt(store, observed_at, ("node-a", "node-b"))
        findings.extend(
            policy.evaluate_metrics(efa_telemetry(f"zero-{index}", value, observed_at))
        )

    assert [item.severity.value for item in findings] == ["warning", "critical"]
    hung = findings[-1]
    assert (
        hung.diagnostic_parameters["diagnostic_reason"] == "EFA_TRAFFIC_HUNG_SUSPECTED"
    )
    assert hung.diagnostic_parameters["capture_process_state"]
    assert hung.diagnostic_parameters["strace_sample_count"] == 3
    assert hung.diagnostic_parameters["strace_duration_seconds"] == 3
    assert hung.diagnostic_parameters["strace_sample_interval_seconds"] == 2
    assert hung.diagnostic_parameters["pyspy_sample_count"] == 3
    assert hung.diagnostic_parameters["pyspy_sample_interval_seconds"] == 2
    assert hung.diagnostic_parameters["pyspy_timeout_seconds"] == 10
    assert (
        hung.diagnostic_parameters["efa_zero_since"]
        == (NOW + timedelta(seconds=15)).isoformat()
    )
    assert hung.job_id == "job-a"
    assert hung.attempt_id == "attempt-a"
    assert hung.diagnostic_parameters["attempt_node_ids"] == ["node-a", "node-b"]
    assert hung.diagnostic_parameters["gpu_uuids_by_node"] == {
        "node-a": ["GPU-0"],
        "node-b": ["GPU-1"],
    }


def test_checkpoint_progress_defers_zero_traffic_escalation(monkeypatch) -> None:
    # Progress at t=20 and t=35 restarts the hang clock, so the fabric
    # being idle from t=15 to t=45 — a checkpoint write or a graph
    # compile — raises nothing. Once progress stops, the clock runs from
    # the last step (t=35) and the attempt still escalates.
    signals = zero_traffic_with_progress(
        monkeypatch, progress_at=(20, 35), samples=(0, 15, 30, 45, 60, 75)
    )

    assert signals == [
        (60, "EFA_TRAFFIC_ZERO_WARNING"),
        (75, "EFA_TRAFFIC_HUNG_SUSPECTED"),
    ]


@pytest.mark.parametrize("token", ["1", "yes", "on"])
def test_zero_traffic_progress_suppression_accepts_every_enabled_token(
    monkeypatch, token: str
) -> None:
    """A default-on switch spelled ``=1`` used to switch the suppression off."""

    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_ENABLED", token)

    signals = zero_traffic_with_progress(
        monkeypatch, progress_at=(20, 35), samples=(0, 15, 30, 45, 60, 75)
    )

    assert signals == [
        (60, "EFA_TRAFFIC_ZERO_WARNING"),
        (75, "EFA_TRAFFIC_HUNG_SUSPECTED"),
    ]


def test_zero_traffic_progress_suppression_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_ENABLED", "false")

    signals = zero_traffic_with_progress(
        monkeypatch, progress_at=(20, 35), samples=(0, 15, 30, 45)
    )

    assert signals == [
        (30, "EFA_TRAFFIC_ZERO_WARNING"),
        (45, "EFA_TRAFFIC_HUNG_SUSPECTED"),
    ]


def test_rank_liveness_defers_zero_traffic_escalation(monkeypatch) -> None:
    # No application instrumentation at all: the node itself measured
    # that a rank advanced until t=35, so the hang clock runs from
    # there instead of from the last packet at t=15.
    emitted = zero_traffic_with_liveness(
        monkeypatch, samples=(0, 15, 30, 45, 60, 75), progress_until=35
    )

    assert [(item[0], item[1]) for item in emitted] == [
        (60, "EFA_TRAFFIC_ZERO_WARNING"),
        (75, "EFA_TRAFFIC_HUNG_SUSPECTED"),
    ]
    assert emitted[-1][2]["status"] == "stalled"
    assert emitted[-1][2]["seconds_since_progress"] == 40


def test_stalled_ranks_still_escalate_to_hung(monkeypatch) -> None:
    # The node reports ranks that have not advanced since t=0, so the
    # probe confirms the hang instead of deferring it.
    emitted = zero_traffic_with_liveness(
        monkeypatch, samples=(0, 15, 30, 45), progress_until=0
    )

    assert [(item[0], item[1]) for item in emitted] == [
        (30, "EFA_TRAFFIC_ZERO_WARNING"),
        (45, "EFA_TRAFFIC_HUNG_SUSPECTED"),
    ]
    assert emitted[-1][2]["status"] == "stalled"


def test_missing_rank_liveness_does_not_suppress_escalation(monkeypatch) -> None:
    # A node that publishes no liveness samples must escalate exactly as
    # before: a broken probe cannot silence hang detection.
    emitted = zero_traffic_with_liveness(
        monkeypatch, samples=(0, 15, 30, 45), progress_until=None
    )

    assert [(item[0], item[1]) for item in emitted] == [
        (30, "EFA_TRAFFIC_ZERO_WARNING"),
        (45, "EFA_TRAFFIC_HUNG_SUSPECTED"),
    ]
    assert emitted[-1][2]["status"] == "unavailable"


def test_continuous_rank_progress_suppresses_within_budget(monkeypatch) -> None:
    # Every sample reports progress, so nothing escalates while the
    # suppression budget lasts.
    emitted = zero_traffic_with_liveness(
        monkeypatch, samples=(0, 15, 30, 45, 60, 75, 90), progress_until=10_000
    )

    assert emitted == []


def test_progress_suppression_is_bounded(monkeypatch) -> None:
    # Same input, but suppression is capped at 30 s past the first zero
    # sample: a signal that always looks like progress cannot silence
    # the detector forever.
    emitted = zero_traffic_with_liveness(
        monkeypatch,
        samples=(0, 15, 30, 45, 60, 75, 90),
        progress_until=10_000,
        max_suppression="30",
    )

    assert [(item[0], item[1]) for item in emitted] == [
        (60, "EFA_TRAFFIC_ZERO_WARNING"),
        (75, "EFA_TRAFFIC_HUNG_SUSPECTED"),
    ]


def test_progress_suppression_budget_must_be_positive(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_MAX_SECONDS", "0")

    with pytest.raises(ValueError, match="PROGRESS_SUPPRESSION_MAX_SECONDS"):
        NodeHealthPolicy(build_store())


def test_rank_liveness_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_RANK_LIVENESS_ENABLED", "false")

    emitted = zero_traffic_with_liveness(
        monkeypatch, samples=(0, 15, 30, 45), progress_until=10_000
    )

    assert [(item[0], item[1]) for item in emitted] == [
        (30, "EFA_TRAFFIC_ZERO_WARNING"),
        (45, "EFA_TRAFFIC_HUNG_SUSPECTED"),
    ]
    assert emitted[-1][2]["status"] == "disabled"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GPU_FAULT_HUNG_STRACE_SAMPLE_COUNT", "1"),
        ("GPU_FAULT_HUNG_STRACE_SAMPLE_COUNT", "6"),
        ("GPU_FAULT_HUNG_STRACE_SAMPLE_DURATION_SECONDS", "0"),
        ("GPU_FAULT_HUNG_STRACE_SAMPLE_DURATION_SECONDS", "31"),
        ("GPU_FAULT_HUNG_STRACE_SAMPLE_INTERVAL_SECONDS", "-1"),
        ("GPU_FAULT_HUNG_STRACE_SAMPLE_INTERVAL_SECONDS", "31"),
        ("GPU_FAULT_HUNG_PYSPY_SAMPLE_COUNT", "2"),
        ("GPU_FAULT_HUNG_PYSPY_SAMPLE_COUNT", "6"),
        ("GPU_FAULT_HUNG_PYSPY_SAMPLE_INTERVAL_SECONDS", "-1"),
        ("GPU_FAULT_HUNG_PYSPY_SAMPLE_INTERVAL_SECONDS", "31"),
        ("GPU_FAULT_HUNG_PYSPY_TIMEOUT_SECONDS", "0"),
        ("GPU_FAULT_HUNG_PYSPY_TIMEOUT_SECONDS", "61"),
    ],
)
def test_hung_strace_configuration_rejects_unsafe_bounds(
    monkeypatch, name, value
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        NodeHealthPolicy(build_store())


def test_metric_policy_emits_only_active_transition() -> None:
    store = build_store()
    first = NodeHealthPolicy(store)
    second = NodeHealthPolicy(store)

    initial = first.evaluate_metrics(
        telemetry("batch-1", "filesystem_used_percent", 99)
    )
    # The claim no longer latches ``notified``; the deliverer does, after commit
    # (F-D11 P0-38B), so the test plays the delivery half here.
    store.mark_health_signal_notified(
        finding_health_signal_key(initial[0]), notified_at=NOW
    )
    repeated = second.evaluate_metrics(
        telemetry("batch-2", "filesystem_used_percent", 99, NOW + timedelta(seconds=15))
    )
    first.evaluate_metrics(
        telemetry("batch-3", "filesystem_used_percent", 50, NOW + timedelta(seconds=30))
    )
    recurring = second.evaluate_metrics(
        telemetry("batch-4", "filesystem_used_percent", 99, NOW + timedelta(seconds=45))
    )

    assert initial[0].recommended_action.value == "QUARANTINE"
    assert not repeated, "expected repeated to be falsy"
    assert recurring[0].event_id.startswith("batch-4"), (
        'expected recurring[0].event_id.startswith("batch-4") to be truthy'
    )


def test_host_validation_metrics_are_persisted_as_latest() -> None:
    store = build_store()
    policy = NodeHealthPolicy(store)
    batch = host_telemetry_batch(
        "validation-latest",
        NOW,
        [
            HostMetricSample(name="load1_per_cpu", value=0.1),
            HostMetricSample(name="memory_used_percent", value=20, unit="percent"),
            HostMetricSample(
                name="filesystem_used_percent", value=30, unit="percent", device="/"
            ),
        ],
    )

    policy.evaluate_metrics(batch)

    latest = store.list_telemetry_metrics_latest("cluster-a", "node-a")
    assert {item.name for item in latest} == {
        "load1_per_cpu",
        "memory_used_percent",
        "filesystem_used_percent",
    }


@pytest.mark.parametrize(
    ("metric_name", "value", "normal_value", "device"),
    [
        ("cpu_usage_percent", 1, 50, None),
        ("host_gpu_utilization_percent", 0, 90, "GPU-a"),
        ("memory_available_percent", 2, 50, None),
        ("page_cache_percent", 95, 20, None),
        ("local_filesystem_used_percent", 92, 40, "/"),
    ],
)
def test_sustained_host_resource_risk_emits_once_and_rearms(
    monkeypatch, metric_name, value, normal_value, device
) -> None:
    monkeypatch.setenv("GPU_FAULT_LOW_UTILIZATION_DURATION_SECONDS", "15")
    monkeypatch.setenv("GPU_FAULT_MEMORY_PRESSURE_DURATION_SECONDS", "15")
    monkeypatch.setenv("GPU_FAULT_PAGE_CACHE_DURATION_SECONDS", "15")
    monkeypatch.setenv("GPU_FAULT_LOCAL_FILESYSTEM_DURATION_SECONDS", "15")
    policy = NodeHealthPolicy(build_store())

    def batch(batch_id, metric_value, observed_at):
        return host_telemetry_batch(
            batch_id,
            observed_at,
            [
                HostMetricSample(
                    name=metric_name, value=metric_value, unit="percent", device=device
                )
            ],
            workload_state="ACTIVE",
            affected_workload_ids=["training/job/job-a"],
        )

    assert not policy.evaluate_metrics(batch("initial", value, NOW)), (
        'expected policy.evaluate_metrics(batch("initial", value, NOW)) to be falsy'
    )
    finding = policy.evaluate_metrics(
        batch("sustained", value, NOW + timedelta(seconds=15))
    )
    assert len(finding) == 1
    assert finding[0].policy_source == "SITE_HOST_RESOURCE_HEALTH"
    assert finding[0].recommended_action is RecoveryAction.RUN_DIAGNOSTICS
    policy.store.mark_health_signal_notified(
        finding_health_signal_key(finding[0]), notified_at=NOW + timedelta(seconds=15)
    )
    assert not policy.evaluate_metrics(
        batch("duplicate", value, NOW + timedelta(seconds=30))
    ), (
        'expected policy.evaluate_metrics( batch("duplicate", value, NOW + timedelta(seconds=30)) ) to be falsy'
    )
    assert not policy.evaluate_metrics(
        batch("normal", normal_value, NOW + timedelta(seconds=45))
    ), (
        'expected policy.evaluate_metrics( batch("normal", normal_value, NOW + timedelta(seconds=45)) ) to be falsy'
    )
    assert not policy.evaluate_metrics(
        batch("rearmed", value, NOW + timedelta(seconds=60))
    ), (
        'expected policy.evaluate_metrics( batch("rearmed", value, NOW + timedelta(seconds=60)) ) to be falsy'
    )
    assert policy.evaluate_metrics(
        batch("repeated", value, NOW + timedelta(seconds=75))
    ), (
        'expected policy.evaluate_metrics( batch("repeated", value, NOW + timedelta(seconds=75)) ) to be truthy'
    )


def test_host_policy_persists_only_metrics_consumed_by_validation() -> None:
    class RecordingStore(InMemoryStore):
        def __init__(self):
            super().__init__()
            self.persisted = []

        def observe_telemetry_metrics(self, items):
            self.persisted.extend(item.name for item in items)
            return super().observe_telemetry_metrics(items)

    store = RecordingStore()
    policy = NodeHealthPolicy(store)
    batch = host_telemetry_batch(
        "persist-filter",
        NOW,
        [
            HostMetricSample(name="cpu_usage_percent", value=50),
            HostMetricSample(name="tcp_retransmits_delta", value=0),
            HostMetricSample(name="gpu_inventory_expected_count", value=8),
            HostMetricSample(name="gpu_inventory_active_count", value=8),
            HostMetricSample(
                name="nvswitch_port_topology",
                value=1,
                device="switch-0/1",
                labels={"trusted": "true"},
            ),
            HostMetricSample(name="network_link_up", value=1, device="eth0"),
            HostMetricSample(name="rdma_link_down", value=0, device="rdmap0s6"),
            HostMetricSample(name="rdma_errors_delta", value=0, device="rdmap0s6"),
            HostMetricSample(name="network_errors_delta", value=0, device="eth0"),
        ],
    )

    policy.evaluate_metrics(batch)

    assert store.persisted == [
        "gpu_inventory_expected_count",
        "gpu_inventory_active_count",
        "nvswitch_port_topology",
        "network_link_up",
        "rdma_link_down",
        "rdma_errors_delta",
        "network_errors_delta",
    ]


def test_low_utilization_does_not_alert_for_idle_node(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_LOW_UTILIZATION_DURATION_SECONDS", "15")
    policy = NodeHealthPolicy(build_store())
    first = telemetry("idle-low-1", "cpu_usage_percent", 0)
    second = telemetry(
        "idle-low-2", "cpu_usage_percent", 0, NOW + timedelta(seconds=30)
    )

    assert not policy.evaluate_metrics(first), (
        "expected policy.evaluate_metrics(first) to be falsy"
    )
    assert not policy.evaluate_metrics(second), (
        "expected policy.evaluate_metrics(second) to be falsy"
    )


def test_sustained_host_resource_state_survives_sqlite_restart(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("GPU_FAULT_MEMORY_PRESSURE_DURATION_SECONDS", "15")
    path = tmp_path / "host-health.db"
    first_store = SqliteStore(str(path))
    first = NodeHealthPolicy(first_store)
    assert not first.evaluate_metrics(
        telemetry("memory-low-before-restart", "memory_available_percent", 2)
    ), (
        'expected first.evaluate_metrics( telemetry( "memory-low-before-restart", "memory_available_percent", 2, ) ) to be falsy'
    )
    first_store.close()

    second_store = SqliteStore(str(path))
    try:
        second = NodeHealthPolicy(second_store)
        finding = second.evaluate_metrics(
            telemetry(
                "memory-low-after-restart",
                "memory_available_percent",
                2,
                NOW + timedelta(seconds=15),
            )
        )
        assert len(finding) == 1
        assert finding[0].diagnostic_parameters["signal"] == ("LOW_MEMORY_AVAILABLE")
    finally:
        second_store.close()


class _IngestionClock:
    """Drive the control plane's receive time for host telemetry (F-M2).

    Sustained-signal windows are measured on the time the control plane
    accepted each batch, not on the node's ``observed_at``; a test that
    wants "fifteen seconds later" has to move this clock, not just the
    payload timestamp.
    """

    def __init__(self, monkeypatch, start: datetime) -> None:
        self.now = start
        clock = self

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return clock.now if tz is not None else clock.now.replace(tzinfo=None)

        monkeypatch.setattr(telemetry_ingest, "datetime", _Frozen)


def test_sustained_host_resource_risk_sends_fixed_email(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_MEMORY_PRESSURE_DURATION_SECONDS", "15")
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    clock = _IngestionClock(monkeypatch, NOW)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            first = await client.post(
                "/v1/collector-events/host-telemetry",
                json=telemetry(
                    "memory-low-initial", "memory_available_percent", 2
                ).model_dump(mode="json"),
            )
            clock.now = NOW + timedelta(seconds=15)
            sustained = await client.post(
                "/v1/collector-events/host-telemetry",
                json=telemetry(
                    "memory-low-sustained",
                    "memory_available_percent",
                    2,
                    NOW + timedelta(seconds=15),
                ).model_dump(mode="json"),
            )

        assert first.json()["notification_ids"] == []
        assert len(sustained.json()["notification_ids"]) == 1

    asyncio.run(scenario())
    assert len(notifier.notifications) == 1
    body = notifier.notifications[0].body_text
    assert "Signal：LOW_MEMORY_AVAILABLE" in body
    assert "Metric：memory_available_percent" in body
    assert "持续时间阈值：15.0 秒" in body
    assert "host-resource-event-zh-v1" in body


def test_node_log_uses_highest_severity_matching_rule() -> None:
    policy = NodeHealthPolicy(build_store())
    findings = policy.evaluate_logs(
        NodeLogBatch(
            batch_id="multi-rule-log",
            cluster_id="cluster-a",
            node_id="node-a",
            collected_at=NOW,
            entries=[
                NodeLogEntry(
                    entry_id="multi-rule",
                    source="journal",
                    observed_at=NOW,
                    message=(
                        "out of memory while EFA RDMA link down reported fatal error"
                    ),
                )
            ],
        )
    )

    assert len(findings) == 1
    assert findings[0].category.value == "RDMA"
    assert findings[0].severity.value == "critical"
    assert findings[0].recommended_action is RecoveryAction.QUARANTINE


def test_node_log_does_not_treat_efa_plugin_name_as_rdma_fault() -> None:
    policy = NodeHealthPolicy(build_store())
    findings = policy.evaluate_logs(
        NodeLogBatch(
            batch_id="efa-plugin-name",
            cluster_id="cluster-a",
            node_id="node-a",
            collected_at=NOW,
            entries=[
                NodeLogEntry(
                    entry_id="operator-error",
                    source="training-log",
                    observed_at=NOW,
                    message=(
                        "Reconcile PyTorchJob gpu-fault-efa-plugin "
                        "error: object has been modified"
                    ),
                )
            ],
        )
    )

    assert findings == []


def _mce_log_batch(source: str) -> NodeLogBatch:
    return NodeLogBatch(
        batch_id=f"mce-from-{source}",
        cluster_id="cluster-a",
        node_id="node-a",
        collected_at=NOW,
        entries=[
            NodeLogEntry(
                entry_id=f"entry-{source}",
                source=source,
                observed_at=NOW,
                message="mce: Hardware Error: Machine check uncorrectable",
            )
        ],
    )


def test_fatal_log_rule_from_training_log_produces_no_trusted_marker() -> None:
    """H-5: a workload-writable origin cannot forge a hardware-fatal fault.

    The MCE rule drives QUARANTINE. A training log is a file the job writes,
    so a job could otherwise print the kernel's machine-check text and have it
    isolate the node. The trusted-source gate must drop the line entirely.
    """

    policy = NodeHealthPolicy(build_store())

    findings = policy.evaluate_logs(_mce_log_batch("training-log"))

    assert findings == []


def test_fatal_log_rule_from_dmesg_produces_a_trusted_quarantine_marker() -> None:
    """H-5 green half: the same text from the kernel ring buffer still fires.

    Gating on origin must not blind the system to a real hardware fault, and
    the resulting marker is trusted and stamped with its tenant (H-14).
    """

    policy = NodeHealthPolicy(build_store())

    findings = policy.evaluate_logs(_mce_log_batch("dmesg"))

    assert len(findings) == 1
    finding = findings[0]
    assert finding.category.value == "MCE"
    assert finding.recommended_action is RecoveryAction.QUARANTINE
    marker = finding.marker()
    assert marker.trusted is True
    assert marker.recommended_action is RecoveryAction.QUARANTINE
    assert marker.cluster_id == "cluster-a"


def test_log_entry_without_a_source_is_treated_as_untrusted() -> None:
    """A line whose origin the collector could not tag never matches.

    ``source`` is a required field on ``NodeLogEntry`` today, but the matcher
    reads it defensively so a future producer that omits it cannot fall
    through to a trusted marker.
    """

    from types import SimpleNamespace

    policy = NodeHealthPolicy(build_store())
    entry = SimpleNamespace(message="mce: Hardware Error: Machine check uncorrectable")
    batch = SimpleNamespace(
        entries=[entry],
        cluster_id="cluster-a",
        node_id="node-a",
        runtime_profile_version=None,
        workload_state=None,
        affected_workload_ids=[],
    )

    assert policy.evaluate_logs(batch) == []


def test_gpu_low_utilization_email_is_aggregated_per_node(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_LOW_UTILIZATION_DURATION_SECONDS", "15")
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    clock = _IngestionClock(monkeypatch, NOW)

    def batch(batch_id: str, observed_at: datetime):
        return host_telemetry_batch(
            batch_id,
            observed_at,
            [
                HostMetricSample(
                    name="host_gpu_utilization_percent",
                    value=0,
                    unit="percent",
                    device="GPU-a",
                ),
                HostMetricSample(
                    name="host_gpu_utilization_percent",
                    value=0,
                    unit="percent",
                    device="GPU-b",
                ),
            ],
            workload_state="ACTIVE",
            affected_workload_ids=["training/pytorchjob/train-a"],
        )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            initial = await client.post(
                "/v1/collector-events/host-telemetry",
                json=batch("gpu-low-initial", NOW).model_dump(mode="json"),
            )
            clock.now = NOW + timedelta(seconds=15)
            sustained = await client.post(
                "/v1/collector-events/host-telemetry",
                json=batch("gpu-low-sustained", NOW + timedelta(seconds=15)).model_dump(
                    mode="json"
                ),
            )
        assert initial.json()["notification_ids"] == []
        assert len(sustained.json()["findings"]) == 2
        assert len(sustained.json()["notification_ids"]) == 1

    asyncio.run(scenario())
    assert len(notifier.notifications) == 1
    assert "GPU-a, GPU-b" in notifier.notifications[0].body_text


def test_rdma_link_down_requests_fabric_diagnostics() -> None:
    finding = NodeHealthPolicy(build_store()).evaluate_metrics(
        telemetry("rdma-down", "rdma_link_down", 1)
    )[0]

    assert finding.category.value == "RDMA"
    assert finding.recommended_action.value == "RUN_DIAGNOSTICS"


def test_rdma_link_error_sends_one_fixed_email() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            first = await client.post(
                "/v1/collector-events/host-telemetry",
                json=telemetry("rdma-link-email", "rdma_link_down", 1).model_dump(
                    mode="json"
                ),
            )
            repeated = await client.post(
                "/v1/collector-events/host-telemetry",
                json=telemetry(
                    "rdma-link-email-repeat",
                    "rdma_link_down",
                    1,
                    NOW + timedelta(seconds=15),
                ).model_dump(mode="json"),
            )

        assert len(first.json()["notification_ids"]) == 1
        assert repeated.json()["notification_ids"] == []

    asyncio.run(scenario())
    assert len(notifier.notifications) == 1
    notification = notifier.notifications[0]
    assert "事件类型：RDMA_LINK_OR_ERROR" in notification.body_text
    assert "Metric：rdma_link_down" in notification.body_text
    assert "FREEZE_EVIDENCE -> VALIDATE_FABRIC" in (notification.body_text)
    assert "efa-rdma-event-zh-v1" in notification.body_text


@pytest.mark.parametrize(
    ("metric_name", "resource", "expected", "observed"),
    [
        ("gpu_inventory_mismatch", "GPU", "8", "7"),
        ("efa_inventory_mismatch", "EFA", "16", "15"),
    ],
)
def test_inventory_card_loss_reboots_then_escalates_to_replacement(
    metric_name, resource, expected, observed
) -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    observe_running_attempt(context.store, NOW)
    batch = host_telemetry_batch(
        f"{resource.lower()}-inventory-loss",
        NOW,
        [
            HostMetricSample(
                name=metric_name,
                value=1,
                labels={
                    "resource": resource,
                    "expected_count": expected,
                    "observed_count": observed,
                    "discovered_count": observed,
                    "missing_count": "1",
                    "excess_count": "0",
                    "consecutive_mismatch_samples": "2",
                    "required_consecutive_samples": "2",
                    "node_instance_type": "ml.p5en.48xlarge",
                    "expected_gpu_count": "8",
                    "expected_efa_device_count": "16",
                },
            )
        ],
        workload_state="ACTIVE",
        affected_workload_ids=["training/job/job-a"],
        runtime_profile_version="simulated-v1",
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/host-telemetry",
                json=batch.model_dump(mode="json"),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["findings"][0]["recommended_action"] == ("REBOOT_NODE")
        workflow = context.store.get_workflow(body["workflow_request_ids"][0])
        operations = [step.operation.value for step in workflow.official_steps]
        assert "STOP_WORKLOADS" in operations
        assert "QUARANTINE" in operations
        assert "RESTART_NODE" in operations
        assert "REPLACE_NODE" not in operations
        assert "RESTART_WORKLOAD" in operations
        validation_operation = (
            WorkflowOperation.VALIDATE_GPU
            if resource == "GPU"
            else WorkflowOperation.VALIDATE_FABRIC
        )
        validation_index = next(
            index
            for index, step in enumerate(workflow.official_steps)
            if step.operation is validation_operation
        )
        requirement = workflow.official_steps[validation_index].parameters[
            "inventory_requirements_by_node"
        ]["node-a"]
        assert requirement["metrics"] == {
            "gpu_inventory_active_count": 8,
            "efa_inventory_active_count": 16,
        }
        for operation in {
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_FABRIC,
        }:
            step = next(
                item for item in workflow.official_steps if item.operation is operation
            )
            assert (
                step.parameters["inventory_requirements_by_node"]["node-a"]["metrics"]
                == requirement["metrics"]
            )

        reboot_index = next(
            index
            for index, step in enumerate(workflow.official_steps)
            if step.operation is WorkflowOperation.RESTART_NODE
        )
        failed = copy_model(
            workflow,
            status=WorkflowStatus.FAILED,
            # The reboot ran and completed before the validation failed; the
            # classifier reads that from the record, not from list position.
            completed_step_indexes=[reboot_index],
            completed_operations=[WorkflowOperation.RESTART_NODE],
            step_executions=[
                workflow_step_execution(reboot_index, WorkflowOperation.RESTART_NODE),
                workflow_step_execution(
                    validation_index,
                    validation_operation,
                    WorkflowStepStatus.FAILED,
                    error="post-reboot inventory mismatch",
                    details={
                        "failed_nodes": ["node-a"],
                        "node_failures": {"node-a": ["post-reboot inventory mismatch"]},
                    },
                ),
            ],
        )
        context.store.save_workflow(failed)
        escalation = context.orchestrator.escalate_failed_hardware_remediation(failed)
        assert escalation is not None
        replacement_incident, replacement = escalation
        assert replacement_incident.effective_action is RecoveryAction.REPLACE_NODE
        replacement_step = next(
            step
            for step in replacement.official_steps
            if step.operation is WorkflowOperation.REPLACE_NODE
        )
        assert (
            replacement_step.parameters["replacement_strategy"]
            == "HEALTHY_WARM_SPARE_ONLY"
        )
        replacement_validation = next(
            step
            for step in replacement.official_steps
            if step.operation is validation_operation
        )
        assert (
            replacement_validation.parameters["inventory_requirements_by_node"][
                "node-a"
            ]["metrics"]
            == requirement["metrics"]
        )

    asyncio.run(scenario())
    assert len(notifier.notifications) == 1
    email = notifier.notifications[0]
    assert f"硬件类型：{resource}" in email.body_text
    assert f"期望数量：{expected}" in email.body_text
    assert f"当前 ACTIVE 数量：{observed}" in email.body_text
    assert "EC2/HyperPod 实例类型：ml.p5en.48xlarge" in email.body_text
    assert "策略动作：REBOOT_NODE" in email.body_text
    assert "hardware-inventory-mismatch-zh-v1" in email.body_text


@pytest.mark.parametrize(
    ("failure_mode", "action"),
    [
        ("PCI_DEVICE_MISSING", RecoveryAction.REBOOT_NODE),
        ("DRIVER_UNBOUND", RecoveryAction.REMEDIATE_EFA_DRIVER),
        ("LINK_INACTIVE", RecoveryAction.RUN_DIAGNOSTICS),
    ],
)
def test_efa_inventory_failure_mode_selects_recovery_action(
    failure_mode, action
) -> None:
    policy = NodeHealthPolicy(build_store())
    batch = host_telemetry_batch(
        f"efa-{failure_mode.lower()}",
        NOW,
        [
            HostMetricSample(
                name="efa_inventory_mismatch",
                value=1,
                labels={
                    "failure_mode": failure_mode,
                    "expected_count": "16",
                    "observed_count": "15",
                    "discovered_count": "15"
                    if failure_mode == "PCI_DEVICE_MISSING"
                    else "16",
                    "driver_bound_count": "15"
                    if failure_mode == "DRIVER_UNBOUND"
                    else "16",
                },
            )
        ],
        runtime_profile_version="simulated-v1",
    )

    finding = policy.evaluate_metrics(batch)[0]

    assert finding.recommended_action is action
    assert finding.diagnostic_parameters["failure_mode"] == failure_mode
