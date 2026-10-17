"""A single failed DCGM scrape does not open a remediation incident (batch 126).

Live 2026-10-17 (gpu-eks-cluster, plain EKS): a release re-rendered the
dcgm-exporter DaemonSet and restarted the exporter Pod on each node. One scrape
of the node collector fell inside the restart and failed; the next, 15 s later,
was clean. The failed scrape became a WARNING ``dcgm_field_completeness``
finding asking for RUN_DIAGNOSTICS, the NODE_HEALTH incident opened at once,
its VALIDATE_GPU ran 3 s before the clean batch cleared the finding and
refused, and the DRAIN successor quarantined a healthy node -- whose WAITING
diagnostic bundle then failed the release's wave-safety probe.

The finding is still booked at once, but with NO_ACTION; it asks for
RUN_DIAGNOSTICS, and so opens its one incident, only once the node's streak of
error batches has lasted ``GPU_FAULT_DCGM_COLLECTION_ERROR_SUSTAIN_SECONDS``.
"""

from __future__ import annotations

import asyncio
import subprocess
from datetime import timedelta

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.collectors.gpu.dcgm import DcgmMetricsCollector
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricsService,
    GpuMetricsThresholds,
)
from gpu_fault.models import RecoveryAction, WorkflowOperation
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.telemetry import CollectorKind
from tests._builders import asgi_client
from tests.metrics._collection_error_streak import (
    CLUSTER,
    COMPLETENESS_KEY,
    DEAD_EXPORTER_OFFSETS,
    NODE,
    NOW,
    WINDOW_SECONDS,
    assert_streak_survives_the_store,
    clean_batch,
    erroring_batch,
)

DIAGNOSTIC_PLAN = [
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
    WorkflowOperation.VALIDATE_GPU,
]


def _post(context: ApplicationContext, batch: GpuMetricBatch) -> dict:
    async def run():
        async with asgi_client(context) as client:
            return await client.post(
                GPU_METRICS_PATH, json=batch.model_dump(mode="json")
            )

    response = asyncio.run(run())
    assert response.status_code == 200, response.text
    return response.json()


def _incident_count(context: ApplicationContext) -> int:
    return sum(context.store.incident_state_counts().values())


def _assert_nothing_opened(context: ApplicationContext, when: str) -> None:
    assert _incident_count(context) == 0, f"an incident opened {when}"
    assert context.store.list_workflows(limit=10) == [], f"a workflow ran {when}"
    assert context.store.list_markers() == [], f"a node marker was set {when}"
    assert context.store.list_remote_commands() == [], f"a command was sent {when}"


def _completeness(context: ApplicationContext):
    state = context.store.get_gpu_finding_state(COMPLETENESS_KEY)
    return None if state is None else state.finding


def test_one_failed_scrape_then_a_clean_one_opens_nothing() -> None:
    """The live sequence: one failed scrape inside the exporter roll, the next
    one clean 15 s later. No incident, no workflow, so nothing to quarantine."""

    context = ApplicationContext(store=InMemoryStore())

    body = _post(context, erroring_batch("dcgm-node-a-1", NOW))

    assert body["new_findings"] == [], "the failed scrape alone asks for nothing"
    (booked,) = body["findings"]
    assert (booked["canonical_name"], booked["severity"]) == (
        "dcgm_field_completeness",
        "WARNING",
    ), "the failed scrape is still booked"
    finding = _completeness(context)
    assert finding is not None, "readers keep seeing the telemetry gap"
    assert finding.automatic_action == RecoveryAction.NO_ACTION.value
    assert finding.policy_source == "SITE_DCGM_FIELD_COMPLETENESS"
    assert "no action before the 180s sustain window" in finding.reason, finding.reason
    (status,) = [
        item
        for item in context.store.list_collector_statuses(CLUSTER, NODE)
        if item.collector is CollectorKind.GPU_METRICS
    ]
    assert status.last_error_at == NOW, "the collector status records the error"
    _assert_nothing_opened(context, "for one failed scrape")

    _post(context, clean_batch("dcgm-node-a-2", NOW + timedelta(seconds=15)))

    assert _completeness(context) is None, "the clean scrape clears the finding"
    _assert_nothing_opened(context, "after the clean scrape")


def test_a_sustained_streak_opens_exactly_one_diagnostic_incident() -> None:
    context = ApplicationContext(store=InMemoryStore())
    batches = [
        erroring_batch(f"dcgm-{offset}", NOW + timedelta(seconds=offset))
        for offset in (*DEAD_EXPORTER_OFFSETS, 300)
    ]

    for batch in batches:
        offset = int((batch.observed_at - NOW).total_seconds())
        body = _post(context, batch)
        if offset < WINDOW_SECONDS:
            assert body["new_findings"] == [], f"+{offset}s is inside the window"
            _assert_nothing_opened(context, f"at +{offset}s")
        elif offset == WINDOW_SECONDS:
            (escalated,) = body["new_findings"]
            assert escalated["automatic_action"] == "RUN_DIAGNOSTICS", escalated
        else:
            assert body["new_findings"] == [], f"+{offset}s activated again"
        assert _incident_count(context) == int(offset >= WINDOW_SECONDS), (
            f"incident count at +{offset}s"
        )

    incident = context.store.get_incident_by_event("gpu-dcgm-0-dcgm-fields")
    assert incident is not None, "the incident is named after the streak's first batch"
    assert incident.policy_source == "SITE_DCGM_FIELD_COMPLETENESS"
    assert incident.effective_action is RecoveryAction.RUN_DIAGNOSTICS
    (workflow,) = context.store.list_workflows(limit=10)
    assert workflow.request_id == incident.workflow_request_id
    assert [step.operation for step in workflow.official_steps] == DIAGNOSTIC_PLAN

    escalation = next(
        batch
        for batch in batches
        if batch.observed_at == NOW + timedelta(seconds=WINDOW_SECONDS)
    )
    replay = _post(context, escalation)
    assert replay["duplicate"] is True, "the same batch id is recognised"
    assert [item["finding_id"] for item in replay["new_findings"]] == [
        "dcgm-0-dcgm-fields"
    ]
    assert _post(context, batches[0])["duplicate"] is True
    assert _incident_count(context) == 1, "a replay reaches the same incident"
    assert len(context.store.list_workflows(limit=10)) == 1, "and the same workflow"


def test_a_clean_batch_breaks_the_streak_and_the_next_error_starts_over() -> None:
    context = ApplicationContext(store=InMemoryStore())
    for offset in (0, 15, 30, 60, 120):
        _post(
            context, erroring_batch(f"dcgm-a{offset}", NOW + timedelta(seconds=offset))
        )
    _post(context, clean_batch("dcgm-clean", NOW + timedelta(seconds=135)))
    assert _completeness(context) is None, "the clean batch ended the streak"

    restart = NOW + timedelta(seconds=150)
    for offset in (0, 15, 30, 60, 120, 165):
        _post(
            context,
            erroring_batch(f"dcgm-b{offset}", restart + timedelta(seconds=offset)),
        )
        finding = _completeness(context)
        assert finding is not None and finding.observed_at == restart, (
            "the second streak is measured from its own first error"
        )
    _assert_nothing_opened(context, "for 315 s of errors broken at 135 s")

    _post(context, erroring_batch("dcgm-b180", restart + timedelta(seconds=180)))

    assert _incident_count(context) == 1, "the second streak reached its window"
    assert context.store.get_incident_by_event("gpu-dcgm-b0-dcgm-fields") is not None


def test_late_batches_neither_extend_nor_end_a_streak() -> None:
    context = ApplicationContext(store=InMemoryStore())
    _post(context, erroring_batch("dcgm-1", NOW))
    _post(context, erroring_batch("dcgm-2", NOW + timedelta(seconds=15)))
    _post(context, clean_batch("dcgm-3", NOW + timedelta(seconds=30)))

    _post(context, erroring_batch("dcgm-late-error", NOW + timedelta(seconds=20)))

    assert _completeness(context) is None, "a late error must not reopen the streak"

    _post(context, erroring_batch("dcgm-4", NOW + timedelta(seconds=45)))
    for offset in (60, 75):
        _post(
            context, erroring_batch(f"dcgm-{offset}", NOW + timedelta(seconds=offset))
        )
    _post(context, clean_batch("dcgm-late-clean", NOW + timedelta(seconds=50)))

    finding = _completeness(context)
    assert finding is not None, "a late clean batch must not end the streak"
    assert finding.observed_at == NOW + timedelta(seconds=45), finding.observed_at
    state = context.store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert state is not None and state.observed_at == NOW + timedelta(seconds=75)
    _assert_nothing_opened(context, "for late batches")


def test_the_real_collector_backoff_opens_one_incident_at_the_sixth_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A dead exporter, as the DCGM collector itself reports it, into the API."""

    boot_id = tmp_path / "boot_id"
    boot_id.write_text("boot-a\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(boot_id))
    clock = [NOW]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += timedelta(seconds=seconds)
        if len(sleeps) > len(DEAD_EXPORTER_OFFSETS):
            raise StopIteration

    def exporter_down(*_args, **_kwargs):
        raise OSError("connection refused")

    def no_nvidia_smi(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="absent")

    class Sink:
        def __init__(self) -> None:
            self.batches: list[GpuMetricBatch] = []

        def post(self, path: str, payload: dict) -> dict:
            if path == GPU_METRICS_PATH:
                self.batches.append(GpuMetricBatch.model_validate(payload))
            return {"accepted": True}

    monkeypatch.setattr("gpu_fault.collectors.gpu.dcgm.time.sleep", sleep)
    monkeypatch.setattr("gpu_fault.collectors.gpu.dcgm.urlopen", exporter_down)
    sink = Sink()
    collector = DcgmMetricsCollector(
        sink,
        CollectorContext(
            cluster_id=CLUSTER,
            runtime_profile_version="simulated-v1",
            product="H100",
            driver_branch=575,
            cuda_version="12.9",
        ),
        node_id=NODE,
        now=lambda: clock[0],
        runner=no_nvidia_smi,
        inventory_interval_seconds=3600,
        startup_grace_seconds=0,
    )
    with pytest.raises(StopIteration):
        collector.run()

    first = sink.batches[0].observed_at
    offsets = [int((item.observed_at - first).total_seconds()) for item in sink.batches]
    assert offsets == list(DEAD_EXPORTER_OFFSETS), f"collector backoff moved: {offsets}"
    context = ApplicationContext(store=InMemoryStore())
    opened = []
    for batch in sink.batches:
        _post(context, batch)
        opened.append(_incident_count(context))
    assert opened == [0, 0, 0, 0, 0, 1, 1], (
        f"one incident, at the sixth failed scrape (180 s): {opened}"
    )


def test_the_window_is_a_validated_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GPU_FAULT_DCGM_COLLECTION_ERROR_SUSTAIN_SECONDS", raising=False)
    assert GpuMetricsThresholds().collection_error_sustain_seconds == 180
    assert (
        GpuMetricsThresholds.from_environment().collection_error_sustain_seconds == 180
    )

    monkeypatch.setenv("GPU_FAULT_DCGM_COLLECTION_ERROR_SUSTAIN_SECONDS", "300")
    assert (
        GpuMetricsThresholds.from_environment().collection_error_sustain_seconds == 300
    )


@pytest.mark.parametrize("raw", ["0", "14", "901", "-15", "1.5", "three minutes"])
def test_the_window_rejects_values_it_cannot_honour(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("GPU_FAULT_DCGM_COLLECTION_ERROR_SUSTAIN_SECONDS", raw)
    with pytest.raises(ValueError):
        GpuMetricsThresholds.from_environment()


def test_a_configured_window_moves_the_escalation() -> None:
    service = GpuMetricsService(
        GpuMetricsThresholds(collection_error_sustain_seconds=60), store=InMemoryStore()
    )
    opened = [
        [
            finding.finding_id
            for finding in service.ingest(
                erroring_batch(f"dcgm-{offset}", NOW + timedelta(seconds=offset))
            ).new_findings
        ]
        for offset in (0, 15, 30, 60, 120)
    ]
    assert opened == [[], [], [], ["dcgm-0-dcgm-fields"], []], opened


def test_a_finding_booked_before_the_window_existed_stays_escalated() -> None:
    """Upgrade: an active finding the previous release booked asks for
    RUN_DIAGNOSTICS and already has its incident; the streak it continues must
    neither downgrade it nor activate it again."""

    store = InMemoryStore()
    legacy = erroring_batch("dcgm-legacy", NOW)
    legacy_finding = (
        GpuMetricsService(store=store)
        .ingest(legacy)
        .findings[0]
        .model_copy(
            update={
                "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
                "reason": legacy.collection_errors[0],
            }
        )
    )
    store.update_gpu_finding(
        COMPLETENESS_KEY, legacy_finding, NOW + timedelta(seconds=1)
    )

    result = GpuMetricsService(store=store).ingest(
        erroring_batch("dcgm-next", NOW + timedelta(seconds=16))
    )

    assert result.new_findings == [], "the previous release opened its incident"
    state = store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert state is not None and state.finding is not None
    assert state.finding.automatic_action == RecoveryAction.RUN_DIAGNOSTICS.value


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_the_streak_start_survives_across_batches(backend: str, tmp_path) -> None:
    if backend == "memory":
        store = InMemoryStore()
        assert_streak_survives_the_store(lambda: store)
        return
    opened: list[SqliteStore] = []

    def reopened() -> SqliteStore:
        if opened:
            opened[-1].close()
        opened.append(SqliteStore(str(tmp_path / "streak.db")))
        return opened[-1]

    try:
        assert_streak_survives_the_store(reopened)
    finally:
        opened[-1].close()
    assert len(opened) > len(DEAD_EXPORTER_OFFSETS), "every batch used a new connection"
