"""VALIDATE_GPU past the warning grace: the verdict follows the data, not the
node's health-summary phase.

A GPU-metrics growth finding (``thermal_violation_total_us``, WARNING by
``thermal_duration_decision``) describes the interval that ends at the batch
that minted or re-graded it. The collector delivers edges only on confirmation
and recovery and otherwise one health summary per node on its deterministic
stable phase (270-330 s apart), so the next batch that can re-judge the
finding -- replace it, clear it, or leave it untouched when the counter reset
-- lands when the phase says, not when the fault does. Live, one physical
sequence read two ways: plain EKS 2026-10-26 (diagnostic workflows created
23:24:17.98Z) PASSED at 23:24:49Z because a summary landed between the
confirmed and the recovered reads and the recovered edge was a zero delta;
HyperPod 2026-10-27 (created 01:48:01.5Z) FAILED at the grace end,
01:50:01.6Z, on the finding the recovered edge had re-minted at 01:48:32Z: the
only newer batch, the 01:49:47Z summary, carried a counter reset the ingest
refuses to grade. 51 of 60 summary phases ended that sequence inconclusive
(the b201 sweep of the live code).

These tests replay both clusters' sequences on their live tick cadence, with
the first read placed so each node's health-summary phase falls where the
live one fell, through the real DCGM collector delivery filter and the real
``GpuMetricsService.ingest`` into one store, and run
``GpuValidationAdapter.execute`` against that store with a controlled clock: inside the grace the wait is the cooldown it always was;
past the grace the step keeps WAITING and names the batch it waits for; the
batch that clears the finding ends the wait in success; a re-grade by newer
data past the grace fails as before; and the executor's per-step cap still
bounds the wait.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters import gpu_validation
from gpu_fault.adapters.gpu_validation import GpuValidationAdapter
from gpu_fault.collectors import CollectorContext, next_stable_phase
from gpu_fault.execution import (
    WorkflowExecutionRequest,
    WorkflowStepContext,
    step_bounds,
)
from gpu_fault.execution.config import ProductionExecutorConfig
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricsService
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import copy_model, workflow_step_execution
from tests.collectors._support import DcgmMetricsCollector, RecordingSink
from tests.execution._support import workflow_state

CLOCK = "clock_throttle_reasons"
THERMAL = "thermal_violation_total_us"
#: The real exporter's GPU_IDLE bit: not a candidate, so pass-through reads
#: before the arm and after the plan are not edges.
IDLE = 1
HW_THERMAL = 0x40
GRACE = timedelta(seconds=120)
STEP_CAP = timedelta(seconds=600)
#: The live plan: four rising reads under the HW thermal bit, three flat
#: reads under no reason; the proxy then served the real counter (0) again.
PASSTHROUGH: list[tuple[int, float]] = [(IDLE, 0.0), (IDLE, 0.0)]
RISING = [(HW_THERMAL, 0.0), (HW_THERMAL, 1e6), (HW_THERMAL, 2e6), (HW_THERMAL, 3e6)]
FLAT = [(0, 3e6)] * 3
#: A tick this long after the last delivery passes the node's stable phase
#: whatever the phase is (``jittered_next_phase`` is at most 330 s out) and
#: does not reach the one after it: exactly one health summary.
ONE_SUMMARY_LATER = 420


#: Where the sites' first reads are placed: a date, not a live timestamp.
ANCHOR = datetime(2026, 10, 27, 1, 47, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Site:
    """A cluster's ids (they fix the node's summary phase) and tick cadence.

    The first read is placed so the node's next stable phase -- its first
    health summary -- falls ``summary_after`` seconds later, which is how the
    live phase happened to fall relative to the live reads (HyperPod: 165 s,
    the summary after the recovered edge and inside the grace; plain EKS:
    70 s, the summary between the confirmed and the recovered reads). The
    phase mechanism itself is the collector's; only the ids are generic.
    """

    cluster: str
    node: str
    gpu: str
    boot: str
    interval: float
    profile: str
    summary_after: float

    @property
    def first_read_at(self) -> datetime:
        phase = next_stable_phase(
            ANCHOR,
            cluster_id=self.cluster,
            node_id=self.node,
            channel="gpu-metrics",
            interval_seconds=300,
        )
        return phase - timedelta(seconds=self.summary_after)


#: GF-ENV-LIVE-006 ``hw_thermal`` as the HyperPod fault node saw it, 2026-10-27.
HYPERPOD = Site(
    cluster="hp-cluster",
    node="hyperpod-node-a",
    gpu="GPU-1a588e85-965e-fc68-40e4-225aaa3315b6",
    boot="3dad2d1c-6eb1-4958-9991-11b4f9c724b0",
    interval=15.37,
    profile="hp-v1",
    summary_after=165,
)
#: The same plan as the plain EKS fault node saw it the day before.
PLAIN_EKS = Site(
    cluster="eks-cluster",
    node="eks-node-a",
    gpu="GPU-9d21f50b-8c80-b4e0-8f24-eff9d453315f",
    boot="b39c4e22-0873-4642-ba92-895dfd9e76ef",
    interval=15.32,
    profile="eks-v1",
    summary_after=70,
)


def exporter_text(site: Site, clock: int, thermal_us: float) -> str:
    """One scrape of an idle H200 whose clock reasons and thermal-violation
    duration are the plan's (the exporter reports the duration in ns)."""

    labels = f'gpu="0",UUID="{site.gpu}",modelName="NVIDIA H200"'
    return "\n".join(
        [
            f"DCGM_FI_DEV_GPU_TEMP{{{labels}}} 30",
            f"DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_FAILURE{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_PENDING{{{labels}}} 0",
            f"DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_CRC{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY{{{labels}}} 0",
            f"DCGM_FI_DEV_CLOCKS_EVENT_REASONS{{{labels}}} {clock}",
            f"DCGM_FI_DEV_POWER_MGMT_LIMIT{{{labels}}} 700",
            "# TYPE DCGM_FI_DEV_THERMAL_VIOLATION counter",
            f"DCGM_FI_DEV_THERMAL_VIOLATION{{{labels}}} {thermal_us * 1000:.0f}",
            "",
        ]
    )


def deliver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    site: Site,
    reads: list[tuple[int, float]],
    times: list[float],
) -> list[GpuMetricBatch]:
    """What the real collector ships for ``reads`` scraped ``times`` seconds
    after the site's first read, with the site's ids and no cadence jitter."""

    path = tmp_path / "boot"
    path.write_text(site.boot + "\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(path))
    monkeypatch.setenv("GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO", "0")
    sink = RecordingSink()
    clock = iter(site.first_read_at + timedelta(seconds=at) for at in times)
    dcgm = DcgmMetricsCollector(
        sink,
        CollectorContext(
            cluster_id=site.cluster,
            runtime_profile_version=site.profile,
            product="H200",
            driver_branch=580,
            cuda_version="13.0",
        ),
        node_id=site.node,
        now=lambda: next(clock),
        health_summary_seconds=300,
        edge_confirmation_samples=3,
    )
    for mask, thermal_us in reads:
        dcgm.collect_text(exporter_text(site, mask, thermal_us))
    return [GpuMetricBatch.model_validate(payload) for _path, payload in sink.requests]


def live_ticks(site: Site, reads: list[tuple[int, float]]) -> list[float]:
    return [site.interval * index for index in range(len(reads))]


def reasons(batches: list[GpuMetricBatch]) -> list[list[str]]:
    return [batch.edge_filter_reasons for batch in batches]


def offsets(batches: list[GpuMetricBatch]) -> list[int]:
    """Each delivery's whole seconds after the first read."""

    first = batches[0].observed_at
    return [int((batch.observed_at - first).total_seconds()) for batch in batches]


def hyperpod_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[GpuMetricBatch]:
    """The HyperPod delivery order on the live cadence: the recovered edge
    re-mints the duration finding (+1e6 over the confirmed read), the node's
    next summary (live 01:49:48Z, 107 s after the confirmed read) carries the
    proxy's real counter -- a reset -- and the one after it, 300 s later, a
    zero delta."""

    reads = [*PASSTHROUGH, *RISING, *FLAT, *([(IDLE, 0.0)] * 26)]
    batches = deliver(
        monkeypatch, tmp_path, HYPERPOD, reads, live_ticks(HYPERPOD, reads)
    )
    assert reasons(batches) == [
        ["initial-baseline"],
        ["candidate-confirmed"],
        ["candidate-recovered"],
        ["health-summary"],
        ["health-summary"],
    ], "the live HyperPod order: confirmed, recovered, reset summary, flat summary"
    assert offsets(batches) == [0, 61, 92, 169, 476], (
        "the live timeline relative to the first read (live: 01:46:59Z, "
        "01:48:01Z, 01:48:32Z, 01:49:48Z; the second summary 300 s on)"
    )
    return batches


class Rig:
    """One store shared by the real ingest and the validation adapter, and a
    VALIDATE_GPU step of the diagnostic workflow the confirmed batch created."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, site: Site, confirmed: GpuMetricBatch
    ) -> None:
        self.monkeypatch = monkeypatch
        self.site = site
        self.store = InMemoryStore()
        self.service = GpuMetricsService(store=self.store)
        self.adapter = GpuValidationAdapter(
            self.service,
            store=self.store,
            temperature_warning_grace=GRACE,
            step_waiting_cap=STEP_CAP,
        )
        # Live: the workflows were created 0.2 s after the confirmed batch.
        created_at = confirmed.observed_at + timedelta(milliseconds=200)
        incident, workflow = workflow_state(
            self.store, [WorkflowOperation.VALIDATE_GPU]
        )
        self.incident = copy_model(
            incident,
            cluster_id=site.cluster,
            created_at=created_at,
            updated_at=created_at,
        )
        self.step = copy_model(
            workflow.official_steps[0],
            execution_owner=self.adapter.owner,
            node_ids=[site.node],
            gpu_uuids=[site.gpu],
        )
        self.workflow = copy_model(
            workflow,
            official_steps=[self.step],
            created_at=created_at,
            updated_at=created_at,
        )
        self.grace_end = created_at + GRACE
        # Live: VALIDATE_GPU's first attempt came 1.5 s after the creation.
        self.step_started_at = created_at + timedelta(seconds=1.5)

    def ingest(self, *batches: GpuMetricBatch) -> None:
        for batch in batches:
            result = self.service.ingest(batch)
            # What the ingestion service records for the node's collector.
            self.store.save_collector_status(
                CollectorStatus(
                    cluster_id=self.site.cluster,
                    node_id=self.site.node,
                    collector=CollectorKind.GPU_METRICS,
                    observed_at=batch.observed_at,
                    ingested_at=batch.observed_at,
                    last_success_at=batch.observed_at,
                    batch_id=batch.batch_id,
                    sample_count=result.accepted_samples,
                )
            )

    def active(self) -> dict[str, Any]:
        """What ``_validate_active_findings`` reads for the GPU."""

        return {
            finding.canonical_name: finding
            for finding in self.service.findings(self.site.cluster, self.site.node)
            if finding.gpu_uuid == self.site.gpu
        }

    def history_ids(self) -> set[str]:
        return {
            finding.finding_id
            for finding in self.service.findings(
                self.site.cluster, self.site.node, active_only=False
            )
        }

    def validate(self, now: datetime, *, first_attempt: bool = False):
        """One VALIDATE_GPU attempt at ``now``. Unless it is the step's first
        attempt, the step's live record (which the cap is measured from)
        says when the step started."""

        self.monkeypatch.setattr(
            gpu_validation, "datetime", SimpleNamespace(now=lambda *_args: now)
        )
        executions = (
            []
            if first_attempt
            else [
                workflow_step_execution(
                    0,
                    WorkflowOperation.VALIDATE_GPU,
                    WorkflowStepStatus.WAITING,
                    started_at=self.step_started_at,
                )
            ]
        )
        return self.adapter.execute(
            WorkflowStepContext(
                workflow=copy_model(self.workflow, step_executions=executions),
                incident=self.incident,
                step=self.step,
                step_index=0,
                request=WorkflowExecutionRequest(expected_fencing_token=3),
                idempotency_key="validation/follows-data",
            )
        )


def test_inside_the_grace_the_wait_is_the_cooldown_as_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    batches = hyperpod_sequence(monkeypatch, tmp_path)
    rig = Rig(monkeypatch, HYPERPOD, batches[1])
    rig.ingest(*batches[:3])

    outcome = rig.validate(rig.workflow.created_at + timedelta(seconds=45))

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_pending"][HYPERPOD.node] == {
        "transient_gpu_warning_cooldown": [rig.active()[THERMAL].finding_id]
    }, "the fixed grace is still the minimum wait"


def test_past_the_grace_with_no_newer_batch_the_step_waits_and_says_for_what(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HyperPod at the grace end, had the 01:49:48Z summary not landed yet:
    nothing newer than the recovered edge has judged the finding, so the step
    waits for that batch instead of failing on the state the grace end saw."""

    batches = hyperpod_sequence(monkeypatch, tmp_path)
    confirmed, recovered = batches[1], batches[2]
    rig = Rig(monkeypatch, HYPERPOD, confirmed)
    rig.ingest(*batches[:3])
    thermal = rig.active()[THERMAL]
    assert thermal.observed_at == recovered.observed_at, (
        "the recovered edge re-minted the finding (+1e6 over the confirmed read)"
    )
    assert f"{confirmed.batch_id}-{HYPERPOD.gpu}-{THERMAL}" in rig.history_ids()
    assert thermal.finding_id in rig.history_ids(), (
        "the same-severity replacement is in the finding history"
    )
    now = rig.grace_end + timedelta(seconds=1)

    outcome = rig.validate(now, first_attempt=True)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["failed_nodes_observed"] == []
    pending = outcome.details["node_pending"][HYPERPOD.node]
    step_cap_at = now + STEP_CAP
    assert pending == {
        "awaiting_newer_gpu_metrics": [thermal.finding_id],
        "awaiting_newer_gpu_metrics_findings": [
            {
                "finding_id": thermal.finding_id,
                "canonical_name": THERMAL,
                "gpu_uuid": HYPERPOD.gpu,
                "observed_at": recovered.observed_at.isoformat(),
                "newest_sample_observed_at": None,
            }
        ],
        "grace_elapsed_at": rig.grace_end.isoformat(),
        "step_cap_at": step_cap_at.isoformat(),
        "reason": (
            f"awaiting a GPU metrics batch newer than finding {thermal.finding_id} "
            f"(observed {recovered.observed_at.isoformat()}) to re-judge it; "
            f"grace elapsed {rig.grace_end.isoformat()}, "
            f"step cap {step_cap_at.isoformat()}"
        ),
    }, "on the step's first attempt the cap counts from now"


def test_a_newer_batch_that_could_not_re_judge_the_finding_keeps_the_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HyperPod at 01:50:01Z, the live grace end: the 01:49:48Z summary carried
    the proxy's real counter, a reset the ingest does not grade. The finding
    is as the recovered edge left it; the step says the newer sample did not
    re-judge it and keeps waiting where the live step failed."""

    batches = hyperpod_sequence(monkeypatch, tmp_path)
    recovered, reset_summary = batches[2], batches[3]
    rig = Rig(monkeypatch, HYPERPOD, batches[1])
    rig.ingest(*batches[:4])
    thermal = rig.active()[THERMAL]
    assert thermal.observed_at == recovered.observed_at, "kept, not re-graded"
    assert reset_summary.observed_at < rig.grace_end, "the live summary was inside"
    now = rig.grace_end + timedelta(milliseconds=100)

    outcome = rig.validate(now)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    pending = outcome.details["node_pending"][HYPERPOD.node]
    assert pending["awaiting_newer_gpu_metrics"] == [thermal.finding_id]
    assert (
        pending["awaiting_newer_gpu_metrics_findings"][0]["newest_sample_observed_at"]
        == reset_summary.observed_at.isoformat()
    )
    assert pending["step_cap_at"] == (rig.step_started_at + STEP_CAP).isoformat(), (
        "the cap counts from the step's first attempt"
    )
    assert pending["reason"] == (
        f"awaiting a GPU metrics batch newer than finding {thermal.finding_id} "
        f"(observed {recovered.observed_at.isoformat()}; the newest sample, "
        f"{reset_summary.observed_at.isoformat()}, did not re-judge it) to "
        f"re-judge it; grace elapsed {rig.grace_end.isoformat()}, "
        f"step cap {pending['step_cap_at']}"
    )


def test_the_batch_that_clears_the_finding_ends_the_wait_in_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The node's next summary, 300 s after the reset, is a zero delta: the
    ingest clears the finding and the next attempt succeeds, inside the step
    cap (live: due 01:54:48Z, cap 01:58:03Z)."""

    batches = hyperpod_sequence(monkeypatch, tmp_path)
    rig = Rig(monkeypatch, HYPERPOD, batches[1])
    rig.ingest(*batches)
    assert rig.active() == {}, "0 after 0 is a zero delta: the finding is cleared"
    clearing = batches[4]
    assert clearing.observed_at - batches[3].observed_at < timedelta(seconds=330)
    now = clearing.observed_at + timedelta(seconds=10)
    assert now < rig.step_started_at + STEP_CAP

    outcome = rig.validate(now)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert outcome.details["validated_nodes"] == [HYPERPOD.node]


def test_a_finding_re_graded_by_newer_data_past_the_grace_fails_as_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The throttling goes on: the summary after the grace still carries the
    HW thermal bit and a higher duration, so the ingest re-grades both
    findings on data newer than the grace end. That is a verdict."""

    reads = [*PASSTHROUGH, *RISING[:3], (HW_THERMAL, 5e6)]
    times = [0, 15, 30, 45, 60, 60 + ONE_SUMMARY_LATER]
    batches = deliver(monkeypatch, tmp_path, HYPERPOD, reads, times)
    assert reasons(batches) == [
        ["initial-baseline"],
        ["candidate-confirmed"],
        ["health-summary"],
    ]
    rig = Rig(monkeypatch, HYPERPOD, batches[1])
    rig.ingest(*batches)
    summary = batches[2]
    assert summary.observed_at > rig.grace_end
    active = rig.active()
    assert {name: item.observed_at for name, item in active.items()} == {
        CLOCK: summary.observed_at,
        THERMAL: summary.observed_at,
    }, "both WARNING findings were re-minted by the summary"

    outcome = rig.validate(summary.observed_at + timedelta(seconds=10))

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details["node_failures"] == {
        HYPERPOD.node: ["active_gpu_health_findings"]
    }
    assert outcome.error is not None and outcome.error.startswith(
        f"validation failed on nodes: {HYPERPOD.node} (active_gpu_health_findings["
    )
    assert f"{THERMAL} WARNING 5e+06>10000 {HYPERPOD.gpu}" in outcome.error


def test_the_plain_eks_order_clears_on_the_recovered_edge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Plain EKS 2026-10-26 on the live cadence: the node's summary (23:24:32Z)
    landed between the confirmed (23:24:17Z) and the recovered (23:24:48Z)
    reads, so the recovered edge was a zero delta and cleared both findings.
    Unchanged: the step succeeds on its next attempt (live: 23:24:49Z)."""

    reads = [*PASSTHROUGH, *RISING, *FLAT, *([(IDLE, 0.0)] * 4)]
    batches = deliver(
        monkeypatch, tmp_path, PLAIN_EKS, reads, live_ticks(PLAIN_EKS, reads)
    )
    assert reasons(batches) == [
        ["initial-baseline"],
        ["candidate-confirmed"],
        ["health-summary"],
        ["candidate-recovered"],
    ]
    assert offsets(batches) == [0, 61, 76, 91], (
        "the live timeline relative to the first read (live: 23:23:16Z, "
        "23:24:17Z, 23:24:32Z, 23:24:48Z)"
    )
    rig = Rig(monkeypatch, PLAIN_EKS, batches[1])
    rig.ingest(*batches)
    assert rig.active() == {}

    outcome = rig.validate(batches[3].observed_at + timedelta(seconds=1))

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome


def test_the_executor_cap_still_bounds_the_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The adapter keeps WAITING; ``step_bounds.bounded_waiting_outcome`` ends
    the wait at the per-step cap with its own failure, as for every other
    step, and the record keeps the adapter's reason."""

    batches = hyperpod_sequence(monkeypatch, tmp_path)
    rig = Rig(monkeypatch, HYPERPOD, batches[1])
    rig.ingest(*batches[:4])
    assert set(rig.active()) == {THERMAL}
    # The step's clock started a whole cap before this attempt at the grace
    # end (the workflow spent that long on the steps before it); the data is
    # fresh (the reset summary, 01:49:48Z).
    now = rig.grace_end + timedelta(milliseconds=100)
    rig.step_started_at = now - STEP_CAP

    outcome = rig.validate(now)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    pending = outcome.details["node_pending"][HYPERPOD.node]
    assert pending["step_cap_at"] == now.isoformat(), (
        "started a cap ago: the executor ends the wait on this attempt"
    )
    executor = SimpleNamespace(
        config=ProductionExecutorConfig.from_mapping({}), store=rig.store
    )
    assert executor.config.step_waiting_limit(WorkflowOperation.VALIDATE_GPU) == (
        int(STEP_CAP.total_seconds())
    )
    capped = copy_model(
        rig.workflow,
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.VALIDATE_GPU,
                WorkflowStepStatus.WAITING,
                started_at=datetime.now(timezone.utc) - STEP_CAP,
                details=outcome.details,
            )
        ],
    )

    bounded = step_bounds.bounded_waiting_outcome(
        executor, capped, rig.step, 0, outcome
    )

    assert bounded.status is WorkflowStepStatus.FAILED, bounded
    assert bounded.error is not None
    assert "past the 600s per-step cap" in bounded.error
    assert bounded.details["step_waiting_timeout_seconds"] == 600
    assert bounded.details["node_pending"][HYPERPOD.node]["reason"].startswith(
        "awaiting a GPU metrics batch newer than finding "
    ), "the capped record keeps the adapter's reason for the wait"
