"""VALIDATE_GPU after a reset: a pending remap/retirement is re-judged on a GPU
batch newer than the reset, not failed on the pre-reset reading (b209).

NVIDIA applies a pending row remap and a pending page retirement at the next GPU
reset (``adapters/pending_repair_rejudge`` quotes the pages), so after the
chain's RESET_GPU (or a GPU + NVSwitch reset, or a node reboot) a pending-flag
finding whose current state is not newer than the reset's completion describes
the GPU before the reset. VALIDATE_GPU failed it at once anyway: the warning
grace and b202's newer-batch wait cover only the thermal, correctable-memory and
power WARNING families. The gap is latent -- found by code reading during the
GF-GENERAL-LIVE-024 analysis, never seen live -- because the freshness gates in
front of the finding check (the GPU_METRICS collector status and every targeted
GPU's metric presence must postdate the reset) wait whenever the latest batch
is still the pre-reset one.

So the race is forced here: the reset completes, the restarted collector's
first scrape after RESTORE_GPU_SERVICES lands, and VALIDATE_GPU reads between
that batch's latest-value rows and its finding-state write -- the fresh
post-reset temperature beside the pre-reset pending finding, the state any path
in which post-reset GPU data passes the gates without re-judging the flag leaves
behind. Both clusters' node shapes run it, through the real DCGM collector
delivery filter and the real ``GpuMetricsService.ingest``: today the step
FAILED at once; now it waits, the batch's write landing decides (cleared ->
SUCCEEDED; still pending -> the existing failure), and the executor's per-step
cap still bounds the wait. The scoping (which steps apply the repair, on which
node and GPU), the unchanged verdicts and the vendor citations are pinned below.
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
from gpu_fault.adapters.pending_repair_rejudge import (
    NVIDIA_GPU_DEBUG_ROW_REMAP_QUOTE,
    NVIDIA_GPU_DEBUG_ROW_REMAP_REFERENCE,
    NVIDIA_PAGE_RETIREMENT_PENDING_QUOTE,
    NVIDIA_PAGE_RETIREMENT_PENDING_REFERENCE,
    NVIDIA_PAGE_RETIREMENT_REATTACH_REFERENCE,
    NVIDIA_PENDING_REPAIR_DOCS_VERSION,
    NVIDIA_ROW_REMAP_PENDING_QUOTE,
    NVIDIA_ROW_REMAP_PENDING_REFERENCE,
    NVIDIA_ROW_REMAP_RESET_QUOTE,
    NVIDIA_ROW_REMAP_RESET_REFERENCE,
    PENDING_REPAIR_METRICS,
    PENDING_REPAIR_RESET_OPERATIONS,
)
from gpu_fault.collectors import CollectorContext, next_stable_phase
from gpu_fault.execution import (
    WorkflowExecutionRequest,
    WorkflowStepContext,
    step_bounds,
)
from gpu_fault.execution.config import ProductionExecutorConfig
from gpu_fault.gpu_memory_policy import MEMORY_METRICS, memory_decision
from gpu_fault.gpu_metric_models import GpuHealthFinding, GpuHealthSeverity
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
    GpuMetricsThresholds,
)
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from tests._builders import copy_model, workflow_step, workflow_step_execution
from tests.collectors._support import DcgmMetricsCollector, RecordingSink
from tests.execution._support import workflow_state

Op = WorkflowOperation
PENDING = "row_remap_pending"
RETIRED = "retired_pages_pending"
GRACE = timedelta(seconds=120)
STEP_CAP = timedelta(seconds=600)
#: The chain the product compiles for a pending-repair finding (the remote
#: step batching module's reset chain); VALIDATE_GPU is step 5.
RESET_CHAIN = [
    Op.MARK_UNSCHEDULABLE,
    Op.QUIESCE_GPU_SERVICES,
    Op.VERIFY_NO_GPU_CLIENTS,
    Op.RESET_GPU,
    Op.RESTORE_GPU_SERVICES,
    Op.VALIDATE_GPU,
    Op.RESTORE_SCHEDULING,
]
VALIDATE_INDEX = RESET_CHAIN.index(Op.VALIDATE_GPU)
RESET_INDEX = RESET_CHAIN.index(Op.RESET_GPU)
#: Seconds after the workflow's creation each chain step completed; the
#: quiesce stops the node's collector, the restore starts it again, and every
#: other step -- the reset, a reboot, any other action -- completes at 25 s.
STEP_DONE = {
    Op.MARK_UNSCHEDULABLE: 1,
    Op.QUIESCE_GPU_SERVICES: 6,
    Op.VERIFY_NO_GPU_CLIENTS: 7,
    Op.RESTORE_GPU_SERVICES: 35,
}
ACTION_DONE = 25


def done_after(operation: WorkflowOperation) -> timedelta:
    return timedelta(seconds=STEP_DONE.get(operation, ACTION_DONE))


#: A date, not a live timestamp.
ANCHOR = datetime(2026, 10, 27, 14, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Site:
    """A cluster's node shape: its ids (they fix the node's health-summary
    phase) and its collector cadence. Both providers run the same DCGM
    collector on the node and the same RESET_GPU node action."""

    cluster: str
    node: str
    gpu: str
    other_gpu: str
    boot: str
    interval: float
    profile: str

    @property
    def first_read_at(self) -> datetime:
        """Placed so the node's first health summary lands 200 s after the
        first read: after the mint, while the chain's quiesce has the
        collector stopped -- the pre-reset deliveries are exactly the baseline
        and the confirmation."""

        phase = next_stable_phase(
            ANCHOR,
            cluster_id=self.cluster,
            node_id=self.node,
            channel="gpu-metrics",
            interval_seconds=300,
        )
        return phase - timedelta(seconds=200)


HYPERPOD = Site(
    cluster="hp-cluster",
    node="hyperpod-node-a",
    gpu="GPU-4f6a2c1e-8d3b-4e7a-9c21-5b0d7e3f9a10",
    other_gpu="GPU-0c9e8d7f-6a5b-4c3d-8e2f-1a0b9c8d7e6f",
    boot="5d7c1b2a-3e4f-4a6b-8c9d-0e1f2a3b4c5d",
    interval=15.37,
    profile="hp-v1",
)
PLAIN_EKS = Site(
    cluster="eks-cluster",
    node="eks-node-a",
    gpu="GPU-7b3e9a2d-1c4f-4b8e-a6d5-3f2e1d0c9b8a",
    other_gpu="GPU-2e1d0c9b-8a7f-4e6d-b5c4-3a2b1c0d9e8f",
    boot="9a8b7c6d-5e4f-4a3b-9c2d-1e0f9a8b7c6d",
    interval=15.32,
    profile="eks-v1",
)
SITES = pytest.mark.parametrize("site", [HYPERPOD, PLAIN_EKS], ids=lambda s: s.cluster)


def exporter_text(site: Site, pending: int) -> str:
    """One scrape of an idle H200 with every field the collector requires and
    the row-remap pending flag as given."""

    labels = f'gpu="0",UUID="{site.gpu}",modelName="NVIDIA H200"'
    return "\n".join(
        [
            f"DCGM_FI_DEV_GPU_TEMP{{{labels}}} 30",
            f"DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_FAILURE{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_PENDING{{{labels}}} {pending}",
            f"DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_CRC{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY{{{labels}}} 0",
            f"DCGM_FI_DEV_CLOCKS_EVENT_REASONS{{{labels}}} 1",
            f"DCGM_FI_DEV_POWER_MGMT_LIMIT{{{labels}}} 700",
            "",
        ]
    )


def deliver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    site: Site,
    start: datetime,
    reads: list[int],
) -> list[GpuMetricBatch]:
    """What one run of the real collector process ships for ``reads`` (the
    pending flag of each scrape), scraped on the site's cadence from
    ``start``. A new call is a new process: the restore's restart."""

    path = tmp_path / f"boot-{site.node}"
    path.write_text(site.boot + "\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(path))
    monkeypatch.setenv("GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO", "0")
    sink = RecordingSink()
    clock = iter(
        start + timedelta(seconds=site.interval * index) for index in range(len(reads))
    )
    collector = DcgmMetricsCollector(
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
    for pending in reads:
        collector.collect_text(exporter_text(site, pending))
    return [GpuMetricBatch.model_validate(payload) for _path, payload in sink.requests]


class SplitWriteStore(InMemoryStore):
    """A store whose finding-state writes can be held back while the latest-
    value rows of the same batch land: the window VALIDATE_GPU is made to read
    in (``GpuMetricsService.ingest`` writes the two separately,
    ``observe_gpu_metrics`` then ``update_gpu_findings``)."""

    def __init__(self) -> None:
        super().__init__()
        self.holding = False
        self.held: list[list[Any]] = []

    def update_gpu_findings(self, items: Any) -> list[bool]:
        if self.holding:
            self.held.append(list(items))
            return [False] * len(items)
        return super().update_gpu_findings(items)

    def land_held_writes(self) -> None:
        self.holding = False
        for items in self.held:
            super().update_gpu_findings(items)
        self.held.clear()


class Rig:
    """One store shared by the real ingest and the adapter, and the reset
    chain of the workflow the pending-flag finding created."""

    def __init__(
        self,
        site: Site,
        created_at: datetime,
        operations: list[WorkflowOperation] | None = None,
        *,
        step_gpus: dict[WorkflowOperation, list[str]] | None = None,
        step_nodes: dict[WorkflowOperation, list[str]] | None = None,
    ) -> None:
        self.site = site
        self.store = SplitWriteStore()
        self.service = GpuMetricsService(store=self.store)
        self.adapter = GpuValidationAdapter(
            self.service,
            store=self.store,
            temperature_warning_grace=GRACE,
            transient_warning_grace=GRACE,
            step_waiting_cap=STEP_CAP,
        )
        operations = list(RESET_CHAIN if operations is None else operations)
        incident, workflow = workflow_state(self.store, operations)
        self.incident = copy_model(incident, cluster_id=site.cluster)
        steps = [
            workflow_step(
                operation,
                execution_owner=(
                    self.adapter.owner if operation is Op.VALIDATE_GPU else "owner-a"
                ),
                node_ids=(step_nodes or {}).get(operation, [site.node]),
                gpu_uuids=(step_gpus or {}).get(operation, [site.gpu]),
            )
            for operation in operations
        ]
        self.index = operations.index(Op.VALIDATE_GPU)
        self.step = steps[self.index]
        self.workflow = copy_model(
            workflow, official_steps=steps, created_at=created_at, updated_at=created_at
        )
        self.created_at = created_at
        self.done = [
            workflow_step_execution(
                index,
                operation,
                started_at=created_at + done_after(operation) - timedelta(seconds=1),
                updated_at=created_at + done_after(operation),
            )
            for index, operation in enumerate(operations[: self.index])
        ]
        self.reset_at = created_at + done_after(Op.RESET_GPU)
        self.restore_at = created_at + done_after(Op.RESTORE_GPU_SERVICES)
        self.step_started_at = self.restore_at + timedelta(seconds=1)

    def ingest(self, *batches: GpuMetricBatch) -> None:
        for batch in batches:
            result = self.service.ingest(batch)
            # What the ingestion service records for the node's collectors.
            for collector in (CollectorKind.GPU_METRICS, CollectorKind.HOST_TELEMETRY):
                self.store.save_collector_status(
                    CollectorStatus(
                        cluster_id=self.site.cluster,
                        node_id=self.site.node,
                        collector=collector,
                        observed_at=batch.observed_at,
                        ingested_at=batch.observed_at,
                        last_success_at=batch.observed_at,
                        batch_id=batch.batch_id,
                        sample_count=result.accepted_samples,
                    )
                )

    def active(self) -> dict[str, Any]:
        return {
            finding.canonical_name: finding
            for finding in self.service.findings(self.site.cluster, self.site.node)
        }

    def validate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        now: datetime,
        *,
        executions: list[Any] | None = None,
        first_attempt: bool = False,
    ):
        """One VALIDATE_GPU attempt at ``now``; unless it is the step's first,
        its live record says when the step started (the cap counts from it)."""

        monkeypatch.setattr(
            gpu_validation, "datetime", SimpleNamespace(now=lambda *_args: now)
        )
        records = list(self.done if executions is None else executions)
        if not first_attempt:
            records.append(
                workflow_step_execution(
                    self.index,
                    Op.VALIDATE_GPU,
                    WorkflowStepStatus.WAITING,
                    started_at=self.step_started_at,
                )
            )
        return self.adapter.execute(
            WorkflowStepContext(
                workflow=copy_model(self.workflow, step_executions=records),
                incident=self.incident,
                step=self.step,
                step_index=self.index,
                request=WorkflowExecutionRequest(expected_fencing_token=3),
                idempotency_key="validation/pending-repair-after-reset",
            )
        )


def reset_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, site: Site
) -> tuple[Rig, GpuMetricBatch]:
    """The pre-reset half on the site's cadence: the baseline, then the
    pending flag held for the confirmation samples mints the finding; the
    workflow is created 0.2 s later, its chain runs up to RESTORE."""

    before = deliver(monkeypatch, tmp_path, site, site.first_read_at, [0, 1, 1, 1])
    assert [batch.edge_filter_reasons for batch in before] == [
        ["initial-baseline"],
        ["candidate-confirmed"],
    ], "the pre-reset deliveries: the baseline and the confirmation"
    minted = before[1]
    rig = Rig(site, minted.observed_at + timedelta(milliseconds=200))
    rig.ingest(*before)
    finding = rig.active()[PENDING]
    assert (finding.severity, finding.automatic_action) == (
        GpuHealthSeverity.WARNING,
        "RESET_GPU",
    ), "a supported part's pending remap is the service-window reset"
    assert finding.observed_at == minted.observed_at < rig.reset_at
    return rig, minted


def after_restore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rig: Rig, pending: int
) -> GpuMetricBatch:
    """The restarted collector's first scrape, 15 s after RESTORE_GPU_SERVICES."""

    batches = deliver(
        monkeypatch,
        tmp_path,
        rig.site,
        rig.restore_at + timedelta(seconds=15),
        [pending],
    )
    assert [batch.edge_filter_reasons for batch in batches] == [["initial-baseline"]]
    assert batches[0].observed_at > rig.reset_at
    return batches[0]


def reason_after_reset(rig: Rig, finding: Any, newest: datetime, cap: datetime) -> str:
    return (
        f"awaiting a GPU metrics batch newer than finding {finding.finding_id} "
        f"(observed {finding.observed_at.isoformat()}; the newest sample, "
        f"{newest.isoformat()}, did not re-judge it) to re-judge it; RESET_GPU "
        f"completed {rig.reset_at.isoformat()} and NVIDIA applies a pending row "
        f"remap at the next GPU reset ({NVIDIA_ROW_REMAP_PENDING_REFERENCE}); "
        f"step cap {cap.isoformat()}"
    )


@SITES
def test_the_pre_reset_flag_waits_for_a_batch_newer_than_the_reset_and_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, site: Site
) -> None:
    """The race: the post-reset batch's latest rows are in, its finding-state
    write is not, so VALIDATE_GPU sees fresh temperature beside the finding the
    pre-reset confirmation minted. That reading predates the reset NVIDIA
    applies the repair at: the step waits on b202's mechanism (it used to fail
    here at once) and succeeds once the batch's write clears the finding."""

    rig, minted = reset_chain(monkeypatch, tmp_path, site)
    finding = rig.active()[PENDING]
    post = after_restore(monkeypatch, tmp_path, rig, pending=0)
    rig.store.holding = True
    rig.ingest(post)
    assert rig.active()[PENDING].finding_id == finding.finding_id, "not re-judged yet"
    now = post.observed_at + timedelta(seconds=2)

    outcome = rig.validate(monkeypatch, now)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["failed_nodes_observed"] == []
    cap = rig.step_started_at + STEP_CAP
    assert outcome.details["node_pending"][site.node] == {
        "awaiting_newer_gpu_metrics": [finding.finding_id],
        "awaiting_newer_gpu_metrics_findings": [
            {
                "finding_id": finding.finding_id,
                "canonical_name": PENDING,
                "gpu_uuid": site.gpu,
                "observed_at": minted.observed_at.isoformat(),
                "newest_sample_observed_at": post.observed_at.isoformat(),
            }
        ],
        "pending_repair_reset": {
            "operation": "RESET_GPU",
            "step_index": RESET_INDEX,
            "completed_at": rig.reset_at.isoformat(),
        },
        "step_cap_at": cap.isoformat(),
        "policy_references": [NVIDIA_ROW_REMAP_PENDING_REFERENCE],
        "reason": reason_after_reset(rig, finding, post.observed_at, cap),
    }

    rig.store.land_held_writes()
    assert PENDING not in rig.active(), "the real post-reset reading clears it"

    passed = rig.validate(monkeypatch, now + timedelta(seconds=1))

    assert passed.status is WorkflowStepStatus.SUCCEEDED, passed
    assert passed.details["validated_nodes"] == [site.node]


@SITES
def test_a_flag_still_pending_after_the_reset_fails_on_the_newer_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, site: Site
) -> None:
    """The reset did not apply the repair: the first post-reset batch still
    carries the flag. Until its write lands the step waits as above; then the
    finding is the one that batch re-graded, newer than the reset, and the
    existing failure path fails it."""

    rig, _minted = reset_chain(monkeypatch, tmp_path, site)
    post = after_restore(monkeypatch, tmp_path, rig, pending=1)
    rig.store.holding = True
    rig.ingest(post)
    now = post.observed_at + timedelta(seconds=2)

    waiting = rig.validate(monkeypatch, now)

    assert waiting.status is WorkflowStepStatus.WAITING, waiting
    assert "pending_repair_reset" in waiting.details["node_pending"][site.node]

    rig.store.land_held_writes()
    regraded = rig.active()[PENDING]
    assert regraded.observed_at == post.observed_at > rig.reset_at

    failed = rig.validate(monkeypatch, now + timedelta(seconds=1))

    assert failed.status is WorkflowStepStatus.FAILED, failed
    assert failed.details["node_failures"] == {
        site.node: ["active_gpu_health_findings"]
    }
    assert failed.error == (
        f"validation failed on nodes: {site.node} (active_gpu_health_findings"
        f"[{PENDING} WARNING 1 {site.gpu}])"
    )


@SITES
def test_with_the_latest_batch_still_pre_reset_the_freshness_gates_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, site: Site
) -> None:
    """Why the gap stayed latent: before any post-reset batch the collector
    gate already waits (its last success predates the reset), as before."""

    rig, _minted = reset_chain(monkeypatch, tmp_path, site)

    outcome = rig.validate(monkeypatch, rig.step_started_at, first_attempt=True)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_pending"][site.node] == {
        "missing_or_stale_collectors": ["GPU_METRICS"]
    }


@SITES
def test_the_executor_cap_still_bounds_the_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, site: Site
) -> None:
    """No new bound: ``step_bounds.bounded_waiting_outcome`` ends the wait at
    the per-step cap with its own failure, and the record keeps the reason."""

    rig, _minted = reset_chain(monkeypatch, tmp_path, site)
    post = after_restore(monkeypatch, tmp_path, rig, pending=0)
    rig.store.holding = True
    rig.ingest(post)
    now = post.observed_at + timedelta(seconds=2)
    rig.step_started_at = now - STEP_CAP

    outcome = rig.validate(monkeypatch, now)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["node_pending"][site.node]["step_cap_at"] == now.isoformat()
    executor = SimpleNamespace(
        config=ProductionExecutorConfig.from_mapping({}), store=rig.store
    )
    capped = copy_model(
        rig.workflow,
        step_executions=[
            workflow_step_execution(
                rig.index,
                Op.VALIDATE_GPU,
                WorkflowStepStatus.WAITING,
                started_at=datetime.now(timezone.utc) - STEP_CAP,
                details=outcome.details,
            )
        ],
    )

    bounded = step_bounds.bounded_waiting_outcome(
        executor, capped, rig.step, rig.index, outcome
    )

    assert bounded.status is WorkflowStepStatus.FAILED, bounded
    assert bounded.error is not None and "past the 600s per-step cap" in bounded.error
    assert bounded.details["node_pending"][site.node]["reason"].startswith(
        "awaiting a GPU metrics batch newer than finding "
    ), "the capped record keeps the adapter's reason for the wait"


# -- The scoping, on the store state the race leaves: a pending-flag finding
# -- from before the reset, fresh GPU (and host) telemetry from after it. -----

T0 = datetime(2026, 10, 27, 14, 0, tzinfo=timezone.utc)
MODELS = {"H200": "NVIDIA H200", "V100": "Tesla V100-SXM2-32GB"}
DCGM_NAMES = {
    "gpu_temperature_c": "DCGM_FI_DEV_GPU_TEMP",
    PENDING: "DCGM_FI_DEV_ROW_REMAP_PENDING",
    RETIRED: "DCGM_FI_DEV_RETIRED_PENDING",
    "ecc_dbe_volatile_total": "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
}


def gpu_batch(
    site: Site, at: datetime, product: str = "H200", **values: float
) -> GpuMetricBatch:
    return GpuMetricBatch(
        batch_id=f"dcgm-{site.node}-{int(at.timestamp() * 1_000_000)}",
        cluster_id=site.cluster,
        node_id=site.node,
        observed_at=at,
        collected_at=at,
        source=GpuMetricSource.DCGM_EXPORTER,
        product=product,
        runtime_profile_version=site.profile,
        samples=[
            GpuMetricSample(
                metric_name=DCGM_NAMES[name],
                canonical_name=name,
                value=value,
                gpu_index="0",
                gpu_uuid=site.gpu,
                labels={"modelName": MODELS[product]},
            )
            for name, value in values.items()
        ],
    )


def scoped(
    operations: list[WorkflowOperation],
    *,
    flag: str = PENDING,
    product: str = "H200",
    **step_scope: Any,
) -> Rig:
    """The race's store state: the flag minted at T0, the workflow created
    0.2 s later, a post-reset batch with fresh temperature but no reading of
    the flag (it has not re-judged it)."""

    rig = Rig(HYPERPOD, T0 + timedelta(milliseconds=200), operations, **step_scope)
    rig.ingest(
        gpu_batch(HYPERPOD, T0, product, gpu_temperature_c=30, **{flag: 1}),
        gpu_batch(
            HYPERPOD,
            rig.restore_at + timedelta(seconds=15),
            product,
            gpu_temperature_c=30,
        ),
    )
    assert set(rig.active()) == {flag}
    return rig


def verdict(outcome: Any, node: str) -> str:
    if outcome.status is WorkflowStepStatus.WAITING:
        return str(
            outcome.details["node_pending"][node]["pending_repair_reset"]["operation"]
        )
    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details["node_failures"] == {node: ["active_gpu_health_findings"]}
    return "FAILED"


RESET_ALL_CHAIN = [Op.QUIESCE_GPU_SERVICES, Op.RESET_ALL_GPUS_NVSWITCHES]
RESET_ALL_CHAIN += [Op.RESTORE_GPU_SERVICES, Op.VALIDATE_GPU]


@pytest.mark.parametrize(
    ("operations", "step_scope", "expected"),
    [
        (RESET_CHAIN, {}, "RESET_GPU"),
        (RESET_ALL_CHAIN, {}, "RESET_ALL_GPUS_NVSWITCHES"),
        ([Op.MARK_UNSCHEDULABLE, Op.RESTART_NODE, Op.VALIDATE_GPU], {}, "RESTART_NODE"),
        (RESET_CHAIN, {"step_gpus": {Op.RESET_GPU: [HYPERPOD.other_gpu]}}, "FAILED"),
        (RESET_CHAIN, {"step_nodes": {Op.RESET_GPU: ["another-node"]}}, "FAILED"),
        (
            [Op.QUIESCE_GPU_SERVICES, Op.RECOVER_DRAIN_P2P, Op.VALIDATE_GPU],
            {},
            "FAILED",
        ),
        ([Op.QUIESCE_GPU_SERVICES, Op.REMEDIATE_DRIVER, Op.VALIDATE_GPU], {}, "FAILED"),
        (
            [Op.QUIESCE_GPU_SERVICES, Op.UPDATE_SOFTWARE_FIRMWARE, Op.VALIDATE_GPU],
            {},
            "FAILED",
        ),
        ([Op.MARK_UNSCHEDULABLE, Op.REPLACE_NODE, Op.VALIDATE_GPU], {}, "FAILED"),
    ],
    ids=[
        "reset-gpu",
        "reset-all-gpus-nvswitches",
        "restart-node",
        "reset-of-another-gpu",
        "reset-on-another-node",
        "drain-p2p-may-skip-the-reset",
        "driver-remediation",
        "firmware-update",
        "replace-node",
    ],
)
def test_only_a_reset_or_reboot_that_reached_the_gpu_defers_the_verdict(
    monkeypatch: pytest.MonkeyPatch,
    operations: list[WorkflowOperation],
    step_scope: dict[str, Any],
    expected: str,
) -> None:
    rig = scoped(operations, **step_scope)

    outcome = rig.validate(monkeypatch, rig.restore_at + timedelta(seconds=20))

    assert verdict(outcome, HYPERPOD.node) == expected


def test_a_reset_that_did_not_succeed_or_comes_later_defers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = scoped(RESET_CHAIN)
    now = rig.restore_at + timedelta(seconds=20)
    failed_reset = [
        copy_model(item, status=WorkflowStepStatus.FAILED)
        if item.operation is Op.RESET_GPU
        else item
        for item in rig.done
    ]
    later_reset = [
        *[item for item in rig.done if item.operation is not Op.RESET_GPU],
        copy_model(
            next(item for item in rig.done if item.operation is Op.RESET_GPU),
            step_index=rig.index + 1,
        ),
    ]

    for executions in (failed_reset, later_reset):
        outcome = rig.validate(monkeypatch, now, executions=executions)
        assert verdict(outcome, HYPERPOD.node) == "FAILED"


def test_a_page_retirement_pending_flag_waits_on_its_own_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``retired_pages_pending`` on a legacy part (V100: the product's reset
    case for it) waits the same way and cites Dynamic Page Retirement."""

    rig = scoped(RESET_CHAIN, flag=RETIRED, product="V100")
    assert rig.active()[RETIRED].automatic_action == "RESET_GPU"

    outcome = rig.validate(monkeypatch, rig.restore_at + timedelta(seconds=20))

    pending = outcome.details["node_pending"][HYPERPOD.node]
    assert pending["policy_references"] == [NVIDIA_PAGE_RETIREMENT_PENDING_REFERENCE]
    assert (
        "NVIDIA applies a pending page retirement at the next GPU reset "
        f"({NVIDIA_PAGE_RETIREMENT_PENDING_REFERENCE})" in pending["reason"]
    )


def test_a_flag_re_judged_after_the_reset_fails_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch newer than the reset that still carries the flag is a verdict."""

    rig = scoped(RESET_CHAIN)
    at = rig.restore_at + timedelta(seconds=16)
    rig.ingest(gpu_batch(HYPERPOD, at, gpu_temperature_c=30, row_remap_pending=1))
    assert rig.active()[PENDING].observed_at == at > rig.reset_at

    outcome = rig.validate(monkeypatch, at + timedelta(seconds=4))

    assert verdict(outcome, HYPERPOD.node) == "FAILED"


def test_a_pending_flag_beside_another_finding_fails_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a set of pending flags waits: an uncorrectable ECC count on the
    same GPU is not something the reset's next reading can excuse."""

    rig = Rig(HYPERPOD, T0 + timedelta(milliseconds=200))
    rig.ingest(
        gpu_batch(HYPERPOD, T0, gpu_temperature_c=30, row_remap_pending=1),
        gpu_batch(HYPERPOD, T0 + timedelta(seconds=1), ecc_dbe_volatile_total=1),
        gpu_batch(
            HYPERPOD, rig.restore_at + timedelta(seconds=15), gpu_temperature_c=30
        ),
    )
    assert set(rig.active()) >= {PENDING, "ecc_dbe_volatile_total"}

    outcome = rig.validate(monkeypatch, rig.restore_at + timedelta(seconds=20))

    assert verdict(outcome, HYPERPOD.node) == "FAILED"


def test_without_a_reset_in_the_workflow_the_flag_fails_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing short of a reset applies the repair: a VALIDATE_GPU with no
    reset ahead of it judges the flag at once, as before."""

    rig = scoped([Op.FREEZE_EVIDENCE, Op.VALIDATE_GPU])

    outcome = rig.validate(monkeypatch, rig.restore_at + timedelta(seconds=20))

    assert verdict(outcome, HYPERPOD.node) == "FAILED"


def test_the_warning_families_keep_the_grace_after_a_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """b202 unchanged in a reset chain: a thermal WARNING finding from before
    the reset gets the grace cooldown, then the wait past the grace end --
    ``grace_elapsed_at``, not the reset."""

    rig = Rig(HYPERPOD, T0 + timedelta(milliseconds=200))
    rig.ingest(
        gpu_batch(
            HYPERPOD, rig.restore_at + timedelta(seconds=15), gpu_temperature_c=30
        )
    )
    thermal = GpuHealthFinding(
        finding_id="thermal-before-the-reset",
        cluster_id=HYPERPOD.cluster,
        node_id=HYPERPOD.node,
        observed_at=T0,
        severity=GpuHealthSeverity.WARNING,
        reason="thermal violation duration grew",
        canonical_name="thermal_violation_total_us",
        value=3e6,
        threshold_value=10000,
        gpu_uuid=HYPERPOD.gpu,
        automatic_action="RUN_DIAGNOSTICS",
    )
    key = (HYPERPOD.cluster, HYPERPOD.node, HYPERPOD.gpu, thermal.canonical_name)
    rig.store.update_gpu_findings([(key, thermal, T0)])

    inside = rig.validate(monkeypatch, rig.restore_at + timedelta(seconds=20))
    past = rig.validate(monkeypatch, rig.created_at + GRACE + timedelta(seconds=1))

    assert inside.details["node_pending"][HYPERPOD.node] == {
        "transient_gpu_warning_cooldown": [thermal.finding_id]
    }
    waiting = past.details["node_pending"][HYPERPOD.node]
    assert waiting["grace_elapsed_at"] == (rig.created_at + GRACE).isoformat()
    assert "pending_repair_reset" not in waiting
    assert waiting["awaiting_newer_gpu_metrics"] == [thermal.finding_id]


# -- Vendor evidence (NVIDIA docs version 615), pinned. ----------------------


def test_the_vendor_evidence_is_pinned() -> None:
    assert NVIDIA_PENDING_REPAIR_DOCS_VERSION == "615"
    assert NVIDIA_ROW_REMAP_PENDING_REFERENCE == (
        "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/"
        "error-recovery-and-response-flags.html#error-recovery-and-response-flags"
    )
    assert NVIDIA_ROW_REMAP_PENDING_QUOTE == (
        "This flag indicates that row-remapping will happen at the next GPU reset."
    )
    assert NVIDIA_ROW_REMAP_RESET_REFERENCE == (
        "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html"
        "#row-remapping"
    )
    assert NVIDIA_ROW_REMAP_RESET_QUOTE == (
        "The process of row-remapping requires a GPU reset to take effect"
    )
    assert NVIDIA_PAGE_RETIREMENT_PENDING_REFERENCE == (
        "https://docs.nvidia.com/deploy/dynamic-page-retirement/visibility.html"
        "#nvidia-smi"
    )
    assert NVIDIA_PAGE_RETIREMENT_PENDING_QUOTE == (
        "The nvidia-smi “Pending Page Blacklist” field indicates whether a page has "
        "been recently retired and will be blacklisted on the next system reboot or "
        "driver load."
    )
    assert NVIDIA_PAGE_RETIREMENT_REATTACH_REFERENCE == (
        "https://docs.nvidia.com/deploy/dynamic-page-retirement/"
        "blacklisting-and-ecc-error-recovery.html#reattaching-the-gpu"
    )
    assert NVIDIA_GPU_DEBUG_ROW_REMAP_REFERENCE == (
        "https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html"
        "#understanding-xid-messages"
    )
    assert NVIDIA_GPU_DEBUG_ROW_REMAP_QUOTE == (
        "until there is a convenient time to reset the GPU (refer to GPU reset "
        "capabilities/limitations section below) or reboot for row remapping to "
        "activate."
    )
    assert PENDING_REPAIR_RESET_OPERATIONS == {
        Op.RESET_GPU,
        Op.RESET_ALL_GPUS_NVSWITCHES,
        Op.RESTART_NODE,
    }


def test_the_flags_are_the_ones_the_memory_policy_resets() -> None:
    """``PENDING_REPAIR_METRICS`` is exactly what ``memory_decision`` grades
    "GPU memory repair pending; reset in a service window" on a supported part
    (a modern one for the row remap, a legacy one for the page retirement)."""

    graded = {
        name
        for name in MEMORY_METRICS
        for product in ("H200", "V100")
        if (
            decision := memory_decision(
                GpuMetricsThresholds(),
                SimpleNamespace(canonical_name=name, value=1.0, labels={}),
                None,
                product,
            )
        )
        is not None
        and decision["reason"] == "GPU memory repair pending; reset in a service window"
    }

    assert graded == PENDING_REPAIR_METRICS


def test_the_docs_carry_the_evidence() -> None:
    root = Path(__file__).resolve().parents[2]
    for document in (
        "docs/详细设计.md",
        "docs/en/detailed-design.md",
        "docs/components/nvidia-policy.md",
        "docs/en/components/nvidia-policy.md",
    ):
        text = (root / document).read_text(encoding="utf-8")
        for reference in (
            NVIDIA_ROW_REMAP_PENDING_REFERENCE,
            NVIDIA_PAGE_RETIREMENT_PENDING_REFERENCE,
        ):
            assert reference in text, (document, reference)
        assert "docs version 615" in text, document
