"""GF-GENERAL-LIVE 019/020/022: a "must fail" VALIDATE_GPU needs an observed verdict.

The diagnostic-only DCGM chains (019 ``corrected_ecc``/``aggregate_dbe``, 020
``crc``/``replay``/``recovery_counter`` and the counter chain of
``independent_fatal_event``, 022 ``over_rate_limit``) expect FREEZE_EVIDENCE ->
VALIDATE_GPU with the validation FAILING while the replayed counter is held.
``general_live_gpu_verdicts.chain_errors`` used to accept any status but
SUCCEEDED, and every one of those variants judged the first read that showed the
incident -- where the product's VALIDATE_GPU is WAITING for evidence newer than
itself (a WARNING memory finding first waits its 120 s transient grace) -- so a
step that had not judged anything satisfied "must fail". These tests pin the
product's own behaviour (real collector delivery, ingest and validation adapter)
and the runner's: WAITING at the deadline is INCONCLUSIVE, a FAILED step must
name the held reading, the wait lasts until the step has a verdict, and 019's
recovery stage reads whether a fresh batch cleared the finding.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters import gpu_validation
from gpu_fault.adapters.gpu_validation import GpuValidationAdapter
from gpu_fault.collectors import CollectorContext
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricsService
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import CollectorKind, CollectorStatus
from scripts.e2e.regional import general_live_gpu_links as links
from scripts.e2e.regional import general_live_gpu_memory as mem
from scripts.e2e.regional.general_live_gpu_replay import over_limit_increment
from scripts.e2e.regional.general_live_gpu_store import GPU_STATE_PROBE
from scripts.e2e.regional.general_live_gpu_verdicts import (
    GpuChain,
    chain_errors,
    inconclusive_unless_failed,
    judge_chain,
    validation_judgeable,
)
from scripts.e2e.regional.general_live_kernel_xid import XID_DECISION_PROBE
from tests._builders import copy_model, workflow_step_execution
from tests.collectors._support import DcgmMetricsCollector, RecordingSink
from tests.execution._support import workflow_state
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    cleared_state,
    dcgm_probe,
    gpu_incident,
    gpu_workflow,
    held_failure,
    waiting_validation,
)

G0 = "GPU-00000000-0000-4000-8000-000000000000"
G1 = "GPU-11111111-1111-4111-8111-111111111111"
SBE = "ecc_sbe_volatile_total"


# --------------------------------------------------------------------------- #
# The product: what VALIDATE_GPU records while the counter is held
# --------------------------------------------------------------------------- #
CLUSTER = "eks-cluster"
NODE = "eks-node-a"
GPU = "GPU-9d21f50b-8c80-b4e0-8f24-eff9d453315f"
START = datetime(2026, 10, 27, 1, 47, tzinfo=timezone.utc)
#: The exporter fields every scrape carries (the completeness profile's set),
#: so the only active finding is the held counter's.
STEADY_FIELDS = {
    "DCGM_FI_DEV_GPU_TEMP": 30,
    "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": 0,
    "DCGM_FI_DEV_ROW_REMAP_FAILURE": 0,
    "DCGM_FI_DEV_ROW_REMAP_PENDING": 0,
    "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": 0,
    "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC": 0,
    "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY": 0,
    "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY": 0,
    "DCGM_FI_DEV_CLOCKS_EVENT_REASONS": 1,
    "DCGM_FI_DEV_POWER_MGMT_LIMIT": 700,
    "DCGM_FI_DEV_THERMAL_VIOLATION": 0,
}


def _scrape(field: str, value: int) -> str:
    labels = f'gpu="0",UUID="{GPU}",modelName="NVIDIA H200"'
    readings = {**STEADY_FIELDS, field: value}
    return "".join(
        f"{name}{{{labels}}} {reading}\n" for name, reading in readings.items()
    )


class HeldCounter:
    """One node's held DCGM counter through the product: the real collector's
    delivery, ``GpuMetricsService.ingest`` and ``GpuValidationAdapter.execute``
    on one in-memory store, on a controlled clock (the setup of
    ``tests/execution/test_validation_follows_data.py``)."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        field: str,
        *,
        step: int = 1,
    ) -> None:
        boot = tmp_path / "boot"
        boot.write_text("b39c4e22-0873-4642-ba92-895dfd9e76ef\n", encoding="ascii")
        monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(boot))
        monkeypatch.setenv("GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO", "0")
        self.monkeypatch = monkeypatch
        self.field = field
        self.increment = step
        self.sink = RecordingSink()
        ticks = (START + timedelta(seconds=15 * index) for index in itertools.count())
        self.collector = DcgmMetricsCollector(
            self.sink,
            CollectorContext(
                cluster_id=CLUSTER,
                runtime_profile_version="eks-v1",
                product="H200",
                driver_branch=580,
                cuda_version="13.0",
            ),
            node_id=NODE,
            now=lambda: next(ticks),
            health_summary_seconds=300,
            edge_confirmation_samples=3,
        )
        self.store = InMemoryStore()
        self.service = GpuMetricsService(store=self.store)
        self.adapter = GpuValidationAdapter(self.service, store=self.store)
        self.delivered = 0
        self.value = 0

    def scrape(self, reads: int, *, rising: bool = True) -> datetime:
        """``reads`` scrapes, the counter ``step`` higher on each; the
        delivered batches ingested. Returns the last delivery's time."""

        for _ in range(reads):
            self.value += self.increment if rising else 0
            self.collector.collect_text(_scrape(self.field, self.value))
        batches = [
            GpuMetricBatch.model_validate(payload)
            for _path, payload in self.sink.requests[self.delivered :]
        ]
        self.delivered = len(self.sink.requests)
        for batch in batches:
            result = self.service.ingest(batch)
            self.store.save_collector_status(
                CollectorStatus(
                    cluster_id=CLUSTER,
                    node_id=NODE,
                    collector=CollectorKind.GPU_METRICS,
                    observed_at=batch.observed_at,
                    ingested_at=batch.observed_at,
                    last_success_at=batch.observed_at,
                    batch_id=batch.batch_id,
                    sample_count=result.accepted_samples,
                )
            )
        return batches[-1].observed_at

    def workflow(self, created: datetime) -> None:
        incident, workflow = workflow_state(
            self.store, [WorkflowOperation.VALIDATE_GPU]
        )
        self.created = created
        self.incident = copy_model(
            incident, cluster_id=CLUSTER, created_at=created, updated_at=created
        )
        self.step = copy_model(
            workflow.official_steps[0],
            execution_owner=self.adapter.owner,
            node_ids=[NODE],
            gpu_uuids=[GPU],
        )
        self.record = copy_model(
            workflow, official_steps=[self.step], created_at=created, updated_at=created
        )

    def validate(self, now: datetime) -> dict[str, Any]:
        """One VALIDATE_GPU attempt at ``now``, as the activity probe shows it."""

        self.monkeypatch.setattr(
            gpu_validation, "datetime", SimpleNamespace(now=lambda *_args: now)
        )
        outcome = self.adapter.execute(
            WorkflowStepContext(
                workflow=copy_model(
                    self.record,
                    step_executions=[
                        workflow_step_execution(
                            0,
                            WorkflowOperation.VALIDATE_GPU,
                            WorkflowStepStatus.WAITING,
                            started_at=self.created,
                        )
                    ],
                ),
                incident=self.incident,
                step=self.step,
                step_index=0,
                request=WorkflowExecutionRequest(expected_fencing_token=3),
                idempotency_key="general-live/must-fail",
            )
        )
        return {
            "operation": "VALIDATE_GPU",
            "status": outcome.status.value,
            "error": outcome.error,
            "details": dict(outcome.details or {}),
        }


def _chain(canonical: str, gpu: str = GPU) -> GpuChain:
    return GpuChain(
        label=f"{canonical} on {gpu}",
        operations=("FREEZE_EVIDENCE", "VALIDATE_GPU"),
        validation="VALIDATE_GPU",
        validation_must_fail=True,
        held_metric=canonical,
        held_gpu=gpu,
    )


def _judged(canonical: str, execution: dict[str, Any], **chain: Any):
    incident = gpu_incident("inc", f"b-{GPU}-{canonical}", state="ACTION_PENDING")
    workflow = gpu_workflow(
        "wf-1", "inc", ["FREEZE_EVIDENCE", "VALIDATE_GPU"], executions=[execution]
    )
    activity = {"incidents": [incident], "workflows": [workflow]}
    return judge_chain(activity, [incident], _chain(canonical, **chain))


def test_corrected_ecc_validation_waits_on_the_first_read_then_fails_on_the_held_finding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """corrected_ecc as the product handles it: the rising SBE counter mints a
    WARNING on the edge, VALIDATE_GPU waits out the transient-warning grace,
    and the first batch after it re-judges the still-active finding: FAILED,
    naming it. The old verdict passed the first state."""

    node = HeldCounter(monkeypatch, tmp_path, "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL")
    node.scrape(2, rising=False)
    created = node.scrape(4)
    node.workflow(created)

    first = node.validate(created + timedelta(seconds=5))
    assert first["status"] == "WAITING", first
    assert "transient_gpu_warning_cooldown" in first["details"]["node_pending"][NODE]
    judgement = _judged(SBE, first)
    assert judgement.errors == [], judgement
    assert judgement.unfinished == [
        "VALIDATE_GPU did not finish: still WAITING "
        f"({NODE}: transient_gpu_warning_cooldown)"
    ], judgement
    assert chain_errors(
        {
            "incidents": [gpu_incident("inc", f"b-{GPU}-{SBE}")],
            "workflows": [
                gpu_workflow("wf-1", "inc", ["VALIDATE_GPU"], executions=[first])
            ],
        },
        [gpu_incident("inc", f"b-{GPU}-{SBE}")],
        _chain(SBE),
    ), "a caller that takes no INCONCLUSIVE reads the waiting step as an error"

    later = node.scrape(10)
    assert later > created + timedelta(seconds=120), "past the grace end"
    final = node.validate(later + timedelta(seconds=5))
    assert final["status"] == "FAILED", final
    assert _judged(SBE, final).errors == [], "the held reading is named"
    assert _judged(SBE, final).unfinished == [], "a verdict, not a wait"
    other = _judged(SBE, final, gpu=G0)
    assert other.errors == [
        f"VALIDATE_GPU failed without naming {SBE} on {G0}: {final['error'][:200]}"
    ], other


@pytest.mark.parametrize(
    "field,canonical",
    [
        ("DCGM_FI_DEV_ECC_DBE_AGG_TOTAL", "ecc_dbe_aggregate_total"),
        ("DCGM_FI_DEV_NVLINK_ERROR_DL_CRC", "nvlink_crc_aggregate_error_total"),
        ("DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY", "nvlink_replay_aggregate_error_total"),
        (
            "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY",
            "nvlink_recovery_aggregate_error_total",
        ),
        ("DCGM_FI_DEV_PCIE_REPLAY_COUNTER", "pcie_replay_total"),
    ],
)
def test_the_failed_validation_names_the_held_counter_in_the_products_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, canonical: str
) -> None:
    """``held_reading_named`` reads what the adapter writes: the failure reason
    and the held finding's canonical name and GPU in ``node_findings``."""

    # The step each variant serves: one per read for ECC and NVLink (every
    # increase grades), 022 over_rate_limit's increment for the 8/min PCIe rule.
    step = over_limit_increment(15) if canonical == "pcie_replay_total" else 1
    node = HeldCounter(monkeypatch, tmp_path, field, step=step)
    node.scrape(2, rising=False)
    created = node.scrape(6)
    node.workflow(created)
    execution = node.validate(created + timedelta(seconds=5))
    assert execution["status"] == "FAILED", execution
    assert execution["details"]["node_failures"] == {
        NODE: ["active_gpu_health_findings"]
    }, execution
    assert _judged(canonical, execution).errors == [], execution
    wrong = _judged("pcie_replay_total" if "nvlink" in canonical else SBE, execution)
    assert wrong.errors, "a failure on another reading does not satisfy must-fail"


# --------------------------------------------------------------------------- #
# The verdict on one read
# --------------------------------------------------------------------------- #
def test_a_cap_failure_or_success_is_no_pass() -> None:
    capped = {
        "operation": "VALIDATE_GPU",
        "status": "FAILED",
        "error": "step 1/VALIDATE_GPU stayed non-terminal for 600s, past the 600s per-step cap",
        "details": {
            "host_diagnostic_failure": True,
            "step_waiting_timeout_seconds": 600,
        },
    }
    judgement = _judged(SBE, capped)
    assert judgement.errors == [], judgement
    assert judgement.unfinished == [
        "VALIDATE_GPU ended on the executor's 600 s per-step cap, not on the held "
        "reading: step 1/VALIDATE_GPU stayed non-terminal for 600s, past the 600s "
        "per-step cap"
    ], judgement
    bare = {"operation": "VALIDATE_GPU", "status": "FAILED", "details": {}}
    assert _judged(SBE, bare).errors == [
        f"VALIDATE_GPU failed without naming {SBE} on {GPU}: "
    ]
    passed = {"operation": "VALIDATE_GPU", "status": "SUCCEEDED", "details": {}}
    assert _judged(SBE, passed).errors == [
        "VALIDATE_GPU succeeded while the replayed reading was held"
    ]


def test_an_unfinished_validation_is_inconclusive_only_when_nothing_failed() -> None:
    unfinished = ["VALIDATE_GPU did not finish: still WAITING (x)"]
    assert inconclusive_unless_failed({"chain": [], "hardware": []}, unfinished) == (
        unfinished
    )
    assert (
        inconclusive_unless_failed({"chain": [], "hardware": ["RESET_GPU"]}, unfinished)
        == []
    ), "a definite error is the verdict whatever the validation's state"
    assert inconclusive_unless_failed({"chain": []}, []) == []


def test_the_wait_is_judgeable_on_a_verdict_or_an_ended_record() -> None:
    incident = gpu_incident("inc", f"b-{GPU}-{SBE}")
    chain = _chain(SBE)

    def activity(status: str, *executions: dict[str, Any]) -> dict[str, Any]:
        workflow = gpu_workflow(
            "wf-1", "inc", ["VALIDATE_GPU"], status=status, executions=executions
        )
        return {"incidents": [incident], "workflows": [workflow]}

    assert not validation_judgeable(activity("RUNNING"), [incident], chain), (
        "no execution yet"
    )
    assert not validation_judgeable(
        activity("RUNNING", waiting_validation()), [incident], chain
    ), "WAITING is still a wait"
    assert validation_judgeable(
        activity("RUNNING", held_failure(SBE, GPU)), [incident], chain
    ), "a FAILED step is a verdict"
    assert validation_judgeable(activity("BLOCKED"), [incident], chain), (
        "a record that ended before its validation is a verdict"
    )
    assert not validation_judgeable({"incidents": [], "workflows": []}, [], chain), (
        "no incident, nothing to judge"
    )


# --------------------------------------------------------------------------- #
# The variants
# --------------------------------------------------------------------------- #
#: variant -> (module, function, survey field / canonical the variant holds)
DIAGNOSTIC_VARIANTS = {
    "019.corrected_ecc": (mem, "corrected_ecc", SBE),
    "019.aggregate_dbe": (mem, "aggregate_dbe", "ecc_dbe_aggregate_total"),
    "020.crc": (links, "crc", "nvlink_crc_flit_error_total"),
    "020.replay": (links, "replay", "nvlink_replay_error_total"),
    "020.recovery_counter": (links, "recovery_counter", "nvlink_recovery_error_total"),
    "022.over_rate_limit": (links, "over_rate_limit", "pcie_replay_total"),
}


def _activity(*executions: dict[str, Any], operations: tuple[str, ...] = ()) -> Any:
    def document(canonical: str) -> dict[str, Any]:
        incident = gpu_incident("inc", f"b-{G1}-{canonical}", state="ACTION_PENDING")
        workflow = gpu_workflow(
            "wf-1",
            "inc",
            ["FREEZE_EVIDENCE", "VALIDATE_GPU", *operations],
            executions=list(executions),
        )
        return {
            "incidents": [incident],
            "workflows": [workflow],
            "batches": [{"observed_at": "t"}],
            "markers": [],
            "commands": [],
            "notifications": [],
            "metric_latest": [],
            "collector_statuses": [],
            "agent": None,
        }

    return document


class Records:
    """``cpu_python``: activity reads served from ``reads`` (the last repeats),
    the GPU state from ``state``; counts the activity reads."""

    def __init__(
        self,
        reads: list[Callable[[str], dict[str, Any]]],
        canonical: str,
        state: dict[str, Any],
    ) -> None:
        self.reads = reads
        self.canonical = canonical
        self.state = state
        self.activity_reads = 0

    def __call__(self, script: str, *arguments: str) -> dict[str, Any]:
        if script == GPU_STATE_PROBE:
            return dict(self.state)
        if script == XID_DECISION_PROBE:
            return {"decisions": []}
        index = min(self.activity_reads, len(self.reads) - 1)
        self.activity_reads += 1
        return self.reads[index](self.canonical)


def _run(
    tmp_path: Path, key: str, reads: list[Any], *, state: dict[str, Any] | None = None
) -> tuple[dict[str, Any], Records]:
    module, function, canonical = DIAGNOSTIC_VARIANTS[key]
    spec = next(
        item
        for item in module.VARIANTS
        if item.variant == function and item.case_id.endswith(key.split(".")[0])
    )
    records = Records(reads, canonical, state or cleared_state(canonical, G1))
    regional = FakeRegional(cpu_python=records)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ticks = itertools.count()
    ctx.clock = lambda: START + timedelta(seconds=20 * next(ticks))
    ctx.probes[f"dcgm@{ctx.settings.node}"] = dcgm_probe(
        audit=audit_records({f"{G1}/{canonical}": 1.0})
    )
    return getattr(module, function)(ctx), records


@pytest.mark.parametrize("key", sorted(DIAGNOSTIC_VARIANTS))
def test_a_validation_still_waiting_at_the_deadline_is_inconclusive(
    tmp_path: Path, key: str
) -> None:
    """The old vacuous pass, read by read: the incident with VALIDATE_GPU
    WAITING on every read. It satisfied "must fail"; now the wait runs its
    budget and the variant is INCONCLUSIVE, naming the waiting step."""

    out, records = _run(tmp_path, key, [_activity(waiting_validation())])
    assert out["verdict"] == "INCONCLUSIVE", out["errors"]
    assert out["stages"]["chain"] == [], out["stages"]
    assert out["inconclusive"] == [
        "VALIDATE_GPU did not finish: still WAITING "
        "(ip-10-0-43-5.us-west-2.compute.internal: transient_gpu_warning_cooldown)"
    ], out
    assert out["detail"] == out["inconclusive"][0], out
    assert records.activity_reads > 2, "the wait ran its budget, not one read"


@pytest.mark.parametrize("key", sorted(DIAGNOSTIC_VARIANTS))
def test_the_chain_is_judged_once_the_validation_failed_on_the_held_reading(
    tmp_path: Path, key: str
) -> None:
    canonical = DIAGNOSTIC_VARIANTS[key][2]
    reads = [_activity(waiting_validation())] * 3 + [
        _activity(held_failure(canonical, G1))
    ]
    out, records = _run(tmp_path, key, reads)
    assert out["verdict"] == "PASS", out["errors"]
    assert records.activity_reads >= 4, "the WAITING reads were not judged"
    assert "inconclusive" not in out, out


@pytest.mark.parametrize("key", sorted(DIAGNOSTIC_VARIANTS))
def test_a_definite_error_beside_a_waiting_validation_is_a_fail(
    tmp_path: Path, key: str
) -> None:
    out, _records = _run(
        tmp_path, key, [_activity(waiting_validation(), operations=("RESET_GPU",))]
    )
    assert out["verdict"] == "FAIL", out
    assert any("RESET_GPU" in error for error in out["errors"]), out["errors"]
    assert any("did not finish" in note for note in out["notes"]), out["notes"]


@pytest.mark.parametrize("key", sorted(DIAGNOSTIC_VARIANTS))
def test_a_validation_that_failed_on_another_reading_is_a_fail(
    tmp_path: Path, key: str
) -> None:
    out, _records = _run(
        tmp_path, key, [_activity(held_failure("gpu_temperature_c", G1))]
    )
    assert out["verdict"] == "FAIL", out
    assert out["stages"]["chain"] == [
        f"VALIDATE_GPU failed without naming {DIAGNOSTIC_VARIANTS[key][2]} on {G1}: "
        + held_failure("gpu_temperature_c", G1)["error"][:200]
    ], out["stages"]


def test_independent_fatal_event_judges_the_counter_chain_on_a_verdict(
    tmp_path: Path,
) -> None:
    """020 ``independent_fatal_event``'s counter chain shares the diagnostic
    chain: a VALIDATE_GPU still WAITING when the XID's handling has been read
    is INCONCLUSIVE, not a pass."""

    canonical = "nvlink_crc_flit_error_total"
    spec = next(
        item for item in links.VARIANTS if item.variant == "independent_fatal_event"
    )
    decision = {
        "event_id": "kernel-log-kmsg-boot-900000064-xid-137-cluster-abc",
        "xid": links.INDEPENDENT_XID,
        "gpu_uuid": G1,
        "event_source": "KERNEL_LOG",
        "decision": True,
        "disposition": "MONITOR_ONLY",
        "official_action": "IGNORE",
        "action": "NO_ACTION",
        "safety_action": None,
        "requires_operator": False,
        "incident_id": "inc-xid-137",
        "incident_state": "RECOVERED",
        "workflow_request_id": None,
        "duplicate": False,
        "correlated_event_id": None,
        "reasons": [],
    }
    records = Records(
        [_activity(waiting_validation())], canonical, cleared_state(canonical, G1)
    )

    def cpu(script: str, *arguments: str) -> dict[str, Any]:
        if script == XID_DECISION_PROBE:
            return {"decisions": [decision]}
        return records(script, *arguments)

    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    object.__setattr__(settings, "approvals", ("KERNEL_SOURCE_REPLAY",))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ticks = itertools.count()
    ctx.clock = lambda: START + timedelta(seconds=20 * next(ticks))
    ctx.probes[f"dcgm@{ctx.settings.node}"] = dcgm_probe(
        audit=audit_records({f"{G1}/{canonical}": 1.0})
    )
    ctx.probes[f"kernel-replay-kmsg@{ctx.settings.node}"] = FakeProbe(
        {
            "install-records": {"records": 1},
            "open-window": {"after": {"ActiveState": "active"}},
            "arm": {"armed": True, "lines_written": 6},
            "status": {"injected": {"mode": "kmsg", "fixture_lines": 6}},
            "close-window": {"clean": True},
        }
    )
    out = links.independent_fatal_event(ctx)
    assert out["verdict"] == "INCONCLUSIVE", out["errors"]
    assert out["stages"]["counter_chain"] == [], out["stages"]
    assert out["inconclusive"][0].startswith("VALIDATE_GPU did not finish"), out


# --------------------------------------------------------------------------- #
# 019: the recovery stage reads the fresh batch
# --------------------------------------------------------------------------- #
def test_019_recovery_needs_the_finding_cleared_by_a_fresh_batch() -> None:
    assert mem.cleared_errors(cleared_state(SBE, G1), SBE, G1) == []
    still = {
        **cleared_state(SBE, G1),
        "active_findings": [
            {
                "finding_id": f"b-{G1}-{SBE}",
                "canonical_name": SBE,
                "gpu_uuid": G1,
                "value": 3.0,
                "delta": 1.0,
                "rate_per_minute": None,
                "observed_at": "2026-10-24T00:03:00+00:00",
            }
        ],
    }
    errors = mem.cleared_errors(still, SBE, G1)
    assert len(errors) == 1 and "still active after the replay ended" in errors[0], (
        errors
    )
    rising = cleared_state(SBE, G1, values=(1.0, 2.0, 3.0))
    assert mem.cleared_errors(rising, SBE, G1) == [
        f"the product accepted no {SBE} reading of {G1} below the replayed peak 3 "
        "after it: the batch that brought the device's own reading back was never "
        "seen"
    ]
    assert mem.cleared_errors({"active_findings": [], "batches": []}, SBE, G1) == [
        f"the product accepted no {SBE} reading of {G1}"
    ]


def test_corrected_ecc_fails_when_no_fresh_batch_cleared_the_finding(
    tmp_path: Path,
) -> None:
    """The old recovery stage was ``[] if ... else []``: a finding that never
    cleared read as recovered."""

    state = {
        **cleared_state(SBE, G1, values=(1.0, 2.0, 3.0)),
        "active_findings": [
            {
                "finding_id": f"b-{G1}-{SBE}",
                "canonical_name": SBE,
                "gpu_uuid": G1,
                "observed_at": "2026-10-24T00:03:00+00:00",
            }
        ],
    }
    out, _records = _run(
        tmp_path, "019.corrected_ecc", [_activity(held_failure(SBE, G1))], state=state
    )
    assert out["verdict"] == "FAIL", out
    assert out["stages"]["chain"] == [], out["stages"]
    assert len(out["stages"]["recovery"]) == 2, out["stages"]["recovery"]
