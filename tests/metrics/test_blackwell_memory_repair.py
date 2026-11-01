"""Batch 235: Blackwell memory repair -- the nvidia-smi XML flags and field 1507.

The tag names are the live H200 R580 output (``<channel_repair_pending>``,
``<tpc_repair_pending>``, ``<unrepairable_memory>``, all ``No``); every other
value here is a synthetic fixture derived from NVIDIA documentation (no
B-series node): ``Yes`` / ``No`` / ``N/A`` and the recovery actions 0-7. Which
recovery action accompanies a channel repair is not asserted.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.collectors.gpu.smi_health import (
    SMI_XML_QUERY,
    SmiHealthProbe,
    merge_unrepairable,
    parse_smi_health,
)
from gpu_fault.gpu_field_rules import ADVISORY_GPU_FINDING_NAMES, blocks_validation
from gpu_fault.gpu_memory_repair_policy import (
    MEMORY_CHANNEL_REPAIR_PENDING,
    MEMORY_TPC_REPAIR_PENDING,
    MEMORY_UNREPAIRABLE,
)
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
)
from gpu_fault.models import IncidentState, RecoveryAction, WorkloadState
from gpu_fault.policy import (
    ActionDisposition,
    GpuFaultPolicyEngine,
    XidEvent,
    parse_xid154_action,
)
from gpu_fault.store import InMemoryStore
from tests._builders import fault_incident

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
CLUSTER = "b200-cluster"
NODE = "p6-b200-node"
GPU = "GPU-b200-3"
PRODUCTS = ("NVIDIA B200", "NVIDIA B300 SXM6 AC")


def gpu_element(uuid: str, index: int, repair: str) -> str:
    return (
        f'<gpu id="00000000:3{index}:00.0"><product_name>NVIDIA H200</product_name>'
        f"<uuid>{uuid}</uuid><minor_number>{index}</minor_number>"
        f"<pci><pci_bus_id>00000000:3{index}:00.0</pci_bus_id></pci>"
        f"<ecc_errors><volatile><dram_correctable>0</dram_correctable></volatile>"
        f"{repair}</ecc_errors></gpu>"
    )


def smi_log(*gpus: str) -> str:
    return (
        '<?xml version="1.0" ?><nvidia_smi_log><driver_version>580.178.04'
        f"</driver_version>{''.join(gpus)}</nvidia_smi_log>"
    )


#: Live H200 R580 (driver 580.178.04), GPU 0, XML lines 194-196.
LIVE_H200_REPAIR = (
    "<channel_repair_pending>No</channel_repair_pending>"
    "<tpc_repair_pending>No</tpc_repair_pending>"
    "<unrepairable_memory>No</unrepairable_memory>"
)


def values_of(samples) -> dict[tuple[str | None, str], float]:
    return {(item.gpu_uuid, item.canonical_name): item.value for item in samples}


def test_the_live_h200_tags_read_no_on_every_flag():
    samples = parse_smi_health(
        smi_log(
            gpu_element("GPU-h0", 0, LIVE_H200_REPAIR),
            gpu_element("GPU-h1", 1, LIVE_H200_REPAIR),
        )
    )

    assert values_of(samples) == {
        (gpu, name): 0.0
        for gpu in ("GPU-h0", "GPU-h1")
        for name in (
            MEMORY_CHANNEL_REPAIR_PENDING,
            MEMORY_TPC_REPAIR_PENDING,
            MEMORY_UNREPAIRABLE,
        )
    }
    assert {item.pci_bdf for item in samples} == {
        "00000000:30:00.0",
        "00000000:31:00.0",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Yes", 1.0), ("No", 0.0), ("N/A", None), ("", None), ("Pending", None)],
)
def test_yes_no_and_anything_else(raw, expected):
    samples = parse_smi_health(
        smi_log(
            gpu_element(
                GPU, 3, f"<channel_repair_pending>{raw}</channel_repair_pending>"
            )
        )
    )

    assert values_of(samples).get((GPU, MEMORY_CHANNEL_REPAIR_PENDING)) == expected


def test_a_missing_tag_publishes_nothing():
    assert parse_smi_health(smi_log(gpu_element(GPU, 3, ""))) == []


def test_the_tag_is_read_by_name_wherever_it_sits():
    moved = gpu_element(GPU, 3, "").replace(
        "</gpu>",
        "<memory><unrepairable_memory>Yes</unrepairable_memory></memory></gpu>",
    )

    assert values_of(parse_smi_health(smi_log(moved))) == {
        (GPU, MEMORY_UNREPAIRABLE): 1.0
    }


class Runner:
    def __init__(self, result=None, error=None) -> None:
        self.result, self.error, self.calls = result, error, []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if self.error is not None:
            raise self.error
        return self.result


def completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(SMI_XML_QUERY, returncode, stdout, "boom")


def test_the_probe_reads_once_per_interval_and_keeps_nothing_on_failure():
    probe = SmiHealthProbe(interval_seconds=60, logger=logging.getLogger("t"))
    good = Runner(completed(smi_log(gpu_element(GPU, 3, LIVE_H200_REPAIR))))

    first = probe.refresh(NOW, good)
    cached = probe.refresh(NOW + timedelta(seconds=30), good)

    assert len(first) == 3 and cached is first and good.calls == [SMI_XML_QUERY]
    for failing in (
        Runner(error=subprocess.TimeoutExpired(SMI_XML_QUERY, 15)),
        Runner(completed("", returncode=9)),
        Runner(completed("<nvidia_smi_log><gpu>")),
    ):
        probe.next_at = None
        assert probe.refresh(NOW + timedelta(seconds=90), failing) == [], (
            "a wedged driver must not keep a stale No alive"
        )


def flag(name: str, value: float, *, gpu: str = GPU) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=f"nvidia_smi_{name}", canonical_name=name, value=value, gpu_uuid=gpu
    )


def test_field_1507_and_the_tag_are_one_reading_at_the_higher_value():
    dcgm = [
        GpuMetricSample(
            metric_name="DCGM_FI_DEV_MEMORY_UNREPAIRABLE_FLAG",
            canonical_name=MEMORY_UNREPAIRABLE,
            value=0,
            gpu_uuid=GPU,
        )
    ]
    kept = merge_unrepairable(
        dcgm, [flag(MEMORY_UNREPAIRABLE, 1), flag(MEMORY_TPC_REPAIR_PENDING, 0)]
    )

    assert [item.canonical_name for item in kept] == [MEMORY_TPC_REPAIR_PENDING]
    assert dcgm[0].value == 1.0, "either source's Yes counts"
    assert merge_unrepairable([], [flag(MEMORY_UNREPAIRABLE, 1)])[0].value == 1.0


# --- the grades ---------------------------------------------------------------


def batch(
    seconds: float, values: dict[str, float], *, product: str = PRODUCTS[0]
) -> GpuMetricBatch:
    return GpuMetricBatch(
        batch_id=f"rep-{seconds}",
        cluster_id=CLUSTER,
        node_id=NODE,
        product=product,
        observed_at=NOW + timedelta(seconds=seconds),
        source=GpuMetricSource.DCGM_EXPORTER,
        workload_state=WorkloadState.IDLE,
        samples=[
            GpuMetricSample(
                metric_name=name,
                canonical_name=name,
                value=value,
                gpu_uuid=GPU,
                pci_bdf="00000000:33:00.0",
                labels={"source_boot_id": "boot-a"},
            )
            for name, value in values.items()
        ],
    )


def graded(result, name):
    return next((item for item in result.findings if item.canonical_name == name), None)


@pytest.mark.parametrize(
    "name", [MEMORY_CHANNEL_REPAIR_PENDING, MEMORY_TPC_REPAIR_PENDING]
)
@pytest.mark.parametrize("action", [{"gpu_recovery_action": 0}, {}])
def test_a_pending_repair_without_a_recovery_action_is_an_advisory(name, action):
    finding = graded(GpuMetricsService().ingest(batch(0, {name: 1, **action})), name)

    assert (finding.severity.value, finding.automatic_action) == (
        "WARNING",
        RecoveryAction.ESCALATE_OPERATOR.value,
    )
    assert finding.official_action == "REBOOT_IN_SERVICE_WINDOW"
    assert name in ADVISORY_GPU_FINDING_NAMES and not blocks_validation(finding), (
        "only the reboot clears a pending repair: no validation can re-judge it"
    )
    assert ("not reported" in finding.reason) is (not action)


@pytest.mark.parametrize(
    ("value", "automatic"),
    [(2, RecoveryAction.REBOOT_NODE), (1, RecoveryAction.RESET_GPU)],
)
def test_a_pending_repair_defers_to_the_recovery_action(value, automatic):
    result = GpuMetricsService().ingest(
        batch(0, {MEMORY_CHANNEL_REPAIR_PENDING: 1, "gpu_recovery_action": value})
    )
    repair = graded(result, MEMORY_CHANNEL_REPAIR_PENDING)
    action = graded(result, "gpu_recovery_action")

    assert repair.automatic_action == RecoveryAction.NO_ACTION.value
    assert "DCGM field 1523" in repair.reason
    assert action.automatic_action == automatic.value, "batch 233's 1523 rule decides"


def test_a_cleared_flag_clears_the_advisory():
    metrics = GpuMetricsService()
    metrics.ingest(batch(0, {MEMORY_TPC_REPAIR_PENDING: 1}))

    metrics.ingest(batch(60, {MEMORY_TPC_REPAIR_PENDING: 0}))

    assert metrics.findings(CLUSTER, NODE) == []


@pytest.mark.parametrize("product", PRODUCTS + ("NVIDIA H200",))
def test_unrepairable_memory_takes_the_rma_drain(product):
    finding = graded(
        GpuMetricsService().ingest(batch(0, {MEMORY_UNREPAIRABLE: 1}, product=product)),
        MEMORY_UNREPAIRABLE,
    )

    assert (finding.severity.value, finding.automatic_action) == (
        "CRITICAL",
        RecoveryAction.DRAIN.value,
    )
    assert finding.official_action == "RUN_FIELD_DIAGNOSTIC_FOR_RMA"
    assert finding.policy_reference.endswith("#c.DCGM_FI_DEV_MEMORY_UNREPAIRABLE"), (
        finding.policy_reference
    )


def test_unrepairable_memory_on_an_unknown_part_is_quarantined_for_review():
    finding = graded(
        GpuMetricsService().ingest(batch(0, {MEMORY_UNREPAIRABLE: 1}, product="")),
        MEMORY_UNREPAIRABLE,
    )

    assert finding.automatic_action == RecoveryAction.QUARANTINE.value
    assert finding.official_action == "REVIEW_GPU_ARCHITECTURE"


@pytest.mark.parametrize(
    "values", [{MEMORY_UNREPAIRABLE: 0}, {MEMORY_CHANNEL_REPAIR_PENDING: 0}, {}]
)
def test_no_or_absent_flags_grade_nothing(values):
    assert (
        GpuMetricsService()
        .ingest(batch(0, {"gpu_temperature_c": 40, **values}))
        .findings
        == []
    )


def xid_incident(store: InMemoryStore, xid: int) -> None:
    event = XidEvent(
        event_id=f"kernel-{xid}",
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW - timedelta(hours=2),
        event_source="KERNEL_LOG",
        xid=xid,
        gpu_uuid=GPU,
        product=PRODUCTS[0],
        driver_branch=580,
        raw_message=f"NVRM: Xid (PCI:0000:33:00): {xid}, synthetic",
    )
    store.save_xid_policy_decision(GpuFaultPolicyEngine().evaluate_xid(event))
    store.save_incident(
        fault_incident(
            f"inc-{xid}",
            event.event_id,
            cluster_id=CLUSTER,
            node_ids=[NODE],
            gpu_uuids=[GPU],
            state=IncidentState.RECOVERED,
            created_at=NOW - timedelta(hours=2),
            updated_at=NOW - timedelta(hours=2),
        )
    )


@pytest.mark.parametrize(
    ("xid", "name", "label"),
    [
        (160, MEMORY_CHANNEL_REPAIR_PENDING, "CHANNEL_RETIREMENT_EVENT"),
        (161, MEMORY_UNREPAIRABLE, "CHANNEL_RETIREMENT_FAILURE"),
    ],
)
def test_an_xid_160_or_161_of_the_gpu_is_named_as_evidence(xid, name, label):
    store = InMemoryStore()
    xid_incident(store, xid)

    finding = graded(GpuMetricsService(store=store).ingest(batch(0, {name: 1})), name)

    assert f"XID {xid} {label}" in finding.reason and f"inc-{xid}" in finding.reason


@pytest.mark.parametrize("product", PRODUCTS)
@pytest.mark.parametrize(
    ("value", "automatic"),
    [
        (5, RecoveryAction.ESCALATE_OPERATOR),
        (6, RecoveryAction.QUARANTINE),
        (7, RecoveryAction.QUARANTINE),
    ],
)
def test_recovery_actions_5_to_7_fail_closed_on_blackwell(product, value, automatic):
    finding = graded(
        GpuMetricsService().ingest(
            batch(0, {"gpu_recovery_action": value}, product=product)
        ),
        "gpu_recovery_action",
    )

    assert (finding.severity.value, finding.automatic_action) == (
        "CRITICAL",
        automatic.value,
    )
    assert finding.automatic_action != RecoveryAction.RESET_GPU.value, (
        "6 BUS_RESET is not the product's Function Level Reset"
    )


def test_an_xid154_line_without_a_known_label_is_blocked_on_blackwell():
    raw = (
        "NVRM: Xid (PCI:0000:33:00): 154, GPU recovery action changed from 0x0 "
        "(None) to 0x9 (Synthetic Unlisted Action)"
    )
    event = XidEvent(
        event_id="kernel-154-unknown",
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW,
        xid=154,
        gpu_uuid=GPU,
        product=PRODUCTS[1],
        driver_branch=580,
        raw_message=raw,
        xid_154_action=parse_xid154_action(raw),
    )

    decision = GpuFaultPolicyEngine().evaluate_xid(event, xid154_window_closed=True)

    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE


def test_a_wedged_driver_backs_the_probe_off_like_the_temperature_probe():
    """``nvidia-smi -q -x`` hangs 15 s on a wedged driver: after three failures
    the wait doubles per failure (capped at an hour); a success restores it."""

    probe = SmiHealthProbe(interval_seconds=60, logger=logging.getLogger("t"))
    wedged = Runner(error=subprocess.TimeoutExpired(SMI_XML_QUERY, 15))
    attempts = []
    for second in range(0, 6 * 3600, 15):
        before = len(wedged.calls)
        probe.refresh(NOW + timedelta(seconds=second), wedged)
        if len(wedged.calls) > before:
            attempts.append(second)

    gaps = [later - earlier for earlier, later in zip(attempts, attempts[1:])]
    assert gaps[:2] == [60, 60], gaps
    assert gaps[2:6] == [120, 240, 480, 960], gaps
    assert max(gaps) == 3600, "the wait is capped at an hour"
    healthy = Runner(completed(smi_log(gpu_element(GPU, 3, LIVE_H200_REPAIR))))
    probe.next_at = None
    probe.refresh(NOW + timedelta(hours=7), healthy)
    assert probe.failures == 0
    assert probe.next_at == NOW + timedelta(hours=7, seconds=60)
