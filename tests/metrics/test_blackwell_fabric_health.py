"""Batch 235: NVLink fabric health, DCGM fields 174/175 and the nvidia-smi XML.

The decode is NVML v615's bit layout (``nvidia-ml-py`` 13.615.71); NOT_SUPPORTED
-- the live H200 reading 0/0, the XML sub-tags all ``N/A`` -- grades nothing.
Fixtures are synthetic, derived from NVIDIA documentation (no B-series node):
healthy mask 0x11AA, route unhealthy 0x119A, degraded bandwidth 0x11A9, H200 0,
summaries 0-3; the XML strings are nvidia-smi(1)'s documented values. Nothing
here asserts that an HGX B200/B300 really reports the mask, or NVL5 Fabric
Manager timing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app.ingest.gpu_findings import gpu_node_health_findings
from gpu_fault.app.ingest.node_health import OPERATOR_REVIEW_METRICS
from gpu_fault.collectors.gpu.dcgm import STATE_CHANGE_REASONS, DcgmMetricsCollector
from gpu_fault.collectors.gpu.smi_health import parse_smi_health
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.gpu_fabric_health import (
    DEGRADED,
    FABRIC_HEALTH,
    FABRIC_HEALTH_HELD_SECONDS,
    FABRIC_HEALTH_MASK,
    FABRIC_HEALTH_SUMMARY,
    FABRIC_HEALTH_XML,
    MASK_SLOTS,
    RECOVERY,
    UNHEALTHY,
    decode_mask,
    fabric_health_state,
    fabric_validation,
)
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
)
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.families.conflicts import (
    FABRIC_MANAGER_OWNED_OPERATIONS,
    INVENTORY_FINDING_VALIDATIONS,
    TRANSIENT_INVENTORY_FINDINGS,
)
from gpu_fault.store import InMemoryStore
from tests._builders import (
    build_context,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
CLUSTER = "b300-cluster"
NODE = "p6-b300-node"
G0, G1 = "GPU-b300-0", "GPU-b300-1"
B300 = "NVIDIA B300 SXM6 AC"
HEALTHY_MASK = 0x11AA
ROUTE_UNHEALTHY_MASK = 0x119A
DEGRADED_BW_MASK = 0x11A9
#: Route recovery TRUE (shift 2) on the healthy fixture.
ROUTE_RECOVERY_MASK = (HEALTHY_MASK & ~(0x3 << 2)) | (1 << 2)
#: Incorrect configuration INSUFFICIENT_NVLINKS (5, shift 8).
INSUFFICIENT_NVLINKS_MASK = (HEALTHY_MASK & ~(0xF << 8)) | (5 << 8)


# --- the decode ----------------------------------------------------------------


def test_the_bit_layout_is_nvml_v615s():
    """``pynvml.py`` 13.615.71: ``NVML_GPU_FABRIC_HEALTH_MASK_SHIFT_*`` /
    ``_WIDTH_*``."""

    assert MASK_SLOTS == {
        "degraded_bw": (0, 0x3),
        "route_recovery": (2, 0x3),
        "route_unhealthy": (4, 0x3),
        "access_timeout_recovery": (6, 0x3),
        "incorrect_configuration": (8, 0xF),
        "partition_assigned": (12, 0x3),
        "gfm_state": (14, 0x3),
    }
    assert decode_mask(HEALTHY_MASK) == {
        "degraded_bw": 2,
        "route_recovery": 2,
        "route_unhealthy": 2,
        "access_timeout_recovery": 2,
        "incorrect_configuration": 1,
        "partition_assigned": 1,
        "gfm_state": 0,
    }


@pytest.mark.parametrize(
    ("mask", "summary", "state", "detail"),
    [
        (0, 0, None, None),
        (HEALTHY_MASK, 1, None, None),
        (HEALTHY_MASK, None, None, None),
        (ROUTE_UNHEALTHY_MASK, 1, UNHEALTHY, "route unhealthy"),
        (HEALTHY_MASK, 2, UNHEALTHY, "summary UNHEALTHY"),
        (0, 2, UNHEALTHY, "summary UNHEALTHY"),
        (
            INSUFFICIENT_NVLINKS_MASK,
            1,
            UNHEALTHY,
            "incorrect configuration INSUFFICIENT_NVLINKS",
        ),
        (ROUTE_RECOVERY_MASK, 1, RECOVERY, "route recovery in progress"),
        (DEGRADED_BW_MASK, 1, DEGRADED, "bandwidth degraded"),
        (HEALTHY_MASK, 3, DEGRADED, "summary LIMITED_CAPACITY"),
        # Partition and global Fabric Manager slots grade nothing on HGX.
        (HEALTHY_MASK | (2 << 14), 1, None, None),
        ((HEALTHY_MASK & ~(0x3 << 12)) | (2 << 12), 1, None, None),
        # Bits NVML may add later are ignored; a value that is no mask too.
        (HEALTHY_MASK | (1 << 20), 1, None, None),
        (1.5, 0, None, None),
    ],
)
def test_the_mask_and_summary_decode(mask, summary, state, detail):
    graded, details = fabric_health_state(mask, summary)

    assert graded == state
    assert (detail in details) if detail else details == []


# --- the node finding -----------------------------------------------------------


def health(
    seconds: float,
    readings: dict[str, tuple[float, float]],
    *,
    held: float | None = None,
    product: str = B300,
    xml: dict[str, dict[str, str]] | None = None,
) -> GpuMetricBatch:
    samples = []
    for gpu, (mask, summary) in readings.items():
        common = {
            "gpu_uuid": gpu,
            "pci_bdf": f"00000000:2{gpu[-1]}:00.0",
            "labels": {"modelName": product, "source_boot_id": "boot-a"},
        }
        samples += [
            GpuMetricSample(
                metric_name="DCGM_FI_DEV_FABRIC_HEALTH_MASK",
                canonical_name=FABRIC_HEALTH_MASK,
                value=mask,
                **common,
            ),
            GpuMetricSample(
                metric_name="DCGM_FI_DEV_FABRIC_HEALTH_SUMMARY",
                canonical_name=FABRIC_HEALTH_SUMMARY,
                value=summary,
                **common,
            ),
        ]
        if held is not None and fabric_health_state(mask, summary)[0] is not None:
            samples.append(
                GpuMetricSample(
                    metric_name="gpu_fault_fabric_health_held_seconds",
                    canonical_name=FABRIC_HEALTH_HELD_SECONDS,
                    value=held,
                    **common,
                )
            )
        if xml and gpu in xml:
            samples.append(
                GpuMetricSample(
                    metric_name="nvidia_smi_fabric_health_xml",
                    canonical_name=FABRIC_HEALTH_XML,
                    value=2,
                    gpu_uuid=gpu,
                    labels={f"fabric_health_{k}": v for k, v in xml[gpu].items()},
                )
            )
    return GpuMetricBatch(
        batch_id=f"fh-{seconds}",
        cluster_id=CLUSTER,
        node_id=NODE,
        product=product,
        observed_at=NOW + timedelta(seconds=seconds),
        source=GpuMetricSource.DCGM_EXPORTER,
        runtime_profile_version="simulated-v1",
        samples=samples,
    )


def node_finding(result):
    return next(
        (item for item in result.findings if item.canonical_name == FABRIC_HEALTH), None
    )


@pytest.mark.parametrize("product", ["NVIDIA H200", B300, "NVIDIA B200"])
def test_not_supported_grades_nothing(product):
    """The live H200 reading, 0 / 0 on every GPU."""

    result = GpuMetricsService().ingest(
        health(0, {G0: (0, 0), G1: (0, 0)}, product=product)
    )

    assert result.findings == []


def test_a_healthy_fabric_grades_nothing():
    result = GpuMetricsService().ingest(health(0, {G0: (HEALTHY_MASK, 1)}))

    assert result.findings == []


def test_an_unhealthy_route_is_evidence_until_it_held_a_minute():
    finding = node_finding(
        GpuMetricsService().ingest(
            health(0, {G0: (ROUTE_UNHEALTHY_MASK, 1), G1: (HEALTHY_MASK, 1)}, held=0)
        )
    )

    assert (finding.severity.value, finding.automatic_action) == (
        "WARNING",
        RecoveryAction.NO_ACTION.value,
    )
    assert finding.threshold_source == "NVIDIA_FABRIC_HEALTH_PENDING"
    assert gpu_node_health_findings(build_context(), [finding]) == []


@pytest.mark.parametrize(
    ("mask", "summary"),
    [(ROUTE_UNHEALTHY_MASK, 1), (HEALTHY_MASK, 2), (INSUFFICIENT_NVLINKS_MASK, 1)],
)
def test_a_confirmed_unhealthy_fabric_takes_the_full_reset(mask, summary):
    metrics = GpuMetricsService()
    metrics.ingest(health(0, {G0: (mask, summary)}, held=0))

    result = metrics.ingest(health(75, {G0: (mask, summary)}, held=75))
    finding = node_finding(result)

    assert (finding.severity.value, finding.automatic_action) == (
        "CRITICAL",
        RecoveryAction.RESET_GPU.value,
    )
    assert finding.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES", (
        "the shape of gpu_fabric_manager.fabric_manager_finding"
    )
    assert finding.gpu_uuid is None and finding.affected_gpu_uuids == [G0], (
        "the node's whole fabric (one GPU here)"
    )
    assert finding in result.new_findings


def store_with_window(operation, started, ended) -> InMemoryStore:
    store = InMemoryStore()
    store.save_workflow(
        workflow_request(
            "wf-fm",
            "inc-fm",
            official_steps=[workflow_step(operation, node_ids=[NODE])],
            step_executions=[
                workflow_step_execution(
                    0,
                    operation,
                    WorkflowStepStatus.SUCCEEDED,
                    started_at=started,
                    updated_at=ended,
                )
            ],
        )
    )
    store.save_incident(
        fault_incident(
            "inc-fm",
            "event-fm",
            cluster_id=CLUSTER,
            node_ids=[NODE],
            workflow_request_id="wf-fm",
            state=IncidentState.RECOVERED,
            created_at=started,
            updated_at=ended,
        )
    )
    return store


def test_a_product_fabric_window_keeps_it_evidence():
    store = store_with_window(
        WorkflowOperation.RESTART_FABRIC_MANAGER,
        NOW + timedelta(seconds=30),
        NOW + timedelta(seconds=60),
    )

    finding = node_finding(
        GpuMetricsService(store=store).ingest(
            health(90, {G0: (ROUTE_UNHEALTHY_MASK, 2)}, held=90)
        )
    )

    assert finding.automatic_action == RecoveryAction.NO_ACTION.value
    assert "RESTART_FABRIC_MANAGER" in finding.reason


def test_a_state_held_long_after_a_window_is_confirmed():
    store = store_with_window(
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        NOW - timedelta(seconds=400),
        NOW - timedelta(seconds=300),
    )

    finding = node_finding(
        GpuMetricsService(store=store).ingest(
            health(0, {G0: (ROUTE_UNHEALTHY_MASK, 2)}, held=600)
        )
    )

    assert finding.automatic_action == RecoveryAction.RESET_GPU.value


def test_route_recovery_is_evidence_and_critical_after_five_minutes():
    metrics = GpuMetricsService()

    early = node_finding(
        metrics.ingest(health(0, {G0: (ROUTE_RECOVERY_MASK, 1)}, held=120))
    )
    stuck = node_finding(
        metrics.ingest(health(180, {G0: (ROUTE_RECOVERY_MASK, 1)}, held=300))
    )

    assert early.automatic_action == RecoveryAction.NO_ACTION.value
    assert (stuck.severity.value, stuck.automatic_action) == (
        "CRITICAL",
        RecoveryAction.RESET_GPU.value,
    )


@pytest.mark.parametrize(
    ("mask", "summary"), [(DEGRADED_BW_MASK, 1), (HEALTHY_MASK, 3)]
)
def test_degraded_capacity_is_a_diagnostic_with_an_operator_notice(mask, summary):
    finding = node_finding(GpuMetricsService().ingest(health(0, {G0: (mask, summary)})))

    assert (finding.severity.value, finding.automatic_action) == (
        "WARNING",
        RecoveryAction.RUN_DIAGNOSTICS.value,
    )
    assert FABRIC_HEALTH in OPERATOR_REVIEW_METRICS
    (node,) = gpu_node_health_findings(build_context(), [finding])
    assert node.metric_name == FABRIC_HEALTH


def test_the_finding_clears_once_the_fabric_is_healthy():
    metrics = GpuMetricsService()
    metrics.ingest(health(0, {G0: (DEGRADED_BW_MASK, 3)}))

    metrics.ingest(health(15, {G0: (HEALTHY_MASK, 1)}))

    assert metrics.findings(CLUSTER, NODE) == []


@pytest.mark.parametrize("product", ["NVIDIA L40S", "NVIDIA GB200", "NVIDIA H100 PCIe"])
def test_parts_without_a_local_nvswitch_fabric_are_not_graded(product):
    result = GpuMetricsService().ingest(
        health(0, {G0: (ROUTE_UNHEALTHY_MASK, 2)}, held=600, product=product)
    )

    assert node_finding(result) is None


def test_the_xml_reading_is_evidence_in_the_reason():
    finding = node_finding(
        GpuMetricsService().ingest(
            health(
                0,
                {G0: (ROUTE_UNHEALTHY_MASK, 2)},
                held=90,
                xml={G0: {"summary": "Unhealthy", "route_unhealthy": "True"}},
            )
        )
    )

    assert "nvidia-smi summary=Unhealthy, route_unhealthy=True" in finding.reason


def test_a_product_window_absorbs_the_finding_into_its_own_validation():
    assert (
        TRANSIENT_INVENTORY_FINDINGS[FABRIC_HEALTH] == FABRIC_MANAGER_OWNED_OPERATIONS
    )
    assert INVENTORY_FINDING_VALIDATIONS[FABRIC_HEALTH] == frozenset(
        {WorkflowOperation.VALIDATE_GPU}
    )


# --- VALIDATE_GPU ----------------------------------------------------------------


def observations(summary: float, *, xml: dict[str, str] | None = None):
    common = {"gpu_uuid": G0, "labels": {"modelName": B300}}
    items = [
        GpuMetricSample(
            metric_name="DCGM_FI_DEV_FABRIC_MANAGER_STATUS",
            canonical_name="fabric_manager_status",
            value=3,
            **common,
        ),
        GpuMetricSample(
            metric_name="DCGM_FI_DEV_FABRIC_HEALTH_SUMMARY",
            canonical_name=FABRIC_HEALTH_SUMMARY,
            value=summary,
            **common,
        ),
    ]
    if xml:
        items.append(
            GpuMetricSample(
                metric_name="nvidia_smi_fabric_health_xml",
                canonical_name=FABRIC_HEALTH_XML,
                value=summary,
                gpu_uuid=G0,
                labels={f"fabric_health_{k}": v for k, v in xml.items()},
            )
        )
    return [(item, NOW) for item in items]


@pytest.mark.parametrize("summary", [0, 1])
def test_validation_accepts_a_healthy_or_unsupported_summary(summary):
    assert fabric_validation(observations(summary)) == (1, {}, None)


@pytest.mark.parametrize(
    ("summary", "name"), [(2, "UNHEALTHY"), (3, "LIMITED_CAPACITY")]
)
def test_validation_fails_any_other_summary_with_the_xml_text(summary, name):
    graded, training, failure = fabric_validation(
        observations(summary, xml={"summary": "Unhealthy", "bandwidth": "Degraded"})
    )

    assert (graded, training) == (1, {})
    assert failure == (
        f"nvlink_fabric_health_not_healthy({G0}={name} "
        "[nvidia-smi summary=Unhealthy, bandwidth=Degraded])"
    )


def test_validation_reports_a_fabric_manager_failure_and_the_health_together():
    items = observations(2)
    failed = (items[0][0].model_copy(update={"value": 4}), NOW)

    _graded, _training, failure = fabric_validation([failed, *items[1:]])

    assert failure.startswith("nvlink_fabric_not_trained(FAILURE"), failure
    assert failure.endswith(f"; nvlink_fabric_health_not_healthy({G0}=UNHEALTHY)"), (
        failure
    )


# --- the collector -----------------------------------------------------------------

SUMMARY_SECONDS = 86_400
START = next_stable_phase(
    datetime(2026, 11, 1, 12, tzinfo=timezone.utc),
    cluster_id=CLUSTER,
    node_id=NODE,
    channel="gpu-metrics",
    interval_seconds=SUMMARY_SECONDS,
) + timedelta(seconds=1)


class Sink:
    def __init__(self) -> None:
        self.service = GpuMetricsService()
        self.batches: list[GpuMetricBatch] = []

    def post(self, path: str, payload: dict) -> dict:
        batch = GpuMetricBatch.model_validate(payload)
        self.batches.append(batch)
        return self.service.ingest(batch).model_dump(mode="json")


def exporter(mask: int, summary: int) -> str:
    labels = f'gpu="0",UUID="{G0}",pci_bus_id="00000000:20:00.0",modelName="{B300}"'
    lines = {
        "DCGM_FI_DEV_GPU_TEMP": 40,
        "DCGM_FI_DEV_POWER_MGMT_LIMIT": 1100,
        "DCGM_FI_DEV_ECC_CURRENT": 1,
        "DCGM_FI_DEV_ECC_PENDING": 1,
        "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": 0,
        "DCGM_FI_DEV_ROW_REMAP_FAILURE": 0,
        "DCGM_FI_DEV_ROW_REMAP_PENDING": 0,
        "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": 0,
        "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": 0,
        "DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS": 0,
        "DCGM_FI_DEV_FABRIC_HEALTH_MASK": mask,
        "DCGM_FI_DEV_FABRIC_HEALTH_SUMMARY": summary,
    }
    return "".join(f"{name}{{{labels}}} {value}\n" for name, value in lines.items())


def test_the_collector_delivers_the_change_and_the_confirmation():
    sink = Sink()
    collector = DcgmMetricsCollector(
        sink,
        CollectorContext(cluster_id=CLUSTER),
        node_id=NODE,
        health_summary_seconds=SUMMARY_SECONDS,
        edge_confirmation_samples=3,
    )
    at = START
    for _tick in range(2):
        collector.collect_text(exporter(HEALTHY_MASK, 1), observed_at=at)
        at += timedelta(seconds=15)
    for _tick in range(10):
        collector.collect_text(exporter(ROUTE_UNHEALTHY_MASK, 2), observed_at=at)
        at += timedelta(seconds=15)

    reasons = [batch.edge_filter_reasons for batch in sink.batches]
    assert reasons[:2] == [["initial-baseline"], ["fabric-health-changed"]]
    assert ["candidate-confirmed"] in reasons[2:], reasons
    held = [
        sample.value
        for sample in sink.batches[-1].samples
        if sample.canonical_name == FABRIC_HEALTH_HELD_SECONDS
    ]
    assert held and held[0] >= 60
    (finding,) = sink.service.findings(CLUSTER, NODE)
    assert finding.canonical_name == FABRIC_HEALTH
    assert finding.automatic_action == RecoveryAction.RESET_GPU.value
    for name in (FABRIC_HEALTH_MASK, FABRIC_HEALTH_SUMMARY, FABRIC_HEALTH_XML):
        assert STATE_CHANGE_REASONS[name] == "fabric-health-changed"


# --- the nvidia-smi XML ------------------------------------------------------------


def smi_xml(health_tags: str, *, repair: str = "") -> str:
    return (
        '<?xml version="1.0" ?><nvidia_smi_log><driver_version>580.178.04'
        '</driver_version><gpu id="00000000:20:00.0"><product_name>'
        f"{B300}</product_name><uuid>{G0}</uuid><minor_number>0</minor_number>"
        "<pci><pci_bus_id>00000000:20:00.0</pci_bus_id></pci>"
        f"<ecc_errors>{repair}</ecc_errors>"
        "<fabric><state>Completed</state><status>Success</status>"
        f"<health>{health_tags}</health></fabric></gpu></nvidia_smi_log>"
    )


#: The live H200 R580 health sub-tags, every one N/A.
LIVE_H200_HEALTH = "".join(
    f"<{tag}>N/A</{tag}>"
    for tag in (
        "summary",
        "bandwidth",
        "route_recovery_in_progress",
        "route_unhealthy",
        "access_timeout_recovery",
        "incorrect_configuration",
        "partition_assigned",
    )
)


def test_the_live_h200_health_tags_publish_nothing():
    assert [
        sample
        for sample in parse_smi_health(smi_xml(LIVE_H200_HEALTH))
        if sample.canonical_name == FABRIC_HEALTH_XML
    ] == []


def test_documented_health_strings_become_one_evidence_sample():
    tags = (
        "<summary>Unhealthy</summary><bandwidth>Degraded</bandwidth>"
        "<route_recovery_in_progress>False</route_recovery_in_progress>"
        "<route_unhealthy>True</route_unhealthy>"
        "<access_timeout_recovery>N/A</access_timeout_recovery>"
        "<incorrect_configuration>None</incorrect_configuration>"
    )

    (sample,) = [
        item
        for item in parse_smi_health(smi_xml(tags))
        if item.canonical_name == FABRIC_HEALTH_XML
    ]

    assert (sample.value, sample.gpu_uuid) == (2.0, G0)
    assert sample.labels["fabric_health_summary"] == "Unhealthy"
    assert sample.labels["fabric_health_route_unhealthy"] == "True"
    assert "fabric_health_access_timeout_recovery" not in sample.labels


def test_an_undocumented_summary_string_is_no_grade():
    (sample,) = [
        item
        for item in parse_smi_health(smi_xml("<summary>Mostly fine</summary>"))
        if item.canonical_name == FABRIC_HEALTH_XML
    ]

    assert sample.value == 0.0
