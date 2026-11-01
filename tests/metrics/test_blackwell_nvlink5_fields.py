"""Batch 235: HGX B200/B300 NVLink completeness and the NVLink5 counter rules.

NVML does not report the data-link counters 497-499 on Blackwell (NVML API
Reference v615, ``NVML_FI_DEV_NVLINK_ERROR_DL_CRC``: "This is unsupported for
Blackwell+ Please use NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_*"), so the link
recovery counters 1211-1213 answer for the NVLink completeness groups, and the
NVLink5 counters 1204-1209, 1211-1214, 1219, 1220 grade on their increments.

Synthetic fixtures derived from NVIDIA documentation, never a live reading (no
B-series hardware): the driver's product strings ``NVIDIA B200`` / ``NVIDIA
B300 SXM6 AC``; a B-series scrape without 409-439, 497-499 and 390-392; healthy
NVLink5 counters at 0; the fault deltas of the Blackwell audit (1212 +1; 1211
+1 with 1213 +1; 1214 +5; an aggregate decrease). Nothing here asserts a real
B-series value, the healthy magnitude of a counter or exporter link labels.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.app.ingest.gpu_findings import gpu_node_health_findings
from gpu_fault.collectors.gpu.completeness import completeness_errors
from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS, parse_dcgm_samples
from gpu_fault.dcgm_fields import (
    DCGM_CORE_CANONICAL_FIELDS,
    DCGM_EXPORTER_FIELD_GROUPS,
    NVLINK5_LINK_RECOVERY_CANONICAL,
    NVLINK5_LINK_RECOVERY_EXPORTER_NAMES,
    dcgm_exporter_required_expression,
    missing_dcgm_metric_groups,
    missing_fabric_metric_groups,
)
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
)
from gpu_fault.gpu_nvlink5_policy import (
    NVLINK5_COUNTERS,
    NVLINK5_ERROR_COUNTERS,
    NVLINK5_FIELDS,
    NVLINK5_RECOVERY_EVENTS,
    NVLINK5_RECOVERY_FAILED,
    NVLINK5_RECOVERY_SUCCESSFUL,
)
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStepStatus,
)
from gpu_fault.policy import GpuFaultPolicyEngine, XidEvent
from gpu_fault.store import InMemoryStore
from tests._builders import (
    build_context,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
CLUSTER = "b200-cluster"
NODE = "p6-b200-node"
G0 = "GPU-b200-0"
G1 = "GPU-b200-1"
PRODUCTS = ("NVIDIA B200", "NVIDIA B300 SXM6 AC")
#: The Hopper NVLink counters a B-series scrape never carries (and 390-392).
HOPPER_ONLY = {
    "nvlink_crc_flit_error_total",
    "nvlink_crc_data_error_total",
    "nvlink_replay_error_total",
    "nvlink_recovery_error_total",
    "nvlink_crc_aggregate_error_total",
    "nvlink_recovery_aggregate_error_total",
    "nvlink_replay_aggregate_error_total",
    "retired_pages_sbe_total",
    "retired_pages_dbe_total",
    "retired_pages_pending",
}


def blackwell_names(*nvlink5: str) -> set[str]:
    return set(DCGM_CORE_CANONICAL_FIELDS) | set(nvlink5)


# --- the completeness blocker ------------------------------------------------


@pytest.mark.parametrize("member", NVLINK5_LINK_RECOVERY_CANONICAL)
def test_any_one_link_recovery_counter_completes_a_blackwell_scrape(member):
    names = blackwell_names(member)

    assert not names & HOPPER_ONLY
    assert missing_dcgm_metric_groups(names) == []
    assert missing_fabric_metric_groups(names) == [], (
        "VALIDATE_FABRIC must not wait for data-link counters NVML never reports "
        "on Blackwell"
    )


def test_a_blackwell_scrape_without_them_is_incomplete_under_the_same_names():
    assert missing_dcgm_metric_groups(blackwell_names()) == [
        "nvlink_crc",
        "nvlink_recovery",
        "nvlink_replay",
    ]


@pytest.mark.parametrize(
    ("names", "missing"),
    [
        (
            {
                "nvlink_crc_aggregate_error_total",
                "nvlink_recovery_aggregate_error_total",
                "nvlink_replay_aggregate_error_total",
            },
            [],
        ),
        (
            {
                "nvlink_crc_flit_error_total",
                "nvlink_recovery_error_total",
                "nvlink_replay_error_total",
            },
            [],
        ),
        ({"nvlink_crc_aggregate_error_total"}, ["nvlink_recovery", "nvlink_replay"]),
        (set(), ["nvlink_crc", "nvlink_recovery", "nvlink_replay"]),
    ],
)
def test_hopper_completeness_is_unchanged(names, missing):
    assert missing_dcgm_metric_groups(set(DCGM_CORE_CANONICAL_FIELDS) | names) == (
        missing
    )


def scrape_line(name: str, gpu: str, value: float, product: str) -> str:
    index = "0" if gpu == G0 else "1"
    labels = (
        f'gpu="{index}",UUID="{gpu}",pci_bus_id="00000000:1{index}:00.0",'
        f'modelName="{product}"'
    )
    return f"{name}{{{labels}}} {value}\n"


def blackwell_scrape(product: str, *, recovery: bool = True) -> str:
    """Every counters-CSV field a B-series GPU serves; Hopper-only ones absent."""

    text = ""
    for name, (canonical, _unit) in DCGM_METRICS.items():
        if canonical in HOPPER_ONLY or name not in CSV_NAMES:
            continue
        if not recovery and canonical in NVLINK5_LINK_RECOVERY_CANONICAL:
            continue
        for gpu in (G0, G1):
            text += scrape_line(name, gpu, 0, product)
    return text


CSV_NAMES = {
    line.split(",")[0].strip()
    for line in (ROOT / "deploy/dataplane/dcgm-counters.csv")
    .read_text(encoding="utf-8")
    .splitlines()
    if line and not line.startswith("#")
}


@pytest.mark.parametrize("product", PRODUCTS)
def test_the_collector_finds_a_blackwell_scrape_complete(product):
    samples = parse_dcgm_samples(blackwell_scrape(product))

    assert completeness_errors(samples) == []
    assert {sample.canonical_name for sample in samples} >= NVLINK5_COUNTERS


def test_the_collector_names_the_gap_when_the_recovery_counters_are_absent():
    samples = parse_dcgm_samples(blackwell_scrape(PRODUCTS[0], recovery=False))

    assert completeness_errors(samples) == [
        "DCGM exporter is missing required fields: "
        "nvlink_crc,nvlink_recovery,nvlink_replay"
    ], "the same message the H200 path has always written"


def test_every_nvlink_exporter_group_accepts_the_link_recovery_names():
    nvlink_groups = [
        group for group in DCGM_EXPORTER_FIELD_GROUPS if "NVLINK" in group[0]
    ]

    assert len(nvlink_groups) == 3
    for group in nvlink_groups:
        assert set(NVLINK5_LINK_RECOVERY_EXPORTER_NAMES) <= set(group), group
        assert group[:2] in {
            (
                "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL",
                "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC",
            ),
            (
                "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL",
                "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY",
            ),
            (
                "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL",
                "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY",
            ),
        }, "the Hopper names keep their place in front"
    for name in NVLINK5_LINK_RECOVERY_EXPORTER_NAMES:
        assert DCGM_METRICS[name][0] in NVLINK5_LINK_RECOVERY_CANONICAL, name


def test_the_deploy_expression_still_parses_in_deploy_sh():
    """``deploy/hyperpod/deploy.sh`` validates the expression with a bash regex
    (``GPU_FAULT_DCGM_REQUIRED_METRICS`` defaults to it); the line runs here as
    the script spells it."""

    script = (ROOT / "deploy/hyperpod/deploy.sh").read_text(encoding="utf-8")
    (check,) = [
        line
        for line in script.splitlines()
        if line.startswith('[[ "${DCGM_REQUIRED_METRICS}" =~ ')
    ]
    harness = f'DCGM_REQUIRED_METRICS="$1"\n{check.removesuffix(" || {")} || exit 3\n'
    expression = dcgm_exporter_required_expression()
    result = subprocess.run(
        ["bash", "-c", harness, "deploy-check", expression],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (check, result.stderr)
    for name in NVLINK5_LINK_RECOVERY_EXPORTER_NAMES:
        assert expression.count(f"|{name}") == 3, name


# --- NVLink5 counter grading ---------------------------------------------------


def batch(
    seconds: float,
    values: dict[str, float],
    *,
    gpu: str = G0,
    product: str = PRODUCTS[0],
    extra: dict[str, dict[str, float]] | None = None,
) -> GpuMetricBatch:
    per_gpu = {gpu: values, **(extra or {})}
    return GpuMetricBatch(
        batch_id=f"bw-{seconds}",
        cluster_id=CLUSTER,
        node_id=NODE,
        product=product,
        observed_at=NOW + timedelta(seconds=seconds),
        source=GpuMetricSource.DCGM_EXPORTER,
        runtime_profile_version="simulated-v1",
        samples=[
            GpuMetricSample(
                metric_name=NVLINK5_FIELDS[name][1] if name in NVLINK5_FIELDS else name,
                canonical_name=name,
                value=value,
                gpu_uuid=device,
                pci_bdf=f"00000000:1{device[-1]}:00.0",
                labels={"source_boot_id": "boot-a", "modelName": product},
            )
            for device, readings in per_gpu.items()
            for name, value in readings.items()
        ],
    )


def finding_of(result, name, gpu=G0):
    return next(
        (
            item
            for item in result.findings
            if item.canonical_name == name and item.gpu_uuid == gpu
        ),
        None,
    )


def service(store: InMemoryStore | None = None) -> GpuMetricsService:
    return GpuMetricsService(store=store or InMemoryStore())


def test_a_counter_that_already_carries_a_count_is_a_baseline():
    metrics = service()

    first = metrics.ingest(batch(0, {NVLINK5_RECOVERY_FAILED: 5}))

    assert first.findings == [], "the reading is never graded, only an increment"


@pytest.mark.parametrize("product", PRODUCTS)
def test_a_failed_link_recovery_is_a_critical_drain(product):
    metrics = service()
    metrics.ingest(batch(0, {NVLINK5_RECOVERY_FAILED: 0}, product=product))

    result = metrics.ingest(batch(15, {NVLINK5_RECOVERY_FAILED: 1}, product=product))
    finding = finding_of(result, NVLINK5_RECOVERY_FAILED)

    assert (finding.severity.value, finding.automatic_action) == (
        "CRITICAL",
        RecoveryAction.DRAIN.value,
    )
    assert finding.official_action == "RUN_FIELD_DIAGNOSTIC"
    assert finding.policy_source == "NVIDIA_DCGM_HEALTH"
    assert finding.policy_reference.endswith(
        "#c.DCGM_FI_DEV_NVLINK_RECOVERY_FAILED_TOTAL"
    ), finding.policy_reference
    assert "DCGM field 1212" in finding.reason
    assert finding in result.new_findings


def h200_grade(name: str = "nvlink_crc_aggregate_error_total"):
    """The NVLink family rule's grade of a Hopper DL aggregate increment."""

    metrics = service()
    metrics.ingest(batch(0, {name: 0}, product="NVIDIA H200"))
    result = metrics.ingest(batch(15, {name: 1}, product="NVIDIA H200"))
    finding = finding_of(result, name)
    return (
        finding.severity,
        finding.automatic_action,
        finding.policy_source,
        finding.official_action,
        finding.reason,
    )


@pytest.mark.parametrize(
    "name",
    sorted(
        NVLINK5_ERROR_COUNTERS | {NVLINK5_RECOVERY_SUCCESSFUL, NVLINK5_RECOVERY_EVENTS}
    ),
)
def test_every_other_nvlink5_counter_takes_the_nvlink_family_rule(name):
    metrics = service()
    metrics.ingest(batch(0, {name: 0}))

    result = metrics.ingest(
        batch(15, {name: 5 if name.endswith("symbol_error_total") else 1})
    )
    finding = finding_of(result, name)

    assert (
        finding.severity,
        finding.automatic_action,
        finding.policy_source,
        finding.official_action,
        finding.reason,
    ) == h200_grade(), name
    assert finding.policy_reference.startswith(
        "https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html#c."
    ), finding.policy_reference


def test_a_successful_recovery_counts_once_not_as_two_error_classes():
    """One successful recovery increments 1211 and 1213: two WARNINGs, but one
    NVLink error class -- no NVLINK_LINK_DEGRADATION of its own."""

    metrics = service()
    metrics.ingest(
        batch(0, {NVLINK5_RECOVERY_SUCCESSFUL: 0, NVLINK5_RECOVERY_EVENTS: 0})
    )

    result = metrics.ingest(
        batch(15, {NVLINK5_RECOVERY_SUCCESSFUL: 1, NVLINK5_RECOVERY_EVENTS: 1})
    )

    assert {item.canonical_name for item in result.new_findings} == {
        NVLINK5_RECOVERY_SUCCESSFUL,
        NVLINK5_RECOVERY_EVENTS,
    }
    assert result.composite_findings == []


def test_symbol_errors_beside_a_recovery_are_a_link_degradation():
    metrics = service()
    metrics.ingest(
        batch(0, {NVLINK5_RECOVERY_EVENTS: 0, "nvlink_rx_symbol_error_total": 0})
    )

    result = metrics.ingest(
        batch(15, {NVLINK5_RECOVERY_EVENTS: 1, "nvlink_rx_symbol_error_total": 5})
    )
    (composite,) = result.new_composite_findings

    assert composite.correlation_rule_id == "NVLINK_LINK_DEGRADATION"
    assert (composite.severity.value, composite.automatic_action) == (
        "WARNING",
        RecoveryAction.RUN_DIAGNOSTICS.value,
    )


def test_one_class_on_two_gpus_is_the_multi_gpu_composite():
    metrics = service()
    metrics.ingest(
        batch(0, {NVLINK5_RECOVERY_EVENTS: 0}, extra={G1: {NVLINK5_RECOVERY_EVENTS: 0}})
    )

    result = metrics.ingest(
        batch(
            15, {NVLINK5_RECOVERY_EVENTS: 1}, extra={G1: {NVLINK5_RECOVERY_EVENTS: 1}}
        )
    )
    (composite,) = result.new_composite_findings

    assert composite.correlation_rule_id == "MULTI_GPU_NVLINK_FABRIC_FAILURE"
    assert composite.affected_gpu_uuids == [G0, G1]
    assert composite.automatic_action == RecoveryAction.RUN_DIAGNOSTICS.value


def test_a_composite_never_swallows_a_failed_link_recovery():
    """A WARNING composite suppresses its components' own incidents; with a
    failed recovery among them it is CRITICAL DRAIN itself."""

    metrics = service()
    metrics.ingest(
        batch(0, {NVLINK5_RECOVERY_FAILED: 0}, extra={G1: {NVLINK5_RECOVERY_EVENTS: 0}})
    )

    result = metrics.ingest(
        batch(
            15, {NVLINK5_RECOVERY_FAILED: 1}, extra={G1: {NVLINK5_RECOVERY_EVENTS: 1}}
        )
    )
    (composite,) = result.new_composite_findings

    assert composite.correlation_rule_id == "MULTI_GPU_NVLINK_FABRIC_FAILURE"
    assert (composite.severity.value, composite.automatic_action) == (
        "CRITICAL",
        RecoveryAction.DRAIN.value,
    )
    assert composite.official_action == "RUN_FIELD_DIAGNOSTIC"
    assert NVLINK5_RECOVERY_FAILED in composite.reason
    assert [item.finding_id for item in result.new_findings] == [
        composite.finding_id
    ], "the components ride on the composite's DRAIN incident"


def test_an_aggregate_decrease_rebaselines_and_grades_nothing():
    """The aggregate sums the *active* ports (DCGM Feature Overview): a link
    that left the active set lowers it."""

    metrics = service()
    metrics.ingest(batch(0, {NVLINK5_RECOVERY_EVENTS: 10}))

    dropped = metrics.ingest(batch(15, {NVLINK5_RECOVERY_EVENTS: 4}))
    still = metrics.ingest(batch(30, {NVLINK5_RECOVERY_EVENTS: 4}))
    raised = metrics.ingest(batch(45, {NVLINK5_RECOVERY_EVENTS: 5}))

    assert dropped.findings == [] and still.findings == []
    assert finding_of(raised, NVLINK5_RECOVERY_EVENTS).delta == 1.0, (
        "the next increase is measured from the decreased reading"
    )


@pytest.mark.parametrize(
    "name",
    [
        "DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER",
        "DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO",
        "DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER",
        "DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO",
        "DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_0",
        "DCGM_FI_DEV_NVLINK_FEC_HISTORY_15_TOTAL",
    ],
)
def test_ber_and_fec_history_are_never_read(name):
    assert name not in DCGM_METRICS
    assert name not in CSV_NAMES
    assert parse_dcgm_samples(scrape_line(name, G0, 7, PRODUCTS[0])) == []


# --- product windows and the XID evidence ------------------------------------


def window_workflow(
    store: InMemoryStore,
    operation: WorkflowOperation,
    started: datetime,
    ended: datetime,
    *,
    status: WorkflowStepStatus = WorkflowStepStatus.SUCCEEDED,
    state: IncidentState = IncidentState.RECOVERED,
) -> None:
    workflow = workflow_request(
        "wf-window",
        "inc-window",
        official_steps=[workflow_step(operation, node_ids=[NODE])],
        step_executions=[
            workflow_step_execution(
                0, operation, status, started_at=started, updated_at=ended
            )
        ],
    )
    store.save_workflow(workflow)
    store.save_incident(
        fault_incident(
            "inc-window",
            "event-window",
            cluster_id=CLUSTER,
            node_ids=[NODE],
            workflow_request_id="wf-window",
            state=state,
            created_at=started,
            updated_at=ended,
        )
    )


@pytest.mark.parametrize(
    "operation",
    [
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTART_FABRIC_MANAGER,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    ],
)
def test_an_increment_across_a_product_window_grades_nothing(operation):
    store = InMemoryStore()
    metrics = service(store)
    metrics.ingest(batch(0, {NVLINK5_RECOVERY_FAILED: 0}))
    window_workflow(
        store, operation, NOW + timedelta(seconds=20), NOW + timedelta(seconds=50)
    )

    result = metrics.ingest(batch(60, {NVLINK5_RECOVERY_FAILED: 2}))

    assert result.findings == [], f"{operation.value} retrained the links"


def test_a_window_still_running_suppresses_too():
    store = InMemoryStore()
    metrics = service(store)
    metrics.ingest(batch(0, {"nvlink_rx_symbol_error_total": 0}))
    window_workflow(
        store,
        WorkflowOperation.RESTART_FABRIC_MANAGER,
        NOW + timedelta(seconds=5),
        NOW + timedelta(seconds=5),
        status=WorkflowStepStatus.WAITING,
        state=IncidentState.ACTION_PENDING,
    )

    result = metrics.ingest(batch(15, {"nvlink_rx_symbol_error_total": 3}))

    assert result.findings == []


def test_an_increment_well_after_a_window_grades():
    store = InMemoryStore()
    metrics = service(store)
    window_workflow(
        store,
        WorkflowOperation.RESET_GPU,
        NOW - timedelta(seconds=300),
        NOW - timedelta(seconds=240),
    )
    metrics.ingest(batch(0, {NVLINK5_RECOVERY_FAILED: 0}))

    result = metrics.ingest(batch(15, {NVLINK5_RECOVERY_FAILED: 1}))

    assert finding_of(result, NVLINK5_RECOVERY_FAILED).automatic_action == "DRAIN"


def nvlink_xid(xid: int, seconds: float, *, gpu: str = G0) -> XidEvent:
    return XidEvent(
        event_id=f"kernel-{xid}-{gpu}",
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW + timedelta(seconds=seconds),
        event_source="KERNEL_LOG",
        xid=xid,
        gpu_uuid=gpu,
        product=PRODUCTS[0],
        driver_branch=580,
        raw_message=f"NVRM: Xid (PCI:0000:10:00): {xid}, synthetic",
    )


@pytest.mark.parametrize("xid", [144, 149, 150, 155])
def test_an_nvlink_xid_event_of_the_gpu_makes_the_counters_evidence(xid):
    store = InMemoryStore()
    store.save_xid_event_if_absent(nvlink_xid(xid, 5))
    metrics = service(store)
    metrics.ingest(batch(0, {NVLINK5_RECOVERY_FAILED: 0}))

    result = metrics.ingest(batch(15, {NVLINK5_RECOVERY_FAILED: 1}))
    finding = finding_of(result, NVLINK5_RECOVERY_FAILED)

    assert finding.automatic_action == RecoveryAction.NO_ACTION.value
    assert finding.threshold_source == "NVLINK5_XID_OWNS_GPU"
    assert f"XID {xid} event" in finding.reason
    assert gpu_node_health_findings(build_context(), result.new_findings) == [], (
        "evidence only: no second workflow beside the XID's"
    )


def test_an_xid_of_another_gpu_owns_nothing_here():
    store = InMemoryStore()
    store.save_xid_event_if_absent(nvlink_xid(149, 5, gpu=G1))
    metrics = service(store)
    metrics.ingest(batch(0, {NVLINK5_RECOVERY_FAILED: 0}))

    result = metrics.ingest(batch(15, {NVLINK5_RECOVERY_FAILED: 1}))

    assert finding_of(result, NVLINK5_RECOVERY_FAILED).automatic_action == "DRAIN"


def test_an_open_nvlink_xid_incident_keeps_the_counters_evidence():
    store = InMemoryStore()
    event = nvlink_xid(149, -600)
    decision = GpuFaultPolicyEngine().evaluate_xid(event)
    store.save_xid_policy_decision(decision)
    store.save_incident(
        fault_incident(
            "inc-xid-149",
            event.event_id,
            cluster_id=CLUSTER,
            node_ids=[NODE],
            gpu_uuids=[G0],
            state=IncidentState.SAFETY_PENDING,
            created_at=NOW - timedelta(minutes=10),
            updated_at=NOW - timedelta(minutes=10),
        )
    )
    metrics = service(store)
    metrics.ingest(batch(0, {"nvlink_rx_symbol_error_total": 0}))

    result = metrics.ingest(batch(15, {"nvlink_rx_symbol_error_total": 5}))
    finding = finding_of(result, "nvlink_rx_symbol_error_total")

    assert finding.automatic_action == RecoveryAction.NO_ACTION.value
    assert "XID 149 incident inc-xid-149" in finding.reason


def test_evidence_only_counters_form_no_composite():
    store = InMemoryStore()
    store.save_xid_event_if_absent(nvlink_xid(149, 5))
    store.save_xid_event_if_absent(nvlink_xid(149, 5, gpu=G1))
    metrics = service(store)
    metrics.ingest(
        batch(0, {NVLINK5_RECOVERY_EVENTS: 0}, extra={G1: {NVLINK5_RECOVERY_EVENTS: 0}})
    )

    result = metrics.ingest(
        batch(
            15, {NVLINK5_RECOVERY_EVENTS: 1}, extra={G1: {NVLINK5_RECOVERY_EVENTS: 1}}
        )
    )

    assert result.composite_findings == [], "the XIDs own both GPUs' recovery"


def test_a_hopper_scrape_has_no_nvlink5_field_and_nothing_new_to_grade():
    text = "".join(
        scrape_line(name, G0, 0, "NVIDIA H200")
        for name in (
            "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC",
            "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY",
            "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY",
        )
    )

    names = {sample.canonical_name for sample in parse_dcgm_samples(text)}

    assert names.isdisjoint(NVLINK5_COUNTERS), names
    assert missing_fabric_metric_groups(names) == []


def test_every_nvlink5_field_is_read_under_both_names():
    for canonical, (field, legacy, current) in NVLINK5_FIELDS.items():
        assert DCGM_METRICS[legacy] == DCGM_METRICS[current], field
        assert DCGM_METRICS[legacy][0] == canonical
        assert legacy in CSV_NAMES and current not in CSV_NAMES, field
        labels = f'gpu="0",UUID="{G0}"'
        samples = parse_dcgm_samples(
            f"{legacy}{{{labels}}} 3\n{current}{{{labels}}} 3\n"
        )
        assert [sample.metric_name for sample in samples] == [legacy], field
