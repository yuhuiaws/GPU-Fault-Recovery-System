"""Batch 233: grading of the DCGM fields the 2026-11-01 field audit added.

Field 1523 (the driver's recovery action) follows the XID 154 path for the
values it shares with it, NVIDIA's documented meaning for RECOVER_IMEX_DOMAIN,
fails closed on everything else, and is evidence only while an XID 154 owns the
GPU. The PCIe replay (202) and correctable-error (1501) rates are judged
against DCGM's expected rate of the current link (237/238), never a fleet
constant; the collector's downtrain verdict (235-238), the SRAM RMA flag (346),
remap headroom (389), the 393 absolute limit, the HW slowdown durations
(1423/1424), ECC mode (300/301) and InfoROM validity grade as the module
docstrings of ``gpu_field_rules`` and its siblings say. Every input here is a
fake batch through the real ``GpuMetricsService.ingest``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.app.ingest.gpu_findings import gpu_node_health_findings
from gpu_fault.app.ingest.node_health import OPERATOR_REVIEW_METRICS
from gpu_fault.gpu_environment_policy import (
    HW_SLOWDOWN_MASK,
    NVIDIA_SMI_HW_SLOWDOWN_QUOTE,
    NVML_HW_SLOWDOWN_QUOTE,
    NVML_UNGRADED_CLOCK_EVENT_BITS,
    clock_reason_decision,
)
from gpu_fault.gpu_field_rules import ADVISORY_GPU_FINDING_NAMES, blocks_validation
from gpu_fault.gpu_memory_policy import (
    NVIDIA_BLACKWELL_CHANNEL_REPAIR_QUOTE,
    NVIDIA_ROW_REMAP_FAILURE_BANK_QUOTE,
)
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
    GpuMetricsThresholds,
)
from gpu_fault.gpu_pcie_link_policy import (
    PCIE_ERRORS_PER_LANE_PER_MINUTE,
    correctable_rate_bound,
    dcgm_diag_aer_rate,
    expected_pcie_error_rate,
)
from gpu_fault.gpu_recovery_action_policy import (
    DCGM_HEALTH_RECOVERY_RANGE_QUOTE,
    GPU_RECOVERY_ACTION_DYNAMIC,
    NVIDIA_SMI_RESET_TYPE_QUOTE,
    NVML_ENUM_ONLY_RECOVERY_ACTIONS,
    NVML_RECOVERY_ACTION_ENUM_MEANINGS,
    NVML_RECOVERY_ACTION_ENUM_REFERENCE,
)
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.policy import GpuFaultPolicyEngine, XidEvent, parse_xid154_action
from gpu_fault.store import InMemoryStore
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
CLUSTER = "cluster-a"
NODE = "node-a"
GPU = "GPU-a"
GEN5_X16 = {"pcie_link_gen_current": 5, "pcie_link_width_current": 16}


def make_batch(
    seconds: float,
    values: dict[str, float],
    *,
    product: str = "H200",
    gpu: str = GPU,
    labels: dict[str, str] | None = None,
) -> GpuMetricBatch:
    return GpuMetricBatch(
        batch_id=f"gap-{gpu}-{seconds}",
        cluster_id=CLUSTER,
        node_id=NODE,
        product=product,
        observed_at=NOW + timedelta(seconds=seconds),
        source=GpuMetricSource.DCGM_EXPORTER,
        samples=[
            GpuMetricSample(
                metric_name=name,
                canonical_name=name,
                value=value,
                gpu_uuid=gpu,
                pci_bdf="0000:59:00.0",
                labels={"source_boot_id": "boot-a", **(labels or {})},
            )
            for name, value in values.items()
        ],
    )


def ingest(service: GpuMetricsService, seconds: float, values, **kwargs):
    return service.ingest(make_batch(seconds, values, **kwargs))


def graded(result, name: str):
    return next((item for item in result.findings if item.canonical_name == name), None)


def service(store: InMemoryStore | None = None, **thresholds: Any):
    return GpuMetricsService(
        GpuMetricsThresholds(**thresholds), store=store or InMemoryStore()
    )


# --- 1523: the driver's GPU recovery action -------------------------------

KERNEL_LABELS = {
    1: "GPU Reset Required",
    2: "Node Reboot Required",
    3: "Drain P2P",
    4: "Drain and Reset",
}


@pytest.mark.parametrize(("value", "label"), sorted(KERNEL_LABELS.items()))
def test_recovery_action_follows_the_xid154_path_for_the_same_label(value, label):
    """The XID 154 engine on the kernel line of the same NVML value decides
    the action, the official action and the grade the field takes."""

    raw = (
        "NVRM: Xid (PCI:0000:59:00): 154, GPU recovery action changed from 0x0 "
        f"(None) to 0x{value} ({label})"
    )
    event = XidEvent(
        event_id=f"kernel-{value}",
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW,
        xid=154,
        gpu_uuid=GPU,
        product="H200",
        driver_branch=580,
        cuda_version="13.0",
        workload_state=WorkloadState.ACTIVE,
        raw_message=raw,
        xid_154_action=parse_xid154_action(raw),
    )
    decision = GpuFaultPolicyEngine().evaluate_xid(event, xid154_window_closed=True)

    finding = graded(
        ingest(service(), 0, {"gpu_recovery_action": value}), "gpu_recovery_action"
    )

    assert decision.action is not None, decision.reasons
    assert finding.automatic_action == decision.action.value
    assert finding.official_action == decision.official_action
    assert finding.severity.value.lower() == decision.severity.value
    assert finding.policy_source == "NVIDIA_DCGM_GPU_RECOVERY_ACTION"


def test_recover_imex_domain_follows_nvidia_documentation_to_an_operator():
    finding = graded(
        ingest(service(), 0, {"gpu_recovery_action": 5}), "gpu_recovery_action"
    )

    assert (finding.severity.value, finding.automatic_action) == (
        "CRITICAL",
        RecoveryAction.ESCALATE_OPERATOR.value,
    )
    assert finding.official_action == "RECOVER_IMEX_DOMAIN"
    assert "IMEX domain" in finding.reason
    assert finding.policy_reference.startswith(
        "https://docs.nvidia.com/datacenter/dcgm/"
    ), finding.policy_reference


@pytest.mark.parametrize(
    ("value", "spelling"),
    [
        (6, "BUS_RESET"),
        (7, "SYSTEM_REBOOT"),
        (8, "unknown(8)"),
        (255, "unknown(255)"),
        (1.5, "unknown(1.5)"),
    ],
)
def test_recovery_actions_nvidia_does_not_document_fail_closed(value, spelling):
    finding = graded(
        ingest(service(), 0, {"gpu_recovery_action": value}), "gpu_recovery_action"
    )

    assert (finding.severity.value, finding.automatic_action) == (
        "CRITICAL",
        RecoveryAction.QUARANTINE.value,
    ), "fail closed: freeze, cordon, quarantine, an operator decides"
    assert spelling in finding.reason
    assert finding.threshold_source == "NVIDIA_GPU_RECOVERY_ACTION_UNKNOWN"


def test_bus_reset_never_takes_the_products_function_level_reset():
    """NVML v615 value 6 means "Reset the GPU’s PCIe bus."; the product's
    RESET_GPU passes no reset type to ``nvidia-smi --gpu-reset``, which is a
    Function Level Reset (pinned in
    ``tests/node_agent/test_reset_is_function_level.py``)."""

    finding = graded(
        ingest(service(), 0, {"gpu_recovery_action": 6}), "gpu_recovery_action"
    )

    assert finding.automatic_action == RecoveryAction.QUARANTINE.value
    assert finding.automatic_action != RecoveryAction.RESET_GPU.value
    assert 6 not in GPU_RECOVERY_ACTION_DYNAMIC, "no XID 154 action for a bus reset"
    assert "Function Level Reset, not a bus reset" in finding.reason
    assert '"Reset the GPU’s PCIe bus."' in finding.reason
    assert finding.policy_reference == NVML_RECOVERY_ACTION_ENUM_REFERENCE
    assert "To issue a Bus Reset, use -r bus." in NVIDIA_SMI_RESET_TYPE_QUOTE


def test_system_reboot_is_not_the_node_reboot_of_value_two():
    finding = graded(
        ingest(service(), 0, {"gpu_recovery_action": 7}), "gpu_recovery_action"
    )

    assert finding.automatic_action == RecoveryAction.QUARANTINE.value
    assert '"Reboot the system."' in finding.reason
    assert "not the node reboot of value 2" in finding.reason


def test_nvidia_documents_only_one_line_meanings_for_values_five_to_seven():
    assert NVML_RECOVERY_ACTION_ENUM_MEANINGS == {
        5: "Recover IMEX Domain.",
        6: "Reset the GPU’s PCIe bus.",
        7: "Reboot the system.",
    }
    assert NVML_ENUM_ONLY_RECOVERY_ACTIONS == frozenset({6, 7})
    assert NVML_RECOVERY_ACTION_ENUM_REFERENCE.endswith(
        "group__nvmlVgpuStructs.html#_CPPv429nvmlDeviceGpuRecoveryAction_t"
    ), "the NVML v615 enum page"
    assert DCGM_HEALTH_RECOVERY_RANGE_QUOTE == "Invalid GPU recovery action value", (
        "DCGM 4.7.0's health check ignores values above 5: no backstop for 6/7"
    )


def test_recovery_action_none_clears_the_finding():
    metrics = service()
    first = ingest(metrics, 0, {"gpu_recovery_action": 1})
    assert graded(first, "gpu_recovery_action"), "a reset asked for is graded"

    cleared = ingest(metrics, 15, {"gpu_recovery_action": 0})

    assert graded(cleared, "gpu_recovery_action") is None
    assert metrics.findings(CLUSTER, NODE) == []


def kernel_xid154(store: InMemoryStore, seconds: float, *, gpu: str = GPU) -> None:
    raw = (
        "NVRM: Xid (PCI:0000:59:00): 154, GPU recovery action changed from 0x0 "
        "(None) to 0x1 (GPU Reset Required)"
    )
    store.save_xid_event_if_absent(
        XidEvent(
            event_id=f"kernel-154-{gpu}",
            cluster_id=CLUSTER,
            node_id=NODE,
            observed_at=NOW + timedelta(seconds=seconds),
            event_source="KERNEL_LOG",
            xid=154,
            gpu_uuid=gpu,
            raw_message=raw,
        )
    )


def test_recovery_action_is_evidence_while_a_kernel_xid154_is_in_its_window():
    """The common order: the kernel line is stored (pending its 30 s window)
    before the exporter's next refresh shows the value."""

    store = InMemoryStore()
    kernel_xid154(store, -5)

    result = ingest(service(store), 0, {"gpu_recovery_action": 1})
    finding = graded(result, "gpu_recovery_action")

    assert finding.automatic_action == RecoveryAction.NO_ACTION.value
    assert finding.threshold_source == "XID_154_OWNS_RECOVERY"
    assert "XID 154 event kernel-154-GPU-a" in finding.reason
    assert result.new_findings == [finding], "recorded as evidence"
    assert gpu_node_health_findings(None, result.new_findings) == [], (
        "an evidence-only finding opens no incident and no second workflow"
    )
    assert not blocks_validation(finding), "evidence never fails a validation"


def test_an_xid154_of_another_gpu_does_not_own_this_gpu():
    store = InMemoryStore()
    kernel_xid154(store, -5, gpu="GPU-other")

    finding = graded(
        ingest(service(store), 0, {"gpu_recovery_action": 1}), "gpu_recovery_action"
    )

    assert finding.automatic_action == RecoveryAction.RESET_GPU.value


def test_an_open_xid154_incident_keeps_the_value_evidence_until_it_settles():
    store = InMemoryStore()
    incident = fault_incident(
        "inc-kernel-154",
        "kernel-154",
        cluster_id=CLUSTER,
        node_ids=[NODE],
        gpu_uuids=[GPU],
        policy_source="NVIDIA_XID_154",
        official_action="RESET_GPU",
        state=IncidentState.ACTION_PENDING,
        created_at=NOW - timedelta(minutes=10),
        updated_at=NOW - timedelta(minutes=10),
    )
    store.save_incident(incident)
    metrics = service(store)

    evidence = graded(
        ingest(metrics, 0, {"gpu_recovery_action": 1}), "gpu_recovery_action"
    )
    assert evidence.automatic_action == RecoveryAction.NO_ACTION.value
    assert "XID 154 incident inc-kernel-154" in evidence.reason

    store.save_incident(
        incident.model_copy(update={"state": IncidentState.RECOVERED}),
        expected=incident,
    )
    later = ingest(metrics, 300, {"gpu_recovery_action": 1})
    actionable = graded(later, "gpu_recovery_action")

    assert actionable.automatic_action == RecoveryAction.RESET_GPU.value
    assert actionable in later.new_findings, (
        "once the XID 154 incident settles a value still asking for recovery is "
        "the backstop: the action change activates it"
    )


# --- 235-238: the collector's downtrain verdict ---------------------------

DOWNTRAIN_LABELS = {
    "pcie_link_gen_current": "5",
    "pcie_link_width_current": "8",
    "pcie_link_gen_max": "5",
    "pcie_link_width_max": "16",
    "consecutive_valid_samples": "3",
}


def test_a_confirmed_downtrain_is_a_warning_diagnostic_and_advisory():
    finding = graded(
        ingest(service(), 0, {"pcie_link_downtrain": 1}, labels=DOWNTRAIN_LABELS),
        "pcie_link_downtrain",
    )

    assert (finding.severity.value, finding.automatic_action) == (
        "WARNING",
        RecoveryAction.RUN_DIAGNOSTICS.value,
    )
    assert finding.threshold_source == "SITE_PCIE_LINK_DOWNTRAIN"
    assert "Gen5 x8 of Gen5 x16 under load for 3 consecutive" in finding.reason
    assert "pcie_link_downtrain" in ADVISORY_GPU_FINDING_NAMES
    assert not blocks_validation(finding), (
        "load-gated: a validation at idle cannot judge it"
    )


def test_a_downtrain_verdict_of_zero_clears():
    metrics = service()
    ingest(metrics, 0, {"pcie_link_downtrain": 1}, labels=DOWNTRAIN_LABELS)

    assert (
        graded(ingest(metrics, 15, {"pcie_link_downtrain": 0}), "pcie_link_downtrain")
        is None
    )
    assert metrics.findings(CLUSTER, NODE) == []


def reset_history(store: InMemoryStore, completed_at: datetime, *, gpu: str = GPU):
    """A workflow on the node whose RESET_GPU of ``gpu`` succeeded."""

    steps = [
        workflow_step(WorkflowOperation.RESET_GPU, node_ids=[NODE], gpu_uuids=[gpu])
    ]
    workflow = workflow_request(
        "wf-reset",
        "inc-reset",
        official_steps=steps,
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.RESET_GPU,
                WorkflowStepStatus.SUCCEEDED,
                updated_at=completed_at,
            )
        ],
    )
    store.save_workflow(workflow)
    store.save_incident(
        fault_incident(
            "inc-reset",
            "event-reset",
            cluster_id=CLUSTER,
            node_ids=[NODE],
            workflow_request_id="wf-reset",
            state=IncidentState.RECOVERED,
            created_at=completed_at,
            updated_at=completed_at,
        )
    )


def test_a_downtrain_that_outlives_a_gpu_reset_escalates_to_an_operator():
    store = InMemoryStore()
    metrics = service(store)
    first = graded(
        ingest(metrics, 0, {"pcie_link_downtrain": 1}, labels=DOWNTRAIN_LABELS),
        "pcie_link_downtrain",
    )
    assert first.automatic_action == RecoveryAction.RUN_DIAGNOSTICS.value
    reset_history(store, NOW + timedelta(minutes=5))

    after = ingest(metrics, 900, {"pcie_link_downtrain": 1}, labels=DOWNTRAIN_LABELS)
    escalated = graded(after, "pcie_link_downtrain")

    assert escalated.automatic_action == RecoveryAction.ESCALATE_OPERATOR.value
    assert escalated.official_action == "RECONNECT_PCIE_AND_RUN_FIELD_DIAGNOSTIC"
    assert "RESET_GPU" in escalated.reason
    assert escalated in after.new_findings, "a new action: an incident of its own"
    sticky = graded(
        ingest(metrics, 1200, {"pcie_link_downtrain": 1}, labels=DOWNTRAIN_LABELS),
        "pcie_link_downtrain",
    )
    assert sticky.automatic_action == RecoveryAction.ESCALATE_OPERATOR.value


@pytest.mark.parametrize("variant", ["no-reset", "other-gpu", "before-finding"])
def test_only_a_reset_of_this_gpu_after_the_finding_escalates(variant):
    store = InMemoryStore()
    metrics = service(store)
    ingest(metrics, 0, {"pcie_link_downtrain": 1}, labels=DOWNTRAIN_LABELS)
    if variant == "other-gpu":
        reset_history(store, NOW + timedelta(minutes=5), gpu="GPU-other")
    elif variant == "before-finding":
        reset_history(store, NOW - timedelta(minutes=5))

    again = graded(
        ingest(metrics, 900, {"pcie_link_downtrain": 1}, labels=DOWNTRAIN_LABELS),
        "pcie_link_downtrain",
    )

    assert again.automatic_action == RecoveryAction.RUN_DIAGNOSTICS.value


# --- 202 and 1501: rates against the current link --------------------------


@pytest.mark.parametrize(
    ("gen", "width", "expected"),
    [
        (1, 16, 3.0),
        (3, 16, 8.0),
        (4, 16, 16.0),
        (5, 16, 31.0),
        (5, 8, 16.0),
        (6, 16, 62.0),
        (7, 16, 62.0),
        (None, 16, None),
        (5, None, None),
        (0, 16, None),
        (5, 0, None),
        (5.5, 16, None),
    ],
)
def test_expected_error_rate_is_dcgms_per_lane_formula(gen, width, expected):
    """DCGM v4.7.0 ``GetExpectedPcieReplayRate``: ceil(per-lane x lanes), a
    newer generation capped at Gen6, nothing decided on a blank."""

    assert PCIE_ERRORS_PER_LANE_PER_MINUTE == (0.15, 0.3, 0.48, 0.96, 1.92, 3.84)
    assert expected_pcie_error_rate(gen, width) == expected


def replay_rate(metrics, replays: float, link: dict[str, float] | None):
    ingest(metrics, 0, {"pcie_replay_total": 0, **(link or {})})
    return graded(
        ingest(metrics, 60, {"pcie_replay_total": replays, **(link or {})}),
        "pcie_replay_total",
    )


@pytest.mark.parametrize(
    ("link", "replays", "threshold", "source"),
    [
        (GEN5_X16, 20, None, None),
        (GEN5_X16, 40, 31.0, "NVIDIA_DCGM_REPLAY_RATE_FOR_LINK"),
        ({"pcie_link_gen_current": 6, "pcie_link_width_current": 16}, 40, None, None),
        (
            {"pcie_link_gen_current": 6, "pcie_link_width_current": 16},
            70,
            62.0,
            "NVIDIA_DCGM_REPLAY_RATE_FOR_LINK",
        ),
        (
            {"pcie_link_gen_current": 3, "pcie_link_width_current": 16},
            9,
            8.0,
            "NVIDIA_DCGM_REPLAY_RATE_FOR_LINK",
        ),
        (None, 9, 8.0, "NVIDIA_DCGM_DOCUMENTED_REPLAY_RATE"),
    ],
    ids=[
        "gen5x16-under",
        "gen5x16-over",
        "gen6x16-under",
        "gen6x16-over",
        "gen3",
        "no-link",
    ],
)
def test_the_replay_threshold_follows_the_current_link(
    link, replays, threshold, source
):
    finding = replay_rate(service(), replays, link)

    if threshold is None:
        assert finding is None, "within DCGM's expected rate for this link"
        return
    assert finding.threshold_value == threshold
    assert finding.threshold_source == source
    assert finding.policy_source == "NVIDIA_DCGM_HEALTH"
    assert finding.reason == "PCIe replay rate exceeded DCGM health threshold"


def test_an_operator_configured_replay_threshold_still_wins():
    finding = replay_rate(service(pcie_replay_rate_warning_per_minute=20), 25, GEN5_X16)

    assert finding.threshold_value == 20
    assert finding.policy_source == "SITE_OVERRIDE_NVIDIA_DCGM_HEALTH"


def correctable(metrics, seconds: float, total: float, link=GEN5_X16):
    return graded(
        ingest(metrics, seconds, {"pcie_correctable_error_total": total, **link}),
        "pcie_correctable_error_total",
    )


def test_correctable_errors_above_the_link_rate_for_two_windows_warn():
    metrics = service()
    assert correctable(metrics, 0, 0) is None

    first = correctable(metrics, 15, 10)
    second = correctable(metrics, 30, 20)
    calm = correctable(metrics, 45, 21)

    assert (first.automatic_action, first.confirmation_count) == (
        RecoveryAction.NO_ACTION.value,
        1,
    ), "the first window above R is evidence only"
    assert first.threshold_value == 31.0
    assert (second.automatic_action, second.confirmation_count) == (
        RecoveryAction.RUN_DIAGNOSTICS.value,
        2,
    )
    assert calm is None, "a window at or below R clears it"


def link_of(current: tuple[int, int], maximum: tuple[int, int] | None = None):
    values = {
        "pcie_link_gen_current": current[0],
        "pcie_link_width_current": current[1],
    }
    if maximum is not None:
        values |= {"pcie_link_gen_max": maximum[0], "pcie_link_width_max": maximum[1]}
    return values


@pytest.mark.parametrize(
    ("current", "maximum", "limit", "source"),
    [
        ((5, 16), (5, 16), 31.0, "NVIDIA_DCGM_ERROR_RATE_FOR_LINK"),
        ((4, 16), (4, 16), 16.0, "NVIDIA_DCGM_ERROR_RATE_FOR_LINK"),
        ((3, 16), (3, 16), 8.0, "NVIDIA_DCGM_ERROR_RATE_FOR_LINK"),
        ((5, 8), (5, 16), 16.0, "NVIDIA_DCGM_ERROR_RATE_FOR_LINK"),
        ((6, 16), (6, 16), 16.0, "NVIDIA_DCGM_DIAG_AER_RATE"),
        ((6, 16), None, 32.0, "NVIDIA_DCGM_DIAG_AER_RATE"),
    ],
    ids=["gen5x16", "gen4x16", "gen3x16", "gen5x8-of-x16", "gen6x16-b300", "no-max"],
)
def test_r_is_the_lower_of_the_link_formula_and_dcgms_diagnostic_limit(
    current, maximum, limit, source
):
    """DCGM v4.7.0 ``BusGrind::GetAERThresholdRate``: 32/min at a Gen5 x16
    maximum link (its default when it cannot read one), 16 otherwise; on Gen6
    x16 (DGX B300) that is 4x tighter than the replay formula's 62."""

    metrics = service()
    link = link_of(current, maximum)
    correctable(metrics, 0, 0, link)
    first = correctable(metrics, 15, 20, link)  # 80/min, above every R here

    assert correctable_rate_bound(link) == (limit, source)
    assert (first.threshold_value, first.threshold_source) == (limit, source)
    assert dcgm_diag_aer_rate(6, 16) == 16.0 and dcgm_diag_aer_rate(5, 16) == 32.0


def test_a_gen6_link_warns_above_the_diagnostics_rate_not_the_formulas():
    metrics = service()
    link = link_of((6, 16), (6, 16))
    correctable(metrics, 0, 0, link)
    first = correctable(metrics, 15, 10, link)  # 40/min: over 16, under 62
    second = correctable(metrics, 30, 20, link)

    assert first.automatic_action == RecoveryAction.NO_ACTION.value
    assert second.automatic_action == RecoveryAction.RUN_DIAGNOSTICS.value, (
        "two windows above R = 16/min"
    )


def test_correctable_errors_without_the_current_link_decide_nothing():
    metrics = service()
    correctable(metrics, 0, 0, {})
    assert correctable(metrics, 15, 1000, {}) is None


# --- 1423/1424: HW slowdown durations --------------------------------------

BRAKE = "hw_power_brake_violation_total_us"


def brake(metrics, seconds: float, total_us: float, mask: int = 0):
    return graded(
        ingest(metrics, seconds, {BRAKE: total_us, "clock_throttle_reasons": mask}),
        BRAKE,
    )


def test_any_brake_time_warns_and_two_samples_at_one_percent_drain():
    metrics = service()
    assert brake(metrics, 0, 0) is None

    small = brake(metrics, 15, 15_000)  # 0.1 % of 15 s
    one = brake(metrics, 30, 15_000 + 150_000)  # 1 % of 15 s
    two = brake(metrics, 45, 15_000 + 300_000)

    assert (small.severity.value, small.automatic_action) == (
        "WARNING",
        RecoveryAction.RUN_DIAGNOSTICS.value,
    )
    assert (one.severity.value, one.automatic_action) == (
        "WARNING",
        RecoveryAction.RUN_DIAGNOSTICS.value,
    ), "the first sample at 1 % is not yet sustained"
    assert (two.severity.value, two.automatic_action) == (
        "CRITICAL",
        RecoveryAction.DRAIN.value,
    )
    assert two.threshold_source == "NVIDIA_HW_POWER_BRAKE", "the existing brake verdict"
    assert two.official_action == "CONTACT_SUPPORT"


def test_brake_time_beside_bit_0x80_is_the_brake_drain_at_once():
    metrics = service()
    brake(metrics, 0, 0, 0x80)

    finding = brake(metrics, 15, 15_000, 0x80)

    assert (finding.severity.value, finding.automatic_action) == (
        "CRITICAL",
        RecoveryAction.DRAIN.value,
    )


@pytest.mark.parametrize(
    ("second", "why"),
    [
        (20_000_000, "more brake time than elapsed is not a duration"),
        (0, "unchanged: no brake time"),
    ],
)
def test_implausible_or_absent_brake_time_grades_nothing(second, why):
    metrics = service()
    brake(metrics, 0, 0)
    assert brake(metrics, 15, second) is None, why


def test_a_decreasing_brake_counter_is_a_reset_and_grades_nothing():
    """The cumulative reading is an assumption (NVML: 'All counters are in
    nanoseconds'); a per-interval series would decrease and grade nothing."""

    metrics = service()
    brake(metrics, 0, 500_000)
    assert brake(metrics, 15, 100_000) is None


def test_hw_thermal_slowdown_time_is_a_thermal_stress_signal():
    metrics = service()
    ingest(
        metrics,
        0,
        {"hw_thermal_slowdown_violation_total_us": 0, "gpu_temperature_c": 87},
    )

    result = ingest(
        metrics,
        15,
        {"hw_thermal_slowdown_violation_total_us": 150_000, "gpu_temperature_c": 87},
    )
    thermal = graded(result, "hw_thermal_slowdown_violation_total_us")
    stress = [
        item
        for item in result.composite_findings
        if item.correlation_rule_id == "THERMAL_STRESS"
    ]

    assert thermal.automatic_action == RecoveryAction.RUN_DIAGNOSTICS.value
    assert (
        stress
        and "hw_thermal_slowdown_violation_total_us" in stress[0].component_metrics
    )


# --- 300/301 ECC mode, InfoROM -------------------------------------------


@pytest.mark.parametrize(
    ("values", "name", "source"),
    [
        (
            {"ecc_mode_current": 0, "ecc_mode_pending": 0},
            "ecc_mode_current",
            "NVIDIA_ECC_MODE_DISABLED",
        ),
        (
            {"ecc_mode_current": 1, "ecc_mode_pending": 0},
            "ecc_mode_pending",
            "NVIDIA_ECC_MODE_PENDING",
        ),
        (
            {"ecc_mode_current": 0, "ecc_mode_pending": 1},
            "ecc_mode_pending",
            "NVIDIA_ECC_MODE_PENDING",
        ),
    ],
)
def test_ecc_disabled_or_a_pending_change_is_an_operator_warning(values, name, source):
    finding = graded(ingest(service(), 0, values), name)

    assert (finding.severity.value, finding.automatic_action) == (
        "WARNING",
        RecoveryAction.ESCALATE_OPERATOR.value,
    )
    assert finding.threshold_source == source
    assert not blocks_validation(finding), "an advisory level"


@pytest.mark.parametrize(
    "values",
    [{"ecc_mode_current": 1, "ecc_mode_pending": 1}, {"ecc_mode_pending": 0}],
    ids=["enabled", "pending-without-current"],
)
def test_ecc_enabled_or_an_unreadable_current_mode_grades_nothing(values):
    result = ingest(service(), 0, values)
    assert result.findings == []


def test_an_invalid_inforom_marks_the_memory_repair_evidence_untrusted():
    result = ingest(service(), 0, {"inforom_checksum_valid": 0})
    finding = graded(result, "inforom_checksum_valid")

    assert (finding.severity.value, finding.automatic_action) == (
        "WARNING",
        RecoveryAction.ESCALATE_OPERATOR.value,
    )
    assert "memory-repair evidence" in finding.reason and "untrusted" in finding.reason
    assert finding.official_action == "FLASH_INFOROM"
    assert ingest(service(), 0, {"inforom_checksum_valid": 1}).findings == []


# --- 346, 389, 393: memory ---------------------------------------------------


@pytest.mark.parametrize("product", ["H200", "B200", "B300", "GB200"])
def test_the_sram_threshold_flag_drains_for_the_field_diagnostic(product):
    finding = graded(
        ingest(service(), 0, {"sram_threshold_exceeded": 1}, product=product),
        "sram_threshold_exceeded",
    )

    assert (finding.severity.value, finding.automatic_action) == (
        "CRITICAL",
        RecoveryAction.DRAIN.value,
    )
    assert finding.official_action == "RUN_FIELD_DIAGNOSTIC_FOR_RMA"
    assert finding.policy_reference.endswith("#gpu-l2-memory-rma-policy"), (
        finding.policy_reference
    )


def test_the_sram_flag_on_an_unrecognised_part_is_quarantined_for_review():
    finding = graded(
        ingest(service(), 0, {"sram_threshold_exceeded": 1}, product="T4"),
        "sram_threshold_exceeded",
    )

    assert finding.automatic_action == RecoveryAction.QUARANTINE.value
    assert finding.official_action == "REVIEW_GPU_ARCHITECTURE"


@pytest.mark.parametrize(
    ("product", "outcome"),
    [
        ("H200", "cannot be remapped and sets the row-remap failure flag"),
        ("A100", "cannot be remapped and sets the row-remap failure flag"),
        ("NVIDIA B200", "invokes HBM channel repair if a spare channel is available"),
        (
            "NVIDIA B300 SXM6 AC",
            "invokes HBM channel repair if a spare channel is available",
        ),
        ("GB200", "invokes HBM channel repair if a spare channel is available"),
        ("T4", "the architecture is not recognised"),
    ],
)
def test_the_no_spare_rows_reason_follows_the_architecture_family(product, outcome):
    """GPU Memory Error Management (615), "GPU DRAM Memory RMA Policy": on
    Blackwell a third remap attempt on a bank invokes HBM channel repair."""

    finding = graded(
        ingest(service(), 0, {"row_remap_banks_none": 1}, product=product),
        "row_remap_banks_none",
    )

    assert outcome in finding.reason, finding.reason
    assert finding.policy_reference.endswith("#gpu-dram-memory-rma-policy"), (
        finding.policy_reference
    )
    assert "invokes HBM channel repair" in NVIDIA_BLACKWELL_CHANNEL_REPAIR_QUOTE
    assert "eight uncorrectable error rows" in NVIDIA_ROW_REMAP_FAILURE_BANK_QUOTE


def test_banks_without_spare_rows_warn_for_diagnostics_and_an_operator():
    finding = graded(
        ingest(service(), 0, {"row_remap_banks_none": 2}), "row_remap_banks_none"
    )

    assert (finding.severity.value, finding.automatic_action) == (
        "WARNING",
        RecoveryAction.RUN_DIAGNOSTICS.value,
    )
    assert "row_remap_banks_none" in OPERATOR_REVIEW_METRICS, "plus an operator notice"
    assert not blocks_validation(finding), "no automatic hardware action follows it"
    assert (
        ingest(
            service(), 0, {"row_remap_banks_none": 0, "row_remap_banks_low": 4}
        ).findings
        == []
    )


@pytest.mark.parametrize(("value", "warns"), [(511, False), (512, True), (600, True)])
def test_uncorrectable_row_remaps_warn_at_dcgms_limit_without_a_delta(value, warns):
    finding = graded(
        ingest(service(), 0, {"row_remap_uncorrectable_total": value}),
        "row_remap_uncorrectable_total",
    )

    if not warns:
        assert finding is None, "below DCGM's limit nothing without an increment"
        return
    assert finding.threshold_value == 512
    assert finding.threshold_source == "NVIDIA_DCGM_ROW_REMAP_LIMIT"
    assert finding.official_action == "RUN_FIELD_DIAGNOSTIC_FOR_RMA"
    assert not blocks_validation(finding), (
        "remaps are permanent: no validation can re-judge the level"
    )


def test_a_new_uncorrectable_remap_above_the_limit_keeps_its_increment_verdict():
    metrics = service()
    ingest(metrics, 0, {"row_remap_uncorrectable_total": 600})

    increased = graded(
        ingest(metrics, 15, {"row_remap_uncorrectable_total": 601}),
        "row_remap_uncorrectable_total",
    )
    level = graded(
        ingest(metrics, 30, {"row_remap_uncorrectable_total": 601}),
        "row_remap_uncorrectable_total",
    )

    assert increased.threshold_source == "CONFIGURED_DELTA", (
        "a new remap is graded as before batch 233"
    )
    assert blocks_validation(increased), "and still blocks a validation"
    assert level.threshold_source == "NVIDIA_DCGM_ROW_REMAP_LIMIT", (
        "between increments the level is DCGM's warning"
    )


# --- 112 bit 0x08 (decision 3c) --------------------------------------------


def test_hw_slowdown_bit_alone_is_not_graded_beside_thermal_and_brake():
    def mask(value: int):
        return clock_reason_decision(
            GpuMetricSample(
                metric_name="DCGM_FI_DEV_CLOCKS_EVENT_REASONS",
                canonical_name="clock_throttle_reasons",
                value=value,
            )
        )

    assert HW_SLOWDOWN_MASK == 0x08
    assert mask(0x08) is None
    assert mask(0x08 | 0x40)["threshold_source"] == "NVIDIA_THROTTLE_REASON_BITMASK"
    assert mask(0x08 | 0x80)["threshold_source"] == "NVIDIA_HW_POWER_BRAKE"
    assert (
        "It is active if either HW Thermal Slowdown or HW Power Brake are active."
        in (NVIDIA_SMI_HW_SLOWDOWN_QUOTE)
    )
    assert (
        "May be also reported during PState or clock change" in NVML_HW_SLOWDOWN_QUOTE
    )


@pytest.mark.parametrize(
    ("mask", "graded_as"),
    [
        (0x200, None),
        (0x400, None),
        (0x600, None),
        (0x200 | 0x80, "NVIDIA_HW_POWER_BRAKE"),
        (0x400 | 0x20, "NVIDIA_THROTTLE_REASON_BITMASK"),
    ],
    ids=["board-limit", "reliability", "both", "with-brake", "with-thermal"],
)
def test_clock_event_bits_the_product_does_not_name_grade_nothing(mask, graded_as):
    """NVML v615 BoardLimit 0x200 and Reliability 0x400 (``nvidia-ml-py``
    13.615.71): a policy limiting the clocks is no fault, and an unknown bit
    neither grades nor hides a bit the product names."""

    decision = clock_reason_decision(
        GpuMetricSample(
            metric_name="DCGM_FI_DEV_CLOCKS_EVENT_REASONS",
            canonical_name="clock_throttle_reasons",
            value=mask,
        )
    )
    result = ingest(service(), 0, {"clock_throttle_reasons": mask})

    assert NVML_UNGRADED_CLOCK_EVENT_BITS == {0x200: "BoardLimit", 0x400: "Reliability"}
    if graded_as is None:
        assert decision is None, f"{mask:#x} is not a fault"
        assert graded(result, "clock_throttle_reasons") is None
        return
    assert decision["threshold_source"] == graded_as


# --- PCIE_XID_LINK_FAILURE ---------------------------------------------------


@pytest.mark.parametrize(
    "component", ["pcie_link_downtrain", "pcie_correctable_error_total"]
)
def test_a_confirmed_link_finding_joins_a_link_xid_into_the_drain_composite(component):
    metrics = service()
    if component == "pcie_link_downtrain":
        first = {"xid_last_error": 0, "pcie_link_downtrain": 0}
        steps = [{"xid_last_error": 0, "pcie_link_downtrain": 1}]
        last = {"xid_last_error": 79, "pcie_link_downtrain": 1}
    else:
        first = {"xid_last_error": 0, component: 0, **GEN5_X16}
        steps = [{"xid_last_error": 0, component: 10, **GEN5_X16}]
        last = {"xid_last_error": 79, component: 20, **GEN5_X16}
    ingest(metrics, 0, first, labels=DOWNTRAIN_LABELS)
    for index, values in enumerate(steps, start=1):
        ingest(metrics, 15 * index, values, labels=DOWNTRAIN_LABELS)

    result = ingest(metrics, 30, last, labels=DOWNTRAIN_LABELS)
    composites = [
        item
        for item in result.composite_findings
        if item.correlation_rule_id == "PCIE_XID_LINK_FAILURE"
    ]

    assert composites, [item.correlation_rule_id for item in result.composite_findings]
    assert (composites[0].severity.value, composites[0].automatic_action) == (
        "CRITICAL",
        RecoveryAction.DRAIN.value,
    )
    assert composites[0].component_metrics == [component, "xid_last_error"]
    assert "correlated with XID 79" in composites[0].reason
