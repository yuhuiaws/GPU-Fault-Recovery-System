"""PCIe link health from DCGM fields 235-238, 202 and 1501 (batch 233).

**Downtraining (235-238).** A link that trains below its capability slows
host<->GPU and GPUDirect traffic and makes the GPU a straggler with no XID.
The collector compares the current generation/width with the maximum the
device *and system* allow -- never with a fleet constant (H200 trains Gen5 x16;
DGX B300's GPUs link at Gen6: NVIDIA DGX B300 User Guide, "Introduction" > "Key
Enhancements Over Previous DGX Systems": "ConnectX-8 networking connects
directly to GPUs via PCIe Gen6.") -- judges only under load and reports the
verdict as ``pcie_link_downtrain`` (``collectors/gpu/pcie_link``): 1 once the
link ran below its maximum for :data:`PCIE_DOWNTRAIN_CONFIRMATION_SAMPLES`
consecutive valid samples in this boot, 0 once a valid sample shows the full
link. NVIDIA's own words, nvidia-smi(1) (v610-era page), "GPU ATTRIBUTES >
PCI > GPU Link information": :data:`NVIDIA_SMI_CURRENT_LINK_QUOTE` (the load
gate) and :data:`NVIDIA_SMI_MAX_LINK_QUOTE` (the maximum is already min(GPU,
system); NVML v615 ``nvmlDeviceGetMaxPcieLinkGeneration``:
:data:`NVML_MAX_LINK_QUOTE`), so a platform that offers less than the GPU is
not a downtrain. "Under load" is ``gpu_utilization_percent`` >=
:data:`PCIE_DOWNTRAIN_LOAD_UTILIZATION_PERCENT` in the same scrape. P-state is
not used: the live H200 (driver 580.178.04) reports P-state 0 at 0 %
utilization on all 16 sampled GPUs, so P0 says nothing utilization does not.
Live, those GPUs held Gen5 x16 = max even at idle; the gate stays because other
parts and drivers downshift.

1 grades WARNING ``RUN_DIAGNOSTICS`` (``SITE_PCIE_LINK_DOWNTRAIN``). When the
finding was already active and the node's history shows a GPU reset
(``RESET_GPU`` naming the GPU, the full GPU + NVSwitch reset) or a reboot
completed since the finding was last judged, the reset did not retrain the
link: ``ESCALATE_OPERATOR``. A component of ``PCIE_XID_LINK_FAILURE``
(``gpu_composites``). Load-gated, so a validation at idle cannot re-judge it:
VALIDATE_GPU does not fail on it (``gpu_field_rules.ADVISORY_GPU_FINDING_NAMES``).

**The expected PCIe error rate of a link (202, 1501).** DCGM's PCIe health
check (DCGM v4.7.0 ``modules/health/DcgmHealthWatch.cpp``
``GetExpectedPcieReplayRate``, read by ``MonitorPcie``) warns when the replays
of the last minute exceed ceil(per-lane rate x lanes) of the *current* link
(fields 237/238), the per-lane rate being a 10^-12 bit error rate at the line
rate: :data:`PCIE_ERRORS_PER_LANE_PER_MINUTE` for Gen1-Gen6, a newer
generation capped at the newest it knows ("Cap the version at the max we know
about.") -- 31/min at Gen5 x16, 62/min at Gen6 x16. NVIDIA's documentation
states only the Gen3-era figure: DCGM Feature Overview (docs 4.6), "Background
Health Checks" > "Failure Conditions": :data:`DCGM_PCIE_REPLAY_DOC_QUOTE`; and
DCGM Diagnostics (docs 4.6), PCIe plugin parameter ``max_pcie_replays``:
:data:`DCGM_DIAG_REPLAY_DOC_QUOTE` (sic, the number is missing in NVIDIA's
text; 80 = 10 x 8). :func:`expected_pcie_error_rate` is that formula;
:func:`replay_decision` grades field 202 on it when the scrape carries the
current link, and on the documented 8/min when it does not (an exporter without
237/238, or a blank). An operator-configured threshold
(``GpuMetricsThresholds.pcie_replay_rate_warning_per_minute`` other than 8)
still wins.

**Correctable errors (1501).** ``DCGM_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS`` is
graded by rate only: occasional correctable errors are normal and the counter
restarts with a GPU reset. NVIDIA documents no health threshold for it. R is
the lower of two NVIDIA figures (:func:`correctable_rate_limit`): the expected
error rate of the current link (the replay formula above, one bit error per
10^12 bits) and the corrected-error rate DCGM's PCIe diagnostic tolerates for
this very field, DCGM v4.7.0 ``nvvs/plugin_src/pcie/Pcie.cpp``
``BusGrind::GetAERThresholdRate`` (:func:`dcgm_diag_aer_rate`): 32 per minute
at a Gen5 x16 *maximum* link (``aer_threshold`` 480 = "32/minute * 15
minutes"; also its default when it cannot read the maximum, "default value for
H100 systems"), 16 at any other (:data:`DCGM_PCIE_AER_RATE_QUOTE`). The two
agree only at Gen5 x16 (31 against 32) and Gen4 x16 (16 against 16); at Gen6
x16 (DGX B300) the formula gives 62 and the diagnostic 16, so R is 16 there.
Without the current link the rule decides nothing. A rate above R for
:data:`PCIE_CORRECTABLE_CONFIRMATION_WINDOWS` consecutive windows grades
WARNING ``RUN_DIAGNOSTICS``; the first window is evidence only (``NO_ACTION``).
Also a component of ``PCIE_XID_LINK_FAILURE``. Live: 0 on every H200; the guest
has no sysfs AER file for the GPU, so DCGM is the only source. Blackwell (B200,
B300): unit-tested from NVIDIA documentation, not live-verified (no B-series
hardware).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from gpu_fault.gpu_metric_models import SITE_METRIC_POLICY_VERSION, GpuHealthSeverity
from gpu_fault.gpu_power_policy import (
    NVIDIA_DCGM_HEALTH_REFERENCE,
    NVIDIA_DCGM_POLICY_VERSION,
)
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStepStatus,
)
from gpu_fault.store.shared.errors import NotFoundError

PCIE_LINK_GEN_MAX = "pcie_link_gen_max"
PCIE_LINK_WIDTH_MAX = "pcie_link_width_max"
PCIE_LINK_GEN_CURRENT = "pcie_link_gen_current"
PCIE_LINK_WIDTH_CURRENT = "pcie_link_width_current"
PCIE_LINK_FIELDS = (
    PCIE_LINK_GEN_MAX,
    PCIE_LINK_WIDTH_MAX,
    PCIE_LINK_GEN_CURRENT,
    PCIE_LINK_WIDTH_CURRENT,
)
PCIE_LINK_DOWNTRAIN = "pcie_link_downtrain"
PCIE_CORRECTABLE_ERRORS = "pcie_correctable_error_total"
PCIE_DOWNTRAIN_LOAD_UTILIZATION_PERCENT = 50.0
PCIE_DOWNTRAIN_CONFIRMATION_SAMPLES = 3
PCIE_CORRECTABLE_CONFIRMATION_WINDOWS = 2
#: The current link a replay or correctable-error rate is judged against.
PCIE_LINK_CONTEXT = frozenset({PCIE_LINK_GEN_CURRENT, PCIE_LINK_WIDTH_CURRENT})
#: ``GetExpectedPcieReplayRate`` (DCGM v4.7.0): errors per minute per lane for
#: PCIe Gen1..Gen6, "(1x10^-12) * (<line rate>x10^9) * 60".
PCIE_ERRORS_PER_LANE_PER_MINUTE = (0.15, 0.3, 0.48, 0.96, 1.92, 3.84)
#: The replay rate NVIDIA's DCGM documentation states (a Gen3-era figure).
DCGM_DOCUMENTED_REPLAY_RATE_PER_MINUTE = 8.0
DCGM_PCIE_REPLAY_DOC_QUOTE = "There were more than 8 PCIe replays in the last minute."
DCGM_DIAG_REPLAY_DOC_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html"
    "#pcie-gpu-bandwidth-plugin"
)
DCGM_DIAG_REPLAY_DOC_QUOTE = (
    "This is based on an expected replay rate less than per minute for PCIe Gen "
    "3.0, assuming this plugin will run for less than a minute and allowing 10x "
    "as many replays before failure."
)
DCGM_REPLAY_RATE_CODE_REFERENCE = (
    "https://github.com/NVIDIA/DCGM/blob/v4.7.0/modules/health/DcgmHealthWatch.cpp"
)
#: ``GetExpectedPcieReplayRate``'s own comments (DCGM v4.7.0).
DCGM_REPLAY_RATE_CODE_QUOTE = (
    "Expected replay rate = Replay rate per lane * Num of pcie Lanes. Note that "
    "value is ceiled."
)
DCGM_REPLAY_RATE_CAP_QUOTE = "Cap the version at the max we know about."
SITE_PCIE_LINK_DOWNTRAIN = "SITE_PCIE_LINK_DOWNTRAIN"
SITE_PCIE_LINK_POLICY_VERSION = "site-pcie-link-policy/v1"
NVIDIA_SMI_REFERENCE = "https://docs.nvidia.com/deploy/nvidia-smi/index.html"
NVIDIA_SMI_CURRENT_LINK_QUOTE = (
    "The current link generation and width. These may be reduced when the "
    "GPU is not in use."
)
NVIDIA_SMI_MAX_LINK_QUOTE = (
    "The maximum link generation and width possible with this GPU and system "
    "configuration."
)
NVML_MAX_LINK_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html"
)
NVML_MAX_LINK_QUOTE = (
    "Retrieves the maximum PCIe link generation possible with this device and system."
)
NVIDIA_DGX_B300_PCIE_REFERENCE = (
    "https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html"
    "#key-enhancements-over-previous-dgx-systems"
)
NVIDIA_DGX_B300_PCIE_QUOTE = (
    "ConnectX-8 networking connects directly to GPUs via PCIe Gen6."
)
DCGM_PCIE_LINK_FIELD_REFERENCE = (
    "https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/"
    "dcgm-api-field-ids.html#c.DCGM_FI_DEV_PCIE_LINK_GEN"
)
NVML_PCIE_CORRECTABLE_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html"
    "#c.NVML_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS"
)
NVML_PCIE_CORRECTABLE_QUOTE = "PCIe Correctable Errors Counter."
DCGM_PCIE_AER_RATE_REFERENCE = (
    "https://github.com/NVIDIA/DCGM/blob/v4.7.0/nvvs/plugin_src/pcie/Pcie.cpp"
)
DCGM_PCIE_AER_RATE_QUOTE = "/* Default is hopper, but fall back for other devices */"
#: ``GetAERThresholdRate``'s values (DCGM v4.7.0): its default link
#: (``PCIE_DEFAULT_PCIE_GEN`` 5, ``PCIE_DEFAULT_LINK_WIDTH`` 16, "default value
#: for H100 systems"), the rate there and the rate at any other link.
DCGM_DIAG_DEFAULT_LINK = (5, 16)
DCGM_DIAG_AER_RATE_DEFAULT_LINK_PER_MINUTE = 32.0
DCGM_DIAG_AER_RATE_OTHER_LINK_PER_MINUTE = 16.0
#: What a downtrain that outlived a reset waits for: a human to reseat or
#: replace the link (DCGM v4.7.0 ``DCGM_FR_PCI_REPLAY_RATE_NEXT``: "Reconnect
#: PCIe card. Run system side PCIE diagnostic utilities to verify hops off the
#: GPU board. If issue is on the board, run the field diagnostic.").
DOWNTRAIN_OFFICIAL_ACTION = "RECONNECT_PCIE_AND_RUN_FIELD_DIAGNOSTIC"
#: The completed node actions that retrain a GPU's PCIe link: a GPU reset
#: naming the GPU (or none), the full GPU + NVSwitch reset, a reboot.
LINK_RETRAINING_OPERATIONS = frozenset(
    {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTART_NODE,
    }
)
#: How many of the node's incidents (newest first) the reset lookup reads.
RESET_LOOKUP_INCIDENT_LIMIT = 50


def device_values(batch: Any, sample: Any) -> dict[str, float]:
    """The batch's values for the sample's GPU, by canonical name."""

    key = sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node"
    return {
        item.canonical_name: float(item.value)
        for item in batch.samples
        if (item.gpu_uuid or item.pci_bdf or item.gpu_index or "node") == key
    }


def expected_pcie_error_rate(gen: float | None, width: float | None) -> float | None:
    """DCGM's expected PCIe errors per minute for a link, or ``None`` unknown.

    ceil(:data:`PCIE_ERRORS_PER_LANE_PER_MINUTE` [gen] x width), a generation
    above Gen6 capped at Gen6 as DCGM does; ``None`` when either value is
    missing or not a positive whole number (no decision on a blank).
    """

    generation, lanes = _link_value(gen), _link_value(width)
    if generation is None or lanes is None:
        return None
    index = min(generation, len(PCIE_ERRORS_PER_LANE_PER_MINUTE)) - 1
    return float(math.ceil(PCIE_ERRORS_PER_LANE_PER_MINUTE[index] * lanes))


def _link_value(value: float | None) -> int | None:
    """A link generation or width as read, ``None`` unless a positive integer."""

    if value is None or not math.isfinite(value) or value < 1:
        return None
    return int(value) if float(value).is_integer() else None


def dcgm_diag_aer_rate(max_gen: float | None, max_width: float | None) -> float:
    """The corrected PCIe errors per minute DCGM's PCIe diagnostic tolerates.

    ``BusGrind::GetAERThresholdRate`` (DCGM v4.7.0) on the *maximum* link
    (fields 235/236), with its own default (Gen5 x16) for a value it cannot
    read: 32 at Gen5 x16, 16 at any other link.
    """

    gen = _link_value(max_gen) or DCGM_DIAG_DEFAULT_LINK[0]
    width = _link_value(max_width) or DCGM_DIAG_DEFAULT_LINK[1]
    if (gen, width) == DCGM_DIAG_DEFAULT_LINK:
        return DCGM_DIAG_AER_RATE_DEFAULT_LINK_PER_MINUTE
    return DCGM_DIAG_AER_RATE_OTHER_LINK_PER_MINUTE


def correctable_rate_bound(values: dict[str, float]) -> tuple[float, str] | None:
    """``(R, threshold source)`` for field 1501, ``None`` without the current
    link: the lower of the current link's expected error rate
    (``NVIDIA_DCGM_ERROR_RATE_FOR_LINK``) and DCGM's PCIe diagnostic limit for
    the maximum link (``NVIDIA_DCGM_DIAG_AER_RATE``); a tie is the link's."""

    expected = expected_pcie_error_rate(
        values.get(PCIE_LINK_GEN_CURRENT), values.get(PCIE_LINK_WIDTH_CURRENT)
    )
    if expected is None:
        return None
    diagnostic = dcgm_diag_aer_rate(
        values.get(PCIE_LINK_GEN_MAX), values.get(PCIE_LINK_WIDTH_MAX)
    )
    if diagnostic < expected:
        return diagnostic, "NVIDIA_DCGM_DIAG_AER_RATE"
    return expected, "NVIDIA_DCGM_ERROR_RATE_FOR_LINK"


def correctable_rate_limit(values: dict[str, float]) -> float | None:
    """R for field 1501 (:func:`correctable_rate_bound`), or ``None``."""

    bound = correctable_rate_bound(values)
    return None if bound is None else bound[0]


def replay_decision(
    thresholds: Any, sample: Any, rate: float | None, link: dict[str, float]
) -> dict[str, Any] | None:
    """Field 202's replay rate against DCGM's threshold for the current link."""

    if sample.canonical_name != "pcie_replay_total" or rate is None:
        return None
    configured = thresholds.pcie_replay_rate_warning_per_minute
    expected = expected_pcie_error_rate(
        link.get(PCIE_LINK_GEN_CURRENT), link.get(PCIE_LINK_WIDTH_CURRENT)
    )
    if configured != DCGM_DOCUMENTED_REPLAY_RATE_PER_MINUTE:
        threshold, threshold_source = configured, "SITE_CONFIGURED_REPLAY_RATE"
    elif expected is not None:
        threshold, threshold_source = expected, "NVIDIA_DCGM_REPLAY_RATE_FOR_LINK"
    else:
        threshold, threshold_source = configured, "NVIDIA_DCGM_DOCUMENTED_REPLAY_RATE"
    if rate <= threshold:
        return None
    source = (
        "SITE_OVERRIDE_NVIDIA_DCGM_HEALTH"
        if threshold_source == "SITE_CONFIGURED_REPLAY_RATE"
        else "NVIDIA_DCGM_HEALTH"
    )
    return {
        "severity": GpuHealthSeverity.WARNING,
        "reason": "PCIe replay rate exceeded DCGM health threshold",
        "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
        "threshold_value": threshold,
        "threshold_source": threshold_source,
        "policy_source": source,
        "policy_version": (
            NVIDIA_DCGM_POLICY_VERSION
            if source == "NVIDIA_DCGM_HEALTH"
            else SITE_METRIC_POLICY_VERSION
        ),
        "policy_reference": NVIDIA_DCGM_HEALTH_REFERENCE,
        "official_action": "EXAMINE_GPU_HEALTH",
    }


def correctable_decision(
    batch: Any, sample: Any, rate: float | None, previous: Any
) -> dict[str, Any] | None:
    """Field 1501 above R for two consecutive windows: WARNING diagnostics."""

    if sample.canonical_name != PCIE_CORRECTABLE_ERRORS or rate is None:
        return None
    bound = correctable_rate_bound(device_values(batch, sample))
    if bound is None or rate <= bound[0]:
        return None
    limit, limit_source = bound
    prior = previous.finding if previous is not None else None
    windows = (
        prior.confirmation_count + 1
        if prior is not None
        and prior.canonical_name == PCIE_CORRECTABLE_ERRORS
        and prior.rate_per_minute is not None
        and prior.rate_per_minute > limit
        else 1
    )
    windows = min(windows, PCIE_CORRECTABLE_CONFIRMATION_WINDOWS)
    confirmed = windows >= PCIE_CORRECTABLE_CONFIRMATION_WINDOWS
    return {
        "severity": GpuHealthSeverity.WARNING,
        "reason": (
            f"PCIe correctable errors at {rate:.1f}/min exceed {limit:g}/min (the "
            "lower of DCGM's expected error rate of the current link and its PCIe "
            "diagnostic's corrected-error limit) for "
            f"{windows}/{PCIE_CORRECTABLE_CONFIRMATION_WINDOWS} consecutive windows"
            + ("" if confirmed else "; evidence until the next window confirms it")
        ),
        "automatic_action": (
            RecoveryAction.RUN_DIAGNOSTICS if confirmed else RecoveryAction.NO_ACTION
        ).value,
        "threshold_value": limit,
        "threshold_source": limit_source,
        "policy_source": "SITE_PCIE_CORRECTABLE_RATE",
        "policy_version": SITE_PCIE_LINK_POLICY_VERSION,
        "policy_reference": DCGM_PCIE_AER_RATE_REFERENCE,
        "official_action": "EXAMINE_GPU_HEALTH",
        "confirmation_count": windows,
    }


def _link_text(labels: dict[str, str]) -> str:
    def part(name: str) -> str:
        return labels.get(name, "?")

    return (
        f"Gen{part(PCIE_LINK_GEN_CURRENT)} x{part(PCIE_LINK_WIDTH_CURRENT)} of "
        f"Gen{part(PCIE_LINK_GEN_MAX)} x{part(PCIE_LINK_WIDTH_MAX)}"
    )


def link_retrained_since(
    store: Any, batch: Any, sample: Any, since: datetime
) -> Any | None:
    """A completed reset or reboot that reached the GPU after ``since``.

    Read from the node's incidents (newest first, any state) and the workflow
    each points at; ``None`` when there is none. Rare: only a confirmed
    downtrain whose finding was already active asks.
    """

    for incident in store.list_incidents_by_state(
        batch.cluster_id,
        set(IncidentState),
        node_ids={batch.node_id},
        limit=RESET_LOOKUP_INCIDENT_LIMIT,
    ):
        if incident.updated_at < since or not incident.workflow_request_id:
            continue
        try:
            workflow = store.get_workflow(incident.workflow_request_id)
        except NotFoundError:
            continue
        steps = workflow.official_steps
        for execution in workflow.step_executions:
            step = (
                steps[execution.step_index]
                if execution.step_index < len(steps)
                else None
            )
            if (
                execution.status is WorkflowStepStatus.SUCCEEDED
                and execution.operation in LINK_RETRAINING_OPERATIONS
                and since < execution.updated_at <= batch.observed_at
                and (
                    step is None or not step.node_ids or batch.node_id in step.node_ids
                )
                and (
                    execution.operation is not WorkflowOperation.RESET_GPU
                    or step is None
                    or not step.gpu_uuids
                    or sample.gpu_uuid in step.gpu_uuids
                )
            ):
                return execution
    return None


def downtrain_decision(
    store: Any, batch: Any, sample: Any, previous: Any
) -> dict[str, Any] | None:
    """The collector's downtrain verdict, graded (see the module docstring)."""

    if sample.canonical_name != PCIE_LINK_DOWNTRAIN or sample.value < 1:
        return None
    link = _link_text(sample.labels)
    count = sample.labels.get("consecutive_valid_samples", "?")
    prior = previous.finding if previous is not None else None
    retrained = None
    if prior is not None and prior.canonical_name == PCIE_LINK_DOWNTRAIN:
        retrained = (
            "an earlier escalation"
            if prior.automatic_action == RecoveryAction.ESCALATE_OPERATOR.value
            else link_retrained_since(store, batch, sample, previous.observed_at)
        )
    common = {
        "severity": GpuHealthSeverity.WARNING,
        "threshold_value": float(PCIE_DOWNTRAIN_CONFIRMATION_SAMPLES),
        "threshold_source": SITE_PCIE_LINK_DOWNTRAIN,
        "policy_source": SITE_PCIE_LINK_DOWNTRAIN,
        "policy_version": SITE_PCIE_LINK_POLICY_VERSION,
        "policy_reference": NVIDIA_SMI_REFERENCE,
        "confirmation_count": PCIE_DOWNTRAIN_CONFIRMATION_SAMPLES,
    }
    if retrained is not None:
        what = (
            retrained
            if isinstance(retrained, str)
            else f"{retrained.operation.value} at {retrained.updated_at.isoformat()}"
        )
        return {
            **common,
            "reason": (
                f"PCIe link still trained at {link} under load after {what}: the "
                "reset did not retrain it; reconnect or replace the link"
            ),
            "automatic_action": RecoveryAction.ESCALATE_OPERATOR.value,
            "official_action": DOWNTRAIN_OFFICIAL_ACTION,
        }
    return {
        **common,
        "reason": (
            f"PCIe link trained at {link} under load for {count} consecutive "
            "valid samples in this boot"
        ),
        "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
        "official_action": "EXAMINE_GPU_HEALTH",
    }
