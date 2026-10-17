"""The control plane grades a node whose NVLink fabric never trained (batch 127).

Live (plain EKS p5en, H200 HGX, DCGM 4.7.0): Fabric Manager failed at boot and
every GPU reported ``DCGM_FI_DEV_FABRIC_MANAGER_STATUS`` 2 (in progress) for
19.5 h while nothing looked at the field; a healthy peer reports 3. The node is
graded once, at node level, after a GPU has held an unsettled state for the
window -- never during boot-time training, never on a GPU with no NVSwitch
fabric, never on a part whose registration semantics differ.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app.ingest.gpu_findings import gpu_node_health_findings
from gpu_fault.gpu_fabric_manager import (
    FABRIC_MANAGER_HELD_SECONDS,
    FABRIC_MANAGER_STATUS,
    FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS,
    FabricManagerStatus,
    fabric_validation,
    nvswitch_fabric_product,
)
from gpu_fault.gpu_metrics import GpuMetricSample, GpuMetricSource, GpuMetricsService
from gpu_fault.models import RecoveryAction
from tests._builders import build_store, gpu_metric_batch

NOW = datetime.now(timezone.utc).replace(microsecond=0)
GPUS = [f"GPU-{index}" for index in range(8)]
KEY = ("cluster-a", "node-a", "node", FABRIC_MANAGER_STATUS)
WINDOW = FABRIC_MANAGER_UNSETTLED_WINDOW_SECONDS


def _status(uuid: str, value: int, model: str = "NVIDIA H200") -> GpuMetricSample:
    return GpuMetricSample(
        metric_name="DCGM_FI_DEV_FABRIC_MANAGER_STATUS",
        canonical_name=FABRIC_MANAGER_STATUS,
        value=float(value),
        gpu_index=uuid.rsplit("-", 1)[-1],
        gpu_uuid=uuid,
        labels={"modelName": model, "UUID": uuid},
    )


def _held(sample: GpuMetricSample, seconds: float) -> GpuMetricSample:
    return sample.model_copy(
        update={
            "metric_name": f"gpu_fault_{FABRIC_MANAGER_HELD_SECONDS}",
            "canonical_name": FABRIC_MANAGER_HELD_SECONDS,
            "value": seconds,
            "unit": "seconds",
        }
    )


def _temperature(uuid: str) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name="DCGM_FI_DEV_GPU_TEMP",
        canonical_name="gpu_temperature_c",
        value=35.0,
        gpu_uuid=uuid,
    )


def _batch(
    batch_id: str,
    *,
    status: int,
    held: float | None,
    at: datetime = NOW,
    model: str = "NVIDIA H200",
    gpus: list[str] = GPUS,
):
    samples: list[GpuMetricSample] = []
    for uuid in gpus:
        reported = _status(uuid, status, model)
        samples += [_temperature(uuid), reported]
        if held is not None:
            samples.append(_held(reported, held))
    return gpu_metric_batch(
        batch_id,
        at,
        GpuMetricSource.DCGM_EXPORTER,
        samples,
        runtime_profile_version="profile-a",
        evidence_ref="prometheus://http://127.0.0.1:9400/metrics",
    )


def _fabric_findings(result) -> list:
    return [
        item
        for item in result.new_findings
        if item.canonical_name == FABRIC_MANAGER_STATUS
    ]


def test_a_fabric_held_in_progress_past_the_window_is_one_node_finding() -> None:
    store = build_store()
    service = GpuMetricsService(store=store)

    result = service.ingest(_batch("dcgm-1", status=2, held=WINDOW + 873))

    (finding,) = _fabric_findings(result)
    assert finding.severity.value == "CRITICAL", "an untrained fabric is critical"
    assert finding.gpu_uuid is None, "one node-level finding, not eight"
    assert finding.affected_gpu_uuids == GPUS, (
        "the full fabric reset needs the node's complete GPU inventory"
    )
    assert finding.automatic_action == RecoveryAction.RESET_GPU.value
    assert finding.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
    assert finding.policy_source == "NVIDIA_FABRIC_MANAGER_STATE"
    assert "IN_PROGRESS" in finding.reason and "8/8 GPUs" in finding.reason
    assert finding.value == WINDOW + 873 and finding.threshold_value == WINDOW
    state = store.get_gpu_finding_state(KEY)
    assert state is not None and state.finding is not None, state

    (node_finding,) = gpu_node_health_findings(None, [finding])
    assert node_finding.recommended_action is RecoveryAction.RESET_GPU
    assert node_finding.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
    assert node_finding.metric_name == FABRIC_MANAGER_STATUS
    assert node_finding.gpu_uuids == GPUS, "the reset targets every GPU"
    assert node_finding.severity.value == "critical"


def test_the_same_untrained_fabric_does_not_open_a_second_incident() -> None:
    service = GpuMetricsService(store=build_store())
    service.ingest(_batch("dcgm-1", status=2, held=WINDOW + 10))

    again = service.ingest(
        _batch("dcgm-2", status=2, held=WINDOW + 310, at=NOW + timedelta(seconds=300))
    )

    assert _fabric_findings(again) == [], "re-graded, not re-activated"
    assert [item.canonical_name for item in again.findings].count(
        FABRIC_MANAGER_STATUS
    ) == 1, "still active"


@pytest.mark.parametrize("held", [None, 0.0, 45.0, WINDOW - 1])
def test_boot_time_training_inside_the_window_is_not_a_finding(held) -> None:
    store = build_store()
    service = GpuMetricsService(store=store)

    result = service.ingest(_batch("dcgm-1", status=2, held=held))

    assert _fabric_findings(result) == []
    assert store.get_gpu_finding_state(KEY) is None, (
        "no state is written for a node that never had a finding"
    )


@pytest.mark.parametrize(
    "status",
    [
        FabricManagerStatus.NOT_SUPPORTED,
        FabricManagerStatus.SUCCESS,
        FabricManagerStatus.UNRECOGNIZED,
        FabricManagerStatus.NVML_TOO_OLD,
        9,
    ],
)
def test_settled_or_unknown_states_are_never_graded(status) -> None:
    service = GpuMetricsService(store=build_store())

    result = service.ingest(_batch("dcgm-1", status=int(status), held=WINDOW * 10))

    assert _fabric_findings(result) == [], f"status {status} is not a fault"


def test_a_failed_registration_past_the_window_is_graded() -> None:
    service = GpuMetricsService(store=build_store())

    result = service.ingest(_batch("dcgm-1", status=4, held=WINDOW))

    (finding,) = _fabric_findings(result)
    assert "FAILURE" in finding.reason


@pytest.mark.parametrize(
    "model",
    [
        "NVIDIA L40S",
        "NVIDIA H100 PCIe",
        "NVIDIA H100 NVL",
        "NVIDIA GH200 480GB",
        "NVIDIA B200",
        "NVIDIA GB200",
        "NVIDIA A100-SXM4-80GB",
        "",
    ],
)
def test_parts_without_a_graded_nvswitch_fabric_never_fire(model) -> None:
    service = GpuMetricsService(store=build_store())

    result = service.ingest(_batch("dcgm-1", status=2, held=WINDOW * 10, model=model))

    assert _fabric_findings(result) == [], (
        f"{model or 'an unknown product'} has no Hopper NVSwitch fabric to grade"
    )
    assert not nvswitch_fabric_product(model), f"{model} is not graded"


@pytest.mark.parametrize(
    "model", ["NVIDIA H100 80GB HBM3", "NVIDIA H200", "NVIDIA H800", "NVIDIA H20"]
)
def test_hopper_sxm_parts_are_graded(model) -> None:
    assert nvswitch_fabric_product(model), model


def test_a_settled_report_clears_the_finding_and_a_silent_batch_keeps_it() -> None:
    store = build_store()
    service = GpuMetricsService(store=store)
    service.ingest(_batch("dcgm-1", status=2, held=WINDOW + 1))

    without_field = gpu_metric_batch(
        "dcgm-2",
        NOW + timedelta(seconds=15),
        GpuMetricSource.DCGM_EXPORTER,
        [_temperature(uuid) for uuid in GPUS],
    )
    service.ingest(without_field)
    state = store.get_gpu_finding_state(KEY)
    assert state is not None and state.finding is not None, (
        "a batch that does not report the field says nothing about the fabric"
    )

    service.ingest(
        _batch("dcgm-3", status=3, held=None, at=NOW + timedelta(seconds=30))
    )

    state = store.get_gpu_finding_state(KEY)
    assert state is not None and state.finding is None, "Success clears it"
    assert all(
        item.canonical_name != FABRIC_MANAGER_STATUS
        for item in store.list_gpu_findings("cluster-a", "node-a", active_only=True)
    ), "and VALIDATE_GPU no longer sees it"


def test_a_restarted_clock_clears_the_finding_until_the_window_passes_again() -> None:
    """After a reset or reboot the collector restarts and training starts over.

    The finding clears on the first report inside the window, so the recovery's
    VALIDATE_GPU waits on the training instead of failing on the finding that
    asked for the recovery; it is raised again if the window passes again.
    """

    store = build_store()
    service = GpuMetricsService(store=store)
    service.ingest(_batch("dcgm-1", status=2, held=WINDOW + 1))

    service.ingest(_batch("dcgm-2", status=1, held=5.0, at=NOW + timedelta(minutes=5)))
    state = store.get_gpu_finding_state(KEY)
    assert state is not None and state.finding is None, state

    again = service.ingest(
        _batch("dcgm-3", status=1, held=WINDOW, at=NOW + timedelta(minutes=10))
    )
    (finding,) = _fabric_findings(again)
    assert "NOT_STARTED" in finding.reason


def test_validation_reading_waits_on_training_and_fails_past_the_window() -> None:
    def observations(status: int, held: float | None, model: str = "NVIDIA H200"):
        batch = _batch("b", status=status, held=held, model=model)
        return [(sample, NOW) for sample in batch.samples]

    graded, training, failure = fabric_validation(observations(3, None))
    assert (graded, training, failure) == (8, {}, None), "Success passes"

    graded, training, failure = fabric_validation(observations(2, 30.0))
    assert failure is None and training == dict.fromkeys(GPUS, "IN_PROGRESS")

    _graded, _training, failure = fabric_validation(observations(2, WINDOW))
    assert failure == "nvlink_fabric_not_trained(IN_PROGRESS: 8/8 GPUs)", failure

    _graded, _training, failure = fabric_validation(observations(4, 0.0))
    assert failure == "nvlink_fabric_not_trained(FAILURE: 8/8 GPUs)", (
        "a failed registration does not wait for the window"
    )

    assert fabric_validation(observations(2, WINDOW, "NVIDIA L40S")) == (0, {}, None)
    assert fabric_validation([]) == (0, {}, None), "nothing reported, nothing graded"


def test_validation_pairs_a_held_value_with_its_own_observation() -> None:
    """The store keeps the last held value after the state settles; a later
    unsettled report must not borrow it."""

    status = _status("GPU-0", 2)
    stale_held = _held(status, WINDOW * 4)

    graded, training, failure = fabric_validation(
        [(status, NOW), (stale_held, NOW - timedelta(minutes=30))]
    )

    assert failure is None and training == {"GPU-0": "IN_PROGRESS"}, (graded, failure)
