"""VALIDATE_GPU after a reset requires the driver's recovery action back at None.

Batch 233: DCGM field 1523 holds the NVML recovery action as a state. After
the workflow's RESET_GPU (or a full GPU + NVSwitch reset, the drain-P2P
recovery, a reboot, a replacement), every target GPU's value read *after* that
step must be 0 (None): a GPU whose driver still asks for a reset or a reboot is
not recovered. A target without a post-step sample waits (the executor's
per-step cap bounds it), an exporter that never served the field is not judged
(exported, never required). Evidence-only findings (``NO_ACTION``) and the
persistent advisory WARNINGs -- ECC mode, InfoROM, remap headroom, the
load-gated downtrain -- cannot be re-judged by a validation and never fail it.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample, GpuMetricSource
from gpu_fault.gpu_recovery_action_policy import RECOVERY_ACTION_RESET_OPERATIONS
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from tests.execution.test_validation_pending_repair_after_reset import (
    HYPERPOD,
    PLAIN_EKS,
    RESET_CHAIN,
    Rig,
    Site,
)

SITES = pytest.mark.parametrize("site", [HYPERPOD, PLAIN_EKS], ids=lambda s: s.cluster)
Op = WorkflowOperation


def scrape(site: Site, at, **values: float) -> GpuMetricBatch:
    return GpuMetricBatch(
        batch_id=f"scrape-{site.node}-{at.timestamp()}",
        cluster_id=site.cluster,
        node_id=site.node,
        observed_at=at,
        source=GpuMetricSource.DCGM_EXPORTER,
        product="H200",
        samples=[
            GpuMetricSample(
                metric_name=name,
                canonical_name=name,
                value=value,
                gpu_uuid=site.gpu,
                labels={"source_boot_id": site.boot},
            )
            for name, value in {"gpu_temperature_c": 30, **values}.items()
        ],
    )


def chain(site: Site, *, recovery_before: float | None = 1.0) -> Rig:
    rig = Rig(site, site.first_read_at)
    before = {} if recovery_before is None else {"gpu_recovery_action": recovery_before}
    rig.ingest(scrape(site, site.first_read_at - timedelta(seconds=1), **before))
    return rig


@SITES
def test_a_fresh_none_after_the_reset_passes(monkeypatch, site):
    rig = chain(site)
    assert "gpu_recovery_action" in rig.active(), (
        "the pre-reset value asked for a reset"
    )
    after = rig.restore_at + timedelta(seconds=15)
    rig.ingest(scrape(site, after, gpu_recovery_action=0))

    outcome = rig.validate(monkeypatch, after + timedelta(seconds=2))

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome


@SITES
@pytest.mark.parametrize(
    ("value", "label"), [(1, "Reset"), (2, "Reboot"), (6, "BUS_RESET")]
)
def test_a_recovery_action_still_set_after_the_reset_fails(
    monkeypatch, site, value, label
):
    rig = chain(site)
    after = rig.restore_at + timedelta(seconds=15)
    rig.ingest(scrape(site, after, gpu_recovery_action=value))

    outcome = rig.validate(monkeypatch, after + timedelta(seconds=2))

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    reasons = outcome.details["node_failures"][site.node]
    reset_index = RESET_CHAIN.index(Op.RESET_GPU)
    assert (
        f"gpu_recovery_action_not_none({site.gpu}={label}; read after step "
        f"{reset_index} RESET_GPU)"
    ) in reasons


@SITES
def test_a_target_without_a_post_reset_reading_waits(monkeypatch, site):
    rig = chain(site)
    after = rig.restore_at + timedelta(seconds=15)
    rig.ingest(scrape(site, after))

    outcome = rig.validate(monkeypatch, after + timedelta(seconds=2))

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    pending = outcome.details["node_pending"][site.node]
    assert pending["awaiting_gpu_recovery_action"] == [site.gpu]
    assert pending["gpu_recovery_action_after"] == rig.reset_at.isoformat()


@SITES
def test_an_exporter_that_never_served_the_field_is_not_judged(monkeypatch, site):
    rig = chain(site, recovery_before=None)
    after = rig.restore_at + timedelta(seconds=15)
    rig.ingest(scrape(site, after))

    outcome = rig.validate(monkeypatch, after + timedelta(seconds=2))

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome


def test_the_steps_that_must_clear_the_recovery_action():
    assert RECOVERY_ACTION_RESET_OPERATIONS == frozenset(
        {
            Op.RESET_GPU,
            Op.RESET_ALL_GPUS_NVSWITCHES,
            Op.RECOVER_DRAIN_P2P,
            Op.RESTART_NODE,
            Op.REPLACE_NODE,
        }
    )


@SITES
@pytest.mark.parametrize(
    "values",
    [
        {"ecc_mode_current": 0, "ecc_mode_pending": 0},
        {"inforom_checksum_valid": 0},
        {"row_remap_banks_none": 3},
        {"pcie_link_downtrain": 1},
    ],
    ids=["ecc-disabled", "inforom-invalid", "no-spare-rows", "downtrain"],
)
def test_advisory_findings_never_fail_a_validation(monkeypatch, site, values):
    rig = chain(site, recovery_before=None)
    after = rig.restore_at + timedelta(seconds=15)
    rig.ingest(scrape(site, after, **values))
    assert rig.active(), "the advisory finding is active"

    outcome = rig.validate(monkeypatch, after + timedelta(seconds=2))

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
