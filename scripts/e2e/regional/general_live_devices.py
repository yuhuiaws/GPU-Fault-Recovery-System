"""GF-GENERAL-LIVE device-signal cases: 007 SMART, 010 EFA/RDMA, 012 GPU missing,
015 BMC.

These four categories are, by the coverage matrix, ``live-replay`` or
``vendor-lab`` only: the product grades them from a raw reading
(``smartctl -H``/``--scan-open``, the EFA ``hw_counters``/``counters`` sysfs
and ``ethtool -S``, the ``nvidia-smi`` GPU inventory and ``/proc/driver/nvidia``,
``ipmitool sensor``), and proving the processing chain means replacing exactly
that targeted reading in the one host-collector unit's private environment, or
inducing a real hardware fault in a vendor lab.

This site has no vendor lab and must never unbind or remove a real GPU, fail a
real NVMe, or disconnect a real EFA link, so the vendor-lab variants stay
NOT_RUN(VENDOR_LAB_ONLY). The command/file replay fixtures that the remaining
variants need are a shared raw-source replay helper that is **not built in this
phase** (building a general host-binary / procfs / sysfs replay toolkit is out
of this lane's scope); they stay NOT_RUN(FIXTURE_NOT_READY) with the exact
reader and field each needs, so the coordinator can add one centrally. Where
the site genuinely lacks the capability (no BMC on EC2, no non-EFA RDMA device)
the reason is CAPABILITY_ABSENT.

The one variant implementable here without any injection is ``007 healthy``: a
read-only observation that the node's real SMART reading grades the drives
healthy and mints no storage finding -- the negative baseline the failing
variants are judged against.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import activity, run_survey
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    CAPABILITY,
    LIVE_REPLAY,
    OBSERVATION,
    VENDOR_LAB,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_verdicts import (
    ChainExpectation,
    chain_errors,
    hardware_errors,
    samples,
    verdict,
)

CASE_007 = "GF-GENERAL-LIVE-007"
CASE_010 = "GF-GENERAL-LIVE-010"
CASE_012 = "GF-GENERAL-LIVE-012"
CASE_015 = "GF-GENERAL-LIVE-015"

SMART_REPLAY = (
    "FIXTURE_NOT_READY: a smartctl command-output replay in the host collector "
    "unit's private PATH (the shared raw-source replay helper) is not built this phase"
)
EFA_REPLAY = (
    "FIXTURE_NOT_READY: an EFA hw_counters/counters sysfs file replay in the host "
    "collector unit's private mount namespace (the shared replay helper) is not built"
)
GPU_REPLAY = (
    "FIXTURE_NOT_READY: an nvidia-smi / NVML / /proc/driver/nvidia read replay in the "
    "host collector unit's private environment (the shared replay helper) is not built"
)
BMC_REPLAY = (
    "FIXTURE_NOT_READY: an ipmitool sensor output replay in the host collector unit's "
    "private PATH (the shared replay helper) is not built this phase"
)
NO_BMC = "CAPABILITY_ABSENT: EC2 instances expose no BMC; ipmitool finds no device"


# --------------------------------------------------------------------------- #
# 007 healthy (read-only observation through the real SMART reader)
# --------------------------------------------------------------------------- #
def smart_healthy(ctx: VariantContext) -> dict[str, Any]:
    """The node's real ``smartctl`` grades the drives healthy -> no finding.

    No injection and no replay: the host collector already runs ``smartctl``
    every cycle, so this reads the product's own records across a window and
    checks that ``smart_health_failed`` minted no storage finding and that any
    latest ``smart_health_failed`` value is 0. It is the negative baseline for
    the failing SMART variants, which need the replay fixture.
    """

    run_survey(ctx)
    started_at = utc_now()
    ctx.hold(ctx.host_summary + ctx.host_interval * 3, name="smart-healthy-hold")
    observed = activity(ctx, started_at, metrics={"smart_health_failed"})
    smart_samples = samples(observed, "smart_health_failed")
    latest = [
        item
        for item in observed.get("metric_latest") or []
        if item.get("name") == "smart_health_failed"
    ]
    stages = {
        "no_finding": chain_errors(
            observed, ChainExpectation(metric="smart_health_failed", present=False)
        ),
        "hardware": hardware_errors(observed),
        "healthy_values": []
        if all(
            not isinstance(item.get("value"), (int, float)) or item["value"] == 0
            for item in [*smart_samples, *latest]
        )
        else ["a smart_health_failed reading was non-zero on a healthy node"],
    }
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": [
            "read-only baseline; the failing SMART variants need the replay fixture"
        ],
        "smart_latest": latest,
    }


VARIANTS = [
    # ----------------------------------------------------------------------- #
    # 007 SMART
    # ----------------------------------------------------------------------- #
    deferred_variant(
        CASE_007,
        "health_failed",
        proof_mode=LIVE_REPLAY,
        reason=SMART_REPLAY,
        lanes=("eks",),
        expected="smart_status.passed=false -> CRITICAL QUARANTINE via the real JSON parser",
    ),
    deferred_variant(
        CASE_007,
        "query_failed",
        proof_mode=LIVE_REPLAY,
        reason=SMART_REPLAY,
        lanes=("eks",),
        expected="a non-zero smartctl -H is smart_query_failed/UNKNOWN, never a healthy zero",
    ),
    deferred_variant(
        CASE_007,
        "scan_failed",
        proof_mode=LIVE_REPLAY,
        reason=SMART_REPLAY,
        lanes=("eks",),
        expected="a --scan-open failure must be surfaced, not treated as no drives (spec FAIL risk)",
    ),
    deferred_variant(
        CASE_007,
        "malformed_json",
        proof_mode=LIVE_REPLAY,
        reason=SMART_REPLAY,
        lanes=("eks",),
        expected="corrupt -H JSON -> smart_query_failed, not a parsed healthy result",
    ),
    deferred_variant(
        CASE_007,
        "tool_missing",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: hiding smartctl from the unit needs a unit-private "
            "PATH shadow (the shared replay helper); a host binary is never removed"
        ),
        lanes=("eks",),
        expected="no smartctl -> no SMART samples and no finding",
    ),
    deferred_variant(
        CASE_007,
        "unsupported_device",
        proof_mode=LIVE_REPLAY,
        reason=SMART_REPLAY,
        lanes=("eks",),
        expected="a device with no smart_status is not a failed query and not quarantined",
    ),
    VariantSpec(
        case_id=CASE_007,
        variant="healthy",
        proof_mode=OBSERVATION,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection="none (read-only): observe the node's real SMART grading",
        expected="a healthy smartctl mints no storage finding and no smart_health_failed > 0",
        cleanup="none",
        max_minutes=10,
        execute=smart_healthy,
    ),
    # ----------------------------------------------------------------------- #
    # 010 EFA / RDMA error counters
    # ----------------------------------------------------------------------- #
    deferred_variant(
        CASE_010,
        "rdma_error",
        proof_mode=LIVE_REPLAY,
        reason=EFA_REPLAY,
        lanes=("eks", "hp"),
        expected="a target counter delta -> rdma_errors_delta WARNING -> VALIDATE_FABRIC (vendor-lab for a real fault)",
    ),
    deferred_variant(
        CASE_010,
        "link_inactive",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: an EFA port state/phys_state sysfs replay (the shared "
            "replay helper) is not built; the physical path is vendor-lab only"
        ),
        lanes=("eks", "hp"),
        expected="state != ACTIVE/LinkUp -> rdma_link_down=1",
    ),
    deferred_variant(
        CASE_010,
        "rnr",
        proof_mode=LIVE_REPLAY,
        reason=EFA_REPLAY,
        lanes=("eks", "hp"),
        expected="efa_rnr_errors_delta is diagnosed as congestion, not a bad card",
    ),
    deferred_variant(
        CASE_010,
        "retry",
        proof_mode=LIVE_REPLAY,
        reason=EFA_REPLAY,
        lanes=("eks", "hp"),
        expected="efa_retry_errors_delta -> diagnostic only, no hardware escalation",
    ),
    deferred_variant(
        CASE_010,
        "cq_error",
        proof_mode=LIVE_REPLAY,
        reason=EFA_REPLAY,
        lanes=("eks", "hp"),
        expected="efa_cq_errors_delta -> WARNING -> VALIDATE_FABRIC",
    ),
    deferred_variant(
        CASE_010,
        "missing_counter",
        proof_mode=LIVE_REPLAY,
        reason=EFA_REPLAY,
        lanes=("eks", "hp"),
        expected="a missing field is UNKNOWN/NOT_SUPPORTED, never a reported zero (spec FAIL risk)",
    ),
    deferred_variant(
        CASE_010,
        "ethtool_failed",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: an ethtool -S failure replay (the shared replay helper) "
            "is not built; a failed ethtool must not read as zero errors (spec FAIL risk)"
        ),
        lanes=("eks", "hp"),
        expected="a failed ethtool is UNKNOWN, not zero congestion counters",
    ),
    deferred_variant(
        CASE_010,
        "non_efa",
        proof_mode=CAPABILITY,
        reason=(
            "CAPABILITY_ABSENT: this site has only EFA RDMA devices; there is no "
            "IB-only device whose counters could be mistaken for EFA readings"
        ),
        lanes=("eks", "hp"),
        expected="an IB-only counter is not treated as a valid EFA reading",
    ),
    deferred_variant(
        CASE_010,
        "counter_reset",
        proof_mode=LIVE_REPLAY,
        reason=EFA_REPLAY,
        lanes=("eks", "hp"),
        expected="a counter reset is not a new fault and not a recovery",
    ),
    # ----------------------------------------------------------------------- #
    # 012 GPU missing / fallen off the bus
    # ----------------------------------------------------------------------- #
    deferred_variant(
        CASE_012,
        "confirmed_missing",
        proof_mode=VENDOR_LAB,
        reason=(
            "VENDOR_LAB_ONLY: a real GPU off the bus; this lane must never unbind or "
            "remove a real GPU, and the 2-sample confirmation feeds a provider reboot"
        ),
        lanes=("eks", "hp"),
        expected="two confirmed missing samples -> approved reboot chain, new boot id",
    ),
    deferred_variant(
        CASE_012,
        "single_sample",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: a one-sample UUID drop from the host collector's "
            "uuid,utilization.gpu query needs a host-query shadow in the shared replay "
            "helper (the existing collector-window shadow matches only the inventory "
            "snapshot query, which is the 026 identity path, not 012)"
        ),
        lanes=("eks", "hp"),
        expected="one missing sample does not confirm and never reaches the reboot chain",
    ),
    deferred_variant(
        CASE_012,
        "query_failed",
        proof_mode=LIVE_REPLAY,
        reason=GPU_REPLAY,
        lanes=("eks",),
        expected="an NVML init failure is UNKNOWN, not zero GPUs (spec FAIL: dual read-failure)",
    ),
    deferred_variant(
        CASE_012,
        "proc_unreadable",
        proof_mode=LIVE_REPLAY,
        reason=GPU_REPLAY,
        lanes=("eks",),
        expected="an unreadable /proc/driver/nvidia is UNKNOWN, not zero GPUs (spec FAIL)",
    ),
    deferred_variant(
        CASE_012,
        "expectation_wrong",
        proof_mode=LIVE_REPLAY,
        reason=GPU_REPLAY,
        lanes=("eks",),
        expected="a wrong expected count is not a real missing GPU; it must not reboot",
    ),
    deferred_variant(
        CASE_012,
        "reboot_verified",
        proof_mode=VENDOR_LAB,
        reason=(
            "VENDOR_LAB_ONLY: a verified reboot of a truly missing GPU needs a real "
            "confirmed-missing fault and an approved provider reboot"
        ),
        lanes=("eks", "hp"),
        expected="STOP, provider reboot, new boot id, full validation, controlled resume",
    ),
    deferred_variant(
        CASE_012,
        "post_reboot_failure",
        proof_mode=VENDOR_LAB,
        reason=(
            "VENDOR_LAB_ONLY: a GPU still missing after a real reboot needs the same "
            "physical fault and provider reboot"
        ),
        lanes=("eks", "hp"),
        expected="a still-missing GPU escalates / stays isolated, no blind reboot loop",
    ),
    # ----------------------------------------------------------------------- #
    # 015 BMC alerts
    # ----------------------------------------------------------------------- #
    deferred_variant(
        CASE_015,
        "critical",
        proof_mode=LIVE_REPLAY,
        reason=BMC_REPLAY,
        lanes=("eks",),
        expected="a cr status in the 4th sensor column -> bmc_critical_sensor -> QUARANTINE",
    ),
    deferred_variant(
        CASE_015,
        "nonrecoverable",
        proof_mode=LIVE_REPLAY,
        reason=BMC_REPLAY,
        lanes=("eks",),
        expected="an nr status -> bmc_critical_sensor -> QUARANTINE",
    ),
    deferred_variant(
        CASE_015,
        "healthy",
        proof_mode=CAPABILITY,
        reason=NO_BMC,
        lanes=("eks",),
        expected="healthy sensors mint no finding",
    ),
    deferred_variant(
        CASE_015,
        "wrong_column",
        proof_mode=LIVE_REPLAY,
        reason=BMC_REPLAY,
        lanes=("eks",),
        expected="a unit string in another column is not read as a status",
    ),
    deferred_variant(
        CASE_015,
        "unavailable_sensor",
        proof_mode=LIVE_REPLAY,
        reason=BMC_REPLAY,
        lanes=("eks",),
        expected="an unavailable sensor is neither healthy nor a false quarantine",
    ),
    deferred_variant(
        CASE_015,
        "command_failure",
        proof_mode=LIVE_REPLAY,
        reason=BMC_REPLAY,
        lanes=("eks",),
        expected="a failed ipmitool query is UNKNOWN, not healthy",
    ),
    deferred_variant(
        CASE_015,
        "unsupported_bmc",
        proof_mode=CAPABILITY,
        reason=NO_BMC,
        lanes=("eks",),
        expected="a node with no BMC is recorded as a capability gap, not a PASS",
    ),
]
