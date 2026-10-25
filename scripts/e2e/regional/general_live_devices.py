"""GF-GENERAL-LIVE device-signal cases: 010 EFA/RDMA and 012 GPU missing.

These two categories are, by the coverage matrix, ``live-replay`` or
``vendor-lab`` only: the product grades them from a raw reading (the EFA
``hw_counters``/``counters`` sysfs and ``ethtool -S``, the ``nvidia-smi`` GPU
inventory and ``/proc/driver/nvidia``), and proving the processing chain means
replacing exactly that targeted reading in the one host-collector unit's
private environment, or inducing a real hardware fault in a vendor lab.

This site has no vendor lab and must never unbind or remove a real GPU or
disconnect a real EFA link, so the vendor-lab variants stay
NOT_RUN(VENDOR_LAB_ONLY). The remaining variants need the shared host source
replay (``general_live_source_replay.py``, approval ``HOST_SOURCE_REPLAY``)
wired for their reader; until a lane wires them they stay
NOT_RUN(FIXTURE_NOT_READY) with the exact reader and field each needs. Where
the site genuinely lacks the capability (no non-EFA RDMA device) the reason is
CAPABILITY_ABSENT. The sibling SMART (007) and BMC (015) categories live in
``general_live_smart.py`` and ``general_live_bmc.py``.
"""

from __future__ import annotations

from scripts.e2e.regional.general_live_registry import (
    CAPABILITY,
    LIVE_REPLAY,
    VENDOR_LAB,
    deferred_variant,
)

CASE_010 = "GF-GENERAL-LIVE-010"
CASE_012 = "GF-GENERAL-LIVE-012"

EFA_REPLAY = (
    "FIXTURE_NOT_READY: an EFA hw_counters/counters sysfs file replay in the host "
    "collector unit's private mount namespace (the shared replay helper) is not built"
)
GPU_REPLAY = (
    "FIXTURE_NOT_READY: an nvidia-smi / NVML / /proc/driver/nvidia read replay in the "
    "host collector unit's private environment (the shared replay helper) is not built"
)


VARIANTS = [
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
]
