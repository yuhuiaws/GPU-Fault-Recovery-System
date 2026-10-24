"""GF-GENERAL-LIVE-013 EFA device inventory and 014 K8s allocatable under-report.

013: the host collector counts EFA devices three ways -- PCI functions
(``GPU_FAULT_PCI_DEVICES_ROOT``), driver-bound RDMA devices and active ports
(``collectors/host/observation.efa_inventory_counts``) -- and a shortfall
resolves by failure mode (``host_health.efa_inventory_rule``): DRIVER_UNBOUND
remediates the driver, PCI_DEVICE_MISSING reboots, LINK_INACTIVE/EXCESS only
diagnose, and an *unreadable* reading is ``INVENTORY_UNAVAILABLE`` (UNKNOWN),
never zero cards. The one safe, reviewed replay on this single node is
``unreadable_state``: a malformed PCI-root fixture repointed into the host
collector's own environment makes the observation UNAVAILABLE without touching a
device. PCI_DEVICE_MISSING drives REBOOT on the only EKS GPU node, and the
driver-binding variants need a reserved native unbind/rebind; those stay NOT_RUN.

014: a Kubernetes allocatable shortfall must come from a *real* device-plugin
registration fault (``collectors/cloud/kubernetes``); the spec is explicit that
patching Node.status or faking allocatable is not valid proof, so there is no
live-replay path. The native device-plugin fault needs the DEVICE_PLUGIN_RESTART
approval and a plugin pod-delete/re-register injection with new-Pod-UID
verification, not built this phase -- every 014 variant stays NOT_RUN with its
reason. The denominator keeps them all.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import activity
from scripts.e2e.regional.general_live_fixture import VariantContext, utc_now
from scripts.e2e.regional.general_live_registry import (
    LIVE_REPLAY,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_source_replay import host_source_replay
from scripts.e2e.regional.general_live_verdicts import (
    finding_incidents,
    hardware_errors,
    node_isolated,
    verdict,
)

CASE_013 = "GF-GENERAL-LIVE-013"
CASE_014 = "GF-GENERAL-LIVE-014"
DETECTION_BUDGET = 420
EFA_METRICS = {"efa_inventory_mismatch", "efa_inventory_active_count", "efa_inventory_expected_count"}


def unreadable_state(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: a malformed PCI reading is UNKNOWN, not zero cards, no reboot."""

    replay = host_source_replay(ctx)
    replay.create()
    replay.build_pci(efa_count=16, malformed=True)
    opened = replay.open_pci_root(restore_seconds=ctx.spec.max_minutes * 60)
    ctx.evidence.write("pci-root-open", opened)
    started = utc_now()
    # Let the host collector poll the malformed PCI tree a few times.
    ctx.hold(ctx.host_interval * 4 + 30, name="unreadable")
    observed = activity(ctx, started, metrics=EFA_METRICS, kinds=("HOST_TELEMETRY",))
    collection_errors = [
        error
        for batch in observed.get("batches") or []
        for error in batch.get("collection_errors") or []
        if "EFA inventory" in error
    ]
    ctx.evidence.append_oracle(
        [{"efa_collection_errors": collection_errors}], source="pci-root"
    )
    missing_findings = finding_incidents(observed, "efa_inventory_mismatch")
    stages = {
        "unknown_observed": []
        if collection_errors
        else ["no 'EFA inventory observation unavailable' collection error was recorded"],
        "no_missing_finding": [
            f"a malformed reading minted an efa_inventory_mismatch incident: {item['incident_id']}"
            for item in missing_findings
        ],
        "no_reboot": hardware_errors(observed),
        "not_isolated": ["node isolated on an UNKNOWN EFA inventory reading"]
        if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
        else [],
    }
    return _result(
        stages,
        notes=["a malformed PCI vendor/device makes the EFA inventory UNAVAILABLE (UNKNOWN), not zero"],
        extra={"collection_errors": collection_errors, "proof_scope": "live-replay"},
    )


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        **(extra or {}),
    }


_PLUGIN_NATIVE = (
    "NOT_AUTHORIZED: a real kubelet device-plugin registration fault needs the "
    "DEVICE_PLUGIN_RESTART approval and a plugin pod-delete/re-register injection with "
    "new-Pod-UID verification; patching Node.status or faking allocatable is not valid "
    "proof (spec 014). Not built in this lane."
)

VARIANTS = [
    # --- 013 EFA inventory --- #
    VariantSpec(
        case_id=CASE_013,
        variant="unreadable_state",
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection="repoint GPU_FAULT_PCI_DEVICES_ROOT at a reviewed tree with a malformed EFA vendor/device",
        expected="EFA inventory observation UNAVAILABLE (UNKNOWN); no missing-device finding, no reboot, no isolation",
        cleanup="close the PCI-root window; confirm clean",
        max_minutes=14,
        approvals=("KERNEL_SOURCE_REPLAY",),
        execute=unreadable_state,
    ),
    deferred_variant(
        CASE_013,
        "pci_missing",
        proof_mode=LIVE_REPLAY,
        reason=(
            "NOT_AUTHORIZED: a PCI-root replay with fewer EFA functions drives the "
            "PCI_DEVICE_MISSING -> REBOOT_NODE chain on the only EKS GPU node; needs a "
            "reboot approval + maintenance window, and the known double-read-failure risk "
            "must be reviewed before executing"
        ),
        lanes=("eks",),
        expected="confirmed PCI_DEVICE_MISSING -> reboot chain with a new boot id (spec FAIL risk: two read failures must not reboot)",
    ),
    deferred_variant(
        CASE_013,
        "driver_unbound",
        proof_mode=LIVE_REPLAY,
        reason=(
            "NOT_AUTHORIZED: the reviewed targeted EFA driver unbind needs the reserved "
            "EFA_DRIVER_UNBIND native probe and an independent rebind deadman, not built this phase"
        ),
        lanes=("eks",),
        expected="DRIVER_UNBOUND -> targeted bind + plugin restart + snapshot + validation; not a reboot",
    ),
    deferred_variant(
        CASE_013,
        "link_inactive",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: active-port shortfall is read from /sys/class/infiniband, which "
            "is not env-repointable; it needs a whole-tree sysfs port-state fixture (not a single "
            "file bind) or a real port-down"
        ),
        lanes=("eks",),
        expected="LINK_INACTIVE -> diagnose first, no reboot",
    ),
    deferred_variant(
        CASE_013,
        "excess_device",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: EXCESS_DEVICE is driven by the active-device count from "
            "/sys/class/infiniband (not env-repointable); the PCI-root replay only moves "
            "pci_discovered"
        ),
        lanes=("eks",),
        expected="EXCESS_DEVICE -> diagnose only, no reboot",
    ),
    deferred_variant(
        CASE_013,
        "unknown_inventory",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: an ambiguous driver-bound/active reading needs an infiniband "
            "sysfs tree fixture the replay helper does not install"
        ),
        lanes=("eks",),
        expected="an unknown inventory shape is not authorized for reboot",
    ),
    deferred_variant(
        CASE_013,
        "bind_failure",
        proof_mode=LIVE_REPLAY,
        reason=(
            "NOT_AUTHORIZED: a bind that fails after an unbind needs the reserved "
            "EFA_DRIVER_UNBIND native path"
        ),
        lanes=("eks",),
        expected="a failed bind has its own receipt and is handled per the approved branch; no global efa unload",
    ),
    deferred_variant(
        CASE_013,
        "post_repair_failure",
        proof_mode=LIVE_REPLAY,
        reason=(
            "NOT_AUTHORIZED: a post-action validation failure needs the reserved "
            "EFA_DRIVER_UNBIND native path plus an injected post-repair failure"
        ),
        lanes=("eks",),
        expected="action succeeds but post-action validation fails -> independent receipt and escalation",
    ),
    # --- 014 Kubernetes allocatable under-report (no valid replay) --- #
    deferred_variant(
        CASE_014,
        "gpu_plugin",
        proof_mode=LIVE_REPLAY,
        reason=_PLUGIN_NATIVE,
        lanes=("eks", "hp"),
        expected="a real GPU device-plugin registration shortfall -> restart the correct plugin; new Pod UID; resources re-registered",
    ),
    deferred_variant(
        CASE_014,
        "efa_plugin",
        proof_mode=LIVE_REPLAY,
        reason=_PLUGIN_NATIVE,
        lanes=("eks", "hp"),
        expected="a real EFA device-plugin registration shortfall -> restart the correct plugin",
    ),
    deferred_variant(
        CASE_014,
        "one_sample",
        proof_mode=LIVE_REPLAY,
        reason=_PLUGIN_NATIVE,
        lanes=("eks",),
        expected="a single allocatable mismatch is not confirmed into an action",
    ),
    deferred_variant(
        CASE_014,
        "replacement_uid",
        proof_mode=LIVE_REPLAY,
        reason=_PLUGIN_NATIVE,
        lanes=("eks",),
        expected="after the plugin restart the new Pod UID and re-Ready are verified",
    ),
    deferred_variant(
        CASE_014,
        "expected_unknown",
        proof_mode=LIVE_REPLAY,
        reason=(
            "CAPABILITY_ABSENT: p5en.48xlarge has a known expected allocatable count; an unknown "
            "expected count needs an instance type absent from INSTANCE_ACCELERATOR_COUNTS"
        ),
        lanes=("eks",),
        expected="an unknown expected count is taken for forensics, not treated as a physical shortfall",
    ),
    deferred_variant(
        CASE_014,
        "mig_timeslicing",
        proof_mode=LIVE_REPLAY,
        reason="CAPABILITY_ABSENT: MIG / time-slicing is not configured on this site",
        lanes=("eks",),
        expected="MIG/time-slicing allocatable is not treated as a physical card shortfall",
    ),
    deferred_variant(
        CASE_014,
        "validation_failure",
        proof_mode=LIVE_REPLAY,
        reason=_PLUGIN_NATIVE,
        lanes=("eks",),
        expected="a post-restart validation failure escalates to reboot under a separate authorization",
    ),
    deferred_variant(
        CASE_014,
        "running_training",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: an allocatable mismatch while training runs needs a managed (scale) "
            "training attempt on the node (lane N territory), not built here"
        ),
        lanes=("eks", "hp"),
        expected="a mismatch while training runs keeps the training identity and valid progress",
    ),
]
