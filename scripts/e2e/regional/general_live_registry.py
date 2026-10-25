"""The variant catalog of the GF-GENERAL-LIVE live acceptance runner.

``testcases/general-live-coverage.yaml`` lists the required variants of every
general fault category; ``docs/components/general-fault-live-acceptance.md``
specifies them. This module binds each variant of an implemented category to
how ``run_general_live.py`` executes it -- or to the explicit reason it is not
executable in this phase -- so a variant can never silently drop out of the
denominator (spec section 9: specification, fixture readiness, execution and
pass rates are reported separately).

Every variant is planned and executed on its own guard key
``<case>.<variant>`` with the case confirmation ``GENERALLIVE<NNN>_EXECUTE``;
dangerous branches additionally need an ``--approve`` token bound into the
plan. A variant whose ``deferred`` reason is set is recorded ``NOT_RUN`` with
that reason and never touches the cluster.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
COVERAGE_PATH = ROOT / "testcases" / "general-live-coverage.yaml"
CASE_PATTERN = re.compile(r"^GF-GENERAL-LIVE-(\d{3})$")
VARIANT_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

NATIVE = "native"
OBSERVATION = "observation"
LIVE_REPLAY = "live-replay"
VENDOR_LAB = "vendor-lab"
CAPABILITY = "capability-absent"
PROOF_MODES = (NATIVE, OBSERVATION, LIVE_REPLAY, VENDOR_LAB, CAPABILITY)
LANES = ("eks", "hp")
RISKS = ("L", "M", "H")
#: Flags of the campaign list file: P lane lock only, S lane + site lock,
#: X every lock, Q at least 600 s after the lane's previous case.
FLAG_PATTERN = re.compile(r"^[PSX]Q?$")

VERDICTS = ("PASS", "FAIL", "NOT_RUN", "BLOCKED", "INCONCLUSIVE")
#: The ``deferred`` reason prefixes. ``COVERED_BY_REGIONAL: <case ids>`` names
#: GF-REGIONAL cases that already PASSED live on this site with the very
#: assertions the variant would make (the same injection on the same product
#: path), so the variant gets no fixture of its own and stays NOT_RUN with the
#: cross-reference; its registry entry keeps lanes and expectations so it can
#: be re-enabled later.
NOT_RUN_REASONS = (
    "VENDOR_LAB_ONLY",
    "CAPABILITY_ABSENT",
    "FIXTURE_NOT_READY",
    "NOT_AUTHORIZED",
    "SCALE",
    "COVERED_BY_REGIONAL",
)

#: ``--approve`` tokens. Each names one dangerous branch; a variant that needs
#: one is NOT_RUN(NOT_AUTHORIZED) without it, whatever else the plan says.
APPROVALS: dict[str, str] = {
    "HOST_MEMORY_PRESSURE": (
        "hold host memory use at about 96 % in a capped test cgroup "
        "(MemoryMax, no swap, OOMPolicy=kill), or exhaust a small capped test "
        "cgroup (at most 1 % of MemTotal) so the kernel OOM-kills the allocator "
        "inside it while host memory use stays normal (002 cgroup_only)"
    ),
    "KERNEL_MODULE_LOAD": (
        "load a kernel module the node does not have loaded (dm_delay, veth, "
        "dummy, sch_netem, vfat for the 004 probe_error FAT16 carrier, nfsv4 for "
        "the 005 test export mount; the dm-error target is built into dm_mod); "
        "unloaded again in cleanup"
    ),
    "ENABLE_NODE_LOG_COLLECTOR": (
        "runtime-enable (/run only) and start the installed but disabled node "
        "log collector for the silence variant; disabled again in cleanup"
    ),
    "WATCHER_NETWORK_ISOLATION": (
        "reject the completion watcher Pod's TCP to the control-plane endpoint "
        "on its host (FORWARD rules matching that Pod's IP only, tcp-reset) with "
        "an automatic rollback timer under the 600 s coverage horizon"
    ),
    "NODE_AGENT_NETWORK_ISOLATION": (
        "reject TCP to one GPU node's Node Agent listening port on that host "
        "(the channel the executor pushes node actions to; heartbeats, "
        "collectors and kubelet untouched) with an automatic rollback timer, so "
        "the hung-triage evidence workflow meets an unreachable node (011 "
        "unreachable_node)"
    ),
    "CONTROL_PLANE_RESTART": (
        "delete one cluster-executor or completion-watcher Pod of the GPU "
        "cluster during an open restart claim (worker restart)"
    ),
    "DECLARE_WARM_SPARE": (
        "declare one warm spare GPU node through gpu-fault-admin config spare "
        "(sets the spare label and cordons it); released and uncordoned in cleanup"
    ),
    "SPARE_KUBELET_STOP": (
        "stop kubelet on the declared warm spare to drive it NotReady, with a "
        "host-side automatic restart timer; restarted in cleanup"
    ),
    # Lane K (phase 2) kernel-log source replay and EFA inventory.
    "KERNEL_SOURCE_REPLAY": (
        "repoint the kernel collector's GPU_FAULT_KMSG_PATH at a reviewed "
        "fixture file for a bounded window (it stops reading real /dev/kmsg "
        "meanwhile); restored by cleanup and a deadman"
    ),
    "EFA_DRIVER_UNBIND": (
        "unbind the driver of one allow-listed EFA data function on a dedicated "
        "node and rebind it, with an independent rebind deadman"
    ),
    "DEVICE_PLUGIN_RESTART": (
        "delete the GPU or EFA device-plugin Pod on the test node so kubelet "
        "de-registers its allocatable, then let the DaemonSet re-register it"
    ),
    "GPU_RESET": (
        "let the product's RESET_GPU chain (quiesce, verify no GPU clients, one "
        "nvidia-smi --gpu-reset of the target GPU, restore GPU services, "
        "validate) run on the idle test node for a replayed row-remap-pending "
        "reading, or -- 029 hardware_chain_continues -- on the test node running "
        "the managed test job for a reviewed kernel XID 46 (the product cordons "
        "and stops that job first; its restart budget is already exhausted)"
    ),
    "GPU_FULL_FABRIC_RESET": (
        "let the product's RESET_ALL_GPUS_NVSWITCHES chain (quiesce incl. Fabric "
        "Manager, one reset of every GPU and NVSwitch, Fabric Manager restart, "
        "GPU and fabric validation) run on the exclusive test node for a "
        "replayed NVLink fabric state"
    ),
    "NODE_REBOOT": (
        "let the product's post-validation-failure ladder reboot the exclusive "
        "test node through the provider (RESTART_NODE) after a failed "
        "post-reset fabric validation"
    ),
    "GPU_FIELD_DIAGNOSTIC": (
        "let the product run the node's configured NVIDIA memory field "
        "diagnostic on the drained test GPU (only when the Node Agent allows "
        "RUN_FIELD_DIAGNOSTIC)"
    ),
    "DCGM_XID_GAUGE": (
        "replay the DCGM last-XID gauge (DCGM_FI_DEV_XID_ERRORS) as XID 32 "
        "(catalog RESTART_APP, monitor-only on an idle node) on one test GPU; "
        "the product turns it into a trusted XID event"
    ),
    # The shared host-collector source replay (general_live_source_replay.py):
    # phase-2 host/device variants 002/007/010/011/012/013/015/032.
    "HOST_SOURCE_REPLAY": (
        "replace one reviewed raw reading inside the host collector unit's "
        "private environment for a bounded window, through a /run drop-in on "
        "gpu-fault-host-collector.service only: a procfs/sysfs file or whole "
        "tree bound read-only into its mount namespace (/proc/meminfo, an EFA "
        "port's hw_counters/counters/state/phys_state, /sys/class/infiniband, "
        "/proc/driver/nvidia and its version file), a procfs file or tool binary "
        "made unopenable there (an unopenable device node bound over it), an "
        "expected GPU/EFA count override, or a reviewed "
        "pass-through wrapper first on its PATH (smartctl, ipmitool, ethtool, "
        "the host collector's nvidia-smi queries) that rewrites only the plan's "
        "allow-listed fields; no host binary is replaced, no global mount is "
        "made, the Node Agent is never touched; removed by cleanup and a deadman"
    ),
    # GF-GENERAL-LIVE-032 missing_version: the version the NVLink5 decoder
    # gates on is the one the KERNEL collector's own CollectorContext stamps.
    "KERNEL_COLLECTOR_SOURCE_REPLAY": (
        "extend the host source replay's command-output mode to "
        "gpu-fault-kernel-collector.service for its nvidia-smi driver-version "
        "discovery query only (--query-gpu=driver_version, "
        "collectors/gpu/discovery.py): the reviewed pass-through wrapper first "
        "on that unit's PATH answers that one query with a reviewed failure or "
        "empty output so the restarted collector's CollectorContext discovers "
        "no driver branch and stamps none on its XID events, "
        "GPU_FAULT_DRIVER_BRANCH is unset in that unit's environment for the "
        "window (the installer writes it empty by default) so no configured "
        "branch stands in, and every other nvidia-smi call (the GPU identity "
        "query included) passes through to the real binary; no file bind, "
        "tree, inaccessible path, count override or PCI root on that unit; "
        "the unit is restarted into the window and restarted back onto the "
        "real binary by cleanup and a deadman; the host collector and the "
        "Node Agent are never touched by this token"
    ),
    # GF-GENERAL-LIVE-009 tcp_only, the 007 SMART replay variants and 004
    # probe_error: the AL2023 EKS node image ships no tc, smartctl or mkfs.vfat.
    "TEMP_TEST_PACKAGE_INSTALL": (
        "install, for this run only, a test package the node lacks with dnf, "
        "from the reviewed allow-list of general_live_deps_probe.py: iproute-tc "
        "when tc is absent (009 tcp_only), smartmontools when smartctl is absent "
        "(the 007 SMART replay variants), dosfstools when mkfs.vfat is absent "
        "(004 probe_error), kernel-modules-extra-$(uname -r) (the running "
        "kernel's module package) when sch_netem cannot be loaded; refused "
        "unless the previewed dnf transaction installs exactly those packages "
        "(no kernel, upgrade, downgrade, removal or other dependency); the "
        "collectors are not restarted for it (they look a tool up every tick); "
        "cleanup and a deadman dnf-remove exactly the packages this run "
        "installed and nothing else"
    ),
}

#: Workflow operations a diagnostic-only chain must never contain.
HARDWARE_OPERATIONS = frozenset(
    {
        "RESET_GPU",
        "RESET_ALL_GPUS_NVSWITCHES",
        "RESTART_NODE",
        "REPLACE_NODE",
        "RESTART_VM",
        "RESTART_FABRIC_MANAGER",
        "REMEDIATE_DRIVER",
        "REMEDIATE_EFA_DRIVER",
        "UPDATE_SOFTWARE_FIRMWARE",
        "RUN_FIELD_DIAGNOSTIC",
        "RECOVER_DRAIN_P2P",
        "QUIESCE_GPU_SERVICES",
    }
)


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class VariantSpec:
    """One required variant: what is injected, what the product must do."""

    case_id: str
    variant: str
    proof_mode: str
    lanes: tuple[str, ...]
    risk: str
    injection: str
    expected: str
    cleanup: str
    max_minutes: int
    flags: str = "S"
    approvals: tuple[str, ...] = ()
    #: Tokens that only unlock an extra branch: without one the variant still
    #: runs and that branch's prerequisite stays NOT_RUN(CAPABILITY_ABSENT).
    optional_approvals: tuple[str, ...] = ()
    expected_outcome: str = "PASS"
    product_gap: str = ""
    deferred: str = ""
    execute: Callable[[Any], Any] | None = None
    preflight: Callable[[Any], list[str]] | None = None

    @property
    def key(self) -> str:
        return f"{self.case_id}.{self.variant}"

    @property
    def number(self) -> int:
        match = CASE_PATTERN.fullmatch(self.case_id)
        if match is None:
            raise RegistryError(f"not a GF-GENERAL-LIVE case: {self.case_id}")
        return int(match.group(1))

    @property
    def confirmation(self) -> str:
        return f"GENERALLIVE{self.number:03d}_EXECUTE"

    @property
    def implemented(self) -> bool:
        return not self.deferred and self.execute is not None

    def describe(self) -> dict[str, Any]:
        """The plan's view of the variant (no callables)."""

        return {
            "key": self.key,
            "case_id": self.case_id,
            "variant": self.variant,
            "proof_mode": self.proof_mode,
            "lanes": list(self.lanes),
            "risk": self.risk,
            "flags": self.flags,
            "approvals": list(self.approvals),
            "optional_approvals": list(self.optional_approvals),
            "injection": self.injection,
            "expected": self.expected,
            "cleanup": self.cleanup,
            "max_minutes": self.max_minutes,
            "expected_outcome": self.expected_outcome,
            "product_gap": self.product_gap,
            "deferred": self.deferred,
            "implemented": self.implemented,
            "confirmation": self.confirmation,
        }


def deferred_variant(
    case_id: str,
    variant: str,
    *,
    proof_mode: str,
    reason: str,
    lanes: tuple[str, ...] = ("eks",),
    expected: str = "",
) -> VariantSpec:
    """A required variant this phase records as NOT_RUN(``reason``)."""

    return VariantSpec(
        case_id=case_id,
        variant=variant,
        proof_mode=proof_mode,
        lanes=lanes,
        risk="L",
        injection="not executed in this phase",
        expected=expected or "see the case specification",
        cleanup="none",
        max_minutes=1,
        flags="P",
        deferred=reason,
    )


#: Category modules, in case order. Each exports ``VARIANTS``.
CATEGORY_MODULES = (
    "scripts.e2e.regional.general_live_host",
    "scripts.e2e.regional.general_live_storage",
    "scripts.e2e.regional.general_live_network",
    "scripts.e2e.regional.general_live_utilization",
    "scripts.e2e.regional.general_live_attempts",
    "scripts.e2e.regional.general_live_collectors",
    "scripts.e2e.regional.general_live_memory",
    "scripts.e2e.regional.general_live_shared",
    "scripts.e2e.regional.general_live_devices",
    "scripts.e2e.regional.general_live_smart",
    "scripts.e2e.regional.general_live_bmc",
    "scripts.e2e.regional.general_live_efa",
    "scripts.e2e.regional.general_live_nccl",
    "scripts.e2e.regional.general_live_nccl_fr",
    "scripts.e2e.regional.general_live_nccl_pair",
    "scripts.e2e.regional.general_live_nccl_roles",
    "scripts.e2e.regional.general_live_progress",
    "scripts.e2e.regional.general_live_spares",
    "scripts.e2e.regional.general_live_kernel",
    "scripts.e2e.regional.general_live_applogs",
    "scripts.e2e.regional.general_live_unknown",
    "scripts.e2e.regional.general_live_inventory",
    "scripts.e2e.regional.general_live_efa_pci",
    "scripts.e2e.regional.general_live_device_plugin",
    "scripts.e2e.regional.general_live_efa_driver",
    "scripts.e2e.regional.general_live_gpu_memory",
    "scripts.e2e.regional.general_live_gpu_links",
    "scripts.e2e.regional.general_live_gpu_fabric",
    "scripts.e2e.regional.general_live_gpu_composite",
    "scripts.e2e.regional.general_live_gpu_environment",
    "scripts.e2e.regional.general_live_gpu_fields",
    "scripts.e2e.regional.general_live_gpu_identity",
    "scripts.e2e.regional.general_live_attempt_hardware",
)
#: The categories phase 1 implements (``cases`` rows of the coverage file).
PHASE_ONE_CATEGORIES = (1, 3, 4, 6, 8, 9, 25, 28, 29, 31)
#: The categories phase 2 adds: 002 memory, 005 shared storage, 007/010/012/015
#: devices, 011 NCCL hung, 013/014 EFA and allocatable inventory, 016 kernel log,
#: 017 application logs, 019-022/024/026 DCGM field and GPU inventory replay
#: (025's field variants live in general_live_gpu_fields), 027 training
#: progress, 030 warm spare, 032 unknown error.
PHASE_TWO_CATEGORIES = (
    2,
    5,
    7,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    21,
    22,
    24,
    26,
    27,
    30,
    32,
)


def coverage_rows(path: Path = COVERAGE_PATH) -> dict[str, list[str]]:
    """``{case_id: [variants]}`` for the single-case GF-GENERAL-LIVE rows."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = document.get("rows") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise RegistryError("general live coverage has no rows")
    result: dict[str, list[str]] = {}
    for row in rows:
        cases = row.get("cases") or []
        if len(cases) == 1 and CASE_PATTERN.fullmatch(str(cases[0])):
            result[str(cases[0])] = [str(item) for item in row.get("variants") or []]
    return result


def validate_spec(spec: VariantSpec) -> None:
    if CASE_PATTERN.fullmatch(spec.case_id) is None:
        raise RegistryError(f"bad case id {spec.case_id!r}")
    if VARIANT_PATTERN.fullmatch(spec.variant) is None:
        raise RegistryError(f"bad variant name {spec.variant!r}")
    if spec.proof_mode not in PROOF_MODES:
        raise RegistryError(f"{spec.key}: unknown proof mode {spec.proof_mode!r}")
    if not spec.lanes or any(lane not in LANES for lane in spec.lanes):
        raise RegistryError(f"{spec.key}: unknown lane in {spec.lanes!r}")
    if spec.risk not in RISKS:
        raise RegistryError(f"{spec.key}: unknown risk {spec.risk!r}")
    if FLAG_PATTERN.fullmatch(spec.flags) is None:
        raise RegistryError(f"{spec.key}: bad list flags {spec.flags!r}")
    unknown = [
        item
        for item in (*spec.approvals, *spec.optional_approvals)
        if item not in APPROVALS
    ]
    if unknown:
        raise RegistryError(f"{spec.key}: unknown approvals {unknown}")
    both = sorted(set(spec.approvals) & set(spec.optional_approvals))
    if both:
        raise RegistryError(f"{spec.key}: approvals both required and optional {both}")
    if spec.expected_outcome not in {"PASS", "FAIL"}:
        raise RegistryError(f"{spec.key}: expected outcome must be PASS or FAIL")
    if spec.expected_outcome == "FAIL" and not spec.product_gap:
        raise RegistryError(f"{spec.key}: an expected FAIL must cite its product gap")
    if spec.deferred:
        reason = spec.deferred.split(":", 1)[0]
        if reason not in NOT_RUN_REASONS:
            raise RegistryError(f"{spec.key}: deferred reason {reason!r} is unknown")
        if spec.execute is not None:
            raise RegistryError(f"{spec.key}: a deferred variant has no executor")
    elif spec.execute is None:
        raise RegistryError(f"{spec.key}: an executable variant needs execute()")
    if spec.proof_mode not in {NATIVE, OBSERVATION, LIVE_REPLAY} and not spec.deferred:
        raise RegistryError(
            f"{spec.key}: only native, observation and live-replay variants are executable"
        )
    if not 1 <= spec.max_minutes <= 180:
        raise RegistryError(f"{spec.key}: max_minutes is outside 1..180")


def build_registry(
    specs: Iterable[VariantSpec],
    *,
    rows: Mapping[str, list[str]] | None = None,
    categories: Iterable[int] = PHASE_ONE_CATEGORIES,
) -> dict[str, VariantSpec]:
    """Validate ``specs`` against the coverage rows of ``categories``."""

    registry: dict[str, VariantSpec] = {}
    for spec in specs:
        validate_spec(spec)
        if spec.key in registry:
            raise RegistryError(f"duplicate variant {spec.key}")
        registry[spec.key] = spec
    expected_rows = coverage_rows() if rows is None else rows
    expected = {
        f"GF-GENERAL-LIVE-{number:03d}.{variant}"
        for number in categories
        for variant in expected_rows.get(f"GF-GENERAL-LIVE-{number:03d}", [])
    }
    missing = sorted(expected - set(registry))
    extra = sorted(set(registry) - expected)
    if missing or extra:
        raise RegistryError(
            f"registry differs from the coverage rows: missing={missing}, extra={extra}"
        )
    return registry


def load_registry() -> dict[str, VariantSpec]:
    specs: list[VariantSpec] = []
    for name in CATEGORY_MODULES:
        module = importlib.import_module(name)
        specs.extend(getattr(module, "VARIANTS"))
    return build_registry(specs, categories=PHASE_ONE_CATEGORIES + PHASE_TWO_CATEGORIES)


def select(
    registry: Mapping[str, VariantSpec], case_id: str, variant: str
) -> VariantSpec:
    key = f"{case_id}.{variant}"
    spec = registry.get(key)
    if spec is None:
        known = sorted(
            item.variant for item in registry.values() if item.case_id == case_id
        )
        if not known:
            raise RegistryError(f"{case_id} is not implemented by this runner")
        raise RegistryError(f"{case_id} has no variant {variant!r}; known: {known}")
    return spec


__all__ = [
    "APPROVALS",
    "CAPABILITY",
    "CATEGORY_MODULES",
    "HARDWARE_OPERATIONS",
    "LANES",
    "LIVE_REPLAY",
    "NATIVE",
    "NOT_RUN_REASONS",
    "OBSERVATION",
    "PHASE_ONE_CATEGORIES",
    "PHASE_TWO_CATEGORIES",
    "RegistryError",
    "VENDOR_LAB",
    "VERDICTS",
    "VariantSpec",
    "build_registry",
    "coverage_rows",
    "deferred_variant",
    "load_registry",
    "select",
    "validate_spec",
]
