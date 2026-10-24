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
NOT_RUN_REASONS = (
    "VENDOR_LAB_ONLY",
    "CAPABILITY_ABSENT",
    "FIXTURE_NOT_READY",
    "NOT_AUTHORIZED",
    "SCALE",
)

#: ``--approve`` tokens. Each names one dangerous branch; a variant that needs
#: one is NOT_RUN(NOT_AUTHORIZED) without it, whatever else the plan says.
APPROVALS: dict[str, str] = {
    "HOST_MEMORY_PRESSURE": (
        "hold host memory use at about 96 % in a capped test cgroup "
        "(MemoryMax, no swap, OOMPolicy=kill)"
    ),
    "KERNEL_MODULE_LOAD": (
        "load a kernel module the node does not have loaded (dm_delay, veth, "
        "dummy, sch_netem); unloaded again in cleanup"
    ),
    "ENABLE_NODE_LOG_COLLECTOR": (
        "runtime-enable (/run only) and start the installed but disabled node "
        "log collector for the silence variant; disabled again in cleanup"
    ),
    "WATCHER_NETWORK_ISOLATION": (
        "reject the completion watcher Pod's TCP/443 to the control plane on its "
        "host with an automatic rollback timer"
    ),
    "CONTROL_PLANE_RESTART": (
        "delete one cluster-executor or completion-watcher Pod of the GPU "
        "cluster during an open restart claim (worker restart)"
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
)
#: The categories phase 1 implements (``cases`` rows of the coverage file).
PHASE_ONE_CATEGORIES = (1, 3, 4, 6, 8, 9, 25, 28, 29, 31)


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
    unknown = [item for item in spec.approvals if item not in APPROVALS]
    if unknown:
        raise RegistryError(f"{spec.key}: unknown approvals {unknown}")
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
    if spec.proof_mode not in {NATIVE, OBSERVATION} and not spec.deferred:
        raise RegistryError(
            f"{spec.key}: only native and observation variants run in phase 1"
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
    return build_registry(specs)


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
