"""Temporary test packages and kernel modules of one variant: the shared step.

Some variants need a host tool or kernel module the node image does not ship
(the AL2023 EKS GPU AMI has no ``tc``, ``smartctl`` or ``mkfs.vfat``).
``ensure_dependencies`` is the one runner-side step over
``probes/general_live_deps_probe.py`` every category uses for that: 009
``tcp_only`` (``tc`` + ``sch_netem``), the 007 SMART replay variants
(``smartctl`` -> ``smartmontools``) and 004 ``probe_error`` (``mkfs.vfat`` ->
``dosfstools``).

Whatever the node has is used as found. A missing tool is installed only
behind ``TEMP_TEST_PACKAGE_INSTALL`` -- exactly the reviewed package of the
probe's ``TOOL_PACKAGES``, after a ``dnf --assumeno`` preview that must install
nothing else (no kernel, upgrade, downgrade, removal or other dependency;
otherwise NOT_RUN(FIXTURE_NOT_READY) with the transaction in the evidence).
Without the token the variant stays NOT_RUN(CAPABILITY_ABSENT) and its detail
names the token that would enable the install. A module that is not loaded is
loaded only behind ``KERNEL_MODULE_LOAD``. The probe arms a deadman and records
the installed NEVRAs and loaded modules on the node *before* the change; the
runner registers the probe's ``cleanup`` on the LIFO stack before the call, so
it runs after every fixture that used the tool is gone and removes exactly what
this run installed. The product's collectors are never restarted for an
install: they look a tool up on every tick (``shutil.which`` in
``collectors/host/system_metrics.py`` ``_smart`` and ``host/bmc.py`` ``_bmc``),
and the host collector unit's ``ProtectSystem=strict`` is a read-only view of
the live ``/usr``, so the next tick sees the new binary.

After a successful step the survey's ``tools`` say what the node has now, so a
later ``require_tools`` of the same category sees the installed tool.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun
from scripts.e2e.regional.general_live_registry import NOT_RUN_REASONS
from scripts.e2e.regional.probes import general_live_deps_probe as deps_probe
from scripts.e2e.regional.regional_commands import RegionalFixtureError

PACKAGE_APPROVAL = "TEMP_TEST_PACKAGE_INSTALL"
MODULE_APPROVAL = "KERNEL_MODULE_LOAD"
#: The probe's install budget (dnf metadata plus one small package) and its
#: cleanup's (a previewed ``dnf remove``).
DEPS_TIMEOUT = 900
DEPS_CLEANUP_TIMEOUT = 600
#: The reviewed tool -> package and module -> package maps are the probe's:
#: one allow-list, reviewed once, shipped to the node.
TOOL_PACKAGES = deps_probe.TOOL_PACKAGES
MODULE_PACKAGES = deps_probe.MODULE_PACKAGES
#: The receipt fields a variant keeps in its result.
RECEIPT_KEYS = ("changed", "installed", "modules_loaded_by_run", "transaction")


def deps_deadman_seconds(fixture_deadman: int) -> int:
    """When the package deadman fires: after the fixture's own deadman.

    The package deadman is armed before the install (at most ``DEPS_TIMEOUT``)
    and the fixture's deadman after it, so the offset covers the install too:
    the package goes last, once nothing uses the tool or module any more.
    """

    return min(
        deps_probe.MAX_DEADMAN_SECONDS, int(fixture_deadman) + DEPS_TIMEOUT + 300
    )


def checked_cleanup(ctx: VariantContext, probe: str, *arguments: str) -> Any:
    """Run a probe cleanup and fail the cleanup stack unless it proved clean."""

    timeout = DEPS_CLEANUP_TIMEOUT if probe == "deps" else 180
    value = getattr(ctx, probe)("cleanup", *arguments, timeout=timeout)
    if value.get("clean") is not True:
        raise RegionalFixtureError(
            f"the {probe} cleanup left residue: {value.get('residue')}"
        )
    return value


def not_run(refused: dict[str, Any], default: str) -> VariantNotRun:
    """The probe's refusal as the variant's NOT_RUN (a refusal is never an error)."""

    reason = str(refused.get("reason") or "")
    return VariantNotRun(
        reason if reason in NOT_RUN_REASONS else "FIXTURE_NOT_READY",
        str(refused.get("detail") or default),
    )


def dependency_summary(result: dict[str, Any]) -> dict[str, Any]:
    """What a variant's result keeps of the dependency step's receipt."""

    return {key: result.get(key) for key in RECEIPT_KEYS}


def ensure_dependencies(
    ctx: VariantContext,
    survey: dict[str, Any],
    *,
    tools: Sequence[str] = (),
    modules: Sequence[str] = (),
    deadman_seconds: int,
    evidence: str = "dependencies",
) -> dict[str, Any]:
    """``tools`` usable and ``modules`` loaded: as found, or for this run only.

    Returns the probe's receipt (``installed`` NEVRAs, ``modules_loaded_by_run``,
    ``transaction``, the tools' and modules' state after the step) with
    ``changed``; ``changed`` is ``False`` when the node had everything, and then
    nothing was touched, recorded or registered for cleanup.
    """

    unknown = [name for name in tools if name not in TOOL_PACKAGES]
    unknown += [name for name in modules if name not in MODULE_PACKAGES]
    if unknown:
        raise RegionalFixtureError(
            f"no reviewed package for {', '.join(unknown)}; the deps probe's "
            "allow-list is the only source of temporary test packages"
        )
    tools_state = survey.setdefault("tools", {})
    modules_state = survey.get("modules") or {}
    approvals = ctx.settings.approvals
    may_install = PACKAGE_APPROVAL in approvals
    missing = [name for name in tools if not tools_state.get(name)]
    to_load = [
        name for name in modules if not (modules_state.get(name) or {}).get("loaded")
    ]
    if missing and not may_install:
        packages = ", ".join(TOOL_PACKAGES[name] for name in missing)
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            f"the node lacks {', '.join(missing)} ({PACKAGE_APPROVAL} installs "
            f"{packages} for one run)",
        )
    for name in to_load:
        if not (modules_state.get(name) or {}).get("available") and not may_install:
            raise VariantNotRun(
                "CAPABILITY_ABSENT",
                f"kernel module {name} is not available ({PACKAGE_APPROVAL} "
                "installs the running kernel's module package)",
            )
    if to_load and MODULE_APPROVAL not in approvals:
        raise VariantNotRun(
            "NOT_AUTHORIZED",
            f"kernel module {', '.join(to_load)} is not loaded; needs "
            f"{MODULE_APPROVAL}",
        )
    if not (missing or to_load):
        return {"changed": False, "installed": [], "modules_loaded_by_run": []}
    arguments = ["ensure"]
    for name in tools:
        arguments += ["--tool", name]
    for name in modules:
        arguments += ["--module", name]
    arguments += ["--deadman-seconds", str(int(deadman_seconds))]
    if may_install:
        arguments.append("--allow-install")
    if MODULE_APPROVAL in approvals:
        arguments.append("--allow-module-load")
    require_action_time(DEPS_TIMEOUT)
    # Registered before the change: a half-finished install is undone too, and
    # LIFO puts the removal after every fixture registered later.
    ctx.cleanup.push("deps", lambda: checked_cleanup(ctx, "deps"))
    result = ctx.deps(*arguments, timeout=DEPS_TIMEOUT)
    ctx.evidence.write(evidence, result)
    if result.get("refused"):
        raise not_run(result["refused"], "the dependency step refused")
    present = result.get("tools") or {}
    states = result.get("modules") or {}
    unusable = [name for name in tools if not present.get(name)]
    unusable += [
        name
        for name in modules
        if not (
            (states.get(name) or {}).get("loaded")
            or (states.get(name) or {}).get("builtin")
        )
    ]
    if unusable:
        raise RegionalFixtureError(
            f"{', '.join(unusable)} still unusable after the dependency step: {result}"
        )
    # The survey now says what the node has: the category's later
    # ``require_tools`` sees the installed tool.
    tools_state.update({name: True for name in tools})
    return {"changed": True, **result}


__all__ = [
    "DEPS_CLEANUP_TIMEOUT",
    "DEPS_TIMEOUT",
    "MODULE_APPROVAL",
    "MODULE_PACKAGES",
    "PACKAGE_APPROVAL",
    "RECEIPT_KEYS",
    "TOOL_PACKAGES",
    "checked_cleanup",
    "dependency_summary",
    "deps_deadman_seconds",
    "ensure_dependencies",
    "not_run",
]
