"""Shared fixtures and builders for split test shards."""

from __future__ import annotations

import functools
import re
from datetime import datetime, timezone

import pytest

from gpu_fault.models import RecoveryAction
from gpu_fault.policy import (
    ActionDisposition,
    GpuFaultPolicyEngine,
    XidEvent,
    load_sxid_policy,
    load_xid_policy,
)

NOW = datetime(2026, 7, 20, 2, 0, tzinfo=timezone.utc)

CATALOG_PRODUCT_COLUMNS = ("A100", "H100", "B100", "GB200")


@functools.lru_cache(maxsize=1)
def _pinned_catalogs():
    return load_xid_policy(), load_sxid_policy()


def _engine() -> GpuFaultPolicyEngine:
    """Fresh engine over the cached pinned catalogs.

    Parsing the two YAML catalogs costs ~0.23s, and the table-driven
    suites below build ~900 engines, so re-parsing per case dominated
    the whole run. The *engine* is still constructed per case, which is
    what matters: each case keeps its own idempotency cache, so a
    verdict cached by one case can never satisfy another.
    """

    xid_policy, sxid_policy = _pinned_catalogs()
    return GpuFaultPolicyEngine(xid_policy, sxid_policy)


def xid(code: int, *, event_id: str, observed_at: datetime = NOW, **kwargs) -> XidEvent:
    product = kwargs.pop("product", "H100")
    driver_branch = kwargs.pop("driver_branch", 575)
    cuda_version = kwargs.pop("cuda_version", "12.9")
    return XidEvent(
        event_id=event_id,
        cluster_id="cluster-a",
        node_id="node-a",
        gpu_uuid="GPU-a",
        product=product,
        driver_branch=driver_branch,
        cuda_version=cuda_version,
        observed_at=observed_at,
        xid=code,
        **kwargs,
    )


def _nvlink5_pattern_matches(value: int, pattern: str) -> bool:
    """Independent re-implementation of the 32-bit mask match.

    Deliberately written from the NVIDIA table's own semantics rather
    than by calling the engine's helper, so a bug in the engine cannot
    make the expectation agree with it.
    """

    if len(pattern) != 32:
        return False
    for index, expected in enumerate(reversed(pattern)):
        if expected == "-":
            continue
        if ((value >> index) & 1) != int(expected):
            return False
    return True


def _nvlink5_error_status_values(raw: str | None) -> list[int]:
    if raw is None:
        return [0]
    return [int(item.strip(), 16) for item in raw.split("/")]


def _nvlink5_error_status_matches(value: int, raw: str | None) -> bool:
    return raw is None or value in set(_nvlink5_error_status_values(raw))


def _nvlink5_expected_action(
    rules, xid_code: int, intr_info: int, error_status: int, is_v1: bool
) -> str:
    actions: set[str] = set()
    for rule in rules:
        if rule.xid != xid_code:
            continue
        pattern = rule.v1_pattern if is_v1 else rule.v2_pattern
        if not _nvlink5_pattern_matches(intr_info, pattern):
            continue
        if not _nvlink5_error_status_matches(error_status, rule.error_status):
            continue
        actions.add(rule.recovery_action)
        action2_pattern = rule.action2_v1_pattern if is_v1 else rule.action2_v2_pattern
        if (
            rule.action2
            and action2_pattern
            and _nvlink5_pattern_matches(intr_info, action2_pattern)
        ):
            actions.add(rule.action2)
    if "RESET_GPU" in actions:
        return "RESET_GPU"
    if "XID_154_EVAL" in actions:
        # No companion XID 154 in these cases, and the correlation
        # window is treated as closed, so the catalog default applies.
        return "RESTART_APP"
    if actions == {"IGNORE"}:
        return "IGNORE"
    return f"CONFLICT:{sorted(actions)}"


def _nvlink5_decode_cases():
    policy = _pinned_catalogs()[0]
    boundary = policy.nvlink5.driver_boundary
    cases = []
    for is_v1 in (True, False):
        branch = boundary - 5 if is_v1 else boundary
        for index, rule in enumerate(policy.nvlink5.decode_rules):
            pattern = rule.v1_pattern if is_v1 else rule.v2_pattern
            intr_info = int(pattern.replace("-", "0"), 2)
            for error_status in _nvlink5_error_status_values(rule.error_status):
                cases.append(
                    pytest.param(
                        index,
                        rule,
                        branch,
                        is_v1,
                        intr_info,
                        error_status,
                        id=(
                            f"xid{rule.xid}-{rule.subcode_name}"
                            f"-{'v1' if is_v1 else 'v2'}"
                            f"-{error_status:#010x}"
                        ),
                    )
                )
    return cases


_EXPECTED_DIRECT_ACTIONS = {
    "IGNORE": RecoveryAction.NO_ACTION,
    "RESTART_APP": RecoveryAction.RESTART_WORKLOAD,
    "RESET_GPU": RecoveryAction.RESET_GPU,
    "RESTART_BM": RecoveryAction.REBOOT_NODE,
}


def _expected_xid_outcome(rule, family: str):
    """Recompute the expected verdict from the pinned catalog alone.

    Assumes an event that carries full version evidence and no extra
    evidence (no registers, no UVM flag, no XID 154 label, no
    companions) -- the version gate and every workflow branch get their
    own dedicated tests below.

    Returns ``(official_action, disposition, action)``.
    """

    if family not in (rule.products or []):
        # Applicability is decided per rule, on that rule's own product
        # column: an action approved for GB200 must not fire on A100.
        return (rule.immediate_action, ActionDisposition.NOT_APPLICABLE, None)
    official = rule.immediate_action
    if rule.xid == 154:
        return ("XID_154", ActionDisposition.BLOCKED_MISSING_EVIDENCE, None)
    if official is None:
        return (None, ActionDisposition.MONITOR_ONLY, RecoveryAction.NO_ACTION)
    if official in _EXPECTED_DIRECT_ACTIONS:
        action = _EXPECTED_DIRECT_ACTIONS[official]
        return (
            official,
            ActionDisposition.MONITOR_ONLY
            if action is RecoveryAction.NO_ACTION
            else ActionDisposition.EXECUTABLE,
            action,
        )
    if official == "WORKFLOW_XID_45":
        # Solo, because no companion is supplied.
        return ("RESTART_FM", ActionDisposition.EXECUTABLE, None)
    if official == "WORKFLOW_XID_48":
        return ("RESET_GPU", ActionDisposition.EXECUTABLE, RecoveryAction.RESET_GPU)
    if official == "WORKFLOW_NVLINK_ERR":
        return (
            "WORKFLOW_NVLINK_ERR",
            ActionDisposition.BLOCKED_MISSING_EVIDENCE,
            RecoveryAction.STOP_WORKLOAD,
        )
    if official in {"WORKFLOW_NVLINK5_ERR", "CHECK_UVM"}:
        return (official, ActionDisposition.BLOCKED_MISSING_EVIDENCE, None)
    if official in {"CONTACT_SUPPORT", "CHECK_MECHANICALS", "UPDATE_SWFW"}:
        return (official, ActionDisposition.EXECUTABLE, None)
    return (official, ActionDisposition.BLOCKED_WORKFLOW, None)


def _catalog_matrix_cases():
    """172 catalog rules x 4 catalog product columns."""

    return [
        pytest.param(rule, family, id=f"xid{rule.xid}-{family}")
        for rule in _pinned_catalogs()[0].catalog_rules
        for family in CATALOG_PRODUCT_COLUMNS
    ]


def _version_gated_rules():
    """Catalog rules whose linkage text declares a minimum version."""

    cases = []
    for rule in _pinned_catalogs()[0].catalog_rules:
        if not rule.products:
            continue
        linkage = rule.xid154_linkage or ""
        driver = re.search(r"GPU driver R(\d+)", linkage)
        cuda = re.search(r"CUDA (\d+\.\d+)", linkage)
        if driver is None and cuda is None:
            continue
        cases.append(
            pytest.param(
                rule,
                int(driver.group(1)) if driver else None,
                cuda.group(1) if cuda else None,
                id=f"xid{rule.xid}",
            )
        )
    return cases
