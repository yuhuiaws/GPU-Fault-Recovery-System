"""Security regressions for the policy engine.

Each test here pins a fail-closed guard that a client holding only a cluster
token could otherwise bypass by shaping the event payload.
"""

from __future__ import annotations

import pytest

from gpu_fault.models import RecoveryAction
from gpu_fault.policy import (
    ActionDisposition,
    Containment,
    GpuFaultPolicyEngine,
    SxidClassification,
    SxidLinkScope,
)
from tests._builders import build_sxid_event
from tests.policy._support import NOW

# A code that is absent from the pinned NVIDIA Fabric Manager SXID table.
UNKNOWN_SXID = 99999


def _fatal_trunk_sxid(code: int, classification_source: str):
    """A fully-evidenced fatal trunk SXID: only the code is unknown."""
    return build_sxid_event(
        f"sxid-unknown-fatal-{code}-{classification_source}",
        NOW,
        code,
        SxidClassification.FATAL,
        classification_source,
        link_scope=SxidLinkScope.TRUNK,
        link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
        fabric_partition="fabric-a",
        participating_gpu_uuids=["GPU-a", "GPU-b"],
    )


@pytest.mark.parametrize(
    "classification_source",
    [
        "NVIDIA_FABRIC_MANAGER",
        "NVIDIA_FABRIC_MANAGER_CATALOG",
        "NVIDIA_FABRIC_MANAGER_RUNTIME",
    ],
)
def test_unknown_fatal_sxid_cannot_authorize_full_fabric_reset(
    classification_source: str,
) -> None:
    """H-9: an uncatalogued code claiming FATAL must fail closed.

    Before the fix, an unknown SXID with ``classification=FATAL`` and a
    trusted TRUNK scope fell through to the runtime terminal branch and was
    granted an EXECUTABLE ``RESET_ALL_GPUS_AND_NVSWITCHES`` with STOP_WORKLOAD
    and MARK_UNSCHEDULABLE pre-actions, while the ALWAYS_FATAL twin was blocked.
    """
    engine = GpuFaultPolicyEngine()
    assert engine.sxid_catalog_rule(UNKNOWN_SXID) is None

    decision = engine.evaluate_sxid(
        _fatal_trunk_sxid(UNKNOWN_SXID, classification_source)
    )

    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.action is None
    assert decision.pre_actions == []
    assert decision.safety_action is RecoveryAction.QUARANTINE
    assert decision.official_action is None
    assert decision.investigatory_action == "VERIFY_SXID_LOG_INTEGRITY"
    assert decision.containment is Containment.UNKNOWN
    assert decision.requires_operator, decision
    assert f"SXID {UNKNOWN_SXID} claims Fatal" in decision.reasons[0]
    assert "absent from pinned NVIDIA" in decision.reasons[0]


def test_unknown_fatal_sxid_with_unknown_scope_still_fails_on_the_catalog() -> None:
    """The catalog check must come before any scope-based reasoning."""
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            "sxid-unknown-fatal-no-scope",
            NOW,
            UNKNOWN_SXID,
            SxidClassification.FATAL,
            "NVIDIA_FABRIC_MANAGER_RUNTIME",
        )
    )

    assert decision.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert decision.action is None
    assert decision.safety_action is RecoveryAction.QUARANTINE
    assert decision.investigatory_action == "VERIFY_SXID_LOG_INTEGRITY"
    assert "absent from pinned NVIDIA" in decision.reasons[0]


def test_unknown_non_fatal_sxid_remains_monitor_only() -> None:
    """The guard is scoped to destructive severities; informational stays."""
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        build_sxid_event(
            "sxid-unknown-non-fatal",
            NOW,
            UNKNOWN_SXID,
            SxidClassification.NON_FATAL,
            "NVIDIA_FABRIC_MANAGER_RUNTIME",
        )
    )

    assert decision.disposition is ActionDisposition.MONITOR_ONLY
    assert decision.action is RecoveryAction.NO_ACTION


def test_catalogued_fatal_trunk_sxid_is_still_executable() -> None:
    """The guard must not regress the legitimate catalogued fatal trunk path."""
    decision = GpuFaultPolicyEngine().evaluate_sxid(
        _fatal_trunk_sxid(11001, "NVIDIA_FABRIC_MANAGER_CATALOG")
    )

    assert decision.disposition is ActionDisposition.EXECUTABLE
    assert decision.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
