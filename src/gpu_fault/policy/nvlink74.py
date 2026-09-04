"""The pinned NVIDIA XID 74 NVLink register decode workflow.

This one catalog workflow lives outside ``engine.py`` because it is the only one
carrying its own evidence model: seven kernel-reported registers, a bit-to-
category table, and per-link occurrence counters. Inside the engine it read as
if resolving an XID normally required all of that, and it crowded out the branch
sequence that every other XID actually follows.

Nothing here reaches for engine state. The one thing the decode needs from the
catalogs -- whether the reporting GPU is Hopper -- is resolved by the caller and
passed in, so this module stays a pure function of the event and its rule.
"""

from __future__ import annotations

from typing import NamedTuple

from gpu_fault.models import RecoveryAction
from gpu_fault.policy.models import (
    NVLINK74_LINK_SCOPED_CATEGORIES,
    NVLINK74_OCCURRENCE_THRESHOLDS,
    NVLINK74_PASSIVE_CATEGORIES,
    NVLINK74_REGISTER_RULES,
    NVLINK74_SUPPORT_CATEGORIES,
    ActionDisposition,
    ActionSource,
    CatalogRule,
    Containment,
    XidEvent,
    _Resolution,
)

# The registers are only defined for Hopper in the pinned workflow.
DECODABLE_FAMILY = "H100"
REGISTER_COUNT = 7


class DecodedBit(NamedTuple):
    """One populated register bit and the category the pinned table gives it.

    ``label`` is also the key ``XidEvent.nvlink_occurrence_counts`` uses, so the
    same-link thresholds are looked up without re-parsing anything.
    """

    label: str
    category: str

    @property
    def rule_id(self) -> str:
        """The decode identity recorded on the decision.

        ``orchestration/workflow_builder.py`` splits this back apart on the
        colon, and the acceptance evidence quotes it, so the text is a contract.
        """

        return f"{self.label}:{self.category}"


def resolve_nvlink74(
    event: XidEvent,
    rule: CatalogRule,
    *,
    family: str | None,
) -> _Resolution:
    """Decode the seven XID 74 registers into one outcome.

    The order below is the workflow's own and is load bearing: evidence gates
    precede the decode, an unknown bit outranks every known one, and the
    same-link occurrence thresholds are only consulted once the link identity is
    known -- counting occurrences without it would aggregate two different links
    into one verdict.
    """

    blocked = _evidence_gate(event, rule, family)
    if blocked is not None:
        return blocked

    matches, unknown = decode(event.registers)
    categories = {bit.category for bit in matches}
    evidence = ", ".join(bit.rule_id for bit in matches) or "no populated bits"

    if unknown or not matches:
        detail = ", ".join(unknown) or "all seven registers are zero"
        return _escalation(
            reason=(
                "NVIDIA XID 74 workflow requires reporting unknown "
                f"or unexpected register state: {detail}"
            ),
            matches=matches,
        )
    if categories <= NVLINK74_PASSIVE_CATEGORIES:
        return _monitoring(
            reason=(
                f"NVIDIA XID 74 register decode permits continued operation: {evidence}"
            ),
            matches=matches,
            investigatory_action=(
                "OPTIONAL_FIELDDIAG" if "corrected_threshold" in categories else None
            ),
        )
    if categories & NVLINK74_LINK_SCOPED_CATEGORIES and event.nvlink_link_id is None:
        return _evidence_block(
            reason=(
                "XID 74 link-scoped resolution requires an explicit "
                "NVLink identity in the source evidence; refusing "
                f"to aggregate or reset at GPU scope: {evidence}"
            ),
            matches=matches,
            investigatory_action=rule.investigatory_action,
            action=None,
        )
    if "fabric_reset_required" in categories:
        return _escalation(
            reason=(
                "NVIDIA XID 74 fourth-register bit 18 requires a "
                "fabric reset; complete local fabric inventory is "
                f"not present in an XID event: {evidence}"
            ),
            matches=matches,
            containment=Containment.FABRIC_PARTITION,
        )
    if categories & NVLINK74_SUPPORT_CATEGORIES:
        return _escalation(
            reason=(
                "NVIDIA XID 74 register decode requires link "
                f"diagnostics and support review: {evidence}"
            ),
            matches=matches,
        )
    if categories - NVLINK74_PASSIVE_CATEGORIES == {"secondary"}:
        return _escalation(
            reason=(
                "NVIDIA XID 74 contains only sympathetic/secondary "
                f"bits and requires reporting when seen solo: {evidence}"
            ),
            matches=matches,
        )
    return _same_link_verdict(event, rule, matches, categories)


def _evidence_gate(
    event: XidEvent,
    rule: CatalogRule,
    family: str | None,
) -> _Resolution | None:
    """Refuse to decode registers the pinned workflow does not describe."""

    if len(event.registers) != REGISTER_COUNT:
        return _evidence_block(
            reason=(
                "XID 74 decode requires exactly seven register "
                "fields from the kernel event"
            ),
            matches=[],
            investigatory_action=rule.investigatory_action,
            action=RecoveryAction.STOP_WORKLOAD,
        )
    if family != DECODABLE_FAMILY:
        return _escalation(
            reason=(
                "the pinned XID 74 register bit workflow identifies "
                "the first, third, fourth and fifth registers as "
                "valid for Hopper products; refusing to apply the "
                f"Hopper decoder to {event.product or 'UNKNOWN'}"
            ),
            matches=[],
        )
    return None


def decode(registers: list[int]) -> tuple[list[DecodedBit], list[str]]:
    """Split the populated register bits into classified and unknown ones.

    Bits are scanned to at least 32 positions even when the reported value is
    narrower, so a bit the pinned table does not cover is still surfaced rather
    than silently dropped by the register's own width.
    """

    matches: list[DecodedBit] = []
    unknown: list[str] = []
    for register_index, value in enumerate(registers):
        rules = NVLINK74_REGISTER_RULES.get(register_index, {})
        for bit in range(max(32, value.bit_length())):
            if not value & (1 << bit):
                continue
            label = f"register{register_index + 1}.bit{bit}"
            categories = [category for category, bits in rules.items() if bit in bits]
            if not categories:
                unknown.append(label)
                continue
            matches.extend(DecodedBit(label, category) for category in categories)
    return matches, unknown


def _same_link_verdict(
    event: XidEvent,
    rule: CatalogRule,
    matches: list[DecodedBit],
    categories: set[str],
) -> _Resolution:
    """Resolve the bits whose outcome depends on same-link repetition.

    Reset comes before support on purpose: for these categories the pinned
    workflow contacts support only after a reset or node reboot fails to clear
    the condition, so escalating first would park a recoverable link on a human.
    """

    counts = event.nvlink_occurrence_counts
    matched_counts = {bit.label: counts.get(bit.label, 0) for bit in matches}
    count_evidence = ", ".join(
        f"{label}=count{value}" for label, value in sorted(matched_counts.items())
    )
    if any(
        matched_counts[bit.label] >= NVLINK74_OCCURRENCE_THRESHOLDS[bit.category]
        for bit in matches
        if bit.category in NVLINK74_OCCURRENCE_THRESHOLDS
    ):
        return _reset(
            reason=(
                "NVIDIA XID 74 same-link threshold reached on "
                f"NVLink {event.nvlink_link_id}: {count_evidence}; "
                "reset the affected GPU first; contact support only "
                "if reset/reboot fails to clear the condition"
            ),
            matches=matches,
            investigatory_action="CONTACT_SUPPORT_AFTER_REMEDIATION",
        )
    if "mechanical_or_hardware" in categories:
        return _reset(
            reason=(
                "NVIDIA XID 74 requires reset of the affected GPU "
                f"on NVLink {event.nvlink_link_id}: "
                f"{count_evidence}; inspect hardware only if reset "
                "or node reboot does not clear the condition"
            ),
            matches=matches,
            investigatory_action="CHECK_MECHANICALS_AFTER_REMEDIATION",
        )
    return _monitoring(
        reason=(
            "NVIDIA XID 74 same-link reporting threshold has not "
            f"been reached on NVLink {event.nvlink_link_id}: "
            f"{count_evidence}; continue monitoring"
        ),
        matches=matches,
        investigatory_action=rule.investigatory_action,
    )


# The four outcomes this workflow can reach. They are built here rather than
# inline so each decision site above shows only what makes it different: which
# condition it fires on and which NVIDIA text it quotes.
def _escalation(
    *,
    reason: str,
    matches: list[DecodedBit],
    containment: Containment = Containment.GPU,
) -> _Resolution:
    """Report to support and stop, with no automated recovery attempted."""

    return _Resolution(
        source=ActionSource.NVIDIA_CATALOG,
        disposition=ActionDisposition.EXECUTABLE,
        official_action="WORKFLOW_NVLINK_ERR",
        investigatory_action="CONTACT_SUPPORT",
        action=RecoveryAction.ESCALATE_OPERATOR,
        containment=containment,
        reasons=[reason],
        matched_decode_rules=[bit.rule_id for bit in matches],
        requires_operator=True,
    )


def _reset(
    *,
    reason: str,
    matches: list[DecodedBit],
    investigatory_action: str,
) -> _Resolution:
    """Reset the affected GPU, leaving the follow-up inspection optional."""

    return _Resolution(
        source=ActionSource.NVIDIA_CATALOG,
        disposition=ActionDisposition.EXECUTABLE,
        official_action="WORKFLOW_NVLINK_ERR",
        investigatory_action=investigatory_action,
        action=RecoveryAction.RESET_GPU,
        containment=Containment.GPU,
        reasons=[reason],
        matched_decode_rules=[bit.rule_id for bit in matches],
        requires_operator=False,
    )


def _monitoring(
    *,
    reason: str,
    matches: list[DecodedBit],
    investigatory_action: str | None,
) -> _Resolution:
    """Keep running: the decode maps to the catalog's own IGNORE."""

    return _Resolution(
        source=ActionSource.NVIDIA_CATALOG,
        disposition=ActionDisposition.MONITOR_ONLY,
        official_action="IGNORE",
        investigatory_action=investigatory_action,
        action=RecoveryAction.NO_ACTION,
        containment=Containment.GPU,
        reasons=[reason],
        matched_decode_rules=[bit.rule_id for bit in matches],
    )


def _evidence_block(
    *,
    reason: str,
    matches: list[DecodedBit],
    investigatory_action: str | None,
    action: RecoveryAction | None,
) -> _Resolution:
    """Fail closed: the evidence cannot support any NVIDIA outcome."""

    return _Resolution(
        source=ActionSource.NVIDIA_CATALOG,
        disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
        official_action="WORKFLOW_NVLINK_ERR",
        investigatory_action=investigatory_action,
        action=action,
        containment=Containment.GPU,
        reasons=[reason],
        matched_decode_rules=[bit.rule_id for bit in matches],
        requires_operator=True,
    )
