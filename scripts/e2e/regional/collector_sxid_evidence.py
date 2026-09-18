"""Code-specific SXID evidence, separate from a generic successful reset."""

from __future__ import annotations

from typing import Any


FULL_RESET_VARIANTS = ((10003, "Fatal"), (19084, "Non-fatal"))


def full_reset_variant_errors(
    state: dict[str, Any], *, sxid: int, classification: str
) -> list[str]:
    events = state.get("fabric_events") or []
    decisions = state.get("decisions") or []
    if len(events) != 1 or len(decisions) != 1:
        return ["full reset has no unique normalized SXID and persisted decision"]
    event, decision = events[0], decisions[0]
    if (
        (sxid, classification) not in FULL_RESET_VARIANTS
        or event.get("sxid") != sxid
        or event.get("classification") != classification.upper().replace("-", "_")
        or event.get("classification_source") != "NVIDIA_FABRIC_MANAGER_CATALOG"
        or decision.get("event_id") != event.get("event_id")
        or decision.get("event_type") != "SXID"
        or decision.get("official_action") != "RESET_ALL_GPUS_AND_NVSWITCHES"
        or decision.get("disposition") != "EXECUTABLE"
        or decision.get("action") is not None
    ):
        return ["full reset decision does not match the exact SXID/severity variant"]
    workflows = [
        row
        for row in state.get("workflows") or []
        if row.get("request_id") == decision.get("workflow_request_id")
        and row.get("incident_id") == decision.get("incident_id")
    ]
    if len(workflows) != 1:
        return ["full reset decision is not bound to one workflow"]
    resets = [
        step
        for step in workflows[0].get("official_steps") or []
        if step.get("operation") == "RESET_ALL_GPUS_NVSWITCHES"
    ]
    if len(resets) != 1 or (resets[0].get("parameters") or {}).get("sxid") != sxid:
        return ["full reset workflow did not retain the triggering SXID"]
    return []


def stable_inventory_errors(
    baseline: dict[str, Any], current: dict[str, Any], node: dict[str, Any]
) -> list[str]:
    before = baseline.get("gpu_inventory") or []
    after = current.get("gpu_inventory") or []
    identities = [
        {str(item.get("uuid") or "") for item in rows} for rows in (before, after)
    ]
    if (
        not before
        or not after
        or any("" in values for values in identities)
        or len(identities[0]) != len(before)
        or len(identities[1]) != len(after)
        or identities[0] != identities[1]
    ):
        return ["full GPU inventory is not stable between SXID reset variants"]
    if (
        not baseline.get("boot_id")
        or current.get("boot_id") != baseline.get("boot_id")
        or node.get("ready") != "True"
        or not node.get("uid")
        or node.get("ownership_annotations")
        or node.get("unschedulable") is not False
        or node.get("taints") != []
    ):
        return ["node is not safely restored between SXID reset variants"]
    return []
