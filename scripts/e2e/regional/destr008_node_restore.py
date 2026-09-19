"""Restore the fault node after a shortage scenario and audit both node baselines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scripts.e2e.regional.destr008_cleanup_identity import (
    require_cleanup_family,
    require_cleanup_node,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.warm_spare_fixture import (
    QUARANTINE_TAINT,
    SPARE_LABEL,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    WarmSpareLiveFixture,
)

if TYPE_CHECKING:
    from scripts.e2e.regional.run_destr008_warm_spare_shortage import Settings


def restore_fault_node(
    warm: WarmSpareLiveFixture,
    *,
    settings: Settings,
    incident_id: str,
    profile_version: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}
    if not incident_id:
        return result
    try:
        original = warm.node_snapshot(settings.fault_node)
        owner = str(original["annotations"].get("gpu-fault.io/incident-id") or "")
        if not isinstance(original.get("uid"), str) or not original["uid"]:
            raise RegionalFixtureError("cleanup fault Node UID is missing")
        require_cleanup_family(
            warm,
            incident_id=incident_id,
            owner=owner,
            cluster_id=settings.regional.cluster_id,
            node=settings.fault_node,
            profile_version=profile_version,
        )
        require_cleanup_node(
            warm, node=settings.fault_node, uid=original["uid"], owner=owner
        )
    except Exception as exc:
        result["errors"].append(f"cleanup ownership: {type(exc).__name__}: {exc}")
        return result
    try:
        warm.release_spares([settings.spare_node], incident_id)
    except Exception as exc:
        result["errors"].append(f"spare rollback: {type(exc).__name__}: {exc}")
    try:
        require_cleanup_node(
            warm, node=settings.fault_node, uid=original["uid"], owner=owner
        )
        warm.reactivate_agent(settings.fault_node)
        warm.wait_agent_active(settings.fault_node)
    except Exception as exc:
        result["errors"].append(f"agent cleanup: {type(exc).__name__}: {exc}")
    try:
        fault = warm.node_snapshot(settings.fault_node)
        if (
            fault["uid"] != original["uid"]
            or str(fault["annotations"].get("gpu-fault.io/incident-id") or "") != owner
        ):
            raise RegionalFixtureError("cleanup quarantine owner or Node UID changed")
        result["quarantine_owner"] = owner or None
        if owner and owner != incident_id:
            # A shortage that blocks recovery is escalated by the product
            # itself: the escalation engine opens a successor support incident
            # over the same node, re-quarantines it and files a ticket, so the
            # node ends up owned by that incident rather than by ours. Cleanup
            # that recognised only its own incident restored nothing and
            # recorded no error at all, and the case then failed in postflight
            # for a condition the cleanup had already seen and skipped.
            result["successor_incident"] = owner
        if owner:
            warm.wait_incident_idle(owner)
            require_cleanup_node(
                warm, node=settings.fault_node, uid=original["uid"], owner=owner
            )
            created = warm.create_restore_workflow(
                incident_id=owner,
                node=settings.fault_node,
                profile_version=profile_version,
                reason="DESTR-008 scenario cleanup",
            )
            restored = warm.wait_workflow_id(str(created["workflow_request_id"]))
            result["restore_workflow"] = restored
            if restored.get("status") != "SUCCEEDED":
                result["errors"].append("fault-node restore workflow failed")
    except Exception as exc:
        result["errors"].append(f"fault restore: {type(exc).__name__}: {exc}")
    return result


def audit_scenario_nodes(
    warm: WarmSpareLiveFixture,
    settings: Settings,
    result: dict[str, Any],
) -> None:
    try:
        result["final_fault_node"] = warm.node_snapshot(settings.fault_node)
        result["final_spare_node"] = warm.node_snapshot(settings.spare_node)
        fault = result["final_fault_node"]
        spare = result["final_spare_node"]
        if (
            fault["ready"] != "True"
            or fault["unschedulable"]
            or any(item.get("key") == QUARANTINE_TAINT for item in fault["taints"])
            or any(
                fault["annotations"].get(key)
                for key in (
                    "gpu-fault.io/incident-id",
                    "gpu-fault.io/fencing-token",
                    "gpu-fault.io/previous-unschedulable",
                )
            )
        ):
            result["verdict"] = "FAIL"
            result.setdefault("postflight_errors", []).append(
                "fault node did not return to Ready/schedulable/unowned"
            )
        if (
            spare["ready"] != "True"
            or not spare["unschedulable"]
            or spare["labels"].get(SPARE_LABEL) != "true"
            or spare["annotations"].get(SPARE_RESERVATION_ANNOTATION)
            or spare["annotations"].get(SPARE_POOL_STATE_ANNOTATION)
            not in {None, "AVAILABLE"}
        ):
            result["verdict"] = "FAIL"
            result.setdefault("postflight_errors", []).append(
                "spare node did not return to the available cordoned pool"
            )
    except Exception as exc:
        result["postflight_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
