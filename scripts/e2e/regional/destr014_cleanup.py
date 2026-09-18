"""Post-reboot probe recovery and bounded cleanup for the DESTR-014 runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import control_plane_env_window as control_window
from scripts.e2e.regional import destr014_recovery as recovery
from scripts.e2e.regional import executor_env_window as env_window
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture
from scripts.e2e.regional.managed_workload_fixture import (
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
)
from scripts.e2e.regional.warm_spare_fixture import (
    QUARANTINE_TAINT,
    WarmSpareLiveFixture,
)

if TYPE_CHECKING:
    from scripts.e2e.regional.run_destr014_branch_exhaustion import Settings


def _ledger(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    state = snapshot.get("state") or {}
    return cast(
        list[dict[str, Any]], state.get("ledger") or snapshot.get("ledger") or []
    )


def recreate_probe(probe: HostProbeFixture) -> None:
    """Replace a host probe Pod after the node it ran on rebooted.

    A RESTART_NODE takes the probe down with its node and leaves it Failed;
    ``kubectl exec`` into a Failed Pod is refused, so deleting and re-applying
    is the only way back to a Pod that can still read and restore the host
    (same rule as ``CollectorAcceptanceFixture.recreate``).
    """

    probe.cleanup()
    probe.create()


def _cleanup(
    *,
    regional: RegionalLiveFixture,
    warm: WarmSpareLiveFixture,
    workload: ManagedWorkloadFixture,
    prewarm: ImagePrewarmFixture,
    fault_probe: HostProbeFixture,
    sibling_probe: HostProbeFixture,
    inject_fault: HostProbeFixture,
    inject_sibling: HostProbeFixture,
    settings: Settings,
    run_id: str,
    incident_id: str,
    env_baseline: Path,
    env_opened: bool,
    control_env_baseline: Path,
    control_env_opened: bool,
    holder_armed: bool,
    agent_disabled: bool,
    profile_version: str,
    recovery_window: recovery.AgentRecoveryWindow | None = None,
    operator_hold: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}

    def guard(label: str, action: Any) -> None:
        try:
            result[label] = action()
            if label.endswith("_cleanup") and isinstance(result[label], dict):
                if any(result[label].values()):
                    raise RegionalFixtureError(f"cleanup left residuals: {label}")
        except ProcessSupervisionLost:
            raise
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    if holder_armed:
        guard("fault_probe_recreate", lambda: recreate_probe(fault_probe))
        guard(
            "holder_disarm",
            lambda: fault_probe.execute("disarm-holder", "--run-id", run_id),
        )
    if recovery_window is not None:
        guard("agent_recovery", recovery_window.cleanup)
    elif agent_disabled:
        result["errors"].append(
            "Agent recovery binding is missing; legacy restore is not authorized"
        )
    if operator_hold:
        result["operator_hold_preserved"] = True
        result["errors"].append(
            "physical outcome unresolved; workload, isolation and configuration "
            "restoration require operator reconciliation"
        )
    if incident_id and not operator_hold:
        guard("quiescence", lambda: warm.wait_incident_idle(incident_id))
    quiescent = not operator_hold and (
        "quiescence" in result or (not incident_id and not holder_armed)
    )
    if quiescent:
        guard("workload_delete", workload.delete)
    else:
        result["workload_cleanup_deferred"] = True
        result["errors"].append("workload cleanup deferred: no quiescence proof")
    if incident_id and quiescent:
        guard(
            "sibling_agent_reactivate",
            lambda: warm.reactivate_agent(settings.sibling_node),
        )
        guard(
            "isolation_restore",
            lambda: _restore_isolated_nodes(
                regional, warm, settings, incident_id, profile_version
            ),
        )
    if env_opened and quiescent:
        guard(
            "env_window_close",
            lambda: env_window.close_window(
                env_window.Settings(baseline=env_baseline, rollout_timeout_seconds=300),
                regional,
                env_window.survey(regional),
            ),
        )
    if control_env_opened and quiescent:
        guard(
            "control_env_window_close",
            lambda: control_window.without_survey(
                control_window.close_window(
                    control_window.Settings(
                        baseline=control_env_baseline, rollout_timeout_seconds=600
                    ),
                    regional,
                    control_window.survey(regional),
                )
            ),
        )
    guard("prewarm_cleanup", prewarm.cleanup)
    for name, probe in (
        ("fault_probe", fault_probe),
        ("sibling_probe", sibling_probe),
        ("inject_fault", inject_fault),
        ("inject_sibling", inject_sibling),
    ):
        guard(f"{name}_cleanup", probe.cleanup)
    return result


def _restore_isolated_nodes(
    regional: RegionalLiveFixture,
    warm: WarmSpareLiveFixture,
    settings: Settings,
    incident_id: str,
    profile_version: str,
) -> dict[str, Any]:
    """Release every node the case -- or the product's own escalation -- left
    isolated, through the incident that owns the isolation.

    The exhaustion escalation opens a support-after incident that re-quarantines
    the node and owns its taint, so a restore through the case's own incident
    is refused (attempt 8, 2026-09-09: "sibling restore workflow did not
    succeed" and both nodes stayed quarantined). Same rule as the DESTR-003/008
    cleanups: read each node's owner annotation, validated-restore through it,
    record the owner, then close the case incident if it is still open. Never
    delete a taint or an annotation by hand.
    """

    report: dict[str, Any] = {"nodes": {}}
    for node in (settings.sibling_node, settings.fault_node):
        snapshot = regional.node_snapshot(node)
        owner = str(
            (snapshot.get("ownership_annotations") or {}).get(
                "gpu-fault.io/incident-id"
            )
            or ""
        )
        isolated = (
            bool(snapshot.get("unschedulable"))
            or any(
                item.get("key") == QUARANTINE_TAINT
                for item in snapshot.get("taints") or []
            )
            or bool(owner)
        )
        entry: dict[str, Any] = {
            "quarantine_owner": owner or None,
            "isolated": isolated,
        }
        if isolated:
            target = owner or incident_id
            if owner and owner != incident_id:
                entry["successor_incident"] = owner
            warm.wait_incident_idle(target)
            created = warm.create_restore_workflow(
                incident_id=target,
                node=node,
                profile_version=profile_version,
                reason="DESTR-014 validated cleanup",
            )
            restored = warm.wait_workflow_id(str(created["workflow_request_id"]))
            entry["restore"] = {
                "workflow_request_id": created["workflow_request_id"],
                "status": restored.get("status"),
                "error": restored.get("error"),
            }
            if restored.get("status") != "SUCCEEDED":
                raise RegionalFixtureError(
                    f"{node} restore workflow did not succeed: "
                    f"{restored.get('status')} {restored.get('error')}"
                )
        report["nodes"][node] = entry
    state = warm.incident_by_id(incident_id).get("state")
    report["incident_state_before_close"] = state
    if state != "RECOVERED":
        warm.wait_incident_idle(incident_id)
        created = warm.create_restore_workflow(
            incident_id=incident_id,
            node=settings.sibling_node,
            profile_version=profile_version,
            reason="DESTR-014 validated cleanup: close the case incident",
        )
        restored = warm.wait_workflow_id(str(created["workflow_request_id"]))
        report["close"] = {
            "status": restored.get("status"),
            "error": restored.get("error"),
        }
        if restored.get("status") != "SUCCEEDED":
            raise RegionalFixtureError("case incident close workflow did not succeed")
    report["incident_state_after"] = warm.incident_by_id(incident_id).get("state")
    return report
