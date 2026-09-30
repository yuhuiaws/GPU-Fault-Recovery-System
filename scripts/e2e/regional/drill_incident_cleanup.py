"""Close the incidents a drill's own injection opened, the way an operator would.

A runner that injects a fault owns the incident the product opens for it.
When the case ends, that incident is closed here through the product's own
closure path (``IncidentClosureService.close_incident`` with node isolation
evidence, the call ``gpu-fault-admin workflow-reconcile --close-quarantined``
makes), never by editing the store:

* ESCALATED and QUARANTINED incidents are operator-closable; a QUARANTINED one
  only once every node it holds is no longer isolated (a validated restore
  releases the node first). A refusal is the service's own wording.
* planning states (DETECTED, ACTION_PENDING, SAFETY_PENDING) are ended by the
  workflow that owns them and cannot be closed by anyone; they are reported as
  a residual so the case result names them.
* an incident whose nodes are not the drill's nodes is never touched.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

QUARANTINE_TAINT = "gpu-fault.io/quarantined"
CLOSABLE_STATES = frozenset({"ESCALATED", "QUARANTINED"})
TERMINAL_STATES = frozenset({"RECOVERED"})
DEFAULT_OPERATOR = "gpu-fault-acceptance"


def isolation_evidence(node: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """``NodeIsolationEvidence.from_mapping`` input from a runner node read
    (``RegionalLiveFixture.node_snapshot`` shape: ``ownership_annotations``)."""

    taint_value = next(
        (
            str(taint.get("value") or "")
            for taint in snapshot.get("taints") or []
            if taint.get("key") == QUARANTINE_TAINT
        ),
        None,
    )
    return {
        "node_id": node,
        "exists": True,
        "unschedulable": bool(snapshot.get("unschedulable")),
        "quarantine_taint_value": taint_value,
        "isolation_annotations": dict(snapshot.get("ownership_annotations") or {}),
    }


def node_isolated(evidence: Mapping[str, Any]) -> bool:
    return bool(
        evidence.get("unschedulable")
        or evidence.get("quarantine_taint_value") is not None
        or evidence.get("isolation_annotations")
    )


def close_drill_incident(
    warm: Any,
    regional: Any,
    incident_id: str,
    *,
    reason: str,
    reference: str,
    nodes: Iterable[str],
    operator: str = DEFAULT_OPERATOR,
) -> dict[str, Any]:
    """Close one drill incident if the product lets an operator close it.

    Returns ``{"incident_id", "state_before", "closed", ...}``; ``residual``
    names why the incident stays open (not closable, still isolated, refused,
    or not this drill's). Nothing here raises for a product refusal; transport
    failures propagate to the caller's cleanup error list.
    """

    incident = warm.incident_by_id(incident_id)
    state = str(incident.get("state") or "")
    report: dict[str, Any] = {
        "incident_id": incident_id,
        "state_before": state,
        "closed": False,
    }
    incident_nodes = [str(item) for item in incident.get("node_ids") or []]
    drill_nodes = {str(item) for item in nodes if item}
    if not incident_nodes or not set(incident_nodes) <= drill_nodes:
        report["residual"] = (
            f"incident nodes {incident_nodes} are not the drill's nodes "
            f"{sorted(drill_nodes)}; a foreign incident is never closed"
        )
        return report
    if state in TERMINAL_STATES:
        report["already_terminal"] = True
        return report
    if state not in CLOSABLE_STATES:
        report["residual"] = (
            f"incident is {state or 'unknown'}; only an ESCALATED or QUARANTINED "
            "incident can be closed by an operator, a planning state only by the "
            "workflow that ends it"
        )
        return report
    evidence = [
        isolation_evidence(node, regional.node_snapshot(node))
        for node in incident_nodes
    ]
    report["evidence"] = evidence
    still_isolated = [item["node_id"] for item in evidence if node_isolated(item)]
    if state == "QUARANTINED" and still_isolated:
        report["residual"] = (
            f"nodes {still_isolated} are still isolated; a validated restore must "
            "release them before the incident can close"
        )
        return report
    outcome = warm.close_incident_with_evidence(
        incident_id,
        reason=reason,
        operator=operator,
        reference=reference,
        evidence=evidence,
    )
    report["closed"] = bool(outcome.get("closed"))
    report["state_after"] = outcome.get("state")
    if not report["closed"]:
        report["refusal"] = outcome.get("refusal")
        report["residual"] = str(outcome.get("refusal") or "close refused")
    return report


def close_drill_incidents(
    warm: Any,
    regional: Any,
    incident_ids: Iterable[str | None],
    *,
    reason: str,
    reference: str,
    nodes: Iterable[str],
    operator: str = DEFAULT_OPERATOR,
) -> dict[str, dict[str, Any]]:
    """``close_drill_incident`` for every distinct non-empty id, keyed by id."""

    drill_nodes = tuple(nodes)
    reports: dict[str, dict[str, Any]] = {}
    for incident_id in incident_ids:
        if not incident_id or incident_id in reports:
            continue
        reports[incident_id] = close_drill_incident(
            warm,
            regional,
            incident_id,
            reason=reason,
            reference=reference,
            nodes=drill_nodes,
            operator=operator,
        )
    return reports


def residual_incidents(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """The incidents a cleanup left open, with the product's reason."""

    return {
        incident_id: str(report["residual"])
        for incident_id, report in reports.items()
        if report.get("residual")
    }
