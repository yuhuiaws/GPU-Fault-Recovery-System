"""``gpu-fault-admin workflow-reconcile --close-incident``, ``--close-escalated``
and ``--close-quarantined``: close incidents from the administrator's shell.

An incident whose remediation was handed to an operator (lifetime exceeded,
support ticket delivered) stays ESCALATED and keeps recording the node's later
faults instead of planning them (F-N1). Once the node is back, the operator
closes it. This is the CLI face of ``IncidentClosureService.close_incident``
-- the same service function ``POST /v1/incidents/{id}/close`` calls -- run
inside the CPU ingress Pod like every other ``workflow-reconcile`` mode, with
the operator's STS caller identity on the audit event and the approved-change
``--reference`` recorded with it. ``--dry-run`` reports the verdict for each
incident and writes nothing.

``--close-incident`` names the ids; ``--close-escalated`` discovers them: every
ESCALATED incident of every registered cluster, oldest first, capped by
``--max-items``. The alert ``GpuFaultIncidentsAwaitingOperator`` only counts
the queue, so before this the ids had to be copied out of escalation e-mails or
read with SQL inside the Pod.

A QUARANTINED incident needs more than a signature: the service closes it only
on node isolation evidence proving the isolation it owned is gone (no cordon,
no ``gpu-fault.io/quarantined`` taint with its value, no isolation annotation
naming it -- ``isolation_verdict``). The Pod has no kubeconfig, so the evidence
is read here, through the site's GPU kubeconfig (the same ``kubectl`` path the
BLOCKED reconcile uses for its node evidence; a site without one is refused),
and handed to the script in a second pass for exactly the incidents the first
pass reported as ``evidence_required``. ``--close-quarantined`` discovers the
QUARANTINED queue the way ``--close-escalated`` discovers the ESCALATED one
(2026-09-10: 24 incidents sat QUARANTINED on nodes a later incident or a
cleanup had already released).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from gpu_fault.admin import operator_identity
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.workflow_reconcile import (
    REFERENCE_PATTERN,
    _run_reconcile,
    cluster_nodes,
    node_isolation_evidence,
)

INCIDENT_CLOSE_HISTORY_PATH = Path("workflow-reconcile/incident-close/history")
REASON_LIMIT = 512
# Per-cluster ceiling of the discovery; a queue past this needs more than one
# run, and the report says so (``discovery_limit_reached``).
DISCOVERY_LIMIT = 1000
ESCALATED_SELECTOR_STATES = ("ESCALATED",)
QUARANTINED_SELECTOR_STATES = ("QUARANTINED",)
SELECTOR_FLAGS = {
    "escalated": "--close-escalated",
    "quarantined": "--close-quarantined",
}

# Runs in the CPU ingress Pod (``_run_reconcile``); the payload arrives on
# stdin. ``context.incident_closure`` is the service the API route uses, so a
# refusal here is word for word the 409 the API would have answered. With a
# ``selector`` instead of ids the script discovers the incidents first
# (``store.list_incidents_by_state``, the read ``GET /v1/incidents`` makes),
# ordered by created_at then id, and then judges or closes them exactly as an
# explicit list would be. ``evidence`` (incident id -> node isolation evidence
# mappings) is what the admin layer read from the GPU cluster for the incidents
# an earlier pass reported as ``evidence_required``; without it a QUARANTINED
# incident is refused and reported with ``evidence_required``, its cluster and
# its nodes, so the admin layer knows what to read.
INCIDENT_CLOSE_SCRIPT = """
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.orchestration.incident_closure import IncidentNotClosable
from gpu_fault.store import NotFoundError

try:
    from gpu_fault.orchestration.incident_closure import NodeIsolationEvidence
except ImportError:  # an image from before evidence-based closes
    NodeIsolationEvidence = None

payload = json.load(sys.stdin)
context = ApplicationContext.from_environment()
service = context.incident_closure
store = context.store
report = {"mode": "incident-close", "dry_run": payload["dry_run"]}
selector = payload.get("selector")
if selector is None:
    incident_ids = list(payload["incident_ids"])
else:
    states = [IncidentState(value) for value in selector["states"]]
    cluster_ids = selector.get("cluster_ids") or store.list_regional_cluster_ids()
    node_ids = set(selector["node_ids"]) if selector.get("node_ids") else None
    per_cluster_limit = int(selector["discovery_limit"])
    discovered = []
    limit_reached = []
    for cluster_id in sorted(set(cluster_ids)):
        rows = store.list_incidents_by_state(
            cluster_id, states, node_ids=node_ids, limit=per_cluster_limit
        )
        if len(rows) >= per_cluster_limit:
            limit_reached.append(cluster_id)
        discovered.extend(rows)
    discovered.sort(key=lambda item: (item.created_at.isoformat(), item.incident_id))
    incident_ids = [item.incident_id for item in discovered]
    max_items = selector.get("max_items")
    report["discovered_total"] = len(incident_ids)
    report["discovered_cluster_ids"] = sorted(set(cluster_ids))
    report["discovery_limit_reached"] = limit_reached
    if max_items is not None and len(incident_ids) > int(max_items):
        incident_ids = incident_ids[: int(max_items)]
    report["discovered_incident_ids"] = incident_ids
evidence_by_incident = payload.get("evidence") or {}


def evidence_for(incident_id):
    items = evidence_by_incident.get(incident_id)
    if items is None or NodeIsolationEvidence is None:
        return None
    return [NodeIsolationEvidence.from_mapping(item) for item in items]


def preview_result(incident_id, preview):
    if preview["state"] == "RECOVERED":
        outcome = "already-recovered"
    elif preview["closable"]:
        outcome = "would-close"
    else:
        outcome = "refused"
    return {
        "incident_id": incident_id,
        "outcome": outcome,
        "state": preview["state"],
        "reason": preview["refusal"],
        "open_workflow_id": preview["open_workflow_id"],
        "cluster_id": preview.get("cluster_id"),
        "node_ids": list(preview.get("node_ids") or []),
        "evidence_required": bool(preview.get("evidence_required")),
        "isolation_reasons": list(preview.get("isolation_reasons") or []),
    }


results = []
for incident_id in incident_ids:
    evidence = evidence_for(incident_id)
    isolation_nodes = (
        sorted(item.node_id for item in evidence) if evidence is not None else None
    )
    if payload["dry_run"]:
        preview = (
            service.preview(incident_id, evidence=evidence)
            if evidence is not None
            else service.preview(incident_id)
        )
        entry = preview_result(incident_id, preview)
        if entry["outcome"] == "would-close" and isolation_nodes:
            entry["isolation_nodes"] = isolation_nodes
        results.append(entry)
        continue
    if evidence is None:
        preview = service.preview(incident_id)
        if preview.get("evidence_required"):
            # The admin layer reads the node evidence and asks again.
            results.append(preview_result(incident_id, preview))
            continue
    try:
        if evidence is not None:
            incident, closed = service.close_incident(
                incident_id,
                reason=payload["reason"],
                operator=payload["operator"],
                reference=payload.get("reference"),
                evidence=evidence,
            )
        else:
            incident, closed = service.close_incident(
                incident_id,
                reason=payload["reason"],
                operator=payload["operator"],
                reference=payload.get("reference"),
            )
    except NotFoundError:
        results.append(
            {"incident_id": incident_id, "outcome": "refused", "state": None,
             "reason": "incident not found"}
        )
    except IncidentNotClosable as exc:
        results.append(
            {"incident_id": incident_id, "outcome": "refused", "reason": str(exc)}
        )
    else:
        entry = {
            "incident_id": incident_id,
            "outcome": "closed" if closed else "already-recovered",
            "state": incident.state.value,
        }
        if closed and isolation_nodes:
            entry["isolation_nodes"] = isolation_nodes
        results.append(entry)
report["results"] = results
print(json.dumps(report, sort_keys=True))
"""


def _selector(
    mode: str,
    states: Sequence[str],
    *,
    cluster_ids: Sequence[str],
    node_ids: Sequence[str],
    max_items: int | None,
) -> dict[str, Any]:
    clusters = sorted({str(item).strip() for item in cluster_ids if str(item).strip()})
    nodes = sorted({str(item).strip() for item in node_ids if str(item).strip()})
    return {
        "mode": mode,
        "states": list(states),
        "cluster_ids": clusters or None,
        "node_ids": nodes or None,
        "max_items": max_items,
        "discovery_limit": DISCOVERY_LIMIT,
        "order": "created_at,incident_id",
    }


def escalated_selector(
    *,
    cluster_ids: Sequence[str] = (),
    node_ids: Sequence[str] = (),
    max_items: int | None = None,
) -> dict[str, Any]:
    """The ``--close-escalated`` discovery: ESCALATED only (a QUARANTINED
    incident is never discovered here; ``--close-quarantined`` is its
    selector), every registered cluster unless ``cluster_ids`` narrows it,
    oldest first."""

    return _selector(
        "escalated",
        ESCALATED_SELECTOR_STATES,
        cluster_ids=cluster_ids,
        node_ids=node_ids,
        max_items=max_items,
    )


def quarantined_selector(
    *,
    cluster_ids: Sequence[str] = (),
    node_ids: Sequence[str] = (),
    max_items: int | None = None,
) -> dict[str, Any]:
    """The ``--close-quarantined`` discovery: QUARANTINED only, same span and
    order as ``escalated_selector``; every discovered incident is then judged
    on node isolation evidence read through the GPU kubeconfig."""

    return _selector(
        "quarantined",
        QUARANTINED_SELECTOR_STATES,
        cluster_ids=cluster_ids,
        node_ids=node_ids,
        max_items=max_items,
    )


def gather_isolation_evidence(
    site: RenderedSite,
    pending: Sequence[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Node isolation evidence for the incidents the Pod reported as
    ``evidence_required``, one ``kubectl get nodes`` per cluster.

    Returns ``(evidence by incident id, refusal by incident id)``: an incident
    whose cluster cannot be read -- no GPU kubeconfig for the site, a cluster
    that is not in the managed site, kubectl failing -- is refused with that
    message instead of aborting the run; the ESCALATED closes of the same run
    already landed and must be reported.
    """

    evidence: dict[str, list[dict[str, Any]]] = {}
    refusals: dict[str, str] = {}
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for item in pending:
        incident_id = str(item.get("incident_id") or "")
        cluster_id = str(item.get("cluster_id") or "")
        node_ids = [str(node) for node in item.get("node_ids") or []]
        if not cluster_id or not node_ids:
            refusals[incident_id] = (
                f"incident {incident_id} names no cluster or no nodes; the "
                "isolation cannot be verified"
            )
            continue
        by_cluster.setdefault(cluster_id, []).append(item)
    for cluster_id, items in sorted(by_cluster.items()):
        try:
            inventory = cluster_nodes(site, cluster_id)
        except BootstrapError as exc:
            for item in items:
                refusals[str(item["incident_id"])] = str(exc)
            continue
        for item in items:
            evidence[str(item["incident_id"])] = node_isolation_evidence(
                site,
                cluster_id,
                [str(node) for node in item["node_ids"]],
                inventory=inventory,
            )
    return evidence, refusals


def _with_evidence(
    site: RenderedSite,
    payload: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """The second pass: read node evidence for every ``evidence_required``
    result and re-judge exactly those incidents with it, replacing their
    entries in place. A run without such results makes no second call."""

    results = list(result.get("results") or [])
    pending = [item for item in results if item.get("evidence_required")]
    if not pending:
        return result
    evidence, refusals = gather_isolation_evidence(site, pending)
    replacements: dict[str, dict[str, Any]] = {}
    if evidence:
        follow_up = {key: value for key, value in payload.items() if key != "selector"}
        follow_up["incident_ids"] = sorted(evidence)
        follow_up["evidence"] = evidence
        second = _run_reconcile(site, follow_up, script=INCIDENT_CLOSE_SCRIPT)
        second_results = second.get("results")
        if not isinstance(second_results, list):
            raise BootstrapError(
                "incident close with node evidence returned no results"
            )
        for item in second_results:
            replacements[str(item.get("incident_id"))] = dict(item)
    for incident_id, message in refusals.items():
        replacements[incident_id] = {
            "incident_id": incident_id,
            "outcome": "refused",
            "state": "QUARANTINED",
            "reason": message,
        }
    result["results"] = [
        replacements.get(str(item.get("incident_id")), item) for item in results
    ]
    result["isolation_evidence"] = evidence
    return result


def run_incident_close(
    site: RenderedSite,
    state_dir: Path,
    *,
    incident_ids: Sequence[str] = (),
    reason: str,
    reference: str | None,
    dry_run: bool,
    actor: str | None = None,
    selector: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Close (or, with ``dry_run``, judge) each incident and return the report.

    Either ``incident_ids`` (``--close-incident``) or a ``selector``
    (``--close-escalated`` / ``--close-quarantined``, see
    ``escalated_selector`` and ``quarantined_selector``) picks the incidents,
    never both. Validation happens before the Pod is reached: a non-empty
    ``reason`` and a well-formed ``reference`` unless ``dry_run``; an explicit
    list needs at least one distinct id. Incidents the Pod reports as
    QUARANTINED get their node isolation evidence read here and are judged
    again with it (``_with_evidence``). The applied report -- evidence
    included -- is archived under the state directory like the other
    reconcile modes; a dry run archives nothing.
    """

    ids = [str(item).strip() for item in incident_ids if str(item).strip()]
    if selector is not None and ids:
        raise BootstrapError(
            f"{SELECTOR_FLAGS.get(str(selector.get('mode')), '--close-escalated')} "
            "and --close-incident cannot be combined"
        )
    if selector is None:
        if not ids:
            raise BootstrapError("--close-incident needs at least one incident id")
        if len(set(ids)) != len(ids):
            raise BootstrapError("--close-incident lists a repeated incident id")
        flag = "--close-incident"
    else:
        max_items = selector.get("max_items")
        if max_items is not None and int(max_items) < 1:
            raise BootstrapError("--max-items must be at least 1")
        flag = SELECTOR_FLAGS.get(str(selector.get("mode")), "--close-escalated")
    normalized_reason = (reason or "").strip()
    if not normalized_reason and not dry_run:
        raise BootstrapError(f"{flag} requires --reason")
    if len(normalized_reason) > REASON_LIMIT:
        raise BootstrapError(f"--reason must be at most {REASON_LIMIT} characters")
    normalized_reference = (reference or "").strip() or None
    if not dry_run and normalized_reference is None:
        raise BootstrapError(
            f"{flag} requires --reference (approved change) unless --dry-run"
        )
    if normalized_reference is not None and not REFERENCE_PATTERN.fullmatch(
        normalized_reference
    ):
        raise BootstrapError("incident close reference is invalid")
    operator = actor or operator_identity.resolve_operator_identity()
    payload: dict[str, Any] = {
        "mode": "incident-close",
        "dry_run": bool(dry_run),
        "incident_ids": ids,
        "reason": normalized_reason,
        "reference": normalized_reference,
        "operator": operator,
    }
    if selector is not None:
        payload["selector"] = selector
    result = _run_reconcile(site, payload, script=INCIDENT_CLOSE_SCRIPT)
    results = result.get("results")
    if not isinstance(results, list):
        raise BootstrapError("incident close returned no results")
    result = _with_evidence(site, payload, result)
    results = result["results"]
    result["actor"] = operator
    result["dry_run"] = bool(dry_run)
    result["reference"] = normalized_reference
    if selector is not None:
        result["selector"] = selector
        discovered = result.get("discovered_incident_ids")
        if not isinstance(discovered, list):
            raise BootstrapError("incident discovery returned no incident ids")
        result["discovered_incident_ids"] = [str(item) for item in discovered]
        result.setdefault("discovered_total", len(result["discovered_incident_ids"]))
    else:
        result["selector"] = {"mode": "incident-ids", "incident_ids": ids}
        result["discovered_incident_ids"] = []
    result["closed_incident_ids"] = [
        str(item["incident_id"]) for item in results if item.get("outcome") == "closed"
    ]
    result["refused_incident_ids"] = [
        str(item["incident_id"]) for item in results if item.get("outcome") == "refused"
    ]
    if not dry_run:
        applied_at = datetime.now(timezone.utc)
        result["applied_at"] = applied_at.isoformat()
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        write_json_atomic(
            state_dir
            / INCIDENT_CLOSE_HISTORY_PATH
            / f"{applied_at.strftime('%Y%m%dT%H%M%SZ')}-{digest}.json",
            result,
        )
    return result


def discovery_lines(result: dict[str, Any]) -> list[str]:
    """The ``--close-escalated`` / ``--close-quarantined`` header: how many
    were discovered, over which clusters, and how many the ``--max-items``
    cap left to process."""

    selector = result.get("selector") or {}
    if selector.get("mode") not in SELECTOR_FLAGS:
        return []
    label = "/".join(str(item) for item in selector.get("states") or []) or "matching"
    total = int(result.get("discovered_total") or 0)
    selected = len(result.get("discovered_incident_ids") or [])
    clusters = result.get("discovered_cluster_ids") or []
    line = (
        f"discovered {total} {label} incident(s) across "
        f"{len(clusters)} cluster(s): {', '.join(clusters) or '-'}"
    )
    if selected < total:
        line += f"; processing {selected} (capped by --max-items {selector.get('max_items')})"
    lines = [line]
    limit_reached = result.get("discovery_limit_reached") or []
    if limit_reached:
        lines.append(
            f"discovery hit the {selector.get('discovery_limit')}-incident ceiling on "
            f"{', '.join(limit_reached)}; rerun after closing to see the rest"
        )
    return lines


def result_lines(result: dict[str, Any]) -> list[str]:
    """One line per incident: ``<id>: closed | already-recovered | would-close
    | refused(<reason>)``; a verdict that rested on node evidence adds
    ``(isolation absent on <nodes>)``; a discovery run is headed by
    ``discovery_lines``."""

    lines = discovery_lines(result)
    for item in result.get("results") or []:
        outcome = str(item.get("outcome"))
        if outcome == "refused":
            outcome = f"refused({item.get('reason') or 'no reason given'})"
        elif item.get("isolation_nodes"):
            nodes = ", ".join(str(node) for node in item["isolation_nodes"])
            outcome = f"{outcome} (isolation absent on {nodes})"
        lines.append(f"{item.get('incident_id')}: {outcome}")
    return lines


def exit_code(result: dict[str, Any]) -> int:
    return 1 if result.get("refused_incident_ids") else 0
