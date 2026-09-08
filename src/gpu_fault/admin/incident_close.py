"""``gpu-fault-admin workflow-reconcile --close-incident``: close an ESCALATED
incident from the administrator's shell.

An incident whose remediation was handed to an operator (lifetime exceeded,
support ticket delivered) stays ESCALATED and keeps recording the node's later
faults instead of planning them (F-N1). Once the node is back, the operator
closes it. This is the CLI face of ``IncidentClosureService.close_incident``
-- the same service function ``POST /v1/incidents/{id}/close`` calls -- run
inside the CPU ingress Pod like every other ``workflow-reconcile`` mode, with
the operator's STS caller identity on the audit event and the approved-change
``--reference`` recorded with it. ``--dry-run`` reports the verdict for each
incident and writes nothing.
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
from gpu_fault.admin.workflow_reconcile import REFERENCE_PATTERN, _run_reconcile

INCIDENT_CLOSE_HISTORY_PATH = Path("workflow-reconcile/incident-close/history")
REASON_LIMIT = 512

# Runs in the CPU ingress Pod (``_run_reconcile``); the payload arrives on
# stdin. ``context.incident_closure`` is the service the API route uses, so a
# refusal here is word for word the 409 the API would have answered.
INCIDENT_CLOSE_SCRIPT = """
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.orchestration.incident_closure import IncidentNotClosable
from gpu_fault.store import NotFoundError

payload = json.load(sys.stdin)
context = ApplicationContext.from_environment()
service = context.incident_closure
results = []
for incident_id in payload["incident_ids"]:
    if payload["dry_run"]:
        preview = service.preview(incident_id)
        if preview["state"] == "RECOVERED":
            outcome = "already-recovered"
        elif preview["closable"]:
            outcome = "would-close"
        else:
            outcome = "refused"
        results.append(
            {
                "incident_id": incident_id,
                "outcome": outcome,
                "state": preview["state"],
                "reason": preview["refusal"],
                "open_workflow_id": preview["open_workflow_id"],
            }
        )
        continue
    try:
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
        results.append(
            {
                "incident_id": incident_id,
                "outcome": "closed" if closed else "already-recovered",
                "state": incident.state.value,
            }
        )
print(
    json.dumps(
        {"mode": "incident-close", "dry_run": payload["dry_run"], "results": results},
        sort_keys=True,
    )
)
"""


def run_incident_close(
    site: RenderedSite,
    state_dir: Path,
    *,
    incident_ids: Sequence[str],
    reason: str,
    reference: str | None,
    dry_run: bool,
    actor: str | None = None,
) -> dict[str, Any]:
    """Close (or, with ``dry_run``, judge) each incident and return the report.

    Validation happens before the Pod is reached: at least one distinct id, a
    non-empty ``reason``, and -- unless ``dry_run`` -- a well-formed
    ``reference``. The applied report is archived under the state directory
    like the other reconcile modes; a dry run archives nothing.
    """

    ids = [str(item).strip() for item in incident_ids if str(item).strip()]
    if not ids:
        raise BootstrapError("--close-incident needs at least one incident id")
    if len(set(ids)) != len(ids):
        raise BootstrapError("--close-incident lists a repeated incident id")
    normalized_reason = (reason or "").strip()
    if not normalized_reason:
        raise BootstrapError("--close-incident requires --reason")
    if len(normalized_reason) > REASON_LIMIT:
        raise BootstrapError(f"--reason must be at most {REASON_LIMIT} characters")
    normalized_reference = (reference or "").strip() or None
    if not dry_run and normalized_reference is None:
        raise BootstrapError(
            "--close-incident requires --reference (approved change) unless --dry-run"
        )
    if normalized_reference is not None and not REFERENCE_PATTERN.fullmatch(
        normalized_reference
    ):
        raise BootstrapError("incident close reference is invalid")
    operator = actor or operator_identity.resolve_operator_identity()
    payload = {
        "mode": "incident-close",
        "dry_run": bool(dry_run),
        "incident_ids": ids,
        "reason": normalized_reason,
        "reference": normalized_reference,
        "operator": operator,
    }
    result = _run_reconcile(site, payload, script=INCIDENT_CLOSE_SCRIPT)
    results = result.get("results")
    if not isinstance(results, list):
        raise BootstrapError("incident close returned no results")
    result["actor"] = operator
    result["dry_run"] = bool(dry_run)
    result["reference"] = normalized_reference
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


def result_lines(result: dict[str, Any]) -> list[str]:
    """One line per incident: ``<id>: closed | already-recovered | would-close
    | refused(<reason>)``."""

    lines = []
    for item in result.get("results") or []:
        outcome = str(item.get("outcome"))
        if outcome == "refused":
            outcome = f"refused({item.get('reason') or 'no reason given'})"
        lines.append(f"{item.get('incident_id')}: {outcome}")
    return lines


def exit_code(result: dict[str, Any]) -> int:
    return 1 if result.get("refused_incident_ids") else 0
