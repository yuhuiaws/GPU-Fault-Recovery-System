"""``gpu-fault-admin remediation-storm release|abort``: the breaker's exits.

A GPU cluster's remediation storm breaker (``gpu_fault.remediation_storm``)
holds the cluster's new destructive remediation once more nodes entered it in
600 s than one fault explains, and never closes by itself. ``release`` closes
and re-arms it: the held workflows run on their next claim, without counting
again. ``abort`` keeps it open and hands every held workflow to people:
SUPERSEDED with a ``REMEDIATION_STORM_ABORTED`` operator event, its incident
ESCALATED; a later ``release`` resumes automatic remediation.

Both run in a control-plane Pod through the exec channel every administrator
command uses (``workflow_reconcile.run_control_plane_script``), under the site
operation lock, with the operator's identity, ``--reason`` and an approved
``--reference`` on the record and an evidence file under
``<state-dir>/remediation-storm/``. ``--dry-run`` reads only. A release of a
closed breaker is a no-op; an abort is idempotent per held workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gpu_fault.admin import operator_identity
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.membership_lock import (
    administrator_operation_lock,
    reload_site_for_mutation,
)
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.workflow_reconcile import (
    REFERENCE_PATTERN,
    cluster_target,
    run_control_plane_script,
)

VERB = "remediation-storm"
STATE_ROOT = "remediation-storm"
REASON_LIMIT = 512
ACTIONS = ("release", "abort")

# Runs in the Pod on the deployed image; one that predates the breaker refuses.
SCRIPT = """
import json
import sys
from datetime import datetime, timezone

payload = json.load(sys.stdin)
try:
    from gpu_fault.workflow_reconcile import remediation_storm_action
except ImportError:
    print(json.dumps({
        "error": "the deployed control plane has no remediation storm breaker",
        "error_kind": "refused",
    }))
    raise SystemExit(1)
from gpu_fault.app import ApplicationContext

store = ApplicationContext.from_environment().store
try:
    result = remediation_storm_action(
        store,
        payload["cluster_id"],
        payload["action"],
        actor=payload.get("actor"),
        reason=payload["reason"],
        reference=payload["reference"],
        now=datetime.now(timezone.utc),
        dry_run=bool(payload.get("dry_run")),
    )
except ValueError as exc:
    print(json.dumps({"error": str(exc), "error_kind": "refused"}))
    raise SystemExit(1)
print(json.dumps(result, sort_keys=True))
"""


def add_command(
    commands: Any, add_managed_site_arguments: Callable[[Any], None]
) -> None:
    """Register ``gpu-fault-admin remediation-storm``; the CLI passes site options."""

    command = commands.add_parser(
        VERB,
        usage=(
            "gpu-fault-admin remediation-storm {release,abort} --state-dir STATE_DIR "
            "--cluster-id CLUSTER_ID --reason TEXT --reference CHANGE_ID [--dry-run]"
        ),
        help=(
            "the exits of a GPU cluster's open remediation storm breaker: release "
            "resumes automatic remediation (the held workflows run), abort keeps it "
            "open and hands the held workflows to the operator"
        ),
    )
    command.add_argument("storm_action", choices=ACTIONS, metavar="{release,abort}")
    add_managed_site_arguments(command)
    command.add_argument("--cluster-id", required=True, metavar="CLUSTER_ID")
    command.add_argument("--reason", required=True, metavar="TEXT")
    command.add_argument(
        "--reference",
        required=True,
        metavar="CHANGE_ID",
        help="approved change or incident reference recorded on the breaker",
    )
    command.add_argument(
        "--dry-run",
        action="store_true",
        help="show the breaker and the held workflows; change nothing",
    )


def _validated(arguments: argparse.Namespace) -> tuple[str, str, str]:
    reason = str(arguments.reason or "").strip()
    reference = str(arguments.reference or "").strip()
    cluster_id = str(arguments.cluster_id or "").strip()
    if not reason or len(reason) > REASON_LIMIT:
        raise BootstrapError(f"--reason must be 1..{REASON_LIMIT} characters")
    if not REFERENCE_PATTERN.fullmatch(reference):
        raise BootstrapError("remediation-storm --reference is invalid")
    if not cluster_id:
        raise BootstrapError("remediation-storm requires --cluster-id")
    return cluster_id, reason, reference


def record_result(site: RenderedSite, result: dict[str, Any], *, now: datetime) -> Path:
    directory = site.source.parent / STATE_ROOT / str(result["cluster_id"])
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
    path = directory / f"{result['action']}-{now:%Y%m%dT%H%M%SZ}-{digest[:16]}.json"
    write_json_atomic(path, result)
    return path


def _operator() -> str:
    return operator_identity.resolve_operator_identity(
        fallback=operator_identity.local_operator_identity()
    )


def run_command(arguments: argparse.Namespace, *, site: RenderedSite) -> int:
    cluster_id, reason, reference = _validated(arguments)
    cluster_target(site, cluster_id)
    payload = {
        "cluster_id": cluster_id,
        "action": arguments.storm_action,
        "reason": reason,
        "reference": reference,
        "actor": _operator(),
        "dry_run": bool(arguments.dry_run),
    }
    if arguments.dry_run:
        result = run_control_plane_script(site, payload, script=SCRIPT)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    recorded_at = datetime.now(timezone.utc)
    with administrator_operation_lock(site.source.parent):
        site = reload_site_for_mutation(site)
        result = run_control_plane_script(site, payload, script=SCRIPT)
        result = {
            **result,
            "operator": payload["actor"],
            "reason": reason,
            "reference": reference,
            "recorded_at": recorded_at.isoformat(),
        }
        result["evidence_path"] = str(record_result(site, result, now=recorded_at))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "partial" else 1
