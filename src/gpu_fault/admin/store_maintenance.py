"""``gpu-fault-admin store-maintenance``: reclaim hot-table bloat.

A day of campaign churn halves control-plane ingestion when the hot tables'
heap and index pages are never returned: the perf verdict on a warm store
failed on the reserved 503 share (675/24576) with the same workload, pods and
Aurora capacity that passed on a fresh database -- the metric upsert walked
bloated indexes and every admission read scanned hundreds of pages for a
handful of rows. Schema v19 stops the growth (fillfactor, threshold-only
autovacuum, HOT-eligible updates); this verb returns the pages already lost.

Plan (no ``--confirm``) is read-only: it prints every hot table's live rows,
pages, index bytes and the proposed action. Execute (``--confirm
STORE_MAINTENANCE`` plus an approved ``--reference``) takes the site
operation lock -- so it refuses while a release or configuration mutation is
in progress -- re-plans in the Pod, refuses a plan whose identity drifted,
runs ``VACUUM (FULL, ANALYZE)`` on the small in-place tables and ``REINDEX
INDEX CONCURRENTLY`` + ``VACUUM (ANALYZE)`` on the large ones through the
product's store connection, and records before/after sizes under
``<state-dir>/store-maintenance/``. Running it twice is harmless.
"""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gpu_fault.admin import operator_identity
from gpu_fault.admin import store_maintenance_script as maintenance_script
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.membership_lock import (
    administrator_operation_lock,
    reload_site_for_mutation,
)
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.workflow_reconcile import (
    REFERENCE_PATTERN,
    run_control_plane_script,
)

CONFIRMATION = "STORE_MAINTENANCE"
STATE_ROOT = "store-maintenance"
SIZE_KEYS = (
    "live_rows",
    "dead_rows",
    "heap_pages",
    "heap_bytes",
    "index_bytes",
    "toast_bytes",
    "total_bytes",
)


def maintenance_source() -> str:
    """The in-Pod script: the maintenance module's own source text."""

    return inspect.getsource(maintenance_script)


def approved_reference(reference: str | None) -> str:
    normalized = (reference or "").strip()
    if not normalized:
        raise BootstrapError(
            "store-maintenance requires --reference (an approved change or "
            "maintenance-window reference)"
        )
    if not REFERENCE_PATTERN.fullmatch(normalized):
        raise BootstrapError("store maintenance reference is invalid")
    return normalized


def evidence_directory(site: RenderedSite) -> Path:
    return site.source.parent / STATE_ROOT


def _validated_plan(value: dict[str, Any]) -> dict[str, Any]:
    tables = value.get("tables")
    server = value.get("server")
    plan_sha256 = value.get("plan_sha256")
    if (
        not isinstance(tables, list)
        or not isinstance(server, dict)
        or not isinstance(plan_sha256, str)
        or len(plan_sha256) != 64
    ):
        raise BootstrapError("store maintenance plan has an unexpected shape")
    for item in tables:
        if not isinstance(item, dict) or not {"table", "action", *SIZE_KEYS} <= set(
            item
        ):
            raise BootstrapError("store maintenance plan has an unexpected shape")
    return value


def plan_store_maintenance(site: RenderedSite) -> dict[str, Any]:
    return _validated_plan(
        run_control_plane_script(site, {"mode": "plan"}, script=maintenance_source())
    )


def plan_lines(plan: dict[str, Any]) -> list[str]:
    """The review: one line per table, sizes in pages and MiB."""

    server = plan["server"]
    lines = [
        f"store: {server.get('database')} on {server.get('version')}",
        (
            "REINDEX INDEX CONCURRENTLY supported: "
            f"{'yes' if server.get('reindex_concurrently_supported') else 'no'}"
        ),
    ]
    for item in plan["tables"]:
        lines.append(
            f"{item['table']}: {item['live_rows']} live rows, "
            f"{item['dead_rows']} dead, {item['heap_pages']} heap pages "
            f"({item['heap_bytes'] / 1048576:.1f} MiB), indexes "
            f"{item['index_bytes'] / 1048576:.1f} MiB, toast "
            f"{item['toast_bytes'] / 1048576:.1f} MiB -> {item['action']}"
        )
    for table in plan.get("missing_tables", []):
        lines.append(f"{table}: absent (skipped)")
    return lines


def record_result(site: RenderedSite, result: dict[str, Any], *, now: datetime) -> Path:
    directory = evidence_directory(site)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{now:%Y%m%dT%H%M%SZ}-{result['plan_sha256'][:16]}.json"
    write_json_atomic(path, result)
    return path


def run_store_maintenance(
    site: RenderedSite,
    *,
    reference: str,
    now: datetime | None = None,
    identity: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Execute the reviewed plan and record the evidence (caller holds the lock)."""

    recorded_at = now or datetime.now(timezone.utc)
    plan = plan_store_maintenance(site)
    executed = run_control_plane_script(
        site,
        {"mode": "execute", "plan_sha256": plan["plan_sha256"]},
        script=maintenance_source(),
    )
    if executed.get("status") not in {"SUCCEEDED", "FAILED"} or not isinstance(
        executed.get("actions"), list
    ):
        raise BootstrapError("store maintenance returned an unexpected result")
    resolve = identity or (
        lambda: operator_identity.resolve_operator_identity(
            fallback=operator_identity.local_operator_identity()
        )
    )
    result = {
        "mode": "execute",
        "status": executed["status"],
        "recorded_at": recorded_at.isoformat(),
        "reference": reference,
        "operator": resolve(),
        "plan_sha256": plan["plan_sha256"],
        "server": executed.get("server", plan["server"]),
        "plan": plan["tables"],
        "actions": executed["actions"],
        "errors": executed.get("errors", []),
    }
    result["evidence_path"] = str(record_result(site, result, now=recorded_at))
    return result


def add_store_maintenance_command(
    commands: Any, add_managed_site_arguments: Callable[[Any], None]
) -> None:
    """Register ``gpu-fault-admin store-maintenance``; the CLI passes site options."""

    command = commands.add_parser(
        "store-maintenance",
        usage=(
            "gpu-fault-admin store-maintenance --state-dir STATE_DIR "
            f"--reference CHANGE_ID [--confirm {CONFIRMATION}]"
        ),
        help=(
            "reclaim hot-table bloat in the PostgreSQL store through the "
            "control plane's own connection: without --confirm prints the "
            "per-table plan (live rows, pages, index bytes, action) and changes "
            f"nothing; --confirm {CONFIRMATION} executes it under the site "
            "operation lock and records before/after sizes as evidence"
        ),
    )
    add_managed_site_arguments(command)
    command.add_argument(
        "--reference",
        required=True,
        metavar="CHANGE_ID",
        help="approved change or maintenance-window reference recorded on the evidence",
    )
    command.add_argument(
        "--confirm",
        default=None,
        metavar="TOKEN",
        help=(
            f"execute requires exactly {CONFIRMATION}; without it the command "
            "prints the plan and changes nothing"
        ),
    )


def run_store_maintenance_command(
    arguments: argparse.Namespace, *, site: RenderedSite
) -> int:
    reference = approved_reference(arguments.reference)
    confirmation = getattr(arguments, "confirm", None)
    if confirmation is None:
        plan = plan_store_maintenance(site)
        print(
            json.dumps(
                {"mode": "plan", "reference": reference, **plan},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if confirmation != CONFIRMATION:
        raise BootstrapError(f"store-maintenance requires --confirm {CONFIRMATION}")
    with administrator_operation_lock(site.source.parent):
        site = reload_site_for_mutation(site)
        result = run_store_maintenance(site, reference=reference)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "SUCCEEDED" else 1
