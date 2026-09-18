"""Custody adapters for administrator commands, site preflight and join state."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapMutationRequired, CommandRunner
from gpu_fault.admin.node_key_custody_admin import (
    CustodyPreparationRequired,
    CustodyReconciliationRequired,
    provision_admin_custody,
    site_custody_context,
)
from gpu_fault.admin.node_key_custody_admin_config import (
    configure_admin_custody,
    load_admin_custody,
    registration_path,
)
from gpu_fault.admin.node_key_custody_admin_probe import CustodyReadRunner
from gpu_fault.admin.node_key_custody_models import CustodyError

if TYPE_CHECKING:
    from gpu_fault.admin.site import RenderedSite

CUSTODY_PAUSE_ERRORS = (
    CustodyPreparationRequired,
    CustodyReconciliationRequired,
    CustodyError,
)


def add_custody_command(commands: Any) -> None:
    custody = commands.add_parser(
        "node-key-custody", help="configure prospective node-key custody evidence"
    )
    actions = custody.add_subparsers(dest="custody_command", required=True)
    configure = actions.add_parser("configure")
    configure.add_argument("--state-dir", type=Path, required=True)
    configure.add_argument("--file", dest="custody_config", type=Path, required=True)
    configure.add_argument("--trust-sha256", required=True)


def run_custody_command(arguments: argparse.Namespace) -> int:
    registration = configure_admin_custody(
        arguments.state_dir, arguments.custody_config, arguments.trust_sha256
    )
    print(
        json.dumps(
            {
                "status": "CONFIGURED",
                "cluster_count": len(registration.selection.clusters),
                "registration": str(registration_path(arguments.state_dir.resolve())),
            },
            sort_keys=True,
        )
    )
    return 0


def assert_site_custody_current(site: RenderedSite, runner: CommandRunner) -> None:
    registration = load_admin_custody(site.source.parent)
    if registration is None:
        return
    from gpu_fault.admin.bootstrap import discover_cluster

    readonly = CustodyReadRunner(runner)
    for entry in site.release_config["clusters"]:
        if entry["eks_cluster_arn"] not in registration.selection.clusters:
            continue
        cluster = discover_cluster(
            readonly,
            cluster_arn=entry["eks_cluster_arn"],
            role="gpu",
            context=entry["context"],
        )
        context = site_custody_context(site, cluster, str(entry["cluster_id"]))
        try:
            provision_admin_custody(
                readonly,
                context,
                fleet_master_file=Path("/unused-by-probe"),
                probe_only=True,
            )
        except BootstrapMutationRequired:
            raise CustodyReconciliationRequired(
                "configured custody is incomplete; public preflight cannot prepare or sign it"
            ) from None


def record_join_custody_pause(
    state_path: Path,
    state: dict[str, Any],
    error: CustodyPreparationRequired | CustodyReconciliationRequired | CustodyError,
) -> None:
    state["phase"] = (
        "CUSTODY_AWAITING_AUTHORIZATION"
        if isinstance(error, CustodyPreparationRequired)
        else "CUSTODY_BLOCKED"
    )
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(state_path, state)
