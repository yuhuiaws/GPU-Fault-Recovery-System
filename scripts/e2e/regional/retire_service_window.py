#!/usr/bin/env python3
"""Retire the stale DESTR-008 service windows a case directory still holds open.

The same retirement GF-GENERAL-LIVE-030 ``not_ready`` runs as its pre-step
(``service_window_retire.retire_stale_windows``), for windows no pending 030
run will reach -- the HyperPod lanes' journals for a spare that is no longer
the lane's candidate, or a DESTR-008 run's own state directory. It goes through
the harness's own host probe like every fixture action and acts only on proof
(see ``service_window_retire`` and ``probes/warm_spare_retire.py``); a journal
that cannot be proven stale is reported with every failed proof and left as it
was. Plan-only by default: the local proofs, the Node and the earlier probe's
objects are read, nothing is created or changed. ``--execute`` needs ``--confirm
RETIRE_STALE_SERVICE_WINDOW`` and a maintenance window; the report is written
under the state directory's ``retire/`` and printed. Exit 0 when every open
journal was retired (or none was open), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_fault.admin.atomic_json import write_json_atomic  # noqa: E402
from gpu_fault.admin.operator_identity import (  # noqa: E402
    local_operator_identity,
    resolve_operator_identity,
)
from scripts.e2e.regional.collector_action_guard import action_window  # noqa: E402
from scripts.e2e.regional.live_driver_guard import add_live_arguments  # noqa: E402
from scripts.e2e.regional.plain_case_identity import (  # noqa: E402
    add_plain_identity_arguments,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    install_abort_signals,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.service_window_node_restore import (  # noqa: E402
    stale_journals,
)
from scripts.e2e.regional.service_window_retire import (  # noqa: E402
    RETIRE_STATE,
    RetireSettings,
    inspect_journal,
    retire_stale_windows,
)
from scripts.e2e.regional.site_profile import install_site_profile  # noqa: E402

CONFIRMATION = "RETIRE_STALE_SERVICE_WINDOW"
DEFAULT_CASE = "GF-GENERAL-LIVE-030.not_ready"
DEFAULT_STATE = "spare-service"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Retire the stale DESTR-008 service windows a case directory holds "
            "open on one node, through the harness host probe and only on proof."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    add_plain_identity_arguments(value)
    value.add_argument("--node", required=True)
    value.add_argument("--host-probe-image", default="")
    value.add_argument(
        "--case-key",
        default=DEFAULT_CASE,
        help=f"the case directory under <run-dir>/cases (default {DEFAULT_CASE})",
    )
    value.add_argument(
        "--state",
        default=DEFAULT_STATE,
        help=f"the window state directory in the case directory (default {DEFAULT_STATE})",
    )
    return value


def deadline(value: str) -> datetime:
    if not value.strip():
        raise RegionalFixtureError(
            "--maintenance-window-end or GPU_FAULT_ACCEPTANCE_WINDOW_END is required"
        )
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RegionalFixtureError("maintenance-window-end must include a timezone")
    return parsed.astimezone(timezone.utc)


def node_uid_reader(regional: Any) -> Any:
    def read(node: str) -> str:
        try:
            return str(regional.node_snapshot(node).get("uid") or "")
        except RegionalFixtureError:
            return ""

    return read


def main(argv: list[str] | None = None) -> int:
    install_site_profile()
    arguments = parser().parse_args(argv)
    os.umask(0o077)
    install_abort_signals()
    settings = settings_from_arguments(arguments)
    regional = RegionalLiveFixture(settings)
    directory = Path(arguments.run_dir) / "cases" / arguments.case_key / arguments.state
    retire = RetireSettings(
        kubeconfig=settings.gpu_kubeconfig,
        context=settings.gpu_context,
        namespace=settings.namespace,
        image=arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
        node_uid=node_uid_reader(regional),
    )
    journals = stale_journals(directory, arguments.node)
    plan: dict[str, Any] = {
        "action": "retire-stale-service-window",
        "directory": str(directory),
        "node": arguments.node,
        "confirmation": CONFIRMATION,
        "journals": [
            inspect_journal(path, arguments.node, settings=retire) for path in journals
        ],
        "mutations": [
            "create one retire host probe on the node (its own Pod/ConfigMap, cleaned)",
            "on the node: remove the window's drop-in, its three units and the claim "
            "by their receipts, systemctl daemon-reload (no service restart), remove "
            "the earlier probe's /run script copy and the window directory",
            "delete the earlier probe's Pod and ConfigMap (proven its own, UID "
            "preconditions) and close its ownership record",
            "mark the journal CLOSED with a retirement record (who, when, proofs)",
        ],
        "hard_stop": (
            "any failed proof (journal phase, lost supervision, Node replaced, a "
            "service invocation other than the one the window restored, an "
            "artifact not matching its receipt, an active unit, another live "
            "window or drop-in, a foreign probe object) leaves everything as it was"
        ),
    }
    if not arguments.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if arguments.confirm != CONFIRMATION:
        raise RegionalFixtureError(f"confirmation must be exactly {CONFIRMATION}")
    window_end = deadline(arguments.maintenance_window_end)
    required(retire.image, "host probe image")
    with action_window(window_end):
        reports = retire_stale_windows(
            directory,
            arguments.node,
            settings=retire,
            by=resolve_operator_identity(fallback=local_operator_identity()),
            via="scripts/e2e/regional/retire_service_window.py",
        )
    result = {
        **plan,
        "reports": reports,
        "retired": all(report["retired"] for report in reports),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json_atomic(directory / RETIRE_STATE / f"retire-report-{stamp}.json", result)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["retired"] else 1


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
