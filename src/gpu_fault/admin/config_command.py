"""The ``gpu-fault-admin config`` verb's arguments.

The apply itself is ``cli._apply_admin_config``; the parser lives here so the
verb's flags sit next to the one that guards a lowered topology
(``capacity_fleet.ACCEPT_FLAG``) instead of growing ``cli.py`` past its size
ratchet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpu_fault.admin import capacity_fleet, warm_spare


def add_admin_config_command(commands: Any) -> None:
    config = commands.add_parser(
        "config",
        usage=(
            "gpu-fault-admin config --state-dir STATE_DIR "
            "[--file ADMIN_CONFIG] [--dry-run] [--reference REFERENCE] "
            f"[{capacity_fleet.ACCEPT_FLAG}]"
        ),
        help=(
            "apply the edited <state-dir>/admin-config.yaml to the live site "
            "(Aurora window and the affected control-plane roles)"
        ),
    )
    # Not ``required``: the ``config spare`` sub-action carries its own
    # ``--state-dir``; ``_managed_site_file`` refuses a bare ``config`` without it.
    config.add_argument("--state-dir", type=Path, metavar="STATE_DIR")
    config.add_argument(
        "--file",
        dest="admin_config_file",
        type=Path,
        metavar="ADMIN_CONFIG",
        help=("private AdminConfig YAML; defaults to <state-dir>/admin-config.yaml"),
    )
    config.add_argument(
        "--reference",
        metavar="REFERENCE",
        help=(
            "change or maintenance-window reference for the audit record; "
            "defaults to the approver identity and the start time"
        ),
    )
    config.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "print the change plan (affected roles, changed fields, Aurora "
            "change) from local state only; nothing is verified or written"
        ),
    )
    capacity_fleet.add_accept_flag(config)
    warm_spare.add_config_spare_command(config)
