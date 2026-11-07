"""``gpu-fault-admin backfill-transparency-log``: log what a deploy signed without it.

A deploy with ``--allow-unlogged-signature --reason`` signs its release with
the site key only, and ``status`` names that release until its signature is in
the public transparency log. Once the log answers again, this verb puts every
such signature of the site there -- the same signature, never a new one -- and
rewrites each bundle with its entry (``scripts/backfill_transparency_log.py``).
It changes nothing on the clusters and needs only the site's public key; it
runs under the site operation lock because it rewrites the state directory's
signature bundles.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from gpu_fault.admin.execution import run_driver
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.site import RenderedSite

VERB = "backfill-transparency-log"


def add_command(
    commands: Any, add_managed_site_arguments: Callable[[Any], None]
) -> None:
    command = commands.add_parser(
        VERB,
        usage=f"gpu-fault-admin {VERB} --state-dir STATE_DIR",
        help=(
            "after a deploy signed with --allow-unlogged-signature, upload each "
            "signature the site key made without a transparency-log entry to the "
            "public transparency log once it answers, and rewrite its bundle with "
            "the entry (the signature itself is unchanged); clears the status "
            "finding. Changes nothing on the clusters"
        ),
    )
    add_managed_site_arguments(command)


def run_command(arguments: argparse.Namespace, *, site: RenderedSite) -> int:
    state_dir = site.source.parent
    with administrator_operation_lock(state_dir):
        completed = run_driver(
            [
                sys.executable,
                str(site.repository_root / "scripts/backfill_transparency_log.py"),
                "--state-dir",
                str(state_dir),
                "--repository-root",
                str(site.repository_root),
                "--public-key",
                str(state_dir / "release-signing" / "cosign.pub"),
            ],
            cwd=site.repository_root,
            check=False,
        )
    return completed.returncode
