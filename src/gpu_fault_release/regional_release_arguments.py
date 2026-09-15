"""Command line of the regional release engine (``python -m gpu_fault_release.rollout``).

Split out of ``rollout.py`` so that module stays under the file budget; the
names are re-exported there, which is where callers and tests still look them
up.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Regional GPU fault release orchestrator"
    )
    value.add_argument(
        "mode",
        choices=(
            "plan",
            "preflight",
            "release-summary",
            "release-diff",
            "status",
            "bootstrap",
            "deploy",
            "upgrade",
            "resume",
            "rollback",
            "commit",
            "stage-noop",
            "join-cluster",
            "activate-cluster",
            "fail-cluster",
            "rollback-cluster",
            "drain-cluster",
            "remove-cluster",
            "sync-state",
            "verify",
            "stability",
        ),
    )
    value.add_argument("--config", required=True, type=Path)
    # Repeatable: drain-cluster publishes every named cluster in one revision;
    # the other cluster modes take exactly one (see rollout._single_cluster_id).
    value.add_argument(
        "--cluster-id", action="append", dest="cluster_ids", metavar="CLUSTER_ID"
    )
    value.add_argument(
        "--plan-mode",
        choices=(
            "bootstrap",
            "deploy",
            "upgrade",
            "rollback",
            "join-cluster",
            "remove-cluster",
        ),
        default="upgrade",
    )
    value.add_argument("--dry-run", action="store_true")
    # Engine-internal: set by `recover_failed_upgrade` (in-process) and by
    # `scripts/release_failure_recovery.py` on `rollback`, never typed by an
    # operator, so hidden from --help; `parse_arguments` refuses it elsewhere.
    # It lets the in-flight install check proceed (logging) when no
    # control-plane Pod can answer the store read.
    value.add_argument("--automatic", action="store_true", help=argparse.SUPPRESS)
    # `status`: every health check instead of the cheap two. `preflight`: the
    # passing checks' details instead of their names. GPU_FAULT_FULL_REPORT=1
    # does the same for a wrapper that cannot add the flag.
    value.add_argument("--full", action="store_true")
    return value
