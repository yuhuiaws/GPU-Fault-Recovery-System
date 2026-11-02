"""``gpu-fault-admin cpu-rolling-update``: run the deploy's last step now.

The deploy ends with the CPU control plane's rolling update
(``cpu_rolling_update.finish_deploy``: the rollback alarms, then the rolling
schedule per ``spec.cpuControlPlane.rollingUpdate``). A step that was deferred
-- an AMI update in flight, a write refusal, the next window too close -- or
failed is run again here without a whole deploy, under the site operation
lock; ``status --full``'s ``cpu_rolling_update`` check names this verb.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.admin.config import load_desired_admin_config
from gpu_fault.admin.cpu_rolling_update import finish_deploy, record_last_run
from gpu_fault.admin.membership_lock import (
    administrator_operation_lock,
    reload_site_for_mutation,
)
from gpu_fault.admin.site import RenderedSite

__all__ = ["VERB", "add_command", "run_command", "run_step"]

VERB = "cpu-rolling-update"


def add_command(
    commands: Any, add_managed_site_arguments: Callable[[Any], None]
) -> None:
    command = commands.add_parser(
        VERB,
        usage=f"gpu-fault-admin {VERB} --state-dir STATE_DIR",
        help=(
            "run the deploy's last step now: create or re-point the CPU rollback "
            "alarms and apply spec.cpuControlPlane.rollingUpdate to the CPU "
            "HyperPod schedule (delete, read back, create, read back)"
        ),
        description=(
            "Writes: CloudWatch put-metric-alarm/tag-resource on the two rollback "
            "alarms and, for mode rolling, sagemaker update-cluster on the CPU "
            "control-plane cluster -- the same step every deploy ends with. "
            "Deferred while an AMI update is in flight or the next scheduled window "
            "is near; never adds a schedule to a group that has none."
        ),
    )
    add_managed_site_arguments(command)


def run_command(arguments: argparse.Namespace, *, site: RenderedSite) -> int:
    """Exit 0 when the step is DONE, 1 when it was deferred or failed."""

    del arguments
    return run_step(site)


def run_step(site: RenderedSite, *, runner: CommandRunner | None = None) -> int:
    """``run_command`` with an injectable runner."""

    state_dir = site.source.parent
    with administrator_operation_lock(state_dir):
        site = reload_site_for_mutation(site)
        result = finish_deploy(
            site.release_config,
            load_desired_admin_config(state_dir).cpu_control_plane,
            state_dir,
            runner=runner,
        )
        record_last_run(state_dir, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    for warning in result.get("warnings") or ():
        print(f"gpu-fault-admin {VERB}: warning: {warning}", file=sys.stderr)
    print(f"gpu-fault-admin {VERB}: {result.get('summary')}", file=sys.stderr)
    return 0 if result.get("status") == "DONE" else 1
