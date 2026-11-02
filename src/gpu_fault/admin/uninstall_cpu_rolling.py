"""Uninstall's journaled RESTORE of the CPU rolling schedule (batch 240).

Before uninstall deletes anything -- the rollback alarms included -- it puts
the CPU instance group's ``ScheduledUpdateConfig`` back to the pre-install
state the deploy recorded (the installation ledger's ``cpu_hyperpod`` row, else
the deploy-host journal), with the same delete-then-create as the deploy step
(``cpu_rolling_update.restore_schedule``). A recorded rolling config with
alarms the product does not own is restored exactly. The restore is refused
-- the uninstall stops before any deletion and says why -- while an AMI update
is in flight. Retiring the CPU cluster (CPU disposition ``delete``) skips
it: the group goes with the cluster. A site whose ledger records neither an
alarm nor a pre-install schedule is left exactly as before: no read, no write.

When a CPU group's schedule still names the product's alarms afterwards (the
deploy found it already rolling and recorded nothing, as on the live site),
uninstall says so and prints the ``hyperpod-scheduled-update`` command that
removes or replaces that configuration -- it does not change the schedule
itself. AWS does not document what a scheduled update does when a listed
alarm does not exist; a reinstall re-creates the alarms under the same names,
which makes the schedule valid again.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gpu_fault.admin import hyperpod_rollback as rollback
from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.admin.config import load_desired_admin_config
from gpu_fault.admin.cpu_rolling_update import (
    Clock,
    Target,
    load_journal,
    restore_schedule,
)
from gpu_fault.admin.resource_registry_cpu_rolling import (
    ALARM_TYPE,
    pre_install_records,
)
from gpu_fault.installation_resources import InstallationResourceSnapshot

__all__ = ["STATE_KEY", "restore_cpu_rolling_update"]

STATE_KEY = "cpu_rolling_update_restore"
Transition = Callable[..., None]


def _follow_up(
    runner: CommandRunner, target: Target, state_dir: Path, eks_arn: str
) -> list[str]:
    described = runner.aws_json(
        target.region, "sagemaker", "describe-cluster", "--cluster-name", target.cluster
    )
    lines = []
    for group in described.get("InstanceGroups") or ():
        names = set(rollback.rollback_alarm_names(group.get("ScheduledUpdateConfig")))
        if not names & set(target.alarm_names):
            continue
        name = group.get("InstanceGroupName")
        lines.append(
            f"the CPU instance group {name} still names the rollback alarms "
            f"{', '.join(sorted(names & set(target.alarm_names)))} in its "
            "AutoRollbackConfiguration, and uninstall deletes them. AWS does not "
            "document what a scheduled update does when a listed alarm does not "
            "exist; a reinstall re-creates the alarms under the same names. To "
            "remove the schedule: gpu-fault-admin hyperpod-scheduled-update "
            f"--state-dir {state_dir} --gpu-cluster-arn {eks_arn} --instance-group "
            f"{name} (add --rolling --rollback-alarm <your alarm> to keep it rolling "
            "on another alarm); the command prints the update-cluster requests"
        )
    return lines


def restore_cpu_rolling_update(
    request: Any,
    runner: CommandRunner,
    snapshot: InstallationResourceSnapshot,
    state_path: Path,
    state: dict[str, Any],
    transition: Transition,
    *,
    clock: Clock | None = None,
) -> None:
    """Journaled under ``STATE_KEY``; a resume after it completed skips it."""

    if STATE_KEY in state:
        return
    site = request.site
    state_dir = site.source.parent
    config = site.release_config
    recorded = pre_install_records(snapshot.resources)
    alarms = [item for item in snapshot.resources if item.resource_type == ALARM_TYPE]
    journal = load_journal(state_dir)
    journaled = any(
        isinstance(entry, dict) and isinstance(entry.get("pre_install"), dict)
        for entry in journal["groups"].values()
    )
    if not (recorded or alarms or journaled):
        return
    if request.cpu_disposition == "delete":
        transition(
            state_path,
            state,
            state["phase"],
            **{
                STATE_KEY: {
                    "outcome": "skipped",
                    "reason": "the CPU cluster is deleted",
                }
            },
        )
        return
    target = Target.of(config, load_desired_admin_config(state_dir).cpu_control_plane)
    results = restore_schedule(
        runner, target, state_dir, recorded=recorded, clock=clock
    )
    follow_up = _follow_up(runner, target, state_dir, str(config["cpu_eks_arn"]))
    for line in follow_up:
        print(f"gpu-fault-admin uninstall: {line}", file=sys.stderr, flush=True)
    transition(
        state_path,
        state,
        state["phase"],
        **{STATE_KEY: {"outcome": "done", "groups": results, "follow_up": follow_up}},
    )
