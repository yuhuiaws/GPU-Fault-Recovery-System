"""Uninstall policy inputs and phases, independent of the execution driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gpu_fault.admin.aws_commands import FinalSnapshotPolicy
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite

CpuDisposition = Literal["keep", "delete"]
UNINSTALL_PHASES = (
    "STARTED",
    "REGISTRY_EXPORTED",
    "KUBERNETES_VERIFIED",
    "NON_AURORA_DELETE_IN_PROGRESS",
    "NON_AURORA_VERIFIED",
    "CPU_DELETE_IN_PROGRESS",
    "CPU_VERIFIED",
    "READY_TO_DELETE_AURORA",
    "AURORA_DELETE_IN_PROGRESS",
    "COMPLETED",
)


@dataclass(frozen=True)
class UninstallRequest:
    """One uninstall.

    ``cpu_disposition="keep"`` is a reinstall: the CPU cluster and the Aurora
    cluster stay, so the site's incident, workflow and registry records (and
    the quarantine taints keyed on those incident ids) survive to the next
    ``deploy``, which adopts the existing cluster. ``reset_database`` is the
    explicit opt-in to wipe Aurora on a reinstall. ``cpu_disposition="delete"``
    is retirement: Aurora goes with the CPU cluster, and
    ``final_snapshot_policy`` decides whether a final snapshot stays for audit.
    """

    site: RenderedSite
    cpu_disposition: CpuDisposition
    confirmation: str
    final_snapshot_policy: FinalSnapshotPolicy = "retain"
    reset_database: bool = False
    repository_root_override: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root_override, bool):
            raise BootstrapError("repository_root_override must be a boolean")
        if self.cpu_disposition not in {"keep", "delete"}:
            raise BootstrapError("invalid CPU cluster disposition")
        if self.final_snapshot_policy not in {"retain", "skip"}:
            raise BootstrapError("invalid Aurora final snapshot policy")
        if not isinstance(self.reset_database, bool):
            raise BootstrapError("reset_database must be a boolean")
        if (
            self.cpu_disposition == "keep"
            and self.final_snapshot_policy == "skip"
            and not self.reset_database
        ):
            # A plain reinstall keeps Aurora; only an explicit database reset
            # can choose to skip its final audit snapshot.
            raise BootstrapError(
                "--aurora-final-snapshot skip is only valid with --cpu-cluster "
                "delete or with --reset-database; a reinstall keeps the Aurora "
                "cluster"
            )
        if self.cpu_disposition == "delete" and self.reset_database:
            raise BootstrapError(
                "--reset-database is only valid with --cpu-cluster keep; "
                "--cpu-cluster delete already deletes the Aurora cluster"
            )
