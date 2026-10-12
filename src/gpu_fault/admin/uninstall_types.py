"""Uninstall policy inputs and phases, independent of the execution driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
    # Last: the ownership label (``cluster_scope.OWNERSHIP_LABEL``) leaves the
    # node groups of every GPU cluster the site listed (``ownership_release``),
    # after everything that still selected nodes by the live label.
    "OWNERSHIP_RELEASED",
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
    #: ``--keep-ownership-label``: the last phase records the label as
    #: ``retained`` on every GPU cluster instead of releasing it. Not part of
    #: the journal identity: a resume stuck at that phase may add it.
    keep_ownership_label: bool = False

    @classmethod
    def from_arguments(cls, site: RenderedSite, arguments: Any) -> UninstallRequest:
        """The request the parsed ``gpu-fault-admin uninstall`` arguments describe.

        Flags an older namespace lacks default to their off state, so a driver
        built against the previous argument set still constructs a request.
        """

        return cls(
            site=site,
            cpu_disposition=arguments.cpu_cluster,
            confirmation=arguments.confirm,
            final_snapshot_policy=arguments.aurora_final_snapshot,
            reset_database=bool(arguments.reset_database),
            repository_root_override=bool(
                getattr(arguments, "accept_repository_root_override", False)
            ),
            keep_ownership_label=bool(
                getattr(arguments, "keep_ownership_label", False)
            ),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root_override, bool):
            raise BootstrapError("repository_root_override must be a boolean")
        if not isinstance(self.keep_ownership_label, bool):
            raise BootstrapError("keep_ownership_label must be a boolean")
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
