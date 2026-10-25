"""The shared mounts a node has had, so one that vanishes is reported.

``_shared_filesystems`` discovers the NFS and Lustre mounts from the mount
table every tick. A table has no memory: a mount that is unmounted simply
stops appearing, so a shared filesystem that went away from under a training
job -- a forced or lazy ``umount`` by an operator, a lifecycle script that did
not bring the mount back after a reboot -- stopped being sampled, and
``shared_filesystem_unavailable`` never rose (GF-GENERAL-LIVE-005
``mount_disappears``). This module is that memory, and the one real data-path
read the collector makes per shared mount (``io_only_failure``).

What is remembered, and why only that:

* Node-scoped shared mounts: those a node carries for its life -- fstab, a
  systemd mount unit, a provisioning script (``/fsx``, ``/home``,
  ``/mnt/nfs``). Their disappearance is never routine.
* Not workload-scoped mounts: a Pod volume under ``/var/lib/kubelet/pods``, a
  CSI staging mount under ``/var/lib/kubelet/plugins``, a container runtime's
  own mounts (:data:`WORKLOAD_MOUNT_ROOTS`). Those come and go with the
  workload -- every job teardown unmounts them, and the control plane already
  follows the Pod lifecycle. Remembering them would raise a CRITICAL storage
  finding on every teardown that straddles a tick.
* Not automounter children: a mount whose parent mount is ``autofs`` expires
  when idle and comes back on access, by design.

What clears a remembered mount:

* It is mounted again at the same mount point (recovery): the record is
  updated and the gauge reads 0 again.
* Its mount point directory is gone from the host. An unmount followed by
  ``rmdir`` is a deliberate teardown, not a failure; a failed or lazy unmount
  leaves the directory in place, which is exactly the dangerous case -- the
  workloads keep writing under the old path, into the local disk. Judged
  with the same bounded directory read as the data-path probe, never a bare
  ``stat`` that could hang on a wedged parent mount.
* An operator removes the state file and restarts the collector.

Nothing clears a remembered mount with time alone: a node held by a vanished
``/fsx`` stays held until someone remounts it or removes the mount point. The
memory is bounded by :data:`MAX_REMEMBERED_MOUNTS` instead; past the cap a new
mount is not remembered, and the collector says so once.

Across restarts: the memory is a JSON file under ``/var/lib/gpu-fault`` (the
unit's ``ReadWritePaths``), rewritten only when it changes. The collector
restarts often -- the systemd watchdog kills a tick that stops advancing, which
is what a hung mount does -- and a memory that restarted empty would forget
the vanished mount exactly when it mattered. The file survives a reboot too: a
mount that fstab failed to bring back is a real failure for the workloads,
while a decommissioned mount whose directory was removed is forgotten on the
first tick. A reimaged node loses the file with the rest of
``/var/lib/gpu-fault``, as a fresh node should. An unreadable or malformed
file starts the memory empty, with one warning: a collector must never refuse
to run because of its own bookkeeping.

Standard library only: this module is part of the node runtime.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from gpu_fault.collectors.host.mount_table import MountEntry, covering_mount

#: The filesystem types ``_shared_filesystems`` samples.
SHARED_FILESYSTEM_TYPES: frozenset[str] = frozenset({"lustre", "nfs", "nfs4"})

#: Mount points at or below these roots belong to a workload or to the
#: container runtime, not to the node (module docstring).
WORKLOAD_MOUNT_ROOTS: tuple[str, ...] = (
    "/var/lib/kubelet/pods",
    "/var/lib/kubelet/plugins",
    "/var/lib/containerd",
    "/run/containerd",
    "/var/lib/docker",
    "/run/docker",
    "/var/lib/containers",
    "/run/k3s",
    "/var/lib/rancher",
)

#: The memory's size bound: one record per node-scoped shared mount ever seen
#: and not torn down. A node with more fstab/systemd shared mounts than this is
#: unheard of; the cap exists so a misclassified mount churn cannot grow the
#: state file without limit.
MAX_REMEMBERED_MOUNTS = 64

STATE_VERSION = 1


def shared_mounts(table: Sequence[MountEntry]) -> list[MountEntry]:
    """The visible NFS/Lustre mount at each mount point of ``table``.

    Several mounts stacked on one mount point are a chain; the one a lookup
    ends on is what the workloads use. A shared mount another filesystem is
    stacked over is not reachable at its path and is not listed.
    """

    result = []
    for point in dict.fromkeys(
        entry.mount_point for entry in table if entry.fs_type in SHARED_FILESYSTEM_TYPES
    ):
        visible = covering_mount(tuple(table), point)
        if (
            visible is not None
            and visible.mount_point == point
            and visible.fs_type in SHARED_FILESYSTEM_TYPES
        ):
            result.append(visible)
    return result


def node_scoped(entry: MountEntry, table: Sequence[MountEntry]) -> bool:
    """Whether ``entry`` is a mount the node keeps for its life (module docstring)."""

    point = entry.mount_point
    if any(
        point == root or point.startswith(root + "/") for root in WORKLOAD_MOUNT_ROOTS
    ):
        return False
    parent = next((item for item in table if item.mount_id == entry.parent_id), None)
    return parent is None or parent.fs_type != "autofs"


def read_directory_entry(mount_point: str) -> None:
    """The smallest real data-path round trip on a mount: one ``getdents``.

    ``statvfs`` is one FSSTAT (NFS) or OBD_STATFS (Lustre) RPC that a server
    answers from its own bookkeeping; an export whose data path is broken --
    permissions changed server-side, a stuck backend -- answers it and reads
    healthy. Opening the mount point and reading one directory entry goes
    through the client's access check and a READDIR (an MDS request on
    Lustre), so a permission failure is an errno here and a stuck path is a
    hang the caller bounds. One entry, not the listing: a training root with a
    hundred thousand files must not cost a page per tick.

    Read-only and non-destructive: no file is created, written or synced on a
    customer's filesystem. The NFS client may answer from its directory cache
    until the attribute cache expires (``acdirmax``, 60 s by default), which is
    the detection latency of a failure that leaves FSSTAT working.
    """

    descriptor = os.open(mount_point, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        with os.scandir(descriptor) as entries:
            next(entries, None)
    finally:
        os.close(descriptor)


@dataclass
class SharedMountRecord:
    """One node-scoped shared mount the memory holds."""

    mount_point: str
    #: The ``source`` field of the mount table: ``10.0.0.2:/export``,
    #: ``10.0.0.1@tcp:/fsx``. Named in the unavailable sample so an operator
    #: (and, later, a failure-domain grouping) knows which export is meant.
    source: str
    fs_type: str
    first_seen: str
    missing_since: str | None = None


class SharedMountObservation(NamedTuple):
    """What one tick's mount table meant for the memory."""

    #: Remembered mounts the table no longer lists, oldest first.
    missing: tuple[SharedMountRecord, ...]
    #: Mounts present this tick that the memory refused because it is full.
    refused: tuple[str, ...]


class SharedMountMemory:
    """The node-scoped shared mounts seen so far, kept in ``state_path``.

    ``state_path`` None keeps the memory in the process only (tests, or a
    deployment that explicitly disables the file): a restart then forgets
    every mount, which the module docstring explains is not the default.
    """

    def __init__(
        self,
        state_path: Path | None,
        *,
        max_entries: int = MAX_REMEMBERED_MOUNTS,
    ) -> None:
        if max_entries < 1:
            raise ValueError("the shared mount memory must hold at least one mount")
        self.state_path = state_path
        self._max_entries = max_entries
        self._records: dict[str, SharedMountRecord] = {}
        self._dirty = False

    @property
    def records(self) -> Mapping[str, SharedMountRecord]:
        return dict(self._records)

    def load(self) -> str | None:
        """Read the state file; the reason the memory starts empty, if it does."""

        self._records = {}
        self._dirty = False
        if self.state_path is None:
            return None
        try:
            if not self.state_path.exists():
                return None
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("mounts"), dict):
                raise ValueError("the shared mount memory must be an object of mounts")
            records = {}
            for mount_point, item in value["mounts"].items():
                if not isinstance(mount_point, str) or not isinstance(item, dict):
                    raise ValueError(
                        f"malformed shared mount record for {mount_point!r}"
                    )
                missing_since = item.get("missing_since")
                records[mount_point] = SharedMountRecord(
                    mount_point=mount_point,
                    source=str(item.get("source", "")),
                    fs_type=str(item.get("fs_type", "")),
                    first_seen=str(item.get("first_seen", "")),
                    missing_since=(
                        str(missing_since) if missing_since is not None else None
                    ),
                )
        except (OSError, ValueError) as error:
            return (
                f"the shared mount memory {self.state_path} cannot be read "
                f"({error}); starting without one -- a shared mount that vanished "
                "before this start is not reported until it is seen again"
            )
        if len(records) > self._max_entries:
            # Written by a release with a larger cap, or edited by hand: keep
            # the oldest-known mounts, which are the ones most likely to be the
            # node's own, and persist the trimmed set.
            kept = sorted(records.values(), key=lambda record: record.first_seen)
            records = {
                record.mount_point: record for record in kept[: self._max_entries]
            }
            self._dirty = True
        self._records = records
        return None

    def observe(
        self, present: Mapping[str, tuple[str, str]], observed_at: datetime
    ) -> SharedMountObservation:
        """Record the node-scoped shared mounts the table lists this tick.

        ``present`` maps each mount point to its ``(source, fs_type)``. A mount
        seen for the first time is remembered (up to the cap); one seen again
        after being missing is cleared; every remembered mount the table no
        longer lists is returned as missing, stamped the first tick it was.
        """

        stamp = observed_at.isoformat()
        refused = []
        for mount_point, (source, fs_type) in present.items():
            record = self._records.get(mount_point)
            if record is None:
                if len(self._records) >= self._max_entries:
                    refused.append(mount_point)
                    continue
                self._records[mount_point] = SharedMountRecord(
                    mount_point, source, fs_type, stamp
                )
                self._dirty = True
                continue
            if (
                record.source != source
                or record.fs_type != fs_type
                or record.missing_since is not None
            ):
                record.source = source
                record.fs_type = fs_type
                record.missing_since = None
                self._dirty = True
        missing = []
        for mount_point, record in self._records.items():
            if mount_point in present:
                continue
            if record.missing_since is None:
                record.missing_since = stamp
                self._dirty = True
            missing.append(record)
        missing.sort(key=lambda record: record.missing_since or "")
        return SharedMountObservation(tuple(missing), tuple(refused))

    def forget(self, mount_point: str) -> None:
        """Drop a mount whose mount point is gone from the host (torn down)."""

        if self._records.pop(mount_point, None) is not None:
            self._dirty = True

    def save(self) -> str | None:
        """Persist the memory if it changed; the reason it could not, if any."""

        if not self._dirty:
            return None
        if self.state_path is None:
            self._dirty = False
            return None
        try:
            _write_atomically(
                self.state_path,
                {
                    "version": STATE_VERSION,
                    "mounts": {
                        mount_point: _record_fields(record)
                        for mount_point, record in sorted(self._records.items())
                    },
                },
            )
        except OSError as error:
            return (
                f"the shared mount memory {self.state_path} cannot be written "
                f"({error}); a shared mount that vanishes is reported until this "
                "collector restarts, not across the restart"
            )
        self._dirty = False
        return None


def _record_fields(record: SharedMountRecord) -> dict[str, Any]:
    fields = asdict(record)
    del fields["mount_point"]
    return fields


def _write_atomically(path: Path, value: dict[str, Any]) -> None:
    """Write-fsync-rename, then fsync the directory: a crash leaves old or new."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
