"""The shared (NFS/Lustre) filesystem contributor of the host collector.

Composed into ``HostTelemetryCollector`` next to ``HostSystemMetricsMixin``,
whose bounded ``_probe_mount`` (one deadline and one tick budget for every
mount syscall) and once-only warnings it uses. The memory of the node-scoped
shared mounts and the data-path read live in ``shared_mounts``; the
``shared_filesystem_unavailable`` vocabulary is documented on
:meth:`HostSharedFilesystemsMixin._shared_filesystem_unavailable`.
"""

from __future__ import annotations

import errno
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gpu_fault.collectors.host.mount_table import MountEntry, host_mount
from gpu_fault.collectors.host.shared_mounts import (
    SharedMountMemory,
    SharedMountRecord,
    node_scoped,
    shared_mounts,
)
from gpu_fault.collectors.host.system_metrics import ProbeStillRunning
from gpu_fault.host_health import HostMetricSample

LOGGER = logging.getLogger(__name__)


class HostSharedFilesystemsMixin:
    # Attributes supplied by the composed concrete implementation.
    proc_root: Path

    _directory_probe: Callable[[str], None]
    _mount_tables: Callable[..., Any]
    _probe_mount: Callable[..., Any]
    _sample: Callable[..., Any]
    _shared_mount_memory: SharedMountMemory
    _warn_filesystems_once: Callable[..., None]

    def _shared_filesystems(self, observed_at: datetime) -> list[HostMetricSample]:
        """Every NFS/Lustre mount the host has, and every one it used to have.

        Discovery reads the host's (PID 1's) mount table, as ``_filesystems``
        does for its read-only verdict: the collector's own table is a sandbox
        view that hides ``ProtectHome=`` paths and never receives a mount made
        under a non-propagating parent. A shared mount the sandbox hides is
        warned about once and neither probed nor reported -- the collector's
        syscalls there would read the sandbox's filesystem, not the export.
        Without a host table the collector's own table stands, as it always
        did.

        Each visible mount is probed twice under one deadline and one budget
        (``_probe_mount``): ``statvfs`` for the capacity reading and the
        FSSTAT round trip, then one directory-entry read for the data path
        (``shared_mounts.read_directory_entry``) -- an export whose
        permissions changed server-side or whose backend is stuck answers
        FSSTAT and used to read healthy (GF-GENERAL-LIVE-005
        ``io_only_failure``).

        The node-scoped mounts seen are remembered (``SharedMountMemory``): a
        remembered mount the table no longer lists is MOUNT_MISSING while its
        mount point directory is still there, and forgotten once the
        directory is gone too (``mount_disappears``, and its legitimate
        counterpart, a teardown). Which mounts are remembered, what clears
        them and what survives a restart is the memory module's docstring.
        """

        result: list[HostMetricSample] = []
        own, host = self._mount_tables()
        host_table = self.proc_root / "1" / "mountinfo"
        table = host if host is not None else own
        if table is None:
            return result
        if host is None:
            self._warn_filesystems_once(
                "\0shared-host-mount-table",
                "the host mount table %s cannot be read; shared mounts are "
                "discovered from the collector's own table",
                host_table,
            )
        present: dict[str, tuple[str, str]] = {}
        for entry in shared_mounts(table):
            mount = entry.mount_point
            if not host_mount(mount, own, host).same_filesystem:
                self._warn_filesystems_once(
                    mount,
                    "shared mount %s (%s) is hidden by the collector's sandbox; it "
                    "is not reported",
                    mount,
                    entry.source,
                )
                continue
            if node_scoped(entry, table):
                present[mount] = (entry.source, entry.fs_type)
            result.extend(self._shared_mount_samples(entry, observed_at))
        observation = self._shared_mount_memory.observe(present, observed_at)
        for mount in observation.refused:
            self._warn_filesystems_once(
                f"\0shared-memory-full:{mount}",
                "the shared mount memory is full; a disappearance of %s will not be "
                "reported",
                mount,
            )
        for record in observation.missing:
            result.extend(self._missing_shared_mount(record, own, host, observed_at))
        warning = self._shared_mount_memory.save()
        if warning is not None:
            self._warn_filesystems_once("\0shared-memory-save", "%s", warning)
        return result

    def _shared_mount_samples(
        self, entry: MountEntry, observed_at: datetime
    ) -> list[HostMetricSample]:
        """The availability gauge and the capacity reading of one visible mount."""

        mount = entry.mount_point
        try:
            stat = self._probe_mount(mount, observed_at)
        except ProbeStillRunning as error:
            # The parked call may be last tick's directory read: keep its name.
            return [
                self._shared_filesystem_unavailable(
                    mount,
                    entry.source,
                    entry.fs_type,
                    "IO_TIMEOUT"
                    if error.operation is self._directory_probe
                    else "PROBE_TIMEOUT",
                )
            ]
        except TimeoutError:
            return [
                self._shared_filesystem_unavailable(
                    mount, entry.source, entry.fs_type, "PROBE_TIMEOUT"
                )
            ]
        except OSError as error:
            # ESTALE, EIO, ENOENT ...: a status that cannot be read is
            # unusable (UNKNOWN), never healthy and never skipped.
            return [
                self._shared_filesystem_unavailable(
                    mount, entry.source, entry.fs_type, "PROBE_FAILED", error
                )
            ]
        result = []
        total = stat.f_blocks * stat.f_frsize
        available = stat.f_bavail * stat.f_frsize
        if total:
            result.append(
                self._sample(
                    "shared_filesystem_used_percent",
                    100 * (total - available) / total,
                    "percent",
                    mount,
                )
            )
        io_failure: str | None = None
        io_error: OSError | None = None
        try:
            self._probe_mount(mount, observed_at, self._directory_probe)
        except TimeoutError:
            # No errno and no verdict: the read did not answer within the
            # deadline. Missing evidence reads unavailable, not healthy.
            io_failure = "IO_TIMEOUT"
        except OSError as exc:
            io_failure, io_error = "IO_ERROR", exc
        result.insert(
            0,
            self._shared_filesystem_unavailable(
                mount, entry.source, entry.fs_type, io_failure, io_error
            ),
        )
        return result

    def _missing_shared_mount(
        self,
        record: SharedMountRecord,
        own: tuple[MountEntry, ...] | None,
        host: tuple[MountEntry, ...] | None,
        observed_at: datetime,
    ) -> list[HostMetricSample]:
        """A remembered mount the table no longer lists: gone, or torn down.

        The mount point is read with the bounded directory probe, never a bare
        ``stat`` (the parent may be a wedged mount). ENOENT on the host's own
        directory is a teardown -- ``umount`` then ``rmdir`` -- and the mount
        is forgotten. Anything else, including a read that hangs or an ENOENT
        the sandbox may have produced, leaves the mount MOUNT_MISSING: the
        table says it is gone and nothing proved it was meant to be.
        """

        mount = record.mount_point
        try:
            self._probe_mount(mount, observed_at, self._directory_probe)
        except (FileNotFoundError, NotADirectoryError):
            if host_mount(mount, own, host).same_directory:
                self._shared_mount_memory.forget(mount)
                LOGGER.info(
                    "shared mount %s (%s) was unmounted and its mount point removed; "
                    "no longer tracked",
                    mount,
                    record.source,
                )
                return []
        except OSError:
            pass
        return [
            self._shared_filesystem_unavailable(
                mount, record.source, record.fs_type, "MOUNT_MISSING"
            )
        ]

    @staticmethod
    def _shared_filesystem_unavailable(
        mount: str,
        source: str,
        fs_type: str,
        failure_mode: str | None,
        error: OSError | None = None,
    ) -> HostMetricSample:
        """Availability gauge for one shared mount (GF-GENERAL-LIVE-005).

        Value 0 with no labels for a mount whose ``statvfs`` and directory read
        both answered; value 1 otherwise, naming the export (``source``,
        ``fs_type``) and the ``failure_mode``: ``PROBE_TIMEOUT`` (``statvfs``
        did not answer), ``PROBE_FAILED`` (it raised; ``errno`` named),
        ``IO_TIMEOUT`` (the directory read did not answer: no verdict, reported
        unavailable rather than healthy), ``IO_ERROR`` (it raised; ``errno``
        named) and ``MOUNT_MISSING`` (a remembered mount the host's table no
        longer lists). The labels are new and optional on the wire; the
        consumer's rule reads the value.
        """

        if failure_mode is None:
            return HostMetricSample(
                name="shared_filesystem_unavailable", value=0.0, device=mount
            )
        labels = {"failure_mode": failure_mode, "source": source, "fs_type": fs_type}
        if error is not None and error.errno:
            labels["errno"] = errno.errorcode.get(error.errno, str(error.errno))
        return HostMetricSample(
            name="shared_filesystem_unavailable",
            value=1.0,
            device=mount,
            labels=labels,
        )
