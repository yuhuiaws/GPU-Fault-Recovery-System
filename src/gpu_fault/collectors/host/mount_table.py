"""Which mount holds a configured path, as the host sees it.

The host collector runs in the private mount namespace systemd builds for
``deploy/systemd/gpu-fault-host-collector.service``: ``ProtectSystem=strict``
re-binds every mount read-only, ``ProtectHome=true`` lays an empty,
inaccessible directory over ``/home``, ``/root`` and ``/run/user``, and
``PrivateTmp=true`` swaps ``/tmp`` and ``/var/tmp`` for private directories.
Inside it ``statvfs`` reports ``ST_RDONLY`` for every healthy ``/`` and
``/var``, and ``ENOENT`` for a path under a hidden directory -- facts about the
collector's sandbox, not about the node. Grading ``ST_RDONLY`` from there made
every GPU node report ``/`` and ``/var`` READ_ONLY the moment its collector
restarted (batch 163).

PID 1 lives in the host mount namespace, and ``/proc/1/mountinfo`` is the
host's own table -- the unit runs as root with no ``ProtectProc=``, and the file
is world-readable besides. Its per-mount options (``ro`` for a read-only bind)
and its superblock options (``ro`` after a ``remount,ro``, or once
``errors=remount-ro`` tripped on an I/O error) are what the node's
workloads are subject to, so whether a configured filesystem is read-only is
taken from there, never from the collector's own bind view. Comparing the
mount that holds a path in each table also says whether the collector's own
``statvfs`` of that path is about the host's filesystem at all.

A configured path is matched against the mount points as written, without
resolving it: resolving would ``lstat`` every component, which blocks on a
hung NFS or Lustre mount exactly like ``statvfs`` does, outside its bounded
probe. A configured path that runs through a symlink is therefore judged by
the mount holding the link, not by the one holding its target: configure the
mount point itself.

Standard library only: this module is part of the node runtime.
"""

from __future__ import annotations

import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

#: ``show_mountinfo`` escapes space, tab, newline and backslash in paths as
#: three-digit octal (``\040`` ...), so a field never contains whitespace.
_OCTAL_ESCAPE = re.compile(r"\\([0-7]{3})")


@dataclass(frozen=True)
class MountEntry:
    """One ``mountinfo`` line (``proc_pid_mountinfo(5)``)."""

    mount_id: str
    parent_id: str
    #: ``major:minor`` of the filesystem (``st_dev``): equal for every bind
    #: view of one filesystem, in every mount namespace.
    device: str
    #: The directory of that filesystem mounted here: ``/`` for a plain mount,
    #: the bound directory for a bind mount.
    root: str
    mount_point: str
    fs_type: str
    mount_options: tuple[str, ...]
    super_options: tuple[str, ...]

    @property
    def read_only(self) -> bool:
        """``ro`` for this mount (a read-only bind) or its superblock.

        The superblock is shared by every namespace: a ``mount -o remount,ro``
        made in a Pod's namespace, or ``errors=remount-ro`` after an I/O
        error, shows as ``ro`` here while the per-mount options stay ``rw``.
        """

        return "ro" in self.mount_options or "ro" in self.super_options


class HostMount(NamedTuple):
    """The host mount holding a path, and whether the collector sees it."""

    #: None when the host table could not be read: no read-only verdict.
    entry: MountEntry | None
    #: The collector's own ``statvfs`` of the path reads this filesystem.
    same_filesystem: bool
    #: ... and the host's directory at the path, so ``ENOENT`` means the path
    #: is missing on the host, not hidden by the collector's sandbox.
    same_directory: bool


def _unescape(value: str) -> str:
    return _OCTAL_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def parse_mountinfo(text: str) -> tuple[MountEntry, ...]:
    """Every well-formed line of a ``mountinfo`` file; malformed lines are skipped."""

    entries = []
    for line in text.splitlines():
        fields = line.split()
        # Fields 1-6 are fixed, then zero or more optional fields
        # (``shared:1``, ``master:2`` ...) and the ``-`` separator, then the
        # filesystem type, the source and the superblock options.
        try:
            separator = fields.index("-", 6)
        except ValueError:
            continue
        if len(fields) < separator + 4:
            continue
        entries.append(
            MountEntry(
                mount_id=fields[0],
                parent_id=fields[1],
                device=fields[2],
                root=_unescape(fields[3]),
                mount_point=_unescape(fields[4]),
                fs_type=fields[separator + 1],
                mount_options=tuple(fields[5].split(",")),
                super_options=tuple(fields[separator + 3].split(",")),
            )
        )
    return tuple(entries)


def read_mount_table(path: Path) -> tuple[MountEntry, ...] | None:
    """The table at ``path``, or None when it cannot be read or holds no mount."""

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return parse_mountinfo(text) or None


def covering_mount(table: tuple[MountEntry, ...], path: str) -> MountEntry | None:
    """The mount a lookup of absolute ``path`` ends on: deepest mount point wins.

    Several mounts stacked on one mount point are a chain of parents; the
    visible one is the mount no other one at that point is stacked on.
    """

    best: list[MountEntry] = []
    for entry in table:
        point = entry.mount_point
        if not (
            point == "/" or path == point or path.startswith(point.rstrip("/") + "/")
        ):
            continue
        if best and len(point) < len(best[0].mount_point):
            continue
        if best and len(point) > len(best[0].mount_point):
            best = []
        best.append(entry)
    if not best:
        return None
    parents = {entry.parent_id for entry in best}
    visible = [entry for entry in best if entry.mount_id not in parents]
    return (visible or best)[-1]


def _location(entry: MountEntry, path: str) -> tuple[str, str]:
    """The filesystem and the directory in it that ``path`` names via ``entry``."""

    relative = path[len(entry.mount_point.rstrip("/")) :].lstrip("/")
    return entry.device, posixpath.normpath(posixpath.join(entry.root, relative))


def host_mount(
    path: str,
    own: tuple[MountEntry, ...] | None,
    host: tuple[MountEntry, ...] | None,
) -> HostMount:
    """The host mount holding ``path``, judged against the collector's own table.

    ``own`` is the collector's ``/proc/self/mountinfo``, ``host`` PID 1's. The
    collector's ``statvfs`` reads the host's filesystem when the mount holding
    the path in its own table is a view of the same device (a
    ``ProtectSystem=`` read-only bind, a ``PrivateTmp=`` directory), and the
    host's directory when it also binds the same directory of it. Without both
    tables nothing can be compared, and the collector's own reading stands as
    it always did.
    """

    target = os.path.abspath(path)
    entry = covering_mount(host, target) if host else None
    mine = covering_mount(own, target) if own else None
    if entry is None or mine is None:
        return HostMount(entry, True, True)
    device, directory = _location(entry, target)
    own_device, own_directory = _location(mine, target)
    same_filesystem = device == own_device
    return HostMount(
        entry, same_filesystem, same_filesystem and directory == own_directory
    )
