"""The loop/dm test filesystem must reach the installed host collector.

GF-GENERAL-LIVE-003/004/006 create a dedicated loop (``fs``) or device-mapper
(``io``) filesystem under ``/mnt/gfrs-live/<digest>/<name>`` and declare it to
the host collector through ``GPU_FAULT_FILESYSTEMS``. The storage probe runs in
its own Pod mount namespace, so a plain ``mount`` there is invisible to the
collector the window restarts (its mount namespace is cloned from the host root
one) and to the ``systemd-run`` I/O writer (which runs in the host root mount
namespace). The probe therefore propagates its mounts back to the host with
``Bidirectional`` mount propagation. Without it the collector ``statvfs`` the
parent (root) filesystem at the appended path -- which is the live finding these
tests pin, driving the product's real ``_filesystems`` reader with a fake
``statvfs``.
"""

from __future__ import annotations

import os
import types
from datetime import datetime, timezone

import pytest

from gpu_fault.collectors.host.system_metrics import HostSystemMetricsMixin
from scripts.e2e.regional import general_live_storage as storage
from scripts.e2e.regional import general_live_utilization as utilization
from scripts.e2e.regional.general_live_fixture import PROBE_HOST_ROOT_PROPAGATION
from scripts.e2e.regional.probes import collector_window_probe as window
from scripts.e2e.regional.probes import general_live_storage_probe as probe
from tests.regional._general_live_fakes import FakeRegional, make_context
from tests.regional._host_probe_support import host_probe

RUN_ID = "gl-capacity-full-a1-deadbeef"
CTX = types.SimpleNamespace(run_id=RUN_ID)
#: The path the harness appends to GPU_FAULT_FILESYSTEMS and the collector
#: statvfs for 003/004 -- computed through the harness helper under test.
FS_MOUNT = storage._fs_mount(CTX)


def _volume_mount(fixture, name):
    _configmap, pod = fixture.manifests()
    container = pod["spec"]["containers"][0]
    matches = [mount for mount in container["volumeMounts"] if mount["name"] == name]
    assert len(matches) == 1, container["volumeMounts"]
    return matches[0], container


# --------------------------------------------------------------------------- #
# The mount-propagation decision on the probe Pod manifest
# --------------------------------------------------------------------------- #
def test_bidirectional_is_set_on_the_privileged_host_root_mount(tmp_path):
    fixture = host_probe(tmp_path, host_root_propagation="Bidirectional")
    host_root, container = _volume_mount(fixture, "host-root")
    assert host_root["mountPropagation"] == "Bidirectional"
    # Bidirectional propagation is only honoured from a privileged container.
    assert container["securityContext"]["privileged"] is True
    # The read-only script mount is never turned into a propagating one.
    script, _container = _volume_mount(fixture, "probe-script")
    assert "mountPropagation" not in script


def test_default_host_root_mount_is_unchanged(tmp_path):
    # The default keeps the manifest byte-identical to before the fix, so every
    # read-only probe's Pod is untouched and its mounts stay Pod-private.
    host_root, _container = _volume_mount(host_probe(tmp_path), "host-root")
    assert host_root == {"name": "host-root", "mountPath": "/host"}


def test_host_to_container_mode_is_available_but_not_the_default(tmp_path):
    host_root, _container = _volume_mount(
        host_probe(tmp_path, host_root_propagation="HostToContainer"), "host-root"
    )
    assert host_root["mountPropagation"] == "HostToContainer"


def test_unknown_propagation_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="propagation"):
        host_probe(tmp_path, host_root_propagation="rshared")


# --------------------------------------------------------------------------- #
# Only the mounting (storage) probe asks for it; every other probe stays private
# --------------------------------------------------------------------------- #
def test_only_the_storage_probe_pod_propagates_to_the_host(tmp_path):
    spec = next(item for item in storage.VARIANTS if item.variant == "capacity_full")
    ctx = make_context(spec, tmp_path=tmp_path, regional=FakeRegional(), probes={})
    assert (
        ctx.probe("storage").fixture.settings.host_root_propagation == "Bidirectional"
    )
    for name in ("window", "node", "net", "dcgm"):
        assert ctx.probe(name).fixture.settings.host_root_propagation == "None"
    # The mounting probe (and only it) is the one named in the mapping.
    assert PROBE_HOST_ROOT_PROPAGATION == {"storage": "Bidirectional"}
    host_root, _container = _volume_mount(ctx.probe("storage").fixture, "host-root")
    assert host_root["mountPropagation"] == "Bidirectional"


# --------------------------------------------------------------------------- #
# The path the collector would statvfs, and why visibility decides the reading
# --------------------------------------------------------------------------- #
def _statvfs(*, total_bytes, avail_bytes, files, ffree):
    frsize = 4096
    blocks = total_bytes // frsize
    bavail = avail_bytes // frsize
    return os.statvfs_result(
        (frsize, frsize, blocks, bavail, bavail, files, ffree, ffree, 0, 255)
    )


#: ~6.1 % used, ~0.1 % inodes: the root filesystem reading the live run saw when
#: the loop mount never reached the collector's namespace.
ROOT_FS = _statvfs(
    total_bytes=100 * 1024**3,
    avail_bytes=94 * 1024**3,
    files=6_553_600,
    ffree=6_547_000,
)
#: The loop filesystem filled to 99 % with its inode table nearly exhausted.
FILLED_LOOP_FS = _statvfs(
    total_bytes=2 * 1024**3, avail_bytes=20 * 1024**2, files=65_536, ffree=100
)


class _FilesystemReader(HostSystemMetricsMixin):
    """Drives the product's real ``_filesystems`` with one scripted ``statvfs``."""

    statvfs_budget_seconds = 15.0
    statvfs_timeout_seconds = 5.0

    def __init__(self, mount: str, reading: os.statvfs_result) -> None:
        self.filesystems = [mount]
        self._statvfs_tick = None
        self._statvfs_spent_seconds = 0.0
        self._unresponsive_mounts: set[str] = set()
        self._reading = reading

    def _statvfs_probe(self, mount: str) -> os.statvfs_result:
        return self._reading

    def _sample(self, name, value, unit=None, device=None):
        return {"name": name, "value": value, "device": device}


def _read(mount: str, reading: os.statvfs_result) -> dict[tuple[str, str], float]:
    reader = _FilesystemReader(mount, reading)
    now = datetime(2026, 10, 24, tzinfo=timezone.utc)
    return {
        (sample["name"], sample["device"]): sample["value"]
        for sample in reader._filesystems(now)
    }


def test_appended_path_is_exactly_what_the_probe_mounts_and_the_window_allows():
    # The 004 byte/inode mount, the 003 scratch mount, and the probe's real
    # mountpoint are the same string -- the path the collector will statvfs.
    assert FS_MOUNT == str(probe.mountpoint(RUN_ID, storage.TEST_FS))
    assert FS_MOUNT == utilization._fs_mount(CTX)
    # The window may only extend GPU_FAULT_FILESYSTEMS with exactly this shape.
    allowed = window.APPENDABLE["GPU_FAULT_FILESYSTEMS"][0]
    assert allowed.fullmatch(FS_MOUNT), (
        f"{FS_MOUNT} is not an allow-listed GPU_FAULT_FILESYSTEMS window entry"
    )
    # 006 shares the digest directory with its own device mount name.
    assert storage._dm_mount(CTX) == str(probe.mountpoint(RUN_ID, storage.TEST_DM))


def test_collector_grades_the_fault_only_when_the_mount_is_visible():
    # Visible: the fixture propagated the loop fs to the host root namespace, so
    # the collector's statvfs reads the filled filesystem and both the capacity
    # and the inode signal cross the 98 % quarantine rule.
    visible = _read(FS_MOUNT, FILLED_LOOP_FS)
    assert visible[("filesystem_used_percent", FS_MOUNT)] >= 98
    assert visible[("filesystem_inodes_used_percent", FS_MOUNT)] >= 98

    # Hidden (the live finding): the mount stayed inside the probe Pod, the
    # directory still exists on the root fs so statvfs does not raise ENOENT, and
    # the collector reads the root filesystem -- far below the rule, so 003/004
    # never fire and time out.
    hidden = _read(FS_MOUNT, ROOT_FS)
    assert hidden[("filesystem_used_percent", FS_MOUNT)] < 10
    assert hidden[("filesystem_inodes_used_percent", FS_MOUNT)] < 1
