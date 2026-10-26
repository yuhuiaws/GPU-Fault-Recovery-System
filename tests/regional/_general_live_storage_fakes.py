"""Scripted node, storage and window probes of the GF-GENERAL-LIVE-004/006 tests.

The storage probe's answers (loop file system, dm device, retarget), the
collector window and the node survey one 004/006 variant needs, shared by
``test_general_live_storage.py`` (004) and ``test_general_live_storage_io.py``
(006).
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import general_live_storage as storage
from tests.regional._general_live_fakes import FakeProbe

SPECS = {spec.variant: spec for spec in storage.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"


def survey(log_collector: str = "inactive", **modules: bool) -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {
            name: True
            for name in (
                "losetup",
                "mkfs.ext4",
                "mkfs.vfat",
                "mount",
                "fallocate",
                "dmsetup",
                "blockdev",
                "systemd-run",
                "nsenter",
                "modprobe",
            )
        },
        "modules": {
            "dm_mod": {"loaded": True, "available": True},
            "dm_delay": {"loaded": modules.get("dm_delay", True), "available": True},
            "vfat": {"loaded": modules.get("vfat", True), "available": True},
        },
        "units": {storage.LOG_COLLECTOR_UNIT: {"ActiveState": log_collector}},
        "backing": [
            {
                "root": "/var/lib/gpu-fault-acceptance",
                "mount": {"fstype": "ext4"},
                "statvfs": {"available_bytes": 400 * 1024**3, "used_percent": 20.0},
            }
        ],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def retarget_answer(arguments: tuple[str, ...]) -> dict[str, Any]:
    """``dm-retarget``: EIO on the error table, a healthy reading on the data one."""

    error = "error" in arguments
    return {
        "name": "err",
        "target": "error" if error else "data",
        "attempts": 1,
        "mounted_on_host": True,
        "faulted": error,
        "fault_dm": "gfrs-deadbeef0000-err-fault",
        "direct_reads": {"boot": "ok", "fat": "OSError: 5"},
        "diagnosis": None,
        "statvfs": {"error": "OSError: 5"} if error else {"used_percent": 1.0},
    }


def storage_probe(**overrides: Any) -> FakeProbe:
    script = {
        "fs-create": {
            "name": "fs",
            "mountpoint": "/mnt",
            "statvfs": {"used_percent": 20.0},
        },
        "fs-fill": {"statvfs": {"used_percent": 99.0}},
        "fs-unfill": {"statvfs": {"used_percent": 20.0}},
        "fs-inodes": {"enospc": True, "created": 256},
        "fs-remount": {"statvfs": {"read_only": True, "used_percent": 20.0}},
        "fs-umount": {"directory_present": False, "statvfs": {"used_percent": None}},
        "fs-mount": {"mounted": "/mnt"},
        "fs-destroy": {"destroyed": True},
        # dm-create only starts the setup unit; dm-status reports the device.
        "dm-create": {"name": "io", "dm": "gfrs-x-io", "phase": "setup"},
        "dm-status": {
            "name": "io",
            "phase": "ready",
            "dm": "gfrs-x-io",
            "device": "dm-7",
            "mountpoint": "/mnt",
        },
        "cleanup": {"clean": True, "residue": {}},
        "dm-suspend": {"suspended": "gfrs-x-io"},
        "dm-resume": {"resumed": "gfrs-x-io"},
        "dm-recreate": {"device_before": "dm-7", "device_after": "dm-8"},
        "dm-retarget": retarget_answer,
        "dm-destroy": {"destroyed": True},
        "io-start": {"unit": "gfrs-x-io-io"},
        "io-stop": {"stopped": "gfrs-x-io-io"},
    }
    script.update(overrides)
    return FakeProbe(script)


def window_probe() -> FakeProbe:
    return FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}, "appended": {}},
            "close-window": {"after": {"ActiveState": "active"}},
            "start-unit": {"unit": "gpu-fault-host-collector.service"},
        }
    )


def node_probe(survey_value: dict[str, Any], **extra: Any) -> FakeProbe:
    script = {
        "survey": survey_value,
        "oracle-start": {"unit": "oracle"},
        "oracle-stop": {},
        "oracle-read": {"records": [], "next_offset": 0, "total": 0},
    }
    script.update(extra)
    return FakeProbe(script)


def probes(
    ctx_node: FakeProbe, store: FakeProbe, win: FakeProbe
) -> dict[str, FakeProbe]:
    return {f"node@{NODE}": ctx_node, f"storage@{NODE}": store, f"window@{NODE}": win}
