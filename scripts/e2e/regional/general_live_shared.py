"""GF-GENERAL-LIVE-005 shared storage unavailable / full.

The host collector grades shared storage through two paths
(``collectors/host/system_metrics.py``): ``_shared_filesystems`` auto-discovers
``/proc/self/mountinfo`` entries of type ``nfs``/``nfs4``/``lustre`` and emits
``shared_filesystem_unavailable`` (CRITICAL RUN_DIAGNOSTICS on a ``statvfs``
timeout/error) and ``shared_filesystem_used_percent``; ``_filesystems`` statvfs
each ``GPU_FAULT_FILESYSTEMS`` entry. A local loop file system is
GF-GENERAL-LIVE-004, not shared storage: it is never an ``nfs``/``lustre``
fstype, so it cannot exercise the shared path.

Decision for this site: neither the plain-EKS GPU node nor the HyperPod nodes
mount an independent NFS/Lustre export that a test may fill, hang or disconnect
without touching a production dependency, and standing up a test NFS/Lustre
server (nfsd, a server process, cross-node export) is out of this phase's
scope and approvals. Every variant therefore stays in the denominator as
NOT_RUN with its reason: CAPABILITY_ABSENT where the site has no shared export,
FIXTURE_NOT_READY where the behaviour is a product gap (statvfs returns but
read/fsync fails) that needs a bounded shared-I/O probe the product does not
have. The spec records these as known gaps ("真实有界 I/O 探针与共享存储故障域聚合").
"""

from __future__ import annotations

from scripts.e2e.regional.general_live_registry import (
    CAPABILITY,
    LIVE_REPLAY,
    deferred_variant,
)

CASE = "GF-GENERAL-LIVE-005"
NO_EXPORT = (
    "CAPABILITY_ABSENT: this site mounts no independent NFS/Lustre export a test "
    "may disturb, and no test shared-storage server is stood up in this phase"
)
NEEDS_IO_PROBE = (
    "FIXTURE_NOT_READY: statvfs-succeeds-but-I/O-fails is a product gap needing a "
    "bounded shared read/fsync probe and a test export; neither exists this phase"
)

VARIANTS = [
    deferred_variant(
        CASE,
        "stat_timeout",
        proof_mode=LIVE_REPLAY,
        reason=NEEDS_IO_PROBE,
        lanes=("eks", "hp"),
        expected="a hung statvfs -> shared_filesystem_unavailable=1 with other metrics still delivered",
    ),
    deferred_variant(
        CASE,
        "capacity_full",
        proof_mode=CAPABILITY,
        reason=NO_EXPORT,
        lanes=("eks", "hp"),
        expected="shared_filesystem_used_percent >= 98 on an nfs/lustre mount",
    ),
    deferred_variant(
        CASE,
        "mount_disappears",
        proof_mode=CAPABILITY,
        reason=NO_EXPORT,
        lanes=("eks", "hp"),
        expected="the vanished shared mount is reported unavailable, not healthy",
    ),
    deferred_variant(
        CASE,
        "io_only_failure",
        proof_mode=LIVE_REPLAY,
        reason=NEEDS_IO_PROBE,
        lanes=("eks", "hp"),
        expected="statvfs returns but read/fsync fails -> the product currently misses it (spec FAIL)",
    ),
    deferred_variant(
        CASE,
        "healthy_other_mount",
        proof_mode=CAPABILITY,
        reason=NO_EXPORT,
        lanes=("eks", "hp"),
        expected="one failing export does not let a second healthy mount mask it",
    ),
    deferred_variant(
        CASE,
        "shared_failure_domain",
        proof_mode=CAPABILITY,
        reason=(
            "CAPABILITY_ABSENT: a shared failure domain needs two task nodes on one "
            "export; this site has no shared export for either lane to share"
        ),
        lanes=("hp",),
        expected="two nodes on one export record a common failure domain, not two bad disks",
    ),
    deferred_variant(
        CASE,
        "recovery",
        proof_mode=CAPABILITY,
        reason=NO_EXPORT,
        lanes=("eks", "hp"),
        expected="a restored export proves fresh samples, real I/O and task progress",
    ),
]
