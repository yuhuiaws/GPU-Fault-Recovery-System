"""A configured filesystem is judged as the host sees it, not as the sandbox does.

Batch 163 read ``statvfs().f_flag & ST_RDONLY`` from inside the host
collector's systemd sandbox (``ProtectSystem=strict``, ``ProtectHome=true``,
``PrivateTmp=true``, ``ReadWritePaths=/var/lib/gpu-fault``), where every mount
is a read-only bind view: right after its collectors restarted, every GPU
node reported ``/`` and ``/var`` as ``filesystem_unavailable`` READ_ONLY, the
product opened "configured filesystem is not usable" incidents on all of them,
and VALIDATE_HOST -- which re-reads the gauge -- could no longer pass anywhere.

The tables below are the lines a transient unit with exactly that hardening
showed in its own ``/proc/self/mountinfo`` and in ``/proc/1/mountinfo``
(captured on a systemd 255 host, trimmed to the mounts these cases use), and
the scripted ``statvfs`` flags are the ones the same sandbox returned. Each
case drives the real ``_filesystems`` through ``collect_once``, grades the batch
with the real ``NodeHealthPolicy`` and re-reads the persisted latest values
with ``validate_host_metrics`` as VALIDATE_HOST does.
"""

from __future__ import annotations

import errno
import logging
import os
from datetime import timedelta

import pytest

from gpu_fault.collectors.host.mount_table import (
    covering_mount,
    host_mount,
    parse_mountinfo,
    read_mount_table,
)
from gpu_fault.host_diagnostics import validate_host_metrics
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch, NodeHealthPolicy
from gpu_fault.models import RecoveryAction
from tests._builders import build_store
from tests.collectors import _cov95_runtime_collect as common
from tests.collectors import _cov95_runtime_collect_host as support

isolated_runtime = common.isolated_runtime
host_case = support.host_case

ROOT_SUPER = "rw,discard,errors=remount-ro,commit=30"
RUN_TMPFS = "rw,size=51704252k,nr_inodes=819200,mode=755,inode64"
PRIVATE = "51dda8ef6251465b93e0e16ec1d9ee46-gpu-fault-host-collector.service"
#: The collector's own table: ``ProtectSystem=strict`` re-bound ``/`` read-only
#: (``NoNewPrivileges=`` adds ``nosuid``) while its superblock stays ``rw``;
#: ``ProtectHome=`` laid systemd's inaccessible directory (on the ``/run``
#: tmpfs) over ``/home``, ``/root`` and ``/run/user``; ``PrivateTmp=`` bound a
#: private directory of the root filesystem over ``/tmp`` and ``/var/tmp``; and
#: ``ReadWritePaths=`` re-bound ``/var/lib/gpu-fault`` onto itself read-write.
SANDBOX = [
    f"2177 76 259:1 / / ro,nosuid,relatime shared:496 master:1 - ext4 /dev/root {ROOT_SUPER}",
    "2183 2177 0:25 / /proc rw,nosuid,nodev,noexec,relatime shared:508 master:5 "
    "- proc proc rw",
    f"2210 2177 0:29 / /run ro,nosuid,nodev shared:522 master:12 - tmpfs tmpfs {RUN_TMPFS}",
    "2216 2210 0:29 /systemd/inaccessible/dir /run/user ro,nosuid,nodev,noexec "
    f"shared:542 master:12 - tmpfs tmpfs {RUN_TMPFS}",
    "2249 2177 0:29 /systemd/inaccessible/dir /home ro,nosuid,nodev,noexec "
    f"shared:585 master:12 - tmpfs tmpfs {RUN_TMPFS}",
    "2250 2177 0:29 /systemd/inaccessible/dir /root ro,nosuid,nodev,noexec "
    f"shared:586 master:12 - tmpfs tmpfs {RUN_TMPFS}",
    f"2217 2177 259:1 /tmp/systemd-private-{PRIVATE}-7a5KHD/tmp /tmp "
    f"rw,nosuid,relatime shared:587 master:1 - ext4 /dev/root {ROOT_SUPER}",
    f"2253 2177 259:1 /var/tmp/systemd-private-{PRIVATE}-MtOohi/tmp /var/tmp "
    f"rw,nosuid,relatime shared:588 master:1 - ext4 /dev/root {ROOT_SUPER}",
    "2254 2177 259:1 /var/lib/gpu-fault /var/lib/gpu-fault rw,nosuid,relatime "
    f"shared:589 master:1 - ext4 /dev/root {ROOT_SUPER}",
]
#: PID 1's table: the same filesystems, read-write, as the node's workloads see
#: them, plus a home directory on NFS the sandbox never shows.
HOST = [
    f"28 1 259:1 / / rw,relatime shared:1 - ext4 /dev/root {ROOT_SUPER}",
    "30 28 0:25 / /proc rw,nosuid,nodev,noexec,relatime shared:5 - proc proc rw",
    f"35 28 0:29 / /run rw,nosuid,nodev shared:12 - tmpfs tmpfs {RUN_TMPFS}",
    "61 28 0:52 / /home rw,relatime shared:40 - nfs4 10.0.0.2:/home "
    "rw,vers=4.1,rsize=1048576,wsize=1048576,hard,proto=tcp",
]
#: The GF-GENERAL-LIVE-004 loop filesystem, mounted (and propagated to the
#: host) before the collector window restarted the unit, so the sandbox holds
#: its own read-only bind of it.
LOOP = "/mnt/gfrs-live/0123456789ab/fs"


def loop_lines(*, host_mount_options: str, superblock: str) -> tuple[str, str]:
    return (
        f"2260 2177 7:15 / {LOOP} ro,nosuid,relatime shared:600 master:590 "
        f"- ext4 /dev/loop15 {superblock}",
        f"590 28 7:15 / {LOOP} {host_mount_options} shared:590 "
        f"- ext4 /dev/loop15 {superblock}",
    )


def write_tables(case, *, own: list[str] | None, host: list[str] | None) -> None:
    for name, lines in (("self/mountinfo", own), ("1/mountinfo", host)):
        target = case.roots.proc / name
        if lines is None:
            target.unlink(missing_ok=True)
        else:
            support.write_file(target, "\n".join(lines) + "\n")


def host_baseline(case) -> None:
    """The rest of VALIDATE_HOST's required baseline, read healthy."""

    support.write_file(case.roots.proc / "stat", "cpu 100 0 0 800 0 0 0 0\n")
    support.write_file(case.roots.proc / "loadavg", "0.1 0.1 0.1 1/10 123\n")
    support.write_file(
        case.roots.proc / "meminfo",
        "MemTotal: 1000 kB\nMemAvailable: 800 kB\nMemFree: 700 kB\n",
    )


#: ``f_flag`` as the sandbox's own ``statvfs`` returns it.
SANDBOX_READ_ONLY_BIND = os.ST_RDONLY | os.ST_NOSUID | os.ST_RELATIME
SANDBOX_WRITABLE_BIND = os.ST_NOSUID | os.ST_RELATIME


def sandbox_statvfs(flags: dict[str, int | OSError], probed: list[str] | None = None):
    def probe(path: str) -> os.statvfs_result:
        if probed is not None:
            probed.append(path)
        value = flags[path]
        if isinstance(value, OSError):
            raise value
        return os.statvfs_result(
            (4096, 4096, 1000, 900, 900, 1000, 900, 900, value, 255)
        )

    return probe


def gauges(batch) -> dict[str, tuple[float, dict[str, str]]]:
    return {
        sample.device: (sample.value, sample.labels)
        for sample in batch.samples
        if sample.name == "filesystem_unavailable"
    }


def validate_host(store, batch) -> tuple[object, dict[str, list[str]]]:
    latest = store.list_telemetry_metrics_latest(batch.cluster_id, batch.node_id)
    failures: dict[str, list[str]] = {}
    return validate_host_metrics(latest, batch.node_id, failures), failures


HEALTHY_FLAGS: dict[str, int | OSError] = {
    "/": SANDBOX_READ_ONLY_BIND,
    "/var": SANDBOX_READ_ONLY_BIND,
    "/tmp": SANDBOX_WRITABLE_BIND,
    "/var/lib/gpu-fault": SANDBOX_WRITABLE_BIND,
}


def test_a_healthy_node_is_usable_through_the_collector_sandbox(host_case):
    """The R13c regression: the sandbox's read-only binds are not a read-only node.

    ``/`` and ``/var`` answer ``ST_RDONLY`` inside the sandbox while the host
    mounts them read-write; ``/tmp`` is the sandbox's private, writable
    directory. All four configured mounts are usable, nothing is graded, and
    VALIDATE_HOST passes -- the readmission the false reading blocked.
    """

    host_baseline(host_case)
    write_tables(host_case, own=SANDBOX, host=HOST)
    collector = host_case.build(
        "_cpu",
        "_memory",
        "_filesystems",
        filesystems=list(HEALTHY_FLAGS),
        statvfs=sandbox_statvfs(HEALTHY_FLAGS),
    )

    batch = collector.collect_once()

    assert gauges(batch) == {mount: (0.0, {}) for mount in HEALTHY_FLAGS}, (
        "the collector's own read-only bind views were reported as read-only "
        f"filesystems: {gauges(batch)}"
    )
    values = support.values(batch)
    assert values[("filesystem_used_percent", "/")] == 10, values
    store = build_store()
    assert NodeHealthPolicy(store).evaluate_metrics(batch) == [], (
        "a healthy node opened a storage finding"
    )
    outcome, failures = validate_host(store, batch)
    assert (outcome, failures) == (None, {}), (
        f"VALIDATE_HOST must pass on a healthy node: {outcome} {failures}"
    )


def test_the_first_fixed_batch_clears_the_r13c_false_reading(host_case):
    """The latest READ_ONLY an R13c collector stored is overwritten, not kept.

    The live nodes hold ``filesystem_unavailable=1`` for ``/`` and ``/var`` from
    the faulty collector, so every VALIDATE_HOST fails. The fixed collector's
    first (baseline) batch reports both usable for the same device, and the
    same validation then passes.
    """

    host_baseline(host_case)
    write_tables(host_case, own=SANDBOX, host=HOST)
    store = build_store()
    policy = NodeHealthPolicy(store)
    r13c = HostTelemetryBatch(
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=host_case.clock.now() - timedelta(minutes=5),
        samples=[
            HostMetricSample(name="load1_per_cpu", value=0.05),
            HostMetricSample(name="memory_used_percent", value=20.0),
            HostMetricSample(name="filesystem_used_percent", value=10.0, device="/"),
            *(
                HostMetricSample(
                    name="filesystem_unavailable",
                    value=1.0,
                    device=mount,
                    labels={"failure_mode": "READ_ONLY"},
                )
                for mount in ("/", "/var")
            ),
        ],
    )
    policy.evaluate_metrics(r13c)
    outcome, failures = validate_host(store, r13c)
    assert failures == {"node-a": ["filesystem_unavailable"]}, (
        f"the stored R13c reading should fail VALIDATE_HOST: {outcome} {failures}"
    )
    collector = host_case.build(
        "_cpu",
        "_memory",
        "_filesystems",
        filesystems=["/", "/var", "/tmp"],
        edge_filter_enabled=True,
        statvfs=sandbox_statvfs(HEALTHY_FLAGS),
    )

    batch = collector.collect_once()

    assert "baseline" in batch.edge_filter_reasons, (
        f"the restarted collector's first batch must be delivered: "
        f"{batch.edge_filter_reasons}"
    )
    assert policy.evaluate_metrics(batch) == [], "the fixed batch graded a fault"
    outcome, failures = validate_host(store, batch)
    assert (outcome, failures) == (None, {}), (
        f"the fixed reading must clear the stored false one: {outcome} {failures}"
    )


@pytest.mark.parametrize(
    ("host_mount_options", "superblock"),
    [
        # 004 read_only: ``mount -o remount,ro`` in the storage probe Pod's
        # namespace flips the shared superblock; the host's own per-mount
        # options stay ``rw``.
        ("rw,relatime", "ro"),
        # A read-only bind of the mount on the host itself.
        ("ro,relatime", "rw"),
        # A superblock remounted read-only on the host (``errors=remount-ro``).
        ("ro,relatime", "ro"),
    ],
    ids=["probe-pod-remount", "host-read-only-bind", "host-remount"],
)
def test_a_mount_the_host_shows_read_only_is_unusable_until_restored(
    host_case, host_mount_options, superblock
):
    """Read-only per mount or per superblock on the host is READ_ONLY (004).

    The collector's own bind of the test mount is read-only before, during
    and after the injection, so it cannot be what decides: the gauge follows
    the host's table, holds VALIDATE_HOST while the mount is read-only and
    clears once it is read-write again.
    """

    host_baseline(host_case)
    own, host = loop_lines(host_mount_options=host_mount_options, superblock=superblock)
    write_tables(host_case, own=[*SANDBOX, own], host=[*HOST, host])
    flags = {**HEALTHY_FLAGS, LOOP: SANDBOX_READ_ONLY_BIND}
    collector = host_case.build(
        "_cpu",
        "_memory",
        "_filesystems",
        filesystems=["/", "/var", LOOP],
        statvfs=sandbox_statvfs(flags),
    )
    store = build_store()
    policy = NodeHealthPolicy(store)

    batch = collector.collect_once()

    assert gauges(batch) == {
        "/": (0.0, {}),
        "/var": (0.0, {}),
        LOOP: (1.0, {"failure_mode": "READ_ONLY"}),
    }, gauges(batch)
    findings = policy.evaluate_metrics(batch)
    assert [(item.metric_name, item.device) for item in findings] == [
        ("filesystem_unavailable", LOOP)
    ], findings
    assert findings[0].recommended_action is RecoveryAction.RUN_DIAGNOSTICS, findings
    outcome, failures = validate_host(store, batch)
    assert failures == {"node-a": ["filesystem_unavailable"]}, (
        f"VALIDATE_HOST must hold the node while the mount is read-only: "
        f"{outcome} {failures}"
    )

    own, host = loop_lines(host_mount_options="rw,relatime", superblock="rw")
    write_tables(host_case, own=[*SANDBOX, own], host=[*HOST, host])
    host_case.clock.seconds += 15
    restored = collector.collect_once()

    assert gauges(restored)[LOOP] == (0.0, {}), gauges(restored)
    assert policy.evaluate_metrics(restored) == [], "the restored mount was graded"
    outcome, failures = validate_host(store, restored)
    assert (outcome, failures) == (None, {}), (
        f"the restored mount must let VALIDATE_HOST pass: {outcome} {failures}"
    )


@pytest.mark.parametrize("host_table", ["absent", "unreadable", "empty"])
def test_an_unreadable_host_mount_table_is_no_evidence_of_read_only(
    host_case, caplog, host_table
):
    """Without PID 1's table the sandbox's ``ST_RDONLY`` is never READ_ONLY.

    A view the collector cannot read is not evidence of a read-only filesystem:
    the mounts keep their capacity readings and the availability gauge stays
    usable, and the condition is logged once, not every tick.
    """

    write_tables(host_case, own=SANDBOX, host=None)
    target = host_case.roots.proc / "1/mountinfo"
    if host_table == "unreadable":
        target.mkdir(parents=True)
    elif host_table == "empty":
        support.write_file(target, "")
    collector = host_case.build(
        "_filesystems",
        filesystems=["/", "/var"],
        statvfs=sandbox_statvfs(HEALTHY_FLAGS),
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.host"):
        batch = collector.collect_once()
        host_case.clock.seconds += 15
        again = collector.collect_once()

    for item in (batch, again):
        assert gauges(item) == {"/": (0.0, {}), "/var": (0.0, {})}, gauges(item)
        assert support.values(item)[("filesystem_used_percent", "/")] == 10, (
            support.values(item)
        )
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "host mount table" in record.getMessage()
    ]
    assert len(warnings) == 1, warnings


def test_a_path_the_collector_sandbox_hides_is_not_reported(host_case, caplog):
    """A hidden path reads neither usable, read-only nor missing (UNKNOWN).

    ``/home`` (here an NFS home on the host) and ``/root`` are systemd's
    inaccessible directory inside the sandbox, so ``statvfs`` would read the
    ``/run`` tmpfs and ``ENOENT`` below it; ``/tmp/scratch`` is missing only
    from the collector's private ``/tmp``; and a mount the host has but the
    sandbox never received is the same blind spot. None of them is probed or
    reported. ``/data`` is visible to the sandbox and really missing, and stays
    MOUNT_MISSING exactly as batch 163 reports it.
    """

    # The loop mount reached the host but never the collector's namespace.
    _, host_loop = loop_lines(host_mount_options="rw,relatime", superblock="rw")
    write_tables(host_case, own=SANDBOX, host=[*HOST, host_loop])
    missing = FileNotFoundError(errno.ENOENT, "No such file or directory")
    probed: list[str] = []
    flags: dict[str, int | OSError] = {
        "/home": SANDBOX_READ_ONLY_BIND,
        "/home/ec2-user": missing,
        "/root": SANDBOX_READ_ONLY_BIND,
        "/tmp/scratch": missing,
        LOOP: SANDBOX_READ_ONLY_BIND,
        "/data": missing,
    }
    collector = host_case.build(
        "_filesystems", filesystems=list(flags), statvfs=sandbox_statvfs(flags, probed)
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.host"):
        batch = collector.collect_once()
        host_case.clock.seconds += 15
        collector.collect_once()

    assert gauges(batch) == {"/data": (1.0, {"failure_mode": "MOUNT_MISSING"})}, gauges(
        batch
    )
    assert {sample.device for sample in batch.samples} == {"/data"}, batch.samples
    assert set(probed) == {"/tmp/scratch", "/data"}, (
        f"a path on another filesystem in the sandbox was probed: {probed}"
    )
    findings = NodeHealthPolicy(build_store()).evaluate_metrics(batch)
    assert [(item.metric_name, item.device) for item in findings] == [
        ("filesystem_unavailable", "/data")
    ], findings
    hidden = sorted(
        record.args[0]
        for record in caplog.records
        if record.getMessage().startswith("configured filesystem")
        and isinstance(record.args, tuple)
    )
    assert hidden == sorted(
        {"/home", "/home/ec2-user", "/root", "/tmp/scratch", LOOP}
    ), f"each hidden path is logged once: {hidden}"


# --------------------------------------------------------------------------- #
# The mount table itself
# --------------------------------------------------------------------------- #
def test_mountinfo_lines_parse_with_optional_fields_and_escapes():
    table = parse_mountinfo(
        "\n".join(
            [
                SANDBOX[0],
                # No optional field at all, and an escaped space in the path.
                "40 2177 0:42 / /mnt/my\\040data rw,relatime - ext4 /dev/xvdb rw",
                "not a mountinfo line",
                "41 2177 0:43 / /short rw - ext4",
            ]
        )
    )

    assert [entry.mount_point for entry in table] == ["/", "/mnt/my data"], table
    root, data = table
    assert (root.device, root.root, root.fs_type) == ("259:1", "/", "ext4"), root
    assert root.mount_options == ("ro", "nosuid", "relatime"), root
    assert root.super_options[0] == "rw", root
    assert root.read_only, "a read-only per-mount bind is read-only"
    assert not data.read_only, "rw per mount and per superblock is read-write"


def test_an_unreadable_or_empty_table_is_no_table(tmp_path):
    assert read_mount_table(tmp_path / "absent") is None, "absent file"
    assert read_mount_table(tmp_path) is None, "a directory cannot be read"
    empty = tmp_path / "empty"
    empty.write_text("")
    assert read_mount_table(empty) is None, "a table without mounts"


def test_the_visible_mount_of_a_stack_holds_the_path():
    # AL2023 mounts /tmp as tmpfs; PrivateTmp stacks the private bind on the
    # sandbox's copy of it, and that bind is what a lookup of /tmp ends on.
    table = parse_mountinfo(
        "\n".join(
            [
                f"2177 76 259:1 / / ro,nosuid,relatime master:1 - ext4 /dev/root {ROOT_SUPER}",
                "2190 2177 0:40 / /tmp ro,nosuid,nodev master:30 - tmpfs tmpfs rw",
                f"2217 2190 0:40 /systemd-private-{PRIVATE}-7a5KHD/tmp /tmp "
                "rw,nosuid,nodev master:30 - tmpfs tmpfs rw",
            ]
        )
    )

    entry = covering_mount(table, "/tmp/scratch")
    assert entry is not None and entry.mount_id == "2217", entry
    root = covering_mount(table, "/var")
    assert root is not None and root.mount_point == "/", root
    assert covering_mount(table[1:], "/var") is None, "no mount holds /var"


@pytest.mark.parametrize(
    ("path", "same_filesystem", "same_directory", "host_mount_point"),
    [
        ("/", True, True, "/"),
        ("/var", True, True, "/"),
        ("/var/lib/gpu-fault", True, True, "/"),
        ("/tmp", True, False, "/"),
        ("/var/tmp/x", True, False, "/"),
        ("/run/user", True, False, "/run"),
        ("/home", False, False, "/home"),
        ("/home/ec2-user", False, False, "/home"),
        ("/root", False, False, "/"),
    ],
)
def test_the_sandbox_view_of_a_path_is_judged_against_the_host(
    path, same_filesystem, same_directory, host_mount_point
):
    view = host_mount(
        path, parse_mountinfo("\n".join(SANDBOX)), parse_mountinfo("\n".join(HOST))
    )

    assert view.entry is not None and view.entry.mount_point == host_mount_point, view
    assert (view.same_filesystem, view.same_directory) == (
        same_filesystem,
        same_directory,
    ), view
    assert not view.entry.read_only, f"{path} is read-write on the host"


def test_without_both_tables_the_collector_reading_stands():
    own = parse_mountinfo("\n".join(SANDBOX))
    host = parse_mountinfo("\n".join(HOST))

    assert host_mount("/home", own, None) == (None, True, True), "no host table"
    view = host_mount("/home", None, host)
    assert view.entry is not None and view.entry.mount_point == "/home", view
    assert (view.same_filesystem, view.same_directory) == (True, True), view
