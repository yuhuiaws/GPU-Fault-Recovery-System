"""A shared mount that vanishes, or whose data path fails, reads unavailable.

GF-GENERAL-LIVE-005 ``mount_disappears`` and ``io_only_failure``, with the
``stat_timeout``, ``healthy_other_mount`` and ``recovery`` variants kept
honest alongside. ``_shared_filesystems`` used to rediscover the NFS/Lustre
mounts from the mount table every tick with no memory -- a mount that was
unmounted simply stopped being sampled and the gauge never rose -- and judged
each one by ``statvfs`` alone, which a server with a broken data path answers.

Each case drives the real ``_shared_filesystems`` through ``collect_once``
against mount tables written under a private ``/proc``, a scripted ``statvfs``
and a scripted directory read, grades the batch with the real
``NodeHealthPolicy`` and re-reads the persisted latest values with
``validate_host_metrics`` as VALIDATE_HOST does.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

import pytest

from gpu_fault.collectors.host.mount_table import parse_mountinfo
from gpu_fault.collectors.host.shared_mounts import (
    SharedMountMemory,
    node_scoped,
    shared_mounts,
)
from gpu_fault.host_diagnostics import validate_host_metrics
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch, NodeHealthPolicy
from gpu_fault.models import RecoveryAction, Severity
from tests._builders import build_store
from tests.collectors import _cov95_runtime_collect as common
from tests.collectors import _cov95_runtime_collect_host as support

isolated_runtime = common.isolated_runtime
host_case = support.host_case

ROOT = "28 1 259:1 / / rw,relatime shared:1 - ext4 /dev/root rw"
FSX = "40 28 0:42 / /fsx rw,relatime shared:40 - lustre 10.0.0.1@tcp:/fsx rw,flock"
SHARED = (
    "41 28 0:43 / /shared rw,relatime shared:41 - nfs4 10.0.0.2:/shared "
    "rw,vers=4.1,hard,proto=tcp"
)
POD_VOLUME = (
    "50 28 0:44 / /var/lib/kubelet/pods/9f1c/volumes/kubernetes.io~nfs/data "
    "rw,relatime shared:50 - nfs4 10.0.0.2:/data rw,vers=4.1"
)
AUTOFS_HOME = "60 28 0:45 / /home rw,relatime shared:60 - autofs systemd-1 rw,fd=40"
AUTOMOUNTED_HOME = (
    "61 60 0:46 / /home/alice rw,relatime shared:61 - nfs4 10.0.0.2:/home/alice rw"
)
#: The export labels every shared sample carries (``source`` identifies the
#: export across nodes for the control plane's failure-domain correlation;
#: the mount point is not part of that identity).
FSX_LABELS = {"source": "10.0.0.1@tcp:/fsx", "fstype": "lustre"}
SHARED_LABELS = {"source": "10.0.0.2:/shared", "fstype": "nfs4"}
POD_LABELS = {"source": "10.0.0.2:/data", "fstype": "nfs4"}
HOME_LABELS = {"source": "10.0.0.2:/home/alice", "fstype": "nfs4"}


def write_tables(case, *lines: str, own: tuple[str, ...] | None = None) -> None:
    """Both tables -- the collector's and PID 1's -- unless ``own`` differs."""

    support.write_file(case.roots.proc / "1/mountinfo", "\n".join(lines) + "\n")
    support.write_file(
        case.roots.proc / "self/mountinfo",
        "\n".join(own if own is not None else lines) + "\n",
    )


def host_baseline(case) -> None:
    """The rest of VALIDATE_HOST's required baseline, read healthy."""

    support.write_file(case.roots.proc / "stat", "cpu 100 0 0 800 0 0 0 0\n")
    support.write_file(case.roots.proc / "loadavg", "0.1 0.1 0.1 1/10 123\n")
    support.write_file(
        case.roots.proc / "meminfo",
        "MemTotal: 1000 kB\nMemAvailable: 800 kB\nMemFree: 700 kB\n",
    )


class DirectoryProbe:
    """A scripted directory read: an error per path, or a read that hangs."""

    def __init__(self) -> None:
        self.failures: dict[str, OSError] = {}
        self.blocking: set[str] = set()
        self.released = threading.Event()
        self.calls: list[str] = []

    def __call__(self, path: str) -> None:
        self.calls.append(path)
        if path in self.blocking:
            self.released.wait()
        error = self.failures.get(path)
        if error is not None:
            raise error


HEALTHY = os.statvfs_result((4096, 4096, 100, 60, 50, 0, 0, 0, 0, 255))


def scripted_statvfs(
    failures: dict[str, OSError] | None = None,
    blocking: set[str] | None = None,
    released: threading.Event | None = None,
):
    def probe(path: str) -> os.statvfs_result:
        if blocking and path in blocking and released is not None:
            released.wait()
        error = (failures or {}).get(path)
        if error is not None:
            raise error
        return HEALTHY

    return probe


def build(case, probe: DirectoryProbe, *contributors: str, **options):
    """A collector running ``_shared_filesystems`` and any other contributors."""

    return case.build(
        *contributors,
        "_shared_filesystems",
        directory_probe=probe,
        shared_filesystem_state_path=str(case.tmp_path / "shared-mounts.json"),
        **options,
    )


def gauges(batch) -> dict[str, tuple[float, dict[str, str]]]:
    return {
        sample.device: (sample.value, sample.labels)
        for sample in batch.samples
        if sample.name == "shared_filesystem_unavailable"
    }


def memory_file(case) -> dict[str, dict]:
    return json.loads((case.tmp_path / "shared-mounts.json").read_text())["mounts"]


def graded_and_validated(batch):
    """The findings, and what VALIDATE_HOST reads back from the latest values."""

    store = build_store()
    findings = NodeHealthPolicy(store).evaluate_metrics(batch)
    latest = store.list_telemetry_metrics_latest(batch.cluster_id, batch.node_id)
    failures: dict[str, list[str]] = {}
    missing = validate_host_metrics(latest, batch.node_id, failures)
    assert missing is None, missing
    return findings, failures.get(batch.node_id, [])


# --------------------------------------------------------------------------- #
# mount_disappears / recovery
# --------------------------------------------------------------------------- #
def test_a_vanished_node_scoped_shared_mount_is_unavailable_until_remounted(host_case):
    """``mount_disappears``: gone from the table is MOUNT_MISSING, not silence.

    The mount is remembered from the tick that saw it; while the table no
    longer lists it and its mount point directory is still there (the
    dangerous case: workloads keep writing under the old path into the local
    disk), it is reported unavailable every tick, graded CRITICAL
    RUN_DIAGNOSTICS, and VALIDATE_HOST keeps refusing the node. Remounting it
    clears the gauge (``recovery``).
    """

    host_baseline(host_case)
    probe = DirectoryProbe()
    write_tables(host_case, ROOT, FSX, SHARED)
    collector = build(host_case, probe, "_cpu", "_memory", "_filesystems")

    first = collector.collect_once()
    assert gauges(first) == {"/fsx": (0, FSX_LABELS), "/shared": (0, SHARED_LABELS)}, (
        gauges(first)
    )
    assert support.values(first)[("shared_filesystem_used_percent", "/fsx")] == 50
    assert set(memory_file(host_case)) == {"/fsx", "/shared"}, memory_file(host_case)

    write_tables(host_case, ROOT, SHARED)
    host_case.clock.sleep(15)
    second = collector.collect_once()
    values = support.values(second)
    assert gauges(second)["/shared"] == (0, SHARED_LABELS), gauges(second)
    assert gauges(second)["/fsx"] == (
        1,
        {
            "failure_mode": "MOUNT_MISSING",
            "source": "10.0.0.1@tcp:/fsx",
            "fstype": "lustre",
        },
    ), gauges(second)
    assert ("shared_filesystem_used_percent", "/fsx") not in values, (
        "a capacity reading was invented for a mount that is gone"
    )
    assert second.collection_errors == [], second.collection_errors
    findings, refused = graded_and_validated(second)
    assert [(item.metric_name, item.device) for item in findings] == [
        ("shared_filesystem_unavailable", "/fsx")
    ], findings
    assert findings[0].severity is Severity.CRITICAL, findings[0]
    assert findings[0].recommended_action is RecoveryAction.RUN_DIAGNOSTICS, findings[0]
    assert findings[0].diagnostic_parameters["failure_mode"] == "MOUNT_MISSING"
    assert findings[0].diagnostic_parameters["source"] == "10.0.0.1@tcp:/fsx"
    assert refused == ["shared_filesystem_unavailable"], (
        f"VALIDATE_HOST must keep refusing the node while the mount is gone: {refused}"
    )
    record = memory_file(host_case)["/fsx"]
    assert record["missing_since"] == host_case.clock.now().isoformat(), record

    host_case.clock.sleep(15)
    third = collector.collect_once()
    assert gauges(third)["/fsx"][0] == 1, "the second missing tick fell silent"

    write_tables(host_case, ROOT, FSX, SHARED)
    host_case.clock.sleep(15)
    recovered = collector.collect_once()
    assert gauges(recovered) == {
        "/fsx": (0, FSX_LABELS),
        "/shared": (0, SHARED_LABELS),
    }, gauges(recovered)
    assert support.values(recovered)[("shared_filesystem_used_percent", "/fsx")] == 50
    assert memory_file(host_case)["/fsx"]["missing_since"] is None, memory_file(
        host_case
    )


def test_the_memory_survives_a_collector_restart(host_case):
    """The watchdog restarts a collector a hung mount wedged: it must not forget.

    A new collector process over the same state file reports the vanished
    mount on its first tick. Without a state file the memory is process-only
    and a restart starts blind -- the documented, non-default mode.
    """

    probe = DirectoryProbe()
    write_tables(host_case, ROOT, FSX)
    build(host_case, probe).collect_once()

    write_tables(host_case, ROOT)
    host_case.clock.sleep(15)
    restarted = build(host_case, probe).collect_once()
    assert gauges(restarted)["/fsx"][1]["failure_mode"] == "MOUNT_MISSING", gauges(
        restarted
    )

    blind = host_case.build(
        "_shared_filesystems", directory_probe=probe, shared_filesystem_state_path=None
    ).collect_once()
    assert gauges(blind) == {}, gauges(blind)


def test_a_torn_down_shared_mount_is_forgotten(host_case):
    """An unmount followed by ``rmdir`` of the mount point is a teardown.

    The directory read of the mount point answers ENOENT: nothing is reported,
    the record is dropped from the file, and the path is not reported later
    either -- a decommissioned export must not hold the node forever.
    """

    probe = DirectoryProbe()
    write_tables(host_case, ROOT, FSX, SHARED)
    collector = build(host_case, probe)
    collector.collect_once()

    write_tables(host_case, ROOT, SHARED)
    probe.failures["/fsx"] = FileNotFoundError(
        errno.ENOENT, "No such file or directory"
    )
    host_case.clock.sleep(15)
    torn_down = collector.collect_once()
    assert gauges(torn_down) == {"/shared": (0, SHARED_LABELS)}, gauges(torn_down)
    assert set(memory_file(host_case)) == {"/shared"}, memory_file(host_case)

    del probe.failures["/fsx"]
    host_case.clock.sleep(15)
    later = collector.collect_once()
    assert "/fsx" not in gauges(later), "a forgotten mount came back from nowhere"


def test_workload_and_automounter_mounts_are_sampled_but_not_remembered(host_case):
    """A Pod volume or an autofs child comes and goes by design.

    While present it is sampled like any shared mount; its disappearance is a
    job teardown or an idle expiry, never MOUNT_MISSING -- that would raise a
    CRITICAL storage finding on every job that ends.
    """

    probe = DirectoryProbe()
    pod_volume = "/var/lib/kubelet/pods/9f1c/volumes/kubernetes.io~nfs/data"
    write_tables(host_case, ROOT, FSX, POD_VOLUME, AUTOFS_HOME, AUTOMOUNTED_HOME)
    collector = build(host_case, probe)

    present = collector.collect_once()
    assert gauges(present) == {
        "/fsx": (0, FSX_LABELS),
        pod_volume: (0, POD_LABELS),
        "/home/alice": (0, HOME_LABELS),
    }, gauges(present)
    assert set(memory_file(host_case)) == {"/fsx"}, memory_file(host_case)

    write_tables(host_case, ROOT, FSX, AUTOFS_HOME)
    host_case.clock.sleep(15)
    gone = collector.collect_once()
    assert gauges(gone) == {"/fsx": (0, FSX_LABELS)}, gauges(gone)
    assert probe.calls.count(pod_volume) == 1, "a departed Pod volume was probed again"


def test_a_shared_mount_the_sandbox_hides_is_neither_reported_nor_remembered(
    host_case, caplog
):
    """``ProtectHome=`` lays an inaccessible directory over an NFS ``/home``.

    The host has the mount; the collector's own ``statvfs``/``open`` there
    would read the sandbox's tmpfs, not the export. Missing evidence: warned
    about once, not sampled, and not remembered -- so its later absence from
    the host table is not MOUNT_MISSING either.
    """

    probe = DirectoryProbe()
    home = "61 28 0:46 / /home rw,relatime shared:61 - nfs4 10.0.0.2:/home rw"
    hidden = (
        "2249 2177 0:29 /systemd/inaccessible/dir /home ro,nosuid,nodev,noexec "
        "shared:585 master:12 - tmpfs tmpfs rw,size=51704252k,mode=755"
    )
    write_tables(host_case, ROOT, FSX, home, own=(ROOT, FSX, hidden))
    collector = build(host_case, probe)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.host"):
        first = collector.collect_once()
        host_case.clock.sleep(15)
        second = collector.collect_once()
    assert gauges(first) == {"/fsx": (0, FSX_LABELS)}, gauges(first)
    assert gauges(second) == {"/fsx": (0, FSX_LABELS)}, gauges(second)
    assert "/home" not in probe.calls, probe.calls
    assert set(memory_file(host_case)) == {"/fsx"}, memory_file(host_case)
    hidden_warnings = [
        record for record in caplog.records if "/home" in record.getMessage()
    ]
    assert len(hidden_warnings) == 1, [item.getMessage() for item in caplog.records]

    write_tables(host_case, ROOT, FSX, own=(ROOT, FSX, hidden))
    host_case.clock.sleep(15)
    third = collector.collect_once()
    assert gauges(third) == {"/fsx": (0, FSX_LABELS)}, gauges(third)


def test_an_unreadable_host_table_falls_back_to_the_collectors_own(host_case):
    """Without PID 1's table the collector's own stands, as it always did."""

    probe = DirectoryProbe()
    support.write_file(host_case.roots.proc / "self/mountinfo", f"{ROOT}\n{FSX}\n")
    collector = build(host_case, probe)

    first = collector.collect_once()
    assert gauges(first) == {"/fsx": (0, FSX_LABELS)}, gauges(first)

    support.write_file(host_case.roots.proc / "self/mountinfo", f"{ROOT}\n")
    host_case.clock.sleep(15)
    second = collector.collect_once()
    assert gauges(second)["/fsx"][1]["failure_mode"] == "MOUNT_MISSING", gauges(second)


def test_a_malformed_memory_file_starts_empty_and_is_rewritten(host_case, caplog):
    """The collector's own bookkeeping must never stop it from running."""

    probe = DirectoryProbe()
    (host_case.tmp_path / "shared-mounts.json").write_text("{not json")
    write_tables(host_case, ROOT, FSX)
    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.host"):
        batch = build(host_case, probe).collect_once()
    assert gauges(batch) == {"/fsx": (0, FSX_LABELS)}, gauges(batch)
    assert set(memory_file(host_case)) == {"/fsx"}, memory_file(host_case)
    assert any(
        "shared mount memory" in record.getMessage() for record in caplog.records
    ), [item.getMessage() for item in caplog.records]


# --------------------------------------------------------------------------- #
# io_only_failure / healthy_other_mount / stat_timeout
# --------------------------------------------------------------------------- #
def test_a_data_path_failure_behind_a_healthy_statvfs_is_unavailable(host_case):
    """``io_only_failure``: FSSTAT answers, the export refuses to be read.

    Server-side permissions changed under a live mount: ``statvfs`` keeps
    answering (the capacity reading is still delivered) while opening the
    mount point and reading one entry is EACCES. The mount is unavailable with
    the errno, graded CRITICAL RUN_DIAGNOSTICS, VALIDATE_HOST refuses the node,
    and the healthy export beside it does not mask it
    (``healthy_other_mount``). Restoring the export clears it.
    """

    host_baseline(host_case)
    probe = DirectoryProbe()
    probe.failures["/shared"] = PermissionError(errno.EACCES, "Permission denied")
    write_tables(host_case, ROOT, FSX, SHARED)
    collector = build(host_case, probe, "_cpu", "_memory", "_filesystems")

    batch = collector.collect_once()
    values = support.values(batch)
    assert gauges(batch)["/fsx"] == (0, FSX_LABELS), gauges(batch)
    assert gauges(batch)["/shared"] == (
        1,
        {
            "failure_mode": "IO_ERROR",
            "errno": "EACCES",
            "source": "10.0.0.2:/shared",
            "fstype": "nfs4",
        },
    ), gauges(batch)
    assert values[("shared_filesystem_used_percent", "/shared")] == 50, (
        "statvfs answered: the capacity reading is still evidence"
    )
    assert batch.collection_errors == [], batch.collection_errors
    findings, refused = graded_and_validated(batch)
    assert [(item.metric_name, item.device) for item in findings] == [
        ("shared_filesystem_unavailable", "/shared")
    ], findings
    assert findings[0].recommended_action is RecoveryAction.RUN_DIAGNOSTICS, findings[0]
    assert findings[0].diagnostic_parameters["errno"] == "EACCES", findings[0]
    assert refused == ["shared_filesystem_unavailable"], refused

    del probe.failures["/shared"]
    host_case.clock.sleep(15)
    restored = collector.collect_once()
    assert gauges(restored) == {
        "/fsx": (0, FSX_LABELS),
        "/shared": (0, SHARED_LABELS),
    }, gauges(restored)


def test_a_data_path_read_that_never_answers_is_unknown_not_healthy(host_case):
    """A stuck export hangs the read; the tick does not hang with it.

    No errno and no verdict: IO_TIMEOUT is missing evidence, reported
    unavailable rather than healthy or dropped, and it stays IO_TIMEOUT while
    the abandoned read is still parked -- the next tick's ``statvfs`` finding
    the mount busy must not relabel the same condition PROBE_TIMEOUT.
    """

    probe = DirectoryProbe()
    probe.blocking.add("/shared")
    write_tables(host_case, ROOT, FSX, SHARED)
    collector = build(host_case, probe, statvfs_timeout_seconds=0.3)

    try:
        started = time.monotonic()
        batch = collector.collect_once()
        elapsed = time.monotonic() - started
        host_case.clock.sleep(15)
        again = collector.collect_once()
    finally:
        probe.released.set()

    assert elapsed < 3, f"the tick waited on the hung read: {elapsed:.1f}s"
    assert gauges(batch)["/fsx"] == (0, FSX_LABELS), gauges(batch)
    assert gauges(batch)["/shared"] == (
        1,
        {"failure_mode": "IO_TIMEOUT", "source": "10.0.0.2:/shared", "fstype": "nfs4"},
    ), gauges(batch)
    assert support.values(batch)[("shared_filesystem_used_percent", "/shared")] == 50
    assert gauges(again)["/shared"][1]["failure_mode"] == "IO_TIMEOUT", gauges(again)
    assert probe.calls.count("/shared") == 1, "the hung read was started again"


def test_a_failed_statvfs_names_its_errno_and_a_hung_one_its_mode(host_case):
    """``stat_timeout`` and ESTALE keep their verdicts, now with a failure mode."""

    probe = DirectoryProbe()
    released = threading.Event()
    write_tables(host_case, ROOT, FSX, SHARED)
    collector = build(
        host_case,
        probe,
        statvfs=scripted_statvfs(
            failures={"/shared": OSError(errno.ESTALE, "Stale file handle")},
            blocking={"/fsx"},
            released=released,
        ),
        statvfs_timeout_seconds=0.3,
    )

    try:
        batch = collector.collect_once()
    finally:
        released.set()

    assert gauges(batch)["/shared"] == (
        1,
        {
            "failure_mode": "PROBE_FAILED",
            "errno": "ESTALE",
            "source": "10.0.0.2:/shared",
            "fstype": "nfs4",
        },
    ), gauges(batch)
    assert gauges(batch)["/fsx"] == (
        1,
        {
            "failure_mode": "PROBE_TIMEOUT",
            "source": "10.0.0.1@tcp:/fsx",
            "fstype": "lustre",
        },
    ), gauges(batch)
    assert probe.calls == [], "the data path was read on a mount statvfs failed"
    values = support.values(batch)
    assert ("shared_filesystem_used_percent", "/shared") not in values, values
    assert ("shared_filesystem_used_percent", "/fsx") not in values, values


def test_batches_from_a_collector_without_labels_still_grade():
    """Wire compatibility: the labels are optional, the rule reads the value."""

    batch = HostTelemetryBatch(
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
        samples=[
            HostMetricSample(
                name="shared_filesystem_unavailable", value=1, device="/fsx"
            )
        ],
    )

    findings = NodeHealthPolicy(build_store()).evaluate_metrics(batch)

    assert [item.recommended_action for item in findings] == [
        RecoveryAction.RUN_DIAGNOSTICS
    ], findings
    assert findings[0].diagnostic_parameters == {}, findings[0]


def test_every_shared_sample_names_its_export_for_failure_domain_correlation(host_case):
    """The control plane correlates nodes on one export by the sample labels.

    ``source`` is the mount table's source field, the same on every node that
    mounts the export; ``fstype`` rides along. Both the availability gauge --
    healthy or any failure mode -- and the capacity reading carry them, and
    the mount point stays out of the labels: nodes may mount one export at
    different paths.
    """

    probe = DirectoryProbe()
    probe.failures["/shared"] = PermissionError(errno.EACCES, "Permission denied")
    write_tables(host_case, ROOT, FSX, SHARED)
    collector = build(host_case, probe)
    batch = collector.collect_once()
    write_tables(host_case, ROOT, SHARED)
    host_case.clock.sleep(15)
    vanished = collector.collect_once()

    shared_samples = [
        sample
        for sample in [*batch.samples, *vanished.samples]
        if sample.name
        in {"shared_filesystem_unavailable", "shared_filesystem_used_percent"}
    ]
    assert len(shared_samples) == 7, [
        (s.name, s.device, s.value) for s in shared_samples
    ]
    expected = {"/fsx": FSX_LABELS, "/shared": SHARED_LABELS}
    for sample in shared_samples:
        assert sample.labels["source"] == expected[sample.device]["source"], sample
        assert sample.labels["fstype"] == expected[sample.device]["fstype"], sample
        assert sample.device not in sample.labels.values(), (
            f"the mount point leaked into the correlation identity: {sample}"
        )
    modes = {
        (sample.device, sample.value): sample.labels.get("failure_mode")
        for sample in shared_samples
        if sample.name == "shared_filesystem_unavailable"
    }
    assert modes == {
        ("/fsx", 0): None,
        ("/shared", 1): "IO_ERROR",
        ("/fsx", 1): "MOUNT_MISSING",
    }, modes


# --------------------------------------------------------------------------- #
# the memory and the table helpers on their own
# --------------------------------------------------------------------------- #
def test_the_memory_is_bounded_and_refuses_rather_than_evicts():
    memory = SharedMountMemory(None, max_entries=2)
    at = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)

    first = memory.observe(
        {"/a": ("s:/a", "nfs4"), "/b": ("s:/b", "nfs4"), "/c": ("s:/c", "nfs4")}, at
    )
    assert first.refused == ("/c",), first
    assert first.missing == (), first
    assert set(memory.records) == {"/a", "/b"}, memory.records

    second = memory.observe({"/b": ("s:/b", "nfs4")}, at)
    assert [record.mount_point for record in second.missing] == ["/a"], second
    assert second.missing[0].missing_since == at.isoformat(), second
    memory.forget("/a")
    assert set(memory.records) == {"/b"}, memory.records
    assert memory.save() is None, "a memory without a file has nothing to fail"


def test_a_memory_file_over_the_cap_keeps_the_oldest_known_mounts(tmp_path):
    path = tmp_path / "shared-mounts.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "mounts": {
                    f"/m{index}": {
                        "source": f"s:/m{index}",
                        "fstype": "nfs4",
                        "first_seen": f"2026-07-{10 + index:02d}T00:00:00+00:00",
                        "missing_since": None,
                    }
                    for index in range(4)
                },
            }
        )
    )
    memory = SharedMountMemory(path, max_entries=2)

    assert memory.load() is None, "a well-formed file over the cap is not an error"
    assert set(memory.records) == {"/m0", "/m1"}, memory.records
    assert memory.save() is None, "the trimmed set could not be written"
    assert set(json.loads(path.read_text())["mounts"]) == {"/m0", "/m1"}


@pytest.mark.parametrize(
    ("line", "scoped"),
    [
        (FSX, True),
        (SHARED, True),
        (POD_VOLUME, False),
        (
            "51 28 0:47 / /var/lib/kubelet/plugins/kubernetes.io/csi/fsx/abc/globalmount "
            "rw - lustre 10.0.0.1@tcp:/fsx rw",
            False,
        ),
        (AUTOMOUNTED_HOME, False),
    ],
    ids=["fstab-lustre", "fstab-nfs", "pod-volume", "csi-global-mount", "autofs-child"],
)
def test_node_scoped_tells_the_nodes_mounts_from_a_workloads(line, scoped):
    table = parse_mountinfo("\n".join([ROOT, AUTOFS_HOME, line]) + "\n")
    entry = next(item for item in table if item.mount_id == line.split()[0])

    assert node_scoped(entry, table) is scoped, (entry, scoped)


def test_shared_mounts_lists_the_visible_mount_at_each_point():
    """A shared mount another filesystem is stacked over is not reachable."""

    table = parse_mountinfo(
        "\n".join(
            [
                ROOT,
                FSX,
                "42 28 0:43 / /shared rw - nfs4 10.0.0.2:/old rw",
                "43 42 0:48 / /shared rw - nfs4 10.0.0.2:/new rw",
                "44 28 0:49 / /covered rw - nfs4 10.0.0.2:/covered rw",
                "45 44 7:1 / /covered rw - ext4 /dev/loop1 rw",
            ]
        )
        + "\n"
    )

    assert [(entry.mount_point, entry.source) for entry in shared_mounts(table)] == [
        ("/fsx", "10.0.0.1@tcp:/fsx"),
        ("/shared", "10.0.0.2:/new"),
    ]
