"""A host read that failed, or never existed, is missing evidence.

GF-GENERAL-LIVE-004/007/010/012: an unreadable source must read UNKNOWN (a
collection error, so validation waits and nothing is graded), a source the
node cannot provide must say NOT_SUPPORTED, and neither may be reported as a
healthy reading or as a zero. Every case drives the real contributor through
``collect_once`` and grades the batch with the real ``NodeHealthPolicy``, so
"never a fault" is checked where a fault would be minted.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from gpu_fault.host_diagnostics import validate_host_metrics
from gpu_fault.host_health import NodeHealthPolicy
from gpu_fault.models import RecoveryAction, Severity
from tests._builders import build_store
from tests.collectors import _cov95_runtime_collect as common
from tests.collectors import _cov95_runtime_collect_host as support

isolated_runtime = common.isolated_runtime
host_case = support.host_case

SCAN = ("smartctl", "--scan-open")


def health(device: str) -> tuple[str, ...]:
    return ("smartctl", "-H", "-j", device)


def verdict(passed: object) -> tuple[str, str, int]:
    return json.dumps({"smart_status": {"passed": passed}}), "", 0


def smart_markers(batch) -> list[tuple[str, str | None, dict[str, str]]]:
    return [
        (sample.name, sample.device, sample.labels)
        for sample in batch.samples
        if sample.name.startswith("smart_")
    ]


def graded(batch):
    return NodeHealthPolicy(build_store()).evaluate_metrics(batch)


# --------------------------------------------------------------------------- #
# 007 SMART
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "answer",
    [
        ("", "smartctl: unrecognized option '--scan-open'\n", 1),
        ("# scan_smart_devices: glob(3) aborted matching pattern /dev/sd*\n", "", 0),
        subprocess.TimeoutExpired(list(SCAN), 20),
    ],
    ids=["exit-status", "scan-error-line", "timeout"],
)
def test_a_failed_drive_scan_is_unknown_not_a_node_without_drives(host_case, answer):
    """``--scan-open`` that could not enumerate is not "no drives to check".

    smartctl prints ``# scan_smart_devices: <error>`` and still exits 0 when the
    platform scan fails, and a non-zero exit was ignored: both parsed as an
    empty drive set, which posted nothing and read as a healthy node.
    """

    support.enable_tools(host_case, "smartctl")
    host_case.outputs[SCAN] = answer
    collector = host_case.build("_smart")

    for _ in range(2):
        batch = collector.collect_once()
        assert smart_markers(batch) == [
            ("smart_query_failed", None, {"failure_mode": "SCAN_FAILED"})
        ], batch.samples
        assert any("--scan-open" in error for error in batch.collection_errors), (
            f"a failed scan must invalidate the host observation: {batch}"
        )
        assert graded(batch) == [], "an unreadable drive set is not a disk fault"
        host_case.clock.sleep(15)
    probes = [argv for argv, _ in host_case.commands if "-H" in argv]
    assert probes == [], f"no drive may be queried from a failed scan: {probes}"


def test_a_drive_the_scan_could_not_open_is_unknown_not_dropped(host_case):
    """``--scan-open`` reports an unopenable drive only as a ``#`` comment.

    The parser read only ``/dev/...`` lines, so a drive that started refusing
    opens -- the dying NVMe the check exists for -- silently left the drive
    set and the node read as healthy.
    """

    support.enable_tools(host_case, "smartctl")
    host_case.outputs[SCAN] = (
        "/dev/nvme0 -d nvme # /dev/nvme0, NVMe device\n"
        "# /dev/nvme1 -d nvme # /dev/nvme1, NVMe device open failed: "
        "Input/output error\n",
        "",
        0,
    )
    host_case.outputs[health("/dev/nvme0")] = verdict(True)
    batch = host_case.build("_smart").collect_once()

    assert sorted(smart_markers(batch), key=lambda item: str(item[1])) == [
        ("smart_health_failed", "/dev/nvme0", {}),
        ("smart_query_failed", "/dev/nvme1", {"failure_mode": "OPEN_FAILED"}),
    ], batch.samples
    assert any("/dev/nvme1" in error for error in batch.collection_errors), (
        f"the unopenable drive must invalidate the observation: {batch}"
    )
    assert graded(batch) == [], "an unopenable drive is not a SMART failure"


def test_a_hung_drive_query_keeps_the_other_drives_verdicts(host_case):
    """One ``smartctl -H`` that times out used to discard the whole tick.

    The exception left ``_smart`` with nothing, so a drive whose
    self-assessment had just FAILED was never reported while its neighbour
    hung -- and the neighbour hangs every tick.
    """

    support.enable_tools(host_case, "smartctl")
    host_case.outputs[SCAN] = ("/dev/nvme0 -d nvme\n/dev/nvme1 -d nvme\n", "", 0)
    host_case.outputs[health("/dev/nvme0")] = verdict(False)
    host_case.outputs[health("/dev/nvme1")] = subprocess.TimeoutExpired(
        list(health("/dev/nvme1")), 20
    )
    batch = host_case.build("_smart").collect_once()

    values = support.values(batch)
    assert values[("smart_health_failed", "/dev/nvme0")] == 1, values
    assert values[("smart_query_failed", "/dev/nvme1")] == 1, values
    assert ("smart_health_failed", "/dev/nvme1") not in values, values
    assert any("/dev/nvme1" in error for error in batch.collection_errors), (
        f"the hung query must invalidate the observation: {batch}"
    )
    findings = graded(batch)
    assert [(item.metric_name, item.device) for item in findings] == [
        ("smart_health_failed", "/dev/nvme0")
    ], findings
    assert findings[0].recommended_action is RecoveryAction.QUARANTINE, findings


@pytest.mark.parametrize("passed", ["false", 0, None, ["no"]])
def test_a_malformed_smart_verdict_is_a_failed_query(host_case, passed):
    """``passed`` must be a JSON boolean; ``"false"`` read as a healthy drive."""

    support.enable_tools(host_case, "smartctl")
    host_case.outputs[SCAN] = ("/dev/nvme0 -d nvme\n", "", 0)
    payload = {"smart_status": {"passed": passed}}
    if passed is None:
        payload = {"smart_status": "PASSED", "smartctl": {"exit_status": 4}}
    host_case.outputs[health("/dev/nvme0")] = (json.dumps(payload), "", 0)
    batch = host_case.build("_smart").collect_once()

    assert smart_markers(batch) == [
        ("smart_query_failed", "/dev/nvme0", {"failure_mode": "QUERY_FAILED"})
    ], batch.samples
    assert batch.collection_errors, f"a malformed verdict is unknown: {batch}"
    assert graded(batch) == [], "a malformed verdict is not a disk fault"


@pytest.mark.parametrize(
    ("tools", "scan", "reason", "device"),
    [
        ((), None, "TOOL_MISSING", None),
        (("smartctl",), ("", "", 0), "NO_DEVICES", None),
        (("smartctl",), ("/dev/sda -d scsi\n", "", 0), "NO_SMART_STATUS", "/dev/sda"),
    ],
    ids=["tool-missing", "no-drives", "device-without-smart"],
)
def test_smart_the_node_cannot_provide_is_reported_not_supported(
    host_case, tools, scan, reason, device
):
    """A node or drive without SMART is a capability gap, not a healthy scan.

    It is reported as ``smart_status_unsupported`` on every tick and never as
    a collection error: the gap is permanent, and an error would hold every
    host validation of the node until its deadline.
    """

    support.enable_tools(host_case, *tools)
    if scan is not None:
        host_case.outputs[SCAN] = scan
    host_case.outputs[health("/dev/sda")] = (
        json.dumps({"smartctl": {"exit_status": 0}}),
        "",
        0,
    )
    collector = host_case.build("_smart")

    for _ in range(2):
        batch = collector.collect_once()
        assert smart_markers(batch) == [
            ("smart_status_unsupported", device, {"reason": reason})
        ], batch.samples
        assert batch.collection_errors == [], batch.collection_errors
        assert graded(batch) == [], "missing SMART support is not a disk fault"
        host_case.clock.sleep(15)
    probes = [argv for argv, _ in host_case.commands if "-H" in argv]
    assert len(probes) == (1 if device else 0), (
        f"a verdict-less drive stays cacheable: {probes}"
    )


# --------------------------------------------------------------------------- #
# 010 EFA ethtool counters
# --------------------------------------------------------------------------- #
ETHTOOL_METRICS = {
    "efa_rnr_errors_delta",
    "efa_retry_errors_delta",
    "efa_cq_errors_delta",
    "network_pfc_pause_delta",
    "network_ecn_marks_delta",
}


def efa_netdev(case, *interfaces: str) -> None:
    device = case.roots.rdma / "rdmap0"
    support.write_file(device / "device/uevent", "DRIVER=efa\n")
    for interface in interfaces:
        (device / "device/net" / interface).mkdir(parents=True)
    support.enable_tools(case, "ethtool")


def ethtool_stats(**counters: int) -> tuple[str, str, int]:
    lines = "".join(f"     {name}: {value}\n" for name, value in counters.items())
    return f"NIC statistics:\n{lines}", "", 0


def counter_samples(batch) -> dict[tuple[str, str | None], float]:
    return {
        key: value
        for key, value in support.values(batch).items()
        if key[0] in ETHTOOL_METRICS
    }


def unsupported(batch) -> dict[str | None, dict[str, str]]:
    return {
        sample.device: sample.labels
        for sample in batch.samples
        if sample.name == "efa_ethtool_counters_unsupported"
    }


def test_the_first_ethtool_reading_is_a_baseline_not_a_zero(host_case):
    """No previous reading means no delta yet, not "zero errors"."""

    efa_netdev(host_case, "eth0")
    host_case.outputs[("ethtool", "-S", "eth0")] = ethtool_stats(
        rnr_naks=3, retry_count=4, cq_err=5, pfc_pause_tx=6, ecn_marked=7
    )
    collector = host_case.build("_efa_network")

    first = collector.collect_once()
    host_case.clock.sleep(15)
    second = collector.collect_once()

    assert counter_samples(first) == {}, (
        f"a counter with no baseline reported a delta: {first.samples}"
    )
    assert first.collection_errors == [], first.collection_errors
    assert counter_samples(second) == {
        (metric, "eth0"): 0.0 for metric in ETHTOOL_METRICS
    }, second.samples
    assert unsupported(second) == {}, second.samples


def test_a_missing_ethtool_counter_is_not_supported_not_zero(host_case):
    """A counter the driver does not expose used to be reported as 0 errors.

    ``VALIDATE_FABRIC`` reads a zero as a clean link, so a NIC without a
    retry counter "passed" a retry check it never took.
    """

    efa_netdev(host_case, "eth0")
    host_case.outputs[("ethtool", "-S", "eth0")] = ethtool_stats(rnr_naks=10)
    collector = host_case.build("_efa_network")
    collector.collect_once()
    host_case.clock.sleep(15)
    host_case.outputs[("ethtool", "-S", "eth0")] = ethtool_stats(rnr_naks=15)

    batch = collector.collect_once()

    assert counter_samples(batch) == {("efa_rnr_errors_delta", "eth0"): 5}, (
        batch.samples
    )
    assert unsupported(batch) == {
        "eth0": {
            "reason": "MISSING_COUNTER",
            "metrics": ",".join(sorted(ETHTOOL_METRICS - {"efa_rnr_errors_delta"})),
        }
    }, batch.samples
    assert batch.collection_errors == [], (
        "a counter the driver never had is a capability, not a read failure"
    )


@pytest.mark.parametrize(
    "answer",
    [
        ("", "Cannot get stats information: Input/output error\n", 97),
        subprocess.TimeoutExpired(["ethtool", "-S", "eth1"], 10),
        FileNotFoundError(2, "No such file or directory", "ethtool"),
    ],
    ids=["exit-status", "timeout", "exec-failure"],
)
def test_a_failed_ethtool_is_unknown_not_zero(host_case, answer):
    """A failed ``ethtool -S`` was skipped silently; a hang lost every netdev."""

    efa_netdev(host_case, "eth0", "eth1")
    host_case.outputs[("ethtool", "-S", "eth0")] = ethtool_stats(rnr_naks=1)
    host_case.outputs[("ethtool", "-S", "eth1")] = ethtool_stats(rnr_naks=1)
    collector = host_case.build("_efa_network")
    collector.collect_once()
    host_case.clock.sleep(15)
    host_case.outputs[("ethtool", "-S", "eth1")] = answer

    batch = collector.collect_once()

    assert counter_samples(batch) == {("efa_rnr_errors_delta", "eth0"): 0}, (
        f"the failed netdev reported counters, or the healthy one lost its: "
        f"{batch.samples}"
    )
    assert any("eth1" in error for error in batch.collection_errors), (
        f"a failed ethtool must invalidate the observation: {batch}"
    )
    assert graded(batch) == [], "an unreadable counter is not an RDMA fault"


@pytest.mark.parametrize(
    "answer",
    [
        ("", "no stats available\n", 94),
        ("", "Cannot get stats strings information: Operation not supported\n", 96),
    ],
    ids=["no-stats", "not-supported"],
)
def test_a_netdev_without_ethtool_statistics_is_not_supported(host_case, answer):
    efa_netdev(host_case, "eth0")
    host_case.outputs[("ethtool", "-S", "eth0")] = answer
    batch = host_case.build("_efa_network").collect_once()

    assert counter_samples(batch) == {}, batch.samples
    assert unsupported(batch) == {
        "eth0": {
            "reason": "NO_STATISTICS",
            "metrics": ",".join(sorted(ETHTOOL_METRICS)),
        }
    }, batch.samples
    assert batch.collection_errors == [], (
        "a driver without statistics is a capability, not a read failure"
    )


def test_an_ethtool_counter_reset_is_unknown_not_a_recovery(host_case):
    """A reset read as delta 0, which clears an active error signal."""

    efa_netdev(host_case, "eth0")
    collector = host_case.build("_efa_network")
    readings = {}
    for value in (10, 15, 2, 4):
        host_case.outputs[("ethtool", "-S", "eth0")] = ethtool_stats(rnr_naks=value)
        readings[value] = collector.collect_once()
        host_case.clock.sleep(15)

    assert counter_samples(readings[15]) == {("efa_rnr_errors_delta", "eth0"): 5}, (
        readings[15].samples
    )
    assert counter_samples(readings[2]) == {}, (
        f"a reset counter reported a delta: {readings[2].samples}"
    )
    assert any("reset" in error for error in readings[2].collection_errors), readings[
        2
    ].collection_errors
    assert counter_samples(readings[4]) == {("efa_rnr_errors_delta", "eth0"): 2}, (
        f"the reading after the reset is measured from the new baseline: "
        f"{readings[4].samples}"
    )


def test_a_non_numeric_ethtool_counter_is_unknown_not_zero(host_case):
    efa_netdev(host_case, "eth0")
    collector = host_case.build("_efa_network")
    host_case.outputs[("ethtool", "-S", "eth0")] = ethtool_stats(rnr_naks=1, cq_err=1)
    collector.collect_once()
    host_case.clock.sleep(15)
    host_case.outputs[("ethtool", "-S", "eth0")] = (
        "NIC statistics:\n     rnr_naks: 3\n     cq_err: N/A\n",
        "",
        0,
    )

    batch = collector.collect_once()

    assert counter_samples(batch) == {("efa_rnr_errors_delta", "eth0"): 2}, (
        batch.samples
    )
    assert any("cq_err" in error for error in batch.collection_errors), (
        batch.collection_errors
    )


# --------------------------------------------------------------------------- #
# 012 GPU inventory
# --------------------------------------------------------------------------- #
NVML_FAILURE = ("", "Failed to initialize NVML: Unknown Error\n", 255)


def gpu_collector(case, *, procfs: str):
    gpus = case.roots.proc / "driver/nvidia/gpus"
    if procfs == "unreadable":
        support.write_file(gpus, "not a directory\n")
    collector = case.build(
        "_gpu_inventory",
        expected_gpu_count=8,
        inventory_mismatch_consecutive_samples=2,
        node_instance_type="ml.p5.48xlarge",
    )
    return collector


@pytest.mark.parametrize("procfs", ["absent", "unreadable"])
def test_two_failed_gpu_inventory_reads_are_unknown_not_zero_gpus(host_case, procfs):
    """NVML failed and ``/proc/driver/nvidia/gpus`` could not be read.

    The procfs fallback returned 0 for an unreadable directory, so two failed
    reads became "0 of 8 GPUs" and, on the second tick, a CRITICAL
    REBOOT_NODE finding (GF-GENERAL-LIVE-012 ``query_failed`` /
    ``proc_unreadable``).
    """

    collector = gpu_collector(host_case, procfs=procfs)
    host_case.outputs[tuple(collector.GPU_QUERY_ARGV)] = NVML_FAILURE
    policy = NodeHealthPolicy(build_store())

    for _ in range(3):
        batch = collector.collect_once()
        names = {sample.name for sample in batch.samples}
        assert names == {"gpu_inventory_expected_count"}, (
            f"an unknown inventory reported a count: {batch.samples}"
        )
        (sample,) = batch.samples
        assert sample.labels["failure_mode"] == "INVENTORY_UNAVAILABLE", sample
        assert "discovered_count" not in sample.labels, sample
        assert any(
            "NVML" in error and "/proc/driver/nvidia/gpus" in error
            for error in batch.collection_errors
        ), batch.collection_errors
        assert policy.evaluate_metrics(batch) == [], (
            "two failed reads must never mint a missing-GPU REBOOT_NODE"
        )
        host_case.clock.sleep(15)


def test_a_partial_gpu_listing_never_reports_a_fabricated_discovered_count(host_case):
    """Seven listed GPUs plus an error line is still evidence of a missing GPU,
    but an unreadable procfs is no evidence of "zero bound GPUs"."""

    collector = gpu_collector(host_case, procfs="absent")
    host_case.outputs[tuple(collector.GPU_QUERY_ARGV)] = (
        "".join(f"GPU-{index}, 10\n" for index in range(7))
        + "Unable to determine the device handle for GPU 0000:07:00.0: Unknown Error\n",
        "Unable to determine the device handle for GPU 0000:07:00.0\n",
        255,
    )

    collector.collect_once()
    host_case.clock.sleep(15)
    batch = collector.collect_once()

    samples = {sample.name: sample for sample in batch.samples}
    assert "gpu_inventory_discovered_count" not in samples, batch.samples
    assert samples["gpu_inventory_active_count"].value == 7, samples
    assert samples["gpu_inventory_mismatch"].value == 1, (
        "the listed shortfall is still graded on its own evidence"
    )
    assert "discovered_count" not in samples["gpu_inventory_mismatch"].labels, samples
    assert samples["gpu_inventory_mismatch"].labels["failure_mode"] == (
        "DRIVER_QUERY_FAILED"
    ), samples


def test_an_unknown_gpu_inventory_neither_confirms_nor_clears_the_missing_streak(
    host_case,
):
    """Only valid samples count toward the consecutive-mismatch confirmation.

    The unknown tick between two valid "7 of 8" readings used to be a third
    "missing" reading of its own (0 of 8) and confirmed the loss by itself;
    it now leaves the streak where the last valid reading put it, as the
    NVML-failure path with a readable procfs already did.
    """

    collector = gpu_collector(host_case, procfs="absent")
    argv = tuple(collector.GPU_QUERY_ARGV)
    listed = ("".join(f"GPU-{index}, 10\n" for index in range(7)), "", 0)
    readings = []
    for answer in (listed, NVML_FAILURE, listed):
        host_case.outputs[argv] = answer
        readings.append(support.values(collector.collect_once()))
        host_case.clock.sleep(15)

    first, unknown, confirmed = readings
    assert first[("gpu_inventory_mismatch", None)] == 0, "one sample is not a loss"
    assert ("gpu_inventory_mismatch", None) not in unknown, (
        f"the unknown tick graded the inventory: {unknown}"
    )
    assert confirmed[("gpu_inventory_mismatch", None)] == 1, (
        f"two valid missing readings confirm the loss: {confirmed}"
    )


# --------------------------------------------------------------------------- #
# 004 local disk inodes
# --------------------------------------------------------------------------- #
def statvfs(*, blocks_free: int, files: int, files_free: int, flag: int = 0):
    return lambda path: os.statvfs_result(
        (
            4096,
            4096,
            100,
            blocks_free,
            blocks_free,
            files,
            files_free,
            files_free,
            flag,
            255,
        )
    )


def test_inode_exhaustion_is_graded_like_capacity_exhaustion(host_case):
    """Free bytes say nothing about an exhausted inode table (004 inode_full)."""

    mount = str(host_case.roots.mount)
    # The rest of VALIDATE_HOST's required baseline, read healthy.
    support.write_file(host_case.roots.proc / "stat", "cpu 100 0 0 800 0 0 0 0\n")
    support.write_file(host_case.roots.proc / "loadavg", "0.1 0.1 0.1 1/10 123\n")
    support.write_file(
        host_case.roots.proc / "meminfo",
        "MemTotal: 1000 kB\nMemAvailable: 800 kB\nMemFree: 700 kB\n",
    )
    collector = host_case.build(
        "_cpu",
        "_memory",
        "_filesystems",
        edge_filter_enabled=True,
        statvfs=statvfs(blocks_free=90, files=1000, files_free=10),
    )

    batch = collector.collect_once()

    values = support.values(batch)
    assert values[("filesystem_used_percent", mount)] == 10, values
    assert values[("filesystem_inodes_used_percent", mount)] == 99, values
    assert "threshold:filesystem_inodes_used_percent" in batch.edge_filter_reasons, (
        batch.edge_filter_reasons
    )
    store = build_store()
    findings = NodeHealthPolicy(store).evaluate_metrics(batch)
    assert [(item.metric_name, item.device) for item in findings] == [
        ("filesystem_inodes_used_percent", mount)
    ], findings
    assert findings[0].severity is Severity.CRITICAL, findings
    assert findings[0].recommended_action is RecoveryAction.QUARANTINE, findings
    latest = store.list_telemetry_metrics_latest(batch.cluster_id, batch.node_id)
    failures: dict[str, list[str]] = {}
    assert validate_host_metrics(latest, batch.node_id, failures) is None, latest
    assert failures == {batch.node_id: ["filesystem_inodes_used_percent"]}, (
        f"VALIDATE_HOST must keep refusing the node while its inodes are "
        f"exhausted, from the persisted latest value: {failures}"
    )


@pytest.mark.parametrize(
    ("files", "files_free", "expected"),
    [(1000, 30, 97), (1000, 1000, 0), (0, 0, None), (1000, 2000, 0)],
    ids=["below-threshold", "empty", "no-inode-table", "free-exceeds-total"],
)
def test_inode_usage_below_the_threshold_or_untracked_is_not_graded(
    host_case, files, files_free, expected
):
    mount = str(host_case.roots.mount)
    collector = host_case.build(
        "_filesystems",
        statvfs=statvfs(blocks_free=90, files=files, files_free=files_free),
    )

    batch = collector.collect_once()

    values = support.values(batch)
    assert values.get(("filesystem_inodes_used_percent", mount)) == expected, values
    assert graded(batch) == [], "inode usage below 98 % is not a storage fault"


# --------------------------------------------------------------------------- #
# 004 read-only / vanished / unreadable configured mount
# --------------------------------------------------------------------------- #
def _host_baseline(host_case):
    """The rest of VALIDATE_HOST's required baseline, read healthy."""

    support.write_file(host_case.roots.proc / "stat", "cpu 100 0 0 800 0 0 0 0\n")
    support.write_file(host_case.roots.proc / "loadavg", "0.1 0.1 0.1 1/10 123\n")
    support.write_file(
        host_case.roots.proc / "meminfo",
        "MemTotal: 1000 kB\nMemAvailable: 800 kB\nMemFree: 700 kB\n",
    )


def test_a_read_only_configured_mount_is_unusable_with_free_capacity(host_case):
    """ST_RDONLY is unusable even when the capacity reading is healthy (004 read_only).

    The spec forbids judging such a mount by its remaining bytes: a remount
    read-only keeps its blocks and inodes, so the capacity signals read fine
    while the mount refuses every write.
    """

    mount = str(host_case.roots.mount)
    _host_baseline(host_case)
    collector = host_case.build(
        "_cpu",
        "_memory",
        "_filesystems",
        edge_filter_enabled=True,
        statvfs=statvfs(blocks_free=90, files=1000, files_free=900, flag=os.ST_RDONLY),
    )

    batch = collector.collect_once()

    values = support.values(batch)
    assert values[("filesystem_used_percent", mount)] == 10, values
    assert values[("filesystem_unavailable", mount)] == 1, values
    marker = next(
        sample for sample in batch.samples if sample.name == "filesystem_unavailable"
    )
    assert marker.labels == {"failure_mode": "READ_ONLY"}, marker
    assert "threshold:filesystem_unavailable" in batch.edge_filter_reasons, (
        batch.edge_filter_reasons
    )
    store = build_store()
    findings = NodeHealthPolicy(store).evaluate_metrics(batch)
    assert [(item.metric_name, item.device) for item in findings] == [
        ("filesystem_unavailable", mount)
    ], findings
    finding = findings[0]
    assert finding.severity is Severity.CRITICAL, finding
    # VALIDATE_HOST-style: a diagnostic, never a reset/reboot or quarantine (004).
    assert finding.recommended_action is RecoveryAction.RUN_DIAGNOSTICS, finding
    assert finding.diagnostic_parameters.get("failure_mode") == "READ_ONLY", finding
    latest = store.list_telemetry_metrics_latest(batch.cluster_id, batch.node_id)
    failures: dict[str, list[str]] = {}
    assert validate_host_metrics(latest, batch.node_id, failures) is None, latest
    assert failures == {batch.node_id: ["filesystem_unavailable"]}, (
        f"VALIDATE_HOST must keep refusing the node while the mount is read-only, "
        f"from the persisted latest value: {failures}"
    )


def _selective_statvfs(bad: str, error: OSError):
    healthy = os.statvfs_result((4096, 4096, 100, 90, 90, 1000, 900, 900, 0, 255))

    def probe(path):
        if path == bad:
            raise error
        return healthy

    return probe


@pytest.mark.parametrize(
    ("error", "failure_mode"),
    [
        (FileNotFoundError(2, "No such file or directory"), "MOUNT_MISSING"),
        (OSError(5, "Input/output error"), "PROBE_FAILED"),
        (PermissionError(13, "Permission denied"), "PROBE_FAILED"),
    ],
    ids=["vanished", "eio", "eacces"],
)
def test_a_configured_mount_whose_statvfs_fails_is_unusable_not_skipped(
    host_case, error, failure_mode
):
    """ENOENT/EIO/EACCES on a configured mount is unusable, never silently skipped.

    A configured mount that vanished (``required_mount_missing``) or whose
    ``statvfs`` raises (``probe_error``) used to fall through the ``except
    OSError`` guard and be dropped from the batch; its unreadable status now
    reads unusable (UNKNOWN), never healthy. The healthy mount alongside it
    still reports available, so the signal has a value to clear to.
    """

    healthy = str(host_case.roots.mount)
    bad = "/mnt/gfrs-test-mount"
    _host_baseline(host_case)
    collector = host_case.build(
        "_cpu",
        "_memory",
        "_filesystems",
        filesystems=[bad, healthy],
        statvfs=_selective_statvfs(bad, error),
    )

    batch = collector.collect_once()

    values = support.values(batch)
    assert values[("filesystem_unavailable", bad)] == 1, values
    assert values[("filesystem_unavailable", healthy)] == 0, values
    assert values[("filesystem_used_percent", healthy)] == 10, values
    marker = next(
        sample
        for sample in batch.samples
        if sample.name == "filesystem_unavailable" and sample.device == bad
    )
    assert marker.labels == {"failure_mode": failure_mode}, marker
    # Unusable, not an unreadable-source collection error that only holds grading.
    assert batch.collection_errors == [], batch.collection_errors
    store = build_store()
    findings = NodeHealthPolicy(store).evaluate_metrics(batch)
    assert [(item.metric_name, item.device) for item in findings] == [
        ("filesystem_unavailable", bad)
    ], findings
    assert findings[0].recommended_action is RecoveryAction.RUN_DIAGNOSTICS, findings
    assert findings[0].diagnostic_parameters.get("failure_mode") == failure_mode
    latest = store.list_telemetry_metrics_latest(batch.cluster_id, batch.node_id)
    failures: dict[str, list[str]] = {}
    assert validate_host_metrics(latest, batch.node_id, failures) is None, latest
    assert failures == {batch.node_id: ["filesystem_unavailable"]}, failures


def test_a_usable_read_write_mount_reports_available_and_is_not_graded(host_case):
    """A usable read-write mount reports available (value 0), never a fault."""

    mount = str(host_case.roots.mount)
    collector = host_case.build(
        "_filesystems",
        statvfs=statvfs(blocks_free=90, files=1000, files_free=900),
    )

    batch = collector.collect_once()

    values = support.values(batch)
    assert values[("filesystem_unavailable", mount)] == 0, values
    assert graded(batch) == [], "a usable read-write mount is not a storage fault"
