"""A host read that failed, or never existed, is missing evidence.

GF-GENERAL-LIVE-004/007/010/012/015: an unreadable source must read UNKNOWN (a
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
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch, NodeHealthPolicy
from gpu_fault.models import RecoveryAction, Severity
from gpu_fault.store.shared.health_signals import (
    finding_health_signal_fingerprint,
    finding_health_signal_key,
)
from scripts.e2e.regional import general_live_bmc as bmc
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
# 010 EFA sysfs counters (the ``_rdma`` reader)
# --------------------------------------------------------------------------- #
RDMA_PORT = "rdmap0/1"


def efa_sysfs_port(case, **counters: int):
    """One EFA port whose ``counters/`` files start at the given values."""

    device = case.roots.rdma / "rdmap0"
    support.write_file(device / "device/uevent", "DRIVER=efa\n")
    port = device / "ports/1"
    support.write_file(port / "state", "4: ACTIVE\n")
    for name, value in counters.items():
        support.write_file(port / "counters" / name, f"{value}\n")
    return port


def rdma_deltas(batch) -> dict[tuple[str, str | None], float]:
    return {
        key: value
        for key, value in support.values(batch).items()
        if key[0].startswith("rdma_") and key[0].endswith("_delta")
    }


def incidents(policy, batches, metric: str) -> list[float]:
    """The values that minted a finding for ``metric`` over a batch sequence.

    Latches every emitted finding the way the ingest's finish half does after
    the incident commits (P0-38B); without that a still-active signal emits on
    every sample and one incident is indistinguishable from a re-minted one.
    """

    minted = []
    for batch in batches:
        for finding in policy.evaluate_metrics(batch):
            if finding.metric_name == metric:
                minted.append(finding.value)
            policy.store.mark_health_signal_notified(
                finding_health_signal_key(finding),
                notified_at=batch.observed_at,
                semantic_fingerprint=finding_health_signal_fingerprint(finding),
            )
    return minted


def test_a_sysfs_rdma_counter_reset_is_unknown_not_a_recovery(host_case):
    """The ``ethtool`` reader's reset rule, applied to the sysfs reader.

    ``_delta`` clamped a counter below its baseline to 0 and ``_rdma`` shipped
    it as ``rdma_errors_delta=0``: a reading that cleared the active signal,
    that VALIDATE_FABRIC accepted as a clean port, and after which the
    continuing increase minted the same fault as a second incident. A reset
    is missing evidence: UNKNOWN for that tick, the baseline re-armed, the
    next increase measured from the new value (GF-GENERAL-LIVE-010
    counter_reset).
    """

    port = efa_sysfs_port(host_case, symbol_error=0)
    collector = host_case.build("_rdma")
    policy = NodeHealthPolicy(build_store())
    readings = []
    for value in (0, 5, 10, 1, 3, 3):
        (port / "counters/symbol_error").write_text(f"{value}\n")
        readings.append(collector.collect_once())
        host_case.clock.sleep(15)

    _, _, ramp, reset, after, quiet = readings
    assert rdma_deltas(ramp) == {
        ("rdma_errors_delta", RDMA_PORT): 5,
        ("rdma_symbol_error_delta", RDMA_PORT): 5,
    }, ramp.samples
    assert rdma_deltas(reset) == {}, (
        f"a reset counter reported a delta: {reset.samples}"
    )
    assert any(
        "_rdma" in error
        and RDMA_PORT in error
        and "symbol_error reset (10 -> 1); delta unknown" in error
        for error in reset.collection_errors
    ), f"the reset tick must name the reader, the port and the counter: {reset}"
    assert rdma_deltas(after) == {
        ("rdma_errors_delta", RDMA_PORT): 2,
        ("rdma_symbol_error_delta", RDMA_PORT): 2,
    }, f"the reading after the reset is measured from the new baseline: {after}"
    assert after.collection_errors == [], after.collection_errors
    assert rdma_deltas(quiet) == {("rdma_errors_delta", RDMA_PORT): 0}, (
        f"a quiet tick after the reset is a real zero reading: {quiet.samples}"
    )
    assert all(sample.value >= 0 for batch in readings for sample in batch.samples), (
        "a counter delta went negative"
    )
    assert incidents(policy, readings, "rdma_errors_delta") == [5], (
        "the reset tick cleared the signal and the next increase re-minted it"
    )


def test_a_sysfs_rdma_counter_total_going_backwards_is_unknown(host_case):
    """A counter the port stopped exposing is not a drop to zero errors.

    The per-port ``rdma_errors_delta`` is the delta of the counter *sum*; a
    counter file that disappears lowers the sum without any counter reading
    below its own baseline, and the clamped 0 shipped as a clean port.
    """

    port = efa_sysfs_port(host_case, symbol_error=4, link_downed=6)
    collector = host_case.build("_rdma")
    collector.collect_once()
    host_case.clock.sleep(15)
    (port / "counters/link_downed").unlink()

    batch = collector.collect_once()

    assert rdma_deltas(batch) == {}, batch.samples
    assert any(
        RDMA_PORT in error and "reset (10 -> 4); delta unknown" in error
        for error in batch.collection_errors
    ), batch.collection_errors


# --------------------------------------------------------------------------- #
# 009 netdev statistics counters (the ``_network`` reader)
# --------------------------------------------------------------------------- #
def physical_netdev(case, name: str, **counters: int):
    interface = case.roots.net / name
    (interface / "device").mkdir(parents=True)
    support.write_file(interface / "operstate", "up\n")
    for counter in ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped"):
        support.write_file(
            interface / "statistics" / counter, f"{counters.get(counter, 0)}\n"
        )
    return interface


def netdev_deltas(batch) -> dict[tuple[str, str | None], float]:
    return {
        key: value
        for key, value in support.values(batch).items()
        if key[0] in {"network_errors_delta", "network_drops_delta"}
    }


def test_a_netdev_statistics_counter_reset_is_unknown_not_a_recovery(host_case):
    """Same clamp, same consumer: ``network_errors_delta`` is graded at 1,
    kept as a latest value for VALIDATE_FABRIC and clears on a 0 reading.

    The reset is per counter: the drop counters did not reset, so their delta
    is still a reading on the reset tick, as is the link state
    (GF-GENERAL-LIVE-009 counter_reset).
    """

    interface = physical_netdev(host_case, "eth0", rx_dropped=7)
    collector = host_case.build("_network")
    policy = NodeHealthPolicy(build_store())
    readings = []
    for value in (0, 5, 10, 1, 3, 3):
        (interface / "statistics/rx_errors").write_text(f"{value}\n")
        readings.append(collector.collect_once())
        host_case.clock.sleep(15)

    _, _, ramp, reset, after, quiet = readings
    assert netdev_deltas(ramp) == {
        ("network_errors_delta", "eth0"): 5,
        ("network_drops_delta", "eth0"): 0,
    }, ramp.samples
    assert netdev_deltas(reset) == {("network_drops_delta", "eth0"): 0}, (
        f"a reset counter reported a delta, or an unrelated one lost its: "
        f"{reset.samples}"
    )
    assert support.values(reset)[("network_link_up", "eth0")] == 1, reset.samples
    assert any(
        "_network" in error
        and "eth0" in error
        and "rx_errors reset (10 -> 1); delta unknown" in error
        for error in reset.collection_errors
    ), f"the reset tick must name the reader, the netdev and the counter: {reset}"
    assert netdev_deltas(after) == {
        ("network_errors_delta", "eth0"): 2,
        ("network_drops_delta", "eth0"): 0,
    }, f"the reading after the reset is measured from the new baseline: {after}"
    assert after.collection_errors == [], after.collection_errors
    assert netdev_deltas(quiet) == {
        ("network_errors_delta", "eth0"): 0,
        ("network_drops_delta", "eth0"): 0,
    }, f"a quiet tick after the reset is a real zero reading: {quiet.samples}"
    assert all(sample.value >= 0 for batch in readings for sample in batch.samples), (
        "a counter delta went negative"
    )
    assert incidents(policy, readings, "network_errors_delta") == [5], (
        "the reset tick cleared the signal and the next increase re-minted it"
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


def listing(count: int) -> tuple[str, str, int]:
    return "".join(f"GPU-{index}, 10\n" for index in range(count)), "", 0


def bound_gpu_collector(case, *, expected: int, bound: int, instance_type: str):
    """A host collector on a node whose driver has ``bound`` GPUs attached."""

    gpus = case.roots.proc / "driver/nvidia/gpus"
    for index in range(bound):
        (gpus / f"0000:0{index}:00.0").mkdir(parents=True)
    return case.build(
        "_gpu_inventory",
        expected_gpu_count=expected,
        inventory_mismatch_consecutive_samples=2,
        node_instance_type=instance_type,
    )


def graded_ticks(case, collector, ticks: int = 3):
    """The findings of ``ticks`` collector rounds graded by one policy over one
    store, plus the ``gpu_inventory_mismatch`` sample of every round.

    Every emitted finding is latched as notified, as the ingestion that carries
    it into an incident does; an un-notified active signal re-emits by design.
    """

    store = build_store()
    policy = NodeHealthPolicy(store)
    findings, mismatches = [], []
    for _ in range(ticks):
        batch = collector.collect_once()
        mismatches.append(
            next(
                item for item in batch.samples if item.name == "gpu_inventory_mismatch"
            )
        )
        emitted = policy.evaluate_metrics(batch)
        for finding in emitted:
            store.mark_health_signal_notified(
                finding_health_signal_key(finding),
                notified_at=batch.observed_at,
                semantic_fingerprint=finding_health_signal_fingerprint(finding),
            )
        findings.extend(emitted)
        case.clock.sleep(15)
    return findings, mismatches


@pytest.mark.parametrize("explicit", [9, 7], ids=["above-the-type", "below-the-type"])
def test_an_expectation_the_instance_type_and_the_node_both_contradict_never_reboots(
    host_case, explicit
):
    """GF-GENERAL-LIVE-012 ``expectation_wrong``: a mistyped
    ``GPU_FAULT_EXPECTED_GPU_COUNT`` on a p5.48xlarge that lists its eight GPUs,
    eight bound, used to be graded as a lost GPU -- CRITICAL REBOOT_NODE on the
    second tick. The configured count is the one reading out of three that
    disagrees: a configuration finding (WARNING, evidence only), never a reboot.
    """

    collector = bound_gpu_collector(
        host_case, expected=explicit, bound=8, instance_type="ml.p5.48xlarge"
    )
    host_case.outputs[tuple(collector.GPU_QUERY_ARGV)] = listing(8)

    findings, mismatches = graded_ticks(host_case, collector)

    for sample in mismatches:
        assert sample.labels["instance_type_gpu_count"] == "8", sample.labels
        assert sample.labels["discovered_count"] == "8", sample.labels
        assert sample.labels["expected_count"] == str(explicit), sample.labels
    assert [item.recommended_action for item in findings] == [
        RecoveryAction.COLLECT_EVIDENCE
    ], f"a configuration typo was graded as hardware: {findings}"
    (finding,) = findings
    assert finding.metric_name == "gpu_expected_count_contradicted"
    assert finding.severity is Severity.WARNING
    for named in (str(explicit), "8", "ml.p5.48xlarge"):
        assert named in finding.reason, finding.reason
    assert finding.diagnostic_parameters["observed_count"] == "8"
    assert finding.diagnostic_parameters["instance_type_gpu_count"] == "8"


@pytest.mark.parametrize(
    ("listed", "bound"),
    [(7, 7), (8, 7)],
    ids=["short-of-the-type", "driver-bound-disagrees"],
)
def test_a_real_shortfall_under_a_wrong_expectation_still_reboots(
    host_case, listed, bound
):
    """Conservative: a node short of its instance type's own GPU count, or whose
    driver-bound count disagrees with the listing, grades exactly as before
    (CRITICAL REBOOT_NODE on the second valid sample) however wrong the
    configured count also is."""

    collector = bound_gpu_collector(
        host_case, expected=9, bound=bound, instance_type="ml.p5.48xlarge"
    )
    host_case.outputs[tuple(collector.GPU_QUERY_ARGV)] = listing(listed)

    findings, mismatches = graded_ticks(host_case, collector, ticks=2)

    assert [item.value for item in mismatches] == [0, 1], "two ticks confirm"
    reboots = [
        item
        for item in findings
        if item.recommended_action is RecoveryAction.REBOOT_NODE
    ]
    assert len(reboots) == 1, f"the real shortfall was not graded: {findings}"
    assert reboots[0].metric_name == "gpu_inventory_mismatch"
    assert reboots[0].severity is Severity.CRITICAL
    assert reboots[0].diagnostic_parameters["instance_type_gpu_count"] == "8"


def test_an_instance_type_the_table_does_not_know_keeps_the_explicit_expectation(
    host_case,
):
    """Without platform knowledge the explicit setting keeps its documented
    meaning: nine expected, eight listed is a mismatch and reboots as before."""

    collector = bound_gpu_collector(
        host_case, expected=9, bound=8, instance_type="ml.p5test.48xlarge"
    )
    host_case.outputs[tuple(collector.GPU_QUERY_ARGV)] = listing(8)

    findings, mismatches = graded_ticks(host_case, collector, ticks=2)

    assert "instance_type_gpu_count" not in mismatches[-1].labels, mismatches[-1]
    assert [(item.metric_name, item.recommended_action) for item in findings] == [
        ("gpu_inventory_mismatch", RecoveryAction.REBOOT_NODE)
    ], findings


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


def _read_only_on_the_host(host_case, mount: str) -> None:
    """``mount`` remounted read-only (superblock ``ro``), as both tables show it.

    The collector's own table adds its ``ProtectSystem=strict`` read-only bind;
    the read-only verdict is taken from the host's (PID 1's) table.
    """

    support.write_file(
        host_case.roots.proc / "self/mountinfo",
        "2177 76 259:1 / / ro,nosuid,relatime master:1 - ext4 /dev/root rw\n"
        f"2260 2177 7:15 / {mount} ro,nosuid,relatime master:590 - ext4 /dev/loop15 ro\n",
    )
    support.write_file(
        host_case.roots.proc / "1/mountinfo",
        "28 1 259:1 / / rw,relatime shared:1 - ext4 /dev/root rw\n"
        f"590 28 7:15 / {mount} rw,relatime shared:590 - ext4 /dev/loop15 ro\n",
    )


def test_a_read_only_configured_mount_is_unusable_with_free_capacity(host_case):
    """Read-only is unusable even when the capacity reading is healthy (004 read_only).

    The spec forbids judging such a mount by its remaining bytes: a remount
    read-only keeps its blocks and inodes, so the capacity signals read fine
    while the mount refuses every write.
    """

    mount = str(host_case.roots.mount)
    _host_baseline(host_case)
    _read_only_on_the_host(host_case, mount)
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
        "_filesystems", statvfs=statvfs(blocks_free=90, files=1000, files_free=900)
    )

    batch = collector.collect_once()

    values = support.values(batch)
    assert values[("filesystem_unavailable", mount)] == 0, values
    assert graded(batch) == [], "a usable read-write mount is not a storage fault"


# --------------------------------------------------------------------------- #
# 015 BMC
# --------------------------------------------------------------------------- #
SENSOR = ("ipmitool", "sensor")
CPU1_ALERT = "CPU1 Temp=96.000 degrees C [cr; unc 90.000, ucr 95.000, unr 100.000]"
FAN1_ALERT = "Fan1=0.000 RPM [cr; lcr 300.000, lnc 500.000]"


def bmc_markers(batch) -> list[tuple[str, str | None, float, dict[str, str]]]:
    return [
        (sample.name, sample.device, sample.value, sample.labels)
        for sample in batch.samples
        if sample.name.startswith("bmc_")
    ]


def bmc_collector(host_case, answer, **kwargs):
    support.enable_tools(host_case, "ipmitool")
    host_case.outputs[SENSOR] = answer
    return host_case.build("_bmc", **kwargs)


def sensor_queries(host_case) -> int:
    return sum(1 for argv, _ in host_case.commands if tuple(argv) == SENSOR)


def test_critical_sensors_are_named_with_reading_status_and_thresholds(host_case):
    """Two ``cr`` rows: the count sample carries both sensors' identity.

    ``bmc_critical_sensor`` used to be a bare count (unit ``sensors``, no
    labels), so the names, readings and thresholds of the alerting rows never
    reached the finding, the incident reasons, the marker or the hardware
    escalation mail: with two critical sensors the operator saw "2". The
    count stays (one finding, one QUARANTINE), and the labels name the
    sensors in table order with the status, reading, units and the threshold
    columns ipmitool printed (GF-GENERAL-LIVE-015 ``critical``).
    """

    collector = bmc_collector(host_case, (bmc.CRITICAL_TABLE, "", 0))

    batch = collector.collect_once()

    assert bmc_markers(batch) == [
        (
            "bmc_critical_sensor",
            None,
            2,
            {
                "sensors": "CPU1 Temp, Fan1",
                "sensor_alerts": f"{CPU1_ALERT}; {FAN1_ALERT}",
            },
        )
    ], batch.samples
    assert batch.samples[0].unit == "sensors", "the count keeps its unit"
    assert batch.collection_errors == [], batch.collection_errors
    findings = graded(batch)
    assert [
        (item.metric_name, item.device, item.value, item.recommended_action)
        for item in findings
    ] == [("bmc_critical_sensor", None, 2, RecoveryAction.QUARANTINE)], findings
    assert findings[0].reason == (
        f"BMC reports 2 critical sensors: {CPU1_ALERT}; {FAN1_ALERT}"
    ), "the reason is what the incident, the marker and the escalation mail keep"
    assert findings[0].diagnostic_parameters == {
        "sensors": "CPU1 Temp, Fan1",
        "sensor_alerts": f"{CPU1_ALERT}; {FAN1_ALERT}",
    }, findings[0].diagnostic_parameters
    assert findings[0].marker().raw_reason == findings[0].reason, (
        "the marker names the sensors through the finding's reason"
    )


def test_a_non_recoverable_sensor_is_named_with_its_status(host_case):
    collector = bmc_collector(host_case, (bmc.NONRECOVERABLE_TABLE, "", 0))

    batch = collector.collect_once()

    alert = "CPU1 Temp=104.000 degrees C [nr; unc 90.000, ucr 95.000, unr 100.000]"
    assert bmc_markers(batch) == [
        (
            "bmc_critical_sensor",
            None,
            1,
            {"sensors": "CPU1 Temp", "sensor_alerts": alert},
        )
    ], batch.samples
    findings = graded(batch)
    assert [item.reason for item in findings] == [
        f"BMC reports 1 critical sensor: {alert}"
    ], findings


def test_a_sensor_listed_twice_is_one_critical_sensor(host_case):
    """A BMC that repeats an SDR record names the sensor once (spec 015: 去重)."""

    table = bmc.CRITICAL_TABLE + bmc.CRITICAL_TABLE.splitlines()[1] + "\n"
    collector = bmc_collector(host_case, (table, "", 0))

    batch = collector.collect_once()

    assert support.values(batch) == {("bmc_critical_sensor", None): 2}, batch.samples
    assert batch.samples[0].labels["sensors"] == "CPU1 Temp, Fan1", batch.samples


@pytest.mark.parametrize(
    "table",
    [bmc.HEALTHY_TABLE, bmc.WRONG_COLUMN_TABLE, bmc.UNAVAILABLE_TABLE],
    ids=["healthy", "wrong-column", "one-unavailable-sensor"],
)
def test_a_table_without_an_alert_is_a_plain_zero(host_case, table):
    """Status words outside the 4th column, a 3-column row and one ``na``
    sensor among readings are no alert: value 0, no labels, no error, no
    finding -- the sample an older control plane already understands."""

    collector = bmc_collector(host_case, (table, "", 0))

    batch = collector.collect_once()

    assert bmc_markers(batch) == [("bmc_critical_sensor", None, 0, {})], batch.samples
    assert batch.collection_errors == [], batch.collection_errors
    assert graded(batch) == [], "a table with no cr/nr status is not a BMC fault"


def test_a_table_of_only_unavailable_sensors_is_unknown_not_zero(host_case):
    """Every status ``na`` is a BMC that answered with no reading at all.

    It used to parse as ``bmc_critical_sensor=0`` -- an all-unavailable BMC
    read as healthy. It is missing evidence: a ``_bmc`` collection error, no
    sample, nothing graded.
    """

    rows = [
        tuple("na" if index not in (0, 2) else value for index, value in enumerate(row))
        for row in (
            ("Inlet Temp", "na", "degrees C", "na", "na", "na", "na", "na", "na", "na"),
            ("Fan1", "na", "RPM", "na", "na", "na", "na", "na", "na", "na"),
        )
    ]
    collector = bmc_collector(host_case, (bmc.sensor_table(rows), "", 0))

    for _ in range(2):
        batch = collector.collect_once()
        assert bmc_markers(batch) == [], batch.samples
        assert any(
            error.startswith("_bmc:") and "no sensor reading" in error
            for error in batch.collection_errors
        ), f"an all-na table must invalidate the observation: {batch.collection_errors}"
        assert graded(batch) == [], "an unreadable sensor table is not a BMC fault"
        host_case.clock.sleep(15)


def test_bmc_without_ipmitool_is_reported_not_supported(host_case):
    """No ``ipmitool``: a capability gap, said so on every tick, never an error."""

    collector = host_case.build("_bmc")

    for _ in range(2):
        batch = collector.collect_once()
        assert bmc_markers(batch) == [
            ("bmc_status_unsupported", None, 1, {"reason": "TOOL_MISSING"})
        ], batch.samples
        assert batch.collection_errors == [], batch.collection_errors
        assert graded(batch) == [], "a missing ipmitool is not a BMC fault"
        host_case.clock.sleep(15)
    assert sensor_queries(host_case) == 0, host_case.commands


def test_a_host_without_an_ipmi_device_is_not_supported_not_an_error(host_case):
    """``ipmitool`` installed, no BMC behind it (EC2): NOT_SUPPORTED.

    ipmitool exits 1 with ``Could not open device at /dev/ipmi0 or ... : No
    such file or directory`` on every tick, and that used to be a ``_bmc``
    collection error every tick -- a permanent capability gap graded as
    missing evidence, which held every host validation of the node forever.
    Like SMART's ``NO_DEVICES`` it is ``bmc_status_unsupported`` NO_DEVICE
    with no error. The query is still made every tick: the gap is read from
    the tool's answer, never assumed.
    """

    collector = bmc_collector(
        host_case,
        ("", bmc.IPMI_NO_DEVICE_STDERR, 1),
        ipmi_device_nodes=[str(host_case.tmp_path / "dev" / "ipmi0")],
    )

    for _ in range(2):
        batch = collector.collect_once()
        assert bmc_markers(batch) == [
            ("bmc_status_unsupported", None, 1, {"reason": "NO_DEVICE"})
        ], batch.samples
        assert batch.collection_errors == [], batch.collection_errors
        assert graded(batch) == [], "a BMC-less host is not a BMC fault"
        host_case.clock.sleep(15)
    assert sensor_queries(host_case) == 2, host_case.commands


@pytest.mark.parametrize(
    "answer",
    [
        ("", bmc.IPMI_TRANSIENT_STDERR, 1),
        ("", "", 1),
        subprocess.TimeoutExpired(list(SENSOR), 20),
        OSError(5, "Input/output error"),
    ],
    ids=["bmc-not-answering", "silent-exit-status", "timeout", "exec-error"],
)
def test_a_failed_sensor_query_is_unknown_not_zero_and_not_unsupported(
    host_case, answer
):
    """A BMC that is there but does not answer is UNKNOWN, as before."""

    collector = bmc_collector(
        host_case, answer, ipmi_device_nodes=[str(host_case.tmp_path / "dev" / "ipmi0")]
    )

    batch = collector.collect_once()

    assert bmc_markers(batch) == [], batch.samples
    assert any(
        error.startswith("_bmc:") and bmc.BMC_QUERY_FAILED in error
        for error in batch.collection_errors
    ), f"a failed query must invalidate the observation: {batch.collection_errors}"
    assert graded(batch) == [], "a failed sensor query is not a BMC fault"


def test_the_no_device_answer_with_a_device_node_present_stays_unknown(host_case):
    """The NOT_SUPPORTED verdict needs both facts: ipmitool's ENOENT text and
    no IPMI device node on the host. A device node that exists while the tool
    cannot open it is not a BMC-less host; it is a failure to read one."""

    node = host_case.tmp_path / "dev" / "ipmi0"
    node.parent.mkdir(parents=True)
    node.write_text("")
    collector = bmc_collector(
        host_case, ("", bmc.IPMI_NO_DEVICE_STDERR, 1), ipmi_device_nodes=[str(node)]
    )

    batch = collector.collect_once()

    assert bmc_markers(batch) == [], batch.samples
    assert any(bmc.BMC_QUERY_FAILED in error for error in batch.collection_errors), (
        batch.collection_errors
    )


def test_a_bare_count_from_an_older_collector_keeps_the_fixed_reason():
    """Wire compatibility: a not-yet-upgraded node posts the unlabelled count."""

    batch = HostTelemetryBatch(
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=common.Clock().now(),
        samples=[HostMetricSample(name="bmc_critical_sensor", value=2, unit="sensors")],
    )

    findings = graded(batch)

    assert [(item.reason, item.recommended_action) for item in findings] == [
        ("BMC reports a critical sensor", RecoveryAction.QUARANTINE)
    ], findings
    assert findings[0].diagnostic_parameters == {}, findings[0].diagnostic_parameters
