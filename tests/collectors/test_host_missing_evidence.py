"""A host read that failed, or never existed, is missing evidence.

GF-GENERAL-LIVE-007: an unreadable source must read UNKNOWN (a
collection error, so validation waits and nothing is graded), a source the
node cannot provide must say NOT_SUPPORTED, and neither may be reported as a
healthy reading or as a zero. Every case drives the real contributor through
``collect_once`` and grades the batch with the real ``NodeHealthPolicy``, so
"never a fault" is checked where a fault would be minted.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from gpu_fault.host_health import NodeHealthPolicy
from gpu_fault.models import RecoveryAction
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
