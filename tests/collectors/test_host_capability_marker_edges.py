"""A capability (NOT_SUPPORTED) marker that appears, changes or clears is an edge.

GF-GENERAL-LIVE-015 on plain EKS, 2026-10-26: ``command_failure`` replayed a
BMC that does not answer (a ``_bmc`` collection error) and then an ``ipmitool``
that finds no IPMI device (``bmc_status_unsupported`` NO_DEVICE); the marker
rode the tick that closed the error and every tick after it, and the read that
started after the replan saw none of them. ``unsupported_bmc`` read a node
without ``ipmitool`` for one health summary period and saw zero batches: the
steady TOOL_MISSING marker was never an edge and the summary that carried it
is not evidence. The edge filter knew only the EFA ethtool marker (b184).

Every case drives the real ``_bmc``/``_smart``/``_efa_network`` contributor
through ``collect_once`` with the edge filter on and reads what the sink was
given, then grades the delivered batches with the real ``NodeHealthPolicy``:
NOT_SUPPORTED never mints a finding, and the marker is kept as the node's
latest record (``host_capability_markers``).
"""

from __future__ import annotations

import json

from gpu_fault.collectors.host import bmc as bmc_module
from gpu_fault.host_capability_markers import (
    CAPABILITY_MARKERS,
    NOT_SUPPORTED_REASON_PREFIX,
    is_capability_marker,
    not_supported_edge_reason,
)
from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from scripts.e2e.regional import general_live_bmc as live_bmc
from tests._builders import build_store
from tests.collectors import _cov95_runtime_collect as common
from tests.collectors import _cov95_runtime_collect_host as support

isolated_runtime = common.isolated_runtime
host_case = support.host_case

SENSOR = ("ipmitool", "sensor")
SCAN = ("smartctl", "--scan-open")
NO_DEVICE_EXIT = ("", live_bmc.IPMI_NO_DEVICE_STDERR, 1)
TRANSIENT_EXIT = ("", live_bmc.IPMI_TRANSIENT_STDERR, 1)
HEALTHY_TABLE = (live_bmc.HEALTHY_TABLE, "", 0)
BMC_NO_DEVICE = f"{NOT_SUPPORTED_REASON_PREFIX}bmc_status_unsupported:node:NO_DEVICE"
SMART_NO_DEVICES = (
    f"{NOT_SUPPORTED_REASON_PREFIX}smart_status_unsupported:node:NO_DEVICES"
)
SMART_NO_STATUS = (
    f"{NOT_SUPPORTED_REASON_PREFIX}smart_status_unsupported:/dev/sda:NO_SMART_STATUS"
)


def ipmitool_installed(case, installed: bool) -> None:
    """Whether ``shutil.which("ipmitool")`` finds the tool; no IPMI node exists."""

    case.monkeypatch.setattr(
        bmc_module.shutil,
        "which",
        lambda name: "/usr/bin/ipmitool" if installed else None,
    )
    case.monkeypatch.setattr(bmc_module.os.path, "exists", lambda path: False)


class Node:
    """One collector ticking every 15 s; what each tick delivered, if anything."""

    def __init__(self, case, collector) -> None:
        self.case = case
        self.collector = collector
        self.policy = NodeHealthPolicy(build_store())
        self.batches = []

    def tick(self):
        batch = self.collector.collect_once()
        self.case.clock.sleep(15)
        self.batches.append(batch)
        return batch

    def delivered(self, batch) -> list[str] | None:
        """The delivery reasons of ``batch``, ``None`` when it never left the node."""

        for _, payload in self.case.sink.requests:
            if payload["batch_id"] == batch.batch_id:
                return list(payload["edge_filter_reasons"])
        return None

    def deliveries(self) -> list[tuple[int, list[str]]]:
        """``(tick number, reasons)`` of every delivered tick, in order."""

        return [
            (number, reasons)
            for number, batch in enumerate(self.batches, start=1)
            if (reasons := self.delivered(batch)) is not None
        ]

    def grade(self, batch):
        """What the control plane does with a delivered batch."""

        return self.policy.evaluate_metrics(batch)

    def grade_delivered(self) -> list:
        """Grade every delivered tick in order, as the control plane saw them;
        an undelivered tick never reaches it. The findings, if any."""

        findings = []
        for batch in self.batches:
            if self.delivered(batch) is not None:
                findings.extend(self.grade(batch))
        return findings

    def latest_marker(self, name: str) -> list[tuple[str | None, str, str]]:
        """``(device, reason, observed_at)`` of the node's latest ``name`` records."""

        return sorted(
            (
                (
                    item.device,
                    item.labels.get("reason", ""),
                    item.observed_at.isoformat(),
                )
                for item in self.policy.store.list_telemetry_metrics_latest(
                    "cluster-a", "node-a"
                )
                if item.name == name
            ),
            key=lambda row: row[0] or "",
        )


def markers(batch) -> list[tuple[str, str | None, str]]:
    return [
        (sample.name, sample.device, sample.labels.get("reason", ""))
        for sample in batch.samples
        if is_capability_marker(sample)
    ]


def bmc_node(host_case, *, installed: bool) -> Node:
    ipmitool_installed(host_case, installed)
    collector = host_case.build(
        "_bmc",
        edge_filter_enabled=True,
        health_summary_seconds=3600,
        ipmi_device_nodes=("/dev/ipmi0", "/dev/ipmi/0", "/dev/ipmidev/0"),
    )
    return Node(host_case, collector)


# --------------------------------------------------------------------------- #
# the BMC marker: appearance, reason change, clearing
# --------------------------------------------------------------------------- #
def test_a_bmc_marker_that_appears_changes_or_clears_is_a_delivered_edge(host_case):
    """TOOL_MISSING -> NO_DEVICE -> a BMC that answers -> NO_DEVICE again.

    The reason change (ipmitool arrives on a host without a BMC), the marker
    clearing (a BMC answers) and the marker reappearing are each delivered
    once; the steady marker is not. None of it mints a finding. Before: only
    the baseline left the node -- the marker had no reason of its own.
    """

    node = bmc_node(host_case, installed=False)
    node.tick()  # 1 baseline: TOOL_MISSING is the node's state
    node.tick()  # 2 steady
    ipmitool_installed(host_case, True)
    host_case.outputs[SENSOR] = NO_DEVICE_EXIT
    changed = node.tick()  # 3 TOOL_MISSING -> NO_DEVICE
    node.tick()  # 4 steady
    host_case.outputs[SENSOR] = HEALTHY_TABLE
    cleared = node.tick()  # 5 the marker clears: a BMC answers
    node.tick()  # 6 steady
    host_case.outputs[SENSOR] = NO_DEVICE_EXIT
    appeared = node.tick()  # 7 the marker appears again

    assert markers(node.batches[0]) == [
        ("bmc_status_unsupported", None, "TOOL_MISSING")
    ]
    assert markers(changed) == [("bmc_status_unsupported", None, "NO_DEVICE")]
    assert markers(cleared) == [] and support.values(cleared) == {
        ("bmc_critical_sensor", None): 0.0
    }, cleared.samples
    assert markers(appeared) == [("bmc_status_unsupported", None, "NO_DEVICE")]
    assert all(batch.collection_errors == [] for batch in node.batches), (
        "a capability gap is never a collection error"
    )
    assert node.deliveries() == [
        (1, ["baseline"]),
        (3, [BMC_NO_DEVICE, "recovered"]),
        (5, ["recovered"]),
        (7, [BMC_NO_DEVICE]),
    ], [payload["edge_filter_reasons"] for _, payload in host_case.sink.requests]
    assert node.grade_delivered() == [], "a NOT_SUPPORTED marker minted a finding"


def test_the_live_015_command_failure_timeline_delivers_the_marker_and_keeps_it_readable(
    host_case,
):
    """Steady NO_DEVICE, then the replayed BMC failure, then NO_DEVICE again.

    The tick that closes the collection error carries the marker and is
    delivered for both edges, the marker's own and ``recovered``; grading
    the delivered batches as the control plane does keeps the node's latest
    ``bmc_status_unsupported`` record with its reason, so the capability state
    is readable from the store while the steady ticks stay undelivered. Live
    (eks, 2026-10-26 04:17Z) that one tick was the only record and the read
    that started after the replan missed it; nothing else ever said NO_DEVICE.
    """

    node = bmc_node(host_case, installed=True)
    host_case.outputs[SENSOR] = NO_DEVICE_EXIT
    steady = node.tick()  # 1 baseline: the node has ipmitool and no BMC
    node.tick()  # 2 steady
    host_case.outputs[SENSOR] = TRANSIENT_EXIT
    failed = node.tick()  # 3 the replayed "Get Device ID command failed"
    node.tick()  # 4 steady error
    host_case.outputs[SENSOR] = NO_DEVICE_EXIT
    back = node.tick()  # 5 the replayed "Could not open device": NO_DEVICE
    node.tick()  # 6 steady

    assert failed.collection_errors and markers(failed) == [], failed
    assert markers(back) == [("bmc_status_unsupported", None, "NO_DEVICE")]
    assert back.collection_errors == [], back.collection_errors
    # Tick 3: the error opens, and the marker that was the node's state ends
    # (``recovered``); tick 5: the marker is a new reason and the error ends.
    assert node.deliveries() == [
        (1, ["baseline"]),
        (3, ["collection-error:_bmc", "recovered"]),
        (5, [BMC_NO_DEVICE, "recovered"]),
    ], [payload["edge_filter_reasons"] for _, payload in host_case.sink.requests]

    assert node.grade(steady) == []
    assert node.latest_marker("bmc_status_unsupported") == [
        (None, "NO_DEVICE", steady.observed_at.isoformat())
    ], "the baseline marker is the node's readable capability state"
    assert node.grade(failed) == []
    assert node.latest_marker("bmc_status_unsupported") == [
        (None, "NO_DEVICE", steady.observed_at.isoformat())
    ], "an error batch carries no marker and leaves the last record as it was"
    assert node.grade(back) == []
    assert node.latest_marker("bmc_status_unsupported") == [
        (None, "NO_DEVICE", back.observed_at.isoformat())
    ], "the delivered marker moves the latest record to its tick"


# --------------------------------------------------------------------------- #
# the SMART marker: reason change at the node, a marker per drive
# --------------------------------------------------------------------------- #
def test_a_smart_marker_reason_change_and_a_per_drive_marker_are_edges(host_case):
    """TOOL_MISSING -> NO_DEVICES -> a drive without a SMART status.

    The node-level reason change and the per-drive marker that replaces the
    node-level one are each delivered once, keyed by the device they name.
    """

    collector = host_case.build(
        "_smart", edge_filter_enabled=True, health_summary_seconds=3600
    )
    node = Node(host_case, collector)
    node.tick()  # 1 baseline: no smartctl
    node.tick()  # 2 steady
    support.enable_tools(host_case, "smartctl")
    host_case.outputs[SCAN] = ("", "", 0)
    no_drives = node.tick()  # 3 TOOL_MISSING -> NO_DEVICES
    node.tick()  # 4 steady
    host_case.outputs[SCAN] = ("/dev/sda -d scsi # /dev/sda, SCSI device\n", "", 0)
    host_case.outputs[("smartctl", "-H", "-j", "/dev/sda")] = (
        json.dumps({"smartctl": {"exit_status": 0}}),
        "",
        0,
    )
    per_drive = node.tick()  # 5 a drive appears, without a SMART status
    node.tick()  # 6 steady (the verdict-less drive is cached)

    assert markers(no_drives) == [("smart_status_unsupported", None, "NO_DEVICES")]
    assert markers(per_drive) == [
        ("smart_status_unsupported", "/dev/sda", "NO_SMART_STATUS")
    ]
    assert node.deliveries() == [
        (1, ["baseline"]),
        (3, [SMART_NO_DEVICES, "recovered"]),
        (5, [SMART_NO_STATUS, "recovered"]),
    ], [payload["edge_filter_reasons"] for _, payload in host_case.sink.requests]
    assert all(batch.collection_errors == [] for batch in node.batches), (
        "a capability gap is never a collection error"
    )
    assert node.grade_delivered() == [], "a NOT_SUPPORTED marker minted a finding"
    assert node.latest_marker("smart_status_unsupported") == [
        (None, "NO_DEVICES", no_drives.observed_at.isoformat()),
        ("/dev/sda", "NO_SMART_STATUS", per_drive.observed_at.isoformat()),
    ], "one latest record per device; the cleared node-level one keeps its time"


# --------------------------------------------------------------------------- #
# the registry: every marker the host collector posts is one, and is kept
# --------------------------------------------------------------------------- #
def efa_netdev(case, interface: str) -> None:
    device = case.roots.rdma / "rdmap0"
    support.write_file(device / "device/uevent", "DRIVER=efa\n")
    (device / "device/net" / interface).mkdir(parents=True)
    support.enable_tools(case, "ethtool")


def test_every_marker_the_host_collector_posts_is_registered_and_kept_latest(host_case):
    """BMC, SMART and EFA ethtool markers: the same edge rule, the same records.

    A ``*_unsupported`` sample a contributor posts that the registry does not
    know would be a state without an edge and without a record again; the EFA
    reason keeps the form the EFA-only rule produced (b184).
    """

    efa_netdev(host_case, "eth0")
    host_case.outputs[("ethtool", "-S", "eth0")] = (
        "NIC statistics:\n     rnr_naks: 0\n",
        "",
        0,
    )
    ipmitool_installed(host_case, True)
    host_case.outputs[SENSOR] = NO_DEVICE_EXIT
    collector = host_case.build("_bmc", "_smart", "_efa_network")
    batch = collector.collect_once()

    posted = [
        sample for sample in batch.samples if sample.name.endswith("_unsupported")
    ]
    assert sorted(sample.name for sample in posted) == sorted(CAPABILITY_MARKERS), (
        posted
    )
    for sample in posted:
        assert is_capability_marker(sample), sample
        assert sample.labels.get("reason"), f"a marker without a reason: {sample}"
        assert sample.name in NodeHealthPolicy.LATEST_METRICS_REQUIRED, sample.name
        reason = not_supported_edge_reason(sample)
        assert reason is not None and reason.startswith(NOT_SUPPORTED_REASON_PREFIX)
    efa = next(sample for sample in posted if sample.name.startswith("efa_"))
    assert not_supported_edge_reason(efa) == (
        "not-supported:efa_ethtool_counters_unsupported:eth0:MISSING_COUNTER:"
        + efa.labels["metrics"]
    )
    assert (
        not_supported_edge_reason(
            HostMetricSample(name="bmc_critical_sensor", value=0.0, unit="sensors")
        )
        is None
    ), "a reading contributes no NOT_SUPPORTED reason"
    assert NodeHealthPolicy(build_store()).evaluate_metrics(batch) == []
