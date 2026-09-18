"""Host filesystem/device failures observed as telemetry, never real probes."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from gpu_fault.collectors.host import collector as host_module
from gpu_fault.collectors.host.system_metrics import BoundedStatvfs
from gpu_fault.collectors.sinks import CollectorError
from tests.collectors import _cov95_runtime_collect as common
from tests.collectors import _cov95_runtime_collect_host as support

isolated_runtime = common.isolated_runtime
host_case = support.host_case


def test_lustre_counter_scan_ignores_noise_and_preserves_reset_semantics(host_case):
    path = host_case.roots.proc / "fs/lustre/llite/client-a/stats"
    support.write_file(
        path,
        "\nunrelated 50\nread_bytes samples [bytes] 4\nwrite_bytes junk\n"
        "dirty_pages_hits events 2\ndirty_pages_misses events 5\n",
    )
    collector = host_case.build("_lustre")
    assert collector.collect_once().samples == []
    host_case.clock.sleep(2)
    path.write_text(
        "read_bytes 5 samples [bytes] 1 2 14\nwrite_bytes junk\n"
        "dirty_pages_hits 7\ndirty_pages_misses 3\n"
    )

    batch = collector.collect_once()

    assert batch.collection_errors == []
    assert support.values(batch) == {
        ("lustre_read_bytes_delta", "client-a"): 10,
        ("lustre_dirty_page_hits_delta", "client-a"): 5,
        ("lustre_dirty_page_misses_delta", "client-a"): 0,
    }
    assert {sample.unit for sample in batch.samples} == {"bytes", "events"}


@pytest.mark.parametrize(
    ("mountinfo", "mount", "local"),
    [
        ("", "/data", True),
        ("bad line\n1 2 3 / /data rw -\n", "/data", True),
        ("1 2 3 / /elsewhere rw - nfs4 remote rw\n", "/data", True),
        ("1 2 3 / /data rw - nfs4 remote rw\n", "/data", False),
        ("1 2 3 / /data\\040one rw - xfs disk rw\n", "/data one", True),
        (
            "1 2 3 / / rw - xfs disk rw\n1 2 3 / /data rw - lustre remote rw\n",
            "/data/sub",
            False,
        ),
    ],
)
def test_filesystem_classification_uses_matching_mount_not_unrelated_entries(
    host_case, mountinfo, mount, local
):
    support.write_file(host_case.roots.proc / "self/mountinfo", mountinfo)
    collector = host_case.build("_filesystems", filesystems=[mount])

    batch = collector.collect_once()

    assert batch.collection_errors == []
    samples = support.values(batch)
    assert samples[("filesystem_used_percent", mount)] == 50
    assert (("local_filesystem_used_percent", mount) in samples) is local


@pytest.mark.parametrize("shared", [False, True])
def test_zero_capacity_or_absent_mounts_do_not_create_usage_percent(host_case, shared):
    support.write_file(
        host_case.roots.proc / "self/mountinfo",
        "unparseable\n1 2 3 / /zero rw - nfs4 remote rw\n"
        "1 2 3 / /missing rw - nfs remote rw\n1 2 3 / /local rw - xfs disk rw\n",
    )

    def probe(path):
        if path == "/missing":
            raise OSError("mount absent")
        return os.statvfs_result((4096, 4096, 0, 0, 0, 0, 0, 0, 0, 255))

    collector = host_case.build(
        "_shared_filesystems" if shared else "_filesystems",
        filesystems=["/zero", "/missing"],
        statvfs=probe,
    )
    batch = collector.collect_once()
    assert batch.collection_errors == []
    if shared:
        assert support.values(batch) == {
            ("shared_filesystem_unavailable", "/zero"): 0,
            ("shared_filesystem_unavailable", "/missing"): 1,
        }
    else:
        assert batch.samples == []


def test_absent_mountinfo_and_short_diskstats_are_ignored(host_case):
    support.write_file(
        host_case.roots.proc / "diskstats",
        "too short\n7 0 loop0 1 0 0 0 1 0 0 0 0 1 2\n1 0 ram0 1 0 0 0 1 0 0 0 0 1 2\n",
    )
    collector = host_case.build("_shared_filesystems", "_diskstats")
    assert collector.collect_once().samples == []
    host_case.clock.sleep(1)
    assert collector.collect_once().samples == []


@pytest.mark.parametrize(
    ("payload", "returncode", "expected", "query_failed"),
    [
        ([], 0, 1, True),
        ({"smartctl": []}, 0, None, False),
        ({"smart_status": None}, 1, 1, True),
        ({"smart_status": {"passed": True}}, 0, 0, False),
        ({"smart_status": {"passed": False}}, 0, 1, False),
    ],
)
def test_smart_reports_query_failure_separately_from_health_verdict(
    host_case, payload, returncode, expected, query_failed
):
    support.enable_tools(host_case, "smartctl")
    host_case.outputs[("smartctl", "--scan-open")] = (
        "\nnot-a-device\n/dev/test0 -d nvme\n/dev/test0 -d nvme\n",
        "",
        0,
    )
    host_case.outputs[("smartctl", "-H", "-j", "/dev/test0")] = (
        json.dumps(payload),
        "",
        returncode,
    )
    collector = host_case.build("_smart")
    first = collector.collect_once()
    host_case.clock.sleep(1)
    second = collector.collect_once()

    assert first.samples == second.samples
    if expected is None:
        assert first.samples == []
    else:
        assert first.samples[0].value == expected
        assert ("failure_mode" in first.samples[0].labels) is query_failed
    probes = [argv for argv, _ in host_case.commands if "-H" in argv]
    assert len(probes) == (2 if query_failed else 1)


@pytest.mark.parametrize("returncode", [0, 1])
def test_bmc_probe_distinguishes_unavailable_from_critical_sensors(
    host_case, returncode
):
    support.enable_tools(host_case, "ipmitool")
    host_case.outputs[("ipmitool", "sensor")] = (
        "short\nTemp | 60 | OK\nFan | 0 | cr\nPower | 0 | non-recoverable\n",
        "",
        returncode,
    )
    collector = host_case.build("_bmc")
    batch = collector.collect_once()
    assert support.values(batch) == (
        {} if returncode else {("bmc_critical_sensor", None): 2}
    )
    assert host_case.commands[0][1]["timeout"] == 20


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("GPU_FAULT_LOW_UTILIZATION_THRESHOLD_PERCENT", "-1", "percentage"),
        ("GPU_FAULT_MEMORY_AVAILABLE_WARNING_PERCENT", "101", "percentage"),
        ("GPU_FAULT_EFA_TRAFFIC_DROP_RATIO", "1", "DROP_RATIO"),
        ("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO", "1", "SPIKE_RATIO"),
        ("GPU_FAULT_RANK_PROGRESS_MIN_WRITE_BPS", "-1", "rank progress"),
        ("GPU_FAULT_RANK_PROGRESS_MIN_CPU_CORES", "0", "rank progress"),
        ("GPU_FAULT_RANK_PROGRESS_GPU_IDLE_PERCENT", "101", "rank progress"),
    ],
)
def test_host_refuses_thresholds_that_would_disable_findings(
    host_case, name, value, message
):
    host_case.monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        host_case.build()
    assert host_case.commands == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"statvfs_timeout_seconds": 0}, "statvfs timeout"),
        ({"statvfs_budget_seconds": 0}, "statvfs budget"),
        ({"inventory_mismatch_consecutive_samples": 0}, "mismatch"),
        ({"history_max_points": 0}, "edge filter"),
        ({"health_summary_seconds": 0}, "edge filter"),
        ({"startup_spread_seconds": 0}, "edge filter"),
    ],
)
def test_host_rejects_invalid_sampling_bounds_before_any_probe(
    host_case, options, message
):
    with pytest.raises(ValueError, match=message):
        host_case.build(**options)
    assert host_case.commands == []


@pytest.mark.parametrize(
    "error", [OSError("fake syscall failure"), RuntimeError("fake backend failure")]
)
def test_statvfs_worker_propagates_fake_probe_error_without_hanging(error):
    def probe(path):
        raise error

    bounded = BoundedStatvfs(probe, timeout_seconds=1)
    with pytest.raises(OSError, match="fake"):
        bounded("/private/mount")


@pytest.mark.parametrize("address", [None, "@private", "/private/notify"])
@pytest.mark.parametrize("fail", [False, True])
def test_systemd_notifications_use_only_the_fake_datagram_boundary(
    monkeypatch, address, fail
):
    calls = []

    class Datagram:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def connect(self, value):
            calls.append(value)
            if fail:
                raise OSError("fake notification unavailable")

        def sendall(self, value):
            calls.append(value)

    if address is None:
        monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    else:
        monkeypatch.setenv("NOTIFY_SOCKET", address)
    monkeypatch.setattr(
        host_module,
        "socket",
        SimpleNamespace(AF_UNIX=1, SOCK_DGRAM=2, socket=lambda *args: Datagram()),
    )
    assert host_module.sd_notify("WATCHDOG=1") is (address is not None and not fail)
    if address is None:
        assert calls == []
    else:
        expected_address = "\0private" if address == "@private" else address
        assert calls[0] == expected_address
        assert calls[-1] == "closed"


def test_force_snapshot_is_retained_on_delivery_failure_until_a_successful_tick(
    host_case, monkeypatch, caplog
):
    collector = host_case.build()
    collector.force_snapshot_path.write_text("requested")
    host_case.sink.error = CollectorError("fake rejected batch")
    notifications = []
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            assert collector.force_snapshot_path.exists(), (
                "failed delivery must retain the force-snapshot request"
            )
            host_case.sink.error = None
            host_case.clock.sleep(seconds)
        else:
            raise common.StopLoop

    monkeypatch.setattr(host_module, "sd_notify", notifications.append)
    monkeypatch.setattr(host_module, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(common.StopLoop):
        collector.run()
    assert not collector.force_snapshot_path.exists(), (
        "successful retry must consume the force-snapshot request"
    )
    assert len(host_case.sink.requests) == 2
    assert notifications == ["READY=1", "WATCHDOG=1", "WATCHDOG=1"]
    assert "fake rejected batch" in caplog.text
