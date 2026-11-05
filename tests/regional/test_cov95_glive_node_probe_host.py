"""GF-GENERAL-LIVE host probe: its read-only facts, read from a fake node.

``general_live_node_probe`` reads the node's ``/proc`` and ``/sys`` itself and
reports them to the runner (``survey``, the oracle sampler, ``efa-traffic``).
Here those reads are answered from a tree under ``tmp_path`` (``NodeHost``):
the tests check the values the probe derives, the lines it skips, and the one
JSON line it prints. Nothing runs systemd or reads the host's counters.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_node_probe as nodep
from tests.regional._cov95_glive_node import RUN, NodeHost, command, node_host

ROOT_MOUNT = {
    "mountpoint": "/",
    "fstype": "xfs",
    "source": "/dev/nvme0n1p1",
    "options": "rw,relatime",
}
MOUNTINFO = (
    "22 1 259:1 / / rw,relatime shared:1 - xfs /dev/nvme0n1p1 rw\n"
    "41 22 0:50 / /var/lib/kubelet rw,nosuid shared:3 - tmpfs\n"
    "40 22 259:3 / /var rw,noatime shared:2 - ext4 /dev/nvme1n1 rw\n"
    "43 22 259:4 / /data rw shared:4 - ext4 /dev/nvme2n1 rw\n"
    "a line some kernel printed without the optional-field separator\n"
)
SNMP_HEADER = (
    "Tcp: RtoAlgorithm RtoMin RtoMax MaxConn ActiveOpens PassiveOpens "
    "AttemptFails EstabResets CurrEstab InSegs OutSegs RetransSegs InErrs OutRsts"
)
SNMP = (
    "Ip: Forwarding DefaultTTL\nIp: 1 64\n"
    f"{SNMP_HEADER}\nTcp: 1 200 120000 -1 5 6 7 8 9 10 11 42 0 1\n"
    "Udp: InDatagrams NoPorts\nUdp: 3 4\n"
)
NET_COUNTERS = (
    "rx_errors",
    "tx_errors",
    "rx_dropped",
    "tx_dropped",
    "rx_packets",
    "tx_packets",
)


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> NodeHost:
    return node_host(monkeypatch, tmp_path)


# --------------------------------------------------------------------------- #
# the probe's own output and command boundary
# --------------------------------------------------------------------------- #
def test_emit_prints_one_compact_sorted_json_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    nodep.emit({"b": [1, 2], "a": Path("/run/gfrs")})
    assert capsys.readouterr().out == '{"a":"/run/gfrs","b":[1,2]}\n'


def test_run_hands_the_command_to_subprocess_and_names_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **options: Any) -> Any:
        seen.append((list(command), options))
        code = 3 if command[0] == "false" else 0
        return subprocess.CompletedProcess(command, code, "out", "  denied  \n")

    fake = SimpleNamespace(run=fake_run, PIPE=subprocess.PIPE)
    monkeypatch.setattr(nodep, "subprocess", fake)
    with pytest.raises(nodep.ProbeError) as caught:
        nodep.run(["false", "two words"])
    assert str(caught.value) == "command failed (3): false 'two words': denied"
    completed = nodep.run(["false"], check=False, timeout=7, input_text="payload")
    assert (completed.returncode, completed.stdout) == (3, "out")
    assert nodep.run(["true"]).returncode == 0
    assert seen[0][1] == {
        "input": None,
        "text": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "timeout": 180,
        "check": False,
    }, "the probe never lets subprocess raise; it judges the return code itself"
    assert (seen[1][1]["input"], seen[1][1]["timeout"]) == ("payload", 7)


def test_unit_state_keeps_assignments_and_the_return_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "ActiveState=failed\nWarning: unit file changed on disk\n\nMainPID=0\n"
    monkeypatch.setattr(
        nodep,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 1, output, ""),
    )
    state = nodep.unit_state("gpu-fault-host-collector.service")
    assert state == {"ActiveState": "failed", "MainPID": "0", "returncode": "1"}


# --------------------------------------------------------------------------- #
# collector env, memory, mounts, backing candidates
# --------------------------------------------------------------------------- #
def test_parse_env_reads_assignments_and_skips_comments_and_noise(
    host: NodeHost,
) -> None:
    assert nodep.parse_env() == {}, "a node without collector.env reports nothing"
    host.write(
        "/etc/gpu-fault/collector.env",
        "# written by the node bootstrap\n\n"
        "GPU_FAULT_METRICS_MODE='dcgm'\n"
        '  GPU_FAULT_FILESYSTEMS="/,/data"  \n'
        "not an assignment\n"
        "GPU_FAULT_DCGM_METRICS_URL=http://127.0.0.1:9400/metrics?a=b\n",
    )
    assert nodep.parse_env() == {
        "GPU_FAULT_METRICS_MODE": "dcgm",
        "GPU_FAULT_FILESYSTEMS": "/,/data",
        "GPU_FAULT_DCGM_METRICS_URL": "http://127.0.0.1:9400/metrics?a=b",
    }


def test_meminfo_converts_kib_and_skips_unparsable_lines(host: NodeHost) -> None:
    host.write(
        "/proc/meminfo",
        "MemTotal:       2048 kB\nMemAvailable:   1024 kB\n"
        "HugePages_Total:       0\nBroken:\nOdd: lots kB\n",
    )
    assert nodep.meminfo() == {
        "MemTotal": 2048.0 * 1024,
        "MemAvailable": 1024.0 * 1024,
        "HugePages_Total": 0.0,
    }


def test_mounts_and_mount_of_pick_the_longest_covering_mountpoint(
    host: NodeHost,
) -> None:
    host.write("/proc/self/mountinfo", MOUNTINFO)
    listed = nodep.mounts()
    assert [item["mountpoint"] for item in listed] == [
        "/",
        "/var/lib/kubelet",
        "/var",
        "/data",
    ], "a line without the ' - ' separator is skipped"
    assert listed[0] == ROOT_MOUNT
    assert listed[1]["source"] == "", "a mount line without a source field"
    assert nodep.mount_of(Path("/var/lib/kubelet/pods/x"))["mountpoint"] == (
        "/var/lib/kubelet"
    ), "a shorter covering mount listed later does not win"
    assert nodep.mount_of(Path("/var"))["source"] == "/dev/nvme1n1"
    assert nodep.mount_of(Path("/srv/other")) == ROOT_MOUNT
    host.write("/proc/self/mountinfo", "43 22 259:4 / /data rw - ext4 /dev/x rw\n")
    assert nodep.mount_of(Path("/srv/other")) is None, "nothing covers the path"


def test_backing_candidates_walk_up_to_the_nearest_existing_directory(
    host: NodeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host.write("/proc/self/mountinfo", MOUNTINFO)
    present = host.path("/var/lib/gpu-fault-acceptance")
    present.mkdir(parents=True)
    absent = host.path("/opt/dlami/nvme/gpu-fault-acceptance")
    orphan = Path("/gfrs-cov95-absent-root/gpu-fault-acceptance")
    monkeypatch.setattr(nodep, "BACKING_ROOTS", (present, absent, orphan))
    first, second, third = nodep.backing_candidates()
    assert first["root"] == first["nearest"] == str(present)
    assert first["exists"] is True and first["parent_present"] is True
    assert first["mount"] == ROOT_MOUNT and first["statvfs"]["path"] == str(present)
    assert (second["exists"], second["nearest"]) == (False, str(host.root))
    assert second["parent_present"] is True, "an existing ancestor below / counts"
    assert (third["exists"], third["nearest"]) == (False, "/")
    assert third["parent_present"] is False, "only / itself exists above it"


# --------------------------------------------------------------------------- #
# processes, modules and the survey
# --------------------------------------------------------------------------- #
def test_process_readers_tolerate_missing_and_short_proc_entries(
    host: NodeHost,
) -> None:
    host.process(812, ["/usr/bin/dcgm-exporter", "-f", "/etc/dcgm/counters.csv"])
    host.write("/proc/91/stat", "91 (short) S 1\n")
    assert nodep.process_cmdline(812) == [
        "/usr/bin/dcgm-exporter",
        "-f",
        "/etc/dcgm/counters.csv",
    ]
    assert nodep.process_cmdline(5555) == [], "a process that is gone"
    assert nodep.process_start_ticks(812) == "8120"
    assert nodep.process_start_ticks(91) is None, "a stat without field 22"
    assert nodep.process_start_ticks(5555) is None


def test_exporter_processes_lists_only_dcgm_exporter_binaries(host: NodeHost) -> None:
    host.process(1, ["/sbin/init"])
    host.process(812, ["/usr/bin/dcgm-exporter", "-f", "/etc/dcgm/counters.csv"])
    host.process(77, ["/bin/dcgm-exporter"])
    host.write("/proc/90/cmdline", b"")
    host.path("/proc/self").mkdir(parents=True)
    assert nodep.exporter_processes() == [
        {"pid": 77, "start_ticks": "770", "argv": ["/bin/dcgm-exporter"]},
        {
            "pid": 812,
            "start_ticks": "8120",
            "argv": ["/usr/bin/dcgm-exporter", "-f", "/etc/dcgm/counters.csv"],
        },
    ]


def test_module_present_asks_modinfo_only_for_unloaded_modules(
    host: NodeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host.path("/sys/module/loop").mkdir(parents=True)
    host.modules = {"dm_delay"}
    monkeypatch.setattr(
        nodep.shutil,
        "which",
        lambda name: "/usr/sbin/modinfo" if name == "modinfo" else None,
    )
    assert nodep.module_present("loop") == {"loaded": True, "available": True}
    assert nodep.module_present("dm_delay") == {"loaded": False, "available": True}
    assert nodep.module_present("vfat") == {"loaded": False, "available": False}
    assert host.calls("modinfo") == [["modinfo", "dm_delay"], ["modinfo", "vfat"]]
    assert {item["timeout"] for item in host.options} == {30}
    monkeypatch.setattr(nodep.shutil, "which", lambda name: None)
    assert nodep.module_present("dm_delay") == {"loaded": False, "available": False}
    assert len(host.calls("modinfo")) == 2, "no modinfo binary: nothing is run"


def test_survey_reports_the_node_as_the_runner_reads_it(
    host: NodeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host.write(
        "/etc/gpu-fault/collector.env",
        "GPU_FAULT_METRICS_MODE=dcgm\nGPU_FAULT_EXPECTED_GPU_COUNT=8\nOTHER=x\n",
    )
    host.write("/proc/meminfo", "MemTotal: 1000 kB\nMemAvailable: 250 kB\n")
    host.write("/proc/sys/kernel/random/boot_id", "6f1c0e3e-boot\n")
    host.write("/proc/self/mountinfo", MOUNTINFO)
    host.path("/sys/module/dm_mod").mkdir(parents=True)
    host.process(812, ["/usr/bin/dcgm-exporter"])
    host.unit("gpu-fault-node-agent.service", MainPID="321")
    present = {"systemd-run", "systemctl", "ip"}
    monkeypatch.setattr(
        nodep.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in present else None,
    )
    command("survey", "--run-id", RUN)
    report = host.emitted[-1]
    assert report["boot_id"] == "6f1c0e3e-boot"
    assert report["cpu_count"] == os.cpu_count()
    assert report["memory"]["MemTotal"] == 1000.0 * 1024
    assert report["memory"]["memory_used_percent"] == 75.0
    assert report["memory"]["SwapTotal"] is None
    assert {name for name, found in report["tools"].items() if found} == present
    assert report["modules"]["dm_mod"] == {"loaded": True, "available": True}
    assert report["modules"]["vfat"] == {"loaded": False, "available": False}
    assert report["collector_env"]["GPU_FAULT_EXPECTED_GPU_COUNT"] == "8"
    assert report["collector_env"]["GPU_FAULT_DCGM_METRICS_URL"] is None
    assert "OTHER" not in report["collector_env"], "only the documented keys"
    assert report["units"]["gpu-fault-node-agent.service"]["MainPID"] == "321"
    assert report["units"]["nvidia-dcgm.service"]["LoadState"] == "not-found"
    assert set(report["units"]) == set(nodep.PRODUCT_UNITS)
    assert [item["root"] for item in report["backing"]] == [
        str(root) for root in nodep.BACKING_ROOTS
    ]
    assert report["root_mount"] == ROOT_MOUNT
    assert report["exporter_processes"] == [{"pid": 812, "start_ticks": "8120"}]
    assert nodep.residue_is_clean(report["residue"]), report["residue"]


# --------------------------------------------------------------------------- #
# oracle sources
# --------------------------------------------------------------------------- #
def test_cpu_counters_sum_the_first_eight_fields(host: NodeHost) -> None:
    host.write("/proc/stat", "cpu  100 5 50 800 20 3 2 1 9 9\ncpu0 1 1 1 1 1 1 1 1\n")
    assert nodep.cpu_counters() == (981.0, 820.0), "idle is idle + iowait"
    host.write("/proc/stat", "cpu 1 2 3 4\n")
    assert nodep.cpu_counters() == (10.0, 4.0), "an old kernel without iowait"


def test_diskstats_fields_finds_the_device_row(host: NodeHost) -> None:
    host.write(
        "/proc/diskstats",
        "   7       0 loop0 0 0 0 0\n"
        " 259       0 nvme0n1 10 0 80 5 20 0 160 9 0 12 14\n"
        " 253       7 dm-7 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17\n",
    )
    assert nodep.diskstats_fields("dm-7") == [float(n) for n in range(1, 18)]
    assert nodep.diskstats_fields("nvme0n1")[:3] == [10.0, 0.0, 80.0]
    assert nodep.diskstats_fields("loop0") is None, "a short row is not a match"
    assert nodep.diskstats_fields("sdz") is None


def test_netdev_reading_reports_presence_counters_and_read_errors(
    host: NodeHost,
) -> None:
    base = "/sys/class/net/veth-gl0"
    host.write(f"{base}/operstate", "up\n")
    host.write(f"{base}/ifindex", "17\n")
    for value, counter in enumerate(NET_COUNTERS):
        host.write(f"{base}/statistics/{counter}", f"{value}\n")
    assert nodep.netdev_reading("veth-gl0") == {
        "present": True,
        "operstate": "up",
        "ifindex": 17,
        **{counter: value for value, counter in enumerate(NET_COUNTERS)},
    }
    assert nodep.netdev_reading("absent0") == {"present": False}
    host.write("/sys/class/net/bad0/operstate", "down\n")
    host.write("/sys/class/net/bad0/ifindex", "not-a-number\n")
    assert nodep.netdev_reading("bad0") == {
        "present": True,
        "operstate": "down",
        "error": "ValueError",
    }
    host.write("/sys/class/net/half0/operstate", "up\n")
    host.write("/sys/class/net/half0/ifindex", "3\n")
    assert nodep.netdev_reading("half0") == {
        "present": True,
        "operstate": "up",
        "ifindex": 3,
        "error": "FileNotFoundError",
    }


class UnresolvablePath(type(Path())):
    """A sysfs tree whose ``device/driver`` links cannot be resolved (EACCES)."""

    def resolve(self, strict: bool = False) -> Any:
        if self.name == "driver":
            raise PermissionError(13, "Permission denied", str(self))
        return super().resolve(strict)


def _efa_device(
    root: Path, device: str, counters: dict[str, str], *, uevent: str | None
) -> None:
    (root / device / "device").mkdir(parents=True)
    if uevent is not None:
        (root / device / "device" / "uevent").write_text(uevent)
    hw = root / device / "ports" / "1" / "hw_counters"
    hw.mkdir(parents=True)
    for name, value in counters.items():
        (hw / name).write_text(value)


def test_efa_traffic_reading_resolves_driver_links_and_marks_bad_counters(
    tmp_path: Path,
) -> None:
    root = tmp_path / "infiniband"
    (tmp_path / "drivers" / "efa").mkdir(parents=True)
    _efa_device(root, "rdmap0", {"rx_bytes": "100\n", "tx_bytes": "40\n"}, uevent=None)
    (root / "rdmap0" / "device" / "driver").symlink_to(tmp_path / "drivers" / "efa")
    _efa_device(root, "rdmap1", {"rx_bytes": "9", "tx_bytes": "9"}, uevent=None)
    _efa_device(
        root, "rdmap2", {"rx_bytes": "garbage", "tx_bytes": "1"}, uevent="DRIVER=efa\n"
    )
    reading = nodep.efa_traffic_reading(root)
    assert reading["ports"] == [
        {
            "device": "rdmap0/1",
            "pair": ("rx_bytes", "tx_bytes"),
            "rx_bytes": 100,
            "tx_bytes": 40,
        },
        {"device": "rdmap2/1", "pair": ("rx_bytes", "tx_bytes"), "error": True},
    ], "rdmap1 has neither an efa driver link nor a uevent: not an EFA device"
    assert reading["pair_bytes_total"] == 140
    assert reading["complete_pairs"] is False, "an unreadable counter pair"


def test_efa_traffic_reading_falls_back_to_uevent_for_an_unresolvable_link(
    tmp_path: Path,
) -> None:
    root = UnresolvablePath(tmp_path / "infiniband")
    _efa_device(
        root, "rdmap0", {"send_bytes": "7", "recv_bytes": "5"}, uevent="DRIVER=efa\n"
    )
    _efa_device(
        root, "mlx5_0", {"rx_bytes": "1", "tx_bytes": "1"}, uevent="DRIVER=mlx5\n"
    )
    reading = nodep.efa_traffic_reading(root)
    assert [port["device"] for port in reading["ports"]] == ["rdmap0/1"]
    assert reading["pair_bytes_total"] == 12 and reading["complete_pairs"] is True


def test_efa_traffic_command_prints_the_reading_of_the_node(
    host: NodeHost, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "infiniband"
    _efa_device(
        root, "rdmap0", {"rx_bytes": "30", "tx_bytes": "12"}, uevent="DRIVER=efa\n"
    )
    reading = nodep.efa_traffic_reading
    monkeypatch.setattr(nodep, "efa_traffic_reading", lambda: reading(root))
    command("efa-traffic", "--run-id", RUN)
    assert host.emitted[-1]["pair_bytes_total"] == 42
    assert host.emitted[-1]["observed_at"] == host.clock.now


def test_tcp_retransmits_reads_the_tcp_counter_pair(host: NodeHost) -> None:
    host.write("/proc/net/snmp", SNMP)
    assert nodep.tcp_retransmits() == 42
    host.write("/proc/net/snmp", "Tcp: RtoAlgorithm RtoMin\nTcp: 1 200\nUdp: 1\n")
    assert nodep.tcp_retransmits() is None, "a Tcp pair without RetransSegs"


def test_oracle_sample_reads_every_requested_source(
    host: NodeHost, tmp_path: Path
) -> None:
    host.write("/proc/loadavg", "6.00 5.00 4.00 2/900 1234\n")
    host.write("/proc/meminfo", "MemTotal: 1000 kB\nMemAvailable: 250 kB\n")
    host.write("/proc/net/snmp", SNMP)
    host.write("/proc/diskstats", " 259 0 nvme0n1 1 2 3 4 5 6 7 8 9 10 11\n")
    host.write("/sys/class/net/veth-gl0/operstate", "down\n")
    spec = {
        "loadavg": True,
        "meminfo": True,
        "statvfs": [str(tmp_path)],
        "diskstats": ["nvme9n1"],
        "netdev": ["veth-gl0", "absent0"],
        "snmp": True,
    }
    previous: dict[str, Any] = {}
    record = nodep.oracle_sample(spec, previous)
    assert record["t"] == host.clock.now
    assert record["load1"] == 6.0
    assert record["load1_per_cpu"] == 6.0 / max(1, os.cpu_count() or 1)
    assert record["memory_used_percent"] == 75.0
    assert record["statvfs"][str(tmp_path)]["path"] == str(tmp_path)
    assert record["diskstats"] == {"nvme9n1": {"present": False}}
    assert previous == {}, "an absent device leaves no baseline for the next sample"
    assert record["netdev"]["absent0"] == {"present": False}
    assert record["netdev"]["veth-gl0"]["operstate"] == "down"
    assert record["tcp_retransmits"] == 42


def test_sample_loop_appends_one_record_per_interval_until_the_deadline(
    host: NodeHost, tmp_path: Path
) -> None:
    host.write("/proc/loadavg", "1.50 1 1 1/1 1\n")
    output = tmp_path / "oracle.jsonl"
    spec = json.dumps({"loadavg": True})
    argv = ["sample-loop", "--output", str(output), "--interval", "0.2", "--spec", spec]
    command(*argv, "--seconds", "10")
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["t"] for record in records] == [1000.0 + n for n in range(10)]
    assert {record["load1"] for record in records} == {1.5}
    assert host.clock.sleeps == [1.0] * 10, "the interval floors at one second"
    with pytest.raises(nodep.ProbeError, match="sampler seconds is outside 10..3600"):
        command(*argv, "--seconds", "5")
    assert len(output.read_text().splitlines()) == 10, "a refused loop writes nothing"
